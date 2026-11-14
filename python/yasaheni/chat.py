# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""One chat request, end to end: messages (and tool definitions) -> prompt ids -> streamed generation -> reasoning / answer text
and tool calls."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import loopguard
from ._native import YasakaniError
from .engine import CONTEXT_MARGIN, Engine, GenResult, Sampling
from .runtime import EOS_IDS
from .textio import ChatFormatter, OutputStream, ToolCallFilter

# defaults of the model (generation_config.json of Qwen3.8)
DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K = 1.0, 0.95, 20
REASONING_EFFORT_ALIASES = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


class RequestError(ValueError):
    """The request cannot be served as asked (maps to HTTP 400)."""


class ContextLengthError(RequestError):
    """The input does not fit into the context (HTTP 400 with the error code ``context_length_exceeded``)."""
    code = "context_length_exceeded"


def context_length_error(n_tokens: int, max_ctx: int) -> ContextLengthError:
    """The error for an input of ``n_tokens`` tokens (images included) that leaves no room for an answer."""
    return ContextLengthError("the input is too long: %d tokens exceeds the model's maximum context length of %d tokens "
                              "(input, thinking and answer together)" % (n_tokens, max_ctx))


@dataclass
class ChatParams:
    messages: List[dict]
    max_tokens: int = 32768
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False
    loop_guard: bool = True
    thinking_budget: int = 0                   # > 0: cut the thinking after this many tokens and go on to the answer (0 = no limit)
    template_kwargs: Dict[str, object] = field(default_factory=dict)
    tools: Optional[List[dict]] = None            # OpenAI-style tool definitions; None = a plain chat request


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    finish: str = "stop"
    passes: int = 0
    prefill_ms: float = 0.0
    gen_ms: float = 0.0
    snap_mode: int = 0
    loop_guard: Optional[dict] = None
    tool_calls: int = 0


class StopFilter:
    """Holds back text that might be the start of a stop string; cuts at the first complete one."""

    def __init__(self, stops: Sequence[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""
        self.hit = False

    def push(self, text: str) -> str:
        if not self.stops or self.hit:
            return "" if self.hit else text
        self.buf += text
        cuts = [self.buf.find(s) for s in self.stops if s in self.buf]
        if cuts:
            out = self.buf[:min(cuts)]
            self.buf, self.hit = "", True
            return out
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        cut = len(self.buf) - keep
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


def _clean_content(c, i: int, images: bool = False):
    if isinstance(c, list):
        parts = []
        for p in c:
            if images and isinstance(p, dict) and p.get("type") == "image_url":
                value = p.get("image_url")
                if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"]:
                    raise RequestError("messages[%d]: image_url needs a nonempty url string" % i)
                if value.get("detail", "auto") not in ("auto", "low", "high"):
                    raise RequestError("messages[%d]: invalid image detail" % i)
                parts.append({"type": "image_url", "image_url": dict(value)})
                continue
            if not isinstance(p, dict) or p.get("type") not in ("text", None) or "text" not in p:
                raise RequestError("messages[%d]: only text content is supported" % i)
            parts.append({"type": "text", "text": str(p["text"])})
        return parts
    if c is None:
        return ""
    if not isinstance(c, str):
        raise RequestError("messages[%d]: content must be a string or a list of text parts" % i)
    return c


def _clean_tool_calls(calls, i: int) -> List[dict]:
    """The tool calls of an assistant message, in the form the chat template reads: the arguments as an object."""
    if not isinstance(calls, list):
        raise RequestError("messages[%d]: tool_calls must be a list" % i)
    out = []
    for k, c in enumerate(calls):
        f = c.get("function") if isinstance(c, dict) else None
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
            raise RequestError("messages[%d].tool_calls[%d]: a function call with a name is expected" % (i, k))
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                raise RequestError("messages[%d].tool_calls[%d]: arguments is not valid JSON" % (i, k))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise RequestError("messages[%d].tool_calls[%d]: arguments must be a JSON object" % (i, k))
        out.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
    return out


def clean_messages(messages, *, images: bool = False) -> List[dict]:
    """Validate OpenAI-style messages and return plain dicts.  ``developer`` is read as
    ``system``; an assistant message may carry ``tool_calls`` and a ``tool`` message carries the result of one call.
    ``image_url`` parts are kept only with ``images`` (image input configured); otherwise they are refused as in 0.1.1."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out = []
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else None
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError("messages[%d]: role must be system, user, assistant or tool" % i)
        msg = {"role": role, "content": _clean_content(m.get("content"), i, images)}
        if role != "user" and isinstance(msg["content"], list) and any(p["type"] == "image_url" for p in msg["content"]):
            raise RequestError("messages[%d]: images are supported only in user messages" % i)
        if role == "assistant":
            if isinstance(m.get("reasoning_content"), str):
                msg["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = _clean_tool_calls(m["tool_calls"], i)
        out.append(msg)
    return out


def clean_tools(tools) -> Optional[List[dict]]:
    """Validate OpenAI-style tool definitions (functions only).  None or an empty list means: no tools."""
    if tools is None or tools == []:
        return None
    if not isinstance(tools, list):
        raise RequestError("tools must be a list")
    out = []
    for i, t in enumerate(tools):
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(t, dict) or t.get("type", "function") != "function" or not isinstance(f, dict):
            raise RequestError("tools[%d]: only tools of type \"function\" are supported" % i)
        if not isinstance(f.get("name"), str) or not f["name"] or any(ch in f["name"] for ch in "<>\n"):
            raise RequestError("tools[%d]: the function needs a name (without <, > or line breaks)" % i)
        if f.get("parameters") is not None and not isinstance(f["parameters"], dict):
            raise RequestError("tools[%d]: parameters must be a JSON schema object" % i)
        fn = {"name": f["name"]}
        if isinstance(f.get("description"), str):
            fn["description"] = f["description"]
        fn["parameters"] = f.get("parameters") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": fn})
    return out


class ChatRunner:
    """Runs chat requests on one engine, one at a time."""

    def __init__(self, engine: Engine, fmt: ChatFormatter, seed0: int = 1, *, vision=None):
        self.engine = engine
        self.fmt = fmt
        self.vision = vision
        self._seed = int(seed0)
        self._seed_lock = threading.Lock()
        self.bridge = fmt.encode(loopguard.BRIDGE)
        if fmt.think_end not in self.bridge.tolist():
            raise YasakaniError("tokenizer: </think> is not a single id in the loop-guard bridge")

    def next_seed(self) -> int:
        with self._seed_lock:
            s = self._seed
            self._seed += 1
            return s

    def warmup(self) -> None:
        plain = self.fmt.encode("warm-up")
        chat, _ = self.fmt.prompt([{"role": "user", "content": "warm-up"}])
        self.engine.warmup([plain, chat], n_gen=32)

    def prepare(self, p: ChatParams):
        from .vision import split_images, PreparedVisual
        messages, sources = split_images(p.messages)
        if sources and self.vision is None:
            raise RequestError("only text content is supported: image input is not configured "
                               "(start the server with --mmproj, --llama-dir and --vision-model)")
        kw = dict(p.template_kwargs)
        if p.tools:
            kw["tools"] = p.tools
            try:
                from jinja2.exceptions import TemplateError
                prompt, in_reasoning = self.fmt.prompt(messages, **kw)
            except TemplateError as ex:                  # e.g. a tool result without a user message before it
                raise RequestError("the chat template does not accept these messages: %s" % ex)
        else:
            prompt, in_reasoning = self.fmt.prompt(messages, **kw)
        if sources:
            if not self.engine.supports_visual:
                raise RequestError("the engine library does not support image input (yskn_set_visual is missing)")
            from .vision import error_text
            try:                                            # reads and encodes the images: the caller holds the generation lock
                prompt, visual = self.vision.prepare(prompt, sources, self.fmt, self.engine.max_ctx - CONTEXT_MARGIN,
                                                     self.engine.max_ctx)
            except RequestError:
                raise                                       # a 400 as it is (ContextLengthError keeps its error code)
            except ValueError as ex:                        # an image that cannot be used
                raise RequestError("image input: %s" % error_text(ex)) from ex
            except Exception as ex:                         # noqa: BLE001 - the image encoder failed: a 500, not a lost connection
                raise YasakaniError("image input failed: %s" % error_text(ex)) from ex
            return PreparedVisual(prompt, in_reasoning, visual)
        return prompt, in_reasoning

    def prepare_checked(self, p: ChatParams):
        """``prepare`` plus the length check: ``ContextLengthError`` if the input leaves no room for an answer.  The message uses
        the words programs look for ("exceeds the model's maximum context length of N tokens"): an agent program that sees them
        shortens the conversation and tries again."""
        prepared = self.prepare(p)
        prompt, in_reasoning = prepared
        if len(prompt) >= self.engine.max_ctx - CONTEXT_MARGIN:
            raise context_length_error(len(prompt), self.engine.max_ctx)
        return prepared

    def run(self, p: ChatParams, emit: Callable[[str, str], None], prepared=None) -> Usage:
        """Generate the reply, calling ``emit("reasoning" | "content", text)`` as text becomes available.  Raises ``RequestError`` for
        a request that cannot be served, ``YasakaniError`` for an engine failure.  ``prepared`` is the result of ``prepare_checked``
        when the caller has already made it (the server does, to answer a request it cannot serve before it starts a stream)."""
        if prepared is None:
            prepared = self.prepare_checked(p)
        visual = getattr(prepared, "visual", None)
        if visual is not None:
            with self.engine.visual_input(**visual):
                return self._run(p, emit, prepared)
        return self._run(p, emit, prepared)

    def close(self):
        if self.vision is not None:
            self.vision.close()

    def _run(self, p, emit, prepared):
        eng = self.engine
        prompt, in_reasoning = prepared if prepared is not None else self.prepare_checked(p)
        max_new = max(1, int(p.max_tokens))
        seed = p.seed if p.seed is not None else self.next_seed()
        eng.set_sampling(Sampling(p.temperature, p.top_p, p.top_k, p.min_p, seed))
        stops = None if p.ignore_eos else EOS_IDS
        usage = Usage(prompt_tokens=len(prompt))
        stopf = StopFilter(p.stop)
        toolf = ToolCallFilter(p.tools) if p.tools else None
        guard_on = p.loop_guard and not p.ignore_eos

        def answer(text):
            if toolf is None:
                emit("content", text)
            else:
                for kind, item in toolf.push(text):          # ("content", text) or ("tool_call", {"name", "arguments"})
                    emit(kind, item)

        def deliver(events):
            for kind, text in events:
                if kind == "content":
                    text = stopf.push(text)
                    if text:
                        answer(text)
                elif text:
                    emit(kind, text)
            if stopf.hit:
                eng.cancel()

        # ---- stage 1: the request as asked ----------------------------------------------------------------------------
        stream = OutputStream(self.fmt, in_reasoning)
        budget = max(0, int(p.thinking_budget)) if (in_reasoning and not p.ignore_eos) else 0
        watch = (loopguard.LoopWatch(self.fmt.think_end, self.fmt.digit_ids, in_reasoning, budget=budget, loops=guard_on)
                 if (guard_on or budget) else None)
        state: Dict[str, object] = {"hit": None}

        def on_tokens(ids):
            deliver(stream.feed(ids))
            if watch is not None and state["hit"] is None:
                hit = watch.check(ids)
                if hit is not None:
                    state["hit"] = hit
                    eng.cancel()

        r1 = eng.generate(prompt, max_new, stop_ids=stops, on_tokens=on_tokens)
        # the engine's final ids may include ones committed after the last poll: feed the complete array once more, then flush
        out_ids = np.asarray(r1.ids, np.int32)
        deliver(stream.feed(out_ids, final=True))
        usage.cached_tokens, usage.snap_mode = r1.cached_tokens, r1.snap_mode
        usage.passes, usage.prefill_ms, usage.gen_ms = r1.passes, r1.prefill_ms, r1.gen_ms
        n_out = len(out_ids)
        reasoning_end = stream.think_end_at
        usage.reasoning_tokens = (reasoning_end if reasoning_end is not None else (n_out if stream.in_reasoning else 0))
        finish = r1.finish
        hit = state["hit"]
        if hit is not None:
            info = dict(where=hit.where, kind=hit.kind, period=hit.period, run=hit.run, at=hit.at, tokens_before=n_out)
            usage.loop_guard = info
            if hit.where == "reasoning" and r1.finish != "stop" and not stopf.hit:
                finish = self._finish_after_loop(p, prompt, out_ids, usage, stops, stopf, emit, deliver, info, max_new)
            else:
                finish = "length" if not stopf.hit else "stop"
        if stopf.hit:
            finish = "stop"
        tail = stopf.flush()
        if tail:
            answer(tail)
        if toolf is not None:
            for kind, item in toolf.flush():
                emit(kind, item)
            usage.tool_calls = toolf.calls
        usage.completion_tokens = n_out + (usage.loop_guard or {}).get("extra_tokens", 0)
        usage.finish = "stop" if finish == "stop" else "length"
        if usage.tool_calls and usage.finish == "stop":
            usage.finish = "tool_calls"
        return usage

    def _finish_after_loop(self, p, prompt, out_ids, usage, stops, stopf, emit, deliver, info, max_new) -> str:
        """The thinking looped: close it with the bridge sentence and let the model write the answer on the same sequence."""
        eng = self.engine
        gen = len(out_ids)
        seq = np.concatenate([prompt, out_ids, self.bridge]).astype(np.int32)
        n2 = min(max_new - 1 - gen - len(self.bridge), eng.max_ctx - len(seq) - CONTEXT_MARGIN)
        if n2 < 16:
            info["answer_tokens"] = 0                      # no budget left for an answer
            return "length"
        emit("reasoning", loopguard.BUDGET_NOTICE if info.get("kind") == "budget" else loopguard.NOTICE)
        stream = OutputStream(self.fmt, False)
        watch = loopguard.LoopWatch(self.fmt.think_end, self.fmt.digit_ids, False, loops=p.loop_guard and not p.ignore_eos)
        hit2: Dict[str, object] = {"hit": None}

        def on_tokens(ids):
            deliver(stream.feed(ids))
            if hit2["hit"] is None:
                h = watch.check(ids)
                if h is not None:
                    hit2["hit"] = h
                    eng.cancel()

        r2 = eng.generate(seq, n2 + 1, stop_ids=stops, on_tokens=on_tokens)
        ids2 = np.asarray(r2.ids, np.int32)
        deliver(stream.feed(ids2, final=True))
        info.update(answer_tokens=len(ids2), answer_cached=r2.cached_tokens, extra_tokens=len(self.bridge) + len(ids2))
        usage.passes += r2.passes
        usage.prefill_ms += r2.prefill_ms
        usage.gen_ms += r2.gen_ms
        if hit2["hit"] is not None:
            info["answer_loop"] = dict(kind=hit2["hit"].kind, period=hit2["hit"].period, at=hit2["hit"].at)
        return r2.finish
