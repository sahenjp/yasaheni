#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Speed harness for YASAKANI and for any other OpenAI-compatible chat server.

One request at a time, streamed, one prompt file per task.  For every run the harness records the answer tokens the *server* reports
(usage.completion_tokens), the time to the first chunk that carried text, the time from that chunk to the last one, and the speed

    tok/s = (completion_tokens - 1) / (time from the first to the last text chunk)

Reading the prompt is not part of that time; it is reported separately as prefill_tok_per_s = prompt_tokens / ttft.  The token count
never comes from the number of chunks: one character can span several tokens and then arrives with the last of them, so a chunk count
would undercount and the speed would come out too high.

Runs that fell into a repetition loop or that ended in a tool call nobody asked for are flagged, kept in the file, and left out of the
summary: a looping answer is written faster than a normal one and would flatter the engine.

usage:  python3 bench/bench.py --base-url http://127.0.0.1:8000/v1 --engine yasakani --out bench/results/yasakani.json
        python3 bench/bench.py --tasks chat,code --runs 1 --max-tokens 512 --out /tmp/quick.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(HERE, "prompts")
RESULT_DIR = os.path.join(HERE, "results")
SCHEMA_VERSION = 1

# The 15 published prompts, in the order the tables of the upstream release use them.  Any other *.txt in prompts/ is appended
# alphabetically, so a new task is added by dropping a file in that folder.
TASK_ORDER = [
    "chat", "explain", "code", "math", "json", "long",
    "prose3_en_cpu", "prose3_en_story", "prose3_ja_essay",
    "japanese_q1", "japanese_q2", "japanese_q3", "japanese_q4", "japanese_q5",
    "sglang_random_8k_1",
]

# Loop detector.  See find_loop() and the rule written out in bench/README.md.
LOOP_WINDOW = 8192        # characters of the tail the detector looks at
LOOP_MAX_PERIOD = 256     # longest repeated unit it recognises, in characters
LOOP_MIN_SPAN = 128       # the repetition must cover at least this many characters
LOOP_REPEATS = 4          # ... and at least this many copies of the unit


# ---- the loop detector -----------------------------------------------------------------------------------------------------------
def find_loop(text: str) -> int:
    """Length of the repeated unit when the tail of text ends in a short cycle, else 0.

    Rule: take the last LOOP_WINDOW characters.  For every period p from 1 to LOOP_MAX_PERIOD, count the copies needed to cover
    LOOP_MIN_SPAN characters (at least LOOP_REPEATS of them) and test whether the last span characters are the last p characters
    repeated.  The smallest p that passes is the period; 0 means no loop.

    The last copy may be cut short (a server that hits its token limit stops mid-cycle), so the comparison is against the repetition
    truncated to the span, not against whole copies only.

    The minimum span is what keeps ordinary text out: a line of dots or four spaces at the end is not a loop, 128 characters of the
    same few tokens in a row is.  A repeated sentence is a loop too, which is why the period goes up to LOOP_MAX_PERIOD.
    """
    tail = text[-LOOP_WINDOW:]
    for period in range(1, LOOP_MAX_PERIOD + 1):
        copies = max(LOOP_REPEATS, -(-LOOP_MIN_SPAN // period))
        span = copies * period
        if span > len(tail):
            break
        unit = tail[-period:]
        if tail[-span:] == (unit * (span // period + 1))[:span]:
            return period
    return 0


# ---- one streamed request --------------------------------------------------------------------------------------------------------
def run_once(url: str, api_key: str, body_base: dict, prompt: str, timeout: float) -> dict:
    """Send one streamed chat completion and measure it.  Returns the fields of one entry of runs."""
    body = dict(body_base)
    body["messages"] = [{"role": "user", "content": prompt}]
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    parts = []            # text in arrival order (reasoning first, then the answer)
    usage = None
    finish = None
    tool_call = False
    t_first = None
    t_last = None

    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=timeout)
    except urllib.error.HTTPError as ex:
        detail = ex.read().decode("utf-8", "replace")[:400]
        raise SystemExit("HTTP %d from %s: %s" % (ex.code, url, detail))
    except urllib.error.URLError as ex:
        raise SystemExit("cannot reach %s: %s\nIs the server running?  See bench/README.md." % (url, ex.reason))

    with resp:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue                      # blank line, comment, or event: line
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            t = time.perf_counter() - t0
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("error"):
                raise SystemExit("the server reported an error: %s" % json.dumps(obj["error"])[:400])
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("tool_calls") or (choice.get("message") or {}).get("tool_calls"):
                    tool_call = True
                text = ""
                for key in ("reasoning_content", "reasoning", "content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        text += value
                if text:
                    parts.append(text)
                    if t_first is None:
                        t_first = t
                    t_last = t
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    t_end = time.perf_counter() - t0

    text = "".join(parts)
    flags = []
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = (usage or {}).get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(parts)        # last resort only; the run is flagged and left out of the summary
        flags.append("no_usage")
    completion_tokens = int(completion_tokens)
    if not text:
        flags.append("empty")
    if tool_call or finish == "tool_calls":
        flags.append("tool_calls")
    period = find_loop(text)
    if period:
        flags.append("loop")

    ttft_ms = (t_first if t_first is not None else t_end) * 1000.0
    gen_ms = ((t_last - t_first) * 1000.0) if (t_first is not None and t_last is not None) else 0.0
    tok_per_s = (completion_tokens - 1) / (gen_ms / 1000.0) if gen_ms > 0 and completion_tokens > 1 else 0.0
    entry = dict(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft_ms, 3),
        gen_ms=round(gen_ms, 3),
        tok_per_s=round(tok_per_s, 3),
        prefill_tok_per_s=round(prompt_tokens / (ttft_ms / 1000.0), 3) if ttft_ms > 0 else 0.0,
        wall_ms=round(t_end * 1000.0, 3),
        finish_reason=finish,
        flags=flags,
    )
    if period:
        entry["loop_period"] = period
        entry["loop_tail"] = text[-80:]
    return entry


# ---- the rest of the harness -----------------------------------------------------------------------------------------------------
def prompt_files() -> dict:
    """Task name to path, in TASK_ORDER first and then alphabetically for anything else in prompts/."""
    found = {}
    if os.path.isdir(PROMPT_DIR):
        for name in os.listdir(PROMPT_DIR):
            if name.endswith(".txt"):
                found[name[:-4]] = os.path.join(PROMPT_DIR, name)
    ordered = {t: found[t] for t in TASK_ORDER if t in found}
    for name in sorted(set(found) - set(ordered)):
        ordered[name] = found[name]
    return ordered


def probe_gpu() -> dict:
    """What nvidia-smi says about the first GPU.  Empty when it is not there; never fatal."""
    info = {"name": "", "driver": None, "memory_total_mib": None, "power_limit_w": None, "note": ""}
    exe = shutil.which("nvidia-smi")
    if not exe:
        info["note"] = "nvidia-smi not found"
        return info
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as ex:
        info["note"] = "nvidia-smi failed: %s" % ex
        return info
    # nvidia-smi prints some failures on stdout and still exits non-zero (a driver/library version mismatch, for one), so the exit
    # code and the shape of the line are both checked before anything is believed.
    lines = (out.stdout or "").strip().splitlines()
    if out.returncode != 0:
        info["note"] = "nvidia-smi exited %d: %s" % (out.returncode, (lines[0] if lines else "").strip()[:120])
        return info
    if not lines:
        info["note"] = "nvidia-smi printed nothing"
        return info
    fields = [f.strip() for f in lines[0].split(",")]
    if len(fields) < 2:
        info["note"] = "nvidia-smi printed %r" % lines[0].strip()[:120]
        return info

    def num(i):
        try:
            return float(fields[i])
        except (IndexError, ValueError):
            return None

    info["name"] = fields[0] if fields else ""
    info["driver"] = fields[1] if len(fields) > 1 else None
    info["memory_total_mib"] = num(2)
    info["power_limit_w"] = num(3)
    return info


def summarise(runs: list, tasks: list) -> dict:
    """Per task: mean/min/max of the runs that carry no flag, and how many were left out."""
    per_task = {}
    for task in tasks:
        used = [r["tok_per_s"] for r in runs if r["task"] == task and not r["flags"]]
        excluded = sum(1 for r in runs if r["task"] == task and r["flags"])
        per_task[task] = dict(
            mean=round(sum(used) / len(used), 3) if used else 0.0,
            min=round(min(used), 3) if used else 0.0,
            max=round(max(used), 3) if used else 0.0,
            n_used=len(used),
            n_excluded=excluded,
        )
    return {"per_task": per_task}


def print_table(tasks: list, summary: dict, runs: list) -> None:
    """Per task, over the runs that were not flagged: output tokens, wall time, TTFT, generation speed, prefill throughput."""
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    print()
    print("%-20s %4s %4s %7s %9s %9s %9s %8s %8s %8s %9s" %
          ("task", "used", "excl", "out tok", "wall ms", "ttft ms", "gen ms", "tok/s", "min", "max", "prefill"))
    print("-" * 118)
    for task in tasks:
        s = summary["per_task"][task]
        clean = [r for r in runs if r["task"] == task and not r["flags"]]
        print("%-20s %4d %4d %7.0f %9.1f %9.1f %9.1f %8.1f %8.1f %8.1f %9.1f" %
              (task, s["n_used"], s["n_excluded"],
               mean([r["completion_tokens"] for r in clean]), mean([r["wall_ms"] for r in clean]),
               mean([r["ttft_ms"] for r in clean]), mean([r["gen_ms"] for r in clean]),
               s["mean"], s["min"], s["max"], mean([r["prefill_tok_per_s"] for r in clean])))
    print("-" * 118)
    print("tok/s = (completion_tokens - 1) / (gen_ms / 1000); prefill = prompt_tokens / (ttft_ms / 1000)")
    flagged = [r for r in runs if r["flags"]]
    if flagged:
        print("left out of the summary (%d run(s)):" % len(flagged))
        for r in flagged:
            extra = ""
            if "loop" in r["flags"]:
                extra = " period %s chars" % r.get("loop_period")
            print("  %-20s run %d  %s%s  %.1f tok/s" % (r["task"], r["run"], ",".join(r["flags"]), extra, r["tok_per_s"]))
    else:
        print("no run was flagged")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the generation speed of an OpenAI-compatible chat server.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible base URL, without /chat/completions (default http://127.0.0.1:8000/v1)")
    ap.add_argument("--engine", default="yasakani", help="engine name for the result file and the comparison table (default yasakani)")
    ap.add_argument("--model", default="qwen3.8-27b", help="model id sent in the request (default qwen3.8-27b)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                    help="sent as Authorization: Bearer (default $OPENAI_API_KEY)")
    ap.add_argument("--tasks", default="all", help="comma list of task names, or all (default all)")
    ap.add_argument("--runs", type=int, default=2, help="runs per task (default 2)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="answer limit of every run (default 4096)")
    ap.add_argument("--temperature", type=float, default=1.0, help="default 1.0; 0 for greedy")
    ap.add_argument("--top-p", type=float, default=0.95, help="default 0.95")
    ap.add_argument("--top-k", type=int, default=None, help="sent only when given (not every server accepts it)")
    ap.add_argument("--seed", type=int, default=None, help="sent only when given; run N uses seed + N")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request (default 1800)")
    ap.add_argument("--warmup", type=int, default=1, help="short requests before measuring, not recorded (default 1)")
    ap.add_argument("--extra", default=None,
                    help='JSON object merged into every request body, e.g. {"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--out", default=None, help="result file (default results/<engine>_<time>.json)")
    args = ap.parse_args(argv)

    files = prompt_files()
    if not files:
        raise SystemExit("no prompt files in %s" % PROMPT_DIR)
    if args.tasks.strip() == "all":
        tasks = list(files)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in tasks if t not in files]
        if missing:
            raise SystemExit("unknown task(s): %s\navailable: %s" % (", ".join(missing), ", ".join(files)))
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    prompts = {}
    fingerprints = {}
    for task in tasks:
        with open(files[task], "rb") as fh:
            raw = fh.read()
        prompts[task] = raw.decode("utf-8")
        fingerprints[task] = hashlib.sha256(raw).hexdigest()

    body_base = dict(model=args.model, max_tokens=args.max_tokens, stream=True, stream_options={"include_usage": True})
    if args.temperature is not None:
        body_base["temperature"] = args.temperature
    if args.top_p is not None:
        body_base["top_p"] = args.top_p
    if args.top_k is not None:
        body_base["top_k"] = args.top_k
    if args.extra:
        try:
            body_base.update(json.loads(args.extra))
        except ValueError as ex:
            raise SystemExit("--extra is not valid JSON: %s" % ex)

    url = args.base_url.rstrip("/") + "/chat/completions"
    gpu = probe_gpu()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print("engine %s  model %s" % (args.engine, args.model))
    print("target %s" % url)
    print("gpu    %s" % (gpu["name"] or "unknown"))
    print("tasks  %d, runs %d each, max_tokens %d" % (len(tasks), args.runs, args.max_tokens))

    for i in range(max(0, args.warmup)):
        warm = dict(body_base, max_tokens=min(16, args.max_tokens))
        run_once(url, args.api_key, warm, "Reply with the single word: ready.", args.timeout)
        print("warm-up %d done" % (i + 1))

    runs = []
    for task in tasks:
        for run in range(args.runs):
            body = dict(body_base)
            if args.seed is not None:
                body["seed"] = args.seed + run
            entry = run_once(url, args.api_key, body, prompts[task], args.timeout)
            entry = dict(task=task, run=run, **entry)
            runs.append(entry)
            mark = ("  [" + ",".join(entry["flags"]) + "]") if entry["flags"] else ""
            print("  %-20s run %d  %5d tok  ttft %7.1f ms  gen %8.1f ms  %8.1f tok/s%s" %
                  (task, run, entry["completion_tokens"], entry["ttft_ms"], entry["gen_ms"], entry["tok_per_s"], mark))
            sys.stdout.flush()

    summary = summarise(runs, tasks)
    result = dict(
        schema_version=SCHEMA_VERSION,
        engine=args.engine,
        model=args.model,
        base_url=args.base_url,
        started_at=started_at,
        max_tokens=args.max_tokens,
        settings=dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed, runs=args.runs),
        prompts=fingerprints,
        tasks=tasks,
        gpu=gpu,
        runs=runs,
        summary=summary,
    )

    out = args.out
    if not out:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.engine) or "engine"
        out = os.path.join(RESULT_DIR, "%s_%s.json" % (safe, stamp))
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print_table(tasks, summary, runs)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
