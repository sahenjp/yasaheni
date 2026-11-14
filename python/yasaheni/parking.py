# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Parked contexts for the HTTP server: pinned host memory, a byte budget, and least-recently-used eviction.

A handle is parked when another request is about to overwrite the live state.  The next request for the same conversation resumes
the handle (by conversation id when one is sent, otherwise by the longest token-id prefix), so the engine does not prefill the
common part again.  Nothing is written anywhere: the state lives in pinned host memory and disappears when the handle is freed.

The budget bounds the handles in this store, not the one spare pinned block the engine keeps after a handle is freed (the largest
block it has seen, used so the next park does not pay ``cudaMallocHost`` again).  Peak pinned memory is therefore at most the budget
plus one handle; set ``YSKN_PARK_REUSE=0`` to free every released block instead.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Optional, Sequence

from ._native import YasakaniError
from .engine import Engine, Parked

DEFAULT_PARK_MB = 2048


def park_budget_bytes() -> int:
    """The pinned host memory budget in bytes, from ``YSKN_PARK_MB`` (megabytes, default 2048)."""
    try:
        mb = int(os.environ.get("YSKN_PARK_MB") or DEFAULT_PARK_MB)
    except ValueError:
        mb = DEFAULT_PARK_MB
    return max(0, mb) * 1024 * 1024


class ParkStore:
    """Handles keyed by conversation id or token ids, with LRU eviction by the pinned bytes they hold."""

    def __init__(self, engine: Engine, budget_bytes: int):
        self.engine = engine
        self.budget = int(budget_bytes)
        self._entries: "OrderedDict[tuple, Parked]" = OrderedDict()
        self._used = 0
        self._live_cid: Optional[str] = None

    # ---- introspection ---------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def used(self) -> int:
        return self._used

    # ---- the live state's conversation -----------------------------------------------------------------------------------------
    def note_live(self, conversation_id: Optional[str]) -> None:
        """Remember which conversation produced the live state (it is parked under that key when it is overwritten)."""
        self._live_cid = conversation_id

    def keep_live(self) -> Optional[Parked]:
        """Park the live state under the conversation remembered by ``note_live``, or under its token ids."""
        h = self.keep(self._live_cid)
        self._live_cid = None
        return h

    # ---- lookup and storage ----------------------------------------------------------------------------------------------------
    def find(self, prompt: Sequence[int], conversation_id: Optional[str] = None) -> Optional[Parked]:
        """The handle that best continues ``prompt``: an exact conversation id first, then the longest id prefix.  A handle that
        diverges from the prompt is never returned."""
        p = list(prompt)
        if conversation_id:
            key = ("c", conversation_id)
            h = self._entries.get(key)
            if h is not None and 0 < len(h.token_ids) < len(p) and p[:len(h.token_ids)] == h.token_ids:
                self._entries.move_to_end(key)
                return h
        best: Optional[Parked] = None
        best_key = None
        for key, h in self._entries.items():
            n = len(h.token_ids)
            if 0 < n < len(p) and n > (len(best.token_ids) if best is not None else 0) and p[:n] == h.token_ids:
                best, best_key = h, key
        if best is not None:
            self._entries.move_to_end(best_key)
        return best

    def keep(self, conversation_id: Optional[str] = None) -> Optional[Parked]:
        """Park the live state.  A handle that does not fit the budget is not parked (one log line, the request goes on)."""
        try:
            ids = self.engine.live_ids()
        except YasakaniError:
            return None
        if not ids:
            return None
        key = ("c", conversation_id) if conversation_id else ("i", tuple(ids))
        old = self._entries.get(key)
        if old is not None and old.token_ids == ids:
            self._entries.move_to_end(key)
            return old
        try:
            size = self.engine.park_size(len(ids))
        except YasakaniError:
            return None
        if size > self.budget:
            self._log("a context of %.1f MB does not fit into the pinned memory budget of %.1f MB; it is not parked"
                      % (size / 1048576.0, self.budget / 1048576.0))
            return None
        need = self._used - (old.nbytes if old is not None else 0) + size
        while need > self.budget and self._entries:
            victim = next((k for k in self._entries if k != key), None)
            if victim is None:
                break
            self._used -= self._entries[victim].nbytes
            self._entries.pop(victim).close()
            need = self._used - (old.nbytes if old is not None else 0) + size
        try:
            h = self.engine.park(ids)
        except YasakaniError as ex:
            self._log("could not park the context: %s" % ex)
            return None
        if old is not None:
            self._used -= old.nbytes
            self._entries.pop(key, None)
            old.close()
        self._entries[key] = h
        self._used += h.nbytes
        return h

    def clear(self) -> None:
        for h in self._entries.values():
            h.close()
        self._entries.clear()
        self._used = 0
        self._live_cid = None

    @staticmethod
    def _log(msg: str) -> None:
        print("park: " + msg, file=sys.stderr, flush=True)
