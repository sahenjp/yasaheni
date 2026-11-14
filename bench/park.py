#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Time parking and resuming a context at 8K, 32K and 128K tokens.

Park copies the durable state (KV rows [0,pos) per head plus the linear S and hist arrays) into pinned host memory; resume copies it
back.  Neither lowers peak GPU memory, so the point of the numbers is the cost of a context switch, not free memory.

The first park of a size allocates a pinned block; later parks reuse the block of the handle that was just closed (the engine
keeps at most one spare).  The table reports the cold and the warm park separately, so the allocation cost is visible.  Pass
``--no-reuse`` to free every block and measure the allocation every time, and ``--debug`` to have the engine print the allocation,
the strided K/V copies and the linear-state copies separately.

usage:  python3 bench/park.py --model-dir /models/qwen3.8-27b
        python3 bench/park.py --model-dir /models/qwen3.8-27b --sizes 8192,32768 --repeat 5
        python3 bench/park.py --model-dir /models/qwen3.8-27b --no-reuse --debug
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from yasaheni.engine import Engine, Sampling          # noqa: E402
from yasaheni.runtime import VOCAB_SIZE, ModelPack, RuntimeProfile   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure park and resume at several context lengths.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model-dir", default=os.environ.get("YASAKANI_TEST_MODEL_DIR"),
                    help="model folder (default YASAKANI_TEST_MODEL_DIR)")
    ap.add_argument("--sizes", default="8192,32768,131072", help="context lengths in tokens (default 8192,32768,131072)")
    ap.add_argument("--repeat", type=int, default=3, help="park/resume pairs per length (default 3)")
    ap.add_argument("--max-ctx", type=int, default=None, help="context size to open with (default: the largest length plus 256)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-reuse", action="store_true",
                    help="free the pinned block of every released handle (the old numbers, allocation included in every park)")
    ap.add_argument("--debug", action="store_true",
                    help="print the engine's park breakdown: pinned allocation, strided K/V copies, linear-state copies")
    args = ap.parse_args(argv)
    if not args.model_dir:
        ap.error("--model-dir or YASAKANI_TEST_MODEL_DIR is required")
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    max_ctx = int(args.max_ctx or max(sizes) + 256)

    overrides = {}
    if args.no_reuse:
        overrides["YSKN_PARK_REUSE"] = "0"
    if args.debug:
        overrides["YSKN_PARK_DEBUG"] = "1"
    pack = ModelPack.load(Path(args.model_dir))
    profile = RuntimeProfile.default(pack, overrides) if overrides else None
    eng = Engine(pack, max_ctx=max_ctx, device=args.device, profile=profile)
    try:
        eng.set_sampling(Sampling(0.0))
        rng = np.random.default_rng(0)
        print("context tokens  handle MiB  park cold ms  park warm ms  resume ms  park GiB/s  resume GiB/s")
        for n in sizes:
            eng.reset()
            prompt = rng.integers(0, VOCAB_SIZE, size=n, dtype=np.int32).tolist()
            eng.generate(prompt, 2, stop_ids=None)                  # prefill so that pos == n + 1
            pos = eng.position()
            nbytes = eng.park_size(pos)
            park_ms, resume_ms = [], []
            for _ in range(max(1, args.repeat)):
                t0 = time.perf_counter()
                h = eng.park()
                park_ms.append((time.perf_counter() - t0) * 1000.0)
                t0 = time.perf_counter()
                eng.resume(h)
                resume_ms.append((time.perf_counter() - t0) * 1000.0)
                h.close()
            cold_ms = park_ms[0]
            warm_ms = None if args.no_reuse else (statistics.median(park_ms[1:]) if len(park_ms) > 1 else None)
            rate_ms = warm_ms if warm_ms is not None else statistics.median(park_ms)
            resume_med = statistics.median(resume_ms)
            gib = nbytes / (1 << 30)
            print("%14d %11.1f %14.2f %14s %11.2f %11.2f %12.2f" % (
                pos, nbytes / (1 << 20), cold_ms, "n/a" if warm_ms is None else "%.2f" % warm_ms, resume_med,
                gib / (rate_ms / 1000.0), gib / (resume_med / 1000.0)))
            print("  park   cold %.2f  min %.2f  med %.2f  max %.2f ms" % (cold_ms, min(park_ms),
                  statistics.median(park_ms), max(park_ms)))
            print("  resume min %.2f  med %.2f  max %.2f ms" % (min(resume_ms), resume_med, max(resume_ms)))
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
