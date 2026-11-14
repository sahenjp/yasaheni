#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""tables_strict.py - the thinking-off speed table made again from the records of bench/20261108 with a stricter rule for
repetition, the same for every engine.

  python tools/speed/tables_strict.py [--out DIR] [--bench DIR]          (Python 3.8+, standard library only)

The records and the first tally are those of bench/20261108 (scripts/tables.py, which this program imports).  That tally left out
a run of the 12 published prompts as a repetition loop when the last 4,000 characters of its answer compress below 12 % with zlib.
An answer that repeats whole paragraphs, but not tightly enough for that test, stayed in.  This program also leaves out, for the
prose prompts (story and translation) and for every engine alike,

    a run in which the lines that occur more than once make up 20 % or more of its lines   (lines of 20 characters or more)

Everything else is as in tables.py: the same records, the same speed of a run, the same averages, the largest value of each column
for an engine measured with several settings.  The thinking-on tables do not leave out any run, so they do not change.

Writes SPEED_TABLES_STRICT.md and speed_tables_strict.json to --out (default: the folder of this file).
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "bench" / "20261108"            # --bench DIR on the command line names another copy of bench/20261108
if "--bench" in sys.argv:
    BENCH = Path(sys.argv[sys.argv.index("--bench") + 1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(BENCH / "scripts"))
import tables as T  # noqa: E402

PROSE = ("story", "translation")
MIN_LINE, SHARE = 20, 0.20
PUBLISHED_FLAG = T.flag


def repeated_share(text):
    """Share of the lines (of MIN_LINE characters or more) that occur more than once in the answer."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= MIN_LINE]
    if not lines:
        return 0.0
    count = collections.Counter(lines)
    return sum(c for c in count.values() if c > 1) / len(lines)


def strict_flag(text, cat):
    f = PUBLISHED_FLAG(text, cat)
    if f == "." and cat in PROSE and repeated_share(text) >= SHARE:
        return "L"
    return f


def per_prompt(n12):
    """fixture -> (mean speed of the runs that are kept, or None; flags of its runs)"""
    out = {}
    for fx in sorted(set(r["fixture"] for r in n12)):
        rows = [r for r in n12 if r["fixture"] == fx]
        kept = [r["decode_tok_s"] for r in rows if r["_flag"] in (".", "?")]
        out[fx] = (statistics.mean(kept) if kept else None, "".join(r["_flag"] for r in rows))
    return out


def build():
    T.flag = PUBLISHED_FLAG
    before = T.build()
    T.flag = strict_flag
    after = T.build()
    prompts = {"yasakani": {}, "engines": {}}
    for name, cfg in T.OURS:
        prompts["yasakani"][name] = per_prompt(T.setting("n12_ysk_" + cfg)[0])
    for eng, off, _ in T.ENGINES:
        best = {}
        for n in off:
            for fx, (v, flags) in per_prompt(T.setting(n)[0]).items():
                if v is not None and (fx not in best or v > best[fx]):
                    best[fx] = v
        prompts["engines"][eng] = best
    T.flag = PUBLISHED_FLAG
    return before, after, prompts


def markdown(before, after, prompts):
    f = lambda v: "-" if v is None else "{:,.0f}".format(v)
    cols = T.COLS
    key = "mean_clean"
    md = ["# Thinking off: the speed table with the stricter rule for repetition", "",
          "Output of `tools/speed/tables_strict.py` from the records of `bench/20261108`. The rule is in the docstring of the script: in",
          "addition to the first tally, a story or translation run is left out when lines that occur more than once make up %d %% or more of"
          % round(100 * SHARE),
          "its lines (lines of %d characters or more). The rule is the same for every engine. tok/s; conditions and limits as in the README of" % MIN_LINE,
          "`bench/`.", "",
          "| Engine | " + " | ".join(cols) + " |", "|---|" + "---|" * 7]
    md += ["| %s | %s |" % (n, " | ".join(f(v) for v in o["off"][key])) for n, o in after["yasakani"].items()]
    md += ["| %s | %s |" % (n, " | ".join(f(v) for v in e["off"][key])) for n, e in after["engines"].items()]
    best = T.colmax([e["off"][key] for e in after["engines"].values()])
    md += ["| (largest value of the other engines) | %s |" % " | ".join(f(v) for v in best)]
    for n, o in after["yasakani"].items():
        md += ["| %s / largest value of the others | %s |" % (n, " | ".join("%.2f" % (a / b) for a, b in zip(o["off"][key], best)))]
    md += ["", "## What changed against the first tally (`bench/20261108/tables/SPEED_TABLES.md`)", ""]
    changed = []
    for group in ("yasakani", "engines"):
        for n in after[group]:
            for c, a, b in zip(cols, before[group][n]["off"][key], after[group][n]["off"][key]):
                if a is not None and b is not None and round(a) != round(b):
                    changed.append("* %s, %s: %s -> %s" % (n, c, f(a), f(b)))
    md += changed or ["* nothing"]
    b0 = T.colmax([e["off"][key] for e in before["engines"].values()])
    md += ["* largest value of the other engines, Story: %s -> %s" % (f(b0[1]), f(best[1])), "",
           "## Runs left out by the added rule", ""]
    for n, o in after["yasakani"].items():
        extra = sorted(set(o["off"]["left_out"]) - set(before["yasakani"][n]["off"]["left_out"]))
        if extra:
            md.append("* %s: %s" % (n, ", ".join(extra)))
    for eng, e in after["engines"].items():
        for s, p in e["off"]["settings"].items():
            extra = sorted(set(p["left_out"]) - set(before["engines"][eng]["off"]["settings"][s]["left_out"]))
            if extra:
                md.append("* %s `%s`: %s" % (eng, s, ", ".join(extra)))
    md += ["", "## The 12 published prompts one by one (thinking off, average of the runs that are kept)", "",
           "`L` = left out as a repetition loop, `T` = left out because the answer is a tool call nobody asked for; two letters = the two runs.", "",
           "| Prompt | " + " | ".join(n for n, _ in T.OURS) + " | Fastest other engine | Fastest model / fastest other |", "|---|---|---|---|---|"]
    fixtures = sorted(prompts["yasakani"][T.OURS[0][0]])
    for fx in fixtures:
        cells = []
        for n, _ in T.OURS:
            v, flags = prompts["yasakani"][n][fx]
            cells.append(f(v) if v is not None else "left out (%s)" % flags)
        others = {eng: best_[fx] for eng, best_ in prompts["engines"].items() if fx in best_}
        top = max(others, key=others.get) if others else None
        v0 = prompts["yasakani"][T.OURS[0][0]][fx][0]
        ratio = "%.2f" % (v0 / others[top]) if (top and v0 is not None) else "-"
        md.append("| %s | %s | %s | %s |" % (fx.replace("scenario_", ""), " | ".join(cells), ("%s (%s)" % (f(others[top]), top)) if top else "-", ratio))
    md += ["", "The thinking-on tables leave out no run, so they are the ones of `bench/20261108/tables/SPEED_TABLES.md`.", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--bench", help="the folder bench/20261108 (default: the one of this repository)")
    a = ap.parse_args()
    before, after, prompts = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(before, after, prompts)
    data = dict(rule=dict(categories=list(PROSE), min_line_chars=MIN_LINE, repeated_line_share=SHARE), columns=T.COLS, strict=after,
                prompts=dict(yasakani={n: {fx: dict(mean=v, flags=fl) for fx, (v, fl) in d.items()} for n, d in prompts["yasakani"].items()},
                             fastest_setting_of_each_engine=prompts["engines"]))
    for name, text in (("speed_tables_strict.json", json.dumps(data, ensure_ascii=False, indent=1) + "\n"), ("SPEED_TABLES_STRICT.md", md)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
