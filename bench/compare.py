#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Compare two or more bench/bench.py result files and render one table.

Rows are engines, columns are tasks, cells are tok/s.  The fastest engine of every column is set in bold.  Before anything is printed
every file is checked against bench/schema.json by hand (no jsonschema dependency) and the files are compared with each other: a
table that mixes answer limits or prompt sets is the main way these numbers lie, so such a set is refused with the reason.

usage:  python3 bench/compare.py bench/results/*.json
        python3 bench/compare.py bench/results/ --format tsv
        python3 bench/compare.py a.json b.json --no-exclude-flagged
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "schema.json")
SCHEMA_VERSION = 1
FLAG_VALUES = ("loop", "tool_calls", "no_usage", "empty")


class BadResult(Exception):
    """A result file that does not match the schema, or a set of files that cannot be compared."""


# ---- validation -------------------------------------------------------------------------------------------------------------------
def _need(obj, key, where, kinds, optional=False):
    if key not in obj:
        if optional:
            return None
        raise BadResult("%s: missing %r" % (where, key))
    value = obj[key]
    if value is None and type(None) in kinds:
        return None
    if isinstance(value, bool) and bool not in kinds:
        raise BadResult("%s.%s: expected %s, found a boolean" % (where, key, _names(kinds)))
    if not isinstance(value, kinds):
        raise BadResult("%s.%s: expected %s, found %s" % (where, key, _names(kinds), type(value).__name__))
    return value


def _names(kinds):
    return " or ".join({bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object",
                        type(None): "null"}[k] for k in kinds)


def validate(result, path="<result>") -> dict:
    """Check one result object against bench/schema.json by hand.  Returns it, raises BadResult with the first problem found."""
    if not isinstance(result, dict):
        raise BadResult("%s: the top level must be an object" % path)
    version = _need(result, "schema_version", path, (int,))
    if version != SCHEMA_VERSION:
        raise BadResult("%s: schema_version %r, this compare.py knows %d" % (path, version, SCHEMA_VERSION))
    for key in ("engine", "model", "base_url", "started_at"):
        _need(result, key, path, (str,))
    _need(result, "max_tokens", path, (int,))
    if result["max_tokens"] < 1:
        raise BadResult("%s: max_tokens must be at least 1" % path)

    settings = _need(result, "settings", path, (dict,), optional=True)
    if settings is not None:
        for key, kinds in (("temperature", (int, float, type(None))), ("top_p", (int, float, type(None))),
                           ("top_k", (int, type(None))), ("seed", (int, type(None))), ("runs", (int,))):
            _need(settings, key, path + ".settings", kinds, optional=True)

    prompts = _need(result, "prompts", path, (dict,))
    for task, digest in prompts.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise BadResult("%s.prompts.%s: expected a SHA-256 hex digest" % (path, task))

    tasks = _need(result, "tasks", path, (list,))
    if not tasks:
        raise BadResult("%s: tasks is empty" % path)
    for task in tasks:
        if not isinstance(task, str):
            raise BadResult("%s.tasks: every entry must be a string" % path)
        if task not in prompts:
            raise BadResult("%s: task %r is not in prompts" % (path, task))

    gpu = _need(result, "gpu", path, (dict,))
    _need(gpu, "name", path + ".gpu", (str,))

    runs = _need(result, "runs", path, (list,))
    if not runs:
        raise BadResult("%s: runs is empty" % path)
    for i, run in enumerate(runs):
        where = "%s.runs[%d]" % (path, i)
        if not isinstance(run, dict):
            raise BadResult("%s: must be an object" % where)
        task = _need(run, "task", where, (str,))
        if task not in tasks:
            raise BadResult("%s.task: %r is not in tasks" % (where, task))
        _need(run, "run", where, (int,))
        for key in ("prompt_tokens", "completion_tokens"):
            if _need(run, key, where, (int,)) < 0:
                raise BadResult("%s.%s: must not be negative" % (where, key))
        for key in ("ttft_ms", "gen_ms", "tok_per_s"):
            if _need(run, key, where, (int, float)) < 0:
                raise BadResult("%s.%s: must not be negative" % (where, key))
        _need(run, "finish_reason", where, (str, type(None)))
        flags = _need(run, "flags", where, (list,))
        for flag in flags:
            if flag not in FLAG_VALUES:
                raise BadResult("%s.flags: unknown flag %r (known: %s)" % (where, flag, ", ".join(FLAG_VALUES)))

    summary = _need(result, "summary", path, (dict,))
    per_task = _need(summary, "per_task", path + ".summary", (dict,))
    for task in tasks:
        if task not in per_task:
            raise BadResult("%s.summary.per_task: no entry for task %r" % (path, task))
        cell = per_task[task]
        where = "%s.summary.per_task.%s" % (path, task)
        if not isinstance(cell, dict):
            raise BadResult("%s: must be an object" % where)
        for key in ("mean", "min", "max"):
            if _need(cell, key, where, (int, float)) < 0:
                raise BadResult("%s.%s: must not be negative" % (where, key))
        for key in ("n_used", "n_excluded"):
            if _need(cell, key, where, (int,)) < 0:
                raise BadResult("%s.%s: must not be negative" % (where, key))
    return result


# ---- loading ----------------------------------------------------------------------------------------------------------------------
def expand(paths: list) -> list:
    """Turn files and directories into a sorted list of .json files."""
    out = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "*.json")))
            if not found:
                raise BadResult("%s: no .json file in this folder" % path)
            out.extend(found)
        elif os.path.isfile(path):
            out.append(path)
        else:
            raise BadResult("%s: no such file or folder" % path)
    seen = set()
    unique = []
    for path in out:
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load(paths: list) -> list:
    """Read and validate every file, then check that the set can be compared at all."""
    results = []
    for path in expand(paths):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as ex:
            raise BadResult("%s: cannot read: %s" % (path, ex))
        except ValueError as ex:
            raise BadResult("%s: not valid JSON: %s" % (path, ex))
        try:
            validate(data, path)
        except BadResult as ex:
            raise BadResult("%s" % ex)
        results.append((path, data))

    if len(results) < 2:
        raise BadResult("need at least two result files to compare, got %d" % len(results))

    limits = {data["max_tokens"] for _, data in results}
    if len(limits) > 1:
        detail = "\n".join("  %s: max_tokens %d" % (path, data["max_tokens"]) for path, data in results)
        raise BadResult("refusing to compare: the answer limits differ, so the runs are not the same test.\n" + detail)

    reference_path, reference = results[0]
    for path, data in results[1:]:
        if data["prompts"] != reference["prompts"]:
            only_a = sorted(set(reference["prompts"]) - set(data["prompts"]))
            only_b = sorted(set(data["prompts"]) - set(reference["prompts"]))
            differ = sorted(t for t in set(data["prompts"]) & set(reference["prompts"])
                            if data["prompts"][t] != reference["prompts"][t])
            lines = ["refusing to compare: the prompt sets differ, so the columns are not the same test.",
                     "  reference %s" % reference_path,
                     "  this file %s" % path]
            if only_a:
                lines.append("  only in the reference: " + ", ".join(only_a))
            if only_b:
                lines.append("  only in this file:   " + ", ".join(only_b))
            if differ:
                lines.append("  same name, different content: " + ", ".join(differ))
            raise BadResult("\n".join(lines))
    return results


# ---- the table --------------------------------------------------------------------------------------------------------------------
def cell_value(data: dict, task: str, exclude_flagged: bool):
    """(tok/s, n_used, n_excluded) for one engine and task; tok/s is None when no run is usable."""
    runs = [r for r in data["runs"] if r["task"] == task]
    if exclude_flagged:
        used = [r["tok_per_s"] for r in runs if not r["flags"]]
        excluded = len(runs) - len(used)
    else:
        used = [r["tok_per_s"] for r in runs]
        excluded = 0
    if not used:
        return None, 0, excluded
    return sum(used) / len(used), len(used), excluded


def build_rows(results: list, exclude_flagged: bool) -> tuple:
    """(tasks, rows) where a row is (label, {task: (value, n_used, n_excluded)})."""
    tasks = list(results[0][1]["tasks"])
    rows = []
    for path, data in results:
        label = data["engine"]
        if label in [r[0] for r in rows]:
            label = "%s (%s)" % (label, os.path.basename(path))
        rows.append((label, {task: cell_value(data, task, exclude_flagged) for task in tasks}))
    return tasks, rows


def fastest_per_task(tasks: list, rows: list) -> dict:
    best = {}
    for task in tasks:
        values = [row[1][task][0] for row in rows if row[1][task][0] is not None]
        best[task] = max(values) if values else None
    return best


def render_markdown(tasks: list, rows: list, best: dict) -> str:
    head = "| Engine | " + " | ".join(tasks) + " |"
    rule = "|---" * (len(tasks) + 1) + "|"
    lines = [head, rule]
    for label, cells in rows:
        out = []
        for task in tasks:
            value, n_used, n_excluded = cells[task]
            if value is None:
                out.append("—" if not n_excluded else "— (%d excl.)" % n_excluded)
                continue
            text = "%.1f" % value
            if best[task] is not None and abs(value - best[task]) < 1e-9 and sum(
                    1 for row in rows if row[1][task][0] is not None) > 1:
                text = "**%s**" % text
            if n_excluded:
                text += " (%d excl.)" % n_excluded
            out.append(text)
        lines.append("| %s | %s |" % (label, " | ".join(out)))
    return "\n".join(lines)


def render_tsv(tasks: list, rows: list) -> str:
    lines = ["engine\t" + "\t".join(tasks)]
    for label, cells in rows:
        out = []
        for task in tasks:
            value, n_used, n_excluded = cells[task]
            out.append("" if value is None else "%.1f" % value)
        lines.append(label + "\t" + "\t".join(out))
    return "\n".join(lines)


def render_json(tasks: list, rows: list, best: dict, results: list, exclude_flagged: bool) -> str:
    payload = dict(
        exclude_flagged=exclude_flagged,
        max_tokens=results[0][1]["max_tokens"],
        tasks=tasks,
        engines=[dict(
            engine=label,
            source=path,
            model=data["model"],
            base_url=data["base_url"],
            started_at=data["started_at"],
            gpu=data["gpu"].get("name", ""),
            tok_per_s={task: (None if cells[task][0] is None else round(cells[task][0], 3)) for task in tasks},
            n_used={task: cells[task][1] for task in tasks},
            n_excluded={task: cells[task][2] for task in tasks},
        ) for (label, cells), (path, data) in zip(rows, results)],
        fastest={task: (None if best[task] is None else round(best[task], 3)) for task in tasks},
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare bench.py result files and render one table.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("results", nargs="+", help="result JSON files, or folders holding them")
    ap.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown", help="output format (default markdown)")
    ap.add_argument("--exclude-flagged", dest="exclude_flagged", action="store_true", default=True,
                    help="drop runs flagged loop or tool_calls (default on)")
    ap.add_argument("--no-exclude-flagged", dest="exclude_flagged", action="store_false",
                    help="average every run in, flagged ones too")
    args = ap.parse_args(argv)

    try:
        results = load(args.results)
    except BadResult as ex:
        print("compare.py: %s" % ex, file=sys.stderr)
        return 2

    tasks, rows = build_rows(results, args.exclude_flagged)
    best = fastest_per_task(tasks, rows)
    if args.format == "markdown":
        print(render_markdown(tasks, rows, best))
    elif args.format == "tsv":
        print(render_tsv(tasks, rows))
    else:
        print(render_json(tasks, rows, best, results, args.exclude_flagged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
