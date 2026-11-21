# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The default runtime: environment settings, speculation settings and the model package.

The values below are the configuration the published numbers were measured with.  Change them only to experiment: a different
setting is a different engine configuration and its output is not covered by the published checks.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ._native import YasakaniError

# ---- Qwen3.8 ids used by the engine settings --------------------------------------------------------------------------------------
EOS_IDS = (248046, 248044)         # <|im_end|>, <|endoftext|>
IM_START = 248045                  # <|im_start|>: the prompt snapshot is taken right after its last occurrence
VOCAB_SIZE = 248320
HIDDEN = 5120

# ---- environment: read by the engine library when the model is opened -------------------------------------------------------------
# Value settings.  The two path settings (YSKN_BDR_SHEAD, YSKN_TB_PROFILE) come from the model package.
DEFAULT_ENV: Dict[str, str] = {
    "YSKN_BDR_SHEAD_M": "512",
    "YSKN_Q35_G6": "1",
    "YSKN_Q35_KV8": "1",
    "YSKN_Q35_KV_FP16": "0",
    "YSKN_Q35_LAYER_OVERLAP": "1",
    "YSKN_Q35_LAYER_ROWS": "4096",
    "YSKN_Q35_PFALT": "1",
    "YSKN_Q35_PMAX": "1024",
    "YSKN_Q35_SHARED_WEIGHTS": "1",
    "YSKN_Q35_ATX": "1",
    "YSKN_TB_CAP": "1",
    "YSKN_TB_VIT2": "1",
    "YSKN_ATX_PK3_PLUT": "1",
    "YSKN_ATX_PKV": "3",
    "YSKN_ATX_VSP": "7",
}
PATH_ENV = ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")
# Process settings that are not part of the model configuration: a profile keeps a value set in the shell instead of removing it.
PASS_ENV = ("YSKN_LNO_CFG", "YSKN_LNO_CFG_TAB", "YSKN_BDR_CFG", "YSKN_G6_TILEGRID")

# ---- tuning-only engine settings ---------------------------------------------------------------------------------------------------
# Read by the engine when a model is opened, but deliberately not part of DEFAULT_ENV: the published numbers were measured without them.
# They are in PASS_ENV, so a value set in the shell survives RuntimeProfile.apply and reaches the engine through the command line
# and the server.
#
# The engine reads the LNORD settings once per process: a profile value therefore decides the table only for the first engine
# opened in that process, and a later engine keeps the value the first one read.
#   YSKN_LNO_CFG      force one LNORD launch configuration (0..2) for every verify-batch size M; it wins over the per-M table.
#                     Values outside 0..2 are ignored with one warning line.
#   YSKN_LNO_CFG_TAB  comma-separated overrides for M=0..8, e.g. "0,1,1,1,1,1,1,1,1". Entries left out keep the device default:
#                     {0,1,1,1,1,1,1,1,1} on compute capability 8.6 (measured on the RTX 3090), {0,1,2,1,2,2,2,2,2} everywhere else,
#                     including the other 8.x parts, which are unmeasured.
#   YSKN_BDR_CFG      the same forced configuration for the drafter head's LNORD launches; values outside 0..2 are also ignored.
# A malformed YSKN_LNO_CFG_TAB is reported once at engine open and ignored.
#
# The q35_g6 setting is read on every launch, so one process can change it between calls.
#   YSKN_G6_TILEGRID  the q35_g6 prefill launch grid: 0 is the split-K grid, 1 is one block per output tile. Unset means the
#                     device default, which is the tile grid on compute capability 8.6 and split-K everywhere else.

# ---- speculation settings: the 44 integers of the generation call, in the order the engine reads them ----------------------------
SPEC_FIELDS = ("gamma", "use_d0", "use_d1", "d0_nmin", "d0_nmax", "d0_priority_n", "extend", "d1_rejected", "auto_gamma", "use_graph",
               "use_d3", "d3_steps", "d1_bigram", "d1_static", "d3_extend", "d3_lazy", "d3_gate", "gamma_long", "long_n", "d1_trigram",
               "eos1", "eos2", "long_d0", "sel", "cost_tok_us", "cost_row_us", "cost_nd_us", "nd_auto", "prmode", "cost_row2_us",
               "lfrow", "lfrow_q_pm", "mark_tok", "cost_tok_k", "cost_row2_k", "cost_nd_k", "bdraft", "bdr_long_n", "bdr_adapt",
               "bdr_probe", "bdr_d0_len", "prefill_head", "snap", "snap_tok")
assert len(SPEC_FIELDS) == 44

DEFAULT_SPEC: Dict[str, int] = dict(
    gamma=7, use_d0=1, use_d1=1, d0_nmin=2, d0_nmax=8, d0_priority_n=3, extend=1, d1_rejected=1, auto_gamma=0, use_graph=1,
    use_d3=0, d3_steps=0, d1_bigram=1, d1_static=0, d3_extend=0, d3_lazy=0, d3_gate=0, gamma_long=15, long_n=6, d1_trigram=1,
    eos1=EOS_IDS[0], eos2=EOS_IDS[1], long_d0=8, sel=0, cost_tok_us=0, cost_row_us=0, cost_nd_us=0, nd_auto=0, prmode=0,
    cost_row2_us=0, lfrow=0, lfrow_q_pm=300, mark_tok=-1, cost_tok_k=0, cost_row2_k=0, cost_nd_k=0, bdraft=1, bdr_long_n=4,
    bdr_adapt=1, bdr_probe=4, bdr_d0_len=15, prefill_head=1, snap=1, snap_tok=IM_START)
assert tuple(DEFAULT_SPEC) == SPEC_FIELDS

# settings that decide how the engine keeps its state between calls; not tunable per request
STATE_FIELDS = ("prefill_head", "snap", "snap_tok")

# statistics returned by the generation call (index in the 48 doubles)
STAT_INDEX = dict(passes=0, tokens=1, produced=30, eos=31, snap_mode=41, cached_tokens=42, bdr_rounds=37, bdr_d0=38)


def spec_array(spec: Dict[str, int]):
    import numpy as np
    unknown = set(spec) - set(SPEC_FIELDS)
    if unknown:
        raise ValueError("unknown speculation setting(s): " + ", ".join(sorted(unknown)))
    return np.array([int(spec[k]) for k in SPEC_FIELDS], np.int32)


@dataclass
class RuntimeProfile:
    """Environment variables applied before the model is opened."""
    env: Dict[str, str] = field(default_factory=dict)
    kept_env: List[str] = field(default_factory=list, init=False, repr=False, compare=False)

    @classmethod
    def default(cls, pack: "ModelPack", overrides: Optional[Dict[str, str]] = None) -> "RuntimeProfile":
        env = dict(DEFAULT_ENV)
        env["YSKN_BDR_SHEAD"] = str(pack.draft_head)
        env["YSKN_TB_PROFILE"] = str(pack.kv_profile)
        for k, v in (overrides or {}).items():
            if not k.startswith("YSKN_"):
                raise ValueError("engine settings start with YSKN_: " + k)
            env[k] = str(v)
        return cls(env)

    def apply(self, environ=None) -> List[str]:
        """Make the process environment this profile: engine settings left over from the shell are removed first (a stale
        setting would silently change the engine), except the tuning-only ``PASS_ENV`` settings, which are kept.  Returns the
        names that were removed; ``kept_env`` holds the kept tuning settings that came from the shell."""
        environ = os.environ if environ is None else environ
        kept = [k for k in PASS_ENV if k in environ and k not in self.env]
        removed = [k for k in list(environ) if k.startswith("YSKN_") and k not in self.env and k not in PASS_ENV]
        for k in removed:
            environ.pop(k)
        environ.update(self.env)
        self.kept_env = kept
        return removed


@dataclass
class ModelPack:
    """A model folder: weights, drafter, drafter head, KV profile, tokenizer files and the manifest that lists them.

    The manifest is ``yasaheni.json`` in this fork; the official YASAKANI folders call it ``yasakani.json``.  Either is read,
    with ``yasaheni.json`` taking precedence when both exist."""
    root: Path
    weights: Path
    draft: Path
    draft_head: Path
    kv_profile: Path
    max_ctx: int = 262144
    model_id: str = "qwen3.8-27b"
    sha256: Dict[str, str] = field(default_factory=dict)
    manifest: Optional[Path] = None

    MANIFESTS = ("yasaheni.json", "yasakani.json")
    DEFAULT_FILES = dict(weights="qwen38_27b_e3x.yskn", draft="bdraft2_k8.yskn", draft_head="bdraft2_k8.shead", kv_profile="kvq_u5.bin")

    @classmethod
    def load(cls, root) -> "ModelPack":
        root = Path(root)
        if not root.is_dir():
            raise YasakaniError("model folder not found: %s" % root)
        files = dict(cls.DEFAULT_FILES)
        meta = {}
        mf = next((root / n for n in cls.MANIFESTS if (root / n).is_file()), None)
        if mf is not None:
            meta = json.loads(mf.read_text(encoding="utf-8"))
            files.update(meta.get("files") or {})
        paths = {k: root / v for k, v in files.items() if k in cls.DEFAULT_FILES}
        missing = [str(p.name) for p in paths.values() if not p.is_file()]
        if missing:
            raise YasakaniError("model folder %s is missing: %s" % (root, ", ".join(missing)))
        for need in ("tokenizer.json", "tokenizer_config.json"):
            if not (root / need).is_file():
                raise YasakaniError("model folder %s is missing %s" % (root, need))
        return cls(root=root, weights=paths["weights"], draft=paths["draft"], draft_head=paths["draft_head"],
                   kv_profile=paths["kv_profile"], max_ctx=int(meta.get("max_ctx", 262144)),
                   model_id=str(meta.get("model_id", "qwen3.8-27b")), sha256=dict(meta.get("sha256") or {}), manifest=mf)

    def verify(self, progress=None) -> Dict[str, bool]:
        """Compare the sha256 of every listed file with the manifest that was read (reads ~15 GB: a minute or two)."""
        res = {}
        for name, want in sorted(self.sha256.items()):
            p = self.root / name
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            res[name] = h.hexdigest() == want
            if progress:
                progress(name, res[name])
        return res
