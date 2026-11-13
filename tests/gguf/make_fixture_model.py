#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Generate a small Qwen3-shaped GGUF fixture for the adapter test.

The container writer and the quantized-block builders are imported from
make_fixture.py so the two fixtures cannot drift apart: this script only
decides *which* tensors and metadata keys exist.

Shapes (deliberately small, all powers of two so Q8_0 blocks divide evenly):

    layers 2, hidden 64, ffn 128, heads 4, kv heads 2, head_dim 16, vocab 128

Variants:
    good     -- a complete, valid model
    broken   -- same, but blk.1.ffn_down.weight has the wrong fan-in
    unknown  -- good, plus one tensor the mapper does not understand

Usage:
    make_fixture_model.py --out gguf_model.gguf --variant good
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fixture as mf  # noqa: E402  (path set up above)

# --- the shape every variant shares ----------------------------------------
LAYERS = 2
HIDDEN = 64
FFN = 128
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 128
EPS = 1e-6
ROPE_THETA = 1000000.0
CONTEXT = 4096
SLIDING = 4096
ALIGNMENT = 32

Q_DIM = HEADS * HEAD_DIM      # 64
KV_DIM = KV_HEADS * HEAD_DIM  # 32


def values(n, seed):
    """Deterministic, exactly-representable test weights."""
    return [((i * seed) % 29 - 14) * 0.25 for i in range(n)]


def q8(name, cols, rows, seed):
    """A Q8_0 matrix in GGUF order: shape is [cols, rows]."""
    payload, expected = mf.build_q8_0(values(cols * rows, seed))
    return {"name": name, "shape": [cols, rows], "type": mf.Q8_0, "payload": payload,
            "expected": expected}


def f32(name, n, seed):
    payload, expected = mf.build_f32(values(n, seed))
    return {"name": name, "shape": [n], "type": mf.F32, "payload": payload,
            "expected": expected}


def build_tensors(variant):
    t = []
    # Global tensors. output.weight is present, so embeddings are not tied.
    t.append(q8("token_embd.weight", HIDDEN, VOCAB, 3))
    t.append(f32("output_norm.weight", HIDDEN, 5))
    t.append(q8("output.weight", HIDDEN, VOCAB, 7))

    for l in range(LAYERS):
        p = "blk.%d." % l
        t.append(f32(p + "attn_norm.weight", HIDDEN, 11 + l))
        t.append(f32(p + "ffn_norm.weight", HIDDEN, 13 + l))
        t.append(q8(p + "attn_q.weight", HIDDEN, Q_DIM, 17 + l))
        t.append(q8(p + "attn_k.weight", HIDDEN, KV_DIM, 19 + l))
        t.append(q8(p + "attn_v.weight", HIDDEN, KV_DIM, 23 + l))
        t.append(q8(p + "attn_output.weight", Q_DIM, HIDDEN, 29 + l))
        # Qwen3 per-head QK norm: one gain per head_dim, F32 (16 is not a
        # whole number of 32-element Q8_0 blocks).
        t.append(f32(p + "attn_q_norm.weight", HEAD_DIM, 31 + l))
        t.append(f32(p + "attn_k_norm.weight", HEAD_DIM, 37 + l))
        t.append(q8(p + "ffn_gate.weight", HIDDEN, FFN, 41 + l))
        t.append(q8(p + "ffn_up.weight", HIDDEN, FFN, 43 + l))
        down_cols = FFN
        if variant == "broken" and l == 1:
            down_cols = HIDDEN  # the deliberate shape error
        t.append(q8(p + "ffn_down.weight", down_cols, HIDDEN, 47 + l))

    if variant == "unknown":
        t.append(q8("blk.0.attn_rot.weight", HIDDEN, HIDDEN, 53))
    return t


def build_metadata():
    kv = [
        ("general.architecture", mf.T_STRING, "qwen3"),
        ("general.name", mf.T_STRING, "yskn gguf model fixture"),
        ("general.alignment", mf.T_UINT32, ALIGNMENT),
        ("general.quantization_version", mf.T_UINT32, 2),
        ("qwen3.block_count", mf.T_UINT32, LAYERS),
        ("qwen3.embedding_length", mf.T_UINT32, HIDDEN),
        ("qwen3.feed_forward_length", mf.T_UINT32, FFN),
        ("qwen3.attention.head_count", mf.T_UINT32, HEADS),
        ("qwen3.attention.head_count_kv", mf.T_UINT32, KV_HEADS),
        ("qwen3.attention.key_length", mf.T_UINT32, HEAD_DIM),
        ("qwen3.attention.layer_norm_rms_epsilon", mf.T_FLOAT32, EPS),
        ("qwen3.attention.sliding_window", mf.T_UINT32, SLIDING),
        ("qwen3.rope.freq_base", mf.T_FLOAT32, ROPE_THETA),
        ("qwen3.rope.dimension_count", mf.T_UINT32, HEAD_DIM),
        ("qwen3.context_length", mf.T_UINT32, CONTEXT),
        ("qwen3.vocab_size", mf.T_UINT32, VOCAB),
    ]
    return kv


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--variant", default="good", choices=("good", "broken", "unknown"))
    args = ap.parse_args(argv)

    tensors = build_tensors(args.variant)

    # Lay the payloads out exactly the way write_gguf() will emit them.
    offset = 0
    for t in tensors:
        t["offset"] = offset
        offset += len(t["payload"])
        offset += (-offset) % ALIGNMENT

    data_base = mf.write_gguf(args.out, 3, ALIGNMENT, build_metadata(), tensors)
    print("wrote %s (%d tensors, variant %s, data base %d)" % (
        args.out, len(tensors), args.variant, data_base))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
