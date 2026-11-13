#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Write a tiny Qwen3-shaped GGUF plus a JSON sidecar of expected values.

This is the fixture for tests/gguf/gguf_container_test.cpp: a real GGUF v3 file
small enough to generate at build time, shaped so that *every* tensor the legacy
Model::load asks for is present and the quantiser can actually encode it.

Why the shape is not hidden=64
------------------------------
The brief for this fixture asked for hidden 64 / ffn 128. That shape cannot be
quantised: src/gguf/gguf_quant.cpp:606 rejects any matrix whose fan-in is not a
multiple of 256 (kSeqCols, the trellis sequence width), and every layer weight's
fan-in is either hidden or ffn. 256 is therefore the smallest legal hidden, and
it is what this script emits.

Sizes (BF16 weights, F32 norms/biases, 32-byte alignment):

    --layers 1   hidden 256 ffn 256 vocab 16   802,816 B   (784 KiB, < 1 MB)
    --layers 2   same shape                  1,589,248 B   (1.52 MiB)

A 256x256 BF16 matrix is 128 KiB and one transformer layer needs five of them
(q, o, gate, up, down), so 2 layers cannot fit in 1 MB at BF16. The default is
2 layers (as requested); pass --layers 1 for a sub-1-MB file.

Head structure follows the brief: 4 heads, 2 kv heads, head_dim 64, so
heads*head_dim == hidden == 256 (the residual width) and kv_heads*head_dim == 128.

lm_head is UNTIED: output.weight is written as its own tensor. Tied embeddings
are supported by the container (--variant tie) but need the embedding payload
materialised twice, once row-major for embed_tokens and once warp-interleaved
for lm_head, so the default fixture keeps them separate.

Variants:
    good      the complete, loadable model                       (default)
    badshape  hidden 96: not a multiple of 256, must be rejected
    missing   good minus blk.<last>.ffn_down.weight, must be rejected
    tie       good with no output.weight (tied embeddings)
    nobias    good with no attn_q/k/v.bias (the container synthesises zeros)

Usage:
    make_tiny_gguf.py --out tiny.gguf
    make_tiny_gguf.py --out tiny1.gguf --layers 1
"""

import argparse
import json
import math
import os
import struct
import sys

# --- GGUF constants ---------------------------------------------------------
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY = 8, 9
F32, BF16 = 0, 30
ALIGNMENT = 32


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def f32(v):
    return struct.pack("<f", float(v))


def gguf_str(s):
    b = s.encode("utf-8")
    return u64(len(b)) + b


def bf16_bytes(v):
    """Round-to-nearest-even float32 -> bfloat16."""
    bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    # Round to nearest even on the 16 discarded bits.
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def bf16_to_f32(raw):
    bits = struct.unpack("<H", raw)[0] << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# --- deterministic weight values -------------------------------------------
def weight_values(n, seed):
    """Deterministic N(0, 0.02) weights, like a real transformer's.

    The distribution matters for the round-trip test: the trellis error is set by
    the codeword spacing over the *pair* distribution, and a uniform [-1, 1]
    matrix is a much harder case than the Gaussian real weights are. Measured on
    256x256 at bits=6 with 4 refinement rounds: Gaussian RMS 5.88, uniform RMS
    8.31. Gaussian is both realistic and the case docs/GGUF_NOTES.md tabulates.
    """
    out = []
    state = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    def next_u64():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return state

    for _ in range(n):
        # Box-Muller from two uniforms; the eps keeps log() finite.
        u1 = ((next_u64() >> 11) + 0.5) / float(1 << 53)
        u2 = (next_u64() >> 11) / float(1 << 53)
        out.append(0.02 * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    return out


def norm_values(n, seed):
    """Exactly representable in f16: multiples of 1/16 in [-1, 1]."""
    return [(((i * 7 + seed * 3) % 33) - 16) * 0.0625 for i in range(n)]


def bias_values(n, seed):
    """Exactly representable in f16: multiples of 1/32 in [-0.5, 0.5]."""
    return [(((i * 5 + seed * 2) % 17) - 8) * 0.03125 for i in range(n)]


# --- GGUF writer ------------------------------------------------------------
def write_gguf(path, kv, tensors):
    buf = bytearray()
    buf += b"GGUF"
    buf += u32(3)
    buf += u64(len(tensors))
    buf += u64(len(kv))
    for key, vtype, value in kv:
        buf += gguf_str(key)
        buf += u32(vtype)
        if vtype == T_UINT32:
            buf += u32(value)
        elif vtype == T_INT32:
            buf += struct.pack("<i", value)
        elif vtype == T_FLOAT32:
            buf += f32(value)
        elif vtype == T_BOOL:
            buf += struct.pack("<B", 1 if value else 0)
        elif vtype == T_STRING:
            buf += gguf_str(value)
        else:
            raise ValueError("bad metadata type %r" % (vtype,))
    for t in tensors:
        buf += gguf_str(t["name"])
        buf += u32(len(t["shape"]))
        for d in t["shape"]:
            buf += u64(d)
        buf += u32(t["type"])
        buf += u64(t["offset"])
    buf += b"\x00" * ((-len(buf)) % ALIGNMENT)
    data_base = len(buf)
    for t in tensors:
        assert len(buf) - data_base == t["offset"], \
            "offset mismatch for %s: %d vs %d" % (
                t["name"], len(buf) - data_base, t["offset"])
        buf += t["payload"]
        buf += b"\x00" * ((-len(buf)) % ALIGNMENT)
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return data_base


class Builder:
    def __init__(self):
        self.tensors = []
        self.vectors = []

    def matrix(self, engine, gguf, cols, rows, seed):
        vals = weight_values(cols * rows, seed)
        payload = b"".join(bf16_bytes(v) for v in vals)
        self.tensors.append({"name": gguf, "shape": [cols, rows], "type": BF16,
                             "payload": payload})
        self.tensors[-1]["engine"] = engine
        self.tensors[-1]["rows"] = rows
        self.tensors[-1]["cols"] = cols

    def vector(self, engine, gguf, values):
        payload = b"".join(f32(v) for v in values)
        self.tensors.append({"name": gguf, "shape": [len(values)], "type": F32,
                             "payload": payload})
        self.vectors.append({"name": engine, "gguf": gguf, "n": len(values),
                             "values": [float(v) for v in values]})

    def finish(self, path):
        offset = 0
        for t in self.tensors:
            t["offset"] = offset
            offset += len(t["payload"])
            offset += (-offset) % ALIGNMENT
        return write_gguf(path, self.metadata(), self.tensors)

    def metadata(self):
        p = "qwen3."
        return [
            ("general.architecture", T_STRING, "qwen3"),
            ("general.name", T_STRING, "yskn tiny gguf container fixture"),
            ("general.alignment", T_UINT32, ALIGNMENT),
            ("general.quantization_version", T_UINT32, 2),
            (p + "block_count", T_UINT32, self.layers),
            (p + "embedding_length", T_UINT32, self.hidden),
            (p + "feed_forward_length", T_UINT32, self.ffn),
            (p + "attention.head_count", T_UINT32, self.heads),
            (p + "attention.head_count_kv", T_UINT32, self.kv_heads),
            (p + "attention.key_length", T_UINT32, self.head_dim),
            (p + "attention.layer_norm_rms_epsilon", T_FLOAT32, self.eps),
            (p + "rope.freq_base", T_FLOAT32, self.rope_theta),
            (p + "rope.dimension_count", T_UINT32, self.head_dim),
            (p + "context_length", T_UINT32, 512),
            (p + "vocab_size", T_UINT32, self.vocab),
        ]


def build(args):
    b = Builder()
    b.layers = args.layers
    b.hidden = args.hidden
    b.ffn = args.ffn
    b.heads = args.heads
    b.kv_heads = args.kv_heads
    b.head_dim = args.head_dim
    b.vocab = args.vocab
    b.eps = 1e-6
    b.rope_theta = 1000000.0

    h, f, v = args.hidden, args.ffn, args.vocab
    qd, kd = args.heads * args.head_dim, args.kv_heads * args.head_dim

    tie = args.variant == "tie"
    want_bias = args.variant != "nobias"

    # Globals.
    b.matrix("embed_tokens", "token_embd.weight", h, v, 3)
    b.vector("norm", "output_norm.weight", norm_values(h, 5))
    if not tie:
        b.matrix("lm_head", "output.weight", h, v, 7)

    for l in range(args.layers):
        p = "blk.%d." % l
        e = "layers.%d." % l
        b.vector(e + "input_layernorm", p + "attn_norm.weight", norm_values(h, 11 + l))
        b.vector(e + "post_attention_layernorm", p + "ffn_norm.weight", norm_values(h, 13 + l))
        b.matrix(e + "q_proj", p + "attn_q.weight", h, qd, 17 + l)
        b.matrix(e + "k_proj", p + "attn_k.weight", h, kd, 19 + l)
        b.matrix(e + "v_proj", p + "attn_v.weight", h, kd, 23 + l)
        b.matrix(e + "o_proj", p + "attn_output.weight", qd, h, 29 + l)
        if want_bias:
            b.vector(e + "q_proj.bias", p + "attn_q.bias", bias_values(qd, 31 + l))
            b.vector(e + "k_proj.bias", p + "attn_k.bias", bias_values(kd, 37 + l))
            b.vector(e + "v_proj.bias", p + "attn_v.bias", bias_values(kd, 41 + l))
        b.matrix(e + "gate_proj", p + "ffn_gate.weight", h, f, 43 + l)
        b.matrix(e + "up_proj", p + "ffn_up.weight", h, f, 47 + l)
        if args.variant == "missing" and l == args.layers - 1:
            pass  # deliberately omit the last layer's down_proj
        else:
            b.matrix(e + "down_proj", p + "ffn_down.weight", f, h, 53 + l)

    return b


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2, dest="kv_heads")
    ap.add_argument("--head-dim", type=int, default=64, dest="head_dim")
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--variant", default="good",
                    choices=("good", "badshape", "missing", "tie", "nobias"))
    args = ap.parse_args(argv)

    if args.variant == "badshape":
        # 96 is a multiple of 32 but not of 256: the quantiser's sequence width.
        args.hidden = 96

    b = build(args)
    data_base = b.finish(args.out)

    sidecar = {
        "model": {
            "arch": "qwen3",
            "layers": args.layers,
            "hidden": args.hidden,
            "ffn": args.ffn,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "vocab": args.vocab,
            "eps": b.eps,
            "rope_theta": b.rope_theta,
            "tie_embeddings": args.variant == "tie",
        },
        "variant": args.variant,
        "data_base": data_base,
        "file_size": os.path.getsize(args.out),
        "tensors": [{"name": t["engine"], "gguf": t["name"],
                     "rows": t["rows"], "cols": t["cols"]}
                    for t in b.tensors if "engine" in t],
        "vectors": b.vectors,
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(sidecar, fh, indent=1, sort_keys=True)

    print("wrote %s (%d tensors, %d vectors, variant %s, %d bytes, data base %d)" % (
        args.out, len(b.tensors), len(b.vectors), args.variant,
        sidecar["file_size"], data_base))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))