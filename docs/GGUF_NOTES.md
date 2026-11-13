# GGUF to YSKN: what it takes to drop the container

This document records what was measured while building the GGUF path, so the numbers are reproducible rather than asserted.

## Where the code is

| File | Role |
|---|---|
| `src/gguf/gguf.hpp`, `gguf.cpp` | GGUF v2/v3 reader, 16 ggml types dequantized to float32 |
| `src/gguf/gguf_model.*` | config extraction, GGUF name to engine role mapping, validation, `describe()` |
| `src/gguf/gguf_quant.*` | rotation, codebook, Viterbi, warp interleave, round-trip oracle |
| `src/gguf/gguf_engine.*` | file-kind detection, `yskn_gguf_probe`, the supported/reason gate |
| `tests/gguf/` | 4 suites, 812 checks, no GPU needed |

## The quantization bridge, measured

`quantize_tensor` runs the pipeline from `docs/GGUF_DESIGN.md` §5.4: rotate, per-row scale, codebook, Viterbi, pack, interleave.

### Viterbi is optimal

Against a brute-force search over all 4096 states on a 256-column sequence:

```
brute-force optimal total cost = 0.001249  (rmse per value 0.002209)
viterbi_quantize cost        = 0.001249  (rmse per value 0.002209)
ratio = 1.000000  (1.0 means optimal)
```

### The trellis is the limit, not the encoder

In the reader a state `s'` can only be reached from states whose low `12 - bits` bits match. At `bits = K` only `2^K` of the 4096 codewords are reachable at each step, so the achievable error is bounded by the codeword spacing, not by the codebook size:

| bits | reachable per step | expected spacing | measured RMS |
|---|---|---|---|
| 4 | 16 | 63.5 | 10.6 |
| 6 | 64 | 31.8 | 5.5 |
| 8 | 256 | 15.9 | 2.9 |

(full scale is 127; RMS over a uniform pair distribution is `spacing / sqrt(12)`)

### Trellis-aware codebook refinement matters

Plain k-means over the pairs ignores that constraint. One Lloyd round over the states the encoder can actually reach closes most of the gap:

```
rows  pairs    plain-rms  refined-rms
   8    2048  bits=4  11.997     5.669
   8    2048  bits=8   3.013     1.306
 512  131072  bits=4  11.104    10.574
 512  131072  bits=8   3.174     2.940
```

More training data narrows the gap because a generalising codebook is already close to the trellis optimum; a codebook trained on too little data overfits and looks better on its own training set than it is.

### Two bugs the oracle caught

`verify_roundtrip` compares the decoded stream against the normalized matrix `B = A / tau`. Both of these were wrong first and the test found them:

1. The Viterbi cost compared `B` against the raw `int8` entries instead of `q * gamma`, which biases every codeword by `1/gamma`.
2. The oracle multiplied the decoded value by `tau`, comparing `A`-units to `B`-units.

A third: `rotate_matrix` is **not** an involution. `A = S_out · Hn · W · Hn · S_in`, and `S` does not commute with `Hn`, so applying it twice does not return `W`. The inverse mirrors the order: `W = S_in · Hn · A · Hn · S_out`.

## It runs

A stock GGUF file loads into the engine and produces tokens. Verified on the RTX 3070 with a 28-head, head_dim-128, hidden-3584, ffn-512, 1-layer synthetic model:

```
open: True
prefill rc: 0  tok: 11
logits rc: 0  finite: True  nonzero: True
decode rc: 0  toks: [8, 0, 2, 14]
```

No YSKN container is involved: GgufContainer dequantizes the BF16 weights, runs the quantization bridge, materializes the image, and Model::load uploads it through the legacy Engine.

### The gate is narrower than "the kernels are templated"

The default attention path is not templated. src/post_kernels.cu:47 writes q16 with a literal stride of 28 heads x 128 slots and src/attention_kv4q.cu:81 reads it back the same way. So the gate requires head_dim == 128 and heads == 28.

A head_dim 256 model was measured doing this:

```
hd256 open: True
prefill rc: 0  tok: 2147483647
logits rc: 0  finite: False  nonzero: True
vals: [nan, nan, nan, nan, nan, nan]
```

It loads, reports success, and returns NaN. That is why the gate refuses it rather than admitting it on the strength of the generic path, which is off by default.

## What still blocks the 27B model

The GGUF path runs through the legacy Engine, which is shape-parametric for `hidden`, `ffn`, `kv_heads` and `vocab`. It is not parametric for the attention shape: `heads` must be 28 and `head_dim` 128. So a 27B Qwen3 GGUF (24 heads, head_dim 256) is refused.

To run the 27B model from GGUF the engine needs one of:

1. **Template the legacy attention path** over `(heads, head_dim)`. The generic path already exists (`YSKN_ATTN_GENERIC=1`, `src/attention_gqa.cu:171-172`) and is instantiated for `(7,128,128)` and `(6,256,64)`; what is missing is making `qkv_post` and the decode attention honour the config instead of the literal 28 x 128 in `src/post_kernels.cu:47` and `src/attention_kv4q.cu:81`, then gating on the generic path being available.
2. **Teach `Engine35` to read GGUF.** It is the fast path and already handles 24 heads x 256, but it speaks a different tensor vocabulary (`layers.N.self_attn.q_proj`), needs `layer_types`, and hard-gates on the 27B shape at construction.

Option 1 is the shorter road and is what the next slice should do: it is one kernel generalisation, not a second engine.
