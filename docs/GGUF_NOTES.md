# GGUF notes

What was measured while building the GGUF path, so the numbers are reproducible rather than asserted.

## Layout

| File | Role |
|---|---|
| `src/gguf/gguf.hpp`, `gguf.cpp` | GGUF v2/v3 reader, 16 ggml types dequantized to float32 |
| `src/gguf/gguf_model.*` | config extraction, GGUF name to engine role mapping, validation |
| `src/gguf/gguf_quant.*` | rotation, codebook, Viterbi, warp interleave, round-trip oracle |
| `src/gguf/gguf_container.*` | the in-memory image `Model::load` reads |
| `src/gguf/gguf_engine.*` | file-kind detection, `yskn_gguf_probe`, the support gate |
| `tests/gguf/` | 6 suites, 1317 checks, no GPU needed |

## The quantization bridge

`quantize_tensor` runs the pipeline from [`GGUF_DESIGN.md`](GGUF_DESIGN.md) §5.4: rotate, per-row scale, codebook, Viterbi, pack, interleave.

### Viterbi is optimal

Against a brute-force search over all 4096 states on a 256-column sequence:

```
brute-force optimal total cost = 0.001249  (rmse per value 0.002209)
viterbi_quantize cost        = 0.001249  (rmse per value 0.002209)
ratio = 1.000000
```

### The trellis is the limit, not the encoder

In the reader a state `s'` can only be reached from states whose low `12 - bits` bits match. At `bits = K` only `2^K` of the 4096 codewords are reachable at each step, so the achievable error is set by the codeword spacing, not by the codebook size:

| bits | reachable per step | spacing | measured RMS |
|---|---|---|---|
| 4 | 16 | 63.5 | 10.6 |
| 6 | 64 | 31.8 | 5.5 |
| 8 | 256 | 15.9 | 2.9 |

Full scale is 127; RMS over a uniform pair distribution is `spacing / sqrt(12)`.

### Trellis-aware refinement

Plain k-means ignores that constraint. Lloyd rounds over the states the encoder can actually reach close most of the gap:

| pairs | bits | plain | refined |
|---|---|---|---|
| 2048 | 4 | 11.997 | 5.669 |
| 2048 | 8 | 3.013 | 1.306 |
| 131072 | 4 | 11.104 | 10.574 |
| 131072 | 8 | 3.174 | 2.940 |

More training data narrows the gap because a generalising codebook is already near the trellis optimum; one trained on too little data overfits and looks better on its own training set than it is.

The default is 2 rounds. The curve at bits=8 is 3.28 (none), 3.11 (1), 3.03 (2), 2.98 (3), 2.94 (4), so 2 captures most of the gain for half the cost of 4.

### Speed

`train_codebook` was the bottleneck: a linear scan over 4096 centroids per pair is `O(pairs * 4096)`, which is 1.4e13 distance evaluations per 27B matrix. An exact k-d tree over the centroids returns the same index (including the lowest-index tie-break), so the codebook is bit-identical, and takes the k-means from 20.9 s to 0.8 s on a 256x5120 matrix.

What remains is Viterbi, at roughly 1.4 G state-steps/s. A 512x5120 matrix at `bits=8`:

| refine rounds | time | throughput | extrapolated 27B |
|---|---|---|---|
| 0 | 5.7 s | 0.23 Mpair/s | ~17 h |
| 1 | 9.5 s | 0.14 Mpair/s | ~28 h |
| 2 | 13.6 s | 0.10 Mpair/s | ~40 h |

That is the honest cost of converting a 27B model on one CPU core. The GPU counterpart is the obvious next step: one sequence per warp, `dp[4096]` and `arg[128][2^(12-bits)]` in shared memory.

### Three bugs the oracle caught

`verify_roundtrip` compares the decoded stream against the normalized matrix `B = A / tau`. All three of these were wrong first:

1. The Viterbi cost compared `B` against the raw `int8` entries instead of `q * gamma`, which biases every codeword by `1/gamma`.
2. The oracle multiplied the decoded value by `tau`, comparing `A`-units to `B`-units.
3. `rotate_matrix` is not an involution. `A = S_out · Hn · W · Hn · S_in`, and `S` does not commute with `Hn`, so applying it twice does not return `W`. The inverse mirrors the order: `W = S_in · Hn · A · Hn · S_out`.

## It runs

A stock GGUF file loads into the engine and produces tokens. Verified on an RTX 3070 with a 28-head, head_dim-128, hidden-3584, ffn-512, 1-layer synthetic model:

```
open: True
prefill rc: 0  tok: 11
logits rc: 0  finite: True  nonzero: True
decode rc: 0  toks: [8, 0, 2, 14]
```

No `YSKN` container is involved: `GgufContainer` dequantizes the weights, runs the quantization bridge, materializes the image, and `Model::load` uploads it through the legacy `Engine`.

### Why the gate is narrower than "the kernels are templated"

The default attention path is not templated. `src/post_kernels.cu:47` writes `q16` with a literal stride of 28 heads x 128 slots and `src/attention_kv4q.cu:81` reads it back the same way. A `head_dim 256` model was measured doing this:

```
hd256 open: True
prefill rc: 0  tok: 2147483647
logits rc: 0  finite: False  nonzero: True
vals: [nan, nan, nan, nan, nan, nan]
```

It loads, reports success, and returns NaN. The gate refuses it rather than admitting it on the strength of the generic path, which is off by default.

## What still blocks the 27B model

The GGUF path runs through the legacy `Engine`, which is parametric for `hidden`, `ffn`, `kv_heads` and `vocab` but not for the attention shape. A 27B Qwen3 GGUF (24 heads, head_dim 256) is therefore refused.

Two ways forward:

1. **Generalise the legacy attention path** over `(heads, head_dim)`. The generic path already exists (`YSKN_ATTN_GENERIC=1`, `src/attention_gqa.cu:171-172`) and is instantiated for `(7,128,128)` and `(6,256,64)`; what is missing is making `qkv_post` and the decode attention honour the config instead of the literal 28 x 128, then requiring the generic path to be enabled.
2. **Teach `Engine35` to read GGUF.** It is the fast path and already handles 24 heads x 256, but it speaks a different tensor vocabulary (`layers.N.self_attn.q_proj`), needs `layer_types`, and hard-gates on the 27B shape at construction.

Option 1 is one kernel generalisation rather than a second engine.
