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

## What still blocks running a GGUF model end to end

`yskn_gguf_probe` reports `supported: false` with a specific reason for anything that is not the 27B shape, because `Engine35` hard-codes `hidden 5120 / 24 heads / 4 kv heads / 256 head_dim` and rejects everything else at construction. The GGUF reader, the adapter and the quantizer are all shape-agnostic; the kernels are not.

To finish the job the engine needs either:

1. **Shape templating.** Instantiate the Qwen3.5 kernels over `(hidden, heads, kv_heads, head_dim)` for the sizes that matter, or
2. **A GGUF-backed `Model` for the legacy `Engine` path**, which is already shape-parametric (`Model::load` reads every dimension from `ModelDesc`), and accept that it is slower than the specialised 27B path.

Option 2 is the shorter road to a working end-to-end run and is what the next slice should do.
