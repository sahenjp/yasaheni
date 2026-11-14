# The GGUF path

This fork adds a GGUF reader and an encoder that produces the engine's own
format. The format itself is documented by upstream, byte for byte, in
[`FORMAT.md`](FORMAT.md); this page covers only what is specific to the bridge.

## What is here

| File | Role |
|---|---|
| `src/gguf/gguf.hpp`, `gguf.cpp` | GGUF v2/v3 reader: header, every metadata type, tensor infos, alignment, `mmap`'d data |
| `src/gguf/gguf_model.*` | hyperparameters, GGUF name to engine role mapping, validation |
| `src/gguf/gguf_quant.*` | rotation, codebook, Viterbi, warp interleave, round-trip oracle |
| `src/gguf/gguf_container.*` | the in-memory image `Model::load` reads |
| `src/gguf/gguf_engine.*` | file-kind detection, `yskn_gguf_probe`, the support gate |

## What the encoder implements

`quantize_tensor` follows the layout in [`FORMAT.md`](FORMAT.md):

1. `A = S_out · Hn · W · Hn · S_in` with `Hn = H/sqrt(512)` and the sign streams from `sign_stream_rule`.
2. A per-row scale `tau`, then `B = A / tau`.
3. A 4096-entry pair codebook, with `max|q| == 127` so the reader's `build_lut8_tau` recovers it exactly.
4. Viterbi over the 12-bit trellis, which is optimal for a given codebook (verified against brute force).
5. The `layout:"warp"` interleave from `src/repack.cu`.

## What it does not implement

The reference encoder in [`tools/convert/`](../tools/convert/) does two things this
one does not, and they are the reason its output is closer to the original
model at the same bit count:

* **The per-row scale is the row RMS**, not `max|A[r]|/127`. RMS leaves the bulk
  of the distribution near the codebook centre instead of pinning the largest
  outlier to 127. See `row_scale16` in `tools/convert/yskn_convert.py`.
* **Error feedback is GPTQ/LDLQ**: a damped block-diagonal Hessian per
  256-column block, Cholesky-factored, with the residual pushed into the
  columns that follow. See `feedback_tables` and `upper_factor_of_inverse`.

A container built without them is still self-consistent and loads and runs; it
is simply lossier. Measured numbers are in [`GGUF_NOTES.md`](GGUF_NOTES.md).

## The support gate

A GGUF is loaded by the legacy `Engine`, which reads `hidden`, `ffn`,
`kv_heads` and `vocab` from the file. The attention shape is not parametric:
`src/post_kernels.cu:47` writes `q16` with a literal stride of 28 heads x 128
slots and `src/attention_kv4q.cu:81` reads it back the same way. The gate
therefore requires `heads == 28`, `head_dim == 128` and `ffn` a multiple of 512,
and names the field and the value when it refuses.

A `head_dim 256` model was measured loading successfully and then returning NaN
logits, which is why it is refused rather than admitted on the strength of the
templated generic path that is off by default.

## Conformance

`tools/testdata/tiny.yskn` and `tiny_expected.json` are upstream's own fixture
and the values a reader must produce from it. `tests/gguf/yskn_conformance_test.cpp`
checks this repository's reader against them: the header, the model object,
every tensor's metadata, the vectors, and the SHA-256 of all five sign streams.
