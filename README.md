# yasaheni

A Linux port of the [YASAKANI](https://github.com/yasakani-dev/yasakani) local LLM engine. Upstream runs on Windows with one RTX 5090 and reads only its own `YSKN` container. This fork runs on Linux across several GPU generations and reads GGUF directly.

Apache License 2.0, same as upstream. The kernels are the work of the YASAKANI authors; what changed here is the build system, the platform layer, and the GGUF input path.

## Status

| Piece | State |
|---|---|
| Linux build (`libyskn.so`) | works; `sm_80/86/89/90/120` in one binary |
| Multi-architecture kernels | works; PDL and bulk prefetch degrade cleanly below `sm_90` |
| GGUF v2/v3 reader | works; 16 ggml types, bit-identical to llama.cpp's dequantizers |
| GGUF model adapter | works; config, name mapping, validation |
| GGUF to engine quantization | works; Viterbi verified optimal against brute force |
| Running a GGUF model | works; open, prefill, decode and logits on a real GPU |

1317 checks across 6 suites, no GPU needed. See [Tests](#tests).

## Build

CMake 3.24+, Ninja, CUDA 12.8+ (13.x recommended), GCC or Clang.

```bash
scripts/check-env.sh                         # driver, nvcc, GPU, python
scripts/build-linux.sh                       # sm_120;86;80
CUDA_ARCH="80;86;89;90;120" scripts/build-linux.sh
CUDA_ARCH=native scripts/build-linux.sh
```

The result is `build-linux/libyskn.so`. The CUDA runtime is linked statically, so the library needs only the NVIDIA driver at run time.

### Why the architectures matter

The upstream kernels target `sm_120` and use two instructions that do not exist below `sm_90`:

* `griddepcontrol.wait` / `.launch_dependents` (Programmatic Dependent Launch)
* `cp.async.bulk.prefetch.L2.global` (bulk L2 prefetch)

Both are behind `__CUDA_ARCH__` guards in `src/pdl.cuh`, and `pdl_enabled()` asks the device for its compute capability before requesting the PDL launch attribute. One binary therefore runs on Ampere, Ada, Hopper and Blackwell.

## GGUF

`src/gguf/` reads GGUF v2/v3: header, every metadata value type including nested arrays, tensor infos, alignment, and an `mmap`'d data section. Dequantization covers F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K and IQ4_NL.

The dequantizers were checked three ways: against `dequantize_row_*` lifted from llama.cpp's `ggml-quants.c` (bit-identical on all 12 quantized types), against the official `gguf` Python package on 19 real GGUF files, and by mutation and fuzz testing.

### What the gate accepts

A GGUF is loaded by the legacy `Engine`, which reads `hidden`, `ffn`, `kv_heads` and `vocab` from the file. The attention shape is not parametric: `src/post_kernels.cu:47` writes `q16` with a literal stride of 28 heads x 128 slots and `src/attention_kv4q.cu:81` reads it back the same way. So the gate requires `heads == 28`, `head_dim == 128`, and `ffn` a multiple of 512. Anything else is refused with a message naming the field, the value, and the kernel line.

A `head_dim 256` model was measured loading successfully and then returning NaN logits, which is why the gate refuses it rather than trusting the templated generic path that is off by default.

## Tests

```bash
cd tests/gguf
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

| Suite | Checks | What it covers |
|---|---|---|
| `gguf_reader` | 304 | parsing, dequantization, malformed input |
| `gguf_model_adapter` | 377 | config, name mapping, validation |
| `gguf_engine_wiring` | 147 | file-kind detection, the support gate, the probe |
| `gguf_quant` | 84 | rotation, codebook, Viterbi optimality, round trip |
| `gguf_container` | 394 | the in-memory image the loader reads |
| `gguf_load_dispatch` | 11 | YSKN and GGUF routing through `Model::load` |

## Documentation

* [`docs/FORMAT.md`](docs/FORMAT.md): upstream's byte-level specification of the `.yskn` format.
* [`docs/GGUF.md`](docs/GGUF.md): what this fork's GGUF reader and encoder do, and what they do not.
* [`docs/GGUF_NOTES.md`](docs/GGUF_NOTES.md): measured behaviour of the quantization bridge, including the trellis bound, the refinement gains, and the bugs the round-trip oracle caught.
* [`docs/COLAB_A100.md`](docs/COLAB_A100.md): both engines measured on an A100, and where the 6.9% gap comes from.
* [`bench/README.md`](bench/README.md): the benchmark harness and how the speed number is defined.

## Upstream

The original engine, its published measurements and its model files are at [yasakani-dev/yasakani](https://github.com/yasakani-dev/yasakani). This port does not change what the kernels compute; it changes where they run and what they can read.

### Where this fork stands against upstream v0.1.2

Upstream released v0.1.1 and v0.1.2 on 2026-11-13/14. This fork started from v0.1.0. Checked against v0.1.2:

| | v0.1.2 | This fork |
|---|---|---|
| MSVC-only `_ftelli64`/`_fseeki64` | fixed (same `#ifndef _WIN32` macro) | fixed |
| `--fmad=false` on Linux builds | added | present |
| `CMAKE_CUDA_ARCHITECTURES` | still hard-set to `120`; `-DCMAKE_CUDA_ARCHITECTURES=80` is ignored | `-DYSKN_CUDA_ARCH=native` / `portable` |
| `griddepcontrol` below sm_90 | still unguarded, ptxas fails | guarded |
| `cp.async.bulk.prefetch` below sm_90 | still unguarded, ptxas fails | guarded |
| PDL on a pre-Hopper device | still requested, measured 6.9% slower on an A100 | disabled by a compute-capability check |
| GGUF input | not supported | reader, adapter, quantizer, container |

Verified by compiling the same two translation units for `sm_80`: v0.1.2 produces `ptxas ... requires .target sm_90 or higher`, this fork produces zero errors.

Upstream v0.1.2 also published the encoder at [`tools/convert/`](tools/convert/) and the format specification at [`docs/FORMAT.md`](docs/FORMAT.md); [docs/GGUF.md](docs/GGUF.md) records where this fork's encoder differs from the reference one.

