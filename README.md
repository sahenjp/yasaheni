# yasakani-sahen

A Linux port of the [YASAKANI](https://github.com/yasakani-dev/yasakani) local LLM engine, with two goals:

1. **Run on more than one GPU.** Upstream targets Windows and a single RTX 5090 (`sm_120`). This port builds for `sm_80 / 86 / 89 / 90 / 120` from one source tree and runs on Linux.
2. **Read GGUF directly.** Upstream only reads its own `YSKN` container, and the converter that writes it is not published. This port adds a GGUF v2/v3 reader so stock GGUF files can be opened without a conversion step.

Apache License 2.0, same as upstream. The engine kernels are the work of the YASAKANI authors; the port is in the build system, the platform layer, and the GGUF path.

## Status

| Piece | State |
|---|---|
| Linux build of the engine (`libyskn.so`) | works; `sm_80/86/89/90/120` in one binary |
| Multi-architecture kernels | works; PDL and bulk prefetch degrade cleanly below `sm_90` |
| GGUF v2/v3 reader | works; 16 ggml types dequantized, verified bit-identical against llama.cpp |
| GGUF model adapter (config, name mapping, validation) | works; 377 checks |
| GGUF to engine tensor quantization | works; Viterbi verified optimal, 84 checks |
| Running a GGUF model end to end | not yet; the 27B kernels are shape-specialised |

## Build

Requires CMake 3.24+, Ninja, a CUDA 12.8+ toolkit (13.x recommended), and GCC or Clang.

```bash
scripts/build-linux.sh                       # sm_120;86;80
CUDA_ARCH="80;86;89;90;120" scripts/build-linux.sh
CUDA_ARCH=native scripts/build-linux.sh      # whatever GPU this machine has
```

The result is `build-linux/libyskn.so`. The CUDA runtime is linked statically, so the library needs only the NVIDIA driver at run time.

Check the machine first with `scripts/check-env.sh`.

## Why the architectures matter

The upstream kernels are written for `sm_120` and use two instructions that do not exist below `sm_90`:

* `griddepcontrol.wait` / `.launch_dependents` (Programmatic Dependent Launch)
* `cp.async.bulk.prefetch.L2.global` (bulk L2 prefetch)

Both are now behind `__CUDA_ARCH__` guards (`src/pdl.cuh`), and the host-side launch path asks the device for its compute capability before enabling PDL, so the same binary runs on Ampere, Ada, Hopper and Blackwell.

## GGUF

`src/gguf/` reads GGUF v2/v3: header, every metadata value type including nested arrays, tensor infos, alignment, and an `mmap`'d data section. Dequantization covers F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K and IQ4_NL.

The dequantizers were checked three ways: against `dequantize_row_*` lifted from llama.cpp's `ggml-quants.c` (bit-identical on all 12 quantized types), against the official `gguf` Python package on 19 real GGUF files (including a full 151,936-token Qwen2 vocabulary), and by mutation and fuzz testing.

`docs/GGUF_DESIGN.md` documents the `YSKN` container the engine was built around, and what a converter from GGUF to that format would need. `docs/GGUF_NOTES.md` records the measured behaviour of the quantization bridge: the trellis bound, the codebook refinement gains, and the bugs the round-trip oracle caught.

## Tests

```bash
cd tests/gguf
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

No GPU is needed for these; they are host-side.

## Upstream

The original engine, its published measurements and its model files are at [yasakani-dev/yasakani](https://github.com/yasakani-dev/yasakani). This port does not change what the kernels compute; it changes where they can run and what they can read.
