# Contributing

This is a fork of [YASAKANI](https://github.com/yasakani-dev/yasakani) that ports the engine to Linux and teaches it to read GGUF. Bug reports, test results and small fixes are welcome.

## What is most useful right now

1. **A run on a GPU other than the RTX 3070 this was developed on.** The library builds for `sm_80/86/89/90/120`; only `sm_86` has actually been executed. If you have an A100, H100, 4090, 5090 or anything else, `scripts/check-env.sh` plus a `cuobjdump --list-elf` of the built library and a note on whether `yskn_open` returns an engine is the report needed.
2. **A real GGUF file that fails to parse.** The reader was checked against 19 real files and llama.cpp's own dequantizers, but more shapes are always useful. Include the file's metadata (`gguf_dump.py` output or the first 4 KB) rather than the whole file.
3. **Anything the GGUF name mapper does not cover.** `yskn_gguf_probe` prints a coverage report with `unmatched` and `missing` lists; a model that shows entries there is a concrete bug.

## Before you send a change

```bash
scripts/check-env.sh
scripts/build-linux.sh
cd tests/gguf && cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python3 -m unittest discover -s python/tests -t python
```

All four must pass. The C++ tests need no GPU.

## What needs its own test results

* **A new GPU architecture.** Add it to `CUDA_ARCH`, build, and show that the kernels run. The two instructions that needed guarding (`griddepcontrol`, `cp.async.bulk.prefetch`) are in `src/pdl.cuh` and are the likely failure point.
* **A change to the quantization bridge.** `docs/GGUF_NOTES.md` records the measured RMS at each `bits` value; a change that makes those numbers worse needs to say why that is acceptable.
* **A change to the reader.** The dequantizers are verified bit-identical against llama.cpp's `dequantize_row_*`. Keep it that way or explain the divergence.

## Style

* C++17, no dependencies beyond the STL and the vendored nlohmann/json.
* Comment the non-obvious arithmetic: the trellis window, the warp interleave, the Hadamard scaling. Those are the parts a reader cannot infer.
* Do not change `--fmad=false`. The numerics were tuned with fused multiply-add contraction disabled.

## License

Apache License 2.0, same as upstream. By sending a contribution you agree to license it that way. Do not send code or data you do not have the right to license.
