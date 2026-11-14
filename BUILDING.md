# Building the engine

The engine is one shared library, `libyskn.so` on Linux and `yskn.dll` on Windows. You do not need a GPU to build it, only to run it.

## Requirements

| | |
|---|---|
| OS | Linux (tested on Fedora 44) or Windows 11 |
| C++ compiler | GCC 13+ or Clang 16+ on Linux; MSVC 19.4x on Windows |
| CUDA | CUDA Toolkit 12.8 or newer (13.5 tested). `sm_120` needs 12.8+. |
| Build tools | CMake 3.24 or newer, Ninja |
| Python | 3.10+ (only for the Python package) |

## Linux

```bash
scripts/check-env.sh          # nvcc, driver, GPU, python
scripts/build-linux.sh
```

`scripts/build-linux.sh` finds `nvcc` on `PATH`, then in `$CUDA_HOME`, then in `/usr/local/cuda`. The architecture list comes from `CUDA_ARCH` (default `120;86;80`):

```bash
CUDA_ARCH="80;86;89;90;120" scripts/build-linux.sh   # one binary for Ampere through Blackwell
CUDA_ARCH=native scripts/build-linux.sh              # only this machine's GPU
```

Or drive CMake directly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DYSKN_CUDA_ARCH="80;86;89;90;120"
cmake --build build -j
```

`-DYSKN_CUDA_ARCH` also accepts `native` and `portable` (`120;90;86;80;75`). Without it, `-DCMAKE_CUDA_ARCHITECTURES` is used, defaulting to `120`.

## Windows

Open the **x64 Native Tools Command Prompt for VS 2022**, go to the source folder and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll`.

## What the build flags are for

* `--fmad=false` is not cosmetic. The engine's numerics were tuned and verified with fused multiply-add contraction disabled; turning it on changes results.
* `-Xptxas -O3` and `--allow-expensive-optimizations=true` are kept on every platform so the register allocation matches what was measured.
* `-DYSKN_WITH_TEST_EXPORTS=ON` adds the test/probe translation units (`src/kvx_probe.cu`, `src/pm_test.cu`, `src/test_hooks.cpp`) to the library. They are off by default: nothing in the engine calls them, the public header does not declare them, and they take raw device pointers.
* On Linux the CUDA runtime is linked statically (`CUDA::cudart_static`), so the `.so` needs only the NVIDIA driver at run time, not the toolkit.
* Visibility is hidden by default and the C ABI is exported explicitly (`include/yskn_abi.h` uses `__attribute__((visibility("default")))`). `-Wl,--no-undefined` turns a missing symbol into a link error rather than a run-time crash.

## Architectures

The kernels need `sm_80` or newer. Two instructions used upstream only exist from `sm_90`:

* `griddepcontrol.wait` / `griddepcontrol.launch_dependents` (PDL)
* `cp.async.bulk.prefetch.L2.global`

Both are behind `__CUDA_ARCH__` guards in `src/pdl.cuh`, and `pdl_enabled()` checks the device's compute capability before asking for the PDL launch attribute. Building for `sm_80` therefore works and produces a correct, slightly slower binary.

## Verifying a build

```bash
nm -D --defined-only build/libyskn.so | grep yskn_open   # the C ABI is exported
cuobjdump --list-elf build/libyskn.so | grep -o 'sm_[0-9]*' | sort -u   # which architectures are inside
cd tests/gguf && cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
```

The GGUF tests need no GPU.
