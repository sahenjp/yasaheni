# Colab A100 comparison: upstream YASAKANI vs yasaheni

Run on a Colab A100-SXM4-40GB (compute capability 8.0, `sm_80`), CUDA 13.0.88, 12 vCPU.
Both repositories cloned at the same time and built on the same VM.

## Summary

| | upstream `yasakani-dev/yasakani` | `sahenjp/yasaheni` |
|---|---|---|
| Builds as shipped | **no** | **yes** |
| Builds after patching | yes, with 2 edits | n/a (not needed) |
| Binary architecture | `sm_120` only | `sm_80` (matches the GPU) |
| Opens a GGUF file | no (`not a YSKN container`) | **yes** |
| `yskn_gguf_probe` export | absent | present |
| GGUF inference on A100 | n/a | **works** |
| Test suite on this machine | none shipped for the engine | 6 suites, 1317 checks, all pass |

## 1. Upstream does not build on an A100

```
/tmp/up/src/engine35.cpp:1724:25: error: '_ftelli64' was not declared in this scope
/tmp/up/src/engine35.cpp:1726:9: error: '_fseeki64' was not declared in this scope
```

Two MSVC-only CRT calls. This is a Linux portability failure independent of the GPU.

## 2. The architecture flag is ignored

Upstream's `CMakeLists.txt:10` contains a hard `set(CMAKE_CUDA_ARCHITECTURES 120)`, which overrides
whatever you pass on the command line. Passing `-DCMAKE_CUDA_ARCHITECTURES=80` changes nothing:

```
CMakeLists.txt: 10:set(CMAKE_CUDA_ARCHITECTURES 120)
actual nvcc arch flags: arch=compute_120
```

The resulting `.so` contains `sm_120` cubins only, which an A100 cannot execute. Even a fully
patched upstream build is unusable on this hardware.

## 3. With a 2-line patch, upstream builds but cannot read GGUF

Patching only `_ftelli64`/`_fseeki64` to `ftello`/`fseeko` makes it compile (201 s). It then
refuses the same GGUF file that yasaheni accepts:

```
upstream+patch   open: False | err: not a YSKN container
yasaheni         open: True  | err:
```

Upstream has no GGUF reader and no `yskn_gguf_probe` export. The `YSKN` converter that would
produce a file it *can* read is not published, so there is no path to running a model on this
machine with upstream.

## 4. yasaheni builds and runs

```
configure  rc=0    3.2 s   (-DYSKN_CUDA_ARCH=native)
build      rc=0  172.8 s
arch       sm_80
exports    38
```

GGUF end-to-end on the A100, with a 28-head / head_dim-128 / hidden-3584 fixture:

```
open: True
prefill rc=0 tok=8
logits rc=0 finite=True nonzero=True
decode rc=0 toks=[56, 31, 8, 56, 31, 4, 31, 4]
closed
```

Its own test suite, run on the same VM:

```
100% tests passed, 0 tests failed out of 6
Total Test time (real) = 34.79 sec
```

## What this does and does not show

It shows that yasaheni is the only one of the two that can build and run on an A100, and that the
GGUF path works on hardware other than the RTX 3070 it was developed on.

It does **not** compare speed. Upstream cannot run here at all, and yasaheni's GGUF path is
limited to 28-head / head_dim-128 models, so no 27B Qwen3 model can be loaded by either on this
GPU. A throughput comparison needs an `sm_120` machine for upstream, which is the hardware
upstream was built for.

## Reproducing

```bash
git clone --depth 1 https://github.com/yasakani-dev/yasakani.git /tmp/up
git clone --depth 1 https://github.com/sahenjp/yasaheni.git /tmp/yh

# upstream: fails on _ftelli64, and even patched produces sm_120 only
cmake -S /tmp/up -B /tmp/up/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/up/build -j

# yasaheni: builds for whatever GPU is present
cmake -S /tmp/yh -B /tmp/yh/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DYSKN_CUDA_ARCH=native
cmake --build /tmp/yh/build -j
```
