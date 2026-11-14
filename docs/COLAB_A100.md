# Colab A100 comparison: upstream YASAKANI vs yasaheni

A100-SXM4-40GB (compute capability 8.0, `sm_80`), CUDA 13.0.88, 12 vCPU.
Model: `yasakani-dev/Qwen3.8-27B-YASAKANI-speed` (10.31 GB `YSKN`).

## Getting upstream to run at all

Upstream does not build on this machine. Four changes were needed, and they are
exactly the portability work yasaheni already contains:

| Change | Why |
|---|---|
| `_ftelli64`/`_fseeki64` -> `ftello`/`fseeko` | MSVC-only CRT calls |
| `set(CMAKE_CUDA_ARCHITECTURES 120)` -> `80` | the hard-set line overrides `-DCMAKE_CUDA_ARCHITECTURES` |
| `griddepcontrol` behind `__CUDA_ARCH__ >= 900` | PDL does not exist below sm_90 |
| `cp.async.bulk.prefetch` behind the same guard | same |

With those four, upstream builds in 176 s and runs the model. yasaheni needs
none of them and builds in 173 s.

## Speed

Decode, 128 tokens, 24 runs each, interleaved A/B/A/B so drift hits both equally.
Same model, same environment, same GPU.

| Engine | mean tok/s | median | max |
|---|---|---|---|
| upstream (as patched) | 54.52 | 54.52 | 54.64 |
| **yasaheni** | **58.27** | **58.26** | **58.38** |

**+6.9%**, with no overlap between the two distributions.

Prefill is unchanged (1808 vs 1804 tok/s at 512 tokens), so the difference is
entirely in the decode path.

## Where the 6.9% comes from

It is not the compiler flags. Three things were ruled out by building upstream
with each one added in isolation:

| Variant | decode tok/s |
|---|---|
| upstream, plain | 54.42 |
| upstream + `--fmad=false` | 54.53 |
| upstream + `--allow-expensive-optimizations` | 54.46 |
| upstream, SASS only (no PTX) | 54.49 |
| **upstream + `YSKN_NO_PDL=1`** | **58.25** |
| yasaheni | 58.22 |

The cause is Programmatic Dependent Launch. On `sm_90` and later, PDL lets a
kernel start before its predecessor finishes. On `sm_80` the hardware does not
have it, but `cudaLaunchKernelEx` with the PDL attribute is still accepted and
the driver still pays for the launch metadata on every kernel launch. Upstream
requests it unconditionally, so it pays that cost on an A100.

yasaheni's `pdl_enabled()` asks the device for its compute capability first:

```cpp
static const bool arch_ok = [] {
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
  return major >= 9;
}();
return on && arch_ok && g_pdl_rows <= max_m;
```

So on an A100 it takes the plain launch path and gets the 6.9% back. On a
5090 it still uses PDL, which is what upstream tuned for.

## Correctness

Both engines produce the same tokens for the same prompt:

```
upstream  toks=[9, 10, 4838, 1665, 25, 33898, 2110, 30, 31, 1249, 61, 14717, 91, 92, 93, 1]
yasaheni  toks=[9, 10, 4838, 1665, 25, 33898, 2110, 30, 31, 1249, 61, 14717, 91, 92, 93, 1]
```

Model load time is also the same (3.56 s vs 3.53 s).

## Test suite

yasaheni's 6 suites, run on this machine: 1317 checks, all pass.

## Caveat

These numbers are for the `YSKN` path, which is what both engines share. yasaheni's
GGUF path is not comparable here: it is limited to 28-head / head_dim-128 models,
so it cannot load a 27B Qwen3 on this GPU. The GGUF path was verified separately
on the same machine with a small fixture (open, prefill, decode, logits all work).
