// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

// Programmatic Dependent Launch (PDL) is a Hopper (sm_90) and later feature: the PTX
// instructions griddepcontrol.wait / griddepcontrol.launch_dependents do not exist on
// older targets. Everything here degrades to an ordinary launch on those targets, so the
// same source builds for sm_75 .. sm_120.
#include <cuda_runtime.h>
#include <utility>
namespace yskn {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#  define YSKN_HAS_PDL 1
#else
#  define YSKN_HAS_PDL 0
#endif

// Bulk L2 prefetch (cp.async.bulk.prefetch) is sm_90+. On older targets we issue an
// ordinary per-warp prefetch so the memory pipeline still gets a hint, just cheaper.
__device__ __forceinline__ void l2_prefetch(const void* p, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
#else
  // One 128 B line per warp; the caller loops over the range anyway.
  if ((threadIdx.x & 31) == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(p) : "memory");
  (void)bytes;
#endif
}

__device__ __forceinline__ void pdl_wait() {
#if YSKN_HAS_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if YSKN_HAS_PDL
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Host side: true only when the compiled binary has PDL and the selected device supports it.
bool pdl_enabled();

template <typename... KArgs, typename... Args>
inline void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  if (!pdl_enabled()) { kern<<<grid, block, smem, st>>>(args...); return; }
  cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}
}
