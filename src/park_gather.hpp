// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace yskn {
// Exact byte-for-byte device-to-device copy, used to pack the park segments into
// the staging buffer.  No cp.async, so it also runs on sm_70.
void park_gather(const void* src, void* dst, size_t n, cudaStream_t st);
}  // namespace yskn
