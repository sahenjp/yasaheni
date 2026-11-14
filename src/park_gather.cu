// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "park_gather.hpp"

namespace yskn {
namespace {

// The vector paths move 16, 8 or 4 bytes per thread when both sides are aligned and a scalar tail in thread 0 keeps the byte
// count exact.  The byte path covers every other alignment and length.
__global__ void park_gather16(const uint4* __restrict__ s, uint4* __restrict__ d, size_t u,
                              const unsigned char* __restrict__ sb, unsigned char* __restrict__ db, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < u; i += stride) d[i] = s[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) for (size_t k = u * 16; k < n; ++k) db[k] = sb[k];
}
__global__ void park_gather8(const uint2* __restrict__ s, uint2* __restrict__ d, size_t u,
                             const unsigned char* __restrict__ sb, unsigned char* __restrict__ db, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < u; i += stride) d[i] = s[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) for (size_t k = u * 8; k < n; ++k) db[k] = sb[k];
}
__global__ void park_gather4(const uint* __restrict__ s, uint* __restrict__ d, size_t u,
                             const unsigned char* __restrict__ sb, unsigned char* __restrict__ db, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < u; i += stride) d[i] = s[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) for (size_t k = u * 4; k < n; ++k) db[k] = sb[k];
}
__global__ void park_gather1(const unsigned char* __restrict__ s, unsigned char* __restrict__ d, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) d[i] = s[i];
}

}  // namespace

void park_gather(const void* src, void* dst, size_t n, cudaStream_t st) {
  if (!n) return;
  const size_t sa = (size_t)src, da = (size_t)dst;
  constexpr int th = 256;
  int width = 1; size_t units = n;
  if (((sa | da) & 15) == 0) { width = 16; units = n / 16; }
  else if (((sa | da) & 7) == 0) { width = 8; units = n / 8; }
  else if (((sa | da) & 3) == 0) { width = 4; units = n / 4; }
  size_t blocks = units ? (units + th - 1) / th : 1;
  if (blocks > 4096) blocks = 4096;
  const unsigned char* sb = static_cast<const unsigned char*>(src); unsigned char* db = static_cast<unsigned char*>(dst);
  switch (width) {
    case 16: park_gather16<<<(unsigned)blocks, th, 0, st>>>(static_cast<const uint4*>(src), static_cast<uint4*>(dst), units, sb, db, n); break;
    case 8:  park_gather8 <<<(unsigned)blocks, th, 0, st>>>(static_cast<const uint2*>(src), static_cast<uint2*>(dst), units, sb, db, n); break;
    case 4:  park_gather4 <<<(unsigned)blocks, th, 0, st>>>(static_cast<const uint*>(src), static_cast<uint*>(dst), units, sb, db, n); break;
    default: park_gather1 <<<(unsigned)blocks, th, 0, st>>>(sb, db, n); break;
  }
}

}  // namespace yskn
