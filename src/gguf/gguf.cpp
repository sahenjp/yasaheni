// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF v2/v3 reader with float32 dequantization.
//
// Block layouts are byte-identical to llama.cpp's ggml.h / ggml-common.h and
// the dequantization math follows ggml/src/ggml-quants.c line for line, so a
// tensor decoded here matches what llama.cpp would produce for the same file.
#include "gguf.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "../mapped_file.hpp"

namespace yskn::gguf {
namespace {

// ---------------------------------------------------------------------------
// Little-endian scalar reads. GGUF is little-endian regardless of host, and
// tensor payloads are not guaranteed to be aligned, so everything goes through
// byte-wise assembly instead of pointer casts.
// ---------------------------------------------------------------------------
inline uint16_t rd_u16(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

inline uint32_t rd_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t rd_u64(const uint8_t* p) {
  return static_cast<uint64_t>(rd_u32(p)) |
         (static_cast<uint64_t>(rd_u32(p + 4)) << 32);
}

inline float rd_f32(const uint8_t* p) {
  const uint32_t bits = rd_u32(p);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline double rd_f64(const uint8_t* p) {
  const uint64_t bits = rd_u64(p);
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return d;
}

// ---------------------------------------------------------------------------
// Half-precision decode (no F16C dependency: this must run on any C++17 host).
// ---------------------------------------------------------------------------
inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal half: renormalize. Shift the mantissa left until the
      // implicit bit reaches position 10, counting the shifts in 'e'.
      uint32_t m = mant;
      int e = -1;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) |
             ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// bfloat16 is just the top 16 bits of a float32.
inline float bf16_to_f32(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// ---------------------------------------------------------------------------
// Type table. Sizes must match sizeof(block_*) in ggml-common.h exactly.
// ---------------------------------------------------------------------------
struct TypeDesc {
  const char* name;
  uint64_t block_size;  // bytes per block
  uint64_t block_elems; // elements per block
};

constexpr uint64_t QK = 32;    // Q4_0 / Q5_0 / Q8_0 / IQ4_NL block
constexpr uint64_t QK_K = 256; // K-quant super-block

TypeDesc type_desc(uint32_t t) {
  switch (t) {
    case TYPE_F32:   return {"F32", 4, 1};
    case TYPE_F16:   return {"F16", 2, 1};
    case TYPE_BF16:  return {"BF16", 2, 1};
    case TYPE_Q4_0:  return {"Q4_0", 2 + QK / 2, QK};              // 18
    case TYPE_Q4_1:  return {"Q4_1", 2 * 2 + QK / 2, QK};          // 20
    case TYPE_Q5_0:  return {"Q5_0", 2 + 4 + QK / 2, QK};          // 22
    case TYPE_Q5_1:  return {"Q5_1", 2 * 2 + 4 + QK / 2, QK};      // 24
    case TYPE_Q8_0:  return {"Q8_0", 2 + QK, QK};                  // 34
    case TYPE_Q2_K:  return {"Q2_K", 16 + QK_K / 4 + 2 + 2, QK_K}; // 84
    case TYPE_Q3_K:  return {"Q3_K", QK_K / 8 + QK_K / 4 + 12 + 2, QK_K};  // 110
    case TYPE_Q4_K:  return {"Q4_K", 2 + 2 + 12 + QK_K / 2, QK_K};         // 144
    case TYPE_Q5_K:  return {"Q5_K", 2 + 2 + 12 + QK_K / 8 + QK_K / 2, QK_K};  // 176
    case TYPE_Q6_K:  return {"Q6_K", QK_K / 2 + QK_K / 4 + QK_K / 16 + 2, QK_K};  // 210
    case TYPE_Q8_K:  return {"Q8_K", 4 + QK_K + 16 * 2, QK_K};     // 292
    case TYPE_IQ4_NL: return {"IQ4_NL", 2 + QK / 2, QK};           // 18
    default:         return {nullptr, 0, 0};
  }
}

// ---------------------------------------------------------------------------
// Dequantization kernels. Each consumes nb = n/block_elems blocks of 'src' and
// writes n floats to 'y'. The bit twiddling mirrors ggml-quants.c.
// ---------------------------------------------------------------------------

void deq_f32(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) y[i] = rd_f32(src + 4 * i);
}

void deq_f16(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) y[i] = fp16_to_f32(rd_u16(src + 2 * i));
}

void deq_bf16(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) y[i] = bf16_to_f32(rd_u16(src + 2 * i));
}

// block_q4_0 { fp16 d; uint8 qs[16]; }  x = d * (q - 8), q in 0..15
void deq_q4_0(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 18;
    const float d = fp16_to_f32(rd_u16(b));
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
      // Low nibbles are elements 0..15, high nibbles 16..31.
      const int x0 = (qs[j] & 0x0F) - 8;
      const int x1 = (qs[j] >> 4) - 8;
      y[j] = x0 * d;
      y[j + 16] = x1 * d;
    }
    y += QK;
  }
}

// block_q4_1 { fp16 d; fp16 m; uint8 qs[16]; }  x = d * q + m
void deq_q4_1(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 20;
    const float d = fp16_to_f32(rd_u16(b));
    const float m = fp16_to_f32(rd_u16(b + 2));
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; ++j) {
      const int x0 = qs[j] & 0x0F;
      const int x1 = qs[j] >> 4;
      y[j] = x0 * d + m;
      y[j + 16] = x1 * d + m;
    }
    y += QK;
  }
}

// block_q5_0 { fp16 d; uint8 qh[4]; uint8 qs[16]; }  x = d * (q - 16), q in 0..31
// The 5th bit of element j lives in bit j of the little-endian qh word, and of
// element j+16 in bit j+16.
void deq_q5_0(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 22;
    const float d = fp16_to_f32(rd_u16(b));
    const uint32_t qh = rd_u32(b + 2);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
      const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      const int x0 = (qs[j] & 0x0F) | xh0;
      const int x1 = (qs[j] >> 4) | xh1;
      y[j] = (x0 - 16) * d;
      y[j + 16] = (x1 - 16) * d;
    }
    y += QK;
  }
}

// block_q5_1 { fp16 d; fp16 m; uint8 qh[4]; uint8 qs[16]; }  x = d * q + m
void deq_q5_1(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 24;
    const float d = fp16_to_f32(rd_u16(b));
    const float m = fp16_to_f32(rd_u16(b + 2));
    const uint32_t qh = rd_u32(b + 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; ++j) {
      const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      const int x0 = (qs[j] & 0x0F) | xh0;
      const int x1 = (qs[j] >> 4) | xh1;
      y[j] = x0 * d + m;
      y[j + 16] = x1 * d + m;
    }
    y += QK;
  }
}

// block_q8_0 { fp16 d; int8 qs[32]; }  x = d * q
void deq_q8_0(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 34;
    const float d = fp16_to_f32(rd_u16(b));
    const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
    for (int j = 0; j < 32; ++j) y[j] = qs[j] * d;
    y += QK;
  }
}

// block_q2_K { uint8 scales[16]; uint8 qs[64]; fp16 d; fp16 dmin; }
// 16 sub-blocks of 16 values: x = d*(sc&0xF)*q - dmin*(sc>>4), q in 0..3.
void deq_q2_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 84;
    const uint8_t* scales = b;
    const uint8_t* q = b + 16;
    const float d = fp16_to_f32(rd_u16(b + 80));
    const float dmin = fp16_to_f32(rd_u16(b + 82));

    int is = 0;
    for (int nb128 = 0; nb128 < 2; ++nb128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        // Two 16-element sub-blocks share one 2-bit plane per 32-byte group.
        uint8_t sc = scales[is++];
        float dl = d * (sc & 0x0F);
        float ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;

        sc = scales[is++];
        dl = d * (sc & 0x0F);
        ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;

        shift += 2;
      }
      q += 32;
    }
  }
}

// block_q3_K { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; fp16 d; }
// 16 sub-blocks of 16 values, 3-bit quants: x = d*(sc-32)*q, q in -4..3.
void deq_q3_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 110;
    const uint8_t* hm = b;
    const uint8_t* q = b + 32;
    const uint8_t* sc12 = b + 96;
    const float d_all = fp16_to_f32(rd_u16(b + 108));

    // Sixteen 6-bit scales are packed into 12 bytes: the first 8 bytes hold the
    // low 4 bits of each scale in one nibble and two high bits of a *different*
    // scale in the other, and the last 4 bytes hold the remaining high bits.
    // Unpack exactly like ggml's dequantize_row_q3_K.
    const uint32_t a0 = rd_u32(sc12 + 0);
    const uint32_t a1 = rd_u32(sc12 + 4);
    const uint32_t a2 = rd_u32(sc12 + 8);
    const uint32_t kmask1 = 0x03030303u;  // two high bits per byte
    const uint32_t kmask2 = 0x0f0f0f0fu;  // low nibble per byte
    const uint32_t w0 = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4);
    const uint32_t w1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4);
    const uint32_t w2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4);
    const uint32_t w3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4);
    const uint32_t words[4] = {w0, w1, w2, w3};
    int8_t scales[16];
    for (int w = 0; w < 4; ++w)
      for (int by = 0; by < 4; ++by)
        scales[w * 4 + by] = static_cast<int8_t>((words[w] >> (8 * by)) & 0xFF);

    uint8_t m = 1;
    int is = 0;
    for (int nb128 = 0; nb128 < 2; ++nb128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        const float dl0 = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          // The high bit is stored inverted in hmask: a set bit means "no
          // borrow", a clear bit subtracts 4.
          const int qv = static_cast<int>((q[l] >> shift) & 3) -
                         ((hm[l] & m) ? 0 : 4);
          *y++ = dl0 * static_cast<float>(qv);
        }
        const float dl1 = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int qv = static_cast<int>((q[l + 16] >> shift) & 3) -
                         ((hm[l + 16] & m) ? 0 : 4);
          *y++ = dl1 * static_cast<float>(qv);
        }
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

// ggml's get_scale_min_k4: pulls one 6-bit scale and one 6-bit min out of the
// 12 packed bytes. Scales 0..3 live in the low 6 bits of q[0..3]; their two
// high bits are stashed in bits 6..7 of q[8..11]. Mins mirror that in q[4..7].
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = static_cast<uint8_t>(q[j] & 63);
    *m = static_cast<uint8_t>(q[j + 4] & 63);
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

// block_q4_K { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qs[128]; }
// 8 sub-blocks of 32: x = d*sc*q - dmin*m, q in 0..15.
void deq_q4_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 144;
    const float d = fp16_to_f32(rd_u16(b));
    const float dmin = fp16_to_f32(rd_u16(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* q = b + 16;

    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc = 0, m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// block_q5_K { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }
// Same as Q4_K plus a 5th bit per value, one bit plane per 64-element group.
void deq_q5_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 176;
    const float d = fp16_to_f32(rd_u16(b));
    const float dmin = fp16_to_f32(rd_u16(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;

    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc = 0, m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (int l = 0; l < 32; ++l) {
        const int qv = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
        *y++ = d1 * static_cast<float>(qv) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        const int qv = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
        *y++ = d2 * static_cast<float>(qv) - m2;
      }
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
}

// block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d; }
// 16 sub-blocks of 16 values: x = d*sc*(q-32), q in 0..63.
void deq_q6_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 210;
    const float d = fp16_to_f32(rd_u16(b + 208));
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);

    for (int nb128 = 0; nb128 < 2; ++nb128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        // Four 6-bit values are spread over two ql bytes and one qh byte.
        const int q1 = static_cast<int>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// block_q8_K { float d; int8 qs[256]; int16 bsums[16]; }  x = d * q
// (bsums are a dot-product helper, not part of the values.)
void deq_q8_K(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK_K;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 292;
    const float d = rd_f32(b);
    const int8_t* qs = reinterpret_cast<const int8_t*>(b + 4);
    for (int j = 0; j < 256; ++j) y[j] = qs[j] * d;
    y += QK_K;
  }
}

// Non-linear 4-bit table used by IQ4_NL (and IQ4_XS).
constexpr int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,    13,   25,  38,  53,  69,  89,  113};

// block_iq4_nl { fp16 d; uint8 qs[16]; }  x = d * kvalues_iq4nl[q]
void deq_iq4_nl(const uint8_t* src, float* y, uint64_t n) {
  const uint64_t nb = n / QK;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* b = src + i * 18;
    const float d = fp16_to_f32(rd_u16(b));
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
      y[j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
      y[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
    }
    y += QK;
  }
}

// ---------------------------------------------------------------------------
// Bounds-checked sequential reader over the mapped file.
// ---------------------------------------------------------------------------
class Reader {
 public:
  Reader(const uint8_t* base, size_t size, const std::string& path)
      : base_(base), size_(size), path_(path) {}

  size_t pos() const { return pos_; }
  size_t remaining() const { return size_ - pos_; }
  void seek(size_t p) {
    if (p > size_) fail("seek past end of file");
    pos_ = p;
  }

  const uint8_t* take(uint64_t n, const char* what) {
    if (n > remaining()) fail(std::string("truncated file while reading ") + what);
    const uint8_t* p = base_ + pos_;
    pos_ += static_cast<size_t>(n);
    return p;
  }

  uint8_t u8(const char* what) { return *take(1, what); }
  uint16_t u16(const char* what) { return rd_u16(take(2, what)); }
  uint32_t u32(const char* what) { return rd_u32(take(4, what)); }
  uint64_t u64(const char* what) { return rd_u64(take(8, what)); }

  // GGUF string: uint64 length followed by raw bytes (v2/v3; v1 used uint32).
  std::string str(const char* what) {
    const uint64_t n = u64(what);
    if (n > remaining()) fail(std::string("string length exceeds file while reading ") + what);
    const uint8_t* p = take(n, what);
    return std::string(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
  }

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("gguf: " + path_ + ": " + msg + " (offset " +
                             std::to_string(pos_) + ")");
  }

 private:
  const uint8_t* base_;
  size_t size_;
  size_t pos_ = 0;
  const std::string& path_;
};

// Upper bound on array nesting; GGUF files in the wild use at most 2 levels.
constexpr int kMaxArrayDepth = 32;

Value read_value(Reader& r, ValueType t, int depth) {
  switch (t) {
    case ValueType::UINT8:  return Value::make_uint(t, r.u8("uint8 value"));
    case ValueType::INT8:   return Value::make_int(t, static_cast<int8_t>(r.u8("int8 value")));
    case ValueType::UINT16: return Value::make_uint(t, r.u16("uint16 value"));
    case ValueType::INT16:  return Value::make_int(t, static_cast<int16_t>(r.u16("int16 value")));
    case ValueType::UINT32: return Value::make_uint(t, r.u32("uint32 value"));
    case ValueType::INT32:  return Value::make_int(t, static_cast<int32_t>(r.u32("int32 value")));
    case ValueType::FLOAT32: {
      const uint32_t bits = r.u32("float32 value");
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return Value::make_float(t, f);
    }
    case ValueType::BOOL: {
      const uint8_t b = r.u8("bool value");
      if (b > 1) r.fail("bool value is neither 0 nor 1");
      return Value::make_bool(b != 0);
    }
    case ValueType::STRING: return Value::make_string(r.str("string value"));
    case ValueType::UINT64: return Value::make_uint(t, r.u64("uint64 value"));
    case ValueType::INT64:  return Value::make_int(t, static_cast<int64_t>(r.u64("int64 value")));
    case ValueType::FLOAT64: {
      const uint64_t bits = r.u64("float64 value");
      double d;
      std::memcpy(&d, &bits, sizeof(d));
      return Value::make_float(t, d);
    }
    case ValueType::ARRAY: {
      if (depth >= kMaxArrayDepth) r.fail("array nesting too deep");
      const ValueType elem = static_cast<ValueType>(r.u32("array element type"));
      const uint64_t count = r.u64("array length");
      // Every element occupies at least one byte, so a count larger than the
      // remaining file is corrupt. This keeps hostile headers from asking for
      // a huge allocation.
      if (count > r.remaining()) r.fail("array length exceeds remaining file size");
      std::vector<Value> items;
      items.reserve(static_cast<size_t>(std::min<uint64_t>(count, 4096)));
      for (uint64_t i = 0; i < count; ++i) items.push_back(read_value(r, elem, depth + 1));
      return Value::make_array(elem, std::move(items));
    }
  }
  r.fail("unknown metadata value type " + std::to_string(static_cast<uint32_t>(t)));
}

}  // namespace

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------
const char* value_type_name(ValueType t) {
  switch (t) {
    case ValueType::UINT8:   return "uint8";
    case ValueType::INT8:    return "int8";
    case ValueType::UINT16:  return "uint16";
    case ValueType::INT16:   return "int16";
    case ValueType::UINT32:  return "uint32";
    case ValueType::INT32:   return "int32";
    case ValueType::FLOAT32: return "float32";
    case ValueType::BOOL:    return "bool";
    case ValueType::STRING:  return "string";
    case ValueType::ARRAY:   return "array";
    case ValueType::UINT64:  return "uint64";
    case ValueType::INT64:   return "int64";
    case ValueType::FLOAT64: return "float64";
  }
  return "unknown";
}

bool Value::is_number() const {
  switch (type) {
    case ValueType::UINT8:
    case ValueType::INT8:
    case ValueType::UINT16:
    case ValueType::INT16:
    case ValueType::UINT32:
    case ValueType::INT32:
    case ValueType::UINT64:
    case ValueType::INT64:
    case ValueType::FLOAT32:
    case ValueType::FLOAT64:
      return true;
    default:
      return false;
  }
}

namespace {
[[noreturn]] void type_mismatch(const char* want, ValueType got) {
  throw std::runtime_error(std::string("gguf: metadata value is ") +
                           value_type_name(got) + ", not " + want);
}
}  // namespace

uint64_t Value::as_uint() const {
  switch (type) {
    case ValueType::UINT8:
    case ValueType::UINT16:
    case ValueType::UINT32:
    case ValueType::UINT64:
      return u;
    case ValueType::BOOL:
      return u ? 1u : 0u;
    case ValueType::INT8:
    case ValueType::INT16:
    case ValueType::INT32:
    case ValueType::INT64:
      if (i < 0) type_mismatch("an unsigned integer", type);
      return static_cast<uint64_t>(i);
    default:
      type_mismatch("an unsigned integer", type);
  }
}

int64_t Value::as_int() const {
  switch (type) {
    case ValueType::INT8:
    case ValueType::INT16:
    case ValueType::INT32:
    case ValueType::INT64:
      return i;
    case ValueType::UINT8:
    case ValueType::UINT16:
    case ValueType::UINT32:
    case ValueType::UINT64:
      if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        type_mismatch("a signed integer", type);
      return static_cast<int64_t>(u);
    default:
      type_mismatch("a signed integer", type);
  }
}

double Value::as_float() const {
  switch (type) {
    case ValueType::FLOAT32:
    case ValueType::FLOAT64:
      return f;
    case ValueType::UINT8:
    case ValueType::UINT16:
    case ValueType::UINT32:
    case ValueType::UINT64:
      return static_cast<double>(u);
    case ValueType::INT8:
    case ValueType::INT16:
    case ValueType::INT32:
    case ValueType::INT64:
      return static_cast<double>(i);
    default:
      type_mismatch("a floating point value", type);
  }
}

bool Value::as_bool() const {
  if (type != ValueType::BOOL) type_mismatch("a bool", type);
  return u != 0;
}

const std::string& Value::as_string() const {
  if (type != ValueType::STRING) type_mismatch("a string", type);
  return str;
}

const std::vector<Value>& Value::as_array() const {
  if (type != ValueType::ARRAY) type_mismatch("an array", type);
  return items;
}

Value Value::make_uint(ValueType t, uint64_t v) {
  Value x;
  x.type = t;
  x.u = v;
  x.i = static_cast<int64_t>(v);
  return x;
}

Value Value::make_int(ValueType t, int64_t v) {
  Value x;
  x.type = t;
  x.i = v;
  x.u = static_cast<uint64_t>(v);
  return x;
}

Value Value::make_float(ValueType t, double v) {
  Value x;
  x.type = t;
  x.f = v;
  return x;
}

Value Value::make_bool(bool v) {
  Value x;
  x.type = ValueType::BOOL;
  x.u = v ? 1u : 0u;
  return x;
}

Value Value::make_string(std::string s) {
  Value x;
  x.type = ValueType::STRING;
  x.str = std::move(s);
  return x;
}

Value Value::make_array(ValueType elem_type, std::vector<Value> items) {
  Value x;
  x.type = ValueType::ARRAY;
  x.elem_type = elem_type;
  x.items = std::move(items);
  return x;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------
const Value* Metadata::find(const std::string& key) const {
  for (const auto& kv : items_)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

const Value& Metadata::at(const std::string& key) const {
  const Value* v = find(key);
  if (!v) throw std::runtime_error("gguf: missing metadata key '" + key + "'");
  return *v;
}

uint64_t Metadata::get_uint(const std::string& key, uint64_t fallback) const {
  const Value* v = find(key);
  return v ? v->as_uint() : fallback;
}

int64_t Metadata::get_int(const std::string& key, int64_t fallback) const {
  const Value* v = find(key);
  return v ? v->as_int() : fallback;
}

double Metadata::get_float(const std::string& key, double fallback) const {
  const Value* v = find(key);
  return v ? v->as_float() : fallback;
}

bool Metadata::get_bool(const std::string& key, bool fallback) const {
  const Value* v = find(key);
  return v ? v->as_bool() : fallback;
}

std::string Metadata::get_string(const std::string& key,
                                 const std::string& fallback) const {
  const Value* v = find(key);
  return v ? v->as_string() : fallback;
}

void Metadata::insert(std::string key, Value v) {
  for (auto& kv : items_) {
    if (kv.first == key) {  // last definition wins, order preserved
      kv.second = std::move(v);
      return;
    }
  }
  items_.emplace_back(std::move(key), std::move(v));
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------
uint64_t TensorInfo::nbytes() const { return type_nbytes(type, n_elements); }

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------
std::string type_name(uint32_t t) {
  const TypeDesc d = type_desc(t);
  if (d.name) return d.name;
  return "unknown(" + std::to_string(t) + ")";
}

uint64_t type_block_size(uint32_t t) {
  const TypeDesc d = type_desc(t);
  if (!d.name)
    throw std::runtime_error("gguf: unknown ggml type " + std::to_string(t));
  return d.block_size;
}

uint64_t type_block_elems(uint32_t t) {
  const TypeDesc d = type_desc(t);
  if (!d.name)
    throw std::runtime_error("gguf: unknown ggml type " + std::to_string(t));
  return d.block_elems;
}

bool type_supported(uint32_t t) { return type_desc(t).name != nullptr; }

uint64_t type_nbytes(uint32_t t, uint64_t n_elements) {
  const TypeDesc d = type_desc(t);
  if (!d.name)
    throw std::runtime_error("gguf: unknown ggml type " + std::to_string(t));
  if (n_elements % d.block_elems != 0)
    throw std::runtime_error("gguf: " + std::string(d.name) + " tensor has " +
                             std::to_string(n_elements) +
                             " elements, which is not a multiple of the block size " +
                             std::to_string(d.block_elems));
  const uint64_t blocks = n_elements / d.block_elems;
  if (blocks > std::numeric_limits<uint64_t>::max() / d.block_size)
    throw std::runtime_error("gguf: tensor byte size overflows uint64");
  return blocks * d.block_size;
}

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------
struct File::Impl {
  ysknio::MappedFile map;
  std::string path;
  uint32_t version = 0;
  uint64_t alignment = 32;
  uint64_t data_base = 0;
  Metadata meta;
  std::vector<TensorInfo> tensors;
  std::unordered_map<std::string, size_t> index;

  void parse();
};

void File::Impl::parse() {
  if (map.size() == 0) throw std::runtime_error("gguf: " + path + ": file is empty");
  if (map.size() < 24)
    throw std::runtime_error("gguf: " + path + ": file is too small to hold a header");

  Reader r(map.data(), map.size(), path);

  const uint32_t magic = r.u32("magic");
  if (magic != 0x46554747u)  // "GGUF" little-endian
    r.fail("bad magic (not a GGUF file)");

  version = r.u32("version");
  if (version != 2 && version != 3)
    r.fail("unsupported GGUF version " + std::to_string(version) + " (need 2 or 3)");

  const uint64_t n_tensors = r.u64("tensor count");
  const uint64_t n_kv = r.u64("metadata count");

  // Sanity: the smallest possible KV is 8 (key length) + 4 (type) + 1 (value)
  // and the smallest tensor info is 8 + 4 + 8 + 4 + 8. Reject counts that
  // cannot possibly fit before allocating anything.
  if (n_kv > r.remaining() / 13 + 1) r.fail("metadata count exceeds file size");
  if (n_tensors > r.remaining() / 24 + 1) r.fail("tensor count exceeds file size");

  // meta holds exactly what the file declares: no synthetic defaults, so
  // meta().size() matches the header's n_kv and a caller can round-trip the
  // metadata faithfully. The alignment default lives in 'alignment' instead.
  meta = Metadata();

  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key = r.str("metadata key");
    const ValueType vt = static_cast<ValueType>(r.u32("metadata value type"));
    Value v = read_value(r, vt, 0);
    meta.insert(std::move(key), std::move(v));
  }

  // Alignment: general.alignment, default 32.
  if (const Value* al = meta.find("general.alignment")) {
    const uint64_t a = al->as_uint();
    if (a == 0) r.fail("general.alignment must be non-zero");
    alignment = a;
  } else {
    alignment = 32;
  }

  tensors.clear();
  tensors.reserve(static_cast<size_t>(std::min<uint64_t>(n_tensors, 1u << 16)));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo t;
    t.name = r.str("tensor name");
    const uint32_t n_dims = r.u32("tensor dimension count");
    if (n_dims > 8) r.fail("tensor '" + t.name + "' has an implausible dimension count");
    t.shape.reserve(n_dims);
    uint64_t n_elem = 1;
    for (uint32_t d = 0; d < n_dims; ++d) {
      const uint64_t dim = r.u64("tensor dimension");
      if (dim == 0) r.fail("tensor '" + t.name + "' has a zero-sized dimension");
      if (n_elem > std::numeric_limits<uint64_t>::max() / dim)
        r.fail("tensor '" + t.name + "' element count overflows uint64");
      n_elem *= dim;
      t.shape.push_back(dim);
    }
    t.type = r.u32("tensor type");
    t.offset = r.u64("tensor offset");
    t.n_elements = n_elem;
    tensors.push_back(std::move(t));
  }

  // Tensor data starts at the next multiple of the alignment after the tensor
  // info table. Files that only carry metadata (vocabulary-only GGUFs, which
  // llama.cpp ships as ggml-vocab-*.gguf) have no tensors and therefore no data
  // section at all: the aligned offset may legitimately sit at or past EOF.
  const uint64_t pos = r.pos();
  const uint64_t rem = pos % alignment;
  data_base = rem == 0 ? pos : pos + (alignment - rem);
  if (!tensors.empty() && data_base > map.size())
    throw std::runtime_error("gguf: " + path +
                             ": tensor data section starts past end of file");

  // Validate every tensor against the mapping so dequant() can never read out
  // of bounds.
  index.clear();
  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorInfo& t = tensors[i];
    if (!type_supported(t.type))
      throw std::runtime_error("gguf: " + path + ": tensor '" + t.name +
                               "' has unsupported ggml type " + type_name(t.type));
    const uint64_t nbytes = type_nbytes(t.type, t.n_elements);
    if (t.offset > map.size() - data_base ||
        nbytes > map.size() - data_base - t.offset)
      throw std::runtime_error("gguf: " + path + ": tensor '" + t.name +
                               "' payload extends past end of file");
    if (!index.emplace(t.name, i).second)
      throw std::runtime_error("gguf: " + path + ": duplicate tensor name '" +
                               t.name + "'");
  }
}

File::File() = default;
File::~File() = default;
File::File(File&&) noexcept = default;
File& File::operator=(File&&) noexcept = default;

void File::open(const std::string& path) {
  auto impl = std::unique_ptr<Impl>(new Impl());
  impl->path = path;
  impl->map.open(path);
  impl->parse();
  impl_ = std::move(impl);
}

uint32_t File::version() const { return impl_->version; }
uint64_t File::alignment() const { return impl_->alignment; }
uint64_t File::data_base() const { return impl_->data_base; }
uint64_t File::file_size() const { return impl_->map.size(); }
const std::string& File::path() const { return impl_->path; }
const Metadata& File::meta() const { return impl_->meta; }
const std::vector<TensorInfo>& File::tensors() const { return impl_->tensors; }

const TensorInfo* File::find(const std::string& name) const {
  const auto it = impl_->index.find(name);
  return it == impl_->index.end() ? nullptr : &impl_->tensors[it->second];
}

const uint8_t* File::tensor_data(const TensorInfo& t) const {
  const uint64_t nbytes = type_nbytes(t.type, t.n_elements);
  const uint64_t size = impl_->map.size();
  // data_base can equal size for a metadata-only file; guard the subtraction.
  if (impl_->data_base > size || t.offset > size - impl_->data_base ||
      nbytes > size - impl_->data_base - t.offset)
    throw std::runtime_error("gguf: tensor '" + t.name +
                             "' payload extends past end of file");
  return impl_->map.data() + impl_->data_base + t.offset;
}

uint64_t File::tensor_nbytes(const TensorInfo& t) const {
  return type_nbytes(t.type, t.n_elements);
}

void File::dequant(const TensorInfo& t, float* out) const {
  if (out == nullptr && t.n_elements != 0)
    throw std::runtime_error("gguf: dequant target buffer is null");
  const uint8_t* src = tensor_data(t);
  const uint64_t n = t.n_elements;
  switch (t.type) {
    case TYPE_F32:   deq_f32(src, out, n); break;
    case TYPE_F16:   deq_f16(src, out, n); break;
    case TYPE_BF16:  deq_bf16(src, out, n); break;
    case TYPE_Q4_0:  deq_q4_0(src, out, n); break;
    case TYPE_Q4_1:  deq_q4_1(src, out, n); break;
    case TYPE_Q5_0:  deq_q5_0(src, out, n); break;
    case TYPE_Q5_1:  deq_q5_1(src, out, n); break;
    case TYPE_Q8_0:  deq_q8_0(src, out, n); break;
    case TYPE_Q2_K:  deq_q2_K(src, out, n); break;
    case TYPE_Q3_K:  deq_q3_K(src, out, n); break;
    case TYPE_Q4_K:  deq_q4_K(src, out, n); break;
    case TYPE_Q5_K:  deq_q5_K(src, out, n); break;
    case TYPE_Q6_K:  deq_q6_K(src, out, n); break;
    case TYPE_Q8_K:  deq_q8_K(src, out, n); break;
    case TYPE_IQ4_NL: deq_iq4_nl(src, out, n); break;
    default:
      throw std::runtime_error("gguf: cannot dequantize tensor '" + t.name +
                               "' of type " + type_name(t.type));
  }
}

std::vector<float> File::dequant(const TensorInfo& t) const {
  std::vector<float> out(static_cast<size_t>(t.n_elements));
  dequant(t, out.data());
  return out;
}

void File::dequant(const std::string& name, float* out) const {
  const TensorInfo* t = find(name);
  if (!t) throw std::runtime_error("gguf: no tensor named '" + name + "'");
  dequant(*t, out);
}

std::vector<float> File::dequant(const std::string& name) const {
  const TensorInfo* t = find(name);
  if (!t) throw std::runtime_error("gguf: no tensor named '" + name + "'");
  return dequant(*t);
}

}  // namespace yskn::gguf
