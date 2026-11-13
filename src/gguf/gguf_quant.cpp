// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF -> engine tensor bridge, host-side encoder. See gguf_quant.hpp for the
// contract and docs/GGUF_DESIGN.md §5.4 for the pipeline this implements.
#include "gguf_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "signs.hpp"  // yskn::sign_stream_rule, yskn::splitmix64

namespace yskn::gguf {
namespace {

constexpr float kInvSqrt512 = 0.044194173824159216f;  // 1/sqrt(512), src/a8_prologue.cuh:49
constexpr float kInf = std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// Config validation. bits is checked here rather than at every call site
// because lnord_build instantiates only K in {4,6,8} (src/lnord.cu:82-86).
// ---------------------------------------------------------------------------
void validate_config(const QuantConfig& cfg) {
  if (cfg.bits != 4 && cfg.bits != 6 && cfg.bits != 8) {
    throw std::invalid_argument("QuantConfig::bits must be 4, 6 or 8 (lnord_build accepts only those)");
  }
  // sign_stream_rule throws on an unknown rule; n = 0 makes it a cheap probe.
  (void)sign_stream_rule(cfg.sign_rule, cfg.sign_seed, 0, 0);
}

// ---------------------------------------------------------------------------
// IEEE binary16 (host side). Round-to-nearest-even, matching __float2half_rn.
// ---------------------------------------------------------------------------
}  // namespace

uint16_t f32_to_f16(float v) {
  uint32_t x = 0;
  std::memcpy(&x, &v, sizeof(x));
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t exp = (x >> 23) & 0xFFu;
  const uint32_t mant = x & 0x7FFFFFu;

  if (exp == 0xFFu) {  // inf / nan
    const uint16_t payload = mant ? static_cast<uint16_t>(0x200u | (mant >> 13)) : 0u;
    return static_cast<uint16_t>(sign | 0x7C00u | payload);
  }
  int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
  if (e <= 0) {                                                 // zero or subnormal
    if (e < -10) return sign;                                   // underflows to zero
    const uint32_t full = mant | 0x800000u;                     // 24-bit significand
    const int shift = 14 - e;                                   // 14..24
    uint32_t m = full >> shift;
    const uint32_t rem = full & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m & 1u))) ++m;
    return static_cast<uint16_t>(sign | m);
  }
  uint32_t m = mant >> 13;
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
    if (++m == 0x400u) {  // carry into the exponent
      m = 0;
      if (++e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(e) << 10) | m);
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +-0
    } else {        // subnormal: renormalise
      int e = -1;
      uint32_t m = mant;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

namespace {

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int clamp_i127(float v) {
  const float r = std::nearbyint(v);
  return r < -127.f ? -127 : (r > 127.f ? 127 : static_cast<int>(r));
}

// LSB-first bit writer over a zero-initialised word array (§2.2).
inline void write_bits(uint32_t* w, int pos, int width, uint32_t v) {
  const int idx = pos >> 5, sh = pos & 31;
  w[idx] |= (v & ((1u << width) - 1u)) << sh;
  if (sh + width > 32) w[idx + 1] |= v >> (32 - sh);
}

// Same read_window as docs/GGUF_DESIGN.md §2.2.
inline uint32_t read_window(const uint32_t* w, int start, int width) {
  const int idx = start >> 5, sh = start & 31;
  uint32_t v = w[idx] >> sh;
  if (sh + width > 32) v |= w[idx + 1] << (32 - sh);
  return v & ((1u << width) - 1u);
}

// The 12-bit trellis window ending at bit bits*(j+1), earliest bit least
// significant -- identical to v2_state<K> (src/v2_decode.cuh:13-19).
inline uint32_t v2_state_host(const uint32_t* w, int bits, int j) {
  const int e = bits * (j + 1);
  if (e < kLutBits) return (w[0] & ((1u << e) - 1u)) << (kLutBits - e);
  return read_window(w, e - kLutBits, kLutBits);
}

// layout:"warp" source index, straight out of k_repack (src/repack.cu:13-14).
inline size_t warp_src(int row, int seq, int word, int rows, int wps) {
  const size_t q = static_cast<size_t>(seq) * rows + row;
  const size_t tile = q / 32, lane = q % 32;
  return ((tile * (wps / 4) + static_cast<size_t>(word / 4)) * 32 + lane) * 4 +
         static_cast<size_t>(word % 4);
}

// dst[c*h + r] = src[r*ld + x0 + c], 32x32 blocked so the strided side stays in
// cache. Used to run the row-direction Hadamard as a contiguous row transform.
void transpose_into(const float* src, int ld, int x0, int w, int h, float* dst) {
  constexpr int T = 32;
  for (int r0 = 0; r0 < h; r0 += T) {
    const int rh = std::min(T, h - r0);
    for (int c0 = 0; c0 < w; c0 += T) {
      const int cw = std::min(T, w - c0);
      for (int r = 0; r < rh; ++r) {
        const float* s = src + static_cast<size_t>(r0 + r) * ld + x0 + c0;
        float* d = dst + static_cast<size_t>(c0) * h + (r0 + r);
        for (int c = 0; c < cw; ++c) d[static_cast<size_t>(c) * h] = s[c];
      }
    }
  }
}

// Inverse of transpose_into: dst[r*ld + x0 + c] = src[c*h + r].
void transpose_out(const float* src, int w, int h, float* dst, int ld, int x0) {
  constexpr int T = 32;
  for (int r0 = 0; r0 < h; r0 += T) {
    const int rh = std::min(T, h - r0);
    for (int c0 = 0; c0 < w; c0 += T) {
      const int cw = std::min(T, w - c0);
      for (int r = 0; r < rh; ++r) {
        float* d = dst + static_cast<size_t>(r0 + r) * ld + x0 + c0;
        const float* s = src + static_cast<size_t>(c0) * h + (r0 + r);
        for (int c = 0; c < cw; ++c) d[c] = s[static_cast<size_t>(c) * h];
      }
    }
  }
}

// (1/sqrt(512)) * H over each 512-row block, every column. Whole blocks only;
// a ragged tail block is left untouched (see rotate_matrix's note).
void hadamard_over_rows(float* A, int rows, int cols) {
  const int blocks = rows / kHadamardBlock;
  if (blocks <= 0) return;
  std::vector<float> tile(static_cast<size_t>(kHadamardBlock) * kHadamardBlock);
  for (int ob = 0; ob < blocks; ++ob) {
    float* base = A + static_cast<size_t>(ob) * kHadamardBlock * cols;
    for (int c0 = 0; c0 < cols; c0 += kHadamardBlock) {
      const int cw = std::min(kHadamardBlock, cols - c0);
      transpose_into(base, cols, c0, cw, kHadamardBlock, tile.data());
      for (int c = 0; c < cw; ++c) {
        float* v = tile.data() + static_cast<size_t>(c) * kHadamardBlock;
        hadamard512(v);
        for (int i = 0; i < kHadamardBlock; ++i) v[i] *= kInvSqrt512;
      }
      transpose_out(tile.data(), cw, kHadamardBlock, base, cols, c0);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Hadamard, natural order. Same loop nest as fwht512_blocks
// (src/a8_prologue.cuh:41-48), just single threaded and unnormalised.
// ---------------------------------------------------------------------------
void hadamard512(float* x) {
  for (int h = 1; h < kHadamardBlock; h <<= 1) {
    const int hm = h - 1;
    for (int k = 0; k < kHadamardBlock / 2; ++k) {
      const int lo = ((k & ~hm) << 1) | (k & hm);
      const float a = x[lo], b = x[lo + h];
      x[lo] = a + b;
      x[lo + h] = a - b;
    }
  }
}

// ---------------------------------------------------------------------------
// Step 3.
// ---------------------------------------------------------------------------
void rotate_matrix(const float* W, int rows, int cols, const QuantConfig& cfg, float* A) {
  validate_config(cfg);
  if (!W || !A) throw std::invalid_argument("rotate_matrix: null buffer");
  if (rows <= 0 || cols <= 0) throw std::invalid_argument("rotate_matrix: empty matrix");

  const size_t n = static_cast<size_t>(rows) * cols;
  // 1. input signs (every column, including any ragged tail).
  if (cfg.use_rot_in) {
    const std::vector<int8_t> si = sign_stream_rule(cfg.sign_rule, cfg.sign_seed, 0, cols);
    for (int r = 0; r < rows; ++r) {
      const float* src = W + static_cast<size_t>(r) * cols;
      float* dst = A + static_cast<size_t>(r) * cols;
      for (int c = 0; c < cols; ++c) dst[c] = src[c] * static_cast<float>(si[c]);
    }
  } else {
    std::memcpy(A, W, n * sizeof(float));
  }

  // 2. input Hadamard over each full 512-column block, per row.
  if (cfg.use_rot_in) {
    const int blocks = cols / kHadamardBlock;
    for (int r = 0; r < rows; ++r) {
      float* row = A + static_cast<size_t>(r) * cols;
      for (int b = 0; b < blocks; ++b) {
        float* blk = row + static_cast<size_t>(b) * kHadamardBlock;
        hadamard512(blk);
        for (int i = 0; i < kHadamardBlock; ++i) blk[i] *= kInvSqrt512;
      }
    }
  }

  // 3. output Hadamard over each full 512-row block, every column.
  if (cfg.use_rot_out) hadamard_over_rows(A, rows, cols);

  // 4. output signs.
  if (cfg.use_rot_out) {
    const std::vector<int8_t> so = sign_stream_rule(cfg.sign_rule, cfg.sign_seed, cols, rows);
    for (int r = 0; r < rows; ++r) {
      const float s = static_cast<float>(so[r]);
      float* row = A + static_cast<size_t>(r) * cols;
      for (int c = 0; c < cols; ++c) row[c] *= s;
    }
  }
}

// ---------------------------------------------------------------------------
// Step 5. Lloyd's algorithm with a deterministic sample init.
//
// GPU: the nearest-centroid scan below is O(pairs * 4096) and dominates the
// converter. Port it as one block per point batch with the 4096 centroids in
// shared memory (32 KiB), or replace it with a 64x64 bucket grid on the host.
// ---------------------------------------------------------------------------
Codebook train_codebook(const float* B, size_t n, int seed) {
  if (!B) throw std::invalid_argument("train_codebook: null buffer");
  const size_t np = n / 2;
  if (np == 0) throw std::invalid_argument("train_codebook: need at least one pair");

  std::vector<float> cx(kLutSize), cy(kLutSize);

  // Init: draw kLutSize pairs from the sample, rejecting duplicates through a
  // small open-addressing set (a duplicate centroid just goes empty later).
  {
    uint64_t st = static_cast<uint64_t>(static_cast<uint32_t>(seed)) * 0x9E3779B97F4A7C15ull;
    std::vector<uint32_t> used(8192, 0u);
    auto probe = [&used](uint32_t idx) {
      uint32_t h = (idx * 2654435761u) & 8191u;
      while (used[h] != 0u && used[h] != idx + 1u) h = (h + 1u) & 8191u;
      return h;
    };
    for (int k = 0; k < kLutSize; ++k) {
      size_t idx = 0;
      for (int attempt = 0; attempt < 8; ++attempt) {
        idx = static_cast<size_t>(splitmix64(st) % np);
        const uint32_t h = probe(static_cast<uint32_t>(idx));
        if (used[h] == 0u) {
          used[h] = static_cast<uint32_t>(idx) + 1u;
          break;
        }
      }
      cx[k] = B[2 * idx];
      cy[k] = B[2 * idx + 1];
    }
  }

  // Lloyd. Fixed iteration cap plus an early exit on a stable partition keeps
  // the result a deterministic function of (B, n, seed).
  constexpr int kIters = 10;
  std::vector<uint32_t> assign(np, 0u);
  std::vector<double> sx(kLutSize), sy(kLutSize);
  std::vector<uint32_t> cnt(kLutSize);
  for (int it = 0; it < kIters; ++it) {
    size_t changed = 0;
    for (size_t t = 0; t < np; ++t) {
      const float x = B[2 * t], y = B[2 * t + 1];
      float best = kInf;
      int bi = 0;
      const float* px = cx.data();
      const float* py = cy.data();
      for (int k = 0; k < kLutSize; ++k) {
        const float dx = x - px[k];
        const float dxx = dx * dx;
        if (dxx >= best) continue;  // dy*dy >= 0, so this cannot win
        const float dy = y - py[k];
        const float d = dxx + dy * dy;
        if (d < best) {  // strict: first minimum wins, keeps ties deterministic
          best = d;
          bi = k;
        }
      }
      if (assign[t] != static_cast<uint32_t>(bi)) {
        assign[t] = static_cast<uint32_t>(bi);
        ++changed;
      }
    }
    std::fill(sx.begin(), sx.end(), 0.0);
    std::fill(sy.begin(), sy.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0u);
    for (size_t t = 0; t < np; ++t) {
      const uint32_t k = assign[t];
      sx[k] += B[2 * t];
      sy[k] += B[2 * t + 1];
      ++cnt[k];
    }
    for (int k = 0; k < kLutSize; ++k) {
      if (cnt[k] == 0) continue;  // empty cluster keeps its previous centroid
      const double inv = 1.0 / static_cast<double>(cnt[k]);
      cx[k] = static_cast<float>(sx[k] * inv);
      cy[k] = static_cast<float>(sy[k] * inv);
    }
    if (it > 0 && changed == 0) break;
  }

  // Rescale so max|q| == 127 exactly: that invariant is what makes
  // build_lut8_tau invertible (§3.2).
  Codebook cb;
  cb.q0.assign(kLutSize, 0);
  cb.q1.assign(kLutSize, 0);
  float mx = 0.f;
  int arg = 0, which = 0;  // which: 0 -> q0, 1 -> q1
  for (int k = 0; k < kLutSize; ++k) {
    const float ax = std::fabs(cx[k]), ay = std::fabs(cy[k]);
    if (ax > mx) { mx = ax; arg = k; which = 0; }
    if (ay > mx) { mx = ay; arg = k; which = 1; }
  }
  if (mx > 0.f) {
    cb.gamma = mx / 127.f;
    const float inv = 1.f / cb.gamma;
    for (int k = 0; k < kLutSize; ++k) {
      cb.q0[k] = static_cast<int8_t>(clamp_i127(cx[k] * inv));
      cb.q1[k] = static_cast<int8_t>(clamp_i127(cy[k] * inv));
    }
    // Force the extreme entry to +-127 so float rounding cannot break the
    // invariant (round(127.0) already does, this is belt and braces).
    const float* c = which == 0 ? cx.data() : cy.data();
    int8_t& q = which == 0 ? cb.q0[arg] : cb.q1[arg];
    q = static_cast<int8_t>(c[arg] < 0.f ? -127 : 127);
  } else {
    // Degenerate all-zero input: keep the invariant with a single +127 entry.
    cb.gamma = 1.f / 127.f;
    cb.q0[0] = 127;
  }
  return cb;
}

// ---------------------------------------------------------------------------
// Step 6+7.
//
// GPU: one sequence per warp, dp[4096] + arg[128][2^(12-bits)] in shared
// memory, __shfl reductions for the per-group min. That is the only way this
// scales to a 27B model (13.5e9 steps x 4096 states).
// ---------------------------------------------------------------------------
void viterbi_quantize(const float* B, int rows, int cols, const Codebook& cb, int bits,
                      std::vector<uint32_t>& words) {
  if (bits != 4 && bits != 6 && bits != 8) {
    throw std::invalid_argument("viterbi_quantize: bits must be 4, 6 or 8");
  }
  if (!B) throw std::invalid_argument("viterbi_quantize: null buffer");
  if (rows <= 0 || cols <= 0) throw std::invalid_argument("viterbi_quantize: empty matrix");
  if (cols % kSeqCols != 0) throw std::invalid_argument("viterbi_quantize: cols must be a multiple of 256");
  if (static_cast<int>(cb.q0.size()) != kLutSize || static_cast<int>(cb.q1.size()) != kLutSize) {
    throw std::invalid_argument("viterbi_quantize: codebook must hold 4096 entries");
  }

  const int seqs = cols / kSeqCols;
  const int wps = 4 * bits;
  const int ngroups = 1 << bits;           // predecessors per state
  const int nbase = 1 << (kLutBits - bits);  // distinct predecessor groups
  const uint32_t lowmask = static_cast<uint32_t>(nbase - 1);
  const int top_shift = kLutBits - bits;

  words.assign(static_cast<size_t>(rows) * seqs * wps, 0u);

  // Float copies of the codebook, built once for the whole tensor.
  std::vector<float> fq0(kLutSize), fq1(kLutSize);
  for (int s = 0; s < kLutSize; ++s) {
    fq0[s] = static_cast<float>(cb.q0[s]);
    fq1[s] = static_cast<float>(cb.q1[s]);
  }

  std::vector<float> dp(kLutSize), ndp(kLutSize), cost(kLutSize), best(nbase);
  std::vector<uint8_t> arg(static_cast<size_t>(kStepsPerSeq) * nbase);
  std::vector<uint8_t> gbits(kStepsPerSeq);

  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < seqs; ++q) {
      const float* b = B + static_cast<size_t>(r) * cols + static_cast<size_t>(q) * kSeqCols;

      // dp over step -1: only state 0 is reachable, which reproduces the
      // zero-padded prefix state of §2.2 for free.
      std::fill(dp.begin(), dp.end(), kInf);
      dp[0] = 0.f;

      for (int j = 0; j < kStepsPerSeq; ++j) {
        const float b0 = b[2 * j], b1 = b[2 * j + 1];

        // Every state s' has the same 2^bits predecessors, base+t with
        // base = s' & lowmask, so one min per base serves all of them.
        for (int g = 0; g < nbase; ++g) {
          const float* d = dp.data() + static_cast<size_t>(g) * ngroups;
          float m = d[0];
          int a = 0;
          for (int t = 1; t < ngroups; ++t) {
            if (d[t] < m) { m = d[t]; a = t; }
          }
          best[g] = m;
          arg[static_cast<size_t>(j) * nbase + g] = static_cast<uint8_t>(a);
        }

        for (int s = 0; s < kLutSize; ++s) {
          const float d0 = b0 - fq0[s], d1 = b1 - fq1[s];
          cost[s] = d0 * d0 + d1 * d1;
        }
        for (int s = 0; s < kLutSize; ++s) {
          ndp[s] = best[static_cast<uint32_t>(s) & lowmask] + cost[s];
        }
        dp.swap(ndp);
      }

      // Best final state, then walk the argmin table backwards.
      int s = 0;
      float m = dp[0];
      for (int t = 1; t < kLutSize; ++t) {
        if (dp[t] < m) { m = dp[t]; s = t; }
      }
      for (int j = kStepsPerSeq - 1; j >= 0; --j) {
        gbits[j] = static_cast<uint8_t>(static_cast<uint32_t>(s) >> top_shift);
        const int base = s & static_cast<int>(lowmask);
        s = (base << bits) | arg[static_cast<size_t>(j) * nbase + base];
      }

      uint32_t* w = words.data() + (static_cast<size_t>(r) * seqs + q) * wps;
      for (int j = 0; j < kStepsPerSeq; ++j) {
        write_bits(w, bits * j, bits, gbits[j]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Step 7b.
// ---------------------------------------------------------------------------
void interleave_warp(const std::vector<uint32_t>& words, int rows, int seqs, int wps,
                     std::vector<uint32_t>& frag) {
  if (rows <= 0 || seqs <= 0 || wps <= 0) throw std::invalid_argument("interleave_warp: empty shape");
  if (rows % kTileRows != 0) throw std::invalid_argument("interleave_warp: rows must be a multiple of 64");
  if (wps % 4 != 0) throw std::invalid_argument("interleave_warp: wps must be a multiple of 4");
  const size_t total = static_cast<size_t>(rows) * seqs * wps;
  if (words.size() != total) throw std::invalid_argument("interleave_warp: word count mismatch");

  frag.assign(total, 0u);
  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < seqs; ++q) {
      const uint32_t* src = words.data() + (static_cast<size_t>(r) * seqs + q) * wps;
      for (int w = 0; w < wps; ++w) frag[warp_src(r, q, w, rows, wps)] = src[w];
    }
  }
}

// ---------------------------------------------------------------------------
// Steps 3..8 chained.
// ---------------------------------------------------------------------------
QuantizedTensor quantize_tensor(const float* W, int rows, int cols, const QuantConfig& cfg,
                                const Codebook* shared) {
  validate_config(cfg);
  if (!W) throw std::invalid_argument("quantize_tensor: null weights");
  if (rows <= 0 || cols <= 0) throw std::invalid_argument("quantize_tensor: empty matrix");
  if (cols % kSeqCols != 0) throw std::invalid_argument("quantize_tensor: cols must be a multiple of 256");
  if (shared && (static_cast<int>(shared->q0.size()) != kLutSize ||
                 static_cast<int>(shared->q1.size()) != kLutSize)) {
    throw std::invalid_argument("quantize_tensor: shared codebook must hold 4096 entries");
  }

  const int bits = cfg.bits;
  const int physical_rows = ((rows + kTileRows - 1) / kTileRows) * kTileRows;
  const int seqs = cols / kSeqCols;
  const int wps = 4 * bits;

  // A doubles as B: rotate into A, then normalise each row in place.
  std::vector<float> A(static_cast<size_t>(physical_rows) * cols, 0.f);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(A.data() + static_cast<size_t>(r) * cols, W + static_cast<size_t>(r) * cols,
                static_cast<size_t>(cols) * sizeof(float));
  }
  rotate_matrix(A.data(), physical_rows, cols, cfg, A.data());

  // Step 4: per-row scale on the rotated matrix (§4.4 -- not on W).
  std::vector<float> tau(physical_rows, 0.f);
  for (int r = 0; r < physical_rows; ++r) {
    float* row = A.data() + static_cast<size_t>(r) * cols;
    float mx = 0.f;
    for (int c = 0; c < cols; ++c) mx = std::max(mx, std::fabs(row[c]));
    if (mx > 0.f) {
      tau[r] = mx / 127.f;
      const float inv = 1.f / tau[r];
      for (int c = 0; c < cols; ++c) row[c] *= inv;
    } else {
      // All-zero row (padding, or a dead row): keep B finite and tau == 0 so
      // the reader contributes nothing for it.
      tau[r] = 0.f;
      std::fill(row, row + cols, 0.f);
    }
  }

  // Step 5: codebook. Train on the real rows only -- the zero padding would
  // otherwise dominate the k-means with a giant (0,0) cluster.
  Codebook cb;
  if (shared) {
    cb = *shared;
  } else {
    const int seed = static_cast<int>((cfg.sign_seed ^ (cfg.sign_seed >> 32)) & 0x7FFFFFFFull);
    cb = train_codebook(A.data(), static_cast<size_t>(rows) * cols, seed);
  }
  if (!(cb.gamma > 0.f) || !std::isfinite(cb.gamma)) {
    throw std::invalid_argument("quantize_tensor: codebook gamma must be finite and positive");
  }

  // Step 6+7: Viterbi over every physical row, then the warp interleave.
  std::vector<uint32_t> logical;
  viterbi_quantize(A.data(), physical_rows, cols, cb, bits, logical);

  QuantizedTensor t;
  t.rows = rows;
  t.cols = cols;
  t.physical_rows = physical_rows;
  t.bits = bits;
  t.seqs = seqs;
  t.wps = wps;
  t.gamma = cb.gamma;
  interleave_warp(logical, physical_rows, seqs, wps, t.stream);

  // Step 8: lut16 = f16(q * gamma), scale16 = f16(tau / gamma). Because
  // max|q| == 127, build_lut8_tau recovers both exactly (§3.2).
  t.lut16.resize(2 * kLutSize);
  for (int s = 0; s < kLutSize; ++s) {
    t.lut16[2 * s] = f32_to_f16(static_cast<float>(cb.q0[s]) * cb.gamma);
    t.lut16[2 * s + 1] = f32_to_f16(static_cast<float>(cb.q1[s]) * cb.gamma);
  }
  t.scale16.resize(physical_rows);
  for (int r = 0; r < physical_rows; ++r) {
    t.scale16[r] = f32_to_f16(tau[r] / cb.gamma);
  }
  return t;
}

// ---------------------------------------------------------------------------
// §5.5 oracle: decode the stream the way the kernels would and diff it
// against the normalised matrix.
// ---------------------------------------------------------------------------
bool verify_roundtrip(const QuantizedTensor& t, const float* B, float* max_abs_err, float* rel_err) {
  auto fail = [&]() {
    if (max_abs_err) *max_abs_err = kInf;
    if (rel_err) *rel_err = kInf;
    return false;
  };
  if (!B) return fail();
  if (t.rows <= 0 || t.cols <= 0 || t.physical_rows <= 0) return fail();
  if (t.cols % kSeqCols != 0) return fail();
  if (t.seqs != t.cols / kSeqCols) return fail();
  if (t.wps != 4 * t.bits) return fail();
  if (static_cast<int>(t.lut16.size()) != 2 * kLutSize) return fail();
  if (static_cast<int>(t.scale16.size()) != t.physical_rows) return fail();
  if (t.stream.size() != static_cast<size_t>(t.physical_rows) * t.seqs * t.wps) return fail();

  // build_lut8_tau (src/loader.cpp:19-25), replayed on the host.
  float mx = 0.f;
  for (int i = 0; i < 2 * kLutSize; ++i) mx = std::max(mx, std::fabs(f16_to_f32(t.lut16[i])));
  if (!(mx > 0.f)) return fail();
  const float gamma = mx / 127.f;
  std::vector<int8_t> q0(kLutSize), q1(kLutSize);
  for (int s = 0; s < kLutSize; ++s) {
    q0[s] = static_cast<int8_t>(clamp_i127(f16_to_f32(t.lut16[2 * s]) / gamma));
    q1[s] = static_cast<int8_t>(clamp_i127(f16_to_f32(t.lut16[2 * s + 1]) / gamma));
  }
  std::vector<float> tau(t.physical_rows);
  for (int r = 0; r < t.physical_rows; ++r) tau[r] = f16_to_f32(t.scale16[r]) * gamma;

  std::vector<uint32_t> w(t.wps);
  float worst = 0.f, bmax = 0.f;
  for (int r = 0; r < t.rows; ++r) {
    for (int q = 0; q < t.seqs; ++q) {
      for (int i = 0; i < t.wps; ++i) w[i] = t.stream[warp_src(r, q, i, t.physical_rows, t.wps)];
      const float s = tau[r];
      const float* b = B + static_cast<size_t>(r) * t.cols + static_cast<size_t>(q) * kSeqCols;
      for (int j = 0; j < kStepsPerSeq; ++j) {
        const uint32_t st = v2_state_host(w.data(), t.bits, j);
        const float v0 = static_cast<float>(q0[st]) * s;
        const float v1 = static_cast<float>(q1[st]) * s;
        const float e0 = std::fabs(v0 - b[2 * j]);
        const float e1 = std::fabs(v1 - b[2 * j + 1]);
        worst = std::max(worst, std::max(e0, e1));
        bmax = std::max(bmax, std::max(std::fabs(b[2 * j]), std::fabs(b[2 * j + 1])));
      }
    }
  }
  if (max_abs_err) *max_abs_err = worst;
  if (rel_err) *rel_err = bmax > 0.f ? worst / bmax : (worst > 0.f ? kInf : 0.f);
  return true;
}

}  // namespace yskn::gguf
