// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for the GGUF -> engine tensor bridge (src/gguf/gguf_quant.cpp).
//
// Everything here is CPU only: no CUDA, no GGUF fixture, no GPU. The suite
// checks the encoder against the *decoder* it targets --
// the round-trip oracle -- by re-implementing the two pieces
// of reader arithmetic the encoder must be exactly compatible with:
//
//   build_lut8_tau   src/loader.cpp:19-25   (f16 LUT -> int8 LUT + gamma)
//   v2_state         src/v2_decode.cuh:13-19 (12-bit trellis window)
//   k_repack         src/repack.cu:10-18     (layout:"warp" permutation)
//
// Exits non-zero on failure and prints a PASS/FAIL summary.
#include "gguf_quant.hpp"
#include "signs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using yskn::gguf::Codebook;
using yskn::gguf::QuantConfig;
using yskn::gguf::QuantizedTensor;

int g_checks = 0;
int g_failures = 0;
std::vector<std::string> g_failure_log;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    g_failure_log.push_back(what);
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void check_lt(double got, double bound, const std::string& what) {
  ++g_checks;
  if (!(got < bound)) {
    ++g_failures;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (got %.6g, want < %.6g)", what.c_str(), got, bound);
    g_failure_log.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

void check_ge(double got, double bound, const std::string& what) {
  ++g_checks;
  if (!(got >= bound)) {
    ++g_failures;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (got %.6g, want >= %.6g)", what.c_str(), got, bound);
    g_failure_log.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

void check_eq(long long got, long long want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (got %lld, want %lld)", what.c_str(), got, want);
    g_failure_log.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

// ---------------------------------------------------------------------------
// Deterministic RNG, the same splitmix64 the engine uses for its sign streams.
// ---------------------------------------------------------------------------
struct Rng {
  uint64_t st;
  explicit Rng(uint64_t seed) : st(seed) {}
  uint64_t next() {
    st += 0x9E3779B97F4A7C15ull;
    uint64_t z = st;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  float uniform() { return static_cast<float>(next() >> 40) * (1.f / 16777216.f); }  // [0,1)
  float sym() { return uniform() * 2.f - 1.f; }                                     // [-1,1)
};

std::vector<float> random_matrix(size_t n, uint64_t seed, float scale = 1.f) {
  Rng rng(seed);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = rng.sym() * scale;
  return v;
}

float max_abs(const std::vector<float>& v) {
  float m = 0.f;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

// ---------------------------------------------------------------------------
// Reader arithmetic, re-implemented from the sources (not linked).
// ---------------------------------------------------------------------------

// build_lut8_tau, src/loader.cpp:19-25, verbatim semantics.
struct Lut8Tau {
  std::vector<int8_t> q0, q1;
  std::vector<float> tau;
  float gamma = 0.f;
};

Lut8Tau build_lut8_tau(const std::vector<uint16_t>& lut16, const std::vector<uint16_t>& s16,
                       int rows) {
  const int S = 1 << 12;
  Lut8Tau out;
  float mx = 0.f;
  for (int i = 0; i < S * 2; ++i) mx = std::max(mx, std::fabs(yskn::gguf::f16_to_f32(lut16[i])));
  out.gamma = mx / 127.f;
  out.q0.resize(S);
  out.q1.resize(S);
  for (int s = 0; s < S; ++s) {
    int q0 = static_cast<int>(
        std::nearbyint(yskn::gguf::f16_to_f32(lut16[2 * s]) / out.gamma));
    int q1 = static_cast<int>(
        std::nearbyint(yskn::gguf::f16_to_f32(lut16[2 * s + 1]) / out.gamma));
    q0 = std::max(-127, std::min(127, q0));
    q1 = std::max(-127, std::min(127, q1));
    out.q0[s] = static_cast<int8_t>(q0);
    out.q1[s] = static_cast<int8_t>(q1);
  }
  out.tau.resize(rows);
  for (int r = 0; r < rows; ++r) out.tau[r] = yskn::gguf::f16_to_f32(s16[r]) * out.gamma;
  return out;
}

// v2_state, src/v2_decode.cuh:13-19.
uint32_t v2_state(const uint32_t* w, int bits, int j) {
  const int e = bits * (j + 1);
  if (e < 12) return (w[0] & ((1u << e) - 1u)) << (12 - e);
  const int start = e - 12, word = start >> 5, sh = start & 31;
  if (sh + 12 <= 32) return (w[word] >> sh) & 0xFFFu;
  return ((w[word] >> sh) | (w[word + 1] << (32 - sh))) & 0xFFFu;
}

// k_repack's source index, src/repack.cu:13-14.
size_t warp_src(int row, int seq, int word, int rows, int wps) {
  const size_t q = static_cast<size_t>(seq) * rows + row;
  const size_t tile = q / 32, lane = q % 32;
  return ((tile * (wps / 4) + static_cast<size_t>(word / 4)) * 32 + lane) * 4 +
         static_cast<size_t>(word % 4);
}

// Inverse of interleave_warp, written here from src/repack.cu:10-18.
void deinterleave(const std::vector<uint32_t>& frag, int rows, int seqs, int wps,
                  std::vector<uint32_t>& words) {
  words.assign(frag.size(), 0u);
  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < seqs; ++q) {
      for (int w = 0; w < wps; ++w) {
        words[(static_cast<size_t>(r) * seqs + q) * wps + w] = frag[warp_src(r, q, w, rows, wps)];
      }
    }
  }
}

// Decode one sequence of the logical stream back to floats, exactly the way the
// kernels do: int8 codebook from build_lut8_tau, per-row tau, 12-bit windows.
void decode_sequence(const std::vector<uint32_t>& logical, int row, int seq, int seqs, int wps,
                     int bits, const Lut8Tau& lut, float tau, std::vector<float>& out) {
  const uint32_t* w = logical.data() + (static_cast<size_t>(row) * seqs + seq) * wps;
  out.resize(256);
  for (int j = 0; j < 128; ++j) {
    const uint32_t s = v2_state(w, bits, j);
    out[2 * j] = static_cast<float>(lut.q0[s]) * tau;
    out[2 * j + 1] = static_cast<float>(lut.q1[s]) * tau;
  }
}

// The exact inverse of rotate_matrix:
//   A = S_out . Hn . W . Hn . S_in   =>   W = S_in . Hn . A . Hn . S_out
// Same two Hadamards, mirrored order, sign roles swapped. Used only by the
// test; the encoder never needs it.
void invert_rotate(const float* A, int rows, int cols, const QuantConfig& cfg, float* W) {
  const size_t n = static_cast<size_t>(rows) * cols;
  std::memcpy(W, A, n * sizeof(float));
  if (cfg.use_rot_out) {
    // undo the output signs, then the output Hadamard
    const std::vector<int8_t> so = yskn::sign_stream_rule(cfg.sign_rule, cfg.sign_seed, cols, rows);
    for (int r = 0; r < rows; ++r) {
      const float s = static_cast<float>(so[r]);
      float* row = W + static_cast<size_t>(r) * cols;
      for (int c = 0; c < cols; ++c) row[c] *= s;
    }
    for (int r0 = 0; r0 + yskn::gguf::kHadamardBlock <= rows; r0 += yskn::gguf::kHadamardBlock) {
      for (int c = 0; c < cols; ++c) {
        std::vector<float> v(yskn::gguf::kHadamardBlock);
        for (int i = 0; i < yskn::gguf::kHadamardBlock; ++i) v[i] = W[static_cast<size_t>(r0 + i) * cols + c];
        yskn::gguf::hadamard512(v.data());
        for (int i = 0; i < yskn::gguf::kHadamardBlock; ++i) W[static_cast<size_t>(r0 + i) * cols + c] = v[i] * yskn::gguf::kInvSqrt512;
      }
    }
  }
  if (cfg.use_rot_in) {
    for (int r = 0; r < rows; ++r) {
      float* row = W + static_cast<size_t>(r) * cols;
      for (int b = 0; b + yskn::gguf::kHadamardBlock <= cols; b += yskn::gguf::kHadamardBlock) {
        float* blk = row + b;
        yskn::gguf::hadamard512(blk);
        for (int i = 0; i < yskn::gguf::kHadamardBlock; ++i) blk[i] *= yskn::gguf::kInvSqrt512;
      }
    }
    const std::vector<int8_t> si = yskn::sign_stream_rule(cfg.sign_rule, cfg.sign_seed, 0, cols);
    for (int r = 0; r < rows; ++r) {
      float* row = W + static_cast<size_t>(r) * cols;
      for (int c = 0; c < cols; ++c) row[c] *= static_cast<float>(si[c]);
    }
  }
}

// The normalised matrix B the encoder quantises: rotate, then scale each row by
// 127/max|A[r]| (docs/GGUF.md).
std::vector<float> normalised_b(const std::vector<float>& W, int rows, int cols,
                                const QuantConfig& cfg) {
  std::vector<float> A = W;
  yskn::gguf::rotate_matrix(A.data(), rows, cols, cfg, A.data());
  for (int r = 0; r < rows; ++r) {
    float* row = A.data() + static_cast<size_t>(r) * cols;
    float mx = 0.f;
    for (int c = 0; c < cols; ++c) mx = std::max(mx, std::fabs(row[c]));
    if (mx > 0.f) {
      const float inv = 127.f / mx;
      for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
  }
  return A;
}

// ---------------------------------------------------------------------------
// 1. Hadamard.
// ---------------------------------------------------------------------------
void test_hadamard() {
  std::printf("Hadamard:\n");

  // H^2 = 512*I on a single random 512-vector.
  std::vector<float> v = random_matrix(512, 0x51ED);
  std::vector<float> orig = v;
  yskn::gguf::hadamard512(v.data());
  yskn::gguf::hadamard512(v.data());
  double worst = 0.0, scale = 0.0;
  for (int i = 0; i < 512; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(v[i]) - 512.0 * orig[i]));
    scale = std::max(scale, std::fabs(512.0 * orig[i]));
  }
  check_lt(worst / scale, 1e-6, "H^2 == 512*I on a random vector");
  std::printf("  H^2 == 512*I: rel err %.3e\n", worst / scale);

  // H is symmetric: <Hx, y> == <x, Hy>.
  std::vector<float> x = random_matrix(512, 0xA1), y = random_matrix(512, 0xA2);
  std::vector<float> hx = x, hy = y;
  yskn::gguf::hadamard512(hx.data());
  yskn::gguf::hadamard512(hy.data());
  double dot_hx_y = 0.0, dot_x_hy = 0.0;
  for (int i = 0; i < 512; ++i) {
    dot_hx_y += static_cast<double>(hx[i]) * y[i];
    dot_x_hy += static_cast<double>(x[i]) * hy[i];
  }
  check_lt(std::fabs(dot_hx_y - dot_x_hy) / std::max(1.0, std::fabs(dot_hx_y)), 1e-5,
           "H is symmetric (<Hx,y> == <x,Hy>)");

  // Natural (Sylvester) order: row i of H is (-1)^popcount(i & j).
  std::vector<float> e(512, 0.f);
  e[3] = 1.f;  // column 3 of H
  yskn::gguf::hadamard512(e.data());
  int mismatches = 0;
  for (int i = 0; i < 512; ++i) {
    const float want = (__builtin_popcount(static_cast<unsigned>(i) & 3u) & 1) ? -1.f : 1.f;
    if (e[i] != want) ++mismatches;
  }
  check_eq(mismatches, 0, "H column 3 is (-1)^popcount(i & 3)");

  // rotate then invert a random 512x512 matrix.
  //
  // rotate_matrix computes A = S_out . Hn . W . Hn . S_in  with Hn = (1/sqrt512)H.
  // It is NOT an involution: S and Hn do not commute, so applying it twice gives
  // S_out Hn S_out Hn W ... != W. The true inverse mirrors the operation order:
  //   W = S_in . Hn . A . Hn . S_out
  // which is the same two Hadamards with the sign roles swapped.
  const int n = 512;
  std::vector<float> W = random_matrix(static_cast<size_t>(n) * n, 0xBEEF);
  std::vector<float> A = W, back = W;
  QuantConfig cfg;
  cfg.bits = 6;
  yskn::gguf::rotate_matrix(A.data(), n, n, cfg, A.data());
  invert_rotate(A.data(), n, n, cfg, back.data());
  double err = 0.0, wmax = 0.0;
  for (size_t i = 0; i < W.size(); ++i) {
    err = std::max(err, std::fabs(static_cast<double>(back[i]) - W[i]));
    wmax = std::max(wmax, std::fabs(static_cast<double>(W[i])));
  }
  check_lt(err / wmax, 1e-4, "rotate then invert a random 512x512 matrix");
  std::printf("  512x512 rotate/invert: rel err %.3e\n", err / wmax);

  // The rotation must actually change the matrix (a no-op would pass the above).
  double changed = 0.0;
  for (size_t i = 0; i < W.size(); ++i) changed = std::max(changed, static_cast<double>(std::fabs(A[i] - W[i])));
  check_ge(changed / wmax, 1e-3, "rotation is not the identity");

  // Ragged tail: 512 rows x 768 cols -> the last 256 columns are untransformed
  // but still signed, so the round trip must stay exact there. The inverse is
  // invert_rotate (rotate_matrix is not an involution; see the note above).
  {
    const int rows = 512, cols = 768;
    std::vector<float> Wr = random_matrix(static_cast<size_t>(rows) * cols, 0xC0FFEE);
    std::vector<float> Ar = Wr, Br = Wr;
    yskn::gguf::rotate_matrix(Ar.data(), rows, cols, cfg, Ar.data());
    invert_rotate(Ar.data(), rows, cols, cfg, Br.data());
    double e2 = 0.0, m2 = 0.0;
    for (size_t i = 0; i < Wr.size(); ++i) {
      e2 = std::max(e2, std::fabs(static_cast<double>(Br[i]) - Wr[i]));
      m2 = std::max(m2, std::fabs(static_cast<double>(Wr[i])));
    }
    check_lt(e2 / m2, 1e-4, "ragged tail (512x768) still round trips");
    std::printf("  512x768 ragged tail: rel err %.3e\n", e2 / m2);
  }

  // use_rot_in / use_rot_out = false must be the identity on that side.
  {
    QuantConfig id;
    id.use_rot_in = false;
    id.use_rot_out = false;
    std::vector<float> W2 = random_matrix(4096, 0xD00D), A2 = W2;
    yskn::gguf::rotate_matrix(A2.data(), 64, 64, id, A2.data());
    double d = 0.0;
    for (size_t i = 0; i < W2.size(); ++i) d = std::max(d, static_cast<double>(std::fabs(A2[i] - W2[i])));
    check_lt(d, 1e-9, "use_rot_in/out == false is the identity");
  }
}

// ---------------------------------------------------------------------------
// 2. Codebook.
// ---------------------------------------------------------------------------
void test_codebook() {
  std::printf("Codebook:\n");

  // f16 helpers first: the codebook invariant depends on them.
  check_eq(yskn::gguf::f32_to_f16(1.0f), 0x3C00, "f16(1.0)");
  check_eq(yskn::gguf::f32_to_f16(-2.0f), 0xC000, "f16(-2.0)");
  check_eq(yskn::gguf::f32_to_f16(0.0f), 0x0000, "f16(0.0)");
  check_eq(yskn::gguf::f32_to_f16(65504.0f), 0x7BFF, "f16(65504)");
  check_eq(yskn::gguf::f32_to_f16(1e30f), 0x7C00, "f16(1e30) overflows to inf");
  check_lt(std::fabs(yskn::gguf::f16_to_f32(0x3555) - 0.33325195f), 1e-7, "f16_to_f32(0x3555)");
  {
    // Round trip every finite f16 through f32 and back.
    int bad = 0;
    for (int h = 0; h < 0x7C00; ++h) {
      const float f = yskn::gguf::f16_to_f32(static_cast<uint16_t>(h));
      if (yskn::gguf::f32_to_f16(f) != static_cast<uint16_t>(h)) ++bad;
    }
    check_eq(bad, 0, "f16 -> f32 -> f16 is exact for all finite values");
  }

  // Train on a synthetic normalised matrix (|B| <= 127).
  const int rows = 64, cols = 256;
  std::vector<float> B = random_matrix(static_cast<size_t>(rows) * cols, 0x1234, 127.f);
  const Codebook cb = yskn::gguf::train_codebook(B.data(), B.size(), 7);

  check_eq(static_cast<long long>(cb.size()), 4096, "codebook has 4096 entries");
  check_ge(cb.gamma, 0.f, "codebook gamma is finite and non-negative");

  int mx = 0;
  for (int s = 0; s < 4096; ++s) {
    mx = std::max(mx, std::abs(static_cast<int>(cb.q0[s])));
    mx = std::max(mx, std::abs(static_cast<int>(cb.q1[s])));
  }
  check_eq(mx, 127, "max|q| == 127 exactly");

  // build_lut8_tau's inverse must recover the integers exactly (§3.2).
  std::vector<uint16_t> lut16(2 * 4096);
  for (int s = 0; s < 4096; ++s) {
    lut16[2 * s] = yskn::gguf::f32_to_f16(static_cast<float>(cb.q0[s]) * cb.gamma);
    lut16[2 * s + 1] = yskn::gguf::f32_to_f16(static_cast<float>(cb.q1[s]) * cb.gamma);
  }
  std::vector<uint16_t> s16(rows, yskn::gguf::f32_to_f16(1.f));
  const Lut8Tau back = build_lut8_tau(lut16, s16, rows);

  int qbad = 0;
  for (int s = 0; s < 4096; ++s) {
    if (back.q0[s] != cb.q0[s] || back.q1[s] != cb.q1[s]) ++qbad;
  }
  check_eq(qbad, 0, "build_lut8_tau recovers q0/q1 exactly");
  check_lt(std::fabs(back.gamma - cb.gamma) / cb.gamma, 1e-3, "build_lut8_tau recovers gamma");

  // Determinism: same seed, same codebook.
  const Codebook cb2 = yskn::gguf::train_codebook(B.data(), B.size(), 7);
  check(cb2.q0 == cb.q0 && cb2.q1 == cb.q1 && cb2.gamma == cb.gamma,
        "train_codebook is deterministic for a fixed seed");

  // A different seed must still satisfy the invariant.
  const Codebook cb3 = yskn::gguf::train_codebook(B.data(), B.size(), 99);
  int mx3 = 0;
  for (int s = 0; s < 4096; ++s) {
    mx3 = std::max(mx3, std::abs(static_cast<int>(cb3.q0[s])));
    mx3 = std::max(mx3, std::abs(static_cast<int>(cb3.q1[s])));
  }
  check_eq(mx3, 127, "max|q| == 127 for a different seed");

  // The codebook must actually fit the data: the nearest-centroid error,
  // measured with the int8 codebook the decoder will see, has to be well under
  // the sample's own variance.
  double sse = 0.0, var = 0.0;
  {
    double m0 = 0.0, m1 = 0.0;
    const double np = static_cast<double>(B.size() / 2);
    for (size_t t = 0; t < B.size() / 2; ++t) {
      m0 += B[2 * t];
      m1 += B[2 * t + 1];
    }
    m0 /= np;
    m1 /= np;
    for (size_t t = 0; t < B.size() / 2; ++t) {
      const double x = B[2 * t], y = B[2 * t + 1];
      var += (x - m0) * (x - m0) + (y - m1) * (y - m1);
    }
  }
  {
    const int S = 4096;
    std::vector<float> fq0(S), fq1(S);
    for (int s = 0; s < S; ++s) {
      fq0[s] = static_cast<float>(cb.q0[s]) * cb.gamma;
      fq1[s] = static_cast<float>(cb.q1[s]) * cb.gamma;
    }
    for (size_t t = 0; t < B.size() / 2; ++t) {
      float best = std::numeric_limits<float>::infinity();
      for (int s = 0; s < S; ++s) {
        const float dx = B[2 * t] - fq0[s], dy = B[2 * t + 1] - fq1[s];
        best = std::min(best, dx * dx + dy * dy);
      }
      sse += best;
    }
  }
  std::printf("  k-means: rmse %.4f (sample rms %.4f)\n", std::sqrt(sse / (B.size() / 2)),
              std::sqrt(var / B.size()));
  check_lt(std::sqrt(sse / (B.size() / 2)), std::sqrt(var / B.size()) * 0.5,
           "k-means rmse is well under the sample rms");
}

// ---------------------------------------------------------------------------
// 3. Viterbi.
// ---------------------------------------------------------------------------
void test_viterbi() {
  std::printf("Viterbi:\n");

  // Synthetic 1-row, 256-column input in the normalised domain.
  const int rows = 1, cols = 256;
  std::vector<float> B(static_cast<size_t>(rows) * cols);
  Rng rng(0x5EED);
  for (int c = 0; c < cols; ++c) {
    // Smooth envelope plus noise: exactly the shape a rotated weight row has.
    const float env = 90.f * std::sin(0.037f * static_cast<float>(c)) +
                      30.f * std::cos(0.011f * static_cast<float>(c));
    B[c] = std::max(-127.f, std::min(127.f, env + 8.f * rng.sym()));
  }
  const Codebook cb = yskn::gguf::train_codebook(B.data(), B.size(), 11);

  std::vector<uint16_t> lut16(2 * 4096);
  for (int s = 0; s < 4096; ++s) {
    lut16[2 * s] = yskn::gguf::f32_to_f16(static_cast<float>(cb.q0[s]) * cb.gamma);
    lut16[2 * s + 1] = yskn::gguf::f32_to_f16(static_cast<float>(cb.q1[s]) * cb.gamma);
  }
  std::vector<uint16_t> s16(rows, yskn::gguf::f32_to_f16(1.f));
  const Lut8Tau lut = build_lut8_tau(lut16, s16, rows);

  double rel[3] = {0, 0, 0};
  const int bits_list[3] = {4, 6, 8};
  for (int bi = 0; bi < 3; ++bi) {
    const int bits = bits_list[bi];
    std::vector<uint32_t> words;
    yskn::gguf::viterbi_quantize(B.data(), rows, cols, cb, bits, words);
    check_eq(static_cast<long long>(words.size()), static_cast<long long>(rows * (cols / 256) * 4 * bits),
             "viterbi word count for bits=" + std::to_string(bits));

    std::vector<float> dec;
    decode_sequence(words, 0, 0, 1, 4 * bits, bits, lut, 1.f, dec);
    double worst = 0.0;
    for (int c = 0; c < cols; ++c) worst = std::max(worst, static_cast<double>(std::fabs(dec[c] - B[c])));
    rel[bi] = worst / 127.0;
    std::printf("  bits=%d: max|err| %.3f (%.2f%% of full scale)\n", bits, worst, rel[bi] * 100.0);

    // The trellis fills the words exactly: wps = 4*bits words == 128*bits bits,
    // so there is no tail to check. (Shifting a uint32_t by 32 is undefined
    // behaviour, which is what the earlier form of this assertion did.) Assert
    // the invariant that actually matters: no word exists beyond the trellis.
    check_eq(static_cast<long long>(words.size()),
             static_cast<long long>(rows * (cols / 256) * 4 * bits),
             "stream holds exactly the trellis words (bits=" + std::to_string(bits) + ")");
  }

  // bits=8 is tight, bits=4 looser, bits=6 in between.
  // The trellis reaches only 2^bits of the 4096 codewords per step, so the best
  // achievable error is ~254/sqrt(2^bits)/2 in full-scale units: 31.8 at bits=4,
  // 15.9 at bits=6, 7.9 at bits=8. Assert against that floor rather than an
  // arbitrary percentage -- measured values sit at 28.8 / 16.6 / 9.2.
  for (int bi = 0; bi < 3; ++bi) {
    const double floor_err = 254.0 / std::sqrt(static_cast<double>(1 << bits_list[bi])) / 2.0;
    check_lt(rel[bi] * 127.0, floor_err * 1.35,
             "bits=" + std::to_string(bits_list[bi]) + " is within 35% of the trellis floor");
  }
  check_lt(rel[0], 0.20, "bits=4 error stays under 20% of full scale");
  check_ge(rel[0], rel[2], "bits=4 is not better than bits=8");
  check_ge(rel[1], rel[2] - 1e-9, "bits=6 is not better than bits=8");
  check_ge(rel[0], rel[1] - 1e-9, "bits=4 is not better than bits=6");

  // Rejecting bits outside {4,6,8}.
  bool threw = false;
  try {
    std::vector<uint32_t> w;
    yskn::gguf::viterbi_quantize(B.data(), rows, cols, cb, 5, w);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "viterbi_quantize rejects bits=5");
}

// ---------------------------------------------------------------------------
// 4. Interleave.
// ---------------------------------------------------------------------------
void test_interleave() {
  std::printf("Interleave:\n");
  const int rows = 128, seqs = 3, wps = 24;  // bits=6
  const size_t n = static_cast<size_t>(rows) * seqs * wps;
  Rng rng(0xF00D);
  std::vector<uint32_t> words(n);
  for (size_t i = 0; i < n; ++i) words[i] = static_cast<uint32_t>(rng.next());

  std::vector<uint32_t> frag;
  yskn::gguf::interleave_warp(words, rows, seqs, wps, frag);
  check_eq(static_cast<long long>(frag.size()), static_cast<long long>(n), "frag size");

  // The permutation must be a bijection: every source word lands exactly once.
  std::vector<uint32_t> back;
  deinterleave(frag, rows, seqs, wps, back);
  check(back == words, "interleave_warp followed by the k_repack inverse is the identity");

  // And it must actually permute (a copy would pass the check above).
  size_t moved = 0;
  for (size_t i = 0; i < n; ++i) {
    if (frag[i] != words[i]) ++moved;
  }
  check_ge(static_cast<double>(moved), static_cast<double>(n) * 0.9,
           "interleave_warp really permutes the stream");

  // Spot-check the layout against src/repack.cu:13-14 for a few coordinates.
  int bad = 0;
  for (int r = 0; r < rows; r += 37) {
    for (int q = 0; q < seqs; ++q) {
      for (int w = 0; w < wps; w += 5) {
        const size_t src = warp_src(r, q, w, rows, wps);
        if (frag[src] != words[(static_cast<size_t>(r) * seqs + q) * wps + w]) ++bad;
      }
    }
  }
  check_eq(bad, 0, "interleave_warp matches k_repack's src index");

  // Shape guards.
  bool threw = false;
  try {
    std::vector<uint32_t> f;
    yskn::gguf::interleave_warp(words, 100, seqs, wps, f);  // 100 % 64 != 0
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "interleave_warp rejects rows % 64 != 0");
}

// ---------------------------------------------------------------------------
// 5. Round trip (the §5.5 oracle).
// ---------------------------------------------------------------------------
void test_roundtrip() {
  std::printf("Round trip:\n");
  // 4096 centroids need far more than 4096 training pairs. A real 27B matrix is
  // 5120x13824 = 35M pairs, so use a size that gives the k-means comparable
  // coverage (512*512/2 = 131k pairs, 32x the codebook). A tiny matrix would
  // measure overfitting, not the encoder.
  const int rows = 512, cols = 512;
  std::vector<float> W = random_matrix(static_cast<size_t>(rows) * cols, 0x7777, 0.05f);

  double rel[2] = {0, 0};
  const int bits_list[2] = {4, 8};
  for (int bi = 0; bi < 2; ++bi) {
    QuantConfig cfg;
    cfg.bits = bits_list[bi];
    const QuantizedTensor t = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg);

    check_eq(t.rows, rows, "tensor rows");
    check_eq(t.cols, cols, "tensor cols");
    check_eq(t.physical_rows, ((rows + 63) / 64) * 64, "tensor physical_rows (padded to 64)");
    check_eq(t.seqs, cols / 256, "tensor seqs");
    check_eq(t.wps, 4 * bits_list[bi], "tensor wps");
    check_eq(static_cast<long long>(t.lut16.size()), 8192, "lut16 size");
    check_eq(static_cast<long long>(t.scale16.size()), t.physical_rows, "scale16 size");
    check_eq(static_cast<long long>(t.stream.size()),
             static_cast<long long>(t.physical_rows) * t.seqs * 4 * bits_list[bi], "stream size");

    const std::vector<float> B = normalised_b(W, rows, cols, cfg);
    float maxerr = 0.f, relerr = 0.f;
    const bool ok = yskn::gguf::verify_roundtrip(t, B.data(), &maxerr, &relerr);
    check(ok, "verify_roundtrip accepts the tensor (bits=" + std::to_string(bits_list[bi]) + ")");
    const float bmax = max_abs(B);

    // RMS is the metric that tracks reconstruction quality; max|err| is set by a
    // handful of outlier pairs and even rises with more training data (a
    // generalising codebook has a worse worst case than an overfitted one).
    float rms = 0.f, rms_max = 0.f;
    const bool ok_rms = yskn::gguf::verify_roundtrip_rms(t, B.data(), &rms, &rms_max);
    check(ok_rms, "verify_roundtrip_rms accepts the tensor");
    check_lt(std::fabs(rms_max - maxerr), 1e-4, "both oracles agree on max|err|");
    rel[bi] = rms;
    std::printf("  bits=%d: rms %.4f (%.2f%% of max|B| = %.0f), max|err| %.2f\n",
                bits_list[bi], rms, rms / bmax * 100.0, bmax, maxerr);

    // The trellis reaches only 2^bits codewords per step, so the RMS floor is
    // roughly 254/sqrt(2^bits)/sqrt(12) for a uniform pair distribution. Measured
    // on this matrix: 10.6 / 5.5 / 2.9 at bits 4 / 6 / 8.
    {
      const double floor_rms = 254.0 / std::sqrt(static_cast<double>(1 << bits_list[bi])) / std::sqrt(12.0);
      check_lt(rms, floor_rms * 1.6,
               "round trip RMS is near the trellis floor (bits=" +
                   std::to_string(bits_list[bi]) + ")");
    }
    check_ge(bmax, 126.0, "the normalised matrix reaches full scale");
  }
  check_ge(rel[0], rel[1] - 1e-9, "bits=4 round trip is not better than bits=8");

  // A zero matrix must not blow up: tau == 0, stream all zero, decode all zero.
  {
    std::vector<float> Z(static_cast<size_t>(rows) * cols, 0.f);
    QuantConfig cfg;
    cfg.bits = 8;
    const QuantizedTensor t = yskn::gguf::quantize_tensor(Z.data(), rows, cols, cfg);
    float maxerr = 0.f, relerr = 0.f;
    const bool ok = yskn::gguf::verify_roundtrip(t, Z.data(), &maxerr, &relerr);
    check(ok, "zero matrix round trips");
    check_lt(maxerr, 1e-6, "zero matrix decodes to zero");
  }
}

// ---------------------------------------------------------------------------
// 6. Determinism.
// ---------------------------------------------------------------------------
void test_determinism() {
  std::printf("Determinism:\n");
  const int rows = 8, cols = 512;
  const std::vector<float> W = random_matrix(static_cast<size_t>(rows) * cols, 0xABCD, 0.02f);
  QuantConfig cfg;
  cfg.bits = 6;

  const QuantizedTensor a = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg);
  const QuantizedTensor b = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg);

  check(a.stream == b.stream, "stream is byte-identical across runs");
  check(a.lut16 == b.lut16, "lut16 is byte-identical across runs");
  check(a.scale16 == b.scale16, "scale16 is byte-identical across runs");
  check(a.gamma == b.gamma, "gamma is identical across runs");
  check_eq(a.physical_rows, b.physical_rows, "physical_rows is identical across runs");

  // Reusing a trained codebook must reproduce the same stream: the codebook is
  // the only state quantize_tensor carries between calls.
  //
  // quantize_tensor refines the codebook it trains itself (cfg.refine_iterations),
  // but a *shared* codebook is used verbatim -- that is the point of sharing it
  // across the 400 matrices of a model. So compare a shared codebook against
  // another run with the same shared codebook, and separately check that the
  // refined path is reproducible.
  {
    const std::vector<float> B = normalised_b(W, rows, cols, cfg);
    const Codebook cb = yskn::gguf::train_codebook(B.data(), B.size(), 7);
    const QuantizedTensor c = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg, &cb);
    const QuantizedTensor d = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg, &cb);
    check(c.stream == d.stream, "a shared codebook is deterministic");
    check(c.gamma == cb.gamma, "a shared codebook keeps its gamma");

    // A refined codebook, shared, must also be reproducible and must beat the
    // unrefined one on the same input.
    // B holds exactly 'rows' rows: refine_codebook must be told the real row
    // count, not the 64-row padded one (the padding is not in this buffer).
    const Codebook ref = yskn::gguf::refine_codebook(B.data(), rows, cols, cb, cfg.bits, 4);
    const QuantizedTensor e = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg, &ref);
    const QuantizedTensor f = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg, &ref);
    check(e.stream == f.stream, "a refined shared codebook is deterministic");
    check(e.stream != c.stream, "refinement changes the stream");

    float e_plain = 0.f, r_plain = 0.f, e_ref = 0.f, r_ref = 0.f;
    yskn::gguf::verify_roundtrip(c, B.data(), &e_plain, &r_plain);
    yskn::gguf::verify_roundtrip(e, B.data(), &e_ref, &r_ref);
    check_lt(r_ref, r_plain, "a refined codebook is closer than plain k-means");
    std::printf("  shared codebook: plain %.4f -> refined %.4f (max|B| units)\n", e_plain, e_ref);
  }

  // Different bits must give different streams.
  QuantConfig cfg8 = cfg;
  cfg8.bits = 8;
  const QuantizedTensor d = yskn::gguf::quantize_tensor(W.data(), rows, cols, cfg8);
  check(d.stream != a.stream, "bits=8 and bits=6 produce different streams");

  // Config validation.
  bool threw = false;
  try {
    QuantConfig bad;
    bad.bits = 5;
    yskn::gguf::quantize_tensor(W.data(), rows, cols, bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "quantize_tensor rejects bits=5");

  threw = false;
  try {
    QuantConfig bad;
    bad.sign_rule = "nope";
    yskn::gguf::quantize_tensor(W.data(), rows, cols, bad);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "quantize_tensor rejects an unknown sign rule");

  threw = false;
  try {
    yskn::gguf::quantize_tensor(W.data(), rows, 100, cfg);  // 100 % 256 != 0
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "quantize_tensor rejects cols % 256 != 0");
}

}  // namespace

int main() {
  std::printf("GGUF -> engine tensor bridge test (host only, no CUDA)\n\n");
  test_hadamard();
  test_codebook();
  test_viterbi();
  test_interleave();
  test_roundtrip();
  test_determinism();

  std::printf("\n");
  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  for (const std::string& f : g_failure_log) std::printf("  - %s\n", f.c_str());
  return 1;
}
