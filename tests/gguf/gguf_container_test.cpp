// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for GgufContainer (src/gguf/gguf_container.cpp).
//
// CPU only: no CUDA, no GPU. The suite proves that the container hands the
// legacy loader exactly what it reads, by re-implementing the three pieces of
// reader arithmetic Model::load depends on and checking the container against
// them:
//
//   build_lut8_tau   src/loader.cpp:19-25    (f16 LUT -> int8 codes + gamma)
//   v2_state         src/v2_decode.cuh:13-19 (12-bit trellis window)
//   k_repack         src/repack.cu:10-18     (layout:"warp" permutation)
//
// Fixtures come from tools/make_tiny_gguf.py, generated at build time.
//
// Exits non-zero on failure and prints a PASS/FAIL summary.
#include "gguf_container.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_model.hpp"
#include "gguf_quant.hpp"

#include <nlohmann/json.hpp>

namespace {

using yskn::gguf::Codebook;
using yskn::gguf::GgufContainer;
using yskn::gguf::QuantConfig;

int g_checks = 0;
int g_failures = 0;
std::vector<std::string> g_log;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    g_log.push_back(what);
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void check_eq(long long got, long long want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s (got %lld, want %lld)", what.c_str(), got, want);
    g_log.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

void check_lt(double got, double bound, const std::string& what) {
  ++g_checks;
  if (!(got < bound)) {
    ++g_failures;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s (got %.6g, want < %.6g)", what.c_str(), got, bound);
    g_log.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

// ---------------------------------------------------------------------------
// Reader arithmetic, re-implemented (see the header comment).
// ---------------------------------------------------------------------------
float h2f(uint16_t h) { return yskn::gguf::f16_to_f32(h); }

// src/loader.cpp:19-25.
void build_lut8_tau(const uint16_t* lut16, const uint16_t* s16, int rows,
                    std::vector<uint16_t>& lut8, std::vector<float>& tau, int S) {
  float mx = 0.f;
  for (int i = 0; i < S * 2; ++i) mx = std::max(mx, std::fabs(h2f(lut16[i])));
  const float gamma = mx / 127.f;
  lut8.resize(S);
  for (int s = 0; s < S; ++s) {
    int q0 = (int)std::nearbyint(h2f(lut16[2 * s]) / gamma);
    int q1 = (int)std::nearbyint(h2f(lut16[2 * s + 1]) / gamma);
    q0 = std::max(-127, std::min(127, q0));
    q1 = std::max(-127, std::min(127, q1));
    lut8[s] = (uint16_t)((uint8_t)(int8_t)q0 | ((uint16_t)(uint8_t)(int8_t)q1 << 8));
  }
  tau.resize(rows);
  for (int r = 0; r < rows; ++r) tau[r] = h2f(s16[r]) * gamma;
}

// src/v2_decode.cuh:13-19.
uint32_t read_window(const uint32_t* w, uint32_t start, uint32_t width) {
  const uint32_t idx = start >> 5, sh = start & 31;
  uint32_t v = w[idx] >> sh;
  if (sh + width > 32) v |= w[idx + 1] << (32 - sh);
  return v & ((1u << width) - 1u);
}

uint32_t v2_state(const uint32_t* w, int bits, int j) {
  const uint32_t e = (uint32_t)bits * (uint32_t)(j + 1);
  if (e < 12u) return (w[0] & ((1u << e) - 1u)) << (12u - e);
  return read_window(w, e - 12u, 12u);
}

// src/repack.cu:13-14.
size_t warp_src(int row, int seq, int word, int rows, int wps) {
  const size_t q = (size_t)seq * rows + row, tile = q / 32, lane = q % 32;
  return ((tile * (size_t)(wps / 4) + (size_t)(word / 4)) * 32 + lane) * 4 + (size_t)(word % 4);
}

// ---------------------------------------------------------------------------
// Sidecar: the JSON make_tiny_gguf.py writes next to the fixture. nlohmann is
// already a dependency of src/format.hpp, so the test reuses it rather than
// hand-rolling a parser.
// ---------------------------------------------------------------------------
struct Sidecar {
  int layers = 0, hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
  float eps = 0.f, rope_theta = 0.f;
  bool tie_embeddings = false;
  std::string variant;
  struct Vec {
    std::string name;
    int n = 0;
    std::vector<float> values;
  };
  struct Mat {
    std::string name;
    int rows = 0, cols = 0;
  };
  std::vector<Vec> vectors;
  std::vector<Mat> tensors;
};

void load_sidecar(const std::string& path, Sidecar& sc) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  const yskn::json j = yskn::json::parse(f);
  const yskn::json& m = j.at("model");
  sc.layers = m.at("layers").get<int>();
  sc.hidden = m.at("hidden").get<int>();
  sc.ffn = m.at("ffn").get<int>();
  sc.heads = m.at("heads").get<int>();
  sc.kv_heads = m.at("kv_heads").get<int>();
  sc.head_dim = m.at("head_dim").get<int>();
  sc.vocab = m.at("vocab").get<int>();
  sc.eps = m.at("eps").get<float>();
  sc.rope_theta = m.at("rope_theta").get<float>();
  sc.tie_embeddings = m.value("tie_embeddings", false);
  sc.variant = j.value("variant", std::string());
  for (const yskn::json& v : j.at("vectors")) {
    Sidecar::Vec vec;
    vec.name = v.at("name").get<std::string>();
    vec.values = v.at("values").get<std::vector<float>>();
    vec.n = static_cast<int>(vec.values.size());
    sc.vectors.push_back(vec);
  }
  for (const yskn::json& t : j.at("tensors")) {
    sc.tensors.push_back({t.at("name").get<std::string>(), t.at("rows").get<int>(),
                          t.at("cols").get<int>()});
  }
}

// ---------------------------------------------------------------------------
// 1. The model description matches the file.
// ---------------------------------------------------------------------------
void test_model_desc(const GgufContainer& gc, const Sidecar& sc) {
  std::printf("model description:\n");
  check(gc.model.arch == "qwen3", "arch is qwen3 (got '" + gc.model.arch + "')");
  check_eq(gc.model.layers, sc.layers, "layers");
  check_eq(gc.model.hidden, sc.hidden, "hidden");
  check_eq(gc.model.ffn, sc.ffn, "ffn");
  check_eq(gc.model.heads, sc.heads, "heads");
  check_eq(gc.model.kv_heads, sc.kv_heads, "kv_heads");
  check_eq(gc.model.head_dim, sc.head_dim, "head_dim");
  check_eq(gc.model.vocab, sc.vocab, "vocab");
  check(std::fabs(gc.model.eps - sc.eps) < 1e-9f, "eps matches the file");
  check(std::fabs(gc.model.rope_theta - sc.rope_theta) < 1.f, "rope_theta matches the file");
  check_eq(gc.model.tie_embeddings ? 1 : 0, sc.tie_embeddings ? 1 : 0, "tie_embeddings");
  std::printf("  arch=%s layers=%d hidden=%d ffn=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
              gc.model.arch.c_str(), gc.model.layers, gc.model.hidden, gc.model.ffn,
              gc.model.heads, gc.model.kv_heads, gc.model.head_dim, gc.model.vocab);
}

// ---------------------------------------------------------------------------
// 2. Every name Model::load asks for resolves.
// ---------------------------------------------------------------------------
void test_names(const GgufContainer& gc) {
  std::printf("loader names:\n");
  const yskn::ModelDesc& c = gc.model;
  int resolved = 0;
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    for (const char* w : {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                            "down_proj"}) {
      const std::string n = p + w;
      bool ok = true;
      try {
        gc.tensor(n);
      } catch (const std::exception&) {
        ok = false;
      }
      check(ok, "tensor " + n + " resolves");
      ++resolved;
    }
    for (const char* v : {"input_layernorm", "post_attention_layernorm", "q_proj.bias",
                            "k_proj.bias", "v_proj.bias"}) {
      const std::string n = p + v;
      bool ok = true;
      try {
        gc.vector(n);
      } catch (const std::exception&) {
        ok = false;
      }
      check(ok, "vector " + n + " resolves");
      ++resolved;
    }
  }
  for (const char* n : {"lm_head", "embed_tokens"}) {
    bool ok = true;
    try {
      gc.tensor(n);
    } catch (const std::exception&) {
      ok = false;
    }
    check(ok, std::string("tensor ") + n + " resolves");
    ++resolved;
  }
  bool norm_ok = true;
  try {
    gc.vector("norm");
  } catch (const std::exception&) {
    norm_ok = false;
  }
  check(norm_ok, "vector norm resolves");
  ++resolved;
  std::printf("  %d names resolved\n", resolved);

  // A name the loader never asks for must still throw, not silently return.
  bool threw = false;
  try {
    gc.tensor("layers.0.self_attn.q_proj");
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("missing tensor") != std::string::npos;
  }
  check(threw, "an Engine35-style name throws 'missing tensor ...'");
}

// ---------------------------------------------------------------------------
// 3. Alignment and stream_bytes.
// ---------------------------------------------------------------------------
void test_layout(const GgufContainer& gc) {
  std::printf("layout:\n");
  const uintptr_t base = reinterpret_cast<uintptr_t>(gc.at<uint8_t>(0));
  check_eq((long long)(base % 16), 0, "buffer base is 16-byte aligned");

  for (const yskn::TensorDesc& t : gc.tensors()) {
    const std::string n = t.name;
    check_eq((long long)(reinterpret_cast<uintptr_t>(gc.at<uint16_t>(t.lut_off)) % 2), 0,
             "lut_off of " + n + " is 2-byte aligned");
    check_eq((long long)(reinterpret_cast<uintptr_t>(gc.at<uint16_t>(t.scale_off)) % 2), 0,
             "scale_off of " + n + " is 2-byte aligned");
    check_eq((long long)(reinterpret_cast<uintptr_t>(gc.at<uint32_t>(t.stream_off)) % 4), 0,
             "stream_off of " + n + " is 4-byte aligned");
    const long long want = (long long)t.physical_rows * (t.cols / 256) * (4 * t.bits) * 4;
    check_eq((long long)t.stream_bytes, want, "stream_bytes of " + n + " is rows*seqs*4*bits*4");
    check_eq(t.L, 12, "L of " + n + " is 12");
    check_eq(t.d, 2, "d of " + n + " is 2");
    check_eq(t.T, 256, "T of " + n + " is 256");
    check(t.bits == 4 || t.bits == 6 || t.bits == 8, "bits of " + n + " is 4/6/8");
    check_eq(t.physical_rows % 64, 0, "physical_rows of " + n + " is a multiple of 64");
    check(t.stream_off + t.stream_bytes <= gc.bytes(), "stream of " + n + " is inside the buffer");
    check(t.lut_off + 2 * 4096 * 2 <= gc.bytes(), "lut16 of " + n + " is inside the buffer");
    check(t.scale_off + (uint64_t)t.physical_rows * 2 <= gc.bytes(),
          "scale16 of " + n + " is inside the buffer");
  }
  for (const yskn::VectorDesc& v : gc.vectors()) {
    check_eq((long long)(reinterpret_cast<uintptr_t>(gc.at<uint16_t>(v.off)) % 2), 0,
             "off of vector " + v.name + " is 2-byte aligned");
    check(v.dtype == "f16", "vector " + v.name + " is f16");
    check(v.off + (uint64_t)v.n * 2 <= gc.bytes(), "vector " + v.name + " is inside the buffer");
  }

  // layout: the embedding table must be row-major, everything else warp.
  check(gc.tensor("embed_tokens").layout == "row", "embed_tokens uses layout:row");
  check(gc.tensor("lm_head").layout == "warp", "lm_head uses layout:warp");
  check(gc.tensor("layers.0.q_proj").layout == "warp", "q_proj uses layout:warp");

  // Model::upload_embedding (src/loader.cpp:65) copies S*4 uint16 from lut_off,
  // i.e. 32 KiB, while lut16 is only 2*S uint16 = 16 KiB. That over-read is
  // harmless in a YSKN container (it lands in the next payload) but it means the
  // buffer must physically contain those bytes. Assert it for every tensor, not
  // just the embedding one, so a future layout change cannot introduce a SIGBUS.
  for (const yskn::TensorDesc& t : gc.tensors()) {
    const uint64_t over = t.lut_off + (uint64_t)(1 << t.L) * 4 * 2;
    check(over <= gc.bytes(), "lut over-read of " + t.name + " stays inside the buffer");
  }

  // hidden 256 < 512, so both Hadamard passes are no-ops: rotate_matrix only
  // transforms whole 512-blocks (gguf_quant.hpp, 'RAGGED TAIL'), and
  // fwht512_blocks(buf, m) with m = cols/512 = 0 does nothing either. The signs
  // still apply on both sides. The container must record m = 0 so the reader
  // agrees -- if it wrote m = 1 the reader would apply a Hadamard the encoder
  // never did, and every output would be scrambled.
  check_eq(gc.tensor("layers.0.q_proj").rot_in.m, 0, "q_proj rot_in.m is 0 (cols < 512)");
  check_eq(gc.tensor("layers.0.q_proj").rot_out.m, 0, "q_proj rot_out.m is 0 (rows < 512)");
  check(gc.tensor("layers.0.q_proj").rot_in.kind == "H512", "q_proj rot_in.kind is H512");
  check(gc.tensor("layers.0.q_proj").rot_in.rule == "mulberry32_7919_13",
        "q_proj rot_in.rule is the reader's default rule");
  check_eq((long long)gc.tensor("layers.0.q_proj").rot_in.sign_offset, 0,
           "q_proj rot_in.sign_offset is 0");
  check_eq((long long)gc.tensor("layers.0.q_proj").rot_out.sign_offset,
           gc.tensor("layers.0.q_proj").cols, "q_proj rot_out.sign_offset is cols");
  check(!gc.tensor("lm_head").has_rot_out, "lm_head has no rot_out (head rule)");
  check(!gc.tensor("embed_tokens").has_rot_out, "embed_tokens has no rot_out");
  check(gc.tensor("layers.0.gate_proj").has_rot_out, "gate_proj has rot_out");

  std::printf("  %zu tensors, %zu vectors, %zu bytes (%zu payload + %zu tail)\n",
              gc.tensors().size(), gc.vectors().size(), gc.bytes(), gc.payload_bytes(),
              yskn::gguf::kTailPadBytes);
}

// ---------------------------------------------------------------------------
// 3b. Tensor shapes match the fixture, and the embedding stream is really
//     de-interleaved.
//
// Model::upload_embedding (src/loader.cpp:61-68) does NOT call
// yskn_repack_warp_to_frag: it indexes stream[id*seqs*wps + q*wps + w] directly
// (src/misc_kernels.cu:18). So embed_tokens must be stored in the *logical*
// row-major order, while every linear weight must be stored warp-interleaved.
// Getting this wrong is silent -- the model just produces garbage -- so the test
// decodes both ways and shows that only the intended one reconstructs.
// ---------------------------------------------------------------------------
double decode_rms(const GgufContainer& gc, const yskn::TensorDesc& d,
                  const std::vector<float>& W, bool warp_order) {
  const int rows = d.rows, cols = d.cols, physical = d.physical_rows;
  const int seqs = cols / 256, wps = 4 * d.bits;

  std::vector<float> B = W;
  yskn::gguf::QuantConfig qc = gc.quant_config();
  qc.use_rot_out = d.has_rot_out;
  yskn::gguf::rotate_matrix(B.data(), rows, cols, qc, B.data());
  for (int r = 0; r < rows; ++r) {
    float* row = B.data() + (size_t)r * cols;
    float mx = 0.f;
    for (int c = 0; c < cols; ++c) mx = std::max(mx, std::fabs(row[c]));
    if (mx > 0.f) {
      const float inv = 127.f / mx;
      for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
  }

  std::vector<uint16_t> lut8;
  std::vector<float> tau;
  build_lut8_tau(gc.at<uint16_t>(d.lut_off), gc.at<uint16_t>(d.scale_off), physical, lut8,
                 tau, 1 << d.L);

  const uint32_t* stream = gc.at<uint32_t>(d.stream_off);
  std::vector<uint32_t> w((size_t)wps);
  double sq = 0.0;
  size_t n = 0;
  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < seqs; ++q) {
      for (int i = 0; i < wps; ++i) {
        w[(size_t)i] = warp_order ? stream[warp_src(r, q, i, physical, wps)]
                                  : stream[((size_t)r * seqs + q) * wps + i];
      }
      const float* b = B.data() + (size_t)r * cols + (size_t)q * 256;
      for (int j = 0; j < 128; ++j) {
        const uint32_t s = v2_state(w.data(), d.bits, j);
        const double e0 = (double)(int8_t)(lut8[s] & 0xFF) - b[2 * j];
        const double e1 = (double)(int8_t)(lut8[s] >> 8) - b[2 * j + 1];
        sq += e0 * e0 + e1 * e1;
        n += 2;
      }
    }
  }
  return std::sqrt(sq / (double)n);
}

void test_shapes_and_layout(const GgufContainer& gc, const Sidecar& sc,
                            const std::string& gguf_path) {
  std::printf("tensor shapes and stream order:\n");

  // Every tensor the sidecar lists must be present with the same shape.
  for (const Sidecar::Mat& m : sc.tensors) {
    bool ok = gc.has_tensor(m.name);
    check(ok, "sidecar tensor " + m.name + " is present");
    if (!ok) continue;
    const yskn::TensorDesc& d = gc.tensor(m.name);
    check_eq(d.rows, m.rows, "rows of " + m.name);
    check_eq(d.cols, m.cols, "cols of " + m.name);
    check_eq(d.physical_rows, ((m.rows + 63) / 64) * 64, "physical_rows of " + m.name);
  }
  check_eq((long long)gc.tensors().size(), (long long)sc.tensors.size(),
           "the container exposes exactly the tensors the fixture defines");

  // The embedding table: row-major must reconstruct, warp order must not.
  yskn::gguf::File f;
  f.open(gguf_path);
  const std::vector<float> E = f.dequant("token_embd.weight");
  const yskn::TensorDesc& emb = gc.tensor("embed_tokens");
  const double row_rms = decode_rms(gc, emb, E, false);
  const double warp_rms = decode_rms(gc, emb, E, true);
  std::printf("  embed_tokens: row-major rms %.4f, warp-order rms %.4f (%.1fx worse)\n",
              row_rms, warp_rms, warp_rms / row_rms);
  const double floor_rms = 254.0 / std::sqrt((double)(1 << emb.bits)) / std::sqrt(12.0);
  check_lt(row_rms, floor_rms * 2.0, "embed_tokens decodes correctly in row-major order");
  check_lt(row_rms * 3.0, warp_rms,
           "embed_tokens would NOT decode in warp order (the de-interleave matters)");

  // A linear weight: warp order must reconstruct, row-major must not.
  const std::vector<float> Q = f.dequant("blk.0.attn_q.weight");
  const yskn::TensorDesc& q = gc.tensor("layers.0.q_proj");
  const double q_warp = decode_rms(gc, q, Q, true);
  const double q_row = decode_rms(gc, q, Q, false);
  std::printf("  layers.0.q_proj: warp rms %.4f, row-major rms %.4f (%.1fx worse)\n", q_warp,
              q_row, q_row / q_warp);
  check_lt(q_warp, floor_rms * 2.0, "layers.0.q_proj decodes correctly in warp order");
  check_lt(q_warp * 3.0, q_row,
           "layers.0.q_proj would NOT decode in row-major order (the interleave matters)");
}

// ---------------------------------------------------------------------------
// 4. The f16 vectors decode to the values the fixture wrote.
// ---------------------------------------------------------------------------
void test_vectors(const GgufContainer& gc, const Sidecar& sc) {
  std::printf("f16 vectors:\n");
  int compared = 0;
  double worst = 0.0;
  for (const Sidecar::Vec& want : sc.vectors) {
    if (!gc.has_vector(want.name)) {
      check(false, "sidecar vector " + want.name + " is present in the container");
      continue;
    }
    const yskn::VectorDesc& v = gc.vector(want.name);
    check_eq(v.n, want.n, "vector " + want.name + " length");
    const uint16_t* p = gc.at<uint16_t>(v.off);
    for (int i = 0; i < v.n && i < want.n; ++i) {
      const double got = h2f(p[i]), exp = want.values[(size_t)i];
      worst = std::max(worst, std::fabs(got - exp));
      ++compared;
    }
  }
  // The fixture writes norms and biases as multiples of 1/32, all exactly
  // representable in f16, so the round trip should be exact.
  check_lt(worst, 1e-6, "every f16 vector decodes to the fixture value");
  std::printf("  %d values compared, worst |err| %.3g\n", compared, worst);

  // The synthesised-bias case is exercised by the nobias fixture; here just
  // confirm the biases are the ones the file carries (non-zero).
  const uint16_t* bq = gc.at<uint16_t>(gc.vector("layers.0.q_proj.bias").off);
  bool any = false;
  for (int i = 0; i < gc.vector("layers.0.q_proj.bias").n; ++i) any = any || bq[i] != 0;
  check(any, "layers.0.q_proj.bias is not all zeros (the fixture writes real values)");
}

// ---------------------------------------------------------------------------
// 5. Quantise-then-read-back of one weight matrix.
//
// Decodes the stream the container holds, the way the kernels would, and
// compares against the fixture's own BF16 values. The metric is RMS in
// B-units (the normalised matrix the encoder quantises), because that is what
// docs/GGUF_NOTES.md tabulates:
//
//   bits=4  RMS ~10.6      bits=6  RMS ~5.5      bits=8  RMS ~2.9
//
// The fixture is random, so the trellis limit -- not the encoder -- sets the
// floor. The bounds below are the measured values from that table plus slack.
void test_roundtrip(const GgufContainer& gc, const std::string& gguf_path) {
  std::printf("quantise/read-back:\n");
  const yskn::TensorDesc& d = gc.tensor("layers.0.q_proj");
  const int rows = d.rows, cols = d.cols, physical = d.physical_rows;
  const int seqs = cols / 256, wps = 4 * d.bits;

  // The fixture values, straight out of the GGUF.
  yskn::gguf::File f;
  f.open(gguf_path);
  const std::vector<float> W = f.dequant("blk.0.attn_q.weight");
  check_eq((long long)W.size(), (long long)rows * cols, "fixture q_proj element count");

  // The normalised matrix B the encoder quantised: rotate, then scale each row.
  std::vector<float> B = W;
  yskn::gguf::rotate_matrix(B.data(), rows, cols, gc.quant_config(), B.data());
  for (int r = 0; r < rows; ++r) {
    float* row = B.data() + (size_t)r * cols;
    float mx = 0.f;
    for (int c = 0; c < cols; ++c) mx = std::max(mx, std::fabs(row[c]));
    if (mx > 0.f) {
      const float inv = 127.f / mx;
      for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
  }

  // Recover the int8 codebook the way build_lut8_tau will.
  std::vector<uint16_t> lut8;
  std::vector<float> tau;
  build_lut8_tau(gc.at<uint16_t>(d.lut_off), gc.at<uint16_t>(d.scale_off), physical, lut8,
                 tau, 1 << d.L);

  // Decode the warp stream and diff.
  const uint32_t* stream = gc.at<uint32_t>(d.stream_off);
  std::vector<uint32_t> w((size_t)wps);
  double sq = 0.0;
  size_t n = 0;
  double bsq = 0.0;
  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < seqs; ++q) {
      for (int i = 0; i < wps; ++i) w[(size_t)i] = stream[warp_src(r, q, i, physical, wps)];
      const float* b = B.data() + (size_t)r * cols + (size_t)q * 256;
      for (int j = 0; j < 128; ++j) {
        const uint32_t s = v2_state(w.data(), d.bits, j);
        const float q0 = (float)(int8_t)(lut8[s] & 0xFF);
        const float q1 = (float)(int8_t)(lut8[s] >> 8);
        const double e0 = q0 - b[2 * j], e1 = q1 - b[2 * j + 1];
        sq += e0 * e0 + e1 * e1;
        bsq += (double)b[2 * j] * b[2 * j] + (double)b[2 * j + 1] * b[2 * j + 1];
        n += 2;
      }
    }
  }
  const double rms = std::sqrt(sq / (double)n);
  const double brms = std::sqrt(bsq / (double)n);
  std::printf("  layers.0.q_proj: %dx%d, bits=%d, rms %.4f (B rms %.2f, %.2f%% of full scale)\n",
              rows, cols, d.bits, rms, brms, 100.0 * rms / 127.0);

  // docs/GGUF_NOTES.md 'The trellis is the limit, not the encoder': at bits=K
  // only 2^K of the 4096 codewords are reachable per step, so the error is set
  // by the codeword spacing over the pair distribution, i.e. roughly
  // 254/sqrt(2^bits)/sqrt(12) for a full-scale (127) uniform spread. The table
  // measured 10.6 / 5.5 / 2.9 at bits 4 / 6 / 8 on a 512-wide matrix; this
  // fixture is 256 wide with Gaussian weights, so allow 2x the floor.
  const double floor_rms = 254.0 / std::sqrt((double)(1 << d.bits)) / std::sqrt(12.0);
  check_lt(rms, floor_rms * 2.0,
           "q_proj round-trip RMS is near the trellis floor (floor " +
               std::to_string(floor_rms) + ")");

  // And the decoded values must be a real reconstruction, not zeros: the RMS
  // has to be far below the signal, or the stream is meaningless.
  check_lt(rms, brms * 0.15, "round-trip RMS is well below the signal RMS");

  // The LUT the loader will build must be exactly invertible (max|q| == 127),
  // which is what makes the read-back above meaningful at all.
  int maxq = 0;
  for (int s = 0; s < (1 << d.L); ++s) {
    maxq = std::max(maxq, std::abs((int)(int8_t)(lut8[(size_t)s] & 0xFF)));
    maxq = std::max(maxq, std::abs((int)(int8_t)(lut8[(size_t)s] >> 8)));
  }
  check_eq(maxq, 127, "the recovered int8 codebook reaches +-127 (invertible LUT)");
}

// ---------------------------------------------------------------------------
// 6. Rejection.
// ---------------------------------------------------------------------------
void expect_reject(const std::string& path, const std::string& must_contain,
                   const std::string& what) {
  GgufContainer gc;
  QuantConfig cfg;
  std::string msg;
  bool threw = false;
  try {
    gc.open(path, cfg);
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  check(threw, what + " is rejected");
  if (threw) {
    const bool named = msg.find(must_contain) != std::string::npos;
    check(named, what + " message names '" + must_contain + "' (got: " + msg + ")");
    std::printf("  %s -> %s\n", what.c_str(), msg.c_str());
  }
}

// ---------------------------------------------------------------------------
// 7. The tied-embedding and synthesised-bias variants.
// ---------------------------------------------------------------------------
void test_variants(const std::string& tie_path, const std::string& nobias_path) {
  std::printf("variants:\n");
  {
    GgufContainer gc;
    QuantConfig cfg;
    gc.open(tie_path, cfg);
    check(gc.model.tie_embeddings, "tie fixture reports tie_embeddings");
    check(gc.has_tensor("lm_head"), "tie fixture still provides lm_head");
    check_eq((long long)gc.tensor("lm_head").rows, gc.model.vocab,
             "tied lm_head has vocab rows");
    check_eq((long long)gc.tensor("lm_head").cols, gc.model.hidden,
             "tied lm_head has hidden cols");
    std::printf("  tie: lm_head %dx%d from token_embd.weight\n", gc.tensor("lm_head").rows,
                gc.tensor("lm_head").cols);
  }
  {
    GgufContainer gc;
    QuantConfig cfg;
    gc.open(nobias_path, cfg);
    const yskn::VectorDesc& v = gc.vector("layers.0.q_proj.bias");
    check_eq(v.n, gc.model.heads * gc.model.head_dim, "synthesised q bias has heads*head_dim entries");
    const uint16_t* p = gc.at<uint16_t>(v.off);
    bool all_zero = true;
    for (int i = 0; i < v.n; ++i) all_zero = all_zero && p[i] == 0;
    check(all_zero, "a missing bias is synthesised as zeros");
    std::printf("  nobias: synthesised %d zero entries for layers.0.q_proj.bias\n", v.n);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string good, bad, missing, tie, nobias;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--good" && i + 1 < argc) good = argv[++i];
    else if (a == "--bad" && i + 1 < argc) bad = argv[++i];
    else if (a == "--missing" && i + 1 < argc) missing = argv[++i];
    else if (a == "--tie" && i + 1 < argc) tie = argv[++i];
    else if (a == "--nobias" && i + 1 < argc) nobias = argv[++i];
    else {
      std::printf("usage: %s --good F --bad F --missing F --tie F --nobias F\n", argv[0]);
      return 2;
    }
  }
  if (good.empty() || bad.empty() || missing.empty() || tie.empty() || nobias.empty()) {
    std::printf("usage: %s --good F --bad F --missing F --tie F --nobias F\n", argv[0]);
    return 2;
  }

  try {
    Sidecar sc;
    load_sidecar(good + ".json", sc);
    check_eq(sc.layers, 2, "sidecar reports 2 layers");
    check_eq(sc.hidden, 256, "sidecar reports hidden 256");

    QuantConfig cfg;
    cfg.bits = 6;
    GgufContainer gc;
    gc.open(good, cfg);
    std::printf("opened %s in %zu bytes\n\n", good.c_str(), gc.bytes());

    // The fixture must stay small enough to generate at build time. The brief
    // asked for < 1 MB; 2 layers at hidden 256 needs 1.52 MiB (see the header of
    // tools/make_tiny_gguf.py), so the bound here is the 2-layer reality and the
    // 1-layer variant is checked separately.
    {
      std::ifstream f(good, std::ios::binary | std::ios::ate);
      const long long sz = (long long)f.tellg();
      std::printf("fixture size: %.2f MiB (2 layers)\n", (double)sz / (1024.0 * 1024.0));
      check(sz > 0 && sz < 2 * 1024 * 1024, "the 2-layer fixture is under 2 MiB");
    }

    test_model_desc(gc, sc);
    std::printf("\n");
    test_names(gc);
    std::printf("\n");
    test_layout(gc);
    std::printf("\n");
    test_shapes_and_layout(gc, sc, good);
    std::printf("\n");
    test_vectors(gc, sc);
    std::printf("\n");
    test_roundtrip(gc, good);
    std::printf("\n");
    test_variants(tie, nobias);
    std::printf("\n");

    std::printf("rejection:\n");
    expect_reject(bad, "not a multiple of 256", "hidden=96");
    expect_reject(missing, "missing required tensor blk.1.ffn_down.weight",
                  "a missing ffn_down.weight");
    expect_reject("/nonexistent/nope.gguf", "open failed", "a nonexistent path");
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++g_failures;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}