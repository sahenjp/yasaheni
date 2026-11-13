// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for the GGUF -> Model::load dispatch: the last wiring slice that lets
// yskn_open("something.gguf") load a real GGUF into the legacy Engine.
//
// ---------------------------------------------------------------------------
// Why this test does not need a GPU
// ---------------------------------------------------------------------------
// Model::load() is two things:
//
//   Model::open_backing()   peek 4 bytes, open Container or GgufContainer,
//                           set cfg and gguf_          <-- pure host code
//   everything after it     cudaMalloc + upload        <-- needs a GPU
//
// The *dispatch* lives entirely in open_backing(), and the accessor resolution
// lives in the free helpers T()/V()/at<>() (src/model.hpp). Both are host-only,
// so this binary links no CUDA and runs anywhere. What it cannot prove is that
// the uploaded weights produce correct logits; that needs head_dim 128 AND
// ffn%512 AND a real tokenizer, and is out of scope for this slice.
//
// ---------------------------------------------------------------------------
// What is checked
// ---------------------------------------------------------------------------
//   1. A file whose first four bytes are "YSKN" still takes the YSKN path:
//      gguf_ stays false and Container::open's own error text is unchanged.
//   2. A real GGUF fixture takes the GGUF path: gguf_ == true and cfg matches
//      the sidecar make_tiny_gguf.py wrote next to it.
//   3. T()/V()/at<T>() resolve the same names whichever container is live.
//   4. The supported gate accepts a head_dim-128 GGUF and rejects a head_dim-64
//      one with the attention-kernel reason.
//
// Exits non-zero on failure and prints a PASS/FAIL summary.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model.hpp"
#include "gguf/gguf_engine.hpp"

namespace {

using json = nlohmann::json;

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

template <class A, class B>
void check_eq(const A& got, const B& want, const std::string& what) {
  const bool ok = (got == want);
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::ostringstream os;
    os << what << " (got " << got << ", want " << want << ")";
    g_failure_log.push_back(os.str());
    std::printf("FAIL: %s\n", os.str().c_str());
  }
}

void check_near(double got, double want, double tol, const std::string& what) {
  const bool ok = (got - want <= tol) && (want - got <= tol);
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::ostringstream os;
    os << what << " (got " << got << ", want " << want << " +-" << tol << ")";
    g_failure_log.push_back(os.str());
    std::printf("FAIL: %s\n", os.str().c_str());
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

void check_contains(const std::string& haystack, const std::string& needle,
                    const std::string& what) {
  check(contains(haystack, needle), what + " contains '" + needle + "'");
}

// Run a callable that is expected to throw std::exception; return what().
template <class F>
std::string error_of(F&& f) {
  try {
    f();
  } catch (const std::exception& ex) {
    return ex.what();
  }
  return std::string();
}

// ---------------------------------------------------------------------------
// A hand-written minimal YSKN container.
//
// src/format.hpp:26-61 is the reader; this writes the smallest file it accepts:
// magic "YSKN", version 1, a JSON header at 0x20, then the payload the JSON's
// offsets point at. One tensor (cols 256 so the cols%256 check passes, L=12,
// d=2, T=256, bits=6) and one f16 vector.
// ---------------------------------------------------------------------------
void put_u32(std::string& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
// Offsets inside a YSKN container are absolute from the start of the file
// (Container::at is base + off), so the payload begins right after the header.
constexpr uint64_t kYsknHeader = 0x20;

void write_yskn(const std::string& path) {
  const int rows = 64, cols = 256, bits = 6;
  const int nvec = 256;

  const uint64_t lut_off = kYsknHeader;                 // 4096*2 f16
  const uint64_t scale_off = lut_off + 4096 * 2;        // rows f16
  const uint64_t stream_off = scale_off + rows * 2;     // rows*seqs*4*bits u32
  const uint64_t vec_off = stream_off + (uint64_t)rows * (cols / 256) * (4 * bits) * 4;

  json j;
  j["model"] = {{"arch", "yskn-test"}, {"layers", 1},   {"hidden", 256},
                {"ffn", 512},          {"heads", 2},     {"kv_heads", 1},
                {"head_dim", 128},     {"vocab", 16},    {"eps", 1e-6},
                {"rope_theta", 10000.0}, {"tie_embeddings", false}};
  j["tensors"] = json::array();
  j["tensors"].push_back({{"name", "layers.0.q_proj"},
                          {"role", "linear"},
                          {"rows", rows},
                          {"cols", cols},
                          {"bits", bits},
                          {"l", 12},
                          {"d", 2},
                          {"t", 256},
                          {"lut_off", lut_off},
                          {"scale_off", scale_off},
                          {"stream_off", stream_off},
                          {"stream_bytes", (uint64_t)rows * (cols / 256) * (4 * bits) * 4}});
  j["vectors"] = json::array();
  j["vectors"].push_back(
      {{"name", "norm"}, {"off", vec_off}, {"n", nvec}, {"dtype", "f16"}});

  const std::string js = j.dump();

  std::string out;
  out += "YSKN";
  put_u32(out, 1);                                  // version
  put_u32(out, static_cast<uint32_t>(js.size()));   // json_len
  out.append(kYsknHeader - out.size(), '\0');       // pad to 0x20
  out += js;

  // lut16: 4096 f16, all 1.0 is fine -- nothing here decodes it.
  out.append(4096 * 2, '\0');
  out.append(rows * 2, '\0');                       // scale16
  out.append((size_t)rows * (cols / 256) * (4 * bits) * 4, '\0');  // stream
  out.append((size_t)nvec * 2, '\0');               // f16 vector

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::printf("FATAL: cannot write %s (is the output directory there?)\n", path.c_str());
    std::exit(2);
  }
  f.write(out.data(), static_cast<std::streamsize>(out.size()));
  if (!f) {
    std::printf("FATAL: short write to %s\n", path.c_str());
    std::exit(2);
  }
}

// A file that starts with "YSKN" but is shorter than the 0x20-byte header, so
// Container::open rejects it with its own message (src/format.hpp:28).
void write_short_yskn(const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::printf("FATAL: cannot write %s\n", path.c_str());
    std::exit(2);
  }
  f.write("YSKN", 4);
}

// ---------------------------------------------------------------------------
// 1. YSKN magic still takes the YSKN path
// ---------------------------------------------------------------------------
void check_yskn_dispatch(const std::string& valid_path, const std::string& short_path) {
  std::printf("\n-- YSKN dispatch\n");

  // A valid container: the YSKN branch must open it and leave gguf_ false.
  {
    yskn::Model m;
    m.open_backing(valid_path);
    check_eq(m.gguf_, false, "yskn: gguf_ stays false");
    check_eq(m.cfg.arch, std::string("yskn-test"), "yskn: cfg.arch");
    check_eq(m.cfg.hidden, 256, "yskn: cfg.hidden");
    check_eq(m.cfg.head_dim, 128, "yskn: cfg.head_dim");
    check_eq(m.cfg.layers, 1, "yskn: cfg.layers");
    check_eq(m.cfg.vocab, 16, "yskn: cfg.vocab");
    // cfg must be c.model, and the Container must be the live one.
    check_eq(m.cfg.hidden, m.c.model.hidden, "yskn: cfg came from c.model");
    check(m.c.base != nullptr, "yskn: the Container is mapped");
    check(m.c.len > 0, "yskn: the mapping has bytes");
  }

  // A truncated container: the error text is Container::open's, unchanged. If
  // the GGUF branch had been taken the message would be GgufContainer's, so
  // this pins the routing as well as the wording.
  {
    yskn::Model m;
    const std::string err = error_of([&] { m.open_backing(short_path); });
    check_eq(m.gguf_, false, "short yskn: gguf_ stays false");
    check_eq(err, std::string("not a YSKN container"),
             "short yskn: the old Container::open message is unchanged");
  }
}

// ---------------------------------------------------------------------------
// 2. GGUF magic takes the GGUF path
// ---------------------------------------------------------------------------
void check_gguf_dispatch(const std::string& gguf_path, const std::string& sidecar_path) {
  std::printf("\n-- GGUF dispatch: %s\n", gguf_path.c_str());

  std::ifstream sf(sidecar_path);
  check(static_cast<bool>(sf), "gguf: the sidecar opens");
  if (!sf) return;
  json side;
  sf >> side;
  const json& want = side.at("model");

  yskn::Model m;
  m.open_backing(gguf_path);

  check_eq(m.gguf_, true, "gguf: gguf_ is true");
  check_eq(m.cfg.arch, want.at("arch").get<std::string>(), "gguf: cfg.arch");
  check_eq(m.cfg.layers, want.at("layers").get<int>(), "gguf: cfg.layers");
  check_eq(m.cfg.hidden, want.at("hidden").get<int>(), "gguf: cfg.hidden");
  check_eq(m.cfg.ffn, want.at("ffn").get<int>(), "gguf: cfg.ffn");
  check_eq(m.cfg.heads, want.at("heads").get<int>(), "gguf: cfg.heads");
  check_eq(m.cfg.kv_heads, want.at("kv_heads").get<int>(), "gguf: cfg.kv_heads");
  check_eq(m.cfg.head_dim, want.at("head_dim").get<int>(), "gguf: cfg.head_dim");
  check_eq(m.cfg.vocab, want.at("vocab").get<int>(), "gguf: cfg.vocab");
  check_near(m.cfg.eps, want.at("eps").get<double>(), 1e-12, "gguf: cfg.eps");
  check_near(m.cfg.rope_theta, want.at("rope_theta").get<double>(), 1.0,
             "gguf: cfg.rope_theta");
  check_eq(m.cfg.tie_embeddings, want.at("tie_embeddings").get<bool>(),
           "gguf: cfg.tie_embeddings");

  // cfg must be gc.model, not a stale copy of c.model.
  check_eq(m.cfg.hidden, m.gc.model.hidden, "gguf: cfg came from gc.model");
  check(m.gc.is_open(), "gguf: the GgufContainer is open");
  check(m.gc.bytes() > m.gc.payload_bytes(), "gguf: the tail pad is present");
}

// ---------------------------------------------------------------------------
// 3. T()/V()/at<T>() resolve the same names for both container kinds
// ---------------------------------------------------------------------------
void check_accessors(const std::string& yskn_path, const std::string& gguf_path) {
  std::printf("\n-- accessor resolution\n");

  // YSKN side: the hand-written container has layers.0.q_proj and norm.
  {
    yskn::Model m;
    m.open_backing(yskn_path);
    check_eq(m.gguf_, false, "T()/yskn: gguf_ false");

    const yskn::TensorDesc& q = yskn::T(m, "layers.0.q_proj");
    check_eq(q.name, std::string("layers.0.q_proj"), "T()/yskn: tensor name");
    check_eq(q.rows, 64, "T()/yskn: tensor rows");
    check_eq(q.cols, 256, "T()/yskn: tensor cols");
    check_eq(q.bits, 6, "T()/yskn: tensor bits");

    const yskn::VectorDesc& v = yskn::V(m, "norm");
    check_eq(v.name, std::string("norm"), "V()/yskn: vector name");
    check_eq(v.n, 256, "V()/yskn: vector n");

    // at<T>() must hand back a pointer inside the mapping at the desc's offset.
    const uint16_t* p = yskn::at<uint16_t>(m, v.off);
    check(p != nullptr, "at<>()/yskn: non-null");
    check_eq(reinterpret_cast<const uint8_t*>(p) - m.c.base,
             static_cast<ptrdiff_t>(v.off), "at<>()/yskn: pointer is base + off");

    // A missing name throws the container's message, not a crash.
    check_eq(error_of([&] { yskn::T(m, "layers.9.q_proj"); }),
             std::string("missing tensor layers.9.q_proj"),
             "T()/yskn: missing tensor message");
    check_eq(error_of([&] { yskn::V(m, "nope"); }),
             std::string("missing vector nope"), "V()/yskn: missing vector message");
  }

  // GGUF side: the same call sites, the same names, a different container.
  {
    yskn::Model m;
    m.open_backing(gguf_path);
    check_eq(m.gguf_, true, "T()/gguf: gguf_ true");

    const yskn::TensorDesc& q = yskn::T(m, "layers.0.q_proj");
    check_eq(q.name, std::string("layers.0.q_proj"), "T()/gguf: tensor name");
    check_eq(q.rows, m.gc.model.heads * m.gc.model.head_dim, "T()/gguf: q_proj rows");
    check_eq(q.cols, m.gc.model.hidden, "T()/gguf: q_proj cols");
    check_eq(q.layout, std::string("warp"), "T()/gguf: linear weights are warp layout");

    const yskn::TensorDesc& e = yskn::T(m, "embed_tokens");
    check_eq(e.layout, std::string("row"), "T()/gguf: the embedding is row layout");

    const yskn::VectorDesc& v = yskn::V(m, "norm");
    check_eq(v.name, std::string("norm"), "V()/gguf: vector name");
    check_eq(v.n, m.gc.model.hidden, "V()/gguf: vector n");

    const uint16_t* p = yskn::at<uint16_t>(m, v.off);
    check(p != nullptr, "at<>()/gguf: non-null");
    check_eq(reinterpret_cast<const uint8_t*>(p) - m.gc.at<uint8_t>(0),
             static_cast<ptrdiff_t>(v.off), "at<>()/gguf: pointer is buf + off");

    // The synthesised zero bias: Model::load always asks for it, Qwen3 has none.
    const yskn::VectorDesc& bq = yskn::V(m, "layers.0.q_proj.bias");
    check_eq(bq.n, m.gc.model.heads * m.gc.model.head_dim, "V()/gguf: q_proj.bias n");

    check_eq(error_of([&] { yskn::T(m, "layers.9.q_proj"); }),
             std::string("missing tensor layers.9.q_proj"),
             "T()/gguf: the same missing-tensor message as the YSKN side");
  }

  // The explicit-container override (the d3 draft model, src/engine.cpp:553-565)
  // must still resolve against the Container it was handed, whatever gguf_ says.
  // A GGUF-backed Model has an empty c, so a hit proves the override was used.
  {
    yskn::Model m;
    m.open_backing(yskn_path);  // c is the live container here
    const yskn::Model g = [&] {
      yskn::Model t;
      t.open_backing(gguf_path);
      return t;
    }();
    check_eq(g.gguf_, true, "override: the model itself is GGUF");
    check_eq(g.c.tensors.size(), size_t{0}, "override: its Container is empty");
    const std::string name = "layers.0.q_proj";
    // With the override the descriptor comes from the YSKN container (rows 64).
    const yskn::TensorDesc& from_c = yskn::T(g, name, &m.c);
    check_eq(from_c.name, name, "override: a src Container short-circuits the dispatch");
    check_eq(from_c.rows, 64, "override: the descriptor came from the YSKN container");
    check_eq(from_c.cols, 256, "override: the YSKN q_proj is 256 wide");
    // Without it, the same call resolves from gc instead: same name, different
    // descriptor. That difference is what proves the override was honoured.
    const yskn::TensorDesc& from_gc = yskn::T(g, name);
    check_eq(from_gc.name, name, "override: without src the name still resolves");
    check_eq(from_gc.rows, g.gc.model.heads * g.gc.model.head_dim,
             "override: without src the rows come from the GGUF model");
    check(from_gc.rows != from_c.rows,
          "override: the two containers really do disagree about q_proj");
  }
}

// ---------------------------------------------------------------------------
// 4. The supported gate
// ---------------------------------------------------------------------------
yskn::gguf::ModelConfig config_from_sidecar(const std::string& sidecar_path) {
  yskn::gguf::ModelConfig c;
  std::ifstream sf(sidecar_path);
  if (!sf) return c;
  json side;
  sf >> side;
  const json& m = side.at("model");
  c.arch = m.at("arch").get<std::string>();
  c.layers = m.at("layers").get<int>();
  c.hidden = m.at("hidden").get<int>();
  c.ffn = m.at("ffn").get<int>();
  c.heads = m.at("heads").get<int>();
  c.kv_heads = m.at("kv_heads").get<int>();
  c.head_dim = m.at("head_dim").get<int>();
  c.vocab = m.at("vocab").get<int>();
  c.eps = m.at("eps").get<float>();
  c.rope_theta = m.at("rope_theta").get<float>();
  c.tie_embeddings = m.at("tie_embeddings").get<bool>();
  return c;
}

void check_gate(const std::string& hd128_sidecar, const std::string& hd64_sidecar) {
  std::printf("\n-- supported gate\n");

  // head_dim 128 with 28 heads: the legacy Engine's kernels can run this shape.
  // 28 x 128 is not a preference, it is what src/post_kernels.cu:47 and
  // src/attention_kv4q.cu:81 literally index with.
  {
    yskn::gguf::ModelConfig c = config_from_sidecar(hd128_sidecar);
    c.heads = 28;
    c.kv_heads = 4;
    check_eq(c.head_dim, 128, "gate: the accept fixture really is head_dim 128");
    check_eq(c.ffn % 512, 0, "gate: the accept fixture has ffn%512 == 0");
    std::string reason;
    check_eq(yskn::gguf::engine_supports(c, reason), true,
             "gate: accepts head_dim 128 with 28 heads");
    // A supported config explains itself: yskn_open loads it instead of
    // returning NULL, so the reason states what was satisfied.
    check_contains(reason, "kernel constraints are met",
                   "gate: a supported config says why it is supported");
    check_contains(reason, "Model::load", "gate: the reason names the load path");
  }

  // head_dim 64: no attention kernel was instantiated for it.
  {
    const yskn::gguf::ModelConfig c = config_from_sidecar(hd64_sidecar);
    check_eq(c.head_dim, 64, "gate: the reject fixture really is head_dim 64");
    std::string reason;
    check_eq(yskn::gguf::engine_supports(c, reason), false,
             "gate: rejects head_dim 64");
    check_contains(reason, "head_dim is 64", "gate: names the head_dim value");
    check_contains(reason, "the default attention path hard-codes 128",
                   "gate: names the attention-kernel constraint");
    check_contains(reason, "post_kernels.cu:47", "gate: cites the qkv_post site");
    check_contains(reason, "engine.cpp:41", "gate: cites the q16 allocation");
  }

  // head_dim 256 must be REFUSED. The generic attention path is templated for
  // it, but that path is off by default, and on the default path a head_dim-256
  // model loads and then returns NaN logits (measured on the RTX 3070). A gate
  // that admits it would hand the user a silently wrong answer.
  {
    yskn::gguf::ModelConfig c = config_from_sidecar(hd128_sidecar);
    c.head_dim = 256;
    c.heads = 28;
    std::string reason;
    check_eq(yskn::gguf::engine_supports(c, reason), false,
             "gate: rejects head_dim 256 (NaN on the default path)");
    check_contains(reason, "head_dim is 256", "gate: names the refused head_dim");
    check_contains(reason, "NaN", "gate: says what happens if it is admitted");
  }

  // ffn must still be a multiple of 512.
  {
    yskn::gguf::ModelConfig c = config_from_sidecar(hd128_sidecar);
    c.ffn = 500;
    std::string reason;
    check_eq(yskn::gguf::engine_supports(c, reason), false,
             "gate: rejects ffn not a multiple of 512");
    check_contains(reason, "ffn is 500", "gate: names the ffn value");
  }

  // Both constraints at once: the reason is the joined list.
  {
    yskn::gguf::ModelConfig c = config_from_sidecar(hd128_sidecar);
    c.head_dim = 96;
    c.ffn = 500;
    std::string reason;
    check_eq(yskn::gguf::engine_supports(c, reason), false,
             "gate: rejects a double mismatch");
    check_contains(reason, "head_dim is 96", "gate: joined reason has head_dim");
    check_contains(reason, "ffn is 500", "gate: joined reason has ffn");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string yskn_path, short_path, gguf_path, gguf_sidecar;
  std::string hd128_path, hd128_sidecar, hd64_path, hd64_sidecar, out_dir = ".";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--yskn") yskn_path = next();
    else if (a == "--short") short_path = next();
    else if (a == "--gguf") gguf_path = next();
    else if (a == "--gguf-sidecar") gguf_sidecar = next();
    else if (a == "--hd128") hd128_path = next();
    else if (a == "--hd128-sidecar") hd128_sidecar = next();
    else if (a == "--hd64") hd64_path = next();
    else if (a == "--hd64-sidecar") hd64_sidecar = next();
    else if (a == "--out-dir") out_dir = next();
    else { std::printf("unknown argument: %s\n", a.c_str()); return 2; }
  }

  if (yskn_path.empty()) yskn_path = out_dir + "/gguf_load_test.yskn";
  if (short_path.empty()) short_path = out_dir + "/gguf_load_test_short.yskn";

  std::printf("gguf_load_test\n");
  std::printf("  yskn       %s\n", yskn_path.c_str());
  std::printf("  gguf       %s\n", gguf_path.c_str());
  std::printf("  hd128      %s\n", hd128_path.c_str());
  std::printf("  hd64       %s\n", hd64_path.c_str());

  write_yskn(yskn_path);
  write_short_yskn(short_path);

  check_yskn_dispatch(yskn_path, short_path);
  if (!gguf_path.empty() && !gguf_sidecar.empty())
    check_gguf_dispatch(gguf_path, gguf_sidecar);
  if (!gguf_path.empty()) check_accessors(yskn_path, gguf_path);
  if (!hd128_sidecar.empty() && !hd64_sidecar.empty())
    check_gate(hd128_sidecar, hd64_sidecar);

  std::remove(yskn_path.c_str());
  std::remove(short_path.c_str());

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  if (g_failures) {
    std::printf("\nFailures:\n");
    for (const std::string& s : g_failure_log) std::printf("  - %s\n", s.c_str());
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
