// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for the GGUF -> engine adapter: config extraction, tensor-name
// mapping, coverage, validation, describe() and the cached float32 accessor.
//
// The fixtures come from make_fixture_model.py (a 2-layer Qwen3-shaped model
// and two deliberately damaged variants). Exits non-zero on failure and prints
// a PASS/FAIL summary.
#include "gguf_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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
  const bool ok = std::fabs(got - want) <= tol;
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::ostringstream os;
    os << what << " (got " << got << ", want " << want << " +-" << tol << ")";
    g_failure_log.push_back(os.str());
    std::printf("FAIL: %s\n", os.str().c_str());
  }
}

// Render a GGUF shape as "[64, 32]" so check_eq can print it on failure.
std::string shape_str(const std::vector<uint64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ", ";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

// Substring search, used for the describe() key lines.
bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

void check_contains(const std::string& haystack, const std::string& needle,
                    const std::string& what) {
  check(contains(haystack, needle), what + " contains '" + needle + "'");
}

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

std::string exe_dir(const char* argv0) {
  const std::string p = argv0 ? argv0 : "";
  const size_t slash = p.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

bool in_list(const std::vector<std::string>& v, const std::string& s) {
  for (const std::string& x : v)
    if (x == s) return true;
  return false;
}

// ---------------------------------------------------------------------------
// 1. Config parsing
// ---------------------------------------------------------------------------
void check_config(const yskn::gguf::ModelConfig& c) {
  check_eq(c.arch, std::string("qwen3"), "arch");
  check_eq(c.layers, 2, "layers");
  check_eq(c.hidden, 64, "hidden");
  check_eq(c.ffn, 128, "ffn");
  check_eq(c.heads, 4, "heads");
  check_eq(c.kv_heads, 2, "kv_heads");
  check_eq(c.head_dim, 16, "head_dim");
  check_eq(c.vocab, 128, "vocab");
  check_near(c.eps, 1e-6, 1e-12, "eps");
  check_near(c.rope_theta, 1000000.0, 1.0, "rope_theta");
  check_eq(c.tie_embeddings, false, "tie_embeddings (output.weight present)");
  check_eq(c.rope_dim, 16, "rope_dim");
  check_eq(c.context_length, 4096, "context_length");
  check_eq(c.sliding_window, 4096, "sliding_window");
  check_eq(c.experts, 0, "experts");
  check_eq(c.has_qk_norm, true, "has_qk_norm");
  check_eq(c.q_dim(), 64, "q_dim");
  check_eq(c.kv_dim(), 32, "kv_dim");

  // No fallback should have fired: every key is present in the fixture.
  check(!c.layers_from_fallback, "block_count read from metadata");
  check(!c.hidden_from_fallback, "embedding_length read from metadata");
  check(!c.ffn_from_fallback, "feed_forward_length read from metadata");
  check(!c.head_dim_from_fallback, "attention.key_length read from metadata");
  check(!c.vocab_from_fallback, "vocab_size read from metadata");
  check(!c.eps_defaulted, "epsilon read from metadata");
  check(!c.rope_theta_defaulted, "rope.freq_base read from metadata");
}

// ---------------------------------------------------------------------------
// 2. Name mapping
// ---------------------------------------------------------------------------
struct Expect {
  const char* gguf;
  int layer;
  const char* role;
  bool bias;
  bool norm;
  const char* engine;
};

const Expect kExpected[] = {
    {"token_embd.weight", -1, "embedding", false, false, "embed_tokens"},
    {"output_norm.weight", -1, "final_norm", false, true, "norm"},
    {"output.weight", -1, "head", false, false, "lm_head"},
    {"blk.0.attn_norm.weight", 0, "ln1", false, true, "layers.0.input_layernorm"},
    {"blk.0.ffn_norm.weight", 0, "ln2", false, true, "layers.0.post_attention_layernorm"},
    {"blk.0.attn_q.weight", 0, "q", false, false, "layers.0.self_attn.q_proj"},
    {"blk.0.attn_k.weight", 0, "k", false, false, "layers.0.self_attn.k_proj"},
    {"blk.0.attn_v.weight", 0, "v", false, false, "layers.0.self_attn.v_proj"},
    {"blk.0.attn_output.weight", 0, "o", false, false, "layers.0.self_attn.o_proj"},
    {"blk.0.ffn_gate.weight", 0, "gate", false, false, "layers.0.mlp.gate_proj"},
    {"blk.0.ffn_up.weight", 0, "up", false, false, "layers.0.mlp.up_proj"},
    {"blk.0.ffn_down.weight", 0, "down", false, false, "layers.0.mlp.down_proj"},
    {"blk.0.attn_q_norm.weight", 0, "q_norm", false, true, "layers.0.self_attn.q_norm"},
    {"blk.0.attn_k_norm.weight", 0, "k_norm", false, true, "layers.0.self_attn.k_norm"},
    {"blk.1.attn_q.weight", 1, "q", false, false, "layers.1.self_attn.q_proj"},
    {"blk.1.ffn_down.weight", 1, "down", false, false, "layers.1.mlp.down_proj"},
    // Biases and Qwen3.5 hybrid names are mapped even when absent from this
    // fixture: the mapper must not depend on the tensor being present.
    {"blk.3.attn_q.bias", 3, "q", true, false, "layers.3.self_attn.q_proj.bias"},
    {"blk.3.attn_k.bias", 3, "k", true, false, "layers.3.self_attn.k_proj.bias"},
    {"blk.3.attn_v.bias", 3, "v", true, false, "layers.3.self_attn.v_proj.bias"},
    {"blk.3.attn_gate.weight", 3, "attn_gate", false, false,
     "layers.3.self_attn.attn_gate"},
    {"blk.3.ssm_a.weight", 3, "ssm_a", false, false, "layers.3.linear_attn.A_log"},
    {"blk.3.ssm_alpha.weight", 3, "ssm_alpha", false, false,
     "layers.3.linear_attn.in_proj_a"},
    {"blk.3.ssm_beta.weight", 3, "ssm_beta", false, false,
     "layers.3.linear_attn.in_proj_b"},
    {"blk.3.ssm_conv1d.weight", 3, "ssm_conv1d", false, false,
     "layers.3.linear_attn.conv1d"},
    {"blk.3.ssm_dt.weight", 3, "ssm_dt", false, false, "layers.3.linear_attn.dt_bias"},
    {"blk.3.ssm_norm.weight", 3, "ssm_norm", false, true, "layers.3.linear_attn.norm"},
    {"blk.3.ssm_out.weight", 3, "ssm_out", false, false,
     "layers.3.linear_attn.out_proj"},
    {"blk.3.ffn_gate_inp.weight", 3, "ffn_gate_inp", false, false,
     "layers.3.ffn_gate_inp"},
    {"blk.3.ffn_gate_exps.weight", 3, "ffn_gate_exps", false, false,
     "layers.3.ffn_gate_exps"},
    {"blk.3.ffn_up_exps.weight", 3, "ffn_up_exps", false, false,
     "layers.3.ffn_up_exps"},
    {"blk.3.ffn_down_exps.weight", 3, "ffn_down_exps", false, false,
     "layers.3.ffn_down_exps"},
};

void check_mapping() {
  for (const Expect& e : kExpected) {
    yskn::gguf::MappedTensor m;
    const bool ok = yskn::gguf::map_tensor(std::string(e.gguf), m);
    check(ok, std::string("map_tensor accepts ") + e.gguf);
    if (!ok) continue;
    check_eq(m.layer, e.layer, std::string("layer of ") + e.gguf);
    check_eq(m.role, std::string(e.role), std::string("role of ") + e.gguf);
    check_eq(m.is_bias, e.bias, std::string("is_bias of ") + e.gguf);
    check_eq(m.is_norm, e.norm, std::string("is_norm of ") + e.gguf);
    check_eq(m.engine_name, std::string(e.engine), std::string("engine name of ") + e.gguf);
    check_eq(m.has_layer(), e.layer >= 0, std::string("has_layer of ") + e.gguf);
  }

  // Names that must not map.
  const char* rejects[] = {
      "blk.0.attn_rot.weight",   // unknown suffix
      "blk.0.attn_q",            // no .weight/.bias
      "blk.0.attn_q.scale",      // unknown qualifier
      "blk.x.attn_q.weight",     // non-numeric layer
      "blk..attn_q.weight",      // empty layer
      "blk.0.",                  // nothing after the index
      "some_other_tensor",       // not a block, not a global
      "token_embd.scale",        // global with an unknown qualifier
      "",
  };
  for (const char* name : rejects) {
    yskn::gguf::MappedTensor m;
    check(!yskn::gguf::map_tensor(std::string(name), m),
          std::string("map_tensor rejects '") + name + "'");
  }

  // The shape-carrying overload copies the tensor info.
  yskn::gguf::TensorInfo t;
  t.name = "blk.1.attn_k.weight";
  t.shape = {64, 32};
  t.n_elements = 2048;
  yskn::gguf::MappedTensor m;
  check(yskn::gguf::map_tensor(t, m), "map_tensor(TensorInfo) accepts attn_k");
  check_eq(m.shape.size(), size_t{2}, "mapped shape rank");
  check_eq(m.cols(), uint64_t{64}, "mapped cols (ne[0] = fan-in)");
  check_eq(m.rows(), uint64_t{32}, "mapped rows (ne[1] = fan-out)");
}

// ---------------------------------------------------------------------------
// 3. Fixture-wide checks
// ---------------------------------------------------------------------------
void check_good_model(const std::string& path) {
  yskn::gguf::File f;
  f.open(path);

  std::printf("\n-- good fixture: %s\n", path.c_str());

  // Config.
  const yskn::gguf::ModelConfig cfg = yskn::gguf::read_config(f);
  check_config(cfg);

  // Every tensor in the file must map, and to the role its name implies.
  for (const yskn::gguf::TensorInfo& t : f.tensors()) {
    yskn::gguf::MappedTensor m;
    const bool ok = yskn::gguf::map_tensor(t, m);
    check(ok, "fixture tensor maps: " + t.name);
    if (!ok) continue;
    check_eq(shape_str(m.shape), shape_str(t.shape), "mapped shape of " + t.name);
  }

  // Coverage: nothing unmatched, nothing missing.
  const yskn::gguf::CoverageReport cov = yskn::gguf::coverage(f, cfg);
  check_eq(cov.unmatched.size(), size_t{0}, "coverage: unmatched tensors");
  check_eq(cov.missing_expected.size(), size_t{0}, "coverage: missing tensors");
  check_eq(cov.matched.size(), f.tensors().size(), "coverage: matched tensor count");
  check(cov.complete(), "coverage report is complete");
  check_contains(cov.to_string(), "unmatched: 0", "coverage text");
  check_contains(cov.to_string(), "missing:   0", "coverage text");

  // Validation.
  const yskn::gguf::ValidationReport val = yskn::gguf::validate(f, cfg);
  check(val.ok, "validate() returns ok");
  check_eq(val.errors.size(), size_t{0}, "validate() error count");
  if (!val.ok)
    for (const std::string& e : val.errors) std::printf("  error: %s\n", e.c_str());
  // The fixture declares an explicit output.weight, so the tie warning must not
  // fire; sliding_window is set, so that warning must.
  bool tie_warning = false;
  for (const std::string& w : val.warnings)
    if (contains(w, "tied")) tie_warning = true;
  check(!tie_warning, "no tied-embeddings warning when output.weight exists");

  // describe().
  const std::string d = yskn::gguf::describe(f);
  check_contains(d, "arch: qwen3", "describe()");
  check_contains(d, "layers: 2", "describe()");
  check_contains(d, "hidden: 64", "describe()");
  check_contains(d, "ffn: 128", "describe()");
  check_contains(d, "heads: 4", "describe()");
  check_contains(d, "kv heads: 2", "describe()");
  check_contains(d, "head_dim: 16", "describe()");
  check_contains(d, "vocab: 128", "describe()");
  check_contains(d, "eps: 1e-06", "describe()");
  check_contains(d, "rope_theta: 1e+06", "describe()");
  check_contains(d, "tie_embeddings: false", "describe()");
  check_contains(d, "tensors: 25", "describe()");
  check_contains(d, "total bytes:", "describe()");
  check_contains(d, "types:", "describe()");
  check_contains(d, "Q8_0:", "describe()");
  check_contains(d, "F32:", "describe()");
  check_contains(d, "coverage:", "describe()");
  check_contains(d, "validation: ok", "describe()");

  // Tensor count: 3 global + 2 blocks x 11 = 25.
  check_eq(f.tensors().size(), size_t{25}, "fixture tensor count");

  // ---- float32 accessor ------------------------------------------------
  yskn::gguf::clear_f32_cache();
  const std::string emb_name = "token_embd.weight";
  const std::vector<float>& emb = yskn::gguf::tensor_f32(f, emb_name);
  check_eq(emb.size(), size_t{64 * 128}, "tensor_f32 size of token_embd.weight");
  // The same call must return the very same buffer (the cache is doing work).
  const std::vector<float>& emb2 = yskn::gguf::tensor_f32(f, emb_name);
  check(&emb == &emb2, "tensor_f32 returns a cached buffer");

  // Values match the reader's own dequantization.
  const std::vector<float> direct = f.dequant("token_embd.weight");
  check_eq(emb.size(), direct.size(), "cached vs direct element count");
  bool same = emb.size() == direct.size();
  for (size_t i = 0; same && i < emb.size(); ++i)
    if (emb[i] != direct[i]) same = false;
  check(same, "cached values equal direct dequantization");

  // A norm vector is F32 and must round-trip exactly. The fixture fills it with
  // values(64, 5) = ((i * 5) % 29 - 14) * 0.25, all exactly representable.
  const std::string norm_name = "output_norm.weight";
  const std::vector<float>& norm = yskn::gguf::tensor_f32(f, norm_name);
  check_eq(norm.size(), size_t{64}, "tensor_f32 size of output_norm.weight");
  check_near(norm[0], -3.5, 1e-6, "output_norm.weight[0]");
  check_near(norm[1], -2.25, 1e-6, "output_norm.weight[1]");
  bool exact = true;
  for (size_t i = 0; i < norm.size(); ++i) {
    const float want = static_cast<float>((static_cast<int>(i) * 5) % 29 - 14) * 0.25f;
    if (norm[i] != want) exact = false;
  }
  check(exact, "output_norm.weight is an exact F32 round trip");

  // Missing names throw.
  bool threw = false;
  try {
    const std::string missing = "no_such_tensor";
    yskn::gguf::tensor_f32(f, missing);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "tensor_f32 throws for an unknown tensor");

  // The cache is keyed per file; a second File object sees its own entries.
  yskn::gguf::File f2;
  f2.open(path);
  check(!yskn::gguf::F32Cache{}.has(f2, "token_embd.weight"),
        "a fresh cache starts empty");
  yskn::gguf::F32Cache local;
  local.get(f2, "output_norm.weight");
  check(local.has(f2, "output_norm.weight"), "F32Cache::has sees a loaded tensor");
  check_eq(local.size(), size_t{1}, "F32Cache size after one load");
  check_eq(local.bytes(), size_t{64 * sizeof(float)}, "F32Cache byte count");
  local.clear();
  check_eq(local.size(), size_t{0}, "F32Cache::clear empties the cache");
}

// ---------------------------------------------------------------------------
// 4. Broken model: one wrong shape
// ---------------------------------------------------------------------------
void check_broken_model(const std::string& path) {
  yskn::gguf::File f;
  f.open(path);

  std::printf("\n-- broken fixture: %s\n", path.c_str());

  const yskn::gguf::ModelConfig cfg = yskn::gguf::read_config(f);
  const yskn::gguf::ValidationReport val = yskn::gguf::validate(f, cfg);

  check(!val.ok, "validate() rejects the broken model");
  check(!val.errors.empty(), "broken model reports at least one error");

  // The error must name the tensor, the bad dimension and the expected one.
  const std::string want = "blk.1.ffn_down.weight has 64 cols, expected 128";
  bool found = false;
  for (const std::string& e : val.errors)
    if (e == want) found = true;
  check(found, "broken model error is exactly: " + want);
  if (!found)
    for (const std::string& e : val.errors) std::printf("  got error: %s\n", e.c_str());

  // Exactly one error: the rest of the model is fine.
  check_eq(val.errors.size(), size_t{1}, "broken model error count");

  // describe() surfaces the failure too.
  const std::string d = yskn::gguf::describe(f);
  check_contains(d, "validation: FAILED", "describe() of the broken model");
  check_contains(d, want, "describe() of the broken model");
}

// ---------------------------------------------------------------------------
// 5. Unknown tensor: coverage must name it, validation must still pass
// ---------------------------------------------------------------------------
void check_unknown_tensor(const std::string& path) {
  yskn::gguf::File f;
  f.open(path);

  std::printf("\n-- unknown-tensor fixture: %s\n", path.c_str());

  const yskn::gguf::ModelConfig cfg = yskn::gguf::read_config(f);
  const yskn::gguf::CoverageReport cov = yskn::gguf::coverage(f, cfg);

  check_eq(cov.matched.size(), size_t{25}, "unknown: matched tensor count");
  check_eq(cov.unmatched.size(), size_t{1}, "unknown: unmatched tensor count");
  check(!cov.complete(), "unknown: coverage is not complete");
  check(in_list(cov.unmatched, "blk.0.attn_rot.weight"),
        "unknown: blk.0.attn_rot.weight is listed as unmatched");
  check_eq(cov.missing_expected.size(), size_t{0}, "unknown: nothing missing");
  check_contains(cov.to_string(), "unmatched tensor: blk.0.attn_rot.weight",
                 "unknown: coverage text names the tensor");

  // Shapes are all correct, so validation passes but warns about the extra.
  const yskn::gguf::ValidationReport val = yskn::gguf::validate(f, cfg);
  check(val.ok, "unknown: validate() still returns ok");
  bool warned = false;
  for (const std::string& w : val.warnings)
    if (contains(w, "blk.0.attn_rot.weight")) warned = true;
  check(warned, "unknown: validate() warns about the unmapped tensor");
}

// ---------------------------------------------------------------------------
// 6. Fallback paths: a file with almost no metadata
// ---------------------------------------------------------------------------
void check_fallbacks(const std::string& out_dir) {
  // Build a tiny GGUF by hand: one token_embd tensor, no hyperparameters.
  // This exercises the "metadata missing -> fallback" branches without needing
  // a second Python generator.
  const std::string path = out_dir + "/gguf_model_minimal.gguf";

  // 4 x 8 F32 embedding: shape [4, 8] in GGUF order.
  const uint32_t n_elem = 4 * 8;
  std::string buf;
  auto u32 = [](uint32_t v) { return std::string(reinterpret_cast<const char*>(&v), 4); };
  auto u64 = [](uint64_t v) { return std::string(reinterpret_cast<const char*>(&v), 8); };
  auto str = [&](const std::string& s) { return u64(s.size()) + s; };

  buf += "GGUF";
  buf += u32(3);
  buf += u64(1);  // one tensor
  buf += u64(1);  // one kv pair
  buf += str("general.architecture");
  buf += u32(8);  // STRING
  buf += str("llama");
  buf += str("token_embd.weight");
  buf += u32(2);  // rank
  buf += u64(4);
  buf += u64(8);
  buf += u32(0);  // F32
  buf += u64(0);  // offset
  while (buf.size() % 32 != 0) buf.push_back('\0');
  for (uint32_t i = 0; i < n_elem; ++i) {
    const float v = static_cast<float>(i) * 0.5f;
    buf.append(reinterpret_cast<const char*>(&v), 4);
  }

  {
    std::ofstream out(path, std::ios::binary);
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  }

  std::printf("\n-- minimal fixture: %s\n", path.c_str());

  yskn::gguf::File f;
  f.open(path);
  const yskn::gguf::ModelConfig cfg = yskn::gguf::read_config(f);

  check_eq(cfg.arch, std::string("llama"), "minimal: arch");
  check(cfg.layers_from_fallback, "minimal: block_count fell back");
  check(cfg.hidden_from_fallback, "minimal: embedding_length fell back");
  check(cfg.head_dim_from_fallback, "minimal: head_dim fell back");
  check(cfg.vocab_from_fallback, "minimal: vocab fell back to embedding rows");
  check_eq(cfg.vocab, 8, "minimal: vocab from token_embd rows");
  check(cfg.eps_defaulted, "minimal: epsilon defaulted");
  check(cfg.rope_theta_defaulted, "minimal: rope_theta defaulted");
  check_near(cfg.eps, 1e-6, 1e-12, "minimal: default epsilon");
  check_near(cfg.rope_theta, 10000.0, 1e-9, "minimal: default rope_theta");
  check_eq(cfg.tie_embeddings, true, "minimal: tied (no output.weight)");

  const yskn::gguf::ValidationReport val = yskn::gguf::validate(f, cfg);
  check(!val.ok, "minimal: validate() fails without hyperparameters");
  check_eq(val.errors.size() > 0, true, "minimal: errors are reported");
  bool named = false;
  for (const std::string& e : val.errors)
    if (contains(e, "block_count")) named = true;
  check(named, "minimal: the error names block_count");

  // The dequantized values must still be reachable.
  yskn::gguf::clear_f32_cache();
  const std::string emb_name = "token_embd.weight";
  const std::vector<float>& v = yskn::gguf::tensor_f32(f, emb_name);
  check_eq(v.size(), size_t{n_elem}, "minimal: tensor_f32 element count");
  check_near(v[3], 1.5, 1e-6, "minimal: tensor_f32 value 3");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = exe_dir(argc > 0 ? argv[0] : nullptr);
  std::string good = dir + "/fixture/gguf_model.gguf";
  std::string broken = dir + "/fixture/gguf_model_broken.gguf";
  std::string unknown = dir + "/fixture/gguf_model_unknown.gguf";
  std::string out_dir = dir;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--good" && i + 1 < argc) {
      good = argv[++i];
    } else if (a == "--broken" && i + 1 < argc) {
      broken = argv[++i];
    } else if (a == "--unknown" && i + 1 < argc) {
      unknown = argv[++i];
    } else if (a == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else {
      std::fprintf(stderr,
                   "usage: %s [--good PATH] [--broken PATH] [--unknown PATH] [--out-dir DIR]\n",
                   argv[0]);
      return 2;
    }
  }

  std::printf("GGUF model adapter test\n");
  std::printf("  good:    %s\n", good.c_str());
  std::printf("  broken:  %s\n", broken.c_str());
  std::printf("  unknown: %s\n", unknown.c_str());

  for (const std::string* p : {&good, &broken, &unknown}) {
    if (!file_exists(*p)) {
      std::printf("FAIL: fixture missing: %s\n", p->c_str());
      std::printf("\nFAIL 1 failure(s), 1 check(s)\n");
      return 1;
    }
  }

  std::printf("\nName mapping:\n");
  check_mapping();
  std::printf("  %s\n", g_failures == 0 ? "ok" : "see failures above");

  check_good_model(good);
  check_broken_model(broken);
  check_unknown_tensor(unknown);
  check_fallbacks(out_dir);

  std::printf("\n");
  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  for (const std::string& f : g_failure_log) std::printf("  - %s\n", f.c_str());
  return 1;
}
