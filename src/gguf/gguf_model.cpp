// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF -> engine adapter: hyperparameter extraction, validation, the info
// dump and the cached float32 accessor. The name mapper lives in gguf_map.cpp.
//
// Everything here is STL-only and CUDA-free so it can be unit tested next to
// the reader. The parser is reused as-is: this file never touches raw bytes.
#include "gguf_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

namespace yskn::gguf {
namespace {

// ---------------------------------------------------------------------------
// Metadata helpers. Missing keys and mistyped values both fall back silently;
// validate() reports which fallbacks fired.
// ---------------------------------------------------------------------------
bool meta_number(const Metadata& md, const std::string& key, double& out) {
  const Value* v = md.find(key);
  if (!v || !v->is_number()) return false;
  out = v->as_float();
  return true;
}

bool meta_u64(const Metadata& md, const std::string& key, uint64_t& out) {
  double d = 0.0;
  if (!meta_number(md, key, d) || d < 0.0) return false;
  out = static_cast<uint64_t>(d + 0.5);
  return true;
}

bool meta_i32(const Metadata& md, const std::string& key, int& out) {
  uint64_t u = 0;
  if (!meta_u64(md, key, u)) return false;
  out = static_cast<int>(u);
  return true;
}

bool meta_f32(const Metadata& md, const std::string& key, float& out) {
  double d = 0.0;
  if (!meta_number(md, key, d)) return false;
  out = static_cast<float>(d);
  return true;
}

bool meta_bool(const Metadata& md, const std::string& key, bool& out) {
  const Value* v = md.find(key);
  if (!v) return false;
  if (v->type == ValueType::BOOL) {
    out = v->u != 0;
    return true;
  }
  if (v->is_number()) {
    out = v->as_float() != 0.0;
    return true;
  }
  return false;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string fmt_g(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return std::string(buf);
}

// Row count of a matrix in GGUF order: ne[0] is the fastest-varying (fan-in)
// dimension, ne[1] the fan-out. A rank-1 tensor has one row.
uint64_t tensor_rows(const TensorInfo& t) {
  return t.shape.size() < 2 ? 1 : t.shape[1];
}

uint64_t tensor_cols(const TensorInfo& t) {
  return t.shape.empty() ? t.n_elements : t.shape[0];
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------
ModelConfig read_config(const File& f) {
  ModelConfig cfg;
  const Metadata& md = f.meta();

  cfg.arch = md.get_string("general.architecture", std::string());
  const std::string p = cfg.arch.empty() ? std::string() : cfg.arch + ".";

  if (!meta_i32(md, p + "block_count", cfg.layers)) cfg.layers_from_fallback = true;
  if (!meta_i32(md, p + "embedding_length", cfg.hidden)) cfg.hidden_from_fallback = true;
  if (!meta_i32(md, p + "feed_forward_length", cfg.ffn)) cfg.ffn_from_fallback = true;
  meta_i32(md, p + "attention.head_count", cfg.heads);
  meta_i32(md, p + "attention.head_count_kv", cfg.kv_heads);

  // head_dim: explicit key_length, else embedding_length / head_count.
  if (!meta_i32(md, p + "attention.key_length", cfg.head_dim)) {
    cfg.head_dim_from_fallback = true;
    cfg.head_dim = (cfg.heads > 0 && cfg.hidden > 0) ? cfg.hidden / cfg.heads : 0;
  }

  // vocab: explicit key, else the embedding table's row count.
  if (!meta_i32(md, p + "vocab_size", cfg.vocab)) {
    cfg.vocab_from_fallback = true;
    if (const TensorInfo* emb = f.find("token_embd.weight"))
      cfg.vocab = static_cast<int>(tensor_rows(*emb));
  }

  if (!meta_f32(md, p + "attention.layer_norm_rms_epsilon", cfg.eps)) {
    cfg.eps_defaulted = true;
    cfg.eps = 1e-6f;
  }
  if (!meta_f32(md, p + "rope.freq_base", cfg.rope_theta)) {
    cfg.rope_theta_defaulted = true;
    cfg.rope_theta = 10000.f;
  }
  if (!meta_i32(md, p + "context_length", cfg.context_length)) cfg.context_length = 0;
  if (!meta_i32(md, p + "attention.sliding_window", cfg.sliding_window)) cfg.sliding_window = 0;
  if (!meta_i32(md, p + "rope.dimension_count", cfg.rope_dim)) cfg.rope_dim = cfg.head_dim;
  if (!meta_i32(md, p + "expert_count", cfg.experts)) cfg.experts = 0;
  if (!meta_i32(md, p + "expert_used_count", cfg.experts_used)) cfg.experts_used = 0;

  // Tied embeddings: no separate output matrix (llama.cpp then reuses
  // token_embd). An explicit tie flag is honoured when present.
  cfg.tie_embeddings = f.find("output.weight") == nullptr;
  bool explicit_tie = false;
  if (meta_bool(md, p + "tie_word_embeddings", explicit_tie) && explicit_tie)
    cfg.tie_embeddings = true;

  // Per-head QK norm is a Qwen3 feature; detect it from the tensor table so
  // validate() can require the tensors when they are supposed to be there.
  for (const TensorInfo& t : f.tensors()) {
    if (ends_with(t.name, "attn_q_norm.weight")) {
      cfg.has_qk_norm = true;
      break;
    }
  }
  cfg.has_output_bias = f.find("output.bias") != nullptr;
  return cfg;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
namespace {

struct Checker {
  ValidationReport rep;

  void error(const std::string& msg) {
    rep.ok = false;
    rep.errors.push_back(msg);
  }
  void warn(const std::string& msg) { rep.warnings.push_back(msg); }

  // Compare one dimension of a tensor. 'what' is "rows" or "cols".
  void dim(const TensorInfo& t, uint64_t got, uint64_t want, const char* what) {
    if (got == want) return;
    error(t.name + " has " + std::to_string(got) + " " + what + ", expected " +
          std::to_string(want));
  }

  void rows(const TensorInfo& t, uint64_t want) { dim(t, tensor_rows(t), want, "rows"); }
  void cols(const TensorInfo& t, uint64_t want) { dim(t, tensor_cols(t), want, "cols"); }
  void count(const TensorInfo& t, uint64_t want) {
    if (t.n_elements == want) return;
    error(t.name + " has " + std::to_string(t.n_elements) + " elements, expected " +
          std::to_string(want));
  }
};

}  // namespace

ValidationReport validate(const File& f, const ModelConfig& cfg) {
  Checker c;

  // ---- config sanity ------------------------------------------------------
  if (cfg.arch.empty())
    c.error("general.architecture is missing");
  else if (cfg.arch != "qwen3" && cfg.arch != "qwen3moe" && cfg.arch != "llama" &&
           cfg.arch != "qwen3_5")
    c.warn("general.architecture is '" + cfg.arch + "', not a known Qwen3/Llama arch");

  if (cfg.layers <= 0) c.error("block_count is " + std::to_string(cfg.layers) + ", expected > 0");
  if (cfg.hidden <= 0) c.error("embedding_length is " + std::to_string(cfg.hidden) + ", expected > 0");
  if (cfg.ffn <= 0 && !cfg.is_moe())
    c.error("feed_forward_length is " + std::to_string(cfg.ffn) + ", expected > 0");
  if (cfg.heads <= 0) c.error("attention.head_count is " + std::to_string(cfg.heads) + ", expected > 0");
  if (cfg.kv_heads <= 0)
    c.error("attention.head_count_kv is " + std::to_string(cfg.kv_heads) + ", expected > 0");
  if (cfg.head_dim <= 0)
    c.error("head_dim resolved to " + std::to_string(cfg.head_dim) + ", expected > 0");
  if (cfg.heads > 0 && cfg.kv_heads > 0 && cfg.heads % cfg.kv_heads != 0)
    c.error("attention.head_count " + std::to_string(cfg.heads) +
            " is not a multiple of attention.head_count_kv " + std::to_string(cfg.kv_heads));
  if (cfg.hidden > 0 && cfg.head_dim > 0 && cfg.heads > 0 &&
      cfg.heads * cfg.head_dim != cfg.hidden)
    c.warn("heads * head_dim = " + std::to_string(cfg.heads * cfg.head_dim) +
           " differs from embedding_length " + std::to_string(cfg.hidden));

  // eps / rope sanity.
  if (!(cfg.eps > 0.f) || !(cfg.eps < 1.f))
    c.error("layer_norm_rms_epsilon is " + fmt_g(cfg.eps) + ", expected 0 < eps < 1");
  if (!(cfg.rope_theta >= 1.f))
    c.error("rope.freq_base is " + fmt_g(cfg.rope_theta) + ", expected >= 1");
  if (cfg.rope_dim <= 0 || (cfg.rope_dim % 2) != 0)
    c.error("rope dimension_count is " + std::to_string(cfg.rope_dim) +
            ", expected a positive even number");
  else if (cfg.head_dim > 0 && cfg.rope_dim > cfg.head_dim)
    c.error("rope dimension_count is " + std::to_string(cfg.rope_dim) +
            ", expected <= head_dim " + std::to_string(cfg.head_dim));

  // ---- required tensors ---------------------------------------------------
  for (const std::string& name : expected_tensors(cfg))
    if (f.find(name) == nullptr) c.error("missing required tensor " + name);

  // ---- global shapes ------------------------------------------------------
  const uint64_t H = static_cast<uint64_t>(cfg.hidden);
  const uint64_t F = static_cast<uint64_t>(cfg.ffn);
  const uint64_t V = static_cast<uint64_t>(cfg.vocab);
  const uint64_t QD = static_cast<uint64_t>(cfg.q_dim());
  const uint64_t KD = static_cast<uint64_t>(cfg.kv_dim());

  if (const TensorInfo* t = f.find("token_embd.weight")) {
    c.cols(*t, H);
    c.rows(*t, V);
    if (V != 0 && tensor_rows(*t) != V)
      c.error("token_embd.weight has " + std::to_string(tensor_rows(*t)) +
              " rows but vocab is " + std::to_string(V));
  }
  if (const TensorInfo* t = f.find("output.weight")) {
    c.cols(*t, H);
    c.rows(*t, V);
  }
  if (const TensorInfo* t = f.find("output_norm.weight")) c.count(*t, H);

  // ---- per-layer shapes ---------------------------------------------------
  for (int l = 0; l < cfg.layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    auto at = [&](const char* suffix) { return f.find(p + suffix); };

    if (const TensorInfo* t = at("attn_norm.weight")) c.count(*t, H);
    if (const TensorInfo* t = at("ffn_norm.weight")) c.count(*t, H);

    if (const TensorInfo* t = at("attn_q.weight")) {
      c.cols(*t, H);
      c.rows(*t, QD);
    }
    if (const TensorInfo* t = at("attn_k.weight")) {
      c.cols(*t, H);
      c.rows(*t, KD);
    }
    if (const TensorInfo* t = at("attn_v.weight")) {
      c.cols(*t, H);
      c.rows(*t, KD);
    }
    if (const TensorInfo* t = at("attn_output.weight")) {
      c.cols(*t, QD);
      c.rows(*t, H);
    }
    if (const TensorInfo* t = at("attn_q.bias")) c.count(*t, QD);
    if (const TensorInfo* t = at("attn_k.bias")) c.count(*t, KD);
    if (const TensorInfo* t = at("attn_v.bias")) c.count(*t, KD);
    if (cfg.has_qk_norm) {
      if (const TensorInfo* t = at("attn_q_norm.weight"))
        c.count(*t, static_cast<uint64_t>(cfg.head_dim));
      if (const TensorInfo* t = at("attn_k_norm.weight"))
        c.count(*t, static_cast<uint64_t>(cfg.head_dim));
    }

    if (const TensorInfo* t = at("ffn_gate.weight")) {
      c.cols(*t, H);
      c.rows(*t, F);
    }
    if (const TensorInfo* t = at("ffn_up.weight")) {
      c.cols(*t, H);
      c.rows(*t, F);
    }
    if (const TensorInfo* t = at("ffn_down.weight")) {
      c.cols(*t, F);
      c.rows(*t, H);
    }
  }

  // ---- dequantizability ---------------------------------------------------
  for (const TensorInfo& t : f.tensors()) {
    MappedTensor m;
    if (!map_tensor(t.name, m)) continue;
    if (!type_supported(t.type))
      c.error(t.name + " has type " + type_name(t.type) + ", which cannot be dequantized");
  }

  // ---- warnings -----------------------------------------------------------
  if (cfg.tie_embeddings) c.warn("output.weight is absent: embeddings are tied to the head");
  if (cfg.layers_from_fallback) c.warn("block_count missing, assuming " + std::to_string(cfg.layers));
  if (cfg.hidden_from_fallback) c.warn("embedding_length missing, assuming " + std::to_string(cfg.hidden));
  if (cfg.ffn_from_fallback) c.warn("feed_forward_length missing, assuming " + std::to_string(cfg.ffn));
  if (cfg.head_dim_from_fallback)
    c.warn("attention.key_length missing, using embedding_length / head_count = " +
           std::to_string(cfg.head_dim));
  if (cfg.vocab_from_fallback)
    c.warn("vocab_size missing, using token_embd.weight rows = " + std::to_string(cfg.vocab));
  if (cfg.eps_defaulted) c.warn("layer_norm_rms_epsilon missing, using 1e-06");
  if (cfg.rope_theta_defaulted) c.warn("rope.freq_base missing, using 10000");
  if (cfg.sliding_window > 0)
    c.warn("attention.sliding_window is " + std::to_string(cfg.sliding_window) +
           "; the adapter ignores it");
  if (cfg.is_moe())
    c.warn("expert_count is " + std::to_string(cfg.experts) +
           "; MoE tensors map but the dense engine path does not consume them");
  if (cfg.context_length > 0 && cfg.context_length < 2048)
    c.warn("context_length is " + std::to_string(cfg.context_length) + ", unusually small");

  const CoverageReport cov = coverage(f, cfg);
  for (const std::string& n : cov.unmatched)
    c.warn("tensor " + n + " is not mapped to any engine role");

  return c.rep;
}

ValidationReport validate(const File& f) { return validate(f, read_config(f)); }

std::string ValidationReport::to_string() const {
  std::ostringstream os;
  os << (ok ? "ok" : "FAILED") << "\n";
  for (const std::string& e : errors) os << "  error:   " << e << "\n";
  for (const std::string& w : warnings) os << "  warning: " << w << "\n";
  return os.str();
}

// ---------------------------------------------------------------------------
// describe(): the body of a future yskn info
// ---------------------------------------------------------------------------
std::string describe(const File& f) {
  const ModelConfig cfg = read_config(f);
  const CoverageReport cov = coverage(f, cfg);
  const ValidationReport val = validate(f, cfg);

  std::map<uint32_t, std::pair<size_t, uint64_t>> by_type;  // type -> (count, bytes)
  uint64_t total_bytes = 0;
  for (const TensorInfo& t : f.tensors()) {
    uint64_t nbytes = 0;
    try {
      nbytes = t.nbytes();
    } catch (...) {
      nbytes = 0;  // a malformed tensor is reported by validate(), not here
    }
    auto& slot = by_type[t.type];
    ++slot.first;
    slot.second += nbytes;
    total_bytes += nbytes;
  }

  std::ostringstream os;
  os << "file: " << f.path() << "\n";
  os << "gguf version: " << f.version() << "\n";
  os << "alignment: " << f.alignment() << "\n";
  os << "arch: " << (cfg.arch.empty() ? "(missing)" : cfg.arch) << "\n";
  os << "layers: " << cfg.layers << "\n";
  os << "hidden: " << cfg.hidden << "\n";
  os << "ffn: " << cfg.ffn << "\n";
  os << "heads: " << cfg.heads << "\n";
  os << "kv heads: " << cfg.kv_heads << "\n";
  os << "head_dim: " << cfg.head_dim << "\n";
  os << "vocab: " << cfg.vocab << "\n";
  os << "eps: " << fmt_g(cfg.eps) << "\n";
  os << "rope_theta: " << fmt_g(cfg.rope_theta) << "\n";
  os << "rope_dim: " << cfg.rope_dim << "\n";
  os << "context_length: " << cfg.context_length << "\n";
  os << "tie_embeddings: " << (cfg.tie_embeddings ? "true" : "false") << "\n";
  if (cfg.is_moe())
    os << "experts: " << cfg.experts << " (used " << cfg.experts_used << ")\n";
  os << "tensors: " << f.tensors().size() << "\n";
  os << "total bytes: " << total_bytes << "\n";
  os << "types:\n";
  for (const auto& kv : by_type)
    os << "  " << type_name(kv.first) << ": " << kv.second.first << " tensors, "
       << kv.second.second << " bytes\n";
  os << cov.to_string();
  os << "validation: " << val.to_string();
  return os.str();
}

// ---------------------------------------------------------------------------
// Cached float32 access
// ---------------------------------------------------------------------------
std::string F32Cache::key_of(const File& f, const std::string& name) {
  // The path alone is not enough: the same path may be reopened after the file
  // changed, so the identity of the mapping is part of the key.
  return f.path() + "|" + std::to_string(f.data_base()) + "|" +
         std::to_string(f.file_size()) + "|" + name;
}

const std::vector<float>& F32Cache::get(File& f, const std::string& name) {
  const std::string k = key_of(f, name);
  auto it = map_.find(k);
  if (it != map_.end()) return it->second;

  const TensorInfo* t = f.find(name);
  if (t == nullptr)
    throw std::runtime_error("gguf_model: no tensor named '" + name + "'");
  if (!type_supported(t->type))
    throw std::runtime_error("gguf_model: tensor '" + name + "' has type " +
                             type_name(t->type) + ", which cannot be dequantized");

  std::vector<float> values = f.dequant(*t);
  return map_.emplace(k, std::move(values)).first->second;
}

bool F32Cache::has(const File& f, const std::string& name) const {
  return map_.find(key_of(f, name)) != map_.end();
}

size_t F32Cache::bytes() const {
  size_t n = 0;
  for (const auto& kv : map_) n += kv.second.size() * sizeof(float);
  return n;
}

namespace {
F32Cache& global_cache() {
  static F32Cache cache;
  return cache;
}
}  // namespace

const std::vector<float>& tensor_f32(File& f, const std::string& name) {
  return global_cache().get(f, name);
}

void clear_f32_cache() { global_cache().clear(); }

}  // namespace yskn::gguf
