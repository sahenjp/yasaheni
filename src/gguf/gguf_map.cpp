// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF tensor-name -> engine role mapping, plus the coverage report.
//
// GGUF uses llama.cpp's flat naming ("blk.3.attn_q.weight"); the engine speaks
// (layer, role) with the role tags from gguf_model.hpp. Everything here is
// pure string work: no file access beyond the tensor table.
#include "gguf_model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>

namespace yskn::gguf {
namespace {

// One row of the layer-tensor table: GGUF suffix -> role + engine suffix.
struct LayerEntry {
  const char* gguf;    // "attn_q"
  const char* role;    // roles::kQ
  const char* engine;  // "self_attn.q_proj"
  bool norm;           // rank-1 norm gain
};

const LayerEntry kLayerTable[] = {
    // Attention.
    {"attn_norm", roles::kLn1, "input_layernorm", true},
    {"ffn_norm", roles::kLn2, "post_attention_layernorm", true},
    {"attn_q", roles::kQ, "self_attn.q_proj", false},
    {"attn_k", roles::kK, "self_attn.k_proj", false},
    {"attn_v", roles::kV, "self_attn.v_proj", false},
    {"attn_output", roles::kO, "self_attn.o_proj", false},
    {"attn_q_norm", roles::kQNorm, "self_attn.q_norm", true},
    {"attn_k_norm", roles::kKNorm, "self_attn.k_norm", true},
    {"attn_gate", roles::kAttnGate, "self_attn.attn_gate", false},
    // Dense MLP.
    {"ffn_gate", roles::kGate, "mlp.gate_proj", false},
    {"ffn_up", roles::kUp, "mlp.up_proj", false},
    {"ffn_down", roles::kDown, "mlp.down_proj", false},
    // Qwen3.5-style linear-attention (SSM) block.
    {"ssm_a", roles::kSsmA, "linear_attn.A_log", false},
    {"ssm_alpha", roles::kSsmAlpha, "linear_attn.in_proj_a", false},
    {"ssm_beta", roles::kSsmBeta, "linear_attn.in_proj_b", false},
    {"ssm_conv1d", roles::kSsmConv, "linear_attn.conv1d", false},
    {"ssm_dt", roles::kSsmDt, "linear_attn.dt_bias", false},
    {"ssm_norm", roles::kSsmNorm, "linear_attn.norm", true},
    {"ssm_out", roles::kSsmOut, "linear_attn.out_proj", false},
    // Mixture of experts (mapped for coverage; the dense path warns).
    {"ffn_gate_inp", roles::kFfnGateInp, "ffn_gate_inp", false},
    {"ffn_gate_exps", roles::kFfnGateExps, "ffn_gate_exps", false},
    {"ffn_up_exps", roles::kFfnUpExps, "ffn_up_exps", false},
    {"ffn_down_exps", roles::kFfnDownExps, "ffn_down_exps", false},
};

// Global (non-layer) tensors: GGUF name -> role + engine name.
struct GlobalEntry {
  const char* gguf;
  const char* role;
  const char* engine;
  bool norm;
};

const GlobalEntry kGlobalTable[] = {
    {"token_embd.weight", roles::kEmbedding, "embed_tokens", false},
    {"token_embd.bias", roles::kEmbedding, "embed_tokens.bias", false},
    {"output_norm.weight", roles::kFinalNorm, "norm", true},
    {"output_norm.bias", roles::kFinalNorm, "norm.bias", false},
    {"output.weight", roles::kHead, "lm_head", false},
    {"output.bias", roles::kHead, "lm_head.bias", false},
};

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parse "blk.<N>.<rest>" into N and rest. Returns false when the prefix is
// absent or the index is not a plain decimal number.
bool split_block(const std::string& name, int& layer, std::string& rest) {
  static const std::string kPrefix = "blk.";
  if (name.compare(0, kPrefix.size(), kPrefix) != 0) return false;
  const size_t dot = name.find('.', kPrefix.size());
  if (dot == std::string::npos || dot == kPrefix.size()) return false;
  const std::string digits = name.substr(kPrefix.size(), dot - kPrefix.size());
  for (char c : digits)
    if (c < '0' || c > '9') return false;
  // Guard against absurd indices (and against strtol overflow).
  if (digits.size() > 9) return false;
  layer = std::atoi(digits.c_str());
  rest = name.substr(dot + 1);
  return true;
}

}  // namespace

bool map_tensor(const std::string& gguf_name, MappedTensor& out) {
  out = MappedTensor{};

  // Global tensors first: they carry no layer index.
  for (const GlobalEntry& g : kGlobalTable) {
    if (gguf_name == g.gguf) {
      out.layer = -1;
      out.role = g.role;
      out.is_norm = g.norm;
      out.is_bias = ends_with(gguf_name, ".bias");
      out.engine_name = g.engine;
      return true;
    }
  }

  int layer = 0;
  std::string rest;
  if (!split_block(gguf_name, layer, rest)) return false;

  // Strip the trailing ".weight" / ".bias" so the table keys stay short.
  std::string base = rest;
  bool bias = false;
  if (ends_with(base, ".bias")) {
    base.resize(base.size() - 5);
    bias = true;
  } else if (ends_with(base, ".weight")) {
    base.resize(base.size() - 7);
  } else {
    return false;  // e.g. "blk.0.some_other_quant_metadata"
  }

  for (const LayerEntry& e : kLayerTable) {
    if (base == e.gguf) {
      out.layer = layer;
      out.role = e.role;
      out.is_norm = e.norm && !bias;
      out.is_bias = bias;
      out.engine_name = "layers." + std::to_string(layer) + "." + e.engine;
      if (bias) out.engine_name += ".bias";
      return true;
    }
  }
  return false;
}

bool map_tensor(const TensorInfo& t, MappedTensor& out) {
  if (!map_tensor(t.name, out)) return false;
  out.shape = t.shape;
  return true;
}

std::vector<std::string> expected_tensors(const ModelConfig& cfg) {
  std::vector<std::string> names;
  names.push_back("token_embd.weight");
  names.push_back("output_norm.weight");
  if (!cfg.tie_embeddings) names.push_back("output.weight");

  for (int l = 0; l < cfg.layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    names.push_back(p + "attn_norm.weight");
    names.push_back(p + "ffn_norm.weight");
    names.push_back(p + "attn_q.weight");
    names.push_back(p + "attn_k.weight");
    names.push_back(p + "attn_v.weight");
    names.push_back(p + "attn_output.weight");
    if (cfg.has_qk_norm) {
      names.push_back(p + "attn_q_norm.weight");
      names.push_back(p + "attn_k_norm.weight");
    }
    if (cfg.is_moe()) {
      names.push_back(p + "ffn_gate_inp.weight");
      names.push_back(p + "ffn_gate_exps.weight");
      names.push_back(p + "ffn_up_exps.weight");
      names.push_back(p + "ffn_down_exps.weight");
    } else {
      names.push_back(p + "ffn_gate.weight");
      names.push_back(p + "ffn_up.weight");
      names.push_back(p + "ffn_down.weight");
    }
  }
  return names;
}

CoverageReport coverage(const File& f, const ModelConfig& cfg) {
  CoverageReport rep;
  std::set<std::string> present;

  for (const TensorInfo& t : f.tensors()) {
    present.insert(t.name);
    MappedTensor m;
    if (map_tensor(t.name, m))
      rep.matched.push_back(t.name);
    else
      rep.unmatched.push_back(t.name);
  }

  for (const std::string& want : expected_tensors(cfg))
    if (present.find(want) == present.end()) rep.missing_expected.push_back(want);

  return rep;
}

CoverageReport coverage(const File& f) { return coverage(f, read_config(f)); }

std::string CoverageReport::to_string() const {
  std::ostringstream os;
  os << "coverage:\n";
  os << "  matched:   " << matched.size() << "\n";
  os << "  unmatched: " << unmatched.size() << "\n";
  os << "  missing:   " << missing_expected.size() << "\n";
  for (const std::string& n : unmatched) os << "  unmatched tensor: " << n << "\n";
  for (const std::string& n : missing_expected) os << "  missing tensor:   " << n << "\n";
  return os.str();
}

}  // namespace yskn::gguf
