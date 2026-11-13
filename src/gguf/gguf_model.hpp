// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF -> engine adapter.
//
// The engine consumes a Container (see src/format.hpp): a ModelDesc of
// hyperparameters plus TensorDesc entries that carry a *role* and a *name*.
// This header presents a plain GGUF file in that shape without going through
// the YSKN container:
//
//   ModelConfig      hyperparameters, read from <arch>.* metadata keys
//   MappedTensor     one GGUF tensor translated to (layer, role, engine name)
//   CoverageReport   which GGUF tensors map, which do not, which are missing
//   ValidationReport shape / sanity checks with specific error messages
//   describe()       the human readable dump a future `yskn info` prints
//   tensor_f32()     dequantized float32 weights, cached
//
// Nothing here touches CUDA or the container reader: it is the bridge layer
// only. Wiring it into Engine35 is a separate step.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.hpp"

namespace yskn::gguf {

// ---------------------------------------------------------------------------
// Role names the engine understands. These mirror the short role tags used by
// the YSKN container ("embedding", "head", "ln1", ...) so a loader written
// against Container keeps working once it is pointed at a GGUF file.
// ---------------------------------------------------------------------------
namespace roles {
inline constexpr const char* kEmbedding = "embedding";
inline constexpr const char* kFinalNorm = "final_norm";
inline constexpr const char* kHead = "head";
inline constexpr const char* kLn1 = "ln1";
inline constexpr const char* kLn2 = "ln2";
inline constexpr const char* kQ = "q";
inline constexpr const char* kK = "k";
inline constexpr const char* kV = "v";
inline constexpr const char* kO = "o";
inline constexpr const char* kGate = "gate";
inline constexpr const char* kUp = "up";
inline constexpr const char* kDown = "down";
inline constexpr const char* kQNorm = "q_norm";
inline constexpr const char* kKNorm = "k_norm";
inline constexpr const char* kAttnGate = "attn_gate";
// Qwen3.5-style hybrid (linear attention) blocks.
inline constexpr const char* kSsmA = "ssm_a";
inline constexpr const char* kSsmAlpha = "ssm_alpha";
inline constexpr const char* kSsmBeta = "ssm_beta";
inline constexpr const char* kSsmConv = "ssm_conv1d";
inline constexpr const char* kSsmDt = "ssm_dt";
inline constexpr const char* kSsmNorm = "ssm_norm";
inline constexpr const char* kSsmOut = "ssm_out";
// Mixture-of-experts blocks: mapped so coverage stays clean, but the dense
// engine path does not consume them (validate() warns).
inline constexpr const char* kFfnGateInp = "ffn_gate_inp";
inline constexpr const char* kFfnGateExps = "ffn_gate_exps";
inline constexpr const char* kFfnUpExps = "ffn_up_exps";
inline constexpr const char* kFfnDownExps = "ffn_down_exps";
}  // namespace roles

// ---------------------------------------------------------------------------
// ModelConfig: the GGUF equivalent of yskn::ModelDesc, plus the extras the
// adapter needs. Every field has a documented fallback; the *_from_fallback /
// *_defaulted flags record which ones were guessed so validate() can warn.
// ---------------------------------------------------------------------------
struct ModelConfig {
  // Mirrors ModelDesc. arch is general.architecture ("qwen3", "qwen3moe",
  // "llama", ...).
  std::string arch;
  int layers = 0;      // <arch>.block_count
  int hidden = 0;      // <arch>.embedding_length
  int ffn = 0;         // <arch>.feed_forward_length
  int heads = 0;       // <arch>.attention.head_count
  int kv_heads = 0;    // <arch>.attention.head_count_kv
  int head_dim = 0;    // <arch>.attention.key_length, else hidden / heads
  int vocab = 0;       // <arch>.vocab_size, else token_embd.weight rows
  float eps = 0.f;         // <arch>.attention.layer_norm_rms_epsilon
  float rope_theta = 0.f;  // <arch>.rope.freq_base
  bool tie_embeddings = false;  // no output.weight, or an explicit tie flag

  // Extras.
  int rope_dim = 0;        // <arch>.rope.dimension_count, else head_dim
  int context_length = 0;  // <arch>.context_length
  int sliding_window = 0;  // <arch>.attention.sliding_window, 0 when absent
  int experts = 0;         // <arch>.expert_count
  int experts_used = 0;    // <arch>.expert_used_count
  bool has_qk_norm = false;     // blk.0.attn_q_norm.weight exists
  bool has_output_bias = false; // output.bias exists

  // Provenance.
  bool layers_from_fallback = false;
  bool hidden_from_fallback = false;
  bool ffn_from_fallback = false;
  bool head_dim_from_fallback = false;
  bool vocab_from_fallback = false;
  bool eps_defaulted = false;
  bool rope_theta_defaulted = false;

  bool is_moe() const { return experts > 0; }
  int q_dim() const { return heads * head_dim; }     // rows of attn_q
  int kv_dim() const { return kv_heads * head_dim; }  // rows of attn_k / attn_v
};

// Read the hyperparameters. Never throws on missing or mistyped metadata: the
// fallbacks fill in, and validate() reports what is wrong.
ModelConfig read_config(const File& f);

// ---------------------------------------------------------------------------
// MappedTensor: one GGUF tensor translated to the engine's vocabulary.
//
//   layer        block index, -1 for a global tensor
//   role         one of roles::k* ("q", "ln1", "embedding", ...)
//   is_bias      1-D additive parameter rather than a weight matrix
//   is_norm      layer-norm / RMS-norm gain
//   shape        GGUF shape, fastest-varying dimension first
//   engine_name  the YSKN-side tensor name ("layers.3.q_proj", "lm_head", ...)
// ---------------------------------------------------------------------------
struct MappedTensor {
  int layer = -1;
  std::string role;
  bool is_bias = false;
  bool is_norm = false;
  std::vector<uint64_t> shape;
  std::string engine_name;

  bool has_layer() const { return layer >= 0; }
  // GGUF stores ne[0] = fastest-varying dimension. For a weight matrix that is
  // the input width, so cols() is the fan-in and rows() the fan-out.
  uint64_t cols() const { return shape.empty() ? 0 : shape[0]; }
  uint64_t rows() const { return shape.size() < 2 ? cols() : shape[1]; }
};

// Map a GGUF tensor name. Returns false when the name is not supported (the
// caller reports it as unmatched). The shape is left empty by this overload.
bool map_tensor(const std::string& gguf_name, MappedTensor& out);
// Same, but also copies the shape out of the tensor info.
bool map_tensor(const TensorInfo& t, MappedTensor& out);

// The tensors a complete model of this shape must contain, in GGUF naming.
std::vector<std::string> expected_tensors(const ModelConfig& cfg);

// ---------------------------------------------------------------------------
// Coverage: what the mapper does and does not understand.
// ---------------------------------------------------------------------------
struct CoverageReport {
  std::vector<std::string> matched;           // mapped to a known role
  std::vector<std::string> unmatched;         // present but not understood
  std::vector<std::string> missing_expected;  // required but absent

  bool complete() const { return unmatched.empty() && missing_expected.empty(); }
  std::string to_string() const;
};

CoverageReport coverage(const File& f, const ModelConfig& cfg);
CoverageReport coverage(const File& f);  // reads the config itself

// ---------------------------------------------------------------------------
// Validation.
// ---------------------------------------------------------------------------
struct ValidationReport {
  bool ok = true;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;

  std::string to_string() const;
};

ValidationReport validate(const File& f, const ModelConfig& cfg);
ValidationReport validate(const File& f);  // reads the config itself

// ---------------------------------------------------------------------------
// Human readable dump: arch, shapes, tensor counts by type, total bytes and
// the coverage report. This is the body of a future `yskn info`.
// ---------------------------------------------------------------------------
std::string describe(const File& f);

// ---------------------------------------------------------------------------
// float32 accessor with an internal cache.
//
// The cache key includes the file path, data base and file size, so two
// different files (or a re-opened, changed file) never share an entry.
// ---------------------------------------------------------------------------
class F32Cache {
 public:
  // Dequantize 'name' once, then hand out the same buffer. Throws when the
  // tensor is absent or its type cannot be dequantized.
  const std::vector<float>& get(File& f, const std::string& name);
  bool has(const File& f, const std::string& name) const;
  void clear() { map_.clear(); }
  size_t size() const { return map_.size(); }
  size_t bytes() const;

 private:
  static std::string key_of(const File& f, const std::string& name);
  std::unordered_map<std::string, std::vector<float>> map_;
};

// Process-wide cache used by the free function below.
const std::vector<float>& tensor_f32(File& f, const std::string& name);
void clear_f32_cache();

}  // namespace yskn::gguf
