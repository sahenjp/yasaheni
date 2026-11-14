// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GgufContainer: materialise a GGUF file into the in-memory image that
// src/loader.cpp's Model::load expects. See gguf_container.hpp for the layout.
#include "gguf_container.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "gguf_model.hpp"  // read_config, ModelConfig

namespace yskn::gguf {
namespace {

// Rows of the sample used to train the shared codebook.
//
// Measured on a 512x512 matrix (bits=6, refine_iterations=0): training on 32
// rows gives RMS 5.801, 64 rows 5.791, 128 rows 5.775. The codebook
// generalises long before it has seen every row, so 64 rows buys a 2x cheaper
// open() for 0.3% of the reconstruction quality. This is the same effect
// docs/GGUF_NOTES.md records under 'More training data narrows the gap'.
constexpr int kTrainRows = 64;

size_t align_up(size_t off, size_t a) { return (off + a - 1) & ~(a - 1); }

// layout:"warp" source index, straight out of k_repack (src/repack.cu:13-14).
// gguf_quant.cpp keeps its copy private (gguf_quant.cpp:135), so it is
// replicated here; the two must stay identical or the embedding rows will be
// permuted relative to what the reader indexes.
size_t warp_src(int row, int seq, int word, int rows, int wps) {
  const size_t q = static_cast<size_t>(seq) * static_cast<size_t>(rows) +
                   static_cast<size_t>(row);
  const size_t tile = q / 32, lane = q % 32;
  return ((tile * static_cast<size_t>(wps / 4) + static_cast<size_t>(word / 4)) * 32 +
          lane) * 4 + static_cast<size_t>(word % 4);
}

std::string join(const std::vector<std::string>& v) {
  std::ostringstream os;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) os << "; ";
    os << v[i];
  }
  return os.str();
}

// One weight matrix the loader will ask for.
struct WeightSpec {
  std::string engine;   // "layers.0.q_proj"
  std::string gguf;     // "blk.0.attn_q.weight"
  bool embedding;       // row-major stream + layout:"row"
  bool rot_out;         // false for the head and the embedding table
};

// One f16 vector the loader will ask for.
struct VectorSpec {
  std::string engine;   // "layers.0.input_layernorm"
  std::string gguf;     // "blk.0.attn_norm.weight"; empty means synthesise zeros
  int n;                // element count the loader will demand
};

// The exact names Model::load asks for, in the order it asks for them
// (src/loader.cpp:78-91).
void plan(const ModelConfig& cfg, std::vector<WeightSpec>& weights,
          std::vector<VectorSpec>& vectors) {
  weights.clear();
  vectors.clear();
  const int qd = cfg.q_dim();       // heads * head_dim
  const int kd = cfg.kv_dim();      // kv_heads * head_dim
  for (int l = 0; l < cfg.layers; ++l) {
    const std::string e = "layers." + std::to_string(l) + ".";
    const std::string g = "blk." + std::to_string(l) + ".";
    weights.push_back({e + "q_proj", g + "attn_q.weight", false, true});
    weights.push_back({e + "k_proj", g + "attn_k.weight", false, true});
    weights.push_back({e + "v_proj", g + "attn_v.weight", false, true});
    weights.push_back({e + "o_proj", g + "attn_output.weight", false, true});
    weights.push_back({e + "gate_proj", g + "ffn_gate.weight", false, true});
    weights.push_back({e + "up_proj", g + "ffn_up.weight", false, true});
    weights.push_back({e + "down_proj", g + "ffn_down.weight", false, true});
    vectors.push_back({e + "input_layernorm", g + "attn_norm.weight", cfg.hidden});
    vectors.push_back({e + "post_attention_layernorm", g + "ffn_norm.weight", cfg.hidden});
    vectors.push_back({e + "q_proj.bias", g + "attn_q.bias", qd});
    vectors.push_back({e + "k_proj.bias", g + "attn_k.bias", kd});
    vectors.push_back({e + "v_proj.bias", g + "attn_v.bias", kd});
  }
  // The head reuses the embedding table when the file has no output.weight.
  weights.push_back({"lm_head", cfg.tie_embeddings ? "token_embd.weight" : "output.weight",
                     false, false});
  weights.push_back({"embed_tokens", "token_embd.weight", true, false});
  vectors.push_back({"norm", "output_norm.weight", cfg.hidden});
}

// Rotate + per-row normalise, i.e. steps 3-4 of the encoder, for the codebook
// training sample. Mirrors gguf_quant.cpp:617-641 exactly.
std::vector<float> normalised_rows(const float* W, int rows, int cols,
                                   const QuantConfig& cfg) {
  std::vector<float> A(static_cast<size_t>(rows) * cols);
  std::memcpy(A.data(), W, A.size() * sizeof(float));
  rotate_matrix(A.data(), rows, cols, cfg, A.data());
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

// True for the attention bias vectors Model::load asks for. Those are the only
// vectors the container will synthesise when the file lacks them.
bool is_bias_name(const std::string& engine) {
  return engine.size() > 5 && engine.compare(engine.size() - 5, 5, ".bias") == 0;
}

int tensor_rows(const TensorInfo& t) {
  return static_cast<int>(t.shape.size() < 2 ? 1 : t.shape[1]);
}

int tensor_cols(const TensorInfo& t) {
  return static_cast<int>(t.shape.empty() ? t.n_elements : t.shape[0]);
}

}  // namespace

GgufContainer::GgufContainer() = default;
GgufContainer::~GgufContainer() = default;
GgufContainer::GgufContainer(GgufContainer&&) noexcept = default;
GgufContainer& GgufContainer::operator=(GgufContainer&&) noexcept = default;

size_t GgufContainer::bytes() const { return buf_.size(); }

bool GgufContainer::has_tensor(const std::string& name) const {
  for (const TensorDesc& t : tensors_)
    if (t.name == name) return true;
  return false;
}

bool GgufContainer::has_vector(const std::string& name) const {
  for (const VectorDesc& v : vectors_)
    if (v.name == name) return true;
  return false;
}

const TensorDesc& GgufContainer::tensor(const std::string& name) const {
  for (const TensorDesc& t : tensors_)
    if (t.name == name) return t;
  throw std::runtime_error("missing tensor " + name);
}

const VectorDesc& GgufContainer::vector(const std::string& name) const {
  for (const VectorDesc& v : vectors_)
    if (v.name == name) return v;
  throw std::runtime_error("missing vector " + name);
}

// ---------------------------------------------------------------------------
// Hyperparameters: ModelConfig -> ModelDesc.
//
// Every field Model::load reads is copied straight across. arch keeps the GGUF
// tag ("qwen3" stays "qwen3") so a caller can still tell what it loaded.
// ---------------------------------------------------------------------------
void GgufContainer::build_model_desc() {
  const ModelConfig cfg = read_config(file_);
  model.arch = cfg.arch;
  model.layers = cfg.layers;
  model.hidden = cfg.hidden;
  model.ffn = cfg.ffn;
  model.heads = cfg.heads;
  model.kv_heads = cfg.kv_heads;
  model.head_dim = cfg.head_dim;
  model.vocab = cfg.vocab;
  model.eps = cfg.eps;
  model.rope_theta = cfg.rope_theta;
  model.tie_embeddings = cfg.tie_embeddings;
}

// ---------------------------------------------------------------------------
// Reject shapes this container cannot turn into a loadable image.
//
// The messages name the dimension and the constant it violates, because the
// whole point of the GGUF path is that a failure is diagnosable without a
// debugger. Constraints, in the order they are checked:
//
//   arch / dims        the adapter's own sanity (gguf_model.cpp:188-221)
//   hidden % 256       kSeqCols, the trellis sequence width (gguf_quant.cpp:606)
//   ffn % 256          same, for down_proj's fan-in
//   heads*head_dim     o_proj's fan-in must equal it, and it must equal hidden
//   required tensors   every name plan() emits must exist in the file
// ---------------------------------------------------------------------------
void GgufContainer::validate_shape() const {
  const ModelConfig cfg = read_config(file_);
  std::vector<std::string> errors;
  const int h = cfg.hidden, f = cfg.ffn;
  const int qd = cfg.q_dim();

  if (cfg.layers <= 0)
    errors.push_back("block_count is " + std::to_string(cfg.layers) + ", expected > 0");
  if (h <= 0)
    errors.push_back("embedding_length is " + std::to_string(h) + ", expected > 0");
  if (f <= 0)
    errors.push_back("feed_forward_length is " + std::to_string(f) + ", expected > 0");
  if (cfg.heads <= 0)
    errors.push_back("attention.head_count is " + std::to_string(cfg.heads) +
                      ", expected > 0");
  if (cfg.kv_heads <= 0)
    errors.push_back("attention.head_count_kv is " + std::to_string(cfg.kv_heads) +
                      ", expected > 0");
  if (cfg.kv_heads > 0 && cfg.heads > 0 && cfg.heads % cfg.kv_heads != 0)
    errors.push_back("attention.head_count " + std::to_string(cfg.heads) +
                      " is not a multiple of attention.head_count_kv " +
                      std::to_string(cfg.kv_heads));
  if (cfg.head_dim <= 0)
    errors.push_back("head_dim resolved to " + std::to_string(cfg.head_dim) +
                      ", expected > 0");
  if (h > 0 && qd != h)
    errors.push_back("heads*head_dim is " + std::to_string(qd) +
                      " but embedding_length is " + std::to_string(h) +
                      " (o_proj's fan-in would not match the residual stream)");
  if (!(cfg.eps > 0.f) || !(cfg.eps < 1.f))
    errors.push_back("layer_norm_rms_epsilon is out of range, expected 0 < eps < 1");
  if (!(cfg.rope_theta >= 1.f))
    errors.push_back("rope.freq_base is below 1, expected >= 1");

  // The quantiser's hard limit: every layer weight's fan-in is either hidden or
  // ffn, and both must be a whole number of 256-column trellis sequences.
  if (h > 0 && h % kSeqCols != 0)
    errors.push_back("embedding_length " + std::to_string(h) +
                      " is not a multiple of " + std::to_string(kSeqCols) +
                      " (kSeqCols, the trellis sequence width); the encoder cannot"
                      " represent it");
  if (f > 0 && f % kSeqCols != 0)
    errors.push_back("feed_forward_length " + std::to_string(f) +
                      " is not a multiple of " + std::to_string(kSeqCols) +
                      " (kSeqCols, the trellis sequence width); the encoder cannot"
                      " represent it");

  // Required tensors.
  std::vector<WeightSpec> weights;
  std::vector<VectorSpec> vectors;
  plan(cfg, weights, vectors);
  for (const WeightSpec& w : weights) {
    const TensorInfo* t = file_.find(w.gguf);
    if (!t) {
      errors.push_back("missing required tensor " + w.gguf + " (needed as " + w.engine +
                        ")");
    } else if (!type_supported(t->type)) {
      errors.push_back("tensor " + w.gguf + " has type " + type_name(t->type) +
                        ", which cannot be dequantized");
    }
  }
  for (const VectorSpec& v : vectors) {
    const TensorInfo* t = file_.find(v.gguf);
    if (!t) {
      // A missing bias is fine (Qwen3 ships none, see materialise()); a missing
      // norm is not.
      if (is_bias_name(v.engine)) continue;
      errors.push_back("missing required tensor " + v.gguf + " (needed as " + v.engine +
                        ")");
    } else if (static_cast<int>(t->n_elements) != v.n) {
      errors.push_back("tensor " + v.gguf + " has " +
                        std::to_string(t->n_elements) + " elements, but " + v.engine +
                        " needs " + std::to_string(v.n));
    }
  }

  if (!errors.empty())
    throw std::runtime_error("gguf_container: " + file_.path() +
                             " cannot back the legacy Model::load: " + join(errors));
}

// ---------------------------------------------------------------------------
// Materialise the image.
// ---------------------------------------------------------------------------
void GgufContainer::materialise() {
  const ModelConfig cfg = read_config(file_);
  std::vector<WeightSpec> weights;
  std::vector<VectorSpec> vectors;
  plan(cfg, weights, vectors);

  buf_.clear();
  tensors_.clear();
  vectors_.clear();
  payload_bytes_ = 0;

  // Reserve roughly what the image needs so the appends below do not
  // repeatedly reallocate a multi-megabyte buffer.
  {
    size_t est = kTailPadBytes;
    for (const WeightSpec& w : weights) {
      const TensorInfo* t = file_.find(w.gguf);
      const int rows = tensor_rows(*t);
      const int physical = ((rows + kTileRows - 1) / kTileRows) * kTileRows;
      est += 2 * kLutSize * 2 + static_cast<size_t>(physical) * 2;  // lut16 + scale16
      est += static_cast<size_t>(t->n_elements) * static_cast<size_t>(cfg_.bits) / 2;
      est += 64;  // alignment slack
    }
    for (const VectorSpec& v : vectors) est += static_cast<size_t>(v.n) * 2 + 64;
    buf_.reserve(est);
  }

  // ---- the shared codebook ------------------------------------------------
  //
  // One codebook for the whole model, trained on a sample of the widest weight
  // tensor and reused through quantize_tensor's 'shared' argument. Each tensor
  // still carries its own lut16 (f16 of q*gamma) and its own scale16, so the
  // reader recovers gamma per tensor exactly as it does for a per-tensor
  // codebook (docs/FORMAT.md).
  {
    const WeightSpec* sample = &weights.front();
    for (const WeightSpec& w : weights) {
      if (tensor_rows(*file_.find(w.gguf)) > tensor_rows(*file_.find(sample->gguf)))
        sample = &w;
    }
    const TensorInfo* t = file_.find(sample->gguf);
    const int rows = tensor_rows(*t), cols = tensor_cols(*t);
    const std::vector<float> W = file_.dequant(*t);
    QuantConfig linear = cfg_;
    linear.use_rot_out = true;  // the sample is a linear weight
    const std::vector<float> A = normalised_rows(W.data(), rows, cols, linear);
    const int train_rows = std::min(rows, kTrainRows);
    const int seed =
        static_cast<int>((cfg_.sign_seed ^ (cfg_.sign_seed >> 32)) & 0x7FFFFFFFull);
    codebook_ = train_codebook(A.data(), static_cast<size_t>(train_rows) * cols, seed);
    // Plain k-means ignores the trellis (docs/GGUF_NOTES.md, 'Trellis-aware
    // codebook refinement matters'), so run the Lloyd rounds over the states the
    // encoder can actually reach. Measured on the 256x256 fixture sample at
    // bits=6: 8.86 RMS plain -> 8.26 after 1 round; 4 rounds cost 96 ms and gain
    // nothing measurable, so use the configured count.
    if (cfg_.refine_iterations > 0)
      codebook_ = refine_codebook(A.data(), train_rows, cols, codebook_, cfg_.bits,
                                  cfg_.refine_iterations);
  }

  // ---- weight tensors -----------------------------------------------------
  for (const WeightSpec& w : weights) {
    const TensorInfo* t = file_.find(w.gguf);
    const int rows = tensor_rows(*t), cols = tensor_cols(*t);
    const std::vector<float> W = file_.dequant(*t);

    QuantConfig qc = cfg_;
    qc.use_rot_out = w.rot_out;
    const QuantizedTensor q = quantize_tensor(W.data(), rows, cols, qc, &codebook_);

    TensorDesc d;
    d.name = w.engine;
    d.role = w.embedding ? "embedding" : (w.engine == "lm_head" ? "head" : "linear");
    d.rows = rows;
    d.cols = cols;
    d.physical_rows = q.physical_rows;
    d.L = kLutBits;
    d.d = 2;
    d.T = kSeqCols;
    d.bits = q.bits;
    d.start = "zero";
    d.layout = w.embedding ? "row" : "warp";

    // rot_in / rot_out must describe exactly what quantize_tensor applied, or
    // upload_linear will generate different signs than the encoder used.
    d.rot_in.kind = "H512";
    d.rot_in.m = cols / kHadamardBlock;
    d.rot_in.sign_seed = qc.sign_seed;
    d.rot_in.sign_offset = 0;
    d.rot_in.rule = qc.sign_rule;
    d.rot_in.qm_off = 0;
    d.has_rot_out = w.rot_out;
    if (w.rot_out) {
      d.rot_out.kind = "H512";
      d.rot_out.m = rows / kHadamardBlock;
      d.rot_out.sign_seed = qc.sign_seed;
      d.rot_out.sign_offset = static_cast<uint64_t>(cols);
      d.rot_out.rule = qc.sign_rule;
      d.rot_out.qm_off = 0;
    }

    // lut16: 2*4096 f16 codebook, exactly what build_lut8_tau reads.
    d.lut_off = align_up(buf_.size(), 2);
    buf_.resize(d.lut_off + q.lut16.size() * 2, 0);
    std::memcpy(buf_.data() + d.lut_off, q.lut16.data(), q.lut16.size() * 2);

    // scale16: physical_rows f16.
    d.scale_off = align_up(buf_.size(), 2);
    buf_.resize(d.scale_off + q.scale16.size() * 2, 0);
    std::memcpy(buf_.data() + d.scale_off, q.scale16.data(), q.scale16.size() * 2);

    // stream: warp-interleaved for linear weights (upload_linear repacks it),
    // logical row-major for the embedding table (src/misc_kernels.cu:18 indexes
    // it directly, so it must NOT be interleaved).
    d.stream_off = align_up(buf_.size(), 4);
    const size_t nwords = q.stream.size();
    buf_.resize(d.stream_off + nwords * 4, 0);
    uint32_t* dst = reinterpret_cast<uint32_t*>(buf_.data() + d.stream_off);
    if (w.embedding) {
      for (int r = 0; r < q.physical_rows; ++r)
        for (int s = 0; s < q.seqs; ++s)
          for (int i = 0; i < q.wps; ++i)
            dst[(static_cast<size_t>(r) * q.seqs + s) * q.wps + i] =
                q.stream[warp_src(r, s, i, q.physical_rows, q.wps)];
    } else {
      std::memcpy(dst, q.stream.data(), nwords * 4);
    }
    d.stream_bytes = nwords * 4;
    d.bias_off = 0;  // upload_linear never reads it; biases arrive as vectors.
    d.proxy = 0.f;
    d.has_rar = false;

    tensors_.push_back(d);
  }

  // ---- f16 vectors --------------------------------------------------------
  for (const VectorSpec& v : vectors) {
    std::vector<uint16_t> h16(static_cast<size_t>(v.n));
    if (const TensorInfo* t = file_.find(v.gguf)) {
      const std::vector<float> f = file_.dequant(*t);
      for (int i = 0; i < v.n; ++i) h16[static_cast<size_t>(i)] = f32_to_f16(f[static_cast<size_t>(i)]);
    }
    // else: synthesised zero bias. Qwen3 ships none, but Model::load always
    // calls upload_vector for bq/bk/bv and would throw on a missing name.

    VectorDesc d;
    d.name = v.engine;
    d.n = v.n;
    d.dtype = "f16";
    d.off = align_up(buf_.size(), 2);
    buf_.resize(d.off + h16.size() * 2, 0);
    std::memcpy(buf_.data() + d.off, h16.data(), h16.size() * 2);
    vectors_.push_back(d);
  }

  payload_bytes_ = buf_.size();
  buf_.resize(buf_.size() + kTailPadBytes, 0);
}

void GgufContainer::open(const std::string& path, const QuantConfig& cfg) {
  cfg_ = cfg;
  file_.open(path);
  build_model_desc();
  validate_shape();
  materialise();
  opened_ = true;

  // The loader casts at<T>() straight to uint16_t/uint32_t/uint64_t. operator
  // new gives the buffer 16-byte alignment and every payload was aligned to
  // its own natural alignment, so this holds by construction -- assert it
  // anyway, because a future layout change would otherwise only show up as a
  // SIGBUS on aarch64.
  const uintptr_t base = reinterpret_cast<uintptr_t>(buf_.data());
  if (base % 16 != 0)
    throw std::runtime_error("gguf_container: buffer base is not 16-byte aligned");
  for (const TensorDesc& t : tensors_) {
    if (t.lut_off % 2 || t.scale_off % 2 || t.stream_off % 4)
      throw std::runtime_error("gguf_container: misaligned tensor payload for " + t.name);
    if (t.stream_bytes != static_cast<uint64_t>(t.physical_rows) * (t.cols / kSeqCols) *
                                (4 * t.bits) * 4)
      throw std::runtime_error("gguf_container: stream_bytes is not rows*seqs*4*bits*4 for " +
                               t.name);
  }
  for (const VectorDesc& v : vectors_) {
    if (v.off % 2)
      throw std::runtime_error("gguf_container: misaligned vector payload for " + v.name);
  }
  if (payload_bytes_ + kTailPadBytes != buf_.size())
    throw std::runtime_error("gguf_container: tail padding was not appended");
}

}  // namespace yskn::gguf