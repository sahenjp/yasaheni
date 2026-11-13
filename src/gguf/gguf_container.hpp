// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF-backed equivalent of yskn::Container (src/format.hpp:23).
//
// The legacy engine path (src/loader.cpp: Model::load) reads a Container as
// mmap'd, already-quantised bytes:
//
//   c.tensor("layers.0.q_proj")  -> TensorDesc with lut_off / scale_off /
//                                   stream_off / stream_bytes / bias_off
//   c.vector("layers.0.input_layernorm") -> VectorDesc with off, n, dtype f16
//   c.at<uint16_t>(d.lut_off)    -> raw pointer into the mapping
//
// A GGUF file has none of that: it stores BF16/Q8_0/... weights and no trellis
// stream. GgufContainer bridges the gap by running the encoder from
// src/gguf/gguf_quant.cpp over every weight the loader asks for and
// materialising the result into ONE owned std::vector<uint8_t> buffer that
// plays the role of the mmap. Offsets in the TensorDesc/VectorDesc are offsets
// into that buffer, so at<T>() hands out pointers the loader can upload exactly
// as it does for a real YSKN file.
//
// Nothing here touches CUDA: it is host code and the test runs without a GPU.
//
// ---------------------------------------------------------------------------
// Buffer layout
// ---------------------------------------------------------------------------
//
// One std::vector<uint8_t>, filled in this order, every payload aligned to its
// natural alignment (2 for f16, 4 for uint32, 8 for uint64):
//
//   per weight tensor, in the order Model::load asks for it:
//     [align 2]  lut16       2*4096 uint16 = 16384 B   -> TensorDesc::lut_off
//     [align 2]  scale16     physical_rows uint16      -> TensorDesc::scale_off
//     [align 4]  stream      physical_rows*seqs*wps uint32
//                                                      -> TensorDesc::stream_off
//                where seqs = cols/256 and wps = 4*bits, so
//                stream_bytes = physical_rows * seqs * 4*bits * 4
//
//   per f16 vector (norms and biases):
//     [align 2]  payload     n uint16                  -> VectorDesc::off
//
//   tail:
//     [align 1]  32768 zero bytes                      (see kTailPadBytes)
//
// TensorDesc::bias_off stays 0: Model::upload_linear never reads it, the
// attention biases arrive as vectors (bq/bk/bv) instead.
//
// All offsets are relative to buf_.data(), which operator new aligns to 16
// bytes, so every natural alignment above holds for the pointers at<T>()
// returns. open() asserts that at runtime.
//
// ---------------------------------------------------------------------------
// layout:warp vs layout:row
// ---------------------------------------------------------------------------
//
// Model::upload_linear feeds the stream through yskn_repack_warp_to_frag
// (src/repack.cu:10-18), so linear weights need the interleaved layout:warp
// order that interleave_warp() produces.
//
// Model::upload_embedding (src/loader.cpp:61-68) instead indexes
// stream[id*seqs*wps + q*wps + w] directly (src/misc_kernels.cu:18), i.e. the
// *logical* row-major order. So embed_tokens is emitted de-interleaved and
// carries layout:row. Both come from the same QuantizedTensor: the permutation
// is inverted with warp_src(), replicated from src/repack.cu:13-14 because
// gguf_quant.cpp keeps it private.
//
// ---------------------------------------------------------------------------
// What the loader asks for, and where it comes from
// ---------------------------------------------------------------------------
//
//   layers.N.q_proj              blk.N.attn_q.weight
//   layers.N.k_proj              blk.N.attn_k.weight
//   layers.N.v_proj              blk.N.attn_v.weight
//   layers.N.o_proj              blk.N.attn_output.weight
//   layers.N.gate_proj           blk.N.ffn_gate.weight
//   layers.N.up_proj             blk.N.ffn_up.weight
//   layers.N.down_proj           blk.N.ffn_down.weight
//   layers.N.input_layernorm     blk.N.attn_norm.weight        (f16 vector)
//   layers.N.post_attention_layernorm  blk.N.ffn_norm.weight   (f16 vector)
//   layers.N.q_proj.bias         blk.N.attn_q.bias             (f16 vector)
//   layers.N.k_proj.bias         blk.N.attn_k.bias             (f16 vector)
//   layers.N.v_proj.bias         blk.N.attn_v.bias             (f16 vector)
//   lm_head                      output.weight, or token_embd.weight when tied
//   embed_tokens                 token_embd.weight
//   norm                         output_norm.weight            (f16 vector)
//
// NOTE the engine names differ from gguf_map.cpp's: that mapper targets
// Engine35's vocabulary (layers.0.self_attn.q_proj, layers.0.mlp.gate_proj),
// while Model::load asks for the flat legacy names above. This file therefore
// carries its own table rather than reusing map_tensor().
//
// Missing *bias* vectors are synthesised as zeros (Qwen3 ships none; the loader
// still calls upload_vector for them and would throw on a missing name).
// Missing *weight* or *norm* entries are a hard error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "format.hpp"      // yskn::ModelDesc, TensorDesc, VectorDesc
#include "gguf.hpp"        // yskn::gguf::File
#include "gguf_quant.hpp"  // yskn::gguf::QuantConfig, Codebook

namespace yskn::gguf {

// Zero padding appended after the last real payload.
//
// src/loader.cpp:65 copies S*4 uint16 (32 KiB) from the embedding LUT, but the
// LUT itself is only 2*S uint16 (16 KiB), so upload_embedding over-reads by
// 16 KiB. In a YSKN container that lands in the neighbouring payload; here the
// buffer has to guarantee the bytes exist. 32 KiB covers the worst case with
// the LUT as the final payload.
constexpr size_t kTailPadBytes = 32768;

class GgufContainer {
 public:
  GgufContainer();
  ~GgufContainer();
  GgufContainer(GgufContainer&&) noexcept;
  GgufContainer& operator=(GgufContainer&&) noexcept;
  GgufContainer(const GgufContainer&) = delete;
  GgufContainer& operator=(const GgufContainer&) = delete;

  // Map the GGUF, read the hyperparameters, validate that the legacy loader can
  // consume this shape, then quantise every weight and materialise the buffer.
  //
  // Throws std::runtime_error with a message that names the offending dimension
  // or tensor when the shape is unusable (see the .cpp for the exact wording).
  // Throws std::invalid_argument when the quantiser rejects a matrix.
  void open(const std::string& path, const QuantConfig& cfg);

  // Same data Model::load reads from Container::model.
  ModelDesc model;

  // Same lookup semantics as Container::tensor / Container::vector: linear scan,
  // throw std::runtime_error("missing tensor <name>") when absent.
  const TensorDesc& tensor(const std::string& name) const;
  const VectorDesc& vector(const std::string& name) const;
  bool has_tensor(const std::string& name) const;
  bool has_vector(const std::string& name) const;

  // Raw pointer into the owned buffer. The loader casts these to uint16_t /
  // uint32_t / uint64_t / uint8_t, so every offset is naturally aligned.
  template <class T>
  const T* at(uint64_t off) const {
    return reinterpret_cast<const T*>(buf_.data() + off);
  }

  // Total size of the materialised image, tail padding included.
  size_t bytes() const;

  // Payload bytes only: everything except the tail padding.
  size_t payload_bytes() const { return payload_bytes_; }

  const std::vector<TensorDesc>& tensors() const { return tensors_; }
  const std::vector<VectorDesc>& vectors() const { return vectors_; }

  // The mapped file, for tests and for a future yskn info.
  const File& file() const { return file_; }
  const QuantConfig& quant_config() const { return cfg_; }
  // The codebook shared by every weight tensor. Trained once on a sample of the
  // widest matrix and reused through quantize_tensor's 'shared' argument; each
  // tensor still carries its own lut16 = f16(q*gamma) and its own scale16, so
  // build_lut8_tau recovers gamma per tensor as usual.
  const Codebook& codebook() const { return codebook_; }

  // True once open() has completed. tensor()/vector()/at() are only meaningful
  // after that.
  bool is_open() const { return opened_; }

 private:
  void build_model_desc();
  void validate_shape() const;
  void materialise();

  File file_;
  QuantConfig cfg_;
  Codebook codebook_;
  std::vector<uint8_t> buf_;
  std::vector<TensorDesc> tensors_;
  std::vector<VectorDesc> vectors_;
  size_t payload_bytes_ = 0;
  bool opened_ = false;
};

}  // namespace yskn::gguf