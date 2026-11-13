// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cuda_runtime.h>
#include "format.hpp"
#include "kernels.hpp"
#include "post_kernels.hpp"
#include "gguf/gguf_container.hpp"
namespace yskn {
struct DevTensor {                       
  TensorDev dev{}; int bits = 6; int kmax = 6;            
  uint4* frag = nullptr; uint16_t* lut8 = nullptr; float* tau = nullptr; uint64_t* tile_off = nullptr; uint8_t* tile_k = nullptr; int8_t* plan = nullptr;
  int8_t *signs_in = nullptr, *signs_out = nullptr; uint16_t* qm_in = nullptr; int m_in = 1;
  uint32_t* row_stream = nullptr; uint16_t* lut16 = nullptr; uint16_t* scale16 = nullptr;   
  float* part = nullptr;                                                                     
  uint4* lno = nullptr;                                                                      
  int nw_big = 0;                                                                            
  std::vector<std::vector<int>> tiles_by_k;                                                 
};
struct DevLayer { DevTensor q, k, v, o, gate, up, down; uint16_t *ln1 = nullptr, *ln2 = nullptr, *bq = nullptr, *bk = nullptr, *bv = nullptr; KvLayer kv{}; };
struct Arena {                           
  int Mmax = 32; float* h = nullptr; float* h2 = nullptr;   uint16_t *q16 = nullptr, *o16 = nullptr, *xff = nullptr; int8_t* qx_ff = nullptr; float* alpha_ff = nullptr;
  float *pm = nullptr, *pl = nullptr, *pacc = nullptr; float* logits = nullptr; float* blk_val = nullptr; int* blk_idx = nullptr; float* blk_max = nullptr; float* blk_sum = nullptr;
  int *d_pos = nullptr, *d_tok = nullptr, *d_next = nullptr, *d_target = nullptr; float* d_nll = nullptr;
  
  
  int *d_toks = nullptr, *d_next8 = nullptr, *d_target8 = nullptr; float* d_nll8 = nullptr;
  float* tk_val = nullptr; int* tk_idx = nullptr; float* top_val = nullptr; int* top_idx = nullptr;
};
// ---------------------------------------------------------------------------
// Model: the legacy (non-qwen3_5) engine's weights.
//
// Two backing containers can be live, never both:
//
//   c      a YSKN container (src/format.hpp), mmap'd, already quantised
//   gc     a GGUF file (src/gguf/gguf_container.hpp), quantised into an owned
//          buffer when it is opened, exposing the same tensor()/vector()/at()
//          surface
//
// gguf_ says which one is live. Everything below the open -- the whole of
// Model::load -- reads through T()/V()/at() (see the free functions after the
// struct), so the loader body is identical for both formats. The YSKN path is
// byte-for-byte the old one: open_backing() only takes the GGUF branch when the
// first four bytes are "GGUF".
// ---------------------------------------------------------------------------
struct Model {
  Container c; gguf::GgufContainer gc; bool gguf_ = false;
  ModelDesc cfg; std::vector<DevLayer> layers; DevTensor emb, head; uint16_t* final_norm = nullptr;
  int max_ctx = 2048, splits = 32, sm_count = 0; size_t stage_bytes_body = 0, stage_bytes_head = 0;
  // Peek the first four bytes and open whichever container they name.
  //   "GGUF" -> gc.open(path, qc), gguf_ = true
  //   anything else (including "YSKN", a short file and an unreadable path)
  //          -> c.open(path), i.e. exactly the previous behaviour and the
  //             previous error text
  // Sets cfg to the live container's ModelDesc. Throws whatever the container
  // throws; gguf_ is left false unless the GGUF opened successfully.
  void open_backing(const std::string& path, const gguf::QuantConfig& qc = gguf::QuantConfig()) {
    gguf_ = false;
    char magic[4] = {0, 0, 0, 0};
    { std::ifstream f(path, std::ios::binary);
      if (f) { f.read(magic, 4); if (f.gcount() < 4) { magic[0] = magic[1] = magic[2] = magic[3] = 0; } } }
    if (std::memcmp(magic, "GGUF", 4) == 0) {
      gc.open(path, qc);
      cfg = gc.model;
      gguf_ = true;
      return;
    }
    c.open(path);
    cfg = c.model;
  }
  void load(const std::string& path, int max_ctx_, int Mmax);
  DevTensor upload_linear(const TensorDesc& d, int Mmax, int want_workers, int force_workers = 0, const Container* src = nullptr);   
  void upload_embedding(const TensorDesc& d);
  uint16_t* upload_vector(const std::string& name, int n);
};
// ---------------------------------------------------------------------------
// Accessor resolution: the four calls the loader makes on its container.
//
// 'src' is the explicit-container override Model::upload_linear already had for
// the d3 draft model (src/engine.cpp:553-565); it is always a plain Container
// and always a YSKN one, so it short-circuits the dispatch.
// ---------------------------------------------------------------------------
inline const TensorDesc& T(const Model& m, const std::string& n, const Container* src = nullptr) {
  if (src) return src->tensor(n);
  return m.gguf_ ? m.gc.tensor(n) : m.c.tensor(n);
}
inline const VectorDesc& V(const Model& m, const std::string& n, const Container* src = nullptr) {
  if (src) return src->vector(n);
  return m.gguf_ ? m.gc.vector(n) : m.c.vector(n);
}
template <class T2> inline const T2* at(const Model& m, uint64_t off, const Container* src = nullptr) {
  if (src) return src->at<T2>(off);
  return m.gguf_ ? m.gc.at<T2>(off) : m.c.at<T2>(off);
}
}
