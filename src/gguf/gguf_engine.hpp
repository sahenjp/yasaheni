// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF <-> engine wiring: file-kind detection, the probe, and the support gate.
//
// This is the slice that lets a caller point the library at a stock .gguf file
// and get an honest answer about what would happen. It does not load weights:
// Engine35 still reads the YSKN container, so a GGUF that passes the shape gate
// is reported as "not runnable yet" instead of being half-loaded.
//
// Nothing here touches CUDA, so the tests link it without a GPU.
#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "gguf_model.hpp"

namespace yskn::gguf {

// ---------------------------------------------------------------------------
// File kind: what the first four bytes say.
// ---------------------------------------------------------------------------
enum class FileKind {
  Yskn,     // "YSKN" -- the native container (src/format.hpp)
  Gguf,     // "GGUF" -- llama.cpp's format (src/gguf/gguf.hpp)
  Unknown,  // anything else
};

struct FileKindInfo {
  FileKind kind = FileKind::Unknown;
  bool readable = false;  // the first four bytes were read
  std::string magic;      // the four bytes, quoted or as 0x........
  std::string error;      // set when kind == Unknown
};

// Peek the first four bytes. Never throws: a file that cannot be opened, or is
// shorter than four bytes, comes back as Unknown with 'error' explaining which.
FileKindInfo detect_file_kind(const std::string& path);

// ---------------------------------------------------------------------------
// The shapes the compiled kernels accept.
//
// These are the LEGACY attention path's numbers, not Engine35's. The legacy
// kernels are not templated over the attention shape: src/post_kernels.cu:47
// writes q16 with a literal stride of 28 heads x 128 slots and
// src/attention_kv4q.cu:81 reads it back with the same assumption. They were
// built for a 28-head, head_dim-128 model and never generalised.
//
// Repeated here rather than included from src/q35.hpp because that header pulls
// in CUDA headers and this translation unit must stay host-only.
// ---------------------------------------------------------------------------
namespace kernel_shape {
inline constexpr int kLegacyHeads = 28;
inline constexpr int kLegacyHeadDim = 128;
inline constexpr int kFfnMultiple = 512;
}  // namespace kernel_shape

// Can the engine load a model with this config *today*? 'reason' names the
// numbers either way, because it is shown to a user verbatim.
bool engine_supports(const ModelConfig& cfg, std::string& reason);

// ---------------------------------------------------------------------------
// The probe: everything yskn_open() knows about a GGUF file, without loading it.
// ---------------------------------------------------------------------------
struct ProbeResult {
  bool ok = false;    // the file was opened and parsed as GGUF
  std::string path;
  std::string error;  // why ok is false

  ModelConfig cfg;
  uint64_t n_tensors = 0;
  std::map<std::string, uint64_t> bytes_by_type;  // "Q8_0" -> payload bytes
  CoverageReport coverage;
  ValidationReport validation;

  bool supported = false;
  std::string reason;  // never empty: the mismatch list, or the remaining blocker
};

// Open, read the config, validate and report. Never throws.
ProbeResult probe(const std::string& path);

// Render a ProbeResult as the JSON document the C ABI returns.
std::string probe_json(const ProbeResult& r);

}  // namespace yskn::gguf
