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
// Mirrors the hard gate in the Engine35 constructor (src/engine35.cpp:216-218)
// and the constants in src/q35.hpp: Q35_H 5120, Q35_NH 24, Q35_KVH 4,
// Q35_HD 256. They are repeated here rather than included because q35.hpp pulls
// in CUDA headers and this translation unit must stay host-only.
// ---------------------------------------------------------------------------
namespace kernel_shape {
inline constexpr const char* kArch = "qwen3_5";
inline constexpr int kHidden = 5120;
inline constexpr int kHeads = 24;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 256;
inline constexpr int kFfnMultiple = 512;
}  // namespace kernel_shape

// Can Engine35 run a model with this config *today*? 'reason' is filled either
// way: the mismatch list when false, the remaining blocker when true. Both are
// meant to be shown to a user verbatim, so they name numbers, not codes.
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
