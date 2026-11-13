// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF <-> engine wiring. Three jobs, all host-only:
//
//   detect_file_kind()  peek the first four bytes ("YSKN" / "GGUF" / neither)
//   probe()             open, read the config, validate, report -- never throws
//   yskn_gguf_probe()   the C ABI entry point that renders probe() as JSON
//
// The support gate mirrors the hard shape check in the Engine35 constructor
// (src/engine35.cpp:216-218) and the constants in src/q35.hpp. Those headers
// pull in CUDA, so the numbers are repeated in gguf_engine.hpp instead of
// included: this translation unit must stay linkable without a GPU.
#include "gguf_engine.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "yskn_abi.h"

namespace yskn::gguf {
namespace {

using json = nlohmann::json;

// Render four magic bytes for an error message: quoted when printable, hex
// otherwise, so a binary file cannot put control characters into a log line.
std::string magic_repr(const char* b, size_t n) {
  bool printable = true;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(b[i]);
    if (c < 0x20 || c > 0x7e) printable = false;
  }
  if (printable) return "\"" + std::string(b, n) + "\"";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%02X%02X%02X%02X",
                static_cast<unsigned char>(b[0]), static_cast<unsigned char>(b[1]),
                static_cast<unsigned char>(b[2]), static_cast<unsigned char>(b[3]));
  return std::string(buf);
}

std::string join(const std::vector<std::string>& parts, const char* sep) {
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out += sep;
    out += parts[i];
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// File-kind detection
// ---------------------------------------------------------------------------
FileKindInfo detect_file_kind(const std::string& path) {
  FileKindInfo info;

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    info.error = "cannot open '" + path + "' for reading";
    return info;
  }

  char magic[4] = {0, 0, 0, 0};
  f.read(magic, 4);
  const std::streamsize got = f.gcount();
  if (got < 4) {
    info.error = "'" + path + "' is " + std::to_string(got) +
                 " byte(s) long, too short to hold a 4-byte magic";
    return info;
  }

  info.readable = true;
  info.magic = magic_repr(magic, 4);
  if (std::memcmp(magic, "YSKN", 4) == 0) {
    info.kind = FileKind::Yskn;
  } else if (std::memcmp(magic, "GGUF", 4) == 0) {
    info.kind = FileKind::Gguf;
  } else {
    info.kind = FileKind::Unknown;
    info.error = "'" + path + "' starts with " + info.magic +
                 ", which is neither a YSKN container (magic \"YSKN\") nor a "
                 "GGUF file (magic \"GGUF\")";
  }
  return info;
}

// ---------------------------------------------------------------------------
// The support gate
// ---------------------------------------------------------------------------
bool engine_supports(const ModelConfig& cfg, std::string& reason) {
  std::vector<std::string> bad;

  if (cfg.arch != kernel_shape::kArch) {
    bad.push_back("arch is \"" + (cfg.arch.empty() ? std::string("(missing)") : cfg.arch) +
                  "\", the engine needs \"" + std::string(kernel_shape::kArch) + "\"");
  }
  if (cfg.hidden != kernel_shape::kHidden) {
    bad.push_back("hidden is " + std::to_string(cfg.hidden) +
                  ", the kernels are built for " + std::to_string(kernel_shape::kHidden));
  }
  if (cfg.heads != kernel_shape::kHeads) {
    bad.push_back("heads is " + std::to_string(cfg.heads) +
                  ", the kernels are built for " + std::to_string(kernel_shape::kHeads));
  }
  if (cfg.kv_heads != kernel_shape::kKvHeads) {
    bad.push_back("kv_heads is " + std::to_string(cfg.kv_heads) +
                  ", the kernels are built for " + std::to_string(kernel_shape::kKvHeads));
  }
  if (cfg.head_dim != kernel_shape::kHeadDim) {
    bad.push_back("head_dim is " + std::to_string(cfg.head_dim) +
                  ", the kernels are built for " + std::to_string(kernel_shape::kHeadDim));
  }
  if (cfg.ffn <= 0 || cfg.ffn % kernel_shape::kFfnMultiple != 0) {
    bad.push_back("ffn is " + std::to_string(cfg.ffn) + ", the kernels need a multiple of " +
                  std::to_string(kernel_shape::kFfnMultiple));
  }

  if (!bad.empty()) {
    reason = join(bad, "; ");
    return false;
  }

  // The shape gate passes, but nothing consumes GGUF weights yet: Engine35
  // still loads a Container. Say so instead of implying the file would run.
  reason =
      "the shape matches the compiled kernels, but GGUF weights are not wired into "
      "Engine35 yet (it still reads YSKN containers)";
  return true;
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------
ProbeResult probe(const std::string& path) {
  ProbeResult r;
  r.path = path;

  const FileKindInfo info = detect_file_kind(path);
  if (info.kind == FileKind::Yskn) {
    r.error = "'" + path +
              "' is a YSKN container (magic \"YSKN\"), not a GGUF file: "
              "yskn_gguf_probe inspects GGUF files only, use yskn_open for .yskn";
    return r;
  }
  if (info.kind == FileKind::Unknown) {
    r.error = info.error;
    return r;
  }

  try {
    File f;
    f.open(path);
    r.cfg = read_config(f);
    r.coverage = coverage(f, r.cfg);
    r.validation = validate(f, r.cfg);
    r.n_tensors = static_cast<uint64_t>(f.tensors().size());

    for (const TensorInfo& t : f.tensors()) {
      uint64_t nbytes = 0;
      try {
        nbytes = t.nbytes();
      } catch (const std::exception&) {
        nbytes = 0;  // a malformed tensor is reported by validate(), not here
      }
      r.bytes_by_type[type_name(t.type)] += nbytes;
    }
    r.ok = true;
  } catch (const std::exception& ex) {
    r.error = ex.what();
    return r;
  }

  r.supported = engine_supports(r.cfg, r.reason);
  return r;
}

std::string probe_json(const ProbeResult& r) {
  json j;
  j["ok"] = r.ok;
  j["path"] = r.path;
  j["error"] = r.error;

  j["arch"] = r.cfg.arch;
  j["layers"] = r.cfg.layers;
  j["hidden"] = r.cfg.hidden;
  j["ffn"] = r.cfg.ffn;
  j["heads"] = r.cfg.heads;
  j["kv_heads"] = r.cfg.kv_heads;
  j["head_dim"] = r.cfg.head_dim;
  j["vocab"] = r.cfg.vocab;
  j["eps"] = static_cast<double>(r.cfg.eps);
  j["rope_theta"] = static_cast<double>(r.cfg.rope_theta);
  j["tie_embeddings"] = r.cfg.tie_embeddings;
  j["n_tensors"] = r.n_tensors;

  json by_type = json::object();
  for (const auto& kv : r.bytes_by_type) by_type[kv.first] = kv.second;
  j["bytes_by_type"] = by_type;

  json cov;
  cov["matched"] = r.coverage.matched.size();
  cov["unmatched"] = r.coverage.unmatched.size();
  cov["missing"] = r.coverage.missing_expected.size();
  // The counts are what a script checks; the names are what a human needs to
  // fix the file, so both travel.
  cov["unmatched_tensors"] = r.coverage.unmatched;
  cov["missing_tensors"] = r.coverage.missing_expected;
  j["coverage"] = cov;

  json val;
  val["ok"] = r.validation.ok;
  val["errors"] = r.validation.errors;
  val["warnings"] = r.validation.warnings;
  j["validation"] = val;

  j["supported"] = r.supported;
  j["reason"] = r.reason;

  return j.dump(2);
}

}  // namespace yskn::gguf

// ---------------------------------------------------------------------------
// C ABI. No exception escapes: probe() and probe_json() are both noexcept in
// practice, and the buffer handling below is plain memcpy.
// ---------------------------------------------------------------------------
extern "C" int yskn_gguf_probe(const char* path, char* out_json, int out_cap) {
  using namespace yskn::gguf;

  if (path == nullptr || path[0] == '\0') {
    if (out_json != nullptr && out_cap > 0) out_json[0] = '\0';
    return -1;
  }

  const ProbeResult r = probe(std::string(path));  // never throws
  const std::string doc = probe_json(r);

  if (out_json == nullptr || out_cap <= 0) return -1;
  if (static_cast<size_t>(out_cap) < doc.size() + 1) {
    // Leave a terminated but truncated string behind; the caller sees a
    // non-zero return and knows to retry with a bigger buffer.
    std::memcpy(out_json, doc.data(), static_cast<size_t>(out_cap) - 1);
    out_json[out_cap - 1] = '\0';
    return -1;
  }

  std::memcpy(out_json, doc.c_str(), doc.size() + 1);
  return r.ok ? 0 : -1;
}
