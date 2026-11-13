// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for the GGUF <-> engine wiring: file-kind detection, the probe report,
// the support gate and the C ABI entry point.
//
// The fixtures come from make_fixture_model.py (a 2-layer/64-hidden Qwen3-shaped
// model plus two damaged variants). That shape is deliberately far from the
// 5120/24/4/256 the kernels are built for, so the good fixture must come back
// supported=false with the exact mismatch list -- that rejection is the point of
// this slice, not a limitation of the test.
//
// Exits non-zero on failure and prints a PASS/FAIL summary.
#include "gguf_engine.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "yskn_abi.h"

namespace {

using json = nlohmann::json;

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
  const bool ok = (got - want <= tol) && (want - got <= tol);
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::ostringstream os;
    os << what << " (got " << got << ", want " << want << " +-" << tol << ")";
    g_failure_log.push_back(os.str());
    std::printf("FAIL: %s\n", os.str().c_str());
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

void check_contains(const std::string& haystack, const std::string& needle,
                    const std::string& what) {
  check(contains(haystack, needle), what + " contains '" + needle + "'");
}

std::string exe_dir(const char* argv0) {
  const std::string p = argv0 ? argv0 : "";
  const size_t slash = p.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// Write a file that starts with the given four magic bytes. Used for the
// "this is a .yskn, not a .gguf" case: the probe must reject on the magic
// alone, before any container parsing.
void write_magic_file(const std::string& path, const char* magic) {
  std::ofstream f(path, std::ios::binary);
  char buf[64] = {0};
  std::memcpy(buf, magic, 4);
  // A plausible YSKN header tail: version 1, json_len 0. The probe must not
  // care -- it rejects on the magic and never parses the container.
  buf[4] = 1;
  f.write(buf, sizeof(buf));
}

// The exact reason the good fixture must produce. Every clause is a separate
// mismatch; keeping the whole string here means a change to the gate or to the
// wording fails loudly instead of drifting.
const char* kGoodReason =
    "arch is \"qwen3\", the engine needs \"qwen3_5\"; "
    "hidden is 64, the kernels are built for 5120; "
    "heads is 4, the kernels are built for 24; "
    "kv_heads is 2, the kernels are built for 4; "
    "head_dim is 16, the kernels are built for 256; "
    "ffn is 128, the kernels need a multiple of 512";

// ---------------------------------------------------------------------------
// 1. File-kind detection
// ---------------------------------------------------------------------------
void check_detection(const std::string& good, const std::string& yskn_path,
                     const std::string& missing) {
  std::printf("\nFile-kind detection:\n");

  const yskn::gguf::FileKindInfo g = yskn::gguf::detect_file_kind(good);
  check_eq(g.kind == yskn::gguf::FileKind::Gguf, true, "detect: good fixture is GGUF");
  check_eq(g.readable, true, "detect: good fixture is readable");
  check_eq(g.magic, std::string("\"GGUF\""), "detect: good fixture magic");
  check_eq(g.error.empty(), true, "detect: no error for a GGUF file");

  const yskn::gguf::FileKindInfo y = yskn::gguf::detect_file_kind(yskn_path);
  check_eq(y.kind == yskn::gguf::FileKind::Yskn, true, "detect: YSKN magic is Yskn");
  check_eq(y.magic, std::string("\"YSKN\""), "detect: YSKN magic text");

  const yskn::gguf::FileKindInfo m = yskn::gguf::detect_file_kind(missing);
  check_eq(m.kind == yskn::gguf::FileKind::Unknown, true, "detect: missing path is Unknown");
  check_eq(m.readable, false, "detect: missing path is not readable");
  check_contains(m.error, "cannot open", "detect: missing path error");

  // A file shorter than the magic.
  const std::string tiny = yskn_path + ".tiny";
  {
    std::ofstream f(tiny, std::ios::binary);
    f.write("GG", 2);
  }
  const yskn::gguf::FileKindInfo t = yskn::gguf::detect_file_kind(tiny);
  check_eq(t.kind == yskn::gguf::FileKind::Unknown, true, "detect: 2-byte file is Unknown");
  check_contains(t.error, "too short", "detect: short file error");
  std::remove(tiny.c_str());
}

// ---------------------------------------------------------------------------
// 2. The good fixture: fields, coverage, validation, the support gate
// ---------------------------------------------------------------------------
void check_good_probe(const std::string& path) {
  std::printf("\n-- good fixture: %s\n", path.c_str());

  const yskn::gguf::ProbeResult r = yskn::gguf::probe(path);
  check(r.ok, "probe: good fixture parses");
  check_eq(r.error.empty(), true, "probe: no error on the good fixture");
  check_eq(r.path, path, "probe: path echoed");

  // Config, straight from the fixture generator.
  check_eq(r.cfg.arch, std::string("qwen3"), "probe: arch");
  check_eq(r.cfg.layers, 2, "probe: layers");
  check_eq(r.cfg.hidden, 64, "probe: hidden");
  check_eq(r.cfg.ffn, 128, "probe: ffn");
  check_eq(r.cfg.heads, 4, "probe: heads");
  check_eq(r.cfg.kv_heads, 2, "probe: kv_heads");
  check_eq(r.cfg.head_dim, 16, "probe: head_dim");
  check_eq(r.cfg.vocab, 128, "probe: vocab");
  check_near(r.cfg.eps, 1e-6, 1e-12, "probe: eps");
  check_near(r.cfg.rope_theta, 1000000.0, 1.0, "probe: rope_theta");
  check_eq(r.cfg.tie_embeddings, false, "probe: tie_embeddings");

  // Tensors: 3 global + 2 blocks x 11.
  check_eq(r.n_tensors, uint64_t{25}, "probe: tensor count");
  check_eq(r.bytes_by_type.count("Q8_0"), size_t{1}, "probe: Q8_0 is counted");
  check_eq(r.bytes_by_type.count("F32"), size_t{1}, "probe: F32 is counted");
  check(r.bytes_by_type.at("Q8_0") > 0, "probe: Q8_0 bytes are non-zero");
  check(r.bytes_by_type.at("F32") > 0, "probe: F32 bytes are non-zero");

  // Coverage: a complete, valid Qwen3 model.
  check_eq(r.coverage.matched.size(), size_t{25}, "probe: coverage matched");
  check_eq(r.coverage.unmatched.size(), size_t{0}, "probe: coverage unmatched");
  check_eq(r.coverage.missing_expected.size(), size_t{0}, "probe: coverage missing");
  check(r.coverage.complete(), "probe: coverage is complete");

  // Validation.
  check(r.validation.ok, "probe: validation ok");
  check_eq(r.validation.errors.size(), size_t{0}, "probe: validation error count");

  // The gate. The fixture is 2 layers / 64 hidden: it must be rejected, with
  // the numbers spelled out.
  check_eq(r.supported, false, "probe: the 2-layer fixture is not supported");
  check_eq(r.reason, std::string(kGoodReason), "probe: exact rejection reason");
  check_contains(r.reason, "hidden is 64, the kernels are built for 5120",
                 "probe: reason names the hidden mismatch");
  check_contains(r.reason, "the kernels are built for", "probe: reason is actionable");

  // engine_supports() must agree with the probe.
  std::string reason;
  check_eq(yskn::gguf::engine_supports(r.cfg, reason), false,
           "engine_supports: rejects the fixture config");
  check_eq(reason, r.reason, "engine_supports: same reason as the probe");
}

// A config that passes every shape check must still report unsupported, because
// no GGUF weights are consumed yet. This is the honest-answer requirement.
void check_support_gate_accepts_27b_shape() {
  std::printf("\n-- support gate at the 27B shape\n");

  yskn::gguf::ModelConfig c;
  c.arch = "qwen3_5";
  c.layers = 64;
  c.hidden = 5120;
  c.ffn = 17408;  // 17408 = 34 * 512
  c.heads = 24;
  c.kv_heads = 4;
  c.head_dim = 256;
  c.vocab = 152064;

  std::string reason;
  check_eq(yskn::gguf::engine_supports(c, reason), true,
           "engine_supports: accepts the 27B shape");
  check_contains(reason, "not wired into", "engine_supports: names the remaining blocker");
  check_contains(reason, "YSKN container", "engine_supports: says what it still reads");

  // ffn must be a multiple of 512.
  c.ffn = 17400;
  check_eq(yskn::gguf::engine_supports(c, reason), false,
           "engine_supports: rejects ffn not a multiple of 512");
  check_contains(reason, "ffn is 17400", "engine_supports: names the ffn value");

  // Each shape field is checked independently.
  c.ffn = 17408;
  c.head_dim = 128;
  check_eq(yskn::gguf::engine_supports(c, reason), false,
           "engine_supports: rejects head_dim 128");
  c.head_dim = 256;
  c.kv_heads = 8;
  check_eq(yskn::gguf::engine_supports(c, reason), false,
           "engine_supports: rejects kv_heads 8");
  c.kv_heads = 4;
  c.heads = 32;
  check_eq(yskn::gguf::engine_supports(c, reason), false,
           "engine_supports: rejects heads 32");
  c.heads = 24;
  c.hidden = 4096;
  check_eq(yskn::gguf::engine_supports(c, reason), false,
           "engine_supports: rejects hidden 4096");
  check_contains(reason, "hidden is 4096, the kernels are built for 5120",
                 "engine_supports: the 4096 message is the documented one");
}

// ---------------------------------------------------------------------------
// 3. Rejections: a .yskn file, a missing path, a truncated file
// ---------------------------------------------------------------------------
void check_yskn_rejected(const std::string& path) {
  std::printf("\n-- .yskn magic: %s\n", path.c_str());

  const yskn::gguf::ProbeResult r = yskn::gguf::probe(path);
  check_eq(r.ok, false, "probe(.yskn): not ok");
  check_contains(r.error, "not a GGUF", "probe(.yskn): error says 'not a GGUF'");
  check_contains(r.error, "YSKN container", "probe(.yskn): error names the YSKN container");
  check_contains(r.error, "yskn_open", "probe(.yskn): error points at yskn_open");
  check_eq(r.supported, false, "probe(.yskn): not supported");
  check_eq(r.n_tensors, uint64_t{0}, "probe(.yskn): no tensors were read");

  // The JSON must still be a parseable report with ok=false.
  const std::string doc = yskn::gguf::probe_json(r);
  json j;
  bool parsed = true;
  try {
    j = json::parse(doc);
  } catch (const std::exception&) {
    parsed = false;
  }
  check(parsed, "probe(.yskn): JSON parses");
  if (parsed) {
    check_eq(j.at("ok").get<bool>(), false, "probe(.yskn): JSON ok is false");
    check_contains(j.at("error").get<std::string>(), "not a GGUF",
                   "probe(.yskn): JSON error");
  }
}

void check_missing_rejected(const std::string& path) {
  std::printf("\n-- missing path: %s\n", path.c_str());

  const yskn::gguf::ProbeResult r = yskn::gguf::probe(path);
  check_eq(r.ok, false, "probe(missing): not ok");
  check_contains(r.error, "cannot open", "probe(missing): error says 'cannot open'");
  check_contains(r.error, path, "probe(missing): error names the path");
  check_eq(r.supported, false, "probe(missing): not supported");
}

// ---------------------------------------------------------------------------
// 4. The damaged fixtures still produce a usable report
// ---------------------------------------------------------------------------
void check_broken_probe(const std::string& path) {
  std::printf("\n-- broken fixture: %s\n", path.c_str());

  const yskn::gguf::ProbeResult r = yskn::gguf::probe(path);
  check(r.ok, "probe(broken): the file still parses");
  check_eq(r.validation.ok, false, "probe(broken): validation fails");
  check(r.validation.errors.size() > 0, "probe(broken): errors are reported");
  bool named = false;
  for (const std::string& e : r.validation.errors)
    if (contains(e, "ffn_down")) named = true;
  check(named, "probe(broken): an error names ffn_down");
  check_eq(r.supported, false, "probe(broken): not supported");
}

void check_unknown_probe(const std::string& path) {
  std::printf("\n-- unknown-tensor fixture: %s\n", path.c_str());

  const yskn::gguf::ProbeResult r = yskn::gguf::probe(path);
  check(r.ok, "probe(unknown): the file still parses");
  check_eq(r.coverage.unmatched.size(), size_t{1}, "probe(unknown): one unmatched tensor");
  if (!r.coverage.unmatched.empty())
    check_eq(r.coverage.unmatched[0], std::string("blk.0.attn_rot.weight"),
             "probe(unknown): the unmatched tensor is named");
  bool warned = false;
  for (const std::string& w : r.validation.warnings)
    if (contains(w, "attn_rot")) warned = true;
  check(warned, "probe(unknown): a warning names the unmatched tensor");
}

// ---------------------------------------------------------------------------
// 5. The C ABI
// ---------------------------------------------------------------------------
void check_c_abi(const std::string& good, const std::string& yskn_path,
                 const std::string& missing) {
  std::printf("\n-- C ABI\n");

  std::vector<char> buf(64 * 1024, 0);
  const int rc = yskn_gguf_probe(good.c_str(), buf.data(), static_cast<int>(buf.size()));
  check_eq(rc, 0, "yskn_gguf_probe: returns 0 on the good fixture");

  const std::string doc(buf.data());
  check_contains(doc, "\"supported\": false", "yskn_gguf_probe: JSON says unsupported");
  check_contains(doc, "\"arch\": \"qwen3\"", "yskn_gguf_probe: JSON has the arch");
  check_contains(doc, "\"n_tensors\": 25", "yskn_gguf_probe: JSON has the tensor count");

  // The JSON must round-trip through nlohmann without throwing.
  json j;
  bool parsed = true;
  try {
    j = json::parse(doc);
  } catch (const std::exception& ex) {
    parsed = false;
    std::printf("  parse error: %s\n", ex.what());
  }
  check(parsed, "yskn_gguf_probe: JSON parses");

  if (parsed) {
    // Every documented field, in the documented shape.
    for (const char* key : {"ok", "path", "arch", "layers", "hidden", "ffn", "heads",
                            "kv_heads", "head_dim", "vocab", "eps", "rope_theta",
                            "tie_embeddings", "n_tensors", "bytes_by_type", "coverage",
                            "validation", "supported", "reason"}) {
      check(j.contains(key), std::string("yskn_gguf_probe: JSON has '") + key + "'");
    }
    check_eq(j.at("ok").get<bool>(), true, "yskn_gguf_probe: ok is true");
    check_eq(j.at("path").get<std::string>(), good, "yskn_gguf_probe: path");
    check_eq(j.at("layers").get<int>(), 2, "yskn_gguf_probe: layers");
    check_eq(j.at("hidden").get<int>(), 64, "yskn_gguf_probe: hidden");
    check_eq(j.at("ffn").get<int>(), 128, "yskn_gguf_probe: ffn");
    check_eq(j.at("heads").get<int>(), 4, "yskn_gguf_probe: heads");
    check_eq(j.at("kv_heads").get<int>(), 2, "yskn_gguf_probe: kv_heads");
    check_eq(j.at("head_dim").get<int>(), 16, "yskn_gguf_probe: head_dim");
    check_eq(j.at("vocab").get<int>(), 128, "yskn_gguf_probe: vocab");
    check_near(j.at("eps").get<double>(), 1e-6, 1e-12, "yskn_gguf_probe: eps");
    check_near(j.at("rope_theta").get<double>(), 1000000.0, 1.0,
               "yskn_gguf_probe: rope_theta");
    check_eq(j.at("tie_embeddings").get<bool>(), false, "yskn_gguf_probe: tie_embeddings");
    check_eq(j.at("n_tensors").get<uint64_t>(), uint64_t{25}, "yskn_gguf_probe: n_tensors");
    check_eq(j.at("supported").get<bool>(), false, "yskn_gguf_probe: supported");
    check_eq(j.at("reason").get<std::string>(), std::string(kGoodReason),
             "yskn_gguf_probe: exact reason text");

    const json& by_type = j.at("bytes_by_type");
    check(by_type.is_object(), "yskn_gguf_probe: bytes_by_type is an object");
    check(by_type.contains("Q8_0"), "yskn_gguf_probe: bytes_by_type has Q8_0");
    check(by_type.contains("F32"), "yskn_gguf_probe: bytes_by_type has F32");
    check(by_type.at("Q8_0").get<uint64_t>() > 0, "yskn_gguf_probe: Q8_0 bytes non-zero");

    const json& cov = j.at("coverage");
    check_eq(cov.at("matched").get<size_t>(), size_t{25}, "yskn_gguf_probe: coverage matched");
    check_eq(cov.at("unmatched").get<size_t>(), size_t{0},
             "yskn_gguf_probe: coverage unmatched");
    check_eq(cov.at("missing").get<size_t>(), size_t{0}, "yskn_gguf_probe: coverage missing");

    const json& val = j.at("validation");
    check_eq(val.at("ok").get<bool>(), true, "yskn_gguf_probe: validation ok");
    check(val.at("errors").is_array(), "yskn_gguf_probe: validation errors is an array");
    check(val.at("warnings").is_array(), "yskn_gguf_probe: validation warnings is an array");
  }

  // A .yskn file: non-zero return, but a parseable report that says why.
  std::vector<char> ybuf(64 * 1024, 0);
  const int yrc = yskn_gguf_probe(yskn_path.c_str(), ybuf.data(),
                                  static_cast<int>(ybuf.size()));
  check_eq(yrc != 0, true, "yskn_gguf_probe: non-zero for a .yskn file");
  const std::string ydoc(ybuf.data());
  check_contains(ydoc, "not a GGUF", "yskn_gguf_probe: .yskn report says 'not a GGUF'");
  bool yparsed = true;
  try {
    const json yj = json::parse(ydoc);
    yparsed = yj.at("ok").get<bool>() == false;
  } catch (const std::exception&) {
    yparsed = false;
  }
  check(yparsed, "yskn_gguf_probe: .yskn report still parses as JSON with ok=false");

  // A missing path: non-zero, no crash, a report that names the path.
  std::vector<char> mbuf(64 * 1024, 0);
  const int mrc = yskn_gguf_probe(missing.c_str(), mbuf.data(),
                                  static_cast<int>(mbuf.size()));
  check_eq(mrc != 0, true, "yskn_gguf_probe: non-zero for a missing path");
  check_contains(std::string(mbuf.data()), "cannot open",
                 "yskn_gguf_probe: missing-path report");

  // A buffer that is too small must fail cleanly instead of overflowing.
  std::vector<char> small(8, static_cast<char>(0x7f));
  const int src = yskn_gguf_probe(good.c_str(), small.data(),
                                  static_cast<int>(small.size()));
  check_eq(src != 0, true, "yskn_gguf_probe: non-zero when the buffer is too small");
  check_eq(small.back(), '\0', "yskn_gguf_probe: the short buffer is NUL terminated");

  // Null arguments.
  check_eq(yskn_gguf_probe(nullptr, buf.data(), static_cast<int>(buf.size())) != 0, true,
           "yskn_gguf_probe: null path fails");
  check_eq(yskn_gguf_probe(good.c_str(), nullptr, 0) != 0, true,
           "yskn_gguf_probe: null buffer fails");
  check_eq(yskn_gguf_probe("", buf.data(), static_cast<int>(buf.size())) != 0, true,
           "yskn_gguf_probe: empty path fails");
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

  std::printf("GGUF engine wiring test\n");
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

  // A file with YSKN magic, written next to the fixtures so the test does not
  // depend on a real container being checked in.
  const std::string yskn_path = out_dir + "/gguf_engine_test_yskn_magic.bin";
  write_magic_file(yskn_path, "YSKN");
  const std::string missing = out_dir + "/gguf_engine_test_does_not_exist.gguf";
  std::remove(missing.c_str());

  check_detection(good, yskn_path, missing);
  check_good_probe(good);
  check_support_gate_accepts_27b_shape();
  check_yskn_rejected(yskn_path);
  check_missing_rejected(missing);
  check_broken_probe(broken);
  check_unknown_probe(unknown);
  check_c_abi(good, yskn_path, missing);

  std::remove(yskn_path.c_str());

  std::printf("\n");
  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  for (const std::string& f : g_failure_log) std::printf("  - %s\n", f.c_str());
  return 1;
}
