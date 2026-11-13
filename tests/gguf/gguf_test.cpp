// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Tests for the GGUF reader: parse the synthetic fixture produced by
// make_fixture.py, check metadata / tensor table / alignment, then dequantize
// every tensor and compare against the expected floats the generator wrote to
// the sidecar JSON.
//
// Exits non-zero on the first hard failure; prints a PASS/FAIL summary.
#include "gguf.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

// ---------------------------------------------------------------------------
// Minimal JSON reader for the sidecar. Supports objects, arrays, strings,
// numbers, true/false/null -- everything make_fixture.py emits. Keeping this
// local avoids pulling a JSON dependency into the reader's test target.
// ---------------------------------------------------------------------------
struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* find(const std::string& key) const {
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  const Json& at(const std::string& key) const {
    const Json* v = find(key);
    if (!v) {
      std::fprintf(stderr, "sidecar: missing key '%s'\n", key.c_str());
      std::exit(2);
    }
    return *v;
  }
  double as_num() const {
    if (kind == Number) return num;
    if (kind == Bool) return b ? 1.0 : 0.0;
    std::fprintf(stderr, "sidecar: value is not a number\n");
    std::exit(2);
  }
  // 64-bit metadata values do not survive a round trip through double, so the
  // integer accessors reparse the literal text instead.
  uint64_t as_u64() const {
    if (kind == Bool) return b ? 1u : 0u;
    if (kind != Number) {
      std::fprintf(stderr, "sidecar: value is not an unsigned integer\n");
      std::exit(2);
    }
    return std::strtoull(str.c_str(), nullptr, 10);
  }
  int64_t as_i64() const {
    if (kind == Bool) return b ? 1 : 0;
    if (kind != Number) {
      std::fprintf(stderr, "sidecar: value is not an integer\n");
      std::exit(2);
    }
    return std::strtoll(str.c_str(), nullptr, 10);
  }
  std::string as_str() const {
    if (kind != String) {
      std::fprintf(stderr, "sidecar: value is not a string\n");
      std::exit(2);
    }
    return str;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : s_(text) {}

  Json parse() {
    Json v = value();
    ws();
    return v;
  }

 private:
  void ws() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
      ++i_;
  }

  [[noreturn]] void fail(const char* msg) const {
    std::fprintf(stderr, "sidecar JSON error at byte %zu: %s\n", i_, msg);
    std::exit(2);
  }

  Json value() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end of input");
    const char c = s_[i_];
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Json v;
      v.kind = Json::String;
      v.str = string();
      return v;
    }
    if (c == 't' || c == 'f') {
      Json v;
      v.kind = Json::Bool;
      if (s_.compare(i_, 4, "true") == 0) {
        v.b = true;
        i_ += 4;
      } else if (s_.compare(i_, 5, "false") == 0) {
        v.b = false;
        i_ += 5;
      } else {
        fail("bad literal");
      }
      return v;
    }
    if (c == 'n') {
      if (s_.compare(i_, 4, "null") != 0) fail("bad literal");
      i_ += 4;
      return Json();
    }
    return number();
  }

  Json number() {
    const size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() &&
           ((s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '.' || s_[i_] == 'e' ||
            s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+'))
      ++i_;
    if (i_ == start) fail("bad number");
    Json v;
    v.kind = Json::Number;
    v.str = s_.substr(start, i_ - start);  // exact literal, for as_u64/as_i64
    v.num = std::strtod(v.str.c_str(), nullptr);
    return v;
  }

  std::string string() {
    if (s_[i_] != '"') fail("expected string");
    ++i_;
    std::string out;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated string");
      const char c = s_[i_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= s_.size()) fail("unterminated escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (i_ + 4 > s_.size()) fail("short \\u escape");
          const unsigned cp = static_cast<unsigned>(
              std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16));
          i_ += 4;
          // Encode as UTF-8 (the fixture only uses BMP code points).
          if (cp < 0x80) {
            out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  Json array() {
    Json v;
    v.kind = Json::Array;
    ++i_;  // '['
    ws();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return v;
    }
    while (true) {
      v.arr.push_back(value());
      ws();
      if (i_ >= s_.size()) fail("unterminated array");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        break;
      }
      fail("expected ',' or ']'");
    }
    return v;
  }

  Json object() {
    Json v;
    v.kind = Json::Object;
    ++i_;  // '{'
    ws();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return v;
    }
    while (true) {
      ws();
      std::string key = string();
      ws();
      if (i_ >= s_.size() || s_[i_] != ':') fail("expected ':'");
      ++i_;
      v.obj.emplace_back(std::move(key), value());
      ws();
      if (i_ >= s_.size()) fail("unterminated object");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        break;
      }
      fail("expected ',' or '}'");
    }
    return v;
  }

  const std::string& s_;
  size_t i_ = 0;
};

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open '%s'\n", path.c_str());
    std::exit(2);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---------------------------------------------------------------------------
// Expected-value comparison.
// ---------------------------------------------------------------------------
bool close_enough(float got, double want) {
  const double g = static_cast<double>(got);
  const double tol = 1e-5 * std::max(1.0, std::fabs(want));
  return std::fabs(g - want) <= tol;
}

void check_tensor_values(const yskn::gguf::File& file,
                         const yskn::gguf::TensorInfo& t,
                         const Json& expected) {
  const std::vector<float> got = file.dequant(t);
  check_eq(got.size(), static_cast<size_t>(t.n_elements),
           t.name + ": dequantized element count");

  const size_t n = std::min(got.size(), expected.arr.size());
  size_t bad = 0;
  double worst = 0.0;
  size_t worst_idx = 0;
  for (size_t i = 0; i < n; ++i) {
    const double want = expected.arr[i].as_num();
    if (!close_enough(got[i], want)) {
      if (bad < 4) {
        std::ostringstream os;
        os << t.name << "[" << i << "]: got " << got[i] << ", want " << want;
        std::printf("FAIL: %s\n", os.str().c_str());
        g_failure_log.push_back(os.str());
      }
      ++bad;
    }
    const double err = std::fabs(static_cast<double>(got[i]) - want);
    if (err > worst) {
      worst = err;
      worst_idx = i;
    }
  }
  ++g_checks;
  if (bad != 0) {
    ++g_failures;
    std::ostringstream os;
    os << t.name << ": " << bad << "/" << n
       << " values mismatched (worst |err| " << worst << " at index "
       << worst_idx << ")";
    g_failure_log.push_back(os.str());
  } else {
    std::printf("  ok   %-12s %-7s %6zu elems, max |err| %.3g\n",
                t.name.c_str(), yskn::gguf::type_name(t.type).c_str(), n, worst);
  }
}

// ---------------------------------------------------------------------------
// Metadata checks driven by the sidecar.
// ---------------------------------------------------------------------------
void check_metadata(const yskn::gguf::File& file, const Json& sidecar) {
  const yskn::gguf::Metadata& meta = file.meta();
  const Json& entries = sidecar.at("metadata");

  check_eq(meta.size(), static_cast<size_t>(sidecar.at("n_kv").as_u64()),
           "metadata entry count");

  for (const Json& e : entries.arr) {
    const std::string key = e.at("key").as_str();
    const uint32_t type = static_cast<uint32_t>(e.at("type").as_u64());
    const yskn::gguf::Value* v = meta.find(key);
    if (!v) {
      check(false, "metadata key present: " + key);
      continue;
    }
    check_eq(static_cast<uint32_t>(v->type), type, "metadata type of " + key);

    switch (type) {
      case 0: case 2: case 4: case 10:  // uint8/16/32/64
        check_eq(v->as_uint(), e.at("value").as_u64(), "metadata value of " + key);
        break;
      case 1: case 3: case 5: case 11:  // int8/16/32/64
        check_eq(v->as_int(), e.at("value").as_i64(), "metadata value of " + key);
        break;
      case 6: case 12:  // float32/64
        check(std::fabs(v->as_float() - e.at("value").as_num()) <= 1e-9,
              "metadata value of " + key);
        ++g_checks;
        break;
      case 7:  // bool
        check_eq(v->as_bool(), e.at("value").b, "metadata value of " + key);
        break;
      case 8:  // string
        check_eq(v->as_string(), e.at("value").as_str(), "metadata value of " + key);
        break;
      case 9: {  // array (possibly nested)
        const Json& items = e.at("items");
        const bool nested = e.at("nested").b;
        check_eq(static_cast<uint32_t>(v->elem_type),
                 static_cast<uint32_t>(e.at("elem_type").as_u64()),
                 "element type of " + key);
        check_eq(v->items.size(), items.arr.size(), "array length of " + key);
        if (nested) {
          for (size_t i = 0; i < v->items.size() && i < items.arr.size(); ++i) {
            const yskn::gguf::Value& inner = v->items[i];
            const Json& want = items.arr[i];
            check_eq(inner.items.size(), want.arr.size(),
                     "nested array length of " + key + "[" + std::to_string(i) + "]");
            for (size_t j = 0; j < inner.items.size() && j < want.arr.size(); ++j) {
              const yskn::gguf::Value& leaf = inner.items[j];
              const Json& w = want.arr[j];
              if (leaf.type == yskn::gguf::ValueType::STRING)
                check_eq(leaf.as_string(), w.as_str(),
                         "nested string of " + key);
              else
                check_eq(leaf.as_int(), w.as_i64(), "nested int of " + key);
            }
          }
        } else {
          for (size_t i = 0; i < v->items.size() && i < items.arr.size(); ++i) {
            const yskn::gguf::Value& leaf = v->items[i];
            const Json& w = items.arr[i];
            switch (static_cast<uint32_t>(v->elem_type)) {
              case 0: case 2: case 4: case 10:
                check_eq(leaf.as_uint(), w.as_u64(), "array item of " + key);
                break;
              case 1: case 3: case 5: case 11:
                check_eq(leaf.as_int(), w.as_i64(), "array item of " + key);
                break;
              case 6: case 12:
                check(std::fabs(leaf.as_float() - w.as_num()) <= 1e-9,
                      "array item of " + key);
                ++g_checks;
                break;
              case 7:
                check_eq(leaf.as_bool(), w.b, "array item of " + key);
                break;
              case 8:
                check_eq(leaf.as_string(), w.as_str(), "array item of " + key);
                break;
              default:
                check(false, "unexpected array element type for " + key);
            }
          }
        }
        break;
      }
      default:
        check(false, "unexpected sidecar metadata type for " + key);
    }
  }
}

// ---------------------------------------------------------------------------
// Negative tests: malformed files must throw std::runtime_error, not crash.
// ---------------------------------------------------------------------------
void write_bytes(const std::string& path, const std::string& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void expect_throw(const std::string& path, const std::string& what) {
  yskn::gguf::File f;
  bool threw = false;
  try {
    f.open(path);
  } catch (const std::runtime_error&) {
    threw = true;
  } catch (...) {
    threw = false;
  }
  check(threw, what);
}

void run_negative_tests(const std::string& out_dir) {
  // bad magic
  {
    std::string s = "NOPE";
    s.append(64, '\0');
    const std::string p = out_dir + "/bad_magic.gguf";
    write_bytes(p, s);
    expect_throw(p, "bad magic is rejected");
  }
  // unsupported version
  {
    std::string s = "GGUF";
    s += std::string("\x01\x00\x00\x00", 4);  // version 1
    s.append(64, '\0');
    const std::string p = out_dir + "/bad_version.gguf";
    write_bytes(p, s);
    expect_throw(p, "version 1 is rejected");
  }
  // truncated header
  {
    const std::string p = out_dir + "/truncated.gguf";
    write_bytes(p, "GGUF\x03\x00\x00\x00");
    expect_throw(p, "truncated header is rejected");
  }
  // empty file
  {
    const std::string p = out_dir + "/empty.gguf";
    write_bytes(p, "");
    expect_throw(p, "empty file is rejected");
  }
}

}  // namespace

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// Directory part of argv[0], so the test can find the fixture that the build
// dropped next to the executable even when the working directory differs.
std::string exe_dir(const char* argv0) {
  const std::string p = argv0 ? argv0 : "";
  const size_t slash = p.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

int main(int argc, char** argv) {
  // Defaults: <exe dir>/fixture/gguf_fixture.gguf, falling back to the plain
  // name in the current directory.
  const std::string dir = exe_dir(argc > 0 ? argv[0] : nullptr);
  std::string fixture = dir + "/fixture/gguf_fixture.gguf";
  if (!file_exists(fixture)) fixture = "gguf_fixture.gguf";
  std::string sidecar = fixture + ".json";
  std::string out_dir = dir;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fixture" && i + 1 < argc) {
      fixture = argv[++i];
    } else if (a == "--sidecar" && i + 1 < argc) {
      sidecar = argv[++i];
    } else if (a == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else {
      std::fprintf(stderr, "usage: %s [--fixture PATH] [--sidecar PATH] [--out-dir DIR]\n",
                   argv[0]);
      return 2;
    }
  }

  std::printf("GGUF reader test\n");
  std::printf("  fixture: %s\n", fixture.c_str());
  std::printf("  sidecar: %s\n\n", sidecar.c_str());

  const Json sidecar_json = JsonParser(read_file(sidecar)).parse();

  yskn::gguf::File file;
  try {
    file.open(fixture);
  } catch (const std::exception& e) {
    std::printf("FAIL: open(%s) threw: %s\n", fixture.c_str(), e.what());
    std::printf("\nFAIL 1 failure(s), 1 check(s)\n");
    return 1;
  }

  // ---- header -----------------------------------------------------------
  check_eq(file.version(), static_cast<uint32_t>(sidecar_json.at("gguf_version").as_u64()),
           "GGUF version");
  check_eq(file.alignment(), sidecar_json.at("alignment").as_u64(), "alignment");
  check_eq(file.data_base(), sidecar_json.at("data_base").as_u64(), "data section base");
  check_eq(file.tensors().size(),
           static_cast<size_t>(sidecar_json.at("n_tensors").as_u64()), "tensor count");
  check(file.file_size() > file.data_base(), "file has a data section");
  check_eq(file.meta().get_string("general.architecture", ""), std::string("llama"),
           "general.architecture");
  check_eq(file.meta().get_uint("general.alignment", 0),
           sidecar_json.at("alignment").as_u64(), "general.alignment via metadata");

  // ---- metadata ---------------------------------------------------------
  check_metadata(file, sidecar_json);

  // ---- tensor table -----------------------------------------------------
  const Json& tensors = sidecar_json.at("tensors");
  check_eq(file.tensors().size(), tensors.arr.size(), "tensor table size");
  for (size_t i = 0; i < tensors.arr.size() && i < file.tensors().size(); ++i) {
    const Json& want = tensors.arr[i];
    const yskn::gguf::TensorInfo& got = file.tensors()[i];
    const std::string name = want.at("name").as_str();
    check_eq(got.name, name, "tensor name #" + std::to_string(i));
    check_eq(got.type, static_cast<uint32_t>(want.at("type").as_u64()),
             "tensor type of " + name);
    check_eq(got.offset, want.at("offset").as_u64(), "tensor offset of " + name);
    check_eq(got.n_elements, want.at("n_elements").as_u64(),
             "tensor element count of " + name);
    check_eq(got.shape.size(), want.at("shape").arr.size(), "tensor rank of " + name);
    for (size_t d = 0; d < got.shape.size() && d < want.at("shape").arr.size(); ++d)
      check_eq(got.shape[d], want.at("shape").arr[d].as_u64(),
               "tensor dim " + std::to_string(d) + " of " + name);

    // find() must return the same object.
    const yskn::gguf::TensorInfo* found = file.find(name);
    check(found == &got, "find(\"" + name + "\") resolves to the same tensor");
    check_eq(got.nbytes(), file.tensor_nbytes(got), "nbytes of " + name);
  }
  check(file.find("no_such_tensor") == nullptr, "find() returns null for a missing tensor");

  // ---- dequantization ---------------------------------------------------
  std::printf("\nDequantization:\n");
  for (const Json& want : tensors.arr) {
    const std::string name = want.at("name").as_str();
    const yskn::gguf::TensorInfo* t = file.find(name);
    if (!t) {
      check(false, "tensor present for dequant: " + name);
      continue;
    }
    check_tensor_values(file, *t, want.at("expected"));
  }

  // Type helpers agree with the block sizes the fixture implies.
  check_eq(yskn::gguf::type_name(yskn::gguf::TYPE_Q4_K), std::string("Q4_K"),
           "type_name(Q4_K)");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q4_K), uint64_t{144},
           "block size of Q4_K");
  check_eq(yskn::gguf::type_block_elems(yskn::gguf::TYPE_Q4_K), uint64_t{256},
           "block elements of Q4_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q6_K), uint64_t{210},
           "block size of Q6_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q8_0), uint64_t{34},
           "block size of Q8_0");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q2_K), uint64_t{84},
           "block size of Q2_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q3_K), uint64_t{110},
           "block size of Q3_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q5_K), uint64_t{176},
           "block size of Q5_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_Q8_K), uint64_t{292},
           "block size of Q8_K");
  check_eq(yskn::gguf::type_block_size(yskn::gguf::TYPE_IQ4_NL), uint64_t{18},
           "block size of IQ4_NL");
  check_eq(yskn::gguf::type_block_elems(yskn::gguf::TYPE_F32), uint64_t{1},
           "block elements of F32");
  check(yskn::gguf::type_supported(yskn::gguf::TYPE_Q6_K), "Q6_K is supported");
  check(!yskn::gguf::type_supported(9999), "an unknown type is not supported");

  // ---- negative tests ---------------------------------------------------
  std::printf("\nNegative tests:\n");
  run_negative_tests(out_dir);
  std::printf("  ok   malformed inputs rejected\n");

  // ---- summary ----------------------------------------------------------
  std::printf("\n");
  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  for (const std::string& f : g_failure_log) std::printf("  - %s\n", f.c_str());
  return 1;
}
