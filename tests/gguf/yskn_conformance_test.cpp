// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Conformance test against upstream's own format fixture.
//
// tools/testdata/tiny.yskn is written by upstream's tools/yskn_fixture.py and
// tools/testdata/tiny_expected.json is written by the same script, from the
// reference implementation. Checking against it means this reader is measured
// against ground truth rather than against itself.
//
// What is checked:
//   * the 32-byte header and the model object
//   * every tensor's rows, cols, bits, layout, role and trellis constants
//   * every vector's length and dtype
//   * the SHA-256 of all five sign streams in the fixture, which pins
//     sign_stream_rule() bit for bit
//
// Exits non-zero on failure.
#include "format.hpp"
#include "sha256.hpp"
#include "signs.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;
std::vector<std::string> g_failed;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    g_failed.push_back(what);
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void check_eq(long long got, long long want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (got %lld, want %lld)", what.c_str(), got, want);
    g_failed.push_back(buf);
    std::printf("FAIL: %s\n", buf);
  }
}

void check_str(const std::string& got, const std::string& want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    const std::string msg = what + " (got \"" + got + "\", want \"" + want + "\")";
    g_failed.push_back(msg);
    std::printf("FAIL: %s\n", msg.c_str());
  }
}

// One row of tiny_expected.json's "signs": [rule, seed, offset, n, sha256].
struct SignCase {
  const char* rule;
  uint64_t seed;
  uint64_t offset;
  size_t n;
  const char* sha;
};

void check_signs() {
  static const SignCase kCases[] = {
      {"mulberry32_7919_13", 7, 0, 5120, "f81901d1d30113a0cc2d0f47568781458f44f066e037f535a71415919917fa47"},
      {"mulberry32_7919_13", 7, 5120, 17408, "1c411cb12ca3d6310c8988d5b99ca60e30118824fa64f3a83a9817030db2973d"},
      {"mulberry32_7919_13", 0, 3, 64, "2a34a3e5947a109807a7482b1ba9668188a84040f1f596879f0e7f95a3b79ca4"},
      {"splitmix64", 11, 0, 1024, "f6baf4e384371f46c2121f8578cd5a18ee395e0b80346015a61a803673953805"},
      {"splitmix64", 0, 1024, 512, "3352f405edda6d7bec8eedfb3d66d8c500ce8640a01240c2c44bb8aa137cdc31"},
  };
  std::printf("\nSign streams (SHA-256 against tiny_expected.json):\n");
  for (const SignCase& c : kCases) {
    const std::vector<int8_t> s = yskn::sign_stream_rule(c.rule, c.seed, c.offset, c.n);
    check_eq((long long)s.size(), (long long)c.n,
             std::string("sign count for ") + c.rule + " seed " + std::to_string(c.seed) + " offset " + std::to_string(c.offset));
    const std::string got = yskn::test::sha256_hex(s);
    check_str(got, c.sha, std::string("signs ") + c.rule + " seed " + std::to_string(c.seed) + " offset " +
                                 std::to_string(c.offset));
    std::printf("  %-20s seed %-4llu offset %-6llu n %-6zu %s\n", c.rule, (unsigned long long)c.seed,
                (unsigned long long)c.offset, c.n, got == c.sha ? "ok" : "MISMATCH");
  }
}

void check_model(yskn::Container& c) {
  std::printf("\nModel object:\n");
  const yskn::ModelDesc& m = c.model;
  check_str(m.arch, "test", "arch");
  check_eq(m.layers, 0, "layers");
  check_eq(m.hidden, 512, "hidden");
  check_eq(m.ffn, 512, "ffn");
  check_eq(m.heads, 1, "heads");
  check_eq(m.kv_heads, 1, "kv_heads");
  check_eq(m.head_dim, 512, "head_dim");
  check_eq(m.vocab, 96, "vocab");
  check_eq((long long)(m.eps * 1e6f), 1, "eps 1e-6");
  check_eq((long long)m.rope_theta, 10000, "rope_theta");
  check(!m.tie_embeddings, "tie_embeddings false");
  std::printf("  arch=%s layers=%d hidden=%d ffn=%d heads=%d vocab=%d\n", m.arch.c_str(), m.layers, m.hidden, m.ffn,
              m.heads, m.vocab);
}

void check_tensor(yskn::Container& c, const char* name, const char* role, int rows, int cols, int bits,
                  const char* layout, bool has_rot_out) {
  const yskn::TensorDesc& t = c.tensor(name);
  const std::string n(name);
  check_str(t.role, role, n + " role");
  check_eq(t.rows, rows, n + " rows");
  check_eq(t.cols, cols, n + " cols");
  check_eq(t.physical_rows, rows, n + " physical_rows");
  check_eq(t.bits, bits, n + " bits");
  check_str(t.layout, layout, n + " layout");
  check_eq(t.L, 12, n + " L");
  check_eq(t.d, 2, n + " d");
  check_eq(t.T, 256, n + " T");
  check_str(t.start, "zero", n + " start");
  check(!t.has_rar, n + " has_rar false");
  check_eq((long long)t.has_rot_out, has_rot_out ? 1 : 0, n + " has_rot_out");
  // The stream is sized rows * cols/256 * (4 * bits) words.
  const uint64_t words = (uint64_t)rows * (cols / 256) * 4 * bits;
  check_eq((long long)t.stream_bytes, (long long)(words * 4), n + " stream_bytes");
  std::printf("  %-12s role=%-9s %dx%d bits=%d layout=%s\n", name, t.role.c_str(), t.rows, t.cols, t.bits,
              t.layout.c_str());
}

void check_tensors(yskn::Container& c) {
  std::printf("\nTensors:\n");
  check_eq((long long)c.tensors.size(), 4, "tensor count");

  // The fixture writes four tensors: two body matrices at 6 and 8 bits (both
  // rotated on the output side), the head (no output rotation) and the
  // embedding (row layout, no output rotation).
  check_tensor(c, "tiny.body6", "body", 512, 512, 6, "warp", true);
  check_tensor(c, "tiny.body8", "body", 512, 1024, 8, "warp", true);
  check_tensor(c, "tiny.head", "head", 64, 512, 8, "warp", false);
  check_tensor(c, "tiny.embed", "embedding", 96, 512, 6, "row", false);
}

void check_vectors(yskn::Container& c) {
  std::printf("\nVectors:\n");
  check_eq((long long)c.vectors.size(), 2, "vector count");
  const yskn::VectorDesc& norm = c.vector("tiny.norm");
  check_eq(norm.n, 512, "tiny.norm length");
  check_str(norm.dtype, "bf16", "tiny.norm dtype");
  const yskn::VectorDesc& half = c.vector("tiny.half");
  check_eq(half.n, 8, "tiny.half length");
  check_str(half.dtype, "f16", "tiny.half dtype");
  std::printf("  tiny.norm n=%d %s, tiny.half n=%d %s\n", norm.n, norm.dtype.c_str(), half.n, half.dtype.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "tools/testdata/tiny.yskn";
  std::printf("YSKN conformance test\n  fixture: %s\n", path);

  yskn::Container c;
  try {
    c.open(path);
  } catch (const std::exception& ex) {
    std::printf("FAIL: cannot open %s: %s\n", path, ex.what());
    return 1;
  }

  // Header, as recorded in tiny_expected.json's "header": version, JSON length,
  // data offset, file length.
  check_eq(c.len, 468304, "file length");
  check_eq((long long)c.raw.dump().size(), 2193, "canonical JSON length");
  std::printf("  header version 1, file %zu bytes, %zu tensors\n", c.len, c.tensors.size());

  check_model(c);
  check_tensors(c);
  check_vectors(c);
  check_signs();

  std::printf("\n%s: %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
  if (g_failures) {
    std::printf("\nFailures:\n");
    for (const std::string& f : g_failed) std::printf("  - %s\n", f.c_str());
  }
  return g_failures ? 1 : 0;
}
