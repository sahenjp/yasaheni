// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Host-only tests for src/q35_g6_tab.h: the YSKN_G6_TILEGRID parser, the 0..1
// validation of the forced setting, the per-device defaults, and precedence.
// No CUDA, no GPU and no engine.
#include "q35_g6_tab.h"

#include <cstdio>
#include <string>

namespace qt = yskn::q35_g6_tab;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void test_defaults() {
  check(qt::default_tilegrid(8, 6), "8.6 defaults to the tile grid");
  for (int major = 0; major <= 12; ++major)
    for (int minor = 0; minor <= 9; ++minor)
      if (!(major == 8 && minor == 6))
        check(!qt::default_tilegrid(major, minor),
              "default outside 8.6 keeps split-K at " + std::to_string(major) + "." + std::to_string(minor));
  for (int minor : {0, 7, 9})
    check(!qt::default_tilegrid(8, minor), "8." + std::to_string(minor) + " keeps split-K");
  check(!qt::default_tilegrid(-1, -1), "unknown compute capability keeps split-K");
}

void test_parse_valid() {
  check(qt::parse_tilegrid("0") == 0, "0 parses");
  check(qt::parse_tilegrid("1") == 1, "1 parses");
  check(qt::parse_tilegrid(" 1 ") == 1, "surrounding spaces");
  check(qt::parse_tilegrid("\t0\t") == 0, "surrounding tabs");
  check(qt::parse_tilegrid("01") == 1, "leading zero");
  check(qt::parse_tilegrid("") == qt::kUnset, "empty is unset");
  check(qt::parse_tilegrid("   ") == qt::kUnset, "blank is unset");
  check(qt::parse_tilegrid(nullptr) == qt::kUnset, "null is unset");
}

void test_parse_rejected() {
  for (const char* s : {"2", "10", "-1", "+1", "abc", "1x", "0 1", "1,0"})
    check(qt::parse_tilegrid(s) == qt::kInvalid, std::string("rejects \"") + s + "\"");
}

void test_precedence() {
  check(!qt::pick_tilegrid(0, 8, 6), "forced 0 beats the 8.6 default (split-K)");
  check(qt::pick_tilegrid(1, 12, 0), "forced 1 beats the other default (tile grid)");
  check(qt::pick_tilegrid(qt::kUnset, 8, 6), "unset keeps the 8.6 default");
  check(!qt::pick_tilegrid(qt::kUnset, 8, 0), "unset keeps split-K on 8.0");
  check(!qt::pick_tilegrid(qt::kInvalid, 8, 0), "invalid falls back to the device default");
  check(qt::pick_tilegrid(qt::kInvalid, 8, 6), "invalid keeps the 8.6 default");
}

}  // namespace

int main() {
  test_defaults();
  test_parse_valid();
  test_parse_rejected();
  test_precedence();
  std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
