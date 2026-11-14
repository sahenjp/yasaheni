// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Host-only tests for src/lnord_tab.h: the YSKN_LNO_CFG_TAB parser, the 0..2
// validation of the forced settings, the per-device defaults, and precedence.
// No CUDA, no GPU and no engine.
#include "lnord_tab.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace lt = yskn::lnord_tab;

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

std::string str(const int* v, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) { if (i) s += ','; s += std::to_string(v[i]); }
  return s;
}

void check_tab(const int* got, const int* want, int n, const std::string& what) {
  check(std::memcmp(got, want, sizeof(int) * (size_t)n) == 0,
        what + " (got " + str(got, n) + ", want " + str(want, n) + ")");
}

void test_defaults() {
  int out[9];
  const int k86[9] = {0, 1, 1, 1, 1, 1, 1, 1, 1};
  const int kOther[9] = {0, 1, 2, 1, 2, 2, 2, 2, 2};
  lt::device_table(8, 6, out);
  check_tab(out, k86, 9, "8.6 default");
  for (int major = 0; major <= 12; ++major)
    for (int minor = 0; minor <= 9; ++minor)
      if (!(major == 8 && minor == 6)) {
        lt::device_table(major, minor, out);
        check_tab(out, kOther, 9, "default outside 8.6 is the 12.x table at " +
                                      std::to_string(major) + "." + std::to_string(minor));
      }
  for (int minor : {0, 7, 9}) {
    lt::device_table(8, minor, out);
    check_tab(out, kOther, 9, "8." + std::to_string(minor) + " default stays {0,1,2,...}");
  }
  lt::device_table(-1, -1, out);
  check_tab(out, kOther, 9, "unknown compute capability uses the 12.x table");
}

void test_parse_valid() {
  int out[9];
  check(lt::parse_tab("0,1,2,1,2,2,2,2,2", out) == 9, "full table parses");
  const int full[9] = {0, 1, 2, 1, 2, 2, 2, 2, 2};
  check_tab(out, full, 9, "full table values");

  check(lt::parse_tab("1,2", out) == 2, "partial override parses");
  const int part[2] = {1, 2};
  check_tab(out, part, 2, "partial override values");

  check(lt::parse_tab(" 0 ,\t1 , 2 ", out) == 3, "spaces and tabs around entries");
  const int spaced[3] = {0, 1, 2};
  check_tab(out, spaced, 3, "spaces and tabs values");

  check(lt::parse_tab("0", out) == 1 && out[0] == 0, "lower bound 0");
  check(lt::parse_tab("2", out) == 1 && out[0] == 2, "upper bound 2");
  check(lt::parse_tab("01", out) == 1 && out[0] == 1, "leading zero");
}

void test_parse_rejected() {
  int out[9];
  const char* bad[] = {
      "", " ", ",", "0,,1", "0,1,", ",0", "0,1,2,3,4,5,6,7,8,9", "0,1,2,3,4,5,6,7,8,9,0",
      "0,a", "a", "1x", "0 1", "-1", "+1", "3", "10", "0\n,1",
  };
  for (const char* s : bad) check(lt::parse_tab(s, out) == -1, std::string("rejects \"") + s + "\"");
  check(lt::parse_tab(nullptr, out) == -1, "rejects null");
}

void test_parse_cfg() {
  check(lt::parse_cfg(nullptr) == lt::kUnset, "cfg null is unset");
  check(lt::parse_cfg("") == lt::kUnset, "cfg empty is unset");
  check(lt::parse_cfg("0") == 0, "cfg 0");
  check(lt::parse_cfg("1") == 1, "cfg 1");
  check(lt::parse_cfg("2") == 2, "cfg 2");
  check(lt::parse_cfg(" 1 ") == 1, "cfg surrounding spaces");
  check(lt::parse_cfg("01") == 1, "cfg leading zero");
  for (const char* s : {"3", "-1", "-2", "abc", "1x", " "})
    check(lt::parse_cfg(s) == lt::kInvalid, std::string("cfg rejects \"") + s + "\"");
}

void test_apply_override() {
  int out[9];
  const int over[2] = {1, 2};
  lt::apply_override(12, 0, over, 2, out);
  const int got12[9] = {1, 2, 2, 1, 2, 2, 2, 2, 2};
  check_tab(out, got12, 9, "partial override on 12.x");

  lt::apply_override(8, 6, over, 2, out);
  const int got86[9] = {1, 2, 1, 1, 1, 1, 1, 1, 1};
  check_tab(out, got86, 9, "partial override on 8.6");

  lt::apply_override(8, 0, over, 0, out);
  const int base[9] = {0, 1, 2, 1, 2, 2, 2, 2, 2};
  check_tab(out, base, 9, "no override keeps the device default");

  const int full[9] = {0, 1, 1, 1, 1, 1, 1, 1, 1};
  lt::apply_override(12, 0, full, 9, out);
  check_tab(out, full, 9, "full override replaces every entry");
}

void test_precedence() {
  const int tab[9] = {0, 1, 1, 1, 1, 1, 1, 1, 1};
  check(lt::pick_cfg(4, 2, tab) == 2, "forced cfg wins at M=4");
  check(lt::pick_cfg(0, 0, tab) == 0, "forced cfg wins at M=0");
  check(lt::pick_cfg(4, -1, tab) == 1, "per-M table when cfg is unset");
  check(lt::pick_cfg(8, -1, tab) == 1, "M=8 uses the table");
  check(lt::pick_cfg(9, -1, tab) == 0, "M>8 uses cfg 0");
  check(lt::pick_cfg(4, lt::kInvalid, tab) == 1, "invalid cfg falls back to the table");

  int out[9];
  const int over[1] = {2};
  lt::apply_override(8, 6, over, 1, out);
  check(lt::pick_cfg(0, -1, out) == 2, "YSKN_LNO_CFG_TAB beats the device default");
  check(lt::pick_cfg(0, 1, out) == 1, "forced cfg beats YSKN_LNO_CFG_TAB");
}

}  // namespace

int main() {
  test_defaults();
  test_parse_valid();
  test_parse_rejected();
  test_parse_cfg();
  test_apply_override();
  test_precedence();
  std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
