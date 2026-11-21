// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Host-only LNORD launch-table policy: parsing the environment overrides,
// choosing the per-device default, and applying the precedence rules.  No CUDA
// call lives here, so tests/lnord_tab can build this header without a toolkit.
#ifndef YSKN_LNORD_TAB_H
#define YSKN_LNORD_TAB_H

namespace yskn {
namespace lnord_tab {

enum {
  kUnset = -1,    // the setting is absent or empty: use the device default
  kInvalid = -2,  // the setting is present but not an integer in 0..2
};

inline bool is_blank(char c) { return c == ' ' || c == '\t'; }

// Parse "0,1,1,1,1,1,1,1,1": one to nine comma-separated integers in 0..2,
// spaces and tabs allowed around each entry.  An empty entry, a trailing
// comma, more than nine entries, a non-digit or a value outside 0..2 rejects
// the whole value.  Returns the number of entries, or -1.
inline int parse_tab(const char* s, int out[9]) {
  if (!s || !*s) return -1;
  int n = 0;
  for (;;) {
    if (n >= 9) return -1;
    while (is_blank(*s)) ++s;
    if (*s < '0' || *s > '9') return -1;
    int v = 0;
    while (*s >= '0' && *s <= '9') {
      v = v * 10 + (*s - '0');
      if (v > 2) return -1;
      ++s;
    }
    out[n++] = v;
    while (is_blank(*s)) ++s;
    if (!*s) return n;
    if (*s != ',') return -1;
    ++s;
  }
}

// Parse YSKN_LNO_CFG / YSKN_BDR_CFG: kUnset when absent or empty, 0..2 when
// valid, kInvalid when present but not an integer in 0..2.
inline int parse_cfg(const char* s) {
  if (!s || !*s) return kUnset;
  while (is_blank(*s)) ++s;
  if (*s < '0' || *s > '9') return kInvalid;
  int v = 0;
  while (*s >= '0' && *s <= '9') {
    v = v * 10 + (*s - '0');
    if (v > 2) return kInvalid;
    ++s;
  }
  while (is_blank(*s)) ++s;
  if (*s) return kInvalid;
  return v;
}

// The per-M default table.  {0,1,1,1,1,1,1,1,1} is the result measured on an
// RTX 3090 (compute capability 8.6, 82 SMs); the other 8.x parts (8.0, 8.7,
// 8.9) are unmeasured, so they keep the 12.x table and opt in with
// YSKN_LNO_CFG_TAB.
inline void device_table(int major, int minor, int out[9]) {
  static const int k86[9] = {0, 1, 1, 1, 1, 1, 1, 1, 1};
  static const int kOther[9] = {0, 1, 2, 1, 2, 2, 2, 2, 2};
  const int* src = (major == 8 && minor == 6) ? k86 : kOther;
  for (int i = 0; i < 9; ++i) out[i] = src[i];
}

// The device default with the leading n entries replaced by an override.
inline void apply_override(int major, int minor, const int* over, int n, int out[9]) {
  device_table(major, minor, out);
  for (int i = 0; i < n && i < 9; ++i) out[i] = over[i];
}

// Precedence: a forced configuration (YSKN_LNO_CFG / YSKN_BDR_CFG) wins over
// the per-M table; M above 8 keeps cfg 0.
inline int pick_cfg(int M, int forced, const int tab[9]) {
  if (forced >= 0 && forced <= 2) return forced;
  return (M >= 0 && M <= 8) ? tab[M] : 0;
}

}  // namespace lnord_tab
}  // namespace yskn

#endif  // YSKN_LNORD_TAB_H
