// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Host-only q35_g6 launch-shape policy: parsing the environment override and
// choosing the per-device default.  No CUDA call lives here, so tests/q35_g6_tab
// can build this header without a toolkit.
#ifndef YSKN_Q35_G6_TAB_H
#define YSKN_Q35_G6_TAB_H

namespace yskn {
namespace q35_g6_tab {

enum {
  kUnset = -1,    // YSKN_G6_TILEGRID is absent or empty: use the device default
  kInvalid = -2,  // the setting is present but not 0 or 1
};

inline bool is_blank(char c) { return c == ' ' || c == '\t'; }

// Parse YSKN_G6_TILEGRID: kUnset when absent or empty, 0 or 1 when valid,
// kInvalid when present but not an integer 0..1.  Surrounding blanks are
// allowed, as on the other YSKN_ switches.
inline int parse_tilegrid(const char* s) {
  if (!s || !*s) return kUnset;
  while (is_blank(*s)) ++s;
  if (!*s) return kUnset;
  if (*s < '0' || *s > '9') return kInvalid;
  int v = 0;
  while (*s >= '0' && *s <= '9') {
    v = v * 10 + (*s - '0');
    if (v > 1) return kInvalid;
    ++s;
  }
  while (is_blank(*s)) ++s;
  if (*s) return kInvalid;
  return v;
}

// The tile-grid shape launches one block per output tile, so a tile finishes its
// whole K loop in one block and never hands a split-K partial to a neighbour.
// Measured on an RTX 3090 (compute capability 8.6, 82 SMs): prefill time to
// first token fell from 54.6 s to 44.5 s at 41K prompt tokens and from 9.39 s to
// 7.51 s at 8K, with identical tokens.  Compute capability 8.0, 8.7 and 8.9 are
// unmeasured and keep the split-K grid until someone measures them.
inline bool default_tilegrid(int major, int minor) { return major == 8 && minor == 6; }

// Precedence: an explicit YSKN_G6_TILEGRID wins over the device default.
inline bool pick_tilegrid(int forced, int major, int minor) {
  return forced == 0 || forced == 1 ? forced != 0 : default_tilegrid(major, minor);
}

}  // namespace q35_g6_tab
}  // namespace yskn

#endif  // YSKN_Q35_G6_TAB_H
