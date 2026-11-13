// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Portable shims for the small set of MSVC-only CRT entry points the engine
// used. Header-only; on Windows these expand to the original calls.
#pragma once

#include <cstdio>
#include <cstdint>

#if defined(_WIN32)
#  define YSKN_FTELL64(f) _ftelli64(f)
#  define YSKN_FSEEK64(f, off, whence) _fseeki64((f), (off), (whence))
#else
#  include <sys/types.h>
#  define YSKN_FTELL64(f) ftello(f)
#  define YSKN_FSEEK64(f, off, whence) fseeko((f), static_cast<off_t>(off), (whence))
#endif
