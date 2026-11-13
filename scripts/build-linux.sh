#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# Build the YASAKANI engine library (libyskn.so) on Linux.
#
#   scripts/build-linux.sh                 # default architectures 120;86;80
#   CUDA_ARCH="120;90;86;80" scripts/build-linux.sh
#   CUDA_ARCH=native scripts/build-linux.sh   # ask the driver for this machine's GPU
#
# Environment:
#   CUDA_ARCH    CUDA architectures passed to -DCMAKE_CUDA_ARCHITECTURES (default "120;86;80")
#   BUILD_DIR    build directory (default "<repo>/build-linux")
#   JOBS         parallel jobs for ninja (default: all cores)
#   CUDA_HOME    toolkit root when nvcc is not on PATH (default /usr/local/cuda)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$root/build-linux}"
cuda_arch="${CUDA_ARCH:-120;86;80}"
jobs="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

die() { printf 'build-linux: %s\n' "$*" >&2; exit 1; }

# ---- nvcc: PATH, then $CUDA_HOME, then /usr/local/cuda ---------------------------------------------------------------
find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    local candidates=()
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME/bin/nvcc")
    candidates+=("/usr/local/cuda/bin/nvcc")
    local c
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { printf '%s\n' "$c"; return 0; }
    done
    return 1
}

nvcc="$(find_nvcc)" || die "nvcc not found. Install the CUDA toolkit, put it on PATH, or set CUDA_HOME (checked \$CUDA_HOME/bin and /usr/local/cuda/bin)."
cuda_bin="$(dirname "$nvcc")"
export PATH="$cuda_bin:$PATH"
export CUDACXX="$nvcc"

command -v cmake >/dev/null 2>&1 || die "cmake not found (need 3.24 or newer)."
command -v ninja >/dev/null 2>&1 || die "ninja not found (install the ninja-build package)."

printf 'nvcc      %s\n' "$("$nvcc" --version | tail -n 1 | sed 's/^ *//')"
printf 'cmake     %s\n' "$(cmake --version | head -n 1)"
printf 'ninja     %s\n' "$(ninja --version)"
printf 'arch      %s\n' "$cuda_arch"
printf 'build dir %s\n' "$build_dir"

# ---- configure + build -----------------------------------------------------------------------------------------------
# CMakeLists.txt lets -DYSKN_CUDA_ARCH win over -DCMAKE_CUDA_ARCHITECTURES.  When CUDA_ARCH is given explicitly we clear that
# cache entry so the requested list is the one that is built; otherwise an existing cache is left exactly as it is (incremental).
extra=()
if [[ -n "${CUDA_ARCH:-}" ]]; then
    extra+=(-DYSKN_CUDA_ARCH=)
fi

mkdir -p "$build_dir"
cmake -S "$root" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    "${extra[@]+"${extra[@]}"}"
# What the build files really ask for: CMakeLists.txt lets YSKN_CUDA_ARCH win, so the cache alone can mislead.  An explicit list
# becomes "arch=compute_N"; "native" becomes a bare "-arch=native".
effective="$(grep -oh -- 'arch=compute_[0-9a-z]*\|-arch=native' "$build_dir/build.ninja" 2>/dev/null |
             sed -e 's/.*compute_//' -e 's/^-arch=//' | sort -un | paste -sd';' - || true)"
printf 'compiling %s\n' "${effective:-unknown}"
cmake --build "$build_dir" --parallel "$jobs"

# ---- report ----------------------------------------------------------------------------------------------------------
so="$build_dir/libyskn.so"
[[ -f "$so" ]] || die "the build finished but $so is missing."
size="$(stat -c %s "$so" 2>/dev/null || stat -f %z "$so")"
printf '\nbuilt %s\n' "$so"
printf 'size  %s bytes (%s MiB)\n' "$size" "$((size / 1024 / 1024))"
printf '\nUse it with:  export YASAKANI_DLL=%s\n' "$so"
