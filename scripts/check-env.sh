#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# Print what the YASAKANI engine needs on this Linux machine: GPU, driver, CUDA toolkit, build tools, python.
# Nothing here is fatal: a missing piece is reported and the script still exits 0.
set -uo pipefail

have() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n== %s ==\n' "$1"; }

section "GPU (nvidia-smi)"
if have nvidia-smi; then
    if out="$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1)"; then
        printf '%s\n' "$out"
    else
        # A driver/library mismatch (the loaded kernel module and the user-space NVML differ) lands here.
        printf 'nvidia-smi could not query the GPUs:\n'
        printf '  %s\n' "$out" | sed '/^$/d' | head -n 5
        printf '  (an "NVML: Driver/library version mismatch" means the kernel module was updated but the user-space driver was not:\n'
        printf '   reboot, or reload the nvidia modules, to make the running driver and libnvidia-ml agree)\n'
    fi
else
    printf 'nvidia-smi not found (it is installed with the NVIDIA driver)\n'
fi

section "driver (/proc/driver/nvidia/version)"
if [[ -r /proc/driver/nvidia/version ]]; then
    head -n 2 /proc/driver/nvidia/version
else
    printf 'not readable: the NVIDIA kernel module does not appear to be loaded\n'
fi

section "CUDA toolkit (nvcc)"
nvcc=""
if have nvcc; then
    nvcc="$(command -v nvcc)"
else
    for c in "${CUDA_HOME:-}/bin/nvcc" /usr/local/cuda/bin/nvcc; do
        [[ -n "$c" && -x "$c" ]] && { nvcc="$c"; break; }
    done
fi
if [[ -n "$nvcc" ]]; then
    printf 'path  %s\n' "$nvcc"
    "$nvcc" --version | tail -n 2
else
    printf 'nvcc not found on PATH, in \$CUDA_HOME/bin or in /usr/local/cuda/bin\n'
    printf '(the engine links the CUDA runtime statically, so the toolkit is only needed to build it)\n'
fi

section "build tools"
for t in cmake ninja gcc g++ make; do
    if have "$t"; then
        case "$t" in
            cmake) printf '%-6s %s\n' "$t" "$(cmake --version 2>/dev/null | head -n 1 | sed 's/^cmake version //')" ;;
            ninja) printf '%-6s %s\n' "$t" "$(ninja --version 2>/dev/null)" ;;
            *)     printf '%-6s %s\n' "$t" "$("$t" --version 2>/dev/null | head -n 1)" ;;
        esac
    else
        printf '%-6s not found\n' "$t"
    fi
done

section "python"
if have python3; then
    printf 'python3 %s (%s)\n' "$(python3 --version 2>&1 | sed 's/^Python //')" "$(command -v python3)"
    python3 - <<'PY' 2>/dev/null || true
import importlib
for mod in ("numpy", "tokenizers", "jinja2"):
    try:
        m = importlib.import_module(mod)
        print("  %-10s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception:
        print("  %-10s MISSING" % mod)
PY
else
    printf 'python3 not found\n'
fi

section "engine library"
found=""
for p in "${YASAKANI_DLL:-}" "$(dirname "${BASH_SOURCE[0]}")/../build-linux/libyskn.so" /usr/local/lib/libyskn.so; do
    [[ -n "$p" && -f "$p" ]] && { found="$p"; break; }
done
if [[ -n "$found" ]]; then
    printf '%s (%s bytes)\n' "$found" "$(stat -c %s "$found" 2>/dev/null || echo '?')"
else
    printf 'libyskn.so not found (build it with scripts/build-linux.sh, or set YASAKANI_DLL)\n'
fi

printf '\n'
exit 0
