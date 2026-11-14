#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""trellis_search.py - the code search of the converter: the symbols whose decoded values are closest to the targets.

Two implementations of the same search, with the same result:

  NativeSearch      the CUDA kernel of native/trellis_search.cu, loaded as a shared library (build it with native/CMakeLists.txt).
                    This is what made the released models.  About 3 ms for 256 sequences.
  reference_search  the same operations in the same order with PyTorch tensors.  Needs nothing but PyTorch and runs on the CPU
                    too, but it is slow (about 0.2 s for 256 sequences on an RTX 5090).  It is the readable description of
                    the kernel and the way to check it: `python yskn_convert.py selftest` compares the two.

What is searched (docs/FORMAT.md, "From codes to values"): one sequence is 256 weights = 128 steps of 2 weights; a step stores
a K-bit symbol v (K = 6 or 8); the 12-bit state is s_j = ((s_(j-1) >> K) | (v_j << (12 - K))) & 4095 with s = 0 before the first
step, and the two weights of the step are table[s_j].  The search returns the symbols that minimise the sum over the steps of
|t_j - table[s_j]|^2 (the Viterbi algorithm).  With feedback, the target t_j of a step is the input z_j minus the errors of the
`taps` steps before it on the same path, each multiplied by a 2 x 2 matrix: the error feedback of a column-by-column rounding,
carried inside the search.
"""
import ctypes
import os
import sys
from pathlib import Path

import torch

T, STEPS, L = 256, 128, 12
FLT_MAX = 3.4028234663852886e38
HERE = Path(__file__).resolve().parent


class NativeSearch:
    """yskn_trellis_search of the shared library; every tensor stays on the GPU."""
    name = "native"

    def __init__(self, path):
        self.path = str(path)
        if os.name == "nt":                                    # the library needs the CUDA runtime it was built with
            for d in (os.environ.get("CUDA_PATH"),):
                if d and os.path.isdir(os.path.join(d, "bin")):
                    os.add_dll_directory(os.path.join(d, "bin"))
                    if os.path.isdir(os.path.join(d, "bin", "x64")):
                        os.add_dll_directory(os.path.join(d, "bin", "x64"))
        self.lib = ctypes.CDLL(self.path)
        f = self.lib.yskn_trellis_search
        f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        f.restype = ctypes.c_int
        self.f = f
        g = self.lib.yskn_trellis_scratch_bytes
        g.argtypes = [ctypes.c_int, ctypes.c_int]
        g.restype = ctypes.c_size_t
        self.scratch_bytes = g
        self.scratch = None

    def __call__(self, z, feedback, table, K, taps):
        """z cuda fp32 [B, 256]; feedback cuda fp32 [128 * taps * 4] or None; table cuda fp32 [4096, 2] -> cuda int16 [B, 128]"""
        assert z.is_cuda and z.dtype == torch.float32 and z.is_contiguous() and z.shape[1] == T
        assert table.is_cuda and table.dtype == torch.float32 and table.is_contiguous()
        B = z.shape[0]
        if feedback is None:
            taps = 0
        need = int(self.scratch_bytes(K, B))
        if self.scratch is None or self.scratch.numel() < need:
            self.scratch = torch.empty(need, dtype=torch.uint8, device="cuda")
        sym = torch.empty((B, STEPS), dtype=torch.int16, device="cuda")
        rc = self.f(K, taps, z.data_ptr(), feedback.data_ptr() if taps else None, table.data_ptr(), sym.data_ptr(),
                    self.scratch.data_ptr(), B, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError("yskn_trellis_search returned %d (K %d, taps %d)" % (rc, K, taps))
        return sym


def reference_search(z, feedback, table, K, taps):
    """The kernel of native/trellis_search.cu, operation by operation.  Same arguments and result as NativeSearch.__call__
    (any device).  The order of the additions and the way ties are broken follow the kernel, so the symbols are the same."""
    assert z.dtype == torch.float32 and z.shape[1] == T and K in (6, 8)
    if feedback is None:
        taps = 0
    B, dev = z.shape[0], z.device
    nstate, nlow, npred = 1 << L, 1 << (L - K), 1 << K
    spt = nstate // 1024                                        # the kernel: 1,024 threads, `slices` threads per class,
    sl0 = npred // spt                                          # each thread scans `ppt` predecessors
    slices = 32 if sl0 > 32 else max(1, sl0)
    ppt = npred // slices
    zz = z.reshape(B, STEPS, 2)
    tab_new = table.reshape(npred, nlow, 2)                     # a new state is v * nlow + low  (low = its class)
    low_idx = torch.arange(nlow, device=dev)
    scan_idx = torch.arange(ppt, device=dev)
    slice_base = (torch.arange(slices, device=dev) * ppt).view(1, 1, slices)
    state_idx = torch.arange(nstate, device=dev)
    big = torch.tensor(FLT_MAX, dtype=torch.float32, device=dev)
    M = torch.full((B, nstate), FLT_MAX, dtype=torch.float32, device=dev)
    M[:, 0] = 0.0                                               # every sequence starts in state 0
    zero2 = torch.zeros((B, nlow, 2), dtype=torch.float32, device=dev)
    err = torch.zeros((B, nlow, taps, 2), dtype=torch.float32, device=dev) if taps else None
    t_prev = zero2
    back = torch.empty((STEPS, B, nlow), dtype=torch.int64, device=dev)
    fb = feedback.reshape(STEPS, taps, 4) if taps else None
    for j in range(STEPS):
        # the best predecessor of every class: the predecessors of class `low` are the states low * npred + q
        v = M.view(B, nlow, slices, ppt)
        m = v.min(dim=3).values
        q = torch.where(v == m.unsqueeze(3), scan_idx, ppt).min(dim=3).values      # the first minimum inside a thread
        q = torch.where(m < big, q + slice_base, 0)                                # nothing below FLT_MAX: q stays 0
        n = slices
        while n > 1:                                                               # threads merged in halves, the lower wins a tie
            h = n // 2
            take = m[..., h:n] < m[..., :h]
            m = torch.where(take, m[..., h:n], m[..., :h])
            q = torch.where(take, q[..., h:n], q[..., :h])
            n = h
        best, bq = m[..., 0], q[..., 0]
        if taps:                                                                   # the class inherits the errors of that path
            s_star = (low_idx << K) | bq
            low_prev = s_star & (nlow - 1)
            if j > 0:
                e = t_prev.gather(1, low_prev.unsqueeze(2).expand(B, nlow, 2)) - table[s_star]
                e = torch.where((best < big).unsqueeze(2), e, zero2)
            else:
                e = zero2
            old = err.gather(1, low_prev.view(B, nlow, 1, 1).expand(B, nlow, taps, 2))
            err = torch.cat([e.unsqueeze(2), old[:, :, :taps - 1]], dim=2)
        t = zz[:, j].unsqueeze(1).expand(B, nlow, 2)                               # the target of the step for every class
        if taps:
            f = fb[j]                                                              # [taps, 4]: 2 x 2 matrices, row-major
            sx = err[..., 0] * f[:, 0] + err[..., 1] * f[:, 2]
            sy = err[..., 0] * f[:, 1] + err[..., 1] * f[:, 3]
            s2 = torch.stack((sx, sy), dim=3)
            for i in range(min(j, taps)):                                          # one subtraction after the other, as the kernel
                t = t - s2[:, :, i]
        t_prev = t
        back[j] = bq
        d = t.unsqueeze(1) - tab_new                                               # [B, npred, nlow, 2]
        Mn = (best.unsqueeze(1) + d[..., 0] * d[..., 0]) + d[..., 1] * d[..., 1]
        M = torch.where((best == big).unsqueeze(1), big, Mn).reshape(B, nstate)
    mm = M.min(dim=1).values                                                       # the best end state, the first one on a tie
    s = torch.where(M == mm.unsqueeze(1), state_idx, nstate).min(dim=1).values
    sym = torch.empty((B, STEPS), dtype=torch.int16, device=dev)
    for j in range(STEPS - 1, -1, -1):
        sym[:, j] = (s >> (L - K)).to(torch.int16)
        low = s & (nlow - 1)
        s = (low << K) | back[j].gather(1, low.unsqueeze(1))[:, 0]
    return sym


reference_search.name = "reference (PyTorch)"


def default_library():
    """Where the build of native/CMakeLists.txt puts the library, seen from this file."""
    names = ("yskn_trellis.dll",) if os.name == "nt" else ("libyskn_trellis.so",)
    roots = [os.environ.get("YSKN_TRELLIS_LIB"), HERE / "native" / "build" / "bin", HERE / "native" / "build",
             HERE.parents[1] / "build_trellis" / "bin", HERE.parents[1] / "build_trellis"]
    for r in roots:
        if not r:
            continue
        r = Path(r)
        if r.is_file():
            return r
        for n in names:
            if (r / n).is_file():
                return r / n
    return None


def load(native=None, allow_reference=True, say=print):
    """The search to use: the native library when it is found (or given), otherwise the PyTorch reference."""
    path = Path(native) if native else default_library()
    if path is not None and Path(path).is_file():
        if not torch.cuda.is_available():
            raise SystemExit("the native search needs a CUDA GPU")
        return NativeSearch(path)
    if native:
        raise SystemExit("no such library: %s" % native)
    if not allow_reference:
        raise SystemExit("the native search library was not found; build tools/convert/native (see its CMakeLists.txt) "
                         "or pass --native PATH, or allow the slow PyTorch search with --reference-search")
    say("note: the native search library was not found; using the PyTorch reference search (slow)")
    return reference_search


if __name__ == "__main__":
    sys.exit("this file is a module of yskn_convert.py; run `python yskn_convert.py selftest`")
