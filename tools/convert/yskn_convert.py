#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""yskn_convert.py - make a YASAKANI model file (.yskn) from the original checkpoint of Qwen3.8-27B.

  python yskn_convert.py selftest
        no model needed: the search kernel against its PyTorch reference, one random matrix encoded, written and read back

  python yskn_convert.py convert  --checkpoint DIR --work DIR [--bits both|6|8|FILE.json] [--parts body,embedding,head]
        the GPU part: statistics of the original model on the calibration windows, then the codes of every matrix
        (DIR of --checkpoint holds config.json, model.safetensors.index.json and the *.safetensors files)

  python yskn_convert.py assemble --checkpoint DIR --work DIR --bits FILE.json|6|8 --out MODEL.yskn
                                  [--head-from OTHER.yskn] [--embedding-from OTHER.yskn]
        the CPU part: one .yskn file from the codes, with the chosen number of bits for every body matrix

  python yskn_convert.py compare  A.yskn B.yskn [--json OUT]
        two files tensor by tensor: equal symbols, row scales, value tables, vectors

What the conversion does is described in README.md of this folder and in docs/TECHNOLOGY.md; the file it writes in docs/FORMAT.md.
In short, for every matrix W of the model:
  1. statistics.  The original model runs on 192 windows of 2,048 tokens.  For the input x and the output gradient g of the
     matrix (labels drawn from the model itself) it keeps  H_in0 = mean x x^T,  H_in = mean |g|^2 x x^T,  H_out = mean w g g^T.
  2. coordinates.  W and the statistics are rotated (blocks of 512, Hadamard with signs), as the file format stores them.
  3. rounding.  Rows are taken in blocks from the last row block upwards; each block is rounded column block by column block
     (256 columns = one sequence of the trellis code), the search of trellis_search.py choosing the codes.  The error of a column
     block is pushed onto the columns that follow (input side, H_in), the error of a row block onto the rows above (output
     side, H_out).
The output matrix is rounded with the input side only, the embedding table with neither (plain squared error).

Needs Python 3.10+, numpy, PyTorch with CUDA and transformers 5.5 (the model classes of Qwen3.8).  The released models were made
with Python 3.14.4, numpy 2.4.4, torch 2.10.0+cu128, transformers 5.5.0 on one RTX 5090 (32 GB) with 93 GB of RAM; the work folder
needs about 65 GB (36 GB of checkpoints, 12 GB of codes for each number of bits) and the run about 40 GB of RAM.
"""
import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SIGN_SEED = 7            # signs of the rotations (the "sign_seed" of the file)
TAPS = 8                 # steps of error feedback inside the search
DAMP_IN = 0.03           # added to the diagonal of the input-side statistic, times its mean
DAMP_OUT = 1.0           # the same for the output-side statistic
SEGMENT = 8              # layers between two checkpoints of the calibration windows
LABEL_SEED = 2026        # the labels of the gradients are drawn from the model with this seed
WINDOW = 2048
T, STEPS, STATE_BITS = 256, 128, 12
HEADER_RESERVE = 8 << 20
ALIGN = 256
SCALE512 = np.float32(0.04419417382415922)          # 1 / sqrt(512)

LINEAR_GROUPS = [["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], ["linear_attn.out_proj"],
                 ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]
FULL_GROUPS = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
               ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]
LINEAR_ORDER = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
FULL_ORDER = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP_ORDER = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
# the small tensors of a layer, stored as they are (bf16): name in the checkpoint -> name in the file
SMALL_LINEAR = ["input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
                "linear_attn.norm.weight"]
SMALL_FULL = ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]
SMALL_NAME = {"input_layernorm.weight": "input_layernorm", "post_attention_layernorm.weight": "post_attention_layernorm",
              "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a", "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b",
              "linear_attn.conv1d.weight": "linear_attn.conv1d", "linear_attn.A_log": "linear_attn.A_log",
              "linear_attn.dt_bias": "linear_attn.dt_bias", "linear_attn.norm.weight": "linear_attn.norm",
              "self_attn.q_norm.weight": "self_attn.q_norm", "self_attn.k_norm.weight": "self_attn.k_norm"}
PREFIX = "model.language_model."

torch = None             # imported by need_torch(): `compare` works without PyTorch


def need_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def sha256_file(path, offset=0, size=None, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        left = os.path.getsize(path) - offset if size is None else size
        while left > 0:
            b = f.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------- signs and rotation
def sign_vectors(n_in, m_out, seed=SIGN_SEED):
    """The signs of the two rotations of one matrix (float32, +1 / -1): one stream, the first n_in values for the input side,
    the next m_out for the output side (docs/FORMAT.md, rule "mulberry32_7919_13")."""
    s0 = ((int(seed) & 0xFFFFFFFF) * 7919 + 13) & 0xFFFFFFFF
    if s0 == 0:
        s0 = 0x9E3779B9
    k = np.arange(1, n_in + m_out + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        t = (np.uint32(s0) + k.astype(np.uint32) * np.uint32(0x6D2B79F5)).astype(np.uint32)
        t = ((t ^ (t >> np.uint32(15))) * (t | np.uint32(1))).astype(np.uint32)
        t = (t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))).astype(np.uint32)
        t = t ^ (t >> np.uint32(14))
    out = np.where(t < np.uint32(0x80000000), -1.0, 1.0).astype(np.float32)
    return out[:n_in].copy(), out[n_in:].copy()


def fwht512_rows(x, s, inverse=False):
    """x: float32 [..., n] with n a multiple of 512; s: float32 [n] signs.  Forward: (x * s) through the Hadamard butterflies of
    every block of 512, times 1 / sqrt(512).  Inverse: butterflies, factor, then * s."""
    assert x.dtype == torch.float32 and x.shape[-1] % 512 == 0
    shp = x.shape
    s = s.to(x.device, torch.float32)
    v = x if inverse else x * s
    v = v.reshape(-1, shp[-1] // 512, 512).clone()
    h = 1
    while h < 512:
        v = v.view(-1, shp[-1] // 512, 512 // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        v = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    v = v.reshape(shp) * torch.tensor(SCALE512, device=x.device)
    return v * s if inverse else v


def rotate_weight(w, si, so):
    """The stored matrix M of W = D_out Hn M Hn D_in: the rows of W through the input rotation, then its columns through the
    output rotation."""
    a = fwht512_rows(w.float().contiguous(), si)
    return fwht512_rows(a.t().contiguous(), so).t().contiguous()


def inverse_rotate(q, si, so):
    a = fwht512_rows(q.t().contiguous(), so, inverse=True).t().contiguous()
    return fwht512_rows(a, si, inverse=True)


def rotate_stat(h, s):
    """A statistic (symmetric) into the stored coordinates of its side."""
    h = fwht512_rows(h.float().contiguous(), s)
    h = fwht512_rows(h.t().contiguous(), s).t().contiguous()
    return (h + h.t()) * 0.5


# ------------------------------------------------------------------------------------------------------------ the checkpoint
class Checkpoint:
    """The tensors of the original model, read with plain file reads (the safetensors header is parsed here)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        index = self.folder / "model.safetensors.index.json"
        if not index.is_file():
            raise SystemExit("%s: no model.safetensors.index.json (give the folder of the original checkpoint)" % folder)
        self.index = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        self.config = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))["text_config"]
        self.headers = {}

    def _header(self, fn):
        if fn not in self.headers:
            with open(self.folder / fn, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.headers[fn] = (8 + n, json.loads(f.read(n)))
        return self.headers[fn]

    def get(self, name):
        dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        fn = self.index[name]
        base, hdr = self._header(fn)
        e = hdr[name]
        b0, b1 = e["data_offsets"]
        with open(self.folder / fn, "rb") as f:
            f.seek(base + b0)
            buf = bytearray(f.read(b1 - b0))
        return torch.frombuffer(buf, dtype=dt[e["dtype"]]).reshape(e["shape"]).clone()

    def layer_state(self, i):
        p = "%slayers.%d." % (PREFIX, i)
        return {k[len(p):]: self.get(k) for k in self.index if k.startswith(p)}

    def source_name(self):
        """<organisation>/<model>@<revision> when the folder is a snapshot of the Hugging Face cache, else the folder name."""
        parts = self.folder.resolve().parts
        if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
            return parts[-3][len("models--"):].replace("--", "/") + "@" + parts[-1]
        return self.folder.name


class Layers:
    """One decoder layer of the original model at a time (the classes of transformers), on the GPU."""

    def __init__(self, ckpt):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        self.config = Qwen3_5TextConfig(**ckpt.config)
        self.config._attn_implementation = "sdpa"
        self.rotary = Qwen3_5TextRotaryEmbedding(self.config).cuda()

    def layer(self, state, i):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(self.config, i)
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.cuda().eval()

    def forward(self, layer, hidden):
        from transformers.masking_utils import create_causal_mask
        batch, length = hidden.shape[:2]
        pos = torch.arange(length, device=hidden.device)[None].expand(batch, -1)
        rotary = self.rotary(hidden, pos[None].expand(3, -1, -1))
        mask = None
        if layer.layer_type == "full_attention":
            mask = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
                                      position_ids=pos)
        r = layer(hidden, position_embeddings=rotary, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False)
        return r[0] if isinstance(r, tuple) else r

    def groups(self, i):
        return LINEAR_GROUPS if self.config.layer_types[i] == "linear_attention" else FULL_GROUPS

    def final_norm_and_head(self, ckpt):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        hidden = self.config.hidden_size
        norm = Qwen3_5RMSNorm(hidden, eps=self.config.rms_norm_eps)
        norm.load_state_dict({"weight": ckpt.get(PREFIX + "norm.weight")}, assign=True)
        norm = norm.cuda().eval()
        norm.requires_grad_(False)
        return norm, ckpt.get("lm_head.weight").cuda().float()


# ---------------------------------------------------------------------------------------------------------------- statistics
def head_gradients(ckpt, st, final):
    """For every window: the gradient of the loss with respect to the last hidden state, the loss being the cross-entropy to
    labels drawn from the model itself (one label per position, seed LABEL_SEED).  Also returns the statistic of the output of
    the final norm (mean x x^T), which the output matrix is rounded with."""
    norm, Wh = st.final_norm_and_head(ckpt)
    hidden = Wh.shape[1]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(LABEL_SEED)
    G = []
    stat = torch.zeros((hidden, hidden), device="cuda", dtype=torch.float32)
    for w in range(final.shape[0]):
        hh = final[w:w + 1].clone().cuda().requires_grad_(True)
        with torch.enable_grad():
            hn = norm(hh)
        with torch.no_grad():
            flat = hn.detach().reshape(-1, hidden).float()
            dh = torch.empty_like(flat)
            for c0 in range(0, flat.shape[0], 256):
                p = torch.softmax(flat[c0:c0 + 256] @ Wh.t(), -1)
                t = torch.multinomial(p, 1, generator=gen)[:, 0]
                p[torch.arange(len(t), device=p.device), t] -= 1.0
                dh[c0:c0 + 256] = p @ Wh
                del p
            stat.addmm_(flat.t(), flat)
        hn.backward(dh.reshape(hn.shape).to(hn.dtype))
        G.append(hh.grad.to(torch.bfloat16).cpu())
        del hh, hn
    stat /= final.shape[0] * final.shape[1]
    stat = stat.cpu()
    del norm, Wh
    gc.collect()
    torch.cuda.empty_cache()
    return G, stat


def layer_stats(st, i, layer, Hin, G):
    """Hin: the inputs of layer i (CPU bf16 [1, 2048, hidden] per window); G: the gradients at its output (same shape).
    Returns the statistics (H_in0 per group of matrices that share an input, H_in and H_out per matrix) and the gradients at the
    input of the layer, which are the G of the layer below."""
    groups = st.groups(i)
    mods = {n: layer.get_submodule(n) for g in groups for n in g}
    grp = {n: g[0] for g in groups for n in g}
    HI0 = {g[0]: torch.zeros((mods[g[0]].in_features,) * 2, device="cuda") for g in groups}
    ntok = 0

    def pre(name):
        def f(m, args):
            x = args[0].reshape(-1, m.in_features).float()
            HI0[name].addmm_(x.t(), x)
        return f

    hs = [mods[g[0]].register_forward_pre_hook(pre(g[0])) for g in groups]
    with torch.no_grad():                                   # pass 1: H_in0 = mean x x^T, two windows at a time
        for w in range(0, len(Hin), 2):
            st.forward(layer, torch.cat(Hin[w:w + 2]).cuda())
            ntok += 2 * WINDOW
    for x in hs:
        x.remove()
    for k in HI0:
        HI0[k] /= ntok
    HO1 = {n: torch.zeros((mods[n].out_features,) * 2, device="cuda") for n in mods}
    HI1 = {n: torch.zeros((mods[n].in_features,) * 2, device="cuda") for n in mods}
    xs, wx = {}, {}

    def fh(name):
        def f(m, args, outp):
            x = args[0].detach().reshape(-1, m.in_features).float()
            xs[name] = x
            w = ((x @ HI0[grp[name]]) * x).sum(1)           # one weight per token: x H_in0 x^T over its median, at most 8
            wx[name] = torch.clamp(w / w.median().clamp(min=1e-30), max=8.0)
        return f

    def bh(name):
        def f(m, gin, gout):
            g = gout[0].detach().reshape(-1, m.out_features).float()
            x = xs.pop(name)
            HO1[name].addmm_(g.t(), g * wx.pop(name)[:, None])
            HI1[name].addmm_(x.t(), x * (g * g).sum(1)[:, None])
        return f

    hs = [mods[n].register_forward_hook(fh(n)) for n in mods] + [mods[n].register_full_backward_hook(bh(n)) for n in mods]
    Gin = []
    for w in range(len(Hin)):                               # pass 2: with gradients, one window at a time
        x = Hin[w].cuda().requires_grad_(True)
        with torch.enable_grad():
            y = st.forward(layer, x)
        y.backward(G[w].cuda().to(y.dtype))
        Gin.append(x.grad.to(torch.bfloat16).cpu())
        del x, y
    for x in hs:
        x.remove()
    n_all = len(Hin) * WINDOW
    for n in mods:
        HO1[n] /= n_all
        HI1[n] /= n_all
    return dict(HI0=HI0, HI1=HI1, HO1=HO1, grp=grp), Gin


# ------------------------------------------------------------------------------------------------------------------ rounding
def feedback_tables(Cb, taps):
    """The 2 x 2 feedback matrices of one block of 256 columns, from the factor C of that block:
    F[j][i-1][rr][cc] = C[2a+rr, 2j+cc] / C[2a+rr, 2a+rr] with a = j - i >= 0 (zero otherwise)."""
    Cb = np.asarray(Cb, np.float32)
    lam = np.zeros((STEPS, taps, 2, 2), np.float32)
    d = np.diag(Cb)
    for i in range(1, taps + 1):
        j = np.arange(i, STEPS)
        a = j - i
        for rr in range(2):
            row = 2 * a + rr
            for cc in range(2):
                lam[j, i - 1, rr, cc] = Cb[row, 2 * j + cc] / d[row]
    return lam.reshape(-1)


def upper_factor_of_inverse(H64, damp):
    """C upper triangular with (H + damp * mean(diag H) * I)^-1 = C^T C, computed in fp64, returned in fp32."""
    H = H64.clone()
    H.diagonal().add_(damp * float(H.diagonal().mean()))
    Lc = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Lc)
    Hinv = 0.5 * (Hinv + Hinv.T)
    return torch.linalg.cholesky(Hinv, upper=True).float()


def row_scale16(W):
    """The fp16 scale of every row: the root mean square of the row (fp64), converted to fp16 by PyTorch."""
    return torch.sqrt((W.double() ** 2).mean(1)).to(torch.float16)


def row_scale16_direct(W):
    """The same, with the fp64 value rounded to fp16 in one step (numpy).  PyTorch on the GPU rounds through fp32 first, which
    gives a neighbouring fp16 value for about 1 row in 10,000.  The embedding table and the output matrix of the released models
    were made by an earlier encoder that rounded in one step; the body of the released models has the PyTorch values."""
    v = torch.sqrt((W.double() ** 2).mean(1)).cpu().numpy()
    return torch.from_numpy(v.astype(np.float16)).to(W.device)


def decode_block(sym, table, s, K):
    """sym int64 [R, 128], the symbols of one block of 256 columns -> fp32 [R, 256] = table[state] * s"""
    assert 2 * K >= STATE_BITS
    prev = torch.zeros_like(sym)
    prev[:, 1:] = sym[:, :-1]
    state = (prev >> (2 * K - STATE_BITS)) | (sym << (STATE_BITS - K))
    return table[state].reshape(sym.shape[0], T) * s[:, None]


def encode_rows(W, Cm, fbs, search, table, K, taps, s16):
    """Rows rounded independently of each other, column block by column block from the left.  After a block, its error
    (times the inverse of the block of C) is subtracted from the columns that follow: the update of GPTQ / LDLQ."""
    R, n = W.shape
    W = W.clone()
    s = s16.float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    sym = torch.empty((R, n // 2), dtype=torch.int16, device=W.device)
    Q = torch.empty_like(W)
    for bi, b0 in enumerate(range(0, n, T)):
        z = (W[:, b0:b0 + T] * inv[:, None]).contiguous()
        sb = search(z, fbs[bi], table, K, taps)
        sym[:, b0 // 2:b0 // 2 + STEPS] = sb
        Qb = decode_block(sb.long(), table, s, K)
        Q[:, b0:b0 + T] = Qb
        if b0 + T < n:
            E = torch.linalg.solve_triangular(Cm[b0:b0 + T, b0:b0 + T], W[:, b0:b0 + T] - Qb, upper=True, left=False)
            W[:, b0 + T:] -= E @ Cm[b0:b0 + T, b0 + T:]
    return sym, Q


def input_side(HI64, taps, damp):
    """The factor C of the input-side statistic and the feedback tables of every block of 256 columns."""
    Cm = upper_factor_of_inverse(HI64, damp)
    Ch = Cm.cpu().numpy()
    n = Cm.shape[0]
    fbs = [torch.from_numpy(feedback_tables(Ch[b0:b0 + T, b0:b0 + T], taps)).to(Cm.device) if taps else None
           for b0 in range(0, n, T)]
    return Cm, fbs


def encode_matrix(Wrot, HI64, search, table, K, taps=TAPS, damp=DAMP_IN, HO64=None, damp_out=DAMP_OUT, R=256, stats=None):
    """Wrot: the rotated matrix (cuda fp32 [m, n]); HI64 / HO64: the rotated statistics of its input / output side (fp64).
    Returns the symbols (numpy uint16 [m, n/2]), the fp16 row scales and the decoded matrix Q (rotated, fp32).
    With HO64 the rounding is two-sided: H_out + damping = U^T U (U upper triangular); the row blocks are taken from the last
    one upwards, and the target of a block is  W + F  with  F[above] += (U[above, block] / diag(U)[above]) (W[block] - Q[block]),
    so that the rows above make up for the error of the rows below.  The scale of a row is that of the original row."""
    Cm, fbs = input_side(HI64, taps, damp)
    m, n = Wrot.shape
    s16 = row_scale16(Wrot)
    if HO64 is None:
        sym, Q = encode_rows(Wrot, Cm, fbs, search, table, K, taps, s16)
        return sym.cpu().numpy().view(np.uint16), s16, Q
    H = HO64.clone()
    H.diagonal().add_(damp_out * float(H.diagonal().mean()))
    U = torch.linalg.cholesky(H, upper=True).float()
    del H
    dU = U.diagonal().clone()
    F = torch.zeros_like(Wrot)
    sym = torch.empty((m, n // 2), dtype=torch.int16, device=Wrot.device)
    Q = torch.empty_like(Wrot)
    fmax = 0.0
    for r1 in range(m, 0, -R):
        r0 = max(0, r1 - R)
        fmax = max(fmax, float(F[r0:r1].abs().max()))
        a, c = encode_rows(Wrot[r0:r1] + F[r0:r1], Cm, fbs, search, table, K, taps, s16[r0:r1])
        sym[r0:r1] = a
        Q[r0:r1] = c
        if r0 > 0:
            F[:r0] += (U[:r0, r0:r1] / dU[:r0, None]) @ (Wrot[r0:r1] - c)
    if stats is not None:
        stats["fmax_rel"] = fmax / max(1e-30, float(torch.sqrt((Wrot.double() ** 2).mean())))
    return sym.cpu().numpy().view(np.uint16), s16, Q


def quad(D, HI, HO=None):
    """tr(HO D HI D^T) (HO = identity when it is not given), products in fp32, the final sum in fp64"""
    D = D.float()
    X = D @ HI.float()
    if HO is None:
        return float((X * D).sum(dtype=torch.float64))
    return float(((HO.float() @ D) * X).sum(dtype=torch.float64))


def encode_body_matrix(Wr, hi0, hi1, ho1, search, table, K, m):
    """One body matrix at K bits per step.  Returns symbols (uint8), scales (uint16 bits of the fp16) and the record."""
    stt = {}
    t1 = time.perf_counter()
    sym, s16, Q = encode_matrix(Wr, hi1, search, table, K=K, HO64=ho1, R=max(64, min(256, m // 8)), stats=stt)
    D = Wr - Q
    den_in, den_two = quad(Wr, hi0), quad(Wr, hi1, ho1)
    r = dict(K=K, p_in=quad(D, hi0) / den_in, p_two=quad(D, hi1, ho1) / den_two, sec=time.perf_counter() - t1, **stt)
    if not (np.isfinite(r["p_in"]) and np.isfinite(r["p_two"]) and r["p_in"] <= 0.05):
        # never taken for the released models (0 of 800 matrices): one-sided rounding with H_in0 when the two-sided error is off
        bad = dict(r)
        del sym, s16, Q, D
        t1 = time.perf_counter()
        sym, s16, Q = encode_matrix(Wr, hi0, search, table, K=K)
        D = Wr - Q
        r = dict(K=K, p_in=quad(D, hi0) / den_in, p_two=quad(D, hi1, ho1) / den_two, sec=time.perf_counter() - t1,
                 fallback="one-sided with H_in0", two_sided=bad)
        if not (np.isfinite(r["p_in"]) and r["p_in"] < 0.5):
            raise SystemExit("the rounding of a matrix diverged: %r" % (r,))
    sym = np.asarray(sym)
    assert sym.shape == (m, Wr.shape[1] // 2) and int(sym.max()) < (1 << K)
    del Q, D
    return sym.astype(np.uint8), s16.cpu().numpy().astype(np.float16).view(np.uint16), r


# ------------------------------------------------------------------------------------------------------------ data of the tool
def load_tables():
    out = {}
    for K in (6, 8):
        p = DATA / ("value_table_k%d.bin" % K)
        raw = np.fromfile(p, dtype="<u2")
        if raw.size != 2 << STATE_BITS:
            raise SystemExit("%s: expected %d bytes" % (p, 4 << STATE_BITS))
        out[K] = raw
    return out


def load_bits(spec, names):
    """--bits: 6, 8, both, or a JSON file {tensor name: 6 or 8}.  Returns {name: [K, ...]}."""
    if spec in ("6", "8"):
        return {n: [int(spec)] for n in names}
    if spec == "both":
        return {n: [6, 8] for n in names}
    m = json.loads(Path(spec).read_text(encoding="utf-8"))
    m = m.get("bits", m)
    if set(m) != set(names) or not set(int(v) for v in m.values()) <= {6, 8}:
        raise SystemExit("%s must name every body matrix with 6 or 8" % spec)
    return {n: [int(m[n])] for n in names}


def body_names(ckpt):
    cfg = ckpt.config
    out = []
    for i in range(cfg["num_hidden_layers"]):
        attn = LINEAR_ORDER if cfg["layer_types"][i] == "linear_attention" else FULL_ORDER
        out += ["layers.%d.%s" % (i, n) for n in attn + MLP_ORDER]
    return out


class Log:
    def __init__(self, work):
        self.path = Path(work) / "convert.log"
        self.json = Path(work) / "convert.json"

    def say(self, s):
        print(s, flush=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(time.strftime("%H:%M:%S ") + s + "\n")

    def load(self):
        return json.loads(self.json.read_text(encoding="utf-8")) if self.json.exists() else {}

    def save(self, d):
        tmp = self.json.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(self.json)


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def environment():
    import platform
    import transformers
    e = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__,
             platform=platform.system())
    if torch.cuda.is_available():
        e["gpu"] = torch.cuda.get_device_name(0)
        e["cuda_of_torch"] = torch.version.cuda
    return e


# ------------------------------------------------------------------------------------------------------------------- convert
def codes_have(path, keys):
    if not path.exists():
        return False
    with np.load(path) as z:
        return all(k in z.files for k in keys)


def convert_body(a, ckpt, log, search, tables):
    work = Path(a.work)
    CK, CODES = work / "ckpt", work / "codes"
    CK.mkdir(parents=True, exist_ok=True)
    CODES.mkdir(parents=True, exist_ok=True)
    res = log.load()
    t0 = time.perf_counter()
    torch.manual_seed(0)
    st = Layers(ckpt)
    NL = ckpt.config["num_hidden_layers"]
    assert NL % SEGMENT == 0
    names = body_names(ckpt)
    plan = load_bits(a.bits, names)
    table32 = {K: torch.from_numpy(tables[K].view(np.float16).astype(np.float32).reshape(-1, 2)).cuda() for K in tables}

    def layer_keys(L):
        return ["%s__sym%d" % (n.split(".", 2)[2].replace(".", "__"), K) for n in names if int(n.split(".")[1]) == L for K in plan[n]]

    def done(L):
        return codes_have(CODES / ("layer_%02d.npz" % L), layer_keys(L))

    if all(done(L) for L in range(NL)):
        log.say("[body] all %d layers encoded" % NL)
        return
    ids = torch.from_numpy(np.load(a.calibration).astype(np.int64))
    NCAL = ids.shape[0]
    assert ids.shape[1] == WINDOW and NCAL % 2 == 0, "the calibration file must hold an even number of windows of 2,048 tokens"
    ids_sha = hashlib.sha256(ids.numpy().astype("<i4").tobytes()).hexdigest()
    res["calibration"] = dict(file=Path(a.calibration).name, windows=NCAL, ids_sha256_int32=ids_sha)
    res["environment"] = environment()
    res["search"] = getattr(search, "name", "?")
    res["settings"] = dict(taps=TAPS, damp_in=DAMP_IN, damp_out=DAMP_OUT, label_seed=LABEL_SEED, sign_seed=SIGN_SEED, segment=SEGMENT)
    log.save(res)
    log.say("=== convert: %d windows, bits %s, search %s; layers done %d ===" % (NCAL, a.bits, res["search"], sum(done(L) for L in range(NL))))

    def ck(i):
        return CK / ("h_%02d.pt" % i)

    gpath = CK / "g_resume.pt"
    B = None
    if gpath.exists():
        g = torch.load(gpath)
        if g.get("ids_sha256_int32") != ids_sha:
            raise SystemExit("the checkpoints of %s were made with other calibration windows" % CK)
        B, G = int(g["bottom"]), g["G"]
        del g
        assert all(ck(i).exists() for i in range(0, B, SEGMENT))
        log.say("[body] resume: the gradient at the input of layer %d loaded" % B)
    if B is None:
        emb = ckpt.get(PREFIX + "embed_tokens.weight")
        bank = torch.nn.functional.embedding(ids, emb).contiguous()        # [windows, 2048, hidden] bf16, on the CPU
        del emb
        with torch.no_grad():
            for i in range(NL):
                if i % SEGMENT == 0:
                    save_atomic(bank, ck(i))
                layer = st.layer(ckpt.layer_state(i), i)
                for w in range(0, NCAL, 2):
                    bank[w:w + 2].copy_(st.forward(layer, bank[w:w + 2].cuda()).cpu())
                del layer
                torch.cuda.empty_cache()
        assert bool(torch.isfinite(bank).all())
        log.say("[body] forward through all layers, the inputs of every %dth layer saved (%.0fs)" % (SEGMENT, time.perf_counter() - t0))
        G, stat = head_gradients(ckpt, st, bank)
        del bank
        gc.collect()
        save_atomic(stat, CK / "final_stat.pt")
        B = NL
        save_atomic(dict(bottom=B, G=G, ids_sha256_int32=ids_sha), gpath)
        log.say("[body] gradients at the top (%.0fs)" % (time.perf_counter() - t0))
    segments = 0
    while B > 0:
        if (work / "STOP").exists() or (a.max_segments and segments >= a.max_segments):
            log.say("=== stopping before layer %d (run the same command again to continue) ===" % (B - 1))
            return
        s0 = B - SEGMENT
        t1 = time.perf_counter()
        x = torch.load(ck(s0))
        Hb = {s0: [x[w:w + 1].clone() for w in range(NCAL)]}
        del x
        with torch.no_grad():
            for i in range(s0, B - 1):
                layer = st.layer(ckpt.layer_state(i), i)
                outp = []
                for w in range(0, NCAL, 2):
                    y = st.forward(layer, torch.cat(Hb[i][w:w + 2]).cuda()).cpu()
                    outp += [y[0:1].clone(), y[1:2].clone()]
                Hb[i + 1] = outp
                del layer
                torch.cuda.empty_cache()
        log.say("[layers %d-%d] layer inputs (%.0fs)" % (s0, B - 1, time.perf_counter() - t1))
        for L in range(B - 1, s0 - 1, -1):
            t2 = time.perf_counter()
            state = ckpt.layer_state(L)
            layer = st.layer({k: v.clone() for k, v in state.items()}, L)
            layer.requires_grad_(False)
            H, G = layer_stats(st, L, layer, Hb[L], G)
            del layer, Hb[L]
            gc.collect()
            torch.cuda.empty_cache()
            ts = time.perf_counter() - t2
            if done(L):
                msg = "already encoded"
            else:
                msg = encode_layer(L, H, state, st, plan, search, table32, res, CODES)
            log.save(res)
            log.say("[layer %d] statistics %.0fs; %s" % (L, ts, msg))
            del H, state
            gc.collect()
            torch.cuda.empty_cache()
        B = s0
        save_atomic(dict(bottom=B, G=G, ids_sha256_int32=ids_sha), gpath)
        segments += 1
        log.say("[layers %d-%d] done (%.0fs; total %.0fs)" % (s0, s0 + SEGMENT - 1, time.perf_counter() - t1, time.perf_counter() - t0))
    res["body_seconds"] = res.get("body_seconds", 0) + time.perf_counter() - t0
    log.save(res)
    log.say("[body] all %d layers encoded (%.0fs)" % (NL, time.perf_counter() - t0))


def encode_layer(L, H, state, st, plan, search, table32, res, CODES):
    rec = res.setdefault("body", {})
    save = {}
    t0 = time.perf_counter()
    line = []
    for g in st.groups(L):
        for name in g:
            full = "layers.%d.%s" % (L, name)
            Wb = state[name + ".weight"]
            m, n = Wb.shape
            si, so = (torch.from_numpy(s).cuda() for s in sign_vectors(n, m))
            Wr = rotate_weight(Wb.cuda().float(), si, so)
            hi0 = rotate_stat(H["HI0"][H["grp"][name]], si).double()
            hi1 = rotate_stat(H["HI1"][name], si).double()
            ho1 = rotate_stat(H["HO1"][name], so).double()
            key = name.replace(".", "__")
            save[key + "__shape"] = np.array([m, n], np.int64)
            for K in plan[full]:
                sym, sc, r = encode_body_matrix(Wr, hi0, hi1, ho1, search, table32[K], K, m)
                rec.setdefault(full, {})[str(K)] = r
                save["%s__sym%d" % (key, K)] = sym
                save["%s__sc%d" % (key, K)] = sc
                line.append("%s K%d %.5f/%.5f%s" % (name.split(".")[-1], K, r["p_in"], r["p_two"], " FALLBACK" if "fallback" in r else ""))
                del sym, sc
            del Wr, hi0, hi1, ho1
            torch.cuda.empty_cache()
    f = CODES / ("layer_%02d.npz" % L)
    tmp = f.with_name(f.name + ".tmp.npz")
    np.savez(tmp, **save)
    tmp.replace(f)
    return "encode %.0fs, error in/two-sided: %s" % (time.perf_counter() - t0, ", ".join(line))


def convert_embedding(a, ckpt, log, search, tables, K=6, chunk=8192):
    """The embedding table: input rotation only, every sequence of 256 weights rounded for plain squared error (no statistic:
    a lookup has no input to weigh), K = 6."""
    out = Path(a.work) / "codes" / "embed_tokens.npz"
    if out.exists():
        log.say("[embedding] already encoded")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    res = log.load()
    t0 = time.perf_counter()
    W = ckpt.get(PREFIX + "embed_tokens.weight")
    V, n = W.shape
    si = torch.from_numpy(sign_vectors(n, 0)[0]).cuda()
    table = torch.from_numpy(tables[K].view(np.float16).astype(np.float32).reshape(-1, 2)).cuda()
    sym = np.empty((V, n // 2), np.uint8)
    sc = np.empty(V, np.uint16)
    num = den = 0.0
    with torch.inference_mode():
        for r0 in range(0, V, chunk):
            r1 = min(V, r0 + chunk)
            wr = fwht512_rows(W[r0:r1].float().cuda(), si)
            s16 = row_scale16_direct(wr)
            s = s16.float()
            inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
            q = torch.empty_like(wr)
            for b0 in range(0, n, T):
                z = (wr[:, b0:b0 + T] * inv[:, None]).contiguous()
                sb = search(z, None, table, K, 0)
                sym[r0:r1, b0 // 2:b0 // 2 + STEPS] = sb.cpu().numpy().astype(np.uint8)
                q[:, b0:b0 + T] = decode_block(sb.long(), table, s, K)
            sc[r0:r1] = s16.cpu().numpy().view(np.uint16)
            d = q - wr
            num += float((d.double() ** 2).sum())
            den += float((wr.double() ** 2).sum())
    np.savez(out.with_name(out.name + ".tmp.npz"), sym=sym, sc=sc, shape=np.array([V, n], np.int64), K=np.array([K], np.int64))
    out.with_name(out.name + ".tmp.npz").replace(out)
    res["embedding"] = dict(K=K, rel_mse=num / den, seconds=time.perf_counter() - t0)
    log.save(res)
    log.say("[embedding] K %d, relative squared error %.5f (%.0fs)" % (K, num / den, time.perf_counter() - t0))


def convert_head(a, ckpt, log, search, tables, K=8, chunk=8192):
    """The output matrix: input rotation only (its rows are the vocabulary), rounded with the input side only; the statistic is
    that of the output of the final norm on the calibration windows (saved by the body stage)."""
    out = Path(a.work) / "codes" / "lm_head.npz"
    if out.exists():
        log.say("[head] already encoded")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    stat_path = Path(a.head_stat) if a.head_stat else Path(a.work) / "ckpt" / "final_stat.pt"
    if not stat_path.exists():
        raise SystemExit("%s is missing: run the body stage first (it saves the statistic of the final norm)" % stat_path)
    res = log.load()
    t0 = time.perf_counter()
    Hm = torch.load(stat_path).cuda()
    n = Hm.shape[0]
    W = ckpt.get("lm_head.weight")
    V = W.shape[0]
    assert W.shape[1] == n
    si = torch.from_numpy(sign_vectors(n, 0)[0]).cuda()
    Hd = rotate_stat(Hm, si).double()
    table = torch.from_numpy(tables[K].view(np.float16).astype(np.float32).reshape(-1, 2)).cuda()
    Cm, fbs = input_side(Hd, TAPS, DAMP_IN)
    sym = np.empty((V, n // 2), np.uint8)
    sc = np.empty(V, np.uint16)
    num = den = mse_n = mse_d = 0.0
    with torch.inference_mode():
        for r0 in range(0, V, chunk):
            r1 = min(V, r0 + chunk)
            wr = fwht512_rows(W[r0:r1].float().cuda(), si)
            s16 = row_scale16_direct(wr)
            sb, q = encode_rows(wr, Cm, fbs, search, table, K, TAPS, s16)
            sym[r0:r1] = sb.cpu().numpy().astype(np.uint8)
            sc[r0:r1] = s16.cpu().numpy().view(np.uint16)
            d = (q - wr).double()
            wd = wr.double()
            num += float(((d @ Hd) * d).sum())
            den += float(((wd @ Hd) * wd).sum())
            mse_n += float((d * d).sum())
            mse_d += float((wd * wd).sum())
    np.savez(out.with_name(out.name + ".tmp.npz"), sym=sym, sc=sc, shape=np.array([V, n], np.int64), K=np.array([K], np.int64),
             proxy=np.array([num / den]))
    out.with_name(out.name + ".tmp.npz").replace(out)
    res["head"] = dict(K=K, p_in=num / den, rel_mse=mse_n / mse_d, statistic=stat_path.name, seconds=time.perf_counter() - t0)
    log.save(res)
    log.say("[head] K %d, error %.5f, relative squared error %.5f (%.0fs)" % (K, num / den, mse_n / mse_d, time.perf_counter() - t0))


def cmd_convert(a):
    need_torch()
    import trellis_search as TS
    if not torch.cuda.is_available():
        raise SystemExit("convert needs a CUDA GPU")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    log = Log(work)
    ckpt = Checkpoint(a.checkpoint)
    search = TS.load(a.native, allow_reference=a.reference_search, say=log.say)
    tables = load_tables()
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    for p in parts:
        if p not in ("body", "embedding", "head"):
            raise SystemExit("--parts: body, embedding, head")
    if "embedding" in parts:
        convert_embedding(a, ckpt, log, search, tables)
    if "body" in parts:
        convert_body(a, ckpt, log, search, tables)
    if "head" in parts:
        convert_head(a, ckpt, log, search, tables)


# ------------------------------------------------------------------------------------------------------------------ the file
def pack_warp(sym, rows, cols, K, chunk=16384):
    """Symbols [rows, cols / 2] -> the uint32 words of a tensor in the "warp" layout (docs/FORMAT.md)."""
    seqs, wps = cols // T, 4 * K
    out = np.empty(rows * seqs * wps, np.uint32)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        n = r1 - r0
        msym = sym[r0:r1].astype(np.uint32).reshape(n, seqs, STEPS)
        bits = ((msym[..., None] >> np.arange(K, dtype=np.uint32)) & 1).astype(np.uint8).reshape(n, seqs, STEPS * K)
        words = np.packbits(bits, axis=-1, bitorder="little").view("<u4").reshape(n, seqs, wps)
        q = np.arange(seqs)[:, None] * rows + np.arange(r0, r1)[None, :]
        tile, lane = q // 32, q % 32
        w = np.arange(wps)
        out[(((tile[..., None] * (wps // 4) + w // 4) * 32 + lane[..., None]) * 4 + w % 4)] = words.transpose(1, 0, 2)
    return out


def pack_row(sym, rows, cols, K, chunk=16384):
    """Symbols [rows, cols / 2] -> the uint32 words of a tensor in the "row" layout."""
    seqs, wps = cols // T, 4 * K
    out = np.empty((rows, seqs, wps), np.uint32)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        n = r1 - r0
        msym = sym[r0:r1].astype(np.uint32).reshape(n, seqs, STEPS)
        bits = ((msym[..., None] >> np.arange(K, dtype=np.uint32)) & 1).astype(np.uint8).reshape(n, seqs, STEPS * K)
        out[r0:r1] = np.packbits(bits, axis=-1, bitorder="little").view("<u4").reshape(n, seqs, wps)
    return out.reshape(-1)


def rotation(m, offset):
    return {"kind": "H512", "m": m, "sign_seed": SIGN_SEED, "sign_offset": offset, "qm_off": 0, "rule": "mulberry32_7919_13"}


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"\0" * HEADER_RESERVE)
        self.pos = HEADER_RESERVE
        self.sha = hashlib.sha256()

    def _write(self, b):
        self.f.write(b)
        self.sha.update(b)
        self.pos += len(b)

    def section(self, data):
        pad = (-self.pos) % ALIGN
        if pad:
            self._write(b"\0" * pad)
        off = self.pos
        self._write(data if isinstance(data, (bytes, bytearray)) else np.ascontiguousarray(data).tobytes())
        return off

    def finish(self, header):
        pad = (-self.pos) % ALIGN
        if pad:
            self._write(b"\0" * pad)
        js = json.dumps(header, separators=(",", ":")).encode("ascii")
        assert 0x20 + len(js) <= HEADER_RESERVE
        self.f.seek(0)
        self.f.write(b"YSKN" + struct.pack("<I", 1) + struct.pack("<QQQ", len(js), HEADER_RESERVE, self.pos) + js)
        self.f.close()
        return self.sha.hexdigest()


class Container:
    """A .yskn file for reading (the layout of docs/FORMAT.md)."""

    def __init__(self, path):
        self.path = str(path)
        self.size = os.path.getsize(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        head = bytes(self.mm[:0x20])
        if head[:4] != b"YSKN":
            raise SystemExit("%s: not a YSKN file" % path)
        self.json_len, self.data_offset, self.file_length = struct.unpack_from("<QQQ", head, 8)
        self.header = json.loads(bytes(self.mm[0x20:0x20 + self.json_len]).decode("utf-8"))
        self.tensors = {t["name"]: t for t in self.header["tensors"]}
        self.vectors = {v["name"]: v for v in self.header["vectors"]}

    def raw(self, off, n):
        return bytes(self.mm[off:off + n])

    def parts(self, t):
        rows = t.get("physical_rows", t["rows"])
        return dict(stream=self.raw(t["stream_off"], t["stream_bytes"]), table=self.raw(t["lut_off"], 4 << STATE_BITS),
                    scales=self.raw(t["scale_off"], rows * 2))

    def symbols(self, t, r0=0, r1=None):
        """uint8 [rows, cols / 2]: the stored symbols of rows r0 .. r1 of a tensor"""
        rows, seqs, K = t.get("physical_rows", t["rows"]), t["cols"] // T, t["bits"]
        wps = 4 * K
        r1 = rows if r1 is None else r1
        words = np.ndarray((rows * seqs * wps,), dtype="<u4", buffer=self.mm, offset=t["stream_off"])
        if t.get("layout", "warp") == "row":
            u = words.reshape(rows, seqs, wps)[r0:r1]
        else:
            a = words.reshape(rows * seqs // 32, K, 32, 4)
            u = a.transpose(0, 2, 1, 3).reshape(seqs, rows, wps).transpose(1, 0, 2)[r0:r1]
        u = np.ascontiguousarray(u)
        bits = np.unpackbits(u.view(np.uint8).reshape(r1 - r0, seqs, wps * 4), axis=-1, bitorder="little")
        return np.packbits(bits.reshape(-1, K), axis=-1, bitorder="little")[:, 0].reshape(r1 - r0, seqs * STEPS)


def model_entry(ckpt):
    cfg = ckpt.config
    theta = float(cfg["rope_parameters"]["rope_theta"]) if "rope_parameters" in cfg else float(cfg.get("rope_theta", 1e7))
    return dict(arch="qwen3_5", layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"], ffn=cfg["intermediate_size"],
                heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                vocab=cfg["vocab_size"], eps=cfg["rms_norm_eps"], rope_theta=theta, tie_embeddings=False,
                layer_types=cfg["layer_types"], rotary_dim=64, linear_num_key_heads=cfg["linear_num_key_heads"],
                linear_num_value_heads=cfg["linear_num_value_heads"], linear_key_head_dim=cfg["linear_key_head_dim"],
                linear_value_head_dim=cfg["linear_value_head_dim"], linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
                attn_output_gate=True, norm_offset=1, source=ckpt.source_name())


def cmd_assemble(a):
    need_torch()
    work = Path(a.work)
    CODES = work / "codes"
    ckpt = Checkpoint(a.checkpoint)
    cfg = ckpt.config
    names = body_names(ckpt)
    if a.bits == "both":
        raise SystemExit("assemble needs one number of bits per matrix: --bits 6, 8 or a JSON file")
    want = {n: ks[0] for n, ks in load_bits(a.bits, names).items()}
    tables = load_tables()
    rec = json.loads((work / "convert.json").read_text(encoding="utf-8"))
    out_path = Path(a.out)
    if out_path.exists() and not a.overwrite:
        raise SystemExit("%s exists (pass --overwrite to replace it)" % out_path)
    other = {}
    for part, src in (("lm_head", a.head_from), ("embed_tokens", a.embedding_from)):
        if src:
            other[part] = Container(src)
    t0 = time.perf_counter()
    partial = out_path.with_name(out_path.name + ".partial")
    w = Writer(partial)
    tensors, vectors = [], []
    z, zL = None, None
    for name in names:
        L = int(name.split(".")[1])
        key = name.split(".", 2)[2].replace(".", "__")
        K = want[name]
        if zL != L:
            if z is not None:
                z.close()
            z, zL = np.load(CODES / ("layer_%02d.npz" % L)), L
        rows, cols = (int(x) for x in z[key + "__shape"])
        words = pack_warp(z["%s__sym%d" % (key, K)], rows, cols, K).astype("<u4")
        stream_off = w.section(words)
        lut_off = w.section(tables[K].astype("<u2"))
        scale_off = w.section(z["%s__sc%d" % (key, K)].astype("<u2"))
        tensors.append(dict(name=name, role="body", rows=rows, cols=cols, physical_rows=rows, l=STATE_BITS, d=2, t=T, bits=K,
                            start="zero", layout="warp", rot_in=rotation(cols // 512, 0), rot_out=rotation(rows // 512, cols),
                            lut_off=lut_off, scale_off=scale_off, stream_off=stream_off, stream_bytes=int(words.nbytes), rar=None,
                            bias_off=0, proxy=float(rec["body"][name][str(K)]["p_in"]), encoder="yskn"))
        del words
    if z is not None:
        z.close()
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    # the output matrix
    if "lm_head" in other:
        c = other["lm_head"]
        t = c.tensors["lm_head"]
        p = c.parts(t)
        e = dict(t)                                                        # the description of the source, with new offsets
        e["stream_off"] = w.section(p["stream"])
        e["lut_off"] = w.section(p["table"])
        e["scale_off"] = w.section(p["scales"])
        tensors.append(e)
    else:
        zh = np.load(CODES / "lm_head.npz")
        K = int(zh["K"][0])
        words = pack_warp(zh["sym"], V, H, K).astype("<u4")
        stream_off = w.section(words)
        lut_off = w.section(tables[K].astype("<u2"))
        scale_off = w.section(zh["sc"].astype("<u2"))
        tensors.append(dict(name="lm_head", role="head", rows=V, cols=H, physical_rows=V, l=STATE_BITS, d=2, t=T, bits=K, start="zero",
                            layout="warp", rot_in=rotation(H // 512, 0), rot_out=None, lut_off=lut_off, scale_off=scale_off,
                            stream_off=stream_off, stream_bytes=int(words.nbytes), rar=None, bias_off=0, proxy=float(zh["proxy"][0]),
                            encoder="yskn"))
        del words
    # the embedding table
    if "embed_tokens" in other:
        c = other["embed_tokens"]
        t = c.tensors["embed_tokens"]
        p = c.parts(t)
        e = dict(t)
        e["stream_off"] = w.section(p["stream"])
        e["lut_off"] = w.section(p["table"])
        e["scale_off"] = w.section(p["scales"])
        tensors.append(e)
    else:
        ze = np.load(CODES / "embed_tokens.npz")
        K = int(ze["K"][0])
        words = pack_row(ze["sym"], V, H, K).astype("<u4")
        e = dict(name="embed_tokens", role="embedding", rows=V, cols=H, physical_rows=V, l=STATE_BITS, d=2, t=T, bits=K, start="zero",
                 layout="row", rot_in=rotation(H // 512, 0), rot_out=None, rar=None, bias_off=0, encoder="yskn")
        e["stream_off"] = w.section(words)
        e["stream_bytes"] = int(words.nbytes)
        e["lut_off"] = w.section(tables[K].astype("<u2"))
        e["scale_off"] = w.section(ze["sc"].astype("<u2"))
        tensors.append(e)
        del words
    # the small tensors, as they are
    for i in range(cfg["num_hidden_layers"]):
        small = SMALL_LINEAR if cfg["layer_types"][i] == "linear_attention" else SMALL_FULL
        for s in small:
            v = ckpt.get("%slayers.%d.%s" % (PREFIX, i, s))
            assert v.dtype == torch.bfloat16
            vectors.append(dict(name="layers.%d.%s" % (i, SMALL_NAME[s]), off=w.section(v.contiguous().view(torch.int16).numpy()),
                                n=int(v.numel()), dtype="bf16"))
    v = ckpt.get(PREFIX + "norm.weight")
    vectors.append(dict(name="norm", off=w.section(v.contiguous().view(torch.int16).numpy()), n=int(v.numel()), dtype="bf16"))
    data_sha = w.finish(dict(model=model_entry(ckpt), tensors=tensors, vectors=vectors))
    partial.replace(out_path)
    kc = {K: sum(1 for n in names if want[n] == K) for K in (6, 8)}
    receipt = dict(file=out_path.name, bytes=out_path.stat().st_size, sha256=sha256_file(out_path), data_sha256=data_sha,
                   body_bits=kc, head="copied from " + Path(a.head_from).name if a.head_from else "converted",
                   embedding="copied from " + Path(a.embedding_from).name if a.embedding_from else "converted",
                   seconds=time.perf_counter() - t0)
    print(json.dumps(receipt, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------------- compare
def cmd_compare(a):
    A, B = Container(a.a), Container(a.b)
    rep = dict(a=Path(a.a).name, b=Path(a.b).name, tensors={}, summary={})
    tot = {}
    for name, ta in A.tensors.items():
        tb = B.tensors.get(name)
        role = ta["role"]
        s = tot.setdefault(role, dict(tensors=0, same_bits=0, identical=0, symbols=0, symbols_different=0, rows=0, scales_different=0,
                                      tables_different=0))
        s["tensors"] += 1
        if tb is None or (tb["rows"], tb["cols"]) != (ta["rows"], ta["cols"]):
            rep["tensors"][name] = dict(status="missing or another shape in B")
            continue
        r = dict(bits_a=ta["bits"], bits_b=tb["bits"])
        if ta["bits"] != tb["bits"]:
            r["status"] = "another number of bits"
            rep["tensors"][name] = r
            continue
        s["same_bits"] += 1
        pa, pb = A.parts(ta), B.parts(tb)
        rows = ta.get("physical_rows", ta["rows"])
        nsym = rows * ta["cols"] // 2
        diff = 0
        if pa["stream"] != pb["stream"]:
            for r0 in range(0, rows, 4096):
                r1 = min(rows, r0 + 4096)
                diff += int((A.symbols(ta, r0, r1) != B.symbols(tb, r0, r1)).sum())
        sd = int((np.frombuffer(pa["scales"], "<u2") != np.frombuffer(pb["scales"], "<u2")).sum())
        td = int(pa["table"] != pb["table"])
        r.update(symbols=nsym, symbols_different=diff, rows=rows, scales_different=sd, table_equal=not td)
        r["status"] = "identical" if (diff == 0 and sd == 0 and not td) else "different"
        s["identical"] += r["status"] == "identical"
        s["symbols"] += nsym
        s["symbols_different"] += diff
        s["rows"] += rows
        s["scales_different"] += sd
        s["tables_different"] += td
        rep["tensors"][name] = r
        if r["status"] != "identical" or a.verbose:
            print("%-42s K%d  symbols different %d of %d (%.4f %%), row scales different %d of %d, table %s"
                  % (name, ta["bits"], diff, nsym, 100.0 * diff / nsym, sd, rows, "equal" if not td else "DIFFERENT"), flush=True)
    vec_same = sum(1 for n, v in A.vectors.items()
                   if n in B.vectors and B.vectors[n]["n"] == v["n"] and A.raw(v["off"], v["n"] * 2) == B.raw(B.vectors[n]["off"], v["n"] * 2))
    for role, s in tot.items():
        s["share_different"] = s["symbols_different"] / s["symbols"] if s["symbols"] else None
        print("%-9s %d tensors, %d with the same number of bits, %d identical; symbols different %d of %d%s; row scales different %d of %d"
              % (role, s["tensors"], s["same_bits"], s["identical"], s["symbols_different"], s["symbols"],
                 "" if not s["symbols"] else " (%.5f %%)" % (100.0 * s["share_different"]), s["scales_different"], s["rows"]))
    hs = dict(a=sha256_file(a.a, A.data_offset), b=sha256_file(a.b, B.data_offset))
    head_equal = A.header == B.header
    rep["summary"] = dict(roles=tot, vectors=len(A.vectors), vectors_identical=vec_same, data_sha256=hs,
                          data_identical=hs["a"] == hs["b"], header_equal=head_equal,
                          file_identical=bool(hs["a"] == hs["b"] and head_equal and A.size == B.size))
    print("vectors: %d of %d identical" % (vec_same, len(A.vectors)))
    print("data (everything after the header): %s" % ("IDENTICAL, SHA-256 " + hs["a"] if hs["a"] == hs["b"] else "different"))
    print("header: %s" % ("equal" if head_equal else "different"))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------------ selftest
def cmd_selftest(a):
    need_torch()
    import trellis_search as TS
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("%-74s %s %s" % (name, "ok" if cond else "FAILED", extra), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tables = load_tables()
    table32 = {K: torch.from_numpy(tables[K].view(np.float16).astype(np.float32).reshape(-1, 2)).to(dev) for K in tables}
    lib = Path(a.native) if a.native else TS.default_library()
    native = TS.NativeSearch(lib) if (lib and dev == "cuda") else None
    print("device %s; native search library: %s" % (dev, lib if native else "not found (the PyTorch reference is used)"))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261114)
    if native:                                              # 1. the kernel against the reference
        for K in (6, 8):
            for taps in (0, TAPS):
                z = torch.randn((48, T), generator=gen).to(dev).contiguous()
                fb = None
                if taps:
                    f4 = torch.randn((STEPS, taps, 4), generator=gen) * 0.15
                    for i in range(1, taps + 1):
                        f4[:i, i - 1] = 0.0
                    fb = f4.reshape(-1).to(dev).contiguous()
                x, y = native(z, fb, table32[K], K, taps), TS.reference_search(z, fb, table32[K], K, taps)
                check("search, K %d, %d taps: kernel and reference give the same symbols" % (K, taps), bool((x == y).all()))
    search = native or TS.reference_search
    # 2. one random matrix with random statistics: two-sided rounding, then through the file layout and back
    m, n = 1024, 1024
    W = torch.randn((m, n), generator=gen).to(dev) * 0.02
    X = torch.randn((4096, n), generator=gen).to(dev) * torch.exp(-torch.linspace(0, 3, n)).to(dev)
    Y = torch.randn((4096, m), generator=gen).to(dev) * torch.exp(-torch.linspace(0, 3, m)).to(dev)
    HI, HO = (X.t() @ X / 4096), (Y.t() @ Y / 4096)
    si, so = (torch.from_numpy(s).to(dev) for s in sign_vectors(n, m))
    Wr = rotate_weight(W, si, so)
    hi, ho = rotate_stat(HI, si).double(), rotate_stat(HO, so).double()
    check("rotation: inverse(rotate(W)) = W", float((inverse_rotate(Wr, si, so) - W).abs().max()) < 1e-6)
    for K in (6, 8):
        sym, s16, Q = encode_matrix(Wr, hi, search, table32[K], K=K, HO64=ho, R=128)
        sym0, s0, Q0 = encode_matrix(Wr, torch.eye(n, dtype=torch.float64, device=dev), search, table32[K], K=K, taps=0)
        e2, e0 = quad(Wr - Q, hi, ho) / quad(Wr, hi, ho), quad(Wr - Q0, hi, ho) / quad(Wr, hi, ho)
        check("K %d: two-sided rounding beats plain rounding in the weighted error" % K, e2 < e0, "(%.5f < %.5f)" % (e2, e0))
        tmp = Path(a.tmp or ".") / ("selftest_k%d.yskn" % K)
        w = Writer(tmp)
        words = pack_warp(sym.astype(np.uint8), m, n, K).astype("<u4")
        t = dict(name="layers.0.test", role="body", rows=m, cols=n, physical_rows=m, l=STATE_BITS, d=2, t=T, bits=K, start="zero",
                 layout="warp", rot_in=rotation(n // 512, 0), rot_out=rotation(m // 512, n))
        t["stream_off"] = w.section(words)
        t["stream_bytes"] = int(words.nbytes)
        t["lut_off"] = w.section(tables[K].astype("<u2"))
        t["scale_off"] = w.section(s16.cpu().numpy().astype(np.float16).view(np.uint16))
        w.finish(dict(model=dict(arch="selftest"), tensors=[t], vectors=[]))
        c = Container(tmp)
        back = c.symbols(c.tensors["layers.0.test"])
        check("K %d: symbols written to a file and read back" % K, np.array_equal(back, sym.astype(np.uint8)))
        try:                                                # the independent decoder of tools/, when it is next to this folder
            sys.path.insert(0, str(HERE.parent))
            import yskn_decode as YD
            cc = YD.Container(str(tmp))
            dec = YD.decode(cc, YD.checked(cc.tensors["layers.0.test"]))
            ref = inverse_rotate(Q, si, so).cpu().numpy()
            err = float(np.abs(dec - ref).max()) / float(np.abs(ref).max())
            check("K %d: tools/yskn_decode.py decodes the file to the rounded matrix" % K, err < 1e-5, "(max difference %.1e)" % err)
        except ImportError:
            print("(tools/yskn_decode.py not found: the decode check is skipped)")
        del c
        try:
            os.remove(tmp)
        except OSError:
            pass
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--native", help="the search library (default: where native/CMakeLists.txt builds it)")
    p.add_argument("--tmp", help="folder for the two small test files")
    p = sub.add_parser("convert")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--bits", default="both", help="6, 8, both (default) or a JSON file {tensor name: 6 or 8}")
    p.add_argument("--parts", default="embedding,body,head")
    p.add_argument("--calibration", default=str(DATA / "calibration_ids.npy"), help="int32 [windows, 2048] token ids")
    p.add_argument("--head-stat", help="another statistic for the output matrix (a saved [hidden, hidden] tensor)")
    p.add_argument("--native", help="the search library (default: where native/CMakeLists.txt builds it)")
    p.add_argument("--reference-search", action="store_true", help="allow the slow PyTorch search when the library is missing")
    p.add_argument("--max-segments", type=int, default=0, help="stop after this many blocks of 8 layers (0 = all)")
    p = sub.add_parser("assemble")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--bits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head-from", help="take the output matrix from this .yskn file instead of the converted one")
    p.add_argument("--embedding-from", help="take the embedding table from this .yskn file instead of the converted one")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--json")
    p.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    return {"selftest": cmd_selftest, "convert": cmd_convert, "assemble": cmd_assemble, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
