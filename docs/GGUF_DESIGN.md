# YSKN container format — specification and encoder feasibility

Scope: everything below is derived by reading the reader side of this repository, which is
the upstream v0.1.0 tree this fork started from. In that tree the converter is **not** present
(`docs/TECHNOLOGY.md:100`, `docs/TECHNOLOGY.md:310`, `README.md:225`), so every statement about
*writing* a container is an inference from what the reader accepts. Citations are `file:line`
against that checkout.

> **Superseded in part by upstream v0.1.2.** Upstream published the encoder at
> `tools/convert/yskn_convert.py` (1241 lines) on 2026-11-14. Where the two disagree, the
> published encoder is authoritative. See section 7 for what it confirmed and what it corrected.

---

## 1. Format layout

### 1.1 File header (32 bytes, little-endian)

| offset | size | field | evidence |
|---|---|---|---|
| 0x00 | 4 | magic `"YSKN"` (ASCII, no NUL) | `src/format.hpp:28` |
| 0x04 | 4 | `uint32 version`, must be exactly `1` | `src/format.hpp:29-30` |
| 0x08 | 4 | `uint32 json_len` — byte length of the JSON metadata | `src/format.hpp:29` |
| 0x0C | 20 | reserved / unread (the reader jumps straight to 0x20) | `src/format.hpp:32` |
| 0x20 | `json_len` | UTF-8 JSON object, **not** NUL-terminated | `src/format.hpp:32` |

Validation the reader performs, in order (`src/format.hpp:26-32`):

1. `len >= 0x20` and `memcmp(base,"YSKN",4)==0` — else `"not a YSKN container"`.
2. `version == 1` — else `"unsupported YSKN version"`.
3. `json_len + 0x20 <= len` — else `"YSKN header truncated"`.
4. `json::parse(std::string(base+0x20, json_len))` — the JSON is parsed from exactly
   `json_len` bytes; trailing bytes after it are payload, not JSON.

There is no checksum, no alignment field, no section table. Every payload is addressed by an
**absolute file offset** (`src/format.hpp:64`: `template <class T2> const T2* at(uint64_t off)
{ return reinterpret_cast<const T2*>(base + off); }`). The file is `mmap`'d read-only
(`src/mapped_file.hpp:63`), so offsets are never rebased. Because payloads are read through
`reinterpret_cast` to `uint16_t`/`uint32_t`/`uint64_t`, every `*_off` must be naturally
aligned (2/4/8 bytes) or the access is undefined behaviour on the host.

### 1.2 `"model"` object

Parsed at `src/format.hpp:33-37` into `ModelDesc` (`src/format.hpp:22`). All keys except
`tie_embeddings` are **required** (`.at()` throws if missing).

| key | type | meaning | evidence |
|---|---|---|---|
| `arch` | string | dispatch tag. `"qwen3_5"` → `Engine35`; anything else → legacy `Engine`. Also `"qwen3_5_mtp"` (MTP drafter) and `"bdraft2"` (draft model) are recognised for auxiliary containers | `src/c_api.cpp:16-17`, `src/engine35.cpp:216`, `src/engine35.cpp:1363`, `src/engine35.cpp:1624` |
| `layers` | int | number of decoder layers | `src/format.hpp:34` |
| `hidden` | int | residual width `H` (5120 for the shipped model) | `src/format.hpp:34`, `src/engine35.cpp:217` |
| `ffn` | int | MLP intermediate width; must be a multiple of 512 | `src/format.hpp:35`, `src/engine35.cpp:217` |
| `heads` | int | query heads (24) | `src/format.hpp:35` |
| `kv_heads` | int | KV heads (4) | `src/format.hpp:35` |
| `head_dim` | int | per-head dim (256) | `src/format.hpp:36` |
| `vocab` | int | vocabulary size (248320) | `src/format.hpp:36` |
| `eps` | float | RMSNorm epsilon | `src/format.hpp:36` |
| `rope_theta` | float | RoPE base | `src/format.hpp:37` |
| `tie_embeddings` | bool | optional, default `false` | `src/format.hpp:37` |
| `layer_types` | array of string | **required by `Engine35` only**, length must equal `layers`. Value `"full_attention"` selects a KV-attention layer; any other value selects a state-carrying (linear-attention / Gated DeltaNet) layer | `src/engine35.cpp:225-227`, `src/engine35.cpp:239` |

`layer_types` is read from the raw JSON, not from `ModelDesc`, so it is easy to miss.

### 1.3 `"tensors"` array

One object per quantised matrix, parsed at `src/format.hpp:41-59` into `TensorDesc`
(`src/format.hpp:17-20`).

| key | type | required | meaning |
|---|---|---|---|
| `name` | string | yes | lookup key; see §5.4 for the names the engines request |
| `role` | string | yes | `"head"` / `"embedding"` / anything else. Only used to suppress the default `rot_out` (`src/format.hpp:51`) and to validate the embedding (`src/engine35.cpp:276`) |
| `rows` | int | yes | logical output rows; used only for the default `rot_out.m = rows/512` (`src/format.hpp:51`) and shape checks |
| `cols` | int | yes | input width; must be a multiple of 256 (`src/format.hpp:56`) |
| `physical_rows` | int | no, default `rows` | rows actually stored. Must be a multiple of 64 because repack requires it (`src/repack.cu:22`, `src/loader.cpp:39`). The shipped model has `physical_rows == rows` (248320 and 5120 are both multiples of 64) |
| `l` | int | no, default 12 | LUT index width. `S = 1<<l` entries. Reader hard-fails unless `l == 12` (`src/loader.cpp:30`) |
| `d` | int | no, default 2 | values per trellis step. Must be 2 (`src/format.hpp:56`); never read again |
| `t` | int | no, default 256 | sequence length per code block. Must be 256 (`src/format.hpp:56`); never read again |
| `bits` | int | yes | bits **per trellis step**, 4..8 (`src/format.hpp:56`). One step encodes `d=2` weights, so bits/weight = `bits/2`: `bits:6` = 3 bit/weight, `bits:8` = 4 bit/weight. See §2.2 |
| `start` | string | no, default `"zero"` | trellis start convention; must be `"zero"` or `"carry"` (`src/format.hpp:56`). Only `"zero"` is actually implemented — see §6 |
| `layout` | string | no, default `"warp"` | `"warp"` = tile/lane interleaved stream (§2.3); `"row"` = plain row-major (embedding only, `src/loader.cpp:62`) |
| `lut_off` | uint64 | yes | absolute offset of the f16 codebook, `2*S` f16 = 16 KiB at `L=12` |
| `scale_off` | uint64 | yes | absolute offset of `physical_rows` f16 per-row scales |
| `stream_off` | uint64 | yes | absolute offset of the code stream |
| `stream_bytes` | uint64 | yes | byte length of the stream. Checked against `rows*seqs*4*bits*4` for embeddings only (`src/engine35.cpp:277`) |
| `bias_off` | uint64 | no, default 0 | **parsed and never used.** Biases live in `"vectors"` instead (`src/loader.cpp:83`) |
| `proxy` | float | no, default 0 | **parsed and never used** |
| `rot_in` | object | no | input rotation; see §4. Default `{kind:"H512", m:cols/512, sign_seed:7, sign_offset:0, rule:"mulberry32_7919_13"}` (`src/format.hpp:48-49`) |
| `rot_out` | object\|null | no | output rotation. If the key is absent **and** `role` is neither `"head"` nor `"embedding"`, a default is synthesised: `{kind:"H512", m:rows/512, sign_seed:7, sign_offset:cols, rule:"mulberry32_7919_13"}` (`src/format.hpp:50-51`). `kind:"none"` disables it |
| `rar` | object\|null | no | per-tile bit-width map; see §2.5. Rejected by `Engine35` (`src/engine35.cpp:159`) |

Validation gate (`src/format.hpp:56`): `d==2 && t==256 && 8<=l<=14 && 4<=bits<=8 &&
start in {zero,carry} && cols%256==0`, otherwise the whole container is rejected.

### 1.4 `"vectors"` array

`{name, off, n, dtype}` (`src/format.hpp:21`, parsed at `src/format.hpp:60`). `off` is an
absolute file offset, `n` is the element count, `dtype` defaults to `"f16"`.

dtype values actually required:

* `"f16"` — used by the legacy `Engine` path (`src/loader.cpp:16` copies `n*2` bytes and the
  kernels decode with `h2f`, e.g. `src/post_kernels.cu:41`).
* `"bf16"` — **mandatory for every vector under `arch:"qwen3_5"`** (`src/engine35.cpp:171`).
  Kernels decode with `bf2f(u) = __uint_as_float(u<<16)` (`src/q35_kernels.cu:35`).
* `"i32"` — only for `mtp.head_ids` / `bdraft2.head_sub_ids` (`src/engine35.cpp:1372`,
  `src/engine35.cpp:1658`).

---

## 2. Tensor payloads

### 2.1 What each offset points at

```
lut_off    -> uint16[2*S]      f16 codebook: S entries x 2 values (16 KiB at L=12)
scale_off  -> uint16[rows]     f16 per-row scale
stream_off -> uint32[rows*seqs*wps]   code words, wps = 4*bits, seqs = cols/256
bias_off   -> (unused)
rar.k_off  -> uint8[rows/64]   bits per 64-row tile
rar.off_off-> uint64[rows/64]  word offset of each tile inside the stream
```

Sizes as the reader computes them:

* LUT: `S = 1<<l` entries; `build_lut8_tau` walks `2*S` f16 (`src/loader.cpp:20`), embedding
  upload copies `S*4` bytes = `2*S` f16 (`src/loader.cpp:65`), `Engine35` copies
  `4096*2*2 = 16384` bytes (`src/engine35.cpp:283`).
* Scales: `rows` f16 (`src/loader.cpp:24`, `src/loader.cpp:65`).
* Stream: `nwords = rows * seqs * wps` words (`src/loader.cpp:41`), `stream_bytes = nwords*4`.

### 2.2 The stream word layout

`wps = 4*bits` 32-bit words per (row, 256-column sequence). Each sequence holds 128 trellis
steps (`256 cols / d=2`), each step costing `bits` bits: `128*bits` bits = `4*bits` words.
Hence **bits/weight = bits/2** — `bits:6` is the 3-bit format, `bits:8` the 4-bit one.

Bit `b` of the sequence lives in word `b/32`, bit `b%32`, **LSB-first**
(`src/misc_kernels.cu:10-14`, `src/q35_kernels.cu:1496-1500`):

```c
read_window(w, start, width): v = w[start>>5] >> (start&31);
                              if ((start&31)+width > 32) v |= w[idx+1] << (32-(start&31));
                              return v & ((1<<width)-1);
```

Step `j` (0..127) occupies bits `[bits*j, bits*(j+1))`. The 12-bit trellis state used to index
the codebook for step `j` is the 12-bit window **ending at** bit `bits*(j+1)`, with the
**earliest** bit as the **least** significant one (little-endian inside the window), and
zero-padded at the bottom for the first steps (`src/v2_decode.cuh:13-19`):

```c
e = bits*(j+1);
state = (e < 12) ? (w[0] & ((1<<e)-1)) << (12-e)     // zero-padded prefix
                 : window of bits [e-12, e) ;         // bit e-12 -> state bit 0
```

So for `bits=4`: step 0 → `state = bits[0..3] << 8`; step 1 → `bits[0..7] << 4`; step 2 →
`bits[0..11]`; step 3 → `bits[4..15]`; …; step 127 → `bits[500..511]`. The step emits
`(v[2j], v[2j+1]) = lut[state]`.

### 2.3 `layout:"warp"` — the on-disk interleave

The stream is **not** `[row][seq][word]`. `src/repack.cu:10-18` is the authoritative
description; solving its index arithmetic for the source side gives

```
src = ((tile*(wps/4) + word/4)*32 + lane)*4 + (word%4)
tile = (seq*rows + row)/32 ,  lane = (seq*rows + row)%32
```

i.e. on disk:

```
for seq   in 0..seqs-1                 # 256-column block
  for tile in 0..rows/32-1             # 32 consecutive rows
    for wg  in 0..wps/4-1              # group of 4 words
      for lane in 0..31                # row within the tile
        for i in 0..3 : uint32 word = wg*4+i
```

Each `(lane, wg)` pair is one 16-byte `uint4`. The loader `cudaMemcpy`s the whole stream and
runs `yskn_repack_warp_to_frag` (`src/loader.cpp:42-43`) into the mma-fragment layout

```
frag[tile64][seq][j=0..7][c=0..wps/4-1][r=0..7]   (uint4)
```

(`src/repack.cu:15-16`, consumed at `src/v2_decode.cuh:130-136` and
`src/gemv_sigma_int.cu:338`). The frag layout is **derived at load time** and is not part of
the file format; `Engine35` discards it after building LNORD (`src/engine35.cpp:163-164`).

### 2.4 `layout:"row"` — embedding stream

Plain `[vocab_row][seq][word]`, indexed `stream + id*seqs*wps + q*wps`
(`src/misc_kernels.cu:18-21`, `src/q35_kernels.cu:1503-1505`). `Engine35` requires
`role=="embedding"`, `layout=="row"`, `l==12`, `start=="zero"`, `rot_in.kind=="H512"`,
`has_rot_out==false`, `has_rar==false` and `stream_bytes == vocab*(hidden/256)*4*bits*4`
(`src/engine35.cpp:276-278`). Optional companion tensors `embed_tokens.res` and
`embed_tokens.res2` add further coded tables that are summed in (`src/engine35.cpp:288-298`,
`src/q35_kernels.cu:1508-1517`).

### 2.5 `rar` — per-tile bit widths

`rar.k_off` is `uint8[rows/64]`, `rar.off_off` is `uint64[rows/64]` (`src/loader.cpp:46-53`).
Tile `i` stores 64 rows at `k[i]` bits, so it occupies `64*seqs*4*k[i]` words, and `off[i]`
is its **word** offset relative to `stream_off` (`src/loader.cpp:51`: `words + off[i]`). The
loader recomputes the tile byte offsets cumulatively (`src/loader.cpp:48`) and does **not**
trust `off[]` for layout — `off[]` must agree with the cumulative packing or the wrong data is
read. `rar.tile_rows` is parsed and never used (64 is hardcoded). `Engine35` rejects any
tensor with `has_rar` (`src/engine35.cpp:159`).

---

## 3. Codebook

### 3.1 Shape

`L = 12` → `S = 4096` entries (`src/loader.cpp:29`, `src/v2_decode.cuh:7`). Each entry encodes
`d = 2` weights — one trellis step — as a pair of signed bytes. On disk the entry is two f16
values (`lut16[2s]`, `lut16[2s+1]`); in GPU shared memory it is one `uint16` holding two
`int8` (`src/loader.cpp:23`: `q0 | (q1<<8)`), 8 KiB total (`src/v2_decode.cuh:27-30`,
`src/gemv_sigma_int.cu:25-26`).

Lookup is a byte offset, not an index: `off = (state<<1) & 0x1FFE`, then
`lut_at(off) = *(uint16*)((char*)g_lut8 + off)` (`src/gemv_sigma_int.cu:26-31`,
`src/v2_decode.cuh:46-51`). Two adjacent entries are then OR-ed into a 32-bit word
`a0 | (a1<<16)` to feed `mma.sync.m16n8k32.s8` (`src/v2_decode.cuh:50`,
`src/v2_decode.cuh:112-115`).

### 3.2 `build_lut8_tau` — the exact derivation

`src/loader.cpp:19-25`, verbatim semantics:

```c
mx    = max over i in [0, 2S) of |f16_to_f32(lut16[i])|;
gamma = mx / 127.0f;
for s in [0,S):
    q0 = clamp( nearbyint(f16_to_f32(lut16[2s  ]) / gamma), -127, 127 );
    q1 = clamp( nearbyint(f16_to_f32(lut16[2s+1]) / gamma), -127, 127 );
    lut8[s] = (uint8)q0 | ((uint8)q1 << 8);
for r in [0,rows):
    tau[r] = f16_to_f32(scale16[r]) * gamma;
```

Three consequences that matter for an encoder:

1. **`gamma` is global to the tensor**, derived from the single largest-magnitude LUT entry.
2. The int8 LUT is a *renormalisation* of the f16 LUT, and `tau` absorbs `gamma` so the
   product `lut8[s] * tau[r] ≈ lut16[s] * scale16[r]` — the f16 LUT is the "true" codebook and
   the int8 one is the tensor-core-friendly projection of it.
3. The projection is **exactly invertible** if the encoder keeps `max|q| == 127`: writing
   `lut16[i] = f16(q_i * gamma)` with integer `q_i` reproduces `lut8[s] == (q0,q1)` bit-exactly
   for any `gamma`. The f16 rounding of the LUT perturbs `gamma` by at most ~2^-11 relative,
   i.e. at most `127 * 2^-11 ≈ 0.062` in `q` units — far below the 0.5 `nearbyint` threshold,
   so the recovery is exact. This is the single most useful fact for building a compatible
   encoder (§5.4 step 8).

### 3.3 Which path uses which table

* **Linear tensors** use only `lut8` + `tau` (`src/loader.cpp:31-32`); `lut16` is never
  uploaded. Its only role is to define `gamma` and the int8 codebook.
* **Embeddings** use `lut16` + `scale16` directly, at full f16 precision, with **no** int8
  step and **no** `gamma` (`src/loader.cpp:65`, `src/misc_kernels.cu:23`,
  `src/q35_kernels.cu:1507`). The embedding codebook is therefore genuinely f16-valued.

---

## 4. Rotations

### 4.1 `RotDesc` fields

`{kind, m, sign_seed, sign_offset, qm_off, rule}` (`src/format.hpp:15`, parsed at
`src/format.hpp:38-40`).

| field | meaning |
|---|---|
| `kind` | `"H512"` (signed Hadamard over 512-element blocks) or `"AYA"` (H512 **plus** an `m×m` mixing matrix). `"none"` disables `rot_out` only |
| `m` | number of 512-blocks: `cols/512` for `rot_in`, `rows/512` for `rot_out` (`src/format.hpp:49-51`). Sizes the `qm` matrix |
| `sign_seed` | seed of the ±1 sign stream |
| `sign_offset` | number of generator outputs to skip before the first sign |
| `qm_off` | absolute offset of an `m*m` f16 matrix; read **only** when `kind=="AYA"` (`src/loader.cpp:35`, `src/loader.cpp:67`) |
| `rule` | `"splitmix64"` or `"mulberry32_7919_13"`; anything else throws (`src/signs.hpp:27-31`) |

Sign stream generators (`src/signs.hpp`):

```c
splitmix64:  st = seed; per k: st += 0x9E3779B97F4A7C15; z = mix(st); bit = z>>63 ? +1 : -1
mulberry32:  s = (uint32)seed*7919 + 13 (0 -> 0x9e3779b9);
             per k: t = (s += 0x6d2b79f5); t = (t^(t>>15))*(t|1); t ^= t + (t^(t>>7))*(t|61); t ^= t>>14;
             bit = t < 0x80000000 ? -1 : +1
```

Both consume `offset+n` outputs and discard the first `offset` (`src/signs.hpp:12`,
`src/signs.hpp:21-24`). Note the generators advance **before** emitting, so output 0 already
reflects one state step.

`rot_in` produces `cols` signs (`src/loader.cpp:34`), `rot_out` produces `rows`
(`src/loader.cpp:36`). The default offsets (`0` for in, `cols` for out) imply the shipped
encoder draws both from one continuous stream per tensor.

### 4.2 Is H512 a Hadamard transform? Yes.

`fwht512_blocks` (`src/a8_prologue.cuh:41-51`) is an in-place radix-2 fast Walsh–Hadamard
transform in natural (Sylvester) order over each 512-element block:

```c
for (h = 1; h < 512; h <<= 1)
  for each k in [0,256): lo = ((k & ~(h-1)) << 1) | (k & (h-1));
                         a = x[lo]; b = x[lo+h]; x[lo] = a+b; x[lo+h] = a-b;
then x[i] *= 0.044194173824159216f;   // == 1/sqrt(512)
```

So `H` is the unnormalised `512×512` Hadamard matrix, `H = Hᵀ`, `H² = 512·I`, and the reader
applies `(1/√512)·H`. The same routine is used for the forward and the inverse direction
(`src/a8_prologue.cuh:68`, `src/post_kernels.cu:17`, `src/q35_kernels.cu:75`), which is
consistent because `(1/√512)H · (1/√512)H = I`.

### 4.3 Where each rotation is applied

**Input side (activations, forward).** `a8_from_fp32` (`src/a8_prologue.cuh:64-81`) and the
fused variants (`src/post_kernels.cu:149-151`, `src/post_kernels.cu:173-174`,
`src/post_kernels.cu:189-199`):

```
x  = round_f16( round_f16(h * rms) * gamma )     # RMSNorm, gamma is the layer-norm vector
x *= signs_in[i]                                  # per-element ±1
x  = fwht512_blocks(x, m)                         # (1/sqrt(512)) * H, per 512-block
if (kind=="AYA") x = qm_mix_inplace(x, qm, m, transpose=false)
alpha[g] = max|x in block g| / 127
qx[i]    = clamp(round(x[i]/alpha[g]), -127, 127)
```

**Output side (activations, inverse).** `inv_h512` (`src/post_kernels.cu:15-20`):

```
y  = sum_k qx[k] * alpha[k/512] * W_enc[n][k]     # int8 tensor-core accumulation
y *= tau[n]                                        # per-row scale, PER ELEMENT
                                                 #   (before the transform, not after)
y  = fwht512_blocks(y, m_out)                      # (1/sqrt(512)) * H over output blocks
y *= signs_out[n]
y  = round_f16(h + round_f16(y))                   # residual add
```

`tau` is applied **before** the output Hadamard and is indexed by the output row `n`, so it
scales the whole row uniformly. Since `H` acts within a 512-block of output rows, `tau`
commutes with `H_out` only if `tau` is constant across each 512-block. It is not — `tau` has
one value per row (`src/loader.cpp:24`). **This is the one place where the encoder's algebra
is not a clean orthogonal conjugation**; see §4.4.

The head has no output rotation — the argmax applies `tau` only
(`src/misc_kernels.cu:69-72`), which is why `role:"head"` suppresses the default `rot_out`.

**Embedding.** `src/misc_kernels.cu:23-28`: decode → `scale16[id]` → (if AYA)
`qm_mix_inplace(..., transpose=true)` → `fwht512_blocks` → `*= signs_in`. Note the AYA
matrix is applied **transposed** and **before** the transform here, whereas the linear input
path applies it untransposed **after** the transform (`src/a8_prologue.cuh:69`). Since `qm`
mixes across blocks and `H` acts within blocks the two commute, but the transpose does not —
see §6.

### 4.4 The transform the encoder must invert

Composing §4.3, the reader computes, for output row `n` and input column `k`,

```
y[n] = tau[n] · Σ_m H_out[n][m] · signs_out[m] · Σ_k W_enc[m][k] · alpha[k/512] · qx[k]
```

with `qx[k] = (1/alpha[k/512]) · Σ_j H_in[k][j] · signs_in[j] · x[j]`. Substituting,

```
y[n] = tau[n] · Σ_m H_out[n][m] · signs_out[m] · Σ_k W_enc[m][k] · Σ_j H_in[k][j] · signs_in[j] · x[j]
```

so the effective weight the network sees is

```
W_eff[n][j] = tau[n] · Σ_m Σ_k H_out[n][m] · signs_out[m] · W_enc[m][k] · H_in[k][j] · signs_in[j]
```

In matrix form, with `H_in`/`H_out` block-diagonal `H` over the input/output 512-blocks:

```
W_eff = diag(tau) · H_out · diag(signs_out) · W_enc · diag(signs_in) · H_in
```

Inverting for the encoder:

```
W_enc = diag(1/signs_out) · H_out⁻¹ · diag(1/tau) · W · H_in⁻¹ · diag(1/signs_in)
      = diag(signs_out) · (1/√512)H · diag(1/tau) · W · (1/√512)H · diag(signs_in)
```

using `H⁻¹ = (1/512)H` and `signs² = 1`.

Two consequences that matter:

* The `diag(1/tau)` sits **between** the two Hadamards and is **not** a scalar, so it does not
  commute past `H_out`. The encoder must therefore fix `tau` on the **rotated** domain, not on
  the original weights: define `A = diag(signs_out)·(1/√512)H·W·(1/√512)H·diag(signs_in)`,
  then `tau[n] = max_k |A[n][k]| / 127` and `W_enc = diag(1/tau)·A`. That is exactly what
  §5.4 step 3–4 does (rotate first, then scale). Choosing `tau` from the *unrotated* `W`
  instead would mis-scale every row and destroy accuracy, because the Hadamard mixes each row
  across 512 columns and changes the row max.
* The ±1 signs cost nothing to store (regenerated from `sign_seed`/`sign_offset`/`rule`), and
  `gamma` does **not** appear: the matmul input is the already-normed activation, so the
  layer-norm weight is applied before the rotation and needs no folding.

---

## 5. Encoder requirements

### 5.1 What the source actually specifies

Fully specified, implementable today from the reader alone:

* header, JSON schema, offset semantics (§1);
* bit packing within a sequence, trellis state extraction (§2.2);
* the on-disk interleave (§2.3);
* the f16→int8 codebook projection and `tau` derivation (§3.2);
* the sign generators, exactly (`src/signs.hpp`);
* the Hadamard transform, exactly (`src/a8_prologue.cuh:41-51`);
* the algebra the encoder must invert (§4.4);
* every tensor/vector name, shape and dtype the engines request (§5.4).

### 5.2 What the source does **not** specify

The prose account is `docs/TECHNOLOGY.md:98-110` (Japanese: `docs/TECHNOLOGY.ja.md:91-101`):
collect statistics → rotate → Viterbi → error feedback, ~75 min on one RTX 5090. None of it
is code. Specifically missing:

1. **Codebook training.** No routine anywhere builds `lut16`. It is a learned 4096-entry
   pair codebook; `grep` for `learn|train|optimi|codebook` over `src/` returns nothing. The
   KV5 path has an *encoder* for a *different* codebook (`src/kvq_vit2.cuh`,
   `src/kvq_fast.cuh`) but it encodes 64-value chains against a blob loaded from
   `kvq_u5.bin`, not the weight codebook.
2. **Viterbi cost function.** Which distortion, whether `tau` is optimised jointly, whether
   the search is exact over 4096 states or beam-limited.
3. **Error feedback (LDLQ/GPTQ-style).** `docs/TECHNOLOGY.md:107-109` says error is pushed
   into not-yet-rounded weights along both columns and rows. No damping, blocking or ordering
   is given.
4. **Calibration statistics.** `docs/TECHNOLOGY.md:102-104`: input second-order statistics
   and output-side gradient statistics, from 96 WikiText windows + 96 self-generated thinking
   windows. The exact estimator, the target tokens, and how the two statistics are combined
   into a per-weight importance are all absent.
5. **`tau` normalisation / `scale16` quantisation.** `tau[r]` is realised as
   `f16(scale16[r]) * gamma` (`src/loader.cpp:24`); the encoder must choose the split and
   round to f16. No policy given. Note the reader applies `tau` *before* the output Hadamard
   (`src/post_kernels.cu:104`), so `tau` is not a free post-hoc rescale — see §4.4.
6. **Per-tensor `bits` allocation.** `docs/TECHNOLOGY.md:89-94` gives the shipped split
   (quality model: 288 matrices at 4 bits, 112 at 3; fastest: all 400 at 3), but not the rule
   that produced it.
7. **Sign seeds.** No policy for `sign_seed`/`sign_offset` per tensor.
8. **The `qm` (AYA) matrices.** No generator, and no statement of which tensors use AYA.

### 5.3 Bit-exact encoder: verdict

**Not derivable from this source alone.** A bit-exact reproduction of a shipped `.yskn` would
require (a) the same codebook, (b) the same calibration corpus and statistics, (c) the same
Viterbi tie-breaking, (d) the same error-feedback ordering, and (e) the same per-tensor bit
allocation and seeds. None of (a)–(e) is recoverable from the reader, and the model files are
not in the tree. Bit-exactness is only achievable by shipping the converter.

### 5.4 Compatible (not bit-identical) encoder: what it needs

This is achievable. The reader is a pure decoder with no integrity checks beyond the shape
gate at `src/format.hpp:56`, so any self-consistent container loads and runs.

**A. Container writer.** Emit the 32-byte header, the JSON, then payloads at naturally
aligned absolute offsets. Required JSON keys per §1.2–1.4. For `arch:"qwen3_5"` also emit
`layer_types`.

**B. Tensor inventory** (`src/engine35.cpp:237-300`), names exactly as requested:

```
per layer L:
  full_attention layer:  layers.L.self_attn.{q,k,v,o}_proj        (tensors)
                         layers.L.self_attn.q_norm, k_norm        (vectors, 256 bf16)
  linear-attention layer: layers.L.linear_attn.in_proj_qkv        (tensor)
                          layers.L.linear_attn.in_proj_z          (tensor)
                          layers.L.linear_attn.out_proj           (tensor)
                          layers.L.linear_attn.in_proj_a, in_proj_b (vectors, 48*5120 bf16)
                          layers.L.linear_attn.conv1d             (vector, 4*10240 bf16, [channel][tap])
                          layers.L.linear_attn.A_log, dt_bias     (vectors, 48 bf16)
                          layers.L.linear_attn.norm               (vector, 128 bf16)
  every layer:            layers.L.mlp.{gate,up,down}_proj        (tensors)
                          layers.L.input_layernorm, post_attention_layernorm (vectors, 5120 bf16)
global:  lm_head (tensor, role "head"), embed_tokens (tensor, role "embedding", layout "row")
         or a plain bf16 vector, norm (vector, 5120 bf16)
```

**C. Norm-weight convention.** `qwen3_5` kernels compute `(1 + bf16_stored)` for every
RMSNorm weight (`src/q35_kernels.cu:158`, `src/q35_kernels.cu:204`,
`src/q35_kernels.cu:624`, `src/q35_kernels.cu:1046`, `src/q35_kernels.cu:1473`). That matches
the Qwen3-Next/Qwen3.5 `RMSNorm` whose parameter is a delta from 1, so the encoder stores the
checkpoint parameter verbatim. The legacy `Engine` path uses the weight directly
(`src/misc_kernels.cu:42`), so the two conventions differ — do not cross them.

**D. Per-matrix pipeline** (the only defensible reading of the code):

1. Load `W` (bf16/f16) as `[rows][cols]`, `cols % 256 == 0`, pad `rows` to a multiple of 64.
2. Generate `signs_in = sign_stream(rot_in.rule, rot_in.sign_seed, rot_in.sign_offset, cols)`
   and `signs_out = sign_stream(rot_out.rule, rot_out.sign_seed, rot_out.sign_offset, rows)`.
3. `A = diag(signs_out)·(1/√512)H·W·(1/√512)H·diag(signs_in)` — apply the output signs and
   the output Hadamard over each 512-row block, then the input Hadamard over each 512-column
   block and the input signs. Do it in f32. This is the inverse of §4.4.
4. Choose a per-row scale **on the rotated matrix**: `tau[r] = max_k |A[r][k]| / 127` (or an
   MSE-optimal scale), then `B[r][k] = A[r][k]/tau[r]`. Because `tau` is applied before the
   output Hadamard in the reader (`src/post_kernels.cu:104`), it must be derived here, not
   from the unrotated `W`.
5. Design a 4096-entry pair codebook `q[s] = (q0,q1) ∈ [-127,127]²` with `max|q| == 127`
   (e.g. k-means over the pairs `(B[r][2t], B[r][2t+1])`).
6. For each row and each 256-column sequence, run Viterbi over the 4096-state trellis with
   transition `state' = ((state<<bits) | b) & 0xFFF`, cost
   `(B[r][2t] - q0[s'])² + (B[r][2t+1] - q1[s'])²`, and emit the winning `bits` per step.
   Seed step 0 with the zero-padded prefix state of §2.2.
7. Pack the 128·`bits` bits per sequence LSB-first, then interleave to `layout:"warp"` (§2.3).
8. Emit `lut16[s] = f16(q0[s]*gamma, q1[s]*gamma)` for a fixed `gamma` (e.g. `1/127`), and
   `scale16[r] = f16(tau[r]/gamma)`. Because `max|q| == 127`, `build_lut8_tau` recovers
   `gamma` and `q` exactly (§3.2), so the GPU's int8 codebook is bit-identical to the one the
   Viterbi searched against — the encoder never has to guess the reader's `gamma`.
9. Set `stream_bytes = rows*seqs*4*bits*4`, `l=12`, `d=2`, `t=256`, `start="zero"`,
   `layout="warp"`, and `bits ∈ {4,6,8}` (= 2, 3, 4 bit/weight) per the allocation policy.

**E. Embedding pipeline.** Same, but `layout:"row"`, `role:"embedding"`, no `rot_out`, and
invert the read order of `src/misc_kernels.cu:23-28`: `e_enc = (1/√512)·H·(signs_in ⊙ e)`
then divide by `scale16[id]` before quantising. The LUT is used as f16, so no int8
projection applies.

**F. What "compatible" buys and costs.** A container built this way loads and runs. Its
quality will be worse than the shipped models because the calibration statistics, the
codebook training and the error feedback in `docs/TECHNOLOGY.md:102-109` are the parts that
actually buy the quality, and none of them are reproduced. Expect a working but noticeably
lossier model.

### 5.5 Minimum viable validation loop

There is no writer to diff against, but the reader can be used as an oracle:
`yskn_repack_warp_to_frag_host` (`src/repack.cu:27-32`) exposes the interleave on the host,
and `Container::open` (`src/format.hpp:26`) validates the header/JSON. A round-trip test can
decode a written stream with the same 12-bit-window arithmetic as `src/v2_decode.cuh:13-19`
and compare to the Viterbi output before ever touching a GPU.

---

## 6. Open questions and ambiguities

1. **`bits` semantics — resolved, but note the `lnord_build` restriction.** `bits` is bits
   per trellis step, so bits/weight = `bits/2` and the shipped 3- and 4-bit models are
   `bits:6` and `bits:8` (`docs/TECHNOLOGY.md:83` says exactly this: a step writes 6 or 8
   bits). `src/format.hpp:56` allows `4..8`, and the GEMV path instantiates `K=4..8`
   (`src/gemv_sigma_int.cu:362-367`), but **`lnord_build` accepts only `K ∈ {4,6,8}`**
   (`src/lnord.cu:82-86`) and `Engine35` always routes through LNORD
   (`src/engine35.cpp:161-164`). So `bits ∈ {5,7}` (2.5 / 3.5 bit-per-weight) parse and load
   but throw at LNORD build. **Use only 4, 6 or 8 for `arch:"qwen3_5"`.**
2. **`start:"carry"` is dead code.** `v2_state_carry` (`src/v2_decode.cuh:21-24`) exists but
   is never called; no kernel passes a carry tail. `Engine35` requires `start=="zero"` for
   embeddings (`src/engine35.cpp:276`). Treat `"carry"` as unimplemented.
3. **`bias_off` and `proxy` are parsed and never used** (`src/format.hpp:53` is the only
   reference). Biases come from `"vectors"`. Whether shipped files populate them is unknown.
4. **`rar.tile_rows` is parsed and never used**; 64 is hardcoded (`src/loader.cpp:48`).
5. **AYA transpose asymmetry.** The embedding applies `qm` transposed and pre-transform
   (`src/misc_kernels.cu:26`), the linear input path applies it untransposed and
   post-transform (`src/a8_prologue.cuh:69`). Either two different matrices are stored, or one
   of the two paths is only exercised with a symmetric `qm`. No shipped `qm` is available to
   check.
6. **Codebook sharing.** `lut_off` is per tensor, so per-tensor codebooks are legal. Whether
   the shipped models share one codebook across tensors (or across the 400 matrices) is not
   observable from the reader. It changes encoder design (one global k-means vs 400).
7. **`physical_rows` semantics when `> rows`.** The stream and `tau` are sized by
   `physical_rows` (`src/loader.cpp:29`), but downstream post-processing indexes by the
   *logical* width (`src/post_kernels.cu:38` uses the literal 3584; the argmax uses
   `cfg.vocab`, `src/misc_kernels.cu:67-72`). Padding therefore looks safe only where the
   logical width already equals the physical width. For the shipped model everything divides
   evenly (248320 and 5120 are multiples of 64), so this path is untested.
8. **`gamma` coupling.** `tau` depends on `gamma`, which depends on the LUT, which depends on
   the quantisation. The encoder must fix the codebook before it can fix `scale16`. The
   workaround in §5.4 step 8 removes the circularity but is an inference, not documented
   behaviour.
9. **`stream_bytes` is only validated for embeddings** (`src/engine35.cpp:277`). For linear
   tensors the reader trusts `bits`; a wrong `stream_bytes` is silently accepted.
10. **`tau`/Hadamard ordering is a real design constraint, not an ambiguity.** Because `tau`
    is per-row and applied before `H_out` (§4.4), the encoder cannot treat `tau` as an
    independent post-scale. Whether the shipped converter iterates this to convergence, or
    folds `tau` into the codebook selection, is not stated anywhere.
11. **`layer_types` is undocumented.** It is required by `Engine35` but absent from
    `ModelDesc` and from every doc in `docs/`. Its exact accepted vocabulary is only
    `"full_attention"` vs "anything else" (`src/engine35.cpp:239`).
12. **KV profile blob.** `kvq_u5.bin` is a separate file outside the container
    (`python/yasakani/runtime.py:116`), parsed by `tbf_parse_blob` (`src/kvq_fast.hpp:46`).
    Its format is a second, independent encoder problem and is out of scope here.
13. **`src/gguf/gguf.hpp` is a reader only.** It parses GGUF v2/v3 and dequantises to f32
    (`src/gguf/gguf.hpp:151-196`); it contains no YSKN writer and no quantiser. It is the
    input half of a converter, not the output half.

---

## 7. What the published encoder corrected

Upstream v0.1.2 shipped `tools/convert/yskn_convert.py`. Reading it against the inferences in
section 5 gives a clean scoreboard.

### Confirmed

| This document | The encoder |
|---|---|
| 2.2 `kSeqCols = 256`, 128 steps | `T, STEPS, STATE_BITS = 256, 128, 12` |
| 4 Hadamard scaled by `1/sqrt(512)` | `SCALE512 = 0.04419417382415922` |
| 4.1 signs applied inside the transform | `fwht512_rows(x, s)`: `(x * s)` through the butterflies |
| 4.4 `rotate` is not an involution; the inverse mirrors the order | `inverse_rotate(q, si, so)` does exactly that |
| 2.2 the 12-bit window ends at the current step, newest symbol in the high bits | `decode_block` builds `state = (sym << (STATE_BITS - K)) | (prev >> (2K - STATE_BITS))` |
| 5.4 step 8 `max|q| == 127` keeps `build_lut8_tau` invertible | `row_scale16` feeds the same `lut16`/`scale16` split |
| 5.2 open question 1: only certain K values are legal | `--bits` accepts only **6** and **8**, not 4 |

### Corrected

**The per-row scale is the RMS, not the max.** Section 5.4 step 4 guessed
`tau[r] = max_k |A[r][k]| / 127`. The encoder uses the root mean square of the row:

```python
def row_scale16(W):
    "The fp16 scale of every row: the root mean square of the row (fp64), converted to fp16."
    return torch.sqrt((W.double() ** 2).mean(1)).to(torch.float16)
```

RMS scaling leaves the bulk of the distribution near the codebook centre instead of pinning the
largest outlier to 127, which is part of why the released models quantise better at the same bit
count. It stays self-consistent either way because `tau` is stored in the file.

There is a rounding subtlety worth knowing: `row_scale16_direct` exists because PyTorch rounds
fp64 to fp16 through fp32 while numpy rounds in one step, and the two differ for about one row in
10,000. The embedding and output matrices of the released files used the one-step form, the body
the PyTorch form.

**Error feedback is GPTQ/LDLQ, not a plain Viterbi.** Section 5.4 treated the quantiser as
independent per 256-column block, with the trellis constraint as the only coupling. The encoder
carries a Hessian and pushes each block's error into the columns that follow:

```python
def upper_factor_of_inverse(H64, damp):
    "C upper triangular with (H + damp * mean(diag H) * I)^-1 = C^T C."
    ...
def feedback_tables(Cb, taps):
    "The 2 x 2 feedback matrices of one block of 256 columns, from the factor C of that block."
```

So the pieces listed as missing in section 5.2 were recoverable after all: the calibration
statistics are a damped block-diagonal Hessian per 256-column block, factored by Cholesky, applied
through `taps` steps of feedback. A container built the way section 5.4 describes loads and runs,
but it leaves this on the table, which is the concrete reason its output is lossier than the
released files rather than merely different.

### What this means for this fork

`src/gguf/gguf_quant.cpp` implements the section 5.4 pipeline: Hadamard rotation, k-means codebook
with the `max|q| == 127` invariant, trellis-aware refinement, Viterbi, warp interleave. It produces
a container the engine loads and runs. It does not implement Hessian feedback, so its output is
compatible but not bit-identical, and lossier than the reference encoder's, as section 5.4
predicted for a pipeline without calibration statistics.

Adopting the real encoder would mean porting `yskn_convert.py`'s Hessian path, which is a larger
job than the GGUF reader and outside what this fork set out to do.
