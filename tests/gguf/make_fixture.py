#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Generate a small synthetic GGUF v3 fixture plus its expected values.

Writes:
  <out>.gguf   -- the container
  <out>.json   -- expected metadata / shapes / dequantized values

Pure standard library (struct only). numpy is deliberately not imported so the
generator runs anywhere Python 3 does.

The quantized tensors are built by hand from the block layouts in
llama.cpp's ggml-common.h, and the expected float values are computed here
independently of the C++ reader, so a bug in either side shows up as a
mismatch rather than cancelling out.
"""

import argparse
import json
import math
import struct
import sys

# --- ggml types -------------------------------------------------------------
F32, F16, BF16 = 0, 1, 30
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 = 2, 3, 6, 7, 8
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
IQ4_NL = 20

QK = 32
QK_K = 256

# --- GGUF metadata value types ---------------------------------------------
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12


# --- little-endian primitives ----------------------------------------------
def u8(v):
    return struct.pack("<B", v & 0xFF)


def i8(v):
    return struct.pack("<b", v)


def u16(v):
    return struct.pack("<H", v & 0xFFFF)


def i16(v):
    return struct.pack("<h", v)


def u32(v):
    return struct.pack("<I", v & 0xFFFFFFFF)


def i32(v):
    return struct.pack("<i", v)


def u64(v):
    return struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF)


def i64(v):
    return struct.pack("<q", v)


def f32(v):
    return struct.pack("<f", v)


def f64(v):
    return struct.pack("<d", v)


def gguf_str(s):
    b = s.encode("utf-8")
    return u64(len(b)) + b


def fp16_bytes(v):
    """Encode a float as IEEE-754 binary16 (exact for the values we use)."""
    return struct.pack("<e", v)


def fp16_to_f32(raw):
    """Decode two raw bytes as binary16 -> Python float."""
    return struct.unpack("<e", raw)[0]


def bf16_bytes(v):
    """Encode a float as bfloat16 by rounding the float32 bit pattern."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    # round-to-nearest-even on the 16 discarded bits
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def bf16_to_f32(raw):
    hi = struct.unpack("<H", raw)[0]
    return struct.unpack("<f", struct.pack("<I", hi << 16))[0]


# --- quantized block builders ----------------------------------------------
def build_q8_0(values):
    """block_q8_0 { fp16 d; int8 qs[32]; }  x = d*q, values pre-quantized."""
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        # Pick a scale that keeps every quant inside int8.
        peak = max(abs(v) for v in block) or 1.0
        d = peak / 127.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]  # quantize the scale
        qs = [max(-128, min(127, int(round(v / d16)))) for v in block]
        out += fp16_bytes(d16)
        out += b"".join(i8(q) for q in qs)
        expected.extend(q * d16 for q in qs)
    return bytes(out), expected


def build_q4_0(values):
    """block_q4_0 { fp16 d; uint8 qs[16]; }  x = d*(q-8), q in 0..15."""
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        peak = max(abs(v) for v in block) or 1.0
        d = peak / 7.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        q = [max(0, min(15, int(round(v / d16)) + 8)) for v in block]
        out += fp16_bytes(d16)
        # low nibbles = elements 0..15, high nibbles = elements 16..31
        for j in range(16):
            out += u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        expected.extend((qi - 8) * d16 for qi in q)
    return bytes(out), expected


def build_q4_1(values):
    """block_q4_1 { fp16 d; fp16 m; uint8 qs[16]; }  x = d*q + m."""
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        lo, hi = min(block), max(block)
        if hi == lo:
            hi = lo + 1.0
        d = (hi - lo) / 15.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        m16 = struct.unpack("<e", struct.pack("<e", lo))[0]
        q = [max(0, min(15, int(round((v - m16) / d16)))) for v in block]
        out += fp16_bytes(d16) + fp16_bytes(m16)
        for j in range(16):
            out += u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        expected.extend(qi * d16 + m16 for qi in q)
    return bytes(out), expected


def build_q5_0(values):
    """block_q5_0 { fp16 d; uint8 qh[4]; uint8 qs[16]; }  x = d*(q-16)."""
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        peak = max(abs(v) for v in block) or 1.0
        d = peak / 15.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        q = [max(0, min(31, int(round(v / d16)) + 16)) for v in block]
        qh = 0
        for j in range(32):
            if q[j] & 0x10:
                qh |= 1 << j  # 5th bit of element j
        out += fp16_bytes(d16) + u32(qh)
        for j in range(16):
            out += u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        expected.extend((qi - 16) * d16 for qi in q)
    return bytes(out), expected


def build_q5_1(values):
    """block_q5_1 { fp16 d; fp16 m; uint8 qh[4]; uint8 qs[16]; }  x = d*q+m."""
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        lo, hi = min(block), max(block)
        if hi == lo:
            hi = lo + 1.0
        d = (hi - lo) / 31.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        m16 = struct.unpack("<e", struct.pack("<e", lo))[0]
        q = [max(0, min(31, int(round((v - m16) / d16)))) for v in block]
        qh = 0
        for j in range(32):
            if q[j] & 0x10:
                qh |= 1 << j
        out += fp16_bytes(d16) + fp16_bytes(m16) + u32(qh)
        for j in range(16):
            out += u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        expected.extend(qi * d16 + m16 for qi in q)
    return bytes(out), expected


def pack_scales_min_k4(scales, mins):
    """Pack 8 six-bit scales and 8 six-bit mins into the 12 bytes of Q4_K/Q5_K.

    Layout (ggml get_scale_min_k4): q[0..3] hold the low 6 bits of scales 0..3,
    q[4..7] the low 6 bits of mins 0..3, q[8..11] the two high bits of scale j
    in bits 0..1 and of min j in bits 4..5 (j = index-4).
    """
    assert len(scales) == 8 and len(mins) == 8
    q = [0] * 12
    for j in range(4):
        q[j] = scales[j] & 0x3F
        q[j + 4] = mins[j] & 0x3F
    for j in range(4, 8):
        q[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4)
        q[j - 4] |= (scales[j] >> 4) << 6
        q[j] |= (mins[j] >> 4) << 6
    return bytes(q)


def build_q4_K(values):
    """block_q4_K { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qs[128]; }

    x = d*sc*q - dmin*m with q in 0..15, sc and m in 0..63. Both d and dmin are
    fixed fp16-exact powers of two so the expected values can be reproduced
    without any rounding surprises; per sub-block we then pick the (sc, m) that
    keeps every quant inside 0..15.
    """
    assert len(values) % QK_K == 0
    d16 = 0.015625   # 1/64, exact in fp16
    dmin16 = 0.015625
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        sub = [block[i * 32:(i + 1) * 32] for i in range(8)]
        scales6, mins6 = [], []
        qs = []
        for s in sub:
            lo, hi = min(s), max(s)
            # The min covers the sub-block's floor; the scale must then span
            # the remaining range within 15 steps.
            m = max(0, min(63, int(round(-lo / dmin16))))
            span = hi + m * dmin16
            scale = max(1, min(63, int(math.ceil(span / (15.0 * d16)))))
            mins6.append(m)
            scales6.append(scale)
            q = [max(0, min(15, int(round((v + m * dmin16) / (scale * d16)))))
                 for v in s]
            qs.append(q)
        out += fp16_bytes(d16) + fp16_bytes(dmin16)
        out += pack_scales_min_k4(scales6, mins6)
        for i in range(0, 8, 2):
            # 32 bytes hold sub-block i in the low nibbles and i+1 in the high.
            for l in range(32):
                out += u8((qs[i][l] & 0xF) | ((qs[i + 1][l] & 0xF) << 4))
        for i, s in enumerate(sub):
            dl = d16 * scales6[i]
            ml = dmin16 * mins6[i]
            expected.extend(q * dl - ml for q in qs[i])
    return bytes(out), expected


def build_q6_K(values):
    """block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d; }"""
    assert len(values) % QK_K == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        sub = [block[i * 16:(i + 1) * 16] for i in range(16)]
        peak = max(max(abs(v) for v in s) for s in sub) or 1.0
        d = peak / (31.0 * 127.0)
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        scales8, qs = [], []
        for s in sub:
            peak_s = max(abs(v) for v in s) or 1.0
            sc = max(1, min(127, int(round(peak_s / (31.0 * d16)))))
            scales8.append(sc)
            q = [max(0, min(63, int(round(v / (d16 * sc))) + 32)) for v in s]
            qs.append(q)
        ql_full = bytearray(128)
        qh_full = bytearray(64)
        # Each 128-element half covers sub-blocks 8*half .. 8*half+7. Within a
        # half, sub-block 2k occupies the low nibbles of ql[0..31] (k=0) or
        # ql[32..63] (k=1), and the high 2 bits of the four sub-blocks that
        # share a qh byte sit at bit offsets 0, 2, 4, 6.
        for half in range(2):
            for l in range(32):
                # ggml: is = l/16, and the four values come from sc[is+0],
                # sc[is+2], sc[is+4], sc[is+6] within this half.
                i0 = half * 8 + (l // 16)
                q1, q2 = qs[i0][l % 16], qs[i0 + 2][l % 16]
                q3, q4 = qs[i0 + 4][l % 16], qs[i0 + 6][l % 16]
                ql_full[half * 64 + l] = (q1 & 0xF) | ((q3 & 0xF) << 4)
                ql_full[half * 64 + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
                qh_full[half * 32 + l] = ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | \
                                         (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6)
        out += bytes(ql_full) + bytes(qh_full)
        out += b"".join(i8(s) for s in scales8)
        out += fp16_bytes(d16)
        for i, s in enumerate(sub):
            expected.extend((q - 32) * d16 * scales8[i] for q in qs[i])
    return bytes(out), expected


def build_q2_K(values):
    """block_q2_K { uint8 scales[16]; uint8 qs[64]; fp16 d; fp16 dmin; }

    x = d*(sc&0xF)*q - dmin*(sc>>4), q in 0..3, 16 sub-blocks of 16 values.
    """
    assert len(values) % QK_K == 0
    d16 = 0.25      # exact in fp16
    dmin16 = 0.25
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        sub = [block[i * 16:(i + 1) * 16] for i in range(16)]
        scales = bytearray(16)
        qs = bytearray(64)
        qall = []
        for i, s in enumerate(sub):
            lo, hi = min(s), max(s)
            m = max(0, min(15, int(round(-lo / dmin16))))
            ml = dmin16 * m
            span = hi + ml
            sc = max(1, min(15, int(math.ceil(span / (3.0 * d16)))))
            scales[i] = (sc & 0xF) | ((m & 0xF) << 4)
            qall.append([max(0, min(3, int(round((v + ml) / (d16 * sc)))))
                         for v in s])
        # 2-bit planes: in each 128-element half, the 32 bytes of qs carry four
        # bit planes (shift = 2j). Bytes 0..15 hold sub-block 2j, bytes 16..31
        # hold sub-block 2j+1, both in the same plane.
        for half in range(2):
            for j in range(4):
                for l in range(32):
                    if l < 16:
                        v = qall[half * 8 + j * 2][l]
                    else:
                        v = qall[half * 8 + j * 2 + 1][l - 16]
                    qs[half * 32 + l] |= (v & 3) << (2 * j)
        out += bytes(scales) + bytes(qs) + fp16_bytes(d16) + fp16_bytes(dmin16)
        for i, s in enumerate(sub):
            sc = scales[i] & 0xF
            m = scales[i] >> 4
            expected.extend(q * d16 * sc - dmin16 * m for q in qall[i])
    return bytes(out), expected


def pack_q3_K_scales(six_bit):
    """Pack 16 six-bit scales into the 12 bytes of block_q3_K.

    Inverse of ggml's unpacking: a0/a1 hold low nibbles (a0 for scales 0..3 and
    8..11, a1 for 4..7 and 12..15) and a2 holds the two high bits of each.
    """
    assert len(six_bit) == 16
    a = [0, 0, 0]
    for i, s in enumerate(six_bit):
        s &= 0x3F
        lo, hi = s & 0xF, (s >> 4) & 3
        if i < 4:
            a[0] |= lo << (8 * i)
            a[2] |= hi << (8 * i)
        elif i < 8:
            a[1] |= lo << (8 * (i - 4))
            a[2] |= hi << (8 * (i - 4) + 2)
        elif i < 12:
            a[0] |= lo << (8 * (i - 8) + 4)
            a[2] |= hi << (8 * (i - 8) + 4)
        else:
            a[1] |= lo << (8 * (i - 12) + 4)
            a[2] |= hi << (8 * (i - 12) + 6)
    return u32(a[0]) + u32(a[1]) + u32(a[2])


def build_q3_K(values):
    """block_q3_K { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; fp16 d; }

    x = d*(sc-32)*q with q in -4..3; the 3rd bit is stored inverted in hmask.
    """
    assert len(values) % QK_K == 0
    d16 = 0.0625    # exact in fp16
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        sub = [block[i * 16:(i + 1) * 16] for i in range(16)]
        mult = []
        qall = []
        for s in sub:
            peak = max(abs(v) for v in s) or 1.0
            m = max(1, min(31, int(math.ceil(peak / (3.0 * d16)))))
            mult.append(m)
            qall.append([max(-4, min(3, int(round(v / (d16 * m))))) for v in s])
        hmask = bytearray(32)
        qs = bytearray(64)
        # hm does not advance between the two 128-element halves; the bit mask
        # m keeps shifting (1,2,4,8 then 16,32,64,128), so the byte index is
        # always l in 0..31 while qs advances by 32 per half.
        for half in range(2):
            for j in range(4):
                bit = 1 << (half * 4 + j)
                for l in range(32):
                    if l < 16:
                        q = qall[half * 8 + j * 2][l]
                    else:
                        q = qall[half * 8 + j * 2 + 1][l - 16]
                    v = q + 4                      # 0..7
                    qs[half * 32 + l] |= (v & 3) << (2 * j)
                    if (v >> 2) & 1:
                        hmask[l] |= bit
        out += bytes(hmask) + bytes(qs)
        out += pack_q3_K_scales([m + 32 for m in mult])
        out += fp16_bytes(d16)
        for i, s in enumerate(sub):
            expected.extend((q) * d16 * mult[i] for q in qall[i])
    return bytes(out), expected


def build_q5_K(values):
    """block_q5_K { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }"""
    assert len(values) % QK_K == 0
    d16 = 0.015625
    dmin16 = 0.015625
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        sub = [block[i * 32:(i + 1) * 32] for i in range(8)]
        scales6, mins6, qall = [], [], []
        for s in sub:
            lo, hi = min(s), max(s)
            m = max(0, min(63, int(round(-lo / dmin16))))
            span = hi + m * dmin16
            sc = max(1, min(63, int(math.ceil(span / (31.0 * d16)))))
            mins6.append(m)
            scales6.append(sc)
            qall.append([max(0, min(31, int(round((v + m * dmin16) / (sc * d16)))))
                         for v in s])
        qh = bytearray(32)
        qs = bytearray(128)
        for g in range(4):
            u1 = 1 << (2 * g)
            u2 = 2 << (2 * g)
            for l in range(32):
                lo_q = qall[g * 2][l]
                hi_q = qall[g * 2 + 1][l]
                qs[g * 32 + l] = (lo_q & 0xF) | ((hi_q & 0xF) << 4)
                if lo_q & 0x10:
                    qh[l] |= u1
                if hi_q & 0x10:
                    qh[l] |= u2
        out += fp16_bytes(d16) + fp16_bytes(dmin16)
        out += pack_scales_min_k4(scales6, mins6)
        out += bytes(qh) + bytes(qs)
        for i, s in enumerate(sub):
            dl = d16 * scales6[i]
            ml = dmin16 * mins6[i]
            expected.extend(q * dl - ml for q in qall[i])
    return bytes(out), expected


def build_q8_K(values):
    """block_q8_K { float d; int8 qs[256]; int16 bsums[16]; }  x = d*q"""
    assert len(values) % QK_K == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK_K):
        block = values[base:base + QK_K]
        d = 0.015625    # exact in fp32
        qs = [max(-128, min(127, int(round(v / d)))) for v in block]
        out += f32(d) + b"".join(i8(q) for q in qs)
        # bsums: sum of every 16 consecutive quants, a dot-product helper.
        for i in range(16):
            out += i16(sum(qs[i * 16:(i + 1) * 16]))
        expected.extend(q * d for q in qs)
    return bytes(out), expected


def build_iq4_nl(values):
    """block_iq4_nl { fp16 d; uint8 qs[16]; }  x = d * kvalues[q]."""
    kv = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    assert len(values) % QK == 0
    out = bytearray()
    expected = []
    for base in range(0, len(values), QK):
        block = values[base:base + QK]
        peak = max(abs(v) for v in block) or 1.0
        d = peak / 127.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        q = []
        for v in block:
            target = v / d16
            q.append(min(range(16), key=lambda k: abs(kv[k] - target)))
        out += fp16_bytes(d16)
        for j in range(16):
            out += u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        expected.extend(kv[qi] * d16 for qi in q)
    return bytes(out), expected


# --- tensor payload helpers -------------------------------------------------
def build_f32(values):
    return b"".join(f32(v) for v in values), [float(v) for v in values]


def build_f16(values):
    out = bytearray()
    exp = []
    for v in values:
        raw = fp16_bytes(v)
        out += raw
        exp.append(fp16_to_f32(raw))
    return bytes(out), exp


def build_bf16(values):
    out = bytearray()
    exp = []
    for v in values:
        raw = bf16_bytes(v)
        out += raw
        exp.append(bf16_to_f32(raw))
    return bytes(out), exp


# --- GGUF writer ------------------------------------------------------------
def encode_value(vtype, value):
    """Encode one metadata value. Arrays carry (elem_type, [values])."""
    if vtype == T_UINT8:
        return u8(value)
    if vtype == T_INT8:
        return i8(value)
    if vtype == T_UINT16:
        return u16(value)
    if vtype == T_INT16:
        return i16(value)
    if vtype == T_UINT32:
        return u32(value)
    if vtype == T_INT32:
        return i32(value)
    if vtype == T_FLOAT32:
        return f32(value)
    if vtype == T_BOOL:
        return u8(1 if value else 0)
    if vtype == T_STRING:
        return gguf_str(value)
    if vtype == T_UINT64:
        return u64(value)
    if vtype == T_INT64:
        return i64(value)
    if vtype == T_FLOAT64:
        return f64(value)
    if vtype == T_ARRAY:
        elem_type, items = value
        body = b"".join(encode_value(elem_type, it) for it in items)
        return u32(elem_type) + u64(len(items)) + body
    raise ValueError("bad value type %r" % (vtype,))


def write_gguf(path, version, alignment, kv, tensors):
    """kv: list of (key, value_type, value); tensors: list of dicts."""
    buf = bytearray()
    buf += b"GGUF"
    buf += u32(version)
    buf += u64(len(tensors))
    buf += u64(len(kv))
    for key, vtype, value in kv:
        buf += gguf_str(key)
        buf += u32(vtype)
        buf += encode_value(vtype, value)
    # tensor info table
    for t in tensors:
        buf += gguf_str(t["name"])
        buf += u32(len(t["shape"]))
        for d in t["shape"]:
            buf += u64(d)
        buf += u32(t["type"])
        buf += u64(t["offset"])
    # pad to the alignment boundary
    pad = (-len(buf)) % alignment
    buf += b"\x00" * pad
    data_base = len(buf)
    # payloads, each padded to the alignment
    for t in tensors:
        assert len(buf) - data_base == t["offset"], \
            "offset mismatch for %s: %d vs %d" % (
                t["name"], len(buf) - data_base, t["offset"])
        buf += t["payload"]
        pad = (-len(buf)) % alignment
        buf += b"\x00" * pad
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return data_base


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="gguf_fixture.gguf",
                    help="output .gguf path (a .json sidecar is written next to it)")
    ap.add_argument("--version", type=int, default=3, choices=(2, 3),
                    help="GGUF version to emit")
    ap.add_argument("--alignment", type=int, default=32,
                    help="general.alignment to emit")
    args = ap.parse_args(argv)

    alignment = args.alignment

    # ---- tensors ----------------------------------------------------------
    # Values are chosen to be exactly representable where possible so the
    # expected numbers stay readable.
    f32_vals = [0.0, 1.0, -2.5, 3.25, -0.125, 7.0, -8.5, 1.5]
    f16_vals = [0.5, -1.25, 2.0, 0.0625, -3.5, 4.75, -0.25, 6.0]
    bf16_vals = [1.0, -2.0, 3.5, 0.25, -0.5, 8.0, -16.0, 0.125]

    # Q8_0: 64 elements = 2 blocks
    q8_vals = [((i * 7) % 31) - 15 + 0.5 * (i % 3) for i in range(64)]
    # Q4_0: 32 elements = 1 block
    q4_0_vals = [((i * 5) % 17) - 8 + 0.25 * (i % 2) for i in range(32)]
    # Q4_1: 32 elements
    q4_1_vals = [((i * 3) % 11) - 5 + 0.5 for i in range(32)]
    # Q5_0 / Q5_1: 32 elements each
    q5_0_vals = [((i * 9) % 41) - 20 + 0.25 for i in range(32)]
    q5_1_vals = [((i * 13) % 29) - 14 + 0.75 for i in range(32)]
    # Q2_K / Q3_K: 256 elements = 1 super-block each
    q2_k_vals = [math.sin(i * 0.31) * 1.5 + ((i % 5) - 2) * 0.25 for i in range(256)]
    q3_k_vals = [math.cos(i * 0.21) * 1.75 + ((i % 9) - 4) * 0.125 for i in range(256)]
    # Q4_K: 256 elements = 1 super-block
    q4_k_vals = [math.sin(i * 0.25) * 3.0 + ((i % 7) - 3) * 0.5 for i in range(256)]
    # Q5_K: 256 elements = 1 super-block
    q5_k_vals = [math.sin(i * 0.13) * 5.0 + ((i % 11) - 5) * 0.25 for i in range(256)]
    # Q8_K: 256 elements = 1 super-block
    q8_k_vals = [math.cos(i * 0.37) * 1.25 + ((i % 13) - 6) * 0.5 for i in range(256)]
    # Q6_K: 256 elements = 1 super-block
    q6_k_vals = [math.cos(i * 0.17) * 2.0 + ((i % 5) - 2) * 0.25 for i in range(256)]
    # IQ4_NL: 32 elements
    iq4_vals = [((i * 11) % 23) - 11 + 0.5 * (i % 4) for i in range(32)]

    plan = [
        ("f32_1d", [8], F32, build_f32(f32_vals)),
        ("f16_2d", [4, 2], F16, build_f16(f16_vals)),
        ("bf16_1d", [8], BF16, build_bf16(bf16_vals)),
        ("q8_0_1d", [64], Q8_0, build_q8_0(q8_vals)),
        ("q4_0_1d", [32], Q4_0, build_q4_0(q4_0_vals)),
        ("q4_1_1d", [32], Q4_1, build_q4_1(q4_1_vals)),
        ("q5_0_1d", [32], Q5_0, build_q5_0(q5_0_vals)),
        ("q5_1_1d", [32], Q5_1, build_q5_1(q5_1_vals)),
        ("q2_k_1d", [256], Q2_K, build_q2_K(q2_k_vals)),
        ("q3_k_1d", [256], Q3_K, build_q3_K(q3_k_vals)),
        ("q4_k_2d", [32, 8], Q4_K, build_q4_K(q4_k_vals)),
        ("q5_k_1d", [256], Q5_K, build_q5_K(q5_k_vals)),
        ("q6_k_1d", [256], Q6_K, build_q6_K(q6_k_vals)),
        ("q8_k_1d", [256], Q8_K, build_q8_K(q8_k_vals)),
        ("iq4_nl_1d", [32], IQ4_NL, build_iq4_nl(iq4_vals)),
    ]

    tensors = []
    offset = 0
    for name, shape, gtype, (payload, expected) in plan:
        n_elem = 1
        for d in shape:
            n_elem *= d
        assert len(expected) == n_elem, "%s: %d values for %d elements" % (
            name, len(expected), n_elem)
        tensors.append({
            "name": name,
            "shape": shape,
            "type": gtype,
            "offset": offset,
            "payload": payload,
            "expected": expected,
        })
        offset += len(payload)
        offset += (-offset) % alignment

    # ---- metadata ---------------------------------------------------------
    kv = [
        ("general.architecture", T_STRING, "llama"),
        ("general.name", T_STRING, "yskn gguf fixture"),
        ("general.alignment", T_UINT32, alignment),
        ("general.quantization_version", T_UINT32, 2),
        ("fixture.u8", T_UINT8, 200),
        ("fixture.i8", T_INT8, -100),
        ("fixture.u16", T_UINT16, 60000),
        ("fixture.i16", T_INT16, -30000),
        ("fixture.u32", T_UINT32, 4000000000),
        ("fixture.i32", T_INT32, -2000000000),
        ("fixture.u64", T_UINT64, 12345678901234567890),
        ("fixture.i64", T_INT64, -1234567890123456789),
        ("fixture.f32", T_FLOAT32, 1.5),
        ("fixture.f64", T_FLOAT64, -2.718281828459045),
        ("fixture.bool_true", T_BOOL, True),
        ("fixture.bool_false", T_BOOL, False),
        ("fixture.string", T_STRING, "hello gguf \u00e4\u00f6\u00fc \u65e5\u672c\u8a9e"),
        ("fixture.empty_string", T_STRING, ""),
        ("fixture.array_i32", T_ARRAY, (T_INT32, [1, -2, 3, -4, 5])),
        ("fixture.array_str", T_ARRAY, (T_STRING, ["alpha", "beta", "gamma"])),
        ("fixture.array_f32", T_ARRAY, (T_FLOAT32, [0.5, -0.25, 2.0])),
        ("fixture.array_bool", T_ARRAY, (T_BOOL, [True, False, True])),
        ("fixture.array_empty", T_ARRAY, (T_UINT32, [])),
        # nested: an array whose elements are themselves arrays
        ("fixture.nested", T_ARRAY, (T_ARRAY, [
            (T_INT32, [1, 2, 3]),
            (T_INT32, []),
            (T_INT32, [7]),
        ])),
        ("fixture.nested_mixed", T_ARRAY, (T_ARRAY, [
            (T_STRING, ["x", "y"]),
            (T_STRING, ["z"]),
        ])),
        # a per-tensor array, the shape llama.cpp uses for tokenizer metadata
        ("tokenizer.ggml.tokens", T_ARRAY, (T_STRING, [
            "a", "b", "c", "d", "e",
        ])),
        ("tokenizer.ggml.scores", T_ARRAY, (T_FLOAT32, [0.0, -1.0, -2.0, -3.0, -4.0])),
        ("tokenizer.ggml.token_type", T_ARRAY, (T_INT32, [1, 1, 1, 1, 1])),
    ]

    gguf_path = args.out
    json_path = gguf_path + ".json"
    data_base = write_gguf(gguf_path, args.version, alignment, kv, tensors)

    # ---- sidecar ----------------------------------------------------------
    def json_scalar(vtype, value):
        """Convert one non-array metadata value to a JSON-friendly Python value."""
        if vtype == T_BOOL:
            return bool(value)
        if vtype in (T_UINT8, T_UINT16, T_UINT32, T_UINT64,
                     T_INT8, T_INT16, T_INT32, T_INT64):
            return int(value)
        if vtype in (T_FLOAT32, T_FLOAT64):
            return float(value)
        if vtype == T_STRING:
            return str(value)
        raise ValueError("not a scalar metadata type: %r" % (vtype,))

    def json_items(elem_type, items):
        """Convert an array's items. Nested arrays become lists of lists."""
        if elem_type == T_ARRAY:
            return [json_items(it[0], it[1]) for it in items]
        return [json_scalar(elem_type, it) for it in items]

    meta_json = []
    for key, vtype, value in kv:
        entry = {"key": key, "type": vtype, "type_name": {
            T_UINT8: "uint8", T_INT8: "int8", T_UINT16: "uint16",
            T_INT16: "int16", T_UINT32: "uint32", T_INT32: "int32",
            T_FLOAT32: "float32", T_BOOL: "bool", T_STRING: "string",
            T_ARRAY: "array", T_UINT64: "uint64", T_INT64: "int64",
            T_FLOAT64: "float64",
        }[vtype]}
        if vtype == T_ARRAY:
            elem_type, items = value
            entry["elem_type"] = elem_type
            entry["items"] = json_items(elem_type, items)
            entry["nested"] = (elem_type == T_ARRAY)
        else:
            entry["value"] = json_scalar(vtype, value)
        meta_json.append(entry)

    sidecar = {
        "gguf_version": args.version,
        "alignment": alignment,
        "data_base": data_base,
        "n_tensors": len(tensors),
        "n_kv": len(kv),
        "metadata": meta_json,
        "tensors": [{
            "name": t["name"],
            "shape": t["shape"],
            "type": t["type"],
            "offset": t["offset"],
            "n_elements": len(t["expected"]),
            "expected": t["expected"],
        } for t in tensors],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=1)

    print("wrote %s (%d tensors, %d kv, version %d, alignment %d)" % (
        gguf_path, len(tensors), len(kv), args.version, alignment))
    print("wrote %s" % json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
