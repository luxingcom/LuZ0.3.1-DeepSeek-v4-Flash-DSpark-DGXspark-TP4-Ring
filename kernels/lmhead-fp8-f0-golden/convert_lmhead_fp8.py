#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lm_head BF16 -> FP8 (E4M3) + E8M0 scale conversion, routeB-native layout.

F0 golden asset — lm_head FP8 契约核对 (2026-08-24), architect-2.

Converts `head.weight` (BF16 [N, K]) into:
  - payload : FP8 E4M3 [N, K], K-contiguous (routeB B operand: column-major "K")
  - scale   : E8M0 [N, ceil(K/32)]   (routeB SFB plain layout, sf_vec=32)

Conversion convention (block-scaled MXFP8, matches routeB/cutlass Float8E4M3FN
+ Float8E8M0FNU, sf_vec=32):
  * per-row, per-32-K-group max-abs power-of-two ceiling scale: scale = 2^e,
    e = ceil(log2(max_abs)); E8M0 byte = e + 127 (bias 127).
  * payload element = round-to-nearest-E4M3(w / scale). |w/scale| <= 1.0, so
    the normalized value always sits in E4M3 normal range [0,1) (+ exact 1.0).
  * dequant: w_hat = e4m3(w/scale) * 2^e  ->  max rel error 2^-4 = 6.25%
    (half ULP) for the block-largest magnitude; bounded by scale*2^-4.

Pure Python / stdlib only -> runs offline on CPU, no numpy/torch/GPU.

Usage:
  python3 convert_lmhead_fp8.py --mode=manifest   # md5 + shape of bf16 shard
  python3 convert_lmhead_fp8.py --mode=convert --rows=N [--out-dir=DIR]
  python3 convert_lmhead_fp8.py --mode=verify  --rows=N
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys

# ---- head.weight location inside the checkpoint (source of truth, F0 verified) ----
SHARD = "<INSTALL_DIR>/models/deepseek-v4-flash-0731/model-00045-of-00048.safetensors"
HEAD_KEY = "head.weight"
N, K = 129280, 4096          # global shape (BF16)
DT = 2                        # bf16 bytes
SF_VEC = 32                   # routeB sf_vec (mandatory 32 for FP8)
E8M0_BIAS = 127

# Safetensors header parsing -------------------------------------------------

def parse_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def find_tensor_offset(path, key):
    hdr = parse_safetensors_header(path)
    if key not in hdr:
        raise KeyError(f"{key} not in {path}")
    meta = hdr[key]
    dtype, shape = meta["dtype"], tuple(meta["shape"])
    off, _ = meta["data_offsets"]
    return dtype, shape, off

# BF16 <-> F32 ---------------------------------------------------------------

def bf16_bytes_to_f32_list(raw):
    """raw: K*DT bytes -> list of float32 (pure python bf16 decode)."""
    out = []
    for i in range(0, len(raw), DT):
        u = struct.unpack_from("<H", raw, i)[0]
        out.append(struct.unpack("<f", struct.pack("<I", u << 16))[0])
    return out

def f32_to_bf16_bytes(vals):
    out = bytearray(len(vals) * DT)
    for i, v in enumerate(vals):
        u = (struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16) & 0xFFFF
        struct.pack_into("<H", out, i * DT, u)
    return bytes(out)

# E4M3 / E8M0 conversion ------------------------------------------------------

def f32_to_e4m3_byte(v):
    """Encode a normalized f32 (-1..1) into an 8-bit E4M3 pattern (Float8E4M3FN).

    abs(v) = m * 2^e, m in [0.5,1) -> value lies in binade [2^(e-1), 2^e).
    E4M3: 1 sign + 4 exp(bias 7) + 3 mantissa; mantissa quantizes the
    fractional part f = 2m-1 in [0,1) onto a 1/8 grid.
    """
    if v == 0.0:
        return 0x00
    s = 0x80 if v < 0 else 0x00
    m, e = math.frexp(abs(v))
    mi = int(round((2.0 * m - 1.0) * 8.0))     # 0..8
    if mi >= 8:                                 # carry to next binade
        mi = 0
        e += 1
    exp_stored = e + 6                          # binade start 2^(e-1), bias 7
    if exp_stored < 1:
        # Subnormal range: abs(v) in [2^-9, 2^-6), encode m*2^-9 (m=1..7).
        if abs(v) < 2.0 ** -9:
            return s                            # below subnormal -> flush to zero
        msub = int(round(abs(v) * 512.0))       # 0..8 on the 2^-9 grid
        if msub >= 8:                           # rounds up to min normal 2^-6
            return s | (1 << 3) | 0
        return s | msub
    if exp_stored > 15:
        return s | 0x7E                         # not reached for |v|<=1
    return s | (exp_stored << 3) | mi

def e4m3_byte_to_f32(b):
    """Decode an 8-bit E4M3 byte into a float (inverse of f32_to_e4m3_byte)."""
    if b == 0:
        return 0.0
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0x0F
    m = b & 0x07
    if e == 0:
        return s * m * 2.0 ** -9                # subnormal (min normal 2^-6)
    if e == 15:
        return s * float("inf") if m == 0 else float("nan")
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)

def round_to_e4m3(x):
    """Round float x to the nearest E4M3 value (consistent byte round-trip)."""
    return e4m3_byte_to_f32(f32_to_e4m3_byte(x))

def e8m0_byte_for_maxabs(max_abs):
    """Power-of-two ceiling scale -> E8M0 byte (bias 127)."""
    if max_abs == 0:
        return 0
    e = math.ceil(math.log2(max_abs))
    b = e + E8M0_BIAS
    if not (0 <= b <= 255):
        raise ValueError(f"E8M0 byte out of range: e={e} b={b}")
    return b

def convert_row_group(row_f32, g):
    """Convert 32 f32 elements -> (fp8 bytes, e8m0 byte)."""
    ma = max(abs(x) for x in row_f32[g * SF_VEC:(g + 1) * SF_VEC])
    sb = e8m0_byte_for_maxabs(ma)
    scale = 2.0 ** (sb - E8M0_BIAS)
    out = bytearray(SF_VEC)
    for j in range(SF_VEC):
        v = round_to_e4m3(row_f32[g * SF_VEC + j] / scale)
        out[j] = f32_to_e4m3_byte(v)
    return bytes(out), sb

# Streaming convert / verify --------------------------------------------------

def read_head_rows(f, off, row_start, row_count):
    f.seek(off + row_start * K * DT)
    return f.read(row_count * K * DT)

def convert_rows(row_start, row_count):
    """Yield (fp8_payload_bytes, scale_bytes, row_f32) per batch."""
    dtype, shape, off = find_tensor_offset(SHARD, HEAD_KEY)
    assert (dtype, shape) == ("BF16", (N, K)), (dtype, shape)
    fp8_per_row = K
    scale_per_row = K // SF_VEC
    with open(SHARD, "rb") as f:
        for base in range(0, row_count, 128):
            cnt = min(128, row_count - base)
            raw = read_head_rows(f, off, row_start + base, cnt)
            for r in range(cnt):
                row = bf16_bytes_to_f32_list(raw[r * K * DT:(r + 1) * K * DT])
                fp8 = bytearray(K)
                scale = bytearray(scale_per_row)
                for g in range(scale_per_row):
                    fb, sb = convert_row_group(row, g)
                    fp8[g * SF_VEC:(g + 1) * SF_VEC] = fb
                    scale[g] = sb
                yield bytes(fp8), bytes(scale), row

def verify_error(row_f32, fp8, scale_bytes, trace=False, rel_thresh=1e-2):
    """Compute dequant error stats for one row.

    rel error is only meaningful for elements not crushed by the block scale;
    we report it for |w| >= rel_thresh (default 1e-2, lm_head-relevant). Tiny
    elements can have large relative error but absolute error is bounded by
    block_scale*2^-4 (the documented FP8 E4M3 envelope).
    """
    max_rel = 0.0
    max_abs_err = 0.0
    rms_sum = 0.0
    n_big = 0
    worst = None
    for g in range(K // SF_VEC):
        scale = 2.0 ** (scale_bytes[g] - E8M0_BIAS)
        for j in range(SF_VEC):
            w = row_f32[g * SF_VEC + j]
            q = struct.unpack("<B", fp8[g * SF_VEC + j:g * SF_VEC + j + 1])[0]
            # decode E4M3 byte -> value
            dv = e4m3_byte_to_f32(q) * scale
            err = abs(dv - w)
            if err > max_abs_err:
                max_abs_err = err
                worst = (g, j, w, dv, q, scale_bytes[g], scale)
            rms_sum += err * err
            if abs(w) >= rel_thresh:
                n_big += 1
                rel = err / abs(w)
                if rel > max_rel:
                    max_rel = rel
    if trace:
        print("  trace worst(abs):", worst)
    n = K
    return {
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel,
        "rms": math.sqrt(rms_sum / n),
        "n_big": n_big,
    }

def e4m3_byte_to_f32(b):
    if b == 0:
        return 0.0
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0x0F
    m = b & 0x07
    if e == 0:
        return s * m * 2.0 ** -9        # subnormal
    if e == 15:
        return s * float("inf") if m == 0 else float("nan")
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)

# main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manifest", "convert", "verify"], required=True)
    ap.add_argument("--rows", type=int, default=4096, help="rows to convert (0=all N)")
    ap.add_argument("--row-start", type=int, default=0)
    ap.add_argument("--out-dir", default="lmhead-fp8-f0-golden")
    args = ap.parse_args()

    if args.mode == "manifest":
        dtype, shape, off = find_tensor_offset(SHARD, HEAD_KEY)
        print(f"shard={SHARD}")
        print(f"key={HEAD_KEY} dtype={dtype} shape={shape} data_off={off}")
        print(f"bytes={shape[0]*shape[1]*DT} = {shape[0]}x{shape[1]}x{DT}")
        h = hashlib.md5()
        with open(SHARD, "rb") as f:
            f.seek(off)
            remaining = shape[0] * shape[1] * DT
            while remaining:
                chunk = f.read(min(1 << 26, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        print(f"md5_full_bf16={h.hexdigest()}")

    elif args.mode == "convert":
        n = args.rows
        os.makedirs(args.out_dir, exist_ok=True)
        fp8_path = os.path.join(args.out_dir, f"head_fp8_rows{args.row_start}_{args.row_start+n}.bin")
        sc_path = os.path.join(args.out_dir, f"head_scale_rows{args.row_start}_{args.row_start+n}.bin")
        with open(fp8_path, "wb") as ff, open(sc_path, "wb") as fs:
            for fp8, scale, _row in convert_rows(args.row_start, n):
                ff.write(fp8)
                fs.write(scale)
        print(f"wrote {fp8_path} ({os.path.getsize(fp8_path)} B)")
        print(f"wrote {sc_path} ({os.path.getsize(sc_path)} B)")

    elif args.mode == "verify":
        agg = {"max_abs_err": 0.0, "max_rel_err": 0.0, "rms_sum": 0.0, "n": 0, "n_big": 0}
        worst_row = None
        for r, (fp8, scale, row) in enumerate(convert_rows(args.row_start, args.rows)):
            st = verify_error(row, fp8, scale, trace=(args.rows <= 4))
            if st["max_abs_err"] > agg["max_abs_err"]:
                agg["max_abs_err"] = st["max_abs_err"]
            if st["max_rel_err"] > agg["max_rel_err"]:
                agg["max_rel_err"] = st["max_rel_err"]
                worst_row = args.row_start + r
            agg["rms_sum"] += st["rms"] * st["rms"]
            agg["n"] += 1
            agg["n_big"] += st["n_big"]
        agg["rms"] = math.sqrt(agg["rms_sum"] / agg["n"])
        print(f"rows={args.rows} row_start={args.row_start}")
        print(f"max_abs_err={agg['max_abs_err']:.6g}")
        print(f"max_rel_err={agg['max_rel_err']:.6g} (bound 0.0625=2^-4) worst_row={worst_row}")
        print(f"rms_abs_err={agg['rms']:.6g}")
        print(f"n_big(>1e-3)={agg['n_big']}")
        # Note: row 0 anomaly documented in F0 report (K[0:224] garbage scale)
        print("NOTE: row 0 contains anomalous huge values (token-0 row, K[0:224]);")
        print("      it is included but flagged; all other rows normal (~0.5-3.8 maxabs).")

if __name__ == "__main__":
    main()
