#!/usr/bin/env python3
"""convert_mini_to_nvfp4.py — 用 Task#20 适配器逻辑把 mini-0731 (MXFP4) 转成 modelopt NVFP4 格式。
验证 A′ 派生链: converted(-0731派生, input_scale=1.0) vs 原生 -nvfp4 checkpoint 在 W4A4 路径等价性。
纯标准库。"""
import json
import os
import struct
import sys

# E8M0 byte -> E4M3 byte LUT (2^(b-127) 精确编码, 域外 clamp 到 0x00/最大)
def e4m3_byte_for_pow2(k):
    if k >= -6 and k <= 8:
        return (k + 7) << 3
    if k == -7: return 0b0000100
    if k == -8: return 0b0000010
    if k == -9: return 0b0000001
    if k < -9:  return 0
    return 0x7F  # 2^8=256 -> 最大 normal (0x7F = 0b01111111 = 256? 实为 240...) 用 saturate
LUT = bytes(e4m3_byte_for_pow2(b - 127) if -9 <= b - 127 <= 8 else (0 if b - 127 < -9 else 0x7E)
            for b in range(256))

SRC = "/tmp/_routea_work/mini0731/model.safetensors"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_routea_work/miniconv_nvfp4"
os.makedirs(OUT, exist_ok=True)

# ---- read source inventory ----
with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

DTYPE_SIZE = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E8M0": 1, "F8_E4M3": 1,
              "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F64": 8, "F16": 2, "BOOL": 1}

def read_raw(name):
    meta = hdr[name]
    off0, off1 = meta["data_offsets"]
    with open(SRC, "rb") as f:
        f.seek(base + off0)
        return f.read(off1 - off0)

# ---- build output tensor list ----
out_entries = []   # (name, dtype, shape, payload_bytes)
n_exp = 0
for name in sorted(hdr):
    if name == "__metadata__":
        continue
    meta = hdr[name]
    dt, shape = meta["dtype"], meta["shape"]
    if name.endswith(".scale") and ".ffn.experts." in name:
        # MXFP4 E8M0 [N, K//32] -> NVFP4 weight_scale E4M3 [N, K//16]
        N, K32 = shape
        raw = read_raw(name)
        t = raw.translate(LUT)              # E8M0 -> E4M3 逐字节
        buf = bytearray(len(t) * 2)         # 16 组 scale: 每 32 组 scale 复制为 2 个 16 组
        buf[0::2] = t
        buf[1::2] = t
        out_entries.append((name.replace(".scale", ".weight_scale"), "F8_E4M3",
                            [N, K32 * 2], bytes(buf)))
        continue
    if name.endswith(".weight") and ".ffn.experts." in name:
        # payload 原样 + 补 weight_scale_2 / input_scale 标量
        raw = read_raw(name)
        out_entries.append((name, dt, shape, raw))
        one = struct.pack("<f", 1.0)
        out_entries.append((name.replace(".weight", ".weight_scale_2"), "F32", [], one))
        out_entries.append((name.replace(".weight", ".input_scale"), "F32", [], one))
        n_exp += 1
        continue
    out_entries.append((name, dt, shape, read_raw(name)))

print(f"[conv] experts converted: {n_exp // 3} x 3 matrices; out tensors: {len(out_entries)}")

# ---- write single safetensors ----
total = 0
header = {}
for name, dt, shape, _ in out_entries:
    nb = DTYPE_SIZE[dt]
    for d in shape:
        nb *= d
    header[name] = {"dtype": dt, "shape": shape, "data_offsets": [total, total + nb]}
    total += nb
hb = json.dumps(header).encode()
hb += b" " * ((-len(hb)) % 8)
out_path = os.path.join(OUT, "model.safetensors")
with open(out_path, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for name, dt, shape, payload in out_entries:
        f.write(payload)
print(f"[conv] wrote {out_path}: {os.path.getsize(out_path)/1e9:.2f} GB")

# ---- config/hf_quant_config: 拷贝 mininvfp4_noclamp 的 (NVFP4 路由, swiglu_limit=None) ----
for fn in ("config.json", "hf_quant_config.json", "tokenizer.json",
           "tokenizer_config.json", "generation_config.json"):
    src = os.path.join("/tmp/_routea_work/mininvfp4_noclamp", fn)
    if os.path.exists(src):
        with open(src, "rb") as g:
            data = g.read()
        with open(os.path.join(OUT, fn), "wb") as g:
            g.write(data)
print("[conv] copied configs from mininvfp4_noclamp (NVFP4 route, swiglu_limit=None)")
print("[conv] DONE")
