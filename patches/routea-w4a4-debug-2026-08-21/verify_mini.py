#!/usr/bin/env python3
"""verify_mini.py — 校验 mini safetensors: 每张量计算字节数 vs 源偏移差; 再用 safetensors 库读回头部。"""
import json
import struct
import sys

DTYPE_SIZE = {"BF16": 2, "F32": 4, "F16": 2, "I8": 1, "U8": 1,
              "F8_E8M0": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I32": 4, "I64": 8,
              "F64": 8, "BOOL": 1, "I16": 2, "U16": 2, "U32": 4, "U64": 8}

src, mini = sys.argv[1], sys.argv[2]

def scan(model):
    inv = {}
    for fn in sorted(__import__('os').listdir(model)):
        if not fn.endswith(".safetensors"): continue
        p = __import__('os').path.join(model, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__": continue
            inv[name] = (meta["dtype"], meta["shape"], meta["data_offsets"][1]-meta["data_offsets"][0])
    return inv

s = scan(src)
with open(mini + "/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
print(f"mini header tensors: {len(hdr)}")
bad = 0
for name, meta in hdr.items():
    if name == "__metadata__": continue
    calc = DTYPE_SIZE[meta["dtype"]]
    for d in meta["shape"]:
        calc *= d
    src_nbytes = s.get(name, (None, None, None))[2]
    if src_nbytes is None:
        print(f"  MISSING in src: {name}"); bad += 1
    elif calc != src_nbytes or calc != meta["data_offsets"][1]-meta["data_offsets"][0]:
        print(f"  SIZE MISMATCH {name}: computed={calc} src={src_nbytes} "
              f"hdr_delta={meta['data_offsets'][1]-meta['data_offsets'][0]} dtype={meta['dtype']} shape={meta['shape']}")
        bad += 1
if bad == 0:
    print("all tensor sizes consistent with source")
    # 尝试 safetensors 库读回
    try:
        from safetensors import safe_open
        with safe_open(mini + "/model.safetensors", framework="pt") as f:
            keys = list(f.keys())
            print(f"safetensors lib open OK, {len(keys)} keys, first={keys[0]}")
            t = f.get_tensor(keys[0])
            print(f"  first tensor: {tuple(t.shape)} {t.dtype}")
    except Exception as e:
        print(f"safetensors lib FAILED: {type(e).__name__}: {e}")
