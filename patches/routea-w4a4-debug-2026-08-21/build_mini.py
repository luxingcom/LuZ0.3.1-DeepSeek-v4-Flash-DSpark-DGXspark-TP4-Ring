#!/usr/bin/env python3
"""build_mini.py — 从真实 checkpoint 抽前 N 层构建 mini 模型（单文件 safetensors）。

用法: python3 build_mini.py <src_model_dir> <out_dir> [n_layers=4]

保留: 全局张量(embed/head/norm/hc_head_*) + layers 0..N-1 全部张量; 丢弃 mtp.*
config 修改: num_hidden_layers=N, compress_ratios[:N], dspark_target_layer_ids=[],
             num_nextn_predict_layers=0; -nvfp4 的 quantized_layers 同步裁剪。
拷贝: tokenizer.json / tokenizer_config.json / generation_config.json / hf_quant_config.json
纯标准库（流式字节拷贝, 无 torch 依赖）。
"""
import json
import os
import struct
import sys


def scan_shards(model_dir):
    """name -> (shard_path, dtype, shape, off0, off1)"""
    inv = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        p = os.path.join(model_dir, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            inv[name] = (p, meta["dtype"], meta["shape"],
                         base + meta["data_offsets"][0],
                         base + meta["data_offsets"][1])
    return inv


def write_safetensors(out_path, entries):
    """entries: list of (name, dtype, shape, src_path, off0, off1)"""
    total = 0
    header = {}
    for name, dtype, shape, _, _, _ in entries:
        nbytes = DTYPE_SIZE[dtype]
        for d in shape:
            nbytes *= d
        # dtype 字节数
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [total, total + nbytes]}
        total += nbytes
    hb = json.dumps(header).encode()
    pad = (-len(hb)) % 8
    hb += b" " * pad
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for i, (name, dtype, shape, src, off0, off1) in enumerate(entries):
            with open(src, "rb") as f:
                f.seek(off0)
                remaining = off1 - off0
                while remaining > 0:
                    chunk = f.read(min(1 << 24, remaining))
                    if not chunk:
                        raise IOError(f"short read {name}")
                    out.write(chunk)
                    remaining -= len(chunk)
            if (i + 1) % 2000 == 0:
                print(f"    ... {i+1}/{len(entries)} tensors, {os.path.getsize(out_path)/1e9:.2f} GB",
                      flush=True)
    return total


DTYPE_SIZE = {"BF16": 2, "F32": 4, "F16": 2, "I8": 1, "U8": 1,
              "F8_E8M0": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I32": 4, "I64": 8,
              "F64": 8, "BOOL": 1, "I16": 2, "U16": 2, "U32": 4, "U64": 8}


def main():
    src, out = sys.argv[1], sys.argv[2]
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out, exist_ok=True)
    print(f"[mini] src={src} out={out} n_layers={n_layers}")
    inv = scan_shards(src)
    print(f"[mini] source tensors: {len(inv)}")

    keep = []
    for name in sorted(inv):
        if name.startswith("mtp."):
            continue
        if name.startswith("layers."):
            l = int(name.split(".")[1])
            if l >= n_layers:
                continue
        keep.append(name)
    entries = []
    for name in keep:
        p, dtype, shape, off0, off1 = inv[name]
        entries.append((name, dtype, shape, p, off0, off1))
    # sanity: dtype sizes known
    for name, dtype, shape, *_ in entries:
        assert dtype in DTYPE_SIZE, f"unknown dtype {dtype} for {name}"
    print(f"[mini] keeping {len(entries)} tensors")

    out_path = os.path.join(out, "model.safetensors")
    total = write_safetensors(out_path, entries)
    print(f"[mini] wrote {out_path}: {total/1e9:.2f} GB")

    # ---- config ----
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = n_layers
    if isinstance(cfg.get("compress_ratios"), list):
        cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    if "dspark_target_layer_ids" in cfg:
        cfg["dspark_target_layer_ids"] = []
    if "num_nextn_predict_layers" in cfg:
        cfg["num_nextn_predict_layers"] = 0
    # prune quantized_layers (nvfp4-style configs)
    qc = cfg.get("quantization_config") or {}
    if isinstance(qc.get("quantized_layers"), dict):
        qc["quantized_layers"] = {
            k: v for k, v in qc["quantized_layers"].items()
            if k.startswith("layers.") and int(k.split(".")[1]) < n_layers
        }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"[mini] config.json: num_hidden_layers={n_layers}, "
          f"compress_ratios={cfg.get('compress_ratios')}, mtp=0, "
          f"quantized_layers={len(qc.get('quantized_layers', {}))}")

    # ---- hf_quant_config.json (modelopt) ----
    hfq = os.path.join(src, "hf_quant_config.json")
    if os.path.exists(hfq):
        with open(hfq) as f:
            h = json.load(f)
        q = h.get("quantization", {})
        if isinstance(q.get("quantized_layers"), dict):
            q["quantized_layers"] = {
                k: v for k, v in q["quantized_layers"].items()
                if k.startswith("layers.") and int(k.split(".")[1]) < n_layers
            }
        with open(os.path.join(out, "hf_quant_config.json"), "w") as f:
            json.dump(h, f, indent=1)
        print(f"[mini] hf_quant_config.json: quantized_layers="
              f"{len(q.get('quantized_layers', {}))}")

    # ---- aux files ----
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            with open(os.path.join(out, fn), "wb") as f:
                f.write(data)
            print(f"[mini] copied {fn} ({len(data)/1e6:.1f} MB)")
    print("[mini] DONE")


if __name__ == "__main__":
    main()
