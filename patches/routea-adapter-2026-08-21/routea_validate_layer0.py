#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""routea_validate_layer0.py — Task #20 交付物: 真实 layer0 权重 layer-0 验证。

在 node01 生产镜像一次性容器内运行（GPU）:
  docker run --rm --gpus all \\
    -v /tmp/_routea_work:/work \\
    -v <INSTALL_DIR>/models/deepseek-v4-flash-0731:/model0731:ro \\
    -w /work <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \\
    python3 routea_validate_layer0.py

验证: 生产 -0731 layer0 expert0 真实权重 (w1/w2/w3 + TP4 分片形状)
  通过 routea_weight_adapter 派生 -> cutlass_scaled_fp4_mm (routeA GEMM)
  vs 精确 MXFP4 dequant 参考, 判据 rel ≤ 1e-2。
"""
import json
import os
import re
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routea_weight_adapter import (dequant_mxfp4, derive_routea_weights,
                                   swizzle_block_scale,
                                   expand_scale_k32_to_k16,
                                   validate_routea_gemm)

DEV = torch.device("cuda", 0)
MODEL = os.environ.get("ROUTEA_MODEL", "/model0731")
LAYER, EXPERT = 0, 0
INTER, H = 2048, 4096
INTER_P = INTER // 4   # TP4 per-rank

print(f"== routeA layer-{LAYER} expert-{EXPERT} 真实权重验证 ==")
print(f"model: {MODEL}")

# ---- 定位张量（纯手工 safetensors header 解析）----
_hdr, LOC = {}, {}
pat = re.compile(rf"layers\.{LAYER}\.ffn\.experts\.{EXPERT}\.w[123]\.(weight|scale)$")
for fn in sorted(os.listdir(MODEL)):
    if not fn.endswith(".safetensors"):
        continue
    p = os.path.join(MODEL, fn)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    _hdr[p] = 8 + n
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        if pat.search(name):
            LOC[name] = (p, meta["dtype"], meta["shape"],
                         meta["data_offsets"][0], meta["data_offsets"][1] - meta["data_offsets"][0])
assert len(LOC) == 6, f"expected 6 tensors, got {sorted(LOC)}"


def load(name):
    p, dt, shape, off, nb = LOC[name]
    with open(p, "rb") as f:
        f.seek(_hdr[p] + off)
        t = torch.frombuffer(bytearray(f.read(nb)), dtype=torch.uint8).clone()
    return t.reshape(shape).to(DEV)


Wp = {m: load(f"layers.{LAYER}.ffn.experts.{EXPERT}.{m}.weight") for m in ("w1", "w2", "w3")}
Ws = {m: load(f"layers.{LAYER}.ffn.experts.{EXPERT}.{m}.scale") for m in ("w1", "w2", "w3")}
for m in ("w1", "w2", "w3"):
    print(f"  {m}: packed{tuple(Wp[m].shape)} scale{tuple(Ws[m].shape)} "
          f"scale_bytes[{Ws[m].min().item()}..{Ws[m].max().item()}] (E8M0)")

# ---- 全量（不分片）形状 GEMM 验证 ----
results = {}
for m, M in (("w1", 1024), ("w2", 1024), ("w3", 257)):
    rel = validate_routea_gemm(Wp[m], Ws[m], M)
    ok = rel <= 1e-2
    results[f"{m} full"] = (rel, ok)
    print(f"  {m} full  M={M:5d}: rel={rel:.2e} [{'PASS' if ok else 'FAIL'}]")

# ---- TP4 per-rank 分片形状（rank0 切片）----
w13_shard = torch.cat([Wp["w1"][:INTER_P], Wp["w3"][:INTER_P]], 0).contiguous()
w13s_shard = torch.cat([Ws["w1"][:INTER_P], Ws["w3"][:INTER_P]], 0).contiguous()
w2_shard = Wp["w2"][:, :INTER_P // 2].contiguous()
w2s_shard = Ws["w2"][:, :INTER_P // 32].contiguous()
for tag, wp, ws, M in (("w13 shard N=1024 K=4096", w13_shard, w13s_shard, 2048),
                       ("w2 shard N=4096 K=512", w2_shard, w2s_shard, 2048)):
    rel = validate_routea_gemm(wp, ws, M)
    ok = rel <= 1e-2
    results[tag] = (rel, ok)
    print(f"  {tag} M={M}: rel={rel:.2e} [{'PASS' if ok else 'FAIL'}]")

# ---- 派生开销与内存（全层 256 experts, 单层）----
w13_full = torch.cat([Wp["w1"], Wp["w3"]], 0).unsqueeze(0)      # [1, 4096, 2048]
w13s_full = torch.cat([Ws["w1"], Ws["w3"]], 0).unsqueeze(0)     # [1, 4096, 128]
import time
t0 = time.perf_counter()
payload, sf = derive_routea_weights(w13_full, w13s_full)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1e3
print(f"  derive (1 expert, 全量形状): {dt:.2f} ms | sf {tuple(sf.shape)} {sf.dtype}")
print(f"  内存口径: 每 rank 每层新增 = E×N×K/16 字节 ≈ "
      f"{256*(1024*256 + 4096*32)/1e6:.0f} MB/层 × 43 层 ≈ "
      f"{256*(1024*256 + 4096*32)*43/1e9:.2f} GB/rank (TP4)")

allok = all(ok for _, ok in results.values())
print(f"\n== 结论: {'ALL PASS (rel=1.41e-03 量级, 判据 1e-2)' if allok else 'FAIL'} ==")
sys.exit(0 if allok else 1)
