#!/usr/bin/env python3
"""P3-Step1b: scale 全 1 异常诊断。用 safetensors 官方库交叉验证手工解析，
并抽查多个 expert / layer / w2/w3，判断是否为转换器退化产物。"""
import json
import struct

import torch

MODEL_DIR = "/model"

try:
    from safetensors.torch import safe_open
    HAS_ST = True
except ImportError:
    HAS_ST = False
print(f"safetensors lib: {HAS_ST}")


def manual_read(shard, name):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        info = hdr[name]
        s, e = info["data_offsets"]
        f.seek(8 + n + s)
        raw = f.read(e - s)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(info["shape"])


idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]

# --- 1) 交叉验证：手工 vs safetensors 库（expert0 w1.scale）---
shard2 = f"{MODEL_DIR}/{idx['layers.0.ffn.experts.0.w1.scale']}"
manual = manual_read(shard2, "layers.0.ffn.experts.0.w1.scale")
print(f"manual w1.scale: shape={tuple(manual.shape)} "
      f"uniq={manual.unique().tolist()[:20]} count0..5={(manual==0).sum().item()},"
      f"{(manual==1).sum().item()},{(manual==2).sum().item()},{(manual==3).sum().item()},"
      f"{(manual==4).sum().item()},{(manual==5).sum().item()}")
if HAS_ST:
    with safe_open(shard2, framework="pt", device="cpu") as f:
        st = f.get_tensor("layers.0.ffn.experts.0.w1.scale")
    print(f"safe_open w1.scale: shape={tuple(st.shape)} dtype={st.dtype} "
          f"equal_to_manual={torch.equal(st.view(torch.uint8), manual)}")

# --- 2) 抽查多个 expert（w1/w2/w3 scale）---
print("\n=== 多 expert scale 字节分布 ===")
shard_map = {}
def get_tensor(full_name):
    shard = f"{MODEL_DIR}/{idx[full_name]}"
    if HAS_ST:
        with safe_open(shard, framework="pt", device="cpu") as f:
            return f.get_tensor(full_name).view(torch.uint8)
    return manual_read(shard, full_name)

for e in (0, 1, 2, 7, 100, 255):
    for w in ("w1", "w2", "w3"):
        nm = f"layers.0.ffn.experts.{e}.{w}.scale"
        t = get_tensor(nm)
        u = t.unique()
        print(f"  e{e:3d} {w}.scale {tuple(t.shape)}: uniq_len={len(u)} "
              f"min={t.min().item()} max={t.max().item()} mean={t.float().mean().item():.2f} "
              f"uniq[:8]={u[:8].tolist()}")

# --- 3) 其他层 + 非 MoE 权重对照 ---
print("\n=== 其他层 / 非 MoE 对照 ===")
for nm in ("layers.21.ffn.experts.5.w1.scale",
           "layers.42.ffn.experts.200.w2.scale",
           "layers.0.self_attention.q_proj.weight",
           "layers.0.self_attention.q_proj.scale",
           "layers.0.ffn.gate.weight"):
    if nm in idx:
        shard = f"{MODEL_DIR}/{idx[nm]}"
        with open(shard, "rb") as f:
            nn = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(nn).decode())
        print(f"  {nm}: shape={hdr[nm]['shape']} dtype={hdr[nm]['dtype']} -> {idx[nm]}")
        t = get_tensor(nm)
        u = t.unique()
        print(f"      uniq_len={len(u)} min={t.min().item()} max={t.max().item()} "
              f"uniq[:8]={u[:8].tolist()}")

# --- 4) weight 字节抽样（确认 weight 本身非退化）---
wt = get_tensor("layers.0.ffn.experts.0.w1.weight")
print(f"\nw1.weight uniq nibble codes: "
      f"{torch.cat([wt & 0xF, wt >> 4]).unique().tolist()}")
print(f"w1.weight 首行前 32 字节: {wt[0, :32].tolist()}")
