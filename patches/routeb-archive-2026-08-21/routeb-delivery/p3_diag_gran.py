#!/usr/bin/env python3
"""P3-Step1g: -hp = 均匀 13 级量化（±3, 步长 0.5）粒度判定 + 重构匹配"""
import json
import struct

import torch

M_HP = "/model"
M_MX = "/model_base"


def hdr_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), 8 + n


def load_raw(path, name):
    hdr, base = hdr_of(path)
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    if len(info["shape"]) == 0:
        return t, info["dtype"]
    return t.reshape(info["shape"]), info["dtype"]


def unpack_e2m1(u8):
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    out = torch.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = torch.where(lo >= 8, -1.0, 1.0) * E2M1_MAG[lo & 7]
    out[..., 1::2] = torch.where(hi >= 8, -1.0, 1.0) * E2M1_MAG[hi & 7]
    return out


E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def shard_for(root, name):
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    return f"{root}/{idx[name]}"


mx_w, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
mx_s, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.scale"),
                   "layers.0.ffn.expn.0.w1.scale" if False else "layers.0.ffn.experts.0.w1.scale")
N, K = mx_w.shape[0], mx_w.shape[1] * 2
codes_mx = unpack_e2m1(mx_w)
sf = torch.pow(2.0, mx_s.float() - 127.0)
W_mx = (codes_mx.reshape(N, K // 32, 32) * sf[:, :, None]).reshape(N, K).t().contiguous()

hp_w, _ = load_raw(shard_for(M_HP, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
codes_hp = unpack_e2m1(hp_w)
K_, N_ = codes_hp.shape

# ---- 粒度判定：多大粒度的 amax 恰好映射到 3.0 ----
print("=== 归一化粒度判定（各级粒度下 max|code|==3.0 的比例）===")
for (kg, ng, label) in ((1, 2048, "row × 全N"),
                        (1, 128, "row × 128N"),
                        (1, 64, "row × 64N"),
                        (1, 32, "row × 32N"),
                        (1, 16, "row × 16N"),
                        (32, 128, "32K × 128N"),
                        (32, 32, "32K × 32N"),
                        (2048, 1, "全K × 单N")):
    if K_ % kg or N_ % ng:
        continue
    g = codes_hp.abs().reshape(K_ // kg, kg, N_ // ng, ng).amax(dim=(1, 3))
    frac = (g == 3.0).float().mean().item()
    print(f"  {label:12s}: {frac:.4f}")

# ---- 重构：W_mx 按 (row × 128N) amax→3 均匀量化 ----
print("\n=== 重构匹配（uniform 13 级, amax→3）===")


def uniform_pack(x):
    """x 已归一化到 ±3。q = round(x*2)/2 → level = q*2 ∈ {0..6}"""
    level = torch.round(x * 2).clamp(-6, 6).long()
    nib = torch.where(level < 0, -level + 8, level)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)


for (kg, ng, label) in ((1, 128, "row × 128N"),
                        (1, 2048, "row × 全N"),
                        (32, 128, "32K × 128N"),
                        (1, 16, "row × 16N"),
                        (1, 8, "row × 8N")):
    blocks = W_mx.reshape(K_ // kg, kg, N_ // ng, ng)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    Wn = (blocks / amax * 3.0).reshape(K_, N_)
    m = (uniform_pack(Wn) == hp_w).float().mean().item()
    print(f"  {label:12s}: match={m:.4f}")

# 隐含真实 scale 的统计（row×128N 粒度）
blocks = W_mx.reshape(K_, N_ // 128, 128)
amax = blocks.abs().amax(dim=-1, keepdim=True)      # [K, N//128, 1]
s_row = (amax / 3.0).squeeze(-1)                    # [K, N//128]
e = torch.log2(s_row)
print(f"\n隐含 per-(row,128N) scale: p5=2^{e.quantile(.05):.2f} p50=2^{e.median():.2f} "
      f"p95=2^{e.quantile(.95):.2f}  幂次比例={(e.round()==e).float().mean():.3f}")
print(f"（若为 E8M0 存储，字节应为 e+127 ∈ [{e.min()+127:.0f},{e.max()+127:.0f}]；"
      f"实际存储全 1）")
