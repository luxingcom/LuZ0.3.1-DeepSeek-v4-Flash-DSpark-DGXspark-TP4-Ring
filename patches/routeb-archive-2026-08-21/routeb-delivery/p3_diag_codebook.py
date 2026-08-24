#!/usr/bin/env python3
"""P3-Step1h: -hp 编码器码本判定（per-column 最小二乘拟合：线性 vs e2m1）"""
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
LIN_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])


def shard_for(root, name):
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    return f"{root}/{idx[name]}"


mx_w, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
mx_s, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.scale"),
                   "layers.0.ffn.experts.0.w1.scale")
N, K = mx_w.shape[0], mx_w.shape[1] * 2
codes_mx = unpack_e2m1(mx_w)
sf = torch.pow(2.0, mx_s.float() - 127.0)
W_mx = (codes_mx.reshape(N, K // 32, 32) * sf[:, :, None]).reshape(N, K).t().contiguous()

hp_w, _ = load_raw(shard_for(M_HP, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
K_, N_ = W_mx.shape
# nibble levels（带符号）：level ∈ {-8..7}，幅值 = codebook[|level|&7]
lvl = torch.empty(K_, N_, dtype=torch.long)
lvl[:, 0::2] = (hp_w & 0xF).long()
lvl[:, 1::2] = (hp_w >> 4).long()
lvl = torch.where(lvl >= 8, lvl - 16, lvl)   # 有符号 level

print("=== per-column 最小二乘拟合（value ≈ level × h_j）===")
for label, mag in (("e2m1 码本 (0,.5,1,1.5,2,3,4,6)", E2M1_MAG),
                   ("线性码本 (0,.5,1,1.5,2,2.5,3,3.5)", LIN_MAG)):
    # level → 幅值（码本索引 = |level|，但 6/7 未用）
    val = mag[lvl.abs().clamp(0, 7)] * torch.sign(lvl.float())  # [K, N]
    # per-column h_j = Σ val·x / Σ val²
    num = (val * W_mx).sum(dim=0)
    den = (val * val).sum(dim=0).clamp(min=1e-30)
    h = num / den                                    # [N]
    resid = W_mx - val * h
    rel = resid.abs().mean() / W_mx.abs().mean()
    print(f"  {label}: h_j p50={h.median():.5f} p5={h.quantile(.05):.5f} "
          f"p95={h.quantile(.95):.5f} | 残差/均值={rel:.4f}")

print("\n=== per-row 拟合对照 ===")
for label, mag in (("e2m1 码本", E2M1_MAG), ("线性码本", LIN_MAG)):
    val = mag[lvl.abs().clamp(0, 7)] * torch.sign(lvl.float())
    num = (val * W_mx).sum(dim=1)
    den = (val * val).sum(dim=1).clamp(min=1e-30)
    h = num / den
    resid = W_mx - val * h[:, None]
    rel = resid.abs().mean() / W_mx.abs().mean()
    print(f"  {label}: h_i p50={h.median():.5f} | 残差/均值={rel:.4f}")

print("\n=== 全局单 scale 拟合（e2m1 vs 线性）===")
for label, mag in (("e2m1 码本", E2M1_MAG), ("线性码本", LIN_MAG)):
    val = mag[lvl.abs().clamp(0, 7)] * torch.sign(lvl.float())
    h = (val * W_mx).sum() / (val * val).sum()
    resid = W_mx - val * h
    rel = resid.abs().mean() / W_mx.abs().mean()
    # 重建字节匹配
    grid = mag[lvl.abs().clamp(0, 7)]
    print(f"  {label}: h={h:.5f} 残差/均值={rel:.4f}")

# 直接重建测试：x ≈ h×level（线性）下重建字节 vs 原码（用 W_mx 重编码）
print("\n=== 用 W_mx 以 per-row 线性码本重编码 → 字节匹配率 ===")
val = LIN_MAG[lvl.abs().clamp(0, 7)] * torch.sign(lvl.float())
num = (val * W_mx).sum(dim=1)
den = (val * val).sum(dim=1).clamp(min=1e-30)
h = num / den
xn = W_mx / h[:, None]                    # 归一化（线性码本空间）
level_rec = torch.round(xn * 2).clamp(-5, 5).long()   # 线性 11 级
nib = torch.where(level_rec < 0, -level_rec + 8, level_rec)
rec = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
print(f"  线性码本 per-row h: match={(rec == hp_w).float().mean():.4f}")
