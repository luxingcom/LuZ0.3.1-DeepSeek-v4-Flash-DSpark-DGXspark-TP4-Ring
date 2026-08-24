#!/usr/bin/env python3
"""P3-Step1d: -hp 权重语义判别——对照 -0731 生产 MXF4 真值
问题：-hp scale 全 1、码本只用 0-5（absmax=3.0）。判定 -hp 到底存了什么、
scale 全 1 是否为转换器缺陷。
"""
import json
import struct

import torch

M_HP = "/model"        # -hp
M_MX = "/model_base"   # -0731 生产 MXF4

E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


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
    return (torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            .reshape(info["shape"]), info["dtype"])


def unpack_e2m1(u8):
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    out = torch.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = torch.where(lo >= 8, -1.0, 1.0) * E2M1_MAG[lo & 7]
    out[..., 1::2] = torch.where(hi >= 8, -1.0, 1.0) * E2M1_MAG[hi & 7]
    return out


def pack_e2m1(x):  # 就近码本（含 tie→低档），用于核对 -hp 码
    mag = E2M1_MAG
    d = (x.abs().unsqueeze(-1) - mag).abs()
    idx = d.argmin(-1)
    nib = torch.where(x < 0, 8, 0) + idx
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


# ---------- 生产 MXF4 真值 ----------
shard = f"{M_MX}/model-00002-of-00048.safetensors"
idx = json.load(open(f"{M_MX}/model.safetensors.index.json"))["weight_map"]
shard = f"{M_MX}/{idx['layers.0.ffn.experts.0.w1.weight']}"
mx_w, dtw = load_raw(shard, "layers.0.ffn.experts.0.w1.weight")
mx_s, dts = load_raw(shard, "layers.0.ffn.experts.0.w1.scale")
print(f"MXF4 w1.weight {tuple(mx_w.shape)} {dtw}; scale {tuple(mx_s.shape)} {dts}")
print(f"MXF4 scale bytes: uniq_len={len(mx_s.unique())} min={mx_s.min()} "
      f"max={mx_s.max()} mean={mx_s.float().mean():.2f}")

N, K = mx_w.shape[0], mx_w.shape[1] * 2          # N=2048, K=4096
codes_mx = unpack_e2m1(mx_w)                      # [N, K] 幅值
sf = torch.pow(2.0, mx_s.float() - 127.0)         # [N, K//32]
W_true = codes_mx.reshape(N, K // 32, 32) * sf[:, :, None]
W_true = W_true.reshape(N, K)
print(f"W_true: mean={W_true.mean():.6f} std={W_true.std():.5f} "
      f"absmax={W_true.abs().max():.4f}")

# ---------- -hp ----------
hp_w, _ = load_raw(f"{M_HP}/model-00002-of-00048.safetensors",
                   "layers.0.ffn.experts.0.w1.weight")
hp_s, _ = load_raw(f"{M_HP}/model-00002-of-00048.safetensors",
                   "layers.0.ffn.experts.0.w1.scale")
codes_hp = unpack_e2m1(hp_w)                      # [K, N]（[K, N//2] 打包）
print(f"\n-hp codes [K={codes_hp.shape[0]}, N={codes_hp.shape[1]}]: "
      f"std={codes_hp.std():.4f} absmax={codes_hp.abs().max():.3f}")

W_true_T = W_true.t().contiguous()                # [K, N]

# ---------- 假设检验 ----------
print("\n=== H1: -hp = W_true / block_amax × 3（块归一化到 ±3）===")
K_, N_ = W_true_T.shape
blocks = W_true_T.reshape(K_ // 32, 32, N_ // 128, 128)
bmax = blocks.abs().amax(dim=(1, 3))              # [K//32, N//128]
print(f"block amax 分布: p5={bmax.quantile(0.05):.4f} p50={bmax.median():.4f} "
      f"p95={bmax.quantile(0.95):.4f}")
W_norm = (blocks / bmax[:, None, :, None] * 3.0).reshape(K_, N_)
# 量化回 e2m1 再比对（避免 tie 细节差异）
repack = pack_e2m1(W_norm)                          # [K, N//2]
match = (repack == hp_w).float().mean().item()
c = torch.corrcoef(torch.stack([W_norm.flatten(), codes_hp.flatten()]))[0, 1].item()
print(f"H1 码字节匹配率: {match:.4f}  corr={c:+.4f}")

print("\n=== H2: -hp = W_true / 2^floor(log2(bmax/3))（幂次块 scale）===")
e = torch.floor(torch.log2(torch.clamp(bmax, min=1e-30) / 3.0))
sfp = torch.pow(2.0, e)
W_norm2 = (blocks / sfp[:, None, :, None]).clamp(-3, 3).reshape(K_, N_)
repack2 = pack_e2m1(W_norm2)
match2 = (repack2 == hp_w).float().mean().item()
c2 = torch.corrcoef(torch.stack([W_norm2.flatten(), codes_hp.flatten()]))[0, 1].item()
print(f"H2 码字节匹配率: {match2:.4f}  corr={c2:+.4f}")

print("\n=== H3: -hp = W_true / 2^floor(log2(bmax/6))（MXF4 floor 语义, clamp 6）===")
e3 = torch.floor(torch.log2(torch.clamp(bmax, min=1e-30) / 6.0))
sfp3 = torch.pow(2.0, e3)
W_norm3 = (blocks / sfp3[:, None, :, None]).clamp(-6, 6).reshape(K_, N_)
repack3 = pack_e2m1(W_norm3)
match3 = (repack3 == hp_w).float().mean().item()
print(f"H3 码字节匹配率: {match3:.4f}")
print(f"H3 隐含 scale 字节 = e+127: min={(e3+127).min():.0f} max={(e3+127).max():.0f} "
      f"mean={(e3+127).mean():.1f}  （实际 -hp scale 全 1）")

print("\n=== H4: -hp 直接 = W_true 原值码（无归一化，scale=1）===")
repack4 = pack_e2m1(W_true_T)
match4 = (repack4 == hp_w).float().mean().item()
c4 = torch.corrcoef(torch.stack([W_true_T.flatten(), codes_hp.flatten()]))[0, 1].item()
print(f"H4 码字节匹配率: {match4:.4f}  corr={c4:+.4f}")

# nibble 顺序对照（在最优假设下测翻转）
print("\n=== nibble 顺序检验（对最优假设做翻转对照）===")
best = {"H1": match, "H2": match2, "H3": match3, "H4": match4}
print(f"  匹配率汇总: {best}")
W_norm_sw = W_norm.clone()
W_norm_sw[:, 0::2], W_norm_sw[:, 1::2] = W_norm[:, 1::2], W_norm[:, 0::2]
repack_sw = pack_e2m1(W_norm_sw)
print(f"  H1 + nibble 翻转: match={(repack_sw == hp_w).float().mean().item():.4f}")
