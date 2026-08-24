#!/usr/bin/env python3
"""P3-Step1f: -hp 编码假设终判——常数 scale + tie 规则穷举"""
import json
import struct

import torch

M_HP = "/model"
M_MX = "/model_base"

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


def pack_e2m1(x, tie="down"):
    """tie=down: argmin（平局取低码，0.25→0）; tie=up: 平局取高码; tie=even: 偶数索引"""
    d = (x.abs().unsqueeze(-1) - E2M1_MAG).abs()
    if tie == "down":
        idx = d.argmin(-1)
    else:
        # 平局取高码：距离相同取较大 index
        idx = d.argmax(-1) * 0
        # 简洁实现：d 反向偏置无穷小 × index
        eps = torch.arange(8, dtype=torch.float32) * 1e-6
        idx = (d - eps).argmin(-1)  # 平局时偏向大 index？不——argmin 仍取第一个最小
        # 正确做法：给小 index 加惩罚
        pen = torch.arange(8, dtype=torch.float32) * 1e-6
        idx = (d + pen).argmin(-1)  # 平局 → 取大 index（pen 随 index 增）
    nib = torch.where(x < 0, 8, 0) + idx
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)


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
codes_hp = unpack_e2m1(hp_w)
K_, N_ = codes_hp.shape

print(f"W_mx [K,N]={tuple(W_mx.shape)} std={W_mx.std():.5f}")
print(f"-hp codes std={codes_hp.std():.4f}  码频率: " +
      " ".join(f"{E2M1_MAG[i].item():.1f}:{(codes_hp.abs()==E2M1_MAG[i]).float().mean():.3f}"
               for i in range(8)))

print("\n=== 假设矩阵：scale 语义 × tie 规则 ===")
for label, Wn in (
    ("W_mx / 2^-4（常数, byte 123 语义）", W_mx / 2 ** -4),
    ("W_mx / 2^-5（常数, byte 122 语义）", W_mx / 2 ** -5),
    ("W_mx / 2^-6（常数, byte 121 语义）", W_mx / 2 ** -6),
):
    for tie in ("down", "up"):
        m = (pack_e2m1(Wn.clamp(-6, 6), tie) == hp_w).float().mean().item()
        print(f"  {label}  tie={tie}: match={m:.4f}")

# 逐块 scale：s = 2^floor(log2(bmax/3))
blocks = W_mx.reshape(K_ // 32, 32, N_ // 128, 128)
bmax = blocks.abs().amax(dim=(1, 3))
for rule, e in (
    ("floor(bmax/3)", torch.floor(torch.log2(bmax.clamp(min=1e-30) / 3.0))),
    ("floor(bmax/6)", torch.floor(torch.log2(bmax.clamp(min=1e-30) / 6.0))),
    ("ceil(bmax/6)", torch.ceil(torch.log2(bmax.clamp(min=1e-30) / 6.0))),
    ("round(bmax/3)", torch.round(torch.log2(bmax.clamp(min=1e-30) / 3.0))),
):
    s = torch.pow(2.0, e)
    Wn = (blocks / s[:, None, :, None]).clamp(-6, 6).reshape(K_, N_)
    for tie in ("down", "up"):
        m = (pack_e2m1(Wn, tie) == hp_w).float().mean().item()
        print(f"  块scale={rule}  tie={tie}: match={m:.4f}  (e 分布: "
              f"{e.unique().tolist()[:6]})")

# -hp 自身块结构统计
hb = codes_hp.abs().reshape(K_ // 32, 32, N_ // 128, 128)
hmax = hb.amax(dim=(1, 3))
print(f"\n-hp 每块最大码分布: " +
      " ".join(f"{E2M1_MAG[i].item():.1f}:{(hmax==E2M1_MAG[i]).float().mean():.3f}"
               for i in range(8) if (hmax == E2M1_MAG[i]).any()))
# 行结构（32-K 组内的行粒度）——检验是否 per-row scale
hr = codes_hp.abs().reshape(K_, N_ // 128, 128).amax(dim=-1)  # [K, N//128]
frac_row_max3 = (hr == 3.0).float().mean()
print(f"每行(单K行×128N)最大码==3.0 的比例: {frac_row_max3:.3f}")
