#!/usr/bin/env python3
"""P3-Step1c: -nvfp4 源格式实证 + -hp scale 语义判别（对照 FP8 原模型）"""
import json
import struct

import torch

M = "/model"          # -hp（ro 挂载点，本脚本专用）
M2 = "/model_src"     # -nvfp4
M3 = "/model_base"    # -0731 原始 FP8

E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def hdr_of(shard):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), 8 + n


def load_raw(path, name):
    hdr, base = hdr_of(path)
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(info["shape"])
    return t, info["dtype"]


def unpack_e2m1(u8):
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    out = torch.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = torch.where(lo >= 8, -1.0, 1.0) * E2M1_MAG[lo & 7]
    out[..., 1::2] = torch.where(hi >= 8, -1.0, 1.0) * E2M1_MAG[hi & 7]
    return out


def idx_of(root):
    return json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]


# ---------- 1) -nvfp4 源格式 ----------
print("=== -nvfp4 源：expert0 张量格式 ===")
idx2 = idx_of(M2)
for nm in ("layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
           "layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w2.scale"):
    if nm in idx2:
        t, dt = load_raw(f"{M2}/{idx2[nm]}", nm)
        u = t.unique()
        print(f"  {nm}: shape={tuple(t.shape)} dtype={dt} "
              f"uniq_len={len(u)} min={t.min().item()} max={t.max().item()} "
              f"uniq[:12]={u[:12].tolist()}")
    else:
        # 找相近命名
        cand = [k for k in idx2 if "layers.0.ffn.experts.0." in k][:8]
        print(f"  ✗ {nm} 不存在；候选: {cand}")

# ---------- 2) FP8 原模型 expert0（ground truth）----------
print("\n=== -0731 FP8 原模型：expert0 w1 ===")
idx3 = idx_of(M3)
cand = sorted(k for k in idx3 if "layers.0.ffn.experts.0." in k)
print(f"  候选: {cand}")
fp8_w, dt_w = load_raw(f"{M3}/{idx3['layers.0.ffn.experts.0.w1.weight']}",
                       "layers.0.ffn.experts.0.w1.weight")
fp8_s, dt_s = load_raw(f"{M3}/{idx3['layers.0.ffn.experts.0.w1.scale']}",
                       "layers.0.ffn.experts.0.w1.scale")
print(f"  weight: {tuple(fp8_w.shape)} {dt_w}; scale: {tuple(fp8_s.shape)} {dt_s}")
# FP8 e4m3 block-scale: scale 为 ue8m0（fp8 scale_fmt=ue8m0, block 128×128）
fp8_s_f = torch.pow(2.0, fp8_s.float() - 127.0)
# dequant: block [128,128]，weight [out,in] 假定 [N?, K?] → 试两种
def deq_fp8(w_u8, s, rgrp, cgrp):
    # e4m3 解码（粗略：用 torch 的 float8_e4m3fn）
    w = w_u8.view(torch.float8_e4m3fn).float()
    R, C = w.shape
    sf = torch.pow(2.0, s.float() - 127.0)
    assert R % rgrp == 0 and C % cgrp == 0 and s.shape == (R // rgrp, C // cgrp)
    return w.reshape(R // rgrp, rgrp, C // cgrp, cgrp) * sf[:, None, :, None]

try:
    W1 = deq_fp8(fp8_w, fp8_s, 128, 128).reshape(fp8_w.shape)
    print(f"  FP8 dequant stats: mean={W1.mean():.5f} std={W1.std():.5f} "
          f"absmax={W1.abs().max():.4f}")
except Exception as ex:
    print(f"  FP8 dequant 失败: {ex}")
    W1 = None

# ---------- 3) -hp 解码 vs FP8 对照（scale 语义判别）----------
print("\n=== -hp w1 解码 × 多种 scale 语义 vs FP8 dequant 相关性 ===")
hp_w, _ = load_raw(f"{M}/model-00002-of-00048.safetensors",
                   "layers.0.ffn.experts.0.w1.weight")
hp_s, _ = load_raw(f"{M}/model-00002-of-00048.safetensors",
                   "layers.0.ffn.experts.0.w1.scale")
Wd = unpack_e2m1(hp_w)  # [4096, 2048] 幅值（未乘 scale）
print(f"  -hp w1 解码（未乘 scale）: mean={Wd.mean():.5f} std={Wd.std():.5f} "
      f"absmax={Wd.abs().max():.3f}")

if W1 is not None:
    # 对齐形状：FP8 w1 [2048, 4096]（out=intermediate, in=hidden），-hp [4096, 2048]
    # -hp 若为 [K, N//2] 则逻辑 W [K=4096, N=2048] = FP8 的 W^T
    W1T = W1.t()  # [4096, 2048]
    for label, sc in (
        ("byte=1 → scale=2^(1-127)（E8M0 标准）", torch.pow(2.0, hp_s.float() - 127.0)),
        ("byte=1 → scale=2^(byte-1)（无偏置+1）", torch.pow(2.0, hp_s.float() - 1.0)),
        ("byte=1 → scale=1.0（字节即指数 0）", torch.ones_like(hp_s, dtype=torch.float32)),
    ):
        W_hp = (Wd.reshape(4096, 16, 128) * sc.reshape(128, 16, 1).transpose(0, 1).reshape(1, 16, 128)
                if False else None)
    # 简化：全 1 scale → 直接三种常数假设
    for label, scale_val in (("2^(1-127)", 2.0 ** (1 - 127)),
                             ("2^(1-1)=1.0", 1.0),
                             ("2^1=2.0", 2.0)):
        W_hp = Wd * scale_val
        c = torch.corrcoef(torch.stack([W_hp.flatten(), W1T.flatten()]))[0, 1].item()
        rel = ((W_hp - W1T).abs().max() / W1T.abs().max()).item()
        print(f"  scale={label}: corr={c:+.4f}  max_rel_err={rel:.4f}")
    # 也试 nibble 反转（高半字节=偶列）
    lo = (hp_w & 0xF).long(); hi = (hp_w >> 4).long()
    Wd_sw = torch.empty_like(Wd)
    Wd_sw[..., 0::2] = torch.where(hi >= 8, -1.0, 1.0) * E2M1_MAG[hi & 7]
    Wd_sw[..., 1::2] = torch.where(lo >= 8, -1.0, 1.0) * E2M1_MAG[lo & 7]
    c = torch.corrcoef(torch.stack([Wd_sw.flatten(), W1T.flatten()]))[0, 1].item()
    print(f"  [nibble 反转, scale=1.0]: corr={c:+.4f}")
    # 以及不转置对照（排除 [N,K//2] 解释）
    c2 = torch.corrcoef(torch.stack([Wd.flatten(), W1.flatten()]))[0, 1].item()
    print(f"  [不转置（-hp 行=FP8 行, 即 [N,K//2] 解释）, scale=1.0]: corr={c2:+.4f}")

    print(f"\n  FP8 w1 absmax per 128-col-block 均值: "
          f"{W1T.abs().reshape(4096, 16, 128).amax(dim=(0, 2)).mean():.4f}")
    print(f"  -hp 解码 absmax per 128-col-block 均值: "
          f"{Wd.abs().reshape(4096, 16, 128).amax(dim=(0, 2)).mean():.4f}")
