#!/usr/bin/env python3
"""P3-Step1e: -hp 块 scale 逐块恢复 + -nvfp4 链路判定
对每个 32(K)×128(N) 块：s_implied = bmax(W_src)/max_code(-hp)，
检验 (a) quant(W_src/s) 与 -hp 码的逐块匹配率 (b) s 是否 2 的幂。
W_src 候选：-0731 MXF4 dequant 与 -nvfp4 dequant。
"""
import json
import struct

import torch

M_HP = "/model"
M_MX = "/model_base"
M_NV = "/model_src"

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


def pack_e2m1(x):
    d = (x.abs().unsqueeze(-1) - E2M1_MAG).abs()
    idx = d.argmin(-1)
    nib = torch.where(x < 0, 8, 0) + idx
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)


def shard_for(root, name):
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    return f"{root}/{idx[name]}"


# ---------- W_src #1: -0731 MXF4 ----------
mx_w, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
mx_s, _ = load_raw(shard_for(M_MX, "layers.0.ffn.experts.0.w1.scale"),
                   "layers.0.ffn.experts.0.w1.scale")
N, K = mx_w.shape[0], mx_w.shape[1] * 2
codes_mx = unpack_e2m1(mx_w)
sf = torch.pow(2.0, mx_s.float() - 127.0)
W_mx = (codes_mx.reshape(N, K // 32, 32) * sf[:, :, None]).reshape(N, K).t().contiguous()

# ---------- W_src #2: -nvfp4 ----------
nv_w, _ = load_raw(shard_for(M_NV, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
nv_s1, dt1 = load_raw(shard_for(M_NV, "layers.0.ffn.experts.0.w1.weight_scale"),
                      "layers.0.ffn.experts.0.w1.weight_scale")
nv_s2, dt2 = load_raw(shard_for(M_NV, "layers.0.ffn.experts.0.w1.weight_scale_2"),
                      "layers.0.ffn.experts.0.w1.weight_scale_2")
print(f"-nvfp4: weight {tuple(nv_w.shape)}, weight_scale {tuple(nv_s1.shape)} {dt1}, "
      f"weight_scale_2 {tuple(nv_s2.shape)} {dt2}")

# NVFP4 两级 scale：weight_scale e4m3 [N, K//16]，weight_scale_2 f32 [N//128, K//128]?
W_nv = None
try:
    if dt1.upper() == "F8_E4M3":
        s1 = nv_s1.view(torch.float8_e4m3fn).float()   # [N, K//16]
    else:
        s1 = nv_s1.float()
    if dt2.upper() == "F32":
        s2 = nv_s2.view(torch.float32).reshape(-1)[0].item()
    elif dt2.upper() == "BF16":
        s2 = nv_s2.view(torch.bfloat16).reshape(-1)[0].item()
    else:
        s2 = nv_s2.reshape(-1)[0].item()
    print(f"  s1 (e4m3) range: [{s1.min():.4g},{s1.max():.4g}]  s2 scalar = {s2:.6g}")
    codes_nv = unpack_e2m1(nv_w)                        # [N, K]
    Nn, Kn = codes_nv.shape
    s1_k = s1.reshape(Nn, Kn // 16, 1).expand(Nn, Kn // 16, 16).reshape(Nn, Kn)
    W_nv = (codes_nv * s1_k * s2).t().contiguous()     # [K, N]
    print(f"  W_nv dequant: std={W_nv.std():.5f} absmax={W_nv.abs().max():.4f}")
except Exception as ex:
    import traceback; traceback.print_exc()
    print(f"  -nvfp4 dequant 失败: {ex}")

# ---------- -hp ----------
hp_w, _ = load_raw(shard_for(M_HP, "layers.0.ffn.experts.0.w1.weight"),
                   "layers.0.ffn.experts.0.w1.weight")
codes_hp = unpack_e2m1(hp_w)                            # [K, N]
K_, N_ = codes_hp.shape
print(f"\n-hp codes [K={K_}, N={N_}]")

# ---------- 逐块隐含 scale 恢复 ----------
def block_recovery(W_src, label):
    blocks_src = W_src.reshape(K_ // 32, 32, N_ // 128, 128)
    blocks_hp = codes_hp.reshape(K_ // 32, 32, N_ // 128, 128)
    bmax = blocks_src.abs().amax(dim=(1, 3))            # [K//32, N//128]
    hmax = blocks_hp.abs().amax(dim=(1, 3))
    s_imp = bmax / hmax.clamp(min=0.5)                  # 隐含 scale
    # 检验幂次性
    e = torch.log2(s_imp)
    frac = (e - torch.round(e)).abs()
    is_pow2 = frac < 1e-4
    # 全局重码匹配：quant(W_src / s_imp)
    Wn = (blocks_src / s_imp[:, None, :, None]).reshape(K_, N_).clamp(-6, 6)
    match = (pack_e2m1(Wn) == hp_w).float().mean().item()
    # 逐块完全匹配率
    blk_match = ((pack_e2m1(Wn) == hp_w)
                 .reshape(K_ // 32, 32, N_ // 128, 512 // 512)
                 if False else None)
    print(f"\n=== {label} ===")
    print(f"  s_implied: p5={s_imp.quantile(.05):.4g} p50={s_imp.median():.4g} "
          f"p95={s_imp.quantile(.95):.4g}")
    print(f"  幂次比例: {is_pow2.float().mean():.3f}  "
          f"log2 范围: [{e.min():.2f}, {e.max():.2f}]")
    print(f"  quant(W_src/s_imp) 码匹配率: {match:.4f}")
    # 隐含 E8M0 字节（若幂次）
    print(f"  隐含字节 e+127: p5={(e.quantile(.05)+127):.0f} "
          f"p50={(e.median()+127):.0f} p95={(e.quantile(.95)+127):.0f}")
    return s_imp, match

s1_imp, m1 = block_recovery(W_mx, "-0731 MXF4 dequant 源")
if W_nv is not None:
    s2_imp, m2 = block_recovery(W_nv, "-nvfp4 dequant 源")

# 块内逐块完全匹配统计（对最优源）
def per_block_full(W_src):
    blocks_src = W_src.reshape(K_ // 32, 32, N_ // 128, 128)
    blocks_hp = codes_hp.reshape(K_ // 32, 32, N_ // 128, 128)
    bmax = blocks_src.abs().amax(dim=(1, 3))
    hmax = blocks_hp.abs().amax(dim=(1, 3))
    s = bmax / hmax.clamp(min=0.5)
    Wn = (blocks_src / s[:, None, :, None]).clamp(-6, 6)
    eq = (pack_e2m1(Wn.reshape(K_, N_)) == hp_w).reshape(K_ // 32, 32, N_ // 128, N_ // 2 // (N_ // 128))
    full = eq.all(dim=(1, 3)).float().mean().item()
    return full

print(f"\n逐块完全匹配率（MXF4 源）: {per_block_full(W_mx):.4f}")
if W_nv is not None:
    print(f"逐块完全匹配率（nvfp4 源）: {per_block_full(W_nv):.4f}")
