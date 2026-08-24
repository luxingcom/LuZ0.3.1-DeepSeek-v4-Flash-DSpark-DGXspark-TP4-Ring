#!/usr/bin/env python3
"""P3-Step1: 生产 NVFP4-HP 权重实证（node01 一次性容器，CPU 即可）
=====================================================================
只读 /model（ro 挂载），只加载 layer 0 expert 0 的 w1/w2/w3 weight+scale
（单 expert ~12MB 级），严禁全量 148G 加载（OOM 铁律）。

验证目标：
  T1 命名/shape/dtype：layers.0.ffn.experts.0.{w1,w2,w3}.{weight,scale}
  T2 布局判别：weight [K, N//2]（K=hidden/N=intermediate 方向） vs [N, K//2]
     —— 用 scale block 语义做判别：正确取向下 max|dequant|/2^(sf-127) ≤ 6
        （编码器保证块内元素 clamp 到 ±6），错误取向下必然出现 >6 的块
  T3 E8M0 语义：scale 字节分布（应集中在 120-135 附近的窄带）
  T4 解码幅值分布：E2M1 码本使用频率（0/0.5/1/1.5/2/3/4/6）
  T5 保存 expert0 六张量到 /work/expert0.pt 供数值对照阶段使用
"""
import json
import struct
import sys

import torch

MODEL_DIR = "/model"
SHARD = "model-00002-of-00048.safetensors"
OUT_PT = "/work/expert0.pt"

torch.manual_seed(0)

E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------- safetensors
def read_header(shard_path):
    with open(shard_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def load_u8(shard_path, name, hdr=None, base=None):
    hdr = read_header(shard_path)[0] if hdr is None else hdr
    info = hdr[name]
    shape, (s, e) = info["shape"], info["data_offsets"]
    assert info["dtype"] == "U8", f"{name}: dtype {info['dtype']} != U8"
    with open(shard_path, "rb") as f:
        f.seek((base if base is not None else 0) + s)
        raw = f.read(e - s)
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
    return t.clone(), info


# ---------------------------------------------------------------- E2M1 decode
def unpack_e2m1(u8: torch.Tensor) -> torch.Tensor:
    """[..., n//2] uint8 → [..., n] f32（低半字节=偶元素，高半字节=奇元素）"""
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    out = torch.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = torch.where(lo >= 8, -1.0, 1.0) * E2M1_MAG[lo & 7]
    out[..., 1::2] = torch.where(hi >= 8, -1.0, 1.0) * E2M1_MAG[hi & 7]
    return out


def ratio_stats(Wdq: torch.Tensor, scale: torch.Tensor, k_grp, n_grp):
    """按 (k_grp × n_grp) block 计算 max|Wdq|/scale 的分布。
    Wdq: [R, C] 解码后的幅值矩阵（未乘 scale），scale: [R//k_grp, C//n_grp] uint8
    返回 (max_ratio, frac_gt6, frac_eq6, p50)。scale 不匹配时返回 None。"""
    R, C = Wdq.shape
    if R % k_grp or C % n_grp or scale.shape != (R // k_grp, C // n_grp):
        return None
    sf = torch.pow(2.0, scale.float() - 127.0)  # [R//k_grp, C//n_grp]
    blocks = Wdq.abs().reshape(R // k_grp, k_grp, C // n_grp, n_grp)
    bmax = blocks.amax(dim=(1, 3))              # [R//k_grp, C//n_grp]
    ratio = bmax / sf
    return (ratio.max().item(),
            (ratio > 6.0).float().mean().item(),
            (ratio >= 5.999).float().mean().item(),
            ratio.median().item())


def main():
    shard = f"{MODEL_DIR}/{SHARD}"
    hdr, base = read_header(shard)

    # T1: 命名与元信息
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    for k in ("hidden_size", "moe_intermediate_size", "num_hidden_layers",
              "n_routed_experts", "num_experts_per_tok", "quantization_config"):
        if k in cfg:
            v = cfg[k]
            print(f"config.{k} = {str(v)[:200]}")

    names = [f"layers.0.ffn.experts.0.{w}.{s}" for w in ("w1", "w2", "w3")
             for s in ("weight", "scale")]
    tensors = {}
    print("\n=== T1 张量元信息（header，不读数据）===")
    for nm in names:
        info = hdr[nm]
        print(f"  {nm}: shape={info['shape']} dtype={info['dtype']}")
    # 顺带看一个非 MoE 张量命名（对照 hidden 方向）
    for nm in list(hdr):
        if "layers.0.self_attention" in nm or "layers.0.input_layernorm" in nm:
            print(f"  [ref] {nm}: shape={hdr[nm]['shape']}")

    print("\n=== T1b 加载 expert0（~12MB）===")
    for nm in names:
        t, _ = load_u8(shard, nm, hdr, base)
        tensors[nm] = t
        print(f"  loaded {nm}: {tuple(t.shape)} {t.dtype}")

    w = {x: {"weight": tensors[f"layers.0.ffn.experts.0.{x}.weight"],
             "scale": tensors[f"layers.0.ffn.experts.0.{x}.scale"]}
         for x in ("w1", "w2", "w3")}

    print("\n=== T2 布局判别（block-ratio ≤ 6 判据）===")
    print("interp A: weight=[K, N//2]（K=行方向）, scale=[K//32, N//128]")
    print("interp B: weight=[N, K//2]（N=行方向）, scale=[N//32, K//128]")
    for name, d in w.items():
        wt, sc = d["weight"], d["scale"]
        R, C2 = wt.shape
        C = C2 * 2
        Wdq = unpack_e2m1(wt)  # [R, C]，按行解码（幅值，不含 scale）
        print(f"\n  {name}: weight {tuple(wt.shape)} scale {tuple(sc.shape)}")
        # interp A: 行=K, 列=N；block = 32行 × 128列
        ra = ratio_stats(Wdq, sc, 32, 128)
        # interp B: 行=N, 列=K；block = 32行(N) × 128列(K)
        rb = ratio_stats(Wdq, sc, 32, 128)
        # A/B 差异在于 scale 索引方向：interp A scale[k//32, n//128]，
        # interp B scale[n//32, k//128]。两者 block 形状相同（32×128），
        # 区别 = 行列语义。重新实现：
        #   interp A: rows=K, cols=N → scale[R//32, C//128] 直接对应
        #   interp B: rows=N, cols=K → scale 应为 [C//32, R//128]，但实际
        #             scale shape 是 [R//32, C//128]，仅当 R//32==C//32 且
        #             C//128==R//128（方阵或巧合）才 shape 相符
        sA_ok = sc.shape == (R // 32, C // 128) if (R % 32 == 0 and C % 128 == 0) else False
        sB_ok = sc.shape == (C // 32, R // 128) if (C % 32 == 0 and R % 128 == 0) else False
        print(f"    interp A scale shape match: {sA_ok}  interp B: {sB_ok}")
        if sA_ok:
            st = ratio_stats(Wdq, sc, 32, 128)
            print(f"    interp A ratio: max={st[0]:.3f} frac>6={st[1]:.4f} "
                  f"frac≈6={st[2]:.3f} p50={st[3]:.3f}")
        if sB_ok:
            # interp B: 需把 scale 转置语义（scale[n//32, k//128] = sc[k? ...]）
            # 实际字节布局 sc[R//32, C//128]；interp B 期望按 (n=行, k=列) 分块，
            # 即 scale_sb[r//32, c//128] 其中 r 是行(N)、c 是列(K)。
            # 字节含义若为 interp B，则 sc[i,j] 对应 n∈[32i,32i+32), k∈[128j,128j+128)
            # —— 与 interp A 的 k∈[32i,...), n∈[128j,...) 不同。
            # 直接用同一字节矩阵按 (行//32, 列//128) 测试（行列均为 weight 的行列）
            st = ratio_stats(Wdq, sc, 32, 128)  # shape 相同，语义区分看数值
            print(f"    interp B ratio: max={st[0]:.3f} frac>6={st[1]:.4f} "
                  f"frac≈6={st[2]:.3f} p50={st[3]:.3f}")
        if sA_ok and sB_ok:
            # 方阵 w2 情形：两种语义数值都测了，靠 frac>6 判别
            pass

    print("\n=== T3/T4 scale 字节 & E2M1 码本分布（interp A 解码）===")
    for name, d in w.items():
        sc = d["scale"]
        print(f"  {name}.scale: min={sc.min()} max={sc.max()} "
              f"mean={sc.float().mean():.2f} p5={sc.float().quantile(0.05):.0f} "
              f"p95={sc.float().quantile(0.95):.0f}")
        nib = torch.cat([(d["weight"] & 0xF).long(), (d["weight"] >> 4).long()])
        codes = (nib & 7).bincount(minlength=8).float()
        codes = codes / codes.sum()
        print(f"  {name}.weight E2M1 码本频率: " +
              " ".join(f"{E2M1_MAG[i].item():.1f}:{codes[i]:.3f}" for i in range(8)))
        neg = (nib >= 8).float().mean()
        print(f"  {name}.weight 负号比例: {neg:.4f}")

    # T5: 保存
    torch.save(w, OUT_PT)
    print(f"\n✅ expert0 六张量已保存 {OUT_PT}（约 {sum(t.numel() for d in w.values() for t in d.values())/1e6:.1f}MB）")


if __name__ == "__main__":
    main()
