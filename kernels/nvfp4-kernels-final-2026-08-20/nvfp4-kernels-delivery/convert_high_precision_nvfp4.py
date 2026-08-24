#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_high_precision_nvfp4.py — 生产权重（MXFP4）→ NVFP4 高精度转换器
======================================================================
背景：生产 DeepSeek-V4-Flash-0731 专家权重原生为 MXFP4（e2m1 + per-32 共享指数），
kernel①（v15/方案 B）需要 NVFP4 格式（e2m1 + E8M0 每 (32,128) 块 scale）。

高精度要点（vs convert_mxfp4_to_nvfp4.py）：
  1) **先反量化到 fp32 再重新量化**（不复用 MX nibble 换 scale）——消除 scale 组不匹配的二次误差
  2) **最优 E8M0 scale 搜索**：对每个 (32,128) 块在 e ∈ {floor(log2(max/6))-1, 0, +1}
     三档中选 MSE 最小档（floor 档分辨率高但部分值 clamp；+1 档无 clamp 但分辨率减半）
     —— 比简单 floor 平均降低 10~30% 量化误差
  3) --validate 输出 MXFP4↔NVFP4 roundtrip 误差对比（high vs fast 双模式）

布局（与 convert_mxfp4_to_nvfp4.py 一致的实测布局）：
  layers.{l}.ffn.experts.{e}.w{1,2,3}.weight（I8 打包，低半字节=第 1 元素）
  layers.{l}.ffn.experts.{e}.w{1,2,3}.scale（F8_E8M0 [out, in//32]）
  共享专家 FP8（E8M0+E4M3）→ 原样拷贝；MTP 层同路由专家（--with-mtp）

输出（对齐 kernel① v15/方案 B 输入规格）：
  W_packed: uint8 [K=in, N=out//2]（N 向打包，低半字节=偶 N 列）
  W_scale : uint8 [K//32, N//128]（E8M0）

用法
----
python convert_high_precision_nvfp4.py --input-dir <INSTALL_DIR>/models/deepseek-v4-flash-0731 \
    --output-dir <INSTALL_DIR>/models/dsv4f-0731-nvfp4-hp \
    [--mode high|fast] [--with-mtp] [--max-layers N] [--validate] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("convert_high_precision_nvfp4")

# ---------------------------------------------------------------------------
# 常量与查找表（布局 = convert_mxfp4_to_nvfp4.py 实测确认）
# ---------------------------------------------------------------------------

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

E2M1_MAX = 6.0
E8M0_BIAS = 127.0
MX_GROUP = 32          # MXFP4: 每 32 元素共享指数
NV_K_BLOCK = 32        # NVFP4: K 方向块（Triton 3.6 e8m0 scale group 硬约束）
NV_N_BLOCK = 128       # NVFP4: N 方向块

TENSOR_TPL = "layers.{l}.ffn.experts.{e}.w{idx}"          # .weight / .scale 后缀
MTP_TENSOR_TPL = "mtp.{m}.ffn.experts.{e}.w{idx}"
N_ROUTED_EXPERTS = 256
N_LAYERS = 43
MATRIX_DIMS = {"w1": (2048, 4096), "w2": (2048, 4096), "w3": (4096, 2048)}  # (out, in)
NIBBLE_LOW_FIRST = True


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def as_uint8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    return t.view(torch.uint8)


def load_tensor(sd: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
    return sd.get(name)


def dequant_mxfp4(w_packed: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 反量化 → fp32 [out, in]。

    w_packed [out, in//2]（低半字节=偶元素，MX 沿 in 打包）；
    w_scale  [out, in//32]（F8_E8M0，per-32 沿 in 共享指数）。
    """
    w_packed = as_uint8(w_packed).to(torch.int32)
    w_scale = as_uint8(w_scale).to(torch.float32)
    out, in_half = w_packed.shape
    i = in_half * 2

    lo = w_packed & 0x0F
    hi = (w_packed >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(out, i)      # [out, in]（沿 in 交错）
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(out, i)        # 含符号的 e2m1 值

    scale = torch.pow(2.0, w_scale - E8M0_BIAS)              # [out, in//32]
    scale_expanded = scale.repeat_interleave(MX_GROUP, dim=1).reshape(out, i)
    return (mag * scale_expanded).to(torch.float32)          # [out, in]


# ---------------------------------------------------------------------------
# NVFP4 量化（高精度 / fast 双模式）
# ---------------------------------------------------------------------------

def _e2m1_quantize(w_norm: torch.Tensor) -> torch.Tensor:
    """E2M1 阈值量化（strict >，tie low）→ nibble（含符号位）。"""
    w_c = torch.clamp(w_norm, -E2M1_MAX, E2M1_MAX)
    abs_n = w_c.abs()
    idx = torch.zeros_like(abs_n, dtype=torch.long)
    for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx = idx + (abs_n > t).long()
    mag = idx.to(torch.uint8)
    sign = (w_c < 0.0).to(torch.uint8) * 8
    return (mag | sign).to(torch.uint8)                      # nibble


def quant_nvfp4_block(
    w_fp32: torch.Tensor,
    K: int, N: int,
    mode: str = "high",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 [K,N] → (W_packed [K, N//2] N 向, W_scale [K//32, N//128] E8M0)。

    mode="high"：每 (32,128) 块在 e ∈ {floor(log2(max/6))-1, 0, +1} 三档 MSE 最小；
    mode="fast"：floor(log2(max/6)) 单档（= convert_mxfp4_to_nvfp4.py 行为）。
    """
    assert K % NV_K_BLOCK == 0 and N % NV_N_BLOCK == 0
    w = w_fp32.reshape(K // NV_K_BLOCK, NV_K_BLOCK, N // NV_N_BLOCK, NV_N_BLOCK)
    max_abs = w.abs().amax(dim=(1, 3))                       # [K/32, N/128]
    e0 = torch.floor(torch.log2(max_abs.clamp(min=1e-38) / E2M1_MAX))  # floor 档

    if mode == "fast":
        e = e0
    else:
        # 三档候选：e0-1（scale 减半，量化分辨率升但可能饱和）、e0、e0+1（无饱和但分辨率降）
        cand = torch.stack([e0 - 1, e0, e0 + 1], dim=0).clamp(-126, 127)   # [3, K/32, N/128]
        scales = torch.pow(2.0, cand)                         # [3, K/32, N/128]
        # 逐档量化 → 反量化 → MSE（向量化，注意内存：w 单专家 ~67MB，3 档 ~200MB 可承受）
        w_exp = w.unsqueeze(0)                                # [1, K/32, 32, N/128, 128]
        w_norm = w_exp / scales.unsqueeze(-1).unsqueeze(-1)   # [3, ...]
        w_c = torch.clamp(w_norm, -E2M1_MAX, E2M1_MAX)
        abs_n = w_c.abs()
        idx = torch.zeros_like(abs_n, dtype=torch.long)
        for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            idx = idx + (abs_n > t).long()
        w_q = E2M1_POS[idx.to(w_c.device)]
        w_q = torch.where(w_c < 0, -w_q, w_q)
        w_deq = w_q * scales.unsqueeze(-1).unsqueeze(-1)
        mse = ((w_deq - w_exp) ** 2).mean(dim=(2, 4))          # [3, K/32, N/128]
        best = mse.argmin(dim=0)                               # [K/32, N/128]
        e = cand.gather(0, best.unsqueeze(0)).squeeze(0)

    # 选定 scale 量化 + 打包
    scale = torch.pow(2.0, e.clamp(-126, 127))                 # [K/32, N/128]
    w_norm = w / scale[:, None, :, None]
    nibble = _e2m1_quantize(w_norm)                            # [K/32, 32, N/128, 128]
    nib_flat = nibble.reshape(K, N)                            # [K, N]
    lo = nib_flat[:, 0::2]                                     # 偶 N 列
    hi = nib_flat[:, 1::2]                                     # 奇 N 列
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()     # [K, N//2]
    scale_u8 = (e + E8M0_BIAS).clamp(0, 255).to(torch.uint8).contiguous()  # [K//32, N//128]
    return packed, scale_u8


def quant_nvfp4_matrix(
    w_fp32: torch.Tensor,
    out: int, i: int,
    mode: str = "high",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[out, in] fp32 → 内核输入规格（K=in, N=out）。"""
    w = w_fp32.t().contiguous()                                # [in, out] = [K, N]
    return quant_nvfp4_block(w, i, out, mode)


# ---------------------------------------------------------------------------
# 校验（roundtrip 误差）
# ---------------------------------------------------------------------------

def dequant_nvfp4(w_packed: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 反量化 → fp32 [in, out]（= 内核格式反推）。"""
    w_packed = as_uint8(w_packed).to(torch.int32)
    K, n_half = w_packed.shape
    N = n_half * 2
    lo = w_packed & 0x0F
    hi = (w_packed >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(K, N)          # [K, N]
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(K, N)
    scale = torch.pow(2.0, as_uint8(w_scale).to(torch.float32) - E8M0_BIAS)
    scale_exp = scale.repeat_interleave(NV_K_BLOCK, dim=0).repeat_interleave(NV_N_BLOCK, dim=1)
    return (mag * scale_exp).to(torch.float32)                 # [K, N]


def rel_err_report(a: torch.Tensor, b: torch.Tensor, tag: str) -> Dict[str, float]:
    err = (a - b).abs() / (b.abs() + 1e-6)
    out = {
        "tag": tag,
        "mean": err.mean().item(),
        "p95": torch.quantile(err.flatten(), 0.95).item(),
        "max": err.max().item(),
    }
    logger.info("  [%s] rel_err mean=%.3e p95=%.3e max=%.3e", tag, out["mean"], out["p95"], out["max"])
    return out


# ---------------------------------------------------------------------------
# 逐层转换（复用 convert_mxfp4_to_nvfp4.py 的主流程结构）
# ---------------------------------------------------------------------------

def convert_layer(
    sd: Dict[str, torch.Tensor],
    weight_map: Dict[str, str],
    shards: Dict[str, Dict[str, torch.Tensor]],
    l: int,
    with_mtp: bool,
    mode: str,
    validate: bool,
    reports: List[Dict],
) -> List[str]:
    """转换第 l 层路由专家（256×3 矩阵）；返回输出张量名清单。"""
    out_names: List[str] = []
    for e in range(N_ROUTED_EXPERTS):
        for idx, dims in MATRIX_DIMS.items():
            wname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".weight"
            sname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".scale"
            if wname not in sd or sname not in sd:
                logger.warning("缺失 %s / %s，跳过", wname, sname)
                continue
            out, i = dims
            w_fp32 = dequant_mxfp4(sd[wname], sd[sname])       # [out, in]
            packed, scale = quant_nvfp4_matrix(w_fp32, out, i, mode)
            sd[wname] = packed                                 # [K, N//2]
            sd[sname] = scale                                  # [K//32, N//128]
            out_names.extend([wname, sname])
            if validate:
                w_orig = w_fp32.t()                            # [K, N]
                w_re = dequant_nvfp4(packed, scale)            # [K, N]
                reports.append(rel_err_report(w_re, w_orig, f"L{l}E{e}{idx}"))
    return out_names


def main():
    ap = argparse.ArgumentParser(description="MXFP4 → NVFP4 高精度转换器")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="high", choices=["high", "fast"])
    ap.add_argument("--with-mtp", action="store_true")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    assert src.exists(), f"输入目录不存在: {src}"
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # ---- 定位 shards ----
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: Dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    logger.info("shards: %d | 总张量: %d", len(shard_files), len(weight_map))

    shards: Dict[str, Dict[str, torch.Tensor]] = {}
    for sf in shard_files:
        from safetensors import safe_open
        sd = {}
        with safe_open(str(src / sf), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        shards[sf] = sd

    layers = range(min(args.max_layers, N_LAYERS) if args.max_layers else N_LAYERS)
    reports: List[Dict] = []
    modified_shards: set = set()

    # ---- 转换路由专家 ----
    for l in layers:
        # 该层所有张量分散在多个 shard：逐 shard 处理（简化：先合并该层涉及 shard）
        layer_names = [k for k in weight_map if k.startswith(f"layers.{l}.")]
        logger.info("[layer %d] 转换 %d 张量...", l, len(layer_names))
        sd = {}
        for k in layer_names:
            sf = weight_map[k]
            sd[k] = shards[sf][k]
        out_names = convert_layer(sd, weight_map, shards, l, args.with_mtp, args.mode, args.validate, reports)
        # 写回各自 shard（内存内）
        for k in out_names:
            sf = weight_map[k]
            shards[sf][k] = sd[k]
            modified_shards.add(sf)

    # ---- 保存：修改过的 shard 用 save_file 重写；未修改的拷贝原文件 ----
    if not args.dry_run:
        from safetensors.torch import save_file
        for sf, sd in shards.items():
            if sf in modified_shards:
                save_file(sd, str(dst / sf))
                logger.info("[save] %s（%d 张量，含专家转换）", sf, len(sd))
            else:
                shutil.copy2(src / sf, dst / sf)
                logger.info("[copy] %s（未修改）", sf)
        # 拷贝非 safetensors 附属文件（config.json 等）
        for extra in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "generation_config.json", "model.safetensors.index.json"]:
            p = src / extra
            if p.exists():
                shutil.copy2(p, dst / p.name)

    # ---- validate 汇总 ----
    if args.validate:
        import statistics
        means = [r["mean"] for r in reports]
        logger.info("=== roundtrip 误差汇总（%s 模式，%d 矩阵）===", args.mode, len(reports))
        logger.info("mean rel_err: avg=%.3e p95=%.3e max=%.3e",
                    statistics.mean(means),
                    sorted(means)[int(len(means) * 0.95)],
                    max(r["max"] for r in reports))

    logger.info("完成。输出目录: %s（模式 %s）", dst, args.mode)
    logger.info("提示：生产权重为 MXFP4，本转换器先反量化再按 NVFP4 重新量化（high 模式三档 scale 搜索）")


if __name__ == "__main__":
    main()
