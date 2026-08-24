#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_high_precision_nvfp4.py — 生产权重（MXFP4）→ NVFP4 高精度转换器（修正版 v2）
====================================================================================
修正点（engineering-assurance 审阅 2026-08-20，基于生产 safetensors 实测）：
  1) MATRIX_DIMS w2/w3 的 (out,in) 写反 → 改为从实际张量动态推断 out/in，
     不再依赖硬编码常量（消除转置错位风险）。
  2) convert_layer 增加 dequant 后 shape 断言（保证 out/in 合法、可被 32×128 块整除）。

背景：生产 DeepSeek-V4-Flash-0731 专家权重原生为 MXFP4（e2m1 + per-32 共享指数 E8M0），
kernel①（routeA）需要 NVFP4 格式（e2m1 + E8M0 每 (32,128) 块 scale）。

高精度要点：
  1) 先反量化到 fp32 再重新量化（消除 scale 组不匹配的二次误差）
  2) 最优 E8M0 scale 搜索：每 (32,128) 块在 e ∈ {floor(log2(max/6))-1, 0, +1} 三档 MSE 最小
  3) --validate 输出 roundtrip 误差对比

生产实测布局（2026-08-20 safetensors）：
  w1.weight [2048,2048] int8 / scale [2048,128] E8M0   → out=2048, in=4096
  w2.weight [4096,1024] int8 / scale [4096,64]  E8M0   → out=4096, in=2048
  w3.weight [2048,2048] int8 / scale [2048,128] E8M0   → out=2048, in=4096
  打包：safetensors 存储 whole-expert 为 [out, K_in_nibbles]，规模 = out × (in//2)。
  MX 沿 in 打包，per-32 共享指数；nibble 沿 in 交错（低半字节=偶第1元素）。

输出（对齐 routeA 内核输入规格 nvfp4_4w4a_mmaf.py）：
  W_packed: uint8 [K=in, N=out//2]（N 向打包，低半字节=偶 N 列）
  W_scale : uint8 [K//32, N//128]（E8M0）

用法
----
python convert_high_precision_nvfp4.py --input-dir /models \
    --output-dir <INSTALL_DIR>/nvfp4/models/dsv4f-0731-nvfp4-hp \
    [--mode high|fast] [--with-mtp] [--max-layers N] [--validate] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("convert_high_precision_nvfp4")

try:
    import torch
except ImportError:
    sys.stderr.write("ERROR: 需要 torch（请在 vllm 容器内运行）\n")
    sys.exit(2)

# ---------------------------------------------------------------------------
# 常量与查找表
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

TENSOR_TPL = "layers.{l}.ffn.experts.{e}.{idx}"          # .weight / .scale 后缀
MTP_TENSOR_TPL = "mtp.{m}.ffn.experts.{e}.{idx}"
N_ROUTED_EXPERTS = 256
N_LAYERS = 43
MX_WEIGHTS = {"w1", "w2", "w3"}      # MXFP4 专家矩阵
NIBBLE_LOW_FIRST = True


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def as_uint8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    return t.view(torch.uint8)


def load_tensor(f, name: str) -> Optional[torch.Tensor]:
    return f.get_tensor(name)


def dequant_mxfp4(w_packed: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 反量化 → fp32 [out, in]。

    w_packed [out, in//2]（低半字节=偶第1元素，MX 沿 in 打包）；
    w_scale  [out, in//32]（F8_E8M0，per-32 沿 in 共享指数）。
    """
    orig_dtype = w_packed.dtype
    if w_packed.dtype.is_floating_point:
        # 个别存储为 fp8_e4m3；先按字节解释（safetensors raw bytes）
        w_packed_b = w_packed.view(torch.uint8)
    else:
        w_packed_b = as_uint8(w_packed)
    w_packed_i = w_packed_b.to(torch.int32)
    w_scale_f = w_scale.float()

    out, in_half = w_packed_i.shape
    i = in_half * 2

    lo = w_packed_i & 0x0F
    hi = (w_packed_i >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(out, i)      # [out, in]
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(out, i)

    scale = torch.pow(2.0, w_scale_f - E8M0_BIAS)            # [out, in//32]
    scale_expanded = scale.repeat_interleave(MX_GROUP, dim=1).reshape(out, i)
    return (mag * scale_expanded).to(torch.float32)


# ---------------------------------------------------------------------------
# NVFP4 量化（高精度 / fast 双模式）
# ---------------------------------------------------------------------------

def _e2m1_quantize(w_norm: torch.Tensor) -> torch.Tensor:
    w_c = torch.clamp(w_norm, -E2M1_MAX, E2M1_MAX)
    abs_n = w_c.abs()
    idx = torch.zeros_like(abs_n, dtype=torch.long)
    for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx = idx + (abs_n > t).long()
    mag = idx.to(torch.uint8)
    sign = (w_c < 0.0).to(torch.uint8) * 8
    return (mag | sign)


def quant_nvfp4_block(w_fp32, K, N, mode="high"):
    w = w_fp32.reshape(K // NV_K_BLOCK, NV_K_BLOCK, N // NV_N_BLOCK, NV_N_BLOCK)
    max_abs = w.abs().amax(dim=(1, 3))                       # [K/32, N/128]
    e0 = torch.floor(torch.log2(max_abs.clamp(min=1e-38) / E2M1_MAX))

    if mode == "fast":
        e = e0
    else:
        cand = torch.stack([e0 - 1, e0, e0 + 1], dim=0).clamp(-126, 127)
        scales = torch.pow(2.0, cand)
        w_exp = w.unsqueeze(0)
        w_norm = w_exp / scales.unsqueeze(-1).unsqueeze(-1)
        w_c = torch.clamp(w_norm, -E2M1_MAX, E2M1_MAX)
        abs_n = w_c.abs()
        idx = torch.zeros_like(abs_n, dtype=torch.long)
        for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            idx = idx + (abs_n > t).long()
        w_q = E2M1_POS[idx.to(w_c.device)]
        w_q = torch.where(w_c < 0, -w_q, w_q)
        w_deq = w_q * scales.unsqueeze(-1).unsqueeze(-1)
        mse = ((w_deq - w_exp) ** 2).mean(dim=(2, 4))
        best = mse.argmin(dim=0)
        e = cand.gather(0, best.unsqueeze(0)).squeeze(0)

    scale = torch.pow(2.0, e.clamp(-126, 127))
    w_norm = w / scale[:, None, :, None]
    nibble = _e2m1_quantize(w_norm)                          # [K/32,32,N/128,128]
    nib_flat = nibble.reshape(K, N)
    lo = nib_flat[:, 0::2]
    hi = nib_flat[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()    # [K, N//2]
    scale_u8 = (e + E8M0_BIAS).clamp(0, 255).to(torch.uint8).contiguous()
    return packed, scale_u8


def quant_nvfp4_matrix(w_fp32, out, i, mode="high"):
    """[out, in] fp32 → 内核输入规格（K=in, N=out）。"""
    w = w_fp32.t().contiguous()                               # [in, out] = [K, N]
    return quant_nvfp4_block(w, i, out, mode)


# ---------------------------------------------------------------------------
# 校验（roundtrip 误差）
# ---------------------------------------------------------------------------

def dequant_nvfp4(w_packed, w_scale):
    """NVFP4 反量化 → fp32 [in, out]（= 内核格式反推）。"""
    w_packed_i = as_uint8(w_packed).to(torch.int32)
    K, n_half = w_packed_i.shape
    N = n_half * 2
    lo = w_packed_i & 0x0F
    hi = (w_packed_i >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(K, N)
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(K, N)
    scale = torch.pow(2.0, w_scale.float() - E8M0_BIAS)
    scale_exp = scale.repeat_interleave(NV_K_BLOCK, dim=0).repeat_interleave(NV_N_BLOCK, dim=1)
    return (mag * scale_exp).to(torch.float32)


def rel_err_report(a, b, tag):
    err = (a - b).abs() / (b.abs() + 1e-6)
    out = {
        "tag": tag,
        "mean": err.mean().item(),
        "p95": torch.quantile(err.flatten(), 0.95).item(),
        "max": err.max().item(),
    }
    logger.info("  [%s] rel_err mean=%.3e p95=%.3e max=%.3e",
                tag, out["mean"], out["p95"], out["max"])
    return out


# ---------------------------------------------------------------------------
# 逐层转换
# ---------------------------------------------------------------------------

def convert_layer(sd, weight_map, l, with_mtp, mode, validate, reports, changed):
    out_names: List[str] = []
    for e in range(N_ROUTED_EXPERTS):
        for idx in sorted(MX_WEIGHTS):
            wname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".weight"
            sname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".scale"
            if wname not in sd or sname not in sd:
                logger.warning("缺失 %s / %s，跳过", wname, sname)
                continue
            w_packed = sd[wname]
            w_scale = sd[sname]
            # 动态推断 (out, in)，不再依赖硬编码 MATRIX_DIMS
            out = w_packed.shape[0]
            in_half = w_packed.shape[1]
            i = in_half * 2
            # 校验可被块整除 + scale 维度一致
            assert i % NV_K_BLOCK == 0, f"{wname} in={i} 不能 32 整除"
            assert out % NV_N_BLOCK == 0, f"{wname} out={out} 不能 128 整除"
            assert w_scale.shape == (out, i // 32), (
                f"{sname} scale shape {tuple(w_scale.shape)} != {(out, i//32)}")
            w_fp32 = dequant_mxfp4(w_packed, w_scale)          # [out, in]
            packed, scale = quant_nvfp4_matrix(w_fp32, out, i, mode)
            sd[wname] = packed                                  # [K, N//2] = [in, out//2]
            sd[sname] = scale                                   # [K//32, N//128] = [in//32, out//128]
            changed.add(wname)
            changed.add(sname)
            out_names.extend([wname, sname])
            if validate:
                w_orig = w_fp32.t()                             # [K, N] = [in, out]
                w_re = dequant_nvfp4(packed, scale)             # [K, N]
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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    assert src.exists(), f"输入目录不存在: {src}"
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    idx_path = src / "model.safetensors.index.json"
    assert idx_path.exists(), f"缺 index: {idx_path}"
    index = json.loads(idx_path.read_text())
    weight_map: Dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    logger.info("[probe] shards=%d, tensors=%d, max_layers=%s",
                len(shard_files), len(weight_map), args.max_layers)

    from safetensors import safe_open
    shards: Dict[str, Dict[str, torch.Tensor]] = {}
    for sf in shard_files:
        sd = {}
        with safe_open(str(src / sf), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        shards[sf] = sd

    layers = range(min(args.max_layers, N_LAYERS) if args.max_layers else N_LAYERS)
    reports: List[Dict] = []
    modified_shards = set()
    changed = set()

    for l in layers:
        layer_names = [k for k in weight_map if k.startswith(f"layers.{l}.")]
        sd = {}
        for k in layer_names:
            sf = weight_map[k]
            sd[k] = shards[sf][k]
        sd = dict(sd)  # 副本，避免污染 shards
        out_names = convert_layer(sd, weight_map, l, args.with_mtp, args.mode,
                                  args.validate, reports, changed)
        for k in out_names:
            sf = weight_map[k]
            shards[sf][k] = sd[k]
            modified_shards.add(sf)
        logger.info("[layer %d] 完成 %d 专家矩阵", l, len(out_names) // 2)

    if not args.dry_run:
        from safetensors.torch import save_file
        for sf, sd in shards.items():
            if sf in modified_shards:
                save_file(sd, str(dst / sf))
                logger.info("[save] %s", sf)
            else:
                shutil.copy2(src / sf, dst / sf)
                logger.info("[copy] %s（未改）", sf)
        for extra in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "generation_config.json"]:
            p = src / extra
            if p.exists():
                shutil.copy2(p, dst / p.name)
        # 改写 index 只保留已转换张量映射（保持文件名不变，映射不变，直接拷贝即可）
        shutil.copy2(idx_path, dst / idx_path.name)
        logger.info("[index] 已拷贝 %s", idx_path.name)

    if args.validate:
        import statistics
        means = [r["mean"] for r in reports]
        logger.info("=== roundtrip 误差汇总（%s 模式，%d 矩阵）===", args.mode, len(reports))
        if means:
            logger.info("mean rel_err: avg=%.3e p95=%.3e max=%.3e",
                        statistics.mean(means),
                        sorted(means)[int(len(means) * 0.95)],
                        max(r["max"] for r in reports))

    logger.info("完成。输出目录: %s（模式 %s，dry_run=%s）", dst, args.mode, args.dry_run)


if __name__ == "__main__":
    main()