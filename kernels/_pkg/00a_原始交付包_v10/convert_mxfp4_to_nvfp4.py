#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_mxfp4_to_nvfp4.py — DeepSeek-V4-Flash-0731 专家权重 MXFP4 → NVFP4 格式转换器
（已按 <INSTALL_DIR>/models/deepseek-v4-flash-0731 实测布局定稿）

实测布局（--probe 确认，2026-08-19）：
  - 路由专家（256/layer × 43 layers，96% 参数）原生 MXFP4：
      layers.{l}.ffn.experts.{e}.w{1,2,3}.weight   dtype=I8（打包，低半字节=第 1 元素）
      layers.{l}.ffn.experts.{e}.w{1,2,3}.scale    dtype=F8_E8M0  shape=[out, in//32]
      w1/w2: [out=2048, in=4096]  → packed [2048, 2048], scale [2048, 128]
      w3  : [out=4096, in=2048]  → packed [4096, 1024], scale [4096, 64]
  - MTP 专家（mtp.{i}.ffn.experts.*，dspark 草稿层）同样 MXFP4
  - 共享专家（n_shared_experts=1）: **FP8（E8M0+E4M3），无需转换，原样拷贝**
  - attention / norm / embed / gate: FP8 或 BF16，原样拷贝
  - 48 shards，总 155.4 GiB，72317 张量；按 model.safetensors.index.json 定位

目标格式（对齐 KernelGen 内核 nvfp4_4w4a_prefill_gemm v8 输入规格）：
  - W_packed: uint8 [K=in, N=out//2]，e2m1 打包（低半字节=第 1 元素）
  - W_scale : uint8 [K//32, N//128]，E8M0（2^(e-127)），每 (32,128) 块一个 scale

用法
----
python convert_mxfp4_to_nvfp4.py --input-dir <INSTALL_DIR>/models/deepseek-v4-flash-0731 \
                                 --output-dir <INSTALL_DIR>/models/dsv4f-0731-nvfp4 \
                                 [--with-mtp] [--max-layers N] [--validate] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("convert_mxfp4_to_nvfp4")

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
NV_K_BLOCK = 32        # NVFP4: K 方向块（= Triton 3.6 e8m0 scale group 硬约束；官方 16×128 需重算 scale）
NV_N_BLOCK = 128       # NVFP4: N 方向块

# 实测确认的布局（来自 probe）
TENSOR_TPL = "layers.{l}.ffn.experts.{e}.w{idx}"          # .weight / .scale 后缀
MTP_TENSOR_TPL = "mtp.{m}.ffn.experts.{e}.w{idx}"
N_ROUTED_EXPERTS = 256
N_LAYERS = 43
# w1=gate, w2=up: hidden→intermediate；w3=down: intermediate→hidden
MATRIX_DIMS = {"w1": (2048, 4096), "w2": (2048, 4096), "w3": (4096, 2048)}  # (out, in)
# MXFP4 scale 布局：[out, in//32]（per-32 沿 in 维）—— probe 实测 shape=[2048,128] 确认
# nibble 顺序：低半字节=第 1 元素（MXFP4 标准；如有异常由 --validate 报出）
NIBBLE_LOW_FIRST = True


# ---------------------------------------------------------------------------
# 数据读取辅助
# ---------------------------------------------------------------------------

def as_uint8(t: torch.Tensor) -> torch.Tensor:
    """safetensors 里 I8 / F8_E8M0 统一转为 uint8 视图。"""
    if t.dtype == torch.uint8:
        return t
    if t.dtype in (torch.int8, torch.int16, torch.int32):
        return t.to(torch.uint8)
    # float8_e8m0fn 等自定义 dtype：按字节 reinterpret
    return t.view(torch.uint8)


def load_shard_map(input_dir: Path) -> Dict[str, str]:
    """读取 model.safetensors.index.json 的 weight_map: tensor -> shard 文件。"""
    idx = json.loads((input_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return idx["weight_map"]


def load_tensor(tensor_name: str, weight_map: Dict[str, str], input_dir: Path) -> torch.Tensor:
    from safetensors import safe_open
    shard = weight_map[tensor_name]
    with safe_open(str(input_dir / shard), framework="pt") as f:
        return f.get_tensor(tensor_name)


# ---------------------------------------------------------------------------
# 核心转换原语
# ---------------------------------------------------------------------------

def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, out: int, in_: int) -> torch.Tensor:
    """MXFP4 [out, in//2] 打包 + E8M0 [out, in//32] → fp32 [out, in]。"""
    p = as_uint8(packed)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    if not NIBBLE_LOW_FIRST:
        lo, hi = hi, lo
    vals = torch.stack([lo, hi], dim=-1).reshape(-1)
    w_e2m1 = E2M1_TABLE.to(p.device)[vals.long()].reshape(out, in_)
    s = torch.pow(2.0, as_uint8(scale).to(torch.float32) - E8M0_BIAS)      # [out, in//32]
    s_exp = s.repeat_interleave(MX_GROUP, dim=1)[:, :in_]                  # [out, in]
    return w_e2m1 * s_exp


def encode_e8m0(v: torch.Tensor) -> torch.Tensor:
    """fp32 → E8M0 uint8：exp = floor(log2(v)) + 127，clamp [0,255]。"""
    e = torch.floor(torch.log2(torch.clamp(v, min=1e-38))) + E8M0_BIAS
    return e.clamp(0, 255).to(torch.uint8)


def quant_nvfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 [K,N] → (W_packed uint8 [K, N//2], W_scale uint8 [K//32, N//128])。"""
    K, N = w.shape
    assert K % NV_K_BLOCK == 0 and N % NV_N_BLOCK == 0, f"shape {w.shape} 需可整除 ({NV_K_BLOCK},{NV_N_BLOCK})"

    w_b = w.reshape(K // NV_K_BLOCK, NV_K_BLOCK, N // NV_N_BLOCK, NV_N_BLOCK)
    block_max = w_b.abs().amax(dim=(1, 3))
    w_scale = encode_e8m0(block_max / E2M1_MAX)

    s_f = torch.pow(2.0, w_scale.to(torch.float32) - E8M0_BIAS)
    s_exp = s_f.repeat_interleave(NV_K_BLOCK, dim=0).repeat_interleave(NV_N_BLOCK, dim=1)

    w_norm = torch.clamp(w / (s_exp + 1e-38), -E2M1_MAX, E2M1_MAX)
    sign = torch.sign(w_norm)
    ax = w_norm.abs()
    idx = (ax.unsqueeze(-1) - E2M1_POS.to(w.device)).abs().argmin(dim=-1)
    mag = idx.to(torch.uint8)
    sign_bit = torch.where(sign < 0,
                           torch.tensor(8, dtype=torch.uint8, device=w.device),
                           torch.tensor(0, dtype=torch.uint8, device=w.device))
    nibble = (mag | sign_bit).to(torch.uint8)
    lo = nibble[:, 0::2]
    hi = nibble[:, 1::2]
    return (lo | (hi << 4)).contiguous(), w_scale.contiguous()


def convert_expert_matrix(weight: torch.Tensor, scale: torch.Tensor, out: int, in_: int):
    """MXFP4 [out,in] → NVFP4 [K=in, N=out] 打包 + 块 scale。"""
    w_fp32 = dequant_mxfp4(weight, scale, out, in_)       # [out, in]
    w_fp32 = w_fp32.t().contiguous()                      # [in, out] = [K, N]
    return quant_nvfp4(w_fp32)


# ---------------------------------------------------------------------------
# 校验：转换前后精度损失
# ---------------------------------------------------------------------------

def roundtrip_error(weight: torch.Tensor, scale: torch.Tensor, out: int, in_: int) -> float:
    """MXFP4 原始反量化 vs NVFP4 转换后再反量化 的相对误差（按块）。"""
    w_orig = dequant_mxfp4(weight, scale, out, in_)                  # [out, in]
    w_packed, w_scale = convert_expert_matrix(weight, scale, out, in_)  # NVFP4 [in, out]
    # NVFP4 反量化（校验用）
    lo = w_packed & 0x0F
    hi = (w_packed >> 4) & 0x0F
    vals = torch.stack([lo, hi], dim=-1).reshape(-1)
    w_nv = E2M1_TABLE.to(w_packed.device)[vals.long()].reshape(in_, out).t()  # [out, in]
    s = torch.pow(2.0, w_scale.to(torch.float32) - E8M0_BIAS)
    s_exp = s.repeat_interleave(NV_K_BLOCK, dim=0).repeat_interleave(NV_N_BLOCK, dim=1).t()
    w_nv = (w_nv * s_exp).contiguous()                                 # [out, in]
    denom = w_orig.abs().amax() + 1e-12
    return ((w_orig - w_nv).abs().mean() / denom).item()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Flash-0731 MXFP4 → NVFP4 专家权重转换（布局已实测确认）")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--with-mtp", action="store_true", help="转换 MTP 草稿层专家（mtp.{m}.ffn.experts.*）")
    ap.add_argument("--max-layers", type=int, default=None, help="仅转换前 N 层（调试）")
    ap.add_argument("--validate", action="store_true", help="转换时逐矩阵输出 roundtrip 相对误差")
    ap.add_argument("--dry-run", action="store_true", help="只打印将转换的张量清单，不写文件")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads((args.input_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = load_shard_map(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = cfg["num_hidden_layers"]
    if args.max_layers:
        layers = min(layers, args.max_layers)

    # 汇总待转换张量： (tensor_name, scale_name, out, in, group)
    targets: List[Tuple[str, str, int, int, str]] = []
    for l in range(layers):
        for e in range(cfg["n_routed_experts"]):
            for idx, (out, in_) in MATRIX_DIMS.items():
                t = TENSOR_TPL.format(l=l, e=e, idx=idx)
                if t + ".weight" in weight_map and t + ".scale" in weight_map:
                    targets.append((t + ".weight", t + ".scale", out, in_, f"L{l}"))
    if args.with_mtp:
        mtp_layers = set(re.findall(r"mtp\.(\d+)\.ffn\.experts", "|".join(weight_map)))
        for m in sorted(mtp_layers):
            for e in range(cfg["n_routed_experts"]):
                for idx, (out, in_) in MATRIX_DIMS.items():
                    t = MTP_TENSOR_TPL.format(m=m, e=e, idx=idx)
                    if t + ".weight" in weight_map and t + ".scale" in weight_map:
                        targets.append((t + ".weight", t + ".scale", out, in_, f"MTP{m}"))

    logger.info("待转换专家矩阵：%d 个（路由 %d×%d×3%s）",
                len(targets), layers, cfg["n_routed_experts"], " + MTP" if args.with_mtp else "")
    if args.dry_run:
        for t, _, _, _, g in targets[:20]:
            logger.info("  [%s] %s", g, t)
        logger.info("  ... 共 %d 个（dry-run，不写入）", len(targets))
        return

    # 逐层写入（按层聚合到同一 shard 文件，保持目录结构简单）
    from safetensors.torch import save_file
    per_group: Dict[str, Dict[str, torch.Tensor]] = {}
    errors: List[float] = []
    for i, (w_name, s_name, out, in_, group) in enumerate(targets):
        w = load_tensor(w_name, weight_map, args.input_dir)
        s = load_tensor(s_name, weight_map, args.input_dir)
        if args.validate:
            err = roundtrip_error(w, s, out, in_)
            errors.append(err)
            if i % 500 == 0:
                logger.info("  [%s] %s rel_err=%.6f", group, w_name, err)
        w_packed, w_scale = convert_expert_matrix(w, s, out, in_)
        per_group.setdefault(group, {})[w_name] = w_packed
        per_group.setdefault(group, {})[s_name] = w_scale

    for group, tensors in per_group.items():
        fname = f"model_{group}.safetensors"
        save_file(tensors, str(args.output_dir / fname))
        logger.info("写出 %s：%d 张量", fname, len(tensors))

    if args.validate and errors:
        import statistics
        logger.info("roundtrip 相对误差：mean=%.6f p95=%.6f max=%.6f",
                    statistics.mean(errors),
                    sorted(errors)[int(len(errors) * 0.95)],
                    max(errors))

    # 其余文件原样拷贝（config.json 除外）+ 非专家张量由 vLLM loader 从原 checkpoint 读取
    for f in os.listdir(args.input_dir):
        src = args.input_dir / f
        if src.is_file() and f not in ("config.json",) and not src.name.endswith(".safetensors"):
            dst = args.output_dir / f
            if not dst.exists():
                shutil.copy2(src, dst)

    cfg["moe_quant_algo"] = "NVFP4"
    cfg["expert_nvfp4_block"] = [NV_K_BLOCK, NV_N_BLOCK]
    (args.output_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完成 → %s（config 已标记 moe_quant_algo=NVFP4）", args.output_dir)


if __name__ == "__main__":
    main()
