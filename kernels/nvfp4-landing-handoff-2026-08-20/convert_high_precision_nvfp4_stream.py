#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_high_precision_nvfp4_stream.py — MXFP4 → NVFP4 高精度转换器（流式逐层版 v3）
====================================================================================
v3 改进（engineering-assurance 落地，2026-08-20）：
  1) MATRIX_DIMS w2/w3 的 (out,in) 写反 → 改为动态推断，不依赖硬编码
  2) 全量加载 48 shard(≈176GB) → **流式逐层**：每层恰 1 shard（layer l → shard 0002+l），
     读单 shard → 转换全部 experts → 写回即释放内存，峰值 = 单层（≈几 GB）
  3) 非专家 shard（embed/head/norm/hc_/mtp）原样拷贝，零转换
  4) dequant 后 shape 断言（out/in 可被 32×128 整除，scale 维度一致）

布局（生产 safetensors 实测）：
  w1.weight [2048,2048] int8 / scale [2048,128] E8M0   → out=2048, in=4096
  w2.weight [4096,1024] int8 / scale [4096,64]  E8M0   → out=4096, in=2048
  w3.weight [2048,2048] int8 / scale [2048,128] E8M0   → out=2048, in=4096
  （safetensors 存 whole-expert 为 [out, in//2]，MX 沿 in 打包，低半字节=偶第1元素）

输出（对齐 routeA 内核 nvfp4_4w4a_mmaf.py）：
  W_packed: uint8 [K=in, N=out//2]（N 打包，低半字节=偶 N 列）
  W_scale : uint8 [K//32, N//128]（E8M0）

用法（容器内, --max-layers N 用于小层试跑）：
python convert_high_precision_nvfp4_stream.py --input-dir /models \
    --output-dir <INSTALL_DIR>/nvfp4/models/dsv4f-0731-nvfp4-hp \
    [--mode high|fast] [--max-layers N] [--validate] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("convert_nvfp4_stream")

try:
    import torch
except ImportError:
    sys.stderr.write("ERROR: 需要 torch（请在 vllm 容器内运行）\n")
    sys.exit(2)

# ---- 常量 ----
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)
E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MAX = 6.0
E8M0_BIAS = 127.0
MX_GROUP = 32
NV_K_BLOCK = 32
NV_N_BLOCK = 128
TENSOR_TPL = "layers.{l}.ffn.experts.{e}.{idx}"
N_ROUTED_EXPERTS = 256
N_LAYERS = 43
MX_WEIGHTS = ("w1", "w2", "w3")


def as_uint8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def dequant_mxfp4(w_packed, w_scale):
    """MXFP4 → fp32 [out, in]。w_packed [out,in//2] 低半字节=偶；scale [out,in//32]。"""
    wp = as_uint8(w_packed).to(torch.int32)
    ws = w_scale.float()
    out, in_half = wp.shape
    i = in_half * 2
    lo = wp & 0x0F
    hi = (wp >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(out, i)
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(out, i)
    scale = torch.pow(2.0, ws - E8M0_BIAS)
    scale_expanded = scale.repeat_interleave(MX_GROUP, dim=1).reshape(out, i)
    return (mag * scale_expanded).to(torch.float32)


def _e2m1_quantize(w_norm):
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
    max_abs = w.abs().amax(dim=(1, 3))
    e0 = torch.floor(torch.log2(max_abs.clamp(min=1e-38) / E2M1_MAX))
    if mode == "fast":
        e = e0
    else:
        cand = torch.stack([e0 - 1, e0, e0 + 1], dim=0).clamp(-126, 127)
        scales = torch.pow(2.0, cand)
        w_exp = w.unsqueeze(0)
        # scales: [3, K//32, N//128] -> 广播 [3, K//32, 1, N//128, 1] 对齐 w_exp [1,K/32,32,N/128,128]
        w_norm = w_exp / scales[:, :, None, :, None]
        w_c = torch.clamp(w_norm, -E2M1_MAX, E2M1_MAX)
        abs_n = w_c.abs()
        idx = torch.zeros_like(abs_n, dtype=torch.long)
        for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            idx = idx + (abs_n > t).long()
        w_q = E2M1_POS[idx.to(w_c.device)]
        w_q = torch.where(w_c < 0, -w_q, w_q)
        w_deq = w_q * scales[:, :, None, :, None]
        mse = ((w_deq - w_exp) ** 2).mean(dim=(2, 4))
        best = mse.argmin(dim=0)
        e = cand.gather(0, best.unsqueeze(0)).squeeze(0)
    scale = torch.pow(2.0, e.clamp(-126, 127))
    w_norm = w / scale[:, None, :, None]
    nibble = _e2m1_quantize(w_norm)
    nib_flat = nibble.reshape(K, N)
    lo = nib_flat[:, 0::2]
    hi = nib_flat[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()
    scale_u8 = (e + E8M0_BIAS).clamp(0, 255).to(torch.uint8).contiguous()
    return packed, scale_u8


def quant_nvfp4_matrix(w_fp32, out, i, mode="high"):
    return quant_nvfp4_block(w_fp32.t().contiguous(), i, out, mode)


def dequant_nvfp4(w_packed, w_scale):
    wp = as_uint8(w_packed).to(torch.int32)
    K, n_half = wp.shape
    N = n_half * 2
    lo = wp & 0x0F
    hi = (wp >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(K, N)
    mag = E2M1_TABLE[nib.reshape(-1)].reshape(K, N)
    scale = torch.pow(2.0, w_scale.float() - E8M0_BIAS)
    scale_exp = scale.repeat_interleave(NV_K_BLOCK, dim=0).repeat_interleave(NV_N_BLOCK, dim=1)
    return (mag * scale_exp).to(torch.float32)


def rel_err_report(a, b, tag):
    err = (a - b).abs() / (b.abs() + 1e-6)
    out = {"tag": tag, "mean": err.mean().item(),
           "p95": torch.quantile(err.flatten(), 0.95).item(), "max": err.max().item()}
    return out


def convert_single_shard(sd, l, mode, validate, reports):
    """转换 shard-0002+l 内该层的所有 expert 权重（in-place 于 sd）。返回改动的张量名集合。"""
    changed = set()
    n_mat = 0
    for e in range(N_ROUTED_EXPERTS):
        for idx in MX_WEIGHTS:
            wname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".weight"
            sname = TENSOR_TPL.format(l=l, e=e, idx=idx) + ".scale"
            if wname not in sd or sname not in sd:
                continue
            w_packed = sd[wname]
            w_scale = sd[sname]
            out = w_packed.shape[0]
            i = w_packed.shape[1] * 2
            assert i % NV_K_BLOCK == 0, f"{wname} in={i}"
            assert out % NV_N_BLOCK == 0, f"{wname} out={out}"
            assert w_scale.shape == (out, i // 32), f"{sname} scale {(tuple(w_scale.shape))}"
            w_fp32 = dequant_mxfp4(w_packed, w_scale)
            packed, scale = quant_nvfp4_matrix(w_fp32, out, i, mode)
            del w_fp32
            sd[wname] = packed
            sd[sname] = scale
            changed.add(wname)
            changed.add(sname)
            n_mat += 1
            if validate:
                w_orig = dequant_mxfp4(w_packed, w_scale).t()
                w_re = dequant_nvfp4(packed, scale)
                reports.append(rel_err_report(w_re, w_orig, f"L{l}E{e}{idx}"))
                del w_orig, w_re
    return changed, n_mat


def main():
    ap = argparse.ArgumentParser(description="MXFP4 → NVFP4 流式转换器")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="high", choices=["high", "fast"])
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
    index = json.loads(idx_path.read_text())
    weight_map: Dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    logger.info("[probe] shards=%d total_tensors=%d", len(shard_files), len(weight_map))

    from safetensors import safe_open
    from safetensors.torch import save_file

    layers = range(min(args.max_layers, N_LAYERS) if args.max_layers else N_LAYERS)
    reports: List[Dict] = []

    # ---- 归纳每 shard 包含的层 ----
    shard_layers: Dict[str, List[int]] = {}
    for k, v in weight_map.items():
        if k.startswith("layers."):
            shard_layers.setdefault(v, []).append(int(k.split(".")[1]))
    shard_layers = {v: sorted(set(ls)) for v, ls in shard_layers.items()}

    for sf in sorted(shard_files):
        ls = shard_layers.get(sf, [])
        target = [l for l in ls if l in layers]
        src_path = src / sf
        dst_path = dst / sf

        if not target:
            # 非本次范围 → 原样拷贝
            if args.dry_run:
                logger.info("[skip-copy] %s（不含本次层）", sf)
            else:
                shutil.copy2(src_path, dst_path)
                logger.info("[copy] %s（原样）", sf)
            continue

        # 读取该 shard
        sd = {}
        with safe_open(str(src_path), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        for l in target:
            changed, n_mat = convert_single_shard(sd, l, args.mode, args.validate, reports)
            logger.info("[layer %d] (%s) 转换 %d 专家矩阵, 改动 %d 张量",
                        l, sf, n_mat, len(changed))
        if not args.dry_run:
            save_file(sd, str(dst_path))
            logger.info("[save] %s", sf)
        del sd

    # ---- 附属文件 ----
    if not args.dry_run:
        for extra in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "generation_config.json"]:
            p = src / extra
            if p.exists():
                shutil.copy2(p, dst / p.name)
        shutil.copy2(idx_path, dst / idx_path.name)
        logger.info("[index] 已拷贝 model.safetensors.index.json")

    if args.validate:
        import statistics
        means = [r["mean"] for r in reports]
        logger.info("=== roundtrip 汇总（%s, %d 矩阵）===", args.mode, len(reports))
        if means:
            logger.info("mean rel_err: avg=%.3e p95=%.3e max=%.3e",
                        statistics.mean(means),
                        sorted(means)[int(len(means) * 0.95)],
                        max(r["max"] for r in reports))

    logger.info("完成。输出目录: %s（%s，max_layers=%s, dry_run=%s）",
                dst, args.mode, args.max_layers, args.dry_run)


if __name__ == "__main__":
    main()