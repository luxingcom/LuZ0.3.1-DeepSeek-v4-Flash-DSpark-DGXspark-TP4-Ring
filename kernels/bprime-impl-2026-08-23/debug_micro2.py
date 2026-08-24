#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根因假设验证: micro direct + e8m0 是否只在 w13(旋转 grid)下坏, w31(未旋转)正常?
   并补采 M=4 micro-off 差异的分位数口径。"""
import torch

from b12x.moe.fused.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_e8m0_native_weights,
    prepare_w4a16_fp4_e8m0_k32_weights,
)
from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

E, H, I, TOPK = 64, 2048, 512, 6


def build(seed=123):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8,
                        generator=g, device="cuda")
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8,
                       generator=g, device="cuda")
    s13 = torch.randint(118, 127, (E, 2 * I, H // 32), dtype=torch.uint8,
                        generator=g, device="cuda")
    s2 = torch.randint(118, 127, (E, H, I // 32), dtype=torch.uint8,
                       generator=g, device="cuda")
    unit = torch.ones(E, dtype=torch.float32, device="cuda")
    packed = prepare_w4a16_fp4_e8m0_k32_weights(
        w13.clone(), s13.clone(), unit, w2.clone(), s2.clone(), unit,
        activation="silu", params_dtype=torch.bfloat16,
        w13_layout="w31", reuse_input_storage=True)
    native31 = prepare_w4a16_e8m0_native_weights(
        w13.clone(), s13.clone(), unit, w2.clone(), s2.clone(), unit,
        activation="silu", params_dtype=torch.bfloat16, w13_layout="w31")
    return packed, native31


def run_one(prepared, M, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(M, H, generator=g, device="cuda",
                     dtype=torch.float32) * 0.5).to(torch.bfloat16)
    ids = torch.stack([
        torch.randperm(E, generator=g, device="cuda")[:TOPK]
        for _ in range(M)]).to(torch.int32)
    wts = torch.rand(M, TOPK, generator=g, device="cuda")
    wts = (wts / wts.sum(-1, keepdim=True)).float().contiguous()
    bufs = make_w4a16_packed_buffers(
        prepared, m=M, topk=TOPK, dtype=torch.bfloat16,
        device=torch.device("cuda"))
    run_w4a16_moe(a, prepared, wts, ids, activation="silu",
                  intermediate_cache13=bufs.intermediate_cache13,
                  intermediate_cache2=bufs.intermediate_cache2,
                  output=bufs.output)
    torch.cuda.synchronize()
    out = bufs.output.clone().float()
    del bufs
    return out


def cmp(tag, o_ref, o):
    d = (o_ref - o).abs()
    rel = (d / o_ref.abs().clamp_min(1e-3))
    print(f"{tag}: nan={int(torch.isnan(o).sum())} max|d|={d.max().item():.3e} "
          f"rel p50={rel.median().item():.2e} p99={rel.quantile(0.99).item():.2e} "
          f"max={rel.max().item():.2e}")


if __name__ == "__main__":
    packed, native31 = build()
    for M in (8, 4):
        o_p = run_one(packed, M)
        o_31 = run_one(native31, M)   # micro direct + w31 (未旋转 grid)
        cmp(f"M={M} packed(mainGEMM) vs native_w31(micro)", o_p, o_31)

    # M=4 micro-off 差异复核（分位数口径, 多种子）
    import os
    os.environ["B12X_W4A16_SMALL_M_DIRECT"] = "0"
    for seed in (7, 8, 9):
        o_p = run_one(packed, 4, seed=seed)
        o_31 = run_one(native31, 4, seed=seed)  # 主 GEMM both
        cmp(f"M=4 seed={seed} packed vs native31 (both 主GEMM, micro off)",
            o_p, o_31)
    os.environ.pop("B12X_W4A16_SMALL_M_DIRECT")
