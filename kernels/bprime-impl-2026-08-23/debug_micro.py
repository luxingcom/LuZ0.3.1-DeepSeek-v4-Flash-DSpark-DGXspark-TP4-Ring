#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G3/M=8 NaN 定位: micro direct vs 主 GEMM vs packed 三方对照 + 几何扫描"""
import os
import sys

import torch

from b12x.moe.fused.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_e8m0_native_weights,
    prepare_w4a16_fp4_e8m0_k32_weights,
)
from b12x.moe.fused.w4a16.kernel import (
    _small_m_direct_supported,
    run_w4a16_moe,
)


def build(E, H, I, seed=123):
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

    w13n, w2n, s13n, s2n = w13.clone(), w2.clone(), s13.clone(), s2.clone()
    tmp = w13n[:, :I].clone()
    w13n[:, :I].copy_(w13n[:, I:]); w13n[:, I:].copy_(tmp)
    tmp = s13n[:, :I].clone()
    s13n[:, :I].copy_(s13n[:, I:]); s13n[:, I:].copy_(tmp)
    native = prepare_w4a16_e8m0_native_weights(
        w13n, s13n, unit, w2n, s2n, unit,
        activation="silu", params_dtype=torch.bfloat16, w13_layout="w13")
    return packed, native, (w13, w2, s13, s2)


def run_one(prepared, M, E, H, I, topk, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(M, H, generator=g, device="cuda",
                     dtype=torch.float32) * 0.5).to(torch.bfloat16)
    ids = torch.stack([
        torch.randperm(E, generator=g, device="cuda")[:topk]
        for _ in range(M)]).to(torch.int32)
    wts = torch.rand(M, topk, generator=g, device="cuda")
    wts = (wts / wts.sum(-1, keepdim=True)).float().contiguous()
    bufs = make_w4a16_packed_buffers(
        prepared, m=M, topk=topk, dtype=torch.bfloat16,
        device=torch.device("cuda"))
    run_w4a16_moe(a, prepared, wts, ids, activation="silu",
                  intermediate_cache13=bufs.intermediate_cache13,
                  intermediate_cache2=bufs.intermediate_cache2,
                  output=bufs.output)
    torch.cuda.synchronize()
    out = bufs.output.clone().float()
    del bufs
    return out


def scenario(E, H, I, topk, Ms=(8, 4, 64)):
    print(f"\n--- geometry E={E} H={H} I={I} topk={topk} ---")
    packed, native, tensors = build(E, H, I)
    try:
        for M in Ms:
            sup = _small_m_direct_supported(
                m=M, hidden_size=H, intermediate_size=I, num_experts=E,
                topk=topk, activation="silu",
                apply_router_weight_on_input=False, swiglu_limit=None,
                element_dtype="bf16", weight_layout="modelopt",
                w13_layout="w13", scale_format="e8m0_k32")
            o_p = run_one(packed, M, E, H, I, topk)
            o_n = run_one(native, M, E, H, I, topk)
            nan_p = int(torch.isnan(o_p).sum())
            nan_n = int(torch.isnan(o_n).sum())
            if nan_p == 0 and nan_n == 0:
                d = (o_p - o_n).abs()
                rel = (d / o_p.abs().clamp_min(1e-3)).max().item()
                print(f"M={M} micro_route={sup}: packed nan={nan_p} native "
                      f"nan={nan_n} max_rel={rel:.2e} max|d|={d.max().item():.2e}")
            else:
                print(f"M={M} micro_route={sup}: packed nan={nan_p}/{o_p.numel()} "
                      f"native nan={nan_n}/{o_n.numel()} "
                      f"(native finite mean={o_n[~torch.isnan(o_n)].abs().mean().item():.3e})"
                      )
                # native micro 关掉再对照（强制主 GEMM）
                os.environ["B12X_W4A16_SMALL_M_DIRECT"] = "0"
                o_n2 = run_one(native, M, E, H, I, topk)
                os.environ.pop("B12X_W4A16_SMALL_M_DIRECT")
                nan_n2 = int(torch.isnan(o_n2).sum())
                if nan_p == 0 and nan_n2 == 0:
                    d = (o_p - o_n2).abs()
                    rel = (d / o_p.abs().clamp_min(1e-3)).max().item()
                    print(f"        native 主GEMM(micro off): nan={nan_n2} "
                          f"max_rel vs packed={rel:.2e} —— micro 臂独立问题")
                else:
                    print(f"        native 主GEMM(micro off): nan={nan_n2} —— "
                          f"非 micro 独有")
    finally:
        del packed, native, tensors
        torch.cuda.empty_cache()


if __name__ == "__main__":
    free_b, _ = torch.cuda.mem_get_info()
    print(f"GPU free: {free_b/2**30:.1f} GiB")
    scenario(32, 512, 256, 4)
    scenario(32, 1024, 512, 4)
    scenario(64, 2048, 512, 6)
    scenario(128, 2048, 512, 6)
