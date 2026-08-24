#!/usr/bin/env python3
"""p1_overhead_bench.py — merged-GEMM host 侧开销量化（Task P1-1 ①②）。

① 权重 N-concat 组装代价（生产 TP4 每 rank 口径）:
   - 每 expert 权重: w13 [1024, 2048]u8 (2MB) + w2 [4096, 256]u8 (1MB)
     + scale E4M3 k16 (0.39MB) ≈ 3.4MB/rank... 任务口径 48×12MB=576MB/层
     （按全模型 per-expert ~12MB 或任务给定口径直接实测两种）
   - 实测: torch.cat 48×12MB / index_select 从 stacked [256, 12MB] / D2D 带宽基线
② token gather/scatter（M=4096 chunk 级）:
   - 重复行 gather: index_select [4096,4096]bf16, idx 24576（含重复）
   - scatter-add: index_add_ [24576,4096]bf16 -> [4096,4096]
   - 朴素 permute: 4096 行 gather+scatter
   - 输出投影: 每 layer 与 43 层/step
"""
import json
import time

import torch

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    out = {}
    GB = 1024 ** 3

    # ---------- ① concat 组装 ----------
    print("=== ① 权重 N-concat 组装代价 ===")
    # D2D 带宽基线: 576MB copy
    src = torch.empty(576 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src))
    bw = 2 * src.numel() / (ms / 1e3) / GB
    print(f"  D2D copy 576MB: {ms:.3f} ms ({bw:.0f} GB/s 双向带宽)")
    out["d2d_copy_576MB_ms"] = round(ms, 3)
    out["d2d_bw_GBs"] = round(bw, 1)

    # 任务口径: 48 expert × 12MB torch.cat
    chunks = [torch.randint(0, 256, (12 * 1024 * 1024,), dtype=torch.uint8, device=DEV)
              for _ in range(48)]
    ms_cat = bench(lambda: torch.cat(chunks, dim=0))
    print(f"  torch.cat 48×12MB: {ms_cat:.3f} ms/层 -> ×43 层 = {ms_cat*43:.1f} ms/step")
    out["cat_48x12MB_ms"] = round(ms_cat, 3)
    out["cat_48x12MB_step_ms"] = round(ms_cat * 43, 1)

    # index_select 从 stacked [256, 12MB]（免 cat 的单 kernel gather）
    stacked = torch.randint(0, 256, (256, 12 * 1024 * 1024), dtype=torch.uint8, device=DEV)
    idx = torch.randperm(256, device=DEV)[:48]
    ms_is = bench(lambda: torch.index_select(stacked, 0, idx))
    print(f"  index_select stacked[256,12MB]×48: {ms_is:.3f} ms/层 -> ×43 = {ms_is*43:.1f} ms/step")
    out["idxsel_48x12MB_ms"] = round(ms_is, 3)
    out["idxsel_48x12MB_step_ms"] = round(ms_is * 43, 1)

    # 生产真实口径: 每 rank 每 expert payload 3MB (w13 2MB + w2 1MB) + scale 0.39MB
    # 40-80 命中专家 -> 60 专家中位: 60×3.4MB = 204MB/层
    for n_exp in (40, 60, 80):
        per = 3 * 1024 * 1024 + 402 * 1024   # payload + scale 近似
        st = torch.randint(0, 256, (256, per), dtype=torch.uint8, device=DEV)
        ix = torch.randperm(256, device=DEV)[:n_exp]
        m = bench(lambda: torch.index_select(st, 0, ix))
        print(f"  index_select {n_exp} expert×3.4MB (rank口径): {m:.3f} ms/层 "
              f"-> ×43 = {m*43:.1f} ms/step")
        out[f"idxsel_{n_exp}exp_rank_ms"] = round(m, 3)
        out[f"idxsel_{n_exp}exp_rank_step_ms"] = round(m * 43, 1)
        del st

    # ---------- ② token gather/scatter ----------
    print("\n=== ② token gather/scatter 开销（M=4096 chunk） ===")
    M, K = 4096, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    # 重复行 gather: 24576 (token,expert) 对 -> 行索引（每 token 6 次，可重复）
    idx_dup = torch.randint(0, M, (24576,), dtype=torch.long, device=DEV)
    ms_g = bench(lambda: torch.index_select(A, 0, idx_dup))
    gb = 24576 * K * 2 * 2 / GB
    print(f"  dup gather index_select [4096,4096]bf16 -> [24576,4096]: {ms_g:.3f} ms "
          f"({gb/(ms_g/1e3):.0f} GB/s) -> ×43 = {ms_g*43:.1f} ms/step")
    out["gather_dup_ms"] = round(ms_g, 3)
    out["gather_dup_step_ms"] = round(ms_g * 43, 1)

    # scatter-add
    B = torch.randn(24576, K, dtype=torch.bfloat16, device=DEV)
    def scat():
        out_ = torch.zeros(M, K, dtype=torch.bfloat16, device=DEV)
        out_.index_add_(0, idx_dup, B)
        return out_
    ms_s = bench(scat)
    print(f"  scatter-add index_add_ [24576,4096]->[4096,4096] (含零初始化): {ms_s:.3f} ms "
          f"-> ×43 = {ms_s*43:.1f} ms/step")
    out["scatter_add_ms"] = round(ms_s, 3)
    out["scatter_add_step_ms"] = round(ms_s * 43, 1)

    # 朴素 permute (无重复): 4096 行
    perm = torch.randperm(M, device=DEV)
    ms_p = bench(lambda: A[perm].contiguous())
    print(f"  permute gather [4096,4096]: {ms_p:.3f} ms -> ×43 = {ms_p*43:.1f} ms/step")
    out["permute_ms"] = round(ms_p, 3)
    out["permute_step_ms"] = round(ms_p * 43, 1)

    # 汇总: merged 路径 host 侧总预算（朴素 concat 方案, rank 口径 60 专家）
    total = out["idxsel_60exp_rank_step_ms"] + out["gather_dup_step_ms"] + out["scatter_add_step_ms"]
    print(f"\n=== 汇总（朴素方案, rank 口径 60 命中专家/层）===")
    print(f"  权重组装 {out['idxsel_60exp_rank_step_ms']} + gather {out['gather_dup_step_ms']}"
          f" + scatter {out['scatter_add_step_ms']} = {total:.1f} ms/step")
    out["naive_total_step_ms"] = round(total, 1)

    with open("/work/p1_overhead.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n[done] wrote /work/p1_overhead.json")


if __name__ == "__main__":
    main()
