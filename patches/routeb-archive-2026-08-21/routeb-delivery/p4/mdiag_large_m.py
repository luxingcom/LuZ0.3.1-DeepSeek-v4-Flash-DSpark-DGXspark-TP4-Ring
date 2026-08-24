#!/usr/bin/env python3
"""M=16384 崩塌诊断：编译复用 vs 全新编译（正确性 + 性能）。"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import routeb_pipe as rp

dev = "cuda"
torch.manual_seed(0)
N, K = 2048, 4096
tile, epi = (128, 128, 128), (128, 128)

Wt = (torch.randn(N, K, device=dev) * 0.3)
wq, ws = rp.torch_w_quant(Wt)

for M in [8192, 16384]:
    A16 = (torch.randn(M, K, device=dev) * 0.5).half()

    # --- 路径 1：复用 M=256 编译（全量矩阵的做法）---
    g0 = rp.RouteBGemm(256, N, K, tile, epi)
    g0.set_W_prepacked(wq, ws)
    g = rp.RouteBGemm(M, N, K, tile, epi, _compile=False)
    g.compiled = g0.compiled
    g.set_W_prepacked(wq, ws)
    aq, asf = g.set_A(A16)
    g.run(); torch.cuda.synchronize()
    ref = rp.dequant_ref(aq, asf, wq, ws)
    rel1 = ((g.out().float() - ref).abs().max() / ref.abs().max()).item()
    t1 = rp.time_ms(g.run, warmup=10, iters=30, rounds=3)
    del g, g0
    torch.cuda.empty_cache()

    # --- 路径 2：在 M=16384 现场全新编译 ---
    g2 = rp.RouteBGemm(M, N, K, tile, epi)
    g2.set_W_prepacked(wq, ws)
    aq2, asf2 = g2.set_A(A16)
    g2.run(); torch.cuda.synchronize()
    ref2 = rp.dequant_ref(aq2, asf2, wq, ws)
    rel2 = ((g2.out().float() - ref2).abs().max() / ref2.abs().max()).item()
    t2 = rp.time_ms(g2.run, warmup=10, iters=30, rounds=3)
    del g2
    torch.cuda.empty_cache()

    print(f"M={M:6d}  复用编译: rel={rel1:.5f} {t1*1e3:8.1f}us"
          f" {rp.tflops(M, N, K, t1):6.1f} TF   |   全新编译: rel={rel2:.5f}"
          f" {t2*1e3:8.1f}us {rp.tflops(M, N, K, t2):6.1f} TF", flush=True)

print("MDIAG_DONE")
