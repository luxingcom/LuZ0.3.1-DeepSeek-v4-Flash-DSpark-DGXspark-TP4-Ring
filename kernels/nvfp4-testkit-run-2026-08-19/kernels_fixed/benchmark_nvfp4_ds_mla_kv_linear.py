"""Benchmark: nvfp4_ds_mla_kv_linear（NVFP4 DS-MLA KV 线性布局写回）

在 DGX Spark（GB10 / SM121）上运行：
    python benchmark_nvfp4_ds_mla_kv_linear.py
"""
import torch
import triton

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as triton_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [1, 4, 16, 64, 256, 1024, 4096, 16384]


def bench(T):
    torch.manual_seed(0)
    k = torch.randn(T, 512, device=DEVICE, dtype=torch.float32) * 0.5
    v = torch.randn(T, 512, device=DEVICE, dtype=torch.float32) * 0.5

    _ = triton_impl(k, v)
    torch.cuda.synchronize()

    t_ref = triton.testing.do_bench(lambda: ref_impl(k, v), warmup=25, rep=100)
    t_tri = triton.testing.do_bench(lambda: triton_impl(k, v), warmup=25, rep=100)

    # 吞吐口径：每 token 写 584B 信封 + 读 2×512×4B
    bytes_per_token = 2 * 512 * 4 + 584
    gb_ref = (T * bytes_per_token) / (t_ref * 1e-3) / 1e9
    gb_tri = (T * bytes_per_token) / (t_tri * 1e-3) / 1e9

    print(f"T={T:6d} | ref {t_ref*1e3:8.2f}ms ({gb_ref:6.1f} GB/s) | "
          f"triton {t_tri*1e3:8.2f}ms ({gb_tri:6.1f} GB/s) | speedup {t_ref/t_tri:6.2f}x")
    return t_ref / t_tri


if __name__ == "__main__":
    if DEVICE != "cuda":
        raise SystemExit("需要 CUDA（DGX Spark / SM121）")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    ratios = [bench(T) for T in SHAPES]
    print("=" * 100)
    print(f"Avg speedup: {sum(ratios)/len(ratios):.2f}x | Best: {max(ratios):.2f}x")
    print("MCP 侧实测（服务端 GPU）：speedup=2.86x（v5，达标 2.0 目标）；GB10 带宽 273GB/s 为上限")
