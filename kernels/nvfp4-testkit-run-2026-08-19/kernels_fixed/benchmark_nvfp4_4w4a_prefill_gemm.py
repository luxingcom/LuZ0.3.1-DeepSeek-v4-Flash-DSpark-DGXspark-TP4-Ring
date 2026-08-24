"""Benchmark: nvfp4_4w4a_prefill_gemm（Triton vs PyTorch 参考实现）

在 DGX Spark（GB10 / SM121）上运行：
    python benchmark_nvfp4_4w4a_prefill_gemm.py
"""
import torch
import triton

from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_triton import nvfp4_4w4a_prefill_gemm as triton_impl
from test_nvfp4_4w4a_prefill_gemm import make_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHAPES = [
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 4096),
    (256, 4096, 16384),
]


def bench(M, K, N, use_bias=True):
    torch.manual_seed(0)
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W_quant, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float32) if use_bias else None

    # warmup
    _ = triton_impl(A, W_quant, W_scale, bias)
    torch.cuda.synchronize()

    t_ref = triton.testing.do_bench(lambda: ref_impl(A, W_quant, W_scale, bias), warmup=25, rep=100)
    t_tri = triton.testing.do_bench(lambda: triton_impl(A, W_quant, W_scale, bias), warmup=25, rep=100)

    flops = 2.0 * M * N * K
    tf_ref = flops / (t_ref * 1e-3) / 1e12
    tf_tri = flops / (t_tri * 1e-3) / 1e12

    print(f"M={M:5d} K={K:5d} N={N:5d} | ref {t_ref*1e3:8.2f}ms ({tf_ref:6.1f} TFLOPS) | "
          f"triton {t_tri*1e3:8.2f}ms ({tf_tri:6.1f} TFLOPS) | speedup {t_ref/t_tri:6.2f}x")
    return t_ref / t_tri


if __name__ == "__main__":
    if DEVICE != "cuda":
        raise SystemExit("需要 CUDA（DGX Spark / SM121）")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    ratios = [bench(*s) for s in SHAPES]
    print("=" * 100)
    print(f"Avg speedup: {sum(ratios)/len(ratios):.2f}x | Best: {max(ratios):.2f}x | Worst: {min(ratios):.2f}x")
    print("注：目标为 GB10 FP4 dense 500 TFLOPS 的 80%（~400 TFLOPS），当前为 fp32-dot 4W4A 语义内核，"
          "如需逼近该目标需切换 tl.dot_scaled(e2m1) 原生 FP4 MMA 路径。")
