"""kernel① v13 复测：正确性（vs torch ref）+ 真实性能（W 缓存 → 纯 GEMM）

环境：DGX Spark GB10 / sm_121a / torch 2.11 / triton 3.6
对比基线：round6/v121 真机终测 v12 GEMM-only = 20.5~45.8 TFLOPS
"""
import sys
import torch
import triton

sys.path.insert(0, ".")

from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
import nvfp4_4w4a_prefill_gemm_v13_triton as M

triton_impl = M.nvfp4_4w4a_prefill_gemm
from test_nvfp4_4w4a_prefill_gemm import make_weights

DEVICE = "cuda"


def main():
    print("=" * 90)
    print("KERNEL1 v13 CORRECTNESS  (vs torch ref, rtol/atol = 5e-2)")
    print("=" * 90)
    SHAPES = [
        (256, 4096, 4096),
        (512, 2048, 4096),
        (1024, 4096, 2048),
        (128, 4096, 4096),
    ]
    pass_n = 0
    total = 0
    for (M_, K, N) in SHAPES:
        for use_bias in (False, True):
            total += 1
            torch.manual_seed(0)
            A = torch.randn(M_, K, device=DEVICE, dtype=torch.float32)
            Wq, Ws = make_weights(K, N, scale=0.5, device=DEVICE)
            bias = torch.randn(N, device=DEVICE, dtype=torch.float32) if use_bias else None
            ref = ref_impl(A, Wq, Ws, bias)
            out = triton_impl(A, Wq, Ws, bias)
            ok = torch.allclose(out, ref, rtol=5e-2, atol=5e-2)
            pass_n += ok
            max_err = (out - ref).abs().max().item()
            print(f"[{'PASS' if ok else 'FAIL'}] M={M_:5d} K={K:5d} N={N:5d} bias={str(use_bias):5s} "
                  f"max_abs_err={max_err:.4f}")
    print(f"CORRECTNESS: {pass_n}/{total} passed\n")

    print("=" * 90)
    print("KERNEL1 v13 PERF  (W 缓存 → 纯 GEMM，含 A 量化；对比基线 20~46 TFLOPS)")
    print("=" * 90)
    BENCH = [
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (256, 8192, 8192),
        (512, 8192, 8192),
        (1024, 8192, 4096),
        (256, 4096, 16384),
    ]
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'time_ms':>9} {'TFLOPS':>8}")
    for (M_, K, N) in BENCH:
        torch.manual_seed(0)
        A = torch.randn(M_, K, device=DEVICE, dtype=torch.float32)
        Wq, Ws = make_weights(K, N, scale=0.5, device=DEVICE)
        bias = torch.randn(N, device=DEVICE, dtype=torch.float32)
        _ = triton_impl(A, Wq, Ws, bias)
        torch.cuda.synchronize()
        t = triton.testing.do_bench(lambda: triton_impl(A, Wq, Ws, bias), warmup=25, rep=100)
        tf = 2.0 * M_ * N * K / (t * 1e-3) / 1e12
        print(f"{M_:>6} {K:>6} {N:>6} | {t * 1e3:9.2f} {tf:8.1f}")


if __name__ == "__main__":
    main()
