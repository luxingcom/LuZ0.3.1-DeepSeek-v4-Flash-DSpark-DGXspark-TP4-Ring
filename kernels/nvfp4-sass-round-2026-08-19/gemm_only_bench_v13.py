"""kernel① v13 GEMM-ONLY 性能（对齐 round6 口径：A 预量化、W 已缓存，仅测 _nvfp4_gemm_kernel 的 dot_scaled MMA 吞吐）

目的：与 round6 真机终测的 20.5~45.8 TFLOPS 基线做苹果对苹果对比。
"""
import sys
import torch
import triton

sys.path.insert(0, ".")

import nvfp4_4w4a_prefill_gemm_v13_triton as M
from test_nvfp4_4w4a_prefill_gemm import make_weights

DEVICE = "cuda"
QUANT_BLOCK_M = 32
QUANT_BLOCK_K = 64


def bench_gemm_only(M_, K, N):
    torch.manual_seed(0)
    A = torch.randn(M_, K, device=DEVICE, dtype=torch.float32)
    Wq, Ws = make_weights(K, N, scale=0.5, device=DEVICE)

    # ① A 预量化（一次，不计入 GEMM 计时）
    A_packed = torch.empty((M_, K // 2), dtype=torch.uint8, device=DEVICE)
    A_scale = torch.empty((M_, K // 32), dtype=torch.uint8, device=DEVICE)
    M._quantize_fp32_to_nvfp4_packed[(triton.cdiv(M_, QUANT_BLOCK_M), triton.cdiv(K, QUANT_BLOCK_K))](
        A, A_packed, A_scale, M_, K,
        A.stride(0), A.stride(1),
        A_packed.stride(0), A_packed.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        BLOCK_M=QUANT_BLOCK_M, BLOCK_K=QUANT_BLOCK_K,
    )
    # ② ③ W 重打包/scale 展开（一次，缓存）
    W_packed_rhs, W_scale_rhs = M.preprocess_weights(Wq, Ws)

    C = torch.empty((M_, N), dtype=torch.float32, device=DEVICE)

    def grid(meta):
        return (triton.cdiv(M_, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    def kernel():
        M._nvfp4_gemm_kernel[grid](
            A_packed, A_scale, W_packed_rhs, W_scale_rhs, C,
            A,  # bias_ptr 占位（HAS_BIAS=False 不读取）
            M_, N, K,
            A_packed.stride(0), A_packed.stride(1),
            W_packed_rhs.stride(0), W_packed_rhs.stride(1),
            W_scale_rhs.stride(0), W_scale_rhs.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=False,
        )

    _ = kernel()
    torch.cuda.synchronize()
    t = triton.testing.do_bench(kernel, warmup=25, rep=100)
    tf = 2.0 * M_ * N * K / (t * 1e-3) / 1e12
    print(f"{M_:>6} {K:>6} {N:>6} | {t * 1e3:9.2f}ms {tf:8.1f} TFLOPS")


if __name__ == "__main__":
    print("KERNEL1 v13 GEMM-ONLY PERF  (A pre-quant, W cached; 对比 round6 基线 20.5~45.8)")
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'time':>12} {'TFLOPS':>8}")
    for (M_, K, N) in [
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (256, 8192, 8192),
        (512, 8192, 8192),
        (1024, 8192, 4096),
        (256, 4096, 16384),
    ]:
        bench_gemm_only(M_, K, N)
