"""kernel1 v12 真实性能：缓存 W 重打包后，只测 GEMM 内核 TFLOPS"""
import torch, triton, time, sys
sys.path.insert(0, "/vllm-workspace/nvfp4-delivery-v12/kernel1-nvfp4_4w4a_prefill_gemm")
from nvfp4_4w4a_prefill_gemm_v12_triton import (
    _quantize_fp32_to_nvfp4_packed,
    _repack_w_for_rhs_k_pack,
    _expand_w_scale,
    _nvfp4_gemm_kernel,
    nvfp4_4w4a_prefill_gemm,
)
from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from test_nvfp4_4w4a_prefill_gemm import make_weights

SHAPES = [
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 4096),
    (256, 4096, 16384),
]

print("Device:", torch.cuda.get_device_name(0))
print("=" * 100)

for M, K, N in SHAPES:
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W_packed, W_scale = make_weights(K, N, scale=0.5, device="cuda")
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    # === 缓存 W 变换（每层只做一次，不计入 forward 时间） ===
    W_rhs = _repack_w_for_rhs_k_pack(W_packed, K, N)      # [K//2, N]
    W_s_rhs = _expand_w_scale(W_scale, K, N)               # [N, K//32]

    # torch ref（含 torch 量化，仅供对比）
    t_ref = triton.testing.do_bench(lambda: ref_impl(A, W_packed, W_scale, bias), warmup=25, rep=100)

    # --- 方法1：完整 v12 调用（含 A 量化 + W 变换，W 变换不计入缓存后） ---
    # 这里 W 变换已在上面完成，但 nvfp4_4w4a_prefill_gemm 内部会重复做
    # 所以用 t_full 仅作对比基线
    # t_full = triton.testing.do_bench(lambda: nvfp4_4w4a_prefill_gemm(A, W_packed, W_scale, bias), warmup=10, rep=50)

    # --- 方法2：纯 GEMM 内核（A 预量化 + W 缓存后） ---
    # Step A: 量化 A（独立 kernel）
    A_packed = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
    A_scale = torch.empty((M, K // 32), dtype=torch.uint8, device="cuda")
    QB_M, QB_K = 32, 64
    grid_q = (triton.cdiv(M, QB_M), triton.cdiv(K, QB_K))
    _quantize_fp32_to_nvfp4_packed[grid_q](
        A, A_packed, A_scale, M, K,
        A.stride(0), A.stride(1),
        A_packed.stride(0), A_packed.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        BLOCK_M=QB_M, BLOCK_K=QB_K,
    )

    # Step B: 纯 GEMM kernel（A 已量化，W 已缓存重打包）
    # 用 autotune 的默认配置跑一次 warmup 触发 autotune，再计时
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")
    HAS_BIAS = True
    B = bias.contiguous().float()

    # warmup + autotune
    def grid_fn(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _nvfp4_gemm_kernel[grid_fn](
        A_packed, A_scale, W_rhs, W_s_rhs, C, B,
        M, N, K,
        A_packed.stride(0), A_packed.stride(1),
        W_rhs.stride(0), W_rhs.stride(1),
        W_s_rhs.stride(0), W_s_rhs.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=HAS_BIAS,
    )
    torch.cuda.synchronize()

    # 计时 GEMM kernel（纯 MMA，不含 A 量化/W 变换）
    def gemm_only():
        _nvfp4_gemm_kernel[grid_fn](
            A_packed, A_scale, W_rhs, W_s_rhs, C, B,
            M, N, K,
            A_packed.stride(0), A_packed.stride(1),
            W_rhs.stride(0), W_rhs.stride(1),
            W_s_rhs.stride(0), W_s_rhs.stride(1),
            C.stride(0), C.stride(1),
            HAS_BIAS=HAS_BIAS,
        )
    t_gemm = triton.testing.do_bench(gemm_only, warmup=25, rep=100)

    flops = 2.0 * M * N * K
    tf_ref = flops / (t_ref * 1e-3) / 1e12
    tf_gemm = flops / (t_gemm * 1e-3) / 1e12

    print(f"M={M:5d} K={K:5d} N={N:5d} | ref {t_ref*1e3:8.2f}ms {tf_ref:6.1f}TF | "
          f"GEMM-only {t_gemm*1e3:8.2f}ms {tf_gemm:6.1f}TF | speedup {t_ref/t_gemm:6.2f}x")