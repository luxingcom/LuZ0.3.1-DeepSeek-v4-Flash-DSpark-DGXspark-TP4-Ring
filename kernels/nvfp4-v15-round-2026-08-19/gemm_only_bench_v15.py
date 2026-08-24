"""kernel① v15 GEMM-only 基准（对齐 round6/round7 口径）：
W 预反量化缓存 + A 预量化一次，仅计时 _bf16_gemm_kernel。
同时给全算子（A量化+bf16 GEMM，W 缓存）数据对比。
"""
import torch
import triton
import time

from nvfp4_4w4a_prefill_gemm_v15_triton import (
    _bf16_gemm_kernel,
    _quantize_activation_triton,
    _preprocess_weights,
    preprocess_weights_clear,
    nvfp4_4w4a_prefill_gemm,
)

DEVICE = "cuda"
E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def make_weights(K, N, scale, device):
    w = (torch.rand(K, N, device=device) * 2 - 1) * scale
    w_scale_raw = w.abs().amax(dim=0).clamp(min=1e-9)
    w_scale_blocks = w_scale_raw.view(N // 128, 128).amax(dim=1)
    exp = torch.floor(torch.log2(w_scale_blocks.clamp(min=1e-30) / 6.0)) + 127.0
    exp = exp.clamp(0, 255).to(torch.uint8)
    w_scale = exp.unsqueeze(0).repeat(K // 32, 1)
    w_scale_f = torch.pow(2.0, w_scale.float() - 127.0)
    w_scale_expanded = w_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)
    w_scaled = w / w_scale_expanded
    signs = torch.sign(w_scaled)
    w_abs = w_scaled.abs()
    pos = E2M1_VALUES[:8].to(device)
    idx = (w_abs.unsqueeze(-1) - pos).abs().argmin(dim=-1)
    w_q = (signs * pos[idx]).nan_to_num(0.0)
    nib = torch.zeros(K, N, dtype=torch.uint8, device=device)
    mag = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    mag_val = torch.where(w_q.abs().unsqueeze(-1) == pos.unsqueeze(0).unsqueeze(0),
                          mag, torch.zeros_like(mag)).sum(dim=-1).to(torch.uint8)
    sign_bit = (w_q < 0).to(torch.uint8) * 8
    nib = (mag_val | sign_bit).to(torch.uint8)
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = lo | (hi << 4)
    return packed.contiguous(), w_scale.contiguous()


def repack_n_to_k(packed_n, K, N):
    lo = packed_n & 0x0F
    hi = (packed_n >> 4) & 0x0F
    nib = torch.empty(K, N, dtype=torch.uint8, device=packed_n.device)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    nib_t = nib.t().contiguous()
    lo_k = nib_t[:, 0::2]
    hi_k = nib_t[:, 1::2]
    return (lo_k | (hi_k << 4)).t().contiguous()


SHAPES = [
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 4096),
    (256, 4096, 16384),
]


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("=" * 78)
    print("kernel① v15 性能基准 | GEMM-only（A预量化+W缓存）vs 全算子（W缓存）")
    print("=" * 78)
    print(f"{'M,K,N':>24} | {'GEMM-only TFLOPS':>17} | {'全算子 TFLOPS':>14}")
    print("-" * 78)
    for (M, K, N) in SHAPES:
        torch.manual_seed(0)
        A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
        W_k = repack_n_to_k(W_n, K, N)
        preprocess_weights_clear()

        # W 预处理缓存一次（v15 内部按 data_ptr 缓存）
        W_dequant = _preprocess_weights(W_k, W_scale, N, K)     # [N, K] fp32
        A_dequant = _quantize_activation_triton(A)               # [M, K] fp32
        A_bf16 = A_dequant.to(torch.bfloat16).contiguous()
        W_bf16 = W_dequant.to(torch.bfloat16).contiguous()
        C = torch.empty((M, N), dtype=torch.float32, device=DEVICE)

        def gemm_only():
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            _bf16_gemm_kernel[grid](
                A_bf16, W_bf16, C, M, N, K,
                A_bf16.stride(0), A_bf16.stride(1),
                W_bf16.stride(0), W_bf16.stride(1),
                C.stride(0), C.stride(1),
            )

        def full_op():
            nvfp4_4w4a_prefill_gemm(A, W_k, W_scale, None)

        # 先跑一次让 autotune 决定最佳 config
        gemm_only()
        full_op()
        torch.cuda.synchronize()

        t_gemm = bench(gemm_only)
        t_full = bench(full_op)
        flops = 2.0 * M * N * K
        tf_gemm = flops / t_gemm / 1e12
        tf_full = flops / t_full / 1e12
        print(f"{M:>5d},{K:>5d},{N:>5d}   | {tf_gemm:>14.1f}    | {tf_full:>12.1f}")


if __name__ == "__main__":
    main()
