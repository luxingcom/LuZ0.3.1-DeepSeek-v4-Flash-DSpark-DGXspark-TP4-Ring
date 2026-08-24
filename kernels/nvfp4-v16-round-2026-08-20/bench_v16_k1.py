"""kernel① v16.1(修复版) 性能基准：W 预处理缓存后全算子 TFLOPS（A 量化在核内，无需分离口径）。
对比基线：v13 GEMM-only 29.6~46.9 / v15 GEMM-only 26.7~81.4。
"""
import torch
import triton
import time

from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_v16_fixed_triton import (
    nvfp4_4w4a_prefill_gemm as v16_impl,
    preprocess_weights_clear,
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


SHAPES = [
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 4096),
    (256, 4096, 16384),
]


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("=" * 70)
    print("kernel① v16.1(修复版) 全算子 TFLOPS（W 缓存，A 量化核内）")
    print("=" * 70)
    print(f"{'M,K,N':>22} | {'TFLOPS':>9} | {'对照v15 GEMM-only':>16}")
    print("-" * 60)
    for (M, K, N) in SHAPES:
        torch.manual_seed(0)
        A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
        preprocess_weights_clear()
        v16_impl(A, W_n, W_scale, None)  # 首次调用：W 预处理 + autotune
        torch.cuda.synchronize()
        t = bench(lambda: v16_impl(A, W_n, W_scale, None))
        flops = 2.0 * M * N * K
        tf = flops / t / 1e12
        print(f"{M:>5d},{K:>5d},{N:>5d} | {tf:>8.1f} |")


if __name__ == "__main__":
    main()
