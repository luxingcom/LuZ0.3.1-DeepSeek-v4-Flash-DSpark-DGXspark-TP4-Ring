# ============================================================================
# test_nvfp4_4w4a_prefill_gemm.py —— kernel① prefill GEMM 正确性测试
# 用途：Triton 实现 vs torch 参考实现数值一致性（4 组 shape × bias 开/关）。
# 运行：python -m pytest test_nvfp4_4w4a_prefill_gemm.py -v
# 期望：8 个用例通过（assert_close rtol=atol=5e-2；4-bit 量化参与计算故容差放宽）。
# ============================================================================
import torch
import pytest

from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_triton import nvfp4_4w4a_prefill_gemm as triton_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def make_weights(K, N, scale, device):
    """构造 NVFP4 打包权重（E2M1 低半字节优先）+ E8M0 block scale [K//32, N//128]。"""
    w = (torch.rand(K, N, device=device) * 2 - 1) * scale
    w_scale_raw = w.abs().amax(dim=0).clamp(min=1e-9)  # per-col max
    w_scale_blocks = w_scale_raw.view(N // 128, 128).amax(dim=1)  # [N//128]
    # 按 (k-group32, n-block128) 组织 scale：K 方向用同一个 32 行组的 max（Triton e8m0 硬约束）
    exp = torch.floor(torch.log2(w_scale_blocks.clamp(min=1e-30) / 6.0)) + 127.0
    exp = exp.clamp(0, 255).to(torch.uint8)  # [N//128]
    w_scale = exp.unsqueeze(0).repeat(K // 32, 1)  # [K//32, N//128]
    w_scale_f = torch.pow(2.0, w_scale.float() - 127.0)
    # 展开到 [K, N]：K 方向每块重复 32 行、N 方向每块重复 128 列
    w_scale_expanded = w_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)
    w_scaled = w / w_scale_expanded
    # 量化到最近 E2M1
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
    # 打包：偶数列低半字节，奇数列高半字节
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = lo | (hi << 4)
    return packed.contiguous(), w_scale.contiguous()


@pytest.mark.parametrize("M,K,N", [
    (256, 4096, 4096),
    (512, 2048, 4096),
    (1024, 4096, 2048),
    (128, 4096, 4096),
])
@pytest.mark.parametrize("use_bias", [False, True])
def test_nvfp4_4w4a_prefill_gemm(M, K, N, use_bias):
    if DEVICE != "cuda":
        pytest.skip("需要 CUDA（DGX Spark / SM121）")
    torch.manual_seed(0)
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W_quant, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float32) if use_bias else None

    ref = ref_impl(A, W_quant, W_scale, bias)
    out = triton_impl(A, W_quant, W_scale, bias)

    # 4-bit 量化参与计算，容差放宽
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
