import torch
import torch.nn.functional as F
from typing import Optional


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NVFP4 4W4A block-scaled GEMM reference implementation in PyTorch.

    Args:
        A: Activations [M, K], any dtype (converted to fp32 internally)
        W_packed: NVFP4 packed weights [K, N//2], uint8, low nibble = first element
        W_scale: E8M0 weight scales [K//16, N//128], uint8, one scale per (16, 128) block
        bias: Optional bias [N], fp32

    Returns:
        C: Output [M, N], fp32
    """
    # Convert activations to fp32
    A_fp32 = A.to(torch.float32)
    M, K = A_fp32.shape
    K2, N_half = W_packed.shape
    N = N_half * 2
    assert K2 == K, f"Weight K dimension {K2} must match activation K {K}"

    # E2M1 FP4 value table: maps nibble 0..15 to float
    # E2M1: sign=1 bit, exponent=2 bits, mantissa=1 bit
    # 注意：必须与权重同设备（修复 CPU 表 + GPU 索引报错）
    e2m1_table = torch.tensor([
        0.0,    # 0000
        0.5,    # 0001
        1.0,    # 0010
        1.5,    # 0011
        2.0,    # 0100
        3.0,    # 0101
        4.0,    # 0110
        6.0,    # 0111
        -0.0,   # 1000 (negative zero -> 0)
        -0.5,   # 1001
        -1.0,   # 1010
        -1.5,   # 1011
        -2.0,   # 1100
        -3.0,   # 1101
        -4.0,   # 1110
        -6.0,   # 1111
    ], dtype=torch.float32, device=W_packed.device)

    # Unpack weights from uint8 [K, N//2] to float32 [K, N]
    W_packed_int = W_packed.to(torch.int32)
    low_nibble = W_packed_int & 0x0F          # first element (low nibble)
    high_nibble = (W_packed_int >> 4) & 0x0F  # second element (high nibble)

    # Dequantize nibbles using e2m1 lookup table
    W_low = e2m1_table[low_nibble.reshape(-1)].reshape(K, N_half)
    W_high = e2m1_table[high_nibble.reshape(-1)].reshape(K, N_half)

    # Interleave: [K, N] with columns alternating low/high
    W_fp32 = torch.stack([W_low, W_high], dim=2).reshape(K, N)

    # Decode E8M0 weight scales: uint8 -> float32
    # E8M0: 8-bit exponent, no sign, no mantissa => value = 2^(exp - 127)
    W_scale_fp32 = torch.pow(2.0, W_scale.to(torch.float32) - 127.0)  # [K//32, N//128]

    # Expand weight scales to full [K, N] shape
    # Each scale covers a (32 along K, 128 along N) block (Triton e8m0 hard constraint)
    W_scale_expanded = W_scale_fp32.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)  # [K, N]

    # Apply weight scales to dequantized weights
    W_scaled = W_fp32 * W_scale_expanded  # [K, N]

    # Quantize activations to NVFP4 E2M1 with E8M0 block scales (group size 32 along K,
    # matching Triton 3.6 tl.dot_scaled e8m0 hard constraint and Blackwell mmaf_scaled)
    group_size = 32
    num_groups_k = K // group_size

    A_grouped = A_fp32.reshape(M, num_groups_k, group_size)  # [M, K//32, 32]

    # Compute per-group max absolute value for E8M0 scale computation
    A_abs_max = A_grouped.abs().amax(dim=2, keepdim=True)  # [M, K//32, 1]

    # Compute E8M0 scales: find exponent such that max_val / scale is in e2m1 range
    # E2M1 max value is 6.0, so scale = max_val / 6.0
    A_scale_val = A_abs_max / 6.0  # [M, K//32, 1]
    A_scale_val = torch.clamp(A_scale_val, min=1e-38)

    # Encode to E8M0 (biased exponent)
    A_exp = torch.floor(torch.log2(A_scale_val)).to(torch.int32)  # [M, K//32, 1]
    A_exp_clamped = torch.clamp(A_exp + 127, 0, 255)

    # Decode E8M0 activation scales back to float
    A_scale_fp32 = torch.pow(2.0, (A_exp_clamped - 127).to(torch.float32))  # [M, K//32, 1]

    # Quantize activations to E2M1 range
    A_normalized = A_grouped / (A_scale_fp32 + 1e-38)  # [M, K//32, 16]

    # Clamp to e2m1 representable range [-6, 6]
    A_normalized = torch.clamp(A_normalized, -6.0, 6.0)

    # Quantize to nearest e2m1 value using the lookup table
    e2m1_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 dtype=torch.float32, device=A_normalized.device)

    A_sign = torch.sign(A_normalized)
    A_abs = A_normalized.abs()

    # Find nearest positive e2m1 value
    A_abs_expanded = A_abs.unsqueeze(-1)  # [..., 1]
    e2m1_pos_view = e2m1_pos.to(A_abs.device)
    diffs = (A_abs_expanded - e2m1_pos_view).abs()
    nearest_idx = diffs.argmin(dim=-1)
    A_quantized_abs = e2m1_pos_view[nearest_idx]

    # Restore sign
    A_quantized = A_sign * A_quantized_abs  # [M, K//32, 32]

    # Rescale quantized activations back to original magnitude
    A_dequant = A_quantized * A_scale_fp32  # [M, K//32, 32]
    A_dequant = A_dequant.reshape(M, K)  # [M, K]

    # Compute GEMM: C = A_dequant @ W_scaled^T
    C = torch.matmul(A_dequant, W_scaled)  # [M, N]

    # Add optional bias
    if bias is not None:
        C = C + bias.to(torch.float32)

    return C
