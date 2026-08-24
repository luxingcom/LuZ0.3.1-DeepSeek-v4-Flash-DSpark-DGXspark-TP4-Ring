import torch
import triton
import triton.language as tl
from typing import Optional

try:
    import triton.experimental.tle.language as tle
    HAS_TLE = True
except ImportError:
    HAS_TLE = False


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'K'],
)
@triton.jit
def _a_quant_kernel(
    A_ptr,
    A_quant_ptr,
    A_scale_ptr,
    M, K,
    stride_am, stride_ak,
    stride_aqm, stride_aqk,
    stride_asm, stride_ask,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    a_ptrs = A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0,
                eviction_policy='evict_last')

    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)

    safe_max = tl.maximum(block_max, 1e-38)
    log2_val = tl.log2(safe_max / 6.0)
    e8m0_f = tl.floor(log2_val) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    e8m0_u8 = e8m0_f.to(tl.uint8)

    scale_ptrs = A_scale_ptr + offs_m.to(tl.int64) * stride_asm + (pid_k * stride_ask).to(tl.int64)
    tl.store(scale_ptrs, e8m0_u8, mask=mask_m)

    a_scale = tl.exp2(e8m0_f - 127.0)
    a_scaled = a / a_scale[:, None]

    a_abs_s = tl.abs(a_scaled)
    idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    idx = idx + (a_abs_s > 0.25).to(tl.int32)
    idx = idx + (a_abs_s > 0.75).to(tl.int32)
    idx = idx + (a_abs_s > 1.25).to(tl.int32)
    idx = idx + (a_abs_s > 1.75).to(tl.int32)
    idx = idx + (a_abs_s > 2.5).to(tl.int32)
    idx = idx + (a_abs_s > 3.5).to(tl.int32)
    idx = idx + (a_abs_s > 5.0).to(tl.int32)

    neg_mask = (a_scaled < 0.0).to(tl.int32)
    sign_bit = tl.where(idx > 0, neg_mask * 8, 0)
    nibble = (idx + sign_bit).to(tl.uint8)

    nibble_i32 = nibble.to(tl.int32)
    nibble_3d = tl.reshape(nibble_i32, [BLOCK_M, BLOCK_K // 2, 2])
    sel_lo_r = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    sel_hi_r = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(nibble_3d * sel_lo_r, axis=2)
    hi_val = tl.sum(nibble_3d * sel_hi_r, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)

    offs_k_packed = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    mask_k_packed = offs_k_packed < (K // 2)
    aq_ptrs = A_quant_ptr + offs_m[:, None].to(tl.int64) * stride_aqm + offs_k_packed[None, :].to(tl.int64) * stride_aqk
    tl.store(aq_ptrs, packed, mask=mask_m[:, None] & mask_k_packed[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp4_gemm_kernel(
    A_quant_ptr,
    A_scale_ptr,
    W_packed_ptr,
    W_scale_ptr,
    C_ptr,
    bias_ptr,
    has_bias: tl.constexpr,
    M, N, K,
    stride_aqm, stride_aqk,
    stride_asm, stride_ask,
    stride_wk, stride_wn,
    stride_wsk, stride_wsn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    W_BLOCK_K: tl.constexpr,
    W_BLOCK_N: tl.constexpr,
    A_BLOCK_K: tl.constexpr,
):
    # PATH ANNOTATION: FALLBACK PATH - bf16 dequant (not native e2m1 MMA)
    # GROUPS_WK and GROUPS_WN are derived at runtime from constexprs
    GROUPS_K: tl.constexpr = BLOCK_K // A_BLOCK_K
    GROUPS_WK: tl.constexpr = BLOCK_K // W_BLOCK_K
    # GROUPS_WN must use the fixed W_BLOCK_N=128 constexpr passed in
    # BLOCK_N is guaranteed >= W_BLOCK_N by autotune configs (min BLOCK_N=128)
    GROUPS_WN: tl.constexpr = BLOCK_N // W_BLOCK_N

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    offs_n_w_packed = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    mask_nw = offs_n_w_packed < (N // 2)

    ws_n_start = (pid_n * BLOCK_N) // W_BLOCK_N
    offs_wsn = ws_n_start + tl.arange(0, GROUPS_WN)
    mask_wsn = offs_wsn < (N // W_BLOCK_N)

    mask_m = offs_m < M

    for k_tile in tl.range(0, num_k_tiles):
        k_start = k_tile * BLOCK_K

        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        mask_k_packed = offs_k_packed < (K // 2)

        aq_ptrs = A_quant_ptr + offs_m[:, None].to(tl.int64) * stride_aqm + offs_k_packed[None, :].to(tl.int64) * stride_aqk
        a_packed = tl.load(aq_ptrs, mask=mask_m[:, None] & mask_k_packed[None, :], other=0,
                           eviction_policy='evict_last')

        offs_as_k = k_start // A_BLOCK_K + tl.arange(0, GROUPS_K)
        mask_as_k = offs_as_k < (K // A_BLOCK_K)
        as_ptrs = A_scale_ptr + offs_m[:, None].to(tl.int64) * stride_asm + offs_as_k[None, :].to(tl.int64) * stride_ask
        a_scale_u8 = tl.load(as_ptrs, mask=mask_m[:, None] & mask_as_k[None, :], other=127,
                              eviction_policy='evict_last')

        offs_k_w = k_start + tl.arange(0, BLOCK_K)
        mask_kw = offs_k_w < K

        w_ptrs = W_packed_ptr + offs_k_w[:, None].to(tl.int64) * stride_wk + offs_n_w_packed[None, :].to(tl.int64) * stride_wn
        w_n_packed = tl.load(w_ptrs, mask=mask_kw[:, None] & mask_nw[None, :], other=0,
                             eviction_policy='evict_first')

        ws_k_start = k_start // W_BLOCK_K
        offs_wsk = ws_k_start + tl.arange(0, GROUPS_WK)
        mask_wsk = offs_wsk < (K // W_BLOCK_K)

        ws_ptrs = W_scale_ptr + offs_wsk[:, None].to(tl.int64) * stride_wsk + offs_wsn[None, :].to(tl.int64) * stride_wsn
        w_scale_u8 = tl.load(ws_ptrs, mask=mask_wsk[:, None] & mask_wsn[None, :], other=127,
                              eviction_policy='evict_first')

        # Dequantize A scale: [BLOCK_M, GROUPS_K] -> fp32
        a_scale_fp32 = tl.exp2(a_scale_u8.to(tl.float32) - 127.0)

        # Unpack A nibbles: [BLOCK_M, BLOCK_K//2] -> lo/hi each [BLOCK_M, BLOCK_K//2]
        a_packed_i32 = a_packed.to(tl.int32)
        a_lo_i32 = a_packed_i32 & 0xF
        a_hi_i32 = (a_packed_i32 >> 4) & 0xF

        a_lo_mag = a_lo_i32 & 7
        a_hi_mag = a_hi_i32 & 7

        # E2M1 magnitude lookup via nested tl.where
        a_lo_mag_f = tl.where(a_lo_mag == 0, 0.0,
                     tl.where(a_lo_mag == 1, 0.5,
                     tl.where(a_lo_mag == 2, 1.0,
                     tl.where(a_lo_mag == 3, 1.5,
                     tl.where(a_lo_mag == 4, 2.0,
                     tl.where(a_lo_mag == 5, 3.0,
                     tl.where(a_lo_mag == 6, 4.0, 6.0)))))))
        a_hi_mag_f = tl.where(a_hi_mag == 0, 0.0,
                     tl.where(a_hi_mag == 1, 0.5,
                     tl.where(a_hi_mag == 2, 1.0,
                     tl.where(a_hi_mag == 3, 1.5,
                     tl.where(a_hi_mag == 4, 2.0,
                     tl.where(a_hi_mag == 5, 3.0,
                     tl.where(a_hi_mag == 6, 4.0, 6.0)))))))

        a_lo_f = tl.where(a_lo_i32 >= 8, -a_lo_mag_f, a_lo_mag_f)
        a_hi_f = tl.where(a_hi_i32 >= 8, -a_hi_mag_f, a_hi_mag_f)

        # Expand A scale from [BLOCK_M, GROUPS_K] to [BLOCK_M, BLOCK_K]
        # Each scale covers A_BLOCK_K=32 consecutive K elements
        a_scale_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(a_scale_fp32, [BLOCK_M, GROUPS_K, 1]),
                [BLOCK_M, GROUPS_K, A_BLOCK_K]
            ),
            [BLOCK_M, BLOCK_K]
        )

        # Interleave lo/hi nibbles back to [BLOCK_M, BLOCK_K]
        # lo nibble = even K index, hi nibble = odd K index
        a_lo_r = tl.reshape(a_lo_f, [BLOCK_M, BLOCK_K // 2, 1])
        a_hi_r = tl.reshape(a_hi_f, [BLOCK_M, BLOCK_K // 2, 1])
        sel = tl.reshape(tl.arange(0, 2), [1, 1, 2])
        a_interleaved = tl.reshape(
            tl.where(sel == 0,
                     tl.broadcast_to(a_lo_r, [BLOCK_M, BLOCK_K // 2, 2]),
                     tl.broadcast_to(a_hi_r, [BLOCK_M, BLOCK_K // 2, 2])),
            [BLOCK_M, BLOCK_K]
        )
        a_full = a_interleaved * a_scale_exp

        # Unpack W nibbles: [BLOCK_K, BLOCK_N//2] -> lo/hi each [BLOCK_K, BLOCK_N//2]
        w_n_i32 = w_n_packed.to(tl.int32)
        w_lo_i32 = w_n_i32 & 0xF
        w_hi_i32 = (w_n_i32 >> 4) & 0xF

        w_lo_mag = w_lo_i32 & 7
        w_hi_mag = w_hi_i32 & 7

        w_lo_mag_f = tl.where(w_lo_mag == 0, 0.0,
                     tl.where(w_lo_mag == 1, 0.5,
                     tl.where(w_lo_mag == 2, 1.0,
                     tl.where(w_lo_mag == 3, 1.5,
                     tl.where(w_lo_mag == 4, 2.0,
                     tl.where(w_lo_mag == 5, 3.0,
                     tl.where(w_lo_mag == 6, 4.0, 6.0)))))))
        w_hi_mag_f = tl.where(w_hi_mag == 0, 0.0,
                     tl.where(w_hi_mag == 1, 0.5,
                     tl.where(w_hi_mag == 2, 1.0,
                     tl.where(w_hi_mag == 3, 1.5,
                     tl.where(w_hi_mag == 4, 2.0,
                     tl.where(w_hi_mag == 5, 3.0,
                     tl.where(w_hi_mag == 6, 4.0, 6.0)))))))

        w_lo_f = tl.where(w_lo_i32 >= 8, -w_lo_mag_f, w_lo_mag_f)
        w_hi_f = tl.where(w_hi_i32 >= 8, -w_hi_mag_f, w_hi_mag_f)

        # Dequantize W scale: [GROUPS_WK, GROUPS_WN] -> fp32
        w_scale_fp32 = tl.exp2(w_scale_u8.to(tl.float32) - 127.0)

        # Expand W scale from [GROUPS_WK, GROUPS_WN] to [BLOCK_K, BLOCK_N]
        w_scale_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(w_scale_fp32, [GROUPS_WK, 1, GROUPS_WN, 1]),
                [GROUPS_WK, W_BLOCK_K, GROUPS_WN, W_BLOCK_N]
            ),
            [BLOCK_K, BLOCK_N]
        )

        # Interleave W lo/hi nibbles to [BLOCK_K, BLOCK_N]
        # W is N-packed: lo nibble = even N col, hi nibble = odd N col
        w_lo_r = tl.reshape(w_lo_f, [BLOCK_K, BLOCK_N // 2, 1])
        w_hi_r = tl.reshape(w_hi_f, [BLOCK_K, BLOCK_N // 2, 1])
        sel_w = tl.reshape(tl.arange(0, 2), [1, 1, 2])
        w_full = tl.reshape(
            tl.where(sel_w == 0,
                     tl.broadcast_to(w_lo_r, [BLOCK_K, BLOCK_N // 2, 2]),
                     tl.broadcast_to(w_hi_r, [BLOCK_K, BLOCK_N // 2, 2])),
            [BLOCK_K, BLOCK_N]
        )

        w_full_scaled = w_full * w_scale_exp

        acc = tl.dot(
            a_full.to(tl.bfloat16),
            w_full_scaled.to(tl.bfloat16),
            acc=acc,
            out_dtype=tl.float32
        )

    mask_n = offs_n < N

    if has_bias:
        bias = tl.load(bias_ptr + offs_n.to(tl.int64), mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    NVFP4 4W4A block-scaled GEMM.
    PATH: FALLBACK - bf16 dequant (not native e2m1 MMA).

    C[M,N] = (A_scale * W_scale) * dot_e2m1(A_quant, W_quant) + bias

    Args:
        A:        fp32/fp16/bf16 [M, K]
        W_packed: uint8 [K, N//2], N-packed (lo nibble=even col, hi nibble=odd col)
        W_scale:  uint8 [K//32, N//128], E8M0 block scales
        bias:     fp32 [N] or None

    Returns:
        C: fp32 [M, N]
    """
    A = A.to(torch.float32).contiguous()

    M, K = A.shape
    K_w, N_half = W_packed.shape
    N = N_half * 2
    assert K == K_w, f"K mismatch: A has K={K}, W_packed has K={K_w}"
    assert K % 32 == 0, "K must be divisible by 32"
    assert N % 128 == 0, "N must be divisible by 128"
    assert M % 32 == 0, "M must be divisible by 32"

    W_packed = W_packed.contiguous()
    W_scale = W_scale.contiguous()

    device = A.device

    A_quant = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    A_scale_out = torch.empty((M, K // 32), dtype=torch.uint8, device=device)

    grid_aq = (triton.cdiv(M, 128), triton.cdiv(K, 32))

    _a_quant_kernel[grid_aq](
        A,
        A_quant,
        A_scale_out,
        M, K,
        A.stride(0), A.stride(1),
        A_quant.stride(0), A_quant.stride(1),
        A_scale_out.stride(0), A_scale_out.stride(1),
    )

    C = torch.empty((M, N), dtype=torch.float32, device=device)

    bias_ptr = bias.to(torch.float32).contiguous() if bias is not None else A

    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fp4_gemm_kernel[grid_gemm](
        A_quant,
        A_scale_out,
        W_packed,
        W_scale,
        C,
        bias_ptr,
        has_bias=(bias is not None),
        M=M, N=N, K=K,
        stride_aqm=A_quant.stride(0), stride_aqk=A_quant.stride(1),
        stride_asm=A_scale_out.stride(0), stride_ask=A_scale_out.stride(1),
        stride_wk=W_packed.stride(0), stride_wn=W_packed.stride(1),
        stride_wsk=W_scale.stride(0), stride_wsn=W_scale.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        W_BLOCK_K=32,
        W_BLOCK_N=128,
        A_BLOCK_K=32,
    )

    return C
