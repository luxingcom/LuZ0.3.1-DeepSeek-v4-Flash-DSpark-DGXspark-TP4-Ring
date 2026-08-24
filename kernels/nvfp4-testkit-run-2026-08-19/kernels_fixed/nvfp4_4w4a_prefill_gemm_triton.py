import torch
import triton
import triton.language as tl
from typing import Optional

try:
    import triton.experimental.tle.language as tle
    HAS_TLE = True
except ImportError:
    HAS_TLE = False


_E8M0_BIAS: tl.constexpr = tl.constexpr(127)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=16, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=8, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=8, num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _nvfp4_gemm_kernel(
    A_ptr,
    W_packed_ptr,
    W_scale_ptr,
    bias_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_ws_kg, stride_ws_ng,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_GROUP_K: tl.constexpr,
    W_GROUP_K: tl.constexpr,
    W_GROUP_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N

    m_range = m_off + tl.arange(0, BLOCK_M)
    n_range = n_off + tl.arange(0, BLOCK_N)

    m_mask = m_range < M
    n_mask = n_range < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    GROUPS_PER_BLOCK_K: tl.constexpr = BLOCK_K // A_GROUP_K

    for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k_tile * BLOCK_K
        k_range = k_base + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        a_ptrs = A_ptr + (
            m_range[:, None].to(tl.int64) * stride_am +
            k_range[None, :].to(tl.int64) * stride_ak
        )
        A_tile = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy='evict_last',
        ).to(tl.float32)

        A_g = tl.reshape(A_tile, [BLOCK_M, GROUPS_PER_BLOCK_K, A_GROUP_K])
        A_abs_max = tl.max(tl.abs(A_g), axis=2)

        scale_val = A_abs_max / 6.0
        scale_val = tl.maximum(scale_val, 1e-38)
        # tl.max 累加会把 fp32 提升为 fp64；bitcast 依赖 32 位布局，需显式回落 fp32
        scale_val = scale_val.to(tl.float32)
        scale_bits = scale_val.to(tl.int32, bitcast=True)
        a_exp = (scale_bits >> 23) - 127
        # tl.clamp 仅支持浮点，int32 需手动 min/max clamp
        a_exp_u8 = tl.minimum(tl.maximum(a_exp + _E8M0_BIAS, 0), 255).to(tl.uint8)

        a_scale_fp32 = tl.exp2(
            (a_exp_u8.to(tl.int32) - _E8M0_BIAS).to(tl.float32)
        )

        A_norm = A_g / (tl.reshape(a_scale_fp32, [BLOCK_M, GROUPS_PER_BLOCK_K, 1]) + 1e-38)
        A_norm = tl.clamp(A_norm, -6.0, 6.0)

        A_sign = tl.where(A_norm < 0.0, -1.0, 1.0)
        A_absnorm = tl.abs(A_norm)

        mag_idx = tl.zeros([BLOCK_M, GROUPS_PER_BLOCK_K, A_GROUP_K], dtype=tl.int32)
        mag_idx = tl.where(A_absnorm >= 0.25, 1, mag_idx)
        mag_idx = tl.where(A_absnorm >= 0.75, 2, mag_idx)
        mag_idx = tl.where(A_absnorm >= 1.25, 3, mag_idx)
        mag_idx = tl.where(A_absnorm >= 1.75, 4, mag_idx)
        mag_idx = tl.where(A_absnorm >= 2.50, 5, mag_idx)
        mag_idx = tl.where(A_absnorm >= 3.50, 6, mag_idx)
        mag_idx = tl.where(A_absnorm >= 5.00, 7, mag_idx)

        sign_bit = tl.where(A_sign < 0.0, 8, 0).to(tl.int32)
        nibble = (sign_bit | mag_idx).to(tl.uint8)

        nibble_flat = tl.reshape(nibble, [BLOCK_M, BLOCK_K])

        n_half_base = n_off // 2
        n_half_range = n_half_base + tl.arange(0, BLOCK_N // 2)
        n_half_mask = n_half_range < (N // 2)

        w_ptrs = W_packed_ptr + (
            k_range[:, None].to(tl.int64) * stride_wk +
            n_half_range[None, :].to(tl.int64) * stride_wn
        )
        W_tile_packed = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_half_mask[None, :],
            other=0,
            eviction_policy='evict_first',
        ).to(tl.uint8)

        w_lo = W_tile_packed & tl.full([BLOCK_K, BLOCK_N // 2], 0x0F, dtype=tl.uint8)
        w_hi = (W_tile_packed >> 4) & tl.full([BLOCK_K, BLOCK_N // 2], 0x0F, dtype=tl.uint8)

        w_lo_r = tl.reshape(w_lo, [BLOCK_K, BLOCK_N // 2, 1])
        w_hi_r = tl.reshape(w_hi, [BLOCK_K, BLOCK_N // 2, 1])

        w_interleaved = tl.reshape(
            tl.join(w_lo_r, w_hi_r),
            [BLOCK_K, BLOCK_N],
        )

        w_t = tl.trans(w_interleaved)
        w_t_r = tl.reshape(w_t, [BLOCK_N, BLOCK_K // 2, 2])
        w_t_split = tl.split(w_t_r)
        w_k_lo = tl.reshape(w_t_split[0], [BLOCK_N, BLOCK_K // 2])
        w_k_hi = tl.reshape(w_t_split[1], [BLOCK_N, BLOCK_K // 2])
        W_rhs_packed = (w_k_lo | (w_k_hi << 4)).to(tl.uint8)

        n_group_range = n_range // W_GROUP_N

        w_scale_rows = tl.zeros([GROUPS_PER_BLOCK_K, BLOCK_N], dtype=tl.float32)
        for g in tl.static_range(GROUPS_PER_BLOCK_K):
            k_abs = k_base + g * A_GROUP_K
            kg = k_abs // W_GROUP_K
            ws_ptrs = W_scale_ptr + (
                tl.full([BLOCK_N], kg, dtype=tl.int64) * stride_ws_kg +
                n_group_range.to(tl.int64) * stride_ws_ng
            )
            ws_raw = tl.load(
                ws_ptrs,
                mask=n_mask,
                other=127,
                eviction_policy='evict_first',
            ).to(tl.uint8)
            ws_fp32 = tl.exp2(
                (ws_raw.to(tl.int32) - _E8M0_BIAS).to(tl.float32)
            )
            g_ids = tl.arange(0, GROUPS_PER_BLOCK_K)
            row_mask = (g_ids[:, None] == g)
            w_scale_rows = tl.where(
                row_mask,
                tl.broadcast_to(tl.reshape(ws_fp32, [1, BLOCK_N]), [GROUPS_PER_BLOCK_K, BLOCK_N]),
                w_scale_rows,
            )

        for g in tl.static_range(GROUPS_PER_BLOCK_K):
            a_nib_g = tl.reshape(
                nibble_flat,
                [BLOCK_M, GROUPS_PER_BLOCK_K, A_GROUP_K],
            )
            a_nib_g_slice = tl.reshape(
                tl.sum(
                    tl.where(
                        tl.reshape(tl.arange(0, GROUPS_PER_BLOCK_K), [1, GROUPS_PER_BLOCK_K, 1]) == g,
                        a_nib_g,
                        tl.zeros([BLOCK_M, GROUPS_PER_BLOCK_K, A_GROUP_K], dtype=a_nib_g.dtype),
                    ),
                    axis=1,
                ),
                [BLOCK_M, A_GROUP_K],
            ).to(tl.uint8)

            a_hi_g_src = tl.reshape(a_nib_g_slice, [BLOCK_M, A_GROUP_K // 2, 2])
            a_hi_g_split = tl.split(a_hi_g_src)
            a_lo_half = tl.reshape(a_hi_g_split[0], [BLOCK_M, A_GROUP_K // 2])
            a_hi_half = tl.reshape(a_hi_g_split[1], [BLOCK_M, A_GROUP_K // 2])
            lhs_g = (a_lo_half | (a_hi_half << 4)).to(tl.uint8)

            w_rhs_g_full = tl.reshape(W_rhs_packed, [BLOCK_N, GROUPS_PER_BLOCK_K, A_GROUP_K // 2])
            w_rhs_g_sum = tl.sum(
                tl.where(
                    tl.reshape(tl.arange(0, GROUPS_PER_BLOCK_K), [1, GROUPS_PER_BLOCK_K, 1]) == g,
                    w_rhs_g_full,
                    tl.zeros([BLOCK_N, GROUPS_PER_BLOCK_K, A_GROUP_K // 2], dtype=w_rhs_g_full.dtype),
                ),
                axis=1,
            )
            rhs_g = tl.reshape(w_rhs_g_sum, [BLOCK_N, A_GROUP_K // 2]).to(tl.uint8)

            a_scale_col = tl.reshape(a_scale_fp32, [BLOCK_M, GROUPS_PER_BLOCK_K])
            lhs_scale_col = tl.sum(
                tl.where(
                    tl.reshape(tl.arange(0, GROUPS_PER_BLOCK_K), [1, GROUPS_PER_BLOCK_K]) == g,
                    a_scale_col,
                    tl.zeros([BLOCK_M, GROUPS_PER_BLOCK_K], dtype=tl.float32),
                ),
                axis=1,
            )
            lhs_scale_g = tl.reshape(
                tl.max(tl.reshape(lhs_scale_col, [BLOCK_M // 32, 32]), axis=1),
                [BLOCK_M // 32, 1],
            )

            rhs_scale_row = tl.sum(
                tl.where(
                    tl.reshape(tl.arange(0, GROUPS_PER_BLOCK_K), [GROUPS_PER_BLOCK_K, 1]) == g,
                    w_scale_rows,
                    tl.zeros([GROUPS_PER_BLOCK_K, BLOCK_N], dtype=tl.float32),
                ),
                axis=0,
            )
            rhs_scale_g = tl.reshape(
                tl.max(tl.reshape(rhs_scale_row, [1, BLOCK_N // 32, 32]), axis=2),
                [1, BLOCK_N // 32],
            )

            tile_out = tl.dot_scaled(
                lhs_g,
                lhs_scale_g,
                rhs_g,
                rhs_scale_g,
                out_dtype=tl.float32,
                lhs_type="e2m1",
                rhs_type="e2m1",
                rhs_k_pack=True,
            )

            acc = acc + tile_out

    if HAS_BIAS:
        bias_vals = tl.load(
            bias_ptr + n_range.to(tl.int64),
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        acc = acc + bias_vals[None, :]

    c_ptrs = C_ptr + (
        m_range[:, None].to(tl.int64) * stride_cm +
        n_range[None, :].to(tl.int64) * stride_cn
    )
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Determine target device — prefer CUDA, fall back to the device of the first
    # non-CPU tensor, and only then use CPU if nothing else is available.
    device = None
    for t in [A, W_packed, W_scale]:
        if t.device.type != 'cpu':
            device = t.device
            break
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        else:
            device = A.device

    A_fp32 = A.to(dtype=torch.float32, device=device).contiguous()
    W_packed_dev = W_packed.to(device=device).contiguous()
    W_scale_dev = W_scale.to(device=device).contiguous()

    M, K = A_fp32.shape
    K2, Nh = W_packed_dev.shape
    N = Nh * 2

    assert K2 == K, f"Weight K={K2} must equal activation K={K}"
    assert K % 16 == 0, f"K must be divisible by 16, got {K}"
    assert N % 128 == 0, f"N must be divisible by 128, got {N}"
    assert M % 32 == 0, f"M must be divisible by 32 for dot_scaled scale granularity"

    C = torch.empty((M, N), dtype=torch.float32, device=device)

    has_bias = bias is not None
    if has_bias:
        bias_fp32 = bias.to(dtype=torch.float32, device=device).contiguous()
        bias_ptr = bias_fp32
    else:
        bias_ptr = C

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _nvfp4_gemm_kernel[grid](
        A_fp32,
        W_packed_dev,
        W_scale_dev,
        bias_ptr,
        C,
        M, N, K,
        A_fp32.stride(0), A_fp32.stride(1),
        W_packed_dev.stride(0), W_packed_dev.stride(1),
        W_scale_dev.stride(0), W_scale_dev.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=has_bias,
        A_GROUP_K=16,
        W_GROUP_K=16,
        W_GROUP_N=128,
    )

    return C
