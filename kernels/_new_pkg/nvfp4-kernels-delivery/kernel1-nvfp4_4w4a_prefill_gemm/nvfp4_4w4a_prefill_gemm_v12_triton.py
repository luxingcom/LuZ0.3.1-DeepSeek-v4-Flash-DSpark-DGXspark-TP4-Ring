import torch
import triton
import triton.language as tl

# ============================================================================
# nvfp4_4w4a_prefill_gemm —— v12（MCP 验证架构 + scale 语义修正）
# 环境：torch 2.11 / Triton 3.6 / CC=(12,1)=sm_121a（DGX Spark GB10）
#
# 架构（采纳 MCP autotune job 18df2864 v3，解决生产 v11 20 TFLOPS 的量化开销瓶颈）：
#   ① _quantize_fp32_to_nvfp4_packed：A fp32 → A_packed [M,K//2] + A_scale [M,K//32]（独立 kernel）
#   ② 主机侧 _repack_w_for_rhs_k_pack：W [K,N//2] → [K//2,N]（K 向 2:1 打包，每层一次可缓存）
#   ③ 主机侧 _expand_w_scale：W_scale [K//32,N//128] → [N,K//32]（uint8 e8m0，不 trans）
#   ④ GEMM kernel：纯 tl.dot_scaled MMA（无内核内量化开销）
#
# Triton 3.6.0/sm121 dot_scaled 三硬约束（生产实测 + MCP 二次确认）：
#   - rhs 布局 [K, N]（K_RHS,N = rhs.shape[-2:]），K 向打包 [BK//2, BN]
#   - lhs_scale / rhs_scale 必须 uint8 e8m0 原始字节（fp32 scale 触发 MLIR 崩溃）
#   - rhs_scale 不 tl.trans（[BN, GROUPS_K] 直接 load）
# scale 语义修正：biased_exp = floor(log2(max/6)) + 127（补 /6，全码本归一化）
# 舍入：阈值链严格 >（等距取低档，与 torch argmin 逐字节一致）
# ============================================================================

_E8M0_BIAS: tl.constexpr = tl.constexpr(127)


# ---------------------------------------------------------------------------
# ① A 量化 kernel：fp32 [M,K] → A_packed [M,K//2] uint8 + A_scale [M,K//32] uint8 e8m0
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_fp32_to_nvfp4_packed(
    A_ptr, A_packed_ptr, A_scale_ptr,
    M, K,
    stride_am, stride_ak,
    stride_pm, stride_pk,
    stride_sm, stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    k_start = pid_k * BLOCK_K

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    a_ptrs = A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0, eviction_policy='evict_last')
    a = a.to(tl.float32)

    GROUPS: tl.constexpr = BLOCK_K // 32

    a_reshaped = tl.reshape(a, (BLOCK_M, GROUPS, 32))
    a_abs_r = tl.abs(a_reshaped)
    abs_max = tl.max(a_abs_r, axis=2)  # [BLOCK_M, GROUPS]

    tiny = 1.175494e-38
    abs_max_clamped = tl.maximum(abs_max, tiny)

    # E8M0 scale: floor(log2(max/6)) + 127，clamp [0,255]（v12 修正：补 /6，全码本归一化）
    log2_val = tl.log2(abs_max_clamped / 6.0)
    floor_log2 = tl.floor(log2_val)
    biased_exp_f = floor_log2 + 127.0
    biased_exp_f = tl.maximum(biased_exp_f, 0.0)
    biased_exp_f = tl.minimum(biased_exp_f, 255.0)
    biased_exp_u8 = biased_exp_f.to(tl.uint8)

    group_k_start = k_start // 32
    offs_gk = group_k_start + tl.arange(0, GROUPS)
    mask_gk = offs_gk < (K // 32)

    s_ptrs = A_scale_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_gk[None, :].to(tl.int64) * stride_sk
    tl.store(s_ptrs, biased_exp_u8, mask=mask_m[:, None] & mask_gk[None, :])

    scale_fp32 = tl.exp2(biased_exp_f - 127.0)  # [BLOCK_M, GROUPS] = 2^floor(log2(max/6))

    # 归一化：x / scale ∈ [1,6]，全码本
    scale_3d = tl.reshape(scale_fp32, (BLOCK_M, GROUPS, 1))
    ones_32 = tl.full((1, 1, 32), 1.0, dtype=tl.float32)
    scale_expanded_3d = scale_3d * ones_32
    scale_expanded = tl.reshape(scale_expanded_3d, (BLOCK_M, BLOCK_K))

    a_norm = a / scale_expanded
    a_norm = tl.minimum(tl.maximum(a_norm, -6.0), 6.0)

    a_abs_norm = tl.abs(a_norm)
    sign_bit = (a_norm < 0.0).to(tl.int32)

    # 阈值链：严格 >（等距取低档，与 torch argmin 一致）
    mag_idx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 0.25).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 0.75).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 1.25).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 1.75).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 2.5).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 3.5).to(tl.int32)
    mag_idx = mag_idx + (a_abs_norm > 5.0).to(tl.int32)

    nibble = (sign_bit << 3) | mag_idx  # [BLOCK_M, BLOCK_K], int32 0..15

    # K 向 2:1 打包：偶索引 = lo，奇索引 = hi
    nibble_3d = tl.reshape(nibble, (BLOCK_M, BLOCK_K // 2, 2))
    lo_basis = tl.reshape(
        tl.broadcast_to(tl.reshape(tl.arange(0, 2) == 0, (1, 1, 2)), (BLOCK_M, BLOCK_K // 2, 2)).to(tl.int32),
        (BLOCK_M, BLOCK_K // 2, 2),
    )
    hi_basis = tl.reshape(
        tl.broadcast_to(tl.reshape(tl.arange(0, 2) == 1, (1, 1, 2)), (BLOCK_M, BLOCK_K // 2, 2)).to(tl.int32),
        (BLOCK_M, BLOCK_K // 2, 2),
    )
    lo_nibble = tl.sum(nibble_3d * lo_basis, axis=2)
    hi_nibble = tl.sum(nibble_3d * hi_basis, axis=2)

    packed = (lo_nibble & 0xF) | ((hi_nibble & 0xF) << 4)
    packed_u8 = packed.to(tl.uint8)

    offs_pk = k_start // 2 + tl.arange(0, BLOCK_K // 2)
    mask_pk = offs_pk < (K // 2)
    p_ptrs = A_packed_ptr + offs_m[:, None].to(tl.int64) * stride_pm + offs_pk[None, :].to(tl.int64) * stride_pk
    tl.store(p_ptrs, packed_u8, mask=mask_m[:, None] & mask_pk[None, :])


# ---------------------------------------------------------------------------
# ④ GEMM kernel：纯 dot_scaled MMA（A_packed/A_scale 预量化，W 主机侧重打包）
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _nvfp4_gemm_kernel(
    A_packed_ptr, A_scale_ptr, W_packed_ptr, W_scale_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_wsn, stride_wsk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # GROUP_M swizzle
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    GROUPS_K: tl.constexpr = BLOCK_K // 32

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    offs_k_scale = tl.arange(0, GROUPS_K)

    for k_start in range(0, K, BLOCK_K):
        k_packed_base = k_start // 2
        k_scale_base = k_start // 32

        cur_offs_kp = k_packed_base + offs_k_packed
        cur_offs_ks = k_scale_base + offs_k_scale

        mask_kp = cur_offs_kp < (K // 2)
        mask_ks = cur_offs_ks < (K // 32)

        # lhs: A_packed [M, K//2]，load [BLOCK_M, BLOCK_K//2]
        lhs_ptrs = A_packed_ptr + offs_m[:, None].to(tl.int64) * stride_am + cur_offs_kp[None, :].to(tl.int64) * stride_ak
        lhs_packed = tl.load(lhs_ptrs, mask=mask_m[:, None] & mask_kp[None, :], other=0, eviction_policy='evict_last')

        # lhs_scale: A_scale [M, K//32]，load [BLOCK_M, GROUPS_K]（uint8 e8m0）
        ls_ptrs = A_scale_ptr + offs_m[:, None].to(tl.int64) * (K // 32) + cur_offs_ks[None, :].to(tl.int64)
        lhs_scale = tl.load(ls_ptrs, mask=mask_m[:, None] & mask_ks[None, :], other=0, eviction_policy='evict_last')

        # rhs: W_packed_rhs [K//2, N]（K 向打包，rhs_k_pack=True），load [BLOCK_K//2, BLOCK_N]
        rhs_ptrs = W_packed_ptr + cur_offs_kp[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
        rhs_packed = tl.load(rhs_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0, eviction_policy='evict_first')

        # rhs_scale: W_scale_rhs [N, K//32]（不 trans），load [BLOCK_N, GROUPS_K]（uint8 e8m0）
        rs_ptrs = W_scale_ptr + offs_n[:, None].to(tl.int64) * stride_wsn + cur_offs_ks[None, :].to(tl.int64) * stride_wsk
        rhs_scale = tl.load(rs_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=0, eviction_policy='evict_first')

        # 原生 FP4 MMA（Triton 3.6 实测签名）
        acc = tl.dot_scaled(
            lhs_packed, lhs_scale, 'e2m1',
            rhs_packed, rhs_scale, 'e2m1',
            acc=acc,
            rhs_k_pack=True,
            out_dtype=tl.float32,
        )

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n.to(tl.int64), mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# 主机侧：W 重打包 [K,N//2] → [K//2,N]（K 向 2:1，供 rhs_k_pack=True）
# ---------------------------------------------------------------------------
def _repack_w_for_rhs_k_pack(W_packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    w_int = W_packed.to(torch.int32)
    w_lo = w_int & 0x0F           # [K, N//2] even cols（低半字节）
    w_hi = (w_int >> 4) & 0x0F    # [K, N//2] odd cols（高半字节）
    W_nibbles = torch.stack([w_lo, w_hi], dim=-1).reshape(K, N)  # [K, N] int32
    # 沿 K 重打包：偶 K 行 → lo，奇 K 行 → hi
    w_even = W_nibbles[0::2, :]   # [K//2, N]
    w_odd = W_nibbles[1::2, :]    # [K//2, N]
    packed_k = ((w_even & 0x0F) | ((w_odd & 0x0F) << 4)).to(torch.uint8)  # [K//2, N]
    return packed_k.contiguous()


# ---------------------------------------------------------------------------
# 主机侧：W_scale [K//32,N//128] → [N,K//32]（每 128 N 行共享一行 scale，uint8 不 trans）
# ---------------------------------------------------------------------------
def _expand_w_scale(W_scale: torch.Tensor, K: int, N: int) -> torch.Tensor:
    Gk = K // 32
    Gn = N // 128
    assert W_scale.shape == (Gk, Gn), f"Expected W_scale shape ({Gk}, {Gn}), got {W_scale.shape}"
    w_scale_expanded = W_scale.repeat_interleave(128, dim=1)  # [K//32, N]
    w_scale_rhs = w_scale_expanded.t().contiguous()           # [N, K//32]
    return w_scale_rhs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert A.is_cuda, "A must be on CUDA"
    assert A.dtype == torch.float32, "A must be fp32"
    assert W_packed.dtype == torch.uint8, "W_packed must be uint8"
    assert W_scale.dtype == torch.uint8, "W_scale must be uint8"

    M, K = A.shape
    K_w, N_half = W_packed.shape
    N = N_half * 2
    assert K == K_w, f"K mismatch: A has K={K}, W has K_w={K_w}"
    assert K % 32 == 0, "K must be divisible by 32"
    assert N % 128 == 0, "N must be divisible by 128 (W_scale block constraint)"
    assert M % 32 == 0, "M must be divisible by 32"

    device = A.device

    # ① A 量化（分离 kernel）
    A_packed = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    A_scale = torch.empty((M, K // 32), dtype=torch.uint8, device=device)

    QUANT_BLOCK_M = 32
    QUANT_BLOCK_K = 64

    grid_quant = (triton.cdiv(M, QUANT_BLOCK_M), triton.cdiv(K, QUANT_BLOCK_K))
    _quantize_fp32_to_nvfp4_packed[grid_quant](
        A, A_packed, A_scale,
        M, K,
        A.stride(0), A.stride(1),
        A_packed.stride(0), A_packed.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        BLOCK_M=QUANT_BLOCK_M,
        BLOCK_K=QUANT_BLOCK_K,
    )

    # ② ③ 主机侧 W 重打包 / scale 展开（每层一次，可缓存）
    W_packed_rhs = _repack_w_for_rhs_k_pack(W_packed, K, N)   # [K//2, N] uint8
    W_scale_rhs = _expand_w_scale(W_scale, K, N)              # [N, K//32] uint8

    C = torch.empty((M, N), dtype=torch.float32, device=device)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _nvfp4_gemm_kernel[grid](
        A_packed, A_scale,
        W_packed_rhs, W_scale_rhs,
        C,
        bias if bias is not None else A,
        M, N, K,
        A_packed.stride(0), A_packed.stride(1),
        W_packed_rhs.stride(0), W_packed_rhs.stride(1),
        W_scale_rhs.stride(0), W_scale_rhs.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return C
