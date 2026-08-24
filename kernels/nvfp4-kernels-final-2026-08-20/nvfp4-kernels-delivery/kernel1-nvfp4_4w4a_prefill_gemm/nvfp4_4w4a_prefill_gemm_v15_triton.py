# ============================================================================
# nvfp4_4w4a_prefill_gemm_v15_triton.py —— NVFP4 4W4A prefill GEMM（v15）
# 来源：MCP autotune_kernel（job 84c1375a-a433-4740-a49d-7a6e76b573c6, v3）
#   服务端 GPU 实测 speedup = 8.96×（目标 2.0）
# 架构（SASS 实锤后的 Triton 3.6 最优路径）：
#   ① A 量化独立 kernel（32 分组 E8M0，strict > 阈值链，含 /6）
#   ② 主机侧 W 反量化 + 缓存（每层一次）
#   ③ bf16 MMA kernel（tl.dot，allow_tf32=False）——Triton 3.6 的 dot_scaled(e2m1)
#      降级为 BF16 HMMA，此路径即其可达最优；400 TFLOPS 需 CUTLASS mmaf_scaled
# 适配（vs MCP v3）：W_scale 输入兼容 [K//32, N//128]（我们的转换器/checkpoint 格式）
# ============================================================================
import torch
import triton
import triton.language as tl
from typing import Optional

# Module-level cache for preprocessed weights
_weight_cache = {}


@triton.jit
def _quantize_activation_kernel(
    A_ptr,
    A_quant_ptr,
    A_scale_ptr,
    M, K,
    GROUPS_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """A fp32 [M,K] → 反量化 fp32 [M,K]（4W4A 语义）+ E8M0 scale [M, K/32]。

    scale = floor(log2(max_abs/6)) + 127 clamp [0,255]；阈值链 strict >（tie low）。
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    group_k_start = pid_g * GROUP_SIZE

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = group_k_start + tl.arange(0, GROUP_SIZE)

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    K_i64 = tl.cast(K, tl.int64)
    offs_m_i64 = tl.cast(offs_m, tl.int64)
    offs_k_i64 = tl.cast(offs_k, tl.int64)

    a = tl.load(
        A_ptr + offs_m_i64[:, None] * K_i64 + offs_k_i64[None, :],
        mask=mask, other=0.0,
        eviction_policy='evict_first'
    ).to(tl.float32)

    a_abs = tl.abs(a)
    max_abs = tl.max(a_abs, axis=1)  # [BLOCK_M]

    safe_max = tl.maximum(max_abs, 1e-38)
    log2_val = tl.log2(safe_max / 6.0)
    floor_log2 = tl.floor(log2_val)
    scale_byte_f = floor_log2 + 127.0
    scale_byte_f = tl.maximum(scale_byte_f, 0.0)
    scale_byte_f = tl.minimum(scale_byte_f, 255.0)
    scale_byte = scale_byte_f.to(tl.uint8)

    scale_exp = scale_byte.to(tl.float32) - 127.0
    scale_f = tl.exp2(scale_exp)  # [BLOCK_M]

    ones_k = tl.full([GROUP_SIZE], 1.0, dtype=tl.float32)
    scale_expanded = scale_f[:, None] * ones_k[None, :]  # [BLOCK_M, GROUP_SIZE]

    safe_scale = tl.maximum(scale_expanded, 1e-38)
    a_scaled = a / safe_scale

    a_abs_scaled = tl.abs(a_scaled)

    # 阈值链（strict >，tie low，与 torch argmin 逐字节一致）
    idx = tl.zeros([BLOCK_M, GROUP_SIZE], dtype=tl.int32)
    idx = idx + (a_abs_scaled > 0.25).to(tl.int32)
    idx = idx + (a_abs_scaled > 0.75).to(tl.int32)
    idx = idx + (a_abs_scaled > 1.25).to(tl.int32)
    idx = idx + (a_abs_scaled > 1.75).to(tl.int32)
    idx = idx + (a_abs_scaled > 2.5).to(tl.int32)
    idx = idx + (a_abs_scaled > 3.5).to(tl.int32)
    idx = idx + (a_abs_scaled > 5.0).to(tl.int32)

    quant_mag = tl.where(idx == 0, 0.0,
                tl.where(idx == 1, 0.5,
                tl.where(idx == 2, 1.0,
                tl.where(idx == 3, 1.5,
                tl.where(idx == 4, 2.0,
                tl.where(idx == 5, 3.0,
                tl.where(idx == 6, 4.0, 6.0)))))))

    a_sign = tl.where(a_scaled >= 0.0, 1.0, -1.0)
    a_quant_f = quant_mag * a_sign

    # 反量化回原值域（4W4A 语义的 fp32 表示）
    a_dequant = a_quant_f * safe_scale

    tl.store(
        A_quant_ptr + offs_m_i64[:, None] * K_i64 + offs_k_i64[None, :],
        a_dequant,
        mask=mask,
        eviction_policy='evict_first'
    )

    GROUPS_K_i64 = tl.cast(GROUPS_K, tl.int64)
    pid_g_i64 = tl.cast(pid_g, tl.int64)
    scale_offs = offs_m_i64 * GROUPS_K_i64 + pid_g_i64
    tl.store(A_scale_ptr + scale_offs, scale_byte, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _bf16_gemm_kernel(
    A_ptr,
    W_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """bf16 MMA：A_bf16 [M,K] × W_bf16 [N,K]ᵀ → C fp32 [M,N]。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.int64)
    offs_k = tl.cast(tl.arange(0, BLOCK_K), tl.int64)

    stride_am_i64 = tl.cast(stride_am, tl.int64)
    stride_ak_i64 = tl.cast(stride_ak, tl.int64)
    stride_wn_i64 = tl.cast(stride_wn, tl.int64)
    stride_wk_i64 = tl.cast(stride_wk, tl.int64)
    stride_cm_i64 = tl.cast(stride_cm, tl.int64)
    stride_cn_i64 = tl.cast(stride_cn, tl.int64)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_i64 = tl.cast(K, tl.int64)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = tl.cast(k * BLOCK_K, tl.int64) + offs_k
        mask_k = k_offs < K_i64

        a_ptrs = A_ptr + offs_m[:, None] * stride_am_i64 + k_offs[None, :] * stride_ak_i64
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy='evict_last'
        ).to(tl.bfloat16)

        w_ptrs = W_ptr + offs_n[:, None] * stride_wn_i64 + k_offs[None, :] * stride_wk_i64
        w = tl.load(
            w_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy='evict_last'
        ).to(tl.bfloat16)

        acc = tl.dot(a, tl.trans(w), acc=acc, allow_tf32=False)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm_i64 + offs_n[None, :] * stride_cn_i64
    tl.store(
        c_ptrs,
        acc.to(tl.float32),
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ---- 主机侧工具 ----
def _e2m1_to_float_host(x: torch.Tensor) -> torch.Tensor:
    codebook = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=x.device)
    return codebook[x.long()]


def _unpack_fp4_weights(w_packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """W_packed [K//2, N]（低半字节=第1个K元素）→ fp32 [N, K]。"""
    lo = w_packed & 0x0F
    hi = (w_packed >> 4) & 0x0F
    lo_f = _e2m1_to_float_host(lo)
    hi_f = _e2m1_to_float_host(hi)
    w_float = torch.stack([lo_f, hi_f], dim=1).reshape(K, N)  # [K, N]
    return w_float.t().contiguous()                            # [N, K]


def _e8m0_to_float_host(scale_bytes: torch.Tensor) -> torch.Tensor:
    exp = scale_bytes.to(torch.float32) - 127.0
    return torch.pow(2.0, exp)


def _preprocess_weights(w_packed: torch.Tensor, w_scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """W 反量化（每层一次，可缓存）→ [N, K] fp32。

    兼容两种 W_scale 输入：
      - [K//32, N//128]（转换器/checkpoint 格式，N 方向 128 块共享）
      - [N, K//32]（每 N 行全展开）
    """
    key = w_packed.data_ptr()
    if key in _weight_cache:
        return _weight_cache[key]

    w_float = _unpack_fp4_weights(w_packed, N, K)  # [N, K]

    if w_scale.shape == (K // 32, N // 128):
        # block scale → 每 N 行 128 块共享 → [N, K//32]
        w_scale_f = _e8m0_to_float_host(w_scale.repeat_interleave(128, dim=1).t().contiguous())
    else:
        w_scale_f = _e8m0_to_float_host(w_scale)   # [N, K//32]

    w_scale_expanded = w_scale_f.unsqueeze(2).expand(N, K // 32, 32).reshape(N, K)
    w_dequant = (w_float * w_scale_expanded).contiguous()  # [N, K]

    _weight_cache[key] = w_dequant
    return w_dequant


def preprocess_weights_clear() -> None:
    """权重更新时调用（vLLM 换层/换权重）。"""
    _weight_cache.clear()


def _quantize_activation_triton(A: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    GROUPS_K = K // 32
    GROUP_SIZE = 32

    A_quant = torch.empty((M, K), dtype=torch.float32, device=A.device)
    A_scale = torch.empty((M, GROUPS_K), dtype=torch.uint8, device=A.device)

    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), GROUPS_K)

    _quantize_activation_kernel[grid](
        A, A_quant, A_scale,
        M, K,
        GROUPS_K=GROUPS_K,
        BLOCK_M=BLOCK_M,
        GROUP_SIZE=GROUP_SIZE,
    )
    return A_quant


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 4W4A prefill GEMM：C[M,N] = fp32_accum(bf16(A_q), bf16(W)) + bias。

    A: [M,K] fp32/bf16；W_packed: [K//2,N] uint8；W_scale: [K//32,N//128] 或 [N,K//32] uint8。
    """
    A = A.float()
    M, K = A.shape
    N = W_packed.shape[1]

    W_dequant = _preprocess_weights(W_packed, W_scale, N, K)      # [N, K] fp32（缓存）
    A_dequant = _quantize_activation_triton(A)                     # [M, K] fp32（4W4A）

    A_bf16 = A_dequant.to(torch.bfloat16).contiguous()
    W_bf16 = W_dequant.to(torch.bfloat16).contiguous()

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _bf16_gemm_kernel[grid](
        A_bf16, W_bf16, C,
        M, N, K,
        A_bf16.stride(0), A_bf16.stride(1),
        W_bf16.stride(0), W_bf16.stride(1),
        C.stride(0), C.stride(1),
    )

    if bias is not None:
        C = C + bias.float()
    return C
