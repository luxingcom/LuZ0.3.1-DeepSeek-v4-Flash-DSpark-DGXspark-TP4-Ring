# ============================================================================
# nvfp4_4w4a_prefill_gemm_v16.1_triton.py —— NVFP4 4W4A prefill GEMM（v16.1 正式版）
# 来源：MCP autotune_kernel（job 23a90d27-e524-47a8-9894-ded97838e452, v8）
#   8 轮正确性全过（v1→v8 连续 passed）；性能待生产 benchmark（MCP harness 限制）
# 两处修正（vs MCP v8）：
#   ① D1（v15 轮生产实测）：W_packed 为 **N 向打包 [K, N//2]**（低半字节=偶 N 列），
#      生成器已正确实现（stack dim=-1 沿 N 交错）——直喂生产转换器格式即可
#   ② scale 语义（人工核对）：A 归一化解码 **去掉 /6.0**——e8m0 编码本就是
#      floor(log2(max/6))+127，解码 2^(raw-127)≈max/6 直接归一化（≤6），
#      MMA 乘回 scale 还原；MCP v8 多除 6 导致输出偏大 6 倍（verify total_tests=0
#      未做数值比对抓不出，生产 pytest 数值比对为终审）
# 核心思路（danielwoz 借鉴）：E2M1 幅值 ⊂ e4m3 无损 → fp8 scaled MMA
# ============================================================================
import torch
import triton
import triton.language as tl
from typing import Optional

# E2M1 幅值 → fp8 e4m3 位模式（0x00=0, 0x30=0.5, 0x38=1.0, 0x3C=1.5,
# 0x40=2.0, 0x44=3.0, 0x48=4.0, 0x4C=6.0 —— e4m3 中精确可表示，无损）
_E2M1_TO_FP8E4M3_TABLE = torch.tensor(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C],
    dtype=torch.uint8,
)

_weight_cache: dict = {}


def _prepare_weights(W_packed: torch.Tensor, W_scale: torch.Tensor, layer_id: int):
    """W 解包（N 向）→ fp8 e4m3 幅值 [K,N] + rhs_scale [N, K//32]（每层一次缓存）。

    W_packed [K, N//2]：低半字节=偶 N 列；W_scale [K//32, N//128]（N 块 128 共享）。
    """
    if layer_id in _weight_cache:
        return _weight_cache[layer_id]

    K, N_half = W_packed.shape
    N = N_half * 2

    W_u8 = W_packed.to(torch.uint8)
    W_lo = (W_u8 & 0x0F).to(torch.int32)                    # [K, N//2] 偶 N 列
    W_hi = ((W_u8 >> 4) & 0x0F).to(torch.int32)             # [K, N//2] 奇 N 列
    W_nibbles = torch.stack([W_lo, W_hi], dim=-1).reshape(K, N)  # 沿 N 交错 [K, N]

    W_sign_bit = (W_nibbles >> 3) & 0x1
    W_mag_idx = W_nibbles & 0x7

    tbl = _E2M1_TO_FP8E4M3_TABLE.to(W_packed.device)
    W_fp8_mag = tbl[W_mag_idx]
    W_fp8 = (W_fp8_mag | (W_sign_bit.to(torch.uint8) << 7)).to(torch.uint8)  # [K, N] fp8 e4m3

    Ws = W_scale
    if Ws.dtype != torch.uint8:
        Ws_f = Ws.float()
        Ws_safe = Ws_f.abs().clamp(min=1e-38)
        Ws_e8m0 = (torch.floor(torch.log2(Ws_safe * 6.0)) + 127.0).clamp(0, 255).to(torch.uint8)
    else:
        Ws_e8m0 = Ws                                    # [K//32, N//128] uint8

    # W_scale [K//32, N//128] -> [K//32, N]（N 块 128 展开）-> transpose [N, K//32]
    Ws_expanded = Ws_e8m0.repeat_interleave(128, dim=1)  # [K//32, N]
    Ws_rhs = Ws_expanded.t().contiguous()                # [N, K//32]（不 trans 的 rhs_scale 布局）

    result = (W_fp8.contiguous(), Ws_rhs)
    _weight_cache[layer_id] = result
    return result


def preprocess_weights_clear() -> None:
    """权重更新时调用（vLLM process_weights_after_loading 集成）。"""
    _weight_cache.clear()


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _nvfp4_gemm_kernel(
    A_ptr,
    W_ptr,
    Ws_ptr,
    C_ptr,
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_wsn, stride_wsk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    GROUPS_K: tl.constexpr = BLOCK_K // 32
    BIAS127: tl.constexpr = tl.constexpr(127)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    K_groups = K // 32

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = offs_k < K

        # ---- A tile [BLOCK_M, BLOCK_K] ----
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_f32 = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0,
                        eviction_policy='evict_last').to(tl.float32)

        # ---- A 量化（32 分组 E8M0 + E2M1 阈值链 → fp8 e4m3 位模式）----
        a_grouped = tl.reshape(a_f32, (BLOCK_M, GROUPS_K, 32))
        a_max = tl.max(tl.abs(a_grouped), axis=2)                 # [BM, GROUPS_K]
        a_max_safe = tl.maximum(a_max, 1e-38)

        a_log2 = tl.log2(a_max_safe / 6.0)                        # e8m0 编码（含 /6）
        a_exp_f = tl.floor(a_log2) + 127.0
        a_exp_clamped = tl.minimum(tl.maximum(a_exp_f, 0.0), 255.0)
        a_scale_raw = a_exp_clamped.to(tl.int32)                  # [BM, GROUPS_K]

        lhs_scale = a_scale_raw.to(tl.uint8)                      # uint8 e8m0 字节

        # ★修正：解码 2^(raw-127)【去掉 /6.0】——归一化到 ≤6 后量化，MMA 乘回还原
        a_scale_dec = tl.exp2((a_scale_raw - BIAS127).to(tl.float32))  # [BM, GROUPS_K]

        a_scale_bc = tl.broadcast_to(
            tl.reshape(a_scale_dec, (BLOCK_M, GROUPS_K, 1)),
            (BLOCK_M, GROUPS_K, 32)
        )
        a_scale_flat = tl.reshape(a_scale_bc, (BLOCK_M, BLOCK_K))

        a_norm = a_f32 / tl.maximum(a_scale_flat, 1e-38)

        a_abs_norm = tl.abs(a_norm)
        a_sign_neg = a_norm < 0.0

        mag_idx = (
            (a_abs_norm > 0.25).to(tl.int32) +
            (a_abs_norm > 0.75).to(tl.int32) +
            (a_abs_norm > 1.25).to(tl.int32) +
            (a_abs_norm > 1.75).to(tl.int32) +
            (a_abs_norm > 2.5).to(tl.int32)  +
            (a_abs_norm > 3.5).to(tl.int32)  +
            (a_abs_norm > 5.0).to(tl.int32)
        )

        fp8_mag = tl.where(mag_idx == 0, 0x00,
                  tl.where(mag_idx == 1, 0x30,
                  tl.where(mag_idx == 2, 0x38,
                  tl.where(mag_idx == 3, 0x3C,
                  tl.where(mag_idx == 4, 0x40,
                  tl.where(mag_idx == 5, 0x44,
                  tl.where(mag_idx == 6, 0x48, 0x4C)))))))

        sign_bits = tl.where(a_sign_neg, 0x80, 0x00).to(tl.int32)
        a_fp8_int = (fp8_mag | sign_bits).to(tl.uint8)            # fp8 e4m3 幅值+符号

        # ---- W tile [BLOCK_K, BLOCK_N]（主机侧已展开缓存）----
        W_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_fp8 = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0,
                        eviction_policy='evict_first').to(tl.uint8)

        # ---- rhs_scale [BLOCK_N, GROUPS_K]（Ws_rhs [N, K//32]，不 trans）----
        k_group_start = k_start // 32
        offs_kg = tl.arange(0, GROUPS_K).to(tl.int64) + k_group_start
        mask_kg = offs_kg < K_groups
        Ws_ptrs = Ws_ptr + (offs_n[:, None] * stride_wsn + offs_kg[None, :] * stride_wsk)
        rhs_scale_i32 = tl.load(Ws_ptrs, mask=mask_n[:, None] & mask_kg[None, :], other=127)
        rhs_scale = rhs_scale_i32.to(tl.uint8)

        # ---- FP8 scaled MMA（e4m3 × e4m3，原生路径，无 e2m1 降级）----
        acc = tl.dot_scaled(
            a_fp8_int, lhs_scale,
            w_fp8,    rhs_scale,
            acc,
            lhs_format='e4m3', rhs_format='e4m3',
            rhs_k_pack=False,
        )

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = mask_m[:, None] & mask_n[None, :]

    out = acc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = out + bias[None, :]

    tl.store(C_ptrs, out, mask=mask_c)


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 4W4A prefill GEMM（v16.1，FP8 scaled MMA）。

    A: [M,K] fp32/bf16；W_packed: [K,N//2] uint8（**N 向打包**，低半字节=偶 N 列）；
    W_scale: [K//32,N//128] uint8；bias: [N] 可选。→ C: [M,N] fp32。
    约束：K%32==0, N%128==0, M%32==0。
    """
    assert A.is_contiguous(), "A must be contiguous"
    M, K = A.shape
    K2, N_half = W_packed.shape
    assert K2 == K, f"K mismatch: A={K}, W_packed={K2}"
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    N = N_half * 2
    assert N % 128 == 0, f"N={N} must be divisible by 128"

    device = A.device

    layer_id = id(W_packed)
    W_fp8_kn, Ws_rhs = _prepare_weights(W_packed, W_scale, layer_id)

    C = torch.empty((M, N), dtype=torch.float32, device=device)

    has_bias = bias is not None
    bias_ptr = bias.contiguous() if has_bias else A

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    _nvfp4_gemm_kernel[grid](
        A, W_fp8_kn, Ws_rhs, C,
        bias_ptr,
        M, N, K,
        A.stride(0), A.stride(1),
        W_fp8_kn.stride(0), W_fp8_kn.stride(1),
        Ws_rhs.stride(0), Ws_rhs.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=has_bias,
    )

    return C
