import torch
import triton
import triton.language as tl
from typing import Optional

# ============================================================================
# nvfp4_4w4a_prefill_gemm —— v10（生产实机修复版）
# 环境：torch 2.11 / Triton 3.6 / CC=(12,1)=sm_121a（DGX Spark GB10）
#
# 针对生产诊断缺陷 #7 的修复（Triton 3.6 dot_scaled 实测签名）：
#   tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
#                 acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=fp32)
#   - lhs_format / rhs_format 必填【位置参数】（'e2m1'）；无 lhs_type/rhs_type 关键字
#   - e8m0 scale group = 32 沿 K（硬性）：lhs_scale [M, K//32]、rhs_scale [N, K//32]（不转置）
#   - rhs_k_pack=True（SM120/121 必需，Triton issue #9678）
#   - lhs [M,K] / rhs [N,K]，e2m1 数据 2:1 打包（每字节 2 元素，K 方向）
#
# 语义变更：A/W scale 分组 16 → 32（对齐 Blackwell mmaf_scaled 原生 K 块粒度）。
# 接口保持 4 参 (A, W_packed, W_scale, bias)，与转换器/test 脚手架一致。
# ============================================================================

_E8M0_BIAS: tl.constexpr = tl.constexpr(127)
_SCALE_K: tl.constexpr = tl.constexpr(32)    # e8m0 scale group 沿 K（硬件/Triton 硬约束）
_SCALE_N: tl.constexpr = tl.constexpr(128)   # NVFP4 W_scale 块沿 N（checkpoint 布局）


@triton.autotune(
    configs=[
        # Triton 3.6.0 sm121：uint8 scale 可编译；BLOCK_N 必须 ≥128 且为 128 倍数（对齐 W_scale N//128 块）
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _nvfp4_4w4a_prefill_kernel(
    A_ptr, W_packed_ptr, W_scale_ptr, bias_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_ws_k, stride_ws_n,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # GROUP_M swizzle：同 N 列的 M 块聚集，提升 W/W_scale 的 L2 复用
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    GROUPS_K: tl.constexpr = BLOCK_K // _SCALE_K

    # Precompute base pointers / 循环不变式
    a_base = A_ptr + offs_m[:, None].to(tl.int64) * stride_am
    w_n_half_base = pid_n * (BLOCK_N // 2)
    w_n_offs = w_n_half_base + tl.arange(0, BLOCK_N // 2)
    n_block = offs_n // _SCALE_N

    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # ---------- A：load（streaming）+ 32 分组量化打包 + lhs_scale ----------
        a_ptrs = a_base + k_offs[None, :].to(tl.int64) * stride_ak
        a_block = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0,
                          eviction_policy="evict_first").to(tl.float32)

        a_g = tl.reshape(a_block, (BLOCK_M, GROUPS_K, _SCALE_K))
        a_absmax = tl.max(tl.abs(a_g), axis=2)                      # [BM, GROUPS_K]
        scale_val = tl.maximum(a_absmax / 6.0, 1e-38).to(tl.float32)
        scale_bits = scale_val.to(tl.int32, bitcast=True)           # fp32 bitcast（先显式回落 fp32）
        a_exp = (scale_bits >> 23) - _E8M0_BIAS
        # e8m0 编码字节（uint8，直接传给 dot_scaled；Triton 3.6 sm121 对 fp32 scale codegen 有 bug）
        lhs_scale_u8 = tl.minimum(tl.maximum(a_exp + _E8M0_BIAS, 0), 255).to(tl.uint8)  # [BM, GROUPS_K]

        a_norm = a_g / (tl.reshape(tl.exp2((lhs_scale_u8.to(tl.int32) - _E8M0_BIAS).to(tl.float32)), (BLOCK_M, GROUPS_K, 1)) + 1e-38)
        a_norm = tl.minimum(tl.maximum(a_norm, -6.0), 6.0)
        a_abs = tl.abs(a_norm)
        a_sign = tl.where(a_norm < 0.0, 8, 0).to(tl.int32)

        # 最近 E2M1 幅值索引（阈值链）
        mag = tl.zeros((BLOCK_M, GROUPS_K, _SCALE_K), dtype=tl.int32)
        mag = tl.where(a_abs >= 0.25, 1, mag)
        mag = tl.where(a_abs >= 0.75, 2, mag)
        mag = tl.where(a_abs >= 1.25, 3, mag)
        mag = tl.where(a_abs >= 1.75, 4, mag)
        mag = tl.where(a_abs >= 2.50, 5, mag)
        mag = tl.where(a_abs >= 3.50, 6, mag)
        mag = tl.where(a_abs >= 5.00, 7, mag)

        nibble = (a_sign | mag).to(tl.uint8)                        # [BM, GROUPS_K, 32]
        nibble_flat = tl.reshape(nibble, (BLOCK_M, BLOCK_K))
        nib2 = tl.reshape(nibble_flat, (BLOCK_M, BLOCK_K // 2, 2))  # 末维 2：idx0=偶(low), idx1=奇(high)
        nib_split = tl.split(nib2)                                  # Triton 3.6 单参 split
        lhs_packed = (nib_split[0] | (nib_split[1] << 4)).to(tl.uint8)  # [BM, BK//2] K 方向 2:1

        # ---------- W：load（L2 复用）→ 解包 → trans → K 方向重打包（rhs_k_pack） ----------
        w_ptrs = W_packed_ptr + k_offs[:, None].to(tl.int64) * stride_wk + w_n_offs[None, :].to(tl.int64) * stride_wn
        w_mask = (k_offs[:, None] < K) & (w_n_offs[None, :] < (N // 2))
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0, eviction_policy="evict_last").to(tl.int32)

        w_lo = w_packed & 0xF                                       # 低半字节 = 第 1 元素
        w_hi = (w_packed >> 4) & 0xF                                # 高半字节 = 第 2 元素
        w_lo_r = tl.reshape(w_lo, (BLOCK_K, BLOCK_N // 2, 1))
        w_hi_r = tl.reshape(w_hi, (BLOCK_K, BLOCK_N // 2, 1))
        pair_sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.float32), (1, 1, 2))
        w_pair = w_lo_r * pair_sel + w_hi_r * (1.0 - pair_sel)      # [BK, BN//2, 2]
        w_full = tl.reshape(w_pair, (BLOCK_K, BLOCK_N)).to(tl.int32)  # 列交替 lo/hi → [BK, BN]

        # rhs 必须为 [K, N] 布局（Triton 3.6 dot_scaled: K_RHS,N = rhs.shape[-2:]）
        # w_full [BK, BN] 已是 [K, N]；沿 K 方向 2:1 打包 -> [BK//2, BN]
        # tl.split 沿末维(须=2)：reshape(BK//2, 2, BN) -> trans(0,2,1) 成 (BK//2, BN, 2)
        w_t3 = tl.reshape(w_full, (BLOCK_K // 2, 2, BLOCK_N))
        w_t3t = tl.trans(w_t3, 0, 2, 1)                         # [BLOCK_K//2, BLOCK_N, 2]
        w_t_split = tl.split(w_t3t)
        rhs_packed = (w_t_split[0] | (w_t_split[1] << 4)).to(tl.uint8)  # [BK//2, BN] K 方向 2:1

        # ---------- rhs_scale：W_scale [K//32, N//128] → 直接展开 [BN, BK//32]（每 128 N 行共享一行）
        # 注意：不 tl.trans（Triton 3.6 sm121 对 trans 后的 scale 有 codegen bug）；直接 load 成 [BN, GROUPS_K]
        n_groups = n_block.to(tl.int64)                    # [BN]
        kg_local = (k_start // _SCALE_K) + tl.arange(0, GROUPS_K)   # [GROUPS_K]
        ws2_ptrs = W_scale_ptr + (
            n_groups[:, None].to(tl.int64) * stride_ws_n +
            kg_local[None, :].to(tl.int64) * stride_ws_k
        )
        ws2_mask = (n_groups[:, None] < (N // _SCALE_N)) & (kg_local[None, :] < (K // _SCALE_K))
        rhs_scale_u8 = tl.load(ws2_ptrs, mask=ws2_mask, other=127, eviction_policy="evict_last").to(tl.uint8)  # [BN, GROUPS_K] e8m0 原始字节

        # ---------- 原生 FP4 MMA（Triton 3.6 实测签名，位置参数） ----------
        acc = tl.dot_scaled(
            lhs_packed, lhs_scale_u8, "e2m1",
            rhs_packed, rhs_scale_u8, "e2m1",
            acc=acc,
            rhs_k_pack=True,
            out_dtype=tl.float32,
        )

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n.to(tl.int64), mask=n_mask, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    c_ptrs = C_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 4W4A prefill GEMM（原生 FP4 MMA 路径，Triton 3.6 dot_scaled 实测签名对齐）。

    Args:
        A: activations [M, K]（任意 dtype，内部转 fp32；内核内 32 分组量化）
        W_packed: NVFP4 packed weights [K, N//2] uint8（低半字节 = 第 1 元素）
        W_scale: E8M0 weight scales [K//32, N//128] uint8（每 (32,128) 块一个 scale）
        bias: optional [N] fp32

    Returns:
        C: [M, N] fp32
    """
    device = A.device if A.device.type != "cpu" else (
        torch.device("cuda", 0) if torch.cuda.is_available() else A.device
    )
    A_fp32 = A.to(dtype=torch.float32, device=device).contiguous()
    W_packed_dev = W_packed.to(device=device).contiguous()
    W_scale_dev = W_scale.to(device=device).contiguous()

    M, K = A_fp32.shape
    K2, N_half = W_packed_dev.shape
    N = N_half * 2
    assert K2 == K, f"Weight K={K2} must equal activation K={K}"
    assert K % 32 == 0, f"K must be divisible by 32 (e8m0 scale group), got {K}"
    assert N % 128 == 0, f"N must be divisible by 128 (NVFP4 block), got {N}"
    assert M % 32 == 0, f"M must be divisible by 32, got {M}"
    assert W_scale_dev.shape == (K // 32, N // 128), (
        f"W_scale must be [K//32={K // 32}, N//128={N // 128}], got {tuple(W_scale_dev.shape)}"
    )

    C = torch.empty((M, N), dtype=torch.float32, device=device)
    has_bias = bias is not None
    bias_tensor = bias.to(dtype=torch.float32, device=device).contiguous() if has_bias else C

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _nvfp4_4w4a_prefill_kernel[grid](
        A_fp32, W_packed_dev, W_scale_dev, bias_tensor, C,
        M=M, N=N, K=K,
        stride_am=A_fp32.stride(0), stride_ak=A_fp32.stride(1),
        stride_wk=W_packed_dev.stride(0), stride_wn=W_packed_dev.stride(1),
        stride_ws_k=W_scale_dev.stride(0), stride_ws_n=W_scale_dev.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        HAS_BIAS=has_bias,
    )
    return C
