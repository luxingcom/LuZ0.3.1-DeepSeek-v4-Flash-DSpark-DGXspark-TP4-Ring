import triton
import triton.language as tl
import torch

try:
    import triton.language.extra.cuda as tle
    HAS_TLE = True
except ImportError:
    try:
        import triton.experimental.tle.language as tle
        HAS_TLE = True
    except ImportError:
        HAS_TLE = False


@triton.autotune(
    configs=[
        triton.Config({'TOKENS_PER_PROG': 1}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 2}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 4}, num_warps=8, num_stages=3),
        triton.Config({'TOKENS_PER_PROG': 8}, num_warps=8, num_stages=4),
        triton.Config({'TOKENS_PER_PROG': 1}, num_warps=8, num_stages=3),
        triton.Config({'TOKENS_PER_PROG': 2}, num_warps=8, num_stages=3),
        triton.Config({'TOKENS_PER_PROG': 4}, num_warps=4, num_stages=3),
        triton.Config({'TOKENS_PER_PROG': 8}, num_warps=4, num_stages=4),
    ],
    key=['T', 'N_GROUPS'],
)
@triton.jit
def _nvfp4_ds_mla_kv_linear_kernel(
    kv_ptr,
    out_ptr,
    T,
    KV_STRIDE_T,
    OUT_STRIDE_T,
    N_GROUPS: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    token_start = pid_t * TOKENS_PER_PROG
    group_idx = pid_d

    lane_arange_16 = tl.arange(0, 16)
    lane_arange_8  = tl.arange(0, 8)

    for ti in tl.static_range(TOKENS_PER_PROG):
        tok = token_start + ti
        mask_tok = tok < T

        base_kv  = tok.to(tl.int64) * KV_STRIDE_T
        base_out = tok.to(tl.int64) * OUT_STRIDE_T

        # ---- process K group ----
        k_col_start = group_idx * 16
        k_offs = k_col_start + lane_arange_16
        k_vals = tl.load(
            kv_ptr + base_kv + k_offs,
            mask=mask_tok,
            other=0.0,
            eviction_policy='evict_first',
        )

        k_abs  = tl.abs(k_vals)
        k_amax = tl.max(k_abs, axis=0)
        k_amax = tl.maximum(k_amax, 1e-30)

        # NVFP4 E8M0 scale：scale = 2^floor(log2(max/6))，使 max/scale ∈ [1,6] 充分利用 E2M1 量程
        # （与生产 v5 / torch 参考逐字节一致的编码；floor 语义）
        k_log2      = tl.log2(k_amax / 6.0)
        k_scale_exp = tl.floor(k_log2).to(tl.int32)
        k_scale_exp = tl.maximum(tl.minimum(k_scale_exp, 127), -126)
        k_scale_float = tl.exp2(k_scale_exp.to(tl.float32))

        k_scaled  = k_vals / k_scale_float
        k_clamped = tl.minimum(tl.maximum(k_scaled, -6.0), 6.0)

        k_sign = (k_clamped < 0.0).to(tl.int32)
        k_abs2 = tl.abs(k_clamped)

        k_me = tl.zeros([16], dtype=tl.int32)
        k_me = tl.where(k_abs2 >= 0.25, 1, k_me)
        k_me = tl.where(k_abs2 >= 0.75, 2, k_me)
        k_me = tl.where(k_abs2 >= 1.25, 3, k_me)
        k_me = tl.where(k_abs2 >= 1.75, 4, k_me)
        k_me = tl.where(k_abs2 >= 2.5,  5, k_me)
        k_me = tl.where(k_abs2 >= 3.5,  6, k_me)
        k_me = tl.where(k_abs2 >= 5.0,  7, k_me)

        k_nibble = (k_sign * 8 + k_me).to(tl.uint8)

        # Pack nibbles using reshape to avoid scalar indexing
        k_nibble_2d = tl.reshape(k_nibble, [8, 2])
        col0_mask = tl.arange(0, 2) == 0  # [2] boolean
        col1_mask = tl.arange(0, 2) == 1  # [2] boolean

        k_lo_2d = tl.where(tl.broadcast_to(col0_mask[None, :], [8, 2]), k_nibble_2d, tl.zeros([8, 2], dtype=tl.uint8))
        k_hi_2d = tl.where(tl.broadcast_to(col1_mask[None, :], [8, 2]), k_nibble_2d, tl.zeros([8, 2], dtype=tl.uint8))

        k_lo_nibbles = tl.sum(k_lo_2d, axis=1).to(tl.uint8)
        k_hi_nibbles = tl.sum(k_hi_2d, axis=1).to(tl.uint8)

        k_packed = (k_lo_nibbles & 0x0F) | ((k_hi_nibbles & 0x0F) << 4)

        k_packed_offs = group_idx * 8 + lane_arange_8
        tl.store(out_ptr + base_out + k_packed_offs, k_packed, mask=mask_tok)

        k_scale_byte = tl.maximum(tl.minimum(k_scale_exp + 127, 255), 0).to(tl.uint8)
        tl.store(out_ptr + base_out + 512 + group_idx, k_scale_byte, mask=mask_tok)

        # ---- process V group ----
        v_col_start = 512 + group_idx * 16
        v_offs = v_col_start + lane_arange_16
        v_vals = tl.load(
            kv_ptr + base_kv + v_offs,
            mask=mask_tok,
            other=0.0,
            eviction_policy='evict_first',
        )

        v_abs  = tl.abs(v_vals)
        v_amax = tl.max(v_abs, axis=0)
        v_amax = tl.maximum(v_amax, 1e-30)

        v_log2      = tl.log2(v_amax / 6.0)
        v_scale_exp = tl.floor(v_log2).to(tl.int32)
        v_scale_exp = tl.maximum(tl.minimum(v_scale_exp, 127), -126)
        v_scale_float = tl.exp2(v_scale_exp.to(tl.float32))

        v_scaled  = v_vals / v_scale_float
        v_clamped = tl.minimum(tl.maximum(v_scaled, -6.0), 6.0)

        v_sign = (v_clamped < 0.0).to(tl.int32)
        v_abs2 = tl.abs(v_clamped)

        v_me = tl.zeros([16], dtype=tl.int32)
        v_me = tl.where(v_abs2 >= 0.25, 1, v_me)
        v_me = tl.where(v_abs2 >= 0.75, 2, v_me)
        v_me = tl.where(v_abs2 >= 1.25, 3, v_me)
        v_me = tl.where(v_abs2 >= 1.75, 4, v_me)
        v_me = tl.where(v_abs2 >= 2.5,  5, v_me)
        v_me = tl.where(v_abs2 >= 3.5,  6, v_me)
        v_me = tl.where(v_abs2 >= 5.0,  7, v_me)

        v_nibble = (v_sign * 8 + v_me).to(tl.uint8)

        v_nibble_2d = tl.reshape(v_nibble, [8, 2])
        v_lo_2d = tl.where(tl.broadcast_to(col0_mask[None, :], [8, 2]), v_nibble_2d, tl.zeros([8, 2], dtype=tl.uint8))
        v_hi_2d = tl.where(tl.broadcast_to(col1_mask[None, :], [8, 2]), v_nibble_2d, tl.zeros([8, 2], dtype=tl.uint8))

        v_lo_nibbles = tl.sum(v_lo_2d, axis=1).to(tl.uint8)
        v_hi_nibbles = tl.sum(v_hi_2d, axis=1).to(tl.uint8)

        v_packed = (v_lo_nibbles & 0x0F) | ((v_hi_nibbles & 0x0F) << 4)

        v_packed_offs = 256 + group_idx * 8 + lane_arange_8
        tl.store(out_ptr + base_out + v_packed_offs, v_packed, mask=mask_tok)

        v_scale_byte = tl.maximum(tl.minimum(v_scale_exp + 127, 255), 0).to(tl.uint8)
        tl.store(out_ptr + base_out + 544 + group_idx, v_scale_byte, mask=mask_tok)


@triton.jit
def _zero_pad_kernel(
    out_ptr,
    OUT_STRIDE_T,
    T,
):
    pid = tl.program_id(0)
    if pid >= T:
        return
    base = pid.to(tl.int64) * OUT_STRIDE_T + 576
    offs = tl.arange(0, 8)
    tl.store(out_ptr + base + offs, tl.zeros([8], dtype=tl.uint8))


def nvfp4_ds_mla_kv_linear(kv: torch.Tensor) -> torch.Tensor:
    kv = kv.to(torch.float32).contiguous()
    T = kv.shape[0]

    out = torch.empty((T, 584), dtype=torch.uint8, device=kv.device)

    N_GROUPS = 32  # 512 / 16

    grid = lambda meta: (
        triton.cdiv(T, meta['TOKENS_PER_PROG']),
        N_GROUPS,
    )

    _nvfp4_ds_mla_kv_linear_kernel[grid](
        kv,
        out,
        T,
        kv.stride(0),
        out.stride(0),
        N_GROUPS=N_GROUPS,
    )

    _zero_pad_kernel[(T,)](
        out,
        out.stride(0),
        T,
    )

    return out


# ---- 兼容入口：K/V 分离输入（内部拼接为 [T, 1024]）----
def nvfp4_ds_mla_kv_linear_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """兼容 (k, v) 分离输入的便捷包装：k/v 各 [T, 512] → 拼接 → nvfp4_ds_mla_kv_linear。"""
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()
    assert k.shape == v.shape and k.ndim == 2 and k.shape[1] == 512, f"k/v 应为 [T, 512]，got {k.shape}/{v.shape}"
    kv = torch.cat([k, v], dim=1)
    return nvfp4_ds_mla_kv_linear(kv)
