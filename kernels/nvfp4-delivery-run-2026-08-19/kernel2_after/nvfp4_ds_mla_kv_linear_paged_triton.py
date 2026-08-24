import triton
import triton.language as tl
import torch

try:
    import triton.experimental.tle.language as tle
    HAS_TLE = True
except ImportError:
    HAS_TLE = False


@triton.autotune(
    configs=[
        triton.Config({'TOKENS_PER_PROG': 1}, num_warps=4,  num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 2}, num_warps=8,  num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 4}, num_warps=8,  num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 8}, num_warps=16, num_stages=2),
    ],
    key=['T'],
)
@triton.jit
def _nvfp4_ds_mla_kv_linear_paged_kernel(
    k_ptr,
    v_ptr,
    seq_ids_ptr,
    positions_ptr,
    block_table_ptr,
    kv_cache_ptr,
    T,
    max_blocks,
    num_blocks,
    stride_k_t,
    stride_v_t,
    stride_cache_b,
    stride_cache_s,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    HALF: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,
    ENVELOPE: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    LOG2_6: tl.constexpr = 2.584962500721156

    pid = tl.program_id(0)
    tok_start = pid * TOKENS_PER_PROG

    for ti in tl.static_range(TOKENS_PER_PROG):
        tok = tok_start + ti
        tok_valid = tok < T

        seq_id   = tl.load(seq_ids_ptr   + tok, mask=tok_valid, other=0).to(tl.int64)
        position = tl.load(positions_ptr + tok, mask=tok_valid, other=0).to(tl.int64)

        block_idx = position // BLOCK_SIZE
        slot      = position  % BLOCK_SIZE

        bid = tl.load(
            block_table_ptr + seq_id * max_blocks + block_idx,
            mask=tok_valid, other=0
        ).to(tl.int64)

        cache_line_ptr = (
            kv_cache_ptr
            + bid  * stride_cache_b
            + slot * stride_cache_s
        )

        k_row = k_ptr + tok * stride_k_t
        v_row = v_ptr + tok * stride_v_t

        for kv in tl.static_range(2):
            src_row         = k_row if kv == 0 else v_row
            packed_base_off = kv * HALF
            scale_base_off  = 2 * HALF + kv * NUM_GROUPS

            for g in tl.static_range(32):
                elem_start = g * GROUP_SIZE
                elem_offs  = elem_start + tl.arange(0, 16)

                vals = tl.load(src_row + elem_offs, mask=tok_valid, other=0.0,
                               eviction_policy='evict_last')

                vals_f32 = vals.to(tl.float32)

                abs_vals = tl.abs(vals_f32)
                max_abs  = tl.max(abs_vals, axis=0)
                safe_max = tl.maximum(max_abs, 1e-38)

                # NVFP4 E8M0：scale = 2^floor(log2(max/6))（与生产 v5 逐字节一致；floor 而非 ceil）
                log2_max      = tl.log2(safe_max)
                scale_exp_f   = log2_max - LOG2_6
                scale_exp_i   = tl.floor(scale_exp_f).to(tl.int32)
                scale_exp_clamped = tl.maximum(tl.minimum(scale_exp_i, 128), -127)
                e8m0_byte = (scale_exp_clamped + 127).to(tl.uint8)

                scale_f32  = tl.exp2(scale_exp_clamped.to(tl.float32))
                x_scaled   = vals_f32 / tl.maximum(scale_f32, 1e-38)
                x_clamped  = tl.minimum(tl.maximum(x_scaled, -6.0), 6.0)

                abs_x = tl.abs(x_clamped)
                signs = (x_clamped < 0.0).to(tl.int32)

                # Codebook: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
                # Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
                mag_idx = tl.where(abs_x < 0.25, 0,
                          tl.where(abs_x < 0.75, 1,
                          tl.where(abs_x < 1.25, 2,
                          tl.where(abs_x < 1.75, 3,
                          tl.where(abs_x < 2.50, 4,
                          tl.where(abs_x < 3.50, 5,
                          tl.where(abs_x < 5.00, 6, 7)))))))

                nibbles = (signs * 8 + mag_idx).to(tl.uint8)

                nib_2d = tl.reshape(nibbles, [8, 2])

                col_sel  = tl.reshape(tl.arange(0, 16) % 2, [8, 2])
                zero_u8  = tl.zeros([8, 2], tl.uint8)

                low_col  = tl.where(col_sel == 0, nib_2d, zero_u8)
                high_col = tl.where(col_sel == 1, nib_2d, zero_u8)

                low_nib  = tl.sum(low_col.to(tl.int32), axis=1).to(tl.uint8)
                high_nib = tl.sum(high_col.to(tl.int32), axis=1).to(tl.uint8)

                packed_byte = (low_nib | (high_nib << 4)).to(tl.uint8)

                byte_offs = packed_base_off + g * BYTES_PER_GROUP + tl.arange(0, 8)
                tl.store(cache_line_ptr + byte_offs, packed_byte, mask=tok_valid)
                tl.store(cache_line_ptr + scale_base_off + g, e8m0_byte, mask=tok_valid)

        # Write 8 zero-padding bytes at offset 576
        tl.store(
            cache_line_ptr + 576 + tl.arange(0, 8),
            tl.zeros([8], dtype=tl.uint8),
            mask=tok_valid,
        )


def nvfp4_ds_mla_kv_linear_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    kv_cache: torch.Tensor,
) -> torch.Tensor:
    T          = k.shape[0]
    max_blocks = block_table.shape[1]
    num_blocks = kv_cache.shape[0]

    k_f32 = k.contiguous().float()
    v_f32 = v.contiguous().float()

    BLOCK_SIZE      = 256
    DIM             = 512
    HALF            = 256
    NUM_GROUPS      = 32
    GROUP_SIZE      = 16
    BYTES_PER_GROUP = 8
    ENVELOPE        = 584

    grid = lambda meta: (triton.cdiv(T, meta['TOKENS_PER_PROG']),)

    _nvfp4_ds_mla_kv_linear_paged_kernel[grid](
        k_ptr           = k_f32,
        v_ptr           = v_f32,
        seq_ids_ptr     = seq_ids.contiguous(),
        positions_ptr   = positions.contiguous(),
        block_table_ptr = block_table.contiguous(),
        kv_cache_ptr    = kv_cache,
        T               = T,
        max_blocks      = max_blocks,
        num_blocks      = num_blocks,
        stride_k_t      = k_f32.stride(0),
        stride_v_t      = v_f32.stride(0),
        stride_cache_b  = kv_cache.stride(0),
        stride_cache_s  = kv_cache.stride(1),
        BLOCK_SIZE      = BLOCK_SIZE,
        DIM             = DIM,
        HALF            = HALF,
        NUM_GROUPS      = NUM_GROUPS,
        GROUP_SIZE      = GROUP_SIZE,
        BYTES_PER_GROUP = BYTES_PER_GROUP,
        ENVELOPE        = ENVELOPE,
    )

    return kv_cache
