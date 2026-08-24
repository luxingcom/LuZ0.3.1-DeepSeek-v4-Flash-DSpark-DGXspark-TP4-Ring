import triton
import triton.language as tl
import torch


# ============================================================================
# nvfp4_ds_mla_kv_linear —— v12（MCP 验证架构 + scale 语义修正）
# 环境：torch 2.11 / Triton 3.6 / CC=(12,1)=sm_121a（DGX Spark GB10）
#
# 架构（采纳 MCP autotune job e3101ddd v2，服务端实测 speedup 59.67×）：
#   2D grid（T/TOKENS_PER_PROG × 32 D-group）；每 program 向量化 load [T,16]；
#   TOKENS_PER_PROG ∈ {1..32} autotune；独立 zero-pad kernel；掩码求和打包（无 slice）。
#
# v12 修正（相对 MCP v2）：scale_factor = 2^(e8m0-127)（去掉 MCP 误加的 ×6），
#   使 x / scale ∈ [1,6] 全码本归一化，与生产 v5 / torch argmin 逐字节一致。
# scale 编码：e8m0 = floor(log2(max/6)) + 127，clamp [0,255]（floor 语义）。
# 舍入：阈值链严格 >（等距取低档）。
# ============================================================================


# ---------------------------------------------------------------------------
# Zero-pad kernel：写入 [576:584] 的 8 字节 pad
# ---------------------------------------------------------------------------
@triton.jit
def _zero_pad_kernel(
    out_ptr,
    T,
    OUT_STRIDE: tl.constexpr,   # 584
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    tok_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = tok_offs < T

    base = tok_offs * OUT_STRIDE + 576  # [BLOCK_T]

    for b in tl.static_range(0, 8):
        tl.store(out_ptr + base + b, tl.zeros([BLOCK_T], dtype=tl.uint8), mask=mask_t)


# ---------------------------------------------------------------------------
# 主 kernel：per-token K/V → NVFP4 E2M1 + E8M0 scale → 584B 线性信封
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'TOKENS_PER_PROG': 1}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 2}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 4}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 8}, num_warps=4, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 16}, num_warps=8, num_stages=2),
        triton.Config({'TOKENS_PER_PROG': 32}, num_warps=8, num_stages=3),
    ],
    key=['T'],
)
@triton.jit
def _nvfp4_ds_mla_kv_linear_kernel(
    kv_ptr,         # [T, 1024] float32
    out_ptr,        # [T, 584]  uint8
    T,
    KV_STRIDE_T,    # stride along token dim of kv  (= 1024)
    OUT_STRIDE_T,   # stride along token dim of out (= 584)
    TOKENS_PER_PROG: tl.constexpr,
):
    pid_t = tl.program_id(0)   # token-tile index
    pid_g = tl.program_id(1)   # D-group index [0, 32)

    # groups 0..15 -> K, 16..31 -> V
    is_v = pid_g >= 16
    g_half = pid_g - (16 * is_v)

    tok_start = pid_t * TOKENS_PER_PROG
    tok_offs = tok_start + tl.arange(0, TOKENS_PER_PROG)   # [TOKENS_PER_PROG]
    tok_mask = tok_offs < T

    kv_col_start = is_v * 512 + g_half * 16

    tok2d = tok_offs[:, None].to(tl.int64)   # [TOKENS_PER_PROG, 1]
    lane2d = tl.arange(0, 16)[None, :]       # [1, 16]

    kv_offs = tok2d * KV_STRIDE_T + kv_col_start + lane2d
    tok_mask2d = tok_mask[:, None]

    xg = tl.load(kv_ptr + kv_offs, mask=tok_mask2d, other=0.0, eviction_policy='evict_last')

    abs_xg = tl.abs(xg)
    max_abs = tl.max(abs_xg, axis=1)         # [TOKENS_PER_PROG]

    # E8M0: floor(log2(max/6)) + 127, clamp [0,255]
    safe_max = tl.maximum(max_abs, 1e-38)
    log2_val = tl.log2(safe_max / 6.0)
    e8m0_f = tl.floor(log2_val) + 127.0
    e8m0_f = tl.where(max_abs == 0.0, tl.zeros_like(e8m0_f), e8m0_f)
    e8m0_f = tl.minimum(tl.maximum(e8m0_f, 0.0), 255.0)
    e8m0_i = e8m0_f.to(tl.int32)

    # v12 修正：scale_factor = 2^(e8m0 - 127)（MCP v2 误乘 6，此处去掉）
    scale_exp = e8m0_f - 127.0
    scale_factor = tl.exp2(scale_exp)        # [TOKENS_PER_PROG]

    sf2d = scale_factor[:, None]
    sf_safe = tl.maximum(sf2d, 1e-38)
    xg_norm = xg / sf_safe

    signs = (xg_norm < 0.0).to(tl.int32)
    abs_norm = tl.abs(xg_norm)

    # 阈值链：严格 >（等距取低档，与 torch argmin 一致）
    mag_idx = tl.zeros([TOKENS_PER_PROG, 16], dtype=tl.int32)
    mag_idx = mag_idx + (abs_norm > 0.25).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 0.75).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 1.25).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 1.75).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 2.5).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 3.5).to(tl.int32)
    mag_idx = mag_idx + (abs_norm > 5.0).to(tl.int32)

    nibble = (signs << 3) | mag_idx          # [TOKENS_PER_PROG, 16]

    packed_col_base = is_v * 256 + g_half * 8
    scale_col = 512 + is_v * 32 + g_half

    # 8 字节/组打包（掩码求和，无 slice）
    for b in tl.static_range(0, 8):
        e_lane = b * 2
        o_lane = b * 2 + 1

        lanes = tl.arange(0, 16)[None, :]
        e_mask = (lanes == e_lane)
        nib_e = tl.sum(nibble * e_mask.to(tl.int32), axis=1)
        o_mask = (lanes == o_lane)
        nib_o = tl.sum(nibble * o_mask.to(tl.int32), axis=1)

        packed_byte = ((nib_o << 4) | nib_e).to(tl.uint8)

        byte_col = packed_col_base + b
        out_offs = tok_offs.to(tl.int64) * OUT_STRIDE_T + byte_col
        tl.store(out_ptr + out_offs, packed_byte, mask=tok_mask)

    scale_byte = e8m0_i.to(tl.uint8)
    scale_offs = tok_offs.to(tl.int64) * OUT_STRIDE_T + scale_col
    tl.store(out_ptr + scale_offs, scale_byte, mask=tok_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def nvfp4_ds_mla_kv_linear(kv: torch.Tensor) -> torch.Tensor:
    assert kv.ndim == 2 and kv.shape[1] == 1024, f"Expected kv shape [T, 1024], got {kv.shape}"

    kv_f32 = kv.float().contiguous()
    T = kv_f32.shape[0]

    out = torch.zeros(T, 584, dtype=torch.uint8, device=kv.device)

    if T == 0:
        return out

    def grid(meta):
        return (triton.cdiv(T, meta['TOKENS_PER_PROG']), 32)

    _nvfp4_ds_mla_kv_linear_kernel[grid](
        kv_f32,
        out,
        T,
        kv_f32.stride(0),
        out.stride(0),
    )

    return out


def nvfp4_ds_mla_kv_linear_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    kv = torch.cat([k, v], dim=-1)
    return nvfp4_ds_mla_kv_linear(kv)
