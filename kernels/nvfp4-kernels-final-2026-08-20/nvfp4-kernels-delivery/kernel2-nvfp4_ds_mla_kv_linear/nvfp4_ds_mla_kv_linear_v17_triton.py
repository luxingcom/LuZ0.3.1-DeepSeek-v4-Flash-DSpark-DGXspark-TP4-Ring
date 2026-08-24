# ============================================================================
# nvfp4_ds_mla_kv_linear_v17_triton.py —— NVFP4 DS-MLA KV 线性信封写回（v17）
# 手写优化版（2026-08-20，基于优化空间分析 S1+M1+S3）
# 目标：打满 HBM（273 GB/s），大 T 预期 120~230 GB/s（v11 53.4 的 2.3~4.3×）
#
# 相对 v11（带宽冠军）的架构改进：
#   S1 每 program 处理 BLOCK_G 组（8/16/32）× TPP token（1/2/4）
#      —— 负载从"1 组 16 元素"提升到"32 组×4 token=2048 元素(8KB)"，
#         grid 从 (T/TPP, 32) 降到 (T/TPP, 64/BLOCK_G)，T=65536 时 ~8192 blocks（-16×）
#   M1 连续 1D load（BLOCK_G*16 元素）+ tl.multiple_of 向量化提示；packed 连续 store
#   S3 pad 内联：out 用 torch.zeros 预分配（零填充一次），移除独立 zero_pad kernel
#   E1 小 T（decode T=6~8）走 TPP=1/BLOCK_G=32 配置（grid≈T，launch 低）
#
# 语义与 v11 逐字节一致（金标准）：E8M0=floor(log2(max/6))+127 clamp[0,255]；
#   阈值链 strict >（tie low）；pack 低半字节=偶元素；信封 584B。
# ============================================================================
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # (BLOCK_G, TPP, warps)
        triton.Config({'BLOCK_G': 32, 'TOKENS_PER_PROG': 1}, num_warps=1, num_stages=2),  # 小 T（decode）
        triton.Config({'BLOCK_G': 32, 'TOKENS_PER_PROG': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_G': 32, 'TOKENS_PER_PROG': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_G': 16, 'TOKENS_PER_PROG': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_G': 16, 'TOKENS_PER_PROG': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_G': 16, 'TOKENS_PER_PROG': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_G': 8,  'TOKENS_PER_PROG': 4}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_G': 8,  'TOKENS_PER_PROG': 8}, num_warps=2, num_stages=4),
    ],
    key=['T'],
)
@triton.jit
def _nvfp4_kv_linear_v17_kernel(
    kv_ptr,
    out_ptr,
    T,
    KV_STRIDE_T,
    OUT_STRIDE_T,
    BLOCK_G: tl.constexpr,
    TOKENS_PER_PROG: tl.constexpr,
):
    """每 program：BLOCK_G 个连续组（每组 16 元素）× TPP 个 token。

    kv [T, 1024] row-major；out [T, 584]（torch.zeros 预分配，pad 已零）。
    组 g ∈ [0,64)：g<32 → K（kv 列 g*16），g≥32 → V（kv 列 512+(g-32)*16）。
    """
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, TOKENS_PER_PROG)
    t_tile = pid % num_t_tiles
    g_block = pid // num_t_tiles

    g0 = g_block * BLOCK_G                              # 组起点 [0, 64)
    is_v = (g0 >= 32).to(tl.int32)
    local_g0 = g0 - is_v * 32
    col_base = is_v * 512 + local_g0 * 16               # kv 列起点（连续 BLOCK_G*16 元素）

    offs_t = t_tile * TOKENS_PER_PROG + tl.arange(0, TOKENS_PER_PROG)
    mask_t = offs_t < T

    lane_c = tl.arange(0, BLOCK_G * 16)
    lane_c = tl.multiple_of(lane_c, 16)                 # 16 元素对齐（64B 连续读）
    lane_p = tl.arange(0, BLOCK_G * 8)
    lane_p = tl.multiple_of(lane_p, 8)
    lane_g = tl.arange(0, BLOCK_G)
    lane_g = tl.multiple_of(lane_g, 4)

    # ---- 连续 load [TPP, BLOCK_G*16]（列全在 [0,1024) 内，仅需 token mask）----
    kv_ptrs = kv_ptr + offs_t[:, None].to(tl.int64) * KV_STRIDE_T + (col_base + lane_c)[None, :].to(tl.int64)
    vals = tl.load(kv_ptrs, mask=mask_t[:, None], other=0.0, eviction_policy='evict_first')

    # ---- 量化 [TPP, BLOCK_G, 16] ----
    g = tl.reshape(vals, (TOKENS_PER_PROG, BLOCK_G, 16))
    amax = tl.max(tl.abs(g), axis=2)                    # [TPP, BLOCK_G]
    amax = tl.maximum(amax, 1e-30)

    log2v = tl.log2(amax / 6.0)                         # 含 /6（与 v11 逐字节一致）
    scale_exp = tl.floor(log2v).to(tl.int32)
    scale_exp = tl.maximum(tl.minimum(scale_exp, 127), -126)
    scale_f = tl.exp2(scale_exp.to(tl.float32))         # [TPP, BLOCK_G]

    scaled = g / scale_f[:, :, None]
    clamped = tl.minimum(tl.maximum(scaled, -6.0), 6.0)

    sign = (clamped < 0.0).to(tl.int32)
    abs2 = tl.abs(clamped)

    me = tl.zeros((TOKENS_PER_PROG, BLOCK_G, 16), dtype=tl.int32)
    me = tl.where(abs2 > 0.25, 1, me)
    me = tl.where(abs2 > 0.75, 2, me)
    me = tl.where(abs2 > 1.25, 3, me)
    me = tl.where(abs2 > 1.75, 4, me)
    me = tl.where(abs2 > 2.5,  5, me)
    me = tl.where(abs2 > 3.5,  6, me)
    me = tl.where(abs2 > 5.0,  7, me)

    nibble = (sign * 8 + me).to(tl.uint8)               # [TPP, BLOCK_G, 16]

    # ---- pack：低半字节=偶元素（reshape+split，无 slice）----
    nib2 = tl.reshape(nibble, (TOKENS_PER_PROG, BLOCK_G, 8, 2))
    lo, hi = tl.split(nib2)                             # 各 [TPP, BLOCK_G, 8]
    packed = ((lo & 0x0F) | ((hi & 0x0F) << 4)).to(tl.uint8)
    packed_flat = tl.reshape(packed, (TOKENS_PER_PROG, BLOCK_G * 8))

    # ---- store packed [TPP, BLOCK_G*8]（连续）----
    p_offs = is_v * 256 + local_g0 * 8 + lane_p
    out_ptrs = out_ptr + offs_t[:, None].to(tl.int64) * OUT_STRIDE_T + p_offs[None, :].to(tl.int64)
    tl.store(out_ptrs, packed_flat, mask=mask_t[:, None])

    # ---- store scale [TPP, BLOCK_G]（E8M0 字节）----
    s_byte = tl.maximum(tl.minimum(scale_exp + 127, 255), 0).to(tl.uint8)
    s_offs = 512 + is_v * 32 + local_g0 + lane_g
    s_ptrs = out_ptr + offs_t[:, None].to(tl.int64) * OUT_STRIDE_T + s_offs[None, :].to(tl.int64)
    tl.store(s_ptrs, s_byte, mask=mask_t[:, None])


def nvfp4_ds_mla_kv_linear(kv: torch.Tensor) -> torch.Tensor:
    """NVFP4 DS-MLA KV 线性信封写回（v17）：kv [T,1024] → out [T,584] uint8。

    信封：data[0:256]=K packed, [256:512]=V packed,
          scale[512:544]=K E8M0, [544:576]=V E8M0, pad[576:584]=0（torch.zeros 预分配）。
    语义与 v11 逐字节一致。
    """
    kv_fp32 = kv.to(torch.float32).contiguous()
    T = kv_fp32.shape[0]
    assert kv_fp32.shape[1] == 1024, f"kv 应为 [T,1024]，got {kv_fp32.shape}"

    out = torch.zeros((T, 584), dtype=torch.uint8, device=kv_fp32.device)  # pad 内联（零填充）

    if T == 0:
        return out

    grid = lambda meta: (
        triton.cdiv(T, meta['TOKENS_PER_PROG']) * (64 // meta['BLOCK_G']),
    )

    _nvfp4_kv_linear_v17_kernel[grid](
        kv_fp32,
        out,
        T,
        kv_fp32.stride(0),
        out.stride(0),
    )
    return out


# ---- 兼容入口：K/V 分离输入 ----
def nvfp4_ds_mla_kv_linear_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """兼容 (k, v) 分离输入：k/v 各 [T, 512] → 拼接 → nvfp4_ds_mla_kv_linear。"""
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()
    assert k.shape == v.shape and k.ndim == 2 and k.shape[1] == 512, \
        f"k/v 应为 [T, 512]，got {k.shape}/{v.shape}"
    kv = torch.cat([k, v], dim=1)
    return nvfp4_ds_mla_kv_linear(kv)
