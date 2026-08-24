#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""triton_moe.py v2.1 — Triton W4A16 grouped MoE（长尾 T3′ + decode，B12X 退出）。

v2.1（fix-rerun, 2026-08-22）两项正确性修复 + 一项瘦身:
  1. [BUG修复] w13 E4M3 scale 解码: v2.0 用 .to(tl.float32) 整数直转（字节值 56
     被当标量 → 56× 系统性误差, 微测试实证）→ 改 bitcast float8e4nv 解码
  2. [BUG修复] 块跨界专家: v2.0 按排序后 pair 序 16 分块, 专家 run 不对齐 16 →
     跨界 pair 用错专家权重（随机路由下大面积错值）→ 改按专家对齐 padding
     （每活跃专家 run 补齐到 BM 倍数, 块内专家纯一）
  3. [P1-A] w2 scale 改 E8M0 逻辑直寻址（零拷贝视图, 免全量 E4M3 派生驻留;
     exp2(u8-127) in-kernel 精确解码）; w13 仍消费 swizzled E4M3 物理字节
     （DSL 壳背衬共用存储）

CUDA graph 兼容（decode 捕获）: grid = ceil((M·T + E·(BM-1))/BM) 静态容量
（M=捕获尺寸静态, T/E 静态）, 尾部哨兵块（expert_id=E）kernel 早退;
全程无 host 同步/动态 shape（invalid id → clamp + 权重清零, 计数静态）。
性能口径: decode/长尾性能未验证, 如实记录。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_w4a16_kernel(
    x_ptr,                    # bf16 [M, K]
    w_payload_ptr,            # u8 [E*N, K/2]（stacked, 行 = e*N + n）
    w13_sf_ptr,               # u8 [phys]（w13 swizzled E4M3, 全局行）
    w2_sf_ptr,                # u8 [E*N2, K2/32]（w2 E8M0 逻辑, 行 = e*N2 + n）
    out_ptr,                  # bf16 [M_pairs, N]
    sorted_token_ptr,         # int32 [M_pairs_pad]（pair 的 token id, pad=M）
    expert_ids_ptr,           # int32 [n_blocks]（block 的 expert; =E 为哨兵早退）
    M, N: tl.constexpr,
    K: tl.constexpr,
    RK13: tl.constexpr,       # w13 K // 64（w13 scale 物理 rk）
    K32_2: tl.constexpr,      # w2 K2 // 32（E8M0 组数）
    N_PER_EXPERT: tl.constexpr,
    EMUL_N: tl.constexpr,     # E*N 总行
    NUM_E: tl.constexpr,      # 专家数（哨兵判定）
    IS_W2: tl.constexpr,      # 0=w13（swizzled E4M3） 1=w2（E8M0 逻辑）
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if e >= NUM_E:            # 哨兵块（容量尾部）早退
        return

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_token = tl.load(sorted_token_ptr + pid_m * BM + tl.arange(0, BM))
    mask_m = offs_token < M
    offs_token = tl.where(mask_m, offs_token, 0)

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    w_row0 = e * N_PER_EXPERT + offs_n        # [BN] 全局行
    if IS_W2 == 0:
        a_ = w_row0 % 32
        b_ = (w_row0 // 32) % 4
        c_ = w_row0 // 128

    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_token[:, None].to(tl.int64) * K + offs_k[None, :],
                    mask=mask_m[:, None], other=0.0)          # [BM, BK] bf16
        # w payload [BN, BK/2] u8 → 解包 [BN, BK]
        wb = tl.load(w_payload_ptr + w_row0[:, None] * (K // 2)
                     + (offs_k[None, :] // 2))                 # [BN, BK]
        lo = wb & 0x0F
        hi = wb >> 4
        # 交错还原 [BN, BK]: 偶 k = lo[:, k//2], 奇 k = hi[:, k//2]
        is_even = (offs_k % 2) == 0                            # [BK]
        code = tl.where(is_even[None, :], lo, hi)              # [BN, BK]
        mag_c = code & 7
        sgn = 1.0 - 2.0 * ((code >> 3) & 1).to(tl.float32)
        e2 = (mag_c >> 1).to(tl.float32)
        m1 = (mag_c & 1).to(tl.float32)
        mag = tl.where(mag_c < 2, mag_c.to(tl.float32) * 0.5,
                       tl.exp2(e2 - 1.0) * (1.0 + m1 * 0.5))
        # scale: w13 = swizzled E4M3 字节 bitcast 解码; w2 = E8M0 逻辑 (exp2)
        g = offs_k // 16                                       # [BK] E4M3 组
        if IS_W2 == 0:
            d_ = g % 4
            ee = g // 4
            sf_off = (512 * RK13) * c_[:, None] + 512 * ee[None, :] \
                + 16 * a_[:, None] + 4 * b_[:, None] + d_[None, :]  # [BN, BK]
            sf_bits = tl.load(w13_sf_ptr + sf_off)             # uint8
            sf = sf_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            g32 = offs_k // 32                                  # [BK] E8M0 组
            sf_u8 = tl.load(w2_sf_ptr + w_row0[:, None] * K32_2
                            + g32[None, :])                     # [BN, BK]
            sf = tl.exp2(sf_u8.to(tl.float32) - 127.0)
        w = (mag * sgn * sf).to(tl.bfloat16)                    # [BN, BK]
        acc += tl.dot(x, tl.trans(w))                           # [BM, BN]

    out_off = (pid_m * BM + tl.arange(0, BM))[:, None].to(tl.int64) * N \
        + offs_n[None, :]
    tl.store(out_ptr + out_off, acc.to(tl.bfloat16),
             mask=mask_m[:, None] & (offs_n[None, :] < N))


def grouped_linear(x, w_payload, sf_ptr, sf_is_w2, expert_ids, sorted_tokens,
                   n_pairs_pad, N_per_expert, K, EMUL_N, num_e,
                   BM=16, BN=64, BK=128, out=None):
    """grouped W4A16: x [M, K] bf16 → out [n_pairs_pad, N] bf16（可外部预分配）。
    sf_is_w2=0: sf_ptr = w13 swizzled E4M3 物理字节; 1: w2 E8M0 逻辑。
    expert_ids 含哨兵（=num_e 早退）; sorted_tokens pad 值 = x 行数 M。"""
    M = x.shape[0]
    N = N_per_expert
    if out is None:
        # [BUGFIX 2026-08-22] 必须零初始化: 哨兵块早退 + pad 行掩码不写 →
        # torch.empty 的未初始化行经 yw=y*w_pad(0) 时 0×NaN/Inf垃圾=NaN 污染
        # index_add 输出（间歇性, 取决于分配器复用内存的位型——p29/p31 实证）
        out = torch.zeros(n_pairs_pad, N, dtype=torch.bfloat16, device=x.device)
    grid = (n_pairs_pad // BM, N // BN)
    _grouped_w4a16_kernel[grid](
        x, w_payload, sf_ptr, sf_ptr, out, sorted_tokens, expert_ids,
        M, N=N, K=K, RK13=K // 64, K32_2=K // 32,
        N_PER_EXPERT=N_per_expert, EMUL_N=EMUL_N, NUM_E=num_e,
        IS_W2=sf_is_w2, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return out


@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, N13: tl.constexpr, I: tl.constexpr,
                   BLK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < I
    gate = tl.load(x_ptr + row.to(tl.int64) * N13 + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row.to(tl.int64) * N13 + I + offs, mask=mask, other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row.to(tl.int64) * I + offs, act, mask=mask)


def swiglu(x, N13):
    Mp, _ = x.shape
    I = N13 // 2
    out = torch.empty(Mp, I, dtype=torch.bfloat16, device=x.device)
    grid = ((I + 255) // 256, Mp)
    _swiglu_kernel[grid](x, out, N13=N13, I=I, BLK=256)
    return out


def _align_pairs(sorted_tokens, sorted_experts, E, block_m, cap, M):
    """按专家对齐 padding（每活跃专家 run 补齐到 block_m 倍数）+ 尾部哨兵补到 cap。
    返回 (tokens_p, expert_ids, valid, vidx); 全部静态形状 [cap]/[cap//BM]
    （searchsorted/gather 构造, 无 repeat_interleave/布尔索引 → CUDA graph 安全;
    cap = n_pairs + E·(block_m-1) 上取整 ≥ 任何对齐 total, M/T/E 静态 → grid 静态）。"""
    dev = sorted_tokens.device
    # [BUGFIX 2026-08-22] bincount 内部含 CPU<->CUDA 拷贝, CUDA graph 捕获期非法
    # （TP4 捕获 03:26 实证）→ 改纯 GPU index_add_
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    counts.index_add_(0, sorted_experts.long(),
                      torch.ones(sorted_experts.numel(), dtype=torch.int32, device=dev))
    pad = (block_m - counts % block_m) % block_m
    seg = counts + pad
    seg_cum = torch.cumsum(seg, 0)                     # [E] 结束位置
    seg_start = seg_cum - seg
    pos_all = torch.arange(cap, device=dev, dtype=torch.int32)
    slots = torch.searchsorted(seg_cum, pos_all, right=True)   # [cap] 位置→专家
    slots_e = slots.clamp(max=E - 1)
    pos_in_seg = pos_all - seg_start[slots_e]
    valid = (slots < E) & (pos_in_seg < counts[slots_e])
    vidx = torch.cumsum(valid.to(torch.int32), 0) - 1  # valid 位置 → pair 序
    tokens_p = torch.where(valid, sorted_tokens[vidx.clamp_min(0)],
                           torch.full((cap,), M, dtype=torch.int32, device=dev))
    n_blocks = cap // block_m
    expert_ids = slots[torch.arange(n_blocks, device=dev) * block_m]
    return tokens_p, expert_ids, valid, vidx


def triton_moe(x, topk_ids, topk_weights,
               w13_payload, w13_sf, N13, w2_payload, w2_sf, N2,
               K, EMUL13, EMUL2, block_m=16):
    """完整 Triton MoE v2.1（专家对齐 + 哨兵容量 grid, CG 安全）。
    w13_sf: swizzled E4M3 物理字节 flat（DSL 壳背衬）;
    w2_sf:  E8M0 逻辑 [E*N2, K2/32] 零拷贝视图。"""
    M, T = topk_ids.shape
    dev = x.device
    if M == 0 or T == 0:
        # [BUGFIX 2026-08-22] 捕获期空批次（CUDA graph capture 传 M=0）:
        # _align_pairs 对空 sorted_tokens 索引越界 → 回退 B12X → B12X 懒准备
        # 撞捕获 RuntimeError → 启动死亡（TP4 03:07 实证）
        return torch.zeros(M, N2, dtype=torch.float32, device=dev)
    E = max(1, EMUL13 // N13)
    flat_experts = topk_ids.reshape(-1).to(torch.int32)
    # 无效 expert id: clamp + 权重清零（计数静态, CG 安全; 不做逐出）
    invalid_pair = (flat_experts < 0) | (flat_experts >= E)
    flat_experts = flat_experts.clamp(0, E - 1)
    flat_tokens = torch.arange(M, device=dev, dtype=torch.int32).repeat_interleave(T)
    order = torch.argsort(flat_experts, stable=True)
    sorted_tokens = flat_tokens[order].to(torch.int32)
    sorted_experts = flat_experts[order]
    sorted_invalid = invalid_pair[order]
    n_pairs = M * T

    # 静态容量: n_pairs + E·(block_m-1) 上取整到 block_m 倍数（CG 安全, grid 静态）
    cap = ((n_pairs + E * (block_m - 1) + block_m - 1) // block_m) * block_m
    tokens_p, expert_ids, valid, vidx = _align_pairs(
        sorted_tokens, sorted_experts, E, block_m, cap, M)

    # w13: K = hidden; scale = w13_sf（swizzled E4M3）; 零填 inter（pad 行 0）
    inter = torch.zeros(cap, N13, dtype=torch.bfloat16, device=dev)
    grouped_linear(x, w13_payload, w13_sf, 0, expert_ids, tokens_p,
                   cap, N13, K, EMUL13, E, BM=block_m, out=inter)
    act_pad = swiglu(inter, N13)               # [cap, I] padded 序（pad 行 0→act 0）

    # w2: K = I; 行 = padded 序; scale = w2_sf（E8M0 逻辑）
    K2 = N13 // 2
    sorted_rows_p = torch.arange(cap, device=dev, dtype=torch.int32)
    y = grouped_linear(act_pad, w2_payload, w2_sf, 1, expert_ids, sorted_rows_p,
                       cap, N2, K2, EMUL2, E, BM=block_m)

    # 加权 scatter（全静态形状）: pad 行权重 0 + token clamp（贡献恒 0）
    w_flat = topk_weights.reshape(-1)[order].float()           # [n_pairs]
    w_flat = w_flat * (~sorted_invalid).float()
    w_pad = torch.where(valid, w_flat[vidx.clamp_min(0)],
                        torch.zeros(cap, device=dev))           # [cap]
    yw = y.float() * w_pad[:, None]
    tokens_scatter = tokens_p.clamp(0, M - 1)
    out = torch.zeros(M, N2, dtype=torch.float32, device=dev)
    out.index_add_(0, tokens_scatter.long(), yw)
    return out
