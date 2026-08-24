"""生成 nvfp4_4w4a_prefill_gemm_v16_fixed_triton.py：修复 Triton 3.6 e4m3 dot_scaled 3 硬约束。
F1: 位置参数顺序（lhs,lhs_scale,'e4m3',rhs,rhs_scale,'e4m3',acc）
F2: k_pack=True（e4m3 必须；fp8 PACKED=1 数据布局不变）
F3: scale 因子 16（fp8e4nv）→ lhs_scale [BM,K//16]、rhs_scale [N,K//16]（32 组重复×2，值等价）
"""
import sys

SRC = "nvfp4_4w4a_prefill_gemm_v16_triton.py"
DST = "nvfp4_4w4a_prefill_gemm_v16_fixed_triton.py"
src = open(SRC).read()

# ---- F3a: host Ws_rhs → [N, K//16]（沿 K 重复×2）----
old1 = "    Ws_expanded = Ws_e8m0.repeat_interleave(128, dim=1)  # [K//32, N]\n    Ws_rhs = Ws_expanded.t().contiguous()                # [N, K//32]（不 trans 的 rhs_scale 布局）"
new1 = "    Ws_expanded = Ws_e8m0.repeat_interleave(128, dim=1)  # [K//32, N]\n    Ws_rhs = Ws_expanded.repeat_interleave(2, dim=0).t().contiguous()  # [N, K//16]（fp8e4nv scale 因子 16，32 组重复值等价）"
assert old1 in src, "anchor1"
src = src.replace(old1, new1)

# ---- F3b: kernel rhs_scale 加载改为 K//16 组 ----
old2 = """    GROUPS_K: tl.constexpr = BLOCK_K // 32
    BIAS127: tl.constexpr = tl.constexpr(127)"""
new2 = """    GROUPS_K: tl.constexpr = BLOCK_K // 32
    GROUPS_K16: tl.constexpr = BLOCK_K // 16
    BIAS127: tl.constexpr = tl.constexpr(127)"""
assert old2 in src, "anchor2"
src = src.replace(old2, new2)

old3 = "    K_groups = K // 32"
new3 = "    K_groups = K // 16"
assert old3 in src, "anchor3"
src = src.replace(old3, new3)

old4 = """        # ---- rhs_scale [BLOCK_N, GROUPS_K]（Ws_rhs [N, K//32]，不 trans）----
        k_group_start = k_start // 32
        offs_kg = tl.arange(0, GROUPS_K).to(tl.int64) + k_group_start
        mask_kg = offs_kg < K_groups"""
new4 = """        # ---- rhs_scale [BLOCK_N, GROUPS_K16]（Ws_rhs [N, K//16]，不 trans）----
        k_group_start = k_start // 16
        offs_kg = tl.arange(0, GROUPS_K16).to(tl.int64) + k_group_start
        mask_kg = offs_kg < K_groups"""
assert old4 in src, "anchor4"
src = src.replace(old4, new4)

# ---- F3c: lhs_scale → [BM, BLOCK_K//16]（32 组重复×2）----
old5 = """        lhs_scale = a_scale_raw.to(tl.uint8)                      # uint8 e8m0 字节
"""
new5 = """        lhs_scale16 = tl.reshape(
            tl.broadcast_to(
                tl.reshape(a_scale_raw.to(tl.uint8), (BLOCK_M, GROUPS_K, 1)),
                (BLOCK_M, GROUPS_K, 2),
            ),
            (BLOCK_M, BLOCK_K // 16),
        )                                                    # [BM, K//16] uint8 e8m0（32 组重复×2）
"""
assert old5 in src, "anchor5"
src = src.replace(old5, new5)

# ---- F1 + F2: dot_scaled 调用 ----
old6 = """        acc = tl.dot_scaled(
            a_fp8_int, lhs_scale,
            w_fp8,    rhs_scale,
            acc,
            lhs_format='e4m3', rhs_format='e4m3',
            rhs_k_pack=False,
        )"""
new6 = """        acc = tl.dot_scaled(
            a_fp8_int, lhs_scale16, 'e4m3',
            w_fp8,    rhs_scale, 'e4m3',
            acc,
            lhs_k_pack=True, rhs_k_pack=True,
        )"""
assert old6 in src, "anchor6"
src = src.replace(old6, new6)

open(DST, "w").write(src)
print("FIXED_MODULE_WRITTEN")
import ast
ast.parse(src)
print("PARSE_OK")
