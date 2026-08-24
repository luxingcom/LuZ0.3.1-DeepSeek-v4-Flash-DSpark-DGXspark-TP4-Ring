"""生成 v16_fixed（正确修复版）：仅 F1 位置参数 + F2 k_pack=True。
F3（scale 16 组）为误读已回退——uint8 e8m0 scale 因子=32，v16 原 scale 设计正确。
"""
SRC = "nvfp4_4w4a_prefill_gemm_v16_triton.py"
DST = "nvfp4_4w4a_prefill_gemm_v16_fixed_triton.py"
src = open(SRC).read()

# ---- F1: dot_scaled 位置参数 ----
old6 = """        acc = tl.dot_scaled(
            a_fp8_int, lhs_scale,
            w_fp8,    rhs_scale,
            acc,
            lhs_format='e4m3', rhs_format='e4m3',
            rhs_k_pack=False,
        )"""
new6 = """        acc = tl.dot_scaled(
            a_fp8_int, lhs_scale, 'e4m3',
            w_fp8,    rhs_scale, 'e4m3',
            acc,
            lhs_k_pack=True, rhs_k_pack=True,
        )"""
assert old6 in src, "anchor6"
src = src.replace(old6, new6)

open(DST, "w").write(src)
print("FIXED_F1F2_WRITTEN")
import ast
ast.parse(src)
print("PARSE_OK")
