#!/usr/bin/env python3
"""P3-Step2: routeB vendored kernel 的 Scale 布局契约探测 + 直配张量构造 API 探测
（node01 一次性容器，GPU 小 shape）
产出：
  P1 tile_atom_to_shape_SF 源码与 layout 对象（atom-swizzle 契约的直接证据）
  P2 cvt_sf_MKL_to_M32x4xrm_K4xrk_L 的字节映射实测（pattern 注入）
  P3 纯 torch swizzle 公式 vs 实测映射（逐字节判定的落锤点）
  P4 from_dlpack+recast 直配构造 API 可行性（uint8 → FP4/E8M0 cute tensor）
  P5 cute_tensor_like FP4 转换的 nibble 顺序
"""
import inspect
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, "/work/routeb_official")

import torch  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
import cutlass.torch as cutlass_torch  # noqa: E402
import cutlass.utils.blockscaled_layout as bs  # noqa: E402

print(f"cutlass {cutlass.__version__}, torch {torch.__version__}")

# ---------------------------------------------------------------- P1
print("\n=== P1: tile_atom_to_shape_SF 源码 ===")
print(inspect.getsource(bs.tile_atom_to_shape_SF))
try:
    print("BlockScaledBasicChunk:", inspect.getsource(bs.BlockScaledBasicChunk))
except Exception:
    pass
try:
    ly = bs.tile_atom_to_shape_SF(cute.make_shape(1, 256, 512), 32)
    print(f"layout(m=256,k=512,v=32): {ly}")
except Exception as ex:
    print(f"(layout 对象构造需 MLIR 上下文，跳过: {ex})")

print("\ncreate_and_permute_torch_tensor 源码:")
print(inspect.getsource(cutlass_torch.create_and_permute_torch_tensor))

# ---------------------------------------------------------------- P2/P3
print("\n=== P2/P3: cvt 字节映射实测 vs 纯 torch 公式 ===")
# DSL 4.5.2 兼容 shim（与 routeb_bench_blockscaled._install_testing_compat 相同）
import cutlass.cute.testing as _cute_testing
if not hasattr(cutlass, "testing"):
    cutlass.testing = _cute_testing
    sys.modules["cutlass.testing"] = _cute_testing
from dense_blockscaled_gemm_persistent_pingpong import (  # noqa: E402
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)

l, mn, k, sf_vec = 1, 256, 512, 32
sf_k = k // sf_vec  # 16

# plain scale 字节模式：b(m, kg) = (m*sf_k + kg) % 250 + 3（可逆、无 0/255 边界）
m_idx = torch.arange(mn).unsqueeze(1)
kg_idx = torch.arange(sf_k).unsqueeze(0)
plain_byte = ((m_idx * sf_k + kg_idx) % 250 + 3).to(torch.uint8)      # [mn, sf_k]
ref_f32 = torch.pow(2.0, plain_byte.float() - 127.0)                  # 可精确转 E8M0
# 例程的 ref 张量：(l, mn, sf_k) contiguous → permute(1,2,0) 视图
ref_view = ref_f32.reshape(1, mn, sf_k).permute(1, 2, 0)              # (mn, sf_k, l)

# 例程的 mma 形张量：mma_shape contiguous → permute(3,4,1,5,2,0) 视图
rm, rk = mn // 128, sf_k // 4
T = torch.zeros((l, rm, rk, 32, 4, 4), dtype=torch.float32)
P = T.permute(3, 4, 1, 5, 2, 0)                                       # (32,4,rm,4,rk,l)
cvt_sf_MKL_to_M32x4xrm_K4xrk_L(from_dlpack(ref_view), from_dlpack(P))

# 纯 torch swizzle 公式（待验证）：
#   buf[l, m//128, kg//4, m%32, (m//32)%4, kg%4] = plain[m, kg]
buf = torch.zeros((l, rm, rk, 32, 4, 4), dtype=torch.uint8)
mm = torch.arange(mn).unsqueeze(1).expand(mn, sf_k)
kk = torch.arange(sf_k).unsqueeze(0).expand(mn, sf_k)
buf[0, mm // 128, kk // 4, mm % 32, (mm // 32) % 4, kk % 4] = plain_byte

actual = (T[0] * 1).to(torch.uint8)  # T 值恰为 2^(b-127)；先取指数
# T 存的是 f32=2^(b-127)，直接比较指数
actual_byte = (torch.log2(T[0]).round() + 127).to(torch.uint8)
match = (actual_byte == buf[0]).float().mean().item()
print(f"纯 torch swizzle 公式 vs cvt 实测: 逐字节匹配率 = {match:.4f}")
print(f"T buffer shape={tuple(T.shape)}  (l, rest_m, rest_k, 32, 4, 4)")

# 连续平坦视图（kernel 实际读的内存）：
flat_actual = actual_byte.flatten()
flat_pred = buf.flatten()
print(f"平坦字节一致: {torch.equal(flat_actual, flat_pred)}")
if not torch.equal(flat_actual, flat_pred):
    diff = (flat_actual != flat_pred).nonzero().flatten()
    print(f"  首个不一致位置: {diff[:5].tolist()}")
    print(f"  actual[:16] = {flat_actual[:16].tolist()}")
    print(f"  pred[:16]    = {flat_pred[:16].tolist()}")

# ---------------------------------------------------------------- P4
print("\n=== P4: from_dlpack + recast 直配构造 ===")
u8 = torch.randint(0, 255, (1, 64, 32), dtype=torch.uint8)  # (l, m, k//2)
t = from_dlpack(u8)
print(f"from_dlpack(uint8): type={type(t)}, elem={t.element_type}, shape={t.shape}")
for cand in ("recast_tensor", "recast"):
    if hasattr(cute, cand):
        print(f"cute.{cand} 存在")
if hasattr(cute, "recast_tensor"):
    try:
        t4 = cute.recast_tensor(t, cutlass.Float4E2M1FN)
        print(f"recast→Float4E2M1FN: shape={t4.shape} elem={t4.element_type}")
    except Exception as ex:
        print(f"recast Float4 失败: {ex}")
try:
    t8 = cute.recast_tensor(t, cutlass.Float8E8M0FNU)
    print(f"recast→Float8E8M0FNU: shape={t8.shape} elem={t8.element_type}")
except Exception as ex:
    print(f"recast E8M0 失败: {ex}")
print(f"tensor methods: {[m for m in dir(t) if 'recast' in m.lower() or 'mark' in m.lower()]}")
print(f"from_dlpack signature: {inspect.signature(from_dlpack)}")
print(f"cute_tensor_like signature: {inspect.signature(cutlass_torch.cute_tensor_like)}")

# ---------------------------------------------------------------- P5
print("\n=== P5: cute_tensor_like FP4 转换 nibble 顺序 ===")
if torch.cuda.is_available():
    m_, k_ = 4, 32
    # a[m, k]: 偶 k = +0.5 (nibble 1), 奇 k = -1.0 (nibble 9)
    a_f32 = torch.zeros(1, m_, k_)
    a_f32[..., 0::2] = 0.5
    a_f32[..., 1::2] = -1.0
    a_cute, a_torch = cutlass_torch.cute_tensor_like(
        a_f32, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
    print(f"a_torch: dtype={a_torch.dtype} shape={tuple(a_torch.shape)}")
    try:
        b = a_torch.reshape(m_, k_ // 2)
        print(f"首行字节: {b[0].tolist()}")
        print("  期望 0x91=145 → lo(偶k)=+0.5, hi(奇k)=-1.0")
    except Exception as ex:
        print(f"reshape 检查失败: {ex}")
else:
    print("无 GPU，跳过 P5")

print("\n✅ probe 完成")
