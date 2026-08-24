#!/usr/bin/env python3
"""P3-Step2b: cutlass_torch 张量构造设施源码探查（写适配器前的最后确认）"""
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

print("=== cutlass_torch.matrix ===")
print(inspect.getsource(cutlass_torch.matrix))

print("=== cutlass_torch.cute_tensor_like ===")
print(inspect.getsource(cutlass_torch.cute_tensor_like))

print("=== convert_cute_tensor 签名 ===")
print(inspect.signature(cutlass_torch.convert_cute_tensor))

# E8M0 uint8 → cute 直配尝试（带 Context）
print("\n=== uint8 直配 E8M0（Context 包装重试）===")
u8 = torch.randint(0, 200, (1, 2, 4, 32, 4, 4), dtype=torch.uint8)
t = from_dlpack(u8, assumed_align=16)
print(f"from_dlpack: elem={t.element_type} shape={t.shape}")
try:
    from cutlass._mlir.dialects import cute as _cute_dialect  # noqa
except Exception:
    pass
try:
    import cutlass.base_dsl.context as _ctx  # noqa
    print("base_dsl.context 存在")
except Exception as ex:
    print(f"base_dsl.context: {ex}")
for ctx_factory in ("Context", "context"):
    if hasattr(cutlass, ctx_factory):
        print(f"cutlass.{ctx_factory} 存在: {getattr(cutlass, ctx_factory)}")
try:
    with cutlass.Context():
        t8 = cute.recast_tensor(t, cutlass.Float8E8M0FNU)
        print(f"recast in Context: {t8.element_type} shape={t8.shape}")
except Exception as ex:
    print(f"cutlass.Context 包装失败: {type(ex).__name__}: {ex}")

# GPU 上 cute_tensor_like(f32→fp4) 行为细查
print("\n=== GPU cute_tensor_like(f32→FP4) 细查 ===")
if torch.cuda.is_available():
    m_, k_ = 4, 32
    a_f32 = torch.zeros(1, m_, k_, device="cuda")
    a_f32[..., 0::2] = 0.5    # nibble 1
    a_f32[..., 1::2] = -1.0   # nibble 9
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_f32, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
    print(f"a_torch: dtype={a_torch.dtype} shape={tuple(a_torch.shape)} "
          f"strides={a_torch.stride()} device={a_torch.device}")
    print(f"a_tensor(cute): elem={a_tensor.element_type} shape={a_tensor.shape} "
          f"strides={a_tensor.stride}")
    raw = a_torch.contiguous().view(-1)
    print(f"a_torch 字节[:16]: {raw[:16].tolist()}")
    print("  若每 fp4 一字节：偶k=+0.5→1, 奇k=-1.0→? （-1.0 的 fp4 码 9，若按有符号 int8 存则 -7?）")
