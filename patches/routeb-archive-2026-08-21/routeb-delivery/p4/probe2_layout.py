#!/usr/bin/env python3
"""P4 探针：验证 routeb_pipe 布局断言 + 量化正确性 + 管线数值 + M 动态性。

P1  cutlass.torch.matrix 形状（(m,k,l) 判定）
P2  fp4 打包布局：官方模板路径底层字节 vs 我方直写
P3  SF swizzle：官方模板路径 buffer vs 我方 sf_scatter（逻辑值 + 视图比对）
P4a triton A 量化 vs torch 参考量化（逐字节）
P4  全管线数值：RouteBGemm(256,512,1024) vs dequant_ref
P5  M=64 部分 tile 边界
P6  编译对象跨 M 复用
P7  计时口径 sanity（vs run_bs testing.benchmark）
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cutlass
import cutlass.cute.testing as _ct_testing
if not hasattr(cutlass, "testing"):
    cutlass.testing = _ct_testing
    sys.modules["cutlass.testing"] = _ct_testing

import cutlass.torch as ct
from cutlass.cute.runtime import from_dlpack

import routeb_pipe as rp  # 先导入（内部挂载 routeb_official 路径）
pp = rp.pp

torch.manual_seed(0)
dev = "cuda"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'✅' if ok else '❌'} {name} {detail}")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
print("=== P1: cutlass.torch.matrix 形状 ===")
a_ref = ct.matrix(1, 4, 16, False, cutlass.Float32)
print("  matrix(l=1,m=4,k=16,k-major) ->", tuple(a_ref.shape), a_ref.stride())
check("P1 matrix 返回 (m,k,l)", tuple(a_ref.shape) == (4, 16, 1),
      f"got {tuple(a_ref.shape)}")

# ---------------------------------------------------------------------------
print("=== P2: fp4 打包布局（官方模板 vs 直写） ===")
m, k = 4, 16
codes = torch.tensor([2, 4, 6, 7, 1, 3, 5, 0, 10, 9, 15, 12, 2, 2, 6, 8], dtype=torch.int32)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
vals = torch.zeros(m, k, 1)
for i, c in enumerate(codes.tolist()):
    v = E2M1[c & 7]
    vals[0, i, 0] = -v if c >= 8 else v
vals[1, :, 0] = 1.0
vals[2, :, 0] = -1.0
vals[3, :, 0] = 6.0

a_cute, a_torch = ct.cute_tensor_like(vals, cutlass.Float4E2M1FN,
                                      is_dynamic_layout=True, assumed_align=16)
print("  a_torch:", tuple(a_torch.shape), a_torch.dtype, "stride", a_torch.stride())
tmpl_bytes = a_torch.cpu().flatten().tolist()
print("  模板 row0 bytes:", tmpl_bytes[:16])

pred = torch.zeros(m * k, dtype=torch.int32)  # 紧凑行布局：行 m 起始偏移 m*(k//2)
for row in range(m):
    if row == 0:
        rowcodes = codes
    else:
        rowcodes = torch.full((k,), [2, 10, 7][row - 1], dtype=torch.int32)
    b = rowcodes[0::2] | (rowcodes[1::2] << 4)
    pred[row * (k // 2): (row + 1) * (k // 2)] = b
mine_bytes = pred.to(torch.int8).flatten().tolist()
n_valid = m * (k // 2)
check("P2 fp4 打包 = 紧凑行布局 lo|hi<<4",
      tmpl_bytes[:n_valid] == mine_bytes[:n_valid])
if tmpl_bytes[:n_valid] != mine_bytes[:n_valid]:
    print("  mine :", mine_bytes[:n_valid])
    print("  tmpl :", tmpl_bytes[:n_valid])
print(f"  ℹ️  模板 buffer 后半（应为未使用）: {tmpl_bytes[n_valid:n_valid + 8]}")

# ---------------------------------------------------------------------------
print("=== P3: SF swizzle（官方模板 vs sf_scatter） ===")
M, K = 256, 512
sf_k = K // 32
rest_m = (M + 127) // 128
rest_k = (sf_k + 3) // 4
S = (127 - 60 + (torch.arange(M).unsqueeze(1) * 31 +
                 torch.arange(sf_k).unsqueeze(0) * 7) % 121).to(torch.uint8)
v = torch.pow(2.0, S.float() - 127.0)  # 精确幂次，E8M0 往返无损

mma_shape = (1, rest_m, rest_k, 32, 4, 4)
ref_base = torch.zeros(1, M, sf_k)
ref_base[0] = v
ref_view = ref_base.permute(1, 2, 0)  # (m, kg, l)
cute_f32 = ct.create_and_permute_torch_tensor(
    mma_shape, torch.float32, permute_order=(3, 4, 1, 5, 2, 0),
    init_type=ct.TensorInitType.SKIP)
pp.cvt_sf_MKL_to_M32x4xrm_K4xrk_L(from_dlpack(ref_view), from_dlpack(cute_f32))
cute_f32_cuda = cute_f32.cuda()
sf_cute, sf_torch = ct.cute_tensor_like(cute_f32_cuda, cutlass.Float8E8M0FNU,
                                        is_dynamic_layout=True, assumed_align=16)
sf_cute = ct.convert_cute_tensor(cute_f32_cuda, sf_cute, cutlass.Float8E8M0FNU,
                                 is_dynamic_layout=True)
print("  模板 sf_torch:", tuple(sf_torch.shape), sf_torch.dtype, "stride", sf_torch.stride())

my_base = torch.zeros(1, rest_m, rest_k, 32, 4, 4, dtype=torch.int8, device=dev)
rp.sf_scatter(S.to(dev), my_base.view(torch.uint8))

# 逻辑取回：模板 (32,4,rest_m,4,rest_k,l) 高级索引
m_ar = torch.arange(M)
kg_ar = torch.arange(sf_k)
ai, bi, ci = m_ar % 32, (m_ar // 32) % 4, m_ar // 128
di, ei = kg_ar % 4, kg_ar // 4
got = sf_torch[ai[:, None], bi[:, None], ci[:, None], di[None, :], ei[None, :], 0].cpu()
check("P3 模板取回逻辑值 == S", torch.equal(got.to(torch.uint8), S))

# 视图级比对（我方 permute 视图 vs 模板）
my_view = my_base.permute(3, 4, 1, 5, 2, 0)
same_view = torch.equal(my_view.cpu(), sf_torch.cpu().to(torch.int8))
print(f"  ℹ️  视图逐元素一致: {same_view}（stride 我方 {my_view.stride()} vs 模板 {sf_torch.stride()}）")
check("P3b sf_scatter 视图 == 模板视图", same_view)

# ---------------------------------------------------------------------------
print("=== P4a: triton A 量化 vs torch 参考 ===")
Mq, Kq = 512, 1024
A = (torch.randn(Mq, Kq, device=dev) * 0.5).half()
aq = torch.zeros(Mq, Kq // 2, dtype=torch.uint8, device=dev)
asf = torch.zeros(Mq, Kq // 32, dtype=torch.uint8, device=dev)
rp.triton_a_quant(A, aq, asf)
# torch 参考（bench 语义）
Af = A.float()
xb = Af.reshape(Mq, Kq // 32, 32)
amax = xb.abs().amax(-1)
e = torch.floor(torch.log2(torch.clamp(amax, min=1e-30) / 6.0)).long() + 127
ref_sf = e.clamp(0, 255).to(torch.uint8)
sfv = torch.pow(2.0, e.clamp(0, 255).float() - 127.0).unsqueeze(-1)
xn = torch.clamp(xb / sfv, -6.0, 6.0)
mag = torch.tensor(E2M1, device=dev)
idx = (xn.abs().unsqueeze(-1) - mag).abs().argmin(-1)
nib = idx | ((xn < 0).long() * 8)
ref_packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8).reshape(Mq, Kq // 2)
mm_sf = (asf != ref_sf).sum().item()
# ±0 编码差异：triton 对 -0.0 不置符号位（code 0），torch 参考置位（code 8）——数值等价
def norm_zero(p):
    p = p.to(torch.int32)
    return torch.where((p & 0x7) == 0, torch.zeros_like(p) & 0xF, p & 0xF)
mm_pk = (norm_zero(aq) != norm_zero(ref_packed)).sum().item()
check("P4a SF 逐字节一致", mm_sf == 0, f"mismatch={mm_sf}")
check("P4a packed 一致（±0 编码等价归一后）", mm_pk == 0, f"mismatch={mm_pk}")

# ---------------------------------------------------------------------------
print("=== P4: 全管线数值 (256,512,1024) ===")
M, N, K = 256, 512, 1024
g = rp.RouteBGemm(M, N, K)
A = (torch.randn(M, K, device=dev) * 0.5).half()
Wt = (torch.randn(N, K, device=dev) * 0.3)
aq, asf = g.set_A(A)
wq, ws = g.set_W(Wt)
g.run()
torch.cuda.synchronize()
out = g.out().float()
ref = rp.dequant_ref(aq, asf, wq, ws)
rel = (out - ref).abs().max() / ref.abs().max()
check("P4 管线输出 rel_err < 0.02", rel < 0.02, f"rel_err={rel:.5f}")

# ---------------------------------------------------------------------------
print("=== P5: M=64 部分 tile ===")
try:
    g64 = rp.RouteBGemm(64, N, K)
    A64 = (torch.randn(64, K, device=dev) * 0.5).half()
    aq64, asf64 = g64.set_A(A64)
    g64.set_W_prepacked(wq, ws)
    g64.run()
    torch.cuda.synchronize()
    o64 = g64.out().float()
    r64 = rp.dequant_ref(aq64, asf64, wq, ws)
    rel64 = (o64 - r64).abs().max() / r64.abs().max()
    check("P5 M=64 可跑且 rel_err < 0.02", rel64 < 0.02, f"rel_err={rel64:.5f}")
except Exception as ex:
    check("P5 M=64 可跑", False, repr(ex)[:200])

# ---------------------------------------------------------------------------
print("=== P6: 编译对象跨 M 复用 ===")
try:
    g2 = rp.RouteBGemm(512, N, K)
    g2.compiled = g.compiled  # 用 M=256 编译的对象跑 M=512
    A2 = (torch.randn(512, K, device=dev) * 0.5).half()
    aq2, asf2 = g2.set_A(A2)
    g2.set_W_prepacked(wq, ws)
    g2.run()
    torch.cuda.synchronize()
    o2 = g2.out().float()
    r2 = rp.dequant_ref(aq2, asf2, wq, ws)
    rel2 = (o2 - r2).abs().max() / r2.abs().max()
    check("P6 M=256 编译对象可跑 M=512 且正确", rel2 < 0.02, f"rel_err={rel2:.5f}")
except Exception as ex:
    print(f"ℹ️  P6 复用失败（驱动将逐 M 编译）: {repr(ex)[:150]}")

# ---------------------------------------------------------------------------
print("=== P7: 计时 sanity（vs run_bs testing.benchmark） ===")
ms_mine = rp.time_ms(g.run, warmup=10, iters=50, rounds=3)
print(f"  我方事件计时: {ms_mine:.4f} ms -> {rp.tflops(M, N, K, ms_mine):.1f} TFLOPS")
try:
    t_us = pp.run_bs((M, N, K, 1), cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
                     cutlass.Float8E8M0FNU, 32, cutlass.Float16, cutlass.Float32,
                     'k', 'k', 'n', (128, 128, 128), (128, 128), 1e-1, 5, 10, True)
    print(f"  run_bs benchmark: {t_us:.2f} us -> {rp.tflops(M, N, K, t_us / 1e3):.1f} TFLOPS")
    ratio = (t_us / 1e3) / ms_mine
    check("P7 两口径差 <15%", 0.85 < ratio < 1.15, f"ratio={ratio:.3f}")
except Exception as ex:
    check("P7 run_bs 对照", False, repr(ex)[:150])

print("\n==== 探针结论 ====")
if FAIL:
    print("❌ 失败项:", FAIL)
    sys.exit(1)
print("✅ 全部通过 —— routeb_pipe 布局与数值验证 OK，可进入全量矩阵")
