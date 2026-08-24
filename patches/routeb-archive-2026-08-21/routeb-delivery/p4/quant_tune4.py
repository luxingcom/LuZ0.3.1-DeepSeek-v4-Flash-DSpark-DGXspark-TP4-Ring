#!/usr/bin/env python3
"""重构量化 kernel：避免物化 a_scaled（scale 折进阈值）、fp32 路径、warp 数扫描。"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

dev = "cuda"


def time_ms(fn, warmup=20, iters=100, rounds=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) / iters)
    res.sort()
    return res[len(res) // 2]


@triton.jit
def _vE(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, USE_F32: tl.constexpr):
    """重构：scale 折进阈值（不物化 a_scaled），SF 数学在归约域。

    idx = sum_j [ |a| > thr_j * scale ]，thr = [.25,.75,1.25,1.75,2.5,3.5,5.0]
    等价于对 a/scale 的码本量化（严格大于语义与原版一致）。
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)
    if USE_F32:
        a = a.to(tl.float32)

    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)
    safe_max = tl.maximum(block_max, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe_max / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    e8m0_u8 = e8m0_f.to(tl.uint8)
    sf_off = (pid_m * REST_K * 512 + (pid_k // 4) * 512
              + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4))
    tl.store(SF_ptr + sf_off, e8m0_u8)

    inv_scale = tl.exp2(127.0 - e8m0_f)  # = 1/scale
    # 阈值预乘 inv_scale（[BLOCK_M] 向量）
    t0 = 0.25 * inv_scale
    t1 = 0.75 * inv_scale
    t2 = 1.25 * inv_scale
    t3 = 1.75 * inv_scale
    t4 = 2.5 * inv_scale
    t5 = 3.5 * inv_scale
    t6 = 5.0 * inv_scale

    idx = ((a_abs > t0[:, None]).to(tl.int32)
           + (a_abs > t1[:, None]).to(tl.int32)
           + (a_abs > t2[:, None]).to(tl.int32)
           + (a_abs > t3[:, None]).to(tl.int32)
           + (a_abs > t4[:, None]).to(tl.int32)
           + (a_abs > t5[:, None]).to(tl.int32)
           + (a_abs > t6[:, None]).to(tl.int32))
    neg = (a < 0.0).to(tl.int32)
    nib = (idx + tl.where(idx > 0, neg * 8, 0)).to(tl.uint8)
    lo = tl.reshape(nib.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed)


def run_variant(M, K, A, aq, sf, rest_k, warps, use_f32):
    g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
    return time_ms(lambda: _vE[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                  BLOCK_M=128, BLOCK_K=32, USE_F32=use_f32,
                                  num_warps=warps, num_stages=2))


print("=== 重构 kernel（K=4096, 无 mask, M%128==0） ===")
for M in [1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    rest_k = (K // 32 + 3) // 4
    sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
    bw = lambda t: (M * K * 2 + M * K // 2) / (t * 1e-3) / 1e9
    r = {}
    for warps in [4, 8]:
        for f32 in [0, 1]:
            t = run_variant(M, K, A, aq, sf, rest_k, warps, f32)
            r[(warps, f32)] = t
    parts = " ".join(f"| w{w}{'f32' if f else 'f16'} {t*1e3:7.1f}us ({bw(t):4.0f}GB/s)"
                     for (w, f), t in r.items())
    print(f"M={M:6d} {parts}", flush=True)

print("\n=== 数值校验（w8 f16 vs torch 参考，±0 归一） ===")


def nz(p):
    p = p.to(torch.int32).flatten()
    return torch.where((p & 7) == 0, torch.zeros_like(p), p)


M, K = 1024, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf = torch.zeros(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
_vE[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, USE_F32=0, num_warps=8, num_stages=2)
torch.cuda.synchronize()
# torch 参考
Af = A.float()
xb = Af.reshape(M, K // 32, 32)
amax = xb.abs().amax(-1)
e = torch.floor(torch.log2(torch.clamp(amax, min=1e-30) / 6.0)).long() + 127
ref_sf = e.clamp(0, 255).to(torch.uint8)
sfv = torch.pow(2.0, e.clamp(0, 255).float() - 127.0).unsqueeze(-1)
xn = torch.clamp(xb / sfv, -6.0, 6.0)
mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
idx = (xn.abs().unsqueeze(-1) - mag).abs().argmin(-1)
nib = idx | ((xn < 0).long() * 8)
ref_pk = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8).reshape(M, K // 2)
# sf 反 swizzle 校验（用 pipe 的映射）
import routeb_pipe as rp
sf2d = torch.zeros(M, K // 32, dtype=torch.uint8, device=dev)
mm = sf.view(-1)
m_ar = torch.arange(M, device=dev)
kg = torch.arange(K // 32, device=dev)
off = ((m_ar // 128)[:, None] * (rest_k * 512) + (kg // 4)[None, :] * 512
       + (m_ar % 32)[:, None] * 16 + ((m_ar // 32) % 4)[:, None] * 4 + (kg % 4)[None, :])
sf2d = mm[off.reshape(-1)].reshape(M, K // 32)
print("packed equal(±0):", torch.equal(nz(aq), nz(ref_pk)),
      " sf equal:", torch.equal(sf2d, ref_sf))
print("TUNE4_DONE")
