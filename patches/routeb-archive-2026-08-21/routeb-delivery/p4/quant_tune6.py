#!/usr/bin/env python3
"""F 系列：B 结构扩展到完整量化语义（uint8 OR 链 / int32 / 双加载）。"""
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
def _F1(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """B 结构 + 完整语义：uint8 OR 链比较 + SF store + inv-scale 乘法。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)
    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)
    safe_max = tl.maximum(block_max, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe_max / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    tl.store(SF_ptr + (pid_m * REST_K * 512 + (pid_k // 4) * 512
                       + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4)),
             e8m0_f.to(tl.uint8))
    inv = tl.exp2(127.0 - e8m0_f)
    a2 = a * inv[:, None]
    b = a2 < 0.0
    a2a = tl.abs(a2)
    idx = ((a2a > 0.25).to(tl.uint8) | ((a2a > 0.75).to(tl.uint8) << 1)
           | ((a2a > 1.25).to(tl.uint8) << 2) | ((a2a > 1.75).to(tl.uint8) << 3)
           | ((a2a > 2.5).to(tl.uint8) << 4) | ((a2a > 3.5).to(tl.uint8) << 5)
           | ((a2a > 5.0).to(tl.uint8) << 6))
    nib = tl.where(idx > 0, idx | (b.to(tl.uint8) << 3), idx)
    lo = tl.reshape(nib.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed)


@triton.jit
def _F4(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """双加载：独立加载喂归约（希望主加载保持向量化）。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ptrs = A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
    a_red = tl.load(ptrs)
    block_max = tl.max(tl.abs(a_red), axis=1)
    safe_max = tl.maximum(block_max, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe_max / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    tl.store(SF_ptr + (pid_m * REST_K * 512 + (pid_k // 4) * 512
                       + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4)),
             e8m0_f.to(tl.uint8))
    a = tl.load(ptrs)
    inv = tl.exp2(127.0 - e8m0_f)
    a2 = a * inv[:, None]
    b = a2 < 0.0
    a2a = tl.abs(a2)
    idx = ((a2a > 0.25).to(tl.uint8) | ((a2a > 0.75).to(tl.uint8) << 1)
           | ((a2a > 1.25).to(tl.uint8) << 2) | ((a2a > 1.75).to(tl.uint8) << 3)
           | ((a2a > 2.5).to(tl.uint8) << 4) | ((a2a > 3.5).to(tl.uint8) << 5)
           | ((a2a > 5.0).to(tl.uint8) << 6))
    nib = tl.where(idx > 0, idx | (b.to(tl.uint8) << 3), idx)
    lo = tl.reshape(nib.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed)


print("=== F 系列（K=4096, 无 mask, M%128==0） ===")
for M in [1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    rest_k = (K // 32 + 3) // 4
    sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
    g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
    bw = lambda t: (M * K * 2 + M * K // 2) / (t * 1e-3) / 1e9
    t1 = time_ms(lambda: _F1[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    t4 = time_ms(lambda: _F4[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    print(f"M={M:6d}  F1(B结构全语义) {t1*1e3:8.1f}us ({bw(t1):5.0f}GB/s) |"
          f" F4(双加载) {t4*1e3:8.1f}us ({bw(t4):5.0f}GB/s)", flush=True)

print("\n=== F1 数值校验（vs torch 参考，±0 归一） ===")


def nz(p):
    p = p.to(torch.int32).flatten()
    return torch.where((p & 7) == 0, torch.zeros_like(p), p)


M, K = 1024, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf = torch.zeros(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
_F1[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
torch.cuda.synchronize()
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
m_ar = torch.arange(M, device=dev)
kg = torch.arange(K // 32, device=dev)
off = ((m_ar // 128)[:, None] * (rest_k * 512) + (kg // 4)[None, :] * 512
       + (m_ar % 32)[:, None] * 16 + ((m_ar // 32) % 4)[:, None] * 4 + (kg % 4)[None, :])
sf2d = sf.view(-1)[off.reshape(-1)].reshape(M, K // 32)
print("packed equal(±0):", torch.equal(nz(aq), nz(ref_pk)),
      " sf equal:", torch.equal(sf2d, ref_sf))
print("TUNE6_DONE")
