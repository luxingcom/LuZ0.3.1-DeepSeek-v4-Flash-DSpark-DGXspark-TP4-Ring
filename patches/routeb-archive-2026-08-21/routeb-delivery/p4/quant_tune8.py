#!/usr/bin/env python3
"""K 系列：双 pass 拆分——K1 纯 SF（归约），K2 量化（SF 从 load 来，无归约）。"""
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
def _K1(A_ptr, SFL_ptr, M, K, sf_stride_m, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """SF pass：load + 组 max + E8M0 编码 + 逻辑 SF store。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak,
                mask=mask_m[:, None], other=0.0)
    amax = tl.max(tl.abs(a), axis=1)
    safe = tl.maximum(amax, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    tl.store(SFL_ptr + offs_m * sf_stride_m + pid_k, e8m0_f.to(tl.uint8),
             mask=mask_m)


@triton.jit
def _K2(A_ptr, SFL_ptr, Aq_ptr, M, K, sf_stride_m, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, THRESH_FOLD: tl.constexpr):
    """量化 pass：load A + load SF + D 结构比较链 + 打包 store。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak,
                mask=mask_m[:, None], other=0.0)
    sf = tl.load(SFL_ptr + offs_m * sf_stride_m + pid_k, mask=mask_m, other=127)
    inv = tl.exp2(127.0 - sf.to(tl.float32))
    if THRESH_FOLD:
        a_abs = tl.abs(a)
        idx = ((a_abs > (0.25 * inv)[:, None]).to(tl.int32)
               + (a_abs > (0.75 * inv)[:, None]).to(tl.int32)
               + (a_abs > (1.25 * inv)[:, None]).to(tl.int32)
               + (a_abs > (1.75 * inv)[:, None]).to(tl.int32)
               + (a_abs > (2.5 * inv)[:, None]).to(tl.int32)
               + (a_abs > (3.5 * inv)[:, None]).to(tl.int32)
               + (a_abs > (5.0 * inv)[:, None]).to(tl.int32))
        neg = (a < 0.0).to(tl.int32)
    else:
        a2 = a * inv[:, None]
        a2a = tl.abs(a2)
        idx = ((a2a > 0.25).to(tl.int32) + (a2a > 0.75).to(tl.int32)
               + (a2a > 1.25).to(tl.int32) + (a2a > 1.75).to(tl.int32)
               + (a2a > 2.5).to(tl.int32) + (a2a > 3.5).to(tl.int32)
               + (a2a > 5.0).to(tl.int32))
        neg = (a2 < 0.0).to(tl.int32)
    nib = (idx + tl.where(idx > 0, neg * 8, 0)).to(tl.uint8)
    lo = tl.reshape(nib.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed, mask=mask_m[:, None])


print("=== K 双 pass（K=4096, BM=128 w8） ===")
for M in [1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    sfl = torch.empty(M, K // 32, dtype=torch.uint8, device=dev)
    g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
    bw = lambda t: (M * K * 2 + M * K // 2 + M * K // 32) / (t * 1e-3) / 1e9

    def k1():
        _K1[g](A, sfl, M, K, sfl.stride(0), A.stride(0), A.stride(1),
               BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)

    def k2():
        _K2[g](A, sfl, aq, M, K, sfl.stride(0), A.stride(0), A.stride(1),
               BLOCK_M=128, BLOCK_K=32, THRESH_FOLD=0, num_warps=8, num_stages=2)

    def k2b():
        _K2[g](A, sfl, aq, M, K, sfl.stride(0), A.stride(0), A.stride(1),
               BLOCK_M=128, BLOCK_K=32, THRESH_FOLD=1, num_warps=8, num_stages=2)

    t1 = time_ms(k1)
    t2 = time_ms(k2)
    t2b = time_ms(k2b)
    tot = t1 + t2
    print(f"M={M:6d}  K1 {t1*1e3:7.1f}us | K2 {t2*1e3:7.1f}us | K2b(阈值折叠) {t2b*1e3:7.1f}us"
          f" | K1+K2 {tot*1e3:7.1f}us (等效 {bw(tot):4.0f}GB/s)", flush=True)

print("\n=== K 双 pass 数值校验（vs torch 参考，±0 归一） ===")


def nz(p):
    p = p.to(torch.int32).flatten()
    return torch.where((p & 7) == 0, torch.zeros_like(p), p)


M, K = 1024, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
sfl = torch.zeros(M, K // 32, dtype=torch.uint8, device=dev)
g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
_K1[g](A, sfl, M, K, sfl.stride(0), A.stride(0), A.stride(1),
       BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
_K2[g](A, sfl, aq, M, K, sfl.stride(0), A.stride(0), A.stride(1),
       BLOCK_M=128, BLOCK_K=32, THRESH_FOLD=0, num_warps=8, num_stages=2)
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
print("sf equal:", torch.equal(sfl, ref_sf),
      " packed equal(±0):", torch.equal(nz(aq), nz(ref_pk)))
print("TUNE8_DONE")
