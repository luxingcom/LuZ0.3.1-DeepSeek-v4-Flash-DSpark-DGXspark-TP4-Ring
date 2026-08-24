#!/usr/bin/env python3
"""数学部分二分：tl.max 归约+broadcast / SF 数学 / 阈值比较 各自代价。"""
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
def _vB(A_ptr, O_ptr, M, K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """A: loadstore + tl.max(axis=1) 归约并 broadcast 回 [BM,BK]。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)
    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)                       # 归约
    scale = tl.exp2(127.0 - tl.floor(tl.log2(block_max)))   # 依赖归约结果
    a2 = a * scale[:, None]                                 # broadcast 回
    idx = ((a2 > 0.25).to(tl.uint8) | ((a2 > 0.75).to(tl.uint8) << 1)
           | ((a2 > 1.25).to(tl.uint8) << 2))
    lo = tl.reshape(idx.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(O_ptr + pk, packed)


@triton.jit
def _vC(A_ptr, O_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """C: A + 完整 SF 数学 + SF swizzle store（但不对 a 缩放，隔离 broadcast-mul）。"""
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
    e8m0_u8 = e8m0_f.to(tl.uint8)
    sf_off = (pid_m * REST_K * 512 + (pid_k // 4) * 512
              + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4))
    tl.store(SF_ptr + sf_off, e8m0_u8)
    idx = ((a > 0.25).to(tl.uint8) | ((a > 0.75).to(tl.uint8) << 1)
           | ((a > 1.25).to(tl.uint8) << 2))
    lo = tl.reshape(idx.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(O_ptr + pk, packed)


@triton.jit
def _vD(A_ptr, O_ptr, M, K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """D: A + 7 段阈值比较（无归约/SF/broadcast）——隔离比较链代价。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)
    a_abs_s = tl.abs(a)
    idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    idx = idx + (a_abs_s > 0.25).to(tl.int32)
    idx = idx + (a_abs_s > 0.75).to(tl.int32)
    idx = idx + (a_abs_s > 1.25).to(tl.int32)
    idx = idx + (a_abs_s > 1.75).to(tl.int32)
    idx = idx + (a_abs_s > 2.5).to(tl.int32)
    idx = idx + (a_abs_s > 3.5).to(tl.int32)
    idx = idx + (a_abs_s > 5.0).to(tl.int32)
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
    tl.store(O_ptr + pk, packed)


print("=== 数学二分（K=4096, 无 mask, M%128==0） ===")
for M in [4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    rest_k = (K // 32 + 3) // 4
    sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
    g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
    tB = time_ms(lambda: _vB[g](A, aq, M, K, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    tC = time_ms(lambda: _vC[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    tD = time_ms(lambda: _vD[g](A, aq, M, K, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    bw = lambda t: (M * K * 2 + M * K // 2) / (t * 1e-3) / 1e9
    print(f"M={M:6d}  B(归约+bcast) {tB*1e3:8.1f}us ({bw(tB):5.0f}GB/s) |"
          f" C(SF数学+store) {tC*1e3:8.1f}us ({bw(tC):5.0f}GB/s) |"
          f" D(7比较链) {tD*1e3:8.1f}us ({bw(tD):5.0f}GB/s)", flush=True)

print("TUNE3_DONE")
