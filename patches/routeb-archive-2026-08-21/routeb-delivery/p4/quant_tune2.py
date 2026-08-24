#!/usr/bin/env python3
"""量化 kernel 根因隔离：mask 加载 vs SF 数学 vs reshape 打包 vs launch 开销。"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl
import routeb_pipe as rp

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
def _empty(A_ptr, M):
    pid = tl.program_id(0)


@triton.jit
def _k_loadstore(A_ptr, O_ptr, M, K, stride_am, stride_ak,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """纯 load + 打包 store（无 SF 数学，无 reshape）——隔离访存。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)  # M%128==0 无 mask
    # 简单阈值量化（无除法/SF）
    idx = ((a > 0).to(tl.uint8) | ((a > 0.5).to(tl.uint8) << 1)
           | ((a > 1.0).to(tl.uint8) << 2))
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
def _k_full_nomask(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K,
                   stride_am, stride_ak,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """完整 kernel 但无 mask（M%128==0）——隔离 mask 影响。"""
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
    a_scaled = a * tl.exp2(127.0 - e8m0_f)[:, None]  # 乘法替代除法
    a_abs_s = tl.abs(a_scaled)
    idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    idx = idx + (a_abs_s > 0.25).to(tl.int32)
    idx = idx + (a_abs_s > 0.75).to(tl.int32)
    idx = idx + (a_abs_s > 1.25).to(tl.int32)
    idx = idx + (a_abs_s > 1.75).to(tl.int32)
    idx = idx + (a_abs_s > 2.5).to(tl.int32)
    idx = idx + (a_abs_s > 3.5).to(tl.int32)
    idx = idx + (a_abs_s > 5.0).to(tl.int32)
    neg_mask = (a_scaled < 0.0).to(tl.int32)
    sign_bit = tl.where(idx > 0, neg_mask * 8, 0)
    nibble = (idx + sign_bit).to(tl.uint8)
    nibble_i32 = nibble.to(tl.int32)
    nibble_3d = tl.reshape(nibble_i32, [BLOCK_M, BLOCK_K // 2, 2])
    sel_lo_r = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    sel_hi_r = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(nibble_3d * sel_lo_r, axis=2)
    hi_val = tl.sum(nibble_3d * sel_hi_r, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk_off = (offs_m[:, None].to(tl.int64) * (K // 2)
              + pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk_off, packed)


print("=== 根因隔离（K=4096, M%128==0） ===")
A_small = torch.zeros(16, device=dev)
t_empty = time_ms(lambda: _empty[(1,)](A_small, 16))
print(f"空核 launch 开销: {t_empty*1e3:.1f} us")

for M in [1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    rest_k = (K // 32 + 3) // 4
    sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)

    t_base = time_ms(lambda: rp._a_quant_fused_kernel[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
        A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    t_ls = time_ms(lambda: _k_loadstore[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
        A, aq, M, K, A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
        num_warps=8, num_stages=2))
    t_nm = time_ms(lambda: _k_full_nomask[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
        A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    bw = lambda t: (M * K * 2 + M * K // 2) / (t * 1e-3) / 1e9
    print(f"M={M:6d}  base(mask+div) {t_base*1e3:8.1f}us ({bw(t_base):5.0f}GB/s) |"
          f" loadstore-only {t_ls*1e3:8.1f}us ({bw(t_ls):5.0f}GB/s) |"
          f" nomask+mul {t_nm*1e3:8.1f}us ({bw(t_nm):5.0f}GB/s)", flush=True)

print("\n=== nomask+mul 数值校验（vs base, ±0 归一） ===")


def nz(p):
    p = p.to(torch.int32).flatten()
    return torch.where((p & 7) == 0, torch.zeros_like(p), p)


M, K = 1024, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq1 = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
aq2 = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf1 = torch.zeros(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
sf2 = torch.zeros_like(sf1)
rp._a_quant_fused_kernel[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq1, sf1, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
_k_full_nomask[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq2, sf2, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
torch.cuda.synchronize()
print("packed equal(±0):", torch.equal(nz(aq1), nz(aq2)),
      " sf equal:", torch.equal(sf1, sf2))
print("TUNE2_DONE")
