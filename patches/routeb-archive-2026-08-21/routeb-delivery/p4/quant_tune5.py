#!/usr/bin/env python3
"""溢出诊断 + 结构变体：n_regs/n_spills + BLOCK_M/warps 扫描。"""
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
def _qsmall(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
            BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """小 BLOCK 结构变体（完整逻辑，无 mask 版）。"""
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
    a_scale = tl.exp2(e8m0_f - 127.0)
    a_scaled = a / a_scale[:, None]
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
    lo = tl.reshape(nibble.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed)


print("=== 溢出诊断 ===")
M, K = 4096, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)

# 当前融合 kernel（BLOCK_M=128）的 regs/spills
g = (triton.cdiv(M, 128), triton.cdiv(K, 32))
h = rp._a_quant_fused_kernel[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
torch.cuda.synchronize()
for dev_fn, name in [(h, "fused BM128 w8")]:
    try:
        print(f"{name}: n_regs={dev_fn.n_regs} n_spills={dev_fn.n_spills}")
    except Exception as ex:
        print(f"{name}: metadata 不可用 {ex}")

print("\n=== 结构扫描（完整逻辑, M=4096, K=4096） ===")
for bm, warps in [(128, 8), (128, 4), (64, 4), (64, 2), (32, 2), (32, 1), (256, 8)]:
    gg = (triton.cdiv(M, bm), triton.cdiv(K, 32))
    try:
        t = time_ms(lambda: _qsmall[gg](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                        BLOCK_M=bm, BLOCK_K=32, num_warps=warps,
                                        num_stages=2))
        bw = (M * K * 2 + M * K // 2) / (t * 1e-3) / 1e9
        print(f"BM={bm:4d} w{warps}: {t*1e3:8.1f}us ({bw:5.0f}GB/s)", flush=True)
    except Exception as ex:
        print(f"BM={bm} w{warps}: FAIL {repr(ex)[:100]}")
print("TUNE5_DONE")
