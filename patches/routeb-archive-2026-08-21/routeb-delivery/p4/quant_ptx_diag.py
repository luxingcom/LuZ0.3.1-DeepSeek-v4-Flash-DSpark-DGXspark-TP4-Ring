#!/usr/bin/env python3
"""PTX 诊断：对比 full vs B 变体的加载指令宽度与 convert_layout 数量。"""
import os
import re
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl
import routeb_pipe as rp

dev = "cuda"
M, K = 256, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)

# full kernel
h_full = rp._a_quant_fused_kernel[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)


@triton.jit
def _vB(A_ptr, O_ptr, M, K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak)
    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)
    scale = tl.exp2(127.0 - tl.floor(tl.log2(block_max)))
    a2 = a * scale[:, None]
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


h_b = _vB[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq, M, K, A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
    num_warps=8, num_stages=2)
torch.cuda.synchronize()


def diag(h, name):
    ttgir = h.asm["ttgir"]
    ptx = h.asm["ptx"]
    n_cvt = len(re.findall(r"convert_layout", ttgir))
    loads = re.findall(r"ld\.global[.\w]*", ptx)
    from collections import Counter
    lc = Counter(loads)
    stores = re.findall(r"st\.global[.\w]*", ptx)
    sc = Counter(stores)
    print(f"--- {name}: convert_layout={n_cvt}")
    print(f"    loads: {dict(lc)}")
    print(f"    stores: {dict(sc)}")
    # shared memory usage
    m = re.search(r"\.shared .*,(\d+)", ptx)
    smem = re.findall(r"\.shared \w+ \w+\[(\d+)\]", ptx)
    print(f"    smem allocs: {smem[:6]}")


diag(h_full, "FULL (slow)")
diag(h_b, "B (fast)")
