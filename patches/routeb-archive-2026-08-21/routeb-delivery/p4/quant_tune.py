#!/usr/bin/env python3
"""量化 kernel 微基准：定位 routeB 融合量化慢的根因。

对照：
- vLLM scaled_fp4_quant（routeA 的 A 量化，目标基准）
- 融合 kernel 变体：evict_last vs 无 / BLOCK_K=32 vs 64（128B 连续行加载）/ warps
- 纯拷贝带宽参考
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl
import routeb_pipe as rp
import vllm._custom_ops as _co

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
def _qf_v2(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K,
           stride_am, stride_ak,
           BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, EVICT: tl.constexpr):
    """变体：EVICT=0 无 evict 策略；BLOCK_K=64 时每行加载 128B 连续。"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    if EVICT:
        a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                    + offs_k[None, :].to(tl.int64) * stride_ak,
                    mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
    else:
        a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                    + offs_k[None, :].to(tl.int64) * stride_ak,
                    mask=mask_m[:, None], other=0.0)

    if BLOCK_K == 32:
        a_abs = tl.abs(a)
        block_max = tl.max(a_abs, axis=1)
    else:
        # BLOCK_K=64: 两个 32 组分别取 max（reshape [BM, G, 32]）
        a3 = tl.reshape(a, [BLOCK_M, BLOCK_K // 32, 32])
        block_max = tl.max(tl.abs(a3), axis=2)  # [BLOCK_M, BLOCK_K//32]

    safe_max = tl.maximum(block_max, 1e-38)
    log2_val = tl.log2(safe_max / 6.0)
    e8m0_f = tl.floor(log2_val) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    e8m0_u8 = e8m0_f.to(tl.uint8)

    if BLOCK_K == 32:
        kg = pid_k
        sf_off = (pid_m * REST_K * 512 + (kg // 4) * 512
                  + (local % 32) * 16 + (local // 32) * 4 + (kg % 4))
        tl.store(SF_ptr + sf_off, e8m0_u8, mask=mask_m)
    else:
        kg = pid_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        sf_off = (pid_m * REST_K * 512 + (kg[None, :] // 4) * 512
                  + (local[:, None] % 32) * 16 + (local[:, None] // 32) * 4
                  + (kg[None, :] % 4))
        tl.store(SF_ptr + sf_off, e8m0_u8, mask=mask_m[:, None])

    a_scale = tl.exp2(e8m0_f - 127.0)
    if BLOCK_K == 32:
        a_scaled = a / a_scale[:, None]
    else:
        a_scaled = a / tl.reshape(
            tl.broadcast_to(a_scale[:, :, None], [BLOCK_M, BLOCK_K // 32, 32]),
            [BLOCK_M, BLOCK_K])
    a_abs_s = tl.abs(a_scaled)
    idx = tl.zeros_like(a_abs_s).to(tl.int32)
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
    tl.store(Aq_ptr + pk_off, packed, mask=mask_m[:, None])


print("=== 量化微基准（K=4096） ===")
gs = torch.tensor([1.0], dtype=torch.float32, device=dev)

for M in [64, 256, 1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq_flat = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    sf_k = K // 32
    rest_k = (sf_k + 3) // 4
    rest_m = (M + 127) // 128
    sf_flat = torch.empty(rest_m * rest_k * 512, dtype=torch.uint8, device=dev)

    # 拷贝带宽参考（同字节数：fp16 -> fp16）
    abuf = torch.empty_like(A)
    t_copy = time_ms(lambda: abuf.copy_(A))
    # vLLM quant（routeA 路径）
    t_vllm = time_ms(lambda: _co.scaled_fp4_quant(A, gs, True, "none", None))
    # 当前融合（evict_last, BK=32, w8）
    t_cur = time_ms(lambda: rp._a_quant_fused_kernel[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
        A, aq_flat, sf_flat, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2))
    # 变体
    t_v2_noevict = time_ms(lambda: _qf_v2[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
        A, aq_flat, sf_flat, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=32, EVICT=0, num_warps=8, num_stages=2))
    t_v2_bk64 = time_ms(lambda: _qf_v2[(triton.cdiv(M, 128), triton.cdiv(K, 64))](
        A, aq_flat, sf_flat, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=64, EVICT=0, num_warps=8, num_stages=2))
    t_v2_bk64w4 = time_ms(lambda: _qf_v2[(triton.cdiv(M, 128), triton.cdiv(K, 64))](
        A, aq_flat, sf_flat, M, K, rest_k, A.stride(0), A.stride(1),
        BLOCK_M=128, BLOCK_K=64, EVICT=0, num_warps=4, num_stages=2))

    bw = lambda t: (M * K * 2 + M * K // 2 + M * K // 32) / (t * 1e-3) / 1e9
    print(f"M={M:6d}  copy {t_copy*1e3:7.1f}us | vLLM {t_vllm*1e3:7.1f}us |"
          f" cur(evict,BK32) {t_cur*1e3:7.1f}us ({bw(t_cur):5.0f}GB/s) |"
          f" noevict {t_v2_noevict*1e3:7.1f}us | BK64 {t_v2_bk64*1e3:7.1f}us ({bw(t_v2_bk64):5.0f}GB/s) |"
          f" BK64w4 {t_v2_bk64w4*1e3:7.1f}us", flush=True)

# 数值正确性：BK64 变体 vs 当前融合（间接经 dequant）
print("\n=== BK64 变体数值校验 ===")
M, K = 256, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq1 = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
sf_k = K // 32
rest_k = (sf_k + 3) // 4
rest_m = (M + 127) // 128
sf1 = torch.zeros(rest_m * rest_k * 512, dtype=torch.uint8, device=dev)
rp._a_quant_fused_kernel[(triton.cdiv(M, 128), triton.cdiv(K, 32))](
    A, aq1, sf1, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2)
aq2 = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
sf2 = torch.zeros(rest_m * rest_k * 512, dtype=torch.uint8, device=dev)
_qf_v2[(triton.cdiv(M, 128), triton.cdiv(K, 64))](
    A, aq2, sf2, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=128, BLOCK_K=64, EVICT=0, num_warps=8, num_stages=2)
torch.cuda.synchronize()
print("packed equal:", torch.equal(aq1, aq2), " sf equal:", torch.equal(sf1, sf2))
print("QUANT_TUNE_DONE")
