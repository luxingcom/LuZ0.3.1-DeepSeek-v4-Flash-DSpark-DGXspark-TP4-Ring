#!/usr/bin/env python3
"""G5：大 K tile（256=8 组）3D 结构——组内归约 + 3D broadcast，期望保持向量化加载。
另含 G1/G2 双 pass 拆分作为对照。"""
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
def _G5(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """BLOCK_K=256（8 个 32 组）。3D 归约 + 3D broadcast + 正确的 int32 累加语义。"""
    NG: tl.constexpr = BLOCK_K // 32
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak,
                mask=mask_m[:, None], other=0.0)
    a3 = tl.reshape(a, [BLOCK_M, NG, 32])
    a3a = tl.abs(a3)
    amax = tl.max(a3a, axis=2)                       # [BM, NG]
    safe = tl.maximum(amax, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    e8m0_u8 = e8m0_f.to(tl.uint8)

    # SF swizzle store [BM, NG]
    kg = pid_k * NG + tl.arange(0, NG)
    sf_off = (pid_m * REST_K * 512 + (kg[None, :] // 4) * 512
              + (local[:, None] % 32) * 16 + (local[:, None] // 32) * 4
              + (kg[None, :] % 4))
    tl.store(SF_ptr + sf_off, e8m0_u8, mask=mask_m[:, None])

    inv = tl.exp2(127.0 - e8m0_f)                    # [BM, NG]
    s3 = a3 * inv[:, :, None]                        # [BM, NG, 32]
    s3a = tl.abs(s3)
    idx = tl.zeros([BLOCK_M, NG, 32], dtype=tl.int32)
    idx = idx + (s3a > 0.25).to(tl.int32)
    idx = idx + (s3a > 0.75).to(tl.int32)
    idx = idx + (s3a > 1.25).to(tl.int32)
    idx = idx + (s3a > 1.75).to(tl.int32)
    idx = idx + (s3a > 2.5).to(tl.int32)
    idx = idx + (s3a > 3.5).to(tl.int32)
    idx = idx + (s3a > 5.0).to(tl.int32)
    neg = (s3 < 0.0).to(tl.int32)
    nib = (idx + tl.where(idx > 0, neg * 8, 0)).to(tl.uint8)   # [BM, NG, 32]
    # 打包：[..., 32] -> [..., 16, 2] -> sum -> [..., 16]
    nib4 = tl.reshape(nib.to(tl.int32), [BLOCK_M, NG, 16, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 1, 2])
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 1, 2])
    lo = tl.sum(nib4 * sel, axis=3)
    hi = tl.sum(nib4 * sel2, axis=3)
    packed = (lo | (hi << 4)).to(tl.uint8)           # [BM, NG, 16]
    packed2 = tl.reshape(packed, [BLOCK_M, BLOCK_K // 2])
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed2, mask=mask_m[:, None])


print("=== G5 大 K tile（K=4096） ===")
for M in [1024, 4096, 16384]:
    K = 4096
    A = (torch.randn(M, K, device=dev) * 0.5).half()
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    rest_k = (K // 32 + 3) // 4
    sf = torch.empty(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
    bw = lambda t: (M * K * 2 + M * K // 2 + M * K // 32) / (t * 1e-3) / 1e9
    for bm, warps in [(32, 4), (64, 4), (64, 8), (128, 8), (32, 2)]:
        g = (triton.cdiv(M, bm), triton.cdiv(K, 256))
        try:
            t = time_ms(lambda: _G5[g](A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
                                        BLOCK_M=bm, BLOCK_K=256, num_warps=warps,
                                        num_stages=2))
            print(f"M={M:6d} BM={bm:4d} w{warps}: {t*1e3:8.1f}us ({bw(t):5.0f}GB/s)",
                  flush=True)
        except Exception as ex:
            print(f"M={M} BM={bm} w{warps}: FAIL {repr(ex)[:80]}")
    print()

print("=== G5 数值校验（vs torch 参考，±0 归一） ===")


def nz(p):
    p = p.to(torch.int32).flatten()
    return torch.where((p & 7) == 0, torch.zeros_like(p), p)


M, K = 1024, 4096
A = (torch.randn(M, K, device=dev) * 0.5).half()
aq = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
rest_k = (K // 32 + 3) // 4
sf = torch.zeros(((M + 127) // 128) * rest_k * 512, dtype=torch.uint8, device=dev)
_G5[(triton.cdiv(M, 64), triton.cdiv(K, 256))](
    A, aq, sf, M, K, rest_k, A.stride(0), A.stride(1),
    BLOCK_M=64, BLOCK_K=256, num_warps=4, num_stages=2)
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
print("TUNE7_DONE")
