#!/usr/bin/env python3
"""K2 数值失配定位。"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
from quant_tune8 import _K1, _K2  # 复用

dev = "cuda"
M, K = 256, 4096
torch.manual_seed(7)
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
e = e.clamp(0, 255)
sfv = torch.pow(2.0, e.float() - 127.0).unsqueeze(-1)
xn = torch.clamp(xb / sfv, -6.0, 6.0)
mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
idx = (xn.abs().unsqueeze(-1) - mag).abs().argmin(-1)
nib = idx | ((xn < 0).long() * 8)
ref_pk = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8).reshape(M, K // 2)

# 解包 got 与 ref 的 nibble 对比
p_g = aq.to(torch.int32)
p_r = ref_pk.to(torch.int32)
lo_g = p_g & 0xF
hi_g = (p_g >> 4) & 0xF
lo_r = p_r & 0xF
hi_r = (p_r >> 4) & 0xF
# ±0 归一（code 0/8 视为等价）
def n0(x):
    return torch.where((x & 7) == 0, torch.zeros_like(x) & 0xF, x)
mism = ((n0(lo_g) != n0(lo_r)) | (n0(hi_g) != n0(hi_r)))
print("mismatch bytes:", mism.sum().item(), "/", mism.numel())
pos = mism.nonzero()[:8]
for p in pos:
    m, kb = p.tolist()
    k0, k1 = kb * 2, kb * 2 + 1
    kg = k0 // 32
    print(f"  m={m} k0={k0} k1={k1} sf={sfl[m,kg].item()} a0={A[m,k0].item():.6f}"
          f" a1={A[m,k1].item():.6f}")
    print(f"    xn0={(Af[m,k0]/sfv[m,kg//32,0]).item():.8f}"
          f" got_lo={lo_g[m,kb].item()} ref_lo={lo_r[m,kb].item()}"
          f" | xn1={(Af[m,k1]/sfv[m,kg//32,0]).item():.8f}"
          f" got_hi={hi_g[m,kb].item()} ref_hi={hi_r[m,kb].item()}")
    d0 = (Af[m, k0].item() / (2.0 ** (sfl[m, kg].item() - 127)))
    print(f"    f32 除法 xn0={d0:.8f} |a|={abs(d0):.8f}")
