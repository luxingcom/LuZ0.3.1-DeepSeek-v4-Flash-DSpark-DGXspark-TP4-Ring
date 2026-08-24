"""cudnn backend + 2D scale 探针：能否跑通原生 FP4 GEMM + 数值对照。"""
import torch
import time
import flashinfer

DEVICE = "cuda"
M, K, N = 256, 4096, 4096
torch.manual_seed(0)
A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)

# 用 flashinfer 原生量化（bf16 输入）
a_gsf = torch.ones(1, device=DEVICE)
a_q, a_sf = flashinfer.nvfp4_quantize(A.bfloat16(), a_gsf, sf_vec_size=16)  # a_q[M,K//2], a_sf[M,K//16] 2D

# W：K 向打包 [K//2, N]
W_n = (torch.rand(K, N // 2, device=DEVICE) * 255).to(torch.uint8)
lo = W_n & 0x0F
hi = (W_n >> 4) & 0x0F
nib = torch.empty(K, N, dtype=torch.uint8, device=DEVICE)
nib[:, 0::2] = lo
nib[:, 1::2] = hi
nib_t = nib.t().contiguous()
w_b = (nib_t[:, 0::2] | (nib_t[:, 1::2] << 4)).t().contiguous()  # [K//2, N]
# b scale：先算 W 的 16 组 e4m3 scale（模拟值）
b_sf = (torch.rand(N, K // 16, device=DEVICE) * 255).to(torch.uint8)

print("a_q", tuple(a_q.shape), "a_sf", tuple(a_sf.shape), a_sf.dtype)
print("w_b", tuple(w_b.shape), "b_sf", tuple(b_sf.shape))

# cudnn：尝试各种 scale 布局组合
for name, a_s in [("a_sf_raw", a_sf.contiguous()),
                  ("a_sf_swz2d", flashinfer.block_scale_interleave(a_sf).view(M, K // 16))]:
    for bname, b_s in [("b_sf_raw", b_sf.contiguous()),
                       ("b_sf_swz2d", flashinfer.block_scale_interleave(b_sf).view(N, K // 16))]:
        try:
            out = flashinfer.mm_fp4(a_q, w_b, a_s, b_s, alpha=None, block_size=16, backend="cudnn")
            print(f"{name}+{bname}: OK shape={tuple(out.shape)} sum={out.float().sum().item():.2f}")
        except Exception as e:
            print(f"{name}+{bname}: {type(e).__name__}: {str(e)[:80]}")
