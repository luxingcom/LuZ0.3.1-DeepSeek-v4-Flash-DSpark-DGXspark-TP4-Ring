"""mm_fp4 各 backend 可用性探针（找非 cute-dsl 的可运行路径）。"""
import torch
import flashinfer

DEVICE = "cuda"
M, K, N = 256, 4096, 4096
torch.manual_seed(0)
A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
a_gsf = torch.ones(1, device=DEVICE)
a_q, a_sf = flashinfer.nvfp4_quantize(A.bfloat16(), a_gsf, sf_vec_size=16)
W_n = (torch.rand(K, N // 2, device=DEVICE) * 255).to(torch.uint8)
w_k = W_n.t().contiguous()  # [N//2? no] 占位形状即可（先测 backend 是否可用）
# b 形状契约 [K//2, N]：用随机打包
w_b = (torch.rand(K // 2, N, device=DEVICE) * 255).to(torch.uint8)
b_sf_raw = (torch.rand(N, K // 16, device=DEVICE) * 255).to(torch.uint8)
b_sf = flashinfer.block_scale_interleave(b_sf_raw)
a_sf_swz = flashinfer.block_scale_interleave(a_sf)

for backend in ["cudnn", "trtllm", "cutlass", "cute-dsl", "b12x"]:
    try:
        out = flashinfer.mm_fp4(a_q, w_b, a_sf_swz, b_sf, alpha=None, block_size=16, backend=backend)
        print(f"backend={backend}: OK shape={tuple(out.shape)} dtype={out.dtype}")
    except Exception as e:
        print(f"backend={backend}: {type(e).__name__}: {str(e)[:90]}")
