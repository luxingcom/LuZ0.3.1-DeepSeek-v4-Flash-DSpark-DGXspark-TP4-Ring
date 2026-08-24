"""v17 信封 ↔ FlashInfer nvfp4_kv_quantize 布局对照（kernel② 替换的消费端兼容性）。"""
import torch
import flashinfer

DEVICE = "cuda"
T = 4
torch.manual_seed(0)
kv = (torch.randn(T, 1024, device=DEVICE) * 0.5).bfloat16()

from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl
out17 = v17_impl(kv.float())

try:
    gsf = torch.ones(1, device=DEVICE)
    kq, ksf = flashinfer.nvfp4_kv_quantize(kv[:, :512].contiguous(), gsf)
    print("nvfp4_kv_quantize k:", tuple(kq.shape), kq.dtype, "scale:", tuple(ksf.shape), ksf.dtype)
    # 对照 v17 的 K 段：data[0:256] K packed + scale[512:544] K E8M0
    print("v17 K data[0:8] =", out17[0, :8].tolist())
    print("fi  K quant[0:8]=", kq[0, :8].tolist() if kq.ndim == 2 else kq.flatten()[:8].tolist())
    if kq.numel() >= 256:
        # 若 fi 也是 256B K 段 + scale，比较 scale 编码
        print("v17 K scale[512:516] =", out17[0, 512:516].tolist())
        s_flat = ksf.flatten()[:4].tolist()
        print("fi  K scale[0:4]    =", s_flat, "dtype=", ksf.dtype)
except Exception as e:
    print(f"nvfp4_kv_quantize FAIL: {type(e).__name__}: {str(e)[:200]}")
