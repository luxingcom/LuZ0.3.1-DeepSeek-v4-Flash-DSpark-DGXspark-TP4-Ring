# ============================================================================
# test_nvfp4_ds_mla_kv_linear_v17.py —— v17 正确性（7 组 T 逐字节 atol=0）
# 参考：v11 torch 参考实现（金标准语义）
# ============================================================================
import torch
import pytest

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl

T_SIZES = [1, 4, 16, 64, 256, 1024, 4096]


def _rand_kv(T, device="cuda", seed=42):
    torch.manual_seed(seed)
    kv = torch.randn(T, 1024, device=device, dtype=torch.float32) * 2.0
    # 注入大/小值考验 scale 边界
    if T > 0:
        kv[0, :8] = kv[0, :8] * 100.0
        kv[T // 2, 500:512] = torch.randn(12, device=device) * 1e-3
    return kv


@pytest.mark.parametrize("T", T_SIZES)
def test_v17_bit_exact(T):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = _rand_kv(T, device)
    ref = ref_impl(kv)
    got = v17_impl(kv)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert torch.equal(got, ref), (
        f"T={T} 逐字节不一致: "
        f"mismatch={(got != ref).sum().item()}/{got.numel()} "
        f"max_diff={(got.float() - ref.float()).abs().max().item()}"
    )


def test_v17_kv_entry():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = torch.randn(64, 512, device=device)
    v = torch.randn(64, 512, device=device)
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear_kv
    out = nvfp4_ds_mla_kv_linear_kv(k, v)
    assert out.shape == (64, 584)
    assert out.dtype == torch.uint8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
