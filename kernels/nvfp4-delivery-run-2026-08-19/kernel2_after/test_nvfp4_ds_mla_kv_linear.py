import torch
import pytest

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as triton_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("T", [1, 4, 16, 64, 256, 1024, 4096])
def test_nvfp4_ds_mla_kv_linear(T):
    if DEVICE != "cuda":
        pytest.skip("需要 CUDA（DGX Spark / SM121）")
    torch.manual_seed(0)
    kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5  # [0:512]=K, [512:]=V

    ref = ref_impl(kv)
    out = triton_impl(kv)

    assert out.shape == (T, 584)
    assert out.dtype == torch.uint8
    # 逐字节精确一致（量化路径纯确定性）
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
