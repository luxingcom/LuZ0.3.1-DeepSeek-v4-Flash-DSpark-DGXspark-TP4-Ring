"""v17 边界值实测：零/±0/-0/1e6/1e30/1e-30 的 scale 字节与 torch 参考对照。"""
import torch
from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl

DEVICE = "cuda"
cases = {
    "zeros": torch.zeros(4, 1024, device=DEVICE),
    "pos_zero": torch.full((4, 1024), 0.0, device=DEVICE),
    "neg_zero": torch.full((4, 1024), -0.0, device=DEVICE),
    "1e6": torch.full((4, 1024), 1e6, device=DEVICE),
    "1e30": torch.full((4, 1024), 1e30, device=DEVICE),
    "1e-30": torch.full((4, 1024), 1e-30, device=DEVICE),
    "6.0": torch.full((4, 1024), 6.0, device=DEVICE),
}

for name, kv in cases.items():
    out_v17 = v17_impl(kv)
    out_ref = ref_impl(kv)
    s17 = out_v17[0, 512]
    s_ref = out_ref[0, 512]
    d17 = out_v17[0, :8].tolist()
    d_ref = out_ref[0, :8].tolist()
    eq = torch.equal(out_v17, out_ref)
    print(f"{name:>9}: v17 scale={s17.item():3d} ref scale={s_ref.item():3d} | "
          f"data v17={d17} ref={d_ref} | byte_equal={eq}")
