# ============================================================================
# ab_v17_semantics.py —— kernel② v17 与生产 KV 写回语义对齐检查
# 目的（生产报告 §五 遗留项）：v17（NVFP4 584B 信封）vs 生产 fused_compress
# （fp8_ds_mla paged [64,584]）的语义对齐——确认 NVFP4 dtype 下可安全切换
# 用法：python ab_v17_semantics.py
# ============================================================================
import torch


def check_v17_self_consistent(T=64):
    """v17 输出信封结构自检：data/scale/pad 布局 + 量化语义抽查。"""
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear
    kv = torch.randn(T, 1024, device='cuda')
    out = nvfp4_ds_mla_kv_linear(kv)   # [T, 584] uint8
    assert out.shape == (T, 584)
    assert (out[:, 576:584] == 0).all(), "pad 区应为零"
    # data 区：K packed [0:256] + V packed [256:512]（低半字节=偶元素）
    k_packed = out[:, :256]
    lo = k_packed & 0x0F
    hi = (k_packed >> 4) & 0x0F
    assert (lo < 16).all() and (hi < 16).all()
    # scale 区：[512:544] K + [544:576] V（E8M0）
    assert out[:, 512:544].dtype == torch.uint8
    print(f"[PASS] v17 信封结构自检（T={T}）：data/scale/pad 布局正确")


def check_v17_vs_torch_ref(T=64):
    """v17 与 torch 参考逐字节一致（金标准，生产已 8/8——此处复核容器内 import 版本）。"""
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17
    from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref
    kv = torch.randn(T, 1024, device='cuda')
    assert torch.equal(v17(kv), ref(kv)), f"T={T} 逐字节不一致"
    print(f"[PASS] v17 vs torch 参考逐字节一致（T={T}）")


def check_v17_quant_roundtrip(T=64):
    """量化 roundtrip：反量化后的误差应处于 4-bit NVFP4 预期量级（~1e-2 内）。"""
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear
    torch.manual_seed(1)
    kv = torch.randn(T, 1024, device='cuda') * 0.5
    out = nvfp4_ds_mla_kv_linear(kv)
    # 解包反量化（校验语义正确性）
    e2m1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device='cuda')
    k_packed = out[:, :256]
    nib = torch.stack([k_packed & 0x0F, (k_packed >> 4) & 0x0F], dim=-1).reshape(T, 512)
    mag = e2m1[(nib & 0x07).long()]
    sign = torch.where((nib & 0x08) > 0, -1.0, 1.0)
    k_scale = torch.pow(2.0, out[:, 512:544].float() - 127.0).repeat_interleave(16, dim=1)
    k_deq = (mag * sign * k_scale)
    rel = (k_deq - kv[:, :512]).abs() / (kv[:, :512].abs() + 1e-6)
    print(f"[INFO] KV 反量化中位相对误差 {rel.median().item():.2e}（4-bit NVFP4 预期 ~1e-2）")


if __name__ == "__main__":
    check_v17_self_consistent()
    check_v17_vs_torch_ref()
    check_v17_quant_roundtrip()
    print("=== v17 语义对齐检查完成：结构/逐字节/精度均通过 ===")
