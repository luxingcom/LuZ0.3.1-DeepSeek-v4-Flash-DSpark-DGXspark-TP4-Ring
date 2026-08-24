# ============================================================================
# test_nvfp4_ds_mla_kv_linear_v17_safety.py —— v17 安全性与可靠性补充测试
# 生产执行：python -m pytest test_nvfp4_ds_mla_kv_linear_v17_safety.py -v
# 覆盖：极端值/饱和/符号零/边界 T/确定性/长期运行
# ============================================================================
import torch
import pytest

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl


def _check(T, kv):
    ref = ref_impl(kv)
    got = v17_impl(kv)
    assert torch.equal(got, ref), (
        f"mismatch={(got != ref).sum().item()}/{got.numel()}"
    )


def test_extreme_values():
    """极端值：全零 / 全±6 / ±1e30 / ±1e-30 / -0.0 逐字节 vs torch 参考。"""
    for T in (4, 64, 1024):
        _check(T, torch.zeros(T, 1024, device="cuda"))
        _check(T, torch.full((T, 1024), 6.0, device="cuda"))
        _check(T, torch.full((T, 1024), -6.0, device="cuda"))
        _check(T, torch.full((T, 1024), 1e30, device="cuda"))
        _check(T, torch.full((T, 1024), 1e-30, device="cuda"))
        _check(T, torch.full((T, 1024), -0.0, device="cuda"))
        # 混合极端
        kv = torch.cat([torch.full((T // 2, 1024), 1e30, device="cuda"),
                        torch.full((T - T // 2, 1024), 1e-30, device="cuda")])
        _check(T, kv)


def test_saturation():
    """大值饱和：quant 后 nibble 全 7（mag）+ sign 位正确，scale 字节 255。"""
    kv = torch.full((64, 1024), 1e6, device="cuda")
    out = v17_impl(kv)
    # 每个字节 = 0x77（mag=7, sign=0 的高半字节）或 0xF7（sign=1）
    data = out[:, :512]
    assert ((data & 0x0F) == 0x07).all(), "低半字节应为 mag=7"
    assert ((data >> 4) & 0x0F == 0x07).all(), "高半字节应为 mag=7"
    assert (out[:, 512:576] == 255).all(), "E8M0 应饱和 255"


def test_sign_zero():
    """-0.0 与 +0.0 产生相同输出（sign=0, mag=0）。"""
    a = v17_impl(torch.zeros(16, 1024, device="cuda"))
    b = v17_impl(torch.full((16, 1024), -0.0, device="cuda"))
    assert torch.equal(a, b)
    assert (a[:, 512:576] == 1).all()  # e8m0 = 127 - 126 = 1（最小 scale）


def test_boundary_T():
    """边界 T：0（空返回）/ 1 / 非整除。"""
    kv0 = torch.zeros(0, 1024, device="cuda")
    assert v17_impl(kv0).shape == (0, 584)
    assert torch.equal(v17_impl(torch.randn(1, 1024, device="cuda")),
                       ref_impl(torch.randn(1, 1024, device="cuda")))
    for T in (3, 100, 65535):
        kv = torch.randn(T, 1024, device="cuda")
        assert torch.equal(v17_impl(kv), ref_impl(kv))


def test_determinism():
    """同输入多次调用逐字节相同（无状态）。"""
    kv = torch.randn(1024, 1024, device="cuda")
    outs = [v17_impl(kv) for _ in range(3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])


def test_long_run_memory():
    """长期运行：1000 次小 T 调用后显存无增长。"""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    kv = torch.randn(64, 1024, device="cuda")
    for _ in range(1000):
        v17_impl(kv)
    torch.cuda.synchronize()
    growth = torch.cuda.memory_allocated() - base
    assert growth < 64 * 1024 * 1024, f"显存增长异常: {growth / 1e6:.1f} MB"


def test_nan_inf_documented():
    """NaN/Inf：文档化行为（输出未定义，前置断言拦截——此处仅验证不崩溃于断言外路径）。"""
    # vLLM 上游保证输入有限；此测试仅确认 kernel 不导致非法内存访问（可运行即可）
    kv = torch.randn(64, 1024, device="cuda")
    kv[0, 0] = float("nan")
    v17_impl(kv)  # 不崩溃（输出值未定义，不在此断言）
    torch.cuda.synchronize()
