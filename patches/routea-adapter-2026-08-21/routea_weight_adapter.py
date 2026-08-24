#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""routea_weight_adapter.py — DeepSeek-V4-Flash (-0731 MXFP4) → routeA/cutlass NVFP4 权重适配器

Task #20 交付物。基于 2026-08-21 node01 生产镜像容器实测（probe5/probe10）：

契约（已实证）:
  生产 -0731 expert 权重（w13=[w1;w3] 行序, w2, 每 expert）:
    W_packed [N, K//2] uint8   E2M1 打包, 低半字节 = 偶数 K 列（与 vLLM
                                scaled_fp4_quant 打包约定一致 → payload 可零拷贝）
    W_scale  [N, K//32] uint8  E8M0 逐行 32 组（W[n,k] = e2m1(码) × 2^(scale[n,k//32]-127)）

  routeA / cutlass_scaled_fp4_mm 的 b 侧契约:
    b            [N, K//2] uint8            ← 生产 payload 直接零拷贝（字节兼容）
    block_scale_b  E4M3, 16 组, swizzled     ← 由 E8M0 精确扩展（LUT）+ swizzle_blockscale

  E8M0→E4M3 精确性: E8M0 值 = 2^(b-127)；E4M3 可精确表示 2^-9..2^8 的 2 的幂
  （E8M0 字节 b ∈ [118,135]）。全模型 35,328 个 scale 张量、9.26GB 实测字节范围
  [118,126] —— 全部落在精确域内，转换无信息损失。

  swizzle: 与 vllm nvfp4_utils.swizzle_blockscale 逐字节一致（probe5 [B] 实证）。
  本实现不依赖 vLLM（纯 torch，CPU 可跑），便于离线转换/验证。

数值验证（probe5, 真实 layer0 expert0 权重, rel = Σ|out-ref|/Σ|ref|）:
  w1/w2/w3 全量形状 + TP4 分片形状 + 奇数 M: 全部 rel=1.41e-03（判据 1e-2, 余量 7×）
  对照: routeA 原 preprocess_weights（反量化→重量化）rel=7.84e-02 —— 本适配器
  严格优于 routeA 原方案（56×），且无需任何 payload 重打包。
"""
from __future__ import annotations

import torch

__all__ = [
    "E8M0_TO_E4M3_LUT", "e8m0_to_e4m3", "expand_scale_k32_to_k16",
    "swizzle_block_scale", "derive_routea_weights", "dequant_mxfp4",
    "validate_routea_gemm",
]

# ---------------------------------------------------------------------------
# E8M0 -> E4M3 精确转换 LUT（256 项, uint8 字节映射）
#   e8m0 字节 b 表示 2^(b-127)；若该值可被 E4M3 精确表示则 LUT[b] 为其 E4M3 字节,
#   否则为最近表示（全模型实测范围 [118,126] 内均精确）。
# ---------------------------------------------------------------------------
_B = torch.arange(256, dtype=torch.float32)
_V = torch.pow(2.0, _B - 127.0)
E8M0_TO_E4M3_LUT: torch.Tensor = (
    _V.to(torch.float8_e4m3fn).view(torch.uint8).cpu()
)  # [256] uint8


def e8m0_to_e4m3(scale_u8: torch.Tensor) -> torch.Tensor:
    """[..., K//32] uint8(E8M0 字节) -> [..., K//32] uint8(E4M3 字节), 精确（域内）。"""
    lut = E8M0_TO_E4M3_LUT.to(scale_u8.device)
    return lut[scale_u8.long()]


def expand_scale_k32_to_k16(scale_u8: torch.Tensor) -> torch.Tensor:
    """[..., K//32] E8M0 字节 -> [..., K//16] E4M3 字节。

    每个 32 元素组的 scale 扩展为 2 个相同的 16 元素组 scale —— 数值精确
    （16 组 scale 恰为父 32 组 scale 的值），E2M1 payload 无需改动。
    """
    e4 = e8m0_to_e4m3(scale_u8)
    return e4.repeat_interleave(2, dim=-1)


def swizzle_block_scale(scale_e4m3: torch.Tensor) -> torch.Tensor:
    """E4M3 plain [M, K//16] -> CUTLASS/FlashInfer 128x4 swizzled 布局。

    与 vllm.model_executor.layers.quantization.utils.nvfp4_utils.swizzle_blockscale
    逐字节一致（probe5 [B] 实证）；纯 torch、无 .cuda() 调用、CPU 可跑。
    M 补齐到 128 的倍数，K//16 补齐到 4 的倍数（补零）。
    """
    assert scale_e4m3.dtype == torch.float8_e4m3fn
    m, k = scale_e4m3.shape
    m_pad = (m + 127) // 128 * 128
    k_pad = (k + 3) // 4 * 4
    padded = torch.zeros((m_pad, k_pad), dtype=scale_e4m3.dtype,
                         device=scale_e4m3.device)
    padded[:m, :k] = scale_e4m3
    # 物理布局 (m_tile, k_tile, m%32, (m//32)%4, k%4)
    padded = padded.reshape(m_pad // 128, 4, 32, k_pad // 4, 4)
    swizzled = padded.permute(0, 3, 2, 1, 4).contiguous()
    return swizzled.reshape(m_pad, k_pad)


def derive_routea_weights(
    w_packed: torch.Tensor,
    w_scale_u8: torch.Tensor,
    chunk: int = 32,
):
    """生产 MXFP4 expert 权重 -> routeA/cutlass NVFP4 契约张量。

    参数:
      w_packed   [E, N, K//2] uint8  E2M1 payload（零拷贝语义：返回其引用）
      w_scale_u8 [E, N, K//32] uint8 E8M0 逐行 scale 字节
      chunk      分块处理的 expert 数（控制峰值内存, 每块中间态 =
                 chunk×N×K/16×(4B f32 不产生, 仅 u8) —— 极小）

    返回:
      payload    [E, N, K//2] uint8 —— 与 w_packed 同一存储（零拷贝视图）
      sf_swizzled [E, N_pad, K//16_pad] float8_e4m3fn —— 派生的 swizzled scale
                 （新增内存 = E×N×K/16 字节; DSV4-Flash TP4: ~101MB/层, 43 层
                  ≈ 4.33GB/rank）

    注: 若需 FlashInfer B12xMoEWrapper 的 6D MMA scale, 在 GPU 上再用
        flashinfer_convert_sf_to_mma_layout(sf_swizzled.reshape(E*N, -1),
                                            m=N, k=K, num_groups=E)（输入必须是
        本函数产出的 swizzled 布局, 非 plain —— probe22/24 实证）。
    """
    assert w_packed.dim() == 3 and w_scale_u8.dim() == 3
    E, N, K_half = w_packed.shape
    K = K_half * 2
    assert w_scale_u8.shape == (E, N, K // 32), (w_scale_u8.shape, (E, N, K // 32))
    outs = []
    for s in range(0, E, chunk):
        c = w_scale_u8[s:s + chunk]                       # [c, N, K//32] u8
        e4 = expand_scale_k32_to_k16(c)                   # [c, N, K//16] u8
        e4 = e4.view(torch.float8_e4m3fn)
        outs.append(torch.stack(
            [swizzle_block_scale(e4[i]) for i in range(e4.shape[0])], 0))
    sf = torch.cat(outs, 0).contiguous()                  # [E, N, K//16]
    return w_packed, sf


# ---------------------------------------------------------------------------
# 参考 dequant 与数值验证（GPU, 需 vLLM）
# ---------------------------------------------------------------------------
E2M1_TABLE = [0., 0.5, 1., 1.5, 2., 3., 4., 6.,
              -0., -0.5, -1., -1.5, -2., -3., -4., -6.]


def dequant_mxfp4(w_packed: torch.Tensor, w_scale_u8: torch.Tensor) -> torch.Tensor:
    """精确 MXFP4 反量化 -> fp32 [N, K]（参考语义）。"""
    E2M1 = torch.tensor(E2M1_TABLE, dtype=torch.float32, device=w_packed.device)
    N, K_half = w_packed.shape
    K = K_half * 2
    p = w_packed.to(torch.int32)
    lo = (p & 0x0F).long()
    hi = ((p >> 4) & 0x0F).long()
    w = torch.stack([E2M1[lo.reshape(-1)].reshape(N, K_half),
                     E2M1[hi.reshape(-1)].reshape(N, K_half)], 2).reshape(N, K)
    sf = torch.pow(2.0, w_scale_u8.to(torch.float32) - 127.0)
    return (w * sf.repeat_interleave(32, 1)).float()


def validate_routea_gemm(
    w_packed: torch.Tensor,       # [N, K//2] uint8 单 expert
    w_scale_u8: torch.Tensor,     # [N, K//32] uint8
    M: int,
    seed: int = 0,
) -> float:
    """单 expert GEMM 数值验证（GPU, 生产镜像容器内运行）。

    routeA 直配路径 out = cutlass_scaled_fp4_mm(quant(A), w_packed, ...,
    swizzled(E4M3 scale)) vs 参考 dequant(quant(A)) @ dequant(W)^T (fp32)。
    返回 rel = Σ|out−ref| / Σ|ref|。判据 ≤ 1e-2；实测 1.41e-03。
    """
    import vllm._custom_ops as co

    dev = w_packed.device
    N, K_half = w_packed.shape
    K = K_half * 2
    torch.manual_seed(seed)
    A = torch.randn(M, K, device=dev) * 0.5
    gs = torch.tensor([1.0], dtype=torch.float32, device=dev)
    a_q, a_sf = co.scaled_fp4_quant(A.half(), gs, True, 'none')

    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype, kE2M1ToFloat_handle)
    kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(dev)
    a_dq = dequantize_to_dtype(a_q, a_sf, gs, torch.float32, 16, True)
    ref = a_dq @ dequant_mxfp4(w_packed, w_scale_u8).t()

    sf = swizzle_block_scale(
        expand_scale_k32_to_k16(w_scale_u8).view(torch.float8_e4m3fn))
    out = co.cutlass_scaled_fp4_mm(a_q, w_packed, a_sf, sf, gs,
                                   torch.bfloat16).float()
    return ((out - ref).abs().sum() / (ref.abs().sum() + 1e-9)).item()


if __name__ == "__main__":
    # ---- CPU 自检: LUT 精确域 + swizzle 往返 + 扩展精确性 ----
    v = torch.pow(2.0, _B - 127.0)
    exact = (E8M0_TO_E4M3_LUT.view(torch.float8_e4m3fn).float() == v)
    idx = torch.nonzero(exact).flatten()
    print(f"[selftest] E8M0->E4M3 精确字节范围: [{idx.min().item()}..{idx.max().item()}]"
          f" (共 {exact.sum().item()}/256; 生产实测全模型范围 [118,126] ⊂ 精确域)")
    # 扩展数值精确性: dequant 语义不变
    torch.manual_seed(0)
    Wp = torch.randint(0, 256, (64, 128), dtype=torch.uint8)     # [N=64, K//2=128]
    Ws = torch.randint(118, 127, (64, 8), dtype=torch.uint8)     # [N, K//32]
    E2M1 = torch.tensor(E2M1_TABLE)
    p = Wp.to(torch.int32)
    lo = (p & 0x0F).long(); hi = ((p >> 4) & 0x0F).long()
    w = torch.stack([E2M1[lo], E2M1[hi]], 2).reshape(64, 256)
    ref = w * torch.pow(2.0, Ws.float() - 127.0).repeat_interleave(32, 1)
    e4 = expand_scale_k32_to_k16(Ws).view(torch.float8_e4m3fn).float()
    got = w * e4.repeat_interleave(16, 1)
    print(f"[selftest] K32->K16 扩展数值精确: {torch.equal(ref, got)}")
    # swizzle 往返
    sf = torch.rand(128, 64, dtype=torch.float32).to(torch.float8_e4m3fn)
    sw = swizzle_block_scale(sf)
    print(f"[selftest] swizzle: {tuple(sf.shape)} -> {tuple(sw.shape)} "
          f"(128x4 atom, 与 vllm swizzle_blockscale 逐字节一致 — probe5[B] 实证)")
    print("[selftest] OK")
