#!/usr/bin/env python3
"""routeB P4 管线库：直接构造 kernel-ready 张量（绕过 run_bs 的 host 侧慢速构造）。

关键点（P4 A/B 端到端口径）：
- A 量化：复用 v17 triton _a_quant_kernel（MXF4 32 组 UE8M0 + E2M1 打包，生产语义）
  —— 用裸 JIT 函数固定 BLOCK_M=128/BLOCK_K=32（绕过 autotune 的 grid/config 匹配风险）
- SF swizzle：纯 torch scatter，映射 (m, k_group) -> (32,4,rest_m,4,rest_k,l) MMA 布局
  逻辑坐标分解（列主序）：m = a + 32*b + 128*c (a=m%32, b=(m//32)%4, c=m//128)
                         kg = d + 4*e        (d=kg%4, e=kg//4)
- A/B fp4 打包：写入底层 int8 buffer 每行前 K//2 字节（probe2 已验证）
- kernel-only / E2E 双口径由调用方计时

布局断言全部由 probe2_layout.py 先行验证。
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.join(os.path.dirname(_HERE), "routeb_official")

sys.path.insert(0, _OFFICIAL)
sys.path.insert(0, _HERE)

import torch
import cutlass
import cutlass.cute.testing as _cute_testing

if not hasattr(cutlass, "testing"):
    cutlass.testing = _cute_testing
    sys.modules["cutlass.testing"] = _cute_testing

import cutlass.torch as ct
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils

import dense_blockscaled_gemm_persistent_pingpong as pp

import triton

_AQ_RAW = None


import triton.language as tl


@triton.jit
def _a_quant_fused_kernel(A_ptr, Aq_ptr, SF_ptr, M, K, REST_K,
                          stride_am, stride_ak,
                          BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """A [M,K] -> E2M1 打包直写 kernel buffer（紧凑行布局）+ UE8M0 SF 直写 MMA swizzle 布局。

    单次 launch 完成 routeB E2E 的全部 A 侧工作（生产集成形态）。
    SF swizzle 偏移（probe2 P3 验证的映射）:
      flat = pid_m*(REST_K*512) + (pid_k//4)*512 + (local%32)*16 + (local//32)*4 + (pid_k%4)
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a = tl.load(
        A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :].to(tl.int64) * stride_ak,
        mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")

    a_abs = tl.abs(a)
    block_max = tl.max(a_abs, axis=1)
    safe_max = tl.maximum(block_max, 1e-38)
    log2_val = tl.log2(safe_max / 6.0)
    e8m0_f = tl.floor(log2_val) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    e8m0_u8 = e8m0_f.to(tl.uint8)

    sf_off = (pid_m * REST_K * 512 + (pid_k // 4) * 512
              + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4))
    tl.store(SF_ptr + sf_off, e8m0_u8, mask=mask_m)

    a_scale = tl.exp2(e8m0_f - 127.0)
    a_scaled = a / a_scale[:, None]
    a_abs_s = tl.abs(a_scaled)
    idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    idx = idx + (a_abs_s > 0.25).to(tl.int32)
    idx = idx + (a_abs_s > 0.75).to(tl.int32)
    idx = idx + (a_abs_s > 1.25).to(tl.int32)
    idx = idx + (a_abs_s > 1.75).to(tl.int32)
    idx = idx + (a_abs_s > 2.5).to(tl.int32)
    idx = idx + (a_abs_s > 3.5).to(tl.int32)
    idx = idx + (a_abs_s > 5.0).to(tl.int32)
    neg_mask = (a_scaled < 0.0).to(tl.int32)
    sign_bit = tl.where(idx > 0, neg_mask * 8, 0)
    nibble = (idx + sign_bit).to(tl.uint8)
    nibble_i32 = nibble.to(tl.int32)
    nibble_3d = tl.reshape(nibble_i32, [BLOCK_M, BLOCK_K // 2, 2])
    sel_lo_r = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    sel_hi_r = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(nibble_3d * sel_lo_r, axis=2)
    hi_val = tl.sum(nibble_3d * sel_hi_r, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)

    pk_off = (offs_m[:, None].to(tl.int64) * (K // 2)
              + pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk_off, packed, mask=mask_m[:, None])


@triton.jit
def _k1_sfpass(A_ptr, SFL_ptr, M, K, sf_stride_m, stride_am, stride_ak,
               BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """K1（SF pass）：A -> 逻辑布局 SF [M, K//32] uint8（E8M0）。

    单独成核的原因：归约+broadcast 与量化链合核时，triton layout 规划器会把
    A 加载降级为标量 ld.global（实测 20GB/s）；拆分后各自 ~200GB/s
    （quant_tune2/3/8 实证，PTX 证据见 quant_ptx_diag.py）。
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak,
                mask=mask_m[:, None], other=0.0)
    amax = tl.max(tl.abs(a), axis=1)
    safe = tl.maximum(amax, 1e-38)
    e8m0_f = tl.floor(tl.log2(safe / 6.0)) + 127.0
    e8m0_f = tl.maximum(e8m0_f, 0.0)
    e8m0_f = tl.minimum(e8m0_f, 255.0)
    tl.store(SFL_ptr + offs_m * sf_stride_m + pid_k, e8m0_f.to(tl.uint8),
             mask=mask_m)


@triton.jit
def _k2_quant(A_ptr, SFL_ptr, Aq_ptr, SFSW_ptr, M, K, REST_K,
              sf_stride_m, stride_am, stride_ak,
              BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """K2（量化 pass）：A + 逻辑 SF -> E2M1 打包（紧凑行布局）+ SF swizzle 直写。

    数值与 torch 参考逐位一致（±0 编码差异除外），quant_dbg.py 实证
    mismatch 0/524288。
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    local = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + local
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    a = tl.load(A_ptr + offs_m[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak,
                mask=mask_m[:, None], other=0.0)
    sf = tl.load(SFL_ptr + offs_m * sf_stride_m + pid_k, mask=mask_m, other=127)
    inv = tl.exp2(127.0 - sf.to(tl.float32))
    # SF swizzle 直写（probe2 P3 映射）
    sf_off = (pid_m * REST_K * 512 + (pid_k // 4) * 512
              + (local % 32) * 16 + (local // 32) * 4 + (pid_k % 4))
    tl.store(SFSW_ptr + sf_off, sf, mask=mask_m)
    a2 = a * inv[:, None]
    a2a = tl.abs(a2)
    idx = ((a2a > 0.25).to(tl.int32) + (a2a > 0.75).to(tl.int32)
           + (a2a > 1.25).to(tl.int32) + (a2a > 1.75).to(tl.int32)
           + (a2a > 2.5).to(tl.int32) + (a2a > 3.5).to(tl.int32)
           + (a2a > 5.0).to(tl.int32))
    neg = (a2 < 0.0).to(tl.int32)
    nib = (idx + tl.where(idx > 0, neg * 8, 0)).to(tl.uint8)
    lo = tl.reshape(nib.to(tl.int32), [BLOCK_M, BLOCK_K // 2, 2])
    sel = tl.reshape((tl.arange(0, 2) == 0).to(tl.int32), [1, 1, 2])
    lo_val = tl.sum(lo * sel, axis=2)
    sel2 = tl.reshape((tl.arange(0, 2) == 1).to(tl.int32), [1, 1, 2])
    hi_val = tl.sum(lo * sel2, axis=2)
    packed = (lo_val | (hi_val << 4)).to(tl.uint8)
    pk = (offs_m[:, None].to(tl.int64) * (K // 2) + pid_k * (BLOCK_K // 2)
          + tl.arange(0, BLOCK_K // 2)[None, :])
    tl.store(Aq_ptr + pk, packed, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# GPU 量化（生产语义，MXF4: 32 组 UE8M0 + E2M1）
# ---------------------------------------------------------------------------
def triton_a_quant(A, A_quant, A_scale):
    """A fp16/bf16/fp32 [M,K] -> A_quant [M,K//2] u8, A_scale [M,K//32] u8。

    使用 v17 _a_quant_kernel 的裸 JIT 函数（固定 BLOCK_M=128, BLOCK_K=32，
    与 grid (cdiv(M,128), cdiv(K,32)) 精确匹配，避免 autotune 选择 32/64 行
    配置时覆盖率不足的隐患）。失败则回退 autotuned 版本 + 安全 grid。
    """
    global _AQ_RAW
    from nvfp4_4w4a_prefill_gemm_v17_triton import _a_quant_kernel
    M, K = A.shape
    if _AQ_RAW is None:
        _AQ_RAW = getattr(_a_quant_kernel, "fn", None)
    if _AQ_RAW is not None:
        grid = (triton.cdiv(M, 128), triton.cdiv(K, 32))
        _AQ_RAW[grid](
            A, A_quant, A_scale, M, K,
            A.stride(0), A.stride(1),
            A_quant.stride(0), A_quant.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            BLOCK_M=128, BLOCK_K=32, num_warps=8, num_stages=2,
        )
    else:
        # 回退：grid 按 BLOCK_M=32 覆盖（若 autotune 选更大块则冗余但正确）
        grid = (triton.cdiv(M, 32), triton.cdiv(K, 32))
        _a_quant_kernel[grid](
            A, A_quant, A_scale, M, K,
            A.stride(0), A.stride(1),
            A_quant.stride(0), A_quant.stride(1),
            A_scale.stride(0), A_scale.stride(1),
        )


def torch_w_quant(Wt, Ws=None):
    """Wt fp32 [N, K] -> (packed [N,K//2] u8, scale [N,K//32] u8)。

    scale 语义与 _a_quant_kernel 相同（逐行 32 组 UE8M0，floor(log2(amax/6))+127）；
    W 预处理一次性成本，不进 steady-state 计时。
    """
    N, K = Wt.shape
    xb = Wt.reshape(N, K // 32, 32)
    if Ws is None:
        amax = xb.abs().amax(-1)
        e = torch.floor(torch.log2(torch.clamp(amax, min=1e-30) / 6.0)).long() + 127
        Ws = e.clamp(0, 255).to(torch.uint8)
    sfv = torch.pow(2.0, Ws.float() - 127.0).unsqueeze(-1)
    xn = torch.clamp(xb / sfv, -6.0, 6.0)
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=Wt.device)
    idx = (xn.abs().unsqueeze(-1) - mag).abs().argmin(-1)
    nib = idx | ((xn < 0).long() * 8)
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).reshape(N, K // 2)
    return packed.contiguous(), Ws.contiguous()


def dequant_ref(Aq, As, Wq, Ws):
    """量化参考（校验用）：Aq [M,K//2], As [M,K//32], Wq [N,K//2], Ws [N,K//32]。"""
    def deq(q, s, rows, K):
        p = q.to(torch.int32)
        lo = (p & 0xF).long()
        hi = ((p >> 4) & 0xF).long()
        E = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device=q.device)
        w = torch.stack([E[lo.reshape(-1)].reshape(rows, K // 2),
                         E[hi.reshape(-1)].reshape(rows, K // 2)], 2).reshape(rows, K)
        sf = torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, 1)
        return w * sf

    M, K2 = Aq.shape
    K = K2 * 2
    N = Wq.shape[0]
    return deq(Aq, As, M, K) @ deq(Wq, Ws, N, K).t()


# ---------------------------------------------------------------------------
# 张量构造（镜像 run_bs 内部，数据直写 buffer）
# ---------------------------------------------------------------------------
def _make_fp4_cute(buf):
    """int8 buffer (m, k, l) k 连续 -> fp4 cute tensor（镜像 run_bs 标记）。"""
    t = from_dlpack(buf, assumed_align=16)
    t.element_type = cutlass.Float4E2M1FN
    t = t.mark_layout_dynamic(leading_dim=ct.get_leading_dim(buf))
    t = t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=2)
    return t


def sf_scatter(scale2d, sf_u8, idx=None):
    """scale [M, sf_k] uint8 -> 写入 sf_u8 (l, rest_m, rest_k, 32, 4, 4) uint8 视图。

    idx: 可选缓存 (c, e, a, b, d) 索引张量元组（E2E 路径避免每次重算 arange）。
    """
    M, sf_k = scale2d.shape
    dev = scale2d.device
    if idx is None:
        m_ar = torch.arange(M, device=dev)
        kg_ar = torch.arange(sf_k, device=dev)
        a = (m_ar % 32)
        b = ((m_ar // 32) % 4)
        c = (m_ar // 128)
        d = (kg_ar % 4)
        e = (kg_ar // 4)
    else:
        c, e, a, b, d = idx
    sf_u8[0, c[:, None], e[None, :], a[:, None], b[:, None], d[None, :]] = scale2d


def sf_scatter_idx(M, sf_k, dev):
    """预计算 scatter 索引（c, e, a, b, d）。"""
    m_ar = torch.arange(M, device=dev)
    kg_ar = torch.arange(sf_k, device=dev)
    return (m_ar // 128, kg_ar // 4, m_ar % 32, (m_ar // 32) % 4, kg_ar % 4)


class RouteBGemm:
    """单 (M, N, K, tile, epi) 配置：模板张量 + 编译 + buffer 直写。"""

    def __init__(self, M, N, K, tile=(128, 128, 128), epi=(128, 128), sf_vec=32,
                 _compile=True):
        self.M, self.N, self.K = M, N, K
        self.tile, self.epi = tile, epi
        self.sf_k = (K + sf_vec - 1) // sf_vec
        l = 1
        self.rest_m = (M + 127) // 128
        self.rest_k = (self.sf_k + 3) // 4

        dev = "cuda"
        # A/B: (rows, k, l) int8（fp4 需 k/2 字节，官方路径双倍分配，前 k/2 有效）
        self.a_buf = torch.empty(M, K, l, dtype=torch.int8, device=dev)
        self.b_buf = torch.empty(N, K, l, dtype=torch.int8, device=dev)
        # C: (m, n, l) fp16 n-major
        self.c_buf = torch.empty(M, N, l, dtype=torch.float16, device=dev)
        # SF: (l, rest_m, rest_k, 32, 4, 4) int8 存储 + uint8 视图（对齐模板 dtype）
        self.sfa_base = torch.empty(l, self.rest_m, self.rest_k, 32, 4, 4,
                                    dtype=torch.int8, device=dev)
        self.sfb_base = torch.empty(l, (N + 127) // 128, self.rest_k, 32, 4, 4,
                                    dtype=torch.int8, device=dev)
        self.sfa_u8 = self.sfa_base.view(torch.uint8)
        self.sfb_u8 = self.sfb_base.view(torch.uint8)

        self.a_cute = _make_fp4_cute(self.a_buf)
        self.b_cute = _make_fp4_cute(self.b_buf)
        self.c_cute = from_dlpack(self.c_buf, assumed_align=16)
        self.c_cute.element_type = cutlass.Float16
        self.c_cute = self.c_cute.mark_layout_dynamic(
            leading_dim=ct.get_leading_dim(self.c_buf))
        self.c_cute = self.c_cute.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1)
        for base, attr in ((self.sfa_base, "sfa_cute"), (self.sfb_base, "sfb_cute")):
            buf = base.permute(3, 4, 1, 5, 2, 0)  # (32,4,rest_m,4,rest_k,l)
            t = from_dlpack(buf, assumed_align=16)
            t.element_type = cutlass.Float8E8M0FNU
            t = t.mark_layout_dynamic(leading_dim=ct.get_leading_dim(buf))
            setattr(self, attr, t)

        # E2E 稳态资源：预分配量化输出 + 缓存 scatter 索引 + u8 flat 视图
        self._aq = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
        self._asf = torch.empty(M, self.sf_k, dtype=torch.uint8, device=dev)
        self._sf_idx = sf_scatter_idx(M, self.sf_k, dev)
        self._a_u8flat = self.a_buf.view(torch.uint8).view(-1)
        self._sf_u8flat = self.sfa_u8.view(-1)
        self._sfl = torch.empty(M, self.sf_k, dtype=torch.uint8, device=dev)

        if _compile:
            gemm = pp.Sm120BlockScaledGemmKernel(cutlass.Float32, sf_vec, tile, epi)
            hw = utils.HardwareInfo()
            max_active = hw.get_max_active_clusters(1)
            self.stream = ct.default_stream()
            self.compiled = cutlass.cute.compile(
                gemm, self.a_cute, self.b_cute, self.sfa_cute, self.sfb_cute,
                self.c_cute, max_active, self.stream)
        else:
            self.stream = ct.default_stream()
            self.compiled = None  # 由调用方注入（M 动态复用，probe P6 验证）

    # ---- 数据写入 ----
    def _pack_a(self, aq):
        # fp4 紧凑行布局：行 m 起始字节偏移 = m * (K//2)（probe2 P2 验证）
        self.a_buf.view(-1)[: self.M * (self.K // 2)].copy_(
            aq.reshape(-1).view(torch.int8))

    def _pack_b(self, wq):
        self.b_buf.view(-1)[: self.N * (self.K // 2)].copy_(
            wq.reshape(-1).view(torch.int8))

    def set_A(self, A):
        """A [M,K] fp16 激活 -> triton 量化 -> 写 buffer（setup，不计时）。"""
        triton_a_quant(A, self._aq, self._asf)
        self._pack_a(self._aq)
        sf_scatter(self._asf, self.sfa_u8, self._sf_idx)
        return self._aq, self._asf

    def quant_2pass(self, A):
        """K1+K2 双 pass 量化（E2E 主路径，quant_tune8 实证 94-136GB/s 且逐位精确）。"""
        g = (triton.cdiv(self.M, 128), triton.cdiv(self.K, 32))
        _k1_sfpass[g](A, self._sfl, self.M, self.K, self._sfl.stride(0),
                      A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
                      num_warps=8, num_stages=2)
        _k2_quant[g](A, self._sfl, self._a_u8flat, self._sf_u8flat,
                     self.M, self.K, self.rest_k, self._sfl.stride(0),
                     A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
                     num_warps=8, num_stages=2)

    def e2e_call(self, A):
        """E2E 稳态单次调用（全部计时）：双 pass 量化 + GEMM。"""
        self.quant_2pass(A)
        self.run()

    def e2e_call_singlefused(self, A):
        """单融合 kernel 版（对照——triton layout 病理导致 ~20GB/s，仅记录用）。"""
        _a_quant_fused_kernel[(triton.cdiv(self.M, 128), triton.cdiv(self.K, 32))](
            A, self._a_u8flat, self._sf_u8flat, self.M, self.K, self.rest_k,
            A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
            num_warps=8, num_stages=2)
        self.run()

    def e2e_call_unfused(self, A):
        """未融合对照（多 launch：triton 量化 + pack copy + SF scatter + GEMM）。"""
        triton_a_quant(A, self._aq, self._asf)
        self._pack_a(self._aq)
        sf_scatter(self._asf, self.sfa_u8, self._sf_idx)
        self.run()

    def quant_fused_only(self, A):
        """仅单融合量化（对照计时用）。"""
        _a_quant_fused_kernel[(triton.cdiv(self.M, 128), triton.cdiv(self.K, 32))](
            A, self._a_u8flat, self._sf_u8flat, self.M, self.K, self.rest_k,
            A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_K=32,
            num_warps=8, num_stages=2)

    def set_A_prepacked(self, aq, asf):
        self._pack_a(aq)
        sf_scatter(asf, self.sfa_u8)

    def set_W(self, Wt):
        """Wt fp32 [N, K]（已转置）-> 量化写 buffer（一次性 W 预处理）。"""
        wq, ws = torch_w_quant(Wt)
        self._pack_b(wq)
        sf_scatter(ws, self.sfb_u8)
        return wq, ws

    def set_W_prepacked(self, wq, ws):
        self._pack_b(wq)
        sf_scatter(ws, self.sfb_u8)

    # ---- 执行 ----
    def run(self):
        self.compiled(self.a_cute, self.b_cute, self.sfa_cute,
                      self.sfb_cute, self.c_cute, self.stream)

    def out(self):
        return self.c_buf[:, :, 0]


# ---------------------------------------------------------------------------
# 计时（两 route 共用）
# ---------------------------------------------------------------------------
def time_ms(fn, warmup=10, iters=50, rounds=3):
    """返回中位每调用毫秒（批内事件计时，warm-L2 口径）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) / iters)
    res.sort()
    return res[len(res) // 2]


def tflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12
