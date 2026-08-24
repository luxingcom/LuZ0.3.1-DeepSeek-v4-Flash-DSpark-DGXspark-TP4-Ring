#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsl_gemm.py v2.1 — merged-GEMM v2 生产 host 侧（fix-rerun 版, 2026-08-22）。

基于 e2e 冒烟版（E6/E12 实证机制, GEMM rel=3.87e-04）+ 三项修复:
  [P0-A] m_pad_tier 档位量化 + M_pad 档预编译支持（编译实测 ~2.5s/档;
         冒烟报告"DSL 冷编译 ≥45min"归因已证伪——p0a_compile_probe 对照实验）
  [P0-B] nvfp4_quant 两级 scale 等价实现（flashinfer 对齐）: 全局 amax 预缩放
         g=2688/amax → 单级量化（block sf 落 E4M3 良好区间, max sf=448, 根治小
         amax block 的 sf 下溢噪声）; 调用方 GEMM 输出 ÷g 还原
  [P1-A] SF 壳/c 壳缓冲复用（_HOLDS 无界泄漏修复——v2.0 每次调用泄漏 SF 壳
         67MB + 268MB f32 瞬态, 为 KV -29% 的主嫌疑）; B 侧 SF 壳由调用方
         （_derive/combo 缓存）一次创建复用, 壳背衬 flat 视图即物理存储

机制不变:
  - fp4 payload 零拷贝: from_dlpack(u8 [rows, K/2, 1]) + traced wrapper 内显式
    packed 布局 make_tensor(recast_ptr(ptr, FP4), make_layout((rows, K, 1), (K, 1, 1)))
  - SF: strided-view 壳 + flat 字节直写, 物理存储序 (l, rm, rk, 32, 4, 4) 连续
惰性导入（EngineCore spawn 安全）; compile 缓存按 (M_pad, N, K, b_rows)。"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")

_V2_PATH = os.environ.get("ROUTEB_V2_PATH", "/routeb-v2")
if _V2_PATH not in sys.path:
    sys.path.insert(0, _V2_PATH)

import torch
import cutlass
import cutlass.cute.testing as _ct
if not hasattr(cutlass, "testing"):
    cutlass.testing = _ct
    sys.modules["cutlass.testing"] = _ct
import cutlass.torch as cutlass_torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import dense_blockscaled_gemm_persistent_pingpong as _v2mod

FP4 = cutlass.Float4E2M1FN
SFDT = cutlass.Float8E4M3FN
CDT = cutlass.Float16

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_THRESH = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

# M_pad 档位（P0-A: 桶 M_g 向上取档, 档内 pad 复制行 0; pad 浪费 ≤2×,
# merged 档位效率 157T@256 仍为 B12X per-expert 6-43T 的 4-56×, 净收益为正）
M_TIERS = (256, 512, 1024, 2048, 4096, 8192, 16384)

_CTX = {}
_COMPILED = {}
_stats = {"gemm_calls": 0, "compiles": 0}


def _ctx():
    if not _CTX:
        torch.zeros(1, device="cuda")
        hw = cutlass.utils.HardwareInfo()
        _CTX["mc"] = hw.get_max_active_clusters(1)
        _CTX["stream"] = cutlass_torch.default_stream()
    return _CTX


def stats():
    return dict(_stats)


def pick_tier(m):
    for t in M_TIERS:
        if t >= m:
            return t
    return ((m + 127) // 128) * 128   # 超大档兜底（>16384, 罕见, 单次编译吸收）


# ---------------- NVFP4 量化（torch, 字节级验证 + 两级 scale 等价） ----------------
# 两级 scale（flashinfer 实证语义, 2026-08-22 p0b_probe）:
#   gs = 2688/amax_global; sf_stored = RTN_E4M3(blk_amax/6 × gs);
#   payload = RTN_E2M1(x × gs / sf_stored); dequant = payload × sf_stored / gs
# 本 kernel 无 epilogue alpha 且 C=fp16 → raw C = gs×(x@W) 有 fp16 溢出风险,
# 故 gs 上限 GS_CAP（默认 16: 下溢防护扩展 16×, raw C ≤ 16×|C_true|）。
GS_CAP = float(os.environ.get("VLLM_MOE_MERGED_GS_CAP", "16"))


def nvfp4_quant(x, two_level=True):
    """[M, K] bf16/f16 → (payload u8 [M, K/2] 低nibble=偶k,
                          scale E4M3 字节 [M, K/16], g 标量张量 CUDA)。
    two_level=True: flashinfer 同式两级 scale（g=clip(2688/amax, 1, GS_CAP) 预缩放,
    payload 满量程 ±6, 根治小 amax block 的 sf 下溢）; 调用方需 GEMM 输出 ÷g。
    two_level=False: 原单级（g=1, 对照口径）。"""
    x = x.float()
    M, K = x.shape
    if two_level:
        amax = x.abs().amax()
        g = (448.0 * 6.0) / amax.clamp_min(1e-30)
        g = g.clamp(1.0, GS_CAP).float()
        x = x * g
    else:
        g = torch.ones(1, device=x.device, dtype=torch.float32)
    amax_blk = x.abs().view(M, K // 16, 16).amax(-1)
    scale = (amax_blk / 6.0).to(torch.float8_e4m3fn)
    scale = scale.view(torch.uint8)
    scale_f = scale.view(torch.float8_e4m3fn).float()
    scale_f = torch.where(scale_f == 0, torch.ones_like(scale_f), scale_f)
    q = x.view(M, K // 16, 16) / scale_f.unsqueeze(-1)
    aq = q.abs()
    code = sum((aq >= t).to(torch.int32) for t in _THRESH).to(torch.uint8)
    nib = code | ((q < 0).to(torch.uint8) << 3)
    pair = nib.view(M, K // 16, 8, 2)
    payload = (pair[..., 0] | (pair[..., 1] << 4)).reshape(M, K // 2)
    return payload.contiguous(), scale.contiguous(), g


def dequant_ref(payload, scale_u8):
    """自检用: [M, K/2] u8 + [M, K/16] u8 → f32 [M, K]。"""
    tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=payload.device)
    M, Kh = payload.shape
    K = Kh * 2
    lo = tab[(payload & 0x0F).long()]
    hi = tab[(payload >> 4).long()]
    out = torch.empty(M, K, dtype=torch.float32, device=payload.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    sf = scale_u8.view(torch.float8_e4m3fn).float()
    return out * sf.repeat_interleave(16, dim=1)


# ---------------- SF swizzle（E12 正确公式） ----------------
def swizzle_scales_u8(scale_u8):
    """[M, G] E4M3 u8 → 物理存储序 (l, rm, rk, 32, 4, 4) 连续 flat（GPU）。
    M 需 128 倍数（调用方 pad）。返回 (flat, rm, rk)。"""
    M, G = scale_u8.shape
    rm, rk = M // 128, G // 4
    v = scale_u8.view(rm, 4, 32, rk, 4)                 # (c, b, a, e, d)
    p = v.permute(0, 3, 2, 1, 4).contiguous()           # (c, e, a, b, d) = 存储序
    return p.reshape(-1), rm, rk


# ---------------- SF 壳（P1-A: 一次创建复用, 壳背衬 = 物理存储） ----------------
def make_sf_cute(phys_flat, rm, rk):
    """一次性 SF 壳: 壳背衬 flat u8 视图即物理存储（back 与 phys 同内容）。
    返回 (cute_tensor, back_flat_u8)。back 供 Triton 等同 layout 消费者复用。"""
    storage = torch.zeros(1, rm, rk, 32, 4, 4, dtype=torch.float32,
                          device="cuda")
    ref = storage.permute(3, 4, 1, 5, 2, 0)             # (32,4,rm,4,rk,1) strided
    cute_tensor, hold = cutlass_torch.cute_tensor_like(
        ref, SFDT, is_dynamic_layout=True, assumed_align=16)
    hb = hold.view(torch.uint8) if hold.dtype != torch.uint8 else hold
    back = hb.permute(5, 2, 4, 0, 1, 3).reshape(-1)     # (1,rm,rk,32,4,4) 连续
    back.copy_(phys_flat)
    return cute_tensor, back


_SF_A_SHELL = {}     # ("A", rm, rk) -> (cute_tensor, back); 内容每调用 copy


def _sf_a_shell(rm, rk):
    ent = _SF_A_SHELL.get((rm, rk))
    if ent is None:
        # 占位创建（内容随后 copy）; 与 make_sf_cute 同构
        dummy = torch.zeros(rm * rk * 512, dtype=torch.uint8, device="cuda")
        cute_tensor, back = make_sf_cute(dummy, rm, rk)
        ent = (cute_tensor, back)
        _SF_A_SHELL[(rm, rk)] = ent
    return ent


# ---------------- traced wrapper（E6 packed 构造） ----------------
@cute.jit
def _gemm_packed_call(gemm: cutlass.Constexpr, a_u8, sfa, b_u8, sfb, c, tile_map,
                      mc, stream):
    am = cute.size(a_u8, mode=[0])
    ak = cute.size(a_u8, mode=[1]) * 2
    a = cute.make_tensor(
        cute.recast_ptr(a_u8.iterator, dtype=cutlass.Float4E2M1FN),
        cute.make_layout((am, ak, 1), stride=(ak, 1, 1)))
    bn = cute.size(b_u8, mode=[0])
    bk = cute.size(b_u8, mode=[1]) * 2
    b = cute.make_tensor(
        cute.recast_ptr(b_u8.iterator, dtype=cutlass.Float4E2M1FN),
        cute.make_layout((bn, bk, 1), stride=(bk, 1, 1)))
    gemm(a, b, sfa, sfb, c, tile_map, mc, stream)


def _u8_cute(payload_u8, rows, k):
    t = from_dlpack(payload_u8.contiguous().view(rows, k // 2, 1), assumed_align=16)
    t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
    return t


def pad_rows(t, M, M_pad):
    if M_pad == M:
        return t
    return torch.cat([t, t[:1].expand(M_pad - M, *t.shape[1:])], 0).contiguous()


# ---------------- c 壳复用（P1-A） ----------------
_C_SHELL = {}


def _c_shell(M, N):
    key = (M, N)
    ent = _C_SHELL.get(key)
    if ent is None:
        c_ref = torch.zeros(M, N, 1, dtype=torch.float32, device="cuda")
        c_tensor, c_hold = cutlass_torch.cute_tensor_like(
            c_ref, CDT, is_dynamic_layout=True, assumed_align=16)
        c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1),
                                            divisibility=1)
        ent = c_tensor
        _C_SHELL[key] = ent
    return ent


def merged_gemm(a_payload, a_scale_u8, b_payload, b_sf_cute, K, N,
                tile_map=None, m_pad_tier=None):
    """一次 merged GEMM（E6+E12 机制 + v2.1 修复）。
    a_payload [M, K/2] u8; a_scale_u8 [M, K/16] u8（M 自动 pad 到 m_pad_tier 档）;
    b_payload [N_b, K/2] u8; b_sf_cute = make_sf_cute 产出的 SF 壳（B 侧内容
    静态, 由调用方持有复用）; tile_map [N/128] int32（None=identity）。
    返回 f32 [M, N]（fp16 计算, 读回 f32）。"""
    M = a_payload.shape[0]
    M_pad = m_pad_tier if m_pad_tier else max(128, (M + 127) // 128 * 128)
    assert M_pad >= M and M_pad % 128 == 0, (M, M_pad)
    a_payload = pad_rows(a_payload, M, M_pad)
    a_scale_u8 = pad_rows(a_scale_u8, M, M_pad)

    a_u8 = _u8_cute(a_payload, M_pad, K)
    b_u8 = _u8_cute(b_payload, b_payload.shape[0], K)
    sfa_phys, sfa_rm, sfa_rk = swizzle_scales_u8(a_scale_u8)
    sfa_cute, sfa_back = _sf_a_shell(sfa_rm, sfa_rk)
    sfa_back.copy_(sfa_phys)

    c_tensor = _c_shell(M_pad, N)
    ctx = _ctx()

    if tile_map is None:
        tm = torch.arange(N // 128, dtype=torch.int32).cuda()
    else:
        tm = tile_map.to(torch.int32).contiguous().cuda()
    map_cute = from_dlpack(tm, assumed_align=16).mark_layout_dynamic(leading_dim=0)

    key = (M_pad, N, K, int(b_payload.shape[0]))
    if key not in _COMPILED:
        gemm = _v2mod.Sm120BlockScaledGemmKernel(
            cutlass.Float32, 16, (128, 128, 128), (128, 128))
        _COMPILED[key] = cute.compile(
            _gemm_packed_call, gemm, a_u8, sfa_cute, b_u8, b_sf_cute, c_tensor,
            map_cute, ctx["mc"], ctx["stream"])
        _stats["compiles"] += 1
    _stats["gemm_calls"] += 1

    _COMPILED[key](a_u8, sfa_cute, b_u8, b_sf_cute, c_tensor, map_cute,
                   ctx["mc"], ctx["stream"])

    out = torch.empty(M_pad, N, 1, dtype=torch.float32).cuda()
    cute.testing.convert(
        c_tensor,
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1))
    return out[:M, :, 0].float()


# ---------------- E8M0 → E4M3 LUT（Task#20 同源） ----------------
_B = torch.arange(256, dtype=torch.float32)
E8M0_TO_E4M3_LUT = (torch.pow(2.0, _B - 127.0)).to(
    torch.float8_e4m3fn).view(torch.uint8).cpu()


def e8m0_to_e4m3_k16(scale_u8):
    """[..., K//32] E8M0 u8 → [..., K//16] E4M3 u8。"""
    lut = E8M0_TO_E4M3_LUT.to(scale_u8.device)
    return lut[scale_u8.long()].repeat_interleave(2, dim=-1)
