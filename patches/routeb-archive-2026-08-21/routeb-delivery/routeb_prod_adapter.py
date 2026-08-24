#!/usr/bin/env python3
"""routeb_prod_adapter.py — routeB kernel 生产权重直配适配器（P3 交付）
=====================================================================
routeB = CUTLASS 4.5 DSL SM120/121 warp-level blockscaled GEMM
        （vendored routeb_official/dense_blockscaled_gemm_persistent_pingpong.py）

【P3 结论前置（2026-08-21，详见 routeb-p3-semantic 报告）】
生产真实对接对象 = node01:/home/<USER>/models/deepseek-v4-flash-0731/
（生产 MXFP4 checkpoint，DeepSeek V4 Flash，43 层 × 256 experts，hidden=4096，
intermediate=2048）。其 expert 权重格式：
    W_packed : uint8 [N, K//2]   E2M1 打包（低半字节 = 偶数 K，N=输出维行）
    W_scale  : uint8 [N, K//32]  E8M0（每行 N、每 32 个 K 一组，值 = 2^(b-127)）
    语义     : W[n, k] = e2m1(W_packed[n, k//2] 的第 k%2 半字节) × 2^(W_scale[n, k//32] - 127)
这正是 routeB vendored kernel 的 B 侧原生格式（b_major="k" + SFB 逐行 vec32），
**权重零重排直配**。scale 张量需做 atom-swizzle 重排（见下）。

⚠ deepseek-v4-flash-0731-nvfp4-hp checkpoint 经实证为缺陷品（全部 scale 字节恒 1，
且权重码本与 E2M1 不符），不可消费——详见报告 §1。

【Scale 布局契约（P3 实证，probe 逐字节 100% 对齐官方 cvt）】
vendored kernel 在 __call__ 内对 sfa/sfb 只取数据指针并以
tile_atom_to_shape_SF 重排布局（BlockScaledBasicChunk: atom ((32,4),(32,4))，
stride ((16,4),(0,1))）。因此 scale 缓冲必须为 atom-swizzle 布局：
    buf[l, m//128, kg//4, m%32, (m//32)%4, kg%4] = plain[m, kg]
    buf 形状 (l, ceil(m/128), ceil(sf_k/4), 32, 4, 4)，连续存储；
    m 需补零到 128 的倍数（补 0x7F 中性值），sf_k = K//32 需为 4 的倍数
    （K=2048/4096 → sf_k=64/128 ✓）。
plain [mn, K//32] 直传 **不可行**（会读错字节 → 数值错）。本文件提供
sf_plain_to_atom / sf_atom_to_plain 双向重排。

【A 侧量化（MXF4 激活量化，对齐 kernel2 v17 语义 + E8M0 金标准）】
    A_scale[m, kg] = clamp(floor(log2(amax/6)) + 127, 0, 255)   amax 下限 1e-30
                     （零输入→24，1e6→144；校准向量与 v17 语义一致）
    A_code = 就近 E2M1 码本量化（tie 取低档，与 v17 阈值 0.25/0.75/.../5.0 一致）
    A_packed[m, k//2]：低半字节 = 偶数 k（与 kernel P5 实测 0xA1 样式一致）

【B-N1 铁律】c_dtype 必须 16-bit（fp16/bf16）；f32 会静默产出 ~50% 垃圾
（vendored 示例侧已加护栏，本适配器默认 fp16）。

用法：
    from routeb_prod_adapter import RouteBProdGEMM
    gemm = RouteBProdGEMM()                       # 默认 tile 128×128×128
    out = gemm(A_bf16, W_packed, W_scale)         # → fp16 [M, N]
    # W_packed [N, K//2] uint8、W_scale [N, K//32] uint8（生产 MXF4 直配）
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL_DIR = os.path.join(_HERE, "routeb_official")

import torch  # noqa: E402

E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SF_VEC = 32


# ===========================================================================
# 1. E2M1 / E8M0 量化原语（纯 torch，数值正确优先）
# ===========================================================================
def _mag_table(device):
    return torch.tensor(E2M1_MAG, device=device, dtype=torch.float32)


def encode_e8m0_32(x: torch.Tensor, k_group: int = SF_VEC) -> torch.Tensor:
    """[..., K] → E8M0 字节 [..., K//32]。
    byte = clamp(floor(log2(amax/6)) + 127, 0, 255)，amax 下限 1e-30。
    校准：零输入 → 24；1e6 → 144（金标准）。"""
    xb = x.float().reshape(*x.shape[:-1], x.shape[-1] // k_group, k_group)
    amax = xb.abs().amax(-1)
    e = torch.floor(torch.log2(torch.clamp(amax, min=1e-30) / 6.0)).to(torch.int64) + 127
    return e.clamp(0, 255).to(torch.uint8)


def quantize_e2m1(x_norm: torch.Tensor) -> torch.Tensor:
    """就近码本量化（输入已归一化，tie 取低档——与 v17 阈值语义一致）。"""
    mag = _mag_table(x_norm.device)
    d = (x_norm.abs().unsqueeze(-1) - mag).abs()
    return x_norm.sign() * mag[d.argmin(-1)]


def quantize_a(A: torch.Tensor):
    """A [M, K]（bf16/fp16/fp32）→ (codes_f32 [M,K], packed u8 [M,K//2],
    scale u8 [M,K//32])。codes ∈ E2M1 码本（供参考反量化与直配共用）。"""
    A = A.detach()
    M, K = A.shape
    assert K % SF_VEC == 0, f"K%{SF_VEC}!=0: {K}"
    scale = encode_e8m0_32(A)                                    # [M, K//32]
    sf = torch.pow(2.0, scale.float() - 127.0).reshape(M, K // SF_VEC, 1)
    a_norm = torch.clamp(A.float().reshape(M, K // SF_VEC, SF_VEC) / sf, -6.0, 6.0)
    codes = quantize_e2m1(a_norm).reshape(M, K)
    packed = pack_e2m1(codes)
    return codes, packed, scale


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 码值 [..., n] → uint8 [..., n//2]（低半字节=偶元素）。"""
    mag = _mag_table(codes.device)
    idx = (codes.abs().unsqueeze(-1) - mag).abs().argmin(-1).to(torch.int32)
    nib = (torch.where(codes < 0, 8, 0) + idx).to(torch.uint8)
    nib = torch.where(idx == 0, torch.zeros_like(nib), nib)      # -0 → +0（v17 语义）
    lo, hi = nib[..., 0::2], nib[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n//2] → E2M1 码值 f32 [..., n]（低半字节=偶元素）。"""
    mag = _mag_table(packed.device)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,),
                      dtype=torch.float32, device=packed.device)
    out[..., 0::2] = torch.where(lo >= 8, -1.0, 1.0) * mag[lo & 7]
    out[..., 1::2] = torch.where(hi >= 8, -1.0, 1.0) * mag[hi & 7]
    return out


# ===========================================================================
# 2. 生产 MXF4 权重反量化（参考实现）
# ===========================================================================
def dequant_w_mxf4(W_packed: torch.Tensor, W_scale: torch.Tensor) -> torch.Tensor:
    """生产 MXF4 → f32 [N, K]：W[n,k] = e2m1(码) × 2^(W_scale[n, k//32]-127)。"""
    N, Kh = W_packed.shape
    K = Kh * 2
    assert W_scale.shape == (N, K // SF_VEC), f"scale shape {W_scale.shape}"
    codes = unpack_e2m1(W_packed)                                # [N, K]
    sf = torch.pow(2.0, W_scale.float() - 127.0)
    return (codes.reshape(N, K // SF_VEC, SF_VEC) * sf[:, :, None]).reshape(N, K)


def dequant_a(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A 量化产物 → f32 [M, K]（参考用）。"""
    M, K = codes.shape
    sf = torch.pow(2.0, scale.float() - 127.0).reshape(M, K // SF_VEC, 1)
    return (codes.reshape(M, K // SF_VEC, SF_VEC) * sf).reshape(M, K)


# ===========================================================================
# 3. ★Scale atom-swizzle 重排（plain ↔ atom，P3 核心契约）
#    buf[l, m//128, kg//4, m%32, (m//32)%4, kg%4] = plain[m, kg]
#    （已对官方 cvt_sf_MKL_to_M32x4xrm_K4xrk_L 逐字节 100% 验证）
# ===========================================================================
def sf_plain_to_atom(plain: torch.Tensor, pad_byte: int = 0x7F) -> torch.Tensor:
    """plain u8 [mn, sf_k] → atom-swizzle 连续缓冲 (1, rm, rk, 32, 4, 4) u8。
    mn 补齐到 128 倍数（补 pad_byte）；sf_k 需 4 的倍数。"""
    mn, sf_k = plain.shape
    assert sf_k % 4 == 0, f"sf_k%4!=0: {sf_k}"
    rm = (mn + 127) // 128
    rk = sf_k // 4
    padded = torch.full((rm * 128, sf_k), pad_byte, dtype=torch.uint8,
                        device=plain.device)
    padded[:mn] = plain
    # (rm, m4, m32, rk, k4) ← plain[m, kg]，m = m32 + 32*m4 + 128*rm_i
    v = padded.reshape(rm, 4, 32, rk, 4)          # [rm, (m//32)%4, m%32, kg//4, kg%4]
    buf = v.permute(0, 3, 2, 1, 4).contiguous()   # → (rm, rk, 32, 4, 4)
    return buf.unsqueeze(0)                       # (1, rm, rk, 32, 4, 4)


def sf_atom_to_plain(buf: torch.Tensor, mn: int) -> torch.Tensor:
    """atom 缓冲 (1, rm, rk, 32, 4, 4) → plain [mn, sf_k]（逆向，调试用）。"""
    rm, rk = buf.shape[1], buf.shape[2]
    sf_k = rk * 4
    v = buf[0].permute(0, 3, 2, 1, 4).contiguous()  # (rm, m4, m32, rk, k4)
    plain = v.reshape(rm * 128, sf_k)
    return plain[:mn]


# ===========================================================================
# 4. routeB kernel 直配封装
# ===========================================================================
class RouteBProdGEMM:
    """routeB 生产 MXF4 权重直配 GEMM。
    gemm(A, W_packed, W_scale) ≈ dequant_a(quant(A)) @ dequant_w(W).T
    """

    def __init__(self, tile_mnk=(128, 128, 128), epi_tile=(128, 128),
                 kernel_variant="pingpong"):
        assert tile_mnk in ((128, 128, 128), (128, 128, 256)), \
            f"tile {tile_mnk} 不受官方 SM120 DSL 示例支持"
        self.tile = tuple(tile_mnk)
        self.epi = tuple(epi_tile)
        self.variant = kernel_variant
        self._compiled = {}
        self._mods = {}

    # ---- 模块懒加载（含 DSL 4.5.2 testing shim）----
    def _load(self):
        if "gemm_cls" in self._mods:
            return self._mods["gemm_cls"]
        import cutlass
        import cutlass.cute.testing as _ct
        if not hasattr(cutlass, "testing"):
            cutlass.testing = _ct
            sys.modules["cutlass.testing"] = _ct
        sys.path.insert(0, _OFFICIAL_DIR)
        try:
            if self.variant == "cooperative":
                from dense_blockscaled_gemm_persistent_cooperative import (
                    Sm120BlockScaledGemmKernel)
            else:
                from dense_blockscaled_gemm_persistent_pingpong import (
                    Sm120BlockScaledGemmKernel)
        finally:
            sys.path.remove(_OFFICIAL_DIR)
        self._mods["cutlass"] = cutlass
        self._mods["gemm_cls"] = Sm120BlockScaledGemmKernel
        return Sm120BlockScaledGemmKernel

    # ---- cute 张量构造（直配：生产字节零拷贝语义）----
    # 官方示例的张量逻辑形状为 (M, K, L)（K 连续、L 最外层存储）——
    # cutlass_torch.matrix 用 empty(l,m,k).permute(1,2,0) 构造。此处镜像之：
    # storage (l, m, k) 连续，view (m, k, l)，fp4 元素步长 (k, 1, m*k) 恰为
    # "k-major 打包"（前 m*k/2 字节 = packed [m, k//2]，probe 0xA1 实证）。
    def _fp4_tensor(self, packed_u8, m, k, cutlass):
        from cutlass.cute.runtime import from_dlpack
        storage = torch.empty(1, m, k, dtype=torch.int8, device="cuda")
        storage.view(torch.uint8).reshape(-1)[: m * k // 2] = \
            packed_u8.reshape(-1).to("cuda")
        view = storage.permute(1, 2, 0)                     # (m, k, l)
        t = from_dlpack(view, assumed_align=16)
        t.element_type = cutlass.Float4E2M1FN
        # 镜像官方示例：mark_layout_dynamic（cute_tensor_like 内部，leading=k）
        # + mark_compact_shape_dynamic（run_bs 显式，k-major: mode=1, order (2,0,1)）
        t = t.mark_layout_dynamic(leading_dim=1)
        t = t.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=2)
        return t, storage

    def _sf_tensor(self, swz_u8, cutlass):
        from cutlass.cute.runtime import from_dlpack
        assert swz_u8.data_ptr() % 16 == 0
        t = from_dlpack(swz_u8.contiguous(), assumed_align=16)
        t.element_type = cutlass.Float8E8M0FNU
        return t

    def _c_tensor(self, m, n, torch_dtype, cutlass):
        from cutlass.cute.runtime import from_dlpack
        storage = torch.empty(1, m, n, dtype=torch_dtype, device="cuda")
        view = storage.permute(1, 2, 0)                     # (m, n, l)，n 连续
        t = from_dlpack(view, assumed_align=16)
        t = t.mark_layout_dynamic(leading_dim=1)
        t = t.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1)
        return t, storage

    # ---- 主入口 ----
    def gemm(self, A: torch.Tensor, W_packed: torch.Tensor,
             W_scale: torch.Tensor, out_dtype: torch.dtype = torch.float16):
        """A [M,K] × 生产 MXF4 W(N,K) → out [M,N]（fp16/bf16，B-N1 铁律）。
        W_packed [N, K//2] uint8、W_scale [N, K//32] uint8（生产直配）。"""
        assert out_dtype in (torch.float16, torch.bfloat16), \
            f"B-N1: c_dtype 必须 16-bit，got {out_dtype}"
        Sm120 = self._load()
        cutlass = self._mods["cutlass"]
        import cutlass.torch as cutlass_torch
        from cutlass.cute.runtime import from_dlpack

        M, K = A.shape
        N, Kh = W_packed.shape
        assert Kh * 2 == K, f"K mismatch: A K={K}, W_packed {tuple(W_packed.shape)}"
        assert W_scale.shape == (N, K // SF_VEC), \
            f"W_scale 应 [N,K//32]=({N},{K // SF_VEC})，got {tuple(W_scale.shape)}"
        assert K % 128 == 0 and N % 128 == 0, "K/N 需 128 倍数（tile/atom 对齐）"
        W_packed = W_packed.contiguous()
        W_scale = W_scale.contiguous()

        # 1. A 侧量化（MXF4 语义）
        codes, a_packed, a_scale = quantize_a(A)

        # 2. scale atom-swizzle（A: M 补齐 128；B: N 天然 128 倍数）
        sfa_swz = sf_plain_to_atom(a_scale).to("cuda")
        sfb_swz = sf_plain_to_atom(W_scale).to("cuda")

        # 3. cute 张量（直配字节）
        a_tensor, _ = self._fp4_tensor(a_packed, M, K, cutlass)
        b_tensor, _ = self._fp4_tensor(W_packed, N, K, cutlass)
        sfa_tensor = self._sf_tensor(sfa_swz, cutlass)
        sfb_tensor = self._sf_tensor(sfb_swz, cutlass)
        c_tensor, storage_c = self._c_tensor(M, N, out_dtype, cutlass)

        # 4. 编译（按 shape 缓存）
        key = (M, N, K, out_dtype, self.tile, self.epi, self.variant)
        if key not in self._compiled:
            if not Sm120.is_valid_tensor_alignment(
                    M, N, K, 1, cutlass.Float4E2M1FN,
                    cutlass.Float16 if out_dtype is torch.float16
                    else cutlass.BFloat16, "k", "k", "n"):
                raise ValueError(f"tensor alignment 校验失败: M={M},N={N},K={K}")
            gemm = Sm120(cutlass.Float32, SF_VEC, self.tile, self.epi)
            hw = cutlass.utils.HardwareInfo()
            max_clusters = hw.get_max_active_clusters(1)
            stream = cutlass_torch.default_stream()
            self._compiled[key] = (
                __import__("cutlass").cute.compile(
                    gemm, a_tensor, b_tensor, sfa_tensor, sfb_tensor,
                    c_tensor, max_clusters, stream),
                stream)
        compiled, stream = self._compiled[key]
        compiled(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, stream)
        torch.cuda.synchronize()
        return storage_c[0]                # [M, N] fp16

    # ---- 参考（判据用）----
    @staticmethod
    def reference(A: torch.Tensor, W_packed: torch.Tensor,
                  W_scale: torch.Tensor) -> torch.Tensor:
        """量化参考：dequant_a(quant(A)) @ dequant_w(W).T，f32（TF32 关闭）。"""
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            codes, _, scale = quantize_a(A)
            A_deq = dequant_a(codes, scale)                    # [M, K]
            W_deq = dequant_w_mxf4(W_packed, W_scale)          # [N, K]
            return A_deq @ W_deq.t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


# ===========================================================================
# 自检（无需 GPU/DSL）：量化语义 + swizzle 往返
# ===========================================================================
def _selfcheck():
    dev = torch.device("cpu")
    # T1 E8M0 校准
    z = torch.zeros(1, 64)
    assert encode_e8m0_32(z).flatten().tolist() == [24, 24]
    big = torch.full((1, 64), 1e6)
    assert encode_e8m0_32(big).flatten().tolist() == [144, 144]
    # T2 swizzle 往返
    plain = torch.randint(0, 256, (257, 16), dtype=torch.uint8)
    buf = sf_plain_to_atom(plain)
    assert buf.shape == (1, 3, 4, 32, 4, 4), buf.shape
    back = sf_atom_to_plain(buf, 257)
    assert torch.equal(back, plain), "swizzle 往返不一致"
    # T3 pack/unpack 往返
    codes = quantize_e2m1(torch.randn(8, 64) * 2)
    assert torch.equal(unpack_e2m1(pack_e2m1(codes)), codes), "pack 往返不一致"
    print("✅ routeb_prod_adapter 自检通过（E8M0 校准 / swizzle 往返 / pack 往返）")


if __name__ == "__main__":
    _selfcheck()
