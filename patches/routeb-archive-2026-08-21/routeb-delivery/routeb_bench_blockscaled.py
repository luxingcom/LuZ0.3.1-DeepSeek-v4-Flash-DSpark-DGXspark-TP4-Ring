#!/usr/bin/env python3
"""
routeB Phase 2：SM121 dense NVFP4 blockscaled GEMM benchmark
（CUTLASS CuTe DSL — 官方 SM120 pingpong 示例编排版）
============================================================================
v2（Task #12 全量修复，2026-08-20）——替换 v1 的自制 kernel 骨架。

复现 baristankut 实证基线：4096×14336×4096 dense NVFP4 = 356 TFLOPS（71% of 500 峰值）
并扫 MoE 真实 shape（tile 128×128×128 / 128×128×256）。

【v1 → v2 修复说明】
v1（交付包原版）声称"仅缺 host launch ~150 行"，经逐行核实**不成立**：v1 的
`make_blockscaled_kernel` 只有 TiledMMA + SMEM 布局描述，缺失整个设备侧 kernel
（约 1000 行：warp 专用化、ab_stage 流水线、ldmatrix S2R、SF CopyUniversal、
pingpong 双 warp-group 调度、TMA epilogue）。向残缺骨架移植 host 段是伪修复。

v2 改为 **vendor 官方完整实现**（routeb_official/ 三件套，取自 NVIDIA/cutlass
main 分支 examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/）：
  - dense_blockscaled_gemm_persistent_pingpong.py  ← 主路径（warp-level MMA，SM120/121）
  - dense_blockscaled_gemm_persistent_cooperative.py ← 备用调度
  - blockscaled_gemm_dispatch.py                   ← MMA op 分派（MXF4 路径）
本脚本只做编排：shape/tile 解析 + 生产语义量化自检（--check）+ TFLOPS 汇总。

【A7 dtype 定案（生产镜像 4.5.2 mma.py 源码已核实）】
MXF4 路径经官方 dispatch 构造：
    MmaMXF4Op(Float4E2M1FN, Float32, Float8E8M0FNU)   # 3 个位置参数
sf_vec_size=32 固化在 MmaMXF4Op 类内（构造函数签名无此参数）。
v1 的 5 关键字参数写法（a_dtype/b_dtype/acc_dtype/sf_dtype/sf_vec_size）
在 4.5.2 下必然 TypeError。
另：4.5.2 的 MmaSM120BlockScaledOp.admissible_archs **原生含 sm_121a**，
P1 patch（sm_121a 放宽）在 DSL ≥4.5 上为 no-op，仅在 4.4.x 需要。

【量化工具修复（服务 P3 生产权重直配验证）】
  - encode_e8m0_32：补 +127，floor 偏置对齐 kernel2 v17 语义
    （零输入→24，1e6→144），clamp 移到 +127 之后
  - pack_e2m1_n：W [K,N] 直出 [K, N//2]（v1 的 W.t() 转置链是布局 bug）
  - make_w_scale：32(K)×128(N) block-max，对齐生产 W_scale [K//32, N//128]
  - 误差度量：相对误差 rel_err = max_err / ref.abs().max()
  - --check：真数值断言（非仅 print 形状）

用法：
  python routeb_bench_blockscaled.py --check                       # 量化语义自检（无需 DSL/GPU）
  python routeb_bench_blockscaled.py --shape 4096,14336,4096       # 复现 356 基线
  python routeb_bench_blockscaled.py --shape 256,512,1024 --tile 128,128,128   # 最小冒烟
  python routeb_bench_blockscaled.py                               # 默认 shape 集 + tile sweep

SASS 门禁（SM12x 硬门槛，不可绕过）：
  nvdisasm <cubin> | grep -c 'mma.*e2m1'   # >0 即原生 FP4 MMA；0 = bf16 回退，No-Go
  ⚠️ SM12x 勿用 tcgen05（那是 SM10x 指令）——本编排 vendored 的是 warp-level
     mma.sync kind::mxf4 路径，非 SM100 tcgen05 示例，两者勿混淆。
"""
import argparse
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL_DIR = os.path.join(_HERE, "routeb_official")

import torch

# ---------------------------------------------------------------------------
# 0. CUTLASS DSL 环境探测（--check 无需 DSL；bench 需要）
# ---------------------------------------------------------------------------
HAS_DSL = False
_CUTLASS_VER = None
try:
    import cutlass
    _CUTLASS_VER = getattr(cutlass, "version", None) or getattr(
        cutlass, "__version__", "unknown"
    )
    from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op  # noqa: F401

    HAS_DSL = True
except Exception as _e:  # pragma: no cover
    _DSL_ERR = _e
    print(f"⚠️  CUTLASS DSL 不可用（{_e}）——仅 --check 模式可用")


def probe_mma_op():
    """现场核对 MmaMXF4Op 签名与 admissible_archs（一次性打印，供现场裁决）。"""
    import inspect

    from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op, MmaSM120BlockScaledOp

    print(f"  cutlass version        : {_CUTLASS_VER}")
    print(f"  MmaMXF4Op.__init__     : {inspect.signature(MmaMXF4Op.__init__)}")
    archs = getattr(MmaSM120BlockScaledOp, "admissible_archs", None)
    print(f"  admissible_archs       : {archs}")
    has_121a = any("121" in str(a) for a in (archs or []))
    if not has_121a:
        print("  ❌ admissible_archs 不含 sm_121a —— 需先跑 patch_cutlass_dsl_sm121a.py（仅 4.4.x 需要）")
        return False
    print("  ✅ sm_121a 原生支持（DSL ≥4.5，无需 patch）")
    return True


# ---------------------------------------------------------------------------
# 1. 生产语义量化工具（torch 参考实现，修复版）
#    —— 语义对齐 kernel2 交付包 v17：E8M0 = floor(log2(amax/6)) + 127，
#       amax 下限 1e-30（零输入 → byte 24）；tie 取低档；信封/打包低半字节=偶元素。
# ---------------------------------------------------------------------------
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # 幅值表（正）
SF_VEC = 32          # MXF4：每 32 个 K 元素共享 1 个 UE8M0 scale（与生产直配）
SMEM_BUDGET = 101376  # SM121 共享内存硬上限（bytes）——仅作参考信息，官方 kernel 自算 stages


def flops(M, N, K):
    return 2.0 * M * N * K


def smem_estimate(tile_m, tile_n, tile_k, num_stages=3):
    """估算 tile 的 SMEM 占用（bytes）——仅信息性参考（官方 kernel 内部自算 stages）。

    v2 修复（A5）：删除 v1 的 acc_bytes 项——F32 累加器驻留寄存器堆（RF）而非 SMEM，
    v1 把它计入 SMEM 导致 256×128×128 被误判 204.5KB（实际 ~76.5KB），
    系统性跳过所有最优 tile。
    """
    a_bytes = tile_m * tile_k // 2            # e2m1 打包 4bit
    b_bytes = tile_n * tile_k // 2
    sf_bytes = (tile_m * tile_k // SF_VEC) + (tile_n * tile_k // SF_VEC)  # UE8M0 1B/元素
    return num_stages * (a_bytes + b_bytes + sf_bytes)


def encode_e8m0_32(x: torch.Tensor, k_group: int = SF_VEC) -> torch.Tensor:
    """fp32 → E8M0 uint8。x: [..., K] → [..., K//32]

    v2 修复（A3）：floor(log2(amax/6)) **+127**，amax 下限 1e-30（v17 语义），
    clamp [0,255] 移到 +127 之后（v1 把 clamp(0,255) 打在指数上且漏 +127，
    scale 系统性偏差）。
    校准向量：零输入 → 24；1e6 → 144。
    """
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // k_group, k_group)
    amax = xb.abs().amax(-1)
    exp_int = torch.floor(
        torch.log2(torch.clamp(amax, min=1e-30) / 6.0)
    ).to(torch.int64) + 127
    exp_int = exp_int.clamp(0, 255)
    return exp_int.to(torch.uint8)


def pack_e2m1_n(x: torch.Tensor) -> torch.Tensor:
    """fp32 [..., n] → uint8 [..., n//2]（N 向打包：低半字节=偶元素，生产格式）。"""
    sgn = (x < 0).to(torch.int32)
    mag = torch.tensor(E2M1_MAG, device=x.device)
    d = (x.abs().unsqueeze(-1) - mag).abs()
    idx = d.argmin(-1).to(torch.int32)
    nib = (sgn * 8 + idx).to(torch.uint8)
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def make_w_scale(W: torch.Tensor) -> torch.Tensor:
    """W [K, N] fp32 → W_scale [K//32, N//128] uint8（E8M0）。

    v2 修复（A8）：v1 的 `encode_e8m0_32(W).reshape(K//32, N//128, 128).amax(-1)`
    是错的——encode 产出 [K, N//32]（K 行、N 向 32 分组），reshape 后既不是
    K 向 32 分组也不是 N 向 128 分组的 block-max。
    生产格式：scale[i,j] = E8M0( max |W[i*32:(i+1)*32, j*128:(j+1)*128]| )。
    """
    K, N = W.shape
    assert K % 32 == 0 and N % 128 == 0, f"W 需 K%32==0 且 N%128==0，got {W.shape}"
    blocks = W.reshape(K // 32, 32, N // 128, 128)
    block_max = blocks.abs().amax(dim=(1, 3))                    # [K//32, N//128]
    exp_int = torch.floor(
        torch.log2(torch.clamp(block_max, min=1e-30) / 6.0)
    ).to(torch.int64) + 127
    return exp_int.clamp(0, 255).to(torch.uint8)


def make_w_packed(W: torch.Tensor) -> torch.Tensor:
    """W [K, N] fp32 → W_packed [K, N//2] uint8（e2m1 N 向打包）。

    v2 修复（A6）：v1 的 `pack_e2m1_n(W.t()).t()` 产出 [K//2, N]（pack 沿
    转置后的末维 K 分组，维度错位）。pack_e2m1_n 本身打包末维，
    直接 `pack_e2m1_n(W)` 即直出 [K, N//2]。
    """
    return pack_e2m1_n(W)


def _quant_e2m1(x_norm: torch.Tensor) -> torch.Tensor:
    """就近码本量化（输入已归一化到 ±6 内）。tie 语义与 argmin 一致（取低档）。"""
    mag = torch.tensor(E2M1_MAG, device=x_norm.device)
    d = (x_norm.abs().unsqueeze(-1) - mag).abs()
    idx = d.argmin(-1)
    return x_norm.sign() * mag[idx]


def torch_reference(A, W, W_scale, bias=None):
    """正确性对照：A/W 分别量化 E2M1(E8M0 分组) → 反量化 matmul。

    W_scale [K//32, N//128]（make_w_scale 产出，生产格式）。
    v2 修复：v1（及本文件首版）W 侧只乘 scale 未做码本量化——
    W_deq = W·W_sf 只是缩放，量化从未发生（潜伏 bug，--check 真断言暴露）。
    正确路径：W/scale → clamp ±6 → E2M1 码本 → ×scale。
    """
    M, K = A.shape
    N = W.shape[1]

    # --- A 量化：32(K) 分组 E8M0 + E2M1 ---
    A_scale = encode_e8m0_32(A)                                   # [M, K//32]
    A_sf = torch.pow(2.0, A_scale.float() - 127.0).reshape(M, K // SF_VEC, 1)
    A_norm = torch.clamp(A.reshape(M, K // SF_VEC, SF_VEC) / A_sf, -6.0, 6.0)
    A_deq = (_quant_e2m1(A_norm) * A_sf).reshape(M, K)

    # --- W 量化：32(K)×128(N) block E8M0 + E2M1 ---
    W_sf = torch.pow(2.0, W_scale.float() - 127.0)                # [K//32, N//128]
    W_norm = torch.clamp(
        W.reshape(K // 32, 32, N // 128, 128) / W_sf[:, None, :, None],
        -6.0, 6.0,
    )
    W_deq = (_quant_e2m1(W_norm) * W_sf[:, None, :, None]).reshape(K, N)

    out = A_deq @ W_deq
    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# 2. --check：量化语义自检（真数值断言，无需 DSL/GPU）
# ---------------------------------------------------------------------------
def run_check():
    print("=== 量化语义自检（torch 参考，无需 DSL/GPU）===")
    failures = []

    # T1: E8M0 编码校准向量（team-lead 指定：零输入→24，1e6→144）
    z = torch.zeros(1, 64)
    b_z = encode_e8m0_32(z)
    assert b_z.shape == (1, 2), f"encode shape: {b_z.shape}"
    assert b_z.flatten().tolist() == [24, 24], \
        f"零输入应 → byte 24，got {b_z.flatten().tolist()}"
    big = torch.full((1, 64), 1e6)
    b_big = encode_e8m0_32(big)
    assert b_big.flatten().tolist() == [144, 144], \
        f"1e6 输入应 → byte 144，got {b_big.flatten().tolist()}"
    print("  ✅ T1 encode_e8m0_32：+127 / floor 1e-30 / clamp 顺序（零→24，1e6→144）")

    # T2: W_packed 布局 [K, N//2]（A6）
    W = torch.randn(128, 256) * 0.3
    Wp = make_w_packed(W)
    assert Wp.shape == (128, 128), f"W_packed 应 [K,N//2]=(128,128)，got {Wp.shape}"
    assert Wp.dtype == torch.uint8
    # 抽查首字节：偶元素低半字节、奇元素高半字节
    sgn = (W[0, 0] < 0).to(torch.int32)
    mag = torch.tensor(E2M1_MAG)
    idx0 = (W[0, 0].abs() - mag).abs().argmin().item()
    idx1 = (W[0, 1].abs() - mag).abs().argmin().item()
    expect = ((sgn * 8 + idx0) | (((W[0, 1] < 0).to(torch.int32) * 8 + idx1) << 4)) & 0xFF
    assert Wp[0, 0].item() == expect, \
        f"pack 字节不符：got {Wp[0, 0].item()}, expect {expect}"
    print("  ✅ T2 W_packed：[K, N//2] 直出 + 低半字节=偶元素")

    # T3: W_scale 32(K)×128(N) block-max（A8）
    K, N = 128, 256
    W = torch.zeros(K, N)
    W[10, 200] = 6.0            # 落在 block (0, 1)，仅该 block 非零
    W[64, 0] = 96.0             # 落在 block (2, 0)
    Ws = make_w_scale(W)
    assert Ws.shape == (K // 32, N // 128), f"W_scale shape {Ws.shape}"
    # block(0,1): max=6 → floor(log2(1))+127 = 127
    assert Ws[0, 1].item() == 127, f"block(0,1) 应 127，got {Ws[0, 1].item()}"
    # block(2,0): max=96 → floor(log2(16))+127 = 131
    assert Ws[2, 0].item() == 131, f"block(2,0) 应 131，got {Ws[2, 0].item()}"
    assert Ws[0, 0].item() == 24, "零 block 应 24"  # 零输入→24
    print("  ✅ T3 W_scale：32(K)×128(N) block-max 对齐生产 [K//32, N//128]")

    # T4: 量化-反量化往返（相对误差，A14 语义）
    # 注：E8M0 取 floor(log2(amax/6))（v17 家族语义）→ 组内最大元素归一化到
    # [6,12) 后 clamp 到 6，含系统性截断偏置，故 NVFP4+floor 的端到端 rel_err
    # 预期在 ~0.1-0.25 区间（阈值 0.3），显著高于 ceil 语义的实现。
    torch.manual_seed(0)
    M, K, N = 256, 1024, 512
    A = torch.randn(M, K) * 0.5
    W = torch.randn(K, N) * 0.3
    Ws = make_w_scale(W)
    ref_q = torch_reference(A, W, Ws)          # 量化参考
    ref_fp = A @ W                              # 全精度参考
    rel_err = (ref_q - ref_fp).abs().max() / ref_fp.abs().max()
    assert rel_err < 0.3, f"量化 vs 全精度 rel_err={rel_err:.4f} 过大（NVFP4+floor 预期 ~0.1-0.25）"
    print(f"  ✅ T4 量化往返 rel_err={rel_err:.4f}（<0.3，NVFP4+floor 语义正常区间）")

    # T5: 自洽性——同一量化参考两次调用逐位一致（确定性）
    r1 = torch_reference(A, W, Ws)
    assert torch.equal(r1, ref_q), "量化参考非确定性"
    print("  ✅ T5 量化参考确定性（同输入恒同输出）")

    print("\n✅ --check 全部通过（T1-T5）——量化语义已对齐 kernel2 v17（零→24/1e6→144）")
    return True


# ---------------------------------------------------------------------------
# 3. bench：编排 vendored 官方 SM120 pingpong kernel
# ---------------------------------------------------------------------------
# 官方示例 CLI 验证允许的 FP4 tile（sf_vec_size=32 时 tile_K 必须是 128 的倍数）
ALLOWED_TILES = {(128, 128, 128), (128, 128, 256)}
EPI_TILES = {(128, 128), (64, 32)}

# v1 的默认 tile 集含 (256,128,128)/(128,256,128)/(64,128,128)——官方 DSL 示例
# 不支持（其 warp-level mma atom_layout=(2,2,1)、permutation 约束 tile M/N=128 的
# 倍数且经 validate 的仅上述两种）。baristankut 的 256×128 最优 tile 来自 SGLang
# C++/builder 路径，非本 DSL 示例——如需 256×128 需另行评估，见修复日志。


def _install_testing_compat():
    """DSL 4.5.2 兼容 shim：main 分支官方示例 `from cutlass import testing`，
    而 4.5.2 的同名设施（JitArguments / benchmark / get_workspace_count /
    convert）在 cutlass.cute.testing 下（已逐一核实存在于安装版 4.5.2）。
    做模块别名 + 属性注入，使 vendored 示例无需改动即可 import。
    """
    import cutlass
    import cutlass.cute.testing as _cute_testing

    if not hasattr(cutlass, "testing"):
        cutlass.testing = _cute_testing
        sys.modules["cutlass.testing"] = _cute_testing
    return _cute_testing


def bench_one(M, N, K, tile, warmup, iterations, skip_ref_check=False,
              epi_tile=(128, 128), sf_vec_size=32, c_dtype_name="Float16"):
    """单 shape 基准：调用官方 run_bs，返回 TFLOPS。

    [B-N1 修复 2026-08-20] c_dtype 默认 Float16（原为 Float32——B-N1 根因）：
    官方 SM120 示例 epilogue 的 C-atom 为 StMatrix8x8x16bOp（16-bit 专用），
    以 f32 实例化时每线程值数 4→2、M tiler 粒度 32→16 行，epilogue 填充循环
    按 32 行粒度索引 → 一半累加器值被丢弃 → 输出结构性 50% 错误。
    fp16 实测 100% 逐位精确（256×256×512 ones 全对）；生产 prefill GEMM
    输出即 fp16/bf16，语义无损失。非 16-bit c_dtype 由 vendored 示例侧护栏
    直接 raise（防静默半错）。
    """
    _install_testing_compat()
    _variant = os.environ.get("ROUTEB_KERNEL", "pingpong")  # pingpong|cooperative
    _mod = ("dense_blockscaled_gemm_persistent_pingpong" if _variant == "pingpong"
            else "dense_blockscaled_gemm_persistent_cooperative")
    sys.path.insert(0, _OFFICIAL_DIR)
    try:
        if _variant == "cooperative":
            from dense_blockscaled_gemm_persistent_cooperative import run_bs
        else:
            from dense_blockscaled_gemm_persistent_pingpong import run_bs
    finally:
        sys.path.remove(_OFFICIAL_DIR)

    import cutlass

    if sf_vec_size == 16:
        sf_dtype = cutlass.Float8E4M3FN     # NVFP4（官方示例默认路径）
    else:
        sf_dtype = cutlass.Float8E8M0FNU    # MXF4（生产直配路径）

    exec_time_us = run_bs(
        mnkl=(M, N, K, 1),
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,             # MXF4: UE8M0+vec32；NVFP4: E4M3+vec16
        sf_vec_size=sf_vec_size,
        c_dtype=getattr(cutlass, c_dtype_name),  # [B-N1] 必须 16-bit（fp16/bf16）
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mnk=tile,
        epi_tile=epi_tile,
        tolerance=1e-1,
        warmup_iterations=warmup,
        iterations=iterations,
        skip_ref_check=skip_ref_check,
    )
    # run_bs 返回每迭代微秒
    tflops = flops(M, N, K) / (exec_time_us * 1e-6) / 1e12
    return tflops


def main():
    ap = argparse.ArgumentParser(description="routeB SM121 NVFP4 blockscaled GEMM 复现（官方示例编排版）")
    ap.add_argument("--shape", default=None, help="单 shape M,N,K（逗号分隔）")
    ap.add_argument("--tile", default=None, help="tile M,N,K（逗号分隔），默认 sweep（128,128,128 与 128,128,256）")
    ap.add_argument("--epi", default="128,128", help="epilogue tile M,N（默认 128,128；官方示例亦支持 64,32）")
    ap.add_argument("--sf-vec", type=int, choices=[16, 32], default=32,
                    help="scale 分组：32=MXF4(UE8M0,生产直配) 16=NVFP4(E4M3,官方示例默认路径)")
    ap.add_argument("--c-dtype", default="Float16",
                    choices=["Float16", "BFloat16"],
                    help="输出 C dtype（[B-N1] 必须 16-bit；f32 会触发示例 epilogue 半错，已由护栏拦截）")
    ap.add_argument("--check", action="store_true", help="仅量化语义自检（无需 DSL/GPU）")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--skip-ref-check", action="store_true", help="跳过官方参考校验（仅计时）")
    args = ap.parse_args()

    if args.check:
        ok = run_check()
        sys.exit(0 if ok else 1)

    if not HAS_DSL:
        print("❌ CUTLASS DSL 未安装——bench 需要 DSL（--check 可无 DSL 运行）")
        sys.exit(1)

    print("=== DSL 环境探测 ===")
    if not probe_mma_op():
        sys.exit(1)

    if args.shape:
        shapes = [tuple(int(x) for x in args.shape.split(","))]
    else:
        shapes = [
            (4096, 14336, 4096),   # 复现 356 基线（主 shape）
            (4096, 4096, 4096),
            (2048, 4096, 4096),    # MoE w1/w3 真实
            (512, 4096, 4096),
            (256, 2048, 4096),     # MoE w2
        ]

    if args.tile:
        tiles = [tuple(int(x) for x in args.tile.split(","))]
    else:
        tiles = [(128, 128, 128), (128, 128, 256)]

    for t in tiles:
        if t not in ALLOWED_TILES:
            print(f"❌ tile {t} 不被官方 SM120 DSL 示例支持（允许：{sorted(ALLOWED_TILES)}）。"
                  f"baristankut 的 256×128 最优 tile 来自 SGLang C++ 路径，见修复日志 §偏离说明。")
            sys.exit(1)

    print(f"\nCUTLASS DSL 就绪 | MXF4 (E2M1×E2M1 + UE8M0, sf_vec=32) | SMEM 预算 {SMEM_BUDGET//1024}KB")
    for M, N, K in shapes:
        print(f"\n=== shape ({M},{N},{K}) ===")
        best = (0.0, None)
        for tile in tiles:
            est = smem_estimate(*tile)
            print(f"  -- tile {tile}（SMEM 估算 {est/1024:.0f}KB，参考值）...")
            tflops = bench_one(M, N, K, tile, args.warmup, args.iterations,
                               skip_ref_check=args.skip_ref_check,
                               epi_tile=tuple(int(x) for x in args.epi.split(",")),
                               sf_vec_size=args.sf_vec,
                               c_dtype_name=args.c_dtype)
            print(f"  tile {tile[0]}x{tile[1]}x{tile[2]}: {tflops:7.1f} TFLOPS")
            if tflops > best[0]:
                best = (tflops, tile)
        if best[1]:
            print(f"  ★ 最优: {best[1]} → {best[0]:.1f} TFLOPS")
            if best[0] < 350.0:
                print(f"  ⚠️  未达 350 TFLOPS 门禁（当前 {best[0]:.1f}）——对照社区基线 356 排查")

    print("\n验收：4096×14336×4096 ≥350 TFLOPS（社区基线 356）")
    print("SASS 门禁：nvdisasm <cubin> | grep 'mma.*e2m1'（>0 命中；0=bf16 回退 No-Go）")
    print("  cubin 位置：cutlass DSL 编译缓存（默认 ~/.cutlass/cache 或 $CUTLASS_CACHE_DIR）")


if __name__ == "__main__":
    main()
