#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""routea_plugin_a1_bprime — b′（native 共享）W4A4 hybrid 插件。

plugin_a1（池集成版 /tmp/_wsdedup_l3/w4a4_experts_pooled.py）的 b′ 演进：
hybrid 模式下 W4A16 侧从破坏性 packed prepare 切到 b12x 0.15.3 内置的非破坏
native（modelopt）路径，W4A16 kernel 与 W4A4 wrapper 共享同一份 FP4 payload
（A3 报告 §3 路线 b′，§4.2 实施骨架）。

env 开关（三开关互相独立，回滚 <10 分钟）:
  VLLM_MOE_W4A4          0=off(默认, 生产原样) 1=hybrid(M≥MIN_M 走 W4A4)
                         2=full(全量 W4A4, 不保留 W4A16)
  VLLM_MOE_W4A4_NATIVE   1=hybrid 的 W4A16 侧走 native 共享路径（仅 mode=1
                         生效; 0/未设 = 与池化版 plugin_a1 行为一致）
  VLLM_B12X_SHARED_WRAPPER  1=wrapper 几何键共享池（overlay 侧, 同 phase3b）
  VLLM_MOE_W4A4_MIN_M    hybrid W4A4 分派阈值（默认 3072）
  VLLM_MOE_W4A4_CG       wrapper use_cuda_graph（默认 1）
  VLLM_MOE_W4A4_DEBUG    1=打印每次分派决策

部署注意（窗口阶段）: 本插件与 plugin_a1 读同一 env VLLM_MOE_W4A4 且都 patch
backend_to_kernel_cls——**两插件不得同时经 entry point 激活**。窗口部署时须
先移除/禁用 plugin_a1 的 vllm.general_plugins 注册（pip uninstall 或替换
PYTHONPATH），否则类解析顺序不确定。

monkey-patch 三处（均在 EngineCore 进程内于插件加载时生效）:
  1. oracle.mxfp4.backend_to_kernel_cls / quantization.mxfp4.
     select_deepseek_v4_mxfp4_moe_backend（同 plugin_a1，注入
     W4A4B12xExperts）
  2. b12x.integration.tp_moe._w4a16_weight_layout_for_source（b′ 新增,
     VLLM_MOE_W4A4_NATIVE=1 且 mode=1 时）: 对 fp4_e8m0_k32 返回
     "modelopt"——plan 侧（_plan_core_workspace :1334 / 冻结 arena prewarm
     :3259）与运行时 prepared.weight_layout（:5319）一致，失配即
     RuntimeError（tp_moe.py:2490）。不 patch 则 plan 预编译 packed launch
     与运行时 native prepared 失配。
"""
import os

_MODE = os.environ.get("VLLM_MOE_W4A4", "0")
_NATIVE = os.environ.get("VLLM_MOE_W4A4_NATIVE", "0")

_installed = False
_native_patch_installed = False


def _install_native_layout_policy_patch() -> bool:
    """b′: b12x serving 布局政策对 fp4_e8m0_k32 改判 "modelopt"（全局,
    进程生命周期内不撤销; 幂等）。仅 native hybrid 模式调用。"""
    global _native_patch_installed
    if _native_patch_installed:
        return True
    from b12x.integration import tp_moe as b12x_tp_moe

    current = getattr(b12x_tp_moe, "_w4a16_weight_layout_for_source", None)
    if current is None:
        return False
    if getattr(current, "_bprime_patched", False):
        _native_patch_installed = True
        return True
    orig = current

    def patched(source_format: str) -> str:
        try:
            fmt = b12x_tp_moe._normalize_fp4_source_format(source_format)
        except Exception:
            return orig(source_format)
        if fmt == "fp4_e8m0_k32":
            return "modelopt"
        return orig(source_format)

    patched._bprime_orig = orig
    patched._bprime_patched = True
    b12x_tp_moe._w4a16_weight_layout_for_source = patched
    _native_patch_installed = True
    from vllm.logger import init_logger
    init_logger("vllm.routea_plugin_a1_bprime").info(
        "[routea_plugin_a1_bprime] b' native layout policy installed: "
        "_w4a16_weight_layout_for_source(fp4_e8m0_k32) -> 'modelopt' "
        "(plan-side launches will match runtime native prepared weights)")
    return True


def install() -> bool:
    """把 W4A4B12xExperts 注入 flashinfer_b12x 的 experts 类解析。幂等。"""
    global _installed
    if _installed:
        return True
    if _MODE not in ("1", "2"):
        return False
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle_mxfp4
    from vllm.model_executor.layers.quantization import mxfp4 as quant_mxfp4
    from routea_plugin_a1_bprime.w4a4_experts import W4A4B12xExperts

    orig_backend_to_cls = oracle_mxfp4.backend_to_kernel_cls

    def patched_backend_to_cls(backend):
        if backend is oracle_mxfp4.Mxfp4MoeBackend.B12X_MXFP4:
            return [W4A4B12xExperts]
        return orig_backend_to_cls(backend)

    oracle_mxfp4.backend_to_kernel_cls = patched_backend_to_cls

    orig_select = getattr(quant_mxfp4, "select_deepseek_v4_mxfp4_moe_backend", None)
    if orig_select is not None:
        def patched_select(config):
            backend, experts_cls = orig_select(config)
            if backend is oracle_mxfp4.Mxfp4MoeBackend.B12X_MXFP4:
                from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (
                    B12xExperts)
                if experts_cls is B12xExperts:
                    experts_cls = W4A4B12xExperts
            return backend, experts_cls

        quant_mxfp4.select_deepseek_v4_mxfp4_moe_backend = patched_select

    _installed = True
    from vllm.logger import init_logger
    init_logger("vllm.routea_plugin_a1_bprime").info(
        "[routea_plugin_a1_bprime] installed: mode=%s native=%s min_m=%s "
        "(W4A4 via B12xMoEWrapper; accepts no swiglu_limit clamp — clamp "
        "effect measured 0.0000, Task#21)",
        _MODE, _NATIVE, os.environ.get("VLLM_MOE_W4A4_MIN_M", "3072"))

    if _NATIVE == "1":
        if _MODE == "1":
            _install_native_layout_policy_patch()
            # e8m0_k32 × micro direct（M≤8）在 b12x 0.15.3 输出错误数值
            # （L1 GPU 实证: E=32/64/128 多几何全 NaN 或 p50 rel~1.1, w13/w31
            # 双布局皆坏——上游 host-check 误放行; 见 bprime-impl 报告 G3）。
            # native 模式强制 M≤8 走主 GEMM（L1 实证与 packed 逐位相等/
            # 孤立元素 ≤2 ULP）。正确性优先于微基准收益, 显式覆盖并告警。
            prev = os.environ.get("B12X_W4A16_SMALL_M_DIRECT")
            if prev != "0":
                os.environ["B12X_W4A16_SMALL_M_DIRECT"] = "0"
                from vllm.logger import init_logger
                init_logger("vllm.routea_plugin_a1_bprime").warning(
                    "[routea_plugin_a1_bprime] B12X_W4A16_SMALL_M_DIRECT "
                    "forced to 0: e8m0 micro direct produces wrong results "
                    "on b12x 0.15.3 (L1-verified); native M<=8 decode uses "
                    "the main GEMM (bit-exact vs packed)"
                    + (f" (was {prev!r})" if prev is not None else ""))
        else:
            init_logger("vllm.routea_plugin_a1_bprime").warning(
                "[routea_plugin_a1_bprime] VLLM_MOE_W4A4_NATIVE=1 ignored: "
                "native sharing only applies to hybrid mode (VLLM_MOE_W4A4=1); "
                "mode=%s runs unchanged", _MODE)
    return True


# import 即安装（env 门控）; entry point 加载路径与显式 import 路径共用此行为
install()
