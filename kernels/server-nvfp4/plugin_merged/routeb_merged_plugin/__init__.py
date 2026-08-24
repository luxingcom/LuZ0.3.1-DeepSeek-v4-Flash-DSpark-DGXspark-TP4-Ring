#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""routeb_merged_plugin — merged-GEMM（专家 N 维合并 + 静态间接寻址）vLLM 插件。
Phase C ⑤ 骨架（Task #26）。B12X W4A16 生产路径上的 prefill 加速分支。

env:
  VLLM_MOE_MERGED        0=off(默认, 生产原样, 零污染) 1=merged prefill 分支
  VLLM_MOE_MERGED_MIN_M  merged 桶最小 M_g（默认 128; mini 验证口径; 生产建议 256-512）
  ROUTEB_V2_PATH         routeb_official_v2 kernel 目录（默认 /routeb/routeb_official_v2）

安装: PYTHONPATH 指向本包父目录后 `import routeb_merged_plugin; install()`。
monkey-patch 同 A′ 插件（Task#22）双挂点: oracle.mxfp4.backend_to_kernel_cls +
quantization.mxfp4.select_deepseek_v4_mxfp4_moe_backend。
"""
import os

_MODE = os.environ.get("VLLM_MOE_MERGED", "0")

_installed = False


def install() -> bool:
    """把 MergedB12xExperts 注入 B12X experts 类解析。幂等。"""
    global _installed
    if _installed:
        return True
    if _MODE != "1":
        return False
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle_mxfp4
    from vllm.model_executor.layers.quantization import mxfp4 as quant_mxfp4
    from routeb_merged_plugin.merged_experts import MergedB12xExperts

    orig_backend_to_cls = oracle_mxfp4.backend_to_kernel_cls

    def patched_backend_to_cls(backend):
        if backend is oracle_mxfp4.Mxfp4MoeBackend.B12X_MXFP4:
            return [MergedB12xExperts]
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
                    experts_cls = MergedB12xExperts
            return backend, experts_cls

        quant_mxfp4.select_deepseek_v4_mxfp4_moe_backend = patched_select

    _installed = True
    from vllm.logger import init_logger
    init_logger(__name__).info(
        "[routeb_merged_plugin] installed: mode=%s min_m=%s "
        "(merged-GEMM v2 间接寻址, Phase C 骨架; kernel/管线级验证见 Phase C 报告)",
        _MODE, os.environ.get("VLLM_MOE_MERGED_MIN_M", "128"))
    return True
