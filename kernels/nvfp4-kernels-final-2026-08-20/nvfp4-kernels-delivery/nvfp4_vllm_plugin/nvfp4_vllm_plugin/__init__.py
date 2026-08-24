# ============================================================================
# nvfp4_vllm_plugin/nvfp4_vllm_plugin/__init__.py
# ============================================================================
"""NVFP4 双算子推理工作流插件（新建路径，非替换生产 B12X/fused_compress）。

原则（依据生产加载报告 2026-08-20）：
  - 生产 MoE prefill 走 B12xExperts（w4a16），KV 走 fused_compress_quant_cache（fp8）
  - 本插件仅注册**旁路/新增**能力，默认不改变任何生产调用点：
      · kernel① v15  → prefill 专用 MoE 路径（--quantization nvfp4_4w4a_sm121 时启用）
      · kernel② v17  → NVFP4 KV 写回路径（--kv-cache-dtype nvfp4_ds_mla 时启用）
  - A/B 证明先行：未验证加速比前保持 B12X/fused_compress 原样
"""

import os

_ENABLED_K1 = os.environ.get("VLLM_NVFP4_K1", "0") == "1"   # v15 prefill 路径开关
_ENABLED_K2 = os.environ.get("VLLM_NVFP4_K2", "0") == "1"   # v17 KV 写回开关


def __getattr__(name):
    # 惰性导入，避免插件加载拖慢 vLLM 启动
    if name == "kv_writer" and _ENABLED_K2:
        from . import kv_writer
        return kv_writer
    if name == "moe_method" and _ENABLED_K1:
        from . import moe_method
        return moe_method
    raise AttributeError(name)


__all__ = ["kv_writer", "moe_method"]
