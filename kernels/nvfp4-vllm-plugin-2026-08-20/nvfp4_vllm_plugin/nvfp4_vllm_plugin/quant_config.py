# ============================================================================
# nvfp4_vllm_plugin/nvfp4_vllm_plugin/quant_config.py
# ============================================================================
"""QuantizationConfig("nvfp4_4w4a_sm121") —— kernel① v15 prefill 专用路径。

依据生产报告：生产 MoE prefill 走 B12xExperts（w4a16），本配置是**新增的
prefill 4W4A 分支**（非替换）：
  - --quantization nvfp4_4w4a_sm121 时，仅 prefill（M≥阈值）走 v15；
  - decode 维持 B12X/Marlin 原路径（M<阈值 → 回落原 MoE 方法）。
"""

import os
import torch
from typing import List, Tuple

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

# prefill/decode 分界（与 MoE 专家 GEMM 算力/带宽拐点对齐，可 env 覆盖）
PREFILL_M_THRESHOLD = int(os.environ.get("NVFP4_PREFILL_M", "256"))


@QuantizationConfig.register  # 兼容 vLLM 0.26 注册机制（无 register 装饰器则用 register_quantization_config）
class Nvfp4W4A4Config(QuantizationConfig):
    def __init__(self, weight_block_size: Tuple[int, int] = (128, 128)):
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "nvfp4_4w4a_sm121"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    def get_min_capability(self) -> int:
        return (12, 0)  # SM12x（GB10）

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    def get_quant_method(self, layer, prefix: str):
        from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinMoEMethod
        # 仅 MoE 层启用；其他层回落默认
        if isinstance(layer, FusedMoEMethodBase) or "moe" in prefix:
            from .moe_method import Nvfp4W4A4MoEMethod
            return Nvfp4W4A4MoEMethod(self)
        return None  # 默认方法

    def get_scaled_act_names(self) -> List[str]:
        return []
