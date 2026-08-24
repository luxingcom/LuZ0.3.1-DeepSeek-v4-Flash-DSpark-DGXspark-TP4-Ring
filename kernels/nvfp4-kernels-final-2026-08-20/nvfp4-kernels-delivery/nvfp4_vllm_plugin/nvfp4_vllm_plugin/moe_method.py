# ============================================================================
# nvfp4_vllm_plugin/nvfp4_vllm_plugin/moe_method.py
# ============================================================================
"""kernel① v15 MoE 方法：prefill 4W4A 分支（新建路径，decode 回落原方法）。

依据生产报告：B12xExperts 走 `_run_b12x_moe_fp4`（w4a16，功能完整）。
本方法**只接管 prefill（M≥阈值）段**，decode 段委托原 B12X/Marlin 方法——
分派在 forward 内做，天然支持 CUDA Graph 按 phase 分别捕获。
"""

import os
import torch
from typing import Optional

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoE
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .quant_config import PREFILL_M_THRESHOLD


class Nvfp4W4A4MoEMethod(FusedMoEMethodBase):
    """MoE 方法：prefill → v15 4W4A；decode → 原方法（_fallback）。"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._fallback: Optional[FusedMoEMethodBase] = None   # 原 B12X/Marlin 方法
        self._w_cached = False

    # ---- 权重处理：v15 需要 [K, N//2] + [K//32, N//128] ----
    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """从 layer 权重解析 NVFP4 格式并预热 v15 W 缓存（每层一次）。

        权重来源：转换器 convert_mxfp4_to_nvfp4.py 产出的
        W_packed [K, N//2] + W_scale [K//32, N//128]（挂在 layer.w13_packed / layer.w2_packed）。
        若权重非 NVFP4 格式（B12X MXFP4 原版权重），保持原路径（_w_cached=False）。
        """
        try:
            from nvfp4_4w4a_prefill_gemm_v15_triton import (
                nvfp4_4w4a_prefill_gemm, preprocess_weights_clear)
            packed = getattr(layer, "w13_packed", None) or getattr(layer, "w1_packed", None)
            if packed is not None:
                preprocess_weights_clear()
                self._w_cached = True
        except Exception:
            self._w_cached = False

    # ---- forward：M 阈值分派（vLLM 0.26 新签名：router 为参数；兼容旧签名）----
    def apply(
        self,
        layer: FusedMoE,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """双签名兼容：
        0.26 新签名 apply(layer, router, x, router_logits)
        旧签名   apply(layer, x, router_logits, top_k, renormalize)
        """
        # 解析新签名（router 对象）
        if len(args) >= 3 and hasattr(args[0], "select_experts"):
            router, x, router_logits = args[0], args[1], args[2]
            M = x.shape[0]
            if self._w_cached and M >= PREFILL_M_THRESHOLD and os.environ.get("VLLM_NVFP4_K1", "0") == "1":
                return self._nvfp4_prefill(layer, router, x)
            return self._fallback(layer, router, x)
        # 旧签名（x, router_logits, top_k, renormalize）
        x, router_logits = args[0], args[1]
        M = x.shape[0]
        if self._w_cached and M >= PREFILL_M_THRESHOLD and os.environ.get("VLLM_NVFP4_K1", "0") == "1":
            return self._nvfp4_prefill_legacy(layer, x, router_logits,
                                              kwargs.get("top_k", 6),
                                              kwargs.get("renormalize", True))
        return self._fallback_legacy(layer, x, router_logits,
                                     kwargs.get("top_k", 6),
                                     kwargs.get("renormalize", True), **kwargs)

    def _fallback(self, layer, router, x):
        """委托原方法（B12X/Marlin）——生产 B12X 链路零改动。"""
        if self._fallback is not None:
            return self._fallback.apply(layer, router, x)
        return layer.default_method.apply(layer, router, x)

    def _fallback_legacy(self, layer, x, router_logits, top_k, renormalize, **kwargs):
        if self._fallback is not None:
            return self._fallback.apply(layer, x, router_logits, top_k,
                                        renormalize=renormalize, **kwargs)
        return layer.default_method.apply(layer, x, router_logits, top_k,
                                          renormalize=renormalize, **kwargs)

    def _nvfp4_prefill(self, layer, router, x):
        """v15 4W4A prefill（0.26 签名）：router 已就绪 → topk 重排 → 专家 GEMM。"""
        from nvfp4_4w4a_prefill_gemm_v15_triton import nvfp4_4w4a_prefill_gemm
        # 骨架：按生产层权重布局（w13_packed/w2_packed 或 w1/w2/w3）补全
        raise NotImplementedError(
            "v15 prefill 路径骨架：先跑 ab_routeA_vs_b12x.py 证明加速比，"
            "再按生产权重布局补全专家分组调用。")

    def _nvfp4_prefill_legacy(self, layer, x, router_logits, top_k, renormalize):
        return self._nvfp4_prefill(layer, None, x)
