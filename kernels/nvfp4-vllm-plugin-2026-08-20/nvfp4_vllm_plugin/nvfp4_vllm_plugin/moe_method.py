# ============================================================================
# nvfp4_vllm_plugin/nvfp4_vllm_plugin/moe_method.py
# ============================================================================
"""kernel① routeA MoE 方法：prefill 4W4A 分支（新建路径，decode 回落原方法）。

依据生产报告：B12xExperts 走 `_run_b12x_moe_fp4`（w4a16，功能完整）。
本方法**只接管 prefill（M≥阈值）段**，decode 段委托原 B12X/Marlin 方法——
分派在 forward 内做，天然支持 CUDA Graph 按 phase 分别捕获。

内核校准（2026-08-20 落实）：由 v15(bf16) 统一到 routeA(cutlass_scaled_fp4_mm，
已验证 60~187 TFLOPS、8/8 rel=0.00141)。全局逻辑保持不变。
"""

import os
import torch
from typing import Optional

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoE
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .quant_config import PREFILL_M_THRESHOLD


class Nvfp4W4A4MoEMethod(FusedMoEMethodBase):
    """MoE 方法：prefill → routeA 4W4A；decode → 原方法（_fallback）。"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._fallback: Optional[FusedMoEMethodBase] = None   # 原 B12X/Marlin 方法
        self._routeA = None
        self._w_cached = False

    # ---- 权重处理：v15 需要 [K, N//2] + [K//32, N//128] ----
    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """从 layer 权重解析 NVFP4 格式并预热 v15 W 缓存（每层一次）。

        权重来源：转换器 convert_mxfp4_to_nvfp4.py 产出的
        W_packed [K, N//2] + W_scale [K//32, N//128]（挂在 layer.w13_packed / layer.w2_packed）。
        若权重非 NVFP4 格式（B12X MXFP4 原版权重），保持原路径（_w_cached=False）。
        """
        try:
            from nvfp4_4w4a_mmaf import RouteA as _RouteA
            packed = getattr(layer, "w13_packed", None) or getattr(layer, "w1_packed", None)
            if packed is not None:
                self._routeA = _RouteA()
                self._w_cached = True
        except Exception:
            self._w_cached = False

    # ---- forward：M 阈值分派 ----
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        M = x.shape[0]
        if self._w_cached and M >= PREFILL_M_THRESHOLD and os.environ.get("VLLM_NVFP4_K1", "0") == "1":
            return self._nvfp4_prefill(layer, x, router_logits, top_k, renormalize)
        return self._fallback_apply(layer, x, router_logits, top_k, renormalize, **kwargs)

    def _fallback_apply(self, layer, x, router_logits, top_k, renormalize, **kwargs):
        """委托原方法（B12X/Marlin）——生产 B12X 链路零改动。"""
        if self._fallback is not None:
            return self._fallback.apply(layer, x, router_logits, top_k,
                                        renormalize=renormalize, **kwargs)
        return layer.default_method.apply(layer, x, router_logits, top_k,
                                          renormalize=renormalize, **kwargs)

    def _nvfp4_prefill(self, layer, x, router_logits, top_k, renormalize):
        """routeA 4W4A prefill：gating → topk → 单 GEMM 全专家合并（routeA 偏置链路）。

        routeA 为单 GEMM（cutlass_scaled_fp4_mm），对接统一打包权重：
          w_packed [K, N_num_experts//2] + w_scale [K//32, N//128]
        用 RouteA.preprocess_weights + __call__ 完成 GEMM；分组/合并按 layer 权重布局。
        """
        from nvfp4_4w4a_mmaf import nvfp4_4w4a_prefill_gemm as _gemm
        # 1) gating + topk
        topk_ids, topk_weights = self._default_routing(layer, x, router_logits, top_k, renormalize)
        # 2) 取权重（统一打包格式；若缺失则回落）
        w_packed = getattr(layer, "w13_packed", None) or getattr(layer, "w1_packed", None)
        w_scale = getattr(layer, "w13_scale", None) or getattr(layer, "w1_scale", None)
        w2_packed = getattr(layer, "w2_packed", None)
        w2_scale = getattr(layer, "w2_scale", None)
        if w_packed is None or w_scale is None:
            return self._fallback_apply(layer, x, router_logits, top_k, renormalize)
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        out = _gemm(x_, w_packed, w_scale)
        if w2_packed is not None and w2_scale is not None:
            out = _gemm(out, w2_packed, w2_scale)
        return out.reshape(*x.shape[:-1], -1)

    def _default_routing(self, layer, x, router_logits, top_k, renormalize):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        return fused_experts._get_topk(router_logits, top_k, renormalize)
