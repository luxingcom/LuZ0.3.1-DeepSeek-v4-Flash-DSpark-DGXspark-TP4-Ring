#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W4A4B12xExperts — B12xExperts 子类：大 M prefill 走 W4A4（flashinfer
B12xMoEWrapper, nvfp4 quant_mode），其余回落生产 W4A16 原路径（或全量 W4A4）。

权重派生链（Task #20 适配器, Task #21 checkpoint 级验证）:
  payload:  loader [w1(gate); w3(up)] → W4A4 需 [w3(up); w1(gate)]
            (hybrid: 副本 +~33GB/rank; full: 原地交换零拷贝)
  w2:       行序无关, hybrid 需副本（b12x w4a16 prepare 就地销毁原始 payload）
  scale:    E8M0 [E,N,K//32] → LUT 精确 → E4M3 [E,N,K//16] → 逐 expert
            swizzle_blockscale → convert_sf_to_mma_layout(num_groups=E)
  alpha/fc2_input_scale: 全 1.0（input_scale 模式; 与原生校准差实测 0.12/0.24%）
"""
import os

import torch

from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (
    B12xExperts,
    _b12x_activation_name,
    _normalize_b12x_moe_topk_ids,
    _normalize_b12x_moe_topk_weights,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout

from vllm.logger import init_logger

logger = init_logger(__name__)

# E8M0 -> E4M3 精确 LUT（E8M0 字节 b 表示 2^(b-127); E4M3 精确表示 2^-9..2^8,
# 即字节域 [118,135]; 全模型实测 scale 字节范围 [118,126] ⊂ 精确域, 零信息损失）
_B = torch.arange(256, dtype=torch.float32)
_E8M0_TO_E4M3 = torch.pow(2.0, _B - 127.0).to(torch.float8_e4m3fn).view(torch.uint8)


def _derive_w4a4_scale_mma(scale_u8: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """[E, N, K//32] uint8 E8M0 → E4M3 swizzled → mma 视图（返回 strided view,
    其底层 swizzled 存储由调用方持有）。行序须已为 [w3; w1]（w13）或原序（w2）。"""
    E = scale_u8.shape[0]
    lut = _E8M0_TO_E4M3.to(scale_u8.device)
    assert scale_u8.shape == (E, N, K // 32), (scale_u8.shape, (E, N, K // 32))
    swz = torch.empty((E,) + swizzle_blockscale(
        torch.empty((N, K // 16), dtype=torch.float8_e4m3fn,
                    device=scale_u8.device)).shape,
        dtype=torch.float8_e4m3fn, device=scale_u8.device)
    for e in range(E):
        e4 = lut[scale_u8[e].long()].view(torch.uint8).repeat_interleave(2, 1)
        swz[e] = swizzle_blockscale(e4.view(torch.float8_e4m3fn))
    return flashinfer_convert_sf_to_mma_layout(
        swz.reshape(E * swz.shape[1], -1), m=N, k=K, num_groups=E), swz


def _get_pooled_wrapper(**kwargs):
    """ws-dedup L3 integration: route B12xMoEWrapper creation through the
    module-level geometry-keyed pool in vllm's flashinfer_b12x_moe overlay
    (env VLLM_B12X_SHARED_WRAPPER=1 gates sharing; default off => per-layer
    wrapper constructed with exactly the original kwargs — zero behavior
    change). Sharing safety rationale identical to the overlay patch:
    run() takes weights per call; output view consumed immediately."""
    from flashinfer.fused_moe import B12xMoEWrapper
    pool_mod = None
    try:
        from vllm.model_executor.layers.fused_moe.experts import (
            flashinfer_b12x_moe as _pool_mod)
        pool_mod = _pool_mod
    except Exception:
        pool_mod = None
    enabled = (getattr(pool_mod, "_b12x_wrapper_pool_enabled", None)
               if pool_mod is not None else None)
    if pool_mod is not None and enabled is not None and enabled():
        key = (
            kwargs["num_experts"],
            kwargs["top_k"],
            kwargs["hidden_size"],
            kwargs["intermediate_size"],
            kwargs["max_num_tokens"],
            kwargs["num_local_experts"],
            kwargs["activation"],
        )
        with pool_mod._B12X_WRAPPER_POOL_LOCK:
            w = pool_mod._B12X_WRAPPER_POOL.get(key)
            if w is None:
                w = B12xMoEWrapper(**kwargs)
                pool_mod._B12X_WRAPPER_POOL[key] = w
                logger.info(
                    "[routea_plugin_a1] B12x shared wrapper pool: created "
                    "wrapper for geometry %s (pool size=%d; cross-layer "
                    "workspace dedup active)", key,
                    len(pool_mod._B12X_WRAPPER_POOL))
            else:
                logger.info(
                    "[routea_plugin_a1] B12x shared wrapper pool: reusing "
                    "wrapper for geometry %s (pool size=%d)", key,
                    len(pool_mod._B12X_WRAPPER_POOL))
            return w
    return B12xMoEWrapper(**kwargs)


class W4A4B12xExperts(B12xExperts):
    """A′ hybrid/full W4A4 experts（env 门控见包 __init__ 文档）。"""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._w4a4_mode = int(os.environ.get("VLLM_MOE_W4A4", "0"))
        self._w4a4_min_m = int(os.environ.get("VLLM_MOE_W4A4_MIN_M", "3072"))
        self._w4a4_debug = os.environ.get("VLLM_MOE_W4A4_DEBUG", "0") == "1"
        self._w4a4_ready = False
        self._wrapper = None
        self._w13 = None
        self._w2 = None
        self._w13_sf_mma = None
        self._w13_sf_store = None
        self._w2_sf_mma = None
        self._w2_sf_store = None
        self._unit = None
        self._fc2_scale = None

    # ---------------- dispatch ----------------
    def _use_w4a4(self, M: int) -> bool:
        if not self._w4a4_ready:
            return False
        if self._w4a4_mode == 2:
            return True
        if self._w4a4_mode != 1:
            return False
        return M >= self._w4a4_min_m

    # ---------------- weights ----------------
    def _derive_w4a4(self, layer) -> None:
        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        s13 = layer.w13_weight_scale.data
        s2 = layer.w2_weight_scale.data
        E, N13, K_half = w13.shape
        n = N13 // 2
        K1 = K_half * 2
        E2, N2, K2_half = w2.shape
        K2 = K2_half * 2
        assert s13.shape == (E, N13, K1 // 32) and s2.shape == (E, N2, K2 // 32)

        if self._w4a4_mode == 2:
            # full: 原地行交换 [w1;w3] -> [w3;w1]（W4A4 kernel up-first 约定）,
            # 零拷贝; layer 参数本身即 W4A4 payload（不调用 super 的 w4a16 prepare）
            tmp = w13[:, :n].clone()
            w13[:, :n].copy_(w13[:, n:])
            w13[:, n:].copy_(tmp)
            tmps = s13[:, :n].clone()
            s13[:, :n].copy_(s13[:, n:])
            s13[:, n:].copy_(tmps)
            self._w13 = w13
            self._w2 = w2
            s13_u8, s2_u8 = s13, s2
        else:
            # hybrid: W4A4 自持副本（super 的 b12x prepare 将就地销毁原始 payload）
            self._w13 = torch.cat([w13[:, n:], w13[:, :n]], dim=1).contiguous()
            self._w2 = w2.clone()
            s13_u8 = torch.cat([s13[:, n:], s13[:, :n]], dim=1).contiguous()
            s2_u8 = s2.clone()

        self._w13_sf_mma, self._w13_sf_store = _derive_w4a4_scale_mma(s13_u8, N13, K1)
        self._w2_sf_mma, self._w2_sf_store = _derive_w4a4_scale_mma(s2_u8, N2, K2)

        dev = w13.device
        self._unit = torch.ones(E, dtype=torch.float32, device=dev)
        self._fc2_scale = torch.ones(E, dtype=torch.float32, device=dev)

        cg = os.environ.get("VLLM_MOE_W4A4_CG", "1") == "1"
        max_tokens = int(getattr(self.moe_config, "max_num_tokens", 0) or 4096)
        activation = getattr(layer, "activation", None)
        if activation is None:
            activation = getattr(self.moe_config, "activation", None)
        act_str = _b12x_activation_name(activation) if activation is not None else "silu"
        self._wrapper = _get_pooled_wrapper(
            num_experts=int(E),
            top_k=int(self.moe_config.experts_per_token),
            hidden_size=int(self.moe_config.hidden_dim),
            intermediate_size=int(self.moe_config.intermediate_size_per_partition),
            use_cuda_graph=cg,
            max_num_tokens=max_tokens,
            num_local_experts=int(E),
            activation=act_str,
        )
        self._w4a4_ready = True
        extra_mb = (self._w13.numel() + self._w2.numel()
                    + self._w13_sf_store.numel() + self._w2_sf_store.numel()) / 1e6
        own_mb = 0.0 if self._w4a4_mode == 2 else extra_mb
        logger.info(
            "[routea_plugin_a1] W4A4 ready: mode=%s min_m=%d layer tensors=%.0fMB "
            "(resident-extra=%.0fMB), wrapper cg=%s max_tokens=%d",
            self._w4a4_mode, self._w4a4_min_m, extra_mb, own_mb, cg, max_tokens)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._w4a4_mode not in ("1", "2") and self._w4a4_mode not in (1, 2):
            super().process_weights_after_loading(layer)
            return
        if self._w4a4_ready:
            return
        self._derive_w4a4(layer)
        if self._w4a4_mode == 2:
            # full: 不做 w4a16 prepare（省 33GB 打包副本）; layer 参数保留为 W4A4
            # payload（本类 apply 忽略传入的 w1/w2 空参数）
            return
        # hybrid: 原始 payload 交给生产 W4A16 路径（b12x 就地重打包）
        super().process_weights_after_loading(layer)

    # ---------------- forward ----------------
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        M = hidden_states.shape[0]
        use_w4a4 = (
            self._use_w4a4(M)
            and not apply_router_weight_on_input  # wrapper 无 input-weight 语义, 回落
        )
        if self._w4a4_debug:
            logger.info("[routea_plugin_a1] dispatch M=%d -> %s", M,
                        "W4A4" if use_w4a4 else "W4A16")
        if use_w4a4:
            assert self._wrapper is not None
            out = self._wrapper.run(
                x=hidden_states,
                w1_weight=self._w13,
                w1_weight_sf=self._w13_sf_mma,
                w2_weight=self._w2,
                w2_weight_sf=self._w2_sf_mma,
                token_selected_experts=_normalize_b12x_moe_topk_ids(topk_ids),
                token_final_scales=_normalize_b12x_moe_topk_weights(topk_weights),
                w1_alpha=self._unit,
                w2_alpha=self._unit,
                fc2_input_scale=self._fc2_scale,
            )
            output.copy_(out)
            return
        return super().apply(
            output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
            workspace2, expert_tokens_meta, apply_router_weight_on_input)
