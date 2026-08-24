#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""routea_plugin_a1 — A′ 插件：DSV4 MXFP4 (-0731) 生产路径上的 W4A4 prefill 分支。

基于 Task #20/#21 实测结论（node01 生产镜像容器）：
  - 生产 MoE 路径: Mxfp4MoEMethod → B12xExperts (b12x w4a16, layout "w31" =
    kernel-native gate_up, loader [w1(gate); w3(up)] 直配)
  - W4A4 路径: flashinfer B12xMoEWrapper (quant_mode 默认 nvfp4), kernel 约定
    [w3(up); w1(gate)] 行序（fork prepare_nvfp4_moe_layer_for_fi_or_cutlass 的
    reorder_w1w3_to_w3w1 等价），scale 链 = E8M0[K//32] --LUT(精确)--> E4M3[K//16]
    --swizzle_blockscale--> swizzled-2D --convert_sf_to_mma_layout--> mma 视图
  - W4A4 语义质量: 总 logprob +0.41% vs W4A16 基线（Task #21 mini 实测）
  - swiglu_limit=10.0: W4A4 路径不施加 clamp —— fork oracle 对 NVFP4 路径有硬闸门,
    本插件在 MXFP4 oracle 内自建 W4A4 分支故不触发该闸门; clamp 效应实测 0.0000
    （W4A16 有/无 clamp 输出逐位一致, Task #21）——由 VLLM_MOE_W4A4=1/2 显式声明
    "接受无 clamp"。

env 开关:
  VLLM_MOE_W4A4        0=off(默认, 生产原样) 1=hybrid(M≥MIN_M 走 W4A4, 其余 W4A16)
                       2=full(全量 W4A4, 不保留 W4A16)
  VLLM_MOE_W4A4_MIN_M  hybrid 模式 W4A4 分派阈值（默认 3072; 生产 prefill chunk
                       上限 4096, W4A4 在 M≥4096 实测 1.32×, 中段 64-2048 0.79-0.95×）
  VLLM_MOE_W4A4_CG     wrapper use_cuda_graph（默认 1; TP4 生产 decode 捕获需要）
  VLLM_MOE_W4A4_DEBUG  1=打印每次分派决策

内存（TP4 per-rank, 43 层, 生产形状）:
  hybrid: +~41GB（W4A4 payload 副本 33GB + E4M3 scale ~8.7GB）——b12x w4a16 prepare
          会就地重打包并销毁原始 payload, W4A4 必须自持副本
  full:   +~8.7GB（payload 原地行交换零拷贝, 仅新增 scale）——无 W4A16 回退

安装: PYTHONPATH 指向本包父目录后 `import routea_plugin_a1`（或 pip install -e 后
      经 vllm.general_plugins entry point 自动加载）。monkey-patch 两处:
      oracle.mxfp4.backend_to_kernel_cls 与 quantization.mxfp4.
      select_deepseek_v4_mxfp4_moe_backend（后者是 Mxfp4MoEMethod 实际调用点）。
"""
import os

_MODE = os.environ.get("VLLM_MOE_W4A4", "0")

_installed = False


def install() -> bool:
    """把 W4A4B12xExperts 注入 flashinfer_b12x 的 experts 类解析。幂等。"""
    global _installed
    if _installed:
        return True
    if _MODE not in ("1", "2"):
        return False
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle_mxfp4
    from vllm.model_executor.layers.quantization import mxfp4 as quant_mxfp4
    from routea_plugin_a1.w4a4_experts import W4A4B12xExperts

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
    init_logger(__name__).info(
        "[routea_plugin_a1] installed: mode=%s min_m=%s (W4A4 via B12xMoEWrapper; "
        "accepts no swiglu_limit clamp — clamp effect measured 0.0000, Task#21)",
        _MODE, os.environ.get("VLLM_MOE_W4A4_MIN_M", "3072"))
    return True


# import 即安装（env 门控）; entry point 加载路径与显式 import 路径共用此行为
install()
