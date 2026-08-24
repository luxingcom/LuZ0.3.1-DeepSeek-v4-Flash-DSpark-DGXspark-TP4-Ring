# RouteA 接入生产 fork（MiaAI vLLM 0.26.1.dev0）modular MoE 架构 —— 接入判定与方案设计

- **文档编号**：nvfp4-kernels-final-2026-08-20 / routeA-integration-review-2026-08-20
- **日期**：2026-08-20
- **作者**：code-reviewer-a（代码审查 · A 路线）
- **范围**：只读研究 + 方案设计；**不修改生产脚本**（改脚本由 sre-oom 执行）
- **目标**：判定 routeA 在 modular MoE 架构下的最优接入点，产出可落地接线链、权重链、forward 分派与回退方案

> **结论速览**：routeA 应作为**「在 ModelOpt NVFP4 量化链路上注册一个自定义 MoE experts 类 + 在其 `process_weights_after_loading`/`apply` 内做 per-shape 分派」**接入，而不是新增一个独立的 `QuantizationConfig` 插件。理由：routeA 是一个**单 GEMM 算子**（A[K]→[N]），不是完整 MoE kernel；它必须复用 fork 内建的路由/sort/finalize 框架。最贴近的模板是 `TritonOrCutlassExperts(FallbackExperts)` 与 `FlashInferB12xExperts`。当前旧插件骨架（`moe_method.py`）用的过时 `apply(layer,x,router_logits,top_k,renormalize)` 签名与 `FusedMoE`（旧式）完全对不上 modular 架构，**必须重写**。

---

## 0. 源码事实核实（非臆断，均来自 fork 源码 /home/<USER>/patch-v026 与 <INSTALL_DIR>/backup/vllm027-src/vllm-0.27.1）

### 0.1 目标生产环境（已从启动脚本核实）
- `start_tp4_worker_b12x.sh`：`--moe-backend flashinfer_b12x --kv-cache-dtype nvfp4_ds_mla --linear-backend deep_gemm`
- 模型：`deepseek-v4-flash-0731-nvfp4`，config.json 关键字段：
  - `model_type: deepseek_v4`, `hidden_size: 4096`, `moe_intermediate_size: 2048`
  - `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 6`
  - `num_hash_layers: 3`, `topk_method: noaux_tc`, `swiglu_limit: 10.0`, `routed_scaling_factor: 1.5`
- 该模型为 **ModelOpt NVFP4** checkpoint：
  - `hf_quant_config.json`: `producer.name=modelopt`, `quant_algo=MIXED_PRECISION`, `moe_quant_algo=NVFP4`, `group_size=16`
  - `config.json` `quantization_config`: `quant_method=fp8`, `moe_quant_algo=NVFP4`, `group_size=16`, `fmt=e4m3`

**关键约束**：`swiglu_limit=10.0` 已被 oracle 强制纳窄可选的 NVFP4 MoE backend。见 0.3。

### 0.2 modular MoE 接口（确认任务描述中的签名）
- `fused_moe_method_base.py` `class FusedMoEMethodBase(QuantizeMethodBase)`：
  - 必实现：`create_weights(...)`、`get_fused_moe_quant_config(layer)`、`apply(...)`、`apply_monolithic(...)`
  - **生产实际 `apply` 签名**（确认）：
    ```python
    def apply(self, layer: "RoutedExperts", x, topk_weights, topk_ids,
              shared_experts, shared_experts_input) -> torch.Tensor
    ```
  - 关键 hook：`get_quant_method(layer, prefix)`（由量化 config 调用来为层分发 method）
  - `is_monolithic`、`method_name`、`topk_indices_dtype`、`mk_can_overlap_shared_experts`
- `RoutedExperts`（fused_moe 层容器）持有 `moe_config: FusedMoEConfig`，权重经 `create_weights` 挂到 layer（`w13_weight/w2_weight/w13_weight_scale/...`）

### 0.3 NVFP4 oracle 与 backend 选择
- `fused_moe/oracle/nvfp4.py`：
  - `NvFp4MoeBackend` 枚举含 `FLASHINFER_B12X / VLLM_CUTLASS / MARLIN / EMULATION / ...`
  - `select_nvfp4_moe_backend(config, weight_key, activation_key) -> (backend, experts_cls)`
  - **`FLASHINFER_B12X` 被刻意排除在 auto 之外**（上游 cutlass SM121 MMA op guard 未解决），需 `--moe-backend flashinfer_b12x` 显式 opt-in —— 生产正是这么做的
  - `backend_to_kernel_cls` 映射：`VLLM_CUTLASS -> CutlassExpertsFp4`、`FLASHINFER_B12X -> FlashInferB12xExperts`、`EMULATION -> Nvfp4QuantizationEmulationTritonExperts`、`MARLIN -> MarlinExperts`
  - `map_nvfp4_backend`: `"cutlass"→VLLM_CUTLASS`, `"flashinfer_b12x"→FLASHINFER_B12X`, `"marlin"→MARLIN`, `"emulation"→EMULATION`
  - **`swiglu_limit` 会过滤**：`NVFP4_BACKENDS_WITH_CLAMP = {FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, MARLIN}`，若 `swiglu_limit` 不为 None，AVAILABLE_BACKENDS 只剩这 3 个！**当前生产 b12x 是显式 opt-in，绕过了 auto 过滤**。这直接决定 routeA 不能走 auto，必须显式 `--moe-backend`。

### 0.4 生产离线 NVFP4 路径（routeA 最相关）
- 模型经 ModelOpt 产生，加载走 `modelopt.py`：
  - `ModelOptNvFp4Config.get_name() -> "modelopt_fp4"`；`override_quantization_method` 把 "NVFP4/FP4" checkpoint 映射到 "modelopt_fp4"
  - `get_quant_method(layer, prefix)`：若是 `RoutedExperts` → `self.FusedMoEMethodCls(quant_config=self, moe_config=layer.moe_config)`
  - `ModelOptNvFp4FusedMoE(FusedMoEMethodBase)` 在 `__init__` 调用 `select_nvfp4_moe_backend(self.moe, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic或None)` 得到 `self.nvfp4_backend, self.experts_cls`
  - `create_weights`: 注册 `w13_weight [E, 2*inter, H//2] uint8`、`w2_weight [E, H, inter//2] uint8`、`w13_weight_scale [E, 2*inter, H//group_size]`、`w2_weight_scale`、`w13_weight_scale_2 / w2_weight_scale_2` (per-tensor fp32)、`w13_input_scale / w2_input_scale`
  - `process_weights_after_loading`: `convert_to_nvfp4_moe_kernel_format(...)`（按 backend 转 kernel 格式）→ `make_nvfp4_moe_kernel(...)` → `self.moe_kernel.fused_experts.process_weights_after_loading(layer)`
  - `apply`: `return self.moe_kernel.apply(x, w13_weight, w2_weight, topk_weights, topk_ids, ...)`（routing 已由 prepare/finalize 处理）

### 0.5 modular kernel 执行链（forward 分派关键）
- `modular_kernel.py`：
  - `FusedMoEKernel.apply` → `prepare_finalize.prepare`（routing + input quant）→ `fused_experts.apply` → `finalize`（topk weight reduction）
  - `FusedMoEExpertsModular`（experts 类基类）必实现：
    - 实例：`apply(...)`、`process_weights_after_loading(layer)`、`workspace_shapes(...)`
    - classmethod 契约（oracle `is_supported_config` 逐个调用）：`activation_format`、`is_monolithic`、`_supports_current_device`、`_supports_no_act_and_mul`、`_supports_quant_scheme(weight_key, activation_key)`、`_supports_activation`、`_supports_parallel_config`、`_supports_routing_method`、`_supports_router_logits_dtype`、`_supports_shape`、`_supports_batch_invariance`、`quant_dtype` 等属性
  - `is_supported_config(cls, moe_config, weight_key, activation_key, activation_format) -> (bool, reason)` 是 oracle 自动挑选的入口
- **per-shape 分派模板**：`experts/fallback.py` `class FallbackExperts(FusedMoEExpertsModular, ABC)`：
  - `__init__(experts, fallback_experts)` 持有两个实现
  - `apply(...)` 调用 `self._select_experts_impl(hidden_states, w1, w2)` 返回选中的实现再 apply
  - 子类 `triton_cutlass_moe.py::TritonOrCutlassExperts` 用 `M <= 8` 判据在 Cutlass/Triton 间切换 —— **这正是 routeA 想要的「prefill M≥threshold 走 A，否则走原路径」的现成模式**
- `FlashInferB12xExperts(mk.FusedMoEExpertsModular)`：`apply` 内走 `B12xMoEWrapper`，管理内部 workspace，**进 kernel 前 BF16 输入 → kernel 内量化**（W4A16 语义）

---

## 1. 接入点判定：routeA 到底接在哪一层？

### 1.1 routeA 的本质
`nvfp4_4w4a_mmaf.py::RouteA` 是**单 expert GEMM 便利算子**：
- `preprocess_weights(W_packed[K,N//2], W_scale[K//32,N//128])`：反量化既有 NVFP4 权重 → fp32[N,K] → `scaled_fp4_quant` 重新量化为官方 NVFP4 格式并缓存
- `__call__(A[M,K], use_cached_w)`：`A` 经 `scaled_fp4_quant` → `cutlass_scaled_fp4_mm(a_q, w_q, a_sf, w_sf, alpha, bf16)` → 可选 bias

**它不含**：routing（hash/noaux_tc）、topk 权重、专家分组/sort、activate-and-mul(swiglu)、shared experts、reduce。这些必须完全复用 fork 的 modular 框架。

### 1.2 两种候选接入方式 trade-off

| 维度 | A. 自定义 QuantizationConfig 插件（旧骨架思路） | B. **在 ModelOpt NVFP4 链路上注册自定义 experts 类** ✓ |
|---|---|---|
| 权重加载 | 需独立处理 checkpoint 读取，与 ModelOpt `hf_quant_config` / `checkpoint reader` 脱节；生产权重就是 ModelOpt NVFP4，重复解析 | 直接复用 `ModelOptNvFp4Config` 的 `create_weights` + `weight_loader`，权重天然是 `w13/w2 uint8` 布局 |
| routing/sort/reduce | 需自己重建（难度高、易错） | 复用 `make_nvfp4_moe_kernel` / `prepare_finalize` 全链路 |
| 回退兼容 | 独立耦合，回退需单独开关 | 天然是 backend 选择（`--moe-backend`）的一部分，回退=换 backend | 
| oracle 集成 | 需要额外注册到 `_QUANTIZATION_CONFIG_REGISTRY` 或 `register_quantization_config`，且与 swiglu_limit 过滤冲突 | 直接插入 `backend_to_kernel_cls` / `select_nvfp4_moe_backend` 的候选表，符合 oracle 选型语义 |
| CUDA Graph | apply 内分派天然按 phase 捕获 | 同上 |
| 改动面 | 新增一个量化方法，侵入启动链路 | 改动集中：1 个 experts 类 + oracle 候选表 + 启动 flag |

**判定：选 B。** routeA 不是一种「量化格式」，它与同一份 ModelOpt NVFP4 checkpoint 兼容，是**在该 checkpoint 上的一个替代 GEMM 内核**。因此最干净的表达是：新增一个 `FusedMoEExpertsModular` 子类（如 `RouteAPrefillExperts`），并让 oracle 在显式 `--moe-backend routea`（或复用 `flashinfer_b12x` 之上做 fallback 包装）时选中它。旧插件骨架 `moe_method.py` 的「新 QuantizationConfig + 重写 apply(layer,x,router_logits,top_k,renormalize)」思路**不适用于 modular 架构，应整体废弃重写**。

### 1.3 推荐的最落地方案（结合 0.3 swiglu_limit 约束）
由于 `swiglu_limit=10.0` 使 auto-selection 只剩 TRTLLM/CUTLASS/MARLIN 三类（b12x 靠显式 opt-in），routeA 必须**显式 backend**。推荐两条落地路径：

- **路径 B1（低侵入，强烈推荐先做）**：自定义 experts 类作为 **`FlashInferB12xExperts` 的 fallback 包装**（`FallbackExperts` 模式），仍以 `--moe-backend flashinfer_b12x` 启动，在 `_select_experts_impl` 里 `M ≥ threshold && hardware_guard` 时选 routeA，否则选 B12x。对生产启动脚本**零 flag 改动**，仅替换 experts 类注入点。
- **路径 B2（干净但需动 oracle）**：在 `NvFp4MoeBackend` 增加 `ROUTEA`，在 `map_nvfp4_backend` 加 `"routea"→ROUTEA`，在 `backend_to_kernel_cls` 加 `ROUTEA -> [RouteAPrefillExperts]`，启动 `--moe-backend routea`。更符合 oracle 语义，但需明确 `swiglu_limit` clamp 兼容判断（routeA 用 `cutlass_scaled_fp4_mm` + 自己处理 act_and_mul，需确认 clamp 行为，见 §6 验证）。

> 建议：**先 B1（wrap flashinfer_b12x）进灰度**，链条短、回退最干净；B2 作为正式命名 backend 的阶段二。

---

## 2. RoutedExperts 装配链：自定义 experts 类需实现哪些方法

以下签名/骨架基于 fork 源码核实（`FusedMoEExpertsModular`、`FallbackExperts`、`FlashInferB12xExperts`、`TrtLlmNvFp4ExpertsBase`）。

```python
# file: vllm/model_executor/layers/fused_moe/experts/routea_moe.py (新增)
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import FlashInferB12xExperts
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

PREFILL_M_THRESHOLD = int(__import__("os").environ.get("NVFP4_PREFILL_M", "256"))

class RouteAPrefillGeMM:
    """薄封装 routeA（nvfp4_4w4a_mmaf.RouteA），按 layer 缓存预处理权重。
    只做单 GEMM；routing/topk/act_and_mul 由外层框架承担。"""

    # preprocess 一次，缓存官方格式 w_q/w_sf
    ...

class RouteAPrefillExperts(FallbackExperts):
    """prefill(M≥threshold) → routeA；decode/小 M → B12x。"""

    def __init__(self, moe_config, quant_config):
        # routeA 单 GEMM 需要权重重排；其“experts 实现”其实就是逐 expert 调 GEMM。
        # 但为复用 prepare/finalize 的 sort/topk，我们让它作为 b12x 的“同继承人”，
        # 在 _select_experts_impl 分派。真正的 routeA GEMM 在 apply 内按 expert 循环调用。
        super().__init__(
            experts=FlashInferB12xExperts(moe_config, quant_config),  # fallback 用
            fallback_experts=None,  # 见下方说明
        )
        self.moe_config = moe_config
        self.quant_config = quant_config

    # ---- oracle 需要的 classmethod 契约（is_supported_config 逐个调用）----
    @classmethod
    def activation_format(cls) -> mk.FusedMoEActivationFormat:
        return FlashInferB12xExperts.activation_format()  # 保持与 fallback 一致

    @staticmethod
    def is_monolithic() -> bool:
        return False  # routeA 是 modular（非 monolithic）

    @classmethod
    def _supports_current_device(cls) -> bool:
        # 仅 GB10 sm_121a 且 routeA/cutlass FP4 op 可用时
        from vllm.platforms import current_platform
        return current_platform.has_device_capability(121)

    @classmethod
    def _supports_no_act_and_mul(cls) -> bool:
        return FlashInferB12xExperts._supports_no_act_and_mul()

    @classmethod
    def _supports_quant_scheme(cls, weight_key, activation_key) -> bool:
        # routeA 接受 NVFP4 static 权重 + 动态/静态激活（W4A4）
        return (weight_key is not None and weight_key.is_nvfp4)  # 按实际 QuantKey 语义
        # 注：需核对 QuantKey（kNvfp4Static/kNvfp4Dynamic）定义，见 §6

    @classmethod
    def _supports_activation(cls, activation: MoEActivation) -> bool:
        return True  # 支持 swiglu（routeA 外层 act_and_mul 由框架/numba 处理）

    @classmethod
    def _supports_parallel_config(cls, moe_parallel_config) -> bool:
        return FlashInferB12xExperts._supports_parallel_config(moe_parallel_config)

    @classmethod
    def _supports_routing_method(cls, routing_method, weight_key, activation_key) -> bool:
        return True  # hash/noaux_tc 由 prepare/finalize 完成

    @classmethod
    def _supports_router_logits_dtype(cls, dtype, routing_method) -> bool:
        return True

    @classmethod
    def _supports_shape(cls, hidden_dim: int) -> bool:
        return hidden_dim == 4096  # routeA 已验证维度

    # ---- workspace 分派（prepare/finalize 用）----
    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        if self._use_routea(M):
            # routeA 逐 expert GEMM：workspace 极小（无需 sort? 需确认 prepare 是否仍需）
            return self.experts.workspace_shapes(M, N, K, topk, global_num_experts,
                                                 local_num_experts, expert_tokens_meta, activation)
        return super().workspace_shapes(...)

    def _use_routea(self, M: int) -> bool:
        import os
        return (M >= PREFILL_M_THRESHOLD
                and os.environ.get("VLLM_NVFP4_ROUTEA", "0") == "1"
                and self._routea_ready)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 先让 fallback(B12x) 完成自身 kernel 格式转换（保回退可用）
        self.experts.process_weights_after_loading(layer)
        # 再为 routeA 准备权重：从 layer 取 w13_weight/w2_weight（已是 [E,2*inter,H//2] uint8 布局）
        self._routea_gemm.provision(layer)   # 内部 RouteA.preprocess_weights 缓存
        self._routea_ready = True

    def _select_experts_impl(self, hidden_states, w1, w2):
        # 实际是轻量分派标记；真正分派在 apply 内（因 routeA 要逐 expert 调 GEMM）
        return self.experts if not self._use_routea(hidden_states.shape[0]) else self

    # ---- 核心 forward ----
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map,
              a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        if self._routea_ready and self._use_routea(hidden_states.shape[0]):
            # routeA：逐 expert 调 cutlass FP4 GEMM，act_and_mul 由现有框架逻辑（b12x 的）
            # 复用 expert permute/sort 结果 —— 需要借助 FlashInferB12xExperts 的 prepare 产物
            routea_dispatch(...)   # 见 §4
            return
        # 回退到 B12x
        self.experts.apply(output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                           global_num_experts, expert_map, a1q_scale, a2_scale,
                           workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input)
```

> **关键提示**：routeA 逐 expert 调 GEMM 需要 `topk_ids` + expert 分组后每个 expert 的 token 子集。这个「permute→GEMM→unpermute」流程正是 `FlashInferB12xExperts`（或 `TritonExperts`）的 `apply` 内部逻辑——**最稳妥的做法是复用 B12x 内部的 B12xMoEWrapper 流程，只把其中「BF16→FP4 GEMM」替换成 routeA 的逐 expert cutlass FP4 GEMM**。因此强烈建议 `RouteAPrefillExperts` 直接**子类化 `FlashInferB12xExperts` 并覆写一个内部 GEMM 调用点**，而不是从一个通用 `FallbackExperts` 白手起家。（见 §4。）

---

## 3. 权重加载链核对

### 3.1 各层权重格式对照

| 概念 | 转换器产出（routeA 期望） | ModelOpt checkpoint（生产磁盘） | fork `RoutedExperts` 挂载后（process 前） |
|---|---|---|---|
| GEMM1 w13 | `W_packed[K=H, N//2=2*inter//2]` uint8 | `w13_weight [E, 2*inter, H//2]` uint8（input 维度打包） | `layer.w13_weight [E, 2*inter, H//2]` |
| GEMM2 w2 | `W_packed[K=2*inter, N//2=H//2]` uint8 | `w2_weight [E, H, inter//2]` uint8 | `layer.w2_weight [E, H, inter//2]` |
| 块 scale | `W_scale[K//32, N//128]` E8M0 | `w13_weight_scale [E, 2*inter, H//group_size]` E8M0-ish | `layer.w13_weight_scale` |
| anti-scale | 每 token quant | `w13_weight_scale_2 / w2_weight_scale_2` per-tensor fp32 | `layer.w13_weight_scale_2` |

**关键发现 ①（组大小不匹配）**：
- routeA 的 `W_scale[K//32, ...]` 按 **group_size=32**（128 列 × 32 行 block，与 MXFP4 一致的 32×128 block）。
- 生产 ModelOpt NVFP4 checkpoint 是 **group_size=16**（`w13_weight_scale` 的 H 维除以 16，即 16 行 block）。
- 因此**不能直接用磁盘上的 `w13_weight_scale` 喂给 routeA 的 `W_scale`**。routeA `preprocess_weights` 先反量化再重量化到 group-16 官方格式（`scaled_fp4_quant`），所以块大小在对齐后由 `scaled_fp4_quant` 重新决定 —— 但 routeA 源码里 `_dequant_w_our` 用 `repeat_interleave(32,0).repeat_interleave(128,1)`，即它**假定输入尺度是 32×128 block**，与 checkpoint group-16 **冲突**。

> **必须解决**：要么 routeA 的 `_dequant_w_our` 改为按 group_size=16（`repeat_interleave(16,0)`）反量化，要么转换器在产出 routeA 权重时重采样为 32×128。当前 `ab_routeA_vs_b12x.py::make_weights` 用的是 `K//32 × N//128`（32×128），且 `w_scale_f` 未加 E8M0 bias（routeA 用原始 2^exp），与生产 ModelOpt（E8M0 带 bias，group-16）**口径不一致**。这是 routeA 真正投入前必须校准的**数学一致性**问题（§6 验证 #V1）。

**关键发现 ②（pack 方向一致）**：
- routeA `W_packed[K, N//2]`：低半字节=偶 N 列，K 维未打包 —— 与 ModelOpt `w13_weight [E, 2*inter, H//2]`（把 2 个 fp4 装入 H 维，即 input 维度打包，`comment: 2 fp4 items are packed in the input dimension`）**pack 方向相同**（沿 input K 打包）。所以层布局方向可对上，只是需要把 routeA 的 `[K,N//2]` 与 layer 的 `[E, 2*inter, H//2]` 做 **K/N 转置+维度重排** 后按 expert 切分。

### 3.2 权重挂载与衔接建议
- 层级：保持 `ModelOptNvFp4FusedMoE.create_weights` 原样（`w13_weight/w2_weight/...` 照常注册），**不改 checkpoint reader / weight_loader**。生产磁盘权重 → `w13_weight` 的通道已有 ModelOpt weight_loader 处理，routeA 无需关心原始 safetensors。
- `process_weights_after_loading`：routeA 从 `layer.w13_weight` / `layer.w2_weight` 切出每 expert 的 `[K, N//2]` + scale，喂给 `RouteA.preprocess_weights`（含修正后的 group_size 反量化），缓存官方 NVFP4 w_q/w_sf。GEMM1/GEMM2 各缓存一份。
- 这样权重链是：`磁盘 NVFP4(ModelOpt group-16) → layer.w13_weight → (group_size 修正+pack 重排) → RouteA.preprocess → 官方 NVFP4 w_q/w_sf → cutlass_scaled_fp4_mm`。**对现有 ModelOpt 加载路径零改动**。

---

## 4. forward 分派（prefill vs decode）

- **推荐：在 experts 类内做 per-shape 分派（`apply` 内），复用 modular kernel 的 M 驱动**。`FusedMoEKernel.apply` 每次调用带着 `hidden_states[M,K]`，`self._use_routea(M)` 直接在 apply 内判 `M >= threshold`。这与 `TritonOrCutlassExperts` 的 `M <= 8` 分派完全同构，是最自然、对 CUDA Graph 友好（按 phase 捕获）的方式。
- **不要在 `FusedMoEMethodBase.apply` 层硬编码** maturity-kill 逻辑；分派判据只应存在于 experts 类。
- 具体 routeA 路径（复用 B12x 的 permute/sort/topk）：
  1. `prepare_finalize.prepare` 已产出 `expert_tokens_meta`, `a1`, `a1_scale`, `topk_ids/topk_weights`（这步两个分支共用）。
  2. routeA 分支按 `topk_ids` 做 expert-grouped 分桶，得到每 expert 的 token 子集索引。
  3. 每 expert：取 `w13[expert]`，经 `RouteA`（含 act_and_mul 的 act 融合 / 或两段 GEMM）产 `hidden'_expert`；再 `w2[expert]` 产该 expert 输出。
  4. `finalize`（TopKWeightAndReduce）用 `topk_weights` 归约多 expert 结果——**框架已支持**，routeA 只需产出未归约的 `[M, N, topk]` 视图。
- **重要实现风险**：`cutlass_scaled_fp4_mm` 输出 BF16 `[M,N]`，需凑成 `[M, N, topk]` 布局给 finalize；且 `swiglu_limit`（W4A16 checkpoint 的 act=gated 上限 clamp）需在 routeA 路径复现，否则数值漂移（见 §6 #V3）。**建议 routeA 首版复用 B12x 的 act_and_mul + clamp 逻辑**（B12x 已正确实现，见 `FlashInferB12xExperts.apply` 与 `b12x_fused_moe` 内部）。

---

## 5. 回退兼容（未达 A/B ≥1.5× 或不满足 hardware guard）

设计要点：**整个 routeA 是「可选、默认关」**，与旧插件 `__init__.py` 的 `VLLM_NVFP4_K1/K2` env 开关思路一致（但替换为正确的 modular 接入方式）。

多级回退：
1. **编译期 hardware guard**：`_supports_current_device` 只认 sm_121a 且有 `cutlass_scaled_fp4_mm`。若 guard 不过，oracle 直接不选 routeA，回落到 b12x（B1 包装的 fallback 自然生效）。
2. **运行期 env 开关**：`VLLM_NVFP4_ROUTEA=1` 才启；默认 0 → apply 永远走 B12x。灰度/回滚只需不设 env。
3. **A/B 未达标自动回退**：入口处做一次性基准（§6 #V2），若 `routeA / b12x < 1.5` 或数值超差，`self._routea_ready=False`，日志告警，此后恒走 B12x。可通过阈值 env 调。
4. **生产回滚（sre-oom 执行）**：
   - B1：仅替换/移除 experts 注入（或关 env），重启 worker 即回 B12x 原样——**对启动脚本零改动**（若用 B2 则回 `--moe-backend flashinfer_b12x`）。
   - 备份：源码改动前由 sre 备份 `fused_moe/` 目标文件（如 `<INSTALL_DIR>/backup/...`）。
   - 与 KV K2 正交：routeA 只动 MoE GEMM，不动 KV 写回；两路可独立灰度。

---

## 6. 验证清单（A/B 与单元验证）

> 运行环境：DGX Spark GB10 干净 GPU（非共享），用 `<INSTALL_DIR>/nvfp4/_nvfp4_verify.py` 与 `ab_routeA_vs_b12x.py` 类脚本。只读研究阶段不实际执行生产改动，以下为 sre-oom 落地后验收项。

### V1 数学一致性（P0，先于一切性能）
- [ ] **group_size 对齐修正**：routeA `_dequant_w_our` 用 32×128 block 反量化，而生产 checkpoint 是 group-16。修正为 `repeat_interleave(group_size,0)`（group_size 从 layer/w13 scale 推断），并以生产 `w13_weight/w13_weight_scale/w13_weight_scale_2` 为输入，验证 `routeA 反量化 → rescaled_fp4_quant` 结果与 B12x 语义一致（逐元素 rel 差）。
- [ ] 与 `scaled_fp4_quant` dequant 数学对齐（kernel1 头注释已声称 rel=0.00141，需在真实 checkpoint 上复核）。

### V2 性能 A/B（P0）
- [ ] 用真实 NVFP4-HP 权重 `[K=4096, N=4096]`，覆盖 prefill 形状 `M∈{256,512,1024,2048,4096}`，`routeA_cutlass vs b12x`，**要求 prefill TFLOPS 提升 ≥1.5×**（routeA 目标 80-180 TFLOPS）。
- [ ] decode 形状 `M∈{1,2,4,8,16,32}` 确认 routeA 不启（走 B12x），verify decode 无回退惩罚。
- [ ] 整层 MoE（含 routing/sort/finalize 开销）的端到端 A/B，非单 GEMM 口径——确认非理想形状/小 M 下 routeA 整体不劣化。

### V3 数值 / 正确性（P0 单元）
- [ ] 选专家一致性：routeA 路径与 B12x 产生**相同 topk_ids/topk_weights**（hash/noaux_tc 两边必须一致）。
- [ ] **swiglu_limit=10.0 复现**：routeA 输出经 act_and_mul clamp 后与 B12x 对齐（否则长序列累积漂移）。
- [ ] `routed_scaling_factor=1.5`、shared_experts(1 个)、`e_score_correction_bias`、`norm_topk_prob` 在 routeA 分支得到保留（复用框架逻辑）。
- [ ] 端到端 logits 一致性：固定 seed 同输入，routeA vs B12x 单层输出 `max_abs_err` 与 `rel_err` 阈值（参考 kernel1 rel=0.00141），及 32 层累计漂移。

### V4 modular 契约 / 装配（P0）
- [ ] `RouteAPrefillExperts` 通过 oracle `is_supported_config` 全部子判定（device/act_and_mul/activation/quant_scheme/parallel/routing/logits_dtype/shape/batch_invariance）。
- [ ] `create_weights`→`process_weights_after_loading`→`apply`→`finalize` 全链 smoke（`_nvfp4_verify.py` 扩展），output shape `[M, hidden]` 正确。
- [ ] monolithic vs modular 路径判定正确（routeA 非 monolithic）。

### V5 回退 / 灰度（P1）
- [ ] env 开关关闭时行为与 B12x 逐字节一致（回退零副作用）。
- [ ] `_routea_ready=False`（guard 未过/A-B 未达）时自动回落 B12x，无内存泄漏、无 CUDA Graph 捕获失败。
- [ ] CUDA Graph：prefill phase（routeA）与 decode phase（B12x）分别捕获，切换无 graph 重捕获开销爆炸。

### V6 生产部署演练（停机窗口，sre-oom）
- [ ] 4 节点 TP4 同权重/同 flag 起 `modelopt_fp4`，routeA 灰度 worker 与 B12x worker 对比首 token 延迟 / prefilled token 数。
- [ ] 事故恢复项：参考团队其他路线（#21/#23）的 KV K2 与停机编排，确认 routeA 与 K2 可联合灰度。

---

## 7. 风险与决策记录

| # | 风险/事实 | 影响 | 决策 |
|---|---|---|---|
| R1 | 生产 checkpoint group_size=16，routeA 假定 32×128 | 数学不一致 → 结果错误 | V1 必须修正 `_dequant_w_our` / 转换器重采样，P0 |
| R2 | `swiglu_limit` 使 NVFP4 auto-selection 只剩 TRTLLM/CUTLASS/MARLIN | b12x 与 routeA 都需显式 opt-in | routeA 走 B1（wrap b12x，复用其显式 flag）或新增 ROUTEA backend |
| R3 | routeA 是单 GEMM，缺 routing/sort/finalize | 直接套用必出错 | 子类化 FlashInferB12xExperts，只替换内部 GEMM；复用其 permute/topk/act_and_mul |
| R4 | 旧插件 moe_method.py 签名过时（FusedMoE 旧式） | 不可用 | 废弃重写为 modular experts 类（§2） |
| R5 | B12x 内部 BF16→FP4 激活量化语义 vs routeA `scaled_fp4_quant` | 激活量化差异 → 精度 | 确认 routeA 激活量化与 b12x 等价（V1/V3） |

---

## 8. 结论

- **接入判定**：routeA 应作为 **ModelOpt NVFP4 链路上的自定义 MoE experts 类**接入（B 方案），并**子类化 `FlashInferB12xExperts` + 覆写内部 GEMM 调用点**以复用 routing/sort/topk/act_and_mul/finalize。灰度走 **B1（wrap flashinfer_b12x，env 开关）**，正式命名 backend（B2/`ROUTEA`）作为阶段二。
- **必须修复 P0**：group_size(32 vs 16) 数学一致性（V1）；后端 A/B 达 1.5× 前不启用。
- **改动面**：新增 1 个 experts 类（~200 行）+ 可选 oracle 候选表条目；生产 ModelOpt 加载链路 / checkpoint reader / weight_loader **零改动**；启动脚本零-feature 改动（env 开关）。
- **回退**：不满足 guard / A/B / env 关 → 自动回落 B12x 原样，可随时回滚。

**文档路径**：`deliverables/engineering-assurance/nvfp4-kernels-final-2026-08-20/routeA-integration-review-2026-08-20.md`