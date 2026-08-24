# vLLM 插件 Entrypoint 注册方式（已核实官方机制）

> 日期：2026-08-20 | 依据：vLLM 官方 docs/features/quantization（Out-of-Tree 量化插件）+ vllm.plugins 插件系统
> 适用：vLLM 0.26（MiaAI fork 兼容——生产报告确认容器为 vLLM 0.26.1）

---

## 一、注册方式（三要素）

### 1. 量化方法注册（`@register_quantization_config`）

```python
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("nvfp4_4w4a_sm121")   # ← 注册名（--quantization 参数）
class Nvfp4W4A4Config(QuantizationConfig):
    def get_name(self) -> str:
        return "nvfp4_4w4a_sm121"
    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]
    @classmethod
    def get_min_capability(cls) -> int:
        return 120                       # SM12x（GB10）；-1 不限
    @staticmethod
    def get_config_filenames() -> list[str]:
        return []                        # 不从模型目录读 config
    @classmethod
    def from_config(cls, config: dict) -> "Nvfp4W4A4Config":
        return cls()
    def get_quant_method(self, layer, prefix):
        # 按层类型分派：仅 MoE 层接管，其余返回 None（默认方法）
        if isinstance(layer, FusedMoE):
            from .moe_method import Nvfp4W4A4MoEMethod
            return Nvfp4W4A4MoEMethod()
        return None
```

### 2. MoE 方法（`FusedMoEMethodBase`，vLLM 0.26 新签名）

```python
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class Nvfp4W4A4MoEMethod(FusedMoEMethodBase):
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        """① 为层创建/登记权重（NVFP4 打包参数：weight_block_size、pack 布局）
           ② 若权重为 MXFP4 原版（非 NVFP4），此处触发高精度转换器（见 convert_high_precision_nvfp4.py）"""
        ...

    def apply(self, layer, router, x, router_logits):
        """M 阈值分派：prefill（M≥256）→ v15 4W4A；decode → 原 B12X/Marlin"""
        if x.shape[0] >= 256 and os.environ.get("VLLM_NVFP4_K1", "0") == "1":
            return self._nvfp4_prefill(layer, router, x)
        return self._fallback(layer, router, x)      # 原方法

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        """返回 MoE 量化配置（None = 不启用本量化路径）"""
        return None  # 或 FusedMoEQuantConfig(...)
```

⚠️ **签名注意（0.26 与旧版差异）**：`apply(layer, router, x, router_logits)`——router 作为参数传入（路由在 layer 外完成），`router_logits` 由上层算好。生产 MiaAI fork 若为旧签名（`apply(layer, x, router_logits, top_k, renormalize)`），以实际 fork 源码为准（插件内做双签名兼容适配）。

### 3. 插件加载（两种方式，二选一）

**A. `vllm.general_plugins` entry point（官方插件系统）**
```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
nvfp4 = "nvfp4_vllm_plugin"
```
```bash
pip install -e nvfp4_vllm_plugin   # 安装后 vLLM 启动自动加载（import nvfp4_vllm_plugin）
```

**B. 启动脚本显式 import（最可靠，生产推荐）**
```bash
# start_tp4_head.sh ENV_ARGS 追加：
# PYTHONPATH 已含 <INSTALL_DIR>/nvfp4 两目录；再加一行：
python -c "import nvfp4_vllm_plugin; print('nvfp4 plugin registered')"
# 或 vLLM 启动前：vllm serve ... --quantization nvfp4_4w4a_sm121
```
> 生产容器已挂载 nvfp4 + PYTHONPATH——只需把 `nvfp4_vllm_plugin` 目录放入挂载路径即可（零编译）。

---

## 二、启用方式（与生产兼容）

| 项 | 值 |
|---|---|
| 量化 | `--quantization nvfp4_4w4a_sm121` |
| MoE 后端 | 保持 `--moe-runner-backend` 原样（B12X）；分派在方法内 |
| KV | `--kv-cache-dtype nvfp4_ds_mla`（NVFP4 KV 时启用 v17；fp8 保持 fused_compress） |
| 开关 | `VLLM_NVFP4_K1=1`（v15 prefill）/ `VLLM_NVFP4_K2=1`（v17 写回） |
| 回滚 | 关闭 env / 去掉 `--quantization` 参数 |

## 三、风险与验证

- **注册名冲突**：`nvfp4_4w4a_sm121` 与 vLLM 内建 `nvfp4` 不同名——无冲突；若 fork 已注册同名，改注册名
- **双签名兼容**：`apply` 用 `*args/**kwargs` 适配新旧签名
- **验证**：`vllm serve --quantization nvfp4_4w4a_sm121` 启动日志出现 `Quantization: nvfp4_4w4a_sm121`；`llm.get_model()` 检查 MoE 层 method 类型
