# DeepSeek-V4-Flash NVFP4 权重调研报告

**日期**：2026-08-13
**需求**：查找 DeepSeek-V4-Flash 最新权重，**全面采用 NVFP4 格式**，来源限定头部贡献者/知名社区
**背景**：项目当前使用 deepseek-v4-flash-0731（FP8，156GB×4）；CUDA 13.2 的 cuBLASLt NVFP4 路径有 3× 大 GEMM 加速潜力，此前因无 NVFP4 权重未验证

---

## 1. 关键概念澄清（先明确"全面 NVFP4"的准确含义）

DeepSeek-V4-Flash 的精度格式有**三种**，容易混淆：

| 格式 | 定义 | 代表 |
|---|---|---|
| **官方 FP4+FP8 混合** | MoE 专家参数 FP4（MXFP4 打包），其余 FP8 | `deepseek-ai/DeepSeek-V4-Flash-0731`（官方原版） |
| **NVFP4**（NVIDIA 风格） | MoE routed-expert 线性层（w1/w2/w3）NVFP4 量化（GS16 block scale），**attention/共享专家/路由/embedding/MTP 保留高精度** | `nvidia/...-NVFP4`、`MJPansa/...-0731-NVFP4` |
| FP8 全精度 | 全部 FP8 | 项目现用 |

> **"全面采用 NVFP4"在业界即指 NVFP4 量化版**：MoE 专家权重（占模型参数 90%+）全部 NVFP4，其余敏感层保留高精度——这是 NVIDIA Model Optimizer 的标准做法，不存在"100% 所有权重 NVFP4"的版本（attention/路由层 NVFP4 精度损失不可接受）。

---

## 2. 候选清单（Hugging Face + ModelScope 双渠道扫描）

### 🔴 首选：`MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（HF）

| 项 | 值 |
|---|---|
| 基础版本 | **DeepSeek-V4-Flash-0731**（2026-07-31 官方最新正式版，非 preview） |
| 量化范围 | **33,024 个 routed-expert 投影（w1/w2/w3）全量 NVFP4**；attention/共享专家/路由/head/embedding/MTP 保留官方原表示 |
| 转换方式 | 官方 MXFP4 打包权重 **无损转换**为 NVIDIA 风格 NVFP4（GS32→GS16 block scale，字节级校验） |
| 规格 | 304B 总参数（含 MTP），**48 shard**（与项目现有 0731 一致） |
| 运行验证 | **2× DGX Spark TP2**，vLLM `0.26.1rc1.dev191`（2026-07-31 构建）实测加载+生成 ✓ |
| 下载量 | 175,279（月）/ 110,233（总） |
| 特点 | **DSpark/MTP 配置保留**——项目现有 dspark 投机解码可继续使用 |
| 许可 | MIT |

### 🔵 备选：`nvidia/DeepSeek-V4-Flash-NVFP4`（HF，官方厂商）

| 项 | 值 |
|---|---|
| 来源 | **NVIDIA 官方**（Model Optimizer v0.44.0），HF 2026-05-28 发布 |
| 基础版本 | DeepSeek-V4-Flash（**5/28 preview 早期版，非 0731**） |
| 量化范围 | transformer block 内 MoE 线性层权重+激活 NVFP4（存储 FP8 + `moe_quant_algo: NVFP4`） |
| 规格 | 284B 总参数，46 shard |
| 运行验证 | GB300 + vLLM nightly 0.22.1rc1（TP4）、SGLang PR#25820 |
| 下载量 | 771,737（总，最高） |
| 精度评测 | NVFP4 vs baseline 近乎无损（GPQA 0.891 vs 0.894 等） |

### 🌐 国内渠道（ModelScope 魔搭）

| 模型 | 备注 |
|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 官方 FP4+FP8 混合版（**非 NVFP4**，与需求不符） |
| `FlagRelease/DeepSeek-V4-Flash-nvidia-FlagOS` | **nvidia NVFP4 版的国内镜像**（296.38B/300.84GB，FlagOpen 智源系，4/27 打包） |
| `unsloth/DeepSeek-V4-Flash` | unsloth 镜像（158.07B，6/29） |

---

## 3. 推荐结论

**首选 `MJPansa/DeepSeek-V4-Flash-0731-NVFP4`**，理由：
1. **版本最新**：0731 正式版（唯一 NVFP4 版基于 0731；nvidia 官方版停留在 5/28 preview）
2. **格式最全**：routed-expert 33,024 个投影全量 NVFP4（= "全面 NVFP4"），且保留 DSpark/MTP（项目投机链路不丢）
3. **硬件匹配**：2×DGX Spark 实测验证（vLLM 0.26.1 与项目同代），48 shard 与项目现有权重对齐
4. **无损转换**：字节级校验（conversion-receipt.json 佐证）

**备选 `nvidia/...-NVFP4`**：若优先"官方厂商背书 + 下载量"，但需接受旧版（非 0731）与 46 shard。

---

## 4. 下载命令（国内网络）

```bash
# 方式一：HF 官方镜像（国内直连）
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  MJPansa/DeepSeek-V4-Flash-0731-NVFP4 \
  --local-dir <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4

# 方式二：ModelScope（备选，nvidia 官方版镜像）
pip install -U modelscope
modelscope download --model FlagRelease/DeepSeek-V4-Flash-nvidia-FlagOS \
  --local_dir <MODELS_DIR>/deepseek-v4-flash-nvidia-nvfp4
```

---

## 5. 项目适配评估

| 维度 | 评估 |
|---|---|
| vLLM 兼容 | MJPansa 验证于 vLLM 0.26.1rc1（与项目 0.26.1.dev0 同代）✓ |
| 加载格式 | 建议 `--load-format instanttensor`（MJPansa 实测）+ 默认 safetensors 亦可 |
| TP4 四机 | 验证为 TP2 双机；TP4 需实测（NVFP4 权重在 TP4 下分布加载，48 shard 可整除） |
| KV | `--kv-cache-dtype fp8` + `--block-size 256`（沿用项目配置） |
| 投机 | **DSpark 保留**（`--speculative-config {"method":"dspark",...}`）——0731 自带投机模块 |
| 硬件前提 | NVFP4 解算依赖 CUDA 13.2+ cuBLASLt（8/7 已评估：需新镜像；595 驱动解锁 NVFP4 完整路径） |
| 预期收益 | MoE GEMM 走 NVFP4 专用 kernel（cuBLASLt 3× 潜力），decode 小 GEMM 需实测；**验证路径 = bench_matrix A/B（NVFP4 vs 现 FP8）** |

**落地建议**：先下载 MJPansa 0731-NVFP4 至 01/02（约 160-170GB 权重），容器内 vLLM 0.26.1 + NVFP4 镜像冒烟 → TP4 全矩阵 A/B 对比现 FP8 基线。

---

## 6. 参考

- HF：`MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（conversion-receipt.json、Technical Report arXiv:2606.19348）
- HF：`nvidia/DeepSeek-V4-Flash-NVFP4`（Model Optimizer v0.44.0）
- 魔搭：`deepseek-ai/DeepSeek-V4-Flash`（官方 FP4+FP8）、`FlagRelease/DeepSeek-V4-Flash-nvidia-FlagOS`
- 项目记忆：8/7 驱动/CUDA 评估（NVFP4 收益、595 驱动解锁）
