# 指标矩阵 + SM121/NVFP4 核实 + DSV4 融合算子 + 开源网关综合报告

**日期**：2026-08-05
**工作流**：工作流 2/5 组合（系统设计调研 + 测试评估）
**参与成员**：Tessa（测试专家）/ Archi（架构师）/ Docu（技术文档师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **指标矩阵**：4 上下文档 × 3 并发档全部补测完成——TTFT 随上下文近似线性、单流 decode 36-45 t/s 不随上下文劣化、c5 聚合输出 512→70.7 t/s vs 32768→6.6 t/s（差一个数量级，长上下文并发增益趋平）。
- **SM121/NVFP4**：5 项 SM121 优化全部生效（nvcc_wrapper 实测 sm_121a cubin）；**NVFP4 KV 已启用**（nvfp4_ds_mla，DSV4 专用格式）。
- **融合算子**：DeepGEMM Mega MoE 值得引入测试（decode MoE 瓶颈，收益最大）；FlashInfer sparse MLA 不建议（格式冲突 + sm121 livelock 缺陷）。
- **开源网关**：主选 **LiteLLM Proxy**（虚拟 key + Admin UI + 8ms@1kRPS + /v1/responses），备选 vLLM 原生双轨；方案 A 字段镜像可保留。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 调研完成，4 项结论齐备 |
| 阻塞项数量 | 0 |
| 关键行动项 | 5 条 |
| 建议下一步 | 网关切换（LiteLLM）+ Mega MoE 引入评估并行推进 |

---

## 📊 一、指标矩阵（Tessa，12 组合补测完成）

**文档**：`bench-matrix-v026-2026-08-05.md`（表 1 完整矩阵 + 逐行解读）

### 核心数据（代表性档位）

| 档位 | TTFT | TPOT | 单流 decode | 聚合输出 |
|------|------|------|------------|---------|
| 512/c1 | **508ms** | 25.8ms | 38.8 t/s | —（交互最佳档） |
| 2048/c1 | ~1s 级 | ~26ms | 40+ t/s | — |
| 8192/c5 | 12.8s | 126ms | 36-45 t/s | 20.8 t/s（c3→c5 仅 +4%） |
| 32768/c5 | 50.0s | 305ms | 36-45 t/s | **6.6 t/s**（最脆弱档，建议并发≤3） |

### 跨行规律（关键洞察）
- **TTFT 随上下文近似线性**（508ms → 50s，32k 档 prefill 主导）
- **单流 decode 36-45 t/s 不随上下文劣化**（decode 非瓶颈）
- **c5 聚合输出 512→70.7 vs 32768→6.6 t/s，差一个数量级**——长上下文下并发增益趋平甚至为负

### DSpark 接受率（探针）
| 上下文 | 512 | 2048 | 8192 | 32768 | 合计 |
|--------|-----|------|------|-------|------|
| 接受率 | 26.3% | 30.7% | 33.2% | 32.6% | 30.6% |

- per-position pos0-4 = 64.8/39.9/23.0/15.1/9.7%（正常衰减）
- ⚠️ 口径修正：固化版原记 52.6% 重算为 51.3%（计数器不闭合，疑笔误）；本次 26-33% 偏低归因合成重复文本可预测性差 + 短输出，非引擎退化

---

## 🏗️ 二、SM121 优化项 + NVFP4 核实（Archi，SSH 实测）

### SM121 优化项清单（GPU=GB10 CC12.1/sm_121a，全部生效 ✅）

| 项 | 机制 | 证据 |
|----|------|------|
| DG_JIT_NVCC_COMPILER=nvcc_wrapper.py | DeepGEMM JIT 调 wrapper 把 sm_120f→sm_121a | 容器 env 确认 |
| nvcc_wrapper 重写真实生效 | cubin 实际架构 | cuobjdump：`kernel.sm_121a.cubin`（fp8_fp4_gemm + mqa_logits） |
| TORCH_CUDA_ARCH_LIST=12.1a | torch/扩展编译目标 | 容器 env 确认 |
| deep_gemm JIT cache 持久卷 | 重启免重编 | 129 内核，cubin 均 sm_121a |
| VLLM_USE_B12X_MOE / TRITON_MLA_SPARSE / BREAKABLE_CUDAGRAPH=1 | fork 定制 DSV4 路径 | 容器 env + envs.py 定义 |

### NVFP4 KV 结论
✅ **已启用 `nvfp4_ds_mla`**（DSV4 专用 FP4 KV 格式，非通用 fp8_ds_mla）。证据：start 脚本 L38 + 容器 cmdline 双确认。模型 deepseek-v4-flash-0731、max-model-len 600000、dspark 投机 5-token。

---

## 🔬 三、DSV4 融合算子比选（Archi）

| 候选 | 来源 | 兼容性 | 收益 | 结论 |
|------|------|--------|------|------|
| vLLM 官方 DSV4 注意力融合组（PR#40760） | 官方 | ✅ 已并入 v0.26 | 大 | 已应用 |
| DeepGEMM FP8xFP4 GEMM/bmm | deepseek-ai | ✅ cache 已编译 sm_121a | 中大 | 已应用 |
| **DeepGEMM Mega MoE**（2026.04） | 官方 | ⚠️ 需 fork 集成 + wrapper 验证 | **大**（decode MoE 瓶颈） | **值得引入测试** |
| DeepGEMM FP4 Indexer scoring | 官方 | ✅ sm_121a 产物在 | 中 | 已应用 |
| FlashInfer sparse MLA | flashinfer | ❌ nvfp4 冲突 + sm121 livelock 已知缺陷 | 中 | **不建议** |
| Triton sparse MLA | fork/社区 | ✅ 已启用 | 中 | 已应用 |
| KT-Kernel DSV4 卸载 | kvcache-ai | ❌ 场景不符 | — | 不适用 |
| vLLM Paged prefill kernel | 官方 | ⏳ 未发布 | 中 | 跟踪 |

**可应用结论**：当前栈已应用 4 项；**Mega MoE 是下一个收益点**（需确认 fork 内 B12X_MOE 路径是否已接，未接则按官方 2026.04 版集成 + wrapper 编译验证 sm121a）。

---

## 🚪 四、开源网关比选（Docu）

### 推荐：主选 **LiteLLM Proxy**，备选 **vLLM 原生双轨**

| 网关 | 端点覆盖 | 鉴权 | 性能 | 结论 |
|------|---------|------|------|------|
| **LiteLLM** | chat/responses/embeddings/audio | 虚拟 key + 预算 + Admin UI | 8ms P95@1kRPS | **主选** |
| vLLM 原生 | 最全（含 audio） | 仅单静态 key | 高 | 备选（内网） |
| Ollama / LM Studio | 基础 | 无 | 低 | 本地工具，不可替代 |
| Higress / Envoy AI GW | chat 为主 | key 托管 | 高 | K8s 运维重 |
| Portkey | 200+ provider | RBAC | <10ms | 观测强，自托管需自建 |

### 替代成本与风险
- **迁移成本低**：config.yaml 声明 model_list（local-v4-flash → hosted_vllm/deepseek-v4-flash-0731）；虚拟 key 可生成 <API_KEY>-*/<API_KEY>-* 分层
- **方案 A 保留**：vLLM 原生已输出 reasoning_content 可直接透传；镜像逻辑放 LiteLLM CustomLogger 回调或轻量 sidecar
- **风险**：① SSE delta 中 reasoning_content 结构需实测 ② /v1/responses 透传程度未经验证 ③ 嵌入批量/维度需测 ④ usage/error 格式差异

### 落地要点
- `docker run -p 4000:4000 -v $PWD/config.yaml:/app/config.yaml ghcr.io/berriai/litellm:v1.83.7-stable --config /app/config.yaml`（虚拟 key 持久化需 PostgreSQL）
- 双轨过渡：LiteLLM(4000) 对外 + vLLM(8001) 内网并行，灰度后下线 8003

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 网关切换评估落地：LiteLLM 部署验证（端点覆盖 + 方案 A 保留 + SSE 实测） | SRE+Cody | P0 | 本周 |
| 2 | DeepGEMM Mega MoE 引入评估（fork B12X_MOE 路径确认 + sm121a 编译验证） | Archi | P1 | 2 周内 |
| 3 | 长上下文（≥32k）并发策略：建议并发≤3（c5 增益趋平为负）写入部署规范 | SRE | P1 | 本周 |
| 4 | DSpark 接受率口径统一（计数器闭合问题核实 + 合成文本测试方法标注） | Testing | P2 | 按需 |
| 5 | 跟踪 vLLM Paged prefill kernel / DeepGEMM FP4 Indexer 上游更新 | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- 指标矩阵使用合成重复文本构造长上下文（真实场景可预测性更高，接受率/吞吐或更优）
- c1 档合成文本偶发提前结束（ct=37~99）致吞吐偏低，已取 2 轮中位标注
- 网关比选基于官方文档/社区资料（未实测 LiteLLM 部署）
- Mega MoE 收益为推断（官方 2026.04 发布口径），未在本环境实测

---

## 📚 数据来源 & 成员产出索引

- **Tessa（测试专家）**：指标矩阵补测（12 组合 + 接受率探针），`bench-matrix-v026-2026-08-05.md`
- **Archi（架构师）**：SM121 清单（cuobjdump 证据）/ NVFP4 确认 / 融合算子比选表
- **Docu（技术文档师）**：网关对比表 + LiteLLM 推荐 + 落地要点（源：vLLM 官方文档 / LiteLLM 文档 / Higress 博客 / Ollama 文档）

---

> 本报告由工程保障团队 AI 协作生成，关键决策（LiteLLM 切换、Mega MoE 引入）请由人类工程负责人复核。
