# DeepSeek V4 Flash 0731 DSpark Draft Head 调研报告

**日期**：2026-08-04
**工作流**：工作流 2（系统设计 - 技术调研专项）
**参与成员**：Archi（架构师）/ Zhen（主理人，编排与汇编）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：DeepSeek-V4-Flash-0731 **内嵌专属 DSpark speculative decoding draft head**（3×MTP 块 + Markov 头 + 置信头），权重与 target 同 checkpoint（48-shard），**当前生产环境已在正确使用该 draft 模块**——无需换模型或"启用新 draft head"。
- **严重度分布**：🟡中 2 项（draft_sample_method 差异 / Anemll 简化版 DSpark）+ 🟢低 2 项（num_spec 差异 / Ollama 无本地 DSpark）
- **阻塞 / 非阻塞**：非阻塞。生产配置合法有效，仅存在可选优化空间（greedy A/B 验证）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（生产配置正确，无需变更） |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（greedy A/B / 长上下文接受率监控 / 升级预案跟踪） |
| 建议下一步 | v0.26 灰度时验证 `draft_sample_method=greedy`（用户已决策） |

---

## 🔍 调研详情

### 1️⃣ Hugging Face 平台结论

| 项 | 内容 |
|----|------|
| 仓库 | `deepseek-ai/DeepSeek-V4-Flash-0731`（MIT、ungated） |
| 版本 | 2026-07-31 发布，rev ≈ 9e165c30（官方 GA 版，取代 preview） |
| 0731 新增 | post-training 增强 agentic 能力；**架构与 DSpark 相同、内嵌投机解码模块** |
| 模型卡声明 | "comes with a speculative decoding module attached" |
| config.json 新增字段 | `dspark_block_size=5`、`dspark_target_layer_ids=[40,41,42]`、`dspark_markov_rank=256`、`dspark_noise_token_id=128799` |
| 权重形态 | 48 个 safetensors，**draft 权重内嵌同 checkpoint，无独立 draft 文件** |
| 性能声明 | DSpark 论文：V4-Flash per-user **+60~85%**（vs MTP-1）、吞吐最高 ~6.6× |
| 官方 vLLM 配方 | `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'` |

### 2️⃣ Ollama 平台结论

| 项 | 内容 |
|----|------|
| 条目 | `ollama.com/library/deepseek-v4-flash` |
| 版本 | tag `cloud`（旧）、`0731-cloud`（3 天前新增） |
| DSpark 支持 | **未发现 Ollama 本地 DSpark / 独立 draft head 支持**；0731 仅云端托管，无本地权重 |
| 与 vLLM 差异 | Ollama 0731 走官方远端后端；本地 GGUF 导入（unsloth/frob 等）基于 preview 且会丢弃 draft 模块；**本地真正跑 DSpark 的是 vLLM + Anemll 路径（即当前生产）** |

### 3️⃣ DSpark draft head 专项结论：**有，且生产已在用（内嵌式）**

**结构**：
- 3 个 MTP 块（挂 backbone 第 40/41/42 层）+ Markov 头（rank 256 低秩）+ 置信头（单线性层）
- 块长 γ=5，借用 target embedding/输出头
- 参数量：304B ≈ 284B base + **~20B draft 模块**

**与当前生产差异**（`deepseek-v4-flash-0731` + `method=dspark` + `num_speculative_tokens=5`）：

| 维度 | 官方配方 | 当前生产 | 判定 |
|------|---------|---------|------|
| draft_sample_method | `greedy` | `probabilistic` | 🟡 官方推荐 greedy（接受率通常更高） |
| num_speculative_tokens | 7（GB300 配方） | 5（= dspark_block_size，2×DGX 社区标准） | ✅ 合法（5 ≥ 块长 5） |
| DSpark 能力 | 完整（置信调度/Stage-C 动态裁剪） | Anemll 0.1.1 简化版（固定 5-token 预算，无动态裁剪） | 🟠 能力简化 |
| 长上下文接受率 | — | 实测退化：thinking 关 ~40% / 开 ~24% | ⚠️ 600-800k 下需关注 TPOT/TTFT |

### 4️⃣ 实测参考（2×DGX Spark / Anemll 0.1.1 / TP=2 / MTP=5，probabilistic）

- thinking 关：接受率 ~40%，55 tok/s
- thinking 开：接受率 ~24%，42 tok/s
- short monologue（accept~0.3）：31~87 tok/s
- **结论**：长推理/agentic 轨迹收益降低——生产 600-800k 长上下文下需关注退化

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | v0.26 灰度时验证 `draft_sample_method=greedy`（用户已决策；测试方案已含 Greedy A/B 专项，阈值 ≥5% 切换） | Testing+SRE | P0 | v0.26 灰度时 |
| 2 | 长上下文（600-800k）下监控接受率与 TPOT/TTFT 退化 | SRE | P1 | 灰度后 |
| 3 | 跟踪 Anemll `nvfp4-a4w4` 分支（draft 拆为 dspark/独立路径）作升级预案 | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- 性能数字（+60~85%、6.6×）来自 DSpark 论文与官方声明，未在本集群实测验证
- v0.26 上游对 dspark 的校验行为（num_spec ≥ block_size）需在灰度时确认
- Anemll 0.1.1 未暴露置信调度器/Stage-C 旋钮——动态裁剪能力缺失为已知简化

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）原始产出**：HF 模型卡 + config.json 字段核对、Ollama 平台条目核查、DSpark draft head 结构分析、与生产配置差异表、实测参考数据
- **关联决策**：用户已拍板「v0.26 灰度验证 greedy」（2026-08-04 17:08）
- **关联报告**：`_tessa_v026_bench_plan.md`（§3.3 Greedy A/B 专项）、`PLAN-v026-gray-deploy-2026-08-04.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（greedy 切换阈值）请由人类工程负责人复核。
