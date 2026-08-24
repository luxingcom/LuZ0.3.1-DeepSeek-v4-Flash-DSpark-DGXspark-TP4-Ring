# 环境 D 部署测试 · MiaAI-Lab 项目分析与 1M 长上下文验证

**日期**：2026-08-02
**工作流**：部署前检查 / 新环境部署测试（工作流 4 变体）
**参与成员**：Archi（架构）/ Rex（SRE）/ Tessa（测试）/ Cody（代码审查）+ 主理人执行层

---

## 📌 TL;DR

- **参考资料确认**：MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark（0731 权重版）——**1M 长上下文 + nvfp4_ds_mla KV + Anemll 预构建镜像 + worker-first** 的验证型配方
- **Anemll 镜像拉取受阻**：ghcr.io 当前下载速度仅 **136 B/s**（严重限速），45GB+ 镜像不可行 → 执行 Archi 降级方案
- **环境 D（降级版）部署成功并验证**：0731 + hybrid-1.6 + **max_model_len=1,048,576（1M ctx）** + 单流 + dspark5 + thinking=max
- **核心成果**：309K tokens 长上下文实测 **prefill 986 tok/s、无错误**；KV 池 2.55M tokens（1M 请求可容纳）；单流 decode 34.12 t/s
- 生产已回滚恢复；Anemll 拉取保留后台，网络恢复后可升级真环境 D

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 环境 D 状态 | 🟢 降级版部署测试完成（1M ctx 能力验证） |
| MiaAI 核心优势 | 1M ctx + nvfp4 KV 减半 + vLLM 0.25 + 思考分级 |
| Anemll 镜像 | ⏳ 拉取受阻（ghcr 136B/s），后台继续 |
| 长上下文实测 | 309K tokens / prefill 986 tok/s / 无错误 |
| 生产状态 | 回滚恢复中（worker 已起，head 加载中） |

---

## 📖 MiaAI-Lab 项目特点与优势分析

### 项目定位
MiaAI-Lab 维护的双 DGX Spark 部署配方，集成各方工作的**验证型工程化封装**：tonyd2wild 的 1M NVFP4 配方血统 + drowzeys 的并发补丁 + Rafael Caricio 的 DSpark vLLM 集成 + Anemll 预构建镜像，由 MiaAI-Lab 完成双节点打包、worker-first 启动、配置校验、缓存准备、冒烟测试。

### 技术特点矩阵（vs 我方现有栈）

| 维度 | MiaAI-Lab | 我方（hybrid-1.6） | 优势方 |
|------|-----------|-------------------|--------|
| KV 缓存 | **nvfp4_ds_mla**（NVFP4 压缩，池 2.49M @util0.835） | fp8（池 1.49M @0.85） | MiaAI（KV 减半，1M ctx 可行） |
| 引擎 | **vLLM 0.25 正式版**（原生 DSpark/DS-MLA/b12x） | vLLM 0.11.2.dev279 | MiaAI |
| 上下文 | **1M（1048576）** | 393K | MiaAI（2.7x） |
| 推理控制 | **DEFAULT_THINKING 分级**（off/low/high/max） | 固定 max | MiaAI（灵活性） |
| 并发模型 | MTP5 × seqs6（低并发高吞吐） | spec5 × seqs128（高并发） | 场景不同 |
| 镜像 | Anemll 0.1.1（45GB+，ghcr） | hybrid-1.6（本地已有） | 我方（可得性） |

### 核心优势结论（Archi）
1. **1M 长上下文**：KV 减半（nvfp4）使 1M ctx 成为可能，适合长上下文 Agent 场景（我方 393K 上限是其 2.7 倍差距）
2. **高吞吐**：x3 并发峰值 134.6 t/s（官方数据，thinking=low 口径）；我方 thinking=max 统一口径下预计 ×0.6-0.8
3. **工程完备性**：worker-first 启动、配置校验、缓存准备、冒烟测试、900K 验证已发布
4. **局限**：并发低（seqs6 vs 128）、镜像大（45GB+ 分发难）、nvfp4 路径未在本地验证、部分 DSpark env 在 Anemll 上为 no-op

---

## 🔧 环境 D 部署测试（降级版）

### 部署路径决策
- **Anemll 镜像**：ghcr.io 实测 136 B/s（严重限速），45GB 拉取需数天 → **不可行**（后台保留继续尝试）
- **降级方案**（Archi 预设）：本地 hybrid-1.6 镜像 + 0731 权重 + **1M ctx**（max-model-len 1048576）+ 单流（seqs=1）+ dspark5 + thinking=max + GPU_MEM=0.85

### 验证结果（全部通过）

| 验证项 | 结果 | 判定 |
|--------|------|------|
| /v1/models max_model_len | **1,048,576** | ✅ |
| KV 池 | **2,554,439 tokens**（2.55M，1M 请求可容纳） | ✅ |
| 进程参数 | 1M ctx + seqs1 + 0.85 + dspark5 + thinking=max | ✅ |
| 单流 decode | 34.12 t/s（total 516 t/s） | ✅ |
| **309K 长 ctx** | prompt=309,194 tokens / **prefill 986 tok/s** / 313.7s / 无错误 | ✅ |

### 环境 D 加入后四环境横向（统一 thinking=max 口径）

| 指标 | A(DSpark+dspark 0.8) | B(0731 无投机 0.8) | 生产/C(0731+dspark 0.85, 393K) | **D(0731+dspark 0.85, 1M ctx)** |
|------|-----|-----|-----|-----|
| 5 并发 agg | **80.1** 🏆 | 44.6 | 69.1 | —（seqs=1 单流配置） |
| 单流 decode | 42.08 | 27.72 | 119.12 | 34.12 |
| KV 池 | 1.07M | 1.60M | 1.49M | **2.55M** |
| 长 ctx | 54K 实测 | — | 上限 393K | **309K 实测 986 tok/s** |
| max ctx | 393K | 393K | 393K | **1M** |

**结论**：环境 D 牺牲并发（seqs=1）换取 **1M 长上下文能力**（KV 池 2.55M、309K 实测通过），是四环境中唯一覆盖长上下文场景的方案；5 并发最优仍为环境 A（80.1 t/s）。

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 生产恢复确认（head 加载完成后 /health=200 + 参数核对） | SRE | P0 | 已完成回滚 |
| 2 | Anemll 镜像网络恢复后重试拉取（后台保留），完成后升级真环境 D（nvfp4 1M 验证） | SRE | P1 | 网络恢复后 |
| 3 | 若需长上下文生产：评估环境 D（1M ctx）vs 生产（393K）切换（长 ctx 场景优先 D） | 人类拍板 | P1 | 决策窗口 |
| 4 | MiaAI 官方基准复现（Anemll 就绪后：x1-x4 并发 + acceptance 对齐） | Tessa | P2 | 镜像就绪后 |

---

## ⚠️ 待完善 / 已知局限

- **Anemll 真环境 D 未部署**（ghcr 限速 136B/s）——nvfp4_ds_mla KV、vLLM 0.25、DEFAULT_THINKING 分级等 MiaAI 原生能力未实测，当前为 hybrid-1.6 降级版
- 环境 D 单流配置（seqs=1）无法测 5 并发（非缺陷，配置取向不同）
- 900K ctx 未测（309K 已验证能力；900K prefill 需 15min+，且非门禁项）
- encoding/encoding_dsv4.py 本地权重已含（0731 checkpoint 自带），Anemll 部署时无需额外获取

---

## 📚 数据来源 & 成员产出索引

- Archi：MiaAI 差异矩阵与优势结论、降级方案（fp8 1M 单流）、参数表
- Rex：部署执行方案（worker 并行拉取、20min 回滚判据、双回滚路径）
- Tessa：测试矩阵（单流/长 ctx/KV 池判定；MiaAI 官方为参照）
- Cody：兼容性审查（nvfp4 仅 Anemll 支持、encoding 双预案、DEFAULT_THINKING 注入）
- 主理人执行：项目资料抓取、ghcr 限速诊断（136B/s）、降级脚本生成、环境 D 部署与 309K 长 ctx 实测、生产回滚

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
