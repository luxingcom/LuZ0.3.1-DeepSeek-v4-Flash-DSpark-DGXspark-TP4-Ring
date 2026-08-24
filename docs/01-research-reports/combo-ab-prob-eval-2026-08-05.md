# 三功能 A/B + DSpark probabilistic 评估 综合报告

**日期**：2026-08-05
**工作流**：工作流 1（测试对比）+ 工作流 4（部署验证）组合
**参与成员**：Rex（SRE 工程师）/ Tessa（测试专家）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **三功能 A/B（regular cudagraph / DeepGEMM Grouped MoE / 组合）均不优于 F**——b12x + breakable + 动态K 已是 GB10 SM121 最优；**不固化任何变体**。
- **DSpark probabilistic 评估：吞吐 +20~47%（结构化负载），建议切换**——但须请求端 temp>0 才生效（temp≈0 回退 greedy）；GSM8K 无退化。
- **真正瓶颈发现**：TileLang 内核每请求 ~5s JIT 重编译（首请求惩罚 10-25x）——建议优先做预热覆盖。
- **阻塞 / 非阻塞**：非阻塞。F 生产已恢复（b12x + greedy + 动态K）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 A/B 完成；probabilistic 值得灰度验证 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条 |
| 建议下一步 | probabilistic 灰度验证（temp>0）+ TileLang 预热 |

---

## 📊 一、三功能 A/B（Rex）

### 前置确认
- grouped MoE 内核：原先无 → G2 测试中已生成 `sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d`（留缓存）
- `--moe-backend deep_gemm` 门控全过：SM121(family120)、quant 匹配 nvfp4_ds_mla、不拦 DeepSeek；实测日志 `Using 'DEEPGEMM_MXFP4' MoE backend` ✅

### A/B 结果（decode 口径 ctx2048/out256）

| 组 | 变更 | c1 | c5 | c10 | 结构化c5 | 接受率 | GSM8K | vs G0F |
|----|------|-----|-----|------|---------|--------|-------|--------|
| G0F | b12x+breakable+动态K（基线） | 23.7 | 76.6 | 85.5 | 67.6 | 0.481 | 0.80 | 基线 |
| G1 | regular FULL cudagraph | 22.9 | 60.9 | 77.2 | 68.7 | 0.484 | 0.80 | c5 -20% |
| G2 | DeepGEMM grouped | 22.1 | 54.0 | 69.3 | 59.2 | 0.469 | 0.90 | c5 -29% |
| G3 | G1+G2 组合 | 23.2 | 55.9 | 72.8 | 58.5 | 0.486 | 0.80 | c5 -27% |

### 结论
- **三功能均负向**（TTFT 批量 shape 重捕获 + TPOT 变差）；组合不叠加
- 社区 +28.6%/+5-20% 基于 SM100/其他堆栈，**本 GB10 SM121+0.26 下 b12x+breakable 已最优**
- 无真实 c1 增益（G0 的 6.9 是 TileLang JIT 异常伪象）

### 冷 vs 热
- 重启后首请求 TTFT 10-26s（TileLang JIT 重编译）vs 热态 0.7-1.0s → **首请求惩罚约 10-25x**
- 预热 2-3 请求后基本吸收；每请求仍有 ~5s JIT 抖动

---

## 🎲 二、DSpark probabilistic 评估（Tessa）

### 评估矩阵（接受率=聚合窗口；吞吐=c5/c10 中位数；GSM8K temp=0.6）

| 组 | 采样+动态K | code agg,acc | json agg,acc | prose agg,acc | GSM8K |
|----|-----------|-------------|-------------|--------------|-------|
| P0 | greedy+动态K temp=0 | 87/-, 82% | 121/106, 86% | 29/59*, 37%* | 18/20 |
| P0t | greedy+动态K temp=0.6 | 99/109, 79% | 113/108, 84% | 80/87, 37% | 16/20 |
| **P1** | **prob+动态K temp=0.6** | **142/150, 72%** | **137/140, 77%** | 90/87, 39% | **18/20** |
| P2 | prob+固定K5 temp=0.6 | 146/150, 73% | 132/143, 79% | 78/81, 30% | 18/20 |

### 权衡结论
| 维度 | greedy | probabilistic | 结论 |
|------|--------|--------------|------|
| 接受率 | 结构化反高（code 78.8% / json 86.9%） | 略低（72-73% / 77-80%） | **prob 无接受率优势** |
| 吞吐 | 基线 | 结构化 **+20~47%**（code +38-47% / json +17-32%），散文持平 | **prob 显著提升** |
| 质量 | temp=0: 18/20 | 18/20（同温下 18/20 > greedy 16/20） | **无退化** |
| 关键 | — | **需请求端 temp>0 才生效**（temp≈0 回退 greedy，SRE 源码确认） | ⚠️ 生产需配合 |

### 建议
- **值得切 probabilistic**，但**生产请求端必须 temp>0**（否则等于没切）；保留动态K
- 可选负载自适应：结构化 → prob+动态K+temp>0；散文/长文 → greedy
- 建议先灰度验证线上真实增益再全量

---

## ⚠️ 重要发现：TileLang JIT 瓶颈

- `mhc_pre_big_fuse_*_tilelang` 内核每请求重编译 ~5s → TTFT 波动大（c1 不可靠）
- 首请求惩罚 10-25x——**这是当前真实性能瓶颈**，优先级高于 MoE backend/cudagraph 切换
- **建议**：做 TileLang 内核预热覆盖（post-readiness warmup 生产形状）

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **probabilistic 灰度验证**：生产请求端 temp>0 + prob 采样，线上真实流量验证 +20~47% 增益 | SRE+Testing | P0 | 本周 |
| 2 | **TileLang 内核预热覆盖**（消除 ~5s JIT 抖动，真实瓶颈） | SRE+Archi | P0 | 本周 |
| 3 | 负载自适应方案设计（结构化→prob / 散文→greedy） | Archi | P1 | 2 周内 |
| 4 | 决策：prob 切换后 F 生产基线更新（保留回滚 greedy+动态K） | 人类负责人 | P1 | 灰度后 |

---

## ⚠️ 待完善 / 已知局限

- c10 超 max-num-seqs=6 有排队（已用多次中位数缓解）
- thinking-on / 超长思考链负载未覆盖（此前 greedy 接受率更高，需专项）
- probabilistic 灰度需生产端 temp>0 配合（工作流变更）

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE 工程师）**：前置确认 + G0-G3 A/B 数据（`~/fparam-exp/{G0,G0F,G1,G2,G3}_*.json`）+ F 恢复
- **Tessa（测试专家）**：P0/P0t/P1/P2 矩阵（`prob-eval-report-2026-08-05.md` + prob_P0_summary.json）
- **前置报告**：`deepgemm-moe-path-research-2026-08-05.md`、`f-param-verify-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（probabilistic 切换、TileLang 预热）请由人类工程负责人复核。
