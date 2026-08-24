# 监控面板"Prefill t/s 8k 假象"与"缓存命中率不体现"原因分析及改进方案

- 编号：RCA-MON-2026-08-14
- 日期：2026-08-14
- 范围：Grafana「DGX Spark vLLM 实时分析」「DGX Spark vLLM 集群」两个仪表盘的 prefill 吞吐与前缀缓存命中率面板
- 数据源：Prometheus <NODE_IP>:8191（vLLM 实例 <NODE_IP>:8001，scrape 5s）、Grafana <NODE_IP>:3000
- 关联测试：4 并发 131K 冷 prefill（随机前缀），`_tessa_tp4_bench/CONC_COMPARE/prefill4_131k_thread.json`

---

## 1. 问题现象

2026-08-14 上午执行 4 并发 131K 冷 prefill 测试（随机前缀，只输出 1 token）期间，出现两个面板读数与实测不符：

| # | 现象 | 实测（可信值） | 面板读数 | 偏差 |
|---|------|---------------|---------|------|
| A | "预填充有效速度 Prefill t/s"面板爆出 8k | 2,409 t/s（墙钟聚合） | 8,168 / 8,108 t/s（06:14–06:15） | 虚高 3.4× |
| B | "前缀缓存命中率"面板未见命中体现 | 754,944 tokens 缓存命中（铁证） | 主测试期间 0% | 漏报 |

两个现象同源（同一批缓存命中 tokens），暴露了面板公式的三个结构性缺陷。

---

## 2. 根因分析（RCA）

### 2.1 现象 A：Prefill t/s 面板虚高 —— 5m 窗口吞入缓存命中批次

**面板公式**（实时分析仪表盘）：
```
sum(increase(vllm:request_prompt_tokens_sum{job="vllm"}[5m]))
/ clamp_min(sum(increase(vllm:request_prefill_time_seconds_sum{job="vllm"}[5m])), 1)
```

**触发时序**（Prometheus 5s 秒级取证）：

| 时刻 UTC | prompt_tokens 计入 | 其中缓存命中 | 命中率 | 引擎耗时 prefill_time |
|---|---|---|---|---|
| 06:12:30 | +181,010 | 171,264 | 94.6% | 仅 6.4s |
| 06:14:00 | +534,375（≈4×133,594） | 450,304 | 84.3% | 仅 50.4s |
| 06:14:05 | +133,600 | 133,376 | 99.8% | 仅 0.4s |
| 06:23:30–40（主测试） | +534,772 | 0（随机前缀） | 0% | 872.1s |

06:14–06:15 面板 5m 滑动窗口 `[06:10, 06:15]`：
- 分子 = 985,728 tokens（其中 **754,944 为缓存命中，占 76.6%**）
- 分母 = 121.5s（缓存命中 tokens 几乎零引擎成本）
- 比值 = **8,113 t/s ≈ 面板读数 8,168/8,108** ✓ 复现吻合

**根因**：`request_prompt_tokens_sum` 按**全部 prompt tokens**（含缓存命中部分）计入分子，而 `request_prefill_time_seconds_sum` 只统计**真实计算耗时**（命中部分几乎为 0）。窗口内一旦混入命中批次，分子虚大、分母虚小，PR 指数级虚高。**面板语义是"有效吞吐"而非"计算吞吐"，且未对命中做任何标注。**

### 2.2 现象 B：命中率面板不体现 —— 窗口时段错位 + 短窗口采样稀疏

**两个命中率面板公式**：

| 面板 | 公式 | 窗口 |
|---|---|---|
| 实时分析 | `sum by (node)(rate(vllm:prefix_cache_hits_total[30s])) / clamp_min(sum by (node)(rate(vllm:prompt_tokens_total[30s])),1)` | 30s |
| 集群 | `sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])),1) * 100` | 5m |

**复现时间线**：

| 时刻 UTC | cluster 5m | realtime 30s | 说明 |
|---|---|---|---|
| 06:12:30 | 77.6% | 66.9% | 命中批次开始 |
| 06:13:30 | 54.4% | **99.7%** | 命中高峰 |
| 06:14:30–06:18:30 | 47.7–75.4% | 0% | 命中计入完毕 |
| **06:20 起** | **0%** | **0%** | 主测试（随机前缀，真 0 命中） |

**根因**（三层）：
1. **命中与观察窗口错位**：754,944 tokens 命中全部发生在 06:12:30–06:14:30（早期缓存命中版测试，前缀相同），最终主测试用随机前缀**本来就 0 命中**。用户在 06:20 后观察时，面板归零是**正确反映**，但无法回溯看到命中。
2. **realtime 30s 窗口过短**：命中仅 ~90 秒可见（3 个采样点 66.9/99.7/74.9%），其余全 0；无人恰好盯住即漏报。
3. **cluster 5m 窗口滑过即归零**：命中 06:14:30 计入完毕，5m 窗口 06:19:30 后不再有增量，无"事件留痕"能力。

**指标口径核对（排除计数 bug）**：
- `vllm:prefix_cache_hits_total` 增量 = 754,944 = `vllm:prompt_tokens_cached_total` 增量（完全同步）→ **hits 是 token 级计数**（非请求次数）
- `vllm:prefix_cache_queries_total` 增量 = 2,105,085 = 窗口内全部 prompt tokens → queries 是 token 级
- 结论：cluster 面板 hits/queries 本质即 token 级命中率，**公式正确、数据无丢失**，问题只在窗口与可视化呈现。

### 2.3 结构性缺陷三：面板口径 = per-request 平均，非聚合吞吐

三数自洽关系验证：
- per-request：133,693 / 219.15s ≈ **610 t/s**
- 面板干净读数：534,772 / 872.1s ≈ **613 t/s**（= per-request 均值，因为 Σprefill_time 是各请求耗时之和，Σtokens/Σtime 分子分母同除并发度后即单请求均值）
- 墙钟聚合：534,772 / 221.98s ≈ **2,409 t/s** = 613 × 3.9（4 请求并行复用引擎）

**面板显示的 613 是"每请求平均速度"**，用户预期的"4 并发总吞吐"是 2,409。面板无任何标注说明，导致口径误读。

---

## 3. 面板缺陷清单

| # | 面板 | 缺陷 | 严重度 |
|---|------|------|--------|
| D1 | 预填充有效速度 Prefill t/s | 分子含缓存命中 tokens，命中批次可致读数虚高数倍（实测 3.4×） | P0 |
| D2 | 预填充有效速度 Prefill t/s | 口径为 per-request 平均，无标注，易误读为聚合吞吐 | P1 |
| D3 | 前缀缓存命中率（实时分析） | 30s 窗口过短，命中事件 90s 后即不可见，采样稀疏 | P1 |
| D4 | 前缀缓存命中率（集群） | 5m 窗口滑过后归零，无事件留痕；空闲期 NoData 与 0% 混淆 | P1 |
| D5 | （缺失）缓存命中 tokens 绝对值面板 | 无 `prompt_tokens_cached_total` 速率面板，命中 burst 只能靠间接推断 | P2 |
| D6 | （缺失）聚合吞吐面板 | 无墙钟口径聚合 PR/DE 面板 | P2 |

---

## 4. 改进方案

### 4.1 P0 —— 修正 Prefill t/s 公式（立即执行）

**目标**：命中批次不再污染 PR 读数。

方案 A（推荐，最小改动）：
```
# 纯计算 PR（排除缓存命中）
sum(increase(vllm:request_prompt_tokens_sum{job="vllm"}[5m]))
  - sum(increase(vllm:prompt_tokens_cached_total{job="vllm"}[5m]))
/ clamp_min(sum(increase(vllm:request_prefill_time_seconds_sum{job="vllm"}[5m])), 1)
  * ((sum(vllm:num_requests_running{job="vllm"}) + sum(vllm:num_requests_waiting{job="vllm"})) > 0)
```

方案 B（更完整）：面板拆两个 Series——
1. `总 PR（含命中）`：原公式不动，标题加注"含前缀缓存命中"
2. `纯计算 PR`：上述减命中公式

**验证标准**：重放 2026-08-14 数据，06:14–06:15 读数应从 ~8,100 回落到 ≤2,500；06:25 后保持 613 ± 10 不变（与既有干净窗口一致）。

### 4.2 P1 —— 命中率面板窗口与留痕

| 改动 | 内容 |
|---|---|
| 统一窗口 | 两面板统一为 5m `increase`（或 rate），避免 30s 采样稀疏 |
| 事件留痕 | 新增"命中 tokens 速率"面板：`sum(rate(vllm:prompt_tokens_cached_total[1m]))`，命中 burst 以绝对值曲线呈现，不随窗口滑过消失 |
| NoData 区分 | 命中率面板加 `or vector(0)` 或在图例标注"无请求时无数据"，避免 0% 与 NoData 混淆 |
| 口径标注 | 面板描述注明"token 级命中率 = 命中 tokens / 全部 prompt tokens" |

### 4.3 P2 —— 聚合吞吐与运维配套

| 改动 | 内容 |
|---|---|
| 聚合 PR 面板 | 新增"聚合 Prefill 吞吐（墙钟）"：`sum(increase(vllm:request_prompt_tokens_sum[5m])) / 300`，注明"窗口平均聚合值，per-request 均值 ≈ 本值/并发度" |
| 清缓存工具 | 记录测试前清前缀缓存操作（vLLM 0.26 支持 `DELETE /prefix_cache` 或重启）到 Runbook；或测试脚本自动换随机前缀（已实践） |
| 告警（可选） | 命中 tokens 速率突增（如 >50K t/s）时提示"存在命中批次，PR 面板读数可能虚高" |

### 4.4 测试报告口径规范（配套）

后续基准报告统一三口径并注明：
- **per-request PR/DE**：每请求 tokens/耗时（面板 613 同源）
- **聚合墙钟 PR**：Σtokens/总墙钟（2409，回答"这批任务多久完成"）
- **命中率**：命中 tokens/全部 tokens（token 级）

---

## 5. 验证与回滚

- **验证**：修改后重放 2026-08-14 06:08–06:26 数据，三个面板读数与本文档第 2 节复现值对照；再用真实 4 并发随机前缀测试冒烟（预期：PR 面板 ~2,400 峰值、命中率 0%、命中 tokens 速率 0）。
- **回滚**：Grafana 面板 JSON 修改前在 `_tessa_tp4_bench/` 备份（realtime/cluster 两文件已存），异常时还原。

## 6. 附录：关键证据数据

- 8k 复现：985,728 tokens（含 754,944 命中）/ 121.5s = 8,113 t/s ≈ 面板 8,168/8,108
- 干净窗口对照：534,772 / 872.1s = 613 t/s（06:25 后稳定）
- 命中率高峰：cluster 77.6%（06:12:30）、realtime 99.7%（06:13:30）
- 指标同步性：hits_total 与 cached_total 增量一致（171,264 → 621,568 → 754,944）
- 前置报告：`prefill-pr-8k-forensics-2026-08-14.md`
