# vLLM 监控指标正确性调查与 Grafana 面板修复报告

**日期**：2026-08-07
**工作流**：工作流 3（事故响应/调查）变体——监控指标正确性
**参与成员**：Rex（SRE，修复方案）/ 主理人实施（源码取证 + 面板更新）

---

## 📌 TL;DR（执行摘要）

- 用户质疑三个指标：① KV Cache 水位运行时比例非常小 ② 生成吞吐受 PR 影响需解耦 ③ 投机解码数据"反了"
- **源码级取证**（vLLM 0.26.1dev 镜像内 /usr/local/lib/python3.12/dist-packages/vllm/）：① **KV 面板换算 bug 确认**（kv_cache_usage_perc 为 0~1 比值未 ×100，Grafana percent 不自动换算 → 显示 0.08 实为 8%）；② 生成吞吐**无 bug**（generation_tokens 已是纯 decode，vLLM 已解耦），缺 ITL 直指标面板；③ 投机解码**方向没反**（面板公式与官方注释公式逐字一致），口径单一致观感偏差
- **已实施修复**：KV ×100、生成吞吐窗口平滑、投机面板重命名、**新增 3 个面板**（ITL 输出间隔 / Mean Acceptance Length / Accepted-Emitted 覆盖率）；顺带巡检其余 16 面板（TTFT/TPOT/DCGM/网络等全部正确）
- Grafana 面板已更新（vllm-realtime，version 8，22 面板），**数值验证须等 vLLM 下次启动**（当前停机配合视频工作流）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（3 质疑全部定位，1 真 bug 已修） |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条（见行动清单） |
| 建议下一步 | vLLM 恢复后按验证流程核对数值并留存基线截图 |

---

## 1. 用户质疑与源码级结论

### 1.1 KV Cache 水位"比例非常小" —— ✅ 面板换算 Bug（已修复）

**源码证据**（vLLM 0.26.1dev）：
- `vllm/v1/core/kv_cache_manager.py` L187：`usage` 属性返回 **0.0~1.0**（注释原文 "between 0.0 and 1.0"）= `block_pool.get_usage()`
- `vllm/v1/metrics/loggers.py` L562：Gauge `vllm:kv_cache_usage_perc`；L1123 `set(scheduler_stats.kv_cache_usage)` 直接 set 0-1 值

**面板问题**：expr `vllm:kv_cache_usage_perc`（无 ×100）+ unit=percent。**Grafana percent 单位只加 % 符号不自动 ×100** → 显示 0.08（实为 8%）。

**修复**：`vllm:kv_cache_usage_perc * 100`（已实施）

**业务侧解释**：seqs=6 + 600K ctx 按最大预留 KV cache，实际并发低（1-2 请求、上下文短）→ **真实水位 10-20% 属正常**，非异常信号。

### 1.2 生成吞吐"受 PR 影响需解耦" —— ✅ 无 Bug，补 ITL 直指标

**源码证据**：
- `vllm/v1/metrics/loggers.py` L705：`vllm:generation_tokens`（Counter → 暴露 `generation_tokens_total`）= **纯 decode 输出**
- L671：`vllm:prompt_tokens`（`prompt_tokens_total`）= prefill 输入 —— **vLLM 已解耦**
- L830：`vllm:inter_token_latency_seconds`（Histogram）= 纯 decode 每 token 间隔，不受 prefill 影响 —— **面板未使用**

**结论**：生成吞吐面板 `sum(rate(vllm:generation_tokens_total))` 本身就是纯 decode，无串扰。用户感知"低"= 大 ctx prefill 主导时 decode 天然稀释（物理事实，此前 ctx-throughput-analysis 已证实 ~90-96%）。

**修复**：窗口 `[30s]`→`[$__rate_interval]` 平滑；**新增 ITL 面板**（P50/P99/均值，histogram_quantile + sum by(le)）——输出速度最直接指标。

### 1.3 投机解码"反了" —— ✅ 方向没反，口径单一

**源码证据**：`vllm/v1/spec_decode/metrics.py` L182-183 官方注释公式 = `rate(spec_decode_num_accepted_tokens_total)/rate(spec_decode_num_draft_tokens_total)`——**与面板公式逐字一致**；L186-190 补充公式：mean acceptance length = `1 + rate(accepted)/rate(drafts)`。

**结论**：公式正确，接受率 ~57-60% 与 bench 基线（512/8K/32K = 60.7/58.9/57.4%）吻合。"反了"观感来自口径单一：面板显示的是**草稿提议接受率**。

**修复**：重命名为"草稿接受率 Draft Acceptance (accepted/drafts)"明确口径；**新增 Mean Acceptance Length**（含 bonus 每轮产出，应 ≈1.6 且恒 ≥1）与 **Accepted/Emitted 覆盖率**（回答"投机节省了多少"）。

## 2. 面板修复清单（已实施，vllm-realtime v8）

| # | 面板 | 变更 | 状态 |
|---|------|------|------|
| 1 | KV Cache 水位 (%) | expr × 100 | ✅ 已实施 |
| 2 | 生成吞吐 (tokens/s) | [30s] → [$__rate_interval] | ✅ 已实施 |
| 3 | 投机解码 Acceptance (%) | 重命名 + $__rate_interval | ✅ 已实施 |
| 4 | **ITL 输出间隔 P50/P99/均值 (s)** | 新增（inter_token_latency_seconds histogram） | ✅ 已实施 |
| 5 | **Mean Acceptance Length (含 bonus)** | 新增（官方补充公式） | ✅ 已实施 |
| 6 | **Accepted/Emitted 覆盖率 (%)** | 新增 | ✅ 已实施 |

## 3. 其余 16 面板巡检（全部正确）

| 面板 | 检查项 | 结论 |
|------|--------|------|
| TTFT P99/P50 | histogram_quantile + sum by(le) | ✅ 正确 |
| TPOT P99/P50 | `request_time_per_output_token_seconds` 确认为 **Histogram**（loggers.py L860 `_histogram_cls`），bucket 用法正确 | ✅ 正确 |
| Prompt 吞吐 | 纯 prefill counter | ✅ 正确 |
| 请求成功率/抢占率 | counter rate | ✅ 正确 |
| 请求队列 | Gauge 直显 | ✅ 正确 |
| 前缀缓存命中率 | hits/queries × 100 | ✅ 正确 |
| GPU 温度/功耗/占用 | DCGM 0-100（不 ×100，正确区分） | ✅ 正确 |
| CPU 占用率 | idle rate × 100 | ✅ 正确 |
| 网络速率 | bytes rate / 1048576 → MBs 明确 | ✅ 正确 |
| 显存占用率 [unified] | MemAvailable 换算 × 100 | ✅ 正确 |

**通用规则**（固化）：Counter 用 rate()；Gauge 直显；vLLM 0-1 比值 ×100；DCGM 0-100 不乘；bytes 需换算。

## 4. 验证方法（待 vLLM 下次启动执行，Rex 方案）

1. `curl -s localhost:8001/metrics | grep '^vllm:'` 全量 dump，核对指标名与后缀（dev 版改名风险）
2. KV：抓 Gauge 原始值（应 ∈[0,1]）→ 面板应显示 值×100
3. 吞吐/ITL 一致性：`rate(generation_tokens_total) ≈ num_requests_running × (1/ITL_avg)`，偏差 <20% 即通过
4. 投机：生产落 55-62%；Mean Acceptance Length ≈1.6 恒 ≥1；draft acceptance ≤100%
5. 双面板并行 15-30 分钟目视一致后删旧，截图留存基线

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | vLLM 恢复后按 §4 验证流程核对 6 处修复数值并留存基线截图 | SRE | P1 | 恢复窗口内 |
| 2 | `curl /metrics | grep spec_decode` 确认 dev 版 spec/drafts 指标确切名称与后缀（新增面板 E 可选） | SRE | P2 | 恢复窗口 |
| 3 | 评估是否加 `--enable-kv-cache-metrics`（仅影响 residency 直方图，usage_perc 不依赖） | SRE | P2 | 下次启动 |
| 4 | 投机解码未启用时新增面板显示 No data 属正常，面板备注说明防误报 | SRE | P3 | 随时 |

---

## ⚠️ 待完善 / 已知局限

- 数值验证依赖 vLLM 运行（当前停机配合视频工作流），本次仅完成配置层修复
- vLLM 0.26.1dev 存在指标改名风险，一切以恢复后 /metrics 实测为准
- 面板修改为显示层改动，不影响推理服务（无需重启）
- 备份留存：<MGMT_OCTET> `~/vllm-dashboard-backup-20260807.json`（v7 原始版）

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE）**：vLLM Grafana 面板修复方案（修复清单 + 16 面板巡检 + 验证方法 + 风险）——消息回传全文，已采信实施
- **源码取证**：vLLM 0.26.1dev 镜像内 `/usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py`（L562/671/705/830/860）、`v1/core/kv_cache_manager.py`（L187）、`v1/spec_decode/metrics.py`（L182-196）
- **面板取证**：Grafana vllm-realtime dashboard JSON（19 面板 → 22 面板，version 8）
- **交叉基准**：bench 接受率 60.7/58.9/57.4%（runbook §1.3）、GSM8K 94.5%

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
