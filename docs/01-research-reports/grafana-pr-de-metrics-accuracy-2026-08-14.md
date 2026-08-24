# Grafana 面板 PR/DE 实时曲线失真调查与准确测量方案

**日期**：2026-08-14
**问题**：Grafana 面板中 prefill（PR）/ decode（DE）实时曲线严重不符合实际
**结论**：根因是 PromQL 窗口选择错误（`rate()[30s]` 太短）+ histogram 短窗口在长请求下无有效样本，非数据采集缺失

---

## 一、调查过程与关键发现

### 1.1 当前版本（0.2.1-v026.0）指标清单核实

已实测 `8001/metrics` 完整指标清单，**关键发现：旧版 vLLM 的 `avg_prompt_throughput_toks_per_s` / `avg_generation_throughput_toks_per_s` 两个指标在当前版本已不存在**（grep `throughput` 零匹配）。

当前可用的 PR/DE 相关指标（实测）：

| 类型 | 指标 | 语义 |
|------|------|------|
| Counter | `vllm:prompt_tokens_total` | 累计 prefill token（当前 8.6M） |
| Counter | `vllm:generation_tokens_total` | 累计生成 token（当前 137775） |
| Counter | `vllm:spec_decode_num_draft_tokens_total` | 投机草稿 token（124631） |
| Counter | `vllm:spec_decode_num_accepted_tokens_total` | 投机接受 token（99240） |
| Histogram | `vllm:time_to_first_token_seconds` | TTFT |
| Histogram | `vllm:request_time_per_output_token_seconds` | TPOT |
| Histogram | `vllm:inter_token_latency_seconds` | ITL |
| Histogram | `vllm:request_prefill_time_seconds` / `request_decode_time_seconds` | prefill/decode 分段耗时 |
| Histogram | `vllm:request_prompt_tokens` / `request_generation_tokens` | 每请求 token 数 |
| Gauge | `vllm:num_requests_running` / `num_requests_waiting` | 运行/等待请求数 |

### 1.2 当前面板实际 PromQL（vllm-realtime.json 实测）

| 面板 | 当前查询 |
|------|---------|
| 预填充吞吐 Prefill t/s | `sum by (node)(rate(vllm:prompt_tokens_total[30s]))` |
| 推理吞吐（decode）t/s | `sum by (node)(rate(vllm:generation_tokens_total[30s]))` |
| TTFT | `histogram_quantile(0.5, rate(vllm:time_to_first_token_seconds_bucket[30s]))` |
| TPOT | `histogram_quantile(0.5, rate(vllm:request_time_per_output_token_seconds_bucket[30s]))` |
| ITL | `histogram_quantile(0.5, rate(vllm:inter_token_latency_seconds_bucket[30s]))` |
| 投机接受率 | `rate(accepted[30s]) / rate(draft[30s])` |

### 1.3 采集链路核实

- Prometheus `scrape_interval: 5s`、`scrape_timeout: 2s`，仅 scrape `<NODE_IP>:8001`（head/rank0）
- **实测验证**：worker rank（02/04/03）不暴露 `/metrics`（TP4 下 worker 走 `--headless`，无 HTTP），head 的 token 计数是**全局累计**（prompt 8.6M、generation 137775）
- **结论**：scrape 单点（head）是正确的，不存在"少算 4 倍"问题

---

## 二、根因分析（4 条，按影响排序）

### 根因 1（核心）：`rate()[30s]` 窗口太短，prefill/decode 交替导致锯齿状失真

scrape 5s → `rate[30s]` 仅 6 个采样点。而 **prefill 是突发性的**：

- 一个 131072 token 请求：prefill 持续 ~65 秒（2013 tok/s），随后转入 decode ~18 秒（110 tok/s）
- prefill 期间 `rate(prompt_tokens_total[30s])` ≈ 2013；prefill 结束 30 秒内跌到 0
- decode 同理：decode 期间 generation rate ≈ 110，prefill 期间 = 0

**结果**：两条吞吐曲线呈"脉冲/锯齿"状——峰值对得上（2000/110），但大部分时间在 0 与峰值间剧烈跳动。用户看到"一会 2000 一会 0"，即判断"严重不符合实际"。

### 根因 2：TTFT/TPOT/ITL 的 histogram 用 [30s] 窗口，长请求下无有效样本

`histogram_quantile(0.5, rate(..._bucket[30s]))` 统计的是"**30 秒内完成的请求**"的 p50。但 131072 prompt 的 TTFT 就 >65 秒，30 秒窗口根本覆盖不到完整请求 → 曲线要么 No data，要么只捕捉到极少数短 prompt（512/8192），严重失真。

### 根因 3：投机解码的吞吐口径（面板用对了，但需理解）

- `generation_tokens_total` = 137775（accepted/实际输出）
- draft token = 124631，accepted = 99240
- 面板用 `generation_tokens_total`（= 用户感知输出吞吐）是**正确的**，不是错误根因
- 但投机解码下"引擎计算吞吐"（含 draft）≠ "输出吞吐"（accepted），若将来有人改用 draft token 会虚高

### 根因 4（次要）：监控配置滞后于架构演进

- Prometheus 注释仍写 "TP=2"、"node-58/60"（08-01 旧配置），未更新到 TP4 四机
- 面板 `sum by (node)` 实际只有 1 个 node（node01），但 head 是全局的，不影响准确性，仅属陈旧

---

## 三、准确测量方案

### 3.1 核心原则

**区分"瞬时速率"与"平均吞吐"，用长窗口平滑**：
- `rate[30s]` = 瞬时速率（捕捉突发，锯齿状）
- `rate[1m~5m]` = 平滑平均（稳定，符合"相对准确"预期）

### 3.2 修正后的 PromQL（直接替换面板查询）

| 面板 | 修正后查询 |
|------|-----------|
| **Prefill 吞吐** | `sum(increase(vllm:prompt_tokens_total[5m])) / 300` |
| **Decode 吞吐** | `sum(rate(vllm:generation_tokens_total[5m]))` |
| **TTFT** | `histogram_quantile(0.5, sum by (le)(rate(vllm:time_to_first_token_seconds_bucket[5m])))` |
| **TPOT** | `histogram_quantile(0.5, sum by (le)(rate(vllm:request_time_per_output_token_seconds_bucket[5m])))` |
| **ITL** | `histogram_quantile(0.5, sum by (le)(rate(vllm:inter_token_latency_seconds_bucket[5m])))` |
| **投机接受率** | `sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1)` |

> 若要"瞬时"观感与"平均"兼顾，可加一组 `[1m]` 面板并列展示，明确标注"1 分钟滚动"。

### 3.3 可选：更精确的 prefill/decode 分离（用直方图 _sum）

```promql
# prefill 精确吞吐（通过 request_prompt_tokens 直方图 _sum 的 rate）
sum(rate(vllm:request_prompt_tokens_sum[5m]))
# decode 精确吞吐
sum(rate(vllm:request_generation_tokens_sum[5m]))
# prefill/decode 分段耗时 p50（长请求也能统计到）
histogram_quantile(0.5, sum by (le)(rate(vllm:request_prefill_time_seconds_bucket[5m])))
histogram_quantile(0.5, sum by (le)(rate(vllm:request_decode_time_seconds_bucket[5m])))
```

### 3.4 建议：加 recording rule 预聚合（提升查询稳定性与性能）

在 `recording_rules.yml` 追加（当前只有 DCGM 预聚合）：

```yaml
- record: vllm:prefill_tokens_rate_5m
  expr: sum(rate(vllm:prompt_tokens_total[5m]))
- record: vllm:decode_tokens_rate_5m
  expr: sum(rate(vllm:generation_tokens_total[5m]))
- record: vllm:spec_accept_rate_5m
  expr: sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1)
```

---

## 四、落地建议（按优先级）

| # | 动作 | 影响 | 风险 |
|---|------|------|------|
| P0 | 将吞吐面板 `[30s]` → `[5m]`（或增加 `[1m]` 并列） | 消除锯齿，曲线贴合实际平均吞吐 | 无 |
| P0 | 将 TTFT/TPOT/ITL 的 histogram `[30s]` → `[5m]` | 长请求能被统计到，消除 No data/失真 | 无 |
| P1 | 加 recording rule 预聚合（3 条） | 查询更稳定、面板加载更快 | 无 |
| P2 | 更新 scrape 配置注释（TP2→TP4）、收敛双 Grafana | 配置一致性 | 低 |

**关键提醒**：修正后曲线是"5 分钟平滑平均"，会有一段滞后（约 2.5 分钟），这是换取"准确稳定"的代价。若要看"瞬时脉冲"，应单独加 `[1m]` 面板并明确标注，而不是用 `[30s]` 当"实时"看。

---

## 五、一句话结论

曲线"严重不符合实际"的根因**不是数据采集缺失**（head 的 counter 计数是全局准确的），而是 **PromQL 的 `rate()[30s]` 窗口太短**：prefill/decode 交替突发时，30 秒窗口只能捕捉到突发的"开始或结束"，无法反映平均吞吐；TTFT/TPOT 的 histogram 同样因 30 秒窗口覆盖不了 65 秒+ 的长 prefill 而失真。**修正 = 把吞吐与延迟查询的窗口统一放宽到 `[5m]`，并加 recording rule 预聚合。**
