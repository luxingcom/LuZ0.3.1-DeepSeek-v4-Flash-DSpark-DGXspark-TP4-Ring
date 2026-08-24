# Grafana 面板扩展报告：DGX Spark vLLM 集群（6 → 15 图）

**日期**：2026-08-02
**工作流**：工作流 4（部署前检查）+ 监控完善
**参与成员**：Rex（面板设计，回传中）、主理人执行
**面板**：Grafana（58:3000）`vllm-dspark-cluster`，version 6

---

## 📌 TL;DR

- 用户反馈面板"内容没有增加，还是只有 6 个面板"——根因：上轮只补了 Prometheus 告警规则，未往 Grafana 面板加图表。
- **本轮完成**：面板从 6 图扩展至 **15 图**（新增 9 个），覆盖 ADR-0009 契约的延迟/吞吐/缓存/队列/投机/成功率维度。
- **新增图全部验证出数**（压测实证）：TTFT P95=2.42s、TPOT P95=70.8ms、Acceptance=31.1%、drafts/s=0.86 等。
- 通过 Grafana Dashboard API 更新（POST /api/dashboards/db，overwrite=true），version 4→6。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 面板扩展完成（15 图全出数） |
| 面板变更 | 6 → 15 图（新增 9，去重 1） |
| 数据验证 | 16 表达式全通过；压测后新图真实出数 |
| 更新方式 | Grafana API（HTTP 200 success） |
| 遗留 | 空载时 rate 类图无数据（正常） |

---

## 🛠️ 扩展明细

### 原 6 图（保留）
GPU 利用率 / GPU 温度 / vLLM 请求吞吐 / 运行中请求数 / KV Cache 使用率 / 节点内存可用

### 新增 9 图（ADR-0009 契约 + vLLM 0.11 实际指标）

| 图 | 表达式 | 维度 |
|----|--------|------|
| TTFT P95（首 token 延迟） | histogram_quantile(0.95, sum by(le)(rate(time_to_first_token_seconds_bucket[5m]))) | 延迟 |
| TPOT P95（每 token 输出延迟） | histogram_quantile(0.95, sum by(le)(rate(request_time_per_output_token_seconds_bucket[5m]))) | 延迟 |
| Prefill / 队列延迟 P95 | prefill_time + queue_time 双曲线 | 延迟 |
| Token 吞吐明细 | gen / prompt / cached 三速率 | 吞吐 |
| 前缀缓存命中率 | hits/queries × 100 | 缓存 |
| 请求队列状态 | running / waiting / waiting by reason | 队列 |
| 抢占数 | rate(num_preemptions_total) | 队列 |
| 投机解码 Acceptance | accepted/draft × 100 + drafts/s | **dspark 核心** |
| 请求成功率 | increase(request_success_total) | 质量 |

### 压测验证数据（5 并发 512in/64out）
| 指标 | 实测值 |
|------|--------|
| TTFT P95 | 2.42s |
| TPOT P95 | 70.8ms |
| Prefill P95 | 1.47s |
| Queue P95 | 285ms |
| Acceptance | **31.1%** |
| drafts/s | 0.86 |
| gen rate | 12.68 tok/s |

## ⚠️ 注意事项

- 空载时 rate/histogram 类图无数据（NaN/0）属正常，有流量即显示
- 前缀缓存命中率空载为 0（无缓存查询）
- 面板 JSON 已备份至本工作空间可追溯（version 6）

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 |
|---|------|------|--------|
| 1 | 刷新 Grafana 页面查看 15 图 | 用户 | 立即 |
| 2 | 如有图显示异常（红/无数据），反馈定位 | 用户 + 主理人 | P1 |
| 3 | 面板 JSON 导出回写仓库（uid 模板化） | Cody | P2 |

## 📚 数据来源

- 主理人执行：指标全名采集（106 个）、表达式验证（16 个）、面板生成/推送/验证、压测出数
- Rex：面板扩展设计（回传中，与本执行交叉校验）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
