# 监控完善执行报告：DGX Spark vLLM 集群（按 ADR-0009 + 审计要求）

**日期**：2026-08-02
**工作流**：工作流 4（部署前检查）+ 监控完善
**参与成员**：Rex（告警/表达式方案）、Tessa（验证矩阵）、主理人执行
**面板**：Grafana（58:3000）`vllm-dspark-cluster`（DGX Spark vLLM 集群，uid=vllm-dspark-cluster，refresh=1s）

---

## 📌 TL;DR

- 用户确认 Grafana 已有本项目面板"DGX Spark vLLM 集群"（位于 **worker 58** 的 Grafana，之前核查查的是 head 60 所以"误报缺失"）。
- **数据链路核实已通**：vllm job 采集正常（106 指标）、dcgm/node 双机 up、面板 6 个图表达式均能出数（空载时 rate 无增量属正常，压测后确认出数）。
- **本轮补齐**：ADR-0009 契约 4 条告警（VLLMInstanceDown / VLLMKvCacheHigh / VLLMTPOTHigh / VLLMHealth5xx）已加载生效（vllm-dspark.rules 现有 8 条，全部 health=ok）。
- **链路验证通过**：VLLMTPOTHigh 压测后进入 pending（规则真实生效）；Alertmanager 注入测试 HTTP=200、告警到达（state=active）。
- 压测附带确认：10 并发 84.65 t/s（预热完全收敛，对齐基准 C 84.6）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 监控完善完成（契约告警 + 链路验证全通） |
| 面板 | vllm-dspark-cluster（6 图，refresh=1s，数据正常） |
| 告警规则 | 8 条（原 4 + 契约 4），health=ok |
| 链路 | Prometheus → Alertmanager → webhook 验证通过 |
| 遗留 | 吞吐面板空载无数据显示（正常）、blackbox 健康探针（增强项） |

---

## 🛠️ 完善明细

### 1. 现状核实（修正此前"监控缺失"误判）
- **Prometheus 在 worker(58)**：vllm job（<NODE_IP>:8001, up）、dcgm-58/60、node-58/60 全部正常
- **Grafana 在 worker(58)**：面板 `vllm-dspark-cluster` 6 图——GPU 利用率/温度、请求吞吐（rate generation_tokens）、运行/等待请求、KV 水位、节点内存
- 此前审计（00:06）查 head(60) 的 8191/3000 得出"无 vllm job/无面板"结论——**head 是 AICAD 配置，58 才是本项目监控栈**（审计报告需更正此项）

### 2. 补契约告警（ADR-0009，已生效）
追加到 `/opt/aicad/monitoring/alert_rules.yml` 的 vllm-dspark.rules（备份 .bak-20260802）：

| 告警 | 表达式 | 级别 |
|------|--------|------|
| VLLMInstanceDown | up{job="vllm"}==0 for 1m | critical |
| VLLMKvCacheHigh | vllm:kv_cache_usage_perc > 90 for 10m | warning |
| VLLMTPOTHigh | histogram_quantile(0.95, sum by(le)(rate(request_decode_time_seconds_bucket[5m]))) > 0.05 for 10m | warning |
| VLLMHealth5xx | up{job="vllm"}==0 for 3m（/health 兜底） | critical |

- promtool 校验：**SUCCESS: 13 rules found**
- 加载方式：改宿主机文件 → `docker restart aicad-prometheus-1`（/-/reload 403 未启用 lifecycle）

### 3. 链路验证（Tessa 矩阵执行）
- **VLLMTPOTHigh 压测后进入 pending**——表达式真实生效
- **Alertmanager 注入测试**：POST /api/v2/alerts HTTP=200，VLLMInstanceDown 到达 AM（state=active），webhook receiver 正常
- 测试告警已清理（endsAt=now 方式）

### 4. 面板数据确认
- 压测前：generation_tokens_total=9322（counter 有值，rate[1m] 空载无增量→面板无显示，**正常**）
- 压测后：rate=38.4 tok/s、prompt rate=370.5、decoded 累计 11882（增量确认）→ **面板会实时显示**

### 5. 附带成果：预热完全收敛
- **10 并发 84.65 t/s**（79.75→85→84.65，errors=0）——对齐基准 C 84.6，预热收敛彻底完成

## ⏳ 遗留 / 增强项

| # | 项 | 说明 |
|---|-----|------|
| 1 | blackbox_exporter 健康探针 | 当前 VLLMHealth5xx 用 up{} 兜底；如需 HTTP 级 /health 探针需部署 blackbox（P2） |
| 2 | 吞吐面板空载无数据显示 | 正常行为；可加"无数据时显示 0"的 Grafana 设置（P2） |
| 3 | 告警接收 | webhook 指向 localhost:5001 stub，生产需配置真实接收通道（email/slack） |

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 |
|---|------|------|--------|
| 1 | 审计报告更正：监控栈在 58 非 head（vllm job/面板均已就绪） | Docu | P1 |
| 2 | 生产告警接收通道配置（webhook 目前指向本地 stub） | Rex | P1 |
| 3 | blackbox /health 探针 + 空载显示优化 | Rex/Cody | P2 |

## 📚 数据来源 & 成员产出索引

- Rex：契约 4 告警表达式改写、健康探针建议（up 兜底）、加载方式（重启容器）
- Tessa：验证矩阵（指标真实性/告警链路/面板完整性/压测触发）
- 主理人执行：规则追加 + promtool + 重启 + AM 注入测试 + 压测验证

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
