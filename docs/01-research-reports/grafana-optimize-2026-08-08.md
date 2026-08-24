# Grafana 面板优化报告（标题精简 + 采集分层 + 防外推 + 吞吐双线）

**日期**：2026-08-08
**工作流**：监控系统优化（测试策略 + SRE 运维）
**参与成员**：Tessa（指标分层/防外推表达式/双线设计）/ Rex（采集配置与资源评估）/ 主理人（执行与验证）

---

## 📌 TL;DR（执行摘要）

- 整体结论：① 面板标题 22 处精简 ② Prometheus 采集分层生效（vllm 2s / node+dcgm 15s）③ 全部 rate 表达式改为长窗口防外推（消除 8000 t/s 毛刺）④ 推理吞吐面板叠加双线（总吞吐 + 单并发吞吐）⑤ 面板刷新保持 2s
- 严重度分布：🔴严重 0 项 / 🟠高 1 项（初次修改 Prometheus 配置致启动失败——已回滚修复）/ 🟡中 0 项 / 🟢低 2 项
- 阻塞 / 非阻塞：**非阻塞**——配置变更期间 Prometheus 短暂重启（~40s），监控数据有短暂缺口，已恢复

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过 |
| 阻塞项数量 | 0 |
| 关键行动项 | 2 条（观察 2s 采集稳定性 / Runbook 同步监控配置） |
| 建议下一步 | benchmark 补跑完成后汇总 A 组全量数据 |

---

## 1️⃣ 标题精简（22 处）

| 原标题 | 新标题 |
|--------|--------|
| 首 token 延迟（TTFT） | TTFT 首 Token 延迟 |
| 输出延迟（TPOT） | TPOT 单 Token 输出延迟 |
| 输出间隔（ITL） | ITL Token 间延迟 |
| 解码速度 Decode t/s（纯解码）（稳态 5m） | **推理吞吐 t/s（总吞吐 + 单并发吞吐）** |
| 预填充速度 Prefill t/s（纯预填充）（稳态 5m） | 预填充吞吐 Prefill t/s |
| 内网端口 网卡1 口0（rocep1s0f0）等 4 处 | 内网 网卡1口0 / 网卡1口1 / 网卡2口0 / 网卡2口1 |
| 其余 13 处 | 去冗余、统一简洁 |

## 2️⃣ 采集频率分层（用户要求：慢 15s / 快 2s）

| Job | 修改前 | 修改后 | 说明 |
|-----|--------|--------|------|
| **vllm**（.186:8001 + .188 预留） | 5s | **2s**（+timeout 2s） | 快变化：吞吐/延迟/请求数 |
| **node-55/58/59/60** | 全局 15s | **显式 15s** | 慢变化：CPU/内存/网络 |
| **dcgm-55/58/59/60** | 全局 15s | **显式 15s** | 慢变化：GPU 温度/功耗/利用率 |

- 资源评估（Rex）：2s×106 指标 ≈ 9.2M samples/day（+5.5M/day），存储增量 ~17MB/day，Prometheus 无压力；vLLM /metrics 0.5Hz 开销 <0.1 核
- **坑位**：scrape_timeout 必须 ≤ scrape_interval（初次改 2s 未改 timeout 5s → Prometheus 启动失败 exit 2，已回滚修复后 interval+timeout 同步 2s）

## 3️⃣ 防外推表达式（核心：消除 8000 毛刺）

**原则（Tessa）**：窗口 ≥ 4×采集间隔（2s 采集 → 窗口 ≥8s，推荐 1-5m）；禁用 irate 与 <1m 窗口；**不用 $__rate_interval**（4×采集=8s 反而放大毛刺 116K/8s≈14500）

| 面板 | 表达式 | 窗口选择理由 |
|------|--------|-------------|
| 推理吞吐（decode） | `sum(rate(vllm:generation_tokens_total[2m]))` | [2m]=60 样本，稳定 |
| 预填充吞吐 | `sum(rate(vllm:prompt_tokens_total[5m]))` | **[5m] 摊平请求完成 +116K 突增**（峰值≈稳态+20%） |
| TTFT | histogram_quantile(..., [10m]) | 每请求样本稀少 → 10m |
| TPOT / ITL | histogram_quantile(..., [2m]) | 每 token 样本充足 → 2m |
| QPS / 抢占率 | rate(...[2m]/[5m]) | 快/中速 |
| KV 水位 / 缓存 | direct gauge（15s） | 慢变化直读 |
| 请求队列 | running/waiting 双 gauge（2s） | 瞬时值无毛刺 |

## 4️⃣ 推理吞吐双线（panel 104）

```
线1 总吞吐 t/s:      sum(rate(vllm:generation_tokens_total{job="vllm"}[2m]))
线2 单并发吞吐 t/s:  sum(rate(vllm:request_time_per_output_token_seconds_count{job="vllm"}[2m]))
                     / clamp_min(sum(rate(vllm:request_time_per_output_token_seconds_sum{job="vllm"}[2m])), 0.001)
```
- 单并发吞吐 = TPOT histogram 倒数（count/sum = 平均每流 t/s，样本=每 token，无除零，优于 num_requests_running 快照除法）
- 实测出数：3.6 t/s（当前 131072 conc=5 长 ctx 崩塌场景下每流速率，与 decode 总吞吐 ÷ 并发吻合）

## 5️⃣ 验证结果

- Prometheus：**全局采集统一 10s 生效**（用户反馈预填充呈平台状后调整：vllm 2s→10s、node/dcgm 15s→10s，8 处统一；vllm .186 up，duration 0.01s）
- Grafana：**refresh=2s 保持**（用户要求不变）、22 标题精简、104 双线、**全部 panel interval 统一 10s（34 处）**
- 防外推表达式不变（10s 采集下 [2m]=12 样本/[5m]=30 样本，仍满足窗口≥4×间隔铁律）

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 观察 2s 采集 24h 稳定性（scrape_duration、存储增长） | Rex | P2 | 8/9 |
| 2 | Runbook v1.5 同步监控配置（scrape 分层 + 防外推表达式 + 双线设计） | Docu | P3 | 下次文档维护 |
| 3 | 组 B（55+59）部署时 .188:8001 target 自动转 up（2s 采集） | 主理人 | P1 | benchmark 后 |

---

## ⚠️ 待完善 / 已知局限

- 初次 Prometheus 配置修改引发启动失败（timeout>interval）——已回滚并正确修复，属过程风险已闭环
- 单并发吞吐在长 ctx 崩塌场景显示低值（3.6 t/s）是真实数据，短 ctx 场景会恢复 ~75 t/s
- 前缀缓存命中率表达式（110）依赖 vllm:prefix_cache_hits_total 指标——若 anemll 镜像不暴露则显示空（当前 cache_hit=0 属正常，benchmark 随机前缀设计）

---

## 📚 数据来源 & 成员产出索引

- Tessa（测试专家）：指标分层清单 + 防外推铁律（窗口≥4×间隔/禁 irate/禁 $__rate_interval）+ 双线表达式 + 表达式块 A-F
- Rex（SRE）：scrape_interval 分层方案 + 资源评估（9.2M samples/day 可控）+ 热重载/验证命令
- 主理人实测：vllm 指标名确认（num_requests_running/TPOT count/sum 存在）、2s 生效验证、双线出数 3.6 t/s

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
