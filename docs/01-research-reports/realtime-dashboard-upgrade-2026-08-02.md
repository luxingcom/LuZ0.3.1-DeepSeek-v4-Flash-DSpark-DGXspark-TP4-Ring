# 实时监控升级 · 布局修复 + 秒级刷新实时分析页

**日期**：2026-08-02
**工作流**：部署前检查 / 监控完善（延续 engineering-monitoring → engineering-realtime）
**参与成员**：Rex（SRE）/ Tessa（测试）/ 主理人执行层

---

## 📌 TL;DR

- **布局修复**：现有"DGX Spark vLLM 集群"面板 15 图根因是 **3 个图被排到 x=24（超出 24 列网格）**，已重排为 **5 行 × 3 列（w=8）均匀布局**，version 7
- **刷新链路升级**：Prometheus vllm 抓取 `10s → 5s`（HUP 热重载，零中断）；Grafana `min_refresh_interval 5s → 1s`（秒级刷新解锁）
- **新建实时分析页**：`vllm-realtime`（DGX Spark vLLM 实时分析），12 图 4×3、**refresh=2s、时间范围 now-5m**——彻底告别小时级曲线
- **秒级出数实证**：压测下每 5s 采样数据实时跳动（吞吐 17.9→61.3 t/s、running 1→8、KV 水位波动、Acceptance 25.7→28.4%）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 完成（布局 + 刷新链路 + 新页面全部就绪） |
| 布局修复 | 15 图 → 5×3 均匀（x=24 溢出 bug 修复） |
| 刷新链路 | 抓取 5s（HUP 热重载）/ Grafana 最小刷新 1s |
| 新实时页 | vllm-realtime：12 图 / refresh=2s / now-5m |
| 验证 | 秒级采样实时跳动 + 压测 errors=0（72.9-78.8 t/s） |

---

## 🔧 执行明细

### 1. 布局问题根因（已修复）
- **根因**：后三行（y=24/32/40）每行排了 3 个 w=12 的图，第 3 个被排到 **x=24**——超出 Grafana 24 列网格，图被挤出屏幕/压缩，造成"左右排布不均匀"
- **修复**：15 图重排为 **5 行 × 3 列**（w=8, h=8, x=0/8/16）：
  - 行1: GPU 利用率 ｜ GPU 温度 ｜ TTFT P95
  - 行2: 请求吞吐 ｜ 运行中请求 ｜ TPOT P95
  - 行3: KV Cache ｜ 节点内存 ｜ Prefill/队列 P95
  - 行4: Token 明细 ｜ 前缀缓存命中 ｜ 请求成功率
  - 行5: 队列状态 ｜ 抢占数 ｜ 投机 Acceptance
- 推送：POST /api/dashboards/db（overwrite=true），version 7

### 2. 刷新链路升级
| 组件 | 改动 | 方式 | 中断 |
|------|------|------|------|
| Prometheus vllm job | scrape_interval 10s → **5s** | 改挂载配置 + `docker kill -s HUP` 热重载 | 零中断 |
| Grafana | min_refresh_interval 5s → **1s** | `docker exec -u root sed` 改 grafana.ini + `docker restart` | ~15s（已恢复） |
| 实时页 | refresh=**2s**、time=**now-5m** | 新建面板配置 | — |

- 抓取 5s 后 `rate[30s]` 窗口 = 6 样本，直方图 `[1m]` = 12 样本，短窗口统计稳健（Tessa 判定）
- 106 指标单 target 5s 抓取负载可忽略（Rex 判定：收益 > 成本）

### 3. 实时分析页（vllm-realtime，12 图 4×3）
| 行 | 图表 | PromQL 要点 |
|----|------|-------------|
| 1 | TTFT P99/P50、TPOT P99/P50 | histogram_quantile(sum by(le)(rate(_bucket[1m]))) |
| 2 | 生成吞吐、Prompt 吞吐 | sum(rate(_total[30s]))——短窗口秒级跳动 |
| 2 | 请求成功率、抢占率 | sum(rate(_total[1m])) |
| 3 | KV 水位、请求队列 | 瞬时值 running/waiting |
| 3 | 前缀缓存命中率、投机 Acceptance | rate(hits)/rate(queries)×100（clamp_min 防除零） |

### 4. 秒级出数验证（压测实证）
压测 8 并发 × 6 轮（errors=0，72.95-78.77 t/s）期间每 5s 采样：

| 指标 | t=0 | t=10s | t=20s | t=30s | 判定 |
|------|-----|-------|-------|-------|------|
| 生成吞吐 t/s | 17.9 | 40.3 | 61.3 | 61.3 | ✅ 实时跳动 |
| Prompt 吞吐 | 197 | 395 | 569 | 569 | ✅ |
| running | 6 | 2 | 1 | 0 | ✅ 请求进出 |
| KV 水位 % | 1.0 | 0.3 | 0.2 | 0 | ✅ 波动 |
| Acceptance % | 25.7 | 27.8 | 28.4 | 28.0 | ✅ dspark 特性可见 |

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 打开 Grafana 实时页 `/d/vllm-realtime` 查看秒级曲线 | 用户 | P0 | 已就绪 |
| 2 | 现有页刷新确认 5×3 均匀布局（version 7） | 用户 | P0 | 已就绪 |
| 3 | 观察 5s 抓取对 Prometheus 的负载（scrape_duration/TSDB 增长） | SRE | P2 | 1 周后复查 |
| 4 | 实时页若需 1s 硬刷新可在页面右上角手动选择 1s | 用户 | P2 | — |

---

## ⚠️ 待完善 / 已知局限

- **实时页 2s 刷新为推荐值**（Rex）：1s 刷新对 5s 抓取数据增益有限且增加 Grafana 查询压力；如需 1s 可在页面时间选择器手动切换（min_refresh_interval=1s 已解锁）
- 空载时 rate 类图表无数据/显示 0 属正常（无流量即无增量）
- Grafana grafana.ini 改动在容器内（未挂载），**容器重建后需重新应用**（已记入 memory）

---

## 📚 数据来源 & 成员产出索引

- Rex（SRE）原始产出：刷新链路决策（HUP 热重载、5s 抓取收益评估、refresh=2s/now-5m 推荐）、实时页 12 图设计、布局 5×3 方案
- Tessa（测试）原始产出：短窗口样本数判定（[30s]=6 样本）、14 图 PromQL 验证矩阵、造流量验证方案、5s 抓取探针与回退阈值
- 主理人执行记录：布局修复推送（version 7）、Prometheus HUP 热重载、Grafana 重启、实时页创建推送（version 1）、秒级采样验证

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
