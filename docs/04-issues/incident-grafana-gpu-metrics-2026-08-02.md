# Grafana GPU 功耗/占用率指标缺失 — 根因排查与修复报告

**日期**：2026-08-02
**工作流**：事故响应 / 监控数据缺失诊断（工作流 3 变体）
**参与成员**：Rex（SRE 诊断与修复）+ 主理人（独立复检交叉验证）

---

## 📌 TL;DR（执行摘要）

- **根因**：Grafana 面板 "DGX Spark vLLM 实时分析"（uid=`vllm-realtime`）中 GPU 设备功耗与 GPU 占用率两个图表使用了**拼写错误的 PromQL 指标名**（`DCGM_FI_DEV_GPU_POWER_USAGE` / `DCGM_FI_DEV_GPU_GPU_UTIL`，各多了一段 `GPU_`），指标不存在导致查询恒空
- **修复**：已通过 Grafana API 修正为正确指标名 `DCGM_FI_DEV_POWER_USAGE` / `DCGM_FI_DEV_GPU_UTIL`（status=success，未触碰任何生产容器）
- **验证**：面板 JSON 复检 + Grafana 数据源代理查询双机返回非空数据 ✅
- 严重度分布：🔴严重 0 项 / 🟠高 0 项 / 🟡中 0 项 / 🟢低 1 项（SEV4 面板级配置错误）
- 阻塞 / 非阻塞：**非阻塞**（仅监控可视化缺失，服务与告警链路正常）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 已修复并验证 |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（用户硬刷新确认、防复发校验、补 dcgm 告警） |
| 建议下一步 | 用户硬刷新面板确认可视化恢复；将"面板推送后 API 校验"纳入流程 |

---

## 事故时间线（诊断时间线）

| 时间 | 事件 |
|------|------|
| 08-01 ~23:40 | dcgm-exporter 容器启动（双机），Prometheus dcgm job 配置生效，DCGM 指标开始采集（回溯验证：23-24h 前起有数据） |
| 08-02 凌晨 | 实时分析面板迭代（6→15→18 图），GPU 功耗/占用率图表推送时**指标名拼写错误**，查询恒空 |
| 08-02 上午 | 用户报告两指标始终无数据 → 本任务启动 |
| 08-02 03:12 | Rex 全链路排查，定位 PromQL 拼写错误，Grafana API 提交修复 |
| 08-02 03:13+ | 主理人独立复检：全链路（exporter→Prometheus→datasource→面板级 ds/query）验证健康，修复确认落地 |

## 影响范围

- **影响面**：仅 Grafana 面板 2 个图表可视化缺失（GPU 设备功耗 W、GPU 占用率 %）
- **未受影响**：vLLM 服务、Prometheus 采集、告警规则（vllm-dspark.rules 8 条 health=ok）、温度/CPU 等其余 16 图

## SEV 评级

**SEV4（低）** — 监控可视化缺失，无服务影响；面板级配置错误，非基础设施故障。

## 根因（5 Why）

1. **现象**：GPU 功耗/占用率图表无数据，其余图表正常
2. **Why 1**：面板查询返回空结果 → 因为查询的指标名在 Prometheus 中不存在
3. **Why 2**：指标名拼写错误（`DCGM_FI_DEV_GPU_POWER_USAGE` 应为 `DCGM_FI_DEV_POWER_USAGE`；`DCGM_FI_DEV_GPU_GPU_UTIL` 应为 `DCGM_FI_DEV_GPU_UTIL`，各多写一段 `GPU_`）
4. **Why 3**：面板 JSON 推送时手工构造表达式，未对照 Prometheus 实际指标名（`/api/v1/label/__name__/values`）校验
5. **Why 4**：推送流程缺少"推送后自动校验"环节（面板更新后未用 API 拉回确认查询可返回数据）

## 全链路排查证据（Rex + 主理人复检交叉一致）

| 排查方向 | 状态 | 证据 |
|------|------|------|
| Grafana 数据源 | ✅ | uid=`PBFA97CFB590B2093` → `prometheus:9090`（=8191 映射），isDefault；其余 16 图正常 |
| Prometheus 采集 | ✅ | targets：`dcgm-58`/`dcgm-60`/`node-58`/`node-60`/`vllm` 全部 **up**；prometheus.yml 含 node label relabel（node01-60/02-58） |
| exporter 启用/权限 | ✅ | dcgm-exporter 双机 Up 23h；`:9400` 直接输出（UTIL=93%、POWER=13.7W 实测）；GB10 SoC 单逻辑 GPU（"仅 gpu=0"属正常，非异常） |
| 指标存在性 | ✅ | `DCGM_FI_DEV_GPU_UTIL`/`DCGM_FI_DEV_POWER_USAGE`/`DCGM_FI_DEV_GPU_TEMP` 存在，连续 23h+ 数据（2 series = 双机） |
| 指标名与 PromQL | ❌→✅ | **面板原表达式拼写错误**（多写 `GPU_` 段）→ 已修正 |
| 面板配置 | ✅ | timeseries、unit=watt/percent、无 transforms/hidden、legend `{{node}}`（node label 存在） |
| 时间范围/刷新 | ✅ | now-5m + refresh 2s + interval 5s 合理 |

## 行动项（已执行）

| # | 行动 | 状态 | 结果 |
|---|------|------|------|
| 1 | Grafana API 修正功耗表达式 `DCGM_FI_DEV_GPU_POWER_USAGE` → `DCGM_FI_DEV_POWER_USAGE` | ✅ 已执行 | status=success |
| 2 | Grafana API 修正占用率表达式 `DCGM_FI_DEV_GPU_GPU_UTIL` → `DCGM_FI_DEV_GPU_UTIL` | ✅ 已执行 | status=success |
| 3 | 复检面板 JSON 确认表达式已更新 | ✅ 已执行 | 面板现查询正确指标名 |
| 4 | Grafana 代理查询验证（instant + range + 面板级 /api/ds/query） | ✅ 已执行 | 双机 2 帧 × 61 行非空数据 |

## 预防措施

1. **面板推送后校验**：推送 Grafana 面板后立即用 API 拉回，逐一核对 target expr 与 `/api/v1/label/__name__/values` 存在性（脚本化）
2. **告警补强**：为 dcgm-58/60 targets 增加 `up == 0` 告警（Alertmanager 已有 vllm-dspark.rules，追加 2 条），防 exporter 掉线无感知
3. **健康自检命令**（可作 runbook 一步）：
   ```bash
   # 1) targets 健康
   curl -s http://127.0.0.1:8191/api/v1/targets | grep -c '"health":"up"'
   # 2) 指标存在性
   curl -s http://127.0.0.1:8191/api/v1/label/__name__/values | grep -c 'DCGM_FI_DEV_POWER_USAGE'
   # 3) 面板数据
   curl -s -u admin:aicad_grafana_dev 'http://127.0.0.1:3000/api/datasources/proxy/uid/PBFA97CFB590B2093/api/v1/query?query=DCGM_FI_DEV_POWER_USAGE'
   ```

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 用户在浏览器硬刷新（Ctrl+Shift+R）确认功耗/占用率图表恢复 | 用户 | P0 | 立即 |
| 2 | 追加 dcgm targets `up==0` 告警到 vllm-dspark.rules | Rex | P1 | 1 周内 |
| 3 | 面板推送流程增加 API 校验步骤（expr 与指标名核对） | Docu | P2 | 下轮面板迭代 |

## ⚠️ 待完善 / 已知局限

- 修复已由 Rex 执行并经主理人独立复检，但**用户端浏览器缓存**可能仍显示旧渲染，需硬刷新确认
- 温度面板（`DCGM_FI_DEV_GPU_TEMP`）拼写本正确故一直正常，佐证了"仅两图拼写错误"的定位
- 历史空窗期数据无法补（指标本身在，只是查询错误），无需补采

## 📚 数据来源 & 成员产出索引

- Rex（SRE）原始产出：全链路排查表 + 根因定位（PromQL 拼写错误）+ 已执行修复 + 验证结果（经 SendMessage 回传）
- 主理人独立复检：targets 状态、指标存在性与历史回溯（23-24h 前起有数据）、datasource 配置、面板 JSON 全量 dump、Grafana 代理 instant/range/面板级 ds/query 三重验证、集群面板对照排查

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
