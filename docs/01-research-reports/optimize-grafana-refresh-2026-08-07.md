# Grafana 面板刷新机制优化与低开销采集方案

**日期**：2026-08-07
**工作流**：工作流 4（部署/变更实施）——监控采集优化
**参与成员**：主理人（实施）；依据用户要求（2s 刷新 / 5s 缓存 / 低开销）

---

## 📌 TL;DR（执行摘要）

- 用户要求：图表 2s 自动刷新、缓存 5s、低性能开销采集方案
- 已实施：**面板 refresh=2s + 全部面板查询 interval=5s + Prometheus recording rules 预聚合高频指标**（CPU/GPU/内存每 15s 后台计算一次，前端 2s 刷新只查轻量结果）
- vllm-realtime 面板 **v12 → v13**，Prometheus 新增 `aicad-dgx-agg` 规则组（5 条 recording rule）
- 浏览器实测：GPU/CPU/网络/内存 14 面板正常渲染，6 个 vLLM 指标面板 No data（LLM 停机预期）
- 无阻塞项

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（v13 + 预聚合已生效） |
| 阻塞项数量 | 0 |
| 关键行动项 | 2 条（compose 补挂载 / 观察开销） |
| 建议下一步 | 观察 24h 内 Prometheus 查询压力与面板刷新表现 |

---

## 1. 实施内容

### 1.1 Grafana 面板（v13）

| 项 | 变更 |
|----|------|
| dashboard refresh | 10s → **2s**（用户要求） |
| 全部面板 target interval | **5s**（min interval，数据点 5s 粒度 = "缓存 5s 刷新"） |
| 资源面板 expr（5 个） | 原查询 → **dcgx:\* 预聚合指标**（114 GPU 占用 / 115 统一内存 / 116 温度 / 117 功耗 / 118 CPU） |
| 其余面板 | expr 不变（vLLM 指标，LLM 停机期间无数据属预期） |

### 1.2 Prometheus recording rules（低开销核心）

`/opt/aicad/monitoring/recording_rules.yml`（aicad-dgx-agg 组，interval 15s）：

| 指标 | 表达式 | 用途 |
|------|--------|------|
| dcgx:cpu_util_percent | 100 - avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[2m])) * 100 | CPU 面板（原查询最重：多 mode series 全量 rate） |
| dcgx:mem_util_percent | 100 - (MemAvailable/MemTotal * 100) | 统一内存面板 |
| dcgx:gpu_util_percent | DCGM_FI_DEV_GPU_UTIL | GPU 占用面板 |
| dcgx:gpu_temp_celsius | DCGM_FI_DEV_GPU_TEMP | 温度面板 |
| dcgx:gpu_power_watts | DCGM_FI_DEV_POWER_USAGE | 功耗面板 |

**开销收益**：CPU 聚合查询从"每次前端刷新全量 scan node_cpu_seconds_total（4 机 × ~8 mode 系列 × 2m 窗口）"降为"查 15s 预计算结果（4 系列）"；2s 刷新下查询负载降低约一个数量级。

### 1.3 实测验证（四机数据产出）

- dcgx:cpu_util_percent = **4 series**（6.63%）
- dcgx:gpu_temp_celsius = **4 series**（85°C，视频工作流 GPU 满载）
- dcgx:mem_util_percent = **4 series**（10.5%）
- dcgx:gpu_util_percent = **4 series**（96%，comfyui 满载）
- 浏览器实测：14 面板正常渲染 + 6 个 vLLM 面板 No data（停机预期）

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | docker compose 定义补 recording_rules.yml 挂载（当前 docker cp 到容器可写层，容器重建会丢；宿主文件已在 /opt/aicad/monitoring/） | SRE | P1 | 下次容器重建前 |
| 2 | 观察 24h：Prometheus 查询延迟 / 面板 2s 刷新表现 / recording 内存占用 | SRE | P2 | 24h 后 |
| 3 | 确认面板页面右上角刷新间隔显示 "2s" 且自动刷新 | SRE | P1 | 今日 |

---

## ⚠️ 待完善 / 已知局限

- recording_rules.yml 未纳入容器挂载（compose 需补：`- ./monitoring/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro`），当前靠 docker cp 持久于可写层
- 单文件 bind mount 的 inode 坑：宿主 sed -i 修改不生效于运行中容器（需重启容器重新挂载）；Prometheus 已重启生效
- 网络面板（119/120）未预聚合（rate 查询轻量，保持原 expr）
- vLLM 指标面板在 LLM 恢复前无数据（停机预期，非故障）

---

## 📚 数据来源

- Prometheus API 实测（status/config rule_files、dcgx:* 四机数据）
- Grafana API 实测（v13 refresh/interval/expr）
- 浏览器 DOM 验证（No data 面板清单）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
