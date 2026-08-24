# Grafana 面板曲线机器名标注（node0X~04）

**日期**：2026-08-07
**工作流**：工作流 4（部署/变更实施）——面板可读性优化
**参与成员**：主理人（实施）

---

## 📌 TL;DR（执行摘要）

- 为 Grafana vllm-realtime 面板全部曲线标注机器名（node0X~04）
- 映射（依据现有 node 标签）：**node0X=<MGMT_OCTET>（head）/ 02=<MGMT_OCTET>（worker）/ 03=<MGMT_OCTET> / 04=<MGMT_OCTET>**
- Prometheus 9 个 job 加 `machine` 静态标签 + recording rules 改 `by(machine)`；面板 v13 → **v14**
- 验证：targets 带 machine（up）、dcgx 预聚合 node0X-04 四 series ✅
- 无阻塞项

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（v14 已生效） |
| 阻塞项数量 | 0 |
| 关键行动项 | 1 条（旧样本过期后复核） |
| 建议下一步 | 5 分钟后强刷确认资源面板 legend 全显 node0X-04 |

---

## 1. 实施内容

| 层 | 变更 |
|----|------|
| Prometheus prometheus.yml | 9 job（vllm + node-55/58/59/60 + dcgm-55/58/59/60）静态标签 `machine: DGXspark0X`；重启生效 |
| recording_rules.yml | `dcgx:cpu_util_percent` / `mem_util_percent` 聚合 `by(instance)` → `by(machine)`（保留机器名标签） |
| Grafana v14 | 资源面板 114-118 legend = `{{machine}}`；网络面板 119/120 expr `by(machine)` + `{{machine}}`；vLLM 面板 101-113 legend 加 "node0X " 前缀（head <MGMT_OCTET> 端点） |

## 2. 机器名映射

| 机器名 | IP | 角色 | 原标签 |
|--------|-----|------|--------|
| node0X | <NODE_IP> | head | node01-60 |
| node0X | <NODE_IP> | worker（大容量中心） | node01-58 |
| node0X | <NODE_IP> | 新增小盘节点 | gx10-3f4d |
| node0X | <NODE_IP> | 新增小盘节点 | gx10-31c4 |

## 3. 验证

- Prometheus targets：node-58=node0X / node-60=node0X 等，全部 up 带 machine ✅
- `dcgx:cpu_util_percent` 预聚合：node0X/02/03/04 四 series ✅
- 面板 v14 API 确认：资源面板 `{{machine}}`、vLLM 面板 "node0X P50/P99/avg" 等 ✅

**注意**：Prometheus 标签变更后，重启前 scrape 的旧样本（无 machine）在 TSDB 保留 5 分钟——此窗口内同指标存在新老两批 series（None + DGXspark），5 分钟后旧样本过期，曲线 legend 全量显示机器名。属 Prometheus 正常行为。

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 5 分钟后强刷面板，复核资源/网络面板 legend 全显 node0X-04 | SRE | P1 | 今日 |
| 2 | vLLM 恢复后复核 101-113 面板 node0X 前缀显示 | SRE | P2 | LLM 恢复窗口 |
| 3 | 后续新增节点沿用 machine 标签规范（DGXspark0X 递增） | SRE | P3 | 下次接入 |

---

## ⚠️ 待完善 / 已知局限

- vLLM 面板标注 node0X 表示数据来自 head 端点（<MGMT_OCTET>:8001）；TP=2 集群实际推理跨 <MGMT_OCTET>/<MGMT_OCTET> 两机，若需精确到 worker 需 vLLM 多端点接入（当前单端点设计）
- 旧样本过渡期（5 分钟）legend 可能显示空机器名

---

## 📚 数据来源

- Prometheus API（targets 标签、dcgx:* 预聚合 machine）
- Grafana API（v14 legendFormat 验证）
- 现有 node 标签（node01-60 / node01-58）作为映射依据

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
