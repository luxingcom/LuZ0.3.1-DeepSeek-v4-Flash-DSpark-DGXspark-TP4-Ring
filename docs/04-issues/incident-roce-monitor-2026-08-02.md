# 内网 TP 互联速率（RoCE）监控指标核实与修复报告

**日期**：2026-08-02
**工作流**：事故响应 / 监控指标失实诊断修复（工作流 3 变体）
**参与成员**：Rex（真机坐实与面板修复）/ 主理人（编排预检与交叉验证）

---

## 📌 TL;DR（执行摘要）

- **根因**：vllm-realtime 面板 "内网交换速率" 使用 `node_network_*`（内核 netdev 计数器）作为指标源，而双机 TP=2 的跨机通信走 **RoCE v2 verbs（RC QP）绕过内核栈**，netdev 恒近零——面板查询的网卡不是数据通道，故"数据流极少"
- **事实**：TP 通信量**很大**——测试窗口实测 100~510 MB/s/链路、rx+tx 合计峰值 **1833 MB/s/节点**；双链路（rocep1s0f1/roceP2p1s0f1，200Gb×2）均满用
- **已修复**：面板改用 `node_infiniband_port_data_received/transmitted_bytes_total`（IB 硬件计数器）双向统计，回查测试窗口 max=1833 MB/s（旧查询仅 0.0004 MB/s），验证通过
- 严重度分布：🔴严重 0 / 🟠高 0 / 🟡中 1（监控失实）/ 🟢低 2（GID 隐患、旁路流量说明）
- 阻塞 / 非阻塞：**非阻塞**（服务正常，监控显示失真）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 已修复并验证 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条（GID 隐患修复、补 IB 告警、iperf 基线、面板备份留存） |
| 建议下一步 | 修正 NCCL_IB_GID_INDEX；补 IB 链路告警；受控带宽对照测试 |

---

## 事故时间线（诊断时间线）

| 时间 | 事件 |
|------|------|
| 08-02 11:35-12:25 | 泰莎四环境基准矩阵执行（A/B/C/D 轮换）——TP 互联实测 100~510 MB/s/链路、峰值 1833 MB/s/节点 |
| 08-02 12:40 | 用户反馈：内网交换速率指标在测试时数据流极少，质疑双机 TP 通信通道 |
| 08-02 12:42 | 主理人预检：面板查询 `node_network_*{enp1s0f1np1|enP2p1s0f1np1}`（只统计 RX）；发现 enp1s0f1np1 累计 TX 367GB、enP2p1s0f1np1 近零；初判"面板未查错网卡、双链路未用"（netdev 视角） |
| 08-02 12:5x | Rex 真机坐实：rdma QP 144 条 RC 全 RTS、IB 硬件计数器字节级对称 → **NCCL 走 verbs，netdev 非数据通道**；双链路均满用；367GB 系 socket 旁路（权重分发） |
| 08-02 13:0x | Rex 修复面板 206（改用 IB 计数器双向统计）并验证 |

## 影响范围

- **影响面**：仅面板 "内网交换速率 (RoCE MB/s)" 显示失实（恒近零），其余图正常
- **未受影响**：vLLM 服务、NCCL 通信（实际 1.5~1.8TB/链路历史累计）、其余 17 图

## SEV 评级

**SEV3（中）** — 监控指标失实导致运维盲区（无法观测 TP 互联带宽），无服务影响。

## 根因（5 Why）

1. **现象**：测试时面板内网交换速率数据流极少（近零）
2. **Why 1**：面板查询 `node_network_*` 计数器 → 该计数器来自 `/proc/net/dev`（内核网络栈）
3. **Why 2**：NCCL 双机通信走 **RoCE v2 verbs（RC QP）**——RDMA 数据面绕过内核网络栈，`/proc/net/dev` 不统计 verbs 流量
4. **Why 3**：面板用错指标源——RoCE 真值在 **`node_infiniband_port_data_*`**（sysfs 端口计数器，node-exporter 已采集，单位已校准 ×4）
5. **Why 4**：面板创建时未对照 node-exporter 可用指标族（netdev vs infiniband 语义差异），且未做真机流量验证

## 真机证据（Rex，已坐实）

| 项 | 证据 |
|---|---|
| NCCL 通道 | `rdma resource show qp`：vllm 进程 144 条 RC QP 全 RTS（comm=VLLM::Worker/Worker_TP，每卡 72 条），worker lqpn2445 ↔ head lqpn2453 精确成对 |
| 数据量 | IB 硬件计数器字节级对称：worker rocep1s0f1 TX=1.59TB == head RX；测试窗口 100~510 MB/s/链路、峰值 1833 MB/s/节点 |
| 双链路 | 每节点 72+72 QPs 均分两条 HCA，均 LinkUp 200000Mb/s——**双链路满用**（enP2p1s0f1np1 netdev=0 是 verbs 绕过内核栈的必然，非故障） |
| 367GB TX | `node_network_*` 的 367GB：worker→head 单向 socket 旁路，峰值 197MB/s@昨日 09:41 UTC，与旧容器 production-ready 崩溃（09:34 TCPStore broken pipe, exit137）时间重叠——权重/校验分发流量，非主 TP 数据 |
| 单位换算 | node_infiniband_port_data_* = sysfs port_*_data × 4（node-exporter 已转字节） |

## 修复方案（已执行并验证）

**面板 vllm-realtime panel 206 "内网交换速率"**（Grafana admin API，备份 `/tmp/vllm-realtime-backup.json` @58）：
- Target A（总）：`sum by(instance)(rate(node_infiniband_port_data_received_bytes_total{device=~"rocep1s0f1|roceP2p1s0f1"}[1m]) + rate(node_infiniband_port_data_transmitted_bytes_total{device=~"rocep1s0f1|roceP2p1s0f1"}[1m])) / 1048576`，legend `{{instance}} 总`
- Target B（按链路）：同式加 `by(instance, device)`，legend `{{instance}} {{device}}`
- 标题：**"内网 TP 互联速率 (RoCE MB/s)"**；单位 MBs；窗口 1m
- **验证**：新查询回查测试窗口 max=**1833 MB/s**；旧查询同窗口仅 0.0004 MB/s

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **修正 NCCL_IB_GID_INDEX**：5 对应 GID 为空（有效 IPv4 GID 在 index 2/3），改 3 或移除，防换库/版本后 verbs 初始化失败回退 socket | Rex | P1 | 下轮部署 |
| 2 | 补告警：`node_infiniband_state_id != 4` / link_downed 增长 / TP 运行期互联速率过低 | Rex | P1 | 1 周内 |
| 3 | 受控 iperf3 / ib_write_bw 对照测试，固化单位换算与带宽基线（预期 ~200Gb/链路） | Rex+Tessa | P2 | 2 周内 |
| 4 | 保留面板备份 `/tmp/vllm-realtime-backup.json` 并纳入 hardened/configs 版本管理 | Docu | P2 | 下轮迭代 |

## ⚠️ 待完善 / 已知局限

- **主理人预检修正记录**：预检曾基于 netdev 视角初判"面板未查错网卡、双链路未用、367GB 系主通道"——Rex 真机证据（QP/IB 计数器）推翻部分推断：面板**确实用错指标源**、双链路**满用**、367GB 系旁路流量。最终结论以真机证据为准
- 面板存于 Grafana DB（非 provisioning 文件），变更需 admin（admin/aicad_grafana_dev），匿名只读
- node_network_* 现仅作 socket 回退旁路指示，如需可另加一图（防复发建议①）
- 空闲时段双通道≈0 属正常（当前会话 GPU 0%）

## 📚 数据来源 & 成员产出索引

- Rex（SRE）原始产出：rdma QP 证据、IB 计数器对称性、流量分布、双链路利用率、面板修复与验证、GID 隐患（SendMessage 回传，含证据附录）
- 主理人编排预检：面板 PromQL、IP 分配、/proc/net/dev 统计、NJU 预检（注：预检推断部分被真机证据修正）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
