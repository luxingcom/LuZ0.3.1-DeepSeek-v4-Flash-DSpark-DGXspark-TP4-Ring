# DGX Spark 四机集群 RoCEv2 低延迟优化专项成果总结（2026-08-08~09）

**日期**：2026-08-09
**工作流**：成果总结（测试成果归档）
**参与成员**：Docu（文档师）主笔，数据源自 benchmark-link-1hop-2026-08-09.md、network-roce-2hop-routing-2026-08-09.md、项目工作日志

> 范围：RoCEv2 低延迟优化专项（2026-08-08 ~ 2026-08-09 定稿）；另附同日完成的 embed OOM 修复与 litellm 吞吐优化两项关联成果。本文档为工程保障团队汇总稿，原始报告索引见 §7。

## ① TL;DR

- 链式拓扑定稿：**04 — 02 — 01**，02 为枢纽，L1（01↔02）与 L2（02↔04）均为 200G 直连，01↔04 需经 02 两跳；03 待固件恢复（注：已于 08-09 自行恢复可见，固件未动）。
- **1 跳 RDMA 16B 延迟基线 avg 3.27µs / p99 3.46µs**，twin 并行带宽 188G（共享物理链路接近 200G 上限）。
- **路径隔离是第一优化手段**：同链路 60% TCP 背景使 RDMA 延迟 +128%（7.46µs / 12.03µs）；TCP 走 L2、RDMA 走 L1 后恢复基线（3.35µs / 3.54µs）。
- **2 跳 RDMA 曾打通**（根因：ConnectX-7 硬件将 UDP 4791 全当 RoCE 拦截 → 卸载 02 的 mlx5_ib 后走内核转发），但延迟约 20×（63.5–65.7µs），且与 02 跑 TP2 互斥。**2026-08-09 12:20 用户拍板：放弃 2 跳转发方案，关闭所有转发功能，恢复纯 1 跳 RoCE + TP2，对角通信改用 NCCL RING 1 跳中继验证**。
- 已落地：isolcpus=18-19（01/02）、QoS DSCP PFC、iptables 内网限 TCP（已 netfilter-persistent 持久化）、Grafana 4 端口面板、rp_filter=1（sysctl.d 持久化）。
- 遗留：QoS PFC 持久化执行待维护窗口、03 接线规划待用户通知、NCCL RING 对角延迟实测待执行。

## ② 拓扑与接线结论（2026-08-09 定稿）

| 项 | 结论 |
|---|---|
| 拓扑 | 链式 04 — 02 — 01；02 为枢纽；01↔04 需经 02 两跳 |
| 01↔02 | module1，200G 直连（1 跳） |
| 02↔04 | module0，200G 直连（1 跳） |
| 03 | 无接线，ConnectX-7 曾消失后自行恢复（2026-08-09 12:30 确认可见），固件未动 |
| 链路能力 | module0/module1 双模块均 200G；插 200G 直通线协商 200000M；此前 100G 为 2-lane H 线限制所致 |
| 网段 | L1(01-02)=<NODE_IP>/30 + <NODE_IP>/30；L2(02-04)=<NODE_IP>/30 + <NODE_IP>/30；L3 预留 |
| MTU | 10.100.x netplan 正式 9000；10.20.0.x 段已还原 1500（测试改动） |
| 旧网段 | 10.100.x 保留给 TP2 |

## ③ 关键数据表

### 3.1 1 跳链路基准（benchmark-link-1hop-2026-08-09.md）

| 指标 | 数值 |
|---|---|
| 带宽 L1-A | 111 G |
| 带宽 L1-B | 110 G |
| twin 并行带宽 | 188 G（共享物理链路，接近 200G 上限） |
| RDMA 16B 延迟基线 | avg **3.27µs** / p99 **3.46µs**（stdev 0.04） |
| 60% TCP 同链路 | avg **7.46µs** / p99 **12.03µs**（**+128%**） |
| 物理链路隔离后（TCP 走 L2、RDMA 走 L1） | avg **3.35µs** / p99 **3.54µs**（恢复基线） |

### 3.2 CPU-延迟关联

| 场景 | softirq | 延迟表现 |
|---|---|---|
| 全速打流 | ~10%（2 核满） | 抖动区 |
| 60% 背景流量 | 0.13% | 延迟稳定 0.219ms，0 丢包 |

### 3.3 1 跳 vs 2 跳 RDMA 延迟对比（network-roce-2hop-routing-2026-08-09.md）

| 场景 | avg | p99 | 备注 |
|---|---|---|---|
| 1 跳（基线） | 3.27µs | 3.46µs | 直连 |
| 2 跳（rdma_cm，已放弃） | 63.5–65.7µs | 68–72µs | 约 20×，软件转发 + rdma_cm 开销 |

### 3.4 优化前后对比（综合）

| 维度 | 优化前 | 优化后 |
|---|---|---|
| 1 跳 16B 延迟 | 基线 3.27µs | 隔离后 3.35µs（受背景流量时从 7.46µs 恢复） |
| 2 跳 RDMA | 不可达（4791 被硬件拦截） | 曾打通 63.5µs；**已放弃转发方案，改 NCCL RING** |
| 内网流量 | 未约束 | 纯 RoCE（内网 TCP 连接=0 实测确认） |
| CPU 隔离 | 无 | isolcpus=18-19 nohz_full rcu_nocbs（01/02 生效） |
| 丢包 | — | 60% 背景 0 丢包 |

## ④ 根因与决策记录

### 4.1 2 跳 RDMA 根因链（已证实）

1. RoCE 启用时，**ConnectX-7 硬件把 UDP 4791 一律当 RoCE 处理**（NVIDIA 官方文档证实）；
2. 02 收到 01→04 的 4791（目标非本机）被**硬件拦截丢弃**（tcpdump / NOTRACK / FORWARD 计数全 0）；
3. **卸载 02 的 mlx5_ib 后**，4791 转由内核转发 → 2 跳打通；
4. perftest 直接寻址（AH）不支持 L3 路由，需 `-R rdma_cm` 模式。

### 4.2 硬件能力边界（2026-08-09 查证，决定性）

- ConnectX-7 的 RoCE 是**整卡级**开关（`echo 0 > /sys/devices/{pci}/roce_enable`，NVIDIA 官方文档 + 官方论坛确认）
- **per-port 级 RoCE 控制不存在**：devlink port function roce 仅适用于 VF/SF，且禁用时全设备 GID 表一起禁用
- 单张卡无法"既当 RDMA 端点（TP2）又透明转发 4791（2 跳）"→ 2 跳与 TP2 并存的唯一方案是双卡分工（卡B off 转发），代价为 TP2 降单口 110G + 失去 LAG + 改 NCCL 脚本
- **用户拍板（2026-08-09 12:20）：放弃 2 跳转发方案，关闭所有转发功能，恢复纯 1 跳 RoCE + TP2；对角通信改用 NCCL RING 1 跳中继**（环网/链式下对角数据由 2×1 跳硬件 RoCE 完成，预期 µs 级）

### 4.3 isolcpus 限制

- 01/02 已重启生效 isolcpus=18-19 + nohz_full + rcu_nocbs；
- **IRQ 绑隔离核在 ARM64 GIC 不可行**（已知平台限制，记录在案）。

### 4.4 内网限流策略

- 01/02/04（及 03 一致化）RoCE 口 INPUT/OUTPUT DROP TCP；02 FORWARD policy DROP；
- GLOO_SOCKET_IFNAME 改 enP7s7（管理口）；
- 实测确认内网纯 RoCE（内网 TCP 连接=0）。

## ⑤ 落地清单

### 已生效（生效中）

| 项 | 状态 |
|---|---|
| 链路升级 200G + 网段规划 | ✅ 生效 |
| 1 跳 RDMA 基线建立 | ✅ 生效 |
| iptables 内网限 TCP（四台） | ✅ 生效 + **netfilter-persistent 持久化** |
| GLOO_SOCKET_IFNAME 管理口 | ✅ 生效（下次 TP2 启动生效） |
| isolcpus=18-19 nohz_full rcu_nocbs（01/02） | ✅ 重启生效 |
| QoS：L1/L2 双端 `mlnx_qos --trust=dscp --pfc=0,0,0,1,0,1,0,0`（P3/P5） | ✅ 运行时生效（持久化待批准窗口） |
| Grafana vllm-realtime.json 网络面板补 4 个 RoCE 端口 | ✅ 生效（备份 .bak_roce） |
| embed OOM 修复（--kv-cache-memory=4GB） | ✅ 生效 |
| rp_filter=1 | ✅ 生效 + sysctl.d 持久化 |

### 已回退（无收益/测试项）

- busy_read / busy_poll 回退 0（无收益）
- tcp_low_latency / netdev_budget / netdev_budget_usecs 还原默认（对 RDMA 无改善）
- 2 跳转发全套配置（ip_forward/proxy_arp/FORWARD/NOTRACK/静态路由/探测 IP）已还原
- 10.20.0.x 段 MTU 还原 1500

### 5.3 关联成果（非 RoCE 专项，同日完成）

| 项 | 详情 |
|---|---|
| embed OOM 根因修复 | VLLM_GPU_MEMORY_UTILIZATION 在 anemll 0.2.1 失效 → 改用 `--kv-cache-memory=4294967296`（4GB）；03/04 各释放 ~108GB；标准启动脚本 start_embed_8022.sh 四台落地（3 条铁律） |
| litellm 吞吐优化 | P0 解除 embedding key rpm：5→380 req/s；P2 `--num_workers 2`：c32 377→651 req/s（**+73%**）；P1 Python batch 化已就绪，但生产 Rust v18-server 需源码重编译 |

## ⑥ 遗留与风险

| 优先级 | 项 | 说明/风险 |
|---|---|---|
| 🟡 P1 | QoS PFC 持久化执行 | 需维护窗口重启验证；PFC 需 L1/L2 双端一致 |
| 🟡 P1 | NCCL RING 01↔04 对角延迟实测 | 方案设计中（Tessa），前置条件已就绪 |
| 🟡 P2 | 03 接线规划 | 留空，等用户通知；接线后需补 netplan/iptables 检查 |
| 🟡 P2 | 10.20.0.x 段 MTU 决策 | 1500（已还原）vs 9000（200G 性能），待用户确认 |
| 🟢 P3 | Grafana 可视化验证 | 面板已补，待数据验证 |

## ⑦ 相关报告索引

| 报告 | 路径/标识 |
|---|---|
| 1 跳链路基准 | benchmark-link-1hop-2026-08-09.md |
| 2 跳 RDMA 打通（历史） | network-roce-2hop-routing-2026-08-09.md |
| 清理还原报告 | cleanup-restore-roce-test-traces-2026-08-09.md |
| Grafana 面板 | vllm-realtime.json（备份 .bak_roce） |
| embed 启动脚本 | start_embed_8022.sh（四台落地） |
| litellm 优化 | report-litellm-optimization-p0p1p2-2026-08-09.md |

---

> 本总结由工程保障团队 AI 协作生成，数据以原始报告为准。
