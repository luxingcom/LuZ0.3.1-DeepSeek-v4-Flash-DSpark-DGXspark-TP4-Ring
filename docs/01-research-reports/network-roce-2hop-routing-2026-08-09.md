# RoCEv2 2 跳路由打通 + 内网限 TCP + CPU 隔离落地报告

**日期**：2026-08-09
**工作流**：系统设计（工作流 2）+ 事故响应（工作流 3）混合
**参与成员**：Rex（SRE 判定中）、Archi（架构方案）、Tessa（测试验收）

---

## 📌 TL;DR（执行摘要）

- **2 跳 RDMA 已打通**：01→04（经 02 转发）ib_read_lat 16B 实测 **avg 63.5-65.7µs / p99 68-72µs**（3 轮稳定），根因是 ConnectX-7 硬件将 UDP 4791 当 RoCE 拦截（NVIDIA 官方文档证实），卸载 02 的 mlx5_ib 后 4791 走内核转发打通
- **内网限 TCP 已实施**：01/02/04 RoCE 口 INPUT/OUTPUT 禁 TCP，02 FORWARD 只放 UDP 4791 + ICMP——实测内网 TCP 被拒、2 跳 ICMP/RDMA 均通
- **Grafana 已修复**：vllm-realtime.json 网络面板设备过滤不含 RoCE 口（仅 wlP9s9|enP7s7），已改为含 4 个 RoCE 口
- **🔴 03 ConnectX-7 整卡 PCIe 消失**（MODULE_SPLIT 写入后遗症，PCIe AER RxErr），需固件级恢复
- **严重度分布**：🔴严重 1 项（03 固件）/ 🟠高 1 项（02 mlx5_ib 卸载与 TP2 互斥）/ 🟡中 2 项（带宽补测、GLOO 改管理口待 TP2 验证）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（2 跳 RDMA 打通，但 02 与 TP2 互斥待决策） |
| 阻塞项 | 1（03 ConnectX-7 固件恢复） |
| 关键行动项 | 6 条 |
| 建议下一步 | 03 固件恢复 → 02 mlx5_ib 最终决策（TP2 vs 2 跳）→ 带宽补测 |

---

## 🔍 2 跳 RDMA 根因链（ConnectX-7 硬件行为）

### 1. 核心根因：RoCE 引擎硬件拦截 UDP 4791

**NVIDIA 官方文档**（MLNX_EN RoCE 章节）明确：
> "When RoCE is enabled, all traffic to UDP port 4791 is treated as RoCE traffic by the device."

**实测证据链**（决定性）：
| 检查点 | 结果 |
|--------|------|
| 01 发出 UDP 4791（目标 MAC=02） | ✅ 01 tcpdump 确认发出 |
| 02 enp1s0f1np1 tcpdump | ❌ 0 包（包未进内核） |
| 02 raw NOTRACK 计数 | ❌ 0 包（最早 hook 未见） |
| 02 FORWARD LOG 规则 | ❌ 0 包 |
| 02 ethtool rx_vport_rdma_unicast | ✅ 110019 包（网卡 RDMA 引擎接收） |

**结论**：02 的 ConnectX-7 收到 UDP 4791（目标 IP 非本机）→ 硬件 RoCE 引擎拦截（RDMA 接收尝试失败）→ 丢弃，**不进内核 IP 转发路径**。ICMP/其他 UDP 正常走内核转发（2 跳 ping 1.5ms 通）。

### 2. 打通方案（3 步）

| 步骤 | 操作 | 作用 |
|------|------|------|
| 1 | **02 卸载 mlx5_ib**（`modprobe -r mlx5_ib`） | 禁用 02 RoCE → 4791 走内核转发（官方支持方式） |
| 2 | **02 FORWARD 放宽**：<NODE_IP>/24 ↔ <NODE_IP>/24 互转 ACCEPT | 01 的 RoCE 数据源 IP 是 GID idx2=<NODE_IP>（netplan IP），需放行 |
| 3 | **perftest 用 -R（rdma_cm 模式）** | 直接寻址（AH）不支持 L3 路由 → READ 发不出（RETRY_EXC_ERR status 12），rdma_cm 支持路由 |

**附带配置**：01 静态路由（<NODE_IP>/30、<NODE_IP>/30 via <NODE_IP>）、04 反向路由、02 proxy_arp=1、02 ip_forward=1。

### 3. 2 跳 RDMA 实测数据（ib_read_lat -R，16B，01→04 经 02）

| 轮次 | t_min | t_avg | p99 | p99.9 | stdev |
|------|-------|-------|-----|-------|-------|
| 轮1 | 15.14µs | 65.53µs | 70.48µs | 490.32µs | 26.58µs |
| 轮2 | 14.66µs | 65.67µs | 69.74µs | 463.82µs | 23.61µs |
| 轮3 | 14.82µs | 63.91µs | 68.03µs | 74.51µs | 8.57µs |
| **防火墙收紧后** | 14.13µs | **63.46µs** | **71.78µs** | 187.09µs | 14.66µs |

**对比**：1 跳基线 3.27µs（avg）→ 2 跳 63-65µs = **约 20× 劣化**（02 软件转发 + rdma_cm 路径开销，非硬件极限）。

**⚠️ 架构权衡（关键）**：
- 02 的 module1 RoCE active（TP2 需要）会拦截**所有** 4791（含转发）→ **2 跳 RDMA 与 02 跑 TP2 互斥**
- 当前 02 mlx5_ib 保持卸载（2 跳可用，02 暂不能跑 TP2/A 组）

---

## 🔒 内网限 TCP 实施（全部走管理网）

| 节点 | 规则 | 状态 |
|------|------|------|
| 01/02/04 | RoCE 口（enp1s0f0np0/enp1s0f1np1/enP2p1s0f0np0/enP2p1s0f1np1）INPUT/OUTPUT **DROP TCP** | ✅ |
| 02 | FORWARD 只放 **UDP 4791 + ICMP**（policy DROP 拦截 TCP） | ✅ |
| 01/02 启动脚本 | **GLOO_SOCKET_IFNAME=enp1s0f1np1 → enP7s7**（TP2 bootstrap 走管理网） | ✅ 语法 OK，下次 TP2 启动生效 |

**验证结果**：
- ✅ 内网 TCP 被拒（1 跳 <NODE_IP>:22、2 跳 <NODE_IP>:22 均拒）
- ✅ 2 跳 ICMP 通（1.5ms）
- ✅ 2 跳 RDMA 通（63.5µs）

---

## 🏗️ CPU 隔离落地（用户要求持久化绑定转发服务）

| 项 | 状态 |
|----|------|
| isolcpus=18-19 + nohz_full + rcu_nocbs | ✅ 01/02 均已重启生效（grub.d/90-isolcpus.cfg + update-grub） |
| 中断绑定隔离核 | ❌ **不可行**：80 个 mlx5 中断写 smp_affinity=0x40000/0x80000 全部 Invalid argument（ARM64 GIC + isolcpus 拒绝隔离核中断亲和）；实测 CPU 非瓶颈（softirq 0.13%），转发性能不受影响 |
| GLOO 改管理口 | ✅（内网限 TCP 前置） |

---

## ⚠️ 03 ConnectX-7 固件恢复（严重待办）

- **症状**：03 重启后 lspci 无 ConnectX-7（仅 Realtek 管理口 + MediaTek WiFi）；dmesg PCIe AER RxErr（10de:22ce）；接口全消失
- **根因**：昨晚 `mlxconfig set MODULE_SPLIT_M0[0]=1 M0[1]=2 M0[2..15]=FF` 写入后，固件 Next Boot 尝试应用拆分失败 → 端口初始化异常 → 设备失联（写入被接受但重启不生效，且破坏端口训练）
- **恢复难点**：设备不在 lspci，MFT 工具无法访问 → 常规 mlxconfig/mstflint 清 NVConfig 不可用
- **候选**：断电重启（已试无效）→ 需 NVIDIA 固件级恢复手段或联系支持

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 03 ConnectX-7 固件恢复（断电/联系 NVIDIA/找恢复模式） | 用户 + Rex | P0 | 尽快 |
| 2 | 02 mlx5_ib 最终决策：保持卸载（2 跳可用，TP2 暂停）vs 恢复（TP2 可跑，2 跳需临时卸载） | 用户 | P0 | 本次决策 |
| 3 | ib_write_bw 2 跳带宽补测（-R 单 QP 0 迭代问题排查） | Tessa/Rex | P1 | 下一窗口 |
| 4 | 内网限 TCP 规则持久化（iptables-save + systemd 恢复脚本，防重启丢失） | Rex | P1 | 下次维护 |
| 5 | TP2 下次启动验证 GLOO 走管理口（enP7s7） | Rex | P1 | TP2 恢复时 |
| 6 | Grafana 面板修改后可视化验证（RoCE 口流量可见） | 用户 | P2 | 观察 |

---

## ⚠️ 待完善 / 已知局限

- **2 跳延迟 65µs** 含 02 软件转发 + rdma_cm 开销，非硬件极限（理论 2 跳应 ~7µs）；若需更优需交换机或 ConnectX 硬转发（P2）
- **02 与 TP2 互斥**是架构级权衡，需用户决策最终拓扑（TP4 环网 vs 链式）
- **03 固件**恢复方案未定（设备不可见死结）
- 带宽测试（ib_write_bw 2 跳）未完成（CM 事件异常）
- iptables 规则未持久化（重启丢失风险）

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE）判定（已完成，与实测一致）**：
  - 硬件行为 (a) 确认：RoCE 引擎默认吞掉全部 4791（与目标 IP 无关），内核层手段（ip_forward/rp_filter/iptables/NOTRACK）全部无效，必须在网卡层关 RoCE
  - Go 方案 = 02 `modprobe -r mlx5_ib`（官方 router 场景做法，可逆：`modprobe mlx5_ib` 恢复）；备选 devlink `enable_roce` param（若存在）、mlxconfig ROCE_EN
  - 流规则（ethtool -U/tc/devlink）确认无效（RoCE 分类先于 steering）
  - NCCL 权衡：2 跳与 TP2 互斥（同一张卡无法"终止本地 RoCE"+"透明转发 4791"）；缓解 = NCCL_ALGO=TREE（仅沿物理邻接通信）或时间分片（modprobe/rmmod 切换）
  - 风险：02 全 RoCE 关闭、rmmod 前确认无 ipoib/perftest 占用、2 跳路径无 PFC/ECN 性能待补验
- **Archi（架构师）**：环网 + 虚拟网段方案、PFC 有效性判定、TP2×DP2 备选
- **Tessa（测试专家）**：验收标准（2 跳延迟/带宽阈值）、测试矩阵
- **NVIDIA 官方文档**：RoCE 章节（"all traffic to UDP port 4791 is treated as RoCE traffic"）
- **实测数据**：本文档全部数据来自 2026-08-09 现场实测（与 Rex 判定交叉验证一致）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
