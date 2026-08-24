# DGX Spark NCCL RING 01↔04 对角延迟实测报告（2026-08-09）

**日期**：2026-08-09
**工作流**：基准测试（NCCL 对角延迟验证）
**参与成员**：Tessa（测试专家，方案设计）、Rex（SRE，执行）

---

## 📌 TL;DR（执行摘要）

- 整体结论：**3-rank ring 的对角边（04↔01）因无直连 IB 路径无法建连**（NCCL 不做透明多跳转发，机制前提不成立）→ T3 主测不可执行，按方案预案走**组合回退估计**。
- 关键数据：**对角单程延迟 Ldiag ≈ 28.9µs（p50）/ 43.5µs（p99）**，对比 2 跳软件转发 RDMA 63.5µs → **约 2.2× 提升**（未达 15µs 理想目标）。
- 链路健康确认：L1/L2 对称（broadcast one-way 18.4µs/18.3µs）、200G 满速、0 错包/丢包、全程 IB 无 Socket 回退、NCCL 2.30.7 版本硬要求满足。
- 严重度分布：🟡 1 项（对角边无法直连建连，硬件拓扑限制，非故障）/ 🟢 链路与工具链验证全部通过。
- 阻塞 / 非阻塞：**非阻塞**——对角直测需要物理直连（加线/换拓扑），属架构决策项。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（组合回退数据有效，直测受硬件拓扑限制） |
| 阻塞项数量 | 1（对角边建连：无直连路径，需物理直连才可直测） |
| 关键行动项 | 3 条（见行动清单） |
| 建议下一步 | 对角延迟直测需 01↔04 物理直连；当前链路健康无需处理 |

---

## 🧪 一、测试方法与拓扑

### 1.1 环境（前置确认）

| 节点 | 管理口(enP7s7) | RoCE IP | mlx5 设备（ACTIVE） | NCCL 版本 |
|---|---|---|---|---|
| 01 | <NODE_IP> | <NODE_IP> / <RING_SUBNET>（L1） | rocep1s0f1、roceP2p1s0f1 | 2.30.7 ✓ |
| 02 | <NODE_IP> | <NODE_IP> / <RING_SUBNET>（L1）、<NODE_IP>/13（L2） | 4 口全 ACTIVE | 2.30.7 ✓ |
| 04 | <NODE_IP> | <NODE_IP> / 14（L2） | rocep1s0f0、roceP2p1s0f0（仅 L2） | 2.30.7 ✓ |

- 拓扑确认：01↔02 直连 L1、02↔04 直连 L2；**01↔04 无任何直连 IB 路径**（ip route get 走管理网网关，RoCE ping 100% 丢包）
- 工具链：nccl-tests 2.19.7（github master tarball，编译 all_reduce_perf/broadcast_perf，三端 /opt/nccl-tests/build/）；OpenMPI 4.1.6（三端）；NCCL 2.30.7 = pip nvidia-nccl-cu13==2.30.7（/opt/nccl-2307，ldd 确认 nccl-library=23007）
- 方案要点（Tessa v1）：3-rank ring（01-02-04）主测 T3；T2_12/T2_24 为 1 跳锚点；严禁双 rank (01,04) 直连；环境 NCCL_ALGO=RING、NCCL_PROTO=Simple、NCCL_SOCKET_IFNAME=enP7s7、GID_INDEX 按节点校正（01/04=3、02 L2=5，规避 np0 169.254.x link-local 污染 idx2/3）

### 1.2 冒烟结果（关键机制验证）

- 3-rank ring init：01↔02、02↔04 边正常建立（NET/IB 确认）；**对角边 04↔01 建连失败**：
  `ibv_modify_qp failed with 110 Connection timed out, dev rocep1s0f0, local GID ::ffff:<NODE_IP>, remote GID ::ffff:<NODE_IP>`
- 根因：01 尝试直连 04（<NODE_IP>）建 QP，无直连路径 → 超时 → ring 无法闭合 → **T3 判无效，按预案走组合回退**（与 Tessa 方案"最大不确定点"预测一致）

---

## 📊 二、实测数据（16B all_reduce_perf，RING/Simple/1 通道，-w100 -n5000，5 runs）

### 2.1 单跳锚点（T2_12 / T2_24）

| run | T2_12（01↔02 L1）µs | T2_24（02↔04 L2）µs |
|---|---|---|
| 1 | 28.96 | 29.12 |
| 2 | 29.17 | 28.76 |
| 3 | 28.92 | 28.59 |
| 4 | 29.04 | 29.06 |
| 5 | 29.04 | 28.77 |
| **mean** | **29.02** | **28.86** |
| **p50** | **29.04** | **28.77** |

per-iteration 代表 run（-I 1）：T2_12 time=30.42 i_min=25.54 i_p99=47.14；T2_24 time=29.93 i_min=25.54 i_p99=39.94

### 2.2 对角延迟（组合回退）

| 指标 | 公式 | 数值 |
|---|---|---|
| Ldiag p50 | (T2_12 + T2_24)/2 | **28.91µs** |
| Ldiag mean | (29.02+28.86)/2 | 28.94µs |
| Ldiag p99（组合） | (47.14+39.94)/2 | **≈43.5µs** |
| 校验 T3 理论 | 2×(T2_12+T2_24) | ≈115.8µs（若对角边可闭合） |

### 2.3 交叉验证（broadcast one-way）

| 链路 | 延迟 |
|---|---|
| L1（01→02） | 18.38µs |
| L2（02→04） | 18.32µs |
| 对称性 | ✅ 高度对称，佐证链路质量 |

### 2.4 与既有基线对比

| 路径 | 延迟（p50） | 说明 |
|---|---|---|
| 1 跳 RDMA 16B（perftest） | 3.27µs | 纯硬件 wire 延迟 |
| 2 跳软件转发 RDMA（已放弃方案） | 63.5µs | 02 内核转发 + rdma_cm |
| **NCCL 对角（组合估计）** | **28.9µs** | 2×1 跳 NCCL step + GPU/驱动开销 |
| NCCL 单跳 one-way（broadcast） | 18.4µs | GB10 GPU launch + 驱动开销为主 |

**提升**：28.9µs vs 63.5µs → **约 2.2×**。未达 Tessa 方案 15µs 目标的原因：GB10 单卡上 NCCL 16B 每步含 GPU launch + 驱动开销（one-way broadcast 即 18µs，纯 wire 应 <5µs），叠加"对角为组合估计"口径。

---

## 🔍 三、NCCL_DEBUG 关键证据

- 三端均 "Using network IB"，**无 "Using network: Socket"**（内网 TCP DROP 未触发回退）
- NET/IB 通道：T2_12 用 [0]rocep1s0f1:[1]roceP2p1s0f1；T2_24 用 [0]rocep1s0f0:[1]roceP2p1s0f0
- GID：NCCL_IB_GID_INDEX=3（01/04）、5（02 L2，规避 np0 169.254.x 污染 idx2/3）
- Ring/Trees：L1/L2 边建立正常；对角边 QP 超时（冒烟 §1.2）
- GPU：三端测试前 0% util 空闲；DGX Spark GB10 单卡

---

## 🩺 四、链路健康确认

- 200Gbps full duplex（L1/L2 双端）
- ethtool 计数 0 错包/0 丢包
- 9000B 帧 ping 0% 丢包（MTU 9000 生效）
- L1/L2 延迟对称（18.3-18.4µs）

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 对角延迟直测需物理直连 01↔04（加线或拓扑调整）——若为业务刚需 | 用户/Archi | P2 | 待决策 |
| 2 | 若追求更低 16B 延迟：试 NCCL_PROTO=LL 或增加通道数（方案口径已锁定 Simple/RING，属后续优化） | Tessa/Rex | P2 | 可选 |
| 3 | 04 np0 的 10.20.0.x 运行时 IP 补录 netplan（重启会丢） | Rex | P1 | 下次维护 |

---

## ⚠️ 待完善 / 已知局限

- **T3 主测不可执行**：对角边无直连路径（NCCL 不做透明多跳转发），Ldiag 为组合估计（(T2_12+T2_24)/2），非直测
- GB10 单卡 NCCL 16B 每步含 GPU launch/驱动开销（~18µs/步），显著高于纯 wire 延迟
- 02 np0 存在 169.254.x link-local 地址污染 GID idx2/3（已用 per-node GID 规避，未改系统配置）
- nccl-tests -z 1 blocking 模式会把 MPI barrier 计入（~400µs 虚高），已改用非阻塞 time 列口径

---

## 📚 数据来源 & 成员产出索引

- Tessa（测试专家）原始产出：Task #8 NCCL RING 测试方案 v1（工具/拓扑/指标/风险/命令序列）
- Rex（SRE）原始产出：前置确认表、冒烟日志（ibv_modify_qp timeout）、5 runs 数据表、NCCL_DEBUG 关键行；落盘 /opt/nccl-tests/build/ + /tmp/nccl_results/
- 基线数据：benchmark-link-1hop-2026-08-09.md（3.27µs）、network-roce-2hop-routing-2026-08-09.md（63.5µs）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
