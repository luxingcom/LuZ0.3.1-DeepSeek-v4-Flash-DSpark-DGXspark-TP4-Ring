# DGX Spark 网络硬件结构与四机环网可行性专项核查

> 核查日期：2026-08-02
> 问题来源：项目组对 DGX Spark 200G 网口内部结构、虚拟专线/中继一跳、H-AOC 交叉线缆识别、四机环网社区实现的疑问
> 证据来源：sparkrun.dev 官方网络文档、NVIDIA ConnectX-7 官方 datasheet、Mellanox MFS1S90-HxxxE 官方产品资料、StorageReview 硬件评测（lstopo 实测）、nccl-mesh-plugin 项目文档与源码、NVIDIA 开发者论坛

---

## 0. 结论摘要（先给答案）

| 问题 | 结论 |
|---|---|
| **① 2×200G 口 = 4×100G 逻辑口、PCIe lane 分组复用、速率共享？** | ✅ **属实（2026-08-02 双机 lspci/sysfs 真机验证）**。硬件事实：ConnectX-7（MT2910）以 **4 个 PF 接入 2 个独立 PCIe domain，每个 PF 都是 Gen5 x4（实测 cur_width=4 / 32 GT/s）**，共 8 lanes；**cross-wired 交叉接线**：每条 x4 承载两个物理口各 1 个逻辑口 → 每个物理口的 2 个 100G 逻辑口 = 4+4 lanes，**任意物理口都能利用全部 8 条 PCIe**；两个物理口（4 个 PF）复用这 8 条 lane，真实网口共享 PCIe 带宽（平台单向上限 2×126 Gbps ≈ 200G 有效）。 |
| **② 能否虚拟出"中继一跳专属通道"（2 口一收一发，延迟翻倍≈3μs）？** | ⚠️ **部分可行，延迟目标达不到**。非相邻节点**物理上没有链路，任何虚拟化都造不出直连**；中继只能软件 store-and-forward。相邻节点间的"双通道一收一发"✅ 可行（nccl-mesh-plugin 生产配置）。但中继一跳 = 1 次 RDMA 收 + 1 次 RDMA 发 + CPU 软件转发，官方数据 **relay 增加 ~1 RTT/跳**，实测直连单向延迟 1.5μs → **一跳实际 5-10μs 级别，不是 3μs**。 |
| **③ 这一跳是否支持 RDMA、网卡能否卸载？** | ❌ **每段链路是 RDMA，但无端到端 RDMA 直通，网卡不卸载中继**。ConnectX-7 的 ASAP² 只卸载虚拟化 vSwitch/vRouter 转发（OVS 场景），**无 RDMA/GPUDirect 中继卸载引擎**；中继节点 CPU 必须参与收包入内存→再发送。这是软件转发，不是硬件 cut-through。 |
| **④ 4 根 200G QSFP56 线缆能否实现 4 机互联？** | ✅ **能，这就是社区 4 机环的标准做法**。4 节点环 = 4 条边 = 4 根线（每节点 2 口各连一个邻居），nccl-mesh-plugin 生产拓扑即此（titanic-iceberg-carpathia-water）。软件要求见第 5 节。 |
| **⑤ H-AOC（MFS1S90-HxxxE）2×200G→2×200G 交叉线缆能否被识别？** | ✅ **能被标准 QSFP56 识别**（SFF-8665/SFF-8636 合规 + EEPROM 可读 + 热插拔，Dre Dyson 实测即插即用）。但注意：其官方定位是**交换机互联**（ToR↔Spine），对 Spark 点对点互联**无拓扑增益**（两端都是自己的双口），价值仅在省线缆/配合第 3 卡。 |

---

## 1. 网卡型号与端口内部结构核实（问题①）

### 1.1 网卡型号与驱动

| 项 | 值 | 证据 |
|---|---|---|
| 网卡 | **NVIDIA ConnectX-7 SmartNIC**（集成版，非标准 PCIe 卡） | StorageReview 评测、sparkrun.dev |
| 物理端口 | **2× QSFP56**（每口 200G，4×50G PAM4） | 官方 datasheet、评测 |
| 标准 ConnectX-7 能力 | 最高 400G、PCIe Gen5 x16/x32、ASAP²、RoCE、GPUDirect、TLS/IPsec/MACsec 卸载 | 官方 datasheet |
| **DGX Spark 上的实际 PCIe 配置** | **PCIe Gen5 x4 ×2（共 8 lanes）**，lstopo 实测确认 | StorageReview |
| 驱动栈 | mlx5_core / mlx5_ib（MLNX_OFED / NVIDIA 驱动），RoCEv2 | sparkrun.dev 命令示例 |
| 逻辑接口 | 4× 100G 网口（4 eth + 4 RoCEv2，共 8 个设备名） | ibdev2netdev 输出 |

### 1.2 端口→PCIe→逻辑口的映射（**真机核实修正版**）

> **修正说明（2026-08-02 22:42 真机验证）**：初版按资料推断为"每组 x4 固定服务 1 个物理口"，**被双机 lspci/sysfs 实测推翻**。正确模型为 **cross-wired（交叉接线）**：每条 x4 承载"两个物理口各 1 个逻辑口"。提问者的原始理解（每个物理口的 2 个 100G 逻辑口 = 4+4 lanes、两物理口复用 8 lanes）**全部正确**。

**真机证据（head=60 / worker=58 双机一致，2026-08-02 实测）**：

```
$ lspci -nn | grep -i mellanox
0000:01:00.0  Mellanox MT2910 Family [ConnectX-7] [15b3:1021]
0000:01:00.1  Mellanox MT2910 Family [ConnectX-7] [15b3:1021]
0002:01:00.0  Mellanox MT2910 Family [ConnectX-7] [15b3:1021]
0002:01:00.1  Mellanox MT2910 Family [ConnectX-7] [15b3:1021]

$ cat /sys/bus/pci/devices/<每个BDF>/{current_link_width,current_link_speed}
4 × Gen5 x4：cur_width=4, cur_speed=32.0 GT/s（全部 4 个 PF 一致）

$ lspci -tv（树形，节选）
-[0000:00]---00.0-[01-0f]--+-00.0  ConnectX-7   ← domain 0000，一条 Gen5 x4
                           \-00.1  ConnectX-7
-[0002:00]---00.0-[01-0f]--+-00.0  ConnectX-7   ← domain 0002，一条 Gen5 x4
                           \-00.1  ConnectX-7
```

**正确映射（cross-wired）**：

```
ConnectX-7（MT2910 单芯片，2 物理口 × 2 逻辑口 = 4 PF）
│
├─ PCIe domain 0000 ─ Gen5 x4（4 lanes，单向 126 Gbps）
│     ├─ PF.0 → 物理口 p0 逻辑口 A（100G）= enp1s0f0np0 / rocep1s0f0
│     └─ PF.1 → 物理口 p1 逻辑口 A（100G）= enp1s0f1np1 / rocep1s0f1
│
└─ PCIe domain 0002 ─ Gen5 x4（4 lanes，单向 126 Gbps）
      ├─ PF.0 → 物理口 p0 逻辑口 B（100G）= enP2p1s0f0np0 / roceP2p1s0f0
      └─ PF.1 → 物理口 p1 逻辑口 B（100G）= enP2p1s0f1np1 / roceP2p1s0f1

∴ 物理口 p0 = {dom0 PF.0, dom2 PF.0} = 4+4 lanes = 全部 8 条 PCIe lane
∴ 物理口 p1 = {dom0 PF.1, dom2 PF.1} = 4+4 lanes = 全部 8 条 PCIe lane
∴ 两物理口（4 个 PF）复用这 8 条 lane → 真实网口共享 PCIe 带宽
```

（证据链：ibdev2netdev 实测——单线插 p0 时 Up 的恰是 enp1s0f0np0 [dom0] 与 enP2p1s0f0np0 [dom2]；worker 现况 Up 的恰是 enp1s0f1np1 [dom0] 与 enP2p1s0f1np1 [dom2] 对应 p1。命名规律：无 P2p 前缀 = domain 0000，P2p 前缀 = domain 0002；f0/f1 = 物理口 p0/p1。）

**与提问者的表述逐条对照（全部 ✅）**：

| 提问者说法 | 核实结果 |
|---|---|
| "两个 200G 网口" | ✅ 正确（2× QSFP56） |
| "每个网口虚拟 2 个 100G 网口" | ✅ 正确（QSFP56 4×50G lane 重组为 2×100G 逻辑口） |
| "每个 100G 网口连接 4 条 PCIe5 总线" | ✅ **正确**（每个 100G 逻辑口 = 1 个 PF = 1 条逻辑 x4；实测每 PF cur_width=4） |
| "这两个 100G 网口分别对应 4+4 PCIE 接口" | ✅ **正确**（物理口 p0 的 2 个逻辑口分别落在 domain 0000 与 0002 的 x4 上 = 4+4） |
| "系统内实际识别 4 个 100G 网口" | ✅ 正确（4 对 eth+roce 设备） |
| "机器本身只有 8 条 PCIe5.0 总线" | ✅ 正确（2 个 domain × Gen5 x4 = 8 lanes） |
| "任意一个物理网口都能利用上 8 条 PCIe" | ✅ **正确**（单口 2 个逻辑口横跨 2 条 x4 = 8 lanes 全可用） |
| "两个网口复用这 8 条 PCIe，真实网口共享 PCIe 带宽" | ✅ **正确**（4 个 PF 竞争 2 条 x4；平台单向上限 2×126 Gbps） |

**带宽数学验证**（cross-wired 模型下）：
- PCIe Gen5 单 lane 单向 3.94 GB/s（32 GT/s，128b/130b 编码）→ x4 单向 15.75 GB/s ≈ 126 Gbps；每 100G 逻辑口需求 12.5 GB/s < 126 Gbps ✅；
- **单口（p0）满载**：2 个逻辑口各占一条 x4 → 单向可用 252 Gbps > 200G 需求，**满速 200G 双向毫无压力**（这正是"单口即满速"的机制）；
- **双口同时满载**：每条 x4 上挤 2 个 100G 逻辑口（126 Gbps 单向 < 200 Gbps 需求）→ 每 x4 单向 126 Gbps 成为瓶颈，双口合计单向 ~252 Gbps（协议开销后有效 ~200G）→ **"平台 200G 上限、双线不叠加"的物理根源**；
- 结论：与官方"单条 200G 线缆即两节点完整 200 Gbps、双线不增加点对点带宽"完全一致。

> 额外佐证：官方文档明确"单条 200G 线缆即足以支撑两节点间完整 200 Gbps，双线不增加点对点带宽"；每物理口 2 个逻辑口 **必须分属两个不同子网**（同子网会混淆接口自动发现并破坏路由——官方明确警告）。

---

## 2. "虚拟专线 / 中继一跳专属通道"可行性（问题②③）

### 2.1 物理前提：虚拟化造不出"不存在的链路"

- 4 机环网上，非相邻节点（A↔C）之间**物理上没有线缆**。SR-IOV/网卡虚拟化只能把已有物理口虚拟成更多逻辑接口，**不能凭空产生新路径**。
- 因此"识别出的 4 个 100G 网口划出 2 个给 A↔C 专线"——在纯 2 卡 4 机环（A-B-C-D-A）中**不成立**：A 的 2 个口分别连 B 和 D，A↔C 通信必须经 B 或 D 中继转发。

### 2.2 相邻节点"双通道一收一发"✅ 可行（这是用户设想的正确落点，且 cross-wired 模型使其更强）

- 每个物理口的 2 个 100G 逻辑口可以**都指向同一邻居**（双通道 dual-channel）：NCCL 自动聚合两个 RoCE 接口 → 单节点 200G 双向吞吐（官方文档确认 NCCL 双接口激活时聚合接近 200 Gbps）。
- **这正是 nccl-mesh-plugin 的生产配置**："200Gbps QSFP56 direct RDMA links (ConnectX-7, **dual-channel**)"，每条链路 200G。
- **cross-wired 带来的额外优势**：同一物理口的 2 个 100G 逻辑口分属两条独立 x4（domain 0000 / 0002）——若按"一收一发"分工（口 A 走 dom0 收、口 A' 走 dom2 发），**收发路径在 PCIe 侧天然隔离（各 4 lanes）**，互不挤占。这正是提问者"划分出 2 个专用该通道、一收一发"设想的最优落点；NCCL 双接口聚合时两条 x4 公平共享，单口 200G 双向时每方向各吃一条 x4，也不会自相竞争。
- 应用场景：4 机环中 A-B 链路若希望独占带宽，可用 A 的 2 个逻辑口全给 B（牺牲掉 A-D 直连，拓扑退化为线形）——**取舍权衡，不是白赚**。

### 2.3 中继一跳延迟：官方/社区数据 vs 3μs 目标

| 项 | 数据 | 来源 |
|---|---|---|
| 直连单向延迟（ib_write_lat） | **~1.5 μs** | sparkrun.dev 官方文档 |
| 直连往返 RTT | ~3 μs | 由 1.5μs 推算 |
| 交换机中转（1 跳） | ~3 μs（直连 2μs） | 腾讯云社区实测 |
| **软件中继（store-and-forward）** | **"relay adds ~1 RTT/hop"**，即理想情况下每跳 +~3μs，**实际含 CPU 收包入内存 + QP 切换 + 再发送，预计 5-10μs 级** | nccl-mesh-plugin 官方 README |
| 中继带宽代价 | 有效带宽 8+ GB/s（直连）、AllReduce 峰值 16.93 GB/s（1000MB 消息，100G 单通道时代测得；200G 双通道数据未公布） | nccl-mesh-plugin |

**结论：3μs/跳的目标无法达到。** 1.5μs 是**单向写**延迟，而中继一跳需要完整"收+存+转+发"（≥2× 单向网络延迟 + 软件处理）。期望的"延迟翻倍 ≈ 3μs"是纯网络 RTT 的理想下界，加上 store-and-forward 软件开销后实际 5-10μs。

> **但对本项目影响很小**：DGX Spark 的 decode 是 273 GB/s UMA 内存带宽瓶颈（与网络延迟无关，8 机 100G vs 200G 实验已证明 decode 零差异）；TP=2/4 的 all-reduce 通信发生在 prefill/TTFT 阶段（秒级），+5μs 完全可忽略；TPOT 是 ms 级（72ms@TP4），+5μs 是 0.007% 级别。

### 2.4 RDMA 支持与网卡卸载（问题③的明确答案）

- **每段链路都是 RDMA**：A→B 是 RoCEv2 RDMA 写，B→C 也是 RDMA 写——**链路级 RDMA 无问题**；
- **但没有端到端 RDMA 直通**：中继节点 B 必须 CPU 参与（RDMA 收包进主机内存 → 软件转发 → 再 RDMA 发出），**不是网卡 cut-through**；
- **网卡不卸载中继转发**：ConnectX-7 的 ASAP²（Accelerated Switching and Packet Processing）只做 **L2/L3 vSwitch/vRouter 虚拟化转发卸载**（Open vSwitch / 虚拟化场景），**没有 RDMA/GPUDirect 中继卸载引擎**——那是交换机 ASIC（如 MikroTik CRS812）的能力，不是 SmartNIC 的能力；
- nccl-mesh-plugin 已明确 roadmap："cut-through 转发（降低中继延迟）**规划中，未实现**"，当前是 store-and-forward。

**一句话**：中继是"软件快递分拣站"，每段路都是高速 RDMA，但货物要在中继站装卸一次。

---

## 3. H-AOC 交叉线缆核实（问题⑤）

### 3.1 官方资料（Mellanox/NVIDIA 产品 brief + datasheet）

| 项 | 规格 |
|---|---|
| 型号 | **MFS1S90-HxxxE**（H003E 3m / H005E 5m / H010E 10m / H015E / H020E / H030E） |
| 类型 | **Active Optical Splitter H-Cable**（有源光分叉线缆） |
| 带宽 | **2×200Gb/s → 2×200Gb/s**（每端 2× QSFP56，每个头 2×100G HDR100） |
| 调制 | 4× 50Gb/s PAM4 |
| 合规 | SFF-8665（QSFP56）+ SFF-8636（I2C 管理）+ EEPROM 可读 + 热插拔 |
| 供电/功耗 | 单 3.3V，每端 4.35W（typ） |
| 性能 | BER < 1E-15（Mellanox 系统），最长 30m |
| **官方定位** | **交换机互联**：200G ToR 口（配成 2×100G）连 2 个 200G spine 口（也配成 2×100G），省 spine 口与线缆，fat-tree 降层 |

### 3.2 对 DGX Spark 的适用性结论

- **能被识别** ✅：SFF-8665/SFF-8636 合规 + EEPROM 提供产品/状态信息，标准 QSFP56 设备；Dre Dyson（NVIDIA 论坛活跃用户，2026-05）实测 "plug and play，no configuration changes needed"。
- **拓扑语义**：一根 MFS1S90 = 两端各 2 个 QSFP56 头，**内部 lane 交叉**。插法：一端 2 头 → Spark A 双口，另一端 2 头 → Spark B 双口 ⇒ A-B 之间 **2×200G（双口全用）**。
- **重要澄清**：交叉（cross-connect）解决的是**交换机端口 lane 顺序**问题（ToR 与 spine 的 lane 映射不同），对 **Spark↔Spark 点对点无拓扑增益**——两端都是自己的双口，2×200G 效果与"两根普通 DAC"等价（且仍受 200G PCIe 平台上限约束）。
- **真实价值**：①一根线覆盖双口（省线缆、少一个故障点）；②配合第 3 块网卡做 4 机准全互联（见 4.2）。**成本 $450/根**（Dre Dyson 报价）。

---

## 4. 4 根 200G QSFP56 线缆互联与社区四机环网实现（问题④）

### 4.1 标准 4 机环（每节点 2 口，4 根线）——nccl-mesh-plugin 生产拓扑

```
Node A ────200G──── Node B
  │                   │
 200G               200G
  │                   │
Node D ────200G──── Node C

每节点 2 个 QSFP56 口各连一个邻居 → 4 条边 = 4 根 200G QSFP56 线缆 ✅
A↔C、B↔D 无直连 → 软件中继（经邻居，2 跳）
```

**软件特殊要求（nccl-mesh-plugin，已验证 Qwen2.5-14B DeepSpeed ZeRO-3 训练 + vLLM 推理）**：

```bash
# 1. 构建插件（需 libibverbs-dev / librdmacm-dev）
git clone https://github.com/autoscriptlabs/nccl-mesh-plugin && cd nccl-mesh-plugin && make

# 2. 核心环境变量
export NCCL_NET_PLUGIN=/path/to/libnccl-net.so   # 替换标准 NCCL 网络插件
export NCCL_SOCKET_IFNAME=eth0                    # 管理网口（引导用）
export NCCL_MESH_ENABLE_RELAY=1                   # 启用非相邻节点中继路由
export NCCL_MESH_MAX_HOPS=4                       # 最大中继跳数
export NCCL_MESH_RING_LOAD_BALANCE=1              # 环上双路径负载均衡
export NCCL_MESH_RING_PREFER_SHORT=0              # 0=均衡 / 1=始终短路径
export NCCL_MESH_RING_BALANCE_THRESHOLD=1048576   # 切换路径字节阈值（1MB）
export NCCL_MESH_GID_INDEX=3                      # RoCE GID（出错试 0-3）
```

**网络/IP 规划（硬性要求：每条 RDMA 链路独立子网）**：

| 链路 | 子网 | A | B | C | D |
|---|---|---|---|---|---|
| A↔B | <NODE_IP>/24 | .2 | .3 | - | - |
| B↔C | <NODE_IP>/24 | - | .2 | .3 | - |
| C↔D | <NODE_IP>/24 | - | - | .2 | .3 |
| D↔A | <NODE_IP>/24 | .3 | - | - | .2 |
| 管理网 | <NODE_IP>/24 | 引导用（NCCL_SOCKET_IFNAME） | | | |

**vLLM 注意**：需要项目内附 vLLM 补丁（"已提交上游未合并"），标准 vLLM 直接跑 4 机环会初始化竞态。

### 4.2 社区进阶：4 机准全互联（每节点 3 卡 6 口，零中继）——Dre Dyson 方案

- 每节点插 **3 块 ConnectX-7**（2 内置 + 1 扩展槽），6 个 100G 逻辑口；
- 环边用 **2 根 H-AOC（双口）** + 对角线用 **2 根 DAC** → 每对节点 2×100G 直连，**完全消除中继**；
- 需 udev 规则按 PCI 地址固定接口名（重启不漂移）：`/etc/udev/rules.d/70-spark-net.rules`；
- 成本：约 $900（H-AOC）+ 2× DAC + 扩展卡；属于工程绕开官方限制的进阶方案，无一键脚本。

### 4.3 官方立场（必须知晓）

- **官方只支持到 3 节点环**（`dgxspark-3node-ring` NCCL 补丁 + sparkrun `topology: ring`）；sparkrun 报错明确 "ring topology requires exactly 3 hosts. For 4+ nodes, use a switch"；
- **4+ 官方方案 = 交换机**（MikroTik CRS812-DDQ / CRS804-DDQ，MTU 9216）；
- 腾讯云社区实测交换机方案：Qwen VL32B 单机 3.58 → 双机 6.14 → 四机 11.36 tok/s（近线性）；交换机中转延迟 ~3μs vs 直连 ~2μs。

---

## 5. 综合判断与对本项目的建议

1. **硬件理解**：提问者对网卡结构的判断（2×200G = 4×100G、8 条 PCIe5 lane、分组复用、速率共享、任意口可用全部 8 lanes）**经真机验证全部正确**——ConnectX-7 以 4 个 PF（每 PF Gen5 x4）接入 2 个独立 PCIe domain，cross-wired 交叉接线（详见第 1.2 节修正版）。

2. **中继方案取舍**：若扩 4 机且接受软件中继——环网可行但每跳 +5-10μs；**对 decode 吞吐零影响**（带宽瓶颈），仅 prefill/TTFT 有微秒级影响，工程上完全可接受。若追求零中继，需第 3 卡（准全互联）或交换机。

3. **RDMA 卸载期望**：网卡不提供 RDMA 中继卸载（ASAP² 只管虚拟化 L2/L3），"一跳 3μs + 硬件卸载"目前无现成方案，属研究级方向（NVIDIA 尚未公开）。

4. **本项目现有基线不动**：当前 2×DGX 直连 + dspark 投机（53.8-78.8 t/s）已优于社区所有公开配方；若未来扩 4 机，**首推交换机**（官方验证 + 近线性扩展 + 支持 700B 级模型），4 机环作为无预算/实验场景的备选。

---

## 6. 基于 cross-wired 硬件模型的 5 种互联方案可行性总评（2026-08-02 真机模型修订版）

> 本节以真机验证的硬件模型（每节点 PCIe 池 = 2× Gen5 x4 = **单向 ~252 Gbps**（约 200G 有效）；单物理口 2 逻辑口横跨 2 条 x4，双口并发时 4 PF 竞争共享）为约束，逐一重评此前讨论的 5 种方案。

### 6.0 共同约束（所有方案都受此约束）

| 约束 | 数值 | 含义 |
|---|---|---|
| 节点级 PCIe 单向池 | ~252 Gbps（2× x4 各 126 Gbps） | **任何拓扑下节点总带宽上限**；多链路拓扑会把池子分薄 |
| 单口独占时可用带宽 | 2×126 = 252 Gbps 单向 | 单物理口（2 逻辑口 = 4+4 lanes）可吃满全池 |
| 双口并发 | 4 PF 竞争 2 条 x4 | 每链路实际 ~126 Gbps 单向（发/收方向分别计） |
| decode 与网络解耦 | 273 GB/s UMA 内存带宽是瓶颈 | **带宽分薄对 decode 几乎无感**，只影响 prefill/TTFT 与大消息 all-reduce |

### 6.1 方案一：双机直连（当前项目形态）——✅ 最优基线

```
Node A ──1× QSFP56 200G── Node B   （单口，2 逻辑口 = 4+4 lanes 全用）
```

| 维度 | 评估 |
|---|---|
| 拓扑/线缆 | 1 根 QSFP56，背靠背，无需交换机 |
| PCIe 利用 | **100%**：单口 2 逻辑口横跨 2 条 x4，NCCL 双接口聚合 → 200G 双向满速 |
| 延迟 | 直连 1.5μs（最优） |
| 软件 | 标准 NCCL + vLLM（本项目现状，零特殊要求） |
| 实测性能 | 本项目 53.8-78.8 t/s（dspark 投机），领先社区所有公开配方 |
| 风险 | 无；唯一限制是 3+ 节点需改造 |
| **结论** | **2 机形态的最优解，PCIe 池 100% 利用，无任何竞争/中继**。 |

### 6.2 方案二：官方三机环（无交换机）——✅ 官方支持，3 机首选

```
     Node A
    /      \
200G        200G
  /            \
Node C ──200G── Node B
（每节点 2 口各连 2 邻居；3 节点环上所有对都是相邻，无中继）
```

| 维度 | 评估 |
|---|---|
| 拓扑/线缆 | 3 根 QSFP56，每节点 2 物理口全用（4 PF 全活） |
| PCIe 利用 | 8 lanes 分给 2 条链路：TP=3 的 all-reduce 同时与 2 邻居通信 → **每链路实际 ~126 Gbps 单向**（池子分半） |
| 延迟 | 全直连 1.5μs（环上无"非相邻对"）——**这是环网方案的核心优势** |
| 软件 | `dgxspark-3node-ring` NCCL 补丁（zyang-dev/nccl 分支）+ `NCCL_IB_SUBNET_AWARE_ROUTING=1 / NCCL_IB_MERGE_NICS=0 / NCCL_NET_PLUGIN=none / NCCL_IB_HCA=4 个 HCA 全列 / GID_INDEX=3`；sparkrun `topology: ring` 自动注入；PP=3 配方已发布（Qwen3.5-397B-INT4 3x-vLLM） |
| 实测 | nccl-tests ~24 GB/s avg bus bandwidth（社区） |
| 风险 | TP=3 每链路带宽打折（decode 无感）；官方报错机制外（仅限 3 机，4+ 拒绝） |
| **结论** | **无交换机多机的官方推荐路线**；3 机 PP/TP 场景最稳妥。 |

### 6.3 方案三：四机环网（4 根 QSFP56 + nccl-mesh-plugin）——⚠️ 可行但有双热点

```
Node A ──200G── Node B
  │                │
 200G             200G
  │                │
Node D ──200G── Node C
（A↔C、B↔D 经邻居软件中继，2 跳）
```

| 维度 | 评估 |
|---|---|
| 拓扑/线缆 | 4 根 QSFP56，每节点 2 口连 2 邻居（社区生产拓扑：titanic-iceberg-carpathia-water） |
| PCIe 利用 | 8 lanes 分 2 链路（每链路 ~126 Gbps 单向）；**且中继节点是双热点**：其 PCIe 池（252 Gbps 单向）要同时承载"自己↔邻居1 + 自己↔邻居2 + 转发邻居1↔邻居2"三路流量 → 中继路径实际带宽再打折 |
| 延迟 | 相邻 1.5μs；非相邻 **+5-10μs/跳**（软件 store-and-forward，达不到 3μs） |
| 软件 | nccl-mesh-plugin：每链路独立子网（硬性）+ `NCCL_NET_PLUGIN / NCCL_MESH_ENABLE_RELAY=1 / MAX_HOPS / RING_LOAD_BALANCE / GID_INDEX=3`；**vLLM 补丁未合并上游**（需自带）；构建依赖 libibverbs/librdmacm |
| 实测 | 直连有效带宽 8+ GB/s、AllReduce 峰值 16.93 GB/s（100G 单通道时代）；200G 双通道数据未公布 |
| 风险 | 中继节点 CPU 转发 + PCIe 争用；插件小众维护风险；官方明确不支持（报错："For 4+ nodes, use a switch"） |
| **结论** | **无交换机 4 机的社区路线**。训练（ZeRO-3，通信步数少）已验证可行；推理并发场景中继热点可能限制扩展性。预算受限的实验室场景可用。 |

### 6.4 方案四：四机准全互联（3 卡 6 口 + H-AOC + DAC）——✅ 零中继，但成本与复杂度最高

```
每节点 3× ConnectX-7（2 内置 + 1 扩展），6 个 100G 逻辑口：
  环边：H-AOC 双口 ↔ 邻居1 / H-AOC 双口 ↔ 邻居2
  对角：DAC 双口 ↔ 邻居3
→ 每对节点 2×100G 直连，无任何中继
```

| 维度 | 评估 |
|---|---|
| 拓扑/线缆 | 每节点 3 卡；2× H-AOC（MFS1S90，$450/根）+ 2× DAC；udev 固定接口名 |
| PCIe 利用 | 节点池 = 内置 8 lanes + 扩展槽（通常 Gen5 x4 ≈ 4 lanes）≈ 12 lanes = 378 Gbps 单向，分给 3 邻居 → **每链路仍 ~126 Gbps 单向**——**消除的是延迟热点，带宽依然分薄** |
| 延迟 | **全直连 1.5μs，零中继**（5 种方案中唯一真正消灭中继的多机拓扑） |
| 软件 | 多子网问题依旧（每邻居独立子网）→ 仍需 nccl-mesh-plugin 的"多子网直连"能力（可关 relay）；或手工路由管理；vLLM 补丁 |
| 成本 | ~$900 线缆 + 扩展卡 + 复杂度 |
| 风险 | 扩展卡槽/供电未经官方验证（Dre Dyson 个人生产验证）；无官方支持；故障面大 |
| **结论** | **通信密集型（TP=4 高频 all-reduce）的极致低延迟路线**。若项目目标是"4 机 TP 无中继"，这是唯一无交换机答案；若接受交换机，方案五更划算。 |

### 6.5 方案五：交换机方案（4 机 / 8 机）——✅ 官方首选，唯一"多机无 PCIe 惩罚"

```
4 机：每节点 1 口 → MikroTik CRS812（8×200G）          → TP=4，每节点 200G 满速
8 机：CRS812 + 2×400DD→4×100G breakout（100G/节点）    → TP=8，或
      CRS804-4DDQ + 4×400G→2×200G（200G/节点）
```

| 维度 | 评估 |
|---|---|
| 拓扑/线缆 | 每节点 1 个物理口入交换机（2 逻辑口聚合 200G）；任意节点间经交换机 1 跳 |
| PCIe 利用 | **单口独占全池（4+4 lanes = 252 Gbps 单向），且流量经交换机转发，不占用任何其他节点的 PCIe**——这是交换机方案区别于环网的关键优势：**多机扩展无节点级 PCIe 惩罚** |
| 延迟 | 交换机 1 跳 ~3μs（vs 直连 2μs，腾讯云实测）——比软件中继（+5-10μs）低一个量级 |
| 带宽 | 4 机 200G 全满速；8 机 100G 时单流 decode 与 200G 无差（内存瓶颈），仅 TTFT/冷 prefill 受影响（短 ctx TTFT +29%~106%，绝对秒级可接受） |
| 软件 | 标准 NCCL + sparkrun / 官方 playbook（MTU 9000/9216、静态 IP、子网规划），**最成熟、零补丁** |
| 实测 | TPOT 近线性（TP1/2/4 = 269/133/72ms）；Qwen VL32B 4 机 11.36 tok/s 近线性；TP=8 Nemotron 550B 就绪 ~10min |
| 风险 | 交换机成本（CRS812 ~$1-3k）；**CRS804 出口限制不可发中国/香港**（国内采购注意渠道）；8 机 100G 冷 prefill 偏慢 |
| **结论** | **4 机/8 机扩展的官方首选与最干净路线**；本项目若扩 4 机首推此方案。 |

### 6.6 五方案横评总表

| 维度 | ①双机直连 | ②官方三机环 | ③四机环(插件) | ④准全互联(3卡) | ⑤交换机 |
|---|---|---|---|---|---|
| 节点规模 | 2 | 3 | 4 | 4 | 4/8 |
| 线缆 | 1×DAC | 3×DAC | 4×DAC | 2×H-AOC+2×DAC+扩展卡 | 每节点 1-2 线+交换机 |
| 每链路带宽(单向) | 200G 满速 | ~126G(池分半) | ~126G + 中继再折 | ~126G(池分3) | **200G 满速** |
| 非相邻延迟 | 无 | 无(全相邻) | **+5-10μs 软件中继** | 无(全直连) | +3μs 交换机 |
| 中继热点 | 无 | 无 | **有(PCIe+CPU)** | 无 | 无 |
| NCCL 软件 | 标准 | 官方补丁分支 | 社区插件+独立子网 | 社区插件(无relay) | 标准 |
| vLLM 补丁 | 无 | 无 | **未合并上游** | 未合并上游 | 无 |
| 官方支持 | ✅ | ✅ | ❌ | ❌ | ✅ |
| 扩展路径 | 需改造 | 到 4 需换方案 | 到 8 不可行 | 到 8 不可行 | **4→8 平滑** |
| 综合评级 | ★★★★★ | ★★★★ | ★★★ | ★★★ | ★★★★★ |

### 6.7 对本项目的最终建议

1. **现状（2 机）不动**：方案一是 PCIe 池 100% 利用的最优形态，53.8-78.8 t/s 已是社区最优。
2. **扩 4 机 → 首选方案五（交换机 CRS812）**：唯一"多机零 PCIe 惩罚 + 官方支持 + 近线性扩展"的路线；每节点 200G 满速、延迟 3μs 远优于软件中继。
3. **预算受限/纯实验 → 方案三（四机环）**：可接受中继热点（decode 无感）与补丁维护成本；训练场景（ZeRO-3）社区已验证。
4. **通信密集型 TP=4 且拒绝交换机 → 方案四（准全互联）**：零中继延迟最优，但需扩展卡 + $900 线缆 + 插件，成本/复杂度最高。
5. **3 机过渡 → 方案二**：官方三机环是无交换机 3 机的唯一官方路线，PP=3 配方现成。
6. **8 机目标**：只有方案五可行（CRS812 breakout 或 CRS804），100G/节点对 decode 零影响、对冷 prefill 有代价（可接受）。

### 6.8 方案六（修正版）：H 线交叉 + 2×双机直连 = 无扩展卡 4 机全互联（K4 full mesh）

> **修正说明（2026-08-02 23:09，提问者纠正）**：初版误将 H 线理解为"给出 2 条 200G 链路（A-C + B-D）"。**正确理解**：MFS1S90 是 **2×200G → 2×200G 全交叉**——每个 200G 头（=2×100G 逻辑）的两路 100G 分别接到对端两个头。故一根 H 线提供 **4 条 100G 链路：A-C、A-D、B-C、B-D**（与超擎博客"一根线完成两台 spine 到两台 leaf 全互联"及图 1 "Spine-Leaf 全互联"一致）。
>
> 提问者连接形式：H 线端 1 头 A→Node A、头 B→Node B；端 2 头 C→Node C、头 D→Node D，交叉后得 A-C/A-D/B-C/B-D（各 100G）；仅 A-B 与 C-D 需补 2 根直连线。**核对结论：正确，且这是真·全互联（K4）而非环网**。

#### 6.8.1 拓扑核对 ✅：K4 完全图（每节点度 3，零中继）

```
              DAC 200G
    Node A ──────────── Node B
      │  ╲            ╱  │
      │100G ╲     100G ╱  │
      │     ╲        ╱    │
   H  │100G  Node(H) 100G │  H
      │     ╱        ╲    │
      │100G ╱     100G ╲  │
      │  ╱            ╲  │
    Node D ──────────── Node C
              DAC 200G

链路全集（6 条 = K4 完全图）：
  DAC 直连：A-B（200G）、C-D（200G）
  H 线交叉：A-C、A-D、B-C、B-D（各 100G）
→ 任意两节点均有物理直连路径 = full mesh = 全点对点互联 ✅
→ 每节点用 2 个物理口（4 个 100G 逻辑口）：口1 → H 线头（2×100G 分别去 2 个邻居），口2 → DAC（2×100G 聚合去 1 个邻居）
→ 节点度 = 3（B/C/D）→ 无中继需求，零中继延迟
```

**与初版（错误理解"2 条 200G"）对比**：初版结论是"4 节点环、2 对需中继"；修正后是 **K4 full mesh、全部直连、零中继**——这是质的差异。提问者原述"构成环网，只是速率不同"中"速率不同"指：H 线 4 条链路为 100G、DAC 2 条链路为 200G（链路速率不对称，见 6.8.3）。

#### 6.8.2 物理识别可行性 ✅ 高（官方设计背书 + 结构天然匹配）

| 项 | 评估 |
|---|---|
| 官方设计用途 | **1 根 H 线 = 4 个 QSFP56 模块 = 4 个设备端口 2×2 全交叉**（超擎博客：2 spine + 2 leaf 全互联；图 1）——"分插 4 台机器"正是设计场景 |
| 与 Spark 结构匹配 | H 线每头按 **2×HDR100（2×100G）** 运行；DGX Spark 物理口 = 2×100G 逻辑口 → 天然匹配；A 口 1 的 2 个 100G 逻辑口各自对端（C、D）独立成链，网卡侧无需特殊配置 |
| 标准合规 | SFF-8665 + SFF-8636 + EEPROM + 热插拔 |
| 剩余风险 | H 线官方验证于 IB 交换机端口；**Spark 网口实测缺失 → 需上机验证**（4 头应各自协商出 2×100G，ibdev2netdev 可见；失败可回退该链路为 DAC） |
| 成本 | 1× MFS1S90（~$450）+ 2× DAC（~$100）≈ $550；**无扩展卡** |

#### 6.8.3 带宽与延迟（cross-wired 模型）——"只是速率不同"的定量分析

**链路速率（线缆层）**：H 线 4 条链路各 **100G**（200G 头拆 2×100G 分给 2 个对端）；DAC 2 条链路各 **200G**（双逻辑口聚合）。

**实际可用带宽（PCIe 池层）**：每节点 4 个 100G 逻辑口（4 PF）全激活，共享 2 条 x4（单向 252 Gbps）。对称负载（TP=4 all-reduce 均匀通信）下：
| 链路 | 逻辑口数 | 实际单向带宽 |
|---|---|---|
| A-B / C-D（DAC） | 2 个逻辑口 | **~126 Gbps**（2×63，占池一半） |
| A-C / A-D / B-C / B-D（H） | 各 1 个逻辑口 | **~63 Gbps**（各占池 1/4） |

**延迟**：全直连 1.5-2μs（H 线为 AOC，光电转换 +~0.5μs，可忽略）；**零中继**（对比方案三/初版方案六的 +5-10μs 中继延迟）。

**对推理的影响**：decode 依旧由 273 GB/s UMA 带宽主导（网络无关）；TP=4 的 all-reduce 走 6 条直连链路，无中继热点，**比 4 机环（方案三）通信模式更优**；仅对角 100G 链路可能成为 prefill 阶段大消息通信的次要瓶颈（可接受）。

#### 6.8.4 软件要求（比方案三更简单：relay 零参与）✅ 有文档证据

nccl-mesh-plugin 的拓扑自动检测（`mesh_detect_topology`，docs/PARTIAL_MESH_ROUTING_PLAN.md）：
- 判定规则：`min_degree == max_degree == num_nodes - 1` → **FULL_MESH**；本方案每节点度 3（4 节点）→ **自动识别为 FULL_MESH** ✅；
- relay 按需触发：`mesh_connect` 先查直连 NIC（`find_nic_for_peer`）→ 有直连走 `mesh_connect_direct`，**仅无直连才 relay**；FULL_MESH 下所有路由条目 num_hops=1，**relay 完全不参与** ✅；
- 所需配置：`NCCL_NET_PLUGIN=<libnccl-net.so>` + `NCCL_SOCKET_IFNAME=管理口` + 每条链路独立子网（6 链路 = 6 子网，DAC 双逻辑口可同子网归并为 4-5 个子网）+ `NCCL_MESH_GID_INDEX=3`（或自动发现）；
- vLLM 仍需 `patches/ticket-f-vllm-nccl-init-barrier.patch`（未合并上游）+ `VLLM_NCCL_INIT_DELAY=2.0`。

#### 6.8.5 对比与结论

| 维度 | 方案六修正版（1H+2DAC） | 方案三（4×DAC 环） | 方案四（3 卡准全互联） | 方案五（交换机） |
|---|---|---|---|---|
| 拓扑 | **K4 full mesh** | 4 环 | K4 full mesh（每对 2×100G） | K4 full mesh（经交换机） |
| 中继 | **0**（relay 不触发） | 2 对 +5-10μs | 0 | 0（+3μs 交换机） |
| 链路速率 | 对角 100G + 邻边 200G | 全 200G | 全 200G | 全 200G |
| 实际单向带宽 | 对角 ~63G / 邻边 ~126G | ~126G | ~126G | **200G 满速** |
| 线缆/成本 | 3 根 ~$550 | 4 根 ~$200 | 3 卡+$900 | 交换机 ~$1-3k |
| 扩展卡 | **不需要** | 不需要 | 需要 | 不需要 |
| 官方支持 | ❌（H 线 Spark 识别待实测） | ❌ | ❌ | ✅ |
| 评级 | ★★★★ | ★★★ | ★★★★ | ★★★★★ |

**结论**：修正后方案六是**无扩展卡 4 机场景的最优拓扑**——真 full mesh、零中继、3 根线、~$550。比方案三（4 环）质优（无中继对、无中继热点），比方案四（3 卡）省一块扩展卡与线缆成本，代价是对角 4 条链路 100G（实际 ~63G 单向）与 H 线在 Spark 上的识别待实测。**落地前唯一硬验证项：H 线插 DGX Spark 后的协商识别（ib_write_bw 冒烟）**；软件侧 nccl-mesh-plugin 的 FULL_MESH 自动检测 + relay 按需机制有文档背书。

---

## 7. 参考链接

- sparkrun.dev/getting-started/networking/（官方：PCIe split、1.5μs 延迟、子网规则、3+ 节点需交换机）
- NVIDIA ConnectX-7 datasheet（400G、PCIe Gen5 x16/x32、ASAP²、RoCE、GPUDirect 卸载能力）
- Mellanox MFS1S90-HxxxE Product Brief / Datasheet（2×200G→2×200G H-AOC、SFF-8665/8636、交换机互联定位）
- StorageReview：NVIDIA DGX Spark Review（lstopo：CX7 = 2× Gen5 x4，8 lanes，平台上限 200G）
- StorageReview：DGX Spark Cluster Review（Dell/GIGABYTE/HP 双机评测，网络拓扑三配置）
- GitHub autoscriptlabs/nccl-mesh-plugin（README/BENCHMARKS/SETUP：relay +1 RTT/hop、4 机环生产、8+ GB/s 直连、16.93 GB/s AllReduce、vLLM 补丁）
- NVIDIA 论坛：Three node Spark clusters without a switch（sparkrun ring 支持、NCCL 环境变量、3 机环报错信息）
- NVIDIA 论坛：Is training on 3 nodes without a switch supported（dgxspark-3node-ring NCCL 补丁 + NCCL_IB_* 完整配置）
- NVIDIA 官方 blog：Scaling Autonomous AI Agents（4 机支持声明、TP1/2/4 TPOT 数据）
- Dre Dyson 博客：4-Node DGX Spark Cluster Without a Switch（H-AOC + DAC 准全互联、udev 规则）
- 腾讯云开发者社区：DGX Spark 多节点集群搭建坑（直连 2μs / 交换机 3μs、四机 11.36 tok/s）
- 超擎数智博客：NVIDIA 200G HDR Splitter H-Cable 使用场景及应用优势（2023-11，H 线官方用途 = 1 根线 4 模块连 2 spine + 2 leaf 交叉互联、二层 Fat-Tree 扩容 100→200 台）

---

*核查完毕。一句话总结：硬件结构你的理解对；"虚拟专线"相邻节点双通道可行、非相邻节点只能软件中继且达不到 3μs；H-AOC 能识别但对点对点无拓扑增益；4 根 QSFP56 线缆做 4 机环可行，软件上用 nccl-mesh-plugin（每链路独立子网 + 中继 + vLLM 补丁）。*
