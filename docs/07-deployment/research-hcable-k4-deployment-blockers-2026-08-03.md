# 定制 4 头交叉 DAC 实现 K4 全互联——软硬件实施堵点清单

> 日期：2026-08-03
> 背景：用户定制到了纯 DAC 的"4 头交叉线"（MFS1S90 的无源铜缆版），用于 4 台 DGX Spark 无交换机 K4 全互联（方案六）。本文档基于 DGX Spark 官方文档、NVIDIA 开发者论坛实测案例、nccl-mesh-plugin 源码，逐层排查软硬件实施堵点。
>
> 拓扑回顾：H 线（4 头交叉 DAC）给 A-C/A-D/B-C/B-D 四条 100G 链路 + 2 根普通 DAC 给 A-B/C-D（各 200G 双通道）= K4 完全图，每节点 2 物理口（4×100G 逻辑口）全用，零中继。

---

## 0. 堵点总览（按严重度）

| # | 堵点 | 层级 | 严重度 | 状态 |
|---|---|---|---|---|
| 1 | ConnectX-7 模块验证（白名单） | 硬件 | 🟡 低→中 | **对无源 DAC 基本不适用**（机制澄清） |
| 2 | 交叉方向正确性 | 硬件 | 🔴 高 | **需逐对打流验收** |
| 3 | nccl-mesh-plugin 4 节点 mesh 实测缺口 | 软件 | 🔴 高 | 检测逻辑已确认成立，实测待做 |
| 4 | A-B 双通道聚合带宽 | 软件 | 🟡 中 | 待实测（可能只 100G） |
| 5 | 幻影节点（Ghost Node）陷阱 | 网络 | 🔴 高 | 已知规避方法 |
| 6 | 子网规划（8 子网/节点 4 逻辑口） | 网络 | 🟡 中 | 有明确规则 |
| 7 | vLLM init-barrier 补丁未合并 | 软件 | 🟡 中 | 已知补丁路径 |
| 8 | NCCL 算法对 K4 链路利用（100G 边瓶颈） | 性能 | 🟡 中 | 影响小（见分析） |
| 9 | CX7 电源管理（hotplug 端口禁用） | 硬件 | 🟢 低 | 已知行为 |
| 10 | DAC 信号完整性（4 头重排 PCB） | 硬件 | 🟡 中 | 需 BER/MTU 9000 实测 |
| 11 | 4 机编排（sparkrun 不支持） | 部署 | 🟡 中 | 自建脚本 |
| 12 | DeepSeek-V4 TP=4 适配 | 部署 | 🟢 低 | 常规参数扩展 |

---

## 1. 硬件层堵点（最硬，全部需上机验证）

### 1.1 🔴→🟡 ConnectX-7 模块验证（白名单）——**对无源 DAC 基本不适用，风险降级**

**机制澄清（2026-08-03 补充核实，修正初版定性）**：ConnectX-7 的 vendor validation（"白名单"）**真实存在，但主要针对有源光模块/AOC**，对无源 DAC 实质不构成门槛：

- **论坛实证（362562）**："The ConnectX-7 firmware enforces strict vendor validation for high-speed links"——严格 vendor 验证针对 optics（第三方光模块 100G 被拒、只能 40G 兜底）；同帖明确："**The DAC works at 100G because passive copper cables don't have firmware/EEPROM vendor-ID negotiation issues the same way active optics do.**"——无源 DAC 不经过 vendor-ID 协商问题；
- **ConnectX-7 官方手册**：模块问题（insufficient power 等）的官方排障建议之一是"**consider switching from Active Optical Cable (AOC) or transceiver to Direct Attached Copper (DAC) connectivity**"——官方推荐用 DAC 规避模块兼容问题；
- **机制原理**：无源 DAC 无 PHY/固件，网卡识别只读 EEPROM 基础字段（SFF-8636：identifier、media type=passive copper、长度、vendor 编码）。字段编程规范即通过；第三方厂商（NADDOD/FS）普遍仿 Mellanox MCP 系列编码实现兼容。

**上一轮"insufficient power 27W"案例的重新定性**：按 ConnectX-7 手册，这是 **PCIe 槽位供电预算（advertised power limit）问题**（平台热插拔供电配置/固件 bug），**不是线缆白名单拒绝**——原装 Amphenol 线同样被报，恰证明与线缆类型无关；升级固件/冷断电 30-60s 恢复（论坛 363193 实证）。

**对本方案的实际风险**：
- 🟢 无源 DAC 被"白名单拒"的概率**远低于 AOC/光模块**；
- 🟡 仍需 EEPROM 基础字段正确（media 不能标成 fiber/active；identifier 合规）——"识别"层要求，非 vendor 锁定；
- 🟡 万一被拒的兜底：`mlxconfig -d <mst设备> q | grep ALLOW` → 若存在 `ALLOW_NON_MLNX_CABLE=false`，`mlxconfig s ALLOW_NON_MLNX_CABLE=true` + 重启（论坛 362562 提供路径）；
- 🔴 **真正的硬件硬风险是"交叉方向正确性 + 信号完整性"（4 头重排 PCB）**，而非白名单（见 1.2/1.5）。

### 1.2 🔴 交叉方向正确性（链路错连风险）

定制线的内部 lane 交叉映射必须符合预期：A 口 1 的 2 个 100G 逻辑口应分别到 **C** 和 **D**（而非都到 C 或 B）。厂商做错方向 → 链路矩阵与预期不符（可能连成非目标拓扑，甚至出现环路/重复对）。

**缓解**：上机后**逐对打流验收**——对全部 6 条预期链路（AB/AC/AD/BC/BD/CD）分别 `ib_write_bw`，确认：
- 每头协商出 2×100G 逻辑口（`ibdev2netdev` 应显示 4 逻辑口/节点）；
- 每条预期链路的带宽/连通性与交叉图一致；
- 未预期的链路（如 A 口 1 两逻辑口都通 C）→ 找厂商重做或调整接线。

### 1.3 🟡 lane-split 协商（物理口可降速，官方确认）

eugr_nv（NVIDIA 论坛 373902）："The port will work at any negotiated speed <=200G... each physical port is represented by two interfaces in the OS - to get full 200G out of a single port, you need to use both."

- H 线每头 200G（2×100G 逻辑）与 Spark 物理口 2×100G 结构匹配 → 每头应协商出 2 个 100G 逻辑口；
- 若定制线做成了"100G 粒度交叉"（每头 2 lanes 激活），ConnectX-7 支持 100G 降速模式（eugr 确认）→ 兼容；
- 验收时检查 `ethtool <iface> | grep Speed` 与 `ibdev2netdev` 逻辑口状态。

### 1.4 🟢 CX7 电源管理（hotplug 端口禁用）

**证据（论坛 365584）**："The Spark has CX7 power savings that disables the CX7 ports when no cable is connected. After connecting the cable, you should see the ports. dgx-spark-mlnx-hotplug enables this feature."

- 插线前端口 Down 是正常行为（省电）；插线后应自动激活；
- 若插线后仍 Down → 检查 `dgx-spark-mlnx-hotplug` 配置（connect-two-sparks playbook 含该组件）+ 冷重启。

### 1.5 🟡 DAC 信号完整性（4 头重排 PCB）

- 无源铜缆 ≤3m 可靠（30AWG）；4 头交叉需内部 PCB lane 重排 → crosstalk/插入损耗高于普通 DAC；
- 需实测：`ib_write_bw` 大消息（4MB/16MB）+ 连续打流 5-10 分钟看 error counter（`ethtool -S <iface> | grep -i err`、`rdma_debug`）；
- MTU 9000 大包（`ping -M do -s 8972`）必须通过；
- 长度 >3m 定制线建议改为混合铜缆（hybrid）或 AOC。

---

## 2. 网络配置层堵点

### 2.1 🔴 幻影节点（Ghost Node）陷阱——禁用官方 autodiscovery

**证据（论坛 359240）**：`discover-sparks / build-and-copy.sh` 会把"同子网双逻辑口"误判为独立物理节点 → 2 台物理机被当成 4 节点 → MPI/Ray 初始化失败或显存超分：
```
Found: <NODE_IP> (spark-2.local)   ← 幻影（同机的另一逻辑口 IP）
Found: <NODE_IP> (spark-1.local)
Found: <NODE_IP> (spark-1.local)   ← 幻影
```
**NVIDIA 员工明确建议**："If you assign IPs to both halves, do not assign IPs from the same subnet!"（同物理口 2 逻辑口严禁同子网）+ "For any NCCL tasks you will use RDMA anyway, so you just need one interface as a control one"（管理面用 1 个接口即可）。

**对本方案的影响**：
- **4 机 K4 部署完全不用官方 discover 脚本**，手工 netplan（本项目一贯做法）；
- **A-B 双通道的 2 个逻辑口（同连 B）也严禁同子网**——配 2 个不同子网（X1/X2）才能规避幻影与路由混乱（见 2.2 子网表）；
- 控制面（管理网 10GbE）与数据面分离，管理网只做 SSH/编排。

### 2.2 🟡 子网规划（K4 需要 7-8 个子网）

基于 nccl-mesh-plugin"子网匹配自动推断邻居"机制，**每条链路的每一端逻辑口必须落在独立子网**（同对端多逻辑口也分不同子网）：

| 链路 | 子网 | A | B | C | D |
|---|---|---|---|---|---|
| A-B（口2 逻辑A） | <NODE_IP>/24 | .2 | .3 | - | - |
| A-B（口2 逻辑B） | <NODE_IP>/24 | .2 | .3 | - | - |
| C-D（口2 逻辑A） | <NODE_IP>/24 | - | - | .2 | .3 |
| C-D（口2 逻辑B） | <NODE_IP>/24 | - | - | .2 | .3 |
| A-C（H） | <NODE_IP>/24 | .2 | - | .3 | - |
| A-D（H） | <NODE_IP>/24 | .2 | - | - | .3 |
| B-C（H） | <NODE_IP>/24 | - | .2 | .3 | - |
| B-D（H） | <NODE_IP>/24 | - | .2 | - | .3 |

- 注意避开 Docker 默认 172.17/172.18 网段、管理网段；
- 每节点 4 个逻辑口 4 个不同子网 → 无同子网逻辑口 → 幻影节点风险归零；
- MTU 9000 全链路（无交换机，无需 9216）。

### 2.3 🟡 GID 配置

- RoCE v2 GID 通常 index 3（`show_gids` 验证）；插件 `NCCL_MESH_GID_INDEX=3`（有自动发现，roadmap ✅）；
- 每个逻辑口（rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1）都要有有效 IPv4 GID——本项目此前已发现 **GID index 5 为空、需用 3** 的坑（8-02 记录），4 机全逻辑口逐一检查。

---

## 3. 软件层堵点

### 3.1 🔴 nccl-mesh-plugin 4 节点 mesh——检测逻辑已确认成立，实测缺口仍在

**好消息（源码级确认，mesh_routing.c）**：
- `count_node_fast_neighbors()` **按"对端节点数"计数**（遍历其他节点，任一共享子网即 count++，同节点多链路只计 1 次）；
- FULL_MESH 判定：`min_degree == max_degree == num_nodes - 1` → 本方案每节点 3 个邻居 → **判定 FULL_MESH 成立** ✅；
- 邻居发现 = **子网匹配自动推断**（IP&mask 相同即相邻）→ 每链路独立子网即可，无需手工路由表；
- relay 按需触发（直连优先）→ FULL_MESH 下 relay 零参与 ✅。

**⚠️ 但插件作者在 SETUP.md 明确断言**："Full mesh with 4 nodes would require 3 NICs per node, which isn't possible on DGX Spark (only 2 ConnectX-7 ports per node). Ring topology is the only option for 4-node Spark clusters."——**作者未考虑"2 物理口 × 2 逻辑口交叉"实现度 3 的场景，4 节点 mesh 无任何测试记录**。可能暴露的问题：
- 握手/注册阶段假设"每节点 fast_addrs ≤ 2"的代码路径（若有则需适配）；
- 连接建立时"子网感知选卡"对"对端 2 个同节点 IP（A-B 双通道 X1/X2）"的处理：可能只建 1 条 QP（见 3.2）；
- **必须实测**：4 机连好后跑 `nccl-tests`（all_reduce/all_gather），看日志 `Detected topology: FULL MESH`、`relay` 不出现、带宽符合预期；若检测异常，回退路径 = 代码适配或 AOC/交换机方案。

### 3.2 🟡 A-B 双通道聚合带宽（可能只跑 100G）

- 插件连接流程：handle 交换所有 IP → connect 时"Find matching subnet"→ 建立 QP；
- README roadmap "Dual-channel per port (200Gbps)" 已实现，但**机制未公开**（多 QP？多 rail？）；
- 若 A-B 双逻辑口（X1/X2）只建 1 条 QP → A-B 实际 100G 而非 200G；
- **影响评估**：TP=4 的 all-reduce 是**小消息高频**（每层 1-2 次，消息 = hidden_size×精度，DS-V4 ~KBs），延迟主导（全直连 1.5μs 是核心优势）；且本项目 **DeepSeek-V4 MLA 的 KV 通信极少**（MLA KV ≈ GQA 的 2%，此前实测 0 fabric 延迟）→ **A-B 100G vs 200G 对推理影响极小**；
- 若追求 A-B 200G：官方建议 bond mode2（balance-XOR）聚合 2 逻辑口（论坛 isdias），但 bond + 插件子网匹配的兼容性需实测（bond 后只剩 1 个 IP，插件按子网选卡可能只认 1 条）；
- **结论**：先按 100G 验收（够用），bond/双通道优化后置。

### 3.3 🟡 vLLM init-barrier 补丁（未合并上游）

- 需 `git apply patches/ticket-f-vllm-nccl-init-barrier.patch` + `VLLM_NCCL_INIT_DELAY=2.0`；
- 本项目 vLLM 是 hybrid-1.6 镜像（vLLM 0.25.2.dev0）——**补丁针对的 vLLM 版本需核对**（patch 改 `vllm/distributed/parallel_state.py` 与 `vllm/envs.py`，若版本差异大需手工移植）；
- 若补丁移植困难：vLLM 侧可用 `NCCL_DEBUG=INFO` + 启动重试/延迟注入替代（社区有绕过方案）。

### 3.4 🟡 NCCL 算法对 K4 的链路利用（100G 边瓶颈分析）

- NCCL ring all-reduce：4 rank 用 4 条边（每 rank 2 邻居）；tree 用 3 条边；**K4 的 6 条边 NCCL 不会全部利用**（ring 拓扑下 100G 对角边若在环内会成为该 stage 瓶颈：~63G vs ~126G）；
- **缓解/影响**：①TP 小消息延迟主导（1.5μs 直连优势远大于带宽差）；②DS-V4 MLA 通信极少；③可 `NCCL_ALGO=Ring` + 观察，必要时调整 rank 顺序让 200G 边优先入环；
- 结论：**性能影响可控，不作为阻塞项**。

### 3.5 🟡 4 机编排（sparkrun 不支持 4 机 mesh）

- 官方 sparkrun：3 机 ring / 交换机；4 机 mesh 无官方工具 → 用本项目 hardened/live 脚本体系扩展 4 机版（start_*_4node.sh），管理面走 10GbE 编排；
- 启动顺序：4 机 NCCL 同时拉起（插件后台握手线程已消除 connect 死锁 ✅）；容器参数：shm 64gb、memlock -1、stack 64MB（本项目已对齐）。

---

## 4. DeepSeek-V4 项目适配（部署层）

| 项 | 值/做法 | 备注 |
|---|---|---|
| TP=4 显存 | 512GB UMA，GPU_MEM 0.80-0.85，模型 160GB（FP8）+ KV | 4 机 KV 池 ~2× 双机基准 |
| 分布式后端 | `--distributed-executor-backend mp`（本项目 D 环境已验证双机；4 机 mp 社区有 TP=8 案例） | 若 mp 4 机不稳，回退 Ray |
| NCCL env | `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`（4 个全列）+ `NCCL_IB_GID_INDEX=3` + `NCCL_SOCKET_IFNAME=管理口` + 插件变量 | 与 3 机环配方一致 |
| MASTER_ADDR/PORT | head 管理 IP + 25000（本项目基线） | - |
| 推理参数 | 沿用双机最优：dspark 投机 + thinking=max + block-size 256 + max-num-seqs | 4 机建议重新 bench |

---

## 5. 验证路径（按顺序执行，每步通过才进下一步）

1. **单根定制线冒烟**（现有 2 机）：插线 → `ibdev2netdev` 确认 4 头各出 2×100G 逻辑口 → `ethtool` 速率 → `ib_write_bw` 逐对打流（**确认交叉方向**）→ MTU 9000 大包 → error counter 5min。
   - 失败动作：冷断电重启（1.1 的 hotplug/供电问题）→ 仍失败则查 EEPROM（要求厂商重编码）→ 仍失败则换 AOC 版。
2. **H 线半拓扑**（2 对双机 A-B、C-D，H 线 2 端各插 1 台）：确认 H 线 4 头在 4 台机器上同时工作、A-C/A-D/B-C/B-D 链路矩阵正确。
3. **4 机全接 + nccl-tests**：`ib_write_bw` 6 条链路逐对验收 → `all_reduce/all_gather` 看日志 `FULL MESH` 检测 + 无 relay + 带宽（直连 ~12GB/s/通道）。
4. **vLLM 补丁移植 + TP=4 启动**：apply patch（版本核对）→ `vllm serve` 4 机 → /health → 单流/并发 bench（对比双机基线 53.8-78.8 t/s 与社区 TP=4 数据）。
5. **性能基准**：prefill/TTFT/decode 全场景，确认 100G 对角链路无异常拖累。

---

## 6. 结论

- **硬件**：最大堵点是 **ConnectX-7 对定制线 EEPROM 的认可（供电白名单）**——这是唯一可能导致"物理不可用"的点，务必先单线冒烟；交叉方向正确性次之（逐对打流可验）。
- **软件**：nccl-mesh-plugin 的 FULL_MESH 检测**已从源码确认对 K4 成立**（按对端节点计数 + 子网匹配），4 节点 mesh 无实测是主要剩余风险，但回退路径清晰（AOC/交换机）；A-B 双通道可能只 100G，对本项目（MLA 低通信）影响可忽略。
- **网络**：幻影节点陷阱通过"全链路独立子网 + 禁用官方 autodiscovery"完全规避。
- **总体**：方案**工程上可行，无不可逾越的堵点**；风险集中在定制线硬件层（EEPROM/交叉方向），建议按第 5 节验证路径逐步推进，单线冒烟应在采购到货后第一时间执行。

---

## 7. 参考来源

- NVIDIA 论坛 363193（ConnectX-7 模块供电拒绝/Cable unplugged/冷重启恢复）
- NVIDIA 论坛 365584（CX7 端口不上线/hotplug 电源管理/NADDOD 线兼容案例）
- NVIDIA 论坛 373902（eugr：端口可协商 ≤200G、双逻辑口结构、bond mode2 聚合）
- NVIDIA 论坛 350417（isdias：GB10 x4 限制 → Cx7 multi-host 模式 2×x4 聚合；官方确认 cross-wired 模型）
- NVIDIA 论坛 359240（Ghost Node：同子网双逻辑口被 autodiscovery 误判为独立节点；官方建议不同子网）
- nccl-mesh-plugin：src/mesh_routing.c（count_node_fast_neighbors 按对端节点计数、FULL_MESH 判定、子网匹配邻居推断）、docs/SETUP.md（4 节点断言"仅 ring"、IP 规划、GID、故障排查）、docs/PARTIAL_MESH_ROUTING_PLAN.md（relay 按需、拓扑检测算法）、patches/ticket-f-vllm-nccl-init-barrier.patch
- sparkrun.dev/networking（子网规则、MTU 9000、1.5μs 延迟）

---

## 附录 A（2026-08-05）：K4 方案最终核对确认（知乎官方实证 + 真机现状）

### A.1 知乎资料《DGX Spark 网络连接》（p/1974520249937307471）实证

| 项 | 知乎实测 | 与本项目结论的关系 |
|---|---|---|
| 结构 | 2 物理口、4 逻辑口；GB10 x4 限制 → CX7 **Socket Direct** 2×x4（路径A/B）；路径A→enp1s0f\*、路径B→enP2p1s0f\* | ✅ 与本项目 lspci 实测（2 domain × Gen5 x4 = cross-wired）完全一致，官方术语"Socket Direct" |
| 单口带宽 | 单物理口双路径并行 **185 Gbit/s**（97+92） | ✅ 验证"单口可吃满全带宽池" |
| 双口带宽 | 双物理口四路径并发 **196 Gbit/s**（4×49） | ✅ 验证"双口并发共享池"（cross-wired 模型） |
| NCCL | 单路径 All-Gather 12.04 GB/s；双路径 22.68 GB/s（~1.9×）、All-Reduce 18.02 GB/s | ✅ 验证"双逻辑口聚合"收益（~12 GB/s ≈ 单 100G 逻辑口实测） |
| 子网 | 双机直连 = **每逻辑口独立子网**（4 子网），NCCL_IB_HCA 列路径 | ✅ 与本项目"每链路独立子网 + 禁同子网"策略一致 |

### A.2 真机现状（2026-08-05，head=60 / worker=58）

```
head:  enp1s0f0np0(Down)  enp1s0f1np1(Up <NODE_IP>)  ← 物理口2/路径A
       enP2p1s0f0np0(Down) enP2p1s0f1np1(Up <NODE_IP>)  ← 物理口2/路径B
worker: enp1s0f0np0(Down) enp1s0f1np1(Up <NODE_IP>)
       enP2p1s0f0np0(Down) enP2p1s0f1np1(Up <NODE_IP>)
```

**关键洞察：当前 A-B 生产链路（8001）= 方案中 DAC#1 的位置（物理口2 双路径 200G，双子网 136/137）**——四机扩展时**直接复用**，只需新增 1 根 H 线（A/B/C/D 的物理口1）+ 1 根 DAC（C-D 物理口2）。

### A.3 lane 映射核对（用户方案逐条验证 ✅）

| 映射 | 逻辑口级对应 | 链路 | 结果 |
|---|---|---|---|
| 头A.lane1-2 → 头C.lane1-2 | A.enp1s0f0np0(路径A) ↔ C.enp1s0f0np0 | A-C 100G | ✅ |
| 头A.lane3-4 → 头D.lane1-2 | A.enP2p1s0f0np0(路径B) ↔ D.enp1s0f0np0 | A-D 100G | ✅ |
| 头B.lane1-2 → 头C.lane3-4 | B.enp1s0f0np0 ↔ C.enP2p1s0f0np0(路径B) | B-C 100G | ✅ |
| 头B.lane3-4 → 头D.lane3-4 | B.enP2p1s0f0np0 ↔ D.enP2p1s0f0np0 | B-D 100G | ✅ |
| DAC#1 A口2↔B口2 | enp1s0f1np1+enP2p1s0f1np1 聚合 | A-B 200G（=现网链路） | ✅ |
| DAC#2 C口2↔D口2 | 同上 | C-D 200G | ✅ |

→ 每节点 4 逻辑口全用、度 3、K4 完全图，**方案成立**。

### A.4 带宽定量（结合知乎实测）

- 单条对角链路（A-C 等）= 单路径单 100G 逻辑口：**~92-97 Gbit/s 单向**（知乎单路径实测）；
- A-B/C-D（双路径聚合）= **~185 Gbit/s**；
- 全并发（A 同时与 B/C/D 通信，4 路径池 ~196 Gbit/s）：对称负载下 A-B ~98、A-C ~49、A-D ~49 Gbit/s——与 cross-wired 池模型一致。

### A.5 子网规划（8 子网，含知乎模式扩展）

| 链路 | 接口对 | 子网 |
|---|---|---|
| A-B | A/B.enp1s0f1np1 | <NODE_IP>/24（现网复用） |
| A-B | A/B.enP2p1s0f1np1 | <NODE_IP>/24（现网复用） |
| C-D | C/D.enp1s0f1np1 | <NODE_IP>/24 |
| C-D | C/D.enP2p1s0f1np1 | <NODE_IP>/24 |
| A-C | A/C.enp1s0f0np0 | <NODE_IP>/24 |
| A-D | A.enP2p1s0f0np0 / D.enp1s0f0np0 | <NODE_IP>/24 |
| B-C | B.enp1s0f0np0 / C.enP2p1s0f0np0 | <NODE_IP>/24 |
| B-D | B/D.enP2p1s0f0np0 | <NODE_IP>/24 |

---

## 附录 B（2026-08-05）：软件层风险清单与消除矩阵（四机 K4 方案）

> 基于当前软件栈真机核实：vLLM 0.26.1.dev0+gd3d3b2cca.d20260805（Anemll v026 今日构建，容器 vllm-envE-node），NCCL 配置 = NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 + NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 + NCCL_IB_GID_INDEX=3 + NCCL_CROSS_NIC=1 + NCCL_PROTO=LL,LL128,Simple + VLLM_DISABLE_PYNCCL=1。

### B.1 风险总览（按严重度）

| ID | 风险 | 层级 | 严重度 | 消除手段（预防/检测/回退） |
|---|---|---|---|---|
| R1 | nccl-mesh-plugin 4 节点 FULL_MESH 零实测（源码逻辑已确认"按对端节点计数 → 度3 → FULL_MESH"，但无 4 机 mesh 运行记录） | NCCL 插件 | 🔴 高 | 预防：每链路独立子网；检测：分阶段实证（B.3）；回退：ring 模式/交换机 |
| R2 | vLLM init-barrier 补丁与 0.26.1 兼容未知（patch 改 parallel_state.py/envs.py，针对旧版） | vLLM | 🔴 高 | 预防：补丁移植 diff 核对；检测：git apply 试打+启动冒烟；回退：VLLM_NCCL_INIT_DELAY=2.0 |
| R3 | A-B 双通道可能只建 1 条 QP（实际 100G 而非 200G） | NCCL 插件 | 🟡 中 | 检测：A-B 双口 ib_write_bw 实测（对比知乎基线 92-97G 单路径/185G 双路径）；影响：MLA 低通信可接受；bond mode2 备选 |
| R4 | Ghost Node/同子网陷阱（discover-sparks 误判） | 网络配置 | 🔴 高 | 预防：8 子网规划+禁用官方 autodiscovery+同物理口逻辑口禁同子网；检测：ip -br a 核对 |
| R5 | NCCL 算法对 K4 链路利用（ring 只用 4 边，100G 对角边可能入环） | NCCL | 🟡 中 | 预防：rank 顺序/NCCL_ALGO 让 200G 边优先入环；影响评估：TP 小消息延迟主导+MLA 低通信；检测：NCCL_DEBUG=INFO |
| R6 | TP=4 多机显存/参数规划（512GB UMA KV 池、mp executor 4 机） | vLLM | 🟡 中 | 预防：GPU_MEM 0.80-0.85 重算；社区 TP=8 先例；回退：Ray executor |
| R7 | 插件构建/分发/版本漂移 | 部署 | 🟡 中 | 预防：容器内构建+镜像版本 pin+SHA 校验+LD_LIBRARY_PATH 统一；检测：日志 Loaded net plugin Mesh (v9) |
| R8 | GID/MTU/子网细节（GID index 5 空坑、MTU 9000） | 网络配置 | 🟢 低 | 预防：show_gids 查 4 逻辑口 IPv4 GID(index 3)+ping -M do -s 8972+静态 IP |
| R9 | 编排与故障恢复（启动顺序、配置持久化、链路降级） | 运维 | 🟢 低 | 预防：worker-first 脚本化+netplan/fstab 持久化+ethtool -S 监控；K4 容错优于环 |
| R10 | 融合算子约束（FlashInfer livelock/DeepGEMM sm121 JIT） | 推理 | 🟡 中 | 预防：attention 锁定 Triton sparse MLA；DeepGEMM NVCC JIT+冒烟（详见 08-05 融合算子报告） |

### B.2 三个"必须实测才能消除"的硬风险（R1/R2/R3）

1. **R1（插件 4 机 mesh）**：双机跑插件（2 节点 mesh，验证插件加载/子网感知/握手）→ 三机环验证 relay → 4 机 K4 全接跑 nccl-tests，日志必须出现 `Detected topology: FULL MESH` 且无 relay 活动；误判 RING 则查 count_node_fast_neighbors（按对端节点数，3 邻居 → FULL_MESH 成立）。
2. **R2（vLLM 补丁）**：下载 patches/ticket-f-vllm-nccl-init-barrier.patch，与 0.26.1 的 parallel_state.py/envs.py 做 diff → 手工移植 → 容器重建 → 4 机启动冒烟；兜底 VLLM_NCCL_INIT_DELAY=2.0。
3. **R3（A-B 双通道）**：A-B 双口同时 ib_write_bw（两条路径并行），对比知乎基线；若只 100G 则评估可接受性（DS-V4 MLA 通信极少，大概率可接受）。

### B.3 软件风险分阶段验证路径

| 阶段 | 动作 | 通过标准 | 失败回退 |
|---|---|---|---|
| S0（现状） | 记录当前 NCCL 基线（all_reduce 双路径 ~18 GB/s） | 基线留存 | - |
| S1 | 双机加载 mesh-plugin（直连模式）跑 nccl-tests | Loaded net plugin Mesh (v9) + 带宽不劣化 | 查 GID/子网/构建 |
| S2 | 三机环验证 relay 路径 | 日志出现 relay 转发 | 查路由表/BFS |
| S3 | 4 机 K4 全接 + nccl-tests | FULL MESH + 无 relay + 6 链路逐对带宽达标 | 回退 4 环 ring/交换机 |
| S4 | vLLM 0.26.1 + 补丁 + TP=4 启动 | /health 就绪 + 推理正确 | VLLM_NCCL_INIT_DELAY/补丁修正 |
| S5 | 全场景 bench（对比双机基线） | decode ≥ 双机 1.5×、无卡死 | 算子/参数调优 |

### B.4 结论

软件层**无不可消除的硬阻塞**：R1/R2 是"未实测"而非"不可行"（有源码/补丁依据 + 明确回退）；R3-R10 全部有预防/检测/回退手段。剩余风险集中在"首次 4 机实测"本身——按 B.3 分阶段推进，每阶段通过标准明确、回退路径清晰，即可逐个关闭。
