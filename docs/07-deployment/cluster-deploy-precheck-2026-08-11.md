# 集群部署前保障报告：环网错误率复测 + 内外自检交叉验证

**日期**：2026-08-11
**工作流**：工作流 4（部署前检查 Go/No-Go）+ 工作流 2（技术调研交叉验证）
**参与成员**：Rex（错误率复测、内部自检）、Archi（外部调研）、Zhen（汇编交叉验证）
**状态**：🟡 有条件 Go（整改 3 个 P1 后即可部署 TP4 集群）

---

## 📌 TL;DR（执行摘要）

- ✅ **任务1 环网错误率：改善**。昨晚 P1（03 module0 FEC 2280）已归零（重插复位），16 口全 200G UP 无降速无 flap，PHY 级错误全 0 → **P1 可解除**（建议先跑一轮高负载 iperf3 确认计数不增长后正式解除）。
- ✅ **任务2 外部调研**：阿奇完成 13 条分级资料收集（T0 官方文档×5 / T1 论坛实测×2 / T2 社区×6）。关键结论：**4 机直连环网为非官方支持拓扑**（官方仅 2-3 机直连或 4 机走交换机），但社区有 4 机直连实测成功案例（vLLM+TP4）；每 QSFP 口含 2 个逻辑接口需全配 IP。
- ⚠️ **任务2 内部自检**：P0 无；**P1×3**（①环 03—01 段 f0 口无 RoCE IP → 环逻辑不闭合，NCCL 全环通信不可用 ②四台 /etc/hosts 无映射 ③01/02 内存仅余 ~10G）；P2×4。
- 🔗 **交叉验证**：3 个 P1 均有外部权威方案支撑修复路径（官方 playbook IP 规划 / NCCL 主机名解析要求 / 资源评估），无内外矛盾；QoS 建议与社区实测一致（直连阶段 DSCP trust，勿配 PFC）。
- 🎯 结论：**有条件 Go** —— 补齐 03—01 段 L3 IP + hosts 映射 + 评估 01/02 内存后，即可按 RING 闭环推进 TP4 部署。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件 Go（部署门禁：P1×3 整改后） |
| 错误率 P1 | ✅ 可解除（2280→0，建议 iperf3 验证） |
| 自检 P0/P1/P2 | 0 / 3 / 4 |
| 外部资料 | 13 条（T0×5、T1×2、T2×6），最新至 2026-07 |
| 关键行动项 | 5 条（P1×3、P2×2） |
| 建议下一步 | 补齐环网 L3 + hosts → iperf3 压测 → TP4 RING 部署 |

---

## ✅ 任务1：四段环网错误率复测（Rex 实测）

| 段 | 口 | 基线 FEC corr | 当前 | PHY/ip 错误 | 状态 |
|---|---|---|---|---|---|
| 03↔01（新线） | 03 f0×2 | **2280【P1】** | **0** | 全 0 | 200G UP |
|  | 01 f0×2 | 3 | 2281/2503* | 全 0 | 200G UP |
| 01↔02 | 01 f1×2 | 15 | 2281/2503* | 全 0 | 200G UP |
|  | 02 f1×2 | 36 | 341/329* | 全 0 | 200G UP |
| 02↔04 | 02 f0×2 | 0/36 | 341/329* | 全 0 | 200G UP |
|  | 04 f0×2 | 305 | 0 | 全 0 | 200G UP |
| 04↔03 | 04 f1×2 | 25 | 0 | 全 0 | 200G UP |
|  | 03 f1×2 | 0 | 0 | 全 0 | 200G UP |

\* 01/02 为设备级共享累计值（同卡两口共享），含两段无法精确归因；无活动错误证据。

**改善判断**：✅ 改善。P1 关键计数 03 module0 2280→0（重插复位，down/up ~24s）；04 端 305/25→0；dmesg 无 FEC/CRC/PHY 告警。
**观察项（P2）**：①01/02 共享累计计数高于基线（2281/2503/341/329），无活动错误证据，需 24h 增量复核；②驱动未暴露 fec_* 计数（基线口径无法用当前 ethtool 复现），建议装 mft/mlxlink 精确读。

## ✅ 任务2a：部署前内部自检（Rex 实测，四机）

### 架构与软件（全部 ✅ 一致）
- CPU：aarch64，20 线程（X925×10@3.9G + A725×10@2.8G），单 NUMA；内存 121Gi+15Gi swap
- GPU：1×GB10/机，驱动 580.173.02，CUDA 13.0，52-53°C，空闲 12W；统一内存 128GB（nvidia-smi 不报 Memory）
- CX-7：2 卡 4 口，FW 28.45.4028 一致，全口 LINK_UP；NVMe smartctl 全 PASSED
- 软件栈四台一致：Ubuntu 24.04.4、内核 6.17.0-1029-nvidia、Docker 29.2.1、nvidia-container-toolkit 1.19.1、容器内 torch 2.11.0+cu130 / NCCL 2.28.9 / vllm 0.26.1.dev0（anemll 0.2.1-v026.0）
- ⚠️ 已知限制：GPU 显存按系统内存估（UMA）；QoS 用 mlnx_qos 报 not supported（需 dcb/mlxconfig 核对）

### 版本差异表（01/02 vs 03/04）
| 项目 | 01/02 | 03/04 | 影响 |
|---|---|---|---|
| 磁盘 | 3.6T NVMe | 916G NVMe | 03/04 容量减半 |
| docker nvidia runtime | 未注册 daemon.json | 已注册 | 功能可用（--gpus 生效），建议对齐 |
| 内存占用 | 已用 110Gi（91%），swap 已用 2.7/4.5G | 已用 13Gi（11%） | **01/02 部署余量不足** |

### 问题清单
| # | 级别 | 项 | 依据 |
|---|---|---|---|
| 1 | 🟠 P1 | **环 03—01 段 f0 口无 RoCE IP**（01 f0 仅 169.254、03 f0 无 IP）→ 环逻辑不闭合，NCCL 全环通信不可用 | 实测 ip addr |
| 2 | 🟠 P1 | 四台 /etc/hosts 无 node01~04 映射（NCCL 跨机 hostname 解析必需） | 实测 getent |
| 3 | 🟠 P1 | 01/02 内存可用仅 ~10-11G 且 swap 已用，升级/扩 TP 前需释放或评估 | 实测 free |
| 4 | 🟡 P2 | 01/02 daemon.json 未注册 nvidia runtime（与 03/04 对齐） | docker info |
| 5 | 🟡 P2 | RoCE 无损配置（PFC/ETS）无法确认（mlnx_qos 不支持此卡） | mlnx_qos |
| 6 | 🟡 P2 | host nvcc 缺失（业务全容器化，非阻断） | which nvcc |
| 7 | 🟡 P2 | sudo 密码暴露面（secrets README），建议部署前轮换 | 审计 |

## ✅ 任务2b：外部调研（Archi，13 条分级资料）

### 关键结论
1. **4 机直连环网非官方支持拓扑**（T0：DGX Spark User Guide / Sync Cluster Assistant 2026-07：官方仅支持 2-3 机直连，4 机须走 QSFP 交换机，≥0.8Tbps）；社区 4 机直连实测成功（T1：MiniMax-M3 on 4×DGX Spark + vLLM，2026-07）
2. **每 QSFP 口 = 2 个逻辑接口**（T0#1/#3）：只配 1 个 IP 带宽减半；官方 playbook 提供环网 IP 规划（每链路独立 /24）+ netplan 40-cx7.yaml 范例 + discover-sparks 脚本（T0#3，2026-03）
3. **RoCE/NCCL**（T1#6 实测）：NCCL_IB_HCA=ibdev2netdev 实际 twin 列表、GID 用 show_gids 逐机核对（mismatch → ncclCommInitRank 失败）、NCCL≥2.21 自动选 GID（系统 2.30+ 更佳）、MTU 全链路统一（4200 或 9216 二选一）、vLLM 多机用 mp 后端（--no-ray 比 Ray 稳）
4. **QoS**（T2#10 实测）：无 DCB 交换机不要配 PFC；直连阶段 mlnx_qos --trust dscp 即可；走交换机才需 MTU9216+FEC91
5. **常见坑**：27W 供电警告正常非故障；NetworkManager 覆盖 IP（需 unmanage）；重配 IP 不重启产生 GID 空洞；容器缺 /dev/infiniband 或 memlock 未放开 → ibv_reg_mr 失败；NCCL_IB_DISABLE=1 → 退化 TCP 降 4×；bond 无效（多 rail 融合用 IB_MERGE_NICS）

## 🔗 内外交叉验证（Zhen 汇编）

| 内部发现（Rex） | 外部依据（Archi） | 交叉结论 |
|---|---|---|
| P1-1：03—01 段 f0 无 IP，环不闭合 | T0#1/#3：每口 2 逻辑接口须全配 IP，官方 playbook 环网 IP 规划；T1#6：GID 逐机核对 | **吻合**：按 playbook 模式补 4 口 IP（建议 10.20.1.x /24 或 /31 点对点），配后重启网络防 GID 空洞 |
| P1-2：hosts 无映射 | T1#6/T2#8：NCCL 跨机 hostname 解析必需；防火墙勿拦 RoCE 子网 | **吻合**：四台补 /etc/hosts，确认 ufw 全空（已核实无规则） |
| P1-3：01/02 内存余量 ~10G | 无直接外部依据（本地资源约束） | 部署前释放内存或评估占用；TP4 扩组前先测 |
| P2：QoS 无法确认（mlnx_qos 不支持） | T2#10：无 DCB 交换机勿配 PFC，直连用 DSCP trust | **吻合**：直连环网阶段 DSCP trust 即可，PFC 留待上交换机 |
| 错误率改善（2280→0） | T0#4：ethtool/perfquery 调试法；mft/mlxlink 精确读 FEC | 补充验证：iperf3 高负载后复核计数；装 mft/mlxlink 建精确基线 |
| 4 机环网非官方拓扑 | T0#2：Sync/playbook 不覆盖 4 机直连 | **预期管理**：手动配置为主，勿依赖 Sync 自动组网 |

## 🚦 Go/No-Go 决策（部署门禁）

**🟡 有条件 Go** —— 硬性门禁（P1×3 清完才可部署 TP4）：
1. 补齐 03—01 段 f0 四口 RoCE IP（环闭合）→ NCCL 全环可达
2. 四台 /etc/hosts 补齐映射 → 跨机 hostname 解析
3. 01/02 内存释放/评估 → 部署余量
软性建议：iperf3 压测确认错误率不增长后正式解除 P1；QoS 按 DSCP trust 配置。

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 | 验收口径 |
|---|------|---------|--------|---------|---------|
| 1 | 补 03—01 段 f0 四口 RoCE IP（按官方 playbook 模式：独立 /24 或 /31，MTU 统一）并 netplan apply + 重启网络防 GID 空洞 | SRE | P1 | 1 次维护窗口 | 四口互通、环上每段 ping 通、ib_write_bw 正常 |
| 2 | 四台 /etc/hosts 补 node01~04=<NODE_IP>~<MGMT_OCTET> | SRE | P1 | 同上 | getent 全返管理网 IP |
| 3 | 评估/释放 01/02 内存（~10G 余量），明确 TP4 部署资源账 | SRE + 用户 | P1 | 部署前 | 部署前后 swap 不增长 |
| 4 | iperf3 高负载压测四段环网 + 复核错误计数，正式解除错误率 P1；装 mft/mlxlink 建 FEC 精确基线 | SRE | P2 | 1 周内 | 计数不增长 |
| 5 | 01/02 daemon.json 注册 nvidia runtime；QoS 按 DSCP trust 落地；sudo 密码轮换 | SRE + 用户 | P2 | 部署前 | 四台配置一致 |

## ⚠️ 待完善 / 已知局限

- 错误率口径：驱动未暴露 fec_* 计数，当前以 rx_prio0_buf_discard + PHY 计数近似，mft/mlxlink 装好后基线更精确。
- 01/02 共享计数（2281/2503/341/329）无法单段归因，24h 增量观察中。
- 4 机直连环网无官方开箱支持，后续升级（驱动/NVIDIA 软件）需验证不破坏手动配置。

---

## 📚 数据来源 & 成员产出索引

- Rex（错误率复测）：teammate-message sre-engineer @ 2026-08-11（ethtool 16 口 + dmesg + PHY 计数实测）
- Rex（内部自检）：teammate-message sre-engineer-2 @ 2026-08-11（lscpu/free/nvidia-smi/smartctl/docker/os-release/ip addr 等全量实测）
- Archi（外部调研）：teammate-message architect @ 2026-08-11（13 条资料：docs.nvidia.com DGX Spark User Guide、NVIDIA Sync Cluster Assistant、github.com/NVIDIA/dgx-spark-playbooks、NCCL User Guide、NVIDIA 博客、forums.developer.nvidia.com×2、DeepWiki、dredyson×2、technotim、keithtyser、ai-infrastructure.net）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
