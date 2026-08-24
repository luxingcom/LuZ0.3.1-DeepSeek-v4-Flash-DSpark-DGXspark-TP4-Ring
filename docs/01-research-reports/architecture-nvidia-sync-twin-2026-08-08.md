# NVIDIA Sync twin 机制与四机直连架构判定

**日期**：2026-08-08
**工作流**：系统设计（架构复核）
**参与成员**：Archi（架构师）、Rex（SRE 工程师）、主理人（编排与汇编）

---

## 📌 TL;DR（执行摘要）

- 整体结论：**ConnectX-7 的 twin 双逻辑口是硬件/Socket Direct 架构固化**，不是 NVIDIA Sync 软件配置的产物；**"spine 功能"（200G 口拆 2×100G）在 DGX 网卡上不可启用**；**四台 DGX 全直连官方不支持**（仅支持 2/3 台直连，4 台必须走交换机）。
- 严重度分布：🔴严重 1 项（四机直连拓扑超出官方支持矩阵）/ 🟠高 1 项（spine 功能不存在，用户目标需调整）/ 🟡中 2 项（H 线非标走线、03 module0 未协商）/ 🟢低 1 项（ARP 行为干扰探测）
- 阻塞 / 非阻塞：**阻塞**——用户的目标拓扑（口A 拆 2×100G 交叉 + 口B 对角 200G）依赖不存在的 lane split 能力。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（官方方案） / 直连方案 🔴 不推荐 |
| 阻塞项数量 | 1（四机直连超官方矩阵） |
| 关键行动项 | 4 条（见行动清单） |
| 建议下一步 | 首选 200G 交换机方案（QM9700/Spectrum）实现 4 节点 TP4；若坚持直连仅作非标实验 |

---

## 🔍 判定正文

### 1️⃣ twin 来源：硬件/Socket Direct 架构固化（与 Sync 无关）

**用户质疑**："module 1 = 2 个 200G 逻辑口——twin 并行并不是一开始就拆好的，这是靠 NVIDIA 软件设置的结果，请分析 NVIDIA Sync 的 clusters 功能。"

**判定结论：twin 是 ConnectX-7 in DGX Spark 的固定硬件架构，NVIDIA Sync 只做管理面配置。**

证据链（三重独立印证）：

| 证据 | 来源 | 结论 |
|------|------|------|
| devlink 四口 `splittable false`，flavour physical | 四机实测 | 驱动层即 4 个独立物理实体，无软件聚合 |
| dmesg 每逻辑口独立 PCIe5 x4（126 Gb/s） | 03 实测 | twin 在 PCIe 枚举层就是独立 Function |
| netplan 99-nvidia-sync-cluster.yaml 仅 IP + MTU 9000 | 01 实测 | 无 bond/team/聚合配置，不存在软件"造双口" |

**NVIDIA Sync clusters 功能实际做了什么**（官方文档 docs.nvidia.com/sync/0.97.6/cluster-assistant.html + 系统实测）：
- 自动发现设备、校验 SSH/sudo/用户名
- 生成 netplan（IP 分配 + MTU 调优）——落地为 `99-nvidia-sync-cluster.yaml`
- 配置节点间 SSH、检查链路
- **不创建、不修改端口模式**——卸载 Sync 后 4 个逻辑口照常存在

### 2️⃣ "spine 功能"不可启用（明确结论）

**用户诉求**：能否像交换机一样把 200G 口拆成 2×100G（MFS1S90 的 cross-connect 能力）。

**判定：不可启用。**

- 该卡的呈现就是"每个物理 QSFP = 2×100G 逻辑口"——**这本身就是"已拆分"形态**，且不可再拆、也不可合并为单 200G 逻辑口
- devlink `splittable=false` + 固件无 `LINK_TYPE_P1/P2`（IB 锁定）+ `MODULE_SPLIT` 在此 SKU 无效
- **任何软件手段（mstconfig / NVIDIA Sync）都改不了**

### 3️⃣ 四机直连拓扑：官方不支持，实测部分生效但脆弱

**官方支持矩阵**：
- 直连：仅支持 2 台、3 台
- **4 台：必须通过交换机**
- "Use only one cable per link. Connecting two devices with two cables will not improve performance."

**当前实测**（23:48）：
- ✅ 01↔02 module1 双 200G UP；03↔04 module1 双 200G UP（直连线已恢复）
- ✅ 01/04 module0 双 100G UP（H 线部分生效）
- ❌ 03 module0 有线但 No partner（链路不完整）

**为什么部分能通**：twin 逻辑口各自独立训练，Y/H 拆分线可把同模块 2 个 100G 引到不同对端——但**属非标走线，NVIDIA 未验证**。03 module0 No partner 即为明证。

**即便 6 边全通的代价**：
- NCCL 可建全 mesh 环，但**环带宽受最慢边（100G）限制 → TP4 有效跨节点带宽≈100G/方向**而非 200G
- RoCE 直连无交换机 PFC，burst 下存在丢包风险
- `arp_announce=2 / arp_ignore=1`（NVIDIA 默认路由器风格 ARP）继续干扰跨口探测

### 4️⃣ 关键决策记录（ADR）

| ADR | 决策 | 理由 |
|-----|------|------|
| ADR-1 | 四机 TP4 首选走 200G 交换机 | 官方支持矩阵 + PFC 保障 + TP4 带宽稳定 |
| ADR-2 | 直连方案降级为非标实验 | 仅当无法获得交换机时；接受 100G 瓶颈 |
| ADR-3 | 停止寻找 twin/spine 软件开关 | 硬件固化，不存在 |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 与用户确认：是否可引入 200G 交换机（QM9700/Spectrum）实现官方支持的 4 节点拓扑 | 主理人 | P0 | 待用户决策 |
| 2 | 若坚持直连：修复 03 module0（换线/查 lane），确保 6 边全 UP 后再跑 NCCL allreduce 实测 | Rex | P1 | 待用户决策后 |
| 3 | 停止 mstconfig/NVIDIA Sync 改端口模式的尝试（已证实无效） | 全员 | P1 | 立即 |
| 4 | 直连实验结论标注"非标、不用于生产"，正式环境走交换机方案 | Docu | P2 | 方案确定后 |

---

## ⚠️ 待完善 / 已知局限

- 03 module0 未协商的具体 lane 原因未完全定位（线缆/插头/交叉段 12↔34 的精确映射）
- H 线（MFS1S90 定制 DAC）4 插头当前物理分布未与用户最终确认
- 交换机方案未实测（需采购/接入 QM9700 或 Spectrum 后才能验证 TP4）

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：四问题判定全文（twin 硬件固化 / spine 不可启用 / 四机直连官方不支持 / 最终建议）——本报告正文依据
- **Rex（SRE 工程师）**：H 线交叉段 lane 分析（12↔34 交叉破坏 twin 配对）与 Socket Direct twin 修正判定
- **主理人实测**：devlink / dmesg / netplan / mstconfig / ethtool / sysctl 系统证据；四机 carrier 矩阵
- **NVIDIA 官方文档**：docs.nvidia.com/sync/0.97.6/cluster-assistant.html、spark-clustering.html、MFS1S90 产品规格

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
