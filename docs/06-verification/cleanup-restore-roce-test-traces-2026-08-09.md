# DGX Spark 集群 RoCE 测试痕迹清理与还原报告（2026-08-09）

**日期**：2026-08-09
**工作流**：还原清理（事故响应 + 技术债评估混合）
**参与成员**：Rex（SRE）、Cody（代码审查师）、Archi（架构师）、Docu（文档师）、Tessa（测试专家，NCCL 方案进行中）

---

## 📌 TL;DR（执行摘要）

- 整体结论：**错误路线的测试改动已全部还原**（2 跳转发配置、探测 IP、临时 sysctl、MTU），**拍板事项已持久化**（内网限 TCP → netfilter-persistent、rp_filter → sysctl.d），03 ConnectX-7 **自行恢复可见**（固件未动，符合用户指示）。
- 严重度分布：🔴 0 项 / 🟠 0 项 / 🟡 2 项（QoS PFC 持久化待批准窗口、MTU 9000 决策待确认）/ 🟢 已完成项不计。
- 阻塞 / 非阻塞：**非阻塞**。生产链路（embed/litellm/TP2）全程未触碰；1 跳 RoCE 验证通过。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（还原完成 + 持久化完成，2 项待用户批准） |
| 阻塞项数量 | 0 |
| 关键行动项 | 5 条（见行动清单） |
| 建议下一步 | 用户批准 QoS PFC 维护窗口 → 执行 mlnx_qos 持久化并重启验证；Tessa NCCL RING 方案到位后实测 01↔04 对角延迟 |

---

## 🔄 一、服务器端还原清单（Phase 1，Rex 执行）

### 1.1 复核已还原项（Task #7 完成后无残留）

| 项 | 01 | 02 | 04 | 验证方式 |
|---|---|---|---|---|
| ip_forward | 1（docker 正常值，保留） | **0 ✅** | 1（docker 正常值，保留） | cat /proc/sys |
| proxy_arp | 0 ✅ | 0 ✅ | 0 ✅ | 三台 cat=0 |
| rp_filter | 1 ✅ | 1 ✅ | 1 ✅ | 三台 cat=1 |
| FORWARD UDP4791 放行 | 0 条 ✅ | 0 条 ✅ | 0 条 ✅ | iptables -S grep -c=0 |
| raw NOTRACK 4791 | 0 条 ✅ | 0 条 ✅ | 0 条 ✅ | iptables -t raw grep -c=0 |
| 2 跳静态路由 | 0 条 ✅ | 0 条 ✅ | 0 条 ✅ | ip route grep via=0 |
| busy_read/busy_poll | 0/0 ✅ | 0/0 ✅ | 0/0 ✅ | /proc 读取 |

### 1.2 本次新还原项

| 项 | 原始值 | 还原动作 | 验证 |
|---|---|---|---|
| 01 叠加探测 IP（<NODE_IP>/30、<NODE_IP>/30） | 无（netplan 仅 10.100） | ip addr del | ✅ 剩 <NODE_IP>/<RING_SUBNET> |
| 02 叠加探测 IP（<NODE_IP>/30、<NODE_IP>/30） | 无 | ip addr del | ✅ 剩 <NODE_IP>/<RING_SUBNET> |
| 04 netdev_budget_usecs | 2000 | sysctl -w 2000 | ✅ cat=2000 |
| 04 module0 双口 MTU | 1500 | ip link set mtu 1500 | ✅ cat=1500 |
| tcp_low_latency / netdev_budget（01/02/04） | 0 / 300 | 早前已回退，复核无残留 | ✅ |

### 1.3 03 设置层面一致化（与 01/02/04 对齐）

| 项 | 03 原值 | 目标 | 结果 |
|---|---|---|---|
| rp_filter | 2 | 1 | ✅ |
| 内网限 TCP（4 口 INPUT/OUTPUT DROP TCP） | 无 | 8 条规则 | ✅ 已加 |
| FORWARD policy | DROP | DROP | ✅ 一致 |
| netplan | <NODE_IP>/<RING_SUBNET> | 保留 | ✅ |
| 生产容器 | — | 未触碰 | ✅ anemll-embed-8022 Up |

### 1.4 保留项（非测试改动）

- ip_forward=1（01/04）：docker 自动设置，非中转用途，保留
- 02↔04 段 <NODE_IP>/13（02）+ <NODE_IP>/14（04）：1 跳链路唯一 IP（无 netplan 替代），保留；MTU 已还原 1500 两侧一致

---

## 💾 二、拍板事项持久化清单（Phase 2，Rex 执行）

| 项 | 落点机制 | 动作 | 验证 |
|---|---|---|---|
| 内网限 TCP（01/02/03/04） | **netfilter-persistent + iptables-persistent 1.0.20**（四台 apt 安装） | netfilter-persistent save | ✅ rules.v4 生成、systemd enabled、01 reload 测试通过 |
| rp_filter=1 | /etc/sysctl.d/99-sec.conf（四台） | sysctl -p 生效 | ✅ |
| QoS mlnx_qos（PFC/DSCP） | systemd oneshot（方案已出） | **未执行**——需维护窗口重启验证 | ⏸️ 待用户批准 |
| isolcpus（01/02） | /etc/default/grub.d/90-isolcpus.cfg | 确认存在且 /proc/cmdline 生效 | ✅ |
| GLOO 管理口 | start_head_v026r.sh:69 GLOO_SOCKET_IFNAME=enP7s7 | 只读确认（未改 <INSTALL_DIR>） | ✅ |

**不持久化项**：netdev_budget/tcp_low_latency/netdev_budget_usecs（实测对 RDMA 无改善，已还原默认，符合用户判断）。

---

## 🧹 三、本机工作区清理（Cody 盘点 + 执行归档）

- **A 类（临时/一次性，已归档）**：**133 个文件 / 827KB** 移入 `_archive_scratch/trash-2026-08-09/`——含 _probe_*（10）、_cap_*（4）、_accept_*（7）、_scratch_*（68+）、_depcheck/_grafana_patch_tput、状态标记（8）、`0` 文件、__pycache__、_audit_20260807（31 文件）、deliverables 下 3 份 superseded raw。
- **凭证迁移**：`_verify_topology.sh` 等含明文密码 `PW='<PASSWORD>'` 的脚本已归档，密码已备份至 `.workbuddy/secrets/README.md`（chmod 600）。⚠️ 建议轮换该密码（曾明文暴露）。
- **B 类（证据文件，保留）**：根目录 `_tessa_abcdef_raw`、`_tessa_def_bench_raw`、`_tessa_final_baseline_raw`、`_archive_scratch/bench_B`、deliverables 下 `_tessa_*_raw`——被 10+ 份正式报告引用为原始证据，**原位保留不移动**（移动会断引用）。
- **C 类（生产，保留）**：`_start_embed_8022.sh`（生产启动脚本，⚠️ 建议改名去掉 `_` 前缀避免误删）、deliverables 正式 .md 报告、.workbuddy/、生产脚本。
- 沙箱安全删除保护说明：环境禁止直接删除（回收站不可用），采用**归档移动**方案，可逆、可恢复。

---

## 🩺 四、03 ConnectX-7 状态（用户指示：固件不动）

- **设备已自行恢复可见**：lspci 4 个 PCI 功能、4 个 IB 设备（rocep1s0f0/.1、roceP2p1s0f0/.1）、4 个 netdev 全部正常。
- MODULE_SPLIT：mlxconfig 可识别设备但参数为 Array[0..15]，**Secure FW 禁止读取 NV config 明细**；未执行任何写入/reset（用户禁止固件级操作 + 风险高）。**结论：设备已恢复，建议不执行 reset**。
- 03 两个 200G 口均未插线（用户确认），**接线规划留空，等用户通知**。
- MFT 备注：mst_pci 内核模块与 kernel 6.17.0-1029-nvidia 不匹配（modprobe FATAL），不影响 mlxconfig 直读设备。

---

## 🧭 五、2 跳 RDMA 决策记录（用户拍板）

| 项 | 决策 |
|---|---|
| 2 跳 RDMA 转发方案（A 双卡分工 / C 按需切换） | **放弃**（用户：关闭所有转发功能） |
| 硬件限制（查证） | ConnectX-7 的 RoCE 为**整卡级**开关（roce_enable sysfs / devlink enable_roce，NVIDIA 官方文档 + 社区确认）；单卡无法"既跑 RDMA 又透明转发 4791"；per-port 控制不存在（devlink port function roce 仅 VF/SF 且禁用时全设备 GID 一起禁） |
| 对角通信路线 | **NCCL RING 1 跳中继**（环网/链式下对角数据由 2×1 跳硬件 RoCE 完成，预期 µs 级 vs 软件转发 63µs）；Tessa 测试方案设计中 |
| 02 mlx5_ib | **已恢复加载**（TP2 能力恢复，4 个 RoCE 设备 PORT_ACTIVE）；01↔04 2 跳不再通为预期行为 |
| 内网限 TCP | 保留（用户批准）+ 已持久化 |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 执行 QoS mlnx_qos PFC/DSCP 持久化（systemd oneshot）——需用户批准维护窗口 + 重启验证 | Rex | P1 | 用户批准后 |
| 2 | Tessa NCCL RING 01↔04 对角延迟测试方案落地 + 实测（前置：方案回传、Rex 配置已还原） | Tessa/Rex | P1 | 方案到位后 |
| 3 | 轮换 sudo 密码 <PASSWORD>（曾明文暴露于已归档脚本） | 用户 | P1 | 尽快 |
| 4 | `_start_embed_8022.sh` 改名去掉 `_` 前缀（避免误删）| 用户/主理人 | P2 | 下次维护 |
| 5 | 确认 10.20.0.x 段 MTU：维持 1500（已还原）或恢复 9000（200G 链路性能） | 用户 | P2 | 待决策 |

---

## ⚠️ 待完善 / 已知局限

- QoS mlnx_qos 持久化**未执行**（需重启验证，等用户批准窗口）；PFC 需 L1/L2 双端一致配置
- 03 MODULE_SPLIT 当前值无法读取确认（Secure FW），默认按"设备已恢复、未改动"处理
- 03 未接线：接线后需补 netplan/iptables 检查
- iptables 规则后续调整需重新 `netfilter-persistent save`
- 2 跳带宽补测（ib_write_bw -R）遗留：2 跳方案已放弃，此补测不再需要
- 本机 trash 归档 133 文件尚未物理删除（沙箱回收站不可用），确认无需后可手动清理

---

## 📚 数据来源 & 成员产出索引

- Rex（SRE）原始产出：Task #1 结论（03 设备恢复可见/未写入）、Task #7 执行摘要（关闭 2 跳转发 9 项）、Task #2 盘点表（A/B/C 分类）、Phase 1 还原清单表、Phase 2 持久化清单表；落盘证据 .workbuddy/tmp/sre-rex/out_*.txt
- Cody（代码审查师）原始产出：Task #3 本机 202 项只读盘点（A/B/C 分类 + 凭证警告 + 引用链核查）
- Archi（架构师）原始产出：Task #5 结构完整性评估（5 维 + 落点矩阵 + 禁区清单 + 还原顺序）、Task #6 三方案判定（A 双卡分工/B 物理直连/C 按需切换）
- Docu（文档师）原始产出：Task #4 RoCE 优化成果总结（另见 roce-optimization-summary-2026-08-09.md）
- Tessa（测试专家）：Task #8 NCCL RING 测试方案（进行中）
- 官方资料：NVIDIA MLNX_EN RoCE 文档（roce_enable 整卡级）、NVIDIA 开发者论坛（Disable RoCE Processing）、Broadcom KB 396622、netdev 内核补丁（devlink port function roce）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
