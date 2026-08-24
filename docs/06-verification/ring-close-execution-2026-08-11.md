# 环网闭环执行与验证报告（参照官方 playbook + 社区案例）

**日期**：2026-08-11
**工作流**：工作流 4（部署前检查执行）/ 变更实施与验证
**参与成员**：Rex（实施与验证）、Zhen（汇编）
**状态**：✅ 部署门禁全部通过（P1×3 闭环 + P2 落地），遗留 4 项后续项

---

## 📌 TL;DR（执行摘要）

- **P1×3 全部闭环并真实验证**：①03—01 段 RoCE IP 补齐（<RING_SUBNET>，四口互 ping 0% 丢包）②四台 /etc/hosts 全解析 ③01/02 内存给出明确结论（TP4 前须降 max-num-seqs/清缓存）。
- **重大发现**：四台 iptables 白名单（INPUT+OUTPUT）只放行旧相邻段 IP → 新链路 TCP 全被 DROP，已修复并写入 rules.v4 持久化 —— 这是历史网络问题的同类根因，本次彻底根治。
- **iperf3 四段双向 99-110Gbps、重传 0-6；压测后 16 口错误计数全 0** → 物理层 P1 **正式解除**。
- QoS DSCP trust 已在 01/03 新口生效（与既有段一致）；01/02 docker runtime 已对齐（待维护窗口重启生效）。
- 残留 4 项：QoS 开机持久化脚本、docker 重启生效、TP4 前内存处理、NCCL 4 机 all_reduce 维护窗口验证。**管理网全程未动，SSH 正常。**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 部署门禁通过（TP4 可推进，前处理内存项） |
| P1 闭环 | 3/3（物理层 P1 已正式解除） |
| P2 落地 | 3/3（iperf3+防火墙、runtime、QoS） |
| 意外发现 | 🔥 iptables 白名单阻断新链路（已根治） |
| 残留 | 4 项（均非阻塞，见行动清单） |
| 回滚 | 备份于 <INSTALL_DIR>/backup/ring-fix-20260811/ |

---

## 🔧 执行逻辑（阶段 0-7）

| 阶段 | 动作 | 改动文件 | 验证 | 结果 |
|---|---|---|---|---|
| 0 | 备份+基线快照 | 四台 netplan/hosts/daemon.json/rules.v4 → <INSTALL_DIR>/backup/ring-fix-20260811/<host>/ | 基线错误计数全 0 | ✅ |
| 1 (P1-1) | 01/03 的 97-roce-mtu.yaml 追加 <RING_SUBNET>（/24，MTU 9000 与既有段一致） | /etc/netplan/97-roce-mtu.yaml | netplan apply 成功；四口互 ping 0% 丢包；ip neigh 有对端；169.254 清除 | ✅ |
| 2 (P1-2) | 四台 /etc/hosts 追加 node01~04=<NODE_IP>~<NODE_IP> | /etc/hosts | getent hosts 全返回管理网 IP | ✅ |
| 3 (P1-3) | 01/02 内存评估（free/ps/docker stats） | - | used 110/121Gi、available ~10G、swap 已用 2.7G/4.5G | ⚠️ 结论见下 |
| 4 (P2) | iperf3 四段双向压测（装 iperf3） | - | **发现防火墙阻断→修复→四段 99-110Gbps、重传 0-6；16 口错误计数仍全 0** | ✅ |
| 5 (P2) | 01/02 daemon.json 追加 runtimes.nvidia（对齐 03） | /etc/docker/daemon.json | jq 校验通过、与 03 diff 一致；docker 未重启 | ✅ |
| 6 (P2) | mlnx_qos --trust dscp（01/03 新口） | 运行时配置 | dcb app show 有完整 dscp-prio 映射（与 136/139 段一致）；未配 PFC | ✅ |
| 7 | 全环验证：8 对口 ping 矩阵 + ip neigh + 容器内 NCCL 可用性 | - | ping 全通、neigh 无 FAILED；torch2.11+NCCL2.28.9 可用、RoCE TCP 通 | ⚠️ all_reduce 未跑（见残留） |

## 📍 IP 规划落定表（全链路 MTU 9000）

| 段 | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| 01↔02（f1） | <NODE_IP> / <RING_SUBNET> 侧 | <NODE_IP> / <RING_SUBNET> 侧 | - | - |
| 02↔04（f0） | - | <NODE_IP> / <NODE_IP> | - | <NODE_IP> / <NODE_IP> |
| 04↔03（f1） | - | - | <NODE_IP> / <RING_SUBNET> 侧 | <NODE_IP> / <RING_SUBNET> 侧 |
| **03↔01（f0）新增** | **<NODE_IP> / <RING_SUBNET> 侧** | - | **<NODE_IP> / <RING_SUBNET> 侧** | - |

## 📊 iperf3 四段实测（双向，-P4）

| 段 | 正向 | 反向 | 重传 |
|---|---|---|---|
| 01↔02 | 106 Gbps | 109 Gbps | 0 / 0 |
| 02↔04 | 110 Gbps | 109 Gbps | 2 / 0 |
| 04↔03 | 110 Gbps | 108.5 Gbps | 0 / 6 |
| 03↔01（新） | 108 Gbps | 110 Gbps | 0 / 0 |

> 注：iperf3 单流受 CPU 限制无法打满 200G，四段对称 ~110G 且重传≈0，说明链路质量良好、无瓶颈段。

## ✅ P 项闭环判定

| 项 | 判定 | 证据 |
|---|---|---|
| P1-1 环网 L3 闭合 | ✅ 闭环 | 环网新段对端接口四口互 ping 0% 丢包 |
| P1-2 hostname 解析 | ✅ 闭环 | 四台 getent 全返 192.168.5.x |
| P1-3 01/02 内存 | ⚠️ 结论 | TP4 部署前须降 max-num-seqs 或清 buff/cache；available ~10G、swap 换页趋势，直接扩 TP 有 OOM 风险 |
| P2-1 错误率 P1 | ✅ **正式解除** | iperf3 压测后 16 口 port_xmit_discards/symbol_error/local_link_integrity_errors 全 0 无增量 |
| P2-2 runtime 对齐 | ✅ 配置闭环 | jq 通过、与 03 一致；维护窗口 restart docker 生效 |
| P2-3 QoS DSCP | ✅ 生效 | dcb app show 完整映射，未配 PFC（符合直连无交换机原则） |

## 🔥 重大发现：iptables 白名单阻断新链路

- 四台 iptables（INPUT+OUTPUT）为**白名单模式**，只放行旧相邻段 IP（如 <RING_SUBNET>、10.20.0.x 对端），新链路（<RING_SUBNET>）TCP 全部 DROP → iperf3 初测失败暴露。
- 已修复：放行 <RING_SUBNET> 段，写入 /etc/iptables/rules.v4（持久化）。
- **启示**：与历史 <NODE_IP>:5000 不通现象高度同类（交换机 ACL 疑云），本次为防火墙层实锤；后续新增链路必须同步放行 iptables，纳入变更清单。

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 | 验收口径 |
|---|------|---------|--------|---------|---------|
| 1 | QoS DSCP trust 开机持久化（udev/systemd 脚本固化 mlnx_qos，防重启丢失） | SRE | P1 | 1 周内 | 重启后 dcb app show 仍完整 |
| 2 | 维护窗口：restart docker 生效 runtime（02 业务容器编排重启验证） | SRE | P1 | 下个维护窗口 | docker info 显示 nvidia runtime；容器全恢复 |
| 3 | TP4 前处理 01/02 内存（降 max-num-seqs 12→8 / 清 buff-cache），明确资源账 | SRE + 用户 | P1 | TP4 部署前 | 部署后 swap 不增长 |
| 4 | 维护窗口：新建测试容器跑 4 机 NCCL all_reduce（容器需 --device /dev/infiniband + memlock） | SRE | P2 | 下个维护窗口 | ncclCommInitRank 成功、busbw 达标 |
| 5 | iptables 变更纳入标准流程：任何新 RoCE 网段配 IP 时同步放行 | SRE | P2 | 流程固化 | 变更清单模板含 iptables 项 |

## ⚠️ 待完善 / 已知局限

- NCCL 4 机完整 all_reduce 未在本窗口执行（03/04 embed 生产容器无 /dev/infiniband 且不宜动）；TCP/RoCE 层已验证，NCCL 层验证留给维护窗口测试容器。
- iperf3 受 CPU 限制 ~110G，非线速 200G；链路健康度以"四段对称+重传≈0+错误计数 0"为准。
- QoS 持久化未固化（残留 1）。
- 回滚方案：备份于 <INSTALL_DIR>/backup/ring-fix-20260811/；netplan 可 revert，其余恢复备份文件后 apply/restart。

---

## 📚 数据来源 & 成员产出索引

- Rex（实施与验证）：teammate-message sre-engineer @ 2026-08-11（阶段 0-7 全量实测：netplan apply/ping/ip neigh/getent/free/iperf3/ethtool 计数/dcb/daemon.json jq+NCCL 容器探测）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
