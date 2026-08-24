# NCCL TCP 放行语义判定 + 03 重启持久化复验（2026-08-09）

**作者**：Archi（架构师）· 工程保障团队
**性质**：只读分析 + 只读复验（未执行任何变更操作）
**复验数据源**：SSH 直连 01/02/03/04 实测（2026-08-09 UTC 10:00-10:20）

---

## 📌 TL;DR

- **判定①（部分符合，有条件确认）**：数据面放行 `<NODE_IP>/24` TCP 是 TP2 多节点 NCCL **必需**（NCCL 控制面走 TCP 且当前脚本把 `NCCL_SOCKET_IFNAME` 指向数据面）；但这是对"内网纯 RoCE（数据面 0 TCP）"的**语义削弱**，且当前放行范围偏宽（整 /24 而非对端主机 IP）并**漏掉 <NODE_IP>/24 子网**。
- **判定②（有更优方案）**：将 `NCCL_SOCKET_IFNAME` 从数据面改为管理口 `enP7s7`（与本集群 nccl-tests matrix **已验证通过**的配置一致），NCCL 控制 TCP 即可走管理网、数据面恢复纯 RoCE → 可撤销数据面 TCP 放行、恢复原语义。NCCL 无独立 `NCCL_BOOTSTRAP_IF` 变量，控制面接口选择就是 `NCCL_SOCKET_IFNAME`。
- **判定③（风险）**：数据面 TCP 放行后，<NODE_IP>/24 间任意 TCP（含误用 bulk，如 scp/rsync 走数据面 IP）可重回 RoCE 链路，重演 benchmark 实测的 **+128% 延迟**（60% TCP 背景 → 7.46µs vs 基线 3.27µs）。缓解 = 收窄到对端主机 IP + 覆盖 <RING_SUBNET> 双子网 + bulk 约定走管理网。
- **03 持久化复验**：**全部通过/符合预期**——isolcpus=16-19 ✅、netfilter-persistent（文件+服务）✅（规则内容需 sudo 复核）、mlnx-qos 未装=预期 ✅、MTU 9000 ✅、sysctl 99-sec ✅、embed 8022 重启后自动恢复 ✅、litellm 池 2 上游 03/04 均在 ✅。
- **附加发现**：03 当前 **hotplug 为 DISABLED**（无 `/etc/nvidia/cx7-hotplug-enabled`），而 01/02/04 为 ENABLED——这与 03 无缆仍可见 CX7 的现象一致；与我此前报告"建议 03 保持热插拔启用"的稳态建议不一致，需团队决策（详见 §5）。

---

## 一、判定①：数据面放行 <NODE_IP>/24 TCP 是否符合预期

### 1.1 结论：**有条件确认**（必要，但属语义削弱 + 范围偏宽）

| 维度 | 评估 |
|------|------|
| 必要性 | **必需**。NCCL 多节点 TP2 的初始化（bootstrapInit）强制使用 TCP 控制环（监听 socket + peerProxy + P2P 地址交换），且当前生产脚本 `NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1` 把控制面明确指向数据面 IP（<NODE_IP>/<RING_SUBNET>）。TCPStore 25000 已放行但 NCCL 内部握手端口（实测 49917）未放 → comm init 挂起。**不放行则 TP2 无法 init** |
| 语义符合度 | **部分削弱**。原"内网纯 RoCE"= 数据面 0 TCP（用户批准，为保护 RDMA 延迟）。现变为"数据面允许 TP2 控制 TCP"。控制面流量很小（KB 级），与"60% TCP 背景压测"的量级完全不同，实际延迟影响有限 |
| 范围问题 | ① 当前 `<NODE_IP>/24` 只覆盖 <RING_SUBNET> 单子网，**漏掉 <NODE_IP>/24**（01/02 的 module1 twin 逻辑口分别在 <RING_SUBNET> 双子网）；NCCL 可能任选其一，存在"今天通、下次不通"的隐患。② 整 /24 放行 = 允许该段任意 TCP，非仅 NCCL/TP2 控制端口 |

### 1.2 判定依据（证据链）

1. **生产脚本指向数据面**：`start_head_v026r.sh` / `start_worker_groupB.sh` / `start_head_groupB.sh` 中 `NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1`（数据面），`VLLM_HOST_IP`/`MASTER_ADDR` 均为 10.100.x。
2. **实测阻塞点**：NCCL bootstrap 走 `<NODE_IP>:49917`（TCP），被数据面 DROP 规则拦截 → comm init 挂起；与"TCPStore 25000 已放行但 NCCL 内部握手端口未放"一致。
3. **可对照证明（同一集群已验证）**：`benchmark-nccl-cpu-pin-4core-matrix` 中 nccl-tests 使用 `NCCL_SOCKET_IFNAME=enP7s7`（管理口）+ `--mca oob/orte/btl_tcp_if_include enP7s7` **成功跑通跨节点 2-rank all_reduce**（RoCE 数据面照常工作）——证明"控制面走管理、数据面纯 RoCE"在本集群可行。

---

## 二、判定②：是否有更优方案保持"限 TCP"语义

### 2.1 结论：**有更优方案**（推荐 Phase-2 验证后采纳）

| 方案 | 做法 | 数据面 TCP | 语义 | 风险/成本 |
|------|------|-----------|------|-----------|
| **现状（Rex 已实施）** | 数据面放行 <NODE_IP>/24 TCP | 允许 | 削弱 | 已生效、改动小；但范围偏宽、漏 137 子网 |
| **更优（推荐验证）** | vLLM 脚本 `NCCL_SOCKET_IFNAME` 改 `enP7s7`（管理口），数据面 TCP 放行撤销 | 仍为 0 TCP | **恢复纯 RoCE** | 需 staging 验证 vLLM TP2 全流程；控制面走管理 1GbE（控制流量小，无瓶颈）；与 matrix 已验证配置一致 |

### 2.2 关键答疑（团队点名问题）

- **"NCCL_SOCKET_IFNAME 为何未覆盖 direct-connect"**：它**覆盖**了——NCCL 的 bootstrap/direct-connect TCP 正是通过 `NCCL_SOCKET_IFNAME` 选定的接口建立（实测走 <NODE_IP>）。问题不是"未覆盖"，而是**该变量被脚本显式指向数据面**。若改为 `enP7s7`，bootstrap 即走管理网。
- **"vllm TP2 场景是否有 NCCL_BOOTSTRAP_IF 类参数"**：**没有**独立控制面变量。NCCL 控制面接口选择 = `NCCL_SOCKET_IFNAME`（数据面 IB 由 `NCCL_IB_HCA` 独立指定，二者可分离）。`NCCL_SOCKET_FAMILY` 只控制 IPv4/IPv6，不控制选卡。
- **能否只改 vLLM 不改防火墙**：可以。`NCCL_SOCKET_IFNAME=enP7s7` + 保留数据面 TCP DROP，即"更优方案"。需确认 vLLM `VLLM_HOST_IP` 相关的 TCPStore（25000）已放行（是，已放行）；ZMQ/mq 若走数据面则需单独评估（见 2.3）。

### 2.3 遗留确认点（建议 staging 验证）

- vLLM 自身 ZMQ/mq 控制（`VLLM_HOST_IP=<NODE_IP>`）是否也产生跨节点 TCP？若是，需在"纯 RoCE"下也放行对应端口，否则 TP2 控制链路仍会被 DROP。**这是从"现状方案"切到"更优方案"前必须实测确认的点**。

---

## 三、判定③：当前放行方案的风险与最终规则形态

### 3.1 风险清单

| 风险 | 说明 | 缓解 |
|------|------|------|
| **延迟回退** | 数据面 TCP（尤其 bulk）与 RoCE 数据同链路 → 实测 +128% 延迟（60% 背景） | 收窄放行 + bulk 约定走管理网 + 监控数据面 TCP 流量 |
| **137 子网缺口** | 当前只放 136，NCCL 若用 137 侧逻辑口则仍被 DROP（时好时坏） | 补 137（或合并为 <NODE_IP>/23） |
| **放行范围过宽** | 整 /24 放行 = 该段任意 TCP 可用，非仅 NCCL 控制 | 收窄到对端主机 IP（.1<->.2），必要时限 dport（25000 + NCCL 高端口段） |
| **B-group 未覆盖** | 03/04（<RING_SUBNET>）成环时若沿用同脚本，需同样规则 | 成环前在 03/04 补放行（或直接切更优方案） |

### 3.2 建议的最终规则形态（现状方案收窄版，01/02）

```bash
# —— 数据面（RoCE 口）控制 TCP：仅放行 TP2 对端主机 ——
# 01 上（对端 02 = <NODE_IP> / <NODE_IP>）
iptables -I INPUT  -i enp1s0f1np1   -s <NODE_IP>/32 -p tcp -j ACCEPT
iptables -I INPUT  -i enp1s0f1np1   -s <NODE_IP>/32 -p tcp -j ACCEPT
iptables -I INPUT  -i enP2p1s0f1np1 -s <NODE_IP>/32 -p tcp -j ACCEPT
iptables -I INPUT  -i enP2p1s0f1np1 -s <NODE_IP>/32 -p tcp -j ACCEPT
# 02 上对称（对端 01 = <NODE_IP> / <NODE_IP>）
# ……（OUTPUT 侧同理；ESTABLISHED,RELATED 已前置放行，保留）
# 其余 TCP 仍落回数据面 DROP
```

> 说明：
> - 若希望进一步限定"仅 NCCL 控制"，可加 `--dport 25000`（TCPStore）+ NCCL 高端口段（实测 49917 属动态高端口，NCCL 无固定端口；如需严格限定需确认端口段，成本较高）。**最稳妥的长期方案仍是判定②（NCCL_SOCKET_IFNAME→enP7s7）。**
> - 03/04 成环（<RING_SUBNET>）时同样处理；若采纳更优方案则 03/04 无需此放行。

---

## 四、03 重启持久化复验清单（只读实测）

03 于 2026-08-09 ~09:37 UTC 重启（rescan 副作用后），以下为重启后实测：

| # | 复验项 | 期望 | 实测 | 判定 |
|---|--------|------|------|------|
| 1 | `isolcpus=16-19`（cmdline） | `/proc/cmdline` 含 isolcpus/nohz_full/rcu_nocbs=16-19 | ✅ 含 `isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19`；`/sys/devices/system/cpu/isolated`=16-19；nproc=16 | ✅ PASS |
| 2 | netfilter-persistent 规则 | `/etc/iptables/rules.v4` 存在 + service enabled | ✅ 文件存在（1864B，8/9 04:48）；`netfilter-persistent` enabled | ⚠️ **文件+服务通过；规则内容需 sudo 复核**（非 root 不可读，见注1） |
| 3 | mlnx-qos | 03 未装 = 预期 inactive/not-found | ✅ `inactive` + `not-found` | ✅ 符合预期 |
| 4 | MTU 9000（netplan 97-roce-mtu） | 4 个 RoCE 口 mtu=9000 | ✅ 4 口均 9000；`/etc/netplan/97-roce-mtu.yaml` 存在（8/9 05:37） | ✅ PASS |
| 5 | sysctl 99-sec（rp_filter=1） | 文件存在 + 生效 | ✅ `99-sec.conf` 存在（rp_filter=1 all/default）；live `rp_filter`=1 | ✅ PASS |
| 6 | embed 8022 恢复 | 容器 Up + 监听 + 可服务 | ✅ `anemll-embed-8022` Up（started 09:37Z, restarts=0）；`0.0.0.0:8022` 监听；`/v1/models` 返回 Qwen3-Embedding-0.6B | ✅ PASS（重启后自动恢复，无 crash-loop） |
| 7 | litellm 池 2 active（03/04 都在） | config 上游指向 03/04 且两端 embed 可用 | ✅ 02 的 `config.yaml`：`local-embedding` upstream = <NODE_IP>:8022（03）+ <NODE_IP>:8022（04）；03/04 embed 容器均 Up、/v1/models 正常 | ✅ PASS（litellm API 鉴权下无法直接读 pool 计数，以配置+两端健康为准） |

> **注1**：rules.v4 为 `root:root 640`，非 root 无法读取；`sudo` 需密码。请 **Rex 用 sudo 复核 03 的 rules.v4 内容**：应含 8 条 RoCE 口 INPUT/OUTPUT DROP TCP + FORWARD policy DROP（以及是否已含 ESTABLISHED,RELATED——按现状 03 未跑 TP2，可不含，但需确认）。
> **注2**：其余项均无持久化失效，**未发现需列入修复清单的问题**。

---

## 五、附加发现：03 hotplug 状态与团队此前建议不一致

| 项 | 01 | 02 | 03 | 04 |
|----|----|----|----|----|
| `/etc/nvidia/cx7-hotplug-enabled` | **ENABLED** | **ENABLED** | **DISABLED（无文件）** | **ENABLED** |
| 当前 CX7 可见性 | 有缆 ACTIVE | 有缆 ACTIVE | 无缆但**可见**（4 PCI fn + 4 IB dev，carrier=0） | 有缆 ACTIVE |

- 03 无缆仍可见 = hotplug 被禁用的**预期结果**（与我此前社区报告"删除 flag 可禁用热插拔、设备恒可见"一致）。这也从反面印证了热插拔机制在此集群真实存在（01/02/04 均 ENABLED）。
- **与之前"建议 03 保持热插拔启用"不一致**：当前 03 实际处于"禁用热插拔（+18W、设备恒可见）"状态。此状态对**当前排障/联调有利**（设备可见、便于 lspci/RoCE 检查），且 03 待成环、无缆，省电意义有限。
- **建议**：03 在成环前的联调窗口**维持当前 DISABLED**（便于观测）；**成环插线后**由团队决策——若追求省电可 `sudo touch /etc/nvidia/cx7-hotplug-enabled` 重新启用（插线后设备自动恢复，无影响）；若追求可观测性可保持禁用（代价 +18W/台）。**四机一致性**：若最终决定统一，需在 01/02/04 同做（当前 01/02/04 均 ENABLED，建议先不动）。

---

## 六、行动清单（供团队参考）

| # | 行动 | 负责 | 优先级 |
|---|------|------|--------|
| 1 | Rex 用 sudo 复核 01/02 新放行规则 + 03 rules.v4 内容（8 条 DROP TCP + FORWARD DROP + 是否已含 ESTABLISHED/RELATED） | Rex | P1 |
| 2 | 决策：短期维持现状放行（需补 137 子网） vs 切更优方案（`NCCL_SOCKET_IFNAME=enP7s7` + 撤销数据面放行） | 用户/团队 | P1 |
| 3 | 若切更优方案：staging 验证 vLLM TP2 全流程（含 VLLM_HOST_IP 相关 ZMQ/mq 是否产生数据面 TCP）后，撤销 01/02 数据面 TCP 放行 | Rex + Tessa | P2 |
| 4 | 03 成环时：按 §3.2 在 03/04 补放行（或按更优方案直接切管理口）；同时完成 hotplug 状态决策 | 用户 + Rex | P2 |
| 5 | 更新监控：数据面 RoCE 口 TCP 流量计数纳入 Grafana（防误用 bulk 影响延迟） | 用户/团队 | P3 |

---

## 📚 数据来源

- 生产脚本：`start_head_v026r.sh`、`start_worker_groupB.sh`、`start_head_groupB.sh`（deliverables/engineering-assurance/）
- 历史报告：`network-roce-2hop-routing`、`roce-optimization-summary`、`cleanup-restore-roce-test-traces`、`benchmark-nccl-cpu-pin-4core-matrix`（2026-08-09）
- 实测：SSH 直连 01/02/03/04 只读检查（cmdline/sysfs/iptables 文件/docker/netplan/litellm config/CX7 hotplug flag）
- 官方：NCCL 用户指南 env 章节（`NCCL_SOCKET_IFNAME` 语义；无独立 bootstrap 控制面变量）
- 社区：NCCL 初始化流程源码剖析（bootstrapInit 控制环走 TCP）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
