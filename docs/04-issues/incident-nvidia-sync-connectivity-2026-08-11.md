# 事故调查：NVIDIA sync 软件连接失败（四机互联后）

**日期**：2026-08-11
**工作流**：工作流 3（事故响应 / 分诊调查）
**参与成员**：Rex（SRE 分诊、5 Why 根因展开）、Docu（文档审查）
**状态**：✅ **已闭环（2026-08-11 晚）** — 用户侧调整后 NVIDIA sync 已正常连接；200G 物理层连线检查另行推进（见 network-200g-physcheck-2026-08-11.md）

---

## 📌 TL;DR（执行摘要）

- 4 台 DGX Spark 集群计算链路全部正常（vllm/litellm/embed 均健康），**无业务中断**；"NVIDIA sync" 客户端连不上属功能不可用，定级 **SEV3**。
- 四台机**均未发现 nvidia-sync 进程/服务/软件包**；唯一相关项是 netplan `99-nvidia-sync-cluster.yaml`（RoCE IP 配置，非同步服务，命名来源存疑）。
- **首要根因（链 A）**：NVIDIA 原厂 `dgx-dashboard-service`（Dashboard）四机均**仅监听 127.0.0.1:11000**，外部任何客户端无法访问 —— 若 sync 软件指向 Dashboard 端口则必然连接失败。
- **次要根因（链 B/C）**：03 号机新接线物理已通（LLDP 证实直连 01）但 01↔03 module0 直连段**两端均无 L3 IP**；且四机 /etc/hosts 无主机映射，主机名会解析到不可路由的 fe80:: 地址。
- 历史遗留问题 <NODE_IP>:5000 TCP 不通**已自行恢复**（现全端口 OPEN），排除出根因。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🔴 未解决（已定位根因方向，待用户确认客户端目标） |
| SEV 评级 | SEV3（功能不可用，无业务中断） |
| 阻塞项数量 | 2（①Dashboard 仅绑 127.0.0.1 ②01↔03 新链路无 L3） |
| 关键行动项 | 3 条（P0×1 / P1×2，另 P2 常规项 3 条不计入关键项） |
| 建议下一步 | 用户确认「NVIDIA sync」客户端连接的目标地址/端口/凭据 → 针对性放通或隧道 |

---

## 🕐 事故时间线（来源标注）

| 时间（UTC+8） | 事件 | 来源 |
|---|---|---|
| 约 00:4x（8/11 凌晨） | 03 号机 module0 两口接线（dmesg Cable plugged + Link up + RoCE ACTIVE，距分诊约 18.9h 推算）⚠️ 与用户"白天接线"表述不符，**待核实** | dmesg 推算 / 用户 |
| 19:25 | 用户报告「NVIDIA sync」软件连接不上，启动事故响应 | 用户 |
| 19:26-19:50 | 雷克斯对四台服务器实地核查（SSH/GPU/进程/端口/网络/接线/LLDP），完成分诊 | Rex 核查 |
| 19:5x | Docu 审查报告，修订定稿 | Docu 审查 |

## 📍 影响范围

- **受影响**：外部访问 11000（DGX Dashboard）全部失败；01↔03 新互联段三层不通（仅物理层 UP）；主机名解析异常（getent 返回 fe80:: link-local，无法解析为管理网 IP）。
- **未受影响**：vllm 推理（8001 /health=200）、litellm 网关（4000）、registry（5000 /v2/ 200）、Grafana（3000）、Prometheus（8191）、embed（03/04 :8022 均 200）、A/B 组 RoCE 组内通信、管理网互 ping。
- **业务判断**：集群作业链路无任何告警或降级，事故仅限 sync/管理类功能。

## ⚠️ SEV 评级

**SEV3**（功能不可用，无用户数据/业务中断）
- 依据：NVIDIA sync 无法连接，但推理/网关/embed 全部正常；未确认 sync 为业务关键路径；无需紧急回滚或人工值守。

---

## 🔍 分诊发现

### 1. 四台服务器状态

| 节点 | SSH | GPU(GB10) | 关键服务 | 网络要点 |
|---|---|---|---|---|
| 01=<NODE_IP> | OK | 1×, 53°C, 0% | vllm-cluster active、8001/health 200 | 管理网通；RoCE <RING_SUBNET>/<RING_SUBNET> UP；module0 两口仅 169.254 |
| 02=<NODE_IP> | OK | 1×, 53°C, 0% | litellm:4000、registry:5000、Grafana:3000、Prom:8191 均 200 | 管理网通；RoCE <RING_SUBNET>/<RING_SUBNET>；<NODE_IP>/13↔04 通 |
| 03=<NODE_IP> | OK | 1×, 52°C, 0% | embed:8022 200、dcgm-exporter | **4×CX-7 口全 200G UP**；<RING_SUBNET>/<RING_SUBNET> 通 04；module0 两口无 IP 仅 fe80 |
| 04=<NODE_IP> | OK | 1×, 52°C, 0% | embed:8022 200 | 管理网通；<RING_SUBNET>/<RING_SUBNET> 通 03；<NODE_IP>/14↔02 通 |

磁盘：01 17% / 02 35% / 03、04 27%（健康）。内存（可用/总）：01、02 各约 110G+/121G（vllm 常驻），03、04 各约 13G。四机防火墙全空（无 ufw/iptables/nft 规则）。

### 2. NVIDIA sync 相关发现

- 四台机**无 nvidia-sync 同名进程 / systemd 服务 / dpkg 包**（`ps|grep sync` 仅 timesyncd/dcgm）。
- netplan 存在 **99-nvidia-sync-cluster.yaml**（01/02 为 328 字节、8/1 修改；03/04 为 248 字节、8/7 修改）—— 内容为 RoCE IP 配置，**命名来源存疑，非同步服务本身**。
- **dgx-dashboard-service（NVIDIA 原厂仪表盘）四机均仅监听 127.0.0.1:11000**（启动参数 `-port 11000 serve`，配置一致，无对外绑定改动）→ 外部不可达，若 sync 软件指向 Dashboard 端口必然失败。
- 01 上 50050/50051 为业务容器（v18-server/m17-model-conf），与 sync 无关。

### 3. 网络核实

- 管理网四机互 ping 全通；A 组 RoCE（<RING_SUBNET>）、B 组 RoCE（<RING_SUBNET>）组内互 ping 通。
- **03 号机接线确认完成（物理层）**：dmesg 显示 module0 两口 Cable plugged + Link up + RoCE ACTIVE；LLDP 证实 03 网卡口直连 01（口对口映射见下）；但 01 对应两口仅 169.254 link-local、03 两口仅 fe80，**均未配置 IP**（netplan 97-roce-mtu.yaml 仅设 mtu + dhcp4:false）→ 物理通、三层不通。

**01↔03 新互联段口对口映射（LLDP）**

| 03 端口 | 01 对端口 | 链路状态 | L3 状态 |
|---|---|---|---|
| enp1s0f0np0 | 01 module0 口 1 | 200G UP / RoCE ACTIVE | ❌ 无 IP（03 fe80 / 01 169.254） |
| enp2p1s0f0np0 | 01 module0 口 2 | 200G UP / RoCE ACTIVE | ❌ 无 IP（03 fe80 / 01 169.254） |

（注：网卡名以实际内核命名为准，本表统一小写）

- **历史问题 187:5000 已恢复**：从 Windows/01/03 探测 5000/4000/3000/8191 全部 TCP OPEN，registry /v2/ 返回 {}。
- 隐患：四机 /etc/hosts 无主机映射，`getent hosts dgxspark0X` 返回 fe80:: IPv6 link-local 地址，**主机名无法解析为管理网 IP**。

---

## 🎯 根因分析（5 Why，三条独立但可叠加的根因链）

### 根因链 A：NVIDIA sync 客户端连不上（无服务端对外可达）— 最直接

| 层级 | 问题 | 证据 |
|---|---|---|
| 现象 | Windows 侧「NVIDIA sync」软件连接失败 | 用户报告；Windows 探测 187:11000/8001 closed |
| Why1 | 连接目标端口在四机上无对外监听 | 四机 `ss -tlnp` 仅见 22/4000/5000/3000/8191/8022/11000；11000 仅绑 127.0.0.1 |
| Why2 | 疑似 sync 依赖的 DGX Dashboard（11000）只绑定回环地址 | 四机均：dashboard-service `-port 11000 serve`，`ss` 显示 `127.0.0.1:11000` |
| Why3 | DGX 原厂 dashboard 默认配置即 localhost-only，部署后未修改 | 四机 /opt/nvidia/dgx-dashboard-service 配置一致，netplan/服务文件无对外绑定改动 |
| Why4 | 集群无任何对外暴露的「sync 服务端」组件/进程 | `ps\|grep sync` 仅 timesyncd/dcgm；无 nvidia-sync 包/二进制/systemd 单元 |
| 结论 | sync 服务端未以对外可达方式运行：要么客户端目标端口本就不应直连该 dashboard，要么需显式配置绑定 0.0.0.0 或 SSH 隧道 | |

### 根因链 B：01↔03 新互联段三层不通 — 仅当 sync 依赖直连段时成立

| 层级 | 问题 | 证据 |
|---|---|---|
| 现象 | 03 号机接线后 01↔03 间 L3 不可达 | 03 无 10.20.0.x 地址；从 03 ping <NODE_IP>/10 均 FAIL |
| Why1 | 03 的 module0 两口只有 link-local，无静态 IP | 03 `ip -br addr`：enp1s0f0np0/enp2p1s0f0np0 仅 fe80；01 对应两口仅 169.254 |
| Why2 | netplan 97-roce-mtu.yaml 只写 mtu+dhcp4:false，未配 addresses | 01/03 均无 addresses 段（02/04 有 <NODE_IP>/10/13/14） |
| Why3 | 03 的 99-nvidia-sync-cluster.yaml（8/7 修改，248 字节）只含 RoCE <RING_SUBNET> module1 口，未纳入新接 module0 口 | 文件内容仅两个 module1 网卡口；01/02 版本（328 字节）才有 module0 |
| Why4 | 接线物理完成后未执行 netplan 配置更新 + apply | dmesg 显示 18.9h 前 Cable plugged/Link up，配置快照（8/7）早于接线且无后续改动 |
| 结论 | 03 号机 CX-7 接线为纯物理层完成；01/03 两端 module0 口缺 L3 规划，需补静态 IP 并 netplan apply | |

### 根因链 C：主机名解析异常（客户端按主机名连接必失败）

| 层级 | 问题 | 证据 |
|---|---|---|
| 现象 | 四机互访主机名 dgxspark0X 解析不到管理网 IP | `getent hosts node01~04` 返回 fe80:: IPv6 link-local |
| Why1 | /etc/hosts 未写入集群主机映射 | 四机 `grep -c dgxspark /etc/hosts` = 0（01/02）或仅本机 127.0.1.1（03/04） |
| Why2 | 解析依赖 mDNS/avahi，返回本接口 link-local 地址 | getent 结果全部为 fe80::/16 地址，对应各网卡 MAC |
| Why3 | 初始部署使用 IP 直连配置，跳过 hosts/DNS 规划 | 现有脚本/服务均以 192.168.5.x 或 10.100.x IP 硬编码（docker、netplan、vllm master-addr） |
| Why4 | 无内部 DNS，也未配置 resolv 指向任何解析服务器 | 各机仅见 127.0.0.53/127.0.0.54 本地解析，无集群 DNS 服务 |
| 结论 | 若 sync 客户端按主机名寻址，必然解析到不可路由的 fe80:: 而失败；需补 /etc/hosts 或部署内部 DNS | |

**交叉结论**：A（服务端暴露缺失）是最直接原因；C（主机名→fe80::）是客户端侧最常见的失败诱因；B（01↔03 L3 缺配）仅在 sync 依赖直连段时成立。三条链共同治理：P0 确认客户端连接方式（IP 或主机名、目标端口）→ P1 补 /etc/hosts + 01↔03 静态 IP + dashboard 绑定决策 → P2 轮换 sudo 密码。

## ✅ 行动清单（按优先级排序，含验收口径）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 | 验收口径 |
|---|------|---------|--------|---------|---------|
| 1 | **用户确认「NVIDIA sync」客户端实际连接的目标地址/端口/凭据**；触发条件：用户提供信息后升为执行态。若是 DGX Dashboard → 将 dgx-dashboard-service 绑到 0.0.0.0 或走 SSH 隧道；若是其他软件 → 按其文档放通 | 用户 + SRE | P0 | 待用户提供信息 | 确认到具体 IP:PORT 与协议；客户端可连通并完成握手 |
| 2 | 为 01↔03 module0 直连段补齐静态 IP（建议 <NODE_IP>/30×2）并 netplan apply | SRE | P1 | 用户确认后 1 次维护窗口 | 两端 IP 生效、01↔03 互 ping 通、ethtool 保持 UP |
| 3 | 四机 /etc/hosts 添加 192.168.5.x ↔ node01~04 映射 | SRE | P1 | 1 次维护窗口 | `getent hosts node01~04` 返回 <NODE_IP>~<MGMT_OCTET> |
| 4 | 核对 99-nvidia-sync-cluster.yaml 命名来源（确认是否 NVIDIA 官方软件残留，决定保留/重命名） | SRE | P2 | 常规 | 确认文件来源并记录决策 |
| 5 | 轮换 sudo 密码（secrets 标注暴露面） | 用户 + SRE | P2 | 常规 | 旧密码失效、secrets 文件已更新 |
| 6 | 监控 5000 端口稳定性（历史 ACL 问题是否复发） | SRE | P2 | 持续 | 连续 7 天无异常记录 |

## 🛡️ 预防措施

- 对外暴露类服务（Dashboard 等）启动后必须校验绑定地址（`ss -tlnp`），并纳入部署检查清单。
- 物理接线变更后必须执行"物理层 → L2 → L3 → 服务"四层验证，避免只确认 Link up 就宣布互联完成。
- 四机主机名解析纳入标准配置（/etc/hosts），防止 link-local 误解析。
- 建议把 11000 端口连通性加入 Prometheus 黑盒监控（probe_success）。

## ⚠️ 待完善 / 已知局限

- 「NVIDIA sync」软件的具体产品名/目标端点未确认 —— 本报告基于服务器侧可观测证据给出最可能根因，**需用户提供客户端连接目标以最终定论**。
- 01↔03 新直连段的 IP 规划未定（<NODE_IP>/30×2 为建议值，需用户/架构确认）。
- 03 号机接线时间点（dmesg 推算约 8/11 凌晨 vs 用户"白天接线"）存在出入，待核实。
- 03 号机接线后 B 组 RoCE（<RING_SUBNET>）仅验证 03↔04；01↔03 段三层不通，跨组 4 机闭环 RoCE 尚不可用。

---

## 📚 数据来源 & 成员产出索引

- Rex（SRE 分诊）原始产出：teammate-message sre-engineer-2 @ 2026-08-11（含已执行命令清单：ssh 四机 uptime/nvidia-smi/df/free、ps/systemctl/ss、docker ps、lspci/rdma link/ethtool/ip addr/dmesg/LLDP、netplan cat、curl 各服务端口、TCP 端口探测、getent hosts、防火墙检查）
- Rex（5 Why 根因展开）：teammate-message sre-engineer-2 @ 2026-08-11（根因链 A/B/C 三条，基于分诊原始数据）
- Docu（技术文档师）审查意见：teammate-message tech-writer @ 2026-08-11（5 项必须修改已全部落实，可选优化项已采纳 4 项）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
