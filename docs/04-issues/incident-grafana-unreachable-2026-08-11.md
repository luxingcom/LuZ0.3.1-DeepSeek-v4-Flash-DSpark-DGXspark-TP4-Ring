# 事故调查：Grafana（<NODE_IP>:3000）外部不可达

**日期**：2026-08-11
**工作流**：工作流 3（事故响应 / 分诊）
**参与成员**：Rex（分诊、P0 修复）、Zhen（汇编）
**状态**：✅ **已修复（2026-08-11 晚）** — P0 修复验证通过，3000/8191 外部恢复 200

---

## 🔧 修复记录（2026-08-11 晚，Rex 执行）

- 执行：`docker compose restart grafana prometheus alertmanager redis postgres`（/opt/aicad compose 7 容器；**vllm worker 不在 compose 内，推理链路未受影响**）
- 结果：外部 <NODE_IP>:3000 refused → **200**（02 自测 + 01 跨机均 200）；8191 000 → **200**；DNAT 全部指向当前容器实际 IP（3000→172.18.0.8、8191→172.18.0.6:9090 等）
- 持久化：`iptables-save > /etc/iptables/rules.v4`（mtime 13:30:10 UTC）覆盖错位快照
- **白名单澄清**：10.100.140/141 并非漏配——140/141 为 01 侧接口网段，01 白名单已含 140.2/141.2（peer=03/04 侧），各机放行各自 peer，ring-close 白名单意图正确，02 无需补规则（分诊时视角误判已纠正）
- 回归：四段环网 ping 全 OK；Grafana /api/health=database ok v13.1.1、/api/search 返回 3 个 dashboard 正常

---

## 📌 TL;DR（执行摘要）

- **症状定性**：Grafana 页面外部打不开（Connection refused），但**服务/数据/面板全部健康**（127.0.0.1:3000=200、/api/health=ok、面板列表完整、Prometheus 数据源 health=OK）→ 用户"功能丢失"= 页面层故障，非面板/数据丢失。
- **根因**：iptables **DOCKER 链 DNAT 容器 IP 错位**——docker daemon 8/10 16:31 UTC 重启后容器 IP 重新分配（grafana=172.18.0.7、prometheus=172.18.0.8），但 DOCKER 链仍指向历史 IP（:3000→172.18.0.8:3000 实为 prometheus 无监听 → RST refused；:8191→172.18.0.6:9090 实为 redis → 同样 refused；9093/6379/8082 同错位）。
- **与 ring-close 强相关**：今晚 ring-close 的 iptables-save/restore 把这套**错位快照持久化**（rules.v4 mtime 今日 12:52 UTC = 持久化时刻）。
- **假象成因**：127.0.0.1:3000=200 是 docker-proxy（userland）直连真实容器绕过 DNAT → 本机自检正常、外部打不开；8/11 早 200 自检很可能是 127.0.0.1。
- 附带发现：生效规则疑似遗漏 10.100.140/141 白名单（待修复时统一核实）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🔴 未修复（P0，外部访问 Grafana 全断） |
| 症状层面 | 页面层（服务/数据/面板健康） |
| 根因 | iptables DNAT 容器 IP 错位（docker 重启后 IP 漂移 + 错位快照被持久化） |
| 影响面 | 3000/8191/9093/6379/8082 端口映射全错位 |
| 关联变更 | ring-close iptables 持久化固化错位快照 |
| 修复 | docker compose restart 重建规则 + 立即 iptables-save 覆盖 |

---

## 🕐 时间线

| 时间（UTC+8） | 事件 |
|---|---|
| 8/10 16:31 UTC | docker daemon 重启，容器 IP 重新分配（21h 前） |
| 8/11 白天 | 127.0.0.1:3000 自检 200（docker-proxy 直连，未暴露外部问题） |
| 8/11 晚 ring-close | iptables-save/restore 持久化**错位 DNAT 快照**（rules.v4 mtime 12:52 UTC） |
| 8/11 21:2x | 用户报告仪表盘功能丢失 → 分诊 |

## 📍 影响范围

- **受影响**：外部访问 <NODE_IP>:3000（Grafana）、:8191（Prometheus 对外）、:9093、:6379、:8082 全部 refused；Windows 本机与跨节点均不可达。
- **未受影响**：Grafana 服务本身、面板数据、Prometheus 取数、容器内互访（bridge 内正常）、推理链路。

## 🔍 根因分析（证据链）

| 层 | 问题 | 证据 |
|---|---|---|
| 现象 | 外部 3000 refused，本机 127.0.0.1:3000=200 | 双路径 curl 对比 |
| Why1 | DOCKER 链 DNAT 指向错误容器 IP | rules.v4：:3000→172.18.0.8（实为 prometheus）、:8191→172.18.0.6（实为 redis）；容器实际 IP：grafana=172.18.0.7、prometheus=172.18.0.8 |
| Why2 | docker 重启后容器 IP 漂移，DOCKER 链快照未随之更新 | docker 8/10 16:31 UTC 重启，容器 Up 21h |
| Why3 | 错位快照被 iptables-save/restore 持久化固化 | rules.v4 mtime 今日 12:52 UTC（ring-close 时刻） |
| Why4 | 自检用 127.0.0.1 掩盖问题（docker-proxy 绕过 DNAT） | 127.0.0.1=200 vs 外部=refused |
| 结论 | **修复**：docker 按当前 IP 重建 DOCKER 链 + 覆盖错误持久化快照 | Rex 分诊 |

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 验收口径 |
|---|------|---------|--------|---------|
| 1 | `cd /opt/aicad && docker compose restart`（02 监控栈容器按当前 IP 重建 DOCKER 链）；**谨慎选项** systemctl restart docker（会中断 21h 服务，不推荐） | SRE | P0 | 外部 <NODE_IP>:3000=200、:8191=200 |
| 2 | 修复后**立即** `iptables-save > /etc/iptables/rules.v4` 覆盖错位快照，防 reboot/reload 复发 | SRE | P0 | rules.v4 含正确 DNAT |
| 3 | 持久化改造：rules.v4 只含自定义白名单，restore 流程不得覆盖 docker 动态 DOCKER/DOCKER-FORWARD 链；compose 端口映射容器固定 ipv4_address 防漂移 | SRE + Archi | P1 | reboot 后 3000 仍可达 |
| 4 | 健康检查改用对外 IP（<NODE_IP>:3000/8191）；补外部可达性监控 | SRE | P2 | 监控告警覆盖 |
| 5 | 核实 10.100.140/141 白名单是否漏配（修复时统一 iptables-save 核对） | SRE | P2 | 140/141 段 reboot 后仍通 |

## ⚠️ 待完善 / 已知局限

- 未执行修复（本报告为分诊结论）；compose restart 对 02 上 vllm worker 的影响面需先确认（worker 由 start_v026r_cluster.sh 管理，独立于 compose，预计无影响，执行前复核）。
- 140/141 白名单疑漏配与 iperf3 四段实测通过存在矛盾，执行修复时以 iptables-save 实际内容为准统一核对。

---

## 📚 数据来源 & 成员产出索引

- Rex（分诊）：teammate-message sre-engineer @ 2026-08-11（双路径 curl、/api/health、/api/search、数据源 health、docker inspect 容器 IP、iptables -L/rules.v4 比对、docker systemctl 时间戳）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
