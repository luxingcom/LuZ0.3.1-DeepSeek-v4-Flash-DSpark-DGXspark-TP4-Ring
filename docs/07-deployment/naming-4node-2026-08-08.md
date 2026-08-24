# DGX 四机集群统一命名规范（node01~04）

**日期**：2026-08-08
**工作流**：系统设计（命名规范 + 执行变更）
**参与成员**：Archi（规范设计）/ Rex（影响面评估）/ Docu（Runbook 更新）

---

## 📌 TL;DR（执行摘要）

- 整体结论：四机命名从 4 套混乱体系（hostname/工作机别名/sync 别名/Prometheus 标签）统一为唯一主标识 **node01~04**（小写），全链路一致
- 严重度分布：🔴严重 0 项 / 🟠高 1 项（hostname 修改联动脚本校验）/ 🟡中 0 项 / 🟢低 3 项（文档/标签/兼容别名）
- 阻塞 / 非阻塞：**非阻塞**——执行期间 TP2 服务全程健康（8001=200），benchmark 不受影响

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过 |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（旧别名 3 个月后清理 / NVIDIA sync 观察 / Runbook 复核） |
| 建议下一步 | benchmark 补跑完成后继续组 B（55+59）部署 |

---

## 🏷️ 统一命名规范（Archi 设计）

**核心原则**：唯一主标识 = DGXspark0N（NVIDIA sync 官方编号），全链路统一为其**小写 dgxspark0N**（hostname/ssh 别名/Prometheus 标签/脚本全用小写，Linux 惯例且避免大小写歧义）。NVIDIA sync UI 显示名与编号映射**不动**。

### 命名对照表（旧→新全量）

| DGXspark编号 | 新hostname | 管理IP | 新ssh别名 | 旧hostname | 旧ssh别名 | 角色 |
|------|----------|--------|-----------|------------|-----------|------|
| 01 | node01 | <NODE_IP> | node01 | spark-05cd | aicad-server60 | head（TP2 rank0） |
| 02 | node01 | <NODE_IP> | node01 | edgexpert-0c69 | aicad-server | worker/监控 |
| 03 | node01 | <NODE_IP> | node01 | gx10-3f4d | gx10-55 | embed/组B head |
| 04 | node01 | <NODE_IP> | node01 | gx10-31c4 | gx10-59 | embed/组B worker |

**不改**：容器名（vllm-envE-node/worker）、RoCE IP（<RING_SUBNET>）、管理 IP（<NODE_IP>~<NODE_IP>）、NVIDIA sync 编号映射。

### 变更范围
1. **hostname**（四机 hostnamectl 三态同改 + /etc/hosts 127.0.1.1 同步）✅
2. **工作机 ssh config**：主别名 node01~04 + 旧别名 deprecated 注释保留（3 个月后移除）✅
3. **四机内部 ssh config**：新增 RoCE 对端别名（旧节点末段→node01、node01→node01），NVIDIA sync 生成的 DGXspark0X 块保留不动（防 sync 重写覆盖）✅
4. **Prometheus 标签**：machine=node=dgxspark0N（10 处统一，4 台 8/8 up）✅
5. **脚本**：start_head_v026r.sh hostname 校验 spark-05cd→node01；start_v026r_cluster.sh HEAD_HOST/WORKER_HOST→node01/02 ✅
6. **文档**：file-registry（四机同步 + 本地副本）、Runbook（Docu 更新中）✅

---

## 🔍 影响面评估（Rex 实机核查）

### 关键发现
- **start_head_v026r.sh L21 hostname 硬校验** `[ "$(hostname)" = "spark-05cd" ]`——改名后不联动会导致 TP2 无法启动（已同步修改）
- NVIDIA sync **无常驻服务/进程/hostname 配置**（dgx-oobe-hostname disabled）→ 改名不会被 sync 回滚
- TP2 运行中容器不受 hostname 修改影响（容器 hostname 独立）；编排脚本别名已同步，重启也不会失败

### 风险与对策
| 风险 | 对策 | 状态 |
|------|------|------|
| sync 覆盖 hostname | 已核实 sync 不管理 hostname（OOBE disabled） | ✅ 排除 |
| 脚本校验失配 | 先改脚本校验再改 hostname | ✅ 已执行 |
| Prometheus 历史 series 断裂 | 标签统一为 dgxspark0N，面板按 node 分线自动适配新值（历史线断裂属预期） | ✅ 已接受 |
| 内部别名被 sync 重写 | 新增块独立于 sync 块（CreatedBy 标记块未动） | ✅ 已规避 |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 旧别名（aicad-server/aicad-server60/gx10-55/gx10-59/DGXspark0X 大写）3 个月后移除 | 主理人 | P3 | 2026-11-08 |
| 2 | 观察 1 个 NVIDIA sync 同步周期，确认不重写别名/hostname | Rex | P2 | 下次 sync 后 |
| 3 | Runbook v1.3 命名复核（Docu 完成后 grep 复查旧命名残留） | Docu | P2 | 今日 |
| 4 | 组 B（55+59）部署脚本引用新命名（node01/04） | 主理人 | P1 | benchmark 补跑后 |

---

## ⚠️ 待完善 / 已知局限

- Grafana 面板历史 series 按旧 node 标签（head-60/node01-58 等）断裂，新数据按 dgxspark0N 分线——历史曲线不可归并，属预期行为
- 工作机旧别名 deprecated 注释已加，但**文档/历史记录中的旧命名**（Runbook 旧版、历史日志）不追溯
- NVIDIA sync 桌面端若重新运行集群同步，可能重写四机内部 ssh config 的 CreatedBy 块（手工新增块不受影响）

---

## 📚 数据来源 & 成员产出索引

- Archi（架构师）：统一命名规范 + 对照表 + 风险对策（engineering-node-naming inbox 06:16）
- Rex（SRE）：全量引用面清单（14 项）+ 执行顺序 + 回滚方案（engineering-node-naming inbox）
- 实测数据：四机 hostname 修改前后验证、Prometheus 4 台 8/8 up、TP2 8001=200 全程健康

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
