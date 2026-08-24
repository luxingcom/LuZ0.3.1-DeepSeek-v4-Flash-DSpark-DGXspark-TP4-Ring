# DGX Spark TP4 四机集群全面工程审查 + 交接文档落实综合报告

**日期**：2026-08-20
**工作流**：全覆盖工程审查（代码+架构+SRE+测试）→ 落实交接文档
**参与成员**：科迪 Cody（code-reviewer）/ 阿奇 Archi（architect）/ 雷克斯 Rex（sre-engineer）/ 泰莎 Tessa（testing-expert）/ 多库 Docu（tech-writer）
**督导**：甄宇航 Zhen（engineering-director，主理人）
**团队**：engineering-audit-handoff（并发 ≤2）

---

## 📌 TL;DR（执行摘要）

- 整体结论：**🟡 有条件通过（CONDITIONAL PASS）**。TP4 集群生产功能已实测稳定，NVFP4 路线A 主路径已打通（8/8 正确性 rel=0.00141、60~187 TFLOPS、零构建）；余下 P0/P1/P2 为**工程化收尾**而非方案不确定。已产出可让新团队直接接管的交接文档。
- 严重度分布：🔴严重 6 项 / 🟠高 10 项 / 🟡中 12 项（四位成员合并去重后）。
- 阻塞项：无硬阻塞（不阻断已上线运行）；🟡有条件项集中在**可复现性与密钥卫生（S1/S2/S3）、生产恢复与自愈未就绪（R1/R2/R3）、验收测试框架未自动化（G1/G2/G3)**。
- 最优先动作：🔴 修密钥泄漏 → 🔴 消除 shim 源码/二进制漂移 → 统一网段权威表 → P0 持久化落 <INSTALL_DIR> → 恢复自愈。具体见交接文档 §5。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过 |
| 🔴 严重项 | 6（科迪 S1/S2/S3 + 雷克斯 R1/R2/R3；泰莎 G1/G2/G3 并入 P0/P1 待办） |
| 阻塞项数量 | 0（不阻断已上线运行） |
| 关键行动项 | P0×5 + P1×8 + P2×8（见交接文档 §5） |
| 需人类裁决项 | 5（见附录） |
| 建议下一步 | 新团队按 `handoff-tp4-cluster-2026-08-20.md` 十章 + 待办清单接管执行，优先清 P0 |

---

## 🔍 审查发现汇总（四位成员合并去重，按严重度排序）

### 🔴 严重（第一优先）

| # | 来源 | 类别 | 对象 | 问题描述 | 建议修复 |
|---|------|------|------|---------|---------|
| 1 | Cody | 安全 | start_tp4_head.sh:77 / worker.sh:76 | **API key 明文泄漏**到启动日志（64-hex 生产密钥） | 打日志脱敏 / 改环境读取 / 日志 chmod 600（P0） |
| 2 | Cody | 可复现性 | ncclpin.c vs libncclpin.so | **shim 源码↔二进制↔文档三处漂移**：kit 内 .c 是 v3（NCCL→0-4），.so 是 v8（→8-9），按源码重编会错绑核 | .c 升 v8 并验证可复现（P0） |
| 3 | Cody | 配置 | netplan / hosts | **三套网段口径未统一**（192.168.5 控制面 / 10.100+10.20.0 数据面），netplan 未覆盖 10.20.0.x MTU | 建单一权威 IP 表 + 同步 netplan（P0） |
| 4 | Rex | 恢复 | 生产 4 rank | **4 rank 未恢复**（GPU 0%），"已上线"仅表示容器拉起非可用 | 走 start_tp4_cluster.sh head-first，禁单机 docker run（P0） |
| 5 | Rex | 可靠性 | monitor/healthcheck | **自愈机制 disable**（08-19 事故期间被关，timer is-enabled 与记录矛盾），集群失去宕机自愈 | 还原 monitor+healthcheck，补 avail<2G 告警（P0） |
| 6 | Rex | 持久化 | /vllm-workspace | **非持久=容器重建即丢**（RPO=∞） | P0 落 <INSTALL_DIR> 后 RPO=0 |

### 🟠 高

| # | 来源 | 对象 | 问题 | 修复 |
|---|------|------|------|------|
| 7 | Cody | 防火墙 | 默认 ACCEPT，未收敛 DROP 白名单 | 评估向 DROP 收敛 + 监控栈白名单 |
| 8 | Cody | prometheus.yml | 仍旧 .58/.60 TP2 拓扑，与生产 8191 漂移 | 同步 TP4 拓扑 + 8191 retention/告警闭环 |
| 9 | Cody | ssh 编排 | 缺 StrictHostKeyChecking/ServerAlive，复合命令拼接解析风险 | 加 ssh 参数 + 命令复用优化 |
| 10 | Cody | start_tp4_cluster | 无 set -e 主路径失败语义 | 强化 set -e |
| 11 | Cody | NCCL_IB_PEER_HCA | 固定 2048 缓冲 + atoi 无校验（低危） | 加边界校验 |
| 12 | Rex | UMA 内存 | **耗尽复发风险**（util 回 0.80，03/04 头寸仅~2.5G） | 恢复 0.70 + avail 告警 + 降 max-num-seqs 后手 |
| 13 | Rex | 监控 | 告警盲区（job 仍旧命名、抓 188:8001 无效、冻死时 Prom 中断） | 修正 job 命名/目标 + 告警覆盖 |
| 14 | Rex | 配置 | 漂移复发（rank 曾颠倒、文档 10+ 处失准） | 以运行态实测为准，NCCL 锚点 b7784b49 |
| 15 | Rex | monitor | 误伤（停机须先停 timer/service） | 固化停机 SOP |
| 16 | Tessa | bench_big | 无 200TFLOPS 断言（PASS/FAIL 缺失） | 加 200 断言门禁 |
| 17 | Tessa | compare_v15 | 用 bf16 matmul 代理非真 v15 | 改真 v15 对照 |
| 18 | Tessa | sass_gate.sh | 仍含 tcgen05（与 ADR-4 冲突） | 改 mma.*e2m1\|mmaf |

### 🟡 中（要点）

- M1 持久化未落 <INSTALL_DIR> / M2-M3 适配层多入口状态耦合 + 每次全量反量化 / M4 isolcpus 与 shim v3 注释语义冲突 / M5 告警规则加载未闭合 / M6 工作区 103 个 tmp_* 残留爆炸 / R8 双 Grafana 未收敛 / R9 .local-backup 删除暂缓 / R10 明文凭据待轮换 / R11 01 时区漂移 / R12 kvssd 不可行不得重启 / G4 kernel① edge 缺失 / G5 见上 / G6 sass 历史脚本陈旧 / G7 自愈+告警+4rank 无回归 / G8 safety 3 脚本缺陷 / G9 routeA 无确定/泄漏测试 / G10 evidence 空。

---

## ✅ 行动清单（关键 5 条，完整见交接文档 §5）

| # | 行动 | 负责角色 | 紧急度 | 完成判据 |
|---|------|---------|--------|---------|
| 1 | 修 API key 日志泄漏（start_tp4_head.sh:77/worker.sh:76） | 运维/接手团队 | P0 | 日志无明文 key，chmod 600 |
| 2 | 消除 shim 源码↔二进制漂移（ncclpin.c 升 v8） | 运维/接手团队 | P0 | .c 与 .so 版本一致，重编可复现 |
| 3 | NVFP4 routeA/v17 落 <INSTALL_DIR> 持久化 | 接手团队 | P0 | 容器重建后 import 成功 |
| 4 | 恢复自愈 monitor+healthcheck + avail<2G 告警 | Sand/运维 | P0 | 自愈探活恢复，RTO 25-40→5-8min |
| 5 | 统一四机网段权威表 + 同步 netplan | 接手团队 | P0 | 单一 hosts/netplan 映射，MTU 全覆盖 |

---

## ⚠️ 待完善 / 已知局限

- 四份成员报告与交接文档均以**工作区文档审阅 + 已有运行态证据**为基础；本次为只读审查，**未连接生产集群执行实时命令**，部分判断依赖已有 report/handoff 记录，最终以运行态实测复核为准。
- 需人类负责人/用户最终裁决的 5 项：①200TFLOPS 是否硬门槛（当前峰值 187 差 7%）；②生产 4 rank 是否恢复；③0.70/0.80 UMA 取舍；④密钥轮换时机；⑤v15 1.5× 对照真实性。
- G1/G2/G3 等验收脚本未落，在补齐前不应宣称"投产验收通过"。

---

## 📚 数据来源 & 成员产出索引

- **主文件·交接文档**（Docu）：`deliverables/engineering-assurance/handoff-tp4-cluster-2026-08-20.md`（十章+附录，347 行）
- 科迪（代码审查）：`code-review-cluster-2026-08-20.md` 🟡有条件
- 阿奇（架构审查）：`architecture-nvfp4-2026-08-20.md`（5 条 ADR）
- 雷克斯（SRE 可靠性）：`sre-ops-reliability-2026-08-20.md` 🟡有条件
- 泰莎（测试策略）：`testing-strategy-2026-08-20.md` 🟡就绪度
- 资料库引用：`deliverables/engineering-assurance/nvfp4-landing/`（README + docs×6 + kernel1 + tests 全量，32 文件）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。