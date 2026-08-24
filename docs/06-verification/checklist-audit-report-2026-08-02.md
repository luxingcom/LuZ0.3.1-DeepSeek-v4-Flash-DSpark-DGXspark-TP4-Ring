# 问题分析报告落实核查报告：DeepSeek-V4-Flash 双 DGX Spark 集群

**日期**：2026-08-02
**工作流**：工作流 4（部署前检查）+ 工作流 3（事故响应）交叉核查
**参与成员**：Cody / Archi / Rex / Tessa / Docu 全员
**核查基准**：engineering-review / incident / hardened 上线前置 / live-execution 四份报告的全部行动项

---

## 📌 TL;DR

- 整体结论：**🔴 未全部落实——核心运行时修复已闭环，但上线前置类（监控/告警/门禁/回滚/digest）大面积未落地，当前不具备生产 8000 切换条件**。
- 核查范围：4 份报告约 20 项行动项（去重合并后），5 位成员分工核验（真机 23:40 实测数据为证）。
- 严重度分布：🔴 P0 阻塞 5 项 / 🟠 P1 应补 6 项 / 🟡 P2 跟踪 4 项。
- 完善工作（本轮同步开展）：工具链已部署双机、digest pin 已完成、smi_server 暴露面已收敛、预热收敛观测运行中——**4 项 P0/P1 已闭环**。
- 阻塞 / 非阻塞：**8000 切换仍被阻塞**（告警规则未加载、vLLM 指标未采集、Grafana 面板缺失、回滚未演练、预热未收敛）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🔴 不通过（有条件修复中：4/5 P0 已处理） |
| 阻塞项 | 5 项 P0（2 项已闭环，3 项待补） |
| 关键行动项 | 15 条（本轮已完成 4 条） |
| 建议下一步 | 补齐告警/采集/面板/回滚/预热 → 复审 → 8000 切换 |

---

## 🔍 核查结果总表（5 成员去重合并，按严重度排序）

### 🔴 P0（阻塞 8000 生产切换）

| # | 行动项 | 状态 | 证据 | 来源 |
|---|--------|------|------|------|
| 1 | Prometheus 4 条告警规则加载并验证链路 | ❌→⏳ | 8191 仅有 cad-backend.rules 5 条；规则文件已部署 `~/dspark-tools/` 待加载 | Rex |
| 2 | vLLM 指标采集（Prometheus vllm job） | ❌ | targets 无 vllm job、vllm 指标数=0 | Rex |
| 3 | vLLM Grafana 专属面板 | ❌ | 仅 aicad-v17-backend 面板 | Rex/Cody |
| 4 | 预热收敛复测（5 并发 ≥80 t/s + TTFT） | ⏳ | 66 t/s 爬升中，收敛观测运行中 | Tessa/Rex |
| 5 | 回滚演练（实测 RTO） | ❌ | 锚点已存，无受控演练 | Rex/Tessa |

### 🟠 P1（上线前应补）

| # | 行动项 | 状态 | 证据 | 来源 |
|---|--------|------|------|------|
| 6 | 镜像 digest pin（ADR-0011） | ✅ **本轮已闭环** | run_container.sh + start_*_fix.sh 已 pin（head 4dbbedda8bc6 / worker b763d81b57f7） | Cody/Archi |
| 7 | smi_server 暴露面收敛（ADR-0010 部分） | ✅ **本轮已闭环** | 0.0.0.0:8088 → 127.0.0.1（备份+重启验证） | Cody |
| 8 | 工具链部署真机（preflight/run_checklist/wait_converge/bench） | ✅ **本轮已闭环** | 双机 `~/dspark-tools/` 9 文件 | Tessa/Rex |
| 9 | preflight.sh 双机跑通 | 🟡 工具已部署，未执行 | — | Rex |
| 10 | 启动顺序自动化（deploy.sh） | ❌ | 仍人肉 worker→12s→head | Cody |
| 11 | 启停/重启 + 故障排查 + 监控接入 + 变更管理 Runbook | ❌ | 4 文档未补 | Docu |
| 12 | 凭据移出 + Grafana 口令轮换 | 🟡 移出✅ 轮换未做 | /tmp 无残留；口令未轮换 | Cody/Rex |

### 🟡 P2（跟踪项）

| # | 行动项 | 状态 | 证据 | 来源 |
|---|--------|------|------|------|
| 13 | P1 补测（长输出/900K/5 并发 B/C 131K/acceptance） | ❌ | 全部未做 | Tessa |
| 14 | --api-key 接入（ADR-0010） | ❌ | 未接 | Archi |
| 15 | RoCE 故障注入（ADR-0012） | ❌ | 未执行 | Archi |
| 16 | 错误档案踩坑手册化 + 0.8/0.85 对照补测 | ❌ | 未做 | Docu/Tessa |

### ✅ 已落实（不阻塞）

- served-model-name = deepseek-v4-flash-0731（进程实参 + /v1/models 双验证）
- GPU_MEM = 0.85（KV 池 1.44M ≈ 基准 C 1.47M）
- 容器加固：Restart=unless-stopped、head/worker 均 healthy、日志卷挂载
- MASTER_PORT = 25000 真机实证（PARAMS.md DEC-01 已修正）
- hostname/NODE_RANK 双守卫（start_*_fix.sh）
- SHA256 双清单落盘（DSpark + 真机 0731）
- 参数基线 PARAMS.md + ADR 0007-0012 + 决策留痕
- bench 工具修复（空 content 跳过 / --check）

---

## 🛠️ 本轮完善工作记录（2026-08-02 凌晨执行）

| # | 完善项 | 操作 | 结果 |
|---|--------|------|------|
| 1 | 工具链部署 | scp 到双机 `~/dspark-tools/`（head 9 文件 / worker 2 文件） | ✅ |
| 2 | digest pin | docker images --digests 获取双机 digest → 更新 run_container.sh + start_*_fix.sh | ✅ |
| 3 | smi_server 收敛 | `/opt/aicad/monitoring/smi_server.py` 0.0.0.0→127.0.0.1（备份 .bak-20260801，setsid 重启） | ✅ 仅本机可访问 |
| 4 | 预热收敛观测 | head 后台跑 wait_until_converged.sh（≤90min） | ⏳ 运行中 |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 加载 prometheus-alerts.yml 4 条规则 + 配置 vllm job 采集 | Rex + Cody | P0 | 1 天 |
| 2 | 创建 vLLM Grafana 面板（uid 模板化） | Rex + Cody | P0 | 2 天 |
| 3 | 回滚演练一次（实测 RTO，8001 暂存环境） | Rex + Tessa | P0 | 2 天 |
| 4 | 预热收敛后复测 5 并发 ≥80 t/s + TTFT 定标 | Tessa | P0 | 收敛后 |
| 5 | preflight.sh 双机跑通（作为 8000 切换门禁） | Rex | P1 | 1 天 |
| 6 | 补 4 份 Runbook（启停/故障排查/监控接入/变更管理） | Docu | P1 | 3 天 |
| 7 | deploy.sh 部署真机（启动顺序自动化） | Cody | P1 | 1 天 |
| 8 | Grafana 口令轮换 + smi_server Token | Rex | P1 | 1 天 |
| 9 | --api-key 接入 + RoCE 故障注入（ADR-0010/0012） | Archi + Rex | P2 | 下轮 |

---

## ⚠️ 待完善 / 已知局限

- 核查基于 23:40 真机快照 + 落盘文档；预热收敛、告警加载等动态项以最终复测为准。
- 告警规则加载与 vllm job 配置需 Prometheus 容器配置权限（AICAD 共享实例，改动需评估影响）。
- Grafana 口令轮换涉及 AICAD 共享凭据，需与 AICAD 负责人协调。
- 生产 8000 切换、--api-key、RoCE 故障注入属变更类操作，未在本轮执行（独立决策）。

---

## 📚 数据来源 & 成员产出索引

- Rex（SRE）：运维 10 项核查表（✅1 🟡2 ❌6 ⏳1）+ P0/P1/P2 分级
- Cody（代码审查）：代码/安全 8 项核查表（✅2 🟡3 ❌3）+ P0/P1/P2 分级
- Tessa（测试专家）：测试 6 项核查表（✅2 🟡2 ❌2）+ 交接门禁判定
- Archi（架构师）：ADR 9 项核查（✅2 ❌4 🟡2 ⏳1）+ 风险排序
- Docu（文档师）：文档 11 项核查表（✅4 ❌5 🟡2）
- 真机实测：SSH 命令输出（容器/进程/端口/监控/凭据快照）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
