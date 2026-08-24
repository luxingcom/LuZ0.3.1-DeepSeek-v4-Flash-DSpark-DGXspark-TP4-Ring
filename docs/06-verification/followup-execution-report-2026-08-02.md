# 收尾推进执行报告：双 DGX Spark 集群（2026-08-02）

**日期**：2026-08-02
**工作流**：工作流 4（部署前检查）+ 工作流 3（事故响应）收尾
**参与成员**：Rex / Cody / Tessa / Docu / Archi（方案）+ 主理人执行
**范围**：核查报告剩余 P0/P1/P2 事项，**跳过 AICAD 共享组件改动**（用户指示）

---

## 📌 TL;DR

- 本轮闭环 6 项：**回滚演练（RTO≈6min）、预热收敛复测、P1 补测、deploy.sh 启动顺序自动化、--api-key 预留、preflight 双机全绿**。
- 剩余待办（AICAD 相关按指示跳过）：告警规则加载、vllm job 采集、Grafana 面板、口令轮换、4 份 Runbook 落盘（Docu 产出回传中）。
- 服务当前状态：双机 healthy、/health 200、单流 decode 94.78 t/s、5 并发 62-65 t/s（预热收敛）。
- 严重度：本轮 P0 完成 2/5（回滚演练、预热复测）、P1 完成 4/6。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（核心保障项已闭环，监控类待 AICAD 授权） |
| 本轮闭环 | 6 项（见 TL;DR） |
| 剩余阻塞 | 3 项 P0（告警/采集/面板，均涉 AICAD） |
| 建议下一步 | 协调 AICAD 负责人授权监控改动 → 8000 切换 |

---

## 🛠️ 本轮执行明细

### 1. preflight.sh 双机全绿（P1 ✅）
- **适配修复 3 处**：镜像检查改用 `docker image inspect`（列格式不匹配）、RoCE 探测加 sysfs 兜底（真机无 ibstat）、digest 日志路径改用户目录（无 /var/log 权限）
- **结果**：head + worker 均 `== preflight PASS ==`（镜像/env/角色/权重 3 片 SHA/RoCE 双链 Active）

### 2. 回滚演练实测 RTO（P0 ✅）
- **流程**：T0=00:15:14 down（head→worker）→ T1=00:15:22 双机停（**中断 8s**）→ worker 重建 → 12s → head 重建 → 权重加载 60%→TileLang 编译 → **T2≈00:21 /health 200**
- **RTO 数据**：中断 8s + 引擎加载 ≈6min；**总恢复 ≈6min，优于 15min 目标**
- **演练后验证**：/v1/models id 正确、容器 healthy、推理正常、**双机 preflight 复跑仍全绿**（闭环）

### 3. 预热收敛复测（P0 ✅）
- 手动数据点：58.36 → 62.73 → 64.97 t/s（连续差 3.4% <5%，收敛）
- **5 并发稳定 62-65 t/s**（reasoning 模式开启，低于基准 C 84.6 属正常——基准测试未开 thinking）
- 单流 decode **94.78 t/s**（门槛 24 ✅）、TTFT 2329ms（reasoning 正常值）
- wait_until_converged.sh 后台观测存在首轮 agg=0 现象（冷启动超时），已修复参数传递（INPUT/OUTPUT/TIMEOUT 透传），手动测速为最终判定依据

### 4. P1 补测（P2 降级执行 ✅）
- 长输出 512：decode 121.94 t/s；长输出 1024：decode 87.54 t/s（errors=0）
- 长 ctx：**54,645 tokens 输入 27.6s 完成**（prefill ≈1980 tok/s）
- 900K ctx：**确认超 max_model_len=393216 不可测**（配置上限实测确认）

### 5. deploy.sh 启动顺序自动化（P1 ✅）
- 按 Cody 审查适配 5 处：ssh 别名（aicad-server/aicad-server60）、脚本路径指向真机 /tmp/start_*_fix.sh、READY_URLS 收敛 8001、探测改 head 侧执行、CRLF 校验
- **验证**：`deploy.sh status` 本机运行正常（双机 healthy + head /health OK）；部署入口在本机（有 ssh config），真机 dspark-tools 同步参考版

### 6. --api-key 预留（P2 ✅ 脚本层）
- start_head_fix.sh / start_worker_fix.sh 增加 `${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"}` 条件注入
- 未设 env 时不加参数（不破坏现状）；上线前 export 即启用（ADR-0010）
- 已同步双机 /tmp/

## ⏳ 剩余待办（AICAD 相关，按用户指示跳过）

| # | 事项 | 状态 | 说明 |
|---|------|------|------|
| 1 | prometheus-alerts.yml 4 条规则加载 | ⏳ 需 AICAD 授权 | Prometheus 8191 为 AICAD 共享实例 |
| 2 | vllm job 采集配置 | ⏳ 需 AICAD 授权 | 同上 |
| 3 | vLLM Grafana 面板 | ⏳ 需 AICAD 授权 | Grafana 3000 为 AICAD 实例 |
| 4 | Grafana 口令轮换 | ⏳ 需 AICAD 协调 | 共享凭据 |
| 5 | 4 份 Runbook 落盘 | 🔄 Docu 产出回传中 | restart/troubleshooting/metrics/change-management |

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 |
|---|------|------|--------|
| 1 | 协调 AICAD 负责人授权监控改动（规则/job/面板） | 工程负责人 | P0 |
| 2 | Runbook 落盘 + 纳入文档体系 | Docu + 主理人 | P1 |
| 3 | 8000 切换决策（served-name 已定，需确认窗口） | 工程负责人 | P0 |

## ⚠️ 待完善 / 已知局限

- wait_until_converged.sh 后台首轮超时现象未完全根治（手动测速为权威判据）；建议后续将收敛逻辑并入 run_checklist 门禁
- 5 并发 62-65 t/s 与基准 C 84.6 的差距归因于 reasoning 模式（thinking 默认开启），如需对齐基准需关 thinking 复测
- 900K ctx 需求若存在，需镜像层调整 max-model-len（超当前 393216 配置）

## 📚 数据来源 & 成员产出索引

- Rex：回滚演练方案（顺序/计时/RTO 定义）、preflight 判定预期
- Cody：deploy.sh 5 处适配审查、--api-key 预留方案（vLLM 0.11 参数 + Bearer 头）
- Tessa：预热复测方案（input 1024 建议）、P1 补测命令集、900K 判定、通过标准表
- Docu：4 份 Runbook 内容（restart/troubleshooting/metrics/change-management，回传中）
- 主理人执行：全部真机 SSH 操作（演练/补测/部署/验证）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
