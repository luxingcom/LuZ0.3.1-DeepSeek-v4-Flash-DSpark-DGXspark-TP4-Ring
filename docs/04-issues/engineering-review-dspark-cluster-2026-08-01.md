# 全面工程审查报告：DeepSeek-V4-Flash 双 DGX Spark 集群（TP=2 vLLM）

**日期**：2026-08-01
**工作流**：工作流 1（综合代码审查）+ 架构评估 + 测试覆盖 + 文档审查（多工作流合并）
**参与成员**：Cody（代码审查主责）、Archi（架构）、Tessa（测试）、Docu（文档）、Rex（SRE 交叉）

---

## 📌 TL;DR（执行摘要）

- 整体结论：**🟡 有条件通过（Conditional Go）**——技术性能已实证（dspark 2.19x 交叉验证），但部署防呆不足、参数多源漂移、监控/文档缺口集中在生产 8000 切换前置区，须修复后再复审。
- 严重度分布：🔴严重 9 项 / 🟠高 7 项 / 🟡中 6 项 / 🟢低 2 项。
- 阻塞 / 非阻塞：**8000 切换前必须修 5 项**（启动顺序强制、双机守卫、参数对齐、镜像 digest pin、served-model-name 决策）+ 4 项 SRE 硬性准入；非阻塞项可后置（监控面板漂移、暴露面收敛）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（Request Changes 后复审） |
| 阻塞项数量 | 5 项必须修 + 4 项硬性准入 |
| 审查范围 | 12 个部署脚本 + 13 个工具链文件 + 6 项 ADR + 基准矩阵 + 交接文档 |
| 代码审查发现 | 15 条（🔴3 / 🟠4 / 🟡6 / 🟢2） |
| 建议下一步 | 修复阻塞项 → 复审 → 执行 Go/No-Go 清单 → 8000 切换 |

---

## 🔍 审查发现（按严重度排序，跨成员去重合并）

### 🔴 严重（9 项）

| # | 类别 | 来源 | 问题描述 | 建议修复 |
|---|------|------|---------|---------|
| 1 | 正确性 | Cody | 启动顺序（worker→12s→head）仅靠人肉，脚本无 sleep/就绪检查；head 早启或不足 12s → TP 初始化整体失败 | deploy 入口脚本：ssh 58 起 worker → 轮询 60:8000 就绪 → ssh 60 起 head；至少内置 sleep + curl 探活 |
| 2 | 正确性 | Cody | #6 互拷防线缺失：head 脚本拷到 58 即双 head 事故；脚本无 hostname/NODE_RANK 守卫 | 脚本头部守卫 `[ "$(hostname)" = "edgexpert-0c69" ] && [ "$NODE_RANK" = "1" ] \|\| exit 1` |
| 3 | 正确性 | Cody | NCCL/RoCE 参数漂移：生产基线 envc 与 dspark 脚本完全无 NCCL_IB_HCA/GID/NCCL_NET（0731 标为"勿改"关键参数） | 补齐或显式注释"继承镜像默认"；与 0731 对齐 |
| 4 | 正确性 | Cody+Docu | 参数多源漂移：GPU_MEM 0.85/0.8/0.82、master-port 25000/25001/25002、SPEC_TOKENS 5/2、batched 4096/8192 多 profile 并存 | 参数单一事实来源（env 模板），master-port 全 profile 唯一，锁定生产值 |
| 5 | 架构 | Archi | head(60) 下线 = 全服务停，无故障转移（TP=2 无容错） | 启动 Runbook + docker restart 策略 + 保留回滚容器 |
| 6 | 架构 | Archi | RoCE 双链同失效 = 集群瘫；单链失效行为未测 | 注入测试验证 NCCL 双 HCA 故障转移；监控 rocep 链路 |
| 7 | 统计 | Tessa | acceptance 口径混乱：A 环境同数据 6.68%（1915 drafts）vs 23.03%（868 drafts）差 3.4x，与 C 14.94% 不可比 | 统一工具/数据集/样本量（每档 ≥5000 drafts），报 CI |
| 8 | 统计 | Tessa | 预热未控制：A 测于 1h 热态、C 测于 30min 内（5 并发 72→85 = +18%）；A/C 残余 35% 归因从未验证 | 固定 warm-up 协议（≥60min 或收敛判据）+ 同热态复测 |
| 9 | 文档 | Docu | MASTER_PORT 三处不一致（交接=25000 vs envc 脚本=25002 vs 索引=25000）；artifacts/ 实际缺 SHA256SUMS 清单文件 | 锁定单一事实来源并落盘；补 SHA256 清单 |

### 🟠 高（7 项）

| # | 类别 | 来源 | 问题描述 | 建议修复 |
|---|------|------|---------|---------|
| 10 | 安全 | Cody | `--privileged --network host --ipc=host` + 第三方镜像仅可变 tag（未 pin digest） | 脚本固定 digest；评估最小权限 |
| 11 | 安全 | Cody | Grafana admin 口令 base64 明文硬编码（fix_dash_kvcache.py:4 / grafana_set_refresh.py:4） | 改 env 注入/Keyring，移出交付物 |
| 12 | 安全 | Cody | smi_server 0.0.0.0:8088 无鉴权暴露 nvidia-smi 全量 + 宿主机信息；ssh 用户/节点 IP 硬编码 | 绑 127.0.0.1 或加 Token；配置化 |
| 13 | 架构 | Archi | 管理网 ssh 为镜像同步唯一通道（16.6MB/s，23.5GB≈20min） | 备选 dockerproxy/离线盘；长期修 CX7 免密 |
| 14 | 架构 | Archi | 控制面随 RoCE 数据面走（torchrun TCPStore），RoCE 断则控制面同步断 | 数据面/控制面分离 ADR |
| 15 | 运维 | Rex | 回滚路径纯手工，RTO 取决于人；无自动恢复 | 回滚 Runbook + 演练实测 RTO |
| 16 | 测试 | Tessa | 工具不一致（urllib 直测 vs bench serve）与对照不一致（A/C 显存+权重变量混淆） | 统一工具/指标定义；归因结论限缩为方向性 |

### 🟡 中（6 项）

| # | 类别 | 来源 | 问题描述 | 建议修复 |
|---|------|------|---------|---------|
| 17 | 正确性 | Cody | bench_matrix_B.sh 头注释/输出文件残留"环境 A"（实际跑 B） | 修正为 B、改 OUT |
| 18 | 正确性 | Cody | `--served-model-name ChatGPTN` 占位名与实际 served-name 无校验 → 400 风险 | served-name 显式化并统一 |
| 19 | 正确性 | Cody | smi_server.py:56,65 DCGM_FI_DRIVER_VERSION 为字符串，`float()` 必失败 → driver 恒 N/A | 按字符串解析 |
| 20 | 可维护性 | Cody | dspark 与 mtp 脚本并存、两种参数风格混用（继承 CMD vs 显式 serve 参数） | 合并为单模板 + 参数文件 |
| 21 | 可维护性 | Cody+Docu | artifacts/start_vllm_node.sh 检出 CRLF（#4 复发 2 次）；Grafana 面板导出与线上漂移（uid 写死 + 旧指标名） | gitattributes eol=lf + pre-commit；uid 模板化、导出回写 |
| 22 | 测试 | Tessa | C 单流 ±20% 无分布（检查清单"≥24"=波动下限形同虚设） | 报 mean±std+n；阈值用稳态中位数 |

### 🟢 低（2 项）

| # | 类别 | 来源 | 问题描述 | 建议修复 |
|---|------|------|---------|---------|
| 23 | 性能 | Cody | smi_server bg_loop 串行抓 2 节点，单节点 SSH 慢拉长整周期 | 线程池并行抓取 |
| 24 | 可维护性 | Cody | check_prom.py:17 依赖 urllib.request 隐式加载 urllib.parse | 显式 import |

## 🏗️ 架构影响评估（Archi）

- **拓扑结论**：双机 TP=2 + RoCE 直连是唯一可行起点（155.4GB > 单机 121GB），roofline 225GB/s÷4GB≈56 t/s，dspark 后 35.3/84 t/s 达标；但**生产可用 ≠ 高可用**，属"可恢复单点"架构。
- **统一内存 0.85**：121GB×0.85≈103GB 契合权重+KV；KV+40%/并发+39% 无负面，同意生产 0.85；但仅剩 ~18GB 余量，需监控宿主内存，1M ctx 前重估。
- **扩展边界**：4 机环网可行但单链断=环断；社区 4→8 仅 +13% 且 DSV4 8 机无先例，**建议止步 4 机**；4 机前提 5 条（双机 MTP 验证、链路故障演练、SSH 互信走管理网、权重 SHA256 全量校验、先环网后补交换机）。

### ADR 复核表

| ADR | 结论 | 备注 |
|-----|------|------|
| ADR-1 驱动/工具链（CUDA13 + b12x） | ✅ 同意 | 错误#13 实证 |
| ADR-2 权重选型（0731=DSpark） | ✅ 同意 | 72317 keys 全等 |
| ADR-3 镜像 hybrid-1.6 | 🟡 附条件 | 镜像 ID 双机不一致需闭环核对；VLLM_DSPARK_* env 语义待确认 |
| ADR-4 投机方式 dspark | ✅ 同意 | method=mtp 必崩 |
| ADR-5 显存 0.85 | 🟡 附条件 | 监控宿主内存后放行 |
| ADR-6 参数来源（继承 CMD） | ✅ 同意 | 补充：沉淀"有效配置快照"防上游漂移 |

### 缺失 ADR（需补充）

1. **ADR 服务命名/端口（P0 阻塞）**：8000 vs 8001、served-model-name——唯一必须人类拍板的决策点
2. **ADR 回滚策略**（触发阈值/命令/验证）
3. **ADR 监控契约**（8191/指标名/告警阈值/节点级 exporter）
4. **ADR 数据面/控制面分离**
5. **ADR 安全暴露面**（--network host 裸 API 无 TLS/认证）
6. **ADR 镜像供应链**（标签 + SHA256 pin）
7. **ADR 4 机扩展**（含 go/no-go 判据）

## 🧪 测试覆盖评估（Tessa）

### 覆盖度矩阵（场景 × 环境）

| 场景 | A：DSpark+dspark(0.8) | B：0731 无投机(0.8) | C：0731+dspark(0.85) |
|---|---|---|---|
| 单流 NL 2K | ✅ 35.3 t/s | ✅ 27.6-27.8 t/s | ✅ 24-34 t/s |
| 单流 random 2K | ⚠️ 29.2 t/s（小样本） | ❌ | ❌ |
| 3 并发 | ✅ 84 t/s | ✅ 38.6 t/s | ⚠️ 60.8 t/s（47-77 波动） |
| 5 并发 | ✅ 108.2 t/s | ❌（P2 待办） | ✅ 84.6 t/s |
| 131K prefill | ✅ 2226 tok/s | ✅ 1985 tok/s | ⚠️ 1449（单次+污染，不可信） |
| 长输出 512/1024 | ❌ | ❌ | ❌（P1 待办） |
| 900K ctx | ❌ | ❌ | ❌（P1 待办） |
| acceptance 大样本 | ❌ 口径矛盾 | N/A | ✅ 14.94%/1.75（3655 drafts） |

**缺口**：长输出、900K ctx、5 并发 B、A/C 同条件复测、C 131K 复测、B 单流 random、acceptance 统一口径。B/C 2.19x 交叉验证是最强证据；A/C 对照被 0.8/0.85 + 热态双重污染。

### 交接检查清单（10 项）验证要点

- 技术项（镜像/权重/RoCE/env 基线）可信；运维项脆弱（脚本同步、启动顺序、角色全凭手工核对）
- 缺 3 项生产前置：①回滚路径验证（profile-0731-nomtp 一键回切演练）②900K/长输出能力验证 ③故障注入（worker 掉线恢复）
- Grafana 项缺 Prometheus 告警规则验证

### 生产 8000 smoke 套件（启动后 10 分钟）

- **Phase1（0-2min）**：`docker images | grep production-hybrid-1.6`；`docker inspect` env 基线；`curl -sf :8000/health && /v1/models`；`ibstat` RoCE 状态
- **Phase2（2-6min）**：urllib 直测 2048in/256out，随机前缀防 cache；断言 TTFT<1500ms、decode≥24 t/s、error=0
- **Phase3（6-10min）**：5 并发 3 轮断言 agg≥80 t/s；`curl :8191/metrics | grep vllm:kv_cache_usage_perc` 断言 KV≥1.4M
- **自动化建议**：Phase1+基线检查 → cron 每小时 + CI 部署门禁；Phase2/3 → CI 门禁 + 每周回归，超阈值接 Alertmanager；预热协议脚本化 `wait_until_converged()`（连续 3 次 5 并发差<5%）；acceptance 采样器固定 min_drafts=5000 + 报 CI + 随机前缀

## 📄 文档可执行性评估（Docu）

- **"10 分钟可行动"不达标**：10 项清单 0 项完整（命令+路径+预期值齐全），9 项 ⚠️ 缺命令/缺预期值
- **6 处文档矛盾**：MASTER_PORT 三处不一致、SPEC_TOKENS 5 vs 2、GPU_MEM 0.8/0.85/0.82、0.8 vs 0.85 归因混淆权重变量、artifacts 缺 SHA256SUMS、"环境 C" vs "envc" 命名不一致
- **P0 文档缺口**：生产 8000 切换 Runbook、回滚 Runbook、profile-envc README（当前基线无 README）、served-model-name 决策留痕、SHA256 清单落盘
- **P1**：启停/重启 Runbook、故障排查 Runbook（head/worker/RoCE/权重决策树）、监控接入文档、变更管理流程

## 🎯 Go/No-Go 决策（生产 8000 切换）

**判定：🟡 有条件 Go（Conditional Go）**

**硬性准入 4 项**（任一不满足 → 降级 No-Go）：
1. served-model-name 决策完成（建议 `deepseek-v4-flash-0731`，由人类拍板）
2. preflight.sh 校验脚本上线并双机跑通（含启动顺序守护、角色断言）
3. 容器 restart:unless-stopped + /health 探针 + 日志持久化卷
4. 至少一条容器/服务级告警生效并验证链路

**同时必须修复（Cody 阻塞项）**：启动顺序强制（#1）、双机 hostname 守卫（#2）、NCCL/GPU_MEM/master-port 参数对齐（#3/#4）、镜像 digest 固定（#10）；强烈建议 CRLF 门禁 + 凭据移出脚本。

**回滚方案**：8001 保留 dspark 暂存 → 8000 切换后观察 30min（错误率/TTFT/KV 水位）→ 劣化即回切 production-ready（profile-0731-nomtp）；上线前实测 RTO。

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | served-model-name 决策（人类拍板）+ 生产 8000 切换 Runbook | 工程负责人 + SRE | P0 | 上线窗口 |
| 2 | 修复脚本阻塞项：启动顺序强制 / hostname 守卫 / NCCL 参数对齐 / digest pin | Cody + SRE | P0 | 上线前 |
| 3 | preflight.sh + 容器自愈 + 告警链路验证 | SRE | P0 | 上线前 |
| 4 | 锁定参数单一事实来源（MASTER_PORT/SPEC_TOKENS/GPU_MEM）+ SHA256 清单落盘 | Docu + SRE | P0 | 上线前 |
| 5 | 回滚演练一次（实测 RTO）+ 交接清单 10 项逐项验证 | SRE + Tessa | P0 | 上线前 |
| 6 | 凭据移出脚本（Grafana/smi_server）+ 暴露面收敛 | Cody | P1 | 1 周 |
| 7 | 补测：长输出 512/1024、900K ctx、5 并发 B、C 131K 复测、acceptance 大样本统一口径 | Tessa | P1 | 2 周 |
| 8 | 补文档：启停/故障排查/监控接入/变更管理 | Docu | P1 | 1 周 |
| 9 | 监控面板导出回写仓库 + uid 模板化 | Cody | P2 | 2 周 |

## ⚠️ 待完善 / 已知局限

- 审查基于静态研读（交接文档 + 脚本 + 基准记录），未连接真实集群；preflight/告警/smoke 需真机验证。
- 0.8 vs 0.85 显存归因、A/C 残余差距归因受热态与变量混淆影响，结论应限缩为方向性。
- 镜像 VLLM_DSPARK_* env 语义、IMAGE ID 双机差异待上游/真机确认。

## 📚 数据来源 & 成员产出索引

- Cody（代码审查师）原始产出：15 条发现清单（含文件:行）、3 条阻塞判定、亮点 3 条
- Archi（架构师）原始产出：架构评估、ADR 复核表、缺失 ADR 7 项、单点故障表、4 机前提
- Tessa（测试专家）原始产出：覆盖度矩阵、统计风险 8 项、清单验证计划、smoke 套件
- Docu（文档师）原始产出：10 项清单评分、缺失文档清单、6 处一致性矛盾
- Rex（SRE 工程师）交叉产出：Go/No-Go 准入条件、运行态风险

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
