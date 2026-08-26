# 生产加固交付：部署能力强化 + 运维能力健壮性强化

**日期**：2026-08-26（00:20 增补内嵌执行集成）
**工作流**：工作流 2/事故衍生——生产加固规范 + 脚本落地 + 启动脚本内嵌执行
**参与成员**：Archi(系统架构师·方案设计) / Rex(SRE·脚本实现 + 内嵌执行集成)
**基线**：start_tp4_cluster.sh v1.5-r12 / start_tp4_head.sh v1.5-r11 / start_tp4_worker.sh v1.5-r12（2026-08-25 稳态）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：已将「分层预防性修复思想」（来自克隆环境事故裁决：运行中 QP 断 vs 重启 GID 空洞 vs 宿主内存三线分离 + connect.cc:317-321 写死 GID_INDEX 盲信）固化为生产加固规范与可执行脚本，覆盖**连接建立前的识别/降级/阻断、崩溃证据必留、卡死被活性探针闭合**。
- 产出：**1 份方案文档正文 + 7 个实现脚本**（6 shell + INTEGRATION.md），置于 `deliverables/engineering-assurance/prod-hardening-2026-08-26/`。
- 严重度：🔴 确定（机制/源码实锤）引出的硬门 P0×8 / 🟡 P1×2；变更遵循 `.bak-<tag>` 留档 + check_vllm_script 通过铁律。
- 待办：脚本已 bash -n 通过，**未连生产实跑**，上生产前需人类工程负责人在目标机按 Go/No-Go 复核输出。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 交付物 | 方案文档 + 6 加固脚本 + 集成说明 |
| 关键决策 | 新增 **exit 3=GID/RoCE 布局 No-Go**、**exit 4=NCCL env 一致性失配** 语义 |
| P0 硬门 | GID 预检 fail-fast / HCA·PEER_HCA·LD_PRELOAD-MD5 一致性 / GID_INDEX 动态化 / 崩溃指纹 / nccl 日志首启核验 / 活性探针 / 看门狗速率窗口 |
| P1 待回归 | distributed-timeout 300→120-180s |
| 建议下一步 | 人类负责人逐机复核 → 打 tag 留档 `.bak-gid-dyn/preflight/crashdump/health-20260826` → 维护窗口落地 |

---

## 一、目标与范围

**目标**：把分层预防思想**固化进生产启动与运维规范**，使：
- 布局不一致在**连接建立前**被识别 / 降级 / 阻断
- **崩溃证据必留**（RCA 不再缺日志）
- 卡死被守卫**活性探针**闭合（不再漏判）
- 环网补丁（nccl-ringonly）加载一致性被校验

**范围**：四支柱 + 运维健壮性（见方案正文章节）。

**非目标**：不改模型/推理路径；不改变既有 NCCL T1aM4+MAX_CH16 延迟优化参数；不动 memdesc 使能条件裁决。

---

## 二、方案设计（架构师 Archi 产出，正文详见 `hardening-design-2026-08-26.md`）

### 支柱 A｜连接建立前的有效识别与管理
- `preflight_roce_gid.sh`（新，P0 硬门）：枚举 RoCE 口 GID index0-3 判洞、判 IPv4 RoCEv2、判 index3 子网前缀一致性（可 --peers 交叉核对）、口名预期集校验。
- 不一致响应（三级）：fail-fast 硬停(exit3) / 降级(退 GID_INDEX=-1 动态，exit0 WARN) / 报错(env 失配 exit4)。

### 支柱 B｜崩溃问题有效记录
- `crash_dump.sh`（新）：崩溃指纹三件套 = docker logs + dmesg(NV_ERR/carrier/flap/IBV) + 时间戳，落 `/opt/aicad-prod/backup/crash-<ts>/`。
- NCCL_DEBUG_FILE 已挂 `~/vllm-logs` 持久卷 ✅（固化核验首启生成 nccl-*.log）。

### 支柱 C｜布局不一致启动检查/降级/报错完善
- 三级启动决策表（Input→判定→动作→exit）。
- **exit code 扩展**：`3=GID/RoCE 布局 No-Go（需人工处理，不可重试）`、`4=NCCL env 一致性失配（需重算 HCA/PEER_HCA）`；写进三脚本头部 EXITCODES + check_vllm_script 白名单。
- **铁律**：exit 3/4 必须在 docker run 之前返回（连接建立前）。

### 支柱 D｜环网补丁与连接建立
- LD_PRELOAD 顺序固定：`/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`；ringonly **MD5=4cc43e3b** 启动前核验，失配→exit4。
- RANK_HCA per-rank / PEER_HCA per-peer 须与环阵(01↔02↔04↔03↔01)一致，预检校验。

### 运维健壮性
- `healthcheck_hardened.sh`（新）：口服活性探针（极小请求 max_tokens=1 带超时），闭合卡死漏判；保留 cold-start 900s。
- `watchdog_hardened.sh`（新）：看门狗改「加载期/运行期分段 + 单位时间 NV_ERR 新增速率」窗口，弃纯累计阈值；补 carrier flap 计数探针。
- `distributed-timeout` 300→120-180s（P1，需回归防误杀长 prefill）。

---

## 三、脚本清单与落点（SRE Rex 实现，详见各脚本头注释）

| 脚本 | 用途 | 插入/作用位置 | 优先级 |
|------|------|--------------|--------|
| `preflight_roce_gid.sh` | 连接前 GID/RoCE 布局预检（空洞/子网/口名/连通），fail-fast | head L157 自检后 / worker D1 门禁后、docker rm 前 | P0 |
| `probe_gid_index.sh` | 探测 RoCEv2/IPv4 实际 index，输出建议(数值/-1/REMOVE)，--print-env 可 source | cluster step2 前 / head+worker | P0 |
| `gid_index_env.sh` | env 决策片段：人工覆写→probe→洞栅栏→-1 动态兜底 | ENV_ARGS 组装前 source，替换 head L102/worker L107 写死=3 | P0 |
| `crash_dump.sh` | 崩溃指纹留存(docker logs+dmesg+时间戳) | systemd ExecStopPost 调用 | P0 |
| `healthcheck_hardened.sh` | 守卫活性探针 + 冷启动宽限，闭合卡死漏判 | 替换 healthcheck.sh | P0 |
| `watchdog_hardened.sh` | 看门狗速率窗口化 + carrier flap 探针 | 替换看门狗逻辑 | P0 |
| `INTEGRATION.md` | 集成说明：插入点/.bak tag/check 验证/参数放宽 | — | — |

> 注：`preflight_roce_gid.sh` 一处 `GCCL→NCCL` typo 已交 SRE 修正确认。

---

## 四、变更铁律（沿用 CHANGE 规范）

1. 每脚本/主脚本改动保留 `.bak-<tag>`（建议 tag：`gid-dyn / preflight / crashdump / health / watchdog`，均加 `-20260826`）。
2. 所有改动须 `bash -n` + `check_vllm_script.sh` 通过。
3. exit 3/4 新语义写入三个脚本头部 EXITCODES 注释 + check_vllm_script 校验白名单。
4. 四机同步改，改后重启验证 + 首启核验（防"声称已落地实测未落地"重复模式）。
5. 更新 REFERENCE.md / runbook 参数表。

---

## 五、风险与回滚

| 风险 | 缓解 |
|------|------|
| GID_INDEX→-1 动态化在双子网(10.20/10.100)下选错 index→busbw 倒退 | preflight 保留 ibping 连通判据；仅 index3 有效且连通才维持写死；动态化后回归验证 busbw 不倒退 |
| fail-fast exit3 过严误停健康启动 | ibping 设 P3 软门(降级而非硬停)；硬停仅限空洞/子网/口名 3 类明确判据 |
| 活性探针开销/误杀长 prefill | 最小 token + cold-start 900s 宽限 + distributed-timeout P1 协同 |
| LD_PRELOAD MD5 误报（ringonly 升级） | MD5 基线更新走明确变更流程，不得静默容忍 |

**回滚**：每改动 `cp .bak-<tag> 原路径` + check_vllm_script 通过即可；GID_INDEX 动态化不稳定→一键回退写死 3（前提 preflight 全过）；distributed-timeout 误杀→立即回退 300。

---

## 六、内嵌执行集成（2026-08-26 增补，用户要求"预检在启动过程执行"）

用户明确要求：**预检要在启动过程执行**（非仅独立脚本），提示不一致警告与降级、避免静默故障；克隆环境不一致→首次部署重建；多层 bug 分层修复。已将预检序列**内嵌进 `start_tp4_*.sh` 启动主流程**（`docker run` 之前强制执行），产出到 `prod-hardening-2026-08-26/integrated/`。

### 内嵌位置对照表
| 脚本 | 内嵌点 | 内容 | 落实检查点 |
|------|--------|------|-----------|
| `start_tp4_head.sh`（内嵌版） | check_vllm_script 后 / docker run 前 | probe → preflight(--expect-index --degrade) → gid_index_env 决策序列；写死 `NCCL_IB_GID_INDEX=3` → `${NCCL_IB_GID_INDEX}` 动态注入 | 检查点 1/2/3 |
| `start_tp4_worker.sh`（内嵌版） | 同上 + 口名/HCA 一致性预检 | 同 head 序列 + 口名非预期→exit4 env 失配 | 检查点 1a/2/3 |
| `start_tp4_cluster.sh`（内嵌版） | step2 启 workers 前（新增 STEP0） | 四机 GID 空洞枚举 + preflight exit3/4 + RINGONLY MD5 跨机一致性，任一异常→集群停启+重建指引 | 检查点 1/3 + D1 |
| `DEPLOY_ASSIST.md` | — | 内嵌位置对照 + 克隆不一致→首次部署重建 6 步 + 多层 T0/L1/L2/观测防护点映射 | 全 |

### 避免静默故障（用户硬要求）
- 每次预检异常打印显式告警：`[preflight-FAIL]` / `[preflight-WARN]` / `[cluster-FAIL]`，绝不 silent pass。
- 预检依赖脚本缺失 → fail-fast exit3（不静默跳过）。
- 降级：probe 建议异常自动退 `NCCL_IB_GID_INDEX=-1`（动态），打印降级提示。

### 克隆环境不一致 → 首次部署重建（内嵌指引）
`_preflight_fail()` / cluster STEP0 打印 6 步重建指引：dump 全机 GID → fix_gid_holes → 重算 HCA/PEER_HCA → 重跑 preflight --expect-index 须 exit0 → 再启动 → 验证。

### 多层 bug 分层修复（对齐报告）
- **T0 carrier flap**：cluster STEP0 + watchdog carrier 探针
- **L1 QP 断**：NCCL_IB_RETRY_CNT 注解 + 活性探针
- **L2 GID 空洞**：preflight fail-fast（最大诱因）
- **观测缺口**：crash_dump + nccl 日志持久卷核验

### 自验
内嵌版 head/worker/cluster 三条 `bash -n` 全部 PASS。check_vllm_script 校验的是 `LD_PRELOAD=/opt/libncclpin.so`（前缀，内嵌版保留故不受影响），**不校验 `NCCL_IB_GID_INDEX` 字面量** → 动态注入不会触发 check 失败，无需放宽。

---

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 | 预期完成 |
|---|------|------|--------|---------|
| 1 | 人类负责人逐机复核 6 脚本输出（Go/No-Go） | 运维 | P0 | 上生产前 |
| 2 | 脚本 scp 四机 `/opt/aicad-prod/scripts/` + .bak 留档 | Rex | P0 | 维护窗口 |
| 3 | preflight/probe/gid_index_env 接入 head/worker/cluster + check 通过 | Rex | P0 | 维护窗口 |
| 4 | ExecStopPost crash_dump + nccl 日志首启核验 | Rex | P0 | 维护窗口 |
| 5 | 守卫活性探针 + 看门狗窗口化替换 | Rex | P0 | 维护窗口 |
| 6 | distributed-timeout 300→120-180 回归验证后落地 | Archi | P1 | 评估后 |
| 7 | 更新 REFERENCE.md/runbook + MEMORY 记忆 | Docu | P1 | 维护窗口 |

---

## ⚠️ 已知局限

- 脚本**未连生产实跑**（按约束只写文件）；上生产前须目标机逐条验证。
- 未实测 GID_INDEX=-1 动态化在现网双子网下的 busbw 数据（需 A/B）。
- 守卫活性探针生产侧曾「暂不设」——本方案评估为 P0，若仍暂缓须如实记录盲区保持开放。

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：`hardening-design-2026-08-26.md`（方案正文：四支柱 + 运维健壮性 + 落地清单 + 风险回滚）。
- **Rex（SRE）**：`prod-hardening-2026-08-26/` 6 脚本 + INTEGRATION.md。
- 上游引用：`incident-clone-roce-prevention-2026-08-25.md`、MEMORY.md 崩溃分层裁决。

---

> 本交付由工程保障团队 AI 协作生成，上生产实施请由人类工程负责人复核后执行。