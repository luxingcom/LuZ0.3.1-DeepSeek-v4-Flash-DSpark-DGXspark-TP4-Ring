# 生产加固脚本 QA 交叉审查与修复验证报告

**日期**：2026-08-26
**工作流**：工作流 1（代码审查）+ 独立验证
**参与成员**：Tessa×2（qa-static-1 静态代码审查 / qa-flow-2 端到端流程审查+独立复核）/ Rex（sre-fix-1 修复）/ Archi（方案基线）
**审查对象**：`prod-hardening-2026-08-26/` 6 加固脚本 + `integrated/` 3 内嵌启动脚本 + 基线对照

---

## 📌 TL;DR（执行摘要）

- **整体结论**：生产加固脚本经两位 QA 从静态代码 + 端到端流程两个视角交叉审查，发现 **4 个阻断级 + 8 个健壮性问题**；已由 SRE 全部修复，并经独立 QA 复核 + 主理人抽查**确认全部闭合**（修复后 9 脚本 bash -n 全过）。
- 严重度分布：🔴 修复前 4 阻断 / 修复后 0 阻断；🟠 8 健壮性已修；✅ 审查还确认了多项"守卫到位正确"。
- **最终判定**：🟢 已闭合可上生产（含 1 项补修的 head hostname 守卫，基于实测机名证据裁断）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 修复后通过（4 阻断 + 8 健壮性全部闭合） |
| 审查视角 | 静态代码（qa-static-1）+ 端到端流程/持久化（qa-flow-2）+ 独立复核（qa-flow-2） |
| 修复项 | 4 阻断 + 8 健壮性 + 1 补修（head hostname）
| 修复后自验 | 9 脚本 bash -n 全过 + R1/I5/HR 有功能实测 |
| 建议下一步 | 人类负责人复核 → 四机同步落地 → 重启验证 |

---

## 🔍 一、两视角交叉审查发现

### 🔴 阻断级（4 项，修复前会致启动失败/静默失效）

| # | 文件:行 | 问题 | 影响 | 修复 | 复核 |
|---|---------|------|------|------|------|
| R1/R2 | `gid_index_env.sh` L42-89 | 降级链 `_finish_nores` 的 `return 1` 触发调用方 `set -e` → 探测失败时整脚本退出而非降级 -1；probe 管道失败同样 errexit | 降级逻辑被破坏 | 调用点补 `|| true` + probe 取值 `set +e; ...||true` | ✅ 实测 source 后 3 分支均存活 |
| I5 | `start_tp4_worker.sh` L227-261 | 遍历 `/sys/class/infiniband` 取 IB 设备名（mlx5_0）却用 netdev 口名比对 → 恒不匹配 → 必 exit4 | 真实 worker 永不启动 | 改遍历 `/sys/class/net` 枚举 RoCE netdev | ✅ 实测 mock 4 口 HCA_OK=1 |
| W1 | `watchdog_hardened.sh` L82-88 | `--since` 窗口计数与累计 LAST_NV 求差 → NEW_ERR≈0 恒不触发 | 卡死期看门狗静默 | 改全量累计 CUR_NV-LAST_NV 求增量 | ✅ 逻辑自洽无死判据 |
| I7 | `start_tp4_cluster.sh` L66-70 | RANK_HOST[1..3] 全=dgxspark01:186（基线继承） | 3 worker 全 ssh 到 head 起错机 | 改 02:187/04:189/03:188 与 CLUSTER_HOST 一致 | ✅ step3/4 用修正映射 |

### 🟠 健壮性/一致性（8 项，已修）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| P2/Q2 | preflight/probe | `--peers` 文档 `:` 与解析 `=` 不符→判据3静默跳过 | 兼容 `:`/`=`，不支持时显式告警 |
| Q1 | probe_gid_index.sh | index3 也有效时未优先 index3 | index3 有效则优先建议 3 |
| I10 | start_tp4_cluster.sh | RINGONLY NOFILE/UNREACH 不拦 | 任一 NOFILE/UNREACH 置 cluster_gate_fail |
| P3 | preflight_roce_gid.sh | 判据3 子网取首 64bit 无区分力 | 改比较 GID 末 32bit(IPv4 段) |
| H1 | healthcheck_hardened.sh | BODY model 硬编码 | `SERVED_MODEL_NAME` env 注入 |
| C1 | crash_dump.sh | 持久化未确认 + L42 `>`覆盖表头 | merge 提示 + 表头 tee + 日志 `>>` |
| I4/I8 | preflight/probe | --peers 判据3 全链路触发契约不明 | 注释+DEPLOY_ASSIST 注明由 cluster 触发 |
| I3 | head/worker | log-opt max-file=5 vs 建议 3 | 统一为 3 |

### ✅ 审查确认"守卫修改到位"的正确项
- 守卫前置到 `docker run` 前（exit3/4 fail-fast）✅；写死 GID_INDEX=3 → 动态注入且 check_vllm_script 全过 ✅；四机一致性门位置在 step1 前 ✅；检查点 1-5 覆盖无漏 ✅；NCCL_IB_GID_INDEX 动态注入不破坏 check ✅。

---

## 🔍 二、持久化稳定性验证结论（qa-flow-2）

**总判断（修复前）**：启动前 fail-fast + 动态 GID index 设计对偶发正确路径成立，但"重启/重建/自愈后依然生效"不成立（watchdog/healthcheck 无 unit、crash_dump 无 service 接入、预检无落盘同步步骤）。

**落地时运维必须做的持久化清单**（缺一即重启后回退）：
1. 6 加固脚本 + 3 integrated 启动脚本 cp 到每机 `/opt/aicad-prod/scripts/` + chmod +x
2. 四机同版本 scp/rsync + md5sum 记录
3. 更新 systemd unit：3 个 .service 加 `ExecStopPost=.../crash_dump.sh vllm-tp4-rank{N}` + `daemon-reload && enable`
4. 装 watchdog timer（`watchdog_hardened.timer` + unit，`enable --now`）；healthcheck_hardened 接 docker `--health-cmd`
5. 改造前留 `.bak-<tag>` 可秒回滚
6. check_vllm_script 通过后签字
7. 原有自愈链（monitor→NO_WAIT 早退、systemd Restart=always）未被破坏——preflight 在 NO_WAIT 早退之前，重启自愈仍先过预检 ✅

---

## 🔍 三、独立复核（qa-flow-2 fresh eyes）+ 补修

**4 阻断 + 8 健壮性 = 判定"已闭合"**，且 R1/R2（降级链）与 I5（口名门）有实测验证（模拟 probe stub、mock 4 twin 口 + 干扰网卡），非仅读码。

**补修 1 项（主理人裁断必须修）**：
- **head hostname 守卫不对齐**：head L51 只认 `dgxspark01`、cluster L90 认 `dgxspark01|spark-05cd`。
- 证据：归档记忆 08-01 明确 `<remote_head_ip> = head = spark-05cd`（`dgxspark01` 是 SSH 别名非 hostname）。
- 修复（sre-fix-1）：head 守卫改为同时认 `dgxspark01 | spark-05cd`，错误消息/头注释/help 同步。**功能验证**：spark-05cd=PASS、dgxspark01=PASS、other-host=FAIL ✅

**残留 minor（不影响运行，可后处理）**：watchdog L91 echo 文案 "window ${LOADING_WIN}s" 语义未同步（纯展示）；healthcheck worker 分支注释承诺的"D3 dmesg 旁证"未实现（doc 过度承诺）——建议后续清理。

---

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 | 预期完成 |
|---|------|------|--------|---------|
| 1 | 人类负责人复核修复后脚本 + 上生产 Go/No-Go | 运维 | P0 | 上生产前 |
| 2 | 按"持久化清单"7 项四机落地（cp/scp/systemd/timer/health-cmd/.bak/check签字） | Rex | P0 | 维护窗口 |
| 3 | 重启验证：改后首启核验（GID 实际注入、nccl-*.log 生成、守卫无误杀） | Rex | P0 | 维护窗口 |
| 4 | minor 清理：watchdog echo 文案 / healthcheck worker 旁证注释 | Rex | P2 | 随手 |

---

## ⚠️ 已知局限

- 修复为交付副本（`integrated/`），未连生产服务器实跑；上生产前人类负责人逐机复核输出。
- QA 以静态审查 + mock 实测为主，未在真实 4 机做端到端启动（需维护窗口）。
- head hostname 裁决基于归档记忆（08-01 记录 spark-05cd），若现机 hostname 有变需现场复核。

---

## 📚 数据来源 & 成员产出索引

- **qa-static-1（代码审查）**：9 脚本逐行审 + bash -n + check_vllm_script 验证；发现 R1/R2、I5、W1 阻断 + 8 健壮性表。
- **qa-flow-2（流程/持久化 + 独立复核）**：流程缺口表 + 持久化清单 + fresh eyes 复核 4 阻断闭环 + 遗留 hostname 项。
- **sre-fix-1（修复）**：4+8+1 项修复实现，R1/I5/HR 功能验证，.bak-qa* 留档。
- **主理人**：抽查 I5/W1/I7 落地 + 裁断 head hostname 必须修（归档证据 spark-05cd）。

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。