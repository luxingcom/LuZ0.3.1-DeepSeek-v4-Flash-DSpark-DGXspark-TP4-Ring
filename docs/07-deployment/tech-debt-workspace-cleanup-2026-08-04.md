# 工作区文档与杂散工具整理归档方案

**日期**：2026-08-04
**工作流**：工作流 5（技术债评估 - 文档债盘点）
**参与成员**：Docu（技术文档师）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：工作区根目录堆积 224 个文件，其中 ~130 个（58%）为调试日志/临时 txt，新旧版本源码共存（v1.1.0 网关与线上 v1.2.0 并存），重复文件 2 对，散落脚本 ~30 个。无 git 仓库，无法通过版本历史核对。
- **严重度分布**：🔴严重 1 项（过期版本误用风险）/ 🟠高 2 项（重复文件、无 git）/ 🟡中 3 项（调试日志堆积、env A-D 快照混杂、基线文档缺失）/ 🟢低 1 项（文件归组）
- **阻塞 / 非阻塞**：非阻塞，但 P0 项（过期网关版本）存在高误用风险。
- **整理效果**：归档后根目录从 224 个文件降至 ~90 个，可读性显著提升。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（需执行归档方案后方可达到整洁基线） |
| 阻塞项数量 | 1（过期网关版本 v1.1.0 与线上 v1.2.0 共存） |
| 关键行动项 | 8 条 |
| 建议下一步 | P0 归档过期版本源码 -> P1 归档调试日志 -> P2 归组脚本 -> 新建基线文档 -> git init |

---

## 🔍 文档债清单（按严重度排序）

| # | 严重度 | 类别 | 位置 | 问题描述 | 建议修复 | 来源 |
|---|--------|------|------|----------|----------|------|
| 1 | 🔴严重 | 过期版本 | 根目录 `hardened_main_gateway.py` (v1.1.0) | 与线上 `hardened/live/responses_gateway_main.py` (v1.4.0) 共存。v1.1.0 仅支持 /v1/responses，v1.4.0 新增 chat/completions + embeddings。误用 v1.1.0 会导致功能缺失 | 归档到 `archive/old-versions/` | Docu |
| 2 | 🟠高 | 重复文件 | 根目录 `start_head_E.sh` / `start_worker_E.sh` | 与 `hardened/live/` 下完全相同（已 diff 确认）。两份并存易造成修改只改一份 | 移到 `archive/duplicates/` 或删除根目录副本 | Docu |
| 3 | 🟠高 | 无版本控制 | 工作区全局 | 未纳入 git，无法通过版本历史核对文件变更。8 对版本关系需手动 diff | 整理完成后 `git init`，调试日志排除在 .gitignore | Docu |
| 4 | 🟡中 | 调试日志堆积 | 根目录 ~130 个 txt/log | sre_v0~v31、head_log_tail、dl/ds/dh 系列、stat/proc/conn/ports 系列，占根目录 58%，严重影响可读性 | 归档到 `archive/debug-logs/` 按类别分子目录 | Docu |
| 5 | 🟡中 | env A-D 快照 | `deliverables/.../hardened/live/` 下 | env A/B/C/D 的启动脚本和修复工具已被 env E 取代，仍与线上文件混杂 | 归档到 `archive/env-A-D-snapshots/` | Docu |
| 6 | 🟡中 | 基线文档缺失 | 全局 | 缺少 TOPOLOGY.md（集群拓扑）、CHANGELOG.md（版本变更）、ENV-MATRIX.md（环境矩阵）、ARCHIVE-INDEX.md（归档索引）、FILE-INVENTORY.md（文件清单） | 新建 5 个基线文档 | Docu |
| 7 | 🟢低 | 脚本散落 | 根目录 ~30 个 | 测试脚本、工具脚本、部署脚本散落根目录无归组 | 归组到 `tests/`、`tools/`、`scripts/` | Docu |

---

## 🏗️ 归档方案

### 建议目录结构

```
集群部署/
├── overview.md                          # 工作区总览（保留）
├── hardened/                            # 当前线上配置（保留）
│   ├── live/                            # env E 线上文件
│   ├── README.md, PARAMS.md             # 基线文档
│   ├── adr/, runbooks/, configs/        # 架构决策/运维手册/配置
│   └── artifacts/                       # 工具脚本和校验和
├── deliverables/engineering-assurance/  # 正式报告（保留）
├── tests/                               # [新建] 测试脚本归组
├── scripts/                             # [新建] 部署/探针脚本归组
│   ├── deploy/                          # dryrun_head.sh, entrypoint_gpu.sh, rollback_cpu.sh
│   └── probe/                           # baseline_*.sh, probe_*.sh
├── tools/                               # [新建] 工具脚本归组
│                                       # fix_toolcall.py, strip_debug.py, patch_debug_v2.py
│                                       # check_*.py, nccl_probe*.py
├── configs/services/                    # [新建] 服务配置归组
└── archive/                             # [新建] 归档区
    ├── old-versions/                    # 过期版本源码（v1.1.0 网关等）
    ├── env-A-D-snapshots/               # 历史 env A-D 快照
    ├── completed-plans/                 # 已完成的计划/台账
    ├── old-checksums/                   # 旧校验和
    ├── work-in-progress/                # 草稿/内部工作文件
    ├── debug-logs/                      # 调试日志（~130个）
    │   ├── sre/                         # sre_v0~v31 + sre_final* + sre_p*
    │   ├── head/                        # head_log* + hlog*
    │   ├── worker/                      # wlog_tail + worker_proc + worker_check
    │   ├── proc/                        # ps_check* + proc_sample* + procs*
    │   ├── gpu/                         # gpu_s* + hcpu*
    │   ├── network/                     # conn* + ports* + ping* + route* + nccl_env + zmq*
    │   ├── nccl/                        # nccl_probe*.log
    │   ├── deploy/                      # deploy_*.log + rm_*
    │   ├── iteration/                   # dh* + ds* + dsa* + dl* + dw* + wl* + wstat* + stat*
    │   ├── code-ctx/                    # coord_* + crpc* + dp_rank + fol_* + gks* + ...
    │   ├── health/                      # health* + hlth + ray_chk
    │   └── misc/                        # probe_out + smoke_out + patch_ev* + rh_* + rw_*
    └── duplicates/                      # 重复文件
```

### 归档优先级

| 优先级 | 操作 | 文件数 | 理由 |
|--------|------|--------|------|
| P0 | 过期版本源码移至 `archive/old-versions/` | 4 | v1.1.0 网关误用风险极高 |
| P0 | 重复文件移至 `archive/duplicates/` | 2 | 与线上完全相同 |
| P1 | 调试日志归档 `archive/debug-logs/` | ~130 | 占根目录 58% |
| P1 | env A-D 历史快照归档 | ~15 | 已被 env E 取代 |
| P2 | 测试脚本归组 `tests/` | ~15 | 散落根目录 |
| P2 | 工具脚本归组 `tools/` | ~9 | 散落根目录 |
| P3 | 部署/探针脚本归组 `scripts/` | ~6 | 散落根目录 |
| P3 | 旧校验和/已完成计划归档 | ~4 | 低风险 |

---

## 🧪 版本关系核对要点（8 对）

| 核对项 | 文件 A | 文件 B | 核对结果 | 风险 |
|--------|--------|--------|----------|------|
| 网关版本 | `hardened_main_gateway.py` (v1.1.0) | `responses_gateway_main.py` (v1.4.0) | **已确认不同** | 高 |
| 启动脚本重复 | `start_head_E.sh` (根目录) | `hardened/live/start_head_E.sh` | **已确认相同** | 低 |
| 启动脚本重复 | `start_worker_E.sh` (根目录) | `hardened/live/start_worker_E.sh` | **已确认相同** | 低 |
| 网关重构版 | `deliverables/.../responses_gateway/main.py` | `hardened/live/responses_gateway_main.py` | 需 diff | 中 |
| Embedding 服务 | `embed_main.py` (根目录) | `deliverables/.../embedding/main.py` | 需 diff | 中 |
| Embedding 网关 | `embed_gateway_main.py` (根目录) | `embedding/gateway_main.py.v1.2.0` | 需 diff | 低 |
| systemd 服务 | `hardened/live/responses-gateway.service` | `responses_gateway/responses-gateway.service` | 需 diff | 低 |
| requirements.txt | `embed_requirements.txt` (根目录) | `embedding/requirements.txt` | 需 diff | 低 |

### Git 化建议

工作区无 git，强烈建议整理完成后：
1. `git init` 在工作区根目录
2. 添加 `.gitignore`：排除 `archive/debug-logs/`
3. 首次提交：以整理后目录结构作为初始 commit
4. 后续变更：所有线上配置变更通过 commit 追踪

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 归档过期版本源码到 `archive/old-versions/`（hardened_main_gateway.py v1.1.0 等 4 个） | Docu+SRE | P0 | 本周 |
| 2 | 移除/归档重复文件（根目录 start_head_E.sh / start_worker_E.sh） | Docu | P0 | 本周 |
| 3 | 归档 ~130 个调试日志到 `archive/debug-logs/` 按类别分子目录 | Docu | P1 | 本周 |
| 4 | 归档 env A-D 历史快照到 `archive/env-A-D-snapshots/` | Docu | P1 | 本周 |
| 5 | 归组测试/工具/部署脚本到 `tests/`、`tools/`、`scripts/` | Docu | P2 | 2 周内 |
| 6 | 新建 5 个基线文档（TOPOLOGY/CHANGELOG/ENV-MATRIX/ARCHIVE-INDEX/FILE-INVENTORY） | Docu+Archi | P2 | 2 周内 |
| 7 | 完成 6 对需 diff 的版本关系核对 | Docu+Cody | P2 | 2 周内 |
| 8 | 整理完成后 `git init` + 首次提交 | SRE | P2 | 2 周内 |

---

## ⚠️ 待完善 / 已知局限

- 本方案为只读分析，未实际移动任何文件。执行归档时需逐批操作并验证。
- 6 对版本关系尚未 diff 确认，执行前需完成核对。
- `hardened/deploy-profiles/` 下有 `.gitattributes` 和 `.githooks/pre-commit`，暗示曾从某 git 仓库复制，需确认上游仓库关系。
- `core.py.orig` / `core.py.patched` 暗示有对上游 vLLM 源码的补丁操作，需确认补丁是否已合入上游。

---

## 📚 数据来源 & 成员产出索引

- **Docu（技术文档师）原始产出**：`_docu_cleanup_plan_raw.md`（224 个文件分类 + 归档方案 + 5 个基线文档建议 + 8 对版本关系核对 + git 化建议）
- **扫描范围**：`C:\Users\novAI\WorkBuddy\集群部署\` 根目录及子目录

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
