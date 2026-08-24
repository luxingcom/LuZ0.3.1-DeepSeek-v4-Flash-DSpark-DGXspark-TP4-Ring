# DGX Spark 四机小问题修复批次汇总报告

**日期**：2026-08-13
**工作流**：修复批次（基于双向审核结论的低风险整改执行）
**参与成员**：Rex（SRE·系统级 8 项）、Cody（代码审查·脚本级 3 项）、Zhen（汇编+抽检验收）
**范围**：node01~04（<NODE_IP>~189）
**红线**：全程未重启/停止任何 vLLM 容器与 aicad 业务容器；唯一例外 Prometheus SIGHUP 优雅重载；每处修改前备份至 `<INSTALL_DIR>/backups/fix-20260813/`

---

## 📌 TL;DR（执行摘要）

- 11 项低风险修复**全部完成并验证通过**：系统级 8 项（Rex）+ 脚本级 3 项（Cody），零回滚、零异常、零业务中断。
- 安全面显著收窄：明文 sudo 密码从文档体系清零、8 份含 master_key 备份归档、sshd 四机硬化（密码认证关闭）、核心脚本权限收敛 750。
- 配置与文档一致化：时区四机统一 UTC、NCCL MD5 陈旧值（4cc43e3b→b7784b49）三处文档更新、部署指南 v1.1 同步 01/02 镜像（md5 三处一致）、Prometheus 失效目标清理。
- 主理人独立抽检：时区/sshd/TP4 服务/明文残留五项复测全部通过；TP4 四容器 Up 8h(healthy)、rank 映射与修正后文档一致（01=0、02=1、03=3、04=2）。
- 涉及决策/停机的事项（P0 密码轮换、API key 环境化、.local-backup 删除、双 Grafana 去重、aicad 端口收敛、registry-mirrors）**未执行**，已明确记录待裁决。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（11/11 完成，独立抽检全过） |
| 修复项 | 系统级 8 + 脚本级 3 |
| 回滚/异常 | 0 / 0 |
| 业务中断 | 无（TP4 容器全程 Up healthy，RestartCount=0） |
| 遗留待裁决 | 6 项（P0×2 / 决策×3 / 策略×1） |
| 备份位置 | 四机 `<INSTALL_DIR>/backups/fix-20260813/` |

---

## 一、修复明细（11 项，全部 ✅）

### 系统级（Rex，8 项）

| # | 修复项 | 关键证据 |
|---|--------|---------|
| 1 | 01 时区 Asia/Hong_Kong → UTC | 四机 `Time zone: Etc/UTC` 一致；list-timers 无异常 |
| 2 | 明文 sudo 密码从 docs 清除 | file-registry.md:102 改占位符；四机 `grep -rc <PASSWORD> docs/` 全部为 0 |
| 3 | 权限收敛 | 四机 scripts 750；02 litellm config.yaml 600；**8 份**含 master_key 的 .bak/dual-bak 全部归档（比预估多 3 份） |
| 4 | 01 回滚锚点 + 陈旧脚本归档 | `.bak-tp4-20260813` 已补；7424B 陈旧 worker 脚本确认无引用后归档（未删） |
| 5 | NCCL MD5 陈旧值更新 | rollback-anchors §2.1 + runbook §A.3 + **quickref §4:68**（额外发现）→ b7784b49...，加"v3 双口已上线"注记；grep 残留=0 |
| 6 | 部署指南 v1.1 同步 01/02 | 本地=01=02 md5 一致（310fd52b...） |
| 7 | sshd 四机硬化 | drop-in `99-hardening.conf`；5 步安全协议全过（改前验证→sshd -t→reload→改后新会话 ALIVE）；`sshd -T` 四机 password/kbd=no、root=prohibit-password |
| 8 | Prometheus 失效目标清理 | 移除 188:8001（worker 无 vLLM API）；SIGHUP 优雅重载；四机 dcgm/node 目标全 up |

### 脚本级（Cody，3 项）

| # | 修复项 | 关键证据 |
|---|--------|---------|
| 9 | monitor_tp4_worker.sh 死代码清理（02/03/04） | 删除 exit 1 后不可达 2 行；三节点 bash -n 通过、diff identical；未重启 monitor（下次自然生效） |
| 10 | shim-deploy.sh 关键步骤失败即停（01） | 未加全局 set -e（保守）；runsudo/md5_of 去吞错误；部署三步+MD5 校验改 `|| return 1` 即停；`check` 只读校验四机 v8 一致 exit=0 |
| 11 | worker unit 死配置清理（02/03/04） | 先验证 start_tp4_worker.sh:106 无条件硬编码 4 口 HCA 后才删除 unit Environment 死配置；daemon-reload（未 restart）；服务 active、容器 Up healthy |

---

## 二、主理人独立抽检（5 项全过）

| 抽检项 | 结果 |
|--------|------|
| shim-deploy.sh 属主统一（<USER>:<USER> 750，与同目录一致） | ✅ |
| 时区四机 `Etc/UTC` | ✅ |
| sshd 四机生效配置（sudo sshd -T） | ✅ |
| TP4 服务：01 head active + 02/03/04 worker active；四容器 Up 8h(healthy)；rank 名与环序一致 | ✅ |
| 明文密码残留（01/02 docs grep） | ✅ =0 |

> 备注：Cody 遗留裁决项"shim-deploy.sh 属主 root:root"已由主理人裁决统一为 <USER>:<USER> 750（与同目录脚本一致，脚本内 sudo 逻辑不受影响，check 校验仍通过）。

---

## ✅ 行动清单（遗留待裁决项）

| # | 行动 | 类别 | 紧急度 | 说明 |
|---|------|------|--------|------|
| 1 | sudo 密码轮换（四机统一口令，暴露面历史存在） | 安全 | **P0** | 需维护窗口；轮换后更新 secrets 指引 |
| 2 | vLLM API key 环境变量化（现硬编码于 start 脚本 --api-key） | 安全 | **P0** | 需重启 TP4 容器，须维护窗口 |
| 3 | .local-backup 312G 删除裁决 | 容灾决策 | P1 | 删= NFS 无兜底全链断风险；建议保留至 NFS 稳定 ≥7 天 |
| 4 | 双 Grafana 去重（保 02 权威） | 运维决策 | P1 | 删除实例属破坏性操作，需确认无面板依赖 |
| 5 | Neo4j/MinIO 端口收敛（0.0.0.0 → 管理网白名单） | 安全 | P1 | 需重建 aicad 容器（业务中断窗口） |
| 6 | registry-mirrors 公网加速器移除 | 策略 | P2 | 需确认镜像拉取依赖后执行 |

---

## ⚠️ 待完善 / 已知局限

- 本批次为**低风险整改**：所有涉及服务重启、容器重建、凭据轮换的操作均未执行，等维护窗口与用户裁决。
- 脚本修改（monitor/shim/unit）已 `bash -n` 与只读校验，但**未做变更后重启演练**——下次自然重启/维护窗口应复核（尤其 shim-deploy 的 deploy 路径）。
- 文档明文密码清除只覆盖 `<INSTALL_DIR>/docs/`；历史备份、shell history、旧归档中可能仍含明文（Rex 已在 .bak 归档时注意，但历史 history 未清）。
- Prometheus job 旧命名（.55/.58/.59/.60）本轮未动（会破坏 Grafana 面板），列为面板协同改造项。

---

## 📚 数据来源 & 成员产出索引

- Rex 原始产出：`deliverables/engineering-assurance/_fix_20260813/sre-fix-batch.md`（8 项操作+前后证据+备份清单）
- Cody 原始产出：`deliverables/engineering-assurance/_fix_20260813/code-fix-batch.md`（3 项 diff+验证+裁决建议）
- 修复依据：`deliverables/engineering-assurance/audit-doc-vs-server-tp4-2026-08-13.md`（双向审核总报告）
- 服务器备份：四机 `<INSTALL_DIR>/backups/fix-20260813/`

---

> 本报告由工程保障团队 AI 协作生成（2026-08-13），P0 安全项（密码轮换/API key）请人类工程负责人尽快安排维护窗口执行。
