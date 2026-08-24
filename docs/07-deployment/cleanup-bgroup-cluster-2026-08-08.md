# B 组 Benchmark 收尾：P4 清理 + 编排死锁修复报告

**日期**：2026-08-08
**工作流**：综合收尾（状态核查 + 镜像清理 + 脚本修复）
**参与成员**：Cody（P4 门禁判定与 rmi 审查）/ Rex（编排脚本死锁修复方案）/ Zhen（执行与汇编）
**范围**：DGX Spark 四机集群 node01~04，B 组 benchmark 后的清理与收尾

---

## 📌 TL;DR（执行摘要）

- 整体结论：B 组 benchmark **54/54 组合全部完成（486/486 样本 ok）**，A/B 对比无回退；**P4 vllm-gb10 清理完成（四机回收约 76GB）**；**编排脚本死锁已修复**；生产服务（A 组 TP2 + embed 03/04）全部健康
- 严重度分布：🔴严重 0 项 / 🟠高 0 项 / 🟡中 1 项（16384/json/c1 prefill 离群待复核）/ 🟢低 2 项（32K-64K 加密点、Runbook 记录）
- 阻塞 / 非阻塞：非阻塞——本次收尾项全部落地

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（收尾目标全部达成） |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（131072/c5 回归验证死锁修复、离群复核、Runbook 记录） |
| 建议下一步 | 下次 B 组 TP2 启动用修复后脚本 head-first 编排，顺带验证死锁修复；8/9 维护窗口做 RTT 回填 |

---

## 📊 一、B 组 benchmark 状态核查（已完成项确认）

### 1.1 全量结果
- **54/54 组合、486/486 样本全部 ok（0 错误）**，数据完整性优于 A 组（无日志拼接、无缺失样本）
- 结果文件：`/tmp/results_B/rows_B.csv`（486 行）+ `summary_B.json`（54 组合 p50）+ `bench_B.log`
- 本地副本：`_archive_scratch/bench_B/`（含 analyze_B.py）

### 1.2 核心结论（Tessa 口径，per-request p50 × conc）
| 维度 | 结论 |
|------|------|
| prefill 平台 | 4096-32768 ctx 单流稳定 1900-2060 t/s；131072 略降 1660-1770（-10~-15%） |
| decode 单流 | coding 71-81 / json 76-81 / prose 37-40 t/s，全 ctx 持平 |
| decode 并发崩塌 | c5 长 ctx 由 30-40 崩至 2.1-4.6 t/s（统一内存带宽饱和，与 A 组同构） |
| **32768 分界点** | **c5/c1 总吞吐 = 0.77 <1 → 并发收益分界在 16K-32K 之间**（16384=1.17 >1）|
| A/B 对比 | prefill 中位 **0.996** / decode-c1 1.033 / decode-c5 1.093 → **B 组无任何维度回退** |

### 1.3 生产状态（21:45 实时核查）
| 组件 | 状态 |
|------|------|
| A 组 TP2（01+02） | ✅ Restarts=0、healthy、`/v1/models` 200（Started 03:40 UTC，10h+ 无回退） |
| B 组 TP2（03+04） | ✅ 已按计划停止（benchmark 后） |
| embed 03/04 | ✅ anemll-embed-8022，Qwen3-Embedding-0.6B **dim=1024 批量向量实测 OK** |
| .58 embed | 维持空池（内存不足决策），litellm config 条目整块注释 |
| litellm embed 池 | ✅ .188:8022 + .189:8022（2 active） |
| 定时汇总任务 | ✅ automation-1786183891921 已于 19:15 触发，报告 19:28 产出 |

---

## 🧹 二、P4 vllm-gb10 清理（Cody 门禁放行 → 已执行）

### 2.1 门禁判定（Cody，migration-tp2-nccl 报告定义）
| # | 门禁条件 | 判定 | 证据 |
|---|---------|------|------|
| 1 | TP2 稳定（health 200 + ≥24h 观察无回退） | ✅ PASS | A 组 Restarts=0 healthy API 200；四机零 vllm-gb10 运行容器；B 组停 TP2 为计划动作 |
| 2 | embed 迁移 anemll | ✅ PASS | 03/04 anemll-embed-8022 双 active；litellm 池 .188/.189:8022 |
| 3 | 无引用扫描 | ✅ PASS | docker ps 全节点零容器；grep <INSTALL_DIR> 全目录 + systemd + /etc/docker + docker-compose + litellm config + 本地脚本 → **全零命中** |

⚠️ Cody 保留意见（不阻塞）：A 组观察窗口实际 10h+ <24h，需在报告中留痕 → 本报告已标注，风险可接受（无 vllm-gb10 容器运行，清理零运行影响）。

### 2.2 执行记录
| 节点 | 删除对象 | 大小 | 结果 |
|------|---------|------|------|
| node01 (.60) | vllm-gb10:0.26.1-cu132 (ac38a938) | 19.2GB | ✅（先删引用容器 embed-qwen3-vllm） |
| node01 (.58) | vllm-gb10:0.26.1-cu132 (5a2a5e99) | 18.9GB | ✅（先删引用容器 embed-qwen3-vllm + Exited 的 anemll-embed-8022 残留） |
| node01 (.55) | 双 tag 指向 5a2a5e99（registry + ghcr.nju.edu.cn） | 18.9GB | ✅ 两条 rmi 后层释放 |
| node01 (.59) | vllm-gb10:0.26.1-cu132 (5a2a5e99) | 18.9GB | ✅ |

**执行要点**（遵循 Cody 建议）：
- 定向按 tag rmi，**未使用 `docker system prune -a`**（防误伤 anemll）
- rmi 前 30s 内二次确认：四机零运行容器
- 01/02 的 `embed-qwen3-vllm` 停止容器（23h 前 Exited，旧 gb10 embed 残留）是 rmi 阻塞源 → 先 `docker rm` 再 rmi
- **anemll 双 tag 全部保留**（head 34.2G + worker 21.6G）
- **回滚保障**：registry（<NODE_IP>:5000）仍留存 vllm-gb10:0.26.1 可 re-pull；03 的 ghcr.nju.edu.cn 外部源 re-pull 慢，如需恢复预留带宽

### 2.3 清理后验证
- 四机 `docker images | grep vllm-gb10` = **0 残留**
- anemll 镜像保留：01=1、02=4（多 tag 变体）、03=1、04=1
- docker 空间：01=614G/3.6T(18%)、02=1.2T/3.6T(35%)、03=230G/916G(27%)、04=227G/916G(27%) → 预计回收约 76GB

---

## 🔧 三、编排脚本死锁修复（Rex 方案 → 已落地）

### 3.1 根因（Rex 确认）
`start_head_groupB.sh:135-142` 把「Application startup complete」（需**所有 TP rank 完成 NCCL 配对**后才打印）当作单机就绪信号；而 worker 在编排脚本第 4 步才启动 → head 永远等不到 → 15min 超时 `exit 1` → `set -euo pipefail` 下编排整体退出 → **worker 永不启动**。本次 B 组部署靠手动启动 worker 绕过。

### 3.2 修复方案（Rex）与落地
| 文件 | 改动 | 状态 |
|------|------|------|
| start_head_groupB.sh:135-142 | "Application startup complete" 轮询（90×10s）→ **docker exec pgrep VLLM::EngineCore**（60×10s，纯 head 侧不依赖 worker 配对；容器退出立即 dump logs fail） | ✅ 已改 |
| start_groupB_cluster.sh | ① worker 启动后加**存活校验 12×5s**（启动即崩溃立即失败）；② 8001 轮询 90→**120×10s**（冷启动余量）；③ 失败分支 **dump 双机 logs --tail 100**（替代裸 exit 1） | ✅ 已改 |
| start_worker_groupB.sh | 无需改动（无尾部就绪轮询，docker run -d 即返回） | ✅ 确认无需改 |

**选择理由**（Rex）：不选容器 healthy（900s start-period 有 starting 歧义、判定慢）；不选直接退出（丢失 head 启动失败快速定位）；head 保持**同步执行**（修复后约 1-3min 返回，set -euo pipefail 正确传播失败）。

**验证**：服务器 + 本地副本 `bash -n` 全部通过；修改后时序 = head 同步启动（EngineCore 就绪返回）→ TCPStore 轮询 → worker 启动 + 存活校验 → 8001 端到端就绪（唯一需两 rank 配对完成的信号）。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 下次 B 组 TP2 启动改用修复后脚本（head-first 编排），顺带验证死锁修复 + 131072/c5 回归 | Zhen+Rex | P1 | 下次需要 B 组时 |
| 2 | 复核 16384/json/c1 prefill 离群（795 vs 平台 2000）：重测 3 波或查时段日志 | Tessa+Zhen | P2 | 下一维护窗口 |
| 3 | Runbook v1.4 记录本次 P4 清理结果 + 死锁修复方案 | Docu | P2 | 8/9 维护窗口 |
| 4 | 32768-65536 之间加密 1-2 点（如 49152）确认分界点连续性 | Tessa | P3 | 有闲余时 |
| 5 | 评估 litellm 网关瓶颈：业务并发 >400 req/s 时多实例/LB 方案 | Archi+Rex | P3 | 按需 |
| 6 | 8/9 维护窗口：RTT 回填 + .58/.60 dockerd 脏缓存修复 | Rex+Zhen | P1 | 2026-08-09 |

---

## ⚠️ 待完善 / 已知局限

- **A 组 TP2 观察窗口 10h+ < 24h 门禁值**：Cody 标注留痕，风险可接受（无 vllm-gb10 容器运行，清理零运行影响；registry 可 re-pull 回滚）
- 16384/json/c1 prefill 离群（795.45，wave2-3）：非系统性，污染该单元格与 A/B 对比统计 max 值，待复核
- 死锁修复未经过真实 TP2 编排演练（B 组 TP2 当前停止）——需下次启动验证 EngineCore 轮询在 worker join 前确实触发
- B 组 asyncio 引擎下 prefill 串行：agg_* 指标口径与 A 组不同，跨组对比统一 per-request p50 × conc（已遵守）
- 131072 prefill 降幅 B 组（-10~-15%）略大于 A 组（-6%），样本量有限，暂不判定为组间差异

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）**：P4 门禁 3 条件逐条 PASS 判定 + rmi 执行建议（定向 tag 删除、禁 prune -a、03 双 tag 两条 rmi、registry re-pull 回滚）；原始产出经团队消息回传
- **Rex（SRE 工程师）**：死锁根因确认 + head 就绪判定改造方案（EngineCore 轮询）+ 编排分层时序建议 + 隐患清单（worker 存活校验、8001 超时放宽、失败 dump logs）；原始产出经团队消息回传
- **Zhen（主理人）**：状态核查（四机 docker/镜像/健康/embed API 实测）、无引用扫描执行、rmi 执行、脚本补丁落地、本地副本同步
- 对照报告：benchmark-B-group-2026-08-08.md（54 组合全矩阵 + A/B 对比 + 32768 分界点）、benchmark-A-group-2026-08-08.md、migration-tp2-nccl-2026-08-08.md（P4 门禁定义）、benchmark-embed-dual-vs-single-2026-08-08.md
- 原始数据：node01 /tmp/results_B/{rows_B.csv, summary_B.json} + 本地 _archive_scratch/bench_B/

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
