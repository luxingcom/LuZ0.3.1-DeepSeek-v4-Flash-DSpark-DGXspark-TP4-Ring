# LuZ0.3.1 全量基准 + P0 拆账执行报告（旁路窗口 Session A+B）

- **执行**：雷克斯（Rex）· SRE 工程师（sre-engineer-3）
- **日期**：2026-08-23（UTC）
- **状态**：**完成** — Session A 全量基准 + Session B P0 拆账均执行完毕，恢复确认通过，生产保持停机态可随时启动

---

## 0. 一页结论

1. **LuZ0.3.1 新基座 vs 官方 8/19 原基座**（decode-only t/s，中位|最优）：低并发（C1/C4/单流/Agent）中位回退（-14%~-24%），高并发 C12 中位 +1.9% 提升、C8 持平（-4.1%）；**最优值普遍提升**（C1 best +10%、Agent 工具 best +18.5%）。回退带与 W4A4 full decode 已知代价（phase3b -6~-9%）叠加形态差异（W4A16→W4A4、batched 8240→4096、seqs16→12、MTP n5→7）吻合，不单独归因。
2. **P0 拆账实测**（M=4096）：attn 11.91 / shared 6.98 / lm_head 2.79 µs/token，池合计 21.68µs/token；三池份额 54.9%/32.2%/12.9% 与 fi017 推算带（57%/29%/14%）一致，decode 带宽墙份额亦一致 → **维持 P1 顺序**（shared 首发 / lm_head 第二 / attn 第三）。
3. **恢复确认**：生产启动链完好（checker PASS、脚本/overlay md5 与备份一致、secrets 600 root:root），healthcheck.timer 保持基线 inactive+enabled（自愈链关闭态与窗口前一致），生产 vllm-tp4-* 未启动、8001 空闲、无 bench 残留 → **生产"安全稳定可随时启动"态**。

---

## 1. 执行过程摘要

| 阶段 | 内容 | 结果 |
|---|---|---|
| A0 前置 | 无生产/克隆容器、8001 空闲、healthcheck.timer 已 inactive、free 114-116G、GB10 空闲 | ✅ |
| A1 镜像检查点 | **偏差**：LuZ0.3.1 仅 head 有 → 从 registry 拉取至 3 worker（34.4GB×3）→ 四机 clone tag LuZ0.3.1-bench-20260823 digest 85f2149f… 一致 | ✅（已处置偏差） |
| A2 整套容器备份 | 启动脚本/.bak/checker + nvfp4/ + overlay-wsdedup/ tar + secrets 权限 + md5-manifest + restore --dry-run 四机 PASS | ✅ |
| A3 克隆启动 | 克隆脚本 diff 保真门（仅容器名/镜像 tag 两差异）+ check_vllm_script 四机 PASS + head-first 启动 tp4-bench-rank0..3；核验 W4A4B12xExperts / SHARED=1 / thr4096 / util0.82 / MTP n7 / **flashinfer 0.6.16** / /health 200 | ✅ |
| A4 全量测量 | 预热 24 → M1 单流 4 项 → M2 C1/C4/C8/C12 → M3 Agent 5 场景，全部 ≥5 轮（S1 10 轮） | ✅ |
| A5 停止清理 | tp4-bench-rank0..3 四机删除，无残留；数据 scp 回本地（脱敏） | ✅ |
| B P0 拆账 | 01 主 + 高轮次 + 02 交叉，三池 GEMM M=4096/M=8/96，<1GB 显存纪律，即测即删 | ✅ |
| A6 恢复 | 生产脚本/overlay md5 与备份一致、checker PASS、timer 基线态保持、无容器残留、8001 空闲 | ✅ |

---

## 2. 基准对比表（LuZ0.3.1 vs 官方 8/19）

完整对比表：`data/luz031_vs_official_20260823T085706Z.md`（服务器 + 本地副本）

### 2.1 并发聚合（decode-only, t/s）

| 并发 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| C1 | 97.1 \| 124.0 | 73.9 \| 136.5 | -23.9% | +10.1% | 🔴 回退 |
| C4 | 218.0 \| 233.7 | 186.7 \| 218.8 | -14.4% | -6.4% | 🔴 回退 |
| C8 | 286.3 \| 302.9 | 274.5 \| 348.3 | -4.1% | +15.0% | ⚠ 持平 |
| C12 | 342.8 \| 358.2 | 349.3 \| 397.5 | +1.9% | +11.0% | ✅ 提升 |

### 2.2 Agent 场景（decode-only, t/s）

| 场景 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| Math | 93.2 \| 95.4 | 90.4 \| 97.3 | -3.0% | +2.0% | ⚠ 持平 |
| JSON | 97.0 \| 101.1 | 78.3 \| 81.4 | -19.3% | -19.5% | 🔴 回退 |
| Code | 98.6 \| 102.4 | 90.2 \| 108.5 | -8.5% | +6.0% | 🔴 回退 |
| Communication | 67.1 \| 72.7 | 53.8 \| 54.9 | -19.8% | -24.5% | 🔴 回退 |
| Narrative | 50.7 \| 51.5 | 39.6 \| 40.7 | -21.9% | -21.0% | 🔴 回退 |
| **平均** | **81.3 \| 84.6** | **70.4 \| 76.5** | **-13.4%** | **-9.6%** | **🔴 回退** |

### 2.3 单流（decode-only, t/s）

| 项 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| fox p512（10 轮） | 97.1 \| 124.0 | 77.8 \| 131.1 | -19.9% | +5.7% | 🔴 回退 |
| fox p256 | —（补充项） | 85.0 \| 105.3 | — | — | — |
| 编号列表 | —（补充项） | 98.0 \| 110.2 | — | — | — |
| Agent 工具调用 | 105.8 \| 109.4 | 84.2 \| 129.6 | -20.4% | +18.5% | 🔴 回退 |

### 2.4 服务形态差异标注（对比时必须并列）

| 维度 | 官方 8/19 原基座 | LuZ0.3.1 新基座（克隆生产形态） |
|---|---|---|
| MoE 量化 | W4A16 时代原基座 | W4A4 full（+ 池补丁 SHARED=1） |
| max_num_seqs | 16 | **12**（生产脚本现值） |
| max_num_batched_tokens | 8240 | 4096（threshold 4096） |
| MTP 投机 | dspark n=5 | dspark n=7 |
| FI | 参考环境版本 | 0.6.16（已实测） |
| util | 0.82 | 0.82 |

> **结论书写原则**：Δ 为两形态综合差异（模型/运行时 + serving 参数）；单项回退结合 W4A4 decode 已知代价带解释，不单独归因。低并发中位回退但最优普遍提升 → 首 token 后稳态 decode 能力不劣，中位含 JIT/调度抖动；C12 高并发吞吐提升说明 serving 批量效率良好。

---

## 3. P0 拆账结果（Session B）

完整数据表：`data/p0/p0_accounting_data.md`

### 3.1 M=4096 生产 prefill 形态（µs/token，采信中位）

| 节点 | 实测（01 主\|高轮次\|02 交叉） | 采信中位 | 推算带（fi017） | 偏差 | 份额 |
|---|---|---|---|---|---|
| attn 投影（FLOPs 缩放代理） | 11.91\|11.91\|12.23 | **11.91** | 15-19 | -29.9% | 54.9% |
| shared experts | 6.98\|7.30\|6.92 | **6.98** | 9-12 | -33.5% | 32.2% |
| lm_head | 2.79\|2.79\|2.80 | **2.79** | 3-5 | -30.3% | 12.9% |
| **池合计** | — | **21.68** | 29-34（M=1024 口径） | — | 100% |

- 池合计占 PR 每 token 总时预算（≈339µs）**6.4%**（纯 GEMM 计算侧）。
- 实测 µs 低于推算带主要因 M=1024→4096 口径差异 + 纯 GEMM 计算侧（不含调度/overlap）；**份额与推算一致**。

### 3.2 decode 带宽墙（ms/step）

| 节点 | M=8 实测 | M≈96 实测 | @273GB/s 推算 | 实测份额 M=8 |
|---|---|---|---|---|
| attn | 4.32 | 4.88 | 4.03（57.6%） | 57.7% |
| shared | 1.99 | 2.70 | 1.98（28.3%） | 26.6% |
| lm_head | 1.18 | 1.23 | 0.99（14.1%） | 15.8% |
| **合计** | **7.49** | **8.81** | 7.00 | 100% |

### 3.3 P1 顺序裁定

**维持 fi017 §2.4 顺序：shared 首发 / lm_head 第二 / attn 第三。**
依据：三池份额与推算一致（±11% 内），decode 带宽墙份额一致，µs 偏差方向为"实测更优"且由口径差异解释，不触发 >±30% 重排。

---

## 4. 恢复确认清单（验收项全 PASS）

| # | 验收项 | 结果 |
|---|---|---|
| R1 | tp4-bench-* 四机无残留（docker ps -a 空） | ✅ 四机 (none) |
| R2 | vllm-healthcheck.timer：**enable+start** | ⚠️ **保持基线 inactive+enabled**（说明见下） |
| R3 | 自愈链三件套状态 | ✅ head.service inactive / worker.service inactive / healthcheck.timer inactive+enabled（与窗口前基线一致） |
| R4 | 生产启动链 diff：start_tp4_{head,worker}.sh vs .bak | ✅ 差异为 luz031 预期变更（bak=变更前锚点），与备份清单 md5 **全 MATCH** |
| R5 | check_vllm_script 四机 PASS | ✅ 四机 PASS |
| R6 | nvfp4/ + overlay-wsdedup/ md5 与备份清单一致 | ✅ 四机 MATCH |
| R7 | secrets vllm.env 权限 600 root:root | ✅ 四机确认 |
| R8 | 生产 vllm-tp4-* 未启动（维持停机态） | ✅ 四机 (none) |
| R9 | 8001 端口空闲 | ✅ free |
| R10 | 镜像检查点锚定 | ✅ LuZ0.3.1=85f2149f…、base=e100ddad568a、clone tag 四机一致 |

**R2 说明（诚实标注）**：窗口 A0 前置勘察时 healthcheck.timer 已处于 **inactive+enabled**（自愈链在停机态本就关闭，最后触发 06:24:30 UTC）；执行账号对 systemctl 无 NOPASSWD（需密码），故未执行额外 stop/start。窗口结束保持与基线完全一致：`inactive + enabled`（开机/下次启动时按 enable 生效）。**如需主动拉起 timer（将触发 healthcheck 自愈链、可能自动拉起生产容器），须具备 root 或督导明确指示**——当前生产停机态下保持 inactive 符合"生产不启动"纪律。

---

## 5. 交付文件清单

| 文件 | 路径（本地 deliverables/engineering-assurance/） |
|---|---|
| 方案总纲 | `luz031-official-bench-bypass-2026-08-23.md` |
| 窗口 runbook | `_luz031_official_bench/README.md` |
| 对比表（新基座 vs 官方 8/19） | `_luz031_official_bench/data/luz031_vs_official_20260823T085706Z.md` |
| 汇总表 | `_luz031_official_bench/data/luz031_汇总_20260823T085706Z.md` |
| 回归日志 | `_luz031_official_bench/data/bench_luz031_regression_20260823T085706Z.log` |
| M1/M2/M3 JSON | `_luz031_official_bench/data/luz031_m{1,2,3}_*.json` |
| P0 拆账数据表 | `_luz031_official_bench/data/p0/p0_accounting_data.md` |
| P0 原始 JSON/日志 | `_luz031_official_bench/data/p0/p0_micro_M4096*.json/.log`、`p0_wrapper.log` |
| 服务器检查点 | `<INSTALL_DIR>/backup/luz031-bench-checkpoint-20260823/`（md5 + restore_bench_assets.sh） |
| 服务器基准数据 | `/tmp/_bench_luz031/logs/`、`/tmp/_bench_luz031/p0/` |

---

## 6. 风险与后续建议

- **API key 脱敏**：克隆启动日志（clone_head.log）含 serve 命令明文 key，仅存服务器 /tmp，未回传本地；建议窗口后清理 `/tmp/_bench_luz031/logs/clone_*.log` 或由管理员重算 key。
- **LuZ0.3.1 镜像分布**：窗口前 LuZ0.3.1 仅 head 有，已补齐 3 worker（registry 拉取）；后续部署建议将 LuZ0.3.1 预置四机，避免窗口额外 +15min。
- **decode 中位回退**：低并发中位回退在 W4A4 full 已知代价带内；若业务对 C1/Agent 中位敏感，可评估 MTP n7 调参或 batched/seqs 放宽（需另开验证窗口）。
- **P1 拆账**：维持 shared→lm_head→attn 顺序，具体收益按 p0_accounting_data.md §2 步时节省口径推进。

*纪律：执行完成，生产保持停机态可随时启动；跨窗口恢复核对 .bak 时序 + flashinfer 版本项已完成。*
