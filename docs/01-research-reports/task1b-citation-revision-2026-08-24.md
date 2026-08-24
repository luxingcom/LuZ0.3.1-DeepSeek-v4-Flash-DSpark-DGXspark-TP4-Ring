# 任务一（阶段 B 前置）· 缺失报告清单 + 引用修订映射表（第一轮）

- **汇编**：多库（Docu）· 技术文档师（tech-writer-1）· 工程保障团队
- **日期**：2026-08-24
- **范围**：18 份被引用但本地缺失报告 + 6 条关键证据链的「旧路径→新路径」引用修订映射（按开源仓库 8 大目录布局）
- **说明**：本清单为**执行前置**——实际文件移动/引用改写在开源副本上执行；本地工作区保持现状（只读纪律）

---

## 1. 缺失报告清单（18 份）

### 1.1 仅存于服务器交付副本（7 份，需回传本地）

来源：`node01:<INSTALL_DIR>/deliverables/engineering-assurance/`（08-15~08-17 era，本地工作区未同步）

| # | 文件名 | 服务器路径（回传源） | 建议新路径（开源库） | 主题 |
|---|---|---|---|---|
| S1 | `b1-compat-adjudication-criteria-architect-2026-08-17.md` | `<INSTALL_DIR>/deliverables/engineering-assurance/b1-compat-adjudication-criteria-architect-2026-08-17.md` | `docs/06-verification/` | B1 兼容性裁定 |
| S2 | `nccl-2hop-s3-final-adjudication-architect-2026-08-17.md` | 同上目录 | `docs/04-issues/` 或 `docs/01-research-reports/` | NCCL 2-hop S3 终裁 |
| S3 | `nccl-final-performance-baseline-2026-08-17.md` | 同上目录 | `docs/02-performance-benchmarks/` | NCCL 最终性能基线 |
| S4 | `nccl-maxch16-e2e-verification-2026-08-16.md` | 同上目录 | `docs/06-verification/` | NCCL MAXCH16 验证 |
| S5 | `nccl-p0-scan-results-2026-08-16.md` | 同上目录 | `docs/06-verification/` | NCCL P0 扫描 |
| S6 | `nccl-proto-threshold-scan-2026-08-16.md` | 同上目录 | `docs/06-verification/` | NCCL proto 阈值扫描 |
| S7 | `nccl-stageb-verification-2026-08-16.md` | 同上目录 | `docs/06-verification/` | NCCL stageB 验证 |

> **回传命令参考**（只读拉取，不删服务器文件）：
> `scp node01:<INSTALL_DIR>/deliverables/engineering-assurance/{S1..S7} ./`
> 回传后建议与服务器 `md5sum` 核对一致性。

### 1.2 两处均缺失 / 疑似命名漂移（11 份，对照 6 条证据链推断正确名）

| # | 被引用名（本地缺失） | 引用它的报告 | 推断正确名 | 依据 / 处置 |
|---|---|---|---|---|
| D1 | `deploy-f-dynamic-k-baseline-2026-08-05.md` | dynk-megamoe-version-ab / dynk-megamoe-research（"部署基线记录"） | **`deploy-f-probabilistic-switch-2026-08-05.md`**（本地存在） | 任务书笔名/旧工作名；F 部署基线实际落地为此文件 |
| D2 | `hf-nvfp4-mirror-survey-2026-08-13.md` | sglang-nvfp4-tp4-setup-plan（`delivery/nvfp4-investigation/hf-nvfp4-mirror-survey-2026-08-13.md`） | **内容并入 nvfp4 系列交付物**（`nvfp4-upgrade-execution-analysis-2026-08-13.md` / `research-nvfp4-alternative-runtimes-2026-08-13.md`） | delivery/ 子路径未在本地/服务器落地为独立文件；需人工确认是否需从 nvfp4 交付包还原 |
| D3 | `kernel-design-2026-08-21.md` | b12x-tail-path-strategy / engineering-assets-report（"routeb-merged-kernel-design-2026-08-21"） | **`routeb-merged-kernel-design-2026-08-21.md`**（本地存在） | 引用省略前缀；实际文件即 routeb-merged-kernel-design |
| D4 | `kvssd-util07-result-2026-08-19.md` | kvssd-util07-verify-plan（"对比表（G3 后补）"） | **`kvssd-rca-memutil07-2026-08-19.md`**（本地存在） | 计划产出被 RCA 报告承接；引用改写为 RCA 报告 |
| D5 | `kvssd-vs-baseline-compare-2026-08-19.md` | kvssd-perf-test-plan（"对比表（模板 §2.4 填完）"） | **`kvssd-200g-execution-report-2026-08-19.md`** 或 `kvssd-mem-composition-final-2026-08-19.md` | 对比结果实际落地于执行报告/最终汇编；需人工确认以哪个为准 |
| D6 | `miaai-2026-08-13.md`（引用为 `research_addendum_miaai-2026-08-13.md`） | research-miaai-dspark-comparison / _fix_20260813/issue22-patch（"8/13 首轮补研"） | **`miaai-cross-verify-2026-08-14.md`** / `research-miaai-dspark-comparison-2026-08-14.md` | addendum 未落地为独立文件；内容并入 08-14 交叉验证/对比报告 |
| D7 | `nvfp4-investigation-2026-08-13.md` | sglang-nvfp4-test-plan / _fix_20260813/l2-build-plan（`delivery/nvfp4-investigation/nvfp4-investigation-2026-08-13.md`） | **内容并入** `nvfp4-upgrade-execution-analysis-2026-08-13.md` / `research-nvfp4-alternative-runtimes-2026-08-13.md` | delivery/ 子路径未落地；nvfp4-investigation 主题（权重转换/Marlin 后端）见 l2-build-plan 引用 |
| D8 | `prde-bottleneck-analysis-2026-08-22.md` | ar-optimization（"昨日 prde-bottleneck 口径"） | **`pr-de-bottleneck-analysis-2026-08-22.md`**（本地存在） | 文件名连写 prde vs pr-de；实际文件即 pr-de-bottleneck-analysis |
| D9 | `prob-eval-report-2026-08-05.md` | combo-ab-prob-eval / bench-f-baseline（"P0/P0t/P1/P2 矩阵"） | **`combo-ab-prob-eval-2026-08-05.md`**（本地存在） | prob-eval 数据实际落在 combo-ab-prob-eval 综合报告 |
| D10 | `v027-nvfp4-acceptance-result-2026-08-15.md` | v027-nvfp4-acceptance-plan（"对比表（模板 §2.2 填完）"） | **`v027-nvfp4-acceptance-2026-08-15.md`**（本地存在） | 计划产出与验收报告同名异写（result 后缀未落地） |
| D11 | `v027-vs-v026-perf-compare-2026-08-14.md` | test-v027-perf-ab-plan（"对比表（模板 §3.1 填完）"） | **实际对比见** `v027-nvfp4-acceptance-2026-08-15.md` / `research-vllm-027-2026-08-12.md` / `vllm-027-tp4-smoke-2026-08-14.md` | 计划产出未落地为独立文件；需人工确认最佳承接报告 |

> **处置约定**：D 类 11 份中 6 份（D1/D3/D4/D8/D9/D10）可**直接改引用为已存在的实际文件**；5 份（D2/D5/D6/D7/D11）为"计划产出被其他报告承接"，需人工确认承接报告后统一改写引用，不可猜测。

---

## 2. 6 条关键证据链的引用修订映射表（旧路径 → 新路径）

> 布局目标（开源仓库 `LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring`）：
> `docs/01-research-reports/`、`docs/02-performance-benchmarks/`、`docs/03-final-metrics/`、`docs/04-issues/`、`docs/05-kernels-patches/`、`docs/06-verification/`、`docs/07-deployment/`、`docs/08-tools/`
> 旧路径基准 = `deliverables/engineering-assurance/<file>.md`

### 链 1 · W4A4 翻案链（threshold 主线）

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `routing-recapture-scheduler-ab-2026-08-22.md` | `docs/02-performance-benchmarks/routing-recapture-scheduler-ab-2026-08-22.md` | 基准 |
| 2 | `p1-p2-research-2026-08-22.md` | `docs/01-research-reports/p1-p2-research-2026-08-22.md` | 研究 |
| 3 | `threshold-retest-2026-08-22.md` | `docs/02-performance-benchmarks/threshold-retest-2026-08-22.md` | 基准 |
| 4 | `threshold-4096-adoption-2026-08-22.md` | `docs/07-deployment/threshold-4096-adoption-2026-08-22.md` | 部署/采纳 |
| 5 | `wsdedup-l3-combo-2026-08-23.md` | `docs/05-kernels-patches/wsdedup-l3-combo-2026-08-23.md` | 补丁 |
| 6 | `w4a4-ext-2026-08-23.md` | `docs/02-performance-benchmarks/w4a4-ext-2026-08-23.md` | 基准 |
| 7 | `luz031-deployment-2026-08-23.md` | `docs/07-deployment/luz031-deployment-2026-08-23.md` | 部署 |

> 链上引用该链的旁支报告：`phase-summary-2026-08-23.md`、`w4a4-vs-w4a16-diff-audit-2026-08-23.md`、`bench-regression-attribution-2026-08-23.md` 引用上述文件处均须同步改路径。

### 链 2 · b′ native 共享路线

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `a3-hybrid-slim-design-2026-08-23.md` | `docs/01-research-reports/a3-hybrid-slim-design-2026-08-23.md` | 研究/设计 |
| 2 | `bprime-impl-2026-08-23/` | `docs/05-kernels-patches/bprime-impl-2026-08-23/` | 补丁实现 |
| 3 | `bprime-window-2026-08-23.md` | `docs/02-performance-benchmarks/bprime-window-2026-08-23.md` | 基准/窗口 |

> 关联：`phase-summary` §3.3、`luz031-deployment` §8 引用 bprime 系列，需同步。

### 链 3 · 环境 stall 调查与判决

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `slowround-rootcause-2026-08-22.md` | `docs/04-issues/slowround-rootcause-2026-08-22.md` | 缺陷 |
| 2 | `expd-r123-2026-08-23.md` | `docs/04-issues/expd-r123-2026-08-23.md` | 缺陷 |
| 3 | `ringonly-w1-2026-08-23.md` | `docs/05-kernels-patches/ringonly-w1-2026-08-23.md` | 补丁+发现 |
| 4 | `envstall-rootcause-2026-08-23.md` | `docs/04-issues/envstall-rootcause-2026-08-23.md` | 缺陷 |
| 5 | `expverdict-verdict-2026-08-23.md` | `docs/04-issues/expverdict-verdict-2026-08-23.md` | 缺陷 |
| 6 | `arstall-production-closure-2026-08-23.md` | `docs/06-verification/arstall-production-closure-2026-08-23.md` | 验证/闭环 |

> 关联资产：`_arstall_closure_20260823/`、`_audit/`、`evdata/` → `docs/08-tools/` 或 `data/`。

### 链 4 · FP8 探索链（shared-FP8 + lm_head）

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `opt-routeb-fp8-2026-08-23.md` | `docs/01-research-reports/opt-routeb-fp8-2026-08-23.md` | 研究 |
| 2 | `fp8-quality-impact-2026-08-23.md` | `docs/06-verification/fp8-quality-impact-2026-08-23.md` | 验证/质量 |
| 3 | `lmhead-fp8-project-2026-08-23.md` | `docs/01-research-reports/lmhead-fp8-project-2026-08-23.md` | 研究 |
| 4 | `lmhead-fp8-f0-contract-2026-08-24.md` | `docs/06-verification/lmhead-fp8-f0-contract-2026-08-24.md` | 验证/契约 |
| 5 | `fp8-f1-window-2026-08-24.md` | `docs/02-performance-benchmarks/fp8-f1-window-2026-08-24.md` | 基准/窗口 |
| 6 | `fp8-quality-gate-toolchain-2026-08-24.md` | `docs/08-tools/fp8-quality-gate-toolchain-2026-08-24.md` | 工具链 |

> 关联资产：`_fp8_qg_toolchain/`、`lmhead-fp8-f0-golden/`、`_rex/` → `docs/08-tools/` + `data/`；服务器 `/tmp/_fp8_f1/`（384MB head bin）不入库（大二进制）。

### 链 5 · issue22 处置链

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `issue22-kv-deopt-2026-08-24.md` | `docs/04-issues/issue22-kv-deopt-2026-08-24.md` | 缺陷 |
| 2 | `issue22-400k-verification-2026-08-24.md` | `docs/06-verification/issue22-400k-verification-2026-08-24.md` | 验证 |
| 3 | `key-rotate-r12-restart-2026-08-24.md` | `docs/07-deployment/key-rotate-r12-restart-2026-08-24.md` | 部署/运维窗口 |
| 4 | `b12x-gate-fix-2026-08-24.md` | `docs/07-deployment/b12x-gate-fix-2026-08-24.md` | 部署修复 |
| 5 | `_issue22_ab_archive_20260824/`、`_issue22_400k_verification_20260824/` | `data/issue22/` | 数据 |

### 链 6 · FI 0.6.16 落地链

| 顺序 | 旧路径（当前） | 新路径（开源库） | 类别 |
|---|---|---|---|
| 1 | `fi-rebase-eugr-prep-2026-08-22.md` | `docs/01-research-reports/fi-rebase-eugr-prep-2026-08-22.md` | 研究 |
| 2 | `windowA-fi-cg-budget-2026-08-23.md` | `docs/02-performance-benchmarks/windowA-fi-cg-budget-2026-08-23.md` | 基准/窗口 |
| 3 | `fi016-replacement-2026-08-23.md` | `docs/07-deployment/fi016-replacement-2026-08-23.md` | 部署替换 |
| 4 | `luz031-deployment-2026-08-23.md` | `docs/07-deployment/luz031-deployment-2026-08-23.md` | 部署（同链 1） |

---

## 3. 高频被引文件的修订（被引次数 ≥5，需在副本上优先更新引用）

| 文件 | 被引次数 | 新路径 |
|---|---|---|
| `rollback-anchors-2026-08-12.md` | 30 | `docs/07-deployment/rollback-anchors-2026-08-12.md` |
| `tp4-service-deployment-guide-2026-08-13.md` | 16 | `docs/07-deployment/tp4-service-deployment-guide-2026-08-13.md` |
| `upstream-check-perf-ceiling-2026-08-23.md` | 10 | `docs/01-research-reports/upstream-check-perf-ceiling-2026-08-23.md` |
| `nvfp4-testkit-diagnosis-2026-08-19.md` | 10 | `docs/06-verification/nvfp4-testkit-diagnosis-2026-08-19.md`（子目录，路径修正） |
| `tp4-r12-final-report-2026-08-13.md` | 9 | `docs/02-performance-benchmarks/tp4-r12-final-report-2026-08-13.md` |
| `litellm-api-key-manual-2026-08-05.md` | 9 | `docs/07-deployment/litellm-api-key-manual-2026-08-05.md`（含 key 纪律，脱敏后入） |
| `bench-regression-attribution-2026-08-23.md` | 9 | `docs/02-performance-benchmarks/bench-regression-attribution-2026-08-23.md` |
| `architecture-nvfp4-2026-08-20.md` | 9 | `docs/01-research-reports/architecture-nvfp4-2026-08-20.md` |
| `slowround-rootcause-2026-08-22.md` | 8 | `docs/04-issues/slowround-rootcause-2026-08-22.md` |
| `luz031-deployment-2026-08-23.md` | 8 | `docs/07-deployment/luz031-deployment-2026-08-23.md` |
| `fi017-p0-accounting-2026-08-23.md` | 8 | `docs/01-research-reports/fi017-p0-accounting-2026-08-23.md` |
| `w4a4-ext-2026-08-23.md` | 7 | `docs/02-performance-benchmarks/w4a4-ext-2026-08-23.md` |
| `a3-hybrid-slim-design-2026-08-23.md` | 7 | `docs/01-research-reports/a3-hybrid-slim-design-2026-08-23.md` |
| `runbook-dspark-vllm-2026-08-06.md` | 7 | `docs/07-deployment/runbook-dspark-vllm-2026-08-06.md` |
| `wsdedup-l3-combo-2026-08-23.md` | 6 | `docs/05-kernels-patches/wsdedup-l3-combo-2026-08-23.md` |
| `e2e-baseline-prod-2026-08-21.md` | 6 | `docs/02-performance-benchmarks/e2e-baseline-prod-2026-08-21.md` |
| `expd-r123-2026-08-23.md` | 6 | `docs/04-issues/expd-r123-2026-08-23.md` |
| `opt-routeb-fp8-2026-08-23.md` | 6 | `docs/01-research-reports/opt-routeb-fp8-2026-08-23.md` |
| `lmhead-fp8-project-2026-08-23.md` | 6 | `docs/01-research-reports/lmhead-fp8-project-2026-08-23.md` |
| `analysis-tp2-tp4-communication-2026-08-09.md` | 6 | `docs/01-research-reports/analysis-tp2-tp4-communication-2026-08-09.md` |

---

## 4. 执行顺序建议（副本上操作）

1. **回传 7 份服务器报告**（§1.1）→ md5 校验 → 归入开源库对应目录。
2. **处理 6 份可直接改名的引用**（D1/D3/D4/D8/D9/D10）：在开源副本中把引用路径改写为实际文件。
3. **人工确认 5 份承接报告**（D2/D5/D6/D7/D11）：由 team-lead 或对应角色裁定承接报告后统一改写。
4. **全库引用替换**：按 §3 高频表 + §2 六链映射，对全部 .md 做「旧文件名 → docs/0X-*/新文件名」的文本替换（在副本上，sed 批量 + 人工抽查）。
5. **更新 `03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`** 内的引用路径与阶段 A 盘点文档中的路径说明。

---

*本清单为只读产出；实际文件移动、引用改写、重命名均在开源副本上执行，本地工作区不做任何修改。*
