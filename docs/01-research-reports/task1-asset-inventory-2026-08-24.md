# 任务一（阶段 A）· 项目资产全量盘点与开源分类结构设计

- **汇编**：多库（Docu）· 技术文档师（tech-writer-1）· 工程保障团队
- **日期**：2026-08-24
- **范围**：本地 `C:\Users\novAI\WorkBuddy\集群部署\deliverables\engineering-assurance\`（全量，只读）+ 服务器 `node01:<INSTALL_DIR>/`（SSH 只读，仅枚举清单/大小，不读大二进制）
- **用途**：为后续交叉核对、敏感信息清理、开源发布（仓库名 `LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring`）做资产底账
- **纪律**：纯只读；未修改/删除任何文件；未读取 checkpoint/镜像等大二进制内容
- **项目形态基线**：DGX Spark 4 节点 TP4（DeepSeek V4 Flash），LuZ0.3.1 生产方案 = W4A4 full + 池补丁 + FI 0.6.16 + threshold 4096 + util 0.82 + Dspark MTP n7

---

## 0. 一页结论（TL;DR）

1. **资产规模**：本地工程保障目录 **2,355 个文件（172.9 MB）**，其中 **.md 报告 477 份**（顶层 311 份 + 子目录 166 份），子目录 51 个；服务器 `<INSTALL_DIR>/` 约 **215 GB**（大头为 kvssd-images 214.7GB 与 backup 54.6GB，属镜像/快照类，不入开源库）。
2. **敏感面**：本地 311 份顶层报告中，**184 份含内部路径、146 份含内网 IP、126 份含主机名 dgxspark0X、85 份含 API key 痕迹、75 份含内部用户名、44 份含密码/secret 痕迹、58 份含 ≥32 位 hex（疑似 key）**——开源前几乎全部报告需经过脱敏（规则见 §4）。
3. **分类设计**：建议开源库 8 大目录（研究报告 / 性能测试 / 最终指标 / 缺陷问题 / 算子源码与补丁 / 验证资料 / 容器与部署 / 工具脚本），每目录给出纳入清单与脱敏要求（§3）。
4. **关联索引**：报告间引用网络密集，**最高被引 3 份**为 `rollback-anchors-2026-08-12.md`(30 次)、`tp4-service-deployment-guide-2026-08-13.md`(16 次)、`upstream-check-perf-ceiling-2026-08-23.md`(10 次)；发现 **18 份被引用但本地缺失**的报告，其中 7 份仅存于服务器交付副本（§5）。
5. **最终指标缺口**：**无单一"最终性能指标"汇总文件**。现分布在各专项报告（luz031-deployment / arstall E5 / fp8-f1-window / g1 对照等）与 `_luz031_official_bench/data/luz031_汇总_20260823T085706Z.md`（半自动汇总）。建议新增 `FINAL-METRICS-LuZ0.3.1.md` 主汇总 + 各窗口数据表（§6）。

---

## 1. 本地资产总清单（Windows）

### 1.1 规模总览

| 维度 | 数量 |
|---|---|
| 总文件数（递归） | 2,355 |
| 其中 .md 报告 | 477 |
| 顶层 .md 报告 | 311 |
| 顶层非 .md 文件 | 33 |
| 子目录数 | 51 |
| 总大小 | 172,887,049 B（~172.9 MB） |
| 报告时间跨度 | 2026-08-01 ~ 2026-08-24 |

### 1.2 顶层报告分类目录（311 份 .md，按主题粗分）

| 主题簇 | 代表性文件（数量） | 时间 |
|---|---|---|
| **最终生产落地（LuZ0.3.1）** | luz031-deployment / luz031-official-bench-bypass / luz031-bench-and-p0-exec / luz031-thr2048-retest / g1-production-restore / arstall-production-closure / phase-summary / b12x-gate-fix / key-rotate-r12-restart | 08-23~08-24 |
| **W4A4 / 翻案链** | wsdedup-l3-combo / w4a4-ext / w4a4-vs-w4a16-diff-audit / threshold-retest / threshold-4096-adoption / threshold-retest / a3-hybrid-slim-design / bprime-window / bprime-impl/ | 08-22~08-23 |
| **FP8 探索链** | fp8-f1-window / fp8-quality-gate-toolchain / fp8-quality-impact / lmhead-fp8-project / lmhead-fp8-f0-contract / opt-routeb-fp8 / upstream-hotfix-vs-ours | 08-23~08-24 |
| **算子/kernel/插件（routeA/routeB/双核）** | architecture-dual-kernel / architecture-nvfp4 / routea-* / routeb-* / nvfp4-*（20+） / ringonly-v5 / ws-dedup-patch | 08-19~08-23 |
| **issue22 处置** | issue22-400k-verification / issue22-kv-deopt / _issue22_ab_archive / _issue22_400k_verification | 08-24 |
| **环境 stall / AR 调查** | slowround-rootcause / expd-r123 / ringonly-w1 / envstall-rootcause / expverdict-verdict / arstall-production-closure | 08-22~08-23 |
| **早期基准/调优（08-01~08-14）** | benchmark-* / bench-* / ab-compare-* / analysis-* / research-* / tp4-r*/tp4-opt* / kvssd-* / nccl-* / audit-* / code-review-* | 08-01~08-14 |
| **运维/runbook/安全** | runbook-tp4-* / ops/ / scripts/ / hardening-* / api-key-* / key-rotate / gateway-* / litellm-* | 08-04~08-24 |

> 完整 311 份文件名清单见附录 A（可按需导出）；此处为粗分，精确归类见 §3 开源库映射。

### 1.3 顶层非 .md 文件（33 个，按大小）

| 大小 | 文件 | 类别 |
|---|---|---|
| 880 KB | nvfp4-v16-round-2026-08-20.zip | 算子交付包归档 |
| 234 KB | routeb-archive-2026-08-21.tar.gz | 算子交付包归档 |
| 140 KB | nvfp4-sass-round-2026-08-19.zip | 算子交付包归档 |
| 65 KB | nvfp4-landing-handoff-2026-08-20.zip | 算子交付包归档 |
| 53 KB | nvfp4-v15-round-2026-08-19.zip | 算子交付包归档 |
| 35 KB | raw_final_matrix.json | 早期基准原始数据（08-05，非最终口径） |
| 13 KB | nvfp4-final-round-2026-08-20.zip | 算子交付包归档 |
| ~8 KB | start_head_v026r.sh / start_head_groupB.sh / start_worker_groupB.sh / start_groupB_cluster.sh | 启动脚本 |
| ~8 KB | tp_comm_analysis.py / tp_comm_analysis_v2.py | TP 通信分析脚本 |
| ~7 KB | phase-summary-2026-08-23-results.csv | 阶段总结结果表 |
| ~5 KB | collect_mem*.sh ×4 | 内存采集脚本 |
| ~4 KB | raw_final_loads.json | 早期负载数据（08-05） |
| ~4 KB | bench_v12_real.py | 基准脚本 |
| ~3 KB | check_vllm_script.sh / test_accel.sh / pull_layers.sh | 巡检/测试脚本 |
| 384 B | _tessa_final_gsm8k_summary.json | 早期 GSM8K 汇总（08-05） |

### 1.4 资产子目录（51 个，按大小 Top 20）

| 大小 | 文件数 | 目录 | 内容 |
|---|---|---|---|
| 60.8 MB | 12 | ringonly-v5-2026-08-23/ | v5 环序补丁 + **libnccl.so.2.30.7 二进制(60.5MB)** + patch 链 |
| 29.9 MB | 20 | _rex/ | lm_head FP8 路由/取证脚本 + 数据 |
| 27.3 MB | 212 | _arstall_closure_20260823/ | AR stall 闭环 e3 取证数据 + 免提权脚本 |
| 17.3 MB | 7 | lmhead-fp8-f0-golden/ | lm_head FP8 golden 资产 |
| 9.5 MB | 14 | nvfp4-v16-round-2026-08-20/ | 算子 round 归档 |
| 4.0 MB | 29 | _audit/ | 四机审计原始数据（tsv） |
| 3.7 MB | 10 | routeb-merged-p1a-2026-08-21/ | merged-GEMM 交付 |
| 3.1 MB | 263 | _routea_work/ | routeA 分析中间产物 |
| 2.1 MB | 15 | _fp8_qg_toolchain/ | FP8 质量门工具链（脚本 + reference_set + runs） |
| 2.1 MB | 568 | _fix_20260813/ | 08-13 修复批次（含 issue22 早期 patch） |
| 1.3 MB | 8 | nvfp4-sass-round-2026-08-19/ | SASS round 归档 |
| 826 KB | 54 | routeb-archive-2026-08-21/ | routeB 归档 |
| 770 KB | 35 | evdata/ | 环境取证数据（affinity） |
| 711 KB | 23 | routea-plugin-p1-2026-08-21/ | routeA 插件 P1 |
| 695 KB | 14 | routea-w4a4-debug-2026-08-21/ | W4A4 调试 |
| 424 KB | 55 | hardened/ | 加固产物（adr/runbooks/deploy-profiles） |
| 397 KB | 11 | nvfp4-v15-round-2026-08-19/ | 算子 round 归档 |
| 366 KB | 67 | _pkg/ | 算子交付包 v10/v12 完整索引 |
| 228 KB | 67 | _luz031_official_bench/ | **LuZ0.3.1 官方口径基准脚本 + data/** |
| 183 KB | 37 | nvfp4-kernels-final-2026-08-20/ | 最终 kernel 交付 |

> 完整 51 目录明细见附录 B；`_luz031_official_bench/data/` 内含 `luz031_汇总_20260823T085706Z.md`（官方口径汇总）与 `luz031_vs_official_20260823T085706Z.md`、`luz031_m1_single/m2_conc/m3_agent.json` 等最终基准数据。

### 1.5 主理人任务书中目录名核对（重要差异）

| 任务书提到 | 实际盘点结果 |
|---|---|
| `_bench_pkg_official/` | **本地不存在**；官方基准包在服务器 `/tmp/_bench_luz031/official/benchmark_package_20260819/`（含 `data/测试数据汇总.md`） |
| `_fp8_f1_window/` | 本地不存在该目录；对应报告 `fp8-f1-window-2026-08-24.md`，数据在服务器 `/tmp/_fp8_f1/`（384 MB） |
| `_fp8_assets/` | 不存在该名；实际为 `_fp8_qg_toolchain/assets/`（golden-bf16-greedy-latest.json）+ 服务器 `/tmp/_fp8_qg_toolchain/` |
| `_luz031_official_bench/` | ✅ 存在（含 data/） |
| `_issue22_ab_archive_20260824/`、`_issue22_400k_verification_20260824/` | ✅ 存在（各 4-5 个 JSON） |

---

## 2. 服务器资产总清单（node01:<INSTALL_DIR>/，SSH 只读）

### 2.1 顶层总览（约 215 GB）

| 大小 | 目录/文件 | 内容 | 开源价值 |
|---|---|---|---|
| 214.7 GB | kvssd-images/ | KV SSD 镜像（root 持有） | ❌ 不入库（大二进制） |
| 54.6 GB | backup/ | prod-snapshot ×3 / vllm-tp4-image-b12x-recovered(12.3GB tar) / luz031 checkpoints / nccl 归档 | ❌ 不入库（快照） |
| 2.0 GB | build/ | 构建产物 | ⚠️ 视许可（部分源码可提取） |
| 109 MB | nvfp4/ | **算子/插件/kernel 源码（开源核心）** | ✅ 高价值 |
| 56.7 MB | envs/ | 运行环境（nvcc_wrapper 等） | ⚠️ 视许可 |
| 1.6 MB | nvfp4-landing-export/ | HANDOFF-TO-TEAM.md + 探针 | ✅ |
| 1.5 MB | verification-logs/ | 32 个子目录基准验证日志 | ⚠️ 可选 |
| 880 KB | deliverables/engineering-assurance/ | **服务器交付副本（含本地缺失的 nccl-* 报告）** | ✅ 补缺 |
| 791 KB | docs/ | 服务器运维手册 / runbook / release-notes / ops/（93 文件） | ✅ 高价值 |
| 639 KB | scripts/ | **启动脚本 + .bak 序列（100+ 文件）** | ✅ 高价值 |
| 595 KB | logs/ | 运行日志（含 nvidia-bug-report） | ❌ 敏感 |
| 424 KB | lib/ | **libncclpin.so v8 + .bak v3~v7（shim）** | ✅ 二进制+源码线索 |
| 321 KB | cache/ | JIT 缓存 | ❌ |
| 255 KB | nvfp4.bak-20260820-1605/ | nvfp4 备份 | ⚠️ |
| 133 KB | backups/ | 备份 | ❌ |
| 54 KB | results_benchopt_smoke/ | 基准结果 | ✅ 数据 |
| 46 KB | results_bt4096_c6verify/ | 基准结果 | ✅ 数据 |
| 39 KB | kvpatch/ | io.py 等 KV patch | ✅ |
| 35 KB | archi-test/ | 架构测试 Dockerfile | ⚠️ |
| 15 KB | overlay-wsdedup/ | **flashinfer_b12x_moe.py（wsdedup 补丁）** | ✅ 核心补丁 |
| 14 KB | overlay-mask/ | **api_utils.py（脱敏 overlay）** | ✅ 核心补丁 |
| 8 KB | results_kvssd_200g/ | KVSSD 基准 | ✅ 数据 |
| 6.6 KB | models/ | checkpoint symlink + 转化脚本 | ⚠️ 只引链接说明 |
| 156 B | secrets/ | **vllm.env + .bak（root-only，含 API key）** | ❌ 高度敏感，严禁入库 |
| — | configs/ | 空 | — |

### 2.2 scripts/（启动脚本家族，含 .bak 演进序列）

核心活动脚本：
- `start_tp4_cluster.sh`（10,026 B，v1.5-r12，08-24）+ **.bak 序列 4 个**（b12xgate-20260820 / b12xgate-fix-20260824 / keyconverge / r12-keyfix 等）
- `start_tp4_head.sh`（9,185 B）+ **.bak 序列约 40 个**（从 F3/F4/F6A-C（08-13）到 wsdedup/w4a4ext/cumem0/overlay/mask（08-24）——**完整记录每次生产变更演进**）
- `start_tp4_worker.sh`（.bak 同 head 族）、`start_sglang_tp4_head/worker.sh`、`start_tp4_head_b12x/combo/cutlass/deepgemm/marlin/nvfp4weight/overlap.sh` 等
- `check_vllm_script.sh`（+ .bak 序列 ~9 个）、`healthcheck-rebuild.sh`、`healthcheck.sh`、`quality_gate.py`、`preflight_sglang.sh`、`shim-deploy.sh`、`monitor_tp4_head.sh`、`mirror_to_02.sh`、`vllm-model-redirect.json`

> ⚠️ **敏感**：脚本含 API key 注入、内网 IP、镜像 registry、内部路径，开源前必须脱敏（尤其 `start_tp4_head.sh` L77 会打印含 `--api-key` 的完整 serve 命令——见 upstream-hotfix-vs-ours §0 安全项）。

### 2.3 docs/（93 文件，~791 KB）

- 运维手册：`服务器运维手册-20260817.md`（44 KB，权威）、`ops/server-maintenance-handbook.md`、`ops/self-recovery.md`、`ops/fault-tolerance.md`、`ops/maintenance-plans.md`、`ops/ops-discipline-quickref.md`、`ops/tools-index.md`、`ops/script-help-template.md`
- Runbook：`runbook-tp4-v1.5-2026-08-12.md`（+ .bak-g3needle-20260824）、`tp4-service-deployment-guide-2026-08-13.md`（+ 4 个 .bak）
- Release notes：**`LuZ0.3.1-release-notes.md`**（6.6 KB，开源核心文档）
- 交付/QA：`B1-*`（4 份）、`P1实施与自恢复演练*`（3 份）、`四机重启自恢复演练-QA报告`、`nccl-*-qa-2026-08-17`（3 份）、`file-registry.md`、`production-self-healing-plan-architect-2026-08-17.md`
- 隐藏备份 `.qa-comment-bak-20260817/`（源码 QA 备份，不入库）

### 2.4 nvfp4/（算子/插件/kernel 源码——开源核心）

| 路径 | 关键文件 | 说明 |
|---|---|---|
| plugin_a1/routea_plugin_a1/ | w4a4_experts.py(11 KB) + .bak-wsdedupl3-20260823 | **W4A4 生产插件**（VLLM_MOE_W4A4=2 门控） |
| plugin_a1_bprime/routea_plugin_a1_bprime/ | w4a4_experts.py(17 KB)、__init__.py(7 KB) | b′ native 共享插件（No-Go 储备） |
| plugin_merged/routeb_merged_plugin/ | merged_experts.py / triton_moe.py / dsl_gemm.py | routeB merged-GEMM 插件 |
| plugin-src/nvfp4_vllm_plugin/ | ab_routeA_vs_b12x.py / ab_v17_semantics.py | A/B 测试插件 |
| kernel1/ | nvfp4_4w4a_mmaf.py(4.3 KB) | kernel1：prefill GEMM |
| kernel2/ | nvfp4_ds_mla_kv_linear_{triton,v17_triton,torch}.py + test/bench | **kernel2：MLA KV linear（issue22 相关）** |
| routeb_official_v2/ | dense_blockscaled_gemm_persistent_pingpong.py(88 KB)、blockscaled_gemm_dispatch.py | **routeB 官方 v2 FP8 稠密 GEMM（fp8 系列核心）** |
| flashinfer-0.6.16/ | flashinfer 源码树 | FI 0.6.16 生产版本 |
| zip-reference-2026-08-20/ | kernel1/plugin/kernel2 引用包 | 参考归档 |
| routeB-delivery-2026-08-20/ | routeB 交付 | 交付归档 |

### 2.5 服务器 /tmp/ 工作数据（只读枚举，重要证据）

| 位置 | 大小 | 内容 |
|---|---|---|
| /tmp/_bench_luz031/ | 604 KB | **官方基准包 benchmark_package_20260819（含 data/测试数据汇总.md 8/19 参考口径）** + p0/data/logs |
| /tmp/_fp8_f1/ | 384 MB | F1 微基准（**head_bf16_32384.bin 265MB / head_fp8_payload 133MB / head_scale 4MB** + 脚本） |
| /tmp/_fp8_qg_toolchain/ | 8.4 MB | FP8 质量门工具链运行产物 |

> /tmp 下另有大量历史实验目录（`_thrst`、`_thr4096`、`_ws_dedup`、`_w4a4_ext`、`_bprime`、`_luz031`、`_fi016`、`_eugr_ab`、`_ringopt`、`_slowround`、`_expverdict`、`_envstall` 等，见 phase-summary §七）——均为阶段证据，开源时按需选取。

---

## 3. 开源仓库分类结构设计（LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring）

### 3.1 建议目录布局

```
LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring/
├── README.md                          # 项目简介/形态基线/快速导航（脱敏）
├── docs/
│   ├── 01-research-reports/           # 研究报告（设计/调研/根因/路线）
│   ├── 02-performance-benchmarks/     # 性能测试报告与原始数据
│   ├── 03-final-metrics/              # 最终性能指标（含新增汇总）
│   ├── 04-issues/                     # 缺陷问题报告
│   ├── 05-kernels-patches/            # 算子源码与补丁
│   ├── 06-verification/               # 验证资料（QA/恢复演练/回归）
│   ├── 07-deployment/                 # 容器与部署（runbook/脚本/镜像说明）
│   └── 08-tools/                      # 工具脚本（基准/采集/取证）
├── kernels/                           # 算子源码（plugin_a1/kernel1/kernel2/routeB 等）
├── patches/                           # 补丁（wsdedup/overlay/ringonly-v5/FI 0.6.16）
├── scripts/                           # 脱敏后的启动/部署/基准脚本
└── data/                              # 关键基准原始数据（json/csv/tsv）
```

### 3.2 各类纳入清单与脱敏标记

| 目录 | 应纳入文件（来源） | 脱敏要求 |
|---|---|---|
| **01 研究报告** | architecture-*（dual-kernel/nvfp4/nvidia-sync-twin）、research-*（deepseek-v4-flash-nvfp4 / tp2-tp4-communication / dgx-networking / comm-overlap / nccl 系列）、analysis-*（tp4-bottleneck / dcp-cost / litellm）、a3-hybrid-slim-design、b12x-tail-path-strategy、pr-de-bottleneck-analysis、upstream-check-perf-ceiling、upstream-hotfix-vs-ours、opt-routeb-fp8、freetoken-research、mtp-tuning、ar-optimization 等 | 🔴 需脱敏：IP/主机名/内部路径/用户名/registry |
| **02 性能基准** | benchmark-*/bench-*（约 40 份）、benchmark-tp2-*、bench-matrix-v026、luz031-official-bench-bypass、luz031-bench-and-p0-exec、_luz031_official_bench/data/*、e2e-baseline-*、e2e-baseline-results/、raw_final_*（标注旧口径）、results_benchopt_smoke、results_bt4096_c6verify | 🟡 数据本身安全；报告文字需脱敏（含 IP/key）；raw_final_* 为 08-05 旧口径须标注 |
| **03 最终指标** | **新增 FINAL-METRICS-LuZ0.3.1.md**（见 §6）+ luz031-deployment、arstall-production-closure（E5 四档复查）、luz031_汇总_20260823T085706Z.md、g1-production-restore（W4A16 对照）、release-notes | 🔴 汇总文件新建即按公开口径书写；来源报告脱敏 |
| **04 缺陷问题** | issue22-*（4 份 + _issue22_*_assets）、incident-*（8 份）、investigation-*、diagnostic-*、dual-kernel-problem-list、review-*、slowround/expd/envstall/expverdict（环境 stall 链）、fp8-quality-impact（质量问题） | 🔴 需脱敏；issue22 报告含 key 纪律说明（key 不回传） |
| **05 算子源码与补丁** | nvfp4/plugin_a1（w4a4_experts.py）、kernel1/、kernel2/、routeb_official_v2/、plugin_merged/、bprime-impl-2026-08-23/、ws-dedup-patch-2026-08-22/、overlay-wsdedup/、overlay-mask/、ringonly-v5-2026-08-23/（patch 与说明，**不含 libnccl.so 二进制**）、_fp8_qg_toolchain/、lmhead-fp8-f0-golden/ | 🔴 源码内可能含路径/注释敏感信息；二进制（libnccl.so 60.5MB、.zip 交付包）默认不入库或单独 tag 发布 |
| **06 验证资料** | B1-*（兼容性/环境质量）、hardening-acceptance-checklist、production-finalize-checklist、test-*/tp4-3test/v027-nvfp4-acceptance*、testing-strategy、nccl-ab-*、kvssd-*（验证计划）、_arstall_closure_20260823/（取证数据） | 🟡 数据安全；脚本内可能含内部路径需清理 |
| **07 容器与部署** | docs/（运维手册/runbook/release-notes）、ops/（handbook/self-recovery/fault-tolerance/maintenance-plans/tools-index）、scripts/（脱敏后）、file-registry、rollback-anchors、生产自愈计划 | 🔴 最高敏感：含密码/IP/路径/key 注入；须逐文件脱敏，secrets/vllm.env **严禁入库** |
| **08 工具脚本** | _luz031_official_bench/*.py、_fp8_qg_toolchain/*.py、evdata/、_rex/、_audit/、collect_mem*.sh、bench_v12_real.py、tp_comm_analysis*.py | 🟡 脚本内 IP/路径需清理；_audit/_rex 取证数据视需要 |

### 3.3 脱敏优先级与规则建议（供阶段 B 执行）

1. **P0（严禁入库）**：`<INSTALL_DIR>/secrets/*`（vllm.env）、任何含明文 API key 的日志/脚本（`start_tp4_head.sh` L77 打印 serve 命令）、`.sudo_pw` 痕迹、`logs/nvidia-bug-report*`。
2. **P1（全局替换）**：内网 IP `192.168.5.x` → `<NODE-IP>`；主机名 `dgxspark0X` → `node0X`；内部用户名 `<USER>`/`novAI` → `<USER>`；内部路径 `<INSTALL_DIR>`、`/home/*/models`、`<MODELS_DIR>` → `<INSTALL_DIR>`；镜像 registry `<NODE_IP>:5000` → `<REGISTRY>`；key 值（≥32 hex）→ `<REDACTED>`。
3. **P2（内容审读）**：所有报告正文引用的 `/tmp/_xxx` 实验路径、`docker logs` 摘录、NCCL 配置值（可能含 site 特征）需人工复核。
4. **P3（二进制处置）**：`libnccl.so.2.30.7`、`nvfp4-*-round-*.zip`、`routeb-archive-*.tar.gz` 不随文档仓库发布；如需发布二进制走 Releases 附件并附源码构建说明。

---

## 4. 敏感信息初步标记统计（本地 .md）

基于 311 份顶层报告的模式扫描（正则，仅统计"出现即算"，未逐文件精读）：

| 敏感类型 | 标记正则 | 命中文件数 | 占顶层报告 |
|---|---|---|---|
| 内部路径 | `<INSTALL_DIR>` / `/home/` / `<MODELS_DIR>` | 184 | 59% |
| 内网 IP | `192.168.x.x` | 146 | 47% |
| 镜像 registry | `IP:port` / `:5000` / docker.io / nvidia.com | 125 | 40% |
| 主机名 | `dgxspark0[1-4]` | 126 | 41% |
| API key 痕迹 | api_key / sk- / VLLM_API_KEY / Bearer | 85 | 27% |
| 内部用户名 | `<USER>` | 75 | 24% |
| 疑似 key 长串 | `[a-f0-9]{32,}` | 58 | 19% |
| 密码/secret | .sudo_pw / password / passwd / secret | 44 | 14% |

> **结论**：开源前**几乎所有报告（≥60%）需过一遍脱敏流水线**；纯公开安全（无需改动即可入库）的报告占比预计 <20%（主要为方法论/纯理论分析类，如 research-*-community、design 类无 IP 引用者）。精确到文件的脱敏清单留待阶段 B（敏感信息清理）输出。

---

## 5. 报告关联索引（交叉引用）

### 5.1 最高被引报告（Top 15，全库 grep 计数）

| 被引报告 | 被引次数 | 说明 |
|---|---|---|
| rollback-anchors-2026-08-12.md | 30 | 回滚锚点/降级权威 |
| tp4-service-deployment-guide-2026-08-13.md | 16 | TP4 部署指南权威 |
| upstream-check-perf-ceiling-2026-08-23.md | 10 | 性能上限/叙事权威 |
| nvfp4-testkit-diagnosis-2026-08-19.md | 10 | 位于子目录（被引用时路径需带子目录） |
| tp4-r12-final-report-2026-08-13.md | 9 | R12 终报 |
| litellm-api-key-manual-2026-08-05.md | 9 | API key 手册 |
| bench-regression-attribution-2026-08-23.md | 9 | 回归归因 |
| architecture-nvfp4-2026-08-20.md | 9 | NVFP4 架构 |
| slowround-rootcause-2026-08-22.md | 8 | 慢轮根因 |
| luz031-deployment-2026-08-23.md | 8 | **LuZ0.3.1 落地权威** |
| fi017-p0-accounting-2026-08-23.md | 8 | FI 0.6.17/P0 |
| w4a4-ext-2026-08-23.md | 7 | W4A4 并发 |
| a3-hybrid-slim-design-2026-08-23.md | 7 | b′ 设计 |
| runbook-dspark-vllm-2026-08-06.md | 7 | 早期 runbook |
| expd-r123 / wsdedup-l3-combo / e2e-baseline-prod / opt-routeb-fp8 / lmhead-fp8-project | 各 6 | — |

### 5.2 关键证据链（修订时需同步更新引用的链）

1. **W4A4 翻案链**：`routing-recapture-scheduler-ab`(-22.5%) → `p1-p2-research`(混杂嫌疑) → `threshold-retest`(翻案) → `threshold-4096-adoption`(采纳) → `wsdedup-l3-combo`(W4A4 翻正) → `w4a4-ext`(并发放大) → `luz031-deployment`(LuZ0.3.1 落地)。
2. **b′ 链**：`a3-hybrid-slim-design` → `bprime-impl/` → `bprime-window`(No-Go)。
3. **环境 stall 链**：`slowround-rootcause` → `expd-r123` → `ringonly-w1` → `envstall-rootcause` → `expverdict-verdict` → `arstall-production-closure`(E5/E3 闭环)。
4. **FP8 链**：`opt-routeb-fp8` → `fp8-quality-impact` → `lmhead-fp8-project` → `lmhead-fp8-f0-contract` → `fp8-f1-window` → `fp8-quality-gate-toolchain`。
5. **issue22 链**：`issue22-kv-deopt` → `issue22-400k-verification` + `key-rotate-r12-restart`(131K A/B)。
6. **FI 0.6.16 链**：`fi-rebase-eugr-prep` → `windowA-fi-cg-budget` → `fi016-replacement` → `luz031-deployment`(误回滚发现)。

> **修订提示**：被引文件若改名/移目录（尤其进入开源库 8 大目录后路径变化），所有引用它的报告须同步改路径。最高优先级 = 5.1 表 Top 10。

### 5.3 被引用但本地缺失的报告（缺口证据）

核对 222 个被引用唯一文件名 vs 本地实际文件，**18 份本地缺失**：

- **仅存于服务器交付副本（7 份，需回传本地）**：`nccl-final-performance-baseline-2026-08-17.md`、`nccl-2hop-s3-final-adjudication-architect-2026-08-17.md`、`nccl-stageb-verification-2026-08-16.md`、`nccl-maxch16-e2e-verification-2026-08-16.md`、`nccl-p0-scan-results-2026-08-16.md`、`nccl-proto-threshold-scan-2026-08-16.md`、`b1-compat-adjudication-criteria-architect-2026-08-17.md`
- **两处均缺失（11 份，疑似命名漂移或从未落盘）**：`deploy-f-dynamic-k-baseline-2026-08-05.md`、`hf-nvfp4-mirror-survey-2026-08-13.md`、`kernel-design-2026-08-21.md`、`kvssd-util07-result-2026-08-19.md`、`kvssd-vs-baseline-compare-2026-08-19.md`、`miaai-2026-08-13.md`、`nvfp4-investigation-2026-08-13.md`、`prde-bottleneck-analysis-2026-08-22.md`（疑为 `pr-de-bottleneck-analysis` 笔误）、`prob-eval-report-2026-08-05.md`、`v027-nvfp4-acceptance-result-2026-08-15.md`、`v027-vs-v026-perf-compare-2026-08-14.md`
- **子目录内存在（2 份，引用路径需修正）**：`nvfp4-delivery-run-report-2026-08-19.md`、`nvfp4-testkit-diagnosis-2026-08-19.md`

---

## 6. 缺口清单

### 6.1 最终性能指标资料缺口（确认）

**无单一最终指标汇总文件**。现有关键指标散落在：

| 数据点 | 所在位置 |
|---|---|
| LuZ0.3.1 采纳 PR 四档（2950.5/2943.6/2834.2/2550.0） | luz031-deployment-2026-08-23.md |
| E5 生产同构复查四档（2959.6/2984.1/2872.2/2642.9） | arstall-production-closure-2026-08-23.md |
| 官方口径 decode-only 单流/并发/Agent（luz031_汇总） | _luz031_official_bench/data/luz031_汇总_20260823T085706Z.md |
| W4A16 vs W4A4 同窗对照（G1） | g1-production-restore-2026-08-24.md |
| KV 容量/质量门/needle | luz031-deployment §验收 |
| FP8 F1 统计窗口 | fp8-f1-window-2026-08-24.md |
| 早期基准（08-05 口径） | raw_final_matrix.json / raw_final_loads.json（**旧口径，勿混用**） |

**建议新增汇总文件**（阶段 B/C 实现，本阶段仅设计）：

```
03-final-metrics/
├── FINAL-METRICS-LuZ0.3.1.md          # 主汇总：形态基线 + 一页指标表 + 引用索引
├── metrics-decode-only.csv            # 单流/并发/Agent decode-only（官方口径）
├── metrics-prefill-pr.csv             # PR 四档 + TTFT（含 E5 复查）
├── metrics-w4a16-vs-w4a4.csv          # G1 对照
└── metrics-kv-quality.csv             # KV 容量 + 质量门 + needle 明细
```

FINAL-METRICS 设计要点：① 形态基线一节（W4A4 full/池补丁/FI 0.6.16/thr4096/util0.82/MTP n7）② 一页指标表（PR/TTFT/decode-only/KV）③ 每行标注测量口径（decode-only vs 端到端、窗口日期、克隆 vs 生产）④ 引用各来源报告（链接而非复制）⑤ 明确"旧口径不混用"警示（raw_final_matrix 为 08-05）。

### 6.2 其他缺口

1. **服务器交付副本回传**：本地缺 7 份 nccl-*/B1 报告（见 §5.3），开源汇编前需从 `node01:<INSTALL_DIR>/deliverables/engineering-assurance/` 回传本地。
2. **官方基准包**：`/tmp/_bench_luz031/official/benchmark_package_20260819/`（含 `测试数据汇总.md`）未在本地，需决定是否纳入 02 目录（建议纳入 data/ 子目录）。
3. **key 日志纪律**：多份报告注明"服务器日志含明文 key 建议清理/重算"（arstall §5、luz031 系列），阶段 B 需对服务器 `/tmp/_bench_luz031/logs/clone_head.log` 等做处置（或重算 key）。
4. **目录名/引用漂移**：`_bench_pkg_official`、`_fp8_f1_window`、`_fp8_assets` 与任务书不一致（§1.5），开源映射时以本清单实际路径为准。

---

## 附录

- **附录 A**：311 份顶层报告完整清单（可由 `ls -1 *.md` 导出，本报告已含代表性归类）
- **附录 B**：51 个资产子目录大小/文件数明细（§1.4 已列 Top 20，完整可由 `du -sb */` 导出）
- **附录 C**：服务器 `<INSTALL_DIR>/` 完整枚举（§2 已列核心，完整可由 SSH find/du 导出）

*本盘点为纯只读产出，未触碰 GPU/集群，未修改任何文件；所有统计基于文件名/元数据/正则模式扫描，正文未做逐字精读，个别误报以阶段 B 人工复核为准。*
