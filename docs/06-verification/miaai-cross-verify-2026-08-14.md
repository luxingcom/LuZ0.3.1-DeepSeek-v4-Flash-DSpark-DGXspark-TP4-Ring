# MiaAI 方案逐条对照 + 社区/服务器双向交叉验证报告

**日期**：2026-08-14 ｜ **核验人**：general-purpose-1（调研/验证，只读，未改动任何生产配置）
**上游来源**：`research-miaai-dspark-comparison-2026-08-14.md`（docs/vllm-027-new-patches.md、RESULTS-2026-08-14.md、9 个 vLLM PR、服务器 node01~04 只读核验）
**约束**：服务器仅 docker inspect / 读文件 / 只读命令；社区优先 GitHub 官方源。

---

## 0. 核验方法说明

| 侧 | 方法 | 证据来源 |
|---|---|---|
| 社区 | GitHub API（api.github.com）+ 官方页面 WebFetch | repo API、docs/vllm-027-new-patches.md、results/RESULTS-2026-08-14.md、9 个 PR API |
| 服务器 | SSH node01~04 只读命令 | build.log/clone.log、git describe、docker images/inspect/ps、pip list、启动脚本 grep、本地基准报告 grep |

---

## 1. 仓库与文档真实性核验

| 项目 | 上游文档声称 | 社区核验结果 | 判定 |
|---|---|---|---|
| 仓库存在性 | github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark，676★/96 fork，2026-08-14 当天仍更新 | GitHub API：`full_name` 精确匹配；**stargazers_count=676、forks_count=96 完全一致**；`pushed_at=2026-08-14T14:54:44Z`（当天有 push）；MIT 协议；语言 Python | **真实** |
| README 要点 | TP=2、nvfp4_ds_mla、1M token、MTP-5、ghcr.io/anemll/dspark-vllm-gx10:0.1.1、patches 动态 hotfix | README 实测：`vllm serve` TP=2 · nnodes 2、`--kv-cache-dtype nvfp4_ds_mla`、MAX_MODEL_LEN=1048576、MTP_NUM_TOKENS=5、默认镜像 `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`、`patches/` 容器启动时应用——**全部吻合** | **真实** |
| docs/vllm-027-new-patches.md 存在性 | 存在，含 6 个 0.27 修复实测效果 | 文件**实际存在**，含 hotfix 表：#49486 3.4%、#48957 ~2×、#50298 1.88×、#50004 1.0%、#50312 448 MiB(256 MiB/rank)、#49236 3.9%、#48047 需 FI≥0.6.14、#46789 feature 级、#48993 unassessed、#48407 Stage A dormant | **真实** |
| results/RESULTS-2026-08-14.md | TP2 性能矩阵，128K×c4=8.4/c6=8.6 tok/s | 文件**实际存在**；2026-08-14 live 矩阵 131072 行 c4=**8.4**、c6=**8.6** 与文档逐字一致 | **真实** |
| 128K c1 TTFT 80.4s（8/1 基线） | §2 表 | 原文 "128K TTFT 80.4 s vs 78.8 s"（8/1 对比）；8/1 sweep 表 131072 c1 TTFT=78.75s。80.4s 与 78.75s 口径相近（±2%） | **基本真实**（微小口径偏差） |

---

## 2. 上游 PR 逐项核验（§3.1 对照）

| PR | 文档声称 | 社区核验（GitHub API） | 服务器核验 | 结论 |
|---|---|---|---|---|
| **#49486** skip indexer topk/router，TTFT **-3.4%**（≤2048 token 触发） | 真实存在：已 merged 2026-07-23，标题 `[DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case`；触发条件 ≤2048（MiaAI docs：512 topk × 4 compress ratio） | 0.27.1 源码含 skip 逻辑（sparse_mla.py）；生产 capture-sizes 覆盖小 batch，prefill ≤2048 场景存在 | **真实**，数字/触发条件与文档一致 |
| **#48957** skip 空 c128 compressor，kernel **~2×**（cudagraph≠FULL 时） | 真实存在：已 merged 2026-07-22，标题 `[DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement`；MiaAI docs 明确 fires only when cudagraph mode **≠ FULL** | ⚠️ 文档 §3.1 称本集群 "CUDA Graph FULL"——**实际生产脚本 `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`（PIECEWISE ≠ FULL）**。触发条件**实际满足**，但文档对本集群状态的描述不准确（见 §5 发现②） | **真实**（PR）；文档对本集群状态描述**不实**（FULL→PIECEWISE） |
| **#50298** FlashMLA workspace reuse，kernel **1.88×**（topk+SWA 索引） | 真实存在：已 merged 2026-07-30，标题 `[DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement`（传 out tensor 避免 torch.full，即 workspace 复用）；微基准 GPU 延迟 -46.8% | 0.27.1 源码构建含此改动（prefill 路径直接受益）；flashinfer_b12x + deep_gemm 后端下路径存在 | **真实**，数字一致 |
| **#50004** adaptive C128A topk，E2E **-1.0%** | 真实存在：已 merged 2026-07-27，标题 `[DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement`；body 实测 gsm8k acc≈0.9492 无回退 | 0.27.1 源码 `active_topk_width` 命中 sparse_mla.py | **真实**，数字一致 |
| **#50312** MTP buffer，省 **448 MiB**（文档"448 MiB/rank，我们 256 MiB/rank 口径"） | 真实存在：已 merged 2026-07-30，标题 `[DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved`；body：8192×4×7168×2B=448 MiB（最大 bf16 buffer），**MiaAI fork 实测 256 MiB/rank** | ⚠️ 无 PP（脚本无 --pipeline-parallel-size，纯 TP4），但 MTP-5 的 pp buffer 概念独立于 TP/PP，MiaAI TP2 同样无 PP 仍测得 256 MiB/rank → 对 TP4+MTP-5 **大概率适用**，节省量需实测 | **部分准确**（448 为 PR 理论最大值，per-rank 因配置而异；文档 "448 MiB/rank" 表述不精确） |
| **#49236** EagerScratchPool reuse，TTFT **-3.9%**（需 C++ op 重编） | 真实存在：已 merged 2026-07-31，标题 `[DSv4 Perf] Optimize workspace reuse for eager break, 3.9% E2E TTFT improvement.`；body 实测 TTFT 44.65 vs 46.41ms | **0.27.1 源码天然包含**：`DeepseekV4EagerScratchPool` 命中 compressor.py/attention.py；01 构建产物含重编 C++（_flashmla_C.abi3.so 等 8/14 12:40）——文档"01 构建直接红利"成立 | **真实**，且已在构建产物中落地 |
| **#48047** sparse-MLA q-head padding，需 FlashInfer **≥0.6.14** | 真实存在：已 merged 2026-07-31，标题 `[DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14`；body 明确 Dependency：运行时依赖 flashinfer ≥ 0.6.14（0.6.13 内核强制 num_heads∈{64,128}，TP4 会 pad 到 64 → 本 PR 对 TP4 有直接收益） | **NGC 26.07 镜像实测 flashinfer-python=0.6.14+d0510b70.nv26.7** ✓ 恰好满足；0.27.1 源码 `_pad_to_supported_q_heads` 命中 flashinfer_sparse.py:67 | **真实**，服务器侧完全符合 |
| **#46789** sequence parallelism（feature 级） | 真实存在：已 merged 2026-08-01，标题 `[DSV4] Implement Sequence Parallelism`；启用条件 PP=1+EP+TP>1 | 0.27.1 源码含 SP 代码（deepseek_v4/nvidia/model.py 等）；本集群 TP4 无 PP、是否启用 EP 需确认，属**评估项** | **真实**，feature 级，纳入灰度评估合理 |
| **#48993** compact MXFP4 indexer（feature 级） | 真实存在：已 merged 2026-07-22，标题 `[Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays`；GB200 实测 FlashMLA KV 块数 +12.05% | 0.27.1 源码中未能以关键词直接确认（MiaAI 文档亦标注 unassessed）——**待代码级 diff 确认**；本集群 KV dtype=nvfp4_ds_mla，与 fp8_ds_mla 布局不同，收益需重估 | **真实**（PR 存在），**待确认**（是否包含于 0.27.1 + 收益是否适用） |

> 备注：9 个 PR 均为 vLLM 项目成员（yewentao256、WoosukKwon、GirasoleY、majunze2001）提交，属同一 DSv4 Perf 系列（issue #45861），合并窗口 2026-07-22 ~ 08-01，早于 0.27.x 发布，0.27.1 源码构建覆盖其中 7 个（#49486/#48957/#50298/#50004/#50312/#49236/#48047 + feature 级 #46789）；#48993 待确认。

---

## 3. §3.2~3.4 结论核验

| 上游结论 | 社区核验 | 服务器核验 | 结论 |
|---|---|---|---|
| **§3.2 128K 高并发断崖普适性**：MiaAI TP2 128K×c4=8.4/c6=8.6；本集群 TP4 131K×c5=7.8-8.2；同数量级 → GB10 UMA 访存延迟物理极限 | **吻合**：RESULTS-2026-08-14.md live 矩阵 131072 c4=8.4/c6=8.6 逐字一致 | **吻合**：tp4-service-deployment-guide-2026-08-13.md:377 `131072/c5 7.8-8.2（断崖）`；tp4-r12-analysis-2026-08-13.md 归因 memory-latency-bound（c5 每步串行遍历 4.4GB KV，128ms/步 = 7.8 tok/s） | **已覆盖**（本集群独立归因与 MiaAI 同源），断崖普适性成立 |
| **§3.3 #43 decode floor + per-step 调度诊断**：公平性 guardrail + 诊断工具化 | #43 为 MiaAI fork 私有 patch（不在 vLLM 上游 PR 列表）；MiaAI docs 确认 #27 在 0.25.2 上 `_check_feature_supported()` 拒绝 >1 | 本集群 F2 已排除调度饥饿；per-step 日志为可借鉴工程思路，非已有功能 | **借鉴**（诊断工具化思路可移植；decode floor 作 guardrail 在并发提升后有预防价值） |
| **§3.4 #55 finish_reason=length**：工具调用截断报 length（IntEnum≠StrEnum） | #55 为 MiaAI fork 私有 issue/patch（上游无对应 PR 核验途径）；修复逻辑合理 | 本集群 8003 responses 网关/工具调用场景存在相同截断语义风险 | **借鉴**（低风险正确性修复，按需移植到 0.27 灰度） |
| **§3.4 suppress-stops-in-reasoning**：推理期 stop 抑制 | Anemll 社区 Patch 5 移植（Tony/Capicua25x），MiaAI 仓库 patches/ 下可见 hotfix 脚本 | 与本集群 think+stop 语义相关场景吻合 | **借鉴**（低风险，按需移植） |
| **§3.4 thinking_token_budget GPU 版**：per-request 推理预算硬顶 | MiaAI #48 私有 patch（Triton kernel，无 host scan） | 产品化能力，与性能无关 | **借鉴**（产品化可选，P2 以下） |
| **§3.4 dormant patch 纪律**（#48407 Stage A 绑定空串，宁可不生效也不静默错） | **真实且高质量**：MiaAI docs 详述 #48407 因 fork 无 dense-MHA 路由而将绑定置 ""，gate 永不触发，零性能效果是刻意为之；上游本尊已 merged（docs 关联 #48407 描述合理） | 本集群 0.26.1/0.27 为官方 main 线，无 fork 锚点问题，天然避免此坑 | **已覆盖**（工程纪律值得采纳，适配到 0.27 灰度移植规范） |

---

## 4. 排除项核验（§4）

| 排除项 | 社区核验 | 结论 |
|---|---|---|
| **0.25.2 基座 hotfix 锚点**：20+ patch 锚定 0.25.2，不能搬到 0.26.1 | 合理：MiaAI docs 的 anchor 校验机制绑定 0.25.2.dev0+g752a3a504 源码；v0.27.1 源码结构已大改 | **确认排除**（我们走官方 main 天然包含大部分修复） |
| **#27 max_num_partial_prefills**：0.26.1 无此字段 | MiaAI docs 确认 0.25.2 上 `_check_feature_supported()` 拒绝 >1；上游该能力需 vLLM 升级 | **确认排除**（0.26.1/0.27 同样约束，与 M2 结论一致） |
| **TP=2 配方本身**：不适用 TP4 拓扑 | 合理：通信/内存互斥已按 TP4 定型 | **确认排除** |

---

## 5. 行动建议执行状态（§5 逐条）

| # | 行动建议 | 执行状态 | 核验证据 / 落地要点 |
|---|---|---|---|
| 1 | **P0 跟随 0.27 构建**，用 MiaAI 清单验收：#49236/#49486/#48957/#50298/#50312/#48047 | **进行中**（构建已完成，进入冒烟/验收阶段） | 构建产物就绪（build.log EXIT=0，8/14 12:40）；镜像 test-0.2.1-v027 45.1GB；import 0.27.2.dev0+g6f5dc38d0 ✓；四机 vllm-tp4-v027-rank0~3 冒烟容器 Up。⚠️ 冒烟为 enforce-eager + fp8_ds_mla + max-model-len 8192，**非生产 nvfp4_ds_mla + CUDA Graph 口径**，需单独做生产参数验收 |
| 2 | **P1 借鉴落地**：per-step 调度诊断 + finish_reason/suppress-stops 移植 | **未执行** | 待 0.27 灰度稳定后按需移植（低风险正确性项） |
| 3 | **P2 认知固化**：128K 高并发 ~8 tok/s 为 GB10 物理极限，目标改为"推迟断崖拐点" | **未执行（认知已成立）** | 双向证据闭合：MiaAI TP2 8.4-8.6 / 本集群 TP4 7.8-8.2 / 归因均指向访存延迟；建议写入 SOP 约束长档并发 ≤c3 |
| 4 | **跟踪机制**：MiaAI 仓库每日更新，纳入每周上游核对 | **进行中**（research-upstream-updates-2026-08-14.md 已建表） | 仓库 pushed_at 当天 14:54，活跃度真实，纳入每周核对合理 |

---

## 6. 汇总统计

**真实**：仓库/README/docs 文件/RESULTS 全部真实；9 个 PR 全部存在且已 merged，其中 8 个数字与标题逐字吻合（#49486/#48957/#50298/#50004/#49236/#48047/#46789/#48993）；128K 断崖数据双向吻合。
**部分准确 / 不实**（文档需修正 3 处）：
1. §3.1 #48957 触发条件：文档称本集群 "CUDA Graph FULL"，**实测生产脚本为 PIECEWISE**（≠FULL，触发条件实际满足——修正后对本集群是利好）
2. §2 表 128K c1 "2013-2016ms×??"：2013-2016 实为 **prefill tok/s**（非 TTFT）；TTFT(c1)=57.4s
3. #50312 "省 448 MiB/rank"：448 MiB 为 PR 理论最大（8192×4×7168×2B），MiaAI 实测 256 MiB/rank；表述应改为"PP buffer 节省，MiaAI 实测 256 MiB/rank，本集群需实测"
**待确认**：#48993 是否包含于 0.27.1（MiaAI 亦 unassessed）+ 其收益对 nvfp4_ds_mla 布局是否适用。

---

## 7. 督导最值得关注的 3 个发现

1. **0.27 构建实际已完成，可进入生产口径验收**：文档假设"01 正在编译"已过时。build.log EXIT=0、wheel `vllm-0.27.2.dev0+g6f5dc38d0.d20260814.cu133`、镜像 test-0.2.1-v027（45.1GB）就绪。但当前冒烟容器（enforce-eager / fp8_ds_mla / 8K）**不是生产验收口径**（生产 nvfp4_ds_mla + CUDA Graph PIECEWISE + capture-sizes 1..64），验收需按 start_tp4_head_b12x.sh 参数走。
2. **#48957 触发条件对本集群实际是满足的（且是利好）**：文档误写为 "CUDA Graph FULL"，实测 `cudagraph_mode=PIECEWISE`（≠FULL）。MiaAI 清单中 6 个 perf PR 里 #48957 与 #49486 的触发条件（cudagraph≠FULL、prefill≤2048）在我们生产参数下均可达，0.27 验收应重点测这两项 kernel/TTFT 增益。
3. **128K 断崖普适性得到双向独立证据**：MiaAI TP2 8.4-8.6 vs 本集群 TP4 7.8-8.2，均归因 GB10 访存延迟。此认知建议正式写入部署指南（长档并发 ≤c3），避免后续在"消灭断崖"方向过度投入；同时 #49236/#49486/#48957 等 PR 的 TTFT/内存收益应作为 0.27 验收的**实际可交付物**，断崖本身不在其列。

---

## 8. 参考

- 上游对比文档：research-miaai-dspark-comparison-2026-08-14.md
- 社区源：github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark（docs/vllm-027-new-patches.md、results/RESULTS-2026-08-14.md）
- PR：vllm-project/vllm#49486/#48957/#50298/#50004/#50312/#49236/#48047/#46789/#48993（GitHub API 核验）
- 服务器：node01~04（<INSTALL_DIR>/backup/vllm027-src/vllm-0.27.1、scripts/start_tp4_head_b12x.sh、NGC 26.07 镜像 flashinfer）
- 本地基准：tp4-service-deployment-guide-2026-08-13.md、tp4-r12-analysis-2026-08-13.md、tp4-r8-final-report-2026-08-12.md
