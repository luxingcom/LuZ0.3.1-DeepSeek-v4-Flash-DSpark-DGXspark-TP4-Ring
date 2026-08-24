# b12x 与 vLLM 上游优化跟踪报告

- 日期：2026-08-22
- 作者：Rex（SRE 工程师，upstream-tracker）
- 范围：b12x 内核库上游 + FlashInfer b12x 树 + vLLM 上游（相对我们 fork 基线 0.26.1-dev @ d3d3b2cca，2026-08-05 快照）
- 口径标注：**[实证]** = 容器内代码/上游仓库/PR 页面直接验证；**[推断]** = 基于证据的合理推断，需落地前二次确认

---

## 0. 执行摘要

1. **b12x 上游身份已确认**：`github.com/local-inference-lab/b12x`（原 `lukealonso/b12x`，曾更名 sparkinfer 后改回），作者 Luke Alonso，PyPI 包名 `b12x`。我们镜像内是 **v0.15.3 @ commit 7dc6fb8（2026-05-30，"Slightly improve fp8 dense gemm perf"）**，上游最新 **v1.2.6（2026-08-20）**——**落后近 3 个月、跨越 3 个主版本号**。
2. **我们的 MoE 栈实际是三层**：vLLM fork → FlashInfer 0.6.15 内的 b12x 树（`flashinfer/fused_moe/cute_dsl/b12x_moe.py`）+ 镜像内独立安装的 b12x 0.15.3 包（被 `b12x_mxfp4_moe.py` 引用做权重准备/route-packing）。**两条上游线（FlashInfer b12x 树、社区 b12x 包）都显著领先我们**。
3. **FlashInfer 0.6.16 已发布**，b12x 树进入统一 MoE API（NVFP4/W4A16 parity + in-kernel routing + CUDA graph/autotune 覆盖）、CuTe-DSL JIT 磁盘缓存（冷启动重载 3–30ms）。
4. **最对症的三件事**：① 跨层 workspace/scratch 去重（社区 b12x "plan/scratch deduplicated by geometry across layers" + lablup/Backend.ai 的共享池方案，直接对症我们 P2 实测的 +28GB×43 层工作区问题）；② FlashInfer 0.6.15→0.6.16 b12x 树升级；③ vLLM 主线 B12X 全家桶集成（WO projection/MHC/FP8 GEMM/sparse indexer/MLA-sparse，fork 目前只有 `VLLM_USE_B12X_MOE` 一个开关）。
5. **三个历史待办 PR 全部已合入 vLLM 上游**：#48957（2026-07-22 合入，**已在 fork 内**[实证]）、#49486（2026-07-23 合入，fork 内**未找到签名**，需 diff 确认[推断]）、"B12x Direct M=1" = b12x micro-kernel/direct-scheduler M=1 decode 路径（社区侧已演进出 K-split FC1：TP12 M=1 94.3µs→39.2µs[实证，w4a8 trellis 路径]；对我们 W4A16 MXFP4 路径的收益需实测[推断]）。
6. **vLLM 上游已到 0.27.1（2026-08-11）**，DSv4 专项优化一整批（workspace reuse 3.9% TTFT、去冗余 kernel 1.88x、adaptive topk 1.0% E2E、PP buffer 省 448MiB、MXFP4 indexer KV cache 等）。
7. **NGC 26.07 vLLM 容器存在但不适合作为版本升级来源**（内含 vLLM 0.24.0 基线，反而落后我们 fork）；真正可行路线是 **Eugr spark-vllm-docker / timothystewart6/vllm-gb10 式的 sm_121a 源码编译流水线**，社区已有 DeepSeek-V4-Flash+DSpark+B12X 全家桶在 SM121 双机上的官方配方帖可参照。

---

## 1. b12x 上游身份确认（证据链）[实证]

在一次性容器（不占 GPU、不动生产）内取证：

| 项 | 值 | 证据 |
|---|---|---|
| 包名/版本 | `b12x 0.15.3` | `pip list`；`b12x-0.15.3.dist-info/METADATA` |
| 定位 | "Unapologetically SM120-only CuTe DSL kernels for NVFP4 GEMM and MoE" | METADATA Summary |
| 安装来源 | `https://github.com/lukealonso/b12x/archive/7dc6fb8f....tar.gz`（经 ghfast.top 代理） | `direct_url.json` |
| 快照日期 | commit `7dc6fb8` = **2026-05-30**，"Slightly improve fp8 dense gemm perf" | GitHub commit 页 |
| 上游现仓库 | `github.com/local-inference-lab/b12x`（lukealonso/b12x 重定向至此） | GitHub |
| 上游最新版 | **1.2.6（2026-08-20）**，"fix: support CUTLASS DSL 4.6.2" | 仓库 commit 36bce2c |
| 依赖现状 | 我们镜像 torch 2.11.0+cu130 / cutlass-dsl ≥4.5；**上游现要求 torch ≥2.12、cutlass-dsl ==4.6.0（1.2.6 兼容 4.6.2）** | METADATA vs 上游 README/pyproject |
| 命名史 | b12x → sparkinfer（2026 年中）→ 2026-08-06 "refactor!: rename sparkinfer back to b12x" | 上游 commit |

镜像内 b12x 包模块结构（0.15.3）：`moe/fused`（含 `w4a16/host.py` 的 route-packing、`route_pack.py`）、`cute/`（fp4.py、scratch.py、compiler.py）、`gemm/`、`integration/`（tp_moe.py、triton_route.py 等）、`quant/`。vLLM fork 的 `b12x_mxfp4_moe.py` 从中导入 `plan_tp_moe_scratch`、`prepare_b12x_fp4_moe_weights`、`select_route_block_size_m`、`pack_topk_routes_by_expert` 等 [实证]。

---

## 2. 社区 b12x：0.15.3 → 1.2.6 差距与可采纳清单

> 时间线：我们快照 2026-05-30；上游 6–7 月为 sparkinfer 时代（W4A16 fused MoE、route packing 成熟期、FP6 栈、SM121 paged attention 调优、indexer workspace 约束），8 月为 trellis/W4A8、TP16 all-reduce、BTX checkpoint、CUTLASS DSL 4.6.2 时代。以下按对我们集群的价值排序。

### A. 直接对症：workspace / 显存优化（★★★★★）

| # | 上游优化 | 内容与证据 | 收益预期 | 采纳方式 | 风险 |
|---|---|---|---|---|---|
| A1 | **跨层 scratch 几何去重** | FP6 提交 77d1cfe（07-27）注明 "MoE plan/scratch **deduplicated by geometry across layers**" [实证] | 我们 P2 实测 +28GB 工作区 ×43 层（每层 ~0.65GB）；按几何去重后理论上限是"所有同形 MoE 层共享一份 scratch"。Backend.ai 在 DGX Spark 上实证同病：48 层各 ~0.5GB → 共享池后问题消除 [实证] | 优先抄两份现成方案之一：① 社区 b12x 1.2.x 的 geometry 去重逻辑（`b12x/_lib/` scratch spine）；② lablup flashinfer fork 的 module-level 共享池（commit 1c623dea）。也可先在我们 fork 内做"同形层共享 workspace"补丁，不动 b12x 版本 | 中：CUDA graph 要求固定地址，共享池必须覆盖 capture 场景（上游 #150 预分配 route histograms 正是为此） |
| A2 | **W4A16 route histograms 预分配**（#150，commit 85d3681） | 为 CUDA graph capture 在 scratch plan 中预留 `int32[route_E]`，每 plan +4×E 字节 [实证] | graph capture 稳定性 + 去掉运行时分配 | 随 A1 一并采纳；或单独 cherry-pick 该提交到我们的 b12x 0.15.3 源码 | 低 |
| A3 | **prefill 尾部复用固定 route arena**（#226，commit 56ab5f4） | "reuse fixed route arenas for prefill tails"，避免每个 tail 长度触发新 Triton 特化编译（含 1536 容量/1177 尾部回归测试）[实证] | 大 M prefill 的尾块不再抖动/JIT 停顿；对 chunked prefill（tail 常见）收益直接 | cherry-pick 提交；注意与 fork 内 `select_route_block_size_m` 的交互 | 低-中 |
| A4 | **indexer two-level fold workspace 约束**（commit 896d5b8，07-29） | fold 候选 slab 超出可配置运行时预算时回退精确流式 carry [实证] | 若我们的 sparse indexer 用到 fold 路径，可封顶其 workspace | 需先确认我们是否走该路径 [推断] | 低 |

### B. 大 M prefill 性能（★★★★☆）

| # | 上游优化 | 内容与证据 | 收益预期 | 采纳方式 | 风险 |
|---|---|---|---|---|---|
| B1 | **W4A16 fused MoE 全部 6–8 月修复族** | #228 tiny decode 忽略 inactive routes、#230 preserve mapped W4A16 route namespaces、#219 route-packed top-k scaling 修复、#214 动态 MoE 忽略 inactive routes、e94d9e7 [实证] | 我们走的正是 W4A16 MXFP4 路径，这批修复直接作用于该路径的正确性与边界性能（尤其小 batch/graph 模式） | 以 b12x 包整体升级到 1.2.6 为主（cherry-pick 单提交到 0.15.3 工作量大且 API 已漂移） | 中：1.2.6 要求 torch≥2.12 + cutlass-dsl 4.6.x，与我们 torch 2.11 冲突（见 §5 路径） |
| B2 | **wave-balanced W4A16 FC2 tile**（#146，commit e68f812） | 验证 `tile_k=32, tile_n=512, cta_threads=256` 的 FC2 tile 配置 [实证] | 大 M FC2 波次平衡；对我们 43 层 MoE 的 prefill 有增益预期 [推断] | fork 已有 `VLLM_B12X_W4A16_FORCE_TILE_CONFIG` 开关 [实证]——可先在我们现有版本上 A/B 该 tile 配置，零升级成本 | 低 |
| B3 | **TMA large-M 激活量化器**（FP6 提交内） | "TMA-based large-M (prefill) and small-M (m≤16) decode/MTP quantizer"，逐行 scale 全融合、零 host 侧 launch [实证] | BF16→FP4 量化在大 M prefill 的带宽瓶颈缓解；MTP n=7 场景的 m≤16 量化路径收益直接 [推断] | 随 B1 整体升级；不适合单拆 | 中 |
| B4 | **trellis W4A8 prefill 5.9x**（commit 524353a/62be481，08-11） | 1024 tokens、TP12、E=896：28.3ms → 4.79ms [实证，w4a8 trellis 新路径] | 这是新量化栈（trellis/BTX）的数字，**不是**我们 W4A16 路径的等比收益；但显示上游 prefill 侧仍有数倍级空间 | 远期选项：若走 W4A8 需重做权重格式（BTX/trellis_t256），工程量大 | 高：新 checkpoint 格式 + SM121 未验证 |

### C. Direct M=1 / micro-kernel decode 路径（★★★★☆）

- 上游 MoE 执行模型：**Direct Scheduler**（micro kernel，M=1 内联路由）→ persistent grid（动态路径）→ W4A16 路径 [实证，deepwiki MoE 文档]。
- 8/12 提交 11fec54：micro path K-split FC1 成为**快速 M=1 W4A8 decode 路径：TP12 形状 94.3µs → 39.2µs**（W4A16 锚点 49.2µs）[实证]；7f6677d：T12 表驻留 smem，M=1 TP12 decode 770µs→571µs [实证]。注意：这些数字是 **w4a8 trellis** 路径，不是我们的 W4A16。
- FlashInfer b12x 树的 micro 路径（Triton compact pre-pass）在 0.6.15 已存在 [实证，Backend.ai 博客提及四后端含 micro path]。
- **对我们的含义**：MTP n=7 时 decode 批 = 1×(1+7) 或小 M，正是 micro/direct 路径射程。历史待办"B12x Direct M=1"对应的优化**在上游已大幅推进**；建议在我们环境实测 FlashInfer 0.6.16 的 micro 路径在 M=1/8/16 的表现 [推断→待实测]。

### D. SM121/GB10 专项（★★★☆☆）

- 2026-07-30 三连提交：`9b852b2` Tune SM121 paged attention capacity bands、`6a2babc` Optimize SM121 FP8 paged attention decode、`b38a60e` Optimize SM121 paged attention decode [实证]。
- FP6 栈注明 "SM12.1 expected compatible but **unvalidated**" [实证]——上游主力验证硬件是 RTX PRO 6000（SM120），SM121 属"预期兼容"，**每项采纳都需我们在 GB10 上自验**。
- MLKV/attention 侧（strided sparse MLA records 195e26c、windowed strided dense MLA ecc4344，08-14）与我们 DSV4 sparse MLA 相关 [实证]。

### E. 其他值得关注（★★☆☆☆）

- **PCIe DMA 线上压缩 all-reduce**（FP8/INT8/MXFP8 共 10 模式，线缆字节 -48.4%）：为**单机多卡 PCIe 系统**设计；我们 TP4 跨 4 节点走 200G 以太，IPC-based PCIe 集合通信**不适用** [实证-不适用]。
- TP16 size-routed all-reduce（08-16）：面向 RTX 6000 Pro 4 卡×4 机，同上不适用。
- MX-FP6 W6A8 栈（07-27）：精度-速度新档位，远期可评估。
- `freeze_kernel_resolution("serving")` 编译冻结 + `B12X_PRINT_COMPILE_PROGRESS`：**运维即用**，防止线上请求被 JIT 阻塞/graph capture 中编译——建议无论是否升级都纳入我们的启动流程 [实证]。

---

## 3. FlashInfer b12x 树：0.6.15 → 0.6.16 [实证]

我们镜像：flashinfer-python **0.6.15**（+cubin 0.6.14 + jit-cache 0.6.15）。上游已发 **0.6.16（含 .post1 tvm-ffi ABI 热修）**。与我们相关：

1. **统一 MoE API parity（#4091 #4026 #3983 #3892）**：SM120/121 B12x NVFP4 与 W4A16 后端进入统一 `MoELayer`/`B12xMoEWrapper` API；in-kernel routing（FromLogits）与预计算路由共用一条路径，带 CUDA-graph 与 autotune 覆盖。
2. **B12xMoEWrapper 新参数**：`source_format` 支持 `compressed_tensors`（原来仅 modelopt）；`num_local_experts`（EP）在 API 文档中出现——但 07-24 时点上游 b12x 树仍无 EP（Backend.ai 实证 + flashinfer commit 817e4bd1），**0.6.16 是否已完整支持 EP 需以 release notes/代码为准**（docs 页可能领先于已发布 tag）[推断]。我们 fork 镜像内的 0.6.15 `b12x_moe.py` 已有 `num_local_experts`/`source_format` 形参 [实证]，需确认是 vanilla 0.6.15 还是 fork 打过补丁（镜像为 "recovered" 版，dist-info 无 direct_url，**建议核查 flashinfer 是否被 MiaAI fork 补丁过**[待办]）。
3. **CuTe-DSL JIT 磁盘缓存**：JitSpecCuteDsl 后端，冷启动重载 3–30ms 替代每进程重编译——对我们 4 节点滚动重启/崩溃恢复的冷启动时间有直接收益。
4. MSA（MiniMax-M3 sparse attention）与 XQA decode 扩展落到 SM120/121——与本集群 DSV4 无直接关系，但证明 b12x 树是 FlashInfer 在 SM12x 上的活跃投送通道。
5. MegaMoE EP（moe_ep，symmetric-memory NVSHMEM）为 SM100+，**不适用 GB10** [实证]。

**采纳方式**：FlashInfer 是 pip wheel（aarch64 cu130 wheel 官方有发），0.6.15→0.6.16 升级成本低；先在测试容器跑 `tests/kernels/moe/test_flashinfer_b12x_moe.py` 等回归。**风险**：我们 fork 的 `flashinfer_b12x_moe.py`/`utils/flashinfer.py` 与 0.6.16 API 的兼容 diff 需过一遍。

---

## 4. vLLM 上游：0.26.1-dev（8/5 快照）→ 0.27.1 / main

### 4.1 三个历史待办 PR 的最终状态 [实证]

| PR | 状态 | 与 fork 的关系 | 建议动作 |
|---|---|---|---|
| **#48957** skip empty c128 kernel launch（~2x kernel） | **已合入**（2026-07-22，yewentao256） | fork 内已存在 `save_partial_states` 模块及调用（compressor.py:349）[实证] → **基本确认已含**（微基准：B300 eager 51.7µs→17.0µs，graph 6.15µs→3.36µs） | 无需动作；如需可做一次代码 diff 复核 skip 判定位置 |
| **#49486** skip topk/router（3.4% E2E TTFT decode） | **已合入**（2026-07-23，MatthewBonanni 合并） | fork 内 `deepseek_v4/` 全目录 grep `num_candidates`/arange-skip **未找到签名** [实证-缺失]；注意我们 fork 的 DSv4 是 MiaAI 自定义实现（nvidia/ 目录含 dspark.py/flashinfer_sparse.py），PR 打的是上游 indexer 代码，可能以不同形态存在 | 43 行小改；先 diff 上游 `deepseek_v4` 与我们 fork 对应 indexer 路径，若缺失则手工 backport。收益口径：上游 B300 上的 3.4% TTFT，GB10 上需自测 |
| **"B12x Direct M=1"** | 不是单一 PR，是 b12x micro-kernel/direct-scheduler 路径（见 §2.C） | fork 走 FlashInfer 0.6.15 micro 路径（存在但旧） | 升级 FlashInfer 0.6.16 后在 GB10 实测 M=1/8/16 |

### 4.2 0.26.0 → 0.27.x 中与我们相关的优化（合并晚于 8/5 的均不在 fork 内）[实证：0.27.0 release notes]

**DSv4 专项（backport 优先级高，均为小-中 PR）**：

| PR | 内容 | 上游实测收益 | 可 backport 性 |
|---|---|---|---|
| #49236 | DSv4 **workspace reuse** | **3.9% E2E TTFT** | 中：与我们 +28GB×43 层 workspace 问题同源！即便代码不能直接搬，思路（workspace 复用）必须对齐 §2.A1 |
| #50298 | 去除冗余 full kernel | 1.88x kernel | 中：需对照我们 fork 的 DSv4 实现确认冗余是否存在 |
| #50004 | adaptive topk width | 1.0% E2E | 中 |
| #50312 | PP buffer 优化 | 省 448MiB 显存 | 低相关（我们无 PP） |
| #48993 | compact MXFP4 indexer KV cache | 显存节省 | 中：我们 KV fp8 + indexer cache 若走 mxfp4 可减负 |
| #48047 | 去 sparse-MLA q-head padding（需 FlashInfer ≥0.6.14，我们 0.6.15 满足） | kernel 效率 | 需确认我们 attention 后端是否命中该路径 [推断] |
| #46789 | DSv4 sequence parallelism | - | 大改，建议随整体升级 |

**平台/基础面（需整体升级或大改）**：PyTorch 2.13 + Triton 3.7.1（**breaking**）；JIT warmup 基建（#47451/#49903，消首次请求编译卡顿——对 b12x CuTeDSL JIT 冷启动痛点同源）；Kimi K3 全栈带来 **DSpark AR fusion（#50242）**——我们正是 DSpark MTP 用户，TP all-reduce 融合与我们 4 节点 TP4 通信优化方向（见 research-comm-overlap-tp4）直接相关 [实证-存在/推断-收益]；共享专家分片选项（#50656）；sm_107 Rubin 前瞻。

**B12X 集成差距（重要）** [实证]：
- 我们 fork：`--moe-backend flashinfer_b12x`、`--linear-backend flashinfer_b12x`（kernel.py 已注册），env 仅 `VLLM_USE_B12X_MOE` + 3 个 W4A16 tile 强制开关。
- vLLM main（NVIDIA 论坛 Eugr 配方，DSV4-Flash-0731 on SM121）：另有 `--moe-backend b12x` / `--linear-backend b12x` / `--attention-backend B12X_MLA_SPARSE`，env 含 `VLLM_USE_B12X_WO_PROJECTION / VLLM_USE_B12X_MHC / VLLM_USE_B12X_FP8_GEMM / VLLM_USE_B12X_MOE / VLLM_USE_B12X_SPARSE_INDEXER / B12X_MLA_SM120_UNIFIED / B12X_MOE_FORCE_A8`——即 **DSV4 的 WO 投影、MHC、FP8 GEMM、sparse indexer、MLA-sparse attention 全部 b12x 化**。这是 fork 与上游在"算力发挥"上的结构性差距之一。
- 来源：https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220

### 4.3 NGC 容器路线现状 [实证]

- `nvcr.io/nvidia/vllm:26.07-py3` 存在且在 DGX Spark（GB10）可用，但**内含 vLLM 0.24.0 基线**（`0.24.0+092c4842.nv26.7`）——比我们 fork（0.26.1-dev, 8/5）**旧**。已知缺陷：tool-calling 500（xgrammar 0.2.0 vs 需 0.2.4，NVIDIA 论坛已确认有内部工单）。
- 结论：NGC 26.07 **不能作为 vLLM 版本升级来源**；其价值是 aarch64 + CUDA 13.3 + sm_121a 的基础环境参考。NGC 26.06 被 Backend.ai 用作 Solar Open 2 镜像基底（`cr.backend.ai/stable/ngc-vllm:26.06-cuda13.3-ubuntu24.04-solaropen2`）。
- aarch64 仍无官方 vLLM wheel；社区成熟路线：**Eugr spark-vllm-docker**（DSV4-Flash+DSpark+B12X 配方即基于它）与 **timothystewart6/vllm-gb10**（全 pin 化 GitHub Actions 构建：vLLM commit SHA + PyTorch/NCCL/FlashInfer + sm_121a，可复现构建验证）。

---

## 5. 升级路径建议（按风险递增排序）

### 路径 0（零升级，本周可做）——配置/开关层收割
1. fork 现有 `VLLM_B12X_W4A16_FORCE_TILE_CONFIG` A/B 上游验证过的 wave-balanced FC2 tile（`tile_k=32, tile_n=512, cta_threads=256`，#146）。
2. 评估在启动预热后启用 b12x 编译冻结（`freeze_kernel_resolution`，0.15.3 已有该 API [实证：`__init__.py` 导出]）+ `B12X_PRINT_COMPILE_PROGRESS` 观测预热覆盖。
3. `flashinfer_b12x_moe.py` 中 workspace 由 b12x 内部管理（`workspace1=(1,), workspace2=(0,)` [实证]）——先做一次 workspace 实测画像（每层实际分配量、是否同形层可共享），为 A1 打样。

### 路径 1（低风险，1–2 周）——FlashInfer 0.6.15 → 0.6.16
- 直接收益：b12x 树统一 MoE API/W4A16 修复族、JIT 磁盘缓存冷启动、compressed_tensors 支持。
- 步骤：测试容器装 0.6.16 wheel → 跑 vLLM fork 的 b12x MoE/nvfp4 测试集 → GB10 上 A/B（非生产实例）。
- 风险：fork `utils/flashinfer.py` 的 API 适配（`has_flashinfer_b12x_moe` 等版本门槛判断）。

### 路径 2（中风险，2–4 周）——workspace 共享池（对症 +28GB×43 层）
- 方案 A：fork 内自做同形 MoE 层 workspace 共享（参照 lablup module-level pool，补丁量小、不动依赖）。
- 方案 B：随 b12x 包升级（0.15.3→1.2.6）获得 geometry 去重 + route arena 复用；**但 1.2.6 要求 torch≥2.12 + cutlass-dsl 4.6.x，与我们 torch 2.11 冲突**——需捆绑 PyTorch 升级或选取中间版本（如 1.0.x/6 月版本，需逐版本核对依赖下限）[推断]。
- 建议先 A 后 B：A 快速止血，B 作为彻底解。

### 路径 3（中高风险，1–2 月）——对齐 vLLM main 的 B12X 全家桶 + DSv4 perf PR 批量 backport
- 以 Eugr spark-vllm-docker / timothystewart6/vllm-gb10 为构建基底，把 fork 的 `deepseek_v4` 自定义模型类移植到较新 vLLM main 基线（目标 0.27.x），吸收：B12X WO/MHC/FP8 GEMM/sparse indexer/MLA-sparse 开关、#49236 workspace reuse、#50298、#50004、#48993、JIT warmup、DSpark AR fusion。
- 明确"可 backport（小 PR 单挑）vs 需整体升级（torch 2.13 breaking、MRV2、seq parallel）"边界：**凡 DSv4 kernel 小 PR 单挑、凡依赖 torch 2.13/MRV2 的整体升级**。
- NGC 26.07 仅作环境参考，不作基线。

### 决策矩阵速览

| 动作 | 成本 | 预期收益 | 风险 | 建议时序 |
|---|---|---|---|---|
| tile config A/B + 编译冻结 | 天级 | 小-中（prefill/稳定性） | 低 | 本周 |
| FlashInfer 0.6.16 | 1–2 周 | 中（MoE 修复族+冷启动） | 低-中 | P2 后 |
| workspace 共享池（fork 内） | 1–2 周 | **大（~28GB 显存）** | 中 | 尽快立项 |
| b12x 1.2.6（捆绑 torch≥2.12） | 月级 | 大（prefill+decode+显存全家桶） | 中-高 | 与路径 3 合并评估 |
| vLLM main 对齐（Eugr 路线） | 1–2 月 | 结构性（算力发挥） | 高 | 立项预研 |

---

## 6. 参考链接

- b12x 上游：https://github.com/local-inference-lab/b12x （commit 7dc6fb8 = 我们快照；36bce2c = 1.2.6）
- b12x 架构文档：https://deepwiki.com/local-inference-lab/sparkinfer/5.1-moe-execution-model-and-scheduling
- FlashInfer 0.6.16 release：https://newreleases.io/project/github/flashinfer-ai/flashinfer/release/v0.6.16 ；API：https://docs.flashinfer.ai/api/fused_moe.html
- vLLM PR：#48957 https://github.com/vllm-project/vllm/pull/48957 ；#49486 https://github.com/vllm-project/vllm/pull/49486 ；b12x 集成 PR #40082
- vLLM 0.27.0 notes（freedom.tech 镜像汇总）
- Backend.ai DGX Spark Solar Open 2（EP+共享池+checkpoint 转换）：https://www.backend.ai/blog/2026-07-serving-solar-open-2-on-dgx-spark ；补丁 https://github.com/lablup/flashinfer/commit/1c623dea557a51e5b92b20ea8a342fd546cc5bf9
- NVIDIA 论坛 DSV4-Flash+DSpark+B12X 配方（Eugr）：https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220
- NGC 26.07 容器问题帖：https://forums.developer.nvidia.com/t/nvcr-io-nvidia-vllm-26-07-py3-tool-calling-requests-500-due-to-xgrammar-transformers-version/378582
- GB10 自建镜像流水线：https://technotim.com/posts/vllm-gb10-docker/ ；https://huggingface.co/Hellohal2064/vllm-dgx-spark-gb10

## 7. 遗留待办（下一步验证点）

1. [ ] 确认镜像内 flashinfer 0.6.15 是否为 MiaAI 补丁版（对比 vanilla 0.6.15 wheel 的 `b12x_moe.py` hash）——影响路径 1 的 diff 基线。
2. [ ] #49486 在我们 fork DSv4 自定义实现中的等价物 diff（43 行）。
3. [ ] P2 场景 workspace 实测画像（每层分配量/同形性），验证 A1 共享池可行规模。
4. [ ] FlashInfer 0.6.16 b12x 树的 EP（`num_local_experts`）实际有效性——我们当前 TP4 不用 EP，但未来 W4A4 路线（arch-w4a4-400t）会用到。
5. [ ] b12x 1.0.x–1.1.x 各版本 torch 依赖下限摸底，寻找"不升级 torch"能拿到的最大 workspace/route 优化集合。

---
*报告基于 2026-08-22 时点的上游状态；生产环境未做任何变更（全部勘察经一次性 `--entrypoint bash` 容器，无 GPU、只读）。*
