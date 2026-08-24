# FlashInfer 0.6.16 Rebase 准备 + Eugr 镜像备料与 A/B 测试计划（2026-08-22）

**执行**: fi-eugr-prep（SRE 工程师）· CPU 一次性容器 + 镜像拉取，零 GPU 占用，生产未受干扰
**输入**: p1-p2-research-2026-08-22.md（研究三/四章节）、upstream-tracking-2026-08-22.md
**口径标注**: 【实证-源码】= 容器内代码直接验证；【实证-数据】= 服务器实测；【实证-上游】= 上游页面/仓库直接验证；【推断】= 基于证据的推断需复测

---

## 0. 执行摘要

1. **FlashInfer 0.6.16 rebase 工作量大幅下修**：对 24 个"自定义冲突文件"逐符号做吸收性探测后确认，**几乎所有 fork 补丁意图已被 vanilla 0.6.16 官方吸收**——"自定义文件"实为 0.6.16-dev 分歧快照回移，非真定制。真正的 fork 独有 delta 仅 **5 个文件**（其中 2 个无功能影响）。**试验性 rebase 已在 CPU 容器完成**：0.6.16 + 5 补丁 + 58 个 fork 新增文件，调用面 import 冒烟 22/23 通过。工作量定论：**代码工作 1-2 天（已基本完成），重心是 GPU 回归验证**，原"1-2 周"口径废弃。
2. **Eugr 镜像已拉取完成**（01 节点，34.5GB 解压后，跨境外层曾卡死一次、重试循环恢复）。镜像勘察确认：vLLM `0.1.dev19023+g30038602b`（**local-inference-lab/vllm `dev/gilded-gnosis` 分支**，非 vllm-project main）+ SparkInfer 1.0.1 @ 272a84bd + **flashinfer 0.6.17**（自建 commit 38bf507f）+ torch 2.12.0 + DSV4/DSpark draft 原生注册 + B12X_MLA_SPARSE 后端 + 全套 B12X env 家族。
3. **A/B 不可与生产共享节点**（生产 gpu-memory-utilization 0.80 ≈ 102/128GB，余量不足 Eugr 0.85 需求）→ **需要全集群停机窗口 3-4h**。计划给出 TP4 配置草案、对照口径、判据（attention b12x 化 PR 增益 ≥5% 才值得 backport 评估）、三条社区警示的针对性检查。**头号对齐风险：权重格式**（生产 = modelopt NVFP4 checkpoint vs Eugr 配方 = 官方 BF16 + FORCE_A8），列 Phase-0 预检。

---

## 1. 研究三推进：FlashInfer 0.6.16 rebase 准备

### 1.1 三方差异全景（重做口径）【实证-源码】

一次性 CPU 容器（fi-rebase-prep，生产镜像基底，已销毁；工件落盘 `/tmp/fi_rebase/`）内，以 pip 官方 wheel 0.6.15/0.6.16 与镜像安装版做全路径 MD5 三方比对：

| 分类 | 数量 | 说明 |
|---|---|---|
| SAME_AS_016（镜像已=0.6.16） | **1011** | 0.6.16-dev 定向回移，升级后零变化 |
| VANILLA_015_016_CHANGED | 166 | 镜像=vanilla 0.6.15、上游 0.6.16 改过 → 直接取 0.6.16，无 fork 冲突 |
| **CUSTOM_CONFLICT（真冲突）** | **24** | 既非 0.6.15 也非 0.6.16（前研究口径 23；多出的 1 个是 `_build_meta.py` 版本串，无功能） |
| IMG_ONLY_016_BACKPORT（分歧回移） | 17 | 仅镜像存在的 0.6.16-dev 回移文件，与 0.6.16 final 有分歧 |
| IMG_ONLY_CUSTOM（fork 新增文件） | 58 | 其中 55 个为 `moe_ep/mega/cutedsl_backend_kernels` vendored MegaMoE EP 内核树（SM100+，GB10 不适用但需携带防 import 错） |
| IMG_MOD_016_REMOVED | 0 | 无"fork 补丁文件被 0.6.16 删除"情况 |

### 1.2 24 个冲突文件分类清单（核心交付）

对每个文件：fork 补丁意图（补丁内容逐行取证）+ 0.6.16 吸收性（符号级 grep 探测：fork 出现次数 vs vanilla 0.6.16 出现次数）+ 处理策略。

**A 组：fork 补丁意图已被 0.6.16 官方吸收（19 个）——直接取 0.6.16，无需 rebase**

| 文件 | fork 补丁意图 | 吸收性证据（fork/016 符号计数） | 处理 |
|---|---|---|---|
| autotuner/autotuner.py | `set_autotune_process_group`（跨 rank tactic 同步，防 NCCL symmetric-memory 死锁） | 5=5 | 取 0.6.16 |
| comm/trtllm_moe_alltoall.py | `moe_a2a_combine_into` 新算子（零拷贝 combine） | 5=5 | 取 0.6.16 |
| data/csrc/trtllm_moe_alltoall.cu | 上述算子的 CUDA 实现 | moeA2ACombineIntoOp 3=3 | 取 0.6.16 |
| decode.py | `q_len_per_req>1` 多 token verify（DFlash 关键路径） | 97<101（016 更全） | 取 0.6.16 |
| prefill.py | cute-dsl→trtllm fmha 路由 + uniform_q_len | `_cute_dsl_use_fmha` 3=3 | 取 0.6.16 |
| mla/_core.py | `require_aligned_block_table` 参数 + multi_ctas 简化 | 4=4（multi_ctas 上游保留） | 取 0.6.16 |
| data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu | `queryPolicyHasCompiledTier` tier 覆盖硬校验 | 3=3 | 取 0.6.16 |
| data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu | launched 检查 + FLASHINFER_CHECK | 3=3 | 取 0.6.16 |
| data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh | tier 策略补充 | patch clean（微差） | 取 0.6.16 |
| fused_moe/cute_dsl/fused_moe.py | `per_token_scale` per-token 量化 | 24=24 | 取 0.6.16 |
| fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py | `a_per_token_scale` | 31=31 | 取 0.6.16 |
| fused_moe/cute_dsl/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py | 同上（blackwell 变体） | 16=16 | 取 0.6.16 |
| trace/templates/moe.py | `per_token_scale` trace 模板 | 4=4 | 取 0.6.16 |
| fused_moe/__init__.py | `bgmv_moe_gemm{1,2}_lora_delta` 导出 | 2=2 | 取 0.6.16 |
| aot.py | `has_sm120 or has_sm121` AOT 放行 | 10=10 | 取 0.6.16 |
| attention/cute_dsl/fmha.py | cubin 不可用时 JIT 回退 | 1=1 | 取 0.6.16 |
| data/build_backend.py | nixl_ep 路径迁移 + NCCL wheel symlink | `_synthesize_nccl_builddir` 2=2 | 取 0.6.16 |
| moe_ep/config.py | EP 配置演进（get_num_tokens thunk 等） | 1=1 | 取 0.6.16（GB10 不走 moe_ep） |
| _build_meta.py | `__git_version__="unknown"` | 构建元数据 | 取 0.6.16（版本串归正） |

**B 组：fork 独有、需保留（4 个）+ 1 个决策项**

| 文件 | fork 补丁意图 | 冲突类型 | 处理策略 | 工时 |
|---|---|---|---|---|
| gemm/gemm_base.py | **cuDNN 9.23.0 bf16 GEMM bug 精确 ban**（`_cudnn_bf16_gemm_usable_or_skip`，27 行函数 + 2 调用点） | 语义冲突（上游 +733/-590 大改）但补丁为纯新增 | ✅ 试验性 rebase 已手工套入并编译通过 | 0.5h（已完成） |
| fused_moe/core.py | SM100 小 batch 早退（`major==10 and num_tokens*top_k<2*num_local_experts: return []`，4 行） | patch **clean**（自动） | ✅ 已自动套入 | 0（已完成） |
| comm/trtllm_mnnvl_ar.py | `torch.inference_mode()` 包装移除 + docstring | patch **clean** | 已自动套入；**需复核**是否仍必要（MNNVL=NVLink 路径，GB10 无 NVLink，可能死代码） | 0.5h 复核 |
| comm/allreduce.py | 纯 docstring 修剪（无功能） | patch **clean** | 可直接丢弃（取 0.6.16），保留亦无害 | 0 |
| artifacts.py | BMM cubin artifact hash pin（481dce07） | 语义决策 | **建议取 0.6.16**（上游已是更新的第三版 5988e15c）；fork pin 是 dev 期产物。**进 GPU 回归清单**（BMM 路径 cubin 变更） | 决策项 |

### 1.3 重大口径修正：diff3 的 212 个冲突块绝大多数是伪冲突

机械 diff3 三方合并报 19 文件/212 冲突块（fused_moe.py 30、decode.py 28、routing_custom.cu 31……），一度支持"手工 rebase 1-2 周"的判断。但吸收性探测证明这些是"0.6.16-dev 快照 vs 0.6.16 final"的演进差异（fork 抄的是中间版），以上游 final 为准即可整体消解。**前研究"真实工作量 = rebase 23 个自定义冲突文件（重点 comm/allreduce 与 trtllm_moe_alltoall）"的结论作废**——这两个重点文件恰恰已被上游完整吸收。

### 1.4 试验性 rebase 结果（CPU 容器内完成）

- 工作树 `/tmp/fi_rebase/rb_test/flashinfer` = vanilla 0.6.16 wheel + B 组补丁 + 58 个 fork 新增文件（mega EP 内核树等）+ 17 个分歧回移文件取 0.6.16 final。
- **最终 delta（`/tmp/fi_rebase/final_delta_manifest.txt`）**：相对 vanilla 0.6.16 恰好 **5 个 patched + 58 个 img-only + 0 removed**。
- **CPU import 冒烟 22/23 通过**：`import flashinfer`（0.6.16）、`B12xMoEWrapper`、b12x 树（b12x_moe.py / moe_dispatch.py）、`from flashinfer.prefill import trtllm_ragged_attention_deepseek`、decode `q_len_per_req` 参数、autotuner `set_autotune_process_group`、gemm_base cuDNN ban、mla/_core、prefill、vLLM fork 四个调用面 import（flashinfer / autotuner / concat_ops / decode）全 PASS。唯一 FAIL 是 `hasattr(moe_a2a_combine_into)`——TVM FFI 惰性注册需 GPU materialize 的**环境伪缺陷**，源码 grep 确认符号在（.py 5 处 + .cu 1 处）。
- 已打包：`/tmp/fi_rebase/flashinfer-0.6.16-rebased-experimental.tar.gz`（20MB，含完整包树）。
- 附带发现：01 上已存在他人构建的 `test-0.2.1-v027-fix121a-dg250-fi016p4` 等系列测试镜像（v027 基底 + fi016 变体），说明团队已有 0.6.16 试验线，本报告补齐了源码级差异定论。

### 1.5 rebase 执行计划（给实施代理）

**顺序（低风险先行）**：
1. （已完成）0.6.16 wheel 为底座 → 套 B 组 4 补丁 + 携带 58 新文件 → tar 归档。
2. 决策 artifacts.py（建议取上游）+ 复核 mnnvl inference_mode 移除必要性（GB10 无 NVLink，倾向取上游、实测回归兜底）。
3. 制作测试镜像：生产 recovered 镜像基底 + 替换 site-packages/flashinfer 为 rebased 树（保留 flashinfer_cubin 0.6.14 与 jit-cache——**需核实 0.6.16 是否要求新版 cubin 包**：0.6.16 依赖面与 0.6.15 一致（前研究已证），cubin 0.6.14→0.6.16 的兼容性列入 GPU 冒烟首查项）。
4. GPU 回归验证（见 §4.6 窗口清单 W2）。

**预估工时**：代码工作合计 ≤1 天（其中 80% 已完成）；GPU 回归 1-2 天。

**回归验证点**：
- b12x_fused_moe 冒烟：`tests/kernels/moe/test_flashinfer_b12x_moe.py`（需 GPU）+ W4A16 数值 golden 对比（升级前后逐 token bit-exact 预期不成立——0.6.16 改了 230 文件，改为容差对比）。
- fork 调用面运行时验证：vllm serve 启动（B12xExperts 路径日志 `Using B12xExperts`）、DSpark verify（decode q_len_per_req）、MLA prefill（trtllm ragged / BMM 新 artifact）。
- 全量 A/B：PR 四档 + DE C1/C12 对生产基线。

---

## 2. 研究四推进：Eugr 镜像备料

### 2.1 镜像来源确认【实证-上游】

- 仓库：https://github.com/eugr/spark-vllm-docker（build-and-copy.sh / run-recipe.py / recipes/deepseek-v4-flash-0731.yaml）
- 镜像（NVIDIA 论坛 #376220 帖 46 canonical recipe pinned 版本）：
  `eugr/spark-vllm-b12x@sha256:eb3ed2bbb0c91dc6d41282d22532267b5a449088c78a032400cd887fe9ddd2c5`（论坛标注 23.4GB；拉取实测解压后 34.5GB）
  https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220/46
- 配方原文（帖 46）：DSV4-Flash-0731 @ TP2 双 Spark、ConnectX-7 直连、B12X 全家桶 env、`--kv-cache-dtype fp8`、block 256、dspark n=5、cudagraph 64、`VLLM_USE_BREAKABLE_CUDAGRAPH=0`。

### 2.2 拉取执行记录【实证-数据】

- 01 节点 13:49 UTC 启动 nohup docker pull（daemon.json 已配 DaoCloud/dockerproxy/1panel 三镜像加速源）。
- 13:56 起外层下载**卡死**（进程 futex 等待、8 秒仅 8KB 读取；已下 ~64GB/20 层中 18 层）→ kill 后以 30 次重试循环重启，已完成层全部复用（Already exists），**14:25 UTC PULL_SUCCESS**。
- 磁盘余量：2.6T（26% 使用），无压力。生产容器全程未受影响（vllm-tp4-rank0 healthy）。

### 2.3 镜像勘察结果【实证-源码】

`docker run --rm --entrypoint bash <img>`（CPU、无 GPU 绑定）：

| 项 | 值 | 备注 |
|---|---|---|
| vLLM | `0.1.dev19023+g30038602b.d20260805` | 构建于 2026-08-05，与我们 fork 基线同日快照 |
| vLLM 源 | **local-inference-lab/vllm `dev/gilded-gnosis`** @ 30038602b | **非 vllm-project main**——Luke Alonso（b12x 作者）的 fork 分支 |
| SparkInfer | 1.0.1 @ 272a84bd（lukealonso/b12x master） | 即更名回 b12x 前的 sparkinfer 1.x |
| flashinfer | **0.6.17**（自建 @ 38bf507f） | 比我们 0.6.15、比官方 0.6.16 都新 |
| torch | 2.12.0+cu130 / cutlass-dsl 4.6.0 | 印证 SparkInfer 1.x 需 torch≥2.12（我们 2.11 冲突点） |
| NCCL | pip nvidia-nccl-cu13 2.29.7 | **非**社区推荐的 2.30.4（见风险 R4） |
| CUDA base | nvidia/cuda:13.0.2-devel-ubuntu24.04，gpu_arch 12.1a | sm_121a 编译 |
| DSV4 支持 | registry：`DeepseekV4ForCausalLM`、`DSparkDraftModel`、`DeepSeekV4MTPModel` @ `vllm/models/deepseek_v4` | **DSV4 + DSpark draft 原生注册**，模型类在 vllm/models/ 而非 model_executor/models/ |
| attention | `v1/attention/backends/mla/b12x_mla_sparse.py` + `b12x_attn.py` + mla/indexer.py；`B12X_MLA_SPARSE` 在 registry | sparse-MLA 后端实证在位 |
| B12X env 家族 | VLLM_USE_B12X_{MOE,WO_PROJECTION,MHC,FP8_GEMM,SPARSE_INDEXER,MINIMAX_M,DCP_A}、B12X_MOE_FORCE_A8、VLLM_USE_V2_MODEL_RUNNER、VLLM_USE_AOT_COMPILE、VLLM_USE_MEGA_AOT_ARTIFACT、VLLM_USE_BREAKABLE_CUDAGRAPH | 全套实证（envs 分散各模块） |
| spec decode | config/speculative.py：DSpark 类型原生 + `dspark_confidence_threshold`、`dspark_budget_frac` 旋钮 | dspark n 可调 |
| KV dtype | cache.py 支持 fp8 族（fp8/e4m3/e5m2/inc…） | **无我们 fork 的 nvfp4_ds_mla**（见 §3.2 口径风险） |
| transformers | transformers 5（build_args transformers_5: true） | 与生产 fork 的 transformers 版本差异注意 API 兼容 |
| mods | 配方的 instanttensor-hybrid-draft-loader、dsv4-reasoning-effort-fix 为**运行时 mod**（仓库 mods/ 目录，非镜像内置） | A/B 需 clone eugr/spark-vllm-docker 或手工等价 |

### 2.4 与我们生产栈对照（更新版）

| 维度 | 生产 fork | Eugr 镜像 |
|---|---|---|
| vLLM | 0.26.1-dev @ d3d3b2cca（MiaAI fork，DSV4 自定义类 + V1 runner） | local-inference-lab dev/gilded-gnosis（DSV4 原生 + **V2 runner + AOT**） |
| MoE | B12xExperts（b12x 0.15.3，W4A16 路径） | b12x 1.0.1 全家桶（A8 强制/或 cutlass W4A8） |
| attention | 自定义 trtllm sparse MLA + flashmla + Triton | B12X_MLA_SPARSE + sparse indexer |
| flashinfer | 0.6.15 混合补丁版 | 0.6.17 自建 |
| torch / NCCL | 2.11 / ring-only 2.30.7 定制 | 2.12 / 2.29.7 |
| KV dtype | nvfp4_ds_mla | fp8 |
| 权重 | /models/deepseek-v4-flash-0731-**nvfp4**（modelopt NVFP4 量化，producer dsv4-nvfp4-experts-mtp-fallback） | 官方 deepseek-ai/DeepSeek-V4-Flash-0731（BF16 + B12X_MOE_FORCE_A8） |
| breakable cudagraph | =1（fork 默认） | =0（社区警示 GB10 伤 prefill） |

---

## 3. Eugr A/B 测试计划

### 3.1 前提与窗口（共享不可行论证）

- 生产 TP4 占满 4 节点；01 上生产实例 gpu-memory-utilization 0.80（≈102/128GB 统一内存）+ aicad 服务族共存。Eugr 配方需 0.85 → **节点内并行共存不可行**（OOM 风险直接威胁生产）。
- **结论：需要全集群停机窗口 3-4 小时**（建议低峰期；生产当前 healthy，可随时恢复）。窗口内 Eugr TP4 全栈 A/B 为主项，可搭车 FI-0.6.16 rebased 冒烟（§4.6 W2）。

### 3.2 头号对齐风险：权重与 KV dtype 不一致（Phase-0 预检）

| 项 | 生产 | Eugr 配方 | 对齐动作 |
|---|---|---|---|
| 权重 | modelopt NVFP4 量化 checkpoint（/models/deepseek-v4-flash-0731-nvfp4） | 官方 BF16 checkpoint + FORCE_A8 运行时量化 | **Phase-0（窗口开始 15min）**：Eugr 栈直接尝试加载我们的 NVFP4 checkpoint（`--model /models/deepseek-v4-flash-0731-nvfp4`）。加载成功 → 同权重公平 A/B；失败 → 两选一：(a) 接受 confound 用官方 BF16 权重跑 Eugr（标注"权重不同"口径），(b) 放弃本轮等权重适配。**严禁**为此临时改 Eugr 镜像内模型代码 |
| KV dtype | nvfp4_ds_mla | fp8（无 nvfp4_ds_mla） | 不可对齐，标注 confound；两臂各自最优配置 |
| dspark n | 7 | 5（speculative.py 支持调参） | Eugr 臂设 n=7 对齐（若启动报错则退 n=5 并标注） |
| max_num_seqs | 12 | 6（配方） | Eugr 臂设 12 对齐 DE C12 |
| threshold / max_num_batched_tokens | 1024 / 4096 | 1024 / 4096（配方相同） | 已对齐 ✓ |

### 3.3 Eugr TP4 serve 命令草案（窗口执行 runbook 用）

```bash
# 4 节点（与生产同拓扑：01 head + 02-04 worker，mp backend，master <NODE_IP>）
docker run --rm --network host --gpus all \
  -v /models:/models \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=-1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e B12X_MOE_FORCE_A8=1 \
  -e NCCL_ALGO=RING -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_BUFFSIZE=8388608 -e NCCL_TUNER_THRESHOLD=40960 \
  -e NCCL_IB_HCA=rocep1s0f1 \
  --entrypoint bash eugr/spark-vllm-b12x@sha256:eb3ed2... -c 'vllm serve /models/deepseek-v4-flash-0731-nvfp4 \
    --served-model-name deepseek-v4-flash-0731 --trust-remote-code \
    --tensor-parallel-size 4 --nnodes 4 --node-rank <N> \
    --master-addr <NODE_IP> --master-port 25999 \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 600000 --max-num-seqs 12 --max-num-batched-tokens 4096 \
    --long-prefill-token-threshold 1024 --gpu-memory-utilization 0.85 \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":7}" \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --max-cudagraph-capture-size 64 --port 8002'
# 注1：NCCL env 先复刻生产 ring 配置；若跨节点崩溃（社区已知问题），按 dredyson 方案容器内 apt 装 nccl 2.30.4 并重试（一次机会，超时即放弃该臂）。
# 注2：配方的两个 runtime mod（instanttensor-hybrid-draft-loader / dsv4-reasoning-effort-fix）若不加也能 serve，仅影响 draft 加载方式与 reasoning 解析——首轮 A/B 可先不加，失败再从 eugr/spark-vllm-docker 仓库取 mods。
# 注3：AOT compile 首次启动编译时间较长（VLLM_USE_AOT_COMPILE=1），窗口预算需含 20-30min 启动编译。
```

### 3.4 对照口径（与生产基线严格对齐）

- **对照组**：生产 fork 基线数值（已有留档：PR 四档 4K/32.8K/65.5K/131K 中位 + DE C1/C12 聚合吞吐）。**窗口内不重跑基线**（省时间），引用 08-22 前留档值；若担心节点状态漂移，窗口结束时恢复生产后补跑一次基线校准（可选）。
- **实验组**：Eugr 臂跑相同脚本（服务器已有 `/tmp/_routea_work/bench_panorama_prefill.py` 轮次机制），同 prompt 集、同 4 档、≥10 轮/档（吸取研究一教训：交错采样、剔除首 2 轮冷启动、记录每轮时间戳）。
- **正确性**：固定 20 条 prompt 集的生成输出 + logprob 采样对比（跨栈 logprob 不逐位可比，用作**劣化哨兵**：Eugr 臂 logprob 分布形态与生产同数量级、无乱码/复读；另做 3-5 条人工可读性检查）。

### 3.5 判据（决策规则）

| 结果 | 判定 | 动作 |
|---|---|---|
| Eugr 臂 PR 四档中位 ≥ +5%（尤其 4K 档 prefill）且 DE C12 无回归（≥-3%）且无 stall/崩溃 | attention b12x 化（及整套 V2+AOT 栈）值得投入 | 启动 upstream 路径 3 立项评估（fork DSV4 类移植 dev/gilded-gnosis 基线，1-2 月） |
| PR 增益 < 5% 或 DE 回归 > 3% | 整体栈不换 | 维持现状，收割研究二（workspace 池）+ 研究三（FI 0.6.16，成本已降至 1-2 天）低风险项 |
| b12x MoE stall 复现（prefill 挂起/超时） | 社区警示在 4 节点 TP4 成立 | 加测 VLLM_USE_B12X_MOE=0 臂（cutlass W4A8 MoE + 保留 b12x attention）一次，区分"MoE 单独问题"vs"attention 仍有增益" |
| 中间态（+2~5%） | 边际 | 结合 FI-0.6.16 rebased 臂结果综合判断；建议同窗口搭车测 breakable cudagraph=0（生产 fork 上，见 W3）|

### 3.6 风险项与针对性检查（社区三警示 + 补充）

| # | 风险 | 来源 | 针对性检查/缓解 |
|---|---|---|---|
| R1 | b12x MoE GB10 prefill stall | Aiden 帖 #372268/188 | 每档首轮 prefill 单独计时+60s 超时守护；stall 即触发 3.5 第三行分支；日志抓 B12X_PRINT_COMPILE_PROGRESS=1 区分编译停顿 vs 真 stall |
| R2 | WO_PROJECTION 高并发不稳 | r0b0tlab 仓库 | DE C12（并发 12）就是压力位：监控该档崩溃/NaN；备选 arm VLLM_USE_B12X_WO_PROJECTION=0 |
| R3 | breakable cudagraph 伤 prefill（我们生产=1，Eugr=0） | 社区多处 | Eugr 臂天然=0；**另列 W3 在生产 fork 单测 =0**（零代码改动 env 翻转，若生产也受益则独立收割） |
| R4 | NCCL 跨节点崩溃（镜像 2.29.7，社区建议 2.30.4） | dredyson 博客 | 首选复刻生产 ring-only env；崩溃一次→容器内 apt nccl 2.30.4 重试一次；仍失败→该臂标注不可用，窗口剩余时间转投 FI-0.6.16 冒烟 |
| R5 | 权重格式不兼容（NVFP4 checkpoint 加载失败） | 本报告 §3.2 | Phase-0 预检；失败走 confound 口径或中止 |
| R6 | AOT 编译冷启动超窗口预算 | 配方特性 | 窗口预算含 20-30min；超时守护 kill 并降级 VLLM_USE_AOT_COMPILE=0 重试 |
| R7 | transformers 5 API 差异导致 tokenizer/parser 报错 | build metadata | Phase-0 serve 启动即暴露；tokenizer-mode 失败时退默认 tokenizer 并标注 |

---

## 4. GPU 窗口需求清单（汇总，供排期）

| 窗口 | 时长 | 占用 | 内容 | 优先级 |
|---|---|---|---|---|
| W1 | 3-4h | **全集群停机** | Eugr TP4 A/B（Phase-0 权重预检 15min → serve 启动 30min → PR 四档+DE 90-120min → 恢复生产 30min） | 高（研究四决策依赖） |
| W2 | 1h | 单节点（可搭车 W1） | FI 0.6.16 rebased 冒烟：b12x_fused_moe 数值 golden + vllm serve 启动 + BMM 新 artifact 回归 | 高（研究三落地前置） |
| W3 | 30min | 生产重启（env 翻转） | 生产 fork breakable cudagraph =0 vs =1 快测（PR 4K 档 + DE C12） | 中（零成本现成差异点） |

---

## 5. 工件索引（服务器 01）

| 项 | 位置 |
|---|---|
| 三方差异分析数据 | `/tmp/fi_rebase/analysis.json`（全路径分类）、`analyze.py` |
| 24 个 fork 补丁工件 | `/tmp/fi_rebase/patches/*.patch`（vanilla015→image 统一 diff） |
| 试验性 rebase 工作树 | `/tmp/fi_rebase/rb_test/flashinfer`（0.6.16 + 5 补丁 + 58 新文件） |
| rebase 归档 | `/tmp/fi_rebase/flashinfer-0.6.16-rebased-experimental.tar.gz`（20MB） |
| 最终 delta 清单 | `/tmp/fi_rebase/final_delta_manifest.txt`（5 patched + 58 img-only + 0 removed） |
| rebase 过程报告 | `/tmp/fi_rebase/rebase_report.md`（patch/diff3 逐文件结果） |
| 吸收性探测脚本 | `/tmp/fi_rebase/probe.py` |
| Eugr 镜像 | 01 本地 `eugr/spark-vllm-b12x@sha256:eb3ed2b...`（34.5GB）；拉取日志 `/tmp/eugr_pull.log` |
| 对照基线数据 | `/tmp/_routea_work/`（PR/DE 既有留档，沿用） |

## 6. 局限声明

- 吸收性探测为符号级 grep（出现次数相同≈语义等价但不逐字节证明）；A 组文件在 GPU 回归中仍需覆盖（上游 final 与 fork 快照可能有微差行为）。
- Eugr 镜像勘察为 CPU 级 import/registry 静态验证，未验证 GPU kernel 可运行性（SparkInfer 1.0.1 attention 栈在 GB10 的实际表现只能 W1 实测）。
- flashinfer_cubin 0.6.14 与 rebased 0.6.16 树的兼容性未验证（0.6.16 依赖声明与 0.6.15 一致，前研究已证，但 cubin 包版本配对列入 W2 首查项）。
- 本任务全程零 GPU、生产未受干扰；一次性容器已销毁，工件落盘 /tmp（注意 /tmp 重启清空风险——建议实施代理窗口前先转存 /home 或推 harbor）。

> 本报告由工程保障团队 fi-eugr-prep 生成，关键决策（W1 停机窗口申请、artifacts.py 取向）请由人类工程负责人复核。
