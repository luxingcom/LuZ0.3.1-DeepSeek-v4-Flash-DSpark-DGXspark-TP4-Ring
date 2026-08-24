# v0.27 + NVFP4 性能验收报告

- **编制**：testing-expert-5（测试专家 Tessa）
- **日期**：2026-08-15（UTC）
- **被测对象**：DGX Spark 四机集群（AICAD）· vLLM v0.27.2.dev0 + NVFP4 权重 + flashinfer_cutlass
- **验收依据**：`v027-nvfp4-acceptance-plan-2026-08-15.md`（用户拍板：唯一目标 = 优于生产基线 PR>1896.4 且 DE>104.1 @ c1@131K）
- **状态**：**执行完成**

---

## 0. 结论摘要（TL;DR）

| 判据             | 单元格            | 生产基线       | 实测          | Δ（绝对/相对）             | 判定        |
| -------------- | -------------- | ---------- | ----------- | -------------------- | --------- |
| **A1 prefill** | c1@131K PR p50 | **1896.4** | **2146.09** | +249.69 / **+13.2%** | **✅ 达标**  |
| **A2 decode**  | c1@131K DE p50 | **104.1**  | **58.04**   | -46.06 / **-44.2%**  | **❌ 未达标** |

> **主判据结论：0.27 + NVFP4 部分达标（prefill 优于生产基线、decode 未达门禁）→ 验收未通过（FAIL）**。  
> decode 差距根因（承 sre-engineer-17 剖析闭环）：GB10 上 NVFP4 路径 MoE 仅可用通用 `cutlass` 内核（`FLASHINFER_CUTLASS`），无 b12x 级专用解码内核（需上游支持）；当前 decode 受 `sparse_mla_sm120_decode_dsv4` + cutlass MoE 通用内核共同限制，DE≈58 较修复前 ~10 已大幅回升但未达 104.1。



---

## 1. 测试环境与配置（实测确认）

| 项              | 值                                                                                                                                                                                                  | 确认方式                        |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 测试镜像           | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:test-0.2.1-v027-fix121a-dg250-ijson-parser`                                                                                                            | docker ps 四机                |
| vLLM 版本        | `v0.27.2.dev0+g6f5dc38d0.d20260814`                                                                                                                                                                | rank0 日志 `core.py:121`      |
| 容器             | `vllm-tp4-v027-rank0~3`（01=rank0/186, 02=rank1/187, 04=rank2/189, 03=rank3/188）                                                                                                                    | docker ps 四机                |
| 权重             | `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4`（01/02）· `<MODELS_DIR>/...-nvfp4`（03/04）→ `/models`                                                                                           | docker inspect rank0 mounts |
| MoE 后端         | **`flashinfer_cutlass`**（`Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of ['FLASHINFER_TRTLLM','FLASHINFER_CUTLASS','MARLIN']`）                                                                | rank0 日志 `nvfp4.py:239`     |
| Linear 后端      | **`deep_gemm`**（UE8M0 生效，`DeepGemmFp8BlockScaledMMKernel`；DeepGEMM warmup 991 iters）                                                                                                               | rank0 日志                    |
| KV cache dtype | `fp8_ds_mla`                                                                                                                                                                                       | rank0 日志 `api_utils.py`     |
| flashinfer     | **0.6.15 修复版 jit-cache**（fi015-full 挂载：flashinfer + flashinfer_jit_cache + cubin 0.6.14）                                                                                                           | docker inspect mounts       |
| 服务参数           | TP=4 / nnodes=4 / max-model-len=400000 / max-num-seqs=6 / max-num-batched-tokens=4096 / gpu-mem-util=0.65 / PIECEWISE cudagraph capture 1..64 / dspark spec 5（per_batch [[1,1,5],[2,4,4],[5,6,3]]） | rank0 日志                    |
| decode 修复      | `patch/deepseek_v4/flashinfer_sparse.py`（`_FLASHINFER_DSV4_DECODE_TOPKS=(128,512,1024)`——C128A topk 宽度 pad 至原生 SM120 内核宽度）+ fi015-full 0.6.15 jit-cache                                            | mount + patch 源码            |
| 预检             | /health=200；/v1/models=`deepseek-v4-flash-0731`；131K 单请求无 KV OOM/preemption；warmup ctx=512 通过                                                                                                      | 实测                          |

---

## 2. 验收矩阵结果（per-request p50 / coding / rounds=3）

| 档位     | ctx      | 指标         | 生产基线 0.26 b12x | 0.27+NVFP4 实测 | Δ%         | 达标/未达标    | 备注          |
| ------ | -------- | ---------- | -------------- | ------------- | ---------- | --------- | ----------- |
| **c1** | **131K** | **PR p50** | **1896.4**     | **2146.09**   | **+13.2%** | **✅ 达标**  | 主判据 A1      |
| **c1** | **131K** | **DE p50** | **104.1**      | **58.04**     | **-44.2%** | **❌ 未达标** | 主判据 A2      |
| c1     | 32K      | PR p50     | 2222.2         | 2360.9        | +6.2%      | 参考        |             |
| c1     | 32K      | DE p50     | 109.7          | 58.0          | -47.1%     | 参考        |             |
| c3     | 131K     | PR p50     | 732.63         | 780.63        | +6.6%      | 参考（优于基线）  | 回归观察        |
| c3     | 131K     | DE p50     | 34.48          | 35.43         | +2.8%      | 参考（优于基线）  | 回归观察        |
| c5     | 131K     | PR p50     | 595.53         | 571.35        | -4.1%      | 对照        | 无恢复预期（符合认知） |
| c5     | 131K     | DE p50     | 7.01           | 7.05          | +0.6%      | 对照        | 无恢复预期（符合认知） |

**逐单元格达标判定**（对照生产基线，主判据仅 c1@131K）：

- c1@131K：PR 达标 ✅（2146.09>1896.4）/ DE 未达标 ❌（58.04<104.1）→ **主判据 FAIL**
- c1@32K：回归观察行——PR 优于基线（2360.9>2222.2），DE 未及（58.0<109.7）
- c3@131K：回归观察行——PR 780.63>732.63（+6.6%）、DE 35.43>34.48（+2.8%），**双双优于基线**
- c5@131K：对照行——PR 571.35<595.53（-4.1%）、DE 7.05≈7.01（+0.6%），**无恢复预期，符合既有认知**（128K 高并发 ~7-8 tok/s 平台共性）

**主判据结论：部分达标（prefill 优于、decode 未达）→ 验收未通过（FAIL）。**  
差距归因：GB10 上 NVFP4 仅可用通用 cutlass MoE 内核（无 b12x 级专用内核，需上游支持）；decode 瓶颈在 cutlass MoE + sparse_mla_sm120_decode_dsv4 共同路径（修复后 DE 由 ~10 回升至 ~58，仍未达 104.1 门禁）。

---

## 3. 原始数据

- 02：`/home/<USER>/results_v027_prod/`
  - `results_v027_nvfp4_c1/rows_V027NVFP4_C1.csv` + `summary_V027NVFP4_C1.json`
  - `results_v027_nvfp4_c35/rows_V027NVFP4_C35.csv` + `summary_V027NVFP4_C35.json`
  - `console_v027_nvfp4_acceptance.log` / `console_v027_nvfp4_c1.log` / `console_v027_nvfp4_c35.log`
- 本地镜像：`results_v027_prod/`（回传）

---

## 4. §4.5 重点优化项 PR 清单（NVFP4 路径实测记录）

> 记录规则：触发条件满足且日志可验证 → 记录实测；不满足/不可见 → N/A / unassessed（不臆测收益）。`flashinfer_b12x` 不可用（swiglu_limit 拒），**#4495 剔除**。

| PR     | 优化                    | 声称           | 触发条件               | NVFP4 路径是否满足                                                                                                                             | 验证方法                     | 实测/预期                                                                        |
| ------ | --------------------- | ------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------- |
| #48957 | skip 空 c128           | kernel ~2×   | cudagraph≠FULL     | 满足（PIECEWISE，capture 1..64）                                                                                                              | grep c128/skip + prefill | 触发条件满足；c128 压缩路径日志不可见 → **N/A（日志不可见）**；PR 2146.09 优于基线已含其收益可能                |
| #49486 | skip topk/router      | TTFT -3.4%   | prefill≤2048       | 部分满足（chunked prefill，long_prefill_token_threshold=1024 → prefill 段 ≤2048；sampler=FlashInfer topk_topp）                                   | c1@32K/131K prefill 对比   | 触发条件满足（chunk≤2048）；skip 日志不可见 → **N/A（日志不可见）**                               |
| #49236 | EagerScratchPool      | TTFT -3.9%   | 已含于构建              | 满足（v0.27.2.dev0 构建天然含 C++ op）                                                                                                            | 启动日志 + prefill           | **满足（构建内含）**；无独立日志                                                           |
| #50298 | FlashMLA workspace    | kernel 1.88× | FlashMLA 路径        | decode 走 `sparse_mla_sm120_decode_dsv4`（DSV4 MLA 路径，patch 含 128MB workspace buffer）                                                      | grep workspace/FlashMLA  | 路径存在（DSV4 workspace buffer in patch）；日志无 FlashMLA 字样 → **unassessed（日志不可见）** |
| #48047 | q-head padding 移除     | 去冗余计算        | flashinfer ≥0.6.14 | 满足（0.6.15 mounted）                                                                                                                       | grep padding + prefill   | 触发条件满足；pass_config `fuse_act_padding=False`；无独立 padding 日志 → **N/A（日志不可见）**  |
| #48993 | compact MXFP4 indexer | KV 减半级       | MXFP4 ≠ NVFP4      | **不可叠加**（NVFP4 权重/KV 路径；实测 `DSA indexer decode path: use_flattening=True (next_n=6, use_fp4_indexer_cache=False)`，indexer 走 FP4 非 MXFP4） | 日志可见则记录                  | **unassessed（NVFP4 路径不适用 MXFP4 indexer，code path 不同）**                       |

**实测补充观察（decode 相关）**：

- rank0 日志出现 autotuner fallback：`No tuned config covers sparse_mla_sm120_decode_dsv4 input_shapes=(...) falling back to runner=... tactic=-1`（如 (60,16,512)/(3,16,512) 等形状超出 tuning bucket）——decode 部分输入形状无 tuned config，回退 tactic=-1，存在 perf cliff 风险；与 DE 未达门禁相关（建议下次 tuning 扩充 buckets / max_num_tokens）。

---

## 5. 生产/测试环境状态

| 项          | 状态                                                                                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 测试容器       | vllm-tp4-v027-rank0~3 四机（验收窗口内运行）→ 验收完成后**已清理**（四机 rm -f，无残留）                                                                                                                                                             |
| 生产 v026    | 已停（窗口），systemd vllm-tp4-head/worker.service 四机 inactive（08:19-08:20 停用，无自愈干扰）                                                                                                                                             |
| embed      | 03/04 `anemll-embed-8022` Up 22h，**全程未触碰**                                                                                                                                                                                |
| 生产 API key | `<KEY_PREFIX_OLD>98...`（<INSTALL_DIR>/secrets/vllm.env，**08-13 轮换后的现行 key**；旧日志中的 `<API_KEY>-11282c...` 已失效）                                                                                                                  |
| 恢复执行       | ① 停测试容器（worker 02/04/03 → head 01）② 8001/25999 释放 ③ `VLLM_API_KEY=c3b4de... bash start_tp4_head.sh`（head）④ 三 worker `VLLM_API_KEY=... NODE_RANK=N VLLM_HOST_IP=... NCCL_IB_HCA=... nohup bash start_tp4_worker.sh` ⑤ 确认就绪 |
| **恢复后确认**  | ✅ 四机容器 Up + healthy（rank0@01 / rank1@02 / rank2@04 / rank3@03）；✅ /health=200；✅ /v1/models 用现行 key=200、旧 key=401；✅ served `deepseek-v4-flash-0731` max_model_len=400000；✅ embed 未触碰                                        |

> **⚠️ 恢复过程发现（重要，建议督导知悉）**：
>
> 1. **API key 已轮换**：旧日志/旧脚本中的 `<API_KEY>-11282c...` 已失效，现行 key 在 `<INSTALL_DIR>/secrets/vllm.env`（`<KEY_PREFIX_OLD>...`，root 0600）。`start_tp4_cluster.sh` 的 worker ssh 段**不传递 VLLM_API_KEY**（worker 脚本 `:?VLLM_API_KEY is not set` 直接退出），首次恢复尝试因此失败——需手动在 ssh 命令内联传 key。
> 2. **systemd 自愈服务仍 inactive**（窗口停用状态）：恢复后的生产容器由 docker 直接托管，**未重新启用** vllm-tp4-head/worker.service。如需恢复自愈（monitor docker-wait 跟随现有容器，安全），请督导指示后再 `systemctl start`。
> 3. 测试产物保留：`<INSTALL_DIR>/scripts/v027-test/`（脚本+patch）、02 `/home/<USER>/results_v027_prod/`（数据）、本地 `results_v027_prod/`（数据镜像）均在位。

---

## 6. 交付物

| 产物   | 路径                                                                            |
| ---- | ----------------------------------------------------------------------------- |
| 本报告  | `deliverables/engineering-assurance/v027-nvfp4-acceptance-2026-08-15.md`      |
| 原始矩阵 | 02 `/home/<USER>/results_v027_prod/` + 本地 `results_v027_prod/`             |
| 验收计划 | `deliverables/engineering-assurance/v027-nvfp4-acceptance-plan-2026-08-15.md` |

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
