# 全天窗口短窗 A：FlashInfer 0.6.16 GPU 冒烟 + breakable cudagraph 快测 + budget 上探

- **日期**：2026-08-22（UTC）16:53 – 18:00
- **执行**：SRE 雷克斯（phase2-windowA，工程保障团队）
- **集群**：DGX Spark 4 节点 TP4（GB10/sm_121a），生产 vLLM 0.26.1 fork（DeepSeek V4 Flash B12X W4A16 + Dspark MTP n=7，threshold 4096 采纳态）
- **前序**：实验 D + R1-R3 修复已完成，生产全绿（threshold 4096）
- **状态**：✅ 三项任务全部完成，生产终态 = 4096 基线（已验证）

---

## 任务 1：FlashInfer 0.6.16 GPU 冒烟（W2）—— **Go**

**对象**：`/tmp/fi_rebase/flashinfer-0.6.16-rebased-experimental.tar.gz`（0.6.16 wheel + 5 fork 补丁 + 58 IMG-only 新增文件 rebase 产物，昨日 CPU 冒烟 22/23 后的最终树）。

**方法**：一次性容器（`--rm --gpus all`，基镜像 = 生产 `anemll/dspark-vllm-gx10:0.2.1-v026.0`，torch 2.11.0+cu130 / cutlass 4.5.2 / GB10 cc(12,1)），tarball 解包后 sys.path 前插安装，与镜像内 dist-packages 0.6.15 混合体同容器对照。共享 GPU 纪律：小 shape 显存 <1GB，实测余量 1.4-2.0GiB，未 OOM。

### 验证矩阵（5/5 PASS）

| # | 项目 | 结果 | 细节 |
|---|------|------|------|
| ① | import cutlass/cute/flashinfer 全链 | **PASS** | v=0.6.16，`__file__` 确认加载自新树 |
| ② | B12xMoEWrapper 实例化 + 小 GEMM（w4a16/modelopt） | **PASS** | E=32/K=4/H=512/I=256/T=64；无 NaN/Inf；**输出与生产 0.6.15 混合体逐位一致**（`torch.equal=True`，max_abs=0）；冷进程首跑 3.3s（旧树 5.1s），进程内二跑 0.5ms |
| ③ | b12x 树关键符号 | **PASS** | `b12x_fused_moe` / `B12xMoEWrapper`（顶层 + fused_moe 导出面）均 callable |
| ④ | trtllm_ragged_attention_deepseek | **PASS** | import + callable |
| ⑤ | CuTe-DSL JIT 磁盘缓存（0.6.16 新特性） | **PASS** | nvfp4_quantize_cute_dsl 双进程对照：冷首调 317ms（编译+落盘 .o/meta.json），暖首调 **32.4ms**（命中 `~/.cache/flashinfer/0.6.16/121a/cached_ops/`，符合 3-30ms 预期带边缘） |

### 附加观察
- **b12x moe_dispatch 内核未接入新磁盘缓存**（仍为进程内内存缓存，跨进程每进程重编译 ~3.3s）——与 0.6.15 行为一致，非回归；生产替换后 b12x 路径享受不到该新特性（仅 gemm_mm_fp4 / nvfp4_quantize 等 CuTe-DSL JIT 路径受益）。
- 磁盘缓存路径：`$HOME/.cache/flashinfer/<ver>/<arch>/cached_ops/`（`FLASHINFER_WORKSPACE_BASE` 可重定向；`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` 可关闭）。生产部署建议将 `~/.cache/b12x` 同款挂载策略扩展到 `~/.cache/flashinfer`。

**结论**：FI 0.6.16 冒烟 **Go**。生产替换排后续窗口（建议窗口内先跑 panorama 四档 + DE 全量 + 24h 稳定性）。

产物：`node01:/tmp/fi_rebase/gpu_smoke/`（res_*.json / out_*.pt / run_*.log / cache_home/）。

---

## 任务 2：breakable cudagraph=0 快测（W3）—— **无增益，记录关闭**

**背景**：fork 生产 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（社区警示伤 prefill；且被其禁用 torch.compile）。env 传递点：`start_tp4_head.sh:141` / `start_tp4_worker.sh:146` 的 `docker run -e` 硬编码；checker `check_vllm_script.sh:75` KEY_PARAMS 同步硬编码（改动需双处同步 + 四机同步，本次全部 .bak 留档）。

### 测量（panorama 4K 档 = 实际 8.2K tokens，2 轮中位 + 模式探针）

| 臂 | 配置 | 模式（首 4K TTFT） | PR 4K | 备注 |
|----|------|------|-------|------|
| 臂 1（基线复核） | =1 | 中簇（3.35s） | **2850**（2853/2848） | 中簇基线带 2842-2853 吻合，复核通过 |
| 臂 2 第 1 次 | =0 | 慢簇（4.53s） | 2735 | 模式不匹配，不可比（弃） |
| 臂 2 第 2 次 | =0 | **中簇（3.18s）** | **2826**（2823/2829） | 与臂 1 同模式可比 |

**同模式对比**：2826 vs 2850 = **-0.84%**，未达 +3% 阈值。跨模式参考：臂 2 慢簇 2735 vs 慢簇基线带 2753-2768 亦低 ~1%。

### 行为变化记录（=0 vs =1）
- **torch.compile 恢复启用**：=1 时有 WARNING "disabling vLLM's torch.compile pipeline, -cc.mode=none"（CompilationMode.NONE）；=0 时无此行，CompilationMode.VLLM——compile 管线回归。
- **CUDA graph 捕获档位不变**：PIECEWISE 16/16、FULL 12/12、dspark 11/11（96 档 decode graph 完整保留），"Breakable CUDA graph enabled" 行消失。
- 内存：KV 6,043,880（-14,440 vs 臂 1），CUDAGraph 内存 0.73GiB（vs 0.64GiB）。

**结论**：=0 无 prefill 增益（-0.8%~-1%），且改变 compile 行为引入额外变量。**保留 =1，记录关闭**。若后续重测，建议连同 -cc.mode 显式矩阵一起做（compile on/off × breakable on/off）。

---

## 任务 3：budget 上探 8192（threshold + batched）—— **PR 无增益 + KV 塌缩，记录关闭**

**背景**：M 供给剂量-反应 1024<2048<4096 单调（4096 采纳 +13.5%）；上探 8192 验证 M_e≈192 后是否继续增益（W4A4 / deferred MoE 决策供数）。

**配置改动**：`--long-prefill-token-threshold 8192` + `--max-num-batched-tokens 8192`（四机 head/worker + checker KEY_PARAMS 同步，.bak 留档 `wA-bud8192`）。

### 臂 3 测量（模式探针：中簇，首 4K TTFT 3.31s——与对照同模式）

| 指标 | 4096 基线（今日采纳态验证） | 8192 臂 3 | Δ |
|------|------|------|-----|
| PR 4K（3 轮中位） | 2849（臂 1 复核 2850） | **2809**（2809/2816/2806） | **-1.4%** |
| PR 16K（3 轮中位） | 2829 | **2826**（2792/2826/2826） | -0.1%（持平） |
| **KV cache tokens** | **6,058,320** | **3,846,277** | **-36.5%**（远低于 5.5M 判据线） |
| KV 内存 | 54.37 GiB | 53.52 GiB | - |
| 峰值激活 | 2.03 GiB | 2.69 GiB | +32% |
| CUDAGraph 内存 | 0.64 GiB | 2.05 GiB | 3.2× |
| DE C1 acc/draft 中位（r1-r3） | 2.66 | 2.62 | 噪声带内 |
| DE C12 acc/draft 中位（r1-r3） | 3.23 | 3.15 | 噪声带内 |
| greedy 门（vs 1024 ref） | FAIL（reason+zh 漂移，已知 chunk 尺寸效应） | FAIL（同两 prompt） | 无鉴别力（需 4096-snapshot ref 才能判 8192 特有漂移） |

### 判定
- PR 4K -1.4%（未达 +3%）、16K 持平——**M 供给增益在 4096 处见顶**，8192 无继续增益；单流 M 上推换不来吞吐。
- KV 塌缩 -36.5%（激活 2× + CUDAGraph 3.2× 挤占），3.85M << 5.5M 判据线——并发/长上下文容量严重受损。
- **记录关闭，不建议采纳**（按任务书不自行采纳，本报告供裁定）。**16384 探臂取消**（KV 风险单调放大，8192 已塌缩，无信息增量）。
- 对 W4A4/deferred MoE 决策的含义：M_e 剂量-反应曲线在 4096 顶部平坦化——"budget+threshold 协同上推单流 M" 路线证伪，后续增益需从 M_e 之外（通信/内核）寻找。

---

## 生产终态（4096 基线恢复 + 验证）

- 四机 `start_tp4_head.sh` / `start_tp4_worker.sh` / `check_vllm_script.sh` 已全部回滚：threshold 4096 / batched 4096 / BREAKABLE=1，`diff` 与窗口前基线（`.bak-wA-cg0-20260822` 留档）逐字节一致。
- 试验臂 .bak 留档：`.bak-wA-cg0-20260822`（=0 臂）、`.bak-wA-bud8192-20260822`（8192 臂），四机齐全。
- 终态重启（restart_tp4.sh head-first 编排）→ 就绪 → 验证全绿（详见下）。
- 自愈链恢复：01 `vllm-healthcheck.timer` + `vllm-tp4-head.service`，02/03/04 `vllm-tp4-worker.service`。

### 终态验证记录（18:00 READY，实测）

| 项目 | 值 | 判定 |
|------|-----|------|
| restart_tp4.sh | READY 18:00:46Z | ✅ |
| KV cache size | 6,036,492 tokens（基线带内，臂 1 = 6,058,320） | ✅ |
| PIECEWISE / FULL / dspark graphs | 16 / 12 / 11 | ✅ |
| Breakable CUDA graph enabled | 是（=1） | ✅ |
| torch.compile | 禁用（WARNING 行在，-cc.mode=none） | ✅ |
| 配置回显 | max_num_batched_tokens=4096 / long_prefill_token_threshold=4096 | ✅ |
| 模式探针 | 首 4K TTFT 3.51s（中簇带内偏上） | ✅ |
| PR 4K（复测 3 轮） | **2839**（2840/2848/2839）——中簇带内 | ✅ |
| /health + /v1/models | 200 / 含 deepseek | ✅ |
| 自愈链 | 01 timer+head 服务 active，02/03/04 worker 服务 active，monitor 进程在 | ✅ |
| 四机容器 | rank0-3 全 healthy，单次 startup（自愈链恢复未触发重启） | ✅ |

注：终态首测 PR 4K 出现一次 2661 瞬态（重启后未稳态），复测 3 轮稳定 2839-2848 中簇带内，四机负载/GPU 进程排查无异常，判定为暖机瞬态而非回归。

## 产物索引
- FI 冒烟：`node01:/tmp/fi_rebase/gpu_smoke/`
- 臂测量：`node01:/tmp/_wA/`（arm{1,2,2b,3}_{probe,quick4k}.json、arm3_de.json、restart_*.log、arm*_measure.log）
- 4096 采纳态基线参照：`node01:/tmp/_thr4096/verify/`（panorama_4096.log / de_4096.json / probe_4096.json / greedy_4096.log）

## 时间线（UTC）
| 时间 | 事件 |
|------|------|
| 16:53 | 任务 1 开始（GPU 余量探测 1.1GiB） |
| 16:59 | FI 冒烟矩阵 5/5 PASS（含 bit-identical 对照） |
| 17:01 | CuTe-DSL 磁盘缓存冷/暖验证 PASS |
| 17:06-17:12 | 停自愈链 → 臂 1（=1）重启 READY |
| 17:14 | 臂 1 出数：中簇 PR 2850 |
| 17:16-17:26 | 臂 2（=0）重启 READY |
| 17:28 | 臂 2 出数：慢簇（不可比）→ 重跑 |
| 17:35-17:39 | 臂 2 第 2 次 READY → 中簇 PR 2826（-0.84%） |
| 17:40-17:45 | 臂 3（8192）重启 READY |
| 17:47-17:52 | 臂 3 测量：PR 2809/2826，KV 3.85M，DE 平 |
| 17:54 | 回滚四机 4096 基线（diff 验证） |
| 17:55-18:04 | 终态重启 + 验证 + 自愈链恢复 |
