# v0.27 + NVFP4 性能验收执行方案（待命版）

- **编制**：testing-expert-4（测试专家 Tessa）
- **日期**：2026-08-15（v1：NVFP4 待命版——验收组合收敛 / bench runner 适配确认 / §4.5 PR 清单模板 / NCCL 与 c5 简化档后续任务）
- **状态**：**待命（standby）**——本方案为执行预案，**不主动启动任何测试**；待性能剖析工程师（general-purpose-7）窗口内 Nsight 剖析完成、decode 劣化根因修复/优化实施后，经 team-lead 指示再执行验收。
- **约束**：不启动测试 / 不停生产 / 不占 GPU（剖析 agent 正在使用窗口）；本文件仅做只读核查与规划落盘。

---

## 0. 目标与判定（用户拍板：唯一目标）

> **不拉 0.26 生产对照、不做同窗 A/B**。本验收的唯一判据 = **0.27 + NVFP4 测试结果优于生产基线**（生产基线 = 0.26 b12x 口径，c1@131K **PR 1896.4 / DE 104.1**）。

**被测对象（NVFP4 路径）**：

| 项 | 值 | 确认状态 |
|---|---|---|
| 权重 | `deepseek-v4-flash-0731-nvfp4`（8/13 modelopt 产出，MTP 全转，164G） | ✅ 四机在位：01/02 `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4`；03/04 NFS `<MODELS_DIR>/deepseek-v4-flash-0731-nvfp4` |
| MoE 后端 | **`flashinfer_cutlass`**（NVFP4 路径可用；`flashinfer_b12x` 被 swiglu_limit 拒——`nvfp4-exploration-final` 已证死路） | ✅ 已确认 |
| linear 后端 | `deep_gemm` **待查**（若其对 NVFP4 支持，优先评估；map 表确认后由 SRE/剖析侧定夺） | ⏳ 待查（见 §6 D1） |
| KV cache dtype | 以剖析/SRE 实际生效值为准（历史冒烟为 `fp8_ds_mla`） | ⏳ 待 SRE 确认 |
| 服务参数 | 生产同参（PIECEWISE cudagraph + capture 1..64 / seqs=6 / util 0.65 / dspark 动态K / max-model-len 400K） | 服务侧由 SRE/剖析方管理 |

**验收组合（用户拍板收敛）**：`c1@32K/coding` + `c1@131K/coding` + `c3@131K/coding` + `c5@131K/coding`（per-request p50，rounds=3）。

**验收标准（主判据，仅 c1@131K）**：

| # | 判据 | 通过条件 | 说明 |
|---|---|---|---|
| A1 | **prefill 达标** | `PR p50 (c1@131K coding) > 1896.4` | 生产基线 1896.4（0.26 b12x 口径） |
| A2 | **decode 达标** | `DE p50 (c1@131K coding) > 104.1` | 生产基线 104.1（0.26 b12x 口径） |

**结论输出**：对每个判据单元格输出 **达标 / 未达标** + **差距**（Δ% 与绝对值）。判定规则：
- A1 且 A2 均达标 → **验收通过（PASS）**，进入后续 NCCL/c5 简化档复测（待督导指示）。
- 任一未达标 → **验收未通过（FAIL）**，记录差距与证据；差距归因（decode 劣化是否已由剖析修复闭环）写入报告。
- c3@131K / c5@131K / c1@32K 作为**回归观察行**（报告但不进主判据）；**c5@131K 无恢复预期**（GB10 物理极限，认知固化：128K 高并发 ~8 tok/s 平台共性）。

> ⚠️ **当前状态提醒**：剖析窗口内快速测试显示 `c1@131K v027 PR 1433.15 / DE 10.95`（R_prefill≈0.756 / R_decode≈0.105），decode 严重劣化。根因疑为 `sparse_mla_sm120` 路径 ≤64 decode dispatch 缺陷（`dg250_decode_ctx` 中 `num_tokens<=64 must go through sparse_mla_sm120_decode_dsv3_2/dsv4`）——**本验收必须在剖析闭环（decode 修复 + 优化实施）之后执行**，否则按当前形态必然 FAIL。

---

## 1. 执行前预检（Preflight，仅剖析/验收窗口开始时执行，非现在）

由测试侧在验收窗口内独立复核（**验收前必做**）：

```bash
# 1) 服务就绪 + 权重路径确认（NVFP4 挂载）
curl -s http://<NODE_IP>:8001/health          # 期望 200/{"status":"OK"}
curl -s http://<NODE_IP>:8001/v1/models | python3 -m json.tool   # served-name = deepseek-v4-flash-0731
ssh node01 "docker inspect vllm-tp4-v027-rank0 --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'" | grep -i models
#    期望 /home/<USER>/models/deepseek-v4-flash-0731-nvfp4 -> /models
# 2) 后端生效确认（NVFP4 路径）
ssh node01 "docker logs vllm-tp4-v027-rank0 2>&1 | grep -iE 'moe.?backend|cutlass|deep.?gemm|nvfp4|FP4' | tail -20"
# 3) TP=4 确认
ssh node01 "docker logs vllm-tp4-v027-rank0 2>&1 | grep -iE 'tensor_parallel_size|tp_size|world_size' | tail -5"
# 4) 131K 单请求无 KV OOM / 无 preemption（max_tokens=16）
python3 bench_prefill_decode_async.py --group V027PRE --endpoint http://<NODE_IP>:8001/v1 \
  --key test-v027-key --model deepseek-v4-flash-0731 --concurrency 1 --ctx 131072 \
  --tasks coding --rounds 1 --engine asyncio --out ./results_v027_nvfp4_pre
# 5) decode 修复闭环确认：单请求 DE ≥ ~80（若仍 ~10 说明 decode 劣化未修复，验收不启动）
python3 bench_prefill_decode_async.py --group V027DECHK --endpoint http://<NODE_IP>:8001/v1 \
  --key test-v027-key --model deepseek-v4-flash-0731 --concurrency 1 --ctx 32768 \
  --tasks coding --rounds 3 --engine asyncio --out ./results_v027_nvfp4_dechk
```

**预检 PASS 门禁**：①–⑤ 全过；NVFP4 权重确认挂载；后端日志出现 `cutlass`/NVFP4 相关生效行；131K 单请求无 KV OOM；**decode 快检 DE ≥ 80**（否则判 decode 未闭环，验收等待）。

---

## 2. 验收矩阵（核心，4 组合 × 3 rounds）

### 2.1 命令（在 02 执行，复用 `run_v027_matrix.sh` 口径）

```bash
# 02 上执行（有 aiohttp）；END/KEY/MODEL 已与 NVFP4 服务对齐（见 §3）
cd /home/<USER>
python3 bench_prefill_decode_async.py \
  --group V027NVFP4 \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key test-v027-key \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,3,5 --ctx 32768,131072 \
  --tasks coding \
  --rounds 3 --engine asyncio \
  --out ./results_v027_nvfp4
```

- **组合数**：conc {1,3,5} × ctx {32K,131K} × coding = **6 组合**；其中**验收 4 组合** = c1@32K + c1@131K + c3@131K + c5@131K（c3/c5 为回归观察行）。
- **口径**：per-request p50（`p50_prefill_tps` / `p50_decode_tps`）；跨组对比禁用 `agg_*`。
- **warmup**：先跑 3 个 512 ctx 请求触发 JIT/cudagraph 编译（或直接以 c1@32K 档充当 warmup）；随机前缀铁律（脚本内置 uuid4 随机，`hit=0` 校验）。
- **总耗时估算**：~40–50 min（c1@131K ~3 min/轮 ×3、c5@131K ~8 min/轮 ×3，含 warmup）。

### 2.2 对比表模板（结果落盘处）

| 档位 | ctx | 指标 | 生产基线 0.26 b12x | 0.27+NVFP4 实测 | Δ% | 达标/未达标 | 差距 |
|---|---|---|---|---|---|---|---|
| c1 | 131K | PR p50 | **1896.4** | | | | |
| c1 | 131K | DE p50 | **104.1** | | | | |
| c1 | 32K | PR p50 | 2222.2 | | | 参考 | |
| c1 | 32K | DE p50 | 109.7 | | | 参考 | |
| c3 | 131K | PR p50 | 732.63 | | | 参考 | |
| c3 | 131K | DE p50 | 34.48 | | | 参考 | |
| c5 | 131K | PR p50 | 595.53 | | | 对照 | |
| c5 | 131K | DE p50 | 7.01 | | | 对照（无恢复预期） | |

> 基线来源：`nvfp4-exploration-final-2026-08-14.md` §3.1/§3.2（0.26 b12x 生产全矩阵，per-request p50 / coding 口径，与验收同口径可比）。
> 单元格值统一取 `summary_V027NVFP4.json` 的 `p50_prefill_tps` / `p50_decode_tps` / `p50_ttft_s`。

---

## 3. Bench runner 适配确认（NVFP4 口径，已核查 ✅）

| 核查项 | 现状 | 结论 |
|---|---|---|
| bench 脚本 | 01 `<INSTALL_DIR>/bench_prefill_decode_async.py`（24.2K，8/14 23:53）；02 `/home/<USER>/bench_prefill_decode_async.py` | ✅ 存在 |
| 封装脚本 | 02 `/home/<USER>/run_v027_matrix.sh`（705B，8/15 00:15）：`ENDPOINT=http://<NODE_IP>:8001/v1`、`KEY=test-v027-key`、`MODEL=deepseek-v4-flash-0731`、`BENCH=/home/<USER>/bench_prefill_decode_async.py`、`OUT=/home/<USER>/results_v027_prod` | ✅ 存在 |
| **endpoint/key** | 脚本用 `<NODE_IP>:8001/v1` + `test-v027-key`；与 0.27 测试服务启动脚本（`start_v027_head.sh` API_KEY=<API_KEY>, PORT=8001）**一致** | ✅ 无需改 |
| **权重路径** | bench 仅经 API 发请求，**不感知权重路径**；NVFP4 权重由服务侧挂载（`MODEL_SRC=/home/<USER>/models/deepseek-v4-flash-0731-nvfp4` → `/models:ro`） | ✅ bench 零改动 |
| **served-model-name** | 服务侧 `--served-model-name deepseek-v4-flash-0731`（不变），bench `--model deepseek-v4-flash-0731` 对齐 | ✅ 无需改 |
| 运行位置 | 02（有 aiohttp）；结果回传 01 | ✅ 沿用 |

> **结论**：`run_v027_matrix.sh` 对 NVFP4 口径**完全适配、零改动**（endpoint/key/model-name 均未变，权重路径服务侧管理）。验收时仅需把 `--concurrency`/`--ctx`/`--tasks` 收敛为 §2.1 的 4 组合验收档（或直接用 6 组合主矩阵，取其中 4 组合判读）。

---

## 4. 重点优化项 PR 清单记录模板（§4.5，NVFP4 路径）

> 验收报告按下表**逐项记录「实测 / 预期」**；触发条件满足与否一并记录（满足才算该 PR 可兑现）。**注意：NVFP4 路径下 `flashinfer_b12x` 不可用（swiglu_limit 拒），#4495（B12x Direct M=1）不适用，从本清单剔除。**

| PR | 优化 | 声称（社区实测） | 触发条件 | NVFP4 路径是否满足 | 验证方法 | 实测/预期 |
|---|---|---|---|---|---|---|
| #48957 | skip 空 c128 | kernel ~2× | **cudagraph≠FULL** | 待确认（生产 PIECEWISE≠FULL；但 NVFP4/cutlass 下 c128 压缩路径是否存在需日志佐证） | grep c128/skip + prefill | |
| #49486 | skip topk/router | TTFT -3.4% | **prefill≤2048** | 待确认（32K/131K 档 prefill 是否触发 ≤2048 段） | c1@32K/131K prefill 对比 | |
| #49236 | EagerScratchPool | TTFT -3.9% | 已含于构建 | 满足（0.27.1 构建天然含 C++ op） | 启动日志 + prefill | |
| #50298 | FlashMLA workspace | kernel 1.88× | FlashMLA 路径 | 待确认（NVFP4/cutlass 下 FlashMLA 路径） | grep workspace/FlashMLA | |
| #48047 | q-head padding 移除 | 去冗余计算 | flashinfer ≥0.6.14 | 满足（镜像含 0.6.14+；**TP4 直接受益**） | grep padding + prefill | |
| #48993 | compact MXFP4 indexer | KV 减半级 | **MXFP4 ≠ NVFP4** | **不可叠加**（NVFP4 权重/KV 路径与 MXFP4 indexer 不同 code path；日志可见相关路径则记录，否则 **unassessed**） | 日志可见则记录；不可见标注 unassessed | |

> 记录规则：触发条件满足且日志可验证 → 记录实测数字；触发条件不满足或日志不可见 → 标注 `N/A` / `unassessed`（不臆测、不假设收益）。

---

## 5. 后续任务（待督导指示执行，非本次待命执行）

> 均为**简化档**方案——保留核心判据单元格与最小执行面，供剖析/验收完成后按指示快速执行。

### 5.1 NCCL A/B（简化档，~45–60 min）

从完整版 `test-nccl-ab-plan-2026-08-14.md` 收敛为 3 档：

| # | 参数 | 生产基线 | 试验值 | 验证 |
|---|---|---|---|---|
| N-A0 | 基线复测 | 生产 env | 同生产 | c1@131K PR/DE |
| N-A1 | `NCCL_MIN_NCHANNELS` | 2 | **4** | busbw + c1@131K PR/DE |
| N-A2 | `NCCL_BUFFSIZE` | 默认 4M | **8M** | busbw + c1@131K PR/DE |

- PASS：busbw ↑ 且 c1@131K PR/DE 不劣化（≥A0×0.97）；否则回滚。
- 执行前置：SRE 确认 0.27/NVFP4 容器内 NCCL 版本与 LD_PRELOAD 行为（分支 A/B/C，见完整版 §2）。

### 5.2 c5 诊断（简化档，~40 min）

从完整版 `test-c5-recovery-plan-2026-08-14.md` 收敛为 2 档（**c5 无恢复预期，仅诊断归因**）：

| # | 参数变更 | 试验值 | 假设 |
|---|---|---|---|
| C-B0 | 基线复测 | 同生产 | 同窗对照 |
| C-B2 | `--max-num-batched-tokens` | **8192** | H1 prefill 切块/插队 |

- 每档记录 c5@131K DE + TTFT 逐 wave；若 TTFT 占主导而纯 decode 段正常 → 证实 H1（prefill 排队稀释），接受「长档并发 ≤c3」处置建议写入报告。
- **不追加** B1/B3/B4 变体（c5 无恢复预期，已拍板）；组合档默认不做。

---

## 6. 决策点与待确认项

- **D1（待查）**：`deep_gemm` 对 NVFP4 支持（map 表）——若支持则优先评估为 linear 后端；否则沿用 cutlass。由 SRE/剖析侧在服务端确认后反馈，测试侧据此锁定 §2.1 服务参数。
- **D2（待 SRE）**：0.27+NVFP4 服务实际 `--kv-cache-dtype` 与生效后端（预检 §1 第 2 步日志确认）。
- **D3（已确认）**：NVFP4 权重四机路径全部在位（01/02 本地 + 03/04 NFS）。
- **D4（已拍板）**：不做 0.26 同窗对照；唯一目标 = 0.27+NVFP4 优于生产基线（1896.4 / 104.1）。
- **D5（已拍板）**：验收组合 = c1@32K/coding + c1@131K/coding + c3@131K/coding + c5@131K/coding（per-request p50，rounds=3）；主判据仅 c1@131K（PR>1896.4 且 DE>104.1），c3/c5 回归观察，c5 无恢复预期。
- **执行条件**：待 team-lead 指示（剖析闭环后）方可启动 §2 验收；当前保持待命。

---

## 7. 交付物

| 产物 | 路径 |
|---|---|
| 验收矩阵原始行 | `results_v027_nvfp4/rows_V027NVFP4.csv` |
| 验收矩阵汇总 | `results_v027_nvfp4/summary_V027NVFP4.json` |
| 对比表（模板 §2.2 填完，含达标/未达标 + 差距） | `v027-nvfp4-acceptance-result-2026-08-15.md` |
| §4.5 PR 清单（逐项实测/预期） | 写入验收结果报告 §4.5 |
| NCCL/c5 简化档结果 | 待执行后补 |

---

> 本方案由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。**当前状态：待命，未启动任何测试。**
