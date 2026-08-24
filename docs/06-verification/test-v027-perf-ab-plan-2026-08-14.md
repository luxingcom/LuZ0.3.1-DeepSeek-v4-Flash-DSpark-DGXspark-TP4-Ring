# ③ vLLM 0.27 性能 A/B 执行方案（v0.27 升级验证）

- **编制**：testing-expert-1（测试专家 Tessa）
- **日期**：2026-08-14（v3 修订：MiaAI 交叉验证——生产参数验收档 / 重点优化项清单 / c5 无恢复预期）
- **前置**：SRE 已部署 `test-0.2.1-v027`（vLLM 0.27.2.dev0）TP4 四机并冒烟通过；**注意：SRE 冒烟为 enforce-eager/fp8_ds_mla/8K 简化口径，不是生产验收口径**
- **GPU**：TP4 全量占用（4 节点 × 1 rank，每节点 1×GB10）
- **预估总耗时**：**4–4.5 h**（核心矩阵 1.5h + 生产参数验收档 0.5h + 同窗 0.26 对照矩阵 8 组合 1.5h + 预检/warmup 20min + 质量抽查 20min）
- **执行顺序**：③ v0.27 主矩阵 → 生产参数验收档（§2.4）→ 同窗 0.26 对照 → ① NCCL A/B → ② c5 诊断（与 SRE 协调窗口）

---

## 0. 目标与范围

对 vLLM 0.27 测试服务（`test-0.2.1-v027`）做性能 A/B 与质量抽查，判定是否进入**生产切换候选**。

**测试对象参数差异（vs 生产）**：

| 项 | 生产 0.26（anemll 0.2.1-v026.0） | 测试 0.27（test-0.2.1-v027） |
|---|---|---|
| kv-cache-dtype | `nvfp4_ds_mla` | **`fp8_ds_mla`（0.27 无 nvfp4_ds_mla）** |
| 其余 serve 参数 | 见下 | 保持生产同参（b12x/deep_gemm/dspark/seqs=6/util 0.65/capture 64） |

**保持不变的生产参数**（0.27 侧沿用）：
```
--max-model-len 400000 --max-num-seqs 6 --max-num-batched-tokens 4096
--long-prefill-token-threshold 1024 --scheduling-policy priority
--gpu-memory-utilization 0.65 --moe-backend flashinfer_b12x --linear-backend deep_gemm
--speculative-config '{"method":"dspark","num_speculative_tokens":5,
  "draft_sample_method":"probabilistic",
  "num_speculative_tokens_per_batch_size":[[1,1,5],[2,4,4],[5,6,3]]}'
--max-cudagraph-capture-size 64 --cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64
--distributed-executor-backend mp --enable-flashinfer-autotune --enable-prefix-caching
--enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4
```
> ⚠️ 0.27 中 `--moe-backend flashinfer_b12x` / `--linear-backend deep_gemm` 的参数名若被改名/移除，以 SRE 冒烟时确认的等价参数为准（见 §4 决策点 D1）。

**生产基线（per-request p50，0.26，coding 口径）**：

| 档位 | ctx | PR (tok/s) | DE (tok/s) | 备注 |
|---|---|---|---|---|
| c1 | 131K | **1896.4** | **104.1** | 主判据单元格 |
| c4 | 131K | 635.96 | 37.45 | 对照 |
| c5 | 131K | 595.53 | **7.01** | GB10 物理极限（认知固化，见下） |
| c5 | 32K | 678.68 | 16.95 | 对照 |

> **认知固化（MiaAI 交叉验证，双向证据闭合）**：128K 高并发 ~8 tok/s 为 **GB10 物理极限**，**c5@131K 不设恢复预期**；断崖相关结论写入最终报告建议「**长档并发 ≤c3**」。③ 判定中 c5 仅作对照单元格，不作为恢复判据（恢复尝试见方案②，H1 证实后按 C3 处置）。

---

## 1. 执行前预检（Preflight，~20 min）

SRE 冒烟通过后，由测试侧独立复核：

```bash
# 1) 服务就绪
curl -s http://<NODE_IP>:8001/health          # 期望 {"status":"OK"} 或 200
curl -s http://<NODE_IP>:8001/v1/models | python3 -m json.tool   # 含 deepseek-v4-flash-0731
# 2) TP=4 确认（head 日志）
grep -E "tensor_parallel_size|tp_size|world_size|All ranks" <head启动日志> | tail -5
# 3) kv-cache dtype 生效确认
grep -iE "kv.cache|fp8_ds_mla|KV cache format" <head启动日志> | tail -5
# 4) fp8 KV 容量预检：util 0.65 下 400K max_model_len 是否成立
#    用 1 个 131K ctx 请求实测（max_tokens=16），观察无 KV OOM / preemption
python3 bench_prefill_decode_async.py --group V027PRE --endpoint http://<NODE_IP>:8001/v1 \
  --key <KEY> --model deepseek-v4-flash-0731 --concurrency 1 --ctx 131072 \
  --tasks coding --rounds 1 --engine asyncio --out ./results_v027_pre
# 5) 参数核对
grep -oE "kv-cache-dtype [a-z0-9_]+|moe-backend [a-z0-9_]+|linear-backend [a-z0-9_]+|speculative-config[^ ]*|max-num-seqs [0-9]+" <head启动日志> | sort -u
```
**预检 PASS 门禁**：①–④ 全过；131K 单请求无 KV OOM / 无 preemption 告警；`kv-cache-dtype` 确为 `fp8_ds_mla`。

> ⚠️ **fp8_ds_mla vs nvfp4_ds_mla 的隔离问题**：fp8（8-bit）KV 容量约为 nvfp4（4-bit）的一半，可能改变 prefill 内存路径与 KV 命中。为把「0.27 优化收益」与「dtype 变化」分离，§3.3 增加一个**0.26+fp8_ds_mla 对照档**（若 0.26 镜像支持该 dtype）。

---

## 2. 性能矩阵（核心，TP4）

### 2.1 命令（v0.27 主矩阵）

```bash
cd <bench脚本目录>   # 01 上 <INSTALL_DIR>/ 下 bench_prefill_decode_async.py
python3 bench_prefill_decode_async.py \
  --group V027 \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <KEY> \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,3,4,5 \
  --ctx 32768,131072 \
  --tasks coding,json,prose \
  --rounds 3 \
  --engine asyncio \
  --sanity-log <head启动日志> \
  --out ./results_v027
```
- **组合数**：4 conc × 2 ctx × 3 task = **24 组合**（本方案核心矩阵）。
- **口径**：per-request p50（脚本 `p50_prefill_tps` / `p50_decode_tps` / `p50_ttft_s`）；**跨组对比禁用 `agg_*`**（批内聚合口径与生产基线不一致）。
- **注意**：本矩阵在 SRE 部署的 0.27 测试服务上跑（冒烟为 enforce-eager/8K 简化口径），作**扫描/横向对比**用；**验收判据（G1/G2）以 §2.4 生产参数验收档为准**。
- **warmup**：矩阵开始前先跑 3 个 512 ctx 请求触发 JIT/cudagraph 编译（或直接先跑 32K 档充当 warmup）；0.27 有 JIT warmup 基建（FA4 深化项），首请求编译延迟应已消除，但仍保留 warmup 步骤。
- **随机前缀铁律**：脚本内置 uuid4 随机填充防 prefix-cache（`hit=0` 校验）；若日志显示 `prefix hit` 增加，重跑该组合。

### 2.2 同窗 0.26 对照矩阵（D3 已拍板：必须执行，8 组合）

> 生产基线数字来自历史窗口，为避免客户端/网络/温度漂移，**同窗口用同脚本重跑 0.26 生产镜像作对照**。D3 已拍板：v0.27 主矩阵后**必须执行** 8 组合对照（c1/c5 × coding），预算 **+1.5h**；与 SRE 协调窗口（停 0.27 → 起 0.26，用生产脚本 `start_tp4_cluster.sh`），避免互踩。

```bash
# 0.26 生产镜像：<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0
# 切换：SRE 顺序停 0.27 测试容器 → bash start_tp4_cluster.sh 起 0.26（head-first ~8min）
# 0.26 侧保持生产参数（nvfp4_ds_mla / b12x / deep_gemm / seqs=6 / util 0.65）
python3 bench_prefill_decode_async.py \
  --group V026 \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <KEY> --model deepseek-v4-flash-0731 \
  --concurrency 1,5 --ctx 32768,131072 \
  --tasks coding --rounds 3 --engine asyncio \
  --sanity-log <head启动日志> --out ./results_v026
```
- **8 组合 = c1/c5 × 32K/131K × coding**（判据单元格全覆盖：G1/G2 主判据 + c5 对照）。
- 切换时**通知 SRE**（同窗互斥，防止 SRE 冒烟/部署动作与对照矩阵冲突）；对照完成后切回 0.27 继续后续项。

### 2.3 可选：dtype 隔离档（仅当 0.26 支持 fp8_ds_mla）

```bash
# 0.26 镜像 + --kv-cache-dtype fp8_ds_mla（其余同生产），group V026F8
# 仅跑 c1/c5 @ 131K coding，用于把「0.27 优化」与「dtype 切换」分离
```

### 2.4 生产参数验收档（MiaAI 口径，必做）

> **SRE 冒烟（enforce-eager / fp8_ds_mla / 8K）不是生产验收口径**。主矩阵之外必须补一档**生产参数验收**：cudagraph 用生产同款（`PIECEWISE` + `--max-cudagraph-capture-size 64` + `--cudagraph-capture-sizes 1..64`），仅 kv-cache-dtype 用 0.27 的 `fp8_ds_mla`，验证 0.27 在生产口径下的真实增益。#48957（skip 空 c128）触发条件为 **cudagraph≠FULL**，本档 PIECEWISE 满足。

```bash
# 0.27 侧：--compilation-config '{"cudagraph_mode":"PIECEWISE"}'
#         + --max-cudagraph-capture-size 64 + --cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64
# 其余同 §0 生产参数（b12x / deep_gemm / dspark / seqs=6 / util 0.65 / long-prefill 1024 / priority）
python3 bench_prefill_decode_async.py --group V027PROD \
  --endpoint http://<NODE_IP>:8001/v1 --key <KEY> \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,3,5 --ctx 32768,131072 \
  --tasks coding --rounds 3 --engine asyncio \
  --sanity-log <head启动日志> --out ./results_v027_prod
```
- **9 组合 = c1/c3/c5 × 32K/131K × coding**（覆盖 G1/G2 判据单元格 + c3 对照）。
- 判读：本档 vs 0.26 生产基线（同为 PIECEWISE 口径）= **0.27 生产口径增益，G1/G2 主判据以本档为准**；本档 vs §2.1 主矩阵（若为非 PIECEWISE 口径）= cudagraph 模式差异。

### 2.5 耗时估算（TP4 全量）

| 段 | 内容 | 预估 |
|---|---|---|
| Preflight | §1 五步 | 20 min |
| V027 矩阵 | 24 组合（131K c5 每 task ~8 min，c4 ~3 min） | 1.2–1.5 h |
| 生产参数验收档（必做） | 9 组合（c1/c3/c5 × 32K/131K × coding） | 0.5 h |
| V026 对照矩阵（D3 必做） | 8 组合（c1/c5 × coding）+ 0.27→0.26 切换 | 1.5 h |
| 质量抽查 | §5 | 20 min |
| 容器切换 | 0.27↔0.26（至少 2 次） | 2×8 min |

---

## 3. 对比表模板与判定

### 3.1 逐项对比表模板（结果落盘处）

| 档位 | ctx | task | 指标 | 生产基线 0.26 | v0.27 实测 | Δ% | 判定 |
|---|---|---|---|---|---|---|---|
| c1 | 131K | coding | PR p50 | 1896.4 | | | |
| c1 | 131K | coding | DE p50 | 104.1 | | | |
| c1 | 131K | json | PR p50 | 2013 | | | |
| c1 | 131K | json | DE p50 | 115 | | | |
| c1 | 32K | coding | PR p50 | 2208 | | | |
| c1 | 32K | coding | DE p50 | 114 | | | |
| c3 | 131K | coding | PR p50 | 808 | | | |
| c3 | 131K | coding | DE p50 | 45 | | | |
| c4 | 131K | coding | PR p50 | 635.96 | | | |
| c4 | 131K | coding | DE p50 | 37.45 | | | |
| c5 | 131K | coding | PR p50 | 595.53 | | | |
| c5 | 131K | coding | DE p50 | 7.01 | | | |
| c5 | 32K | coding | PR p50 | 678.68 | | | |
| c5 | 32K | coding | DE p50 | 16.95 | | | |
| …（json/prose 全量由 results_v027/summary_V027.json 展开） | | | | | | | |

> 单元格值统一取 `summary_<group>.json` 的 `p50_prefill_tps` / `p50_decode_tps` / `p50_ttft_s`；prose 档投机接受率低，作为辅助参考不参与主判据。

### 3.2 主判据（Gate）

| # | 判据 | 通过条件 |
|---|---|---|
| G1 | **prefill 提升（分档，D2 已拍板）** | 以 `R = v0.27 生产参数验收档（§2.4）c1@131K coding PR p50 / 0.26 同窗 c1@131K coding PR p50` 分档（见下） |
| G2 | **decode 不回退** | v0.27 生产参数验收档 c1@131K coding **DE p50 ≥ 104.1 × 0.95**（≥98.9）；c5@131K 仅作对照（GB10 物理极限，**无恢复预期**，见 §0 认知固化） |
| G3 | **质量无劣化** | GSM8K-20 acc 不劣于 0.26（§5），且 tool-call/reasoning parser 抽查无回归 |
| G4 | **稳定性** | 矩阵 0 错误 / 0 超时 / 无 KV OOM；131K 档 preemption 增量 = 0 |

**G1 分档判定（用户拍板）**：

| R 档位 | 结论 |
|---|---|
| **R ≥ 1.20**（PR ≥20%） | **强切换候选**（进入方案② NCCL/c5 复测后统一评审） |
| **1.05 ≤ R < 1.20**（5–20%） | **中等收益**——逐项呈报用户决策（附同窗 V026 对照、重点优化项清单 §4.5、dtype 隔离结论） |
| **R < 1.05**（<5%） | **不切换**（维持 0.26），记录证据 |

**结论规则**：
- 强切换候选：G1 强档 + G2–G4 全过 → 生产切换候选（进入方案②统一评审）。
- 中等收益：G1 中档 + G2–G4 过 → 呈报用户逐项决策；用户批准后方可进入生产切换候选。
- 任一质量/稳定性项劣化（G3/G4 失败）→ **不切换**，记录回归证据。
- **长档并发建议**：c5@131K 断崖（~8 tok/s）为 GB10 物理极限，**「长档并发 ≤c3」建议写入最终报告**（认知固化）。

> ⚠️ **口径提醒**：`1.668×` 是 B12x Direct M=1 的 kernel 级声称，**不是 E2E 阈值**，不能直接当 G1 用；E2E R 受限于 prefill 耗时结构（通信 ~6.7% 等），R ≥ 1.3 即视为 B12x 批显著兑现（§4.3）。

---

## 4. B12x Direct M=1（#4495）生效验证

### 4.1 目标
确认 0.27 的 B12x **Direct M=1** prefill 路径真实生效（而非仅在代码里），并量化其对 prefill 的贡献。

### 4.2 方法一：日志/配置检查（生效性证据）

```bash
# a) MoE backend 确认：启动日志应出现 b12x 相关行
grep -iE "b12x|b12|moe.?backend|flashinfer.*moe" <head启动日志> | tail -20
# b) autotune 内核选择（--enable-flashinfer-autotune 已开）：b12x 内核配置
grep -iE "autotune|kernel.*selected|b12x" <head启动日志> | tail -30
# c) M=1 特征：0.27 b12x 若打印 Direct/M=1 模式行则直接确认
grep -iE "direct|M=1|m.?=?1.*batch|batch.*m.?=?1" <head启动日志> | tail -10
# d) 参数面核对（0.27 是否仍接受并应用）
grep -oE "moe-backend [a-z0-9_]+" <head启动日志> | sort -u
```
**生效性判定**：出现 b12x backend + autotune 选择记录（或显式 Direct/M=1 行）→ 判定「路径生效」；若 0.27 日志不打印内核级信息，则标注「仅参数生效，内核级证据缺失，以 4.3 比例佐证」。

### 4.3 方法二：prefill 提升比例（收益量化）

- **主指标**：`R = v0.27 c1@131K coding PR p50 / 0.26 同窗 c1@131K coding PR p50`
  - 参考：0.26 基线 1896.4；若 R ∈ [1.3, 1.7] → 与「prefill 优化批（Direct M=1 + 空 c128 skip + topk/router skip + q-head padding 移除 + adaptive topk）合计」的量级一致，判定收益兑现。
  - **1.668× 为 kernel 级声称**，E2E 受限于 prefill 耗时结构（通信 ~6.7% 等），预期 R < 1.668；R ≥ 1.3 即视为显著兑现。
- **辅助指标**：c1@32K coding PR p50 的 R'（32K 档 B12x Direct M=1 收益更接近 kernel 口径）。
- **隔离归因（尽力而为）**：
  - 若 0.27 暴露 b12x 开关（env/flag，如 `VLLM_USE_B12X_MOE` 或等价），跑一档**关闭 b12x**（回退默认 MoE backend）的 c1@131K 对比，差值即 B12x 批贡献；
  - 若无开关，则以「0.26 b12x vs 0.27 b12x 的 R」作为 prefill 优化批总量，**不强行拆分到单一 PR**（文档记录此局限）。

### 4.4 记录模板

| 证据类别 | 检查项 | 结果 |
|---|---|---|
| 参数 | `moe-backend` 应用 | |
| 日志 | b12x / autotune / Direct 行 | |
| 内核 | M=1 显式证据 | |
| E2E | R (c1@131K) / R' (c1@32K) | |
| 隔离 | b12x on/off 差值（若可行） | |

### 4.5 重点优化项验证清单（MiaAI 交叉验证，社区已核验）

> 验收报告按下表**逐项记录「实测 / 预期」**；触发条件满足与否一并记录（满足才算该 PR 可兑现）。

| PR | 优化 | 声称（实测数字，社区已核验） | 触发条件 | 本环境是否满足 | 验证方法 | 实测/预期 |
|---|---|---|---|---|---|---|
| #4495 | B12x Direct M=1 | prefill 1.668×（kernel 级） | b12x backend + prefill M=1 | 满足（`--moe-backend flashinfer_b12x`） | §4.2/4.3 日志 + R | |
| #48957 | skip 空 c128 | kernel ~2× | **cudagraph≠FULL** | 满足（PIECEWISE，§2.4） | grep c128/skip + prefill | |
| #49486 | skip topk/router | TTFT -3.4% | **prefill≤2048** | 满足（512/2048 档） | c1@32K prefill 对比 | |
| #49236 | EagerScratchPool | TTFT -3.9% | 已含于构建 | 满足（构建内） | 启动日志 + prefill | |
| #50298 | FlashMLA workspace | kernel 1.88× | FlashMLA 路径 | 待确认（MLA 路径） | grep workspace/FlashMLA | |
| #48047 | q-head padding 移除 | 去冗余计算 | flashinfer ≥0.6.14 | 满足（0.27 自带 0.6.14+，**TP4 直接受益**） | grep padding + prefill | |
| #48993 | compact MXFP4 indexer | KV 减半级 | **待确认是否含于 0.27.1 + 对 fp8_ds_mla 收益** | 未知 | 日志可见相关路径则记录；不可见标注 **unassessed** | |

> **#48993**：验收时若日志可见 MXFP4 indexer 相关路径则记录实测；不可见则标注 `unassessed`（不臆测、不假设收益）。

---

## 5. 质量抽查（GSM8K-20 + parser）

### 5.1 GSM8K 抽样 20 题（对比 0.26 输出自洽性）

复用既有脚本 `gsm8k_eval.py`（exact-match 数字提取，temperature=0）：

```bash
# a) 固定题单：从 300 题池取 idx0-19（seed42 定序，保证与 0.26 历史评估同题）
head -20 gsm8k_300_seed42.jsonl > gsm8k_20_seed42.jsonl   # 01 上 sre_work 目录

# b) v0.27 侧评估
python3 gsm8k_eval.py --endpoint http://<NODE_IP>:8001/v1 \
  --key <KEY> --model deepseek-v4-flash-0731 \
  --qs gsm8k_20_seed42.jsonl --out gsm8k_v027_20.json --workers 5

# c) 0.26 侧评估（同窗切换 0.26 后执行，或直接采用历史 gsm8k_300_seed42 同题答案）
python3 gsm8k_eval.py --endpoint http://<NODE_IP>:8001/v1 \
  --key <KEY> --model deepseek-v4-flash-0731 \
  --qs gsm8k_20_seed42.jsonl --out gsm8k_v026_20.json --workers 5
```
**自洽性判定**（对 20 题逐题比对 `gsm8k_v027_20.json` vs `gsm8k_v026_20.json`）：
- 逐题 gold-extract 一致 → 自洽；不一致 → 记入差异清单（最多留 50 条 mistakes）。
- **通过条件**：v0.27 acc ≥ v0.26 acc（或差距 ≤ 1 题 / 5% 绝对值），且无「系统性输出格式回归」（如 reasoning 结构、tool-call 格式变化）。

### 5.2 Tool-call / Reasoning parser 抽查（~10 min）

0.27 移除/改名了部分 flags（research 提示 partial-prefill flags、CPUOffloadingSpec 改名），须验证 parser 行为不变：

```bash
# 3 个 tool-call 请求（auto-tool-choice + deepseek_v4 parser）
python3 smoke_chat.py --endpoint http://<NODE_IP>:8001/v1 --key <KEY> \
  --model deepseek-v4-flash-0731 --tool-call   # 若脚本支持；否则 curl 直发
# 3 个 reasoning 请求（deepseek_v4 reasoning parser）
# 判定：tool_calls 字段结构、reasoning_content 分离与 0.26 一致
```

### 5.3 判定汇总（G3）

| 项 | 0.26 | 0.27 | 判定 |
|---|---|---|---|
| GSM8K-20 acc | | | |
| 输出自洽（逐题一致数） | — | | |
| tool-call 结构 | | | |
| reasoning 结构 | | | |

---

## 6. 回滚与恢复

- 本方案不修改生产脚本；0.27/0.26 切换由 SRE 按 systemctl 顺序停机 SOP 执行。
- 任意环节出现严重劣化（G4 失败 / 服务不可用 / 数据正确性回归）→ 立即停止矩阵，恢复生产：
```bash
ssh node01 "cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh"   # head-first，约 8 min
# 验证：8001=200 + 四机 healthy + PSR（NCCL→8-9、Engine→15-19）
```
- 结果文件一律保留：`results_v027/`、`results_v026/`、`gsm8k_v027_20.json`、`gsm8k_v026_20.json`。

---

## 7. 交付物

| 产物 | 路径 |
|---|---|
| v0.27 矩阵原始行 | `results_v027/rows_V027.csv` |
| v0.27 矩阵汇总 | `results_v027/summary_V027.json` |
| v0.27 生产参数验收档 | `results_v027_prod/summary_V027PROD.json` |
| 0.26 对照矩阵 | `results_v026/summary_V026.json` |
| 对比表（模板 §3.1 填完） | `v027-vs-v026-perf-compare-2026-08-14.md` |
| B12x Direct M=1 证据（§4.4） | 写入对比表 §4 |
| 重点优化项清单（§4.5 逐项） | 写入对比表 §4.5（含 #48993 unassessed 标注） |
| GSM8K-20 结果 | `gsm8k_v027_20.json` / `gsm8k_v026_20.json` |
| 最终结论（含长档并发 ≤c3 建议） | 随方案②/① 汇总至统一验证报告 |

---

## 8. 决策点（督导/用户已拍板）

- **D1（已拍板）**：`--moe-backend flashinfer_b12x` / `--linear-backend deep_gemm` / `--kv-cache-dtype fp8_ds_mla` 参数名以 SRE 冒烟实际为准；**`fp8_ds_mla` 已确认**。
- **D2（已拍板：分档判定）**：PR≥20% → 强切换候选；5–20% → 中等收益逐项呈报用户；<5% → 不切换。已写入 §3.2。
- **D3（已拍板：执行同窗对照）**：v0.27 主矩阵后**必须**用 0.26 生产镜像（`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`）同窗复跑 8 组合（c1/c5 × coding），预算 +1.5h，与 SRE 协调窗口。已写入 §2.2/§2.4。
- **D4（已拍板）**：GSM8K 对照基准接受「历史 0.26 同题答案」（seed42 定序可追溯）；若 0.26 能同窗重跑则优先同窗。
- **D5（MiaAI 交叉验证已固化）**：① SRE 冒烟（enforce-eager/8K）非生产验收口径，**生产参数验收档（§2.4）必做**；② 重点优化项按 **§4.5 清单**逐项记录实测/预期（#48957/#49486/#49236/#50298/#48047 触发条件均已核对）；③ **#48993** 日志可见则记录、不可见标注 `unassessed`；④ **c5@131K 不设恢复预期**（GB10 物理极限），长档并发 **≤c3** 建议写入最终报告。
- **执行顺序（team-lead 传达）**：③ v0.27 主矩阵 → 生产参数验收档（§2.4）→ 同窗 0.26 对照 → ① NCCL A/B → ② c5 诊断。

> 本方案由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
