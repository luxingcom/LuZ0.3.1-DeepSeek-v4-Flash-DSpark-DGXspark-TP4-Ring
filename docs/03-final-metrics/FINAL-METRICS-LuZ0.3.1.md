# LuZ0.3.1 最终性能指标汇总（FINAL METRICS）

- **汇编**：多库（Docu）· 技术文档师（tech-writer-1）· 工程保障团队
- **日期**：2026-08-24
- **状态**：**LuZ0.3.1 生产采纳后最终指标主汇总**（对应开源仓库 `03-final-metrics/` 目录）
- **用途**：开源发布最终性能指标权威口径；管理层/社区报告引用基准
- **口径警示**：⚠️ 旧口径（08-05 `raw_final_matrix.json` 等）与本表**不可混用**；每行标注测量口径与来源报告

---

## 1. 被测形态基线（LuZ0.3.1，2026-08-23 生产采纳）

| 维度 | 值 | 说明 |
|---|---|---|
| 集群 | 4× DGX Spark GB10（sm_121a）TP4，环网 01-02-04-03 | |
| 模型 | DeepSeek V4 Flash（0731 ckpt） | |
| 基座 | anemll 0.2.1-v026.0（= vLLM 0.26.1 fork） | |
| **MoE 量化** | **W4A4 full（`VLLM_MOE_W4A4=2`）** | MXFP4 payload |
| **池补丁** | **SHARED=1（`VLLM_B12X_SHARED_WRAPPER=1`）** | 跨层 wrapper 去重 |
| **FlashInfer** | **0.6.16**（bind-mount + JIT 缓存） | |
| **threshold** | **4096**（`--long-prefill-token-threshold 4096`） | |
| **util** | **0.82**（`--gpu-memory-utilization 0.82`） | |
| **MTP** | **dspark n=7**（`--speculative-config`） | |
| **max_model_len** | **600000**（600K） | |
| **max_num_seqs** | **12** | |
| **max_num_batched_tokens** | **4096** | |
| **CUMEM** | **NCCL_CUMEM_HOST_ENABLE=0**（08-24 落地） | |
| kv_cache_dtype | nvfp4_ds_mla | |
| cudagraph capture | 1..96（16 档） | |
| NCCL | ring-only 2.30.7 + 4 通道（MIN/MAX=4） | |
| 镜像 | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`（digest sha256:85f2149f…） | |

> 基线变更记录：vs W4A16 B1 基线仅 9 项实际变更（W4A4 full、MIN_M、CG、SHARED、util 0.82、FI 0.6.16 补回、plugin_a1 前缀、池 overlay、checker 同步）——详见 `w4a4-vs-w4a16-diff-audit-2026-08-23.md`。

---

## 2. 一页指标总表（最终采纳口径）

### 2.1 PR 单流 prefill（tok/s，3 轮中位；prompt 实际长度）

| 档位 | 实际 prompt | **LuZ0.3.1** | vs W4A16 B1 | 来源 |
|---|---|---|---|---|
| 4K | 8.2K | **2950.5** | +6.6% | luz031-deployment §3.1 |
| 16K | 32.8K | **2943.6** | +6.3% | 同上 |
| 32K | 65.5K | **2834.2** | +10.5% | 同上 |
| 64K | 131K | **2550.0** | +15.1% | 同上 |

> E5 生产同构复查（arstall-production-closure）：2959.6 / 2984.1 / 2872.2 / 2642.9（+0.3~+3.6% vs 采纳带）→ 生产形态未中招环境 stall。

### 2.2 并发聚合 prefill（4K 档，3 轮中位，tok/s）

| 并发 | **LuZ0.3.1** | vs W4A16 B1 | med TTFT | 来源 |
|---|---|---|---|---|
| C6 | **3057**（3023/3060/3057） | +11.4% | 10.47s | luz031-deployment §3.2 |
| C12 | **3056**（3059/3056/3034） | +11.6% | 18.39s | 同上 |

### 2.3 DE（decode 接受率归一 step_eff，4 轮中位）

| 指标 | **LuZ0.3.1** | vs W4A16 B1 | 判读 |
|---|---|---|---|
| C1 step_eff | **18.2** | +2.8% | ✓ 中性 |
| C12 step_eff | **80.2** | -8.0% | ⚠ 落 W4A4 full decode 代价带 |

### 2.4 官方口径 decode-only（Session A，t/s 中位|最优）

| 场景 | 官方 8/19 参考 | **LuZ0.3.1** | Δ中位 | 来源 |
|---|---|---|---|---|
| C1 | 97.1 \| 124.0 | **73.9 \| 136.5** | -23.9%（+10.1% best） | luz031-bench §2.1 |
| C4 | 218.0 \| 233.7 | **186.7 \| 218.8** | -14.4% | 同上 |
| C8 | 286.3 \| 302.9 | **274.5 \| 348.3** | -4.1%（+15.0% best） | 同上 |
| C12 | 342.8 \| 358.2 | **349.3 \| 397.5** | **+1.9%（+11.0% best）** | 同上 |
| Agent 平均 | 81.3 \| 84.6 | **70.4 \| 76.5** | -13.4% | 同上 §2.2 |
| 单流 fox p512 | 97.1 \| 124.0 | **77.8 \| 131.1** | -19.9%（+5.7% best） | 同上 §2.3 |

### 2.5 资源 / 质量 / 长上下文

| 项 | **LuZ0.3.1** | vs W4A16 B1 | 说明 |
|---|---|---|---|
| weight（MoE 权重显存） | **45.32 GiB** | +11.9%（40.5） | W4A4 full 执行体积更大（池化后 vs 未池化 68.15 省 22.83） |
| KV tokens | **5,730,000** | -5.1%（6,037,164） | ≥5.7M 门过 |
| KV 内存 | 50.81 GiB | -5.1%（53.53） | |
| peak 激活 | 2.03 GiB | 0% | |
| CUDAGraph | 1.4 GiB | +0.7（~0.7） | |
| 质量门（greedy 逐字一致） | **4/4 exact match** | — | |
| needle 64K | **3/3 PASS**（128K 1/2，已知抖动） | — | |
| 回归日志 | **0 error/exception/traceback** | — | |
| 400K decode | **≈99.6 t/s**（11 轮合并中位） | — | issue22-400k：无塌陷 |
| 400K 冷 prefill TTFT | **39.0s** | — | issue22-400k |
| 400K QA 抽验 | **PASS**（tail + deep-tail） | — | 同上 |

---

## 3. 详细指标表（按来源报告）

### 3.1 P0 拆账（Session B，M=4096 生产 prefill 形态，µs/token 采信中位）

| 节点 | 实测中位 | 推算带（fi017） | 份额 |
|---|---|---|---|
| attn 投影 | **11.91** | 15-19 | 54.9% |
| shared experts | **6.98** | 9-12 | 32.2% |
| lm_head | **2.79** | 3-5 | 12.9% |
| **池合计** | **21.68** | 29-34（M=1024 口径） | 100% |

> 池合计占 PR 每 token 总时预算（≈339µs）**6.4%**。decode 带宽墙（M=8 实测）：attn 4.32 / shared 1.99 / lm_head 1.18 ms/step。来源：luz031-bench §3、`_luz031_official_bench/data/p0/p0_accounting_data.md`。

### 3.2 W4A4 vs W4A16 同窗并发代价（G1 对照，decode-only t/s 中位|最优）

| 并发 | W4A4（LuZ0.3.1） | W4A16（G1） | Δ中位 |
|---|---|---|---|
| C1 | 73.9 \| 136.5 | 71.6 \| 93.9 | -3.1% |
| C4 | 186.7 \| 218.8 | 203.5 \| 213.2 | +9.0% |
| C8 | 274.5 \| 348.3 | 306.4 \| 320.3 | +11.6% |
| C12 | 349.3 \| 397.5 | 411.8 \| 441.9 | **+17.9%** |

> 来源：g1-production-restore §1.2（同窗克隆，仅差 `VLLM_MOE_W4A4`）。W4A4 full decode 代价随并发放大；业务以 prefill+并发为主，**不改变 W4A4 生产采纳结论**。

### 3.3 400K 长上下文明细（issue22-400k）

| 项 | 值 | 说明 |
|---|---|---|
| decode set1（3 轮中位） | 38.89 t/s | range 37.46–110.74 |
| decode set2（5 轮中位） | 119.89 t/s | range 59.34–177.52 |
| decode set3（受控 3 轮中位） | 99.61 t/s | |
| **400K 合并（11 轮中位）** | **≈99.6 t/s** | range 37.46–177.52 |
| 冷 prefill TTFT@400K | **39.0s** | 深尾 QA 冷启动 |
| warm TTFT@400K | 4.0–4.4s | prefix cache hit |
| tail-fact QA | PASS（BRAVO-4829） | 400K 末尾 ~58 token |
| deep-tail-fact QA | PASS（CHARLIE-7315） | ~50K token 前 |
| prompt_tokens | 409,605（校准） | |

---

## 4. CSV 数据文件索引

| 文件 | 内容 |
|---|---|
| `metrics-prefill-pr.csv` | PR 单流四档 + TTFT（含 E5 复查） |
| `metrics-concurrency.csv` | 并发 C6/C12（prefill 聚合）+ 官方 decode-only C1/C4/C8/C12 |
| `metrics-decode.csv` | DE step_eff + 官方单流/Agent decode-only + W4A16 对照 |
| `metrics-resources.csv` | weight/KV/内存 + P0 拆账 + 400K 明细 |

---

## 5. 引用来源（链接不复制）

- 主报告：`luz031-deployment-2026-08-23.md`（采纳验收）、`luz031-bench-and-p0-exec-report-2026-08-23.md`（Session A+B）、`w4a4-vs-w4a16-diff-audit-2026-08-23.md`（差异核对）、`arstall-production-closure-2026-08-23.md`（E5 复查）、`g1-production-restore-2026-08-24.md`（G1 对照）、`issue22-400k-verification-2026-08-24.md`（400K）
- 数据：`_luz031_official_bench/data/`（luz031_汇总、luz031_vs_official、p0_accounting_data）、`_issue22_400k_verification_20260824/`
- 官方参考：`/tmp/_bench_luz031/official/benchmark_package_20260819/data/测试数据汇总.md`（8/19 原基座）
- 版本说明：`LuZ0.3.1-release-notes.md`（docs/）

*本汇总为只读汇编，所有数字取自上述报告原文；[推算] 类数字保留原报告口径标注。*
