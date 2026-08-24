# MLA 压缩有效性 + decode 并发崩塌归因 复核报告

**日期**：2026-08-09
**工作流**：性能归因复核（基于 vLLM 日志 + 权重文件实读 + rows_B.csv 实测）
**参与成员**：主理人（复核与建模）
**前置**：analysis-tp2-tp4-communication-2026-08-09.md（原分析）、benchmark-B-group-2026-08-08.md（原归因）

---

## 📌 TL;DR（执行摘要）

- **MLA 压缩：真实启用且有效** ✅
  - `nvfp4_ds_mla` KV cache dtype 确认生效（vLLM 日志 "Using DeepSeek V4 padded nvfp4_ds_mla KV cache format"）
  - **KV cache 13.61 GiB / 2,129,743 tokens = 6.7 KB/token**（含 padded 存储）——比此前估算还低
  - DEEPSEEK_SPARSE_SWA + hash 层 + compress_ratios(4/128) 三重机制均激活
- **decode 并发崩塌：原"统一内存带宽饱和"归因错误** ❌
  - **决定性证据：c1 单流纯 decode 在所有 ctx（512→131K）恒定 70-76 t/s，完全不随序列长度下降**
  - 真实根因：**asyncio 引擎 prefill 完全串行** → 长 ctx prefill 超长耗时（131K 每请求 70s+）→ 并发下 decode 请求被 prefill 排队阻塞 → decode_tps 被 TTFT 稀释
  - 证据链：TTFT = 纯 prefill 时间（超出倍恒 1.00×）；prefill_tps 随 wave 从 1632→323（引擎 5 等分）
- **硬件带宽结论不变**：UMA 实测 206 GB/s，KV 读带宽 <10% UMA，200G 互联带宽余量充足

---

## 1. MLA 压缩有效性复核 ✅

### 1.1 三层压缩机制（全部生效）

| 机制 | 配置 | 证据 |
|------|------|------|
| KV cache dtype | `--kv-cache-dtype nvfp4_ds_mla` | 日志 "Using DeepSeek V4 padded nvfp4_ds_mla KV cache format" |
| 结构压缩 | `compress_ratios`（4/128 交替，层 2-41） | config.json 实读 |
| Sparse SWA | `Setting kv cache block size to 256 for DEEPSEEK_SPARSE_SWA` | 日志确认（hash 层稀疏） |

### 1.2 量化验证

```
KV cache 13.61 GiB / 2,129,743 tokens = 6.70 KB/token
```

- 相比标准 GQA（64 heads × 512 head_dim × 2B = 64KB/token），**压缩比 ~10×**
- 相比 MLA 理论 latent（(1024+512)/128 ≈ 12 dims × 2B ≈ 24B/token），padded 存储后 6.7KB 含稀疏 SWA 块结构开销
- **结论：压缩真实生效，KV 每 token 仅 6.7KB**

### 1.3 权重构成核查（附带发现）

磁盘权重 156GB 实读发现：
- **MoE 专家权重为 I8（INT8）**：35328 个 tensor，148GB（非 FP8）
- 专家 FFN 形状 `[2048,2048]`（等宽，3 矩阵 w1/w2/w3）
- attention 5.1GB（FP8）+ dense 1.16GB
- 运行态 vLLM 实际加载 79GB（进一步量化/稀疏化）

> 磁盘 156GB ≠ 运行 79GB，说明 dspark 稀疏化在加载时裁剪了大部分专家权重。

---

## 2. decode 并发崩塌归因复核 ❌（推翻原结论）

### 2.1 决定性证据：单流纯 decode 不随 ctx 下降

| ctx | 512 | 4K | 16K | 32K | 65K | 131K |
|-----|-----|-----|-----|-----|-----|------|
| **c1 decode t/s** | 71.1 | 74.0 | 72.9 | 70.0 | 76.5 | **73.8** |

→ **恒定 70-76 t/s，与序列长度无关** → 若为 KV 读带宽或 UMA 饱和，131K 单流必然显著下降

### 2.2 崩塌只发生在并发（c3/c5）

| ctx | c1 | c3 | c5 | 崩塌比 |
|-----|-----|-----|-----|--------|
| 4K | 74.0 | 36.9 | 26.8 | 2.8× |
| 32K | 70.0 | 19.9 | 9.8 | 7.1× |
| 131K | 73.8 | 6.8 | 4.3 | **17×** |

### 2.3 根因证据链：prefill 串行 + decode 阻塞

rows_B.csv 131K c5 coding 实测：

| wave | TTFT (s) | prefill_tps | 纯 prefill 时间 | 超出倍 |
|------|----------|-------------|----------------|--------|
| 1 | 70.9 | 1632 | 70.9s | **1.00×** |
| 2 | 142.2 | 814 | 142.2s | **1.00×** |
| 3 | 215.5 | 538 | 215.5s | **1.00×** |
| 4 | 286.9 | 404 | 286.9s | **1.00×** |
| 5 | 358.5 | 323 | 358.5s | **1.00×** |

- **TTFT = 纯 prefill 时间（1.00×）** → decode 请求必须等 prefill 完全结束
- **prefill_tps 从 1632 降到 323（5 等分）** → asyncio 引擎把 prefill 在 5 请求间串行调度
- **decode_tps = completion/(total-TTFT) 的分母被 TTFT 膨胀** → 平均值虚低

### 2.4 归因判定

| 候选归因 | 判定 |
|---------|------|
| ~~统一内存带宽饱和~~ | ❌ c1 单流 131K 恒定 74 t/s，排除 |
| ~~KV 读带宽~~ | ❌ KV 仅 6.7KB/token，131K@5 并发 19GB/s = 9% UMA |
| ~~200G 互联带宽~~ | ❌ 网络通信 <5% 链路能力 |
| **asyncio 引擎 prefill 串行** | ✅ **根因**：长 ctx prefill 70s+ × 并发排队 → decode 阻塞 |
| 引擎调度（prefill/decode 争用） | ✅ 次因：同一引擎线程处理 prefill 时 decode 无 slot |

---

## 3. 影响与建议

### 3.1 对已出报告的影响

- `analysis-tp2-tp4-communication-2026-08-09.md`：**互联带宽结论不受影响**（本就与 decode 崩塌归因解耦），但 §3.4 中"归因差异待复核"标注应更新为本报告结论
- `benchmark-B-group-2026-08-08.md`：**原"统一内存带宽饱和"归因需修订** → 改为"引擎串行 + prefill 阻塞 decode"

### 3.2 后续验证建议（如需 100% 坐实）

1. **vLLM metrics 复核**：`collect_detailed_traces`/`kv_cache_metrics` 开启后实测 KV 命中率与 per-step 耗时
2. **TP2 恢复后对照**：单请求无并发测 decode（排除引擎干扰），与 c1 数据对照
3. **chunked prefill 优化**：当前 `enable_chunked_prefill=True`，但长 ctx 仍整体串行——评估是否拆分 prefill 让 decode 插队（vLLM V1 调度器）
4. **引擎并行**：vLLM V1 的 `decode_context_parallel` 或独立 prefill/decode 引擎（评估中）

### 3.3 业务侧影响

- **长 ctx 并发场景的真实瓶颈 = 首 token 延迟（TTFT），非吞吐**：131K c5 时 5 请求 TTFT 依次 70/142/215/287/358s，体验极差
- **缓解方向**：
  - ① prefix caching 命中（共享前缀）→ TTFT 降 10×（前一报告已分析）
  - ② 降低并发或控制长 ctx 请求数量
  - ③ 升级 vLLM V1 调度（chunked prefill 让 decode 插队）

---

## 📚 数据来源

- vLLM 日志：vllm-envE-node（KV cache dtype、13.61GiB/2.13M tokens、sparse SWA、权重 79GB）
- 权重文件实读：model-00002-of-00048.safetensors（I8 专家、形状、dtype 分布）
- UMA 带宽实测：embed 容器 torch 测 205.9 GB/s
- benchmark 明细：_archive_scratch/bench_B/rows_B.csv（486 行，TTFT/prefill/decode 逐轮）
- 复核脚本：deliverables/engineering-assurance/tp_comm_analysis_v2.py（扩展）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
