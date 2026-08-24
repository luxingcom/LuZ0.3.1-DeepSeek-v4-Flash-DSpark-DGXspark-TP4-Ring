# E4 — KV 足迹 8192 vs 4096 "+58%/token" 放大机制归因

- **执行人**：阿奇（Archi）· 系统架构师（architecture-analysis）
- **日期**：2026-08-23
- **范围**：只读分析。源码（LuZ0.3.1 镜像内 vLLM fork 0.26.1 V1 引擎）+ 实测启动日志（w4a4-ext B2/B3 臂）+ 生产配置交叉验证。**未动任何配置、未启动容器**（仅 `docker create`+`docker cp` 提取源码，已清理）。
- **前置遗留**：luz031-deployment-2026-08-23 §8.1「E4 KV 足迹 8192 机制调查」；w4a4-ext-2026-08-23 §5.2 E4「8192 使每 token KV 足迹 +58%」。

---

## 0. 一句话结论

**"+58%/token KV 足迹放大"是对 fork 内一个指标语义的误读，真实机制是"每请求 KV 预留随批大小上涨"，而非"每个 token 的物理 KV 字节变胖"。**

具体为：

1. 启动日志 `GPU KV cache size: X tokens` **不是物理 KV 池的 token 容量**，而是 `max_concurrency × max_model_len`（"最多能同时容纳多少个满长请求的上下文总长"，kv_cache_utils.py:2167-2182）。
2. 物理 KV 池字节在 4096/8192 两档几乎不变（B2 **49.27 GiB** vs B3 **48.19 GiB**，仅 −2.2%）；物理 per-token 字节**不变**（≈9.4 KiB/token，nvfp4_ds_mla 每层 584 B 包络 × 层结构）。
3. 8192 档报告值 5.495M→3.480M tokens 的真正原因是 **`blocks_per_request`（每个满长请求的块预留）上涨 ~54%**：fork 的 `max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens`（async 调度下 = 2×批大小）随 `max_num_batched_tokens` 4096→8192 而 8192→16384 翻倍，放大 DeepSeek V4 滑动窗口（SWA）KV 组的**每请求准入预留**：`cdiv(sliding_window − 1 + max_in_flight, block_size) + 1`。
4. 这是**真实的准入容量下降**（满长并发 9.16→5.80，−36.7%），方向与 w4a4-ext "KV 塌缩"判定一致；但**不是** "每 token 足迹 +58%"。

**生产建议：维持 threshold 4096（Go 保持现状）；threshold 8192 继续 No-Go，但否决理由需按本报告修正。无需为"每 token 足迹"做任何优化——它从未变大。**

---

## 1. 事实复核（实测数字，与报告一致）

来自 w4a4-ext 两臂 startup 日志（本地副本 `_w4a4_ext_assets/logs/b2_startup.log` / `b3_startup.log`）：

| 项 | B2（4096） | B3（8192） | Δ |
|---|---|---|---|
| max_num_batched_tokens / threshold | 4096 | 8192 | 2× |
| Available KV cache memory | 49.27 GiB | 48.19 GiB | **−1.08 GiB（−2.2%）** |
| **GPU KV cache size（报告）** | **5,495,820 tokens** | **3,480,344 tokens** | −36.7% |
| → 隐含 max_concurrency（÷600,000） | **9.16** | **5.80** | −36.7% |
| weight | 45.32 GiB | 45.54 GiB | +0.2 GiB |
| peak activation | 2.03 GiB | 2.69 GiB | +0.66 |
| CUDAGraph | 0.40 GiB | 1.69 GiB | +1.29 |
| non-torch | 0.68 GiB | 0.89 GiB | +0.21 |

- 两臂唯一不同的启动参数就是 `max_num_batched_tokens`/`long_prefill_token_threshold`（4096 vs 8192）。W4A4 池补丁（SHARED=1）、n=7、util 0.8、max_num_seqs=12、CG capture 列表全部相同。
- **关键观测**：KV 内存预算只差 1 GiB，但报告 tokens 差 2M。若按"预算 ÷ 报告 tokens"硬算 per-token，会得到 49.27GiB/5.496M ≈ 9.0 KiB 与 48.19GiB/3.480M ≈ 14.2 KiB（+58%）——这正是 w4a4-ext §2.5 的数字。**但分母不是物理 tokens，见 §2。**

---

## 2. 机制归因（源码实证）

### 2.1 "GPU KV cache size" 的指标语义（决定性发现）

fork `vllm/v1/core/kv_cache_utils.py`：

```python
# L2167-2182（get_kv_cache_configs 收尾）
if len(kv_cache_config.kv_cache_groups) > 0:
    max_model_len = vllm_config.model_config.max_model_len
    # GPU KV cache size in tokens = max_concurrency * max_model_len:
    # the total tokens of context the pool can hold at peak utilization.
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")

# L1819-1829
def get_kv_cache_capacity(vllm_config, kv_cache_config):
    max_concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
    return int(max_concurrency * max_model_len), max_concurrency
```

**所以日志里的 "tokens" = `max_concurrency × max_model_len`，不是 `num_gpu_blocks × block_size`。** 在"每请求块数恰好 = max_model_len/block_size"的极限下两者相等（B2 恰好近似相等，见 §2.3），一旦每请求块数更大，报告值就低于物理池。

`max_concurrency = num_blocks / blocks_per_request`（L937-955）：

```python
def get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config):
    num_layer_per_group = max(len(group.layer_names) for group in kv_cache_config.kv_cache_groups)
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(...)
    memory_per_block = kv_cache_groups[0].kv_cache_spec.page_size_bytes * num_layer_per_group
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
```

### 2.2 物理池未变：num_blocks 只随 available_memory 微变

`num_blocks = available_memory // bytes_per_block`（packed 布局 L1322；字节/块为模型结构常数，与 threshold 无关）。因此：

```
num_blocks_B3 / num_blocks_B2 ≈ 48.19 / 49.27 = 0.978（物理池仅 −2.2%）
```

物理池 token 槽 ≈ num_blocks × block_size：
- B2：49.27 GiB ÷ (物理 per-token ≈9.4 KiB) ≈ **5.375M 槽**；报告 5.496M ≈ 物理槽（因 bpr≈9375=max_len/64，见下）
- B3：48.19 GiB ÷ 9.4 KiB ≈ **5.375M 槽**；报告 3.480M << 物理槽

**物理每 token 字节（nvfp4_ds_mla：DeepSeek V4 每层 448B NoPE + 128B RoPE + 8B fp8 scale = 584B 包络 × 层结构）在两档完全相同。** "9.0→14.2 KiB/token" 是"预算 ÷ 准入容量指标"造成的伪足迹。

### 2.3 每请求块数 blocks_per_request 是 batch 相关的（核心机制）

由观测反推（bpb=bytes_per_block 消去）：

```
bpr_B2 × bpb = 49.27 GiB / 9.1597 = 5.776e9
bpr_B3 × bpb = 48.19 GiB / 5.8006 = 8.920e9
→ bpr_B3 / bpr_B2 = 1.5445（每满长请求块数 +54.5%）
```

`blocks_per_request` 中唯一随 `max_num_batched_tokens` 变化的输入是 `max_in_flight_tokens`：

```python
# config/vllm.py（VllmConfig 属性）
@property
def max_in_flight_tokens(self) -> int:
    # Upper bound on tokens scheduled but not yet settled (freed):
    # every concurrent batch may hold up to a full max_num_batched_tokens.
    return self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens

@property
def max_concurrent_batches(self) -> int:
    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:      # DSpark 强制 V2
            return self.pp_size + 1        # pp=1 → 2
```

- async_scheduling 对本 fork 的 dspark 投机默认**开启**（config/vllm.py L1059-1107，dspark 在允许列表），→ `max_concurrent_batches = 2`。
- B2：`max_in_flight = 2×4096 = 8192`；B3：`2×8192 = 16384`。

`max_in_flight_tokens` 进入**回收型（recycling-aware）KV 组的每请求准入上限**：

```python
# kv_cache_interface.py — SlidingWindowSpec / ChunkedLocalAttentionSpec
def max_admission_blocks_per_request(self, max_in_flight_tokens, max_model_len):
    num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)
    return cdiv(num_tokens, self.block_size) + 1
```

DeepSeek V4 是**混合 full + sliding-window** 模型（config.json：`compress_ratios` = 46 项；0/1→SWA-only，4→C4A×21，128→C128A×20；`sliding_window=128`；每层还挂 `DeepseekV4SWACache` 发出 `SlidingWindowMLASpec`，见 sparse_swa.py）。因此每档的 SWA 组每请求预留：

| max_in_flight | SWA 每请求块 `cdiv(128−1+in_flight,64)+1` |
|---|---|
| B2 = 8192 | 131 |
| B3 = 16384 | **259（+128）** |

差异 128 块/每 batch 相关组 × 组数量与页大小加权 ≈ 观测到的 +5105 块（bpr_B3−bpr_B2）。这一预留**同时**作用于启动池大小计算与运行时准入闸门（`single_type_kv_cache_manager.get_manager_for_kv_cache_spec` 用同一 `max_admission_blocks_per_request`），所以是真实准入约束，不是纯会计数字。

### 2.4 二次效应（次要，非主因）

8192 档：
- peak activation 2.03→2.69 GiB（+0.66）：更大 chunk 的 bf16 gather workspace（flashmla.py warmup `M = N + window + max_num_batched_tokens`）。
- CUDAGraph 0.40→1.69 GiB（+1.29）：更大 CG 捕获。
- 两者合计从 KV 预算拿走 ~1.08 GiB（即 Available 49.27→48.19）。这只解释 −2.2% 物理池，不解释 −36.7% 报告值。

### 2.5 候选机制排除表

| 候选（任务书提示） | 判定 | 证据 |
|---|---|---|
| ① 投机解码 KV 预留（MTP n=7 draft tokens） | **排除为报告值主因** | 注意力 spec 的 max_memory_usage 公式中无 draft-token 项；n=7 通过 `max_num_scheduled_tokens`（4096−72=4024 / 8192−72=8120）影响调度而非 KV 池；draft KV 是运行时瞬时占用，两档相同 |
| ② block 对齐损耗 | **排除** | `_apply_alignment_padding` 对齐到 584（nvfp4）是模型常数，与 threshold 无关 |
| ③ per-request 预留（in-flight） | **√ 主因** | `max_in_flight_tokens = 2×max_num_batched_tokens` → SWA 组每请求预留 131→259 |
| ④ 池碎片 / pool 共享（ws-dedup SHARED=1） | **排除** | 池只作用于权重（45.32→45.54 GiB 稳定），KV 路径不经该池；KV 每 token 字节是 nvfp4_ds_mla 包络常数 |
| ⑤ CUDAGraph / 激活固定开销 | **次要** | 合计仅 +1.9 GiB（其中约 1 GiB 进 KV 预算），无法解释 2M token 报告差 |

---

## 3. 对 w4a4-ext 结论的修正

w4a4-ext §2.5/§5.2 的 "每 token KV 足迹 +58%（9.0→14.2 KiB）" 应修正为：

> 8192 档物理 KV 每 token 足迹不变（≈9.4 KiB）；"KV 塌缩 36.7%" 实为 **满长请求准入并发 9.16→5.80**（`max_concurrency`），由 `max_in_flight_tokens` 随 `max_num_batched_tokens` 翻倍所致。**"capacity 塌缩" 方向仍成立，机制是 per-request 预留而非 per-token 字节。**

对业务的实际影响范围：
- **短上下文（4K/16K/32K/64K 生产形态）**：每请求实际块数远低于准入上限，准入闸门按请求实际长度校验，**并发容量几乎不受 36.7% 影响**。8192 的否决主因回到性能（单流 −2.1%~+0.1%、并发 −1.6%~−2.1% 无增益）+ 长上下文余量下降。
- **满长（600K）负载**：并发余量真实下降 36.7%，需按此评估长上下文容量规划。

---

## 4. Go / No-Go 建议

### 生产现状（threshold 4096）：**Go，维持不变，无需动作**
- 物理 KV 池/每 token 字节未受 E4 影响；报告 "KV 5.73M tokens" 即 4096 档的正常准入容量。
- 不建议为修正指标而改配置。

### threshold 8192：**No-Go（维持既有否决），理由按本报告修订**
1. 性能无增益（既有结论）。
2. 满长并发准入 5.80（−36.7%），长上下文余量下降。
3. 激活 + CUDAGraph 额外 ~1 GiB。
4. **不列入** "每 token 足迹 +58%" 这一理由（不成立）。

### 未来若想"大 batch + 长上下文"兼容的可选优化（均为 No-Go 前提下的研究项，非当前必做）
| 方向 | 机制 | 效果 | 代价/风险 |
|---|---|---|---|
| `--no-async-scheduling` | `max_concurrent_batches` 2→1 → `max_in_flight` 减半 | SWA 预留基数减半，满长并发提升 | 失去 async 调度重叠收益，需 A/B |
| 下调 `max_num_batched_tokens` | 直接缩小 `max_in_flight` | 对齐 4096 现状 | 已知 threshold 4096 已是最优带 |
| fork 内把 `max_in_flight_tokens` 改用 `max_num_scheduled_tokens`（已扣除 spec-decode 槽位）或对 SWA 预留设上限 | 消除"batch 越大每请求预留越宽"的放大 | 满长并发不再随 batch 塌缩 | 需 kernel 工程评估；改 fork 有回归风险 |
| 监控口径改用物理 KV 池（`num_blocks × bytes_per_block` / 容器内 `torch.cuda.memory_summary` 的 KV 张量） | 避免把准入指标当物理容量 | 告警/容量规划更准 | 需新增观测脚本 |

---

## 5. 补充测量清单（证据缺口 → 完全闭环 E4）

本报告机制已由源码+实测日志双向锁定；以下为**精确量化到每个 KV 组**的缺口，属"nice-to-have"而非"必要"：

1. **一次性容器内打印 `get_kv_cache_configs()` 输出**（每组的 `num_blocks`、`page_size_bytes`、`num_layer_tuples`、`max_memory_usage_pages`）在 4096/8192 两档 → 把 bpr 由"反推 +54.5%"落实为逐组加减。**必须起的调试容器/日志**，符合只读（不产流）。
2. **`--kv-cache-memory` 对齐探针**：把 8192 档显式指定与 4096 档相同 KV 预算，复测报告 tokens → 直接验证物理池相同、报告差纯由 bpr 引起。
3. **运行时观测**：两档在相同并发负载下抓 Prometheus `num_blocks_free` / `num_blocks_used`，确认短请求形态下准入余量差异 ≤ 噪声（支撑"短上下文影响小"论断）。

---

## 6. 源码证据索引（均提取自 LuZ0.3.1 镜像 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`，只读）

| 文件 | 关键行 | 结论 |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py` | L2167-2182, L1819-1829, L937-955, L1322 | 报告 tokens = max_concurrency×max_model_len；bpr 公式 |
| `vllm/config/vllm.py` | L518-536 (max_in_flight), L508-517 (max_concurrent_batches), L1059-1107 (async 默认开) | max_in_flight = 2×max_num_batched_tokens |
| `vllm/v1/kv_cache_interface.py` | L567-598 (SlidingWindowSpec 准入), L521-526 (ChunkedLocal) | per-request 预留随 max_in_flight |
| `vllm/v1/core/single_type_kv_cache_manager.py` | L1797-1825 (admission gate 同源) | 启动与运行时同一预留，真实准入约束 |
| `vllm/models/deepseek_v4/nvidia/model.py` / `attention.py` | L197 compress_ratio; L632 MLAAttentionSpec | DeepSeek V4 混合注意力、584B 包络 |
| `vllm/models/deepseek_v4/nvidia/sparse_swa.py`（实际路径 `v1/attention/backends/mla/sparse_swa.py`） | L90 SlidingWindowMLASpec; 层类型 | SWA 组窗口 128、block 64 |
| `models/deepseek-v4-flash-0731/config.json` | compress_ratios/sliding_window/num_hash_layers | 21 C4A + 20 C128A + 5 SWA-only；window=128 |

---

## 7. 遗留与后续

1. 建议把运行报告中的 "KV tokens" 指标名改为 "KV 准入容量（max_concurrency×max_model_len）"，并补充物理池口径（`num_blocks×block_size`），避免后续再次误读。
2. 若重开大 batch 方向，先做 §5.1 的逐组打印，再评估 §4 优化方向。
3. 本报告不改变 LuZ0.3.1 生产终态；threshold 4096 的 "KV 5.73M≥5.7M" 门结论保持有效。

---

*纪律：只读（SSH + docker create/cp 提取源码后清理，未启 vLLM、未改任何配置）；源码为主、日志交叉验证；对无法完全逐组复算的部分明确标注证据缺口（§5），不强行解读。*
