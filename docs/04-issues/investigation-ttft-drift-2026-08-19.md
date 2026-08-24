# vLLM TTFT "越用越慢"调查报告

- **日期**：2026-08-19
- **环境**：DGX Spark TP4（vLLM 0.26.1 定制版 / anemll，DSpark + sparse_mla），容器 vllm-tp4-rank* 连续运行 18h+
- **报告人**：Infrastructure Operations Expert

---

## 一、直接结论

**当前稳态实测 TTFT 健康且稳定（84ms ±2ms，连续 6 轮无爬升）。** 通过 18 小时监控时间序列 + 实机打点，**未发现任何"越用越慢"的单调累积机制**。用户观察到的变慢，绝大多数来自**请求本身特征的变化**（prompt 更长 / prefix 未命中需全量 prefill），以及少量**推理期间偶发 JIT 编译尖峰**——两者都与"运行时长"无关，可通过 warmup 覆盖与负载口径区分。

---

## 二、数据证据（Prometheus 18h 序列 + 实机实测）

### 2.1 当前实测 TTFT（01 上连续 6 轮短请求 streaming 首 token）
```
round1: 110ms  round2: 84ms  round3: 82ms
round4: 85ms   round5: 84ms  round6: 83ms
```
→ 84ms 稳定，无递增趋势。

### 2.2 经典累积退化指标：全部健康 ⚪
| 指标 | 18h 趋势 | 判定 |
|---|---|---|
| `kv_cache_usage_perc` | **全程 0-8%** | ⚪ 远未顶满，无碎片压力 |
| `num_requests_waiting_by_reason`(capacity/deferred) | **全 0** | ⚪ 无调度队列堆积 |
| `num_preemptions_total` | **全 0** | ⚪ 无抢占 |
| prefix cache 命中率 | 业务时段 **90%+**，低峰无请求 | ⚪ 缓存工作正常 |
| 容器内存 | 2.99GiB（121.6GiB 配额），稳定 | ⚪ 无泄漏放大 |
| 容器 CPU | 2.12% | ⚪ 无异常 |
| e2e latency | 波动 55-2674ms，**23:55 最低 55ms** | ⚪ 无累积上升 |

→ 排除了**显存/KV 顶满、调度排队、抢占、prefix 缓存退化、内存泄漏**这几类最典型的"越跑越慢"根因。

### 2.3 平均 prefill 时长：有波动、非单调递增
```
01:53 24ms  04:53 315ms  05:53 1216ms  06:53 1331ms
08:53 147ms 10:53 904ms  12:53 30ms    15:53 90ms
23:53 17ms
```
对应 KV computed tokens：波动时段多为 1K（长 prompt / 未命中 prefill 全量计算），低时段 <1K（短查询 / 缓存命中走快速路径）。**波动由请求长短驱动，不随运行时长上升。**

### 2.4 ⚠️ 唯一确凿的"偶发变慢"信号：推理期间 JIT 编译
日志（vllm-tp4-rank1，18h 内）发现 **4 次 "JIT compilation during inference... causes a latency spike"**：
```
06:08:22 TileLang mhc_pre_big_fuse_broadcast_with_norm_tilelang
06:08:22 TileLang mhc_pre_big_fuse_with_norm_tilelang
06:08:30 Triton  _compute_global_topk_indices_and_lens_kernel
06:16:17 Triton  _prepare_dflash_inputs_kernel
13:54:38 Triton  _compute_prefill_metadata_kernel
```
这些是**首次遇到未覆盖 shape 时的 kernel 动态编译**，会产生一次性 TTFT 尖峰（秒级）。后续同 shape 复用已编译 kernel 则正常。

---

## 三、根因判定

| 用户感知 | 实际机制 | 与运行时长的关系 |
|---|---|---|
| "越用越久 TTFT 变长" | **请求变长**（prompt/上下文更长）→ prefill 需算的 KV 更多 | ❌ 无关，是负载特征 |
| | **prefix 缓存未命中**（新前缀/冷缓存时段）→ 全量 prefill | ❌ 无关，是缓存复用率 |
| | **偶发 JIT 尖峰**（新 shape 首次触发 Triton/TileLang 编译） | ⚠️ 弱相关，随使用遇到更多 shape 而增多 |
| 短请求变慢 | **当前稳态无此现象**（实测 84ms 稳定） | — |

**结论**：不存在"运行时长短 *正相关* 的 TTFT 漂移"。若业务侧感知变慢，优先怀疑**请求输入变长 / 并发上来**或**命中冷前缀**，其次偶发 JIT。

---

## 四、建议动作（按 ROI）

1. **P0（零成本，诊断口径）**：业务侧感知模型 + 打点对齐。在网关（8003 responses_gateway / liteLLM）加 request 记账：记录每条请求 `prompt_tokens` + `TTFT`。只有同 prompt 长度、同缓存命中状态下对比，才能区分"引擎变慢"与"请求变长"。
2. **P1（低风险，消 JIT 尖峰）**：扩展 warmup 覆盖更多 shape。日志已明示"consider extending warmup to cover this shape/config"。把 **长 prompt / 大 batch / 不同 seq-len 档** 加进 warmup 预编译范围，可消除大部分推理期 JIT 尖峰（消除秒级 TTFT spike 的主要来源）。
3. **P2（观察）**：sparse MLA / DSpark speculator 的 autotune cache 已命中（/root/.cache/vllm/flashinfer_autotune_cache/0.6.15/121a/...），确认该缓存持久化、重启不被清，避免每次重启重新 autotune。
4. **不做**：性能调参 / 显存扩容（KV 仅用 8%，无意义）；重启（无累积漂移可清除）。

---

## 五、附带风险提示

- 容器**已连续运行 18h+ 且 Docker 日志仅 235 行**（stdout 无滚动压力），无告警。可继续跑，无强制重启需求。
- `kv_cache_usage_perc` 全程 ≤8% 说明**生产负载极低**（每 2h 前缀查询百万级但 KV 占用极小）→ 若后续要升负载，余量充足。

---

*数据源：Prometheus（02:8191，`{job=\"vllm\"}`）+ vllm-tp4-rank0/1 实机打点与 Docker 日志。*
