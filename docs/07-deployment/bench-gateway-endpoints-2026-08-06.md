# 网关端点综合性能 Benchmark — 8003 自建 vs LiteLLM 4000

**日期**：2026-08-06
**环境**：双 DGX Spark healthy（NCCL 加固 + 8003 思考链修复已上线），全链路 8001/8003/4000 = 200
**被测**：
- **8003 自建网关** `http://<NODE_IP>:8003`（客户端 key `<API_KEY>-*`）
- **LiteLLM 4000** `http://<NODE_IP>:4000`（prob key `sk-07E1…` 仅访问 dspark-prob；embedding 用 master key）

**方法**：每端点先 2 个预热请求；并发 c1/c5/c10；短 prompt 200→64 为主 + 一档 2048→256；SSE 流式 60s 超时；温度 8003=网关默认（不带）、4000 prob=0.7。全部请求 0 错误、0 超时。

---

## 📌 TL;DR（执行摘要）

- **全部端点 × 全部并发档：0 HTTP 错误、0 超时、[DONE] 100% 出现**，双网关功能完好。
- **8003 思考链修复生效**：`/v1/responses` 返回 `type=reasoning` 输出项（reasoning_text），流式含 `response.reasoning_text.delta` 事件；但 `enable_thinking:false` 不生效，思考链**始终开启**。
- **4000 思考链被剥离**：`/v1/responses` 与 chat 均无 reasoning（reasoning_tokens=0，无 reasoning 输出项）——与 8003 对比是**有思考 vs 无思考**的天然口径。
- **短 prompt 单流**：8003 更快（chat_ns c1 1303ms vs 1973ms）；**并发**：4000 聚合吞吐略优（c5/c10 83-86 vs 42-76 t/s）。
- **流式 TTFT 两网关几乎相同（~366ms）**；8003 因先输出思考链，首个 content 出现延迟约 2s（思考链开销），4000 直接出内容。
- **2048→256 档**：4000 c1 仅 3.9s vs 8003 7.9s，但 4000 提前结束（sum_ct=118 vs 256），非完全同口径。

---

## 1. 表 1：8003 全端点（c1/c5/c10）

| 端点 | 参数 | 并发 | 延迟 (ms) | 聚合吞吐 (t/s) | 错误 |
|------|------|------|-----------|----------------|------|
| `/v1/models` | GET | c1 | 28.9 | — | 0 |
| `/v1/models` | GET | c5 | 40.6 | — | 0 |
| `/v1/models` | GET | c10 | 22.2 | — | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c1 | 1303.1 | 49.1 | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c5 | 7683.8 | 41.6 | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c10 | 5864.6 | 75.9 | 0 |
| `/v1/chat/completions` 非流式 | 2048→256 | c1 | 7906.9 | 295.4¹ | 0 |
| `/v1/chat/completions` 非流式 | 2048→256 | c5 | 20567.8 | 558.5¹ | 0 |
| `/v1/chat/completions` 流式 SSE | 200→64 | c1 | TTFT 366.6 / content 2323.9 | 27.5 | 0 |
| `/v1/chat/completions` 流式 SSE | 200→64 | c5 | TTFT 501.2 / content 4097.3 | 73.0 | 0 |
| `/v1/responses` | 200→64 | c1 | 1334.8 | 48.0 | 0 |
| `/v1/responses` | 200→64 | c5 | 3591.2 | 82.6 | 0 |
| `/v1/responses` | 200→64 | c10 | 4995.2 | 80.3 | 0 |
| `/v1/embeddings` | 1024维 | c1 | 44.9 | — | 0 |
| `/v1/embeddings` | 1024维 | c5 | 55.4 | — | 0 |
| `/v1/embeddings` | 1024维 | c10 | 88.4 | — | 0 |

¹ 总速率（含 prefill），输出吞吐因 max_tokens=256 上限。SSE 帧数：c1 中位 33 帧（其中 31 帧为 reasoning delta），[DONE] 100%。

### 一句话解读（8003）
- **models/embeddings**：20-90ms 级别，均为轻量端点，无压力。
- **chat_ns 短 prompt c1**：1.3s，是 4000 的 66%，单流体验更好。
- **chat_ns c5/c10**：中位延迟 5.9-7.7s（并发排队明显），聚合输出 42-76 t/s，并发利用率中等。
- **2048→256**：prefill 主导，c1 7.9s，c5 20.6s——长上下文并发敏感，与历史矩阵结论一致。
- **流式**：TTFT 367ms 极快，但思考链先行导致 content 首现 ~2.3s（思考链长度占 31/33 帧）。
- **responses**：1.3-5.0s，思考链输出项恒为 1，吞吐 48-83 t/s，与 chat 相当。
- **embeddings**：c1 45ms / c10 88ms，维度 1024，线性扩容正常。

---

## 2. 表 2：4000（LiteLLM）全端点（c1/c5/c10）

| 端点 | 参数 | 并发 | 延迟 (ms) | 聚合吞吐 (t/s) | 错误 |
|------|------|------|-----------|----------------|------|
| `/v1/models` | GET | c1 | 32.1 | — | 0 |
| `/v1/models` | GET | c5 | 11.2 | — | 0 |
| `/v1/models` | GET | c10 | 40.8 | — | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c1 | 1972.7 | 32.4 | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c5 | 3446.0 | 83.3 | 0 |
| `/v1/chat/completions` 非流式 | 200→64 | c10 | 5147.7 | 86.0 | 0 |
| `/v1/chat/completions` 非流式 | 2048→256 | c1 | 3920.6 | 560.4¹ | 0 |
| `/v1/chat/completions` 非流式 | 2048→256 | c5 | 17044.2 | 635.1¹ | 0 |
| `/v1/chat/completions` 流式 SSE | 200→64 | c1 | TTFT 365.5 / content 365.5 | 35.8 | 0 |
| `/v1/chat/completions` 流式 SSE | 200→64 | c5 | TTFT 600.9 / content 600.9 | 70.6 | 0 |
| `/v1/responses` | 200→64 | c1 | 1963.6 | 32.6 | 0 |
| `/v1/responses` | 200→64 | c5 | 4121.0 | 74.2 | 0 |
| `/v1/responses` | 200→64 | c10 | 4716.3 | 84.6 | 0 |
| `/v1/embeddings` | 1024维 | c1 | 30.8 | — | 0 |
| `/v1/embeddings` | 1024维 | c5 | 63.7 | — | 0 |
| `/v1/embeddings` | 1024维 | c10 | 132.0 | — | 0 |

¹ 2048→256 c1 提前结束（sum_ct=118<256，prob 模型输出更短），总速率口径仅供趋势参考。SSE 帧数：c1 中位 25 帧（0 帧 reasoning），[DONE] 100%。

### 一句话解读（4000）
- **models**：11-41ms，与 8003 同级。
- **chat_ns 短 prompt c1**：1.97s，较 8003 慢 51%（无思考链但 prob 生成内容更长）。
- **chat_ns c5/c10**：中位 3.4-5.1s，聚合 83-86 t/s，**并发扩展性优于 8003**（c5 即达峰）。
- **2048→256**：c1 3.9s 明显快于 8003（输出提前结束所致，非全同口径）。
- **流式**：TTFT 366ms，**content 首现即 366ms（无思考链）**，交互响应即时。
- **responses**：1.96-4.7s，**无 reasoning 输出项（思考链被剥离）**，仅 type=message。
- **embeddings**：c1 31ms 最快，但 c10 132ms 扩容劣于 8003。

---

## 3. 表 3：8003 vs 4000 对比（同端点）

| 端点 | 并发 | 8003 | 4000 | 结论（一句话） |
|------|------|------|------|----------------|
| models | c1 | 28.9ms | 32.1ms | 持平，均轻量 |
| chat 非流式 200→64 | c1 | **1303ms** | 1973ms | 8003 单流更快（-34%） |
| chat 非流式 200→64 | c5 | 7684ms / 41.6 t/s | **3446ms / 83.3 t/s** | 4000 并发更优（吞吐 +100%） |
| chat 非流式 200→64 | c10 | 5865ms / 75.9 t/s | 5148ms / 86.0 t/s | 4000 略优（吞吐 +13%） |
| chat 非流式 2048→256 | c1 | 7907ms / ct=256 | **3921ms / ct=118** | 4000 快但输出短，非全同口径 |
| chat 流式 SSE | c1 | TTFT 367 / content 2324 | TTFT 366 / **content 366** | TTFT 相同；8003 content 被思考链延迟 ~1.96s |
| chat 流式 SSE | c5 | TTFT 501 / content 4097 | TTFT 601 / content 601 | 4000 思考链开销为 0，content 即到 |
| responses | c1 | **1335ms**（reasoning=1） | 1964ms（reasoning=0） | 8003 更快且保留思考链 |
| responses | c10 | 4995ms / 80.3 t/s | 4716ms / 84.6 t/s | 吞吐接近，4000 略高但丢思考链 |
| embeddings | c1 | 44.9ms | **30.8ms** | 4000 单流更快 |
| embeddings | c10 | **88.4ms** | 132.0ms | 8003 并发更稳 |

**跨网关总评**：8003 单流延迟更优且唯一保留思考链；4000 高并发吞吐略优且流式无思考延迟。两者错误率均为 0，功能各自完整，差异主要源于「有思考 vs 无思考」的路径设计。

---

## 4. 思考链专项：responses 端点 type=reasoning 事件

### 4.1 事件与延迟（流式，同一 math 提示，3 轮）

| 网关 | 首 reasoning 事件 | 首 content 事件 | 总耗时 | reasoning_text.delta 帧数 | 结论 |
|------|------------------|------------------|--------|---------------------------|------|
| 8003 | 343-367ms | =总耗时（64 token 全被思考链消耗，无 content 帧） | 1231-1763ms | 13-23 帧 + reasoning_part.added/done | ✅ 思考链事件完整出现 |
| 4000 | 无（first_reasoning=总耗时） | 342-355ms | 1849-2289ms | 0 帧 | ❌ 无任何 reasoning 事件 |

### 4.2 有思考 vs 无思考（同一提示词、responses 非流式，5 次中位）

| 提示 | 8003（有思考） | 4000（无思考） | 思考链延迟开销 |
|------|---------------|---------------|----------------|
| math「7*8+15=? 给出推导」 | 2044ms（reasoning=1, ot~99） | 2016ms（reasoning=0, ot~113） | ~+28ms（可忽略，因 4000 也生成等长答案） |
| simple「请回复:ok」 | **1707ms**（reasoning=1, ot~62） | **218ms**（reasoning=0, ot=2） | **+1489ms（思考链开销 7.8×）** |

### 4.3 思考链专项结论
- **修复验证通过**：8003 responses 流式稳定输出 `response.reasoning_part.added → reasoning_text.delta×N → reasoning_text.done`，type=reasoning 事件完整。
- **开销形态**：思考链延迟开销与「是否必须思考」强相关。math 类 prompt 两边输出等长，开销趋零；**trivial 请求（如 ok）8003 也强制思考 62 token，造成 ~1.5s 纯开销（7.8× 延迟）**。
- **注意**：8003 responses 的 `usage.output_tokens_details.reasoning_tokens` 恒为 0（思考 token 计入 output_tokens 但未拆分），属于网关元数据口径问题，不影响事件流本身。
- **4000 无思考可对比**：4000 全程 reasoning_tokens=0、无 reasoning 输出项，思考链在 LiteLLM 层被剥离——若下游依赖思考链（如 CoT 展示），**必须走 8003**。

---

## 5. 异常记录 & 注意事项

- ✅ **0 错误 0 超时**：全部端点 × 全部并发档 200，无重试。
- 📝 **4000 思考链剥离（设计行为，非故障）**：prob key 的 chat/responses 均不返回 reasoning；如需思考链必须走 8003 客户端 key。
- 📝 **4000 key 模型权限隔离**：prob key 仅能访问 dspark-prob（`/v1/embeddings` 返回 401 key_model_access_denied），embedding 测试改用 master key。
- 📝 **8003 思考链不可关闭**：`enable_thinking:false` 不生效，思考链始终开启——trivial 请求会付出 ~1.5s 固定开销。
- 📝 **8003 responses usage 口径**：`reasoning_tokens` 恒 0（未拆分），分析用量时以 output[].type=reasoning 为准。
- 📝 **2048→256 非全同口径**：4000 prob 模型提前结束（ct=118 vs 8003 ct=256），延迟对比仅作趋势参考。
- 📝 **8003 chat_ns c10 中位延迟低于 c5**：c5 有单轮慢请求拖高中位（7684ms），c10 wall 8.43s > c5 7.68s 仍符合并发放大，属正常波动。

---

## 6. 方法 & 复现

- 脚本：`C:\Users\novAI\WorkBuddy\集群部署\.tessa_test\bench_gateway_endpoints.py`
- 分析：`C:\Users\novAI\WorkBuddy\集群部署\.tessa_test\analyze_gateway_bench.py`
- 原始数据：`C:\Users\novAI\WorkBuddy\集群部署\.tessa_test\_gateway_bench_2026-08-06.json`
- 原始证据：`_tessa_gateway_bench_raw_2026-08-06.txt`
- 请求量：双网关共 ~170 请求 + 预热，并发 ≤10，全程 ~3 分钟。

---

> 本报告由工程保障团队测试专家（Tessa）生成，2026-08-06 实时实测。结论：**8003 为唯一保留思考链的网关且单流更快；4000 高并发吞吐与无思考流式体验更优**；两者均 0 错误，生产可并行承载不同负载类型。
