# vLLM V1 Chunked Prefill 排队/优先级问题：社区方案调查报告

**日期**：2026-08-09
**工作流**：社区技术调研（vLLM V1 调度器 prefill 阻塞 decode 的解决方案）
**关联**：review-mla-compression-decode-collapse-2026-08-09.md（已确认根因 = 引擎串行 + prefill 阻塞 decode）

---

## 📌 TL;DR（执行摘要）

- **问题普遍性确认**：这是 vLLM 社区广泛讨论的经典问题——"one long prompt stalls every decode"（单长 prompt 冻结所有 decode），有专门的技术文章、GitHub issue、官方文档覆盖
- **vLLM V1 原生已内置解决方案**：5 个调度参数即可缓解/解决，生产 0.26.1 版本完全支持
- **生产配置缺口确认**：当前 A 组 TP2 启动参数**全部缺失**这 5 个参数（用默认值 1/1/0/fcfs）→ 正是 131K prefill 独占预算的根因
- **关键机制**：V1 调度器 Phase 1 先处理 running（decode），Phase 2 处理 waiting（prefill），但 `max_num_batched_tokens`（当前 2024，spec decoding 限制）被单个长 prefill 独占时，decode 请求仍会被 starve——需要 `max-num-partial-prefills`/`max-long-partial-prefills`/`long-prefill-token-threshold` 控制并发 prefill，防止 head-of-line blocking
- **priority 调度**：`--scheduling-policy priority` + 请求 `priority` 字段（PR #5958/#19057）可让交互请求优先于 batch
- **进阶方案**：DCP（decode context parallel）+ MTP（含 bug 修复）在 4×DGX Spark 上 128K 可达 23-24 t/s（社区实测，需 fork 分支）

---

## 1. 问题普遍性（社区证据）

| 来源 | 内容 |
|------|------|
| dev.to《Chunked Prefill: Why One Long Prompt Stalls Every Decode》 | **精确描述我们的场景**：单个 200k-token prefill 独占整个 prefill 预算，后面的 300-token 短请求要等完 100 个 chunk |
| vLLM 官方文档（Optimization and Tuning） | 确认 V1 chunked prefill 默认启用 + 调度策略**优先 decode**（先批所有 pending decode 再调度 prefill） |
| GitHub Issue #16528 | `--long-prefill-token-threshold` V1 实现 bug（已修），证明该参数是社区常用配置 |
| Gitee MindSpore Feature | 大并发长输入场景下 V1 调度优化请求（TTFT 瓶颈） |
| vLLM 内核探秘（juejin） | 确认 V1 统一 token 调度 + `num_computed_tokens` 断点续传实现 chunked prefill |

---

## 2. vLLM V1 调度机制（源码级确认）

### 2.1 两阶段调度（scheduler.py）

- **Phase 1**：处理 `self.running`（decode 请求），优先分配 token 预算
- **Phase 2**：处理 `self.waiting`（prefill 请求），用剩余预算

### 2.2 关键参数（官方文档 + API 参考确认）

| 参数 | 默认 | 作用 |
|------|------|------|
| `max_num_batched_tokens` | 2048（V1 自动推断） | 每步总 token 硬上限；spec decoding 时被限制（当前 2024） |
| `max_num_partial_prefills` | **1** | 并发 partial prefill 请求数上限 |
| `max_long_partial_prefills` | **1** | 长 prefill（>threshold）并发上限；**设小可让短请求插队** |
| `long_prefill_token_threshold` | **0（禁用）** | 判定"长"prefill 的阈值；>0 时启用并发 partial prefill |
| `scheduling_policy` | fcfs | `fcfs` 或 `priority`（lower=优先） |
| `enable_chunked_prefill` | True（V1 默认） | 长 prompt 自动分块 |

### 2.3 机制本质

```
问题: 长 prefill 独占 max_num_batched_tokens → 后续 decode/短请求被 head-of-line 阻塞
      （我们实测 131K c5: TTFT 70/142/215/287/358s, decode 被 prefill 排队饿死）

vLLM 默认 max_num_partial_prefills=1 → 同一时间只允许 1 个 partial prefill
      → 长 prefill 一旦开始就独占直到完成

修复: long_prefill_token_threshold>0 → 长 prefill 也分块且与其他 prefill 并行推进
      max_long_partial_prefills < max_num_partial_prefills → 短请求可插队长请求
```

---

## 3. 社区解决方案（按落地难度排序）

### 方案 A：启用并发 partial prefill（首选，官方原生，零侵入）

```bash
vllm serve /models ... \
  --max-num-batched-tokens 2048 \
  --max-num-partial-prefills 4 \
  --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 1024
```

- **效果**：最多 4 个 prefill 并发推进；长 prefill（>1024 token）最多 1 个并发，其余资源让给短请求
- **对应问题**：131K prefill 不再独占预算，decode/短请求可插入 chunk 间隙
- **参考**：vLLM CLI 文档 + dev.to 文章（最后 3 个 flag "matter more than they look"）

### 方案 B：priority 优先级调度（交互优先于 batch）

```bash
# 服务端启用
vllm serve /models ... --scheduling-policy priority

# 客户端（OpenAI 兼容 API，extra_body）
# priority 越小越优先；0=交互/加急, 10=batch, 20=后台
{"model":"...", "messages":[...], "priority": 10}
```

- **机制**（PR #5958 + #19057）：
  - waiting 队列按 priority 升序出队
  - 抢占时先移除 priority 最大（最低优先级）的请求
  - **注意**：priority≠0 时若未启用 priority 调度会报错
- **适用**：交互式请求（chat/agent）与批量任务（RAG 索引/离线）混跑时

### 方案 C：DCP + MTP（长 ctx 吞吐终极方案，需 fork 分支）

- **社区实测**：4×DGX Spark + 128K + DCP4 + MTP3 → **23-24 t/s**（Dre Dyson 实战文，修复前仅 14.6）
- 关键配置：`VLLM_DCP_SHARD_DRAFT=1`、`MAX_CUDAGRAPH_CAPTURE_SIZE=10`、`KZ_TRIM_AFTER_LOAD=1`、`NCCL_MAX/MIN_NCHANNELS=4`
- **⚠️ 有 bug**：V1 draft 路径缺少 `decode_context_parallel_size` 复制（PR #72 修复）——需 rebase `eldritch-head66` 分支
- **注意**：DCP 解决的是 decode 本身的长 ctx 吞吐（KV 跨节点分片），**与 prefill 排队问题正交**，可与方案 A/B 叠加

### 方案 D：业务侧缓解（无侵入，立即生效）

- ① 提升 prefix cache 命中率（RAG 前缀规范化/系统提示词固定）→ TTFT 降 10×（前报告已分析）
- ② 控制长 ctx 请求并发（max 并发或队列分层）
- ③ 拆分长文档为多个短请求

---

## 4. 生产落地建议（针对本集群）

### 4.1 当前配置缺口

```
A 组 TP2 启动命令（vllm-envE-node 实查）：
- max_num_batched_tokens = 2024（spec decoding 自动限制）
- max_num_partial_prefills   = 1（默认）
- max_long_partial_prefills  = 1（默认）
- long_prefill_token_threshold = 0（禁用）
- scheduling_policy = fcfs（默认）
→ 5 个参数全部默认，长 prefill 独占预算问题无防护
```

### 4.2 推荐变更（维护窗口执行）

```bash
# 追加到 TP2 启动命令：
--max-num-partial-prefills 4 \
--max-long-partial-prefills 1 \
--long-prefill-token-threshold 2048 \
# 可选（若交互/batch 混跑）：
--scheduling-policy priority
```

**预期效果**：131K 长 prefill 不再独占 2024 token 预算；并发 5 请求的 TTFT 从"排队 70-358s"变为"并行推进 + 短请求插队"；decode 请求获得稳定插槽。

### 4.3 验证方式

- 复测 B 组 131K c5：TTFT 分布应显著收窄（不再 1.00× 线性排队）
- 观察 `Prefix cache hit rate` 与 `GPU KV cache usage` 指标

---

## 5. 参考资源索引

| 资源 | 链接/位置 |
|------|----------|
| dev.to 长 prefill 阻塞文章 | dev.to/ji_ai/chunked-prefill-why-one-long-prompt-stalls-every-decode |
| vLLM 官方 Optimization | docs.vllm.ai → configuration/optimization（chunked prefill 节） |
| vLLM SchedulerConfig API | vllm/config/scheduler.py（max_num_partial_prefills 等） |
| DeepWiki Scheduler 分析 | deepwiki.com/lee20/vllm_v0.10.2/2.3-request-scheduler |
| Priority PR | github.com/vllm-project/vllm/pull/5958 + #19057 |
| long-prefill bug | github.com/vllm-project/vllm/issues/16528 |
| DCP+MTP 4×Spark 实战 | dredyson.com（DCP 修复文）+ PR #72 |
| 本集群 root cause | review-mla-compression-decode-collapse-2026-08-09.md |

---

## ⚠️ 局限

- 方案 C（DCP/MTP）需要 fork 分支与额外补丁，本集群若仅需解决 prefill 排队，方案 A 已足够
- 优先级语义注意：官方文档 lower=higher priority；个别中文教程写反（本报告以官方 API 参考为准）
- 具体参数值（partial=4/threshold=2048）为经验值，需在维护窗口实测调优
- vLLM 0.26.1.dev 版本较新，需确认这些参数在该版本的完整支持（建议启动时 `--help` 核对）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
