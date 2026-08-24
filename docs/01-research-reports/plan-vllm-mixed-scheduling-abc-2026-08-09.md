# vLLM 混合业务调度（A/B/C 三方案）落地方案

**日期**：2026-08-09
**工作流**：部署变更（vLLM 调度优化落地）
**前置调研**：research-vllm-chunked-prefill-priority-2026-08-09.md、review-mla-compression-decode-collapse-2026-08-09.md
**范围**：A 组 TP2（node01+02，deepseek-v4-flash-0731，vLLM 0.26.1.dev）

---

## 📌 TL;DR（执行摘要）

- **目标**：业务混合跑（交互式请求 + 长 ctx RAG/batch），解决长 prefill 阻塞 decode + 长 ctx decode 吞吐
- **三方案组合落地**：
  - **A**：并发 partial prefill（`max_num_partial_prefills=4` + `max_long_partial_prefills=1` + `long_prefill_token_threshold=2048`）→ 消除长 prefill head-of-line 阻塞
  - **B**：priority 优先级调度（`scheduling_policy=priority` + 请求 `priority` 字段）→ 交互请求优先
  - **C**：DCP 长 ctx 加速（`decode_context_parallel_size=2`，分阶段验证）→ 长 ctx decode 吞吐
- **可行性已确认**：vLLM 0.26.1.dev **原生支持全部参数**（源码级验证，非推断）
- **⚠️ 关键风险**：DCP 与 dspark MTP 的组合兼容性**未经验证**（SpeculativeConfig 无 DCP 引用，社区同类 bug 存在）→ 方案 C 必须**分阶段**：先 DCP 单独验证，再叠加
- **⚠️ 前置事件**：A 组 TP2 当前**异常停机**（日志 08-09 01:19 多次 EngineCore 初始化 + ERROR dump input，6h 前 Exited）——落地前需先恢复或确认停机原因
- **总时长预估**：3 个阶段窗口，每阶段 1 个维护窗口

---

## 1. 现状基线（2026-08-09 16:00 实测）

### 1.1 生产配置

| 项 | 当前值 | 问题 |
|----|--------|------|
| vLLM | 0.26.1.dev0+gd3d3b2cca.d20260805 | 原生支持 A/B/C 全部参数 ✅ |
| TP | tensor_parallel_size=2 | DCP=2 满足约束（2%2==0）✅ |
| max_num_batched_tokens | **2024**（spec decoding 自动限制） | 预算小，长 prefill 更易独占 |
| max_num_partial_prefills | 1（默认） | **缺失** → 长 prefill 独占 |
| max_long_partial_prefills | 1（默认） | **缺失** → 无短请求插队 |
| long_prefill_token_threshold | 0（禁用） | **缺失** → 不分块并行 |
| scheduling_policy | fcfs | **缺失** → 无优先级 |
| decode_context_parallel_size | 1（未启用） | 长 ctx decode 无 DCP 加速 |
| dcp_comm_backend | ag_rs | 已配（默认） |
| prefix caching | enable_prefix_caching=True | 已启用 ✅ |
| chunked prefill | enable_chunked_prefill=True | 已启用 ✅ |
| spec decoding | dspark 5-token | 与 DCP 兼容性待验证 ⚠️ |
| kv_cache_dtype | nvfp4_ds_mla | MLA 压缩有效（6.7KB/token）✅ |

### 1.2 现状问题（已确认根因）

- **131K c5 TTFT 70/142/215/287/358s 线性排队** → asyncio 引擎 prefill 串行 + 长 prefill 独占预算
- **decode 被 prefill 阻塞**（c1 单流 74 t/s 恒定，c5 崩至 4.3）→ 软件调度问题非硬件

### 1.3 ⚠️ 前置：TP2 停机状态

- `vllm-envE-node` 容器 **Exited (1)** 6 小时前（08-09 01:20 前后）
- 日志：01:19 多次 EngineCore 初始化 + ERROR "Dumping input data" → **疑似异常退出**
- 落地前必须：①确认停机原因（查看 01:19 前后完整日志）②恢复或按方案重建

---

## 2. 方案设计

### 2.1 方案 A：并发 partial prefill（消除 prefill 阻塞）

**配置变更**：
```bash
--max-num-partial-prefills 4 \
--max-long-partial-prefills 1 \
--long-prefill-token-threshold 2048
```

**机制**（源码确认）：Phase 1 先调度 running（decode），Phase 2 调度 waiting（prefill）。`long_prefill_token_threshold>0` 后，长 prefill 被切块且最多 `max_long_partial_prefills=1` 个长 prefill 并行，其余资源让给短请求——**短请求可插队长 prefill**。

**预期效果**：
- 131K 长 prefill 不再独占 2024 token 预算
- 并发 5 请求的 TTFT 从"排队 70-358s"变为"并行推进 + 短请求插队"
- decode 请求获得稳定插槽

### 2.2 方案 B：priority 优先级调度（交互优先）

**服务端配置**：
```bash
--scheduling-policy priority
```

**客户端契约**（OpenAI 兼容 API extra_body，PR #5958/#19057）：
```json
{
  "model": "deepseek-v4-flash-0731",
  "messages": [{"role": "user", "content": "..."}],
  "priority": 0
}
```
- **数值越小优先级越高**（官方 API 参考确认）
- 推荐分级：`0`=交互（chat/agent）、`10`=RAG/常规 batch、`20`=后台批量（离线索引）
- 抢占语义：KV 不足时先抢占 priority 最大（最低优先级）请求，完全重计算

**⚠️ 注意**：priority≠0 且未启用 priority 调度会报错 → 服务端必须同时配 `--scheduling-policy priority`

### 2.3 方案 C：DCP 长 ctx 加速（分阶段验证）

**原理**：`decode_context_parallel_size=2` 把 KV cache 沿序列维度分片到 2 个 TP rank，**消除 KV 冗余副本**（当前 MLA 1 KV head + TP2 = 2× KV 冗余），为长 ctx 腾出 KV 空间，增加 batch 能力。

**约束**（官方 Ascend 文档）：MLA 模型 `tp % dcp == 0` → tp=2, dcp=2 ✅

**⚠️ 风险与验证门槛**：
- `SpeculativeConfig` 源码**无 DCP 引用** → dspark MTP 与 DCP 组合**未经官方保证**
- 社区 4×DGX Spark 文章确认：V1 draft 路径存在 DCP size 不传播 bug（需补丁/新分支）
- **必须分阶段**：
  - **Stage C1**：DCP=2 单独验证（**临时关闭 dspark MTP**，spec 先禁用）→ 验证 DCP 正确性与长 ctx decode 增益
  - **Stage C2**：DCP=2 + dspark MTP 组合 → 若崩溃/性能退化，回退 C1 或评估补丁

---

## 3. 落地方案（分阶段执行）

### 阶段 0：TP2 恢复与基线（1 个窗口）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 0.1 | 排查 TP2 异常停机（08-09 01:19 日志） | 定位 Exited 根因 |
| 0.2 | head-first 恢复 TP2（start_v026r_cluster.sh） | /health 200 + 双节点 |
| 0.3 | 基线复测：131K c1/c5 decode + TTFT | 记录当前性能 |

### 阶段 1：方案 A + B 落地（1 个窗口，低风险）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 1.1 | TP2 启动命令追加 A 参数 + `--scheduling-policy priority` | 容器启动无参数报错 |
| 1.2 | 复测 131K c5：TTFT 分布应**显著收窄**（不再 1.00× 线性排队） | TTFT p50 改善 |
| 1.3 | 交互/batch 混跑测试：priority=0 请求应优先于 priority=10 | 优先级生效 |
| 1.4 | 观察 24h：GPU KV usage、Prefix hit rate、无回归 | 稳定性 |

**回滚**：移除 4 个参数 + 重启，恢复 fcfs 默认。

### 阶段 2：方案 C 验证（1-2 窗口，高风险）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 2.1 **C1** | DCP=2 单独启用（**临时 `--num-speculative-tokens 0` 关闭 MTP**） | DCP 正确性：KV 分片日志 + 长 ctx decode 提升 |
| 2.2 | C1 基准：131K decode t/s vs DCP=1 | 量化 DCP 增益 |
| 2.3 **C2** | DCP=2 + dspark MTP=5 组合 | 若崩溃/退化 → 评估补丁或保持 C1 |
| 2.4 | 混合场景终测：A+B+C 全开 | 全链路验证 |

**回滚**：`decode_context_parallel_size=1` 恢复。

---

## 4. 风险矩阵

| 风险 | 等级 | 缓解 |
|------|------|------|
| DCP + dspark MTP 不兼容（崩溃/错误） | **高** | 分阶段 C1/C2；C2 失败回退 C1 |
| DCP=2 增加跨 rank KV 通信，长 ctx 反而变慢 | 中 | C1 基准量化；若退化不启用 |
| priority 语义误用（教程写反） | 中 | 按官方 API 文档：**lower=higher priority** |
| max_num_batched_tokens 过小（2024）限制收益 | 中 | 评估调大（注意 spec decoding 影响） |
| TP2 停机根因未明导致恢复失败 | 中 | 阶段 0 先排查 |
| 参数与镜像版本不兼容（已源码级确认支持） | 低 | 已排除 |

---

## 5. 预期收益（量化）

| 指标 | 现状 | 预期 | 依据 |
|------|------|------|------|
| 131K c5 TTFT 最差 | 358s | **<100s**（并行推进+插队） | A 方案机制 |
| 短请求（<2K）在长 prefill 后的等待 | 等完整个长 prefill | **可插队** | A+B 方案 |
| 交互请求响应 | 与 batch 同队列 | **优先调度** | B 方案 |
| 长 ctx decode 吞吐（131K） | 4.3 t/s (c5) | **待 C1 实测**（DCP 消除 KV 冗余 → 更大 batch） | C 方案 |
| 长 ctx 并发能力 | 受 KV cache 限制 | **KV 冗余减半 → batch 可增大** | C 方案 |

---

## 6. 资源与前置

| 项 | 要求 |
|----|------|
| 维护窗口 | 3-4 个（阶段 0/1/2），每窗口 1-2h |
| 业务配合 | ①确认交互/batch 请求的 priority 分级 ②batch 调用侧加 `extra_body.priority` |
| 人员 | SRE（执行）+ BE（客户端 priority 适配）+ 主理人（验收） |
| 回滚预案 | 每阶段独立回滚参数，无需重建镜像 |

---

## 7. 行动清单

| # | 行动 | 负责 | 紧急度 |
|---|------|------|--------|
| 1 | 排查 TP2 异常停机（阶段 0.1） | SRE | **P0** |
| 2 | 阶段 0 恢复 + 基线复测 | SRE | P0 |
| 3 | 阶段 1：A+B 参数落地 + 混跑验证 | SRE+BE | P1 |
| 4 | 阶段 2：C1 DCP 单独验证 | SRE | P2 |
| 5 | 阶段 2：C2 DCP+MTP 组合验证 | SRE+Archi | P2 |
| 6 | BE 侧 priority 字段适配（extra_body） | BE | P1 |

---

## 8. 关键决策点（需用户拍板）

1. **priority 分级方案**：0=交互 / 10=RAG / 20=后台 是否可接受？
2. **DCP 引入节奏**：建议先 A+B（阶段 1，低风险快速见效），DCP（阶段 2）单独验证后再上——是否同意分两批？
3. **max_num_batched_tokens**：当前 2024（spec 限制），是否同意在阶段 1 评估调大到 4096（需复核 spec decoding 影响）？
4. **TP2 停机**：确认是否需要本方案落地前先恢复生产？

---

## 📚 数据来源

- 本集群实测：vLLM 0.26.1.dev 参数源码级验证（check_vllm_dcp.py）、生产启动命令、TP2 日志（DCP=1、spec dspark）
- 社区调研：research-vllm-chunked-prefill-priority-2026-08-09.md（A/B 方案）、DCP 官方文档/4×DGX 实战文（C 方案）
- 根因分析：review-mla-compression-decode-collapse-2026-08-09.md

---

> 本落地方案由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
