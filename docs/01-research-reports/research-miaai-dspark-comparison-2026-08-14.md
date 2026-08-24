# MiaAI-Lab DeepSeek-v4-Flash-DSpark-2x-DGX-Spark 技术方案对比与借鉴点

**日期**：2026-08-14 ｜ **仓库**：github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark（676★/96 fork，2026-08-14 当天仍更新）
**对比对象**：本集群（4× DGX Spark TP4 环网，anemll 0.2.1-v026.0 / vLLM 0.26.1.dev0）

---

## 1. MiaAI 方案速览

- **拓扑**：2× DGX Spark，TP=2，RoCE/NCCL，MTP-5 投机，`nvfp4_ds_mla` KV，1M token 上限
- **基座**：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（vLLM 0.25.2.dev0）——**比我们的 0.26.1 旧**
- **核心方法论**：**20+ 独立 hotfix 动态应用**（`patches/`），对 vLLM 源码做锚点字节校验 + 幂等应用 + `--status` 查询；**不重编镜像、不重启服务**
- **配方组织**：`recipe/overlay/vllm/...` 覆盖官方 main 源码；`recipe/official-main` = 官方 0731 权重配方；另有 abliterated/keys 变体
- 附带能力：dspark_vision_mcp 插件、thinking_token_budget（GPU 版）、suppress-stops-in-reasoning

## 2. 关键对比表

| 维度 | MiaAI（2x Spark TP2） | 本集群（4x Spark TP4） | 差异影响 |
|---|---|---|---|
| 并行规模 | TP=2 | TP=4（环网 + NCCL ring-only） | 我们内存池/带宽更大 |
| 基座 vLLM | 0.25.2.dev0（anemll 0.1.1） | **0.26.1.dev0（anemll 0.2.1-v026.0）** | 我们更接近官方 main |
| patch 方式 | hotfix 动态打（幂等/可查状态） | 镜像内置 + 自研 shim/ring-only | 我们升级靠重建镜像 |
| 调度公平 | #27 串行 prefill + **#43 decode floor** | 断崖已归因访存延迟（非饥饿） | 见 §4 |
| **128K 高并发 decode** | **~8.4–8.6 tok/s**（128K×c4/c6，TP2） | **7.8–8.2 tok/s**（131K×c5，TP4） | **同数量级 → 断崖是 GB10 物理极限，跨 TP 规模一致** |
| 128K c1 TTFT | 80.4s（8/1 基线） | prefill 2013-2016 tok/s（TTFT(c1)=57.4s，非 TTFT 口径） | — |
| 更新频率 | 每日多 commit | — | 借鉴窗口短，需持续跟踪 |

## 3. 可借鉴点（按价值排序）

### 3.1 ⭐ vLLM 0.27 修复效果清单（直接服务我们 0.27 升级）
MiaAI `docs/vllm-027-new-patches.md` 给出 6 个 0.27 修复的**实测效果**，我们的 0.27 构建（01 正在 NGC 26.07 容器内编译）应逐项核对：

| 上游 PR | 效果（MiaAI 实测） | 本集群 0.27 升级要点 |
|---|---|---|
| #49486 skip indexer topk/router | **TTFT -3.4%**（≤2048 token 触发） | 直接受益（prefill 优化） |
| #48957 skip 空 c128 compressor | **kernel ~2×**（cudagraph≠FULL 时） | 我们 capture-sizes 1..64 + CUDA Graph PIECEWISE（≠FULL，触发条件实际满足，对本集群是利好），需验证触发条件 |
| #50298 FlashMLA workspace reuse | **kernel 1.88×**（topk+SWA 索引） | prefill 路径直接受益 |
| #50004 adaptive C128A topk | **E2E -1.0%** | 小收益 |
| #50312 MTP buffer | **PP buffer 节省：PR 理论最大 448 MiB，MiaAI 实测 256 MiB/rank，本集群需实测** | 内存余量 |
| #49236 EagerScratchPool reuse | **TTFT -3.9%**（需 C++ op 重编） | **0.27 源码构建天然包含**——01 构建的直接红利 |
| #48047 sparse-MLA q-head padding | 需 FlashInfer ≥0.6.14 | NGC 26.07 内置 flashinfer 0.6.14 **恰好满足** ✓ |
| #46789 sequence parallelism / #48993 compact MXFP4 indexer | feature 级 | 评估是否纳入 0.27 灰度范围 |

→ **行动**：01 的 0.27 构建完成后，以上述清单为验收项（TTFT/内存/触发条件）。

### 3.2 128K 高并发断崖的普适性确认
- MiaAI TP2：128K×c4=8.4 / c6=8.6 tok/s；本集群 TP4：131K×c5=7.8–8.2 tok/s
- **两个独立实现、不同 TP 规模、同一数量级** → 断崖不是我们集群特有（非网络/非配置问题），是 **GB10 UMA 访存延迟的物理极限**，进一步坐实 F2 归因
- 含义：131K 高并发优化空间的真实边界就在 ~8 tok/s 附近，**不应期望任何调度/内核优化把它拉到 c1 水平**；合理目标是把"断崖拐点"推后（如 #27/#43 类公平性手段延长中等并发可用的上下文区间）

### 3.3 #43 decode fairness + per-step 调度诊断
- `DSPARK_ISSUE43_SCHED_DIAG=1` 输出每步每请求的 scheduled prefill/decode tokens + 零 token skip 记录
- 我们 F2 已排除调度饥饿，但该**诊断工具化思路**可借鉴（把断崖观测做成可复现的 per-step 日志），且 decode floor 作为**公平性 guardrail** 在"部分 prefill 并发提升后"有预防价值（#43 作者明言 floor 在 cap=1 时是中性，真正的解法需 vLLM 升级——与我们的 M2 结论一致）

### 3.4 正确性/产品化修复（低风险可借鉴）
- **#55 finish_reason=length**：max_tokens 截断工具调用时报 length（发现 FinishReason 是 IntEnum 非 StrEnum）→ 我们 8003 responses 网关/工具调用场景可参考
- **suppress-stops-in-reasoning**（Anemll 移植 Tony/Capicua25x Patch 5）：推理期 stop strings 抑制，think+stop 语义正确
- **thinking_token_budget GPU 版**（#48）：客户端 per-request 硬顶推理预算（Triton kernel，无 host scan）——产品化能力，与性能无关
- **dormant patch 纪律**（#48407 Stage A 绑定空串）：移植修复时"宁可不生效也不引入静默错误"——工程纪律值得采纳

## 4. 不借鉴点（明确排除）

- **0.25.2 基座 hotfix 锚点**：MiaAI 的 20+ patch 锚定 0.25.2 源码，**不能直接搬到 0.26.1**（锚点失配）；我们已用 0.26.1/0.27 天然包含大部分官方修复，无需重复移植
- **#27 max_num_partial_prefills**：已核实 0.26.1 无此字段（MiaAI 的 #43 提交也确认 0.25.2 上 `_check_feature_supported()` 拒绝 >1）——双方共同的硬约束
- **TP=2 配方本身**：不适用我们 TP=4 拓扑（通信/内存互斥已定）

## 5. 行动建议

1. **P0（跟随 0.27 构建）**：01 vLLM 0.27 构建完成后，用 MiaAI 清单验收：#49236（EagerScratchPool，TTFT -3.9%）、#49486、#48957、#50298、#50312、#48047（flashinfer 0.6.14 已具备）
2. **P1（借鉴落地）**：per-step 调度诊断（#43 思路）纳入断崖观测工具；finish_reason/suppress-stops 正确性修复按需移植到 0.27 灰度
3. **P2（认知固化）**：128K 高并发 ~8 tok/s 为 GB10 物理极限（跨 TP 规模验证），优化目标改为"推迟断崖拐点"而非"消灭断崖"
4. **跟踪机制**：该仓库每日更新，建议纳入每周上游核对（与 litellm/vLLM/SGLang 同表）

## 6. 参考

- 仓库：github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- docs/vllm-027-new-patches.md（0.27 修复效果清单）
- results/RESULTS-2026-08-14.md（TP2 性能矩阵）
- 关联：8/13 research_addendum_miaai-2026-08-13.md（Issue #22/#27 首轮补研）

---

## 7. 修订注记（2026-08-14 交叉验证后，依据 miaai-cross-verify-2026-08-14.md §6）

督导评审通过后，对 3 处不实/部分准确项做修正（其余内容未改动）：

1. **§3.1 #48957 行**：本集群 cudagraph 状态由 "CUDA Graph FULL" 改为 **"CUDA Graph PIECEWISE（≠FULL，触发条件实际满足，对本集群是利好）"**。
   依据：`start_tp4_head_b12x.sh` 实测 `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`，非 FULL；MiaAI docs 明确 #48957 fires only when cudagraph mode ≠ FULL。
2. **§2 表 128K c1 行**：由 "2013-2016ms×??（TP4 口径不同）" 改为 **"prefill 2013-2016 tok/s（TTFT(c1)=57.4s，非 TTFT 口径）"**。
   依据：tp4-service-deployment-guide-2026-08-13.md 中 131072/c1 prefill=2013-2016 tok/s、TTFT(c1)=57.4s，原值实为 prefill 速率而非 TTFT。
3. **§3.1 #50312 行**：由 "省 448 MiB/rank（我们 256 MiB/rank 口径）" 改为 **"PP buffer 节省：PR 理论最大 448 MiB，MiaAI 实测 256 MiB/rank，本集群需实测"**。
   依据：上游 PR #50312 声称 448 MiB（8192×4×7168×2B bf16 最大 buffer），MiaAI fork 实测 256 MiB/rank；448 为理论上限而非 per-rank 常量。
