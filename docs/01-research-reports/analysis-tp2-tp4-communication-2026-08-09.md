# DeepSeek-V4 TP2/TP4 互联通信带宽需求与延时影响分析

**日期**：2026-08-09
**工作流**：架构分析（基于模型架构参数 + NCCL 实测 + TP2 benchmark 三方数据）
**参与成员**：主理人（计算与综合）/ Tessa（实测口径）/ Rex（链路实测）
**模型**：DeepSeek-V4-Flash（dspark 变体，deepseek-v4-flash-0731，config.json 实读）

---

## 📌 TL;DR（执行摘要）

- **带宽结论**：**200G 链路远够用（prefill 峰值仅用到 25-45%），100G 在长 prefill 下会接近饱和（86-90%）**。DeepSeek-V4 是"通信轻量"模型——MLA 128× KV 压缩 + 窄 MoE（intermediate 2048）+ FP8 共同把通信量压得很低
- **缓存命中进一步减压（90% 常态）**：prefix caching 命中时 prefill 只算增量 token，互联带宽需求再降 ~10×（200G 的 1-4%，100G 的 5-9%）——生产已启用 `enable_prefix_caching=True`，"100G 饱和"仅存在于 0% 命中冷启动
- **延时结论**：**延迟主导（非带宽）**。每层 2 次 all-reduce，TP2（1跳）每层 ~58µs / TP4（2跳 ring）~116µs → 全模型纯通信 TP2≈2.6ms / TP4≈5.1ms；但 vLLM 流水线能部分掩蔽
- **KV 通信可忽略**：MLA 压缩后 KV all-gather 仅 ~11KB/token（≈标准 GQA 的 2%，与 8/9 研究文档 §3.2 实测结论一致）
- **TP4 相对 TP2 的真实代价**：主要不在带宽（都够用），而在 **ring 拓扑 2 跳延迟 + 消息频次不变但每跳延迟累加**；4 机当前是链式（04—02—01，03 未接线），TP4 需补 03 接线且 ring 存在对角边无直连的建连限制（8/9 NCCL 实测已确认）
- **结论**：若 TP4 目标是为 512GB 内存池/更高单请求吞吐，200G 互联不构成瓶颈；若仅为降低延迟，TP4 在 2 跳 ring 下延迟反而劣于 TP2

---

## 1. 模型架构参数（决定通信量）

来自 `deepseek-v4-flash-0731/config.json`（2026-08-09 实读）：

| 参数 | 值 | 对通信的影响 |
|------|-----|-------------|
| num_hidden_layers | 43（+1 MTP 层） | all-reduce 频次 = 44 层 × 2 |
| hidden_size | 4096 | 单次 AR 消息主项 = 4096×tokens×精度 |
| num_attention_heads | 64，head_dim=512 | 大 head_dim（非标准 128） |
| num_key_value_heads | 1（MLA 低秩） | **KV 通信极低** |
| q_lora_rank / o_lora_rank | 1024 / 1024 | KV latent 维度 |
| compress_ratios | 4/128 交替（层 2-41），首尾 dense | **KV 128× 压缩**（nvfp4_ds_mla + 结构压缩） |
| n_routed_experts / topk | 256 / 6 | MoE 窄（intermediate 2048） |
| moe_intermediate_size | 2048 | 窄 FFN → AR 消息小 |
| o_groups | 8 | 输出分组 |
| vocab_size | 129280 | 末层 logits AR 大消息 |
| 量化 | FP8 (e4m3) | 权重 1B/元素 |
| dspark_block_size / markov_rank | 5 / 256 | 投机解码（MTP 5 token） |
| num_nextn_predict_layers | 1 | MTP 额外 1 层通信 |
| max_position_embeddings | 1M | 超长 ctx 场景 |

**运行配置补充**（vLLM-envE-node 实查）：
- `--kv-cache-dtype nvfp4_ds_mla`：KV cache 用 FP4 + MLA 专用压缩
- `--speculative-config dspark 5 token`：投机解码一次前向处理 6 位置
- `--moe-backend flashinfer_b12x`：MoE 用 flashinfer 批量专家

---

## 2. 通信量计算（每 token）

### 2.1 每 token 全模型通信字节

| 通信类型 | 公式 | 每 token | 说明 |
|---------|------|---------|------|
| all-reduce（QKV/FFN） | 44 层 × 2 次 × 4096 × 1B(FP8) | **352KB** | 主项，与 seq 无关 |
| all-gather（KV, MLA 压缩） | 44 层 × (1024+512)/128 × 1B | **~11KB** | 可忽略（≈GQA 的 2%） |
| 合计 | — | **~368KB/token** | BF16 口径翻倍 ~736KB |

### 2.2 不同 seq 的整批通信量

| seq | all-reduce | KV-ag | 合计 |
|-----|-----------|-------|------|
| 1 (decode step) | 0.36MB | 0.02MB | **0.38MB** |
| 512 | 185MB | 8MB | 193MB |
| 4,096 | 1,476MB | 65MB | **1.54GB** |
| 32,768 | 11.8GB | 0.52GB | 12.3GB |
| 131,072 | 47.2GB | 2.1GB | 49.3GB |

> 131K 场景 decode 受限（实测 2-7 t/s），通信量大但不构成额外瓶颈。

---

## 3. 带宽需求 vs 链路能力

### 3.1 计算口径

- **per-layer 视角**（流水线并行）：单层通信字节 ÷ 单层计算时间
- 单层计算时间 = seq ÷ (vLLM 实测 prefill t/s) ÷ 43 层
- 实测 prefill：512→1118 t/s、4K→1900、32K→2000、131K→1660（B 组 benchmark 2026-08-08）

### 3.2 结果（BF16 保守口径 = 上限）

| 场景 | 单层 AR | 需求带宽 | **100G 占比** | **200G 占比** |
|------|--------|---------|--------------|--------------|
| prefill seq=512 | 8.4MB | 6.3 Gbps | 50% | 25% |
| prefill seq=4K | 67.1MB | 10.7 Gbps | **86%** | 43% |
| prefill seq=32K | 536.9MB | 11.3 Gbps | **90%** | **45%** |
| prefill seq=131K | 2.1GB | 9.4 Gbps | 75% | 37% |
| decode（任意 seq） | 16KB | ~0.4 Gbps | ~3% | ~2% |

> FP8 通信口径为上述一半（约 200G 的 22%），BF16 是最保守上限。

### 3.4 缓存命中情形（90% 命中常态——业务主场景）

> ⚠️ 上述 §3.2 为 **0% 命中（全量重算）** 的最坏口径。业务场景（共享系统提示词 / RAG 文档前缀 / 多轮对话历史）前缀高度复用，**prefix caching 命中率可达 90%**。生产 vLLM 已启用 `enable_prefix_caching=True`（V1 默认，日志可见该配置项）。

**机制**：prefix caching 命中时，命中的前缀 KV 直接从 cache 读取，**不重算前向**——只有增量（未命中）token 走完整前向 + all-reduce。TP 下 KV cache 按 rank 分片存储，命中请求各 rank 读本地分片，**跨节点 KV 传输为 0**。

**重算结果（BF16 保守口径）**：

| seq | 0% 命中 | 50% | **90%** | 99% | 100G@90% | 200G@90% |
|-----|---------|-----|---------|-----|----------|----------|
| 512 | 6.3 Gbps | 3.2 | **0.6** | 0.1 | 5% | 3% |
| 4,096 | 10.7 Gbps | 5.4 | **1.1** | 0.1 | 9% | 4% |
| 32,768 | 11.3 Gbps | 5.6 | **1.1** | 0.1 | 9% | 5% |
| 131,072 | 9.4 Gbps | 4.7 | **0.9** | 0.1 | 7% | 4% |

**关键影响**：
- 90% 命中 → 只算 10% 增量 token：131K prefill 从 47GB 通信降到 ~4.7GB，**互联带宽需求降 ~10×（200G 的 1-4%）**
- **100G 链路在 90% 命中下也仅 5-9%**——缓存命中彻底消除带宽担忧（此前"100G 长 prefill 饱和"的结论仅适用于 0% 命中冷启动）
- **TTFT 收益**：131K 全量 prefill 79s → 90% 命中仅 ~8s（增量 prefill），缓存读取近 0 延迟
- **decode 不变**：每步 1 新 token 必须计算，单步通信 ~0.7MB/step，与命中率无关

**⚠️ 命中率救不了的两个场景**：
1. **冷启动/随机长上下文**（0% 命中）：131K prefill 需全量，200G 仍够（37-45%）但 100G 饱和
2. **未来开 decode context parallel（DCP>1）**：KV 分片跨节点，每步需 all-gather 新 KV——命中率无法救（当前 DCP=1 未启用，无此问题）

> **与 benchmark 归因的差异说明（已复核，2026-08-09）**：原 B 组报告将 131K decode 并发崩塌（c5 2-7 t/s）归因"统一内存带宽饱和"。复核确认（详见 review-mla-compression-decode-collapse-2026-08-09.md）：**c1 单流纯 decode 512→131K 恒定 70-76 t/s 不随 ctx 下降** → 排除内存带宽/KV 读/UMA 饱和；真实根因 = **asyncio 引擎 prefill 串行 + 长 ctx prefill 阻塞 decode**（TTFT=纯 prefill 时间 1.00×，prefill_tps 并发下 5 等分）。MLA 压缩确认有效（KV=6.7KB/token），MoE 专家权重实为 I8(INT8) 148GB。

### 3.3 关键结论

- **200G 链路：prefill 峰值 25-45%，decode 仅 2%——带宽余量 >2 倍，完全够用**
- **100G 链路：4K-32K prefill 达 86-90%，接近饱和**——若未来降级到 100G（如 H 线 2-lane），长 prefill 会成为瓶颈
- 本项目实测链路为 **200G 满速**（01↔02、02↔04，ethtool 200000Mb，0 错包），无带宽压力

---

## 4. 延时影响（TP2 vs TP4）

### 4.1 实测锚点（2026-08-09 NCCL 测试）

| 指标 | 值 | 说明 |
|------|-----|------|
| NCCL 16B all-reduce 单跳 | **29.0µs** | 含 GPU launch + 驱动（wire 仅 3.27µs） |
| one-way broadcast | 18.4µs | 链路对称 |
| 1 跳 RDMA wire | 3.27µs | 纯硬件 |

### 4.2 每层与全模型延迟（未掩蔽口径）

| 维度 | TP2（1跳） | TP4（2跳 ring） |
|------|-----------|----------------|
| 每层 all-reduce 次数 | 2 次 | 2 次 |
| 每层延迟 | 2×29 = **58µs** | 2×2×29 = **116µs** |
| 全模型（44 层） | **2.55ms** | **5.10ms** |

### 4.3 大消息时带宽/延迟权衡

- 单次 all-reduce 总时间 = 固定延迟（29µs/跳）+ 传输时间（字节÷带宽）
- seq=4K 单次 AR=33.6MB：200G 传输 1.3ms、100G 传输 2.7ms → **带宽主导**（固定延迟占比 <2%）
- seq=512 单次 AR=4.2MB：200G 传输 0.17ms vs 延迟 29µs → 延迟占比 ~15%
- **短序列延迟主导，长序列带宽主导，但 200G 下两者都不构成瓶颈**

### 4.4 vLLM 流水线掩蔽

- 43+1 层前向为流水线，all-reduce 可与下一层计算重叠（NCCL 异步 + CUDA 流并发）
- 实测 TP2 prefill 1900 t/s 说明流水掩蔽效果良好；**TP4 的 2 跳延迟增量大概率被部分掩盖**
- 纯增量估算：TP4 相对 TP2 全模型通信延迟 +2.55ms，若 70% 被掩蔽 → 实际影响 ~0.8ms/请求

---

## 5. TP4 拓扑专项分析

### 5.1 当前物理拓扑（2026-08-09 布线报告）

```
[04] ═══ 200G ═══ [02] ═══ 200G ═══ [01]      （03 未接线）
```

- TP4 需 4 机环：**03 补线后为链式 03—01—02—04**，或理想 ring
- **8/9 NCCL 实测确认**：对角边（无直连）`ibv_modify_qp timeout`——NCCL 不做透明多跳转发，ring 需物理闭环

### 5.2 TP4 通信特征（相对 TP2）

| 项 | TP2 | TP4 | 影响 |
|----|-----|-----|------|
| all-reduce 每层 | 1 跳 | ring 2 跳（4 rank） | 延迟 ×2（小消息） |
| 每 rank 权重 | 1/2 | 1/4 | 计算更快，通信占比相对上升 |
| ring 每边负载 | 1×消息 | 2×消息（4 rank 环） | 带宽需求 ×2，但仍 <90% of 200G |
| KV cache（MLA） | rank 内分片 | rank 内分片 | 无额外通信（KV 已压缩 128×） |
| 内存池 | 2×121G | 4×121G | **TP4 核心价值：512GB 统一内存池** |

### 5.3 TP4 真实价值与代价

- **价值**：①内存池 ×2（当前 131K 场景 decode 仅 2-7 t/s 受限于单机内存/KV；TP4 可承载更大 KV）②单请求吞吐可拆分到 4 卡
- **代价**：ring 2 跳延迟（每层 58→116µs，全模型 +2.55ms）+ 布线/编排复杂度 + 对角边建连限制
- **200G 带宽不是 TP4 的瓶颈**（最坏 45%×2 场景仍 <90%）

---

## 6. 结论与建议

| 问题 | 结论 |
|------|------|
| 200G 够用吗？ | **够**：prefill 最坏 45%（BF16 口径），decode 2%，余量 >2× |
| 100G 够用吗？ | **0% 命中冷启动会饱和（86-90%）**；**90% 命中常态仅 5-9%，够用** |
| 缓存命中（90%）影响？ | **互联带宽再降 ~10×（200G 的 1-4%）**，TTFT 降 10×；命中时 KV 无跨节点传输 |
| TP2 的瓶颈是带宽还是延迟？ | **延迟**（GPU launch 主导，wire 仅 3.27µs） |
| TP4 相对 TP2 快吗？ | **带宽上无差异；延迟上 TP4 2 跳 ring 反而劣化**；价值在内存池 |
| KV 通信要担心吗？ | **不用**：MLA 128× 压缩后仅 ~11KB/token（命中时 0 传输） |

### 行动建议（按价值排序）

1. **若 TP4 目标 = 512GB 内存池 / 更长 ctx**：值得做，200G 互联满足要求；补 03 接线 + ring 闭环（需对角直连或换拓扑）
2. **若 TP4 目标 = 降延迟**：不推荐——2 跳 ring 延迟劣化，收益在内存不在速度
3. **短 ctx（<4K）场景优化**：方向是减少每跳延迟（NCCL_PROTO=LL/LL128、多通道）而非加带宽——wire 3.27µs vs NCCL 29µs 的 26µs 差距在 GPU launch/驱动
4. **禁止降级 100G**：**仅 0% 命中冷启动**（长 prefill 全量）在 100G 会饱和；若业务命中率高（≥90%）可接受 100G，但 200G 仍是安全垫
5. **优先提升 prefix cache 命中率**（RAG 前缀规范化/系统提示词固定化）——比任何网络优化都更直接：带宽需求与 TTFT 同时降 10×

---

## ⚠️ 局限与假设

- 通信字节按全模型同步前向估算；实际 vLLM 的 chunked prefill / 流水线会分摊
- decode 带宽按 per-token 瞬时峰值算，未计 batch 内多序列并发（batch 增大 → decode 带宽需求线性上升；batch=6 时 decode AR 带宽 ~2%×6=12% of 200G，仍充裕）
- NCCL 16B 延迟 29µs 为单通道 RING/Simple 实测；LL 协议/多通道可能更低（行动项已列）
- dspark 投机解码（5 token）使通信量 ×6，但同时吞吐 ×~4-5，净带宽需求基本不变
- 模型 187B 粗估为 FP8 权重全量；实际加载按 kv-cache 预算运行

---

## 📚 数据来源

- 模型架构：`/home/<USER>/models/deepseek-v4-flash-0731/config.json`（实读，2026-08-09）
- 运行参数：vLLM-envE-node 容器 Cmd（nvfp4_ds_mla / dspark 5-token / flashinfer_b12x）
- NCCL 实测：benchmark-nccl-diagonal-2026-08-09.md（16B 单跳 29µs、200G 满速）
- prefill/decode 速率：benchmark-B-group-2026-08-08.md、benchmark-tp2-prefill-decode-protocol-2026-08-08.md
- 拓扑/布线：network-200g-wiring-survey-2026-08-09.md、migration-tp2-nccl-2026-08-08.md
- 计算脚本：tp_comm_analysis.py / tp_comm_analysis_v2.py（deliverables/engineering-assurance/）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
