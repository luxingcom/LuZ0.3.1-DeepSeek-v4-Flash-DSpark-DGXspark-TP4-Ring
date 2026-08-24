# NCCL 算子级通信延迟深度分析（小消息瓶颈 · 优化空间评估）

**日期**：2026-08-17 ｜ **分析人**：KernelGen（多芯片算子开发）｜ **对象**：生产 TP4 vLLM + NCCL 2.30.7 ring-only hardened（2be94172）
**前置**：stageB per-size tuner 已上线（368KB allreduce 923->173µs，-81%）；2hop 双 Primitives 路线已机制级否决

---

## 0. TL;DR

| 发现 | 定量 | 结论 |
|---|---|---|
| **368KB（prefill 主消息）仍有 3.8× 提升空间** | 实测 173µs vs 双口线速下界 22.6µs（效率 26%） | 有空间，但剩余差距主体是**软件固定延迟**（launch/同步/协议握手），非线速 |
| **decode 真正的瓶颈 = 61 次/层串行 14KB 小 allreduce** | 61 × ~40µs ≈ **2.4ms/token（占 10.1ms/token 的 ~24%）** | 小消息 = 纯延迟主导，带宽无关（呼应用户"内网带宽充足"判断） |
| **tuner 阈值与 16 通道存在粒度错配** | 368KB/16ch = **23KB/通道，落在 40KB 阈值之上被路由到 Simple** | Simple 处理 23KB 分片是延迟不友好选择--**可实测的最具体优化点** |
| **LL128 协议从未被 A/B**（T2 遗留） | LL128 = 6.25% 开销 + 低于 Simple 的延迟 | 23-92KB 分片区间的空白选项 |
| 环网拓扑的结构性天花板 | 非相邻 rank 无直连路径，one-shot 需 3 发（仅 2 口） | 2hop 已否决；**跨机 one-shot/two-shot 自定义 kernel 在此拓扑下不成立** |

---

## 1. 生产通信算子结构（实测还原）

### 1.1 消息分布（两种负载画像）

**Prefill/大消息**：每 MLA 层 1 次 368KB allreduce（bf16 hidden 7168 × 26 token 批）--Simple 协议，16 通道并发。
**Decode/小消息**：每 token 61 层 × 14KB（hidden 7168 × bf16）--tuner 路由 LL（<40KB 阈值），**逐层串行**（层间计算依赖，无融合空间）。

### 1.2 理论下界 vs 实测

| 消息 | 线速下界（双口） | 实测 | 效率 | 差距构成 |
|---|---|---|---|---|
| 368KB Simple/16ch | 22.6µs | **173µs** | 26% | 每通道 23KB 分片 + 协议/launch/同步固定延迟 |
| 14KB LL | ~6 hops × 2.5µs ≈ 15µs + LL 2× 流量 | ~40µs（推算） | ~37% | LL 线速翻倍开销 + 6 跳串行传播 + 同步 |

### 1.3 NCCL 日志实证（stageB 运行态）

- 16 通道全建立（`Channel 00/16 ~ 15/16`），双 HCA 轮换（IB/1↔IB/3 交替）✓
- **RING-ONLY v4** 硬编码映射生效（chan 级 dev 重定向日志）
- `VLLM_DISABLE_PYNCCL=1`（vLLM 直连 NCCL，无自定义 P2P 路径）

---

## 2. 优化空间清单（按 ROI 排序）

### 🎯 P1：tuner 通道感知阈值校准（最小成本、可实测）
**问题**：`NCCL_TUNER_THRESHOLD=40960` 按**消息总量**路由，但 16 通道下 368KB 的**每通道分片仅 23KB**--Simple 协议为 23KB 建数据通路（ handshake + chunk 转发）是延迟不友好的。
**方案**：tuner 决策改为 **per-channel size = total/nChannels**：
- 阈值不变（40KB），但判据用 `size/channels`：368KB@16ch -> 23KB -> **LL**
- 或反向：368KB 档改 4 通道（92KB/通道，Simple 合理区）+ 保持 decode 14KB 走 LL
**预期**：368KB 173µs -> **~120-140µs**（LL 在 23KB 分片的延迟优势）；
**验证**：nccl-tests 368KB 单尺寸 A/B（LL/Simple/4ch/16ch 2×2 矩阵，半小时）。

### 🎯 P2：LL128 填补协议空白（T2 遗留项，从未 A/B）
**问题**：LL（2× 流量、最低延迟）与 Simple（全带宽、高固定延迟）之间，**LL128（6.25% 开销、低延迟）从未实测**。
**方案**：`NCCL_PROTO=LL128` 固定 + 368KB/14KB 两档 nccl-tests；若 tuner 化，加第三分支（≤40KB LL / 40KB-2MB LL128 / >2MB Simple）。
**预期**：23-92KB 分片区间可能拿到 **10-20%** 延迟改善；LL128 对 RDMA 写内联（448B doorbell）友好。
**风险**：RoCE 无损队列（PFC prio5）需确认 LL128 的 128B 序列对齐无 PFC 风暴敏感性。

### P3：decode 通信-计算 overlap（vLLM 侧，收益上限最大但工程量大）
61 次串行 allreduce 占 decode ~24%。数学上不可全消（层间依赖），但 **MoE expert 计算与 allreduce 的 overlap**（dual-batch overlap 思路，SGLang TBO 类似）可隐藏 30-50% 通信。vLLM 0.26 无现成实现，属引擎级改造，**建议列为 0.27+ 上游跟踪项**（vLLM 社区 async-TP/DBO 正在演进）。

### P4：每跳延迟微优化（收益小、可打包进 P1/P2 验证）
- `NCCL_IB_QPS_PER_CONNECTION=2` + `NCCL_IB_SPLIT_DATA_ON_QPS=1`（T3 遗留，深管道降低 per-chunk 排队）
- `NCCL_LL_BUFFSIZE`（LL 缓冲对 14KB 单包化的影响）
- inline 补丁已就位（stageB INLINE 构建），确认运行态生效

### ❌ 已排除/不可行（明确关闭）
| 选项 | 排除原因 |
|---|---|
| **跨机 one-shot/two-shot 自定义 allreduce kernel** | 环网拓扑：每 rank 仅 2 个直连邻居，one-shot 需同时触达 3 peer；2hop 双 Primitives 已被 S3 终审**机制级否决**（SIMPLE 下崩溃、LL 未实例化） |
| vLLM custom_all_reduce（P2P 版） | 依赖 NVLink/P2P 映射，GB10 跨机仅 RDMA，不适用 |
| NVLS/多播 | 需交换机 SHARP，无交换机直连环不支持 |
| 消息融合（61 层合并） | 层间计算串行依赖，数学不可行 |
| 继续 BUFFSIZE/通道数调参 | stageB 后已饱和（16ch 为 R14v4 6-channel 异常排查后的定版） |

---

## 3. 建议执行序列

1. **本周可做**（无代码风险，纯 env/参数 A/B，nccl-tests 半小时/档）：
   - P1 tuner 通道感知（或 368KB 档 4ch 变体）
   - P2 LL128 两档实测
   - P4 QPS=2/SPLIT 打包验证
2. **判定门槛**：368KB allreduce <150µs 且 c1@131K PR/DE 不劣化（±3%）才切换生产
3. **中期跟踪**：P3 overlap 归入 vLLM 0.27+ 上游能力评估（与 #46789 sequence parallelism 同表跟踪）
4. **认知锚定**：decode 每 token 2.4ms 通信中，**~0.9ms（15µs×61）是 6 跳物理传播下界**（环网不可消），软件可压缩空间约 1.5ms，其中 overlap 是唯一大头手段

## 4. 数据与证据

- 生产 NCCL 日志：`vllm-tp4-rank0:/var/log/vllm/nccl-node01.log`（16ch/v4 映射实证）
- 基线报告：`nccl-final-performance-baseline-2026-08-17.md`（A0->stageB 三级演进、173µs 定版）
- 2hop 否决：`nccl-2hop-s3-final-adjudication-architect-2026-08-17.md`（机制级）
- 理论计算：ring allreduce 1.5× 放大、双口 16.67GB/s、6 跳传播模型（本报告 §1.2）
- 备注：KernelGen MCP（kernelgen-server）未配置，本报告为纯分析产出；若 P3 决定走自定义 kernel/Triton 通信算子路线，需先配置 MCP 再生成代码
