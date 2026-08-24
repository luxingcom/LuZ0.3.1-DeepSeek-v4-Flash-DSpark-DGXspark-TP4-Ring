# FreeToken 开源边缘端推理系统调研报告

**日期:** 2026-08-22
**作者:** 阿奇（Archi）· 系统架构师 / freetoken-researcher
**任务来源:** 用户指示——"社区有一套开源的 FreeToken 全新边缘端推理系统，其动态路由载入机制对我们研究的几个方案有很大助力，请安排调查研究"
**调研方式:** 纯 Web 调研（WebSearch + WebFetch），无 GPU / 服务器操作

---

## 0. 结论速览（TL;DR）

1. **FreeToken 真实存在，且真实开源。** UC Berkeley / MIT / UT Austin 联合团队（Ion Stoica、Matei Zaharia、韩松、Kurt Keutzer 等），论文 arXiv:2608.16157（2026-08-17 提交），GitHub `FlashML-org/FreeToken`（Apache-2.0），官网 flashml.ai，提供 Windows/Linux 桌面 App 与 `uv pip install "freetoken[accel]"` CLI。【口径：官方实证】
2. **"动态路由载入机制"的准确画像**：FreeToken 的核心是 **"路由感知的专家驻留 + 带宽自适应 miss 分流"**——GPU 端全局 LRU 专家缓存（跨全部 MoE 层共享，逻辑 `(layer, expert)` 粒度）+ 设备端路由控制内核（去重→驻留检查→fetch count→victim 选择→槽位 ID 重写，全程在 CUDA Graph 内）+ miss 二分（`M = F ∪ C`，一部分 PCIe 填充进缓存、一部分直接 CPU 原地执行，分流比 `q* ≈ m·B_P/B_H`）。
3. **关键澄清（对我们最重要）**：FreeToken **明确不做路由预测、不做跨步/跨请求路由聚合**——它刻意选择"纯 LRU 捕捉时间局部性 + 改变残余 miss 的服务方式"，并把"预测预取 + LRU + q* 分流的组合"列为 open problem。**因此它不能直接改写我们 merged-GEMM 热桶方案的覆盖率数学**，但它提供的 miss 率实证数据（decode 路由存在显著短期局部性）给"跨步聚合"路线提供了新的实验依据与明确的否定性边界。
4. **对我们环境（GB10 UMA）的适配性警告**：FreeToken 的全部核心机制都围绕** PCIe 带宽瓶颈**设计（离散 GPU + host DRAM 两级内存）。我们的 DGX Spark 是 **UMA 架构——CPU/GPU 共享同一 121GB LPDDR5x，不存在 host→device 权重搬运瓶颈**。因此其"专家按需载入/分流"主机制**不适用于我们的常驻权重场景**；真正可移植的是四个工程思想：设备端路由控制、弹性内存再分配、语义锚点状态复用、预打包权重格式。
5. **最大助力点排序**：方案 3（专家驻留/内存管理，弹性再配置思想，中价值）> 方案 4（KV cache，语义锚点 checkpoint，中价值）> 方案 1（merged-GEMM，miss 率证据 + 设备端路由内核模式，间接价值）> 方案 2（W4A4，基本无关）。

---

## 1. FreeToken 系统调研

### 1.1 系统定位

| 维度 | 内容 | 口径 |
|------|------|------|
| 定位 | 边缘原生（edge-native）MoE 推理服务系统：把个人电脑（GPU+CPU+host 内存+PCIe）当作**统一的弹性推理平台**，而非"一块小显存 GPU" | 官方论文 |
| 团队 | 并列一作：杨硕（Berkeley EECS 博士）、范晓泽（UT Austin），作者含 Ion Stoica、Matei Zaharia、韩松、Kurt Keutzer、Chenfeng Xu 等；受 mini-sglang 启发，复用/借鉴了 SGLang、vLLM、FlashInfer、flash-linear-attention、LightLLM、llama.cpp 的设计与代码 | GitHub README + 社区报道 |
| 发布 | 论文 2026-08-17；代码首次开源 2026-08-11；v0.1.2（2026-08-19）；30 commits | GitHub |
| 许可证 | **Apache License 2.0**（对我们的 cherry-pick 友好） | GitHub LICENSE |
| 支持模型 | 20+ MoE 模型：DeepSeek-V4-Flash（284B/13B 激活，MXFP4）、Qwen3.6-35B-A3B（BF16/NVFP4）、GLM-5.2（753B/40B 激活，NVFP4）等 | 官方论文 + README |
| 量化格式 | MXFP4 / NVFP4 / FP8 / BF16；原则是"不超出所提供模型格式降低精度"（与 HOBBIT/SiDA/SMoE 的专家替换/降精度路线划清界限） | 官方论文 |
| 硬件覆盖 | RTX 30/40/50 系列，8GB 笔记本 GPU 到 96GB 工作站卡；实测 6 台机器 | 官方论文 |
| 接口 | Anthropic/OpenAI 兼容 API，实测接入 Codex、Claude Code、OpenCode、OpenClaw、DeepSeek Harness 等真实 agent | 官方论文 |

### 1.2 核心成果数据【口径：官方论文实证】

**能力边界改变：**
- 8GB RTX 4060 笔记本跑 Qwen3.6-35B：**39.3 tok/s**（NVFP4；论文引用的 Codex 生产 trace decode 中位数为 33 tok/s）
- RTX 5090 跑 DeepSeek-V4-Flash 284B（满血 MXFP4）：**22–25 tok/s**（需 32GB 显存 + 建议 192GB host 内存）
- 单卡 RTX PRO 6000（96GB）跑 GLM-5.2 753B：**14.9 tok/s**（vs llama.cpp 7.3，2.0×）

**加速比（vs 最强基线 llama.cpp/Ollama/KTransformers/MoE-Infinity）：**
- Qwen3.6-35B：1.3×（3090/4090）～ 2.1×（5090 desktop）；RTX 5090 上 77–83 tok/s（1.8–2.3×）
- DeepSeek-V4-Flash：1.5–1.9×
- TTFT：最差轮次 < 44s（基线全部 > 150s：llama.cpp 232s、Ollama 179s、KTransformers 946s）；多轮 agent 后续轮次 TTFT 减少 65–80%
- Agent 负载下 decode 吞吐与单轮差距 < 12%（KTransformers 在 OpenCode 上损失 31%）
- 双缓冲消融：去掉第二个 buffer，4k/8k/16k prompt 吞吐分别 -19%/-25%/-26%
- 内存压力韧性：显存被后台应用抢占 4–8GB 时，零重启热收缩 LRU 缓存，服务降级不中断

### 1.3 动态路由载入机制深析（用户点名的核心）

FreeToken 的"动态路由载入"是一套**四件套协同机制**，而非单一技术：

#### (a) 两级专家存储 + 全局 LRU 专家缓存（驻留策略）
- 完整专家池驻留 **pinned host memory**（source of truth）；非专家权重常驻 GPU。
- 剩余 GPU 显存成为**跨所有 MoE 层共享的单一弹性缓存**；每个 slot 持有完整 `(layer, expert)` 的全部张量。
- **纯 LRU 驻留**（刻意不做静态"热专家"放置、不做预测）：cache hit 刷新 recency，miss 准入新专家并逐出最久未被路由的专家。论文实证"token 级时间局部性比 prefill 推导的放置更具预测性"。
- **实测 miss 率**（同一 routing trace 回放，RTX 5090 服务容量：Qwen 缓存=专家池 37%，DSV4-Flash 缓存=11%）：

| 引擎/策略 | Qwen3.6 miss | DSV4-Flash miss |
|---|---|---|
| **FreeToken 全局 LRU** | **16%** | **39%** |
| KTransformers（prefill 更新放置） | 41% | 59% |
| llama.cpp（路由盲静态切分） | 62% | 89% |

#### (b) 设备端路由控制内核（"载入"的调度中枢）
每个 MoE 层由**单个 GPU kernel** 完成全部 routing 依赖控制流：① 去重被路由专家 → ② 对照驻留表分类命中 H/缺失 M → ③ 推导带宽感知 fetch count q → ④ **单遍**选出 K 个 LRU victim 槽（victim 发现代价恒定，与 miss 数无关）→ ⑤ 把逻辑专家 ID **重写为物理缓存槽 ID 或 CPU 分配标志**。配合固定形状 work buffer、device-resident valid count、捕获的 host-function node，使异构 CPU-GPU 步骤可在 **CUDA Graph 内重放**，无逐 token Python 调度。

#### (c) 带宽自适应 miss 分流（q* 策略）——机制的理论核心
对每步 m 个缺失专家：**M = F ∪ C，q = |F|**。
- F（cache-fill 集）：PCIe 传入缓存槽，GPU 执行，保持驻留；
- C（CPU 执行集）：直接从 CPU 驻留池原地执行（pinned C++ 线程池 + 物理 core 绑定 + SIMD + in-kernel 反量化），产出 gate-weighted 部分和；
- 推导（论文式 2–4）：设 B_P=PCIe 传输带宽、B_H=host 专家处理带宽、B_R=max(B_H−B_P,0)（DMA 与 CPU 执行读同一 host 内存子系统），平衡两并发分支得 **q* ≈ m·B_P/B_H**；B_H→B_P 时退化为纯按需填充。
- 两路**并发**执行，层延迟=较慢分支；始终保留 ≥1 个 fill 使缓存持续预热；F 的具体选择委托给替换策略。
- B_P/B_H 在**部署时于目标硬件、实际 tensor shape 上实测**（Table 1：5090 server 52.7/77.3，4060 笔记本 11.8/47.5 GB/s 等）。
- **精确性保证**：CPU/GPU 部分输出用原始 routing/gating 合并，不改 router、不换专家、不降精度。

#### (d) 双缓冲全层 Prefill 流水线（载入与计算重叠）
- 关键洞察：**prefill 摧毁 MoE 稀疏性**——长 prompt 各 token 路由并集近似覆盖每层全部专家，按需取专家得不偿失，因此**整层预载**：GPU 算第 l 层时，专用 transfer stream 同时加载第 l+1 层全部专家，双 buffer 轮换；传输在路由未知时即可开始，权重搬运完全藏进计算（RTX 5090 上 8192-token chunk 1.19–1.22s ≈ 以 52.7 GB/s 流完 64.4GB 专家池的时间，16k prompt prefill 达 6.7k tok/s）。
- 两个 full-layer buffer 与 decode 缓存**共享同一 slot pool**：prefill 存活条目直接 seed decode 阶段。
- GPU 内存装不下两个整层时回退 on-demand prefill loading。

#### 配套机制（简述）
- **语义锚点状态复用**（agentic state reuse）：在 thinking 段、tool call/output、对话轮次等 special-token 边界设轻量 recurrent-state checkpoint + full-attention KV 用 radix prefix tree + paged KV；agent 编辑上下文后仅从最深存活锚点恢复并增量 prefill（后续轮 TTFT -65~80%）。
- **弹性内存管理**：任意 scheduler safe point 可按修订后的 VRAM 预算重建专家缓存（KV 页 ↔ expert slot 再平衡），不重启、不重载；"GPU 内存只影响性能，不影响正确性"。
- **FTW 权重格式**：离线把专家权重预合并为 bank 布局（`lE+e` 为 leading dimension），启动时并行 direct I/O 直读精确大小 host bank、填满后才 pin（避免 pin 空 buffer 触发 GB 级缺页清零），跳过 tensor discovery/repacking，且**无需预热**（冷缓存直接服务，自然升温）。

#### 关于"路由预测/跨步聚合"的明确立场（重要！）
论文原话（Related Work）：预测式系统（ProMoE、ExpertFlow、FineMoE）的差异"在于预测 miss 的好坏，而非如何服务 miss——**every miss is ultimately a PCIe transfer**……FreeToken 改变的是残余 miss 的服务方式，而非 miss 被预测得多准"。
- §3.2 利用**相邻 token 路由重叠**（引 Liang et al. 2025 跨模型族 routing consistency 测量）作为 LRU 有效性的依据；
- 跨请求复用仅针对 **prefix/recurrent state**（radix tree），**未做跨请求专家路由聚合**；
- "预测预取 + 共享 LRU + q* miss 执行的组合"被明确列为 **open problem**；
- 无投机预取（对比 Mixtral-offloading 的 LRU+speculative prefetching 组合，FreeToken 未采用）。

### 1.4 依赖与工程形态

- **自研运行时**（llama.cpp/Ollama/KTransformers/MoE-Infinity 均为对比基线），但明确"学习设计并复用了"SGLang、vLLM、FlashInfer、flash-linear-attention、LightLLM、llama.cpp 的代码，受 mini-sglang 启发——**代码形态对我们（vLLM fork 栈）并不陌生**。
- 交付形态：PyPI wheel（`freetoken[accel]`）、桌面 App（GUI）、`freetoken-kernel-cache` 目录（预编译 kernel 缓存）、FTW 模型格式。
- 局限（论文自认/第三方指出）：q* 带宽模型忽略 NUMA/CPU 争用/非线性饱和；CUDA Graph 对变化 batch/多并发会话的处理未说明；**多 GPU 完全未评测**（仅单卡）；CPU 分支实验限 6–8 线程；能耗未报告。社区（AI Beat）也指出"与朴素 offloading 的同硬件对比增益幅度"论证不充分。

---

## 2. 对我们四个研究方案的助力评估

**我们的环境基线**：DGX Spark 4 节点 TP4，GB10（sm_121a）**UMA 121GB/节点**（CPU/GPU 共享 LPDDR5x，无 PCIe host→device 搬运），DeepSeek V4 Flash W4A16 MXFP4 ~40.5GB/rank 权重**全常驻**，vLLM 0.26.1 fork + B12X + TP4 环网，6M tokens KV 池。

> **先说根本性架构差异**：FreeToken 的全部核心机制（专家载入、PCIe/CPU 分流、双缓冲预取）都是为"**显存放不下专家池、host↔device 有窄管道**"的离散 GPU 边缘机设计的。我们在 UMA 上权重已全常驻、无窄管道，**FreeToken 的"载入"主机制没有直接对应物**。因此评估焦点是"思想可否移植"而非"代码可否集成"。

### 2.1 方案 1：merged-GEMM 热桶（已 e2e No-Go）

**背景死因**：真实路由下单步内专家集合合并覆盖率极低（dense 层 gate 逐 token 多样化，59K distinct sets，单步组合频率中位=1）。

**FreeToken 能否改写覆盖率数学？——不能直接改写，但有三点重要输入：**

1. **否定性证据（关键）**：FreeToken 是该领域最懂路由局部性的团队之一（Stoica/Zaharia/Han 阵容），他们**刻意不做路由预测与跨步聚合**，且明确把"预测+LRU+分流组合"列为 open problem。这从侧面印证：**可靠的跨步路由聚合至今没有公开可复用的成熟方案**——我们不应期待从 FreeToken 直接拿到"路由预判"技术。
2. **专家粒度局部性 vs 集合粒度覆盖率的区分（最重要的间接助力）**：FreeToken 实测 decode 阶段专家级 miss 仅 16%（Qwen，缓存 37%）/39%（DSV4-Flash，缓存仅 11%）——**单个专家的复访局部性是真实存在的**（相邻 token 路由重叠）。但我们 merged-GEMM 死在**专家集合（N 维组合）的匹配频率**，集合中位数频率=1 与专家级高局部性**并不矛盾**（集合=6 专家的组合空间 256^6，即使每个专家 40% 时间活跃，特定组合仍可几乎不重复）。这给我们的教训：**任何"跨步聚合"重试应把聚合粒度从"集合"降到"专家"**——例如跨步按专家（而非集合）重排 token 分组（本质是更大的 grouped GEMM M 维聚合，等价于把 FusedMoE 的 sort-by-expert 窗口从单步扩到多步）。这不是 FreeToken 的机制，但其 miss 数据是这个方向可行性的最强公开论据。
3. **设备端路由控制内核是可移植的工程模式**：我们的 merged-GEMM/路由调度若做多步窗口聚合，FreeToken 的"去重→驻留/分组检查→work list 生成→单遍 victim/槽位选择→逻辑 ID 重写，全部设备端、CUDA Graph 兼容"模式，正是把动态路由控制从 Python/宿主侧移进图内的成熟做法（vLLM 的 FusedMoE 已部分如此，但 FreeToken 的"captured host-function node + device-resident valid count"处理异构分支的模式值得借鉴）。

**移植路径**：借鉴思路 cherry-pick。**工作量**：(a) 用我们的 routing trace 回放"多步窗口专家级聚合覆盖率"——纯 CPU 离线分析，**约 1–2 人日**，建议先做（决策性实验）；(b) 若覆盖率可观，设备端聚合调度内核改造约 2–4 人周。**价值判断：间接但真实，优先做离线覆盖率实验再决定是否复活方案 1。**

### 2.2 方案 2：W4A4 量化路线（kernel 就绪、workspace 死结已解）

- FreeToken 与 W4A4 **基本无关**：它不做量化研究（只消费模型自带格式），其 CPU 执行分支用的也是 in-kernel 反量化，不涉及 4 比特激活。
- 唯一间接输入：FreeToken 证实 **prefill 阶段路由并集近似稠密**——这意味着 prefill 大 M 下 grouped GEMM 每个 expert 的 M 天然饱满，**prefill 恰是 W4A4 扩 M（threshold 4096）受益最大的阶段**，与我们"threshold 4096 实测 +12%"方向一致，可作为扩 M 主张的旁证（口径：推断，借官方 prefill 稠密性结论）。
- **移植路径**：不适用/仅旁证。**工作量**：0。

### 2.3 方案 3：专家权重载入/内存管理（KV 池与权重竞争 121GB UMA）

这是我们与 FreeToken **思想重合度最高**的方案，但移植的是**内存治理思想**而非载入机制：

1. **弹性再分配（最值得抄）**：FreeToken 的"任意 scheduler safe point 重建专家缓存、KV 页 ↔ expert slot 动态再平衡、不重启不重载、GPU 内存只影响性能不影响正确性"——映射到我们：**KV 池（6M tokens）与 W4A4 workspace 的动态再平衡**。我们的 workspace 补丁方案解决了 W4A4 内存死结，但 workspace 大小是静态配置；FreeToken 模式提示可做成"按实际 batch/M 水位在 safe point 弹性伸缩 workspace ↔ KV 页"，在长上下文低并发与短上下文高并发负载间自适应。**口径：借鉴思想的推断，需 PoC。**
2. **驻留策略本身不适用**：我们每 rank 全部 64 experts × 43 层常驻（TP4 切分后 40.5GB，UMA 放得下），没有"驱逐/按需载入"需求。仅当未来出现"单节点多模型常驻"或"KV 池扩到挤压权重"场景时，其 LRU 驻留 + slot pool 设计才有直接参照价值。
3. **FTW 预打包格式**：离线合并 expert bank、启动跳过 discovery/repacking、direct I/O 并行直读、"填满才 pin"（UMA 上即避免无用缺页/清零）——我们 4 节点 TP4 每次重启的权重加载路径可以直接借鉴，缩短冷启动。**工作量：约 1 人周（离线打包脚本 + 加载路径改造），收益中等。**
4. **Grace CPU 分流（记录但不推荐）**：GB10 的 10 个 Grace CPU 核与 GPU 共享 LPDDR5x 带宽，理论上存在"decode 阶段把部分 memory-bound 专家 GEMM 分给 CPU"的 FreeToken 式玩法。但 (a) Grace 核算力弱、(b) 与 GPU 争同一 LPDDR 带宽（无"残余带宽"红利，FreeToken 的 B_R 依赖 PCIe 与 DRAM 双通道分离，UMA 上不成立）、(c) vLLM 无 CPU 专家执行路径。**口径：推断，预期收益低、工程量大，不建议投入。**

**移植路径**：cherry-pick 思想（弹性内存治理 + FTW 加载）。**工作量**：FTW 加载约 1 人周；弹性 workspace↔KV 再平衡 PoC 约 2–3 人周。

### 2.4 方案 4：KV cache 管理（nvfp4_ds_mla，6M tokens 池）

- vLLM 已有 prefix caching/radix attention，FreeToken 的增量在于**语义锚点 recurrent-state checkpoint**：锚定在 tool call / thinking 段 / 对话轮次边界，agent 编辑上下文后从最深存活锚点增量恢复，多轮 TTFT -65~80%（官方实证，Claude Code/OpenCode/OpenClaw 负载）。
- **适用性**：我们若有 agent/多轮工具调用负载（迹象：mtp-tuning、TTFT drift 调查等），把 vLLM 的 prefix cache 失效粒度从"字节级 diff"升级为"语义块级锚点"（在 tool_call/tool_result/thinking 结束 token 处主动建立可恢复点），可直接削减多轮 TTFT。对纯单轮 batch 负载无收益。
- **注意**：FreeToken 的 checkpoint 机制很大程度是为 gated DeltaNet/KDA 等**recurrent 层**（状态无法从 KV prefix 重建）设计的；我们是 MLA（标准压缩 KV），vLLM prefix caching 已覆盖大部分场景，**收益上限低于其论文数字**。**口径：官方实证机制 + 推断的适用上限。**
- **移植路径**：借鉴思路（vLLM 侧实现语义锚点，非引入 FreeToken 代码）。**工作量**：约 2 人周（scheduler/ prefix-cache manager 改造 + agent trace 回放验证）。

### 2.5 汇总矩阵

| 方案 | FreeToken 直接助力 | 可移植内容 | 路径 | 工作量 | 优先级 |
|------|------|------|------|------|------|
| 1 merged-GEMM | ✗（明确不做预测/聚合） | miss 率证据 + 专家粒度聚合启示 + 设备端路由控制内核模式 | cherry-pick 思想 | 离线覆盖率实验 1–2 人日；若复活 2–4 人周 | **先做实验** |
| 2 W4A4 | ✗ | prefill 稠密性旁证（支持扩 M） | 旁证引用 | 0 | 低 |
| 3 专家驻留/内存 | 部分（驻留本身不适用，治理思想适用） | 弹性 KV↔workspace 再平衡；FTW 预打包加载 | cherry-pick 思想 | 1 人周（FTW）/ 2–3 人周（弹性） | **中高** |
| 4 KV cache | 部分 | 语义锚点 checkpoint（多轮 agent TTFT） | 借鉴思路自研 | ~2 人周 | 中（视 agent 负载占比） |

---

## 3. 移植建议与行动项

1. **P0（决策性，1–2 人日）**：用现有 DSV4-Flash routing trace 离线回放"跨步窗口专家级聚合覆盖率"（窗口 w=2/4/8/16 步，聚合粒度=专家而非专家集合）。FreeToken 的 miss 数据（11% 缓存容量下专家 miss 39%，即专家级局部性显著）是我们所缺的最后一块可行性证据。若 w=4 时 per-expert 平均 M 增益可观，方案 1 以"多步窗口 grouped GEMM"形式复活（而非原教旨 merged-GEMM）；否则彻底关闭。
2. **P1（1 人周）**：FTW 式权重预打包 + direct I/O 并行加载，缩短 4 节点 TP4 冷启动（参考其"填满才 pin"细节）。
3. **P1.5（2–3 人周，PoC）**：workspace↔KV 页弹性再平衡（safe point 重建，"内存只影响性能不影响正确性"原则），服务 W4A4 主线在不同负载形态下的内存压力。
4. **P2（视负载画像，2 人周）**：若 agent/多轮占比高，语义锚点 prefix-cache 增强。
5. **不做**：专家按需载入/LRU 驻留（UMA 无瓶颈）、CPU 专家分流（无残余带宽红利）、引入 FreeToken 运行时本身（单卡边缘定位，与 TP4 集群栈不匹配）。

---

## 4. 引用 URL

- 论文：https://arxiv.org/abs/2608.16157 （HTML 全文：https://arxiv.org/html/2608.16157v1）
- GitHub：https://github.com/FlashML-org/FreeToken （Apache-2.0）
- 官网/下载：https://www.flashml.ai/
- 论文深度摘要（Emergent Mind）：https://www.emergentmind.com/papers/2608.16157
- 机器之心报道（腾讯新闻转载）：https://news.qq.com/rain/a/20260822A09NNU00
- 社区评论：https://ai-beat.github.io/news/2026/08/freetoken-edge-moe-serving ；https://dev.to/breachprotocol/a-753-billion-parameter-model-ran-on-a-single-workstation-gpu-bh6
- 微博速报（含 22–25/39.3/14.9 tok/s 数字）：https://weibo.com/2194035935/5334653265252306

**口径标注**：§1.1–1.3 数据均为官方论文/README 实证；§2 各项评估除标明"官方实证"外均为本人基于我方栈特征的推断；社区媒体数字（微博等）与论文一致时以论文为准。

---

# §P0 决定性实验结果：跨步窗口专家级聚合覆盖率离线回放（2026-08-22 续）

**目的**：判定方案 1（merged-GEMM 热桶，已 e2e No-Go）是否以"多步窗口聚合"形式复活。
**数据源**：`01:/tmp/_routea_work/routing_capture_gsm8k.jsonl`（已拉取至本地 `_rex/`）；分析脚本 `_rex/analyze_routing7.py`（主实验）、`analyze_routing8*.py`（稳健性检查），原始输出 `_rex/routing7_out.txt`、`routing7b_out.txt`、`routing8*_out.txt`。

## P0.1 数据质量与口径校准（先于一切结论）

- **文件构成**：896 records 总计；**前 16 条为 warmup 退化记录**（全 token 同一路由的合成探针，4 prompt × 4 layer，占 3.2% token）——**已剔除**，否则会虚增 set 覆盖率。剔除后 **220 个真实 GSM8K 8-shot prompt × 4 层**，246,755 tokens/层，prompt 均长 1122 tokens。
- **层假设验证**：record i 的层 = i%4（rec0-3 为同一 prompt 的 layer0-3，首 token 路由核对一致）；报告 layer0（hash）与 layer3（dense）两个代表层。
- **无共享前缀伪影**：真实 prompt 间无确定性共享路由前缀（vs 首条公共前缀中位=0）；相邻 prompt 仅共享 ~126 token 头部（模板头，占比 11%）。**§2.1 担心的"8-shot 模板跨 prompt 复用推高 set 覆盖"伪影不存在**——观测到的重复来自内容级路由集中，而非 prefix-cache 可吃掉的确定性前缀。
- **步进模拟**：每层按采集顺序拼接 token 流（连续打包多 prompt，对应生产连续批调度），按 **4096 tokens/step** 切分（threshold 4096 已采纳，= 当前 max-num-batched-tokens 上限）；每层 60 步。窗口 w=1/2/4/8/16 个连续 step，主口径滑窗 stride=1（对应生产连续延迟流水），附非重叠批次口径。
- **采集范围局限（如实标注）**：mini 模型（真实 router 权重构建的 4 层）、prefill-only、GSM8K 评测流量、220/1319 prompts（16.7%）、仅 2 个代表层。decode 阶段与 agent 多轮流量不在本数据覆盖内。

## P0.2 主结果

**M_e = 窗口内每 expert 收到的 token 数**（B12X 单步 M_e≈96 为基准；T_step(4096)=960ms 实测见 threshold-retest-2026-08-22）。

| w | tok/win | hash: M_e mean/med | assign≥384 / ≥768 | dense: M_e mean/med | assign≥384 / ≥768 | 延迟代价(延迟流水) |
|---|---------|--------------------|--------------------|---------------------|--------------------|--------------------|
| 1 | 4,096 | 96 / 72 | 12.6% / 0.0% | 106 / 50 | 34.0% / 16.3% | 0 |
| 2 | 8,192 | 192 / 144 | 32.5% / 12.6% | 205 / 96 | 59.1% / 33.9% | +0.96s |
| 4 | 16,384 | 384 / 286 | **68.1% / 32.6%** | 401 / 185 | **81.3% / 58.9%** | +2.88s |
| 8 | 32,768 | 768 / 571 | 91.0% / 68.1% | 792 / 365 | 93.6% / 81.3% | +6.72s |
| 16 | 65,536 | 1536 / 1133 | 96.2% / 91.0% | 1575 / 712 | 97.2% / 93.7% | +14.4s |

（assign 口径 = ΣM_e(≥th)/ΣM_e，即吞吐加权——大 expert 承担更多工作量，是经济效益的正确口径；per-expert 非加权口径见 P0.4）

**四问逐一回答**：

1. **M_e 增长曲线**：均值**严格线性 96w**（hash w=1 均值 96=均匀理论值；dense 105w，重尾 max 42K@w=16 显示 sink expert）。**w=4 → 68–81% 的 MoE GEMM 工作量进入 M_e≥384；w=8 → 68–81% 工作量进入 M_e≥768（B12X/W4A4 算力甜区）**。斜率没有衰减——窗口扩 M 没有"局部性耗尽"问题，本质是 4096 步进下路由近均匀 + 重尾集中。
2. **distinct experts 饱和**：**单步即饱和**——hash 每步 256/256 全活跃、dense 231/256，相邻步复用率 96–100%（Jaccard 93–100%）。这**复证了 FreeToken 的"prefill 摧毁稀疏性"论断**：prefill 粒度上专家级"局部性"无意义（全集饱和），权重读流量在 UMA 常驻场景本来也不构成瓶颈。FreeToken 的 LRU 局部性故事是 decode 现象，本采集（prefill-only）不可测。
3. **N-merge（set 级）可行性残留**：
   - dense 层 w=1 set≥64 覆盖 = **0.0%**——原死刑判决（单步组合频率中位=1）在 4096 步进口径下**精确复证**。
   - 窗口确实"救活"了 set 覆盖的数字：w=4 时 set≥64 = 80.4%（hash）/ 64.5%（dense），w=16 时 dense ≥256 也达 64.2%。**但分析上这无价值**：对任何 set 桶，expert 粒度的 M_e 恒 ≥ set 桶频率（grouped GEMM 按 expert 聚合天然包含 set 聚合），且两者算术强度相同（M×6d 与 6×(M×d) 的 FLOPs/权重元素均 = M）——**set 级 N-merge 相对 expert 级 grouped GEMM 无任何额外 M 增益**，仅剩 activation 读摊销与 kernel 融合的边际收益（M≥384 后可忽略）。
   - **结论：原教旨 N-merge/merged-GEMM 死透，永久关闭；窗口的价值全部由 expert 粒度 grouped GEMM 承接。**
4. **延迟代价**（T_step=960ms 实测）：延迟流水下稳态吞吐不变（MoE 与后续步 attention 重叠），代价是 **TTFT +（w−1)×T_step**：w=4 → **+2.9s**，w=8 → +6.7s，w=16 → +14.4s。对评测/批处理 prefill（PR tok/s 口径）零影响；对交互式长 prompt prefill，w=4 边界可接受、w=8 起不可接受；**decode 阶段 ITL×w，任何 w>2 均不可接受——本方案仅限 prefill/batch 路径**。挂起激活内存 ≈ (w−1)×4096×hidden×2B/层 ≈ 数百 MB，可忽略。

## P0.3 判定

**有条件复活——但复活形态不是 merged-GEMM，而是"跨块 MoE 聚合"（deferred MoE：MoE GEMM 的 M 与调度块 4096 解耦，缓冲 w 块激活后按 expert 粒度 grouped GEMM 一次执行）**：

- ✅ 团长判定阈值逐项对照：w=4 ≤ 4 ✓；M_e≥384（吞吐加权口径）68.1%/81.3% ✓（但 per-expert 中位口径 286/185 未达——诚实标注：中位 expert GEMM 仍属带宽型，甜区主要靠重尾 expert 贡献）；延迟 +2.9s 仅 prefill 路径 ✓、decode ✗（范围限定）。
- ✅ 与 W4A4 主线（方案 2）直接协同：W4A4 需要 M≥768 进算力甜区——单步 4096 给不了（M_e≈96），w=8 窗口给到 68–81% 工作量 ≥768。这是 W4A4 从"kernel 就绪"走向"e2e 兑现"的 M 供给方案。
- ❌ N-merge/set 级桶：永久关闭（P0.2-3）。
- ⚠️ **先做一个更便宜的对照实验再立项**：threshold-retest 已指出"M>4096 需先抬 `--max-num-batched-tokens`"（激活峰值/KV 预算变化，workspace 死结已解）。**直接抬 budget 到 8192/16384 与"保持 4096 块 + w=4/8 延迟 MoE"是同一 M_e 增益的两条路径**——前者零调度改造但抬激活峰值/粗化 TTFT 粒度，后者保持细粒度块（平滑 TTFT、attention 仍 4096 步进）但需 scheduler 改造（MoE 延迟缓冲 + 层内流水）。**建议：先测 budget 8192/16384（约 0.5–1 人日，纯配置实验），若激活峰值/慢轮问题把 budget 顶死，再立"跨块 MoE 聚合"工程项（预估 2–4 人周：B12X 前向缓冲 + vLLM scheduler 延迟点 + per-layer 流水）**。
- 工作量量级：预算上探实验 0.5–1 人日（决策闸门）；若走延迟 MoE 路线 2–4 人周（vLLM 0.26.1 fork 内改造，无上游依赖）。

## P0.4 附：per-expert 非加权口径（诚实对照）

| w | hash ≥384 / ≥768 | dense ≥384 / ≥768 |
|---|---|---|
| 4 | 34.8% / 9.7% | 30.5% / 14.6% |
| 8 | 65.1% / 34.8% | 48.7% / 30.2% |

含义：即便 w=8，仍有 35–51% 的活跃 expert 处于 M<384 的小 GEMM（grouped GEMM 单 launch 内以小 tile 处理，开销可控但甜区不覆盖它们）；吞吐加权口径（P0.2 表）显示这些小 expert 只承担 7–19% 的总工作量。**甜区覆盖率的两口径差距本身就是"重尾路由"的量化画像**，与 merged-GEMM 研究期的 59K distinct sets 结论同源。

## P0.5 口径与风险标注

- 以上全部数字：**离线回放实证**（220 真实 prompt、2 代表层、GSM8K 8-shot prefill-only）；未在 GPU 上做过端到端验证，M_e→实际 kernel 效率的映射依赖 threshold-retest 的 Roofline 结论（+15–20% 预测、4096 档兑现 2/3）。
- 负载外推风险：GSM8K 评测流量的路由集中度（top32 sets 覆盖 hash 62%/dense 21%）可能高于生产混合流量；agent 多轮/decode 流量未覆盖。建议立项时同步在生产行为档（pr-de 负载画像）上回放一次同款分析（脚本已就绪，换数据文件即可）。
- 与 FreeToken 的关系收束：本实验**证实**了 FreeToken 论文的两个论断（prefill 路由并集稠密；miss 服务方式比 miss 预测更重要——我们没有用任何预测就拿到了线性 M 增长），FreeToken 对本实验的贡献是"把问题从'预测路由'重新框定为'组织已发生的路由'"。

