# lm_head FP8 立项文档（2026-08-23）

- **作者**: 阿奇（Archi）· 系统架构师（architect-1）
- **状态**: Proposed（待 P0 拆账/微基准后转 Accepted）
- **日期**: 2026-08-23（本地，纯只读调查 + 立项设计，不碰 GPU/集群）
- **上游输入**: opt-routeb-fp8-2026-08-23.md（F0-F5 路径、lm_head 池内唯一 BF16 [129280,4096] 实测、decode ÷2 省 0.5ms/step 或 ÷4 0.7ms/step）、fi017-p0-accounting-2026-08-23.md（P1 lm_head 第二、P0 拆账实测 2.79µs/12.9%）、upstream-check-perf-ceiling-2026-08-23.md（lm_head 唯一同时作用于 decode 带宽墙）、luz031-bench-and-p0-exec-report-2026-08-23.md（P0 拆账实测 + Session A 基准）、bench-regression-attribution-2026-08-23.md（四因素归因）
- **口径标注**: 【实测-源码】= 服务器 checkpoint/容器源码直接验证；【实测】= 本团队既有实测数据；【推算】= 基于 Roofline/形状/FLOPs 推算；【待窗口验证】= 必须由窗口执行
- **纪律**: 只读分析；立项批准后 F0/F1 起由窗口执行

---

## 0. 立项摘要（ADR 风格）

**决策**: 立项 **lm_head 权重 BF16 → FP8（E4M3，routeB FP8 引擎）**，作为 bf16 稠密池性能释放的第二优先节点（P1 第二，shared-FP8 之后）。

**核心理由**:
1. **lm_head 是池内唯一 BF16 节点**【实测-源码】：`head.weight` = BF16 [129280,4096]，TP4 分片后每 rank [32320,4096]。
2. **唯一同时作用于 decode 带宽墙的节点**【推算+实测】：decode 每步全量读权重，M=8 实测 1.18ms/step、M≈96 实测 1.23ms/step（P0 拆账）。
3. **FP8 路径 decode 字节 ÷2（省 ~0.5ms/step @C12）、W4A4 路径 ÷4（省 ~0.7ms/step @C12）**——直接抵消 W4A4 full 的 C12 decode 代价（-8.0%）。
4. **A-quant 每步仅 1 次**（[M,4096]），占比远低于 shared 的 ×43——A-quant delta 风险相对可控。

**不做（本项目范围外）**: 不做 W4A4（FP4）立项（该路径需校准 + 有损量化 + 质量门更重，作为 FP8 之后的"后段"选项保留）；不做 attn 投影（P2，decode 侧小 M 效率反转明确不做）。

**一句话**: lm_head FP8 是"唯一同时打 decode 带宽墙 + prefill GEMM 效率 + 显存"的节点，收益结构完整，但质量门槛最高（logits 直驱采样分布）——**校准 + KL 门必须前置**。

---

## 1. 目标与预期收益

### 1.1 目标

1. **decode 步时削减**（主目标）：C12 decode 步时 -5~6%（FP8 ÷2）或 -7~8%（W4A4 ÷4，后段选项），抵消 LuZ0.3.1 采纳的 W4A4 full 已知代价带（-6~-9%）。
2. **prefill GEMM 效率提升**（次目标）：prefill M=4096 生产形态节点级 3.5-5×（routeB 350T 平台 vs 当前 55-65T）。
3. **显存节省**：FP8 -135MB/rank，W4A4 -190MB/rank（TP4 分片后）。
4. **验证质量门**：KL(量化‖bf16) 分布门 + 困惑度 + greedy 逐字 4/4 + 温度采样抽验——**证明量化对 token 分布无退化**。

### 1.2 预期收益表（定量）

| 收益项 | FP8（÷2） | W4A4（÷4，后段） | 依据 | 强度 |
|---|---|---|---|---|
| decode 字节/rank/步 | 0.27GB → 0.135GB | → 0.07GB | 【实测-源码】BF16 0.27GB；【推算】÷2/÷4 | 【推算】 |
| **C12 步时节省（M≈96）** | **~0.6ms/step（~5%）** | ~0.92ms/step（~7.7%） | P0 拆账 lm_head M≈96 = 1.23ms；C12 步时 ~12ms 口径 | 【实测+推算】 |
| **C1 步时节省（M=8）** | **~0.59ms/step（~1.4%）** | ~0.89ms/step（~2.2%） | P0 拆账 lm_head M=8 = 1.18ms；C1 步时 ~41ms 口径 | 【实测+推算】 |
| prefill GEMM 效率 | 节点级 3.5-5×（55-65T → 200-300T，N=32320 大 N 乐观） | 同左（W4A4 更高） | routeB 350T 平台 + P4 FP4 外推 | 【推算】 |
| 显存 | -135MB/rank | -190MB/rank | 【实测-源码】0.27GB→0.135GB | 【推算】 |
| **C12 抵消 W4A4 代价** | **+5% 步时 对 -8.0% 代价 → 抵消 62%** | +7.7% 对 -8.0% → 抵消 96% | luz031-deployment DE C12 -8.0% | 【实测+推算】 |

> 诚实标注：P0 拆账 ms/step 为纯 GEMM 计算时间（CUDA event 计时，不含调度/overlap）；decode 步可能存在 lm_head 与其他算子部分重叠，E2E 实际改善可能低于纯 GEMM 口径。实际以 F4 窗口 A/B 实测为准。

### 1.3 对 Session A 低并发/Agent 中位回退的理论改善量

| 场景 | Session A 实测中位 | lm_head FP8 理论改善 | 改善后理论值 | 说明 |
|---|---|---|---|---|
| C1 单流（M=8） | 73.9 | **+1.4%**（0.59ms/41ms） | ~75.0 | 改善量有限——**中位回退主因是 F4 不稳定（-15~-20%），非步时** |
| Agent 工具调用（M≈8） | 84.2 | **+1.4%** | ~85.4 | 同上 |
| Agent 平均（5 场景） | 70.4 | **+1.4%** | ~71.4 | 同上 |
| C12 并发（M≈96） | 349.3 | **+5%** | ~366 | 高并发改善显著，抵消 W4A4 decode 代价 |

**排程含义**：lm_head FP8 对 Session A 低并发中位回退的恢复能力有限（~1.4%），**不能指望它修复不稳定波动**；低并发中位回退的治理应依赖重测协议 + E3/E5 环境闭环（见 env-random-factors-tracking 报告）。lm_head FP8 的真实价值在 **C12/高并发（+5%）与 prefill GEMM 效率**。

---

## 2. 任务分解（F0-F5）

### F0 契约核对（纯离线，0.5 天）

**目标**: 确认 BF16→FP8 转化的契约与生产内存布局，产出 golden 与布局清单。

| 子项 | 内容 | 判定 |
|---|---|---|
| F0.1 | **head.weight 转化**: BF16 [129280,4096]（TP4 分片 [32320,4096]）→ FP8 E4M3，K-contiguous（routeB B 操作数契约）；scale 布局选型：**per-channel（每输出 channel 一个 scale，[N,1]）或 128×128 块（同 shared）**——需 F0.3 按 routeB kernel 约束定 | golden: 转化后 dequant == BF16 参考（rel<1e-6） |
| F0.2 | **scale 布局**: routeB 期望每行/每 32-K 组 E8M0（[N, ceil(K/32)]），tile_k%128=0 约束；若选 per-channel 需确认 routeB 支持或走最小 staging 适配 | 布局清单 + 转换脚本 |
| F0.3 | **生产内存 weight 实际布局**: `process_weights_after_loading` 是否重排（Cutlass 路径 `B.T` 处理）；确认 [K,N] vs [N,K] 与零拷贝 view 可行性 | 逐层核对内存布局【待窗口验证】 |
| F0.4 | **CPU golden**: 转化 + 反量化 == 原始 BF16（rel<1e-6） | golden PASS |

**关键风险**: 生产内存布局与假设不符（需回退 view 方案为 staging 复制，仍可接受但损失零拷贝收益）。

### F1 kernel 微基准 go/no-go（GPU 一次性容器，0.5-1 天）

**目标**: 验证 routeB FP8 引擎在 lm_head 形状（N=32320 大 N）下是否优于当前 cutlass FP8/BF16。

| 子项 | 内容 | 判定 |
|---|---|---|
| F1.1 | routeB FP8 same-dtype（FP8×FP8+E8M0+sf_vec=32）× lm_head 形状 × M∈{8,96,512,1024,4096}；KO + E2E（含 FP8 A-quant 适配器）vs 当前 cutlass | 数据表 |
| F1.2 | **go/no-go 门**: **E2E(M=4096) ≥ 1.1× 当前 cutlass**；A-quant delta 占比 < GEMM 增益；M=8/96 decode 侧 ≥ 中性 | go/no-go 裁定 |
| F1.3 | 与 shared-FP8 微基准（opt-routeb-fp8 W3）**共享一次 GPU 窗口**（一次性容器 <1GB 显存纪律） | 复用调度 |

**关键 go/no-go 决策点**: routeB FP8 E2E(M=4096) 不达 1.1× 当前 cutlass，或 A-quant delta 吃光 GEMM 增益 → **lm_head FP8 降级为设计储备**，回退到 FI 0.6.17 小 batch 路径或维持 BF16（沉没成本 = F0/F1 半天窗口）。

### F2 零拷贝/最小 staging 适配器 + golden（GPU 容器，1-2 天）

| 子项 | 内容 | 判定 |
|---|---|---|
| F2.1 | `RouteBFp8BlockScaledMMKernel` 接入 lm_head：A-quant（group=32）+ scale 布局适配 + swizzle + routeB FP8 GEMM | rel_err ≤ 1e-2 vs BF16 参考 |
| F2.2 | 零拷贝路径（from_dlpack 直配，无 payload 复制）实测确认；若 F0.3 布局不符 → 最小 staging（一次加载时转换） | 零拷贝 or staging 实证 |
| F2.3 | golden vs 当前 Fp8LinearMethod/BF16 | golden PASS |

### F3 集成 L1（无生产，2 天）

| 子项 | 内容 | 判定 |
|---|---|---|
| F3.1 | 并入 kernel 候选（env 门控 `VLLM_LMHEAD_FP8=1`）；off 路径 byte-equivalent | off 路径逐字等价 |
| F3.2 | env checker 四机同步 + cudagraph 捕获 + 启动核验 | L1 全过（0 ERROR） |
| F3.3 | 数值包络：on 路径输出在 BF16 带内（logits rel 差阈值） | 数值包络 PASS |

### F4 窗口 A/B（生产窗口，0.5-1 天）

**目标**: 对照 LuZ0.3.1 基线，验证 decode 步时削减与并发收益，质量门全过。

| 门 | 判据 | 参考值（LuZ0.3.1 实测） |
|---|---|---|
| 性能门 PR 四档 | ≥-3%（目标 +0.5~1.5%） | 2950.5 / 2943.6 / 2834.2 / 2550.0 |
| 性能门 C6/C12 | ±5% | 3057 / 3056（并发口径；**重点看 C12 decode 改善**） |
| 性能门 DE C1/C12 step_eff | C1 ±5%；C12 ≥-3%（目标 +3~5%） | 18.2 / 80.2 |
| **质量门 golden 4/4** | greedy 稳定 4 prompt 逐字一致（fox_repeat/count/code/list） | vs LuZ0.3.1 参考 |
| **质量门 KL 门** | KL(FP8 量化 logits ‖ BF16 logits) < 阈值（校准集） | 待 F0 校准确定基线 |
| 质量门 困惑度 Δ | ≤0.05 | 待校准 |
| 回归观察门 | 日志 ERROR/Traceback=0；needle 64K 3/3 | — |

### F5 质量门 + 发布（1-2 天）

- 扩展质量：golden 4 逐字 + logprob 对齐 + 困惑度 + 接受率不降 + 温度采样抽验 + needle 抽验。
- 回滚锚点：`.bak` 快照 + head-first 重建核验（沿用 luz031 §2 教训：恢复必须用注入前快照，重建后核版本项）。
- 发布后维持 env 门控（默认 on 需团队裁定；初始 off 灰度）。

---

## 3. 验证门与判定阈值（引用既有口径）

| 门 | 阈值 | 引用口径 |
|---|---|---|
| F1 go/no-go | E2E(M=4096) ≥ 1.1× 当前 cutlass；A-quant delta < GEMM 增益 | opt-routeb-fp8 §3.1（shared 同款门槛） |
| F2 golden | rel_err ≤ 1e-2；payload 零拷贝实证 | opt-routeb-fp8 §3.1 |
| F3 off 等价 | byte-equivalent；0 ERROR | opt-routeb-fp8 §3.1 |
| F4 PR | 四档 ≥-3%（目标 +0.5~1.5%） | fi017 §1.5 三门口径 |
| F4 DE | C1 ±5%；C12 ≥-3%（目标 +3~5%） | fi017 §1.5 |
| F4 质量 | **KL 门 + greedy 4/4 + 困惑度 Δ ≤0.05 + 温度采样抽验** | upstream-check §2.3 P2 门 |
| F5 发布 | 全门 PASS + 回滚锚点就绪 | — |

**KL 门设计（关键）**：校准集上对比 FP8 量化后 logits 分布 vs BF16 logits 分布，KL 散度 < 预设阈值（建议先在 CPU 校准集测 BF16 vs BF16 的 KL 基线，再设 2-3× 基线为门限）；**校准必须 per-channel**（upstream-check §2.3）。贪心输出最敏感 → greedy 逐字 4/4 为硬门。

---

## 4. 依赖与前置

| 前置 | 状态 | 说明 |
|---|---|---|
| **P0 拆账（已完成）** | ✅ 【实测】 | lm_head prefill 2.79µs/12.9%（M=4096）；decode M=8 1.18ms/15.8%、M≈96 1.23ms | 
| **FP8 转化工具链** | ⚠️ 部分就位 | vLLM `Fp8LinearMethod`/`process_weights_after_loading` 已存在（shared 在用）；但 lm_head 当前是 BF16，需新增转化路径（离线转换 or 加载时转换）；scale 布局决策待 F0 |
| **routeB FP8 kernel** | ✅ 源码级就位 | dispatch 表 `(FP8,FP8,E8M0,32)→MmaMXF8Op` 存在；运行级微基准未测（F1） |
| **质量门流程** | ⚠️ 待建 | greedy 稳定 prompt 集已定义（4/4）；KL 门基线待 F0 校准 |
| **A-quant 适配器** | ⚠️ 与 shared-FP8 共用 | 建议与 shared-FP8 F1/F2 绑定投入（共用前置工程） |
| **FI 0.6.17（U1）** | Go（有条件） | 其 W4A16 小 batch 路径可作为 lm_head decode 侧折中选项；与 lm_head FP8 可叠加 |

---

## 5. 窗口估算（每阶段）

| 阶段 | 预计 | 窗口类型 | 前置 |
|---|---|---|---|
| F0 契约核对 | 0.5 天 | 纯离线 | 无 |
| F1 kernel 微基准 | 0.5-1 天 | GPU 一次性容器（与 shared-FP8 W3 共享） | F0 |
| F2 适配器 + golden | 1-2 天 | GPU 容器 | F1 数据为正 |
| F3 集成 L1 | 2 天 | 无生产（测试容器） | F2 |
| F4 窗口 A/B | 0.5-1 天 | 生产窗口 | F3 |
| F5 质量门 + 发布 | 1-2 天 | 生产窗口 | F4 |
| **合计** | **~5.5-7.5 天（日历，含窗口排队）** | — | — |

**窗口编排建议**：F0 纯离线可立即启动；F1 与 shared-FP8 W3 共享一次 GPU 微基准窗口；F4/F5 需生产窗口（**必须先采纳重测协议 P1-P5，否则测量噪声 ±8-13% 淹没 ~5% 收益**）。

---

## 6. 风险

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | **精度损失 KL（最高）**：logits 直接决定采样分布，贪心输出最敏感 | **高** | 校准 per-channel + KL 门前置 + greedy 逐字 4/4 硬门 + 温度采样抽验 |
| R2 | **质量门最高**：与 shared 相比，lm_head 数值风险更高（直接作用 token 分布） | 高 | F0 校准集先行；KL 门阈值基于 BF16 基线设 |
| R3 | **A-quant delta 吃光 GEMM 增益**（P4 教训迁移） | 中 | F1 硬 go/no-go：E2E ≥ 1.1× 且 A-quant delta < GEMM 增益；每步仅 1 次 A-quant（优于 shared ×43） |
| R4 | **运行级零拷贝未实证**（from_dlpack 对齐/divisibility/编译路径） | 中 | F0.3 布局核对 + F2 golden 门；失败回退最小 staging |
| R5 | **N=32320 大 N 形状 routeB 效率未知**（P4 只测 N=2048/12288） | 中 | F1 覆盖 N=32320 形状；若大 N 效率不足 → 降级为 decode-only 或设计储备 |
| R6 | **decode 步重叠效应**：lm_head 可能与其他算子部分重叠，E2E 改善低于纯 GEMM 口径 | 中 | F4 实测为准；收益表标注纯 GEMM 口径 |
| R7 | **与 shared-FP8 竞争窗口/资产**：A-quant 适配器共用 | 低 | F1/F2 绑定投入；F0 并行 |

---

## 7. 与 shared-FP8 / W3 的排程关系

### 7.1 排序（P0 拆账后维持）

| 排序 | 节点 | 理由 | 状态 |
|---|---|---|---|
| **1** | **shared → routeB FP8**（首发） | 资产零缺口、M=4096 甜点、无损 payload/scale（无校准门）、全 token | opt-routeb-fp8 修正后仍第一 |
| **2** | **lm_head → FP8（本项目）** | 唯一 BF16 节点 + decode 字节削减成立 + A-quant 每步 1 次；**P0 后与 shared 差距显著收窄，反超概率最高** | 本立项 |
| 3 | attn 投影 | FLOPs 份额最大但仅 prefill 半场、decode 明确不做 | P2 |

### 7.2 与 shared-FP8 的协同/复用

1. **A-quant 适配器共用**：shared-FP8 与 lm_head-FP8 的 A-quant（group=32 + E8M0 + swizzle）是**共用前置工程**，建议与 shared F1/F2 绑定投入——先 shared 后 lm_head 顺次复用。
2. **F1 微基准共享窗口**：lm_head F1 与 shared W3 共享一次 GPU 微基准窗口（一次性容器，形状不同但引擎同族）。
3. **F0 可并行**：lm_head 契约核对（F0）与 shared F0 均为纯离线，可并行。

### 7.3 条件触发

- P0 拆账实测 lm_head decode 墙份额 >20%（当前 15.8%）或 prefill µs 份额 >25%（当前 12.9%）→ **lm_head 可上提首发**（质量门工作量更高，仍建议 shared 首发做快赢 + 资产复用）。
- F1 微基准 E2E < 1.1× → **lm_head FP8 降级为设计储备**，shared 顺势前移；lm_head 走 FI 0.6.17 小 batch 路径折中。
- 若 W4A4（FP4）后段路径最终采纳（需校准 + 有损量化 + 更重质量门）→ 在 FP8 稳定后另立小项目，复用 F2/F3 资产。

### 7.4 与 FI 0.6.17（U1）的关系

- FI 0.6.17 的 **W4A16 小 batch 张量核 decode 路径**（#4255）作用于现生产 W4A16 decode 段，与 lm_head FP8 **正交可叠加**——U1 落地后 lm_head decode 侧可选择 FP8（本项目）或 W4A16 小 batch（更低风险折中），A/B 裁定。
- FI 0.6.17 的 NVFP4 W4A4 精度修复（#4285/#3932）是把 C12 decode 中性度拉回正值的头号候选，与 lm_head FP8 叠加可进一步改善 C12。

---

## 8. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| 【实测-源码】 | `head.weight` = BF16 [129280,4096]（池内唯一 BF16）；TP4 分片 [32320,4096]；routeB FP8 dispatch `(FP8,FP8,E8M0,32)→MmaMXF8Op`；vLLM Fp8LinearMethod 现役 |
| 【实测】 | P0 拆账：lm_head prefill 2.79µs/12.9%（M=4096）；decode M=8 1.18ms/15.8%、M≈96 1.23ms；C1 带宽墙 7.49ms/step；C12 step_eff 80.2（-8.0% vs B1）；LuZ0.3.1 三门数字 |
| 【推算】 | decode ÷2 省 ~0.5ms/step（C12 ~5%）、÷4 省 ~0.7ms/step（C12 ~7.7%）；C1 FP8 改善 ~1.4%；prefill 节点级 3.5-5×；显存 -135MB/rank（FP8）/-190MB/rank（W4A4）；C12 抵消 W4A4 代价 62%（FP8）/96%（W4A4） |
| 【待窗口验证】 | F0 生产内存布局；F1 微基准（N=32320 形状）；运行级零拷贝 golden；F2-F5 集成与窗口 A/B；KL 门阈值校准 |

**诚实声明**：
1. 本项目收益以 P0 拆账实测（纯 GEMM 口径）为基数，decode 步重叠效应未计入——E2E 收益以 F4 实测为准。
2. lm_head FP8 对 Session A 低并发中位回退的理论改善仅 ~1.4%，**不能修复不稳定波动**；判定本项目生效的 A/B 窗口必须先采纳重测协议（30 轮/交错 A/B）。
3. 质量门槛为全池最高（logits 直驱采样分布），**校准 + KL 门先行**为硬前置，不可跳过。
4. 所有 [推算] 数字在 F1/F4 实测后须替换为实测值。

---

## 9. 引用索引

- 本项目关键实测源: `opt-routeb-fp8-2026-08-23.md`、`fi017-p0-accounting-2026-08-23.md`、`upstream-check-perf-ceiling-2026-08-23.md`、`luz031-bench-and-p0-exec-report-2026-08-23.md`、`bench-regression-attribution-2026-08-23.md`、`env-random-factors-tracking-2026-08-23.md`（本团队同日产出）
- P0 拆账原始数据: `_luz031_official_bench/data/p0/p0_accounting_data.md`、`p0_micro_M4096.json`

*本报告由工程保障团队（系统架构师 architect-1）生成；立项批准后 F0 可立即离线启动，F1 与 shared-FP8 W3 共享窗口；最终采纳请由人类工程负责人结合 F1 微基准与 F4 窗口 A/B 裁定。*
