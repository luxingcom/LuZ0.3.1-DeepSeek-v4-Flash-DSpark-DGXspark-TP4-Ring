# FP8（E4M3）量化对输出质量影响评估 — lm_head 与 shared experts（2026-08-23）

- **作者**: 阿奇（Archi）· 系统架构师（architect-1）
- **任务**: 调研 lm_head 与 shared experts 采用 FP8（E4M3）对**输出质量**的影响（机制 + 敏感度 + 校准需求），设计验证门（F2/F4 用），给出下一步方案（含任务排序建议）。纯只读分析，不碰 GPU/集群。
- **上游输入**: lmhead-fp8-project-2026-08-23.md（lm_head FP8 立项 F0-F5/R1）、opt-routeb-fp8-2026-08-23.md（shared 走 routeB FP8、现役 Fp8LinearMethod→CutlassFp8BlockScaledMM）、fi017-p0-accounting-2026-08-23.md + `_luz031_official_bench/data/p0/p0_accounting_data.md`（P0 实测拆账）、quality_gate.py（现役质量门）、bench-regression-attribution-2026-08-23.md（KL 门建议/假阳性史）、upstream-check-perf-ceiling-2026-08-23.md（§2.3 P2 门）、opt-objective-research-2026-08-23.md（FP8 修正/质量门先例）、mtp-tuning-2026-08-22.md（draft/接受率机制）、a3-hybrid-slim-design-2026-08-23.md（G1 显存账）、arstall-production-closure-2026-08-23.md（CUMEM=0）
- **口径标注**: 【实测-源码】= 服务器 checkpoint/容器源码直接验证；【实测】= 本团队既有实测数据；【推算】= 基于数值格式/形状/既有数据的推算；【上游实证】= 上游仓库核验；【待窗口验证】= 必须由窗口执行
- **纪律**: 只读分析；所有【推算】项在质量门工具链 + F1 微基准后替换为实测

---

## 0. 一页结论

1. **机制**：E4M3 是 3-bit 尾数 + 8-bit 指数（联合 E8M0 幂次块 scale），逐元素最大相对舍入误差 2^-4=6.25%（round-to-nearest 半 ULP）、RMS ≈3.6%【推算-格式性质】。块缩放误差**有界、块内相关、可带偏置**（非独立零均值）——lm_head 的 logits 是 K=4096 项内积，误差**不因大 K 自动相消**；推算 logits 绝对误差为 logits 幅值的 ~0.1-2%【推算】。**风险增量的本质**：shared/routed MoE 已是 FP8 E4M3 block-scaled 且质量门过（shared golden 4/4【实测】）——但那是**中间层**，误差被后续非线性/43 层部分吸收；**lm_head 是最终层、logits 直接驱动采样、且从无损 BF16 首次引入有损量化**——无下游吸收、无既有 FP8 先例覆盖。这是 R1/KL 风险的真正来源。
2. **敏感度**：**greedy 最敏感**（argmax 翻转集中在近并列 token；repo 已有 reason/zh 近平票翻转先例【实测】）→ greedy 4/4 硬门 + 包络兜底必须；温度采样敏感度中（逐 token KL 小，用 KL 门 + logprob 漂移度量）；**top-k/top-p 会放大 logits 尾部误差**（截断边界恰是近并列区，边界 token 排序翻转直接改变候选集）→ 应加 top-p/top-k 抽验；**MTP draft 验证用 target logits 判接受**——量化后 p_target 微扰、边界 token 接受翻转，预期 tokens/step 漂移 <1-2%【推算】，用既有 DE 接受率门实测。
3. **校准**：**权重变换数据无关（max-abs 幂次 scale，无需校准集）**——这是 FP8 E4M3 相对 W4A4/FP4 路径（需校准+有损）的核心优势【实测-源码判定与 opt-routeb-fp8 §2.3 一致】。routeB 契约布局 `[N, K/32] E8M0` **同时比 per-channel [N,1] 和 shared 的 128×128 块更细** → **采用 routeB 原生布局 = 零 staging + 更细粒度 + 精度最优，三重占优**（lm_head 相对 shared 的隐藏顺风）。**KL/困惑度门需要的是"参考数据集 + BF16 噪声底"而非"校准"**——先测 BF16 vs BF16 噪声底，设 2-3× 为门限。
4. **验证门（F2/F4）**：greedy 4/4 硬门 + KL 门（参考集、BF16 噪声底 2-3×）+ 困惑度 Δ≤0.05 + 温度采样/top-p 抽验 + 长上下文 64K 抽验 + DE 接受率门 + off 等价。**质量门工具链先行落地（零 GPU）**是 lm_head 硬前置，shared-FP8 F4 复用。
5. **下一步**：**shared-FP8 窗口 A/B 先行、F0 顺手带 lm_head 契约核对**（shared=引擎置换无新量化、风险最低、先例可迁移；lm_head=新量化质量门最高，须质量门工具链 + KL 基线先行）。与 CUMEM=0（AR stall 缓解 env）/G1（native 派生显存账）**并行正交**；但 FP8 A/B 窗口**必须先把 CUMEM=0/重测协议纳入测量基线**，否则 ±8-13% 噪声淹没 ~5% 收益。

---

## 1. FP8 量化对输出质量的影响机制

### 1.1 E4M3 格式精度（格式性质）【推算】

| 项 | 值 | 说明 |
|---|---|---|
| 位宽 | 1s + 4e + 3m | 尾数 3-bit |
| binade 内 ULP | 2^-3（如 [1,2) 内步长 0.125） | 3 bit 尾数 |
| 最大相对误差（RNE） | 2^-4 = **6.25%** | 半 ULP |
| 均匀舍入 RMS | ≈ 6.25%/√3 ≈ **3.6%** | 相对量 |
| 动态范围（联合 E8M0） | ~2^±127 | 8-bit 指数纯幂次 scale |

- **动态范围不是问题，尾数精度才是**。对比 BF16（7-bit 尾数，相对误差 0.4%）→ 单元素口径 FP8 E4M3 比 BF16 粗糙 ~16×（4 bit 尾数差）。
- 但"单元素"口径会误导：块缩放 + 内积的**实际**误差见 §1.2/§1.3，不是线性乘 16×。

### 1.2 块缩放（block-scaled 128×128 + E8M0）的误差分布

- 机制：每块一个 E8M0 scale（= 块内 max-abs 向上取幂次），块内元素量化到 `scale·2^-3` 网格。
- **误差三性（关键）**：
  1. **有界**：|Δ| ≤ scale·2^-4；
  2. **块内相关**：同一块共享 scale → 块内误差同阶、非独立；
  3. **可带偏置**：块内大值主导 scale → 小值元素相对精度受损（绝对步长由大值决定）。
- **routeB 契约布局 `[N, ceil(K/32)]`（每行、每 32-K 一组 E8M0）更细**：K 组 32 vs 128×128 块 K 组 128，每块仅 32 元素 → max-abs 更贴近局部、块内动态范围更小 → **相对精度优于 128×128 块，更优于 per-channel [N,1]（整行 1 个 scale）**【推算-布局粒度】。
- **已知弱点**：块内动态范围大时，小值元素被大值 scale 决定步长 → 相对精度劣化。**不能假设，必须用参考集实测**（KL/困惑度门）【待窗口验证】。

### 1.3 logits 误差的推算模型（lm_head 专属）

```
logits = x[4096] · W^T[4096, 32320]      （TP4 分片后 per-rank [4096,32320]）
量化后: Ŵ = W + Δ,  |Δ_ik| ≤ s_i · 2^-4   （s_i = 该元素所在块 scale）
logits 误差: δy_i = Σ_k x_k · Δ_ik
```

- **有界性**：|δy_i| ≤ 2^-4 · Σ_k |x_k|·s_ik。由 x 幅度与 W 块 scale 的关联，量级 ≈ 2^-4·‖x‖_1·(典型 W 幅度) ≈ **logits 幅值的 ~0.1-2%**【推算】。以生产 logits 幅值 ~10-40 计 → 绝对误差 ~0.01-0.8。
- **不自动相消的核心点**：块内舍入误差是"值的确定性函数"，不是独立零均值随机量 → 可在行/块内**相干叠加**，不会按 1/√K 随机游走式相消。**这与"中心极限直觉"相反**：FP8 权重误差对 logits 的污染是**系统性、有界、可偏置**的，其大小只能靠参考集实测。
- **A-quant 叠加**：routeB w8a8 路径同时量化激活 x（per-token group=32 FP8）。A 误差数据依赖、逐 token 变化，与 W 误差叠加进 logits 噪声。**KL/困惑度门测的是 W+A 联合效应**（正是生产形态，正确）。
- **诚实标注**：0.1-2% 是推算量级，非实测；logits 噪声对**输出文本**的影响通过 argmax/softmax/top-k/投机验证四个暴露面传递（§2），最终判定以质量门实测为准。

### 1.4 与现役 Fp8LinearMethod 数值基线对比（风险增量定位）

**现役基线（FP8 E4M3 数值正确性已有实证）**：
- shared experts 生产走 FP8 E4M3 block-scaled（Fp8LinearMethod → CutlassFp8BlockScaledMMKernel）【实测-源码】；质量门过（golden 4/4 逐字一致，bprime-window/wsdedup-l3 先例【实测】）。
- routed MoE 在 LuZ0.3.1 生产形态为 W4A4（NVFP4），质量门 4/4 + logprob 级 **0.36-0.41%**【实测】（routeb-merged-phasec / routea-w4a4-debug）。
- 结论："FP8 E4M3 block-scaled 本身不可用"的担心不成立——**该格式在本池已有 golden 实证**。

**lm_head 新增风险在哪（三层）**：
| 层 | 差异 | 影响 |
|---|---|---|
| **层级位置** | shared/routed 是**中间层**，误差经 SiLU/gating/后续 43 层去相关、被非线性吸收；lm_head 是**最终投影**，logits 直接进采样 | 无下游吸收 → 误差直接暴露到输出分布 |
| **基线性质** | lm_head 当前是**无损 BF16**（池内唯一 BF16 节点【实测-源码】）；shared 当前**已是 FP8** | shared-FP8 是"引擎置换"（GEMM 换快，不引入新量化）；**lm_head-FP8 是"首次对有损量化到无损节点"**——质量风险结构完全不同 |
| **采样暴露** | logits 噪声直接作用 argmax/softmax/top-k/投机验证 | 每个采样语义都被量化噪声触碰 |

**风险等级裁定**：lm_head-FP8 是池内**质量门最高**节点（与 lmhead-fp8 R1/R2 登记一致）。但**绝对误差预期仍在有界小量级**（§1.3 推算 0.1-2%），不是"量化即毁"，而是**需要比 shared 更重的质量门 + 参考集实测背书**。

---

## 2. 敏感度分析

### 2.1 greedy decode vs 温度采样

| 维度 | greedy（temp=0） | 温度采样（temp>0） |
|---|---|---|
| 暴露面 | argmax 翻转（只关心 top-2 gap 是否 < 噪声） | 全分布经 softmax 平移 |
| 敏感度 | **高**：129280 vocab 长尾存在大量近并列；repo 已有 reason/zh 近平票翻转先例【实测】 | 中：分布展平后单 token 权重变化小，逐 token KL 小 |
| 度量 | 逐字 4/4 硬门 + 包络 top-1 logprob sum drift ≤1% 兜底 + own_stable 交叉复跑 | KL 门 + logprob 漂移 + 采样多样性指标 |
| 预期 | argmax 翻转率 0.1-1%/token【推算，待实测】 | 可接受（KL < 门限即过） |

**关键**：greedy 是"最坏情形暴露"→ 硬门；但 **greedy 4/4 对近平票翻转天然敏感，必须用 own_stable 交叉复跑区分"量化噪声 vs 运行级非确定"**（quality_gate.py 已内置 own_stable【实测-源码】，reason/zh 已除名——历史假阳性教训【实测】）。

### 2.2 top-k/top-p 截断是否放大 logits 尾部误差

- **会放大，机制明确**【推算】：top-k/top-p 在排序/累积概率**边界**截断——**边界 token 恰是近并列区**（logits gap < 噪声幅值）。FP8 噪声使边界 token 排序/概率翻转 → 直接改变候选集成员 → **把 logits 尾部的小误差"放大"为离散的候选集跳变**（比温度采样更敏感）。
- 方向：top-p 比 top-k 略稳（累积概率阈值对小幅扰动不敏感），但边界 token 的进/出仍是离散跳变。
- **验证门含义**：需加 top-p（可选 top-k）抽验，度量 **token 候选集重叠率 / distinct-n / logprob 漂移**，而非只看生成文本。

### 2.3 MTP 投机解码 draft 验证的影响

- 机制【实测-源码（mtp-tuning）】：MTP draft 头提出 draft token，target lm_head logits 计算 p_target 做接受判定（`accept if u ≤ min(1, p_t/p_d)`）。
- 影响拆解：
  - **提议侧不变**：draft 提议来自独立 MTP 头，不受 lm_head 量化影响；
  - **验证侧扰动**：lm_head 量化 → p_target 微扰 → 边界 token（p_t/p_d ≈ 1）接受/拒绝翻转 → **接受率漂移**；
  - 量级【推算】：logits 噪声 ~0.1-0.8 → p_t 乘性扰动 e^{δl} ~1.1-2.2；但接近接受边界的 token 占比小 → **预期 tokens/step 漂移 <1-2%**。
- **验证门**：DE C1/C12 tokens/step（step_eff，接受率归一）±3% 带内——**既有 DE 门直接覆盖**【实测-既有门】。注意：probabilistic draft 运行级非确定（temp-0 下=argmax【实测-源码】），FP8 A/B 必须同 draft 配置、用重测协议。

### 2.4 长上下文 64K

- 机制【推算】：量化误差逐 token 近似 i.i.d.，**不沿上下文累积进权重**（无递归权重），但**采样漂移可跨 64K 步复合**（微小分布漂移经长链生成放大）。
- 风险等级：低-中。需 needle 64K 抽验（3/3）+ 长上下文生成 sanity（主题保持/无崩溃），作为 F4/F5 观察门。

---

## 3. 校准需求

### 3.1 per-channel vs per-block vs routeB 原生布局

| 布局 | K 方向粒度 | routeB 契约 | staging | 相对精度【推算】 |
|---|---|---|---|---|
| per-channel [N,1] | 整行 1 个 scale（K=4096） | ❌ 与 sf_vec=32 不符 | 需适配 | 粗 |
| 128×128 块（同 shared） | K 组 128 | ⚠️ 需扩展为 [N,K/32]（无损复制） | 小（~11MB/rank） | 中 |
| **routeB 原生 [N,ceil(K/32)] E8M0** | **K 组 32** | ✅ 直接满足 | **零 staging** | **细（最优）** |

**裁定**：**采用 routeB 原生 `[N, ceil(K/32)]`**——同时满足"零 staging + 最细粒度 + 精度最优"。这比 lmhead-fp8 F0.1 列的"per-channel or 128×128 块"两选项更优（F0.2 已指出 routeB 期望 [N,K/32]）。**这是 lm_head 相对 shared 的隐藏顺风**：shared 从 128×128 块转 routeB 需 scale 扩展；lm_head 从 BF16 直接生成 routeB 原生布局，一步到位。

### 3.2 是否需要校准集

- **权重变换：不需要校准集**。max-abs 幂次 scale（E8M0）是**数据无关的静态变换**，从 BF16 权重统计直接计算，无训练/无优化目标。与 FP4/W4A4 路径的"校准 + 有损量化"不同——**FP8 E4M3 路径无校准门**（与 opt-routeb-fp8 §2.3 判定一致：FP8 相对 FP4 最大优势即"无损 payload/scale、无校准门"）。
- **质量门：需要参考数据集（不是校准）**。KL/困惑度门必须在**固定参考集**上测 BF16 vs FP8 分布差异，并先测 BF16 vs BF16 噪声底。参考集要求：
  - 覆盖多类负载（fox 文本、代码、列表、数数、reason/zh——后者仅用于分布度量不用于逐字门，因运行级非确定）；
  - 长度覆盖（短 128-512 tok + 长 64K 抽验）；
  - 规模：建议 200-500 条 prompt × 128-256 tok，足够统计 KL/困惑度【推算，待窗口定标】；
  - 复用既有质量门 prompt 集 + bench 参考集，**不新增采集负担**。

### 3.3 KL 门基线如何建立（困惑度 Δ≤0.05 口径）

1. **BF16 vs BF16 噪声底**：同一参考集跑两次（同配置背靠背），算 `KL(p_bf16 ‖ p_bf16_2)` + 困惑度差。因 probabilistic draft 运行级非确定 + kernel ULP，噪声底**非零**（reason/zh 近平票翻转已证非确定存在【实测】）。预期 KL 噪声底 0.01-0.1 nats/token 量级【推算，待实测】。
2. **门限 = 2-3× 噪声底**（沿用 lmhead-fp8 §3 + upstream-check §2.3 P2 门）：
   - `KL(p_bf16 ‖ p_fp8) < 3 × KL(p_bf16 ‖ p_bf16_2)`（有向 KL，方向 = "用 fp8 近似 bf16 的信息损失"）；
   - 困惑度 `Δ = PPL_fp8 − PPL_bf16 ≤ 0.05`（绝对值，非相对）；
   - 逐字门 + 包络（quality_gate.py 现有，≤1%）。
3. **Δ≤0.05 可达性**：logits 噪声 ~0.1-2% → 单 token 困惑度扰动 e^{2δ} 量级 ~1.0-1.04，跨 200-500 条平均后 Δ 应远小于 0.05【推算】——0.05 合理但需实测确认；**若实测 BF16 噪声底本身 >0.05，门限须按噪声底重标定**（先测底再定门）。

---

## 4. 验证门设计（F2/F4 用）

### 4.1 门矩阵

| # | 门 | 判据 | 载体 | 阶段 | 来源 |
|---|---|---|---|---|---|
| G1 | **greedy 硬门** | 4 稳定 prompt（fox_repeat/count/code/list）temp=0 逐字 4/4；不一致时包络 top-1 logprob sum drift ≤1% 兜底；own_stable 交叉复跑标注 | quality_gate.py（现成） | F2/F3/F4/F5 | 既有固化口径 |
| G2 | **KL 门** | `KL(p_bf16‖p_fp8) < 3×噪声底`（参考集） | 新建 kl_gate.py | F4/F5（参考集先建） | 本报告 §3.3 + lmhead-fp8 §3 |
| G3 | **困惑度门** | `PPL Δ ≤ 0.05`（参考集） | 新建 ppl_gate.py | F4/F5 | upstream-check §2.3 |
| G4 | **温度采样抽验** | temp∈{0.6,0.9} 参考子集：logprob sum drift ≤1% + token 集重叠率 ≥ 阈值 + distinct-n 不降 | 新建 temp_sampling_gate.py | F4/F5 | bench-regression + 本报告 §2.1 |
| G5 | **top-p/top-k 抽验** | top-p=0.9 / top-k=50：候选集重叠率 ≥ 阈值；logprob drift ≤1% | 并入 G4 脚本 | F4/F5 | 本报告 §2.2 |
| G6 | **MTP 接受率门** | DE C1/C12 tokens/step（step_eff）±3% 带内（同 draft 配置 A/B） | de_bench.py（现成） | F4 | fi017 三门 |
| G7 | **长上下文 64K** | needle 64K 3/3 + 长上下文生成 sanity（无崩溃/主题保持） | 既有 needle 流程 | F4/F5 | 回归观察门 |
| G8 | **off 等价 / 数值包络** | env off 路径 byte-equivalent；on 路径 logits rel 差阈值（F2 golden rel_err ≤1e-2） | L1 | F2/F3 | opt-routeb §3.1 |

### 4.2 F2/F4 门序建议

- **F2（适配器 + golden）**：G1（golden 4/4）+ 数值包络（logits rel 差阈值）+ G8。**F2 阶段即可先跑 CPU 参考集 KL 初值（BF16 vs FP8 离线 dequant 对比）——不占 GPU，先于 F4 拿到风险读数**。
- **F4（窗口 A/B）**：G1-G7 全过 + 性能门（PR 四档 ≥-3% / DE C1±5% C12≥-3% / 内存门）+ 回归观察门。
- **F5（发布）**：G1-G7 复跑 + 回滚锚点就绪（`.bak` 快照 + head-first 重建核验）。

### 4.3 工具链先行建议

质量门工具链（G2-G5 + 参考集）**零 GPU、纯离线可立即建**，是 lm_head F2/F4 的硬前置；同时 shared-FP8 F4 复用 G1-G3（shared 虽无损但 F4 窗口要求质量门全过）。

---

## 5. 下一步方案建议

### 5.1 立项顺序（维持 P0 裁定 + 质量风险分层）

| 排序 | 项 | 质量风险 | 理由 | 前置 |
|---|---|---|---|---|
| 0 | **质量门工具链 + 参考集/KL 基线先行**（离线） | — | G2-G5 是 lm_head 硬前置；shared-FP8 F4 复用 | 无 |
| 1 | **shared → routeB FP8 窗口 A/B**（首发） | 低（引擎置换、无新量化、质量门先例可迁移） | 资产零缺口、M=4096 甜点、无损 payload/scale【实测-源码】 | F0/F1（可与 0 并行） |
| 2 | **lm_head → FP8**（第二） | **最高**（唯一 BF16 无损节点 + 最终层直接采样） | 唯一打 decode 带宽墙 + prefill + 显存 | 0 + F0（顺手带）+ F1 微基准共享 |
| 3 | attn 投影 | 中-高 | 仅 prefill 半场 | 后段 |

> 排序依据与 P0 裁定一致（shared 首发 / lm_head 第二 / attn 第三【实测-裁定】）；质量风险分层是 shared-FP8 先行的**额外**理由（先低风险跑通引擎 + 质量门流程，再攻高风险 lm_head）。

### 5.2 "先做 shared-FP8 窗口 A/B 顺手带 lm_head F0 契约核对"——是，推荐

- **shared-FP8 是低风险快赢**：把已存在的 FP8 GEMM 换成更快的 FP8 GEMM（修正后 PR +0.5~1.5%【推算】），无新量化、无校准门、质量门先例可迁移；**先跑通 routeB FP8 引擎 + A-quant 适配器 + scale 扩展**，为 lm_head 复用。
- **lm_head F0（契约核对）纯离线、与 shared F0 并行**：确认 head.weight 生产内存布局 + scale 布局决策（本报告裁定 routeB 原生 [N,K/32]）+ CPU golden。**不占窗口、不阻塞**。
- **条件触发**：质量门工具链先行完成 + shared F1 微基准 E2E≥1.1× → lm_head F1 与 shared W3 **共享一次 GPU 微基准窗口**（形状 N=32320 大 N）；顺序仍是 shared 先 F4 窗口、lm_head F2/F4 后置。

### 5.3 质量门先行落地建议（本周可做，零 GPU）

1. 建参考数据集（复用既有 prompt + bench 集，200-500 条）；
2. 建 kl_gate.py / ppl_gate.py / temp_sampling_gate.py / top_p_gate.py（并入 quality_gate.py 同目录）；
3. 在现有 LuZ0.3.1 生产形态上跑 BF16 vs BF16 噪声底（**需一次窗口旁路采集，其余离线**）；
4. 产出 KL 门限值（2-3× 噪声底）+ 困惑度 Δ 门限确认（若噪声底 >0.05 则重定标）。

### 5.4 与 CUMEM=0 / G1 补齐的并行关系

- **CUMEM_HOST_ENABLE=0（AR stall 缓解 env）**：来自 arstall 生产闭环【实测-源码】，与 FP8 量化**正交、独立可落地**。但**是 FP8 A/B 测量基线的前置**：lm_head FP8 收益仅 ~5%（C12 口径【实测+推算】），而当前测量噪声 ±8-13%【实测-重测协议前提】；A/B 窗口若不先把 CUMEM=0/重测协议（30 轮/交错 A/B）纳入基线，FP8 收益会被噪声淹没、判定失真。**顺序：CUMEM=0 先定，再开 FP8 A/B**。
- **G1（native 派生显存账补齐）**：a3-hybrid G1 验证 native 零拷贝权重路径显存账【实测-源码】，与 routeB FP8 零拷贝适配器**同族但独立**（一个 W4A4/native 派生、一个 FP8/routeB 布局）——并行推进、无相互阻塞；两者共享 from_dlpack/对齐的工程经验，可互作旁证。

### 5.5 关键 go/no-go 汇总（供窗口对照）

- shared-FP8 F1：`E2E(M=4096) ≥ 1.1× 当前 cutlass` 且 `A-quant delta < GEMM 增益` → go；否则降级设计储备（沉没成本 = F0/F1 半天）。
- lm_head-FP8 F1：`E2E(M=4096) ≥ 1.1×` + N=32320 大 N 形状效率 → go；质量门 G1-G7 F4 全过 → 发布。
- **质量门先行（门限值）产出后**：若 BF16 噪声底本身超预期（KL > 0.5 nats/token 或困惑度跨次复跑 Δ > 0.05）→ 质量门口径需重新设计（可降级到 JS 散度或 token 集重叠度量），**先于 lm_head F2 解决**。

---

## 6. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| 【实测-源码】 | head.weight = BF16 [129280,4096]（唯一 BF16 节点）；shared 权重 F8_E4M3 block-scaled + F8_E8M0 [16,32]/[32,16]；routeB FP8 dispatch (FP8,FP8,E8M0,32)→MmaMXF8Op + SF 布局 [N,ceil(K/32)] + tile_k%128=0；shared 现役 Fp8LinearMethod→CutlassFp8BlockScaledMMKernel（生产未开 DeepGEMM）；quality_gate.py 4 稳定 prompt + 包络 ≤1% + own_stable；draft_sample_method probabilistic（temp-0 下 = argmax）；CUMEM_HOST_ENABLE=0 env；a3 G1 native 派生显存账 |
| 【实测】 | P0 拆账：lm_head prefill 2.79µs/12.9%、decode M=8 1.18ms/15.8%、M≈96 1.23ms；shared 6.98µs/32.2%、decode 1.99/2.70ms；三池份额 attn 54.9%/shared 32.2%/lm_head 12.9%；shared golden 4/4 逐字；routed W4A4 logprob 级 0.36-0.41%；reason/zh 近平票翻转（运行级非确定）；测量噪声 ±8-13%；BF16 vs BF16 KL/困惑度噪声底**无直接实测**（【待窗口验证】） |
| 【推算】 | E4M3 相对误差 6.25%（max）/3.6%（RMS）；logits 绝对误差 ~0.1-2%；argmax 翻转率 0.1-1%/token；MTP 接受率漂移 <1-2%；greedy/温度/top-k 敏感度分层；routeB [N,K/32] 布局精度最优；困惑度 Δ≤0.05 可达性；参考集规模 200-500 条 |
| 【待窗口验证】 | BF16 vs BF16 噪声底实测值；KL 门限定标；困惑度门限重标定；参考集规模定标；N=32320 微基准；F2/F4 全门；G1 全量显存账 |

**诚实声明**：
1. §1.3 的 logits 误差 0.1-2% 是推算量级（格式性质 + 内积模型），**不是实测**；最终以质量门工具链 + F2 参考集 KL/困惑度实测为准。
2. lm_head 当前是无损 BF16，引入 FP8 是"首次有损量化到最终层"——即使误差有界，**质量门工作量也必须是全池最重**（G1-G7），不可沿用 shared 的轻量门。
3. greedy 4/4 对近平票翻转天然敏感，量化引入的逐字差异**不一定代表质量退化**——必须用包络兜底 + KL/困惑度门区分"可接受噪声"与"分布退化"。
4. 所有【推算】数字在质量门工具链 + F1 微基准后须替换为实测值。

---

## 7. 引用索引

- 关键实测源：`lmhead-fp8-project-2026-08-23.md`、`opt-routeb-fp8-2026-08-23.md`、`fi017-p0-accounting-2026-08-23.md`、`_luz031_official_bench/data/p0/p0_accounting_data.md`、`bench-regression-attribution-2026-08-23.md`、`upstream-check-perf-ceiling-2026-08-23.md`、`opt-objective-research-2026-08-23.md`、`mtp-tuning-2026-08-22.md`、`a3-hybrid-slim-design-2026-08-23.md`、`arstall-production-closure-2026-08-23.md`
- 质量门：`tmp/quality_gate.py`（固化版，LuZ0.3.1 起唯一口径）

*本报告由工程保障团队（系统架构师 architect-1）生成；纯只读分析，未执行 GPU/集群操作；立项顺序与质量门阈值请由人类工程负责人结合质量门工具链实测与 F1 微基准裁定。*
