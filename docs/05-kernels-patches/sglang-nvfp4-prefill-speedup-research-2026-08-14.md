# SGLang × NVFP4 权重 vs vLLM 官方 0731 MXFP4：Prefill 加速量化区间调研

**日期**：2026-08-14 ｜ **作者**：阿奇（Archi）· 系统架构师
**任务**：调查 SGLang 环境采用 NVFP4 权重，相对当前 vLLM 官方 0731 MXFP4 原版，prefill（PR）速度能带来多大加速——给出量化区间与依据链
**状态**：调研完成（本地既有证据 + WebSearch/WebFetch 补充）｜ **报告落盘**：本文件

---

## 0. TL;DR（一句话）

**prefill 三档加速区间（SGLang 0.5.14 + NVFP4 TP4 vs vLLM 0.26.1.dev0 + MXFP4 0731 TP4，DGX Spark SM121 四机环网）**：

| 档位 | prefill 加速区间 | 一句话依据 |
|---|---|---|
| **保守** | **1.05 – 1.15×** | 格式层收益按 tsarihan 低端（1.14×）兑现，被栈层差异/通信暴露/autotune 首跑稀释 0–10% |
| **中性** | **1.15 – 1.30×** | 格式层收益按 tsarihan 中枢（1.23×）兑现，SGLang vs vLLM 栈层近乎持平（±3%） |
| **乐观** | **1.30 – 1.45×** | 格式层收益按 tsarihan 高端（1.32×）+ SGLang SM120 调优（PR #26496 ~17% TPS）部分叠加 |

**置信度**：中性档 **中-高**（同硬件 SM121 有 tsarihan 直接实测锚点）；保守档下限受 **SGLang 0.5.14 的 `flashinfer_cutlass` 可用性**这一单一高风险变量钳制（PR #26496 晚于 0.5.14，见 §7.1）；乐观档上限受"格式收益与 SGLang 调优可叠加"假设制约，**中-低**。

**一句话定性**：用户问的"采用 NVFP4 相对 vLLM 原版 MXFP4"是**跨栈+跨格式**的混合对比；主因（~全部收益）在**权重格式层**（NVFP4 vs MXFP4 内核路径成熟度），**运行时栈层（SGLang vs vLLM）prefill 上基本打平、±10% 内**。NVFP4 相对 MXFP4 并非"4bit vs 8bit"的算术收益（两者同为 FP4 E2M1、同一 tensor core 峰值），收益来自 **NVFP4 在 SM12x 有专属、成熟的 CUTLASS fp4_gemm 内核路径**，而 MXFP4 在 SM121 依赖 b12x（decode 向调优）。

---

## 1. 核心结论卡片

| 项目 | 内容 |
|---|---|
| 对比基准 | vLLM TP4（anemll 0.2.1-v026.0，vLLM 0.26.1.dev0，FlashInfer b12x）官方 0731 MXFP4 |
| 被调查对象 | SGLang 0.5.14（NGC 26.07-py3）+ NVFP4（modelopt，group_size=16，164GB 已四机分发）+ TP4 环网 |
| prefill 预期 | **中性 1.15–1.30×；保守 ≥1.05×；乐观 ≤1.45×**（同 ctx×task×conc 的 per-request p50） |
| 主收益来源 | **权重格式层**（NVFP4 专属 CUTLASS fp4_gemm sm120 模板 + cuBLASLt 大 M×N 优化 vs MXFP4 b12x 路径），tsarihan 同栈实测 1.14–1.32× |
| 栈层贡献 | SGLang vs vLLM prefill 基本打平（8×B300 实测 ±10%；SGLang 0.5.14 无 DSPARK 反可省 MTP prefill 计算） |
| 不可引用数字 | cuBLASLt 13.2 "DGX Spark NVFP4/MXFP8 大 M×N up to 3×" —— 库级上限，SGLang 0.5.14 NVFP4 MoE 走 FlashInfer 不走 cuBLASLt，且为 dense GEMM 非 grouped MoE |
| 最大风险 | SGLang 0.5.14 可能无 `flashinfer_cutlass` 合法取值（#26496 晚于 0.5.14）→ 降级 flashinfer→marlin，prefill 收益打折 |
| Phase-A 判定 | 测试计划 `目标 ≥1.1× / PASS ≥1.05×` **合理，建议保留**；加解释规则（<1.05 FAIL / 1.05–1.15 WEAK / 1.15–1.30 STRONG / >1.30 EXCEPTIONAL） |

---

## 2. 证据分层表

> 分层：A=同硬件同栈直接实测；B=同硬件跨栈实测/权威声明；C=异硬件/理论推导/库级数字。可信度：高/中/低。

| # | 证据 | 数值 | 来源 | 硬件 | 可信度 |
|---|---|---|---|---|---|
| E1 | **NVFP4 vs MXFP4 prefill（同 vLLM 栈，cu130）** | **1.14 – 1.32×** | tsarihan 实测（工作区已引证，research-nvfp4-alternative-runtimes-2026-08-13） | DGX Spark SM121 | **A / 高** |
| E2 | **NVFP4 vs MXFP4 吞吐（SGLang 栈，官方语境）** | **~1.4×** | mr.technology 2026-08-12 + SGLang PR #25702 语境（arch-design V1 §1.1） | Blackwell（GB300 向，非 Spark） | **B / 中** |
| E3 | GB10 NVFP4 dense tensor-core 峰值 | ~511 TFLOPS（spec 500）；**tuned CUTLASS NVFP4 GEMM ~375 TFLOPS** | Alfonso De Gregorio 实测（LinkedIn，CUDA 13，driver 580） | GB10 SM121 | A / 高 |
| E4 | NVFP4 精度/内存特性 vs MXFP4 | NVFP4 E2M1 + E4M3 16 元素块；**NVFP4 精度显著优于 MXFP4**（同 loss 训练 MXFP4 需 +36% token） | NVIDIA research / NVFP4 pretraining blog | —（格式定义） | A / 高 |
| E5 | SGLang SM120 NVFP4 调优（autoselect flashinfer_cutlass + autotune） | **~17% TPS 提升** | SGLang PR #26496 | RTX PRO 6000 / SM120 | B / 中 |
| E6 | FlashInfer CUTLASS NVFP4 在 SM120/SM121 可跑 | SM120 157.9 tok/s / SM121 65.0 tok/s（FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"） | NVIDIA 论坛 "We unlocked NVFP4 on the DGX Spark" | SM120/SM121 | B / 中 |
| E7 | vLLM flashinfer-cutlass 在 SM121 的坑 | 默认 autoselect 选 FLASHINFER_CUTLASS → **"Failed to initialize cutlass TMA WS grouped gemm" 静默降级**，Marlin 兜底反而快 16% | NVIDIA 论坛 Marlin Fix / Avarok blog | SM121（vLLM 栈） | A / 高 |
| E8 | **vLLM vs SGLang prefill 打平**（8×B300，Kimi-K3，chunked 32768） | 两者聚合 prefill 均 ~19K–23K tok/s（±10%） | GPUStack Day 0 实测 | 8×B300 | B / 中 |
| E9 | SGLang RadixAttention 前缀复用收益 | 多轮/共享前缀 10–30% 更优；**但 uuid 随机前缀 bench 无此收益** | aiwiki / Spheron 对照 | 通用 | C / 中 |
| E10 | DSV4 通信轻量 | all-reduce 352KB/token；MLA KV 压缩 ~11KB/token；200G 链路 prefill 峰值用 25–45%（BF16 口径）；TP4 ring 2 跳全模型 ~5.1ms（未掩蔽） | 工作区 analysis-tp2-tp4-communication-2026-08-09（config.json 实读 + NCCL 实测） | 本集群拓扑 | A / 高 |
| E11 | cuBLASLt 13.2 DGX Spark NVFP4/MXFP8 大 M×N | **up to 3×**（库级、dense 非 grouped、相对旧版 cuBLASLt，非 vs MXFP4） | cuBLAS 13.4.0 release notes（工作区已核实） | SM120/SM121 | A / 高（但不适用于 SGLang 路径，见 §7.2） |
| E12 | 当前 vLLM 基线 prefill p50 | 32768×c1 coding **2208** / 8192×c1 **2222** / 131072×c1 **2015**（r12 正式全矩阵） | tp4-r12-final-report-2026-08-13 | 本集群 | A / 高 |
| E13 | FlashInfer b12x（CuteDSL）NVFP4 MoE 定位 | **decode 小 M 场景**相对 CUTLASS 1.3–1.6×；prefill 大 M 非其强项 | vLLM PR #40082 摘要 | SM12x | B / 中 |

---

## 3. 两层贡献拆解（关键：必须讲清）

用户问的"采用 NVFP4 权重 vs 当前 vLLM 原版 MXFP4"是 **跨栈 + 跨格式**的混合对比。加速来源分两层，**主因在格式层**：

### 3.1 权重格式层：NVFP4 vs MXFP4（贡献 ~全部可预期收益）

**重要纠偏：NVFP4 vs MXFP4 不是"4bit vs 8bit"的算术收益。** 两者元素同为 FP4 E2M1、同一 Blackwell tensor core（NVFP4 与 MXFP4 的 mma 峰值相同，GB10 dense 均 ~511 TFLOPS spec）。NVFP4 相对 MXFP4 的吞吐优势来自三处：

1. **内核路径成熟度（主因）**：NVFP4 在 FlashInfer/CUTLASS 有**专属 `fp4_gemm_cutlass_template_sm120.h`**（大 M prefill 用 128×128 cooperative tile）与 cuBLASLt 13.x 的 NVFP4 大 M×N 优化；MXFP4 在 SM121 上历史路径较差——本集群 vLLM 用 b12x（CuteDSL，**decode 小 M 向调优**，E13），且 vLLM 默认选 flashinfer-cutlass 在 SM121 会静默降级（E7）。tsarihan 的 1.14–1.32× prefill（E1）本质上就是这个内核路径差异的端到端体现。
2. **精度余量**：NVFP4 用 E4M3（FP8）块缩放 + 16 元素块，精度显著优于 MXFP4 的 E8M0（E4）；精度余量允许工作负载稳定留在快速 FP4 原生路径上（避免因精度/质量劣化触发 W4A16 反量化 + FP8 GEMM 慢路径）。
3. **内存足迹**：NVFP4 ~4.5 bits/value vs MXFP4 ~4.25 bits/value，**基本持平**；prefill 大 M 是 compute-bound，内存减半（相对 FP8）的收益在"NVFP4 vs FP8"才显著，在"NVFP4 vs MXFP4"上≈0。**因此不要用"数据搬运减半"论证 NVFP4 vs MXFP4 的 prefill 加速。**

**定量锚点**：E1（tsarihan，SM121 同栈）**1.14–1.32×** 是格式层最直接、最相关的数字；E2（mr.technology 1.4×）是 SGLang 语境但偏 GB300。**格式层贡献 ≈ 1.14–1.32×（SM121）/ ≤1.4×（数据中心 Blackwell）。**

### 3.2 运行时栈层：SGLang vs vLLM（贡献 ≈ ±10%，方向不确定）

1. **prefill 能力实证基本打平**：8×B300 Kimi-K3 实测两者聚合 prefill 均 ~19K–23K tok/s（E8），SGLang 无明显 prefill 栈优势。
2. **SGLang 0.5.14 无 DSPARK → 反而省 MTP 计算**：0.5.14 不消费 MTP 模块（实测 `--speculative-algorithm` 无 DSPARK）；而 vLLM 基线带 DSpark，prefill 阶段需跑 MTP head 前向。→ SGLang 在 prefill 上可能有**小幅正贡献**（~3–8%，机制性推断非实测）。
3. **SGLang SM120 调优（PR #26496 ~17% TPS）**：但 #26496 **晚于 0.5.14**，0.5.14 未必吃到（§7.1 风险）——这是乐观档上限的依赖项。
4. **RadixAttention 前缀复用**：bench 用 uuid 随机前缀（防 cache），**SGLang 此优势在 A/B 中不生效**（E9）。

**栈层净贡献 ≈ 0.95 – 1.10×（prefill）**，与 GPUStack 的 ±10% 一致。

### 3.3 合成

```
预期 = 格式层(1.14–1.32× SM121) × 栈层(0.95–1.10×)
     → 保守 1.05–1.15× / 中性 1.15–1.30× / 乐观 1.30–1.45×
```

（叠加后向保守端打折，理由：收益非完全独立、通信暴露、autotune/首跑、0.5.14 版本缺口。）

---

## 4. 量化结论（乐观/中性/保守 + 依据链）

### 4.1 三档区间

| 档位 | 区间 | 依据链（证据→推理） | 适用情形 |
|---|---|---|---|
| **乐观** | **1.30 – 1.45×** | E1 高端 1.32× → 栈层假设 SGLang 调优部分生效（E5 +17% 部分叠加、E8 栈层 ≥0）+ 0.5.14 跳过 MTP 正贡献 → 上限 1.45× | 0.5.14 有 `flashinfer_cutlass`、autotune 正常、环网 comm 全隐藏、无首跑 JIT 干扰 |
| **中性** | **1.15 – 1.30×** | E1 中枢 1.23×（1.14+1.32 中值）→ 栈层持平（E8 ±3%），comm 大部分隐藏（E10 200G 余量 >2×） | 主推荐预期；`flashinfer_cutlass` 可用、栈层互相对消 |
| **保守** | **1.05 – 1.15×** | E1 低端 1.14× → 被栈层 -5%（E8 波动、comm 暴露、autotune 首跑、0.5.14 缺 #26496）→ 下限 1.05× | 0.5.14 走通用 `flashinfer`（无专属 cutlass 模板）、或环网 comm 暴露、或 JIT/autotune 干扰 |

### 4.2 与"3×"的关系（必须写明）

- cuBLASLt 13.2 的 **up to 3×**（E11）是 **dense `cublasLtMatmul` 大 M×N** 相对**旧版 cuBLASLt** 的库级上限，且是 NVFP4/MXFP8 **对 FP8/BF16** 的提速；**不是** NVFP4 对 MXFP4、更不是端到端。
- SGLang 0.5.14 的 NVFP4 MoE **不走 cuBLASLt**（走 FlashInfer fp4_gemm CUTLASS）→ **3× 不可直接引用**，仅可作为"NVFP4 相对 FP8 的 GEMM 理论空间"背景（GB10 dense NVFP4 峰值 ~511 TFLOPS vs FP8 ~250 TFLOPS，E3）。
- 即便按 E3 的 tuned CUTLASS NVFP4 GEMM ~375 TFLOPS 计算，prefill 若 100% GEMM-bound，理论上限约 1.5–2×（vs 当前 ~2200 tok/s 的 FP4 实现）；**1.45× 乐观档已在此之下，不越理论界**。

### 4.3 对 prefill 计算/带宽属性的判断

- **prefill 大 M 是 compute-bound 主导**（chunked prefill 4096 → M=4096；GB10 临界算术强度 AI*≈915 FLOP/byte，prefill 算术强度约 2T → T≥~460 token 即进入 compute-bound，E10 引证 + 工作区 tp4-nccl-prefill-analysis 模型）。
- 但**当前 FP4 实现远未达峰值**：vLLM 基线 32768×c1 ~2208 tok/s 折算约 60–80 TFLOPS 有效（vs 250 TFLOPS FP8 / 511 TFLOPS FP4 spec），即**内核效率 ~25–30%**。→ 收益空间在"换更好内核路径"，而非单纯格式算术。
- 结论与既有"NVFP4 主要改善 prefill/带宽饱和档"**一致但需修正表述**：NVFP4 改善的是 **prefill 的 GEMM 效率**（大 M compute-bound 档）；在 decode 小 M / 带宽饱和档，NVFP4 相对 MXFP4 的内存减半≈0（同为 4bit），**decode 收益主要靠 b12x/投机而非 NVFP4 格式**。

---

## 5. 与 vLLM 基线衔接（prefill p50 参照系）

### 5.1 基线（当前生产 vLLM TP4 MXFP4）

| ctx | conc | task=coding prefill p50 | task=json | task=prose | 来源 |
|---|---|---|---|---|---|
| 8192 | c1 | 2222 | 2210 | 2217 | tp4-r12-final（正式全矩阵） |
| 16384 | c1 | ~2200–2400（插值 8192↔32768） | — | — | 推断 |
| 32768 | c1 | **2208** | **2228** | **2217** | tp4-r12-final |
| 65536 | c1 | ~2409（r8 旧配置） | ~2409 | ~2405 | tp4-r8-final（r12 未测 65536） |
| 131072 | c1 | 2015 | 2013 | 2016 | tp4-r12-final |

> ⚠️ 数据口径差异：r12（正式）与 r8（旧配置 util 0.6）prefill 有 5–10% 系统差；**严格 A/B 应在同一测试窗口用同一 bench 脚本重跑 V 组**（test-plan §6.2 P5 已如此规划），r12/r8 仅作参照锚点。

### 5.2 加速后投影（中性 1.15–1.30×，同口径 per-request p50）

| ctx | 基线（coding c1） | 中性加速后 | 全带（1.05–1.45×） |
|---|---|---|---|
| 16384 | ~2200–2400 | 2530–3120 | 2310–3480 |
| 32768 | 2208 | 2540–2870 | 2318–3202 |
| 65536 | ~2409 | 2770–3130 | 2529–3493 |
| 131072 | 2015 | 2317–2620 | 2116–2922 |

---

## 6. Phase-A 判定阈值评估（test-plan §6.3 / §10.4.6）

测试计划已定：**prefill 目标 ≥1.1×，PASS ≥1.05×**。

**评估结论：阈值合理，建议保留，并加解释规则。**

- **合理**：中性预期 1.15–1.30× 高于 1.1× 目标；PASS 线 1.05× 恰在保守档下限，容纳测量噪声（该基准历史波动 ±5–10%）。
- **建议加解释规则**（防止"过了 PASS 线就推广"的误判）：
  - **<1.05×** → FAIL（无可信收益）
  - **1.05–1.15×** → WEAK PASS：收益在/接近噪声区，**推广前必须补同栈 A/B 归因**（见 §9 第 2×2 矩阵），不能直接归因 NVFP4
  - **1.15–1.30×** → STRONG PASS：与中性预期吻合，格式层归因由 tsarihan（E1）支撑
  - **>1.30×** → EXCEPTIONAL：核查测量伪影（前缀缓存命中、MTP 跳过优势、autotune 方差、conc 口径不一致）
- **归因警告**：这是跨栈+跨格式 A/B，PASS 不等于"NVFP4 带来 1.2×"。**若需把功劳归给 NVFP4 而非 SGLang**，必须补第 3 臂/第 4 臂（§9）。

---

## 7. 风险与不确定性

### 7.1 NVFP4 prefill 收益在 SM121 会被 kernel 支持度打折吗？——会，这是最大风险

| 风险 | 等级 | 说明 | 缓解 |
|---|---|---|---|
| **0.5.14 可能无 `flashinfer_cutlass` 合法取值**（#26496 晚于 0.5.14，arch-design V2-R8） | 🔴 高 | 若无 → 只能走通用 `flashinfer`（fp4_gemm fallback）或 `marlin`（W4A16，性能不作数）→ prefill 收益直接落到保守档下沿甚至更差 | sre 先 `launch_server --help` 确认 choices；无 → Phase-A 接受保守预期或提前切 0.5.16 |
| **vLLM 栈的"flashinfer-cutlass 在 SM121 静默降级"教训在 SGLang 重演** | 🟠 高 | E7：SM121 上 CUTLASS TMA WS grouped GEMM 初始化失败会静默走慢路径（不报错） | TP1 冒烟 K3 核对实际 backend/kernel 名；跑通后先小样本 TTFT sanity，再进全矩阵 |
| **0.5.14 缺 #26496 的 SM120 autotune** | 🟡 中 | fp4_gemm 不 autotune → 可能用默认 tile，效率打折 | 若 0.5.14 不支持 → 接受中低档预期；Phase-B 升 0.5.16 复测 |
| **flashinfer JIT/autotune 首跑** | 🟡 中 | 首次 batch 触发 JIT 编译/autotune → 污染首轮 TTFT/早段 prefill | bench 前 warm-up 若干请求；用中位数非均值 |

### 7.2 cuBLASLt 13.2 的 3× 不能直接引用

- SGLang 0.5.14 NVFP4 MoE 走 **FlashInfer fp4_gemm（CUTLASS）**，不走 cuBLASLt；3× 是 dense `cublasLtMatmul` 库级上限（E11）。**引用链必须写"格式理论空间参考，非本方案端到端依据"。** 若未来 SGLang 换 cudnn/cuBLASLt backend（#26496 提到 flashinfer_cudnn 在 SM120 有 NaN 问题），届时再评估。

### 7.3 环网通信对 prefill 加速的稀释

- DSV4 通信轻量（E10）：all-reduce 352KB/token、MLA KV ~11KB/token；200G 链路 prefill 峰值 25–45%（BF16 保守口径）、余量 >2×；TP4 ring 2 跳全模型 ~5.1ms（未掩蔽）基本被流水掩蔽。
- **NVFP4 权重减半不会等比降低通信量**——TP 的 all-reduce 传的是**激活**（FP8/BF16），不是权重；权重格式只影响 GEMM 读权重的字节，不影响 all-reduce 字节。但 GEMM 变快 → 通信/计算比上升 → **若 comm 未被完全掩蔽，会稀释 GEMM 收益**。估算：comm 若暴露 10–20% 计算时间，则 GEMM 收益 1.3× 稀释为 ~1.25×。**中性档已计入此稀释。**
- 另：NVFP4 权重在 TP4 下每 rank 读 42GB÷4 权重更少 → **KV/激活内存更宽裕**（可提高 chunk 大小），但 bench 不调参，此点不改变 A/B 结论。

### 7.4 其他

- **0.5.14 无 DSPARK**：prefill-only A/B 公平；且 vLLM 基线带 DSpark（prefill 跑 MTP）→ 轻微有利于 SGLang（机制性，未实测，量级 ~3–8%）。
- **MTP 权重（"experts-mtp-fallback" 命名）**：0.5.14 不消费 MTP → 只需"加载不报错"（R9-A）；若 Phase-B 升 0.5.16 启用 DSPARK，MTP 权重策略将重新成为接受率风险（已在 test-plan V2 记录）。
- **精度差异**：NVFP4 vs MXFP4 输出非逐位一致（E4M3×2^-G vs E8M0），R6 门槛（greedy 一致率 ≥0.90）需执行以排除格式精度导致业务不可用。

---

## 8. Phase-A 实测建议（最小实验设计，与 testing-expert P2 矩阵衔接）

### 8.1 核心建议：2×2 归因矩阵（推荐，成本可控）

| 臂 | 运行时 | 权重 | 目的 |
|---|---|---|---|
| **A（目标）** | SGLang 0.5.14 `flashinfer_cutlass` | NVFP4 | 主测：跨栈+跨格式加速 |
| **B（基线）** | vLLM 0.26.1.dev0（b12x） | MXFP4 0731 | 现有 r12 基线（P5 同窗口重跑） |
| **C（可选，归因格式层）** | SGLang 0.5.14 `flashinfer_mxfp4` | MXFP4 0731 | 隔离"栈层"：A/C 比值 = 格式层贡献 |
| **D（可选，归因栈层）** | vLLM b12x | NVFP4（MJPansa/本地转换） | 隔离"格式层"同栈：D/B 比值 = 同栈格式收益（tsarihan 对照） |

- **最少实验 = 臂 A + 臂 B**（回答"是否加速、多少"）；**推荐加臂 C**（回答"收益是不是 NVFP4 的"）——臂 C 成本低（同一 SGLang 容器换权重路径 + `flashinfer_mxfp4`），却能杜绝"跨栈功劳错配"。
- 臂 D 属锦上添花（vLLM NVFP4 路径在 SM121 有 E7 风险），时间允许再补。

### 8.2 测量矩阵（沿用 test-plan P2，最小化）

- ctx：**16384 / 32768 / 65536** × task：coding / json / prose × conc：**1 / 4**（SGLang 侧加 8/16/32 作参考，但与 vLLM 基线仅 conc 1/4 严格可比，因 vLLM max-num-seqs=6）→ 核心 18 组合 + 参考组合
- 指标：per-request p50 prefill TPS / TTFT；**禁用 agg_***；3 轮取中位 + 90% CI；uuid 随机前缀防 cache；每组合前 warm-up
- 判定：比率下界 >1.0 判胜；按 §6 解释规则分级

### 8.3 前置门槛（缺一不可）

1. **TP1 冒烟确认 kernel**：日志必须显示 `flashinfer_cutlass` / fp4_gemm CUTLASS sm120 模板被选用（K3），并确认非静默降级到 marlin/慢路径（E7 教训）。
2. **R5 输出 sanity**（防 garbage）与 R6 精度对比先过。
3. **同一窗口 A/B**：vLLM 停 → SGLang 测 → 切回 vLLM 重跑 V 组，消除漂移。
4. **0.5.14 flag 实测**：`launch_server --help` 确认 `flashinfer_cutlass` 存在（§7.1 最大风险前置）。

### 8.4 预期结果分档应对

| 实测比值 | 结论 | 行动 |
|---|---|---|
| <1.05× | FAIL | 排查 kernel 降级/autotune/comm 暴露；考虑 0.5.16 |
| 1.05–1.15× | WEAK PASS | 补臂 C 归因后再定推广 |
| 1.15–1.30× | STRONG PASS | 与 tsarihan 吻合，可向 Phase-B（0.5.16+DSPARK）推进 |
| >1.30× | EXCEPTIONAL | 查伪影；若真实 → 写入生产切换评估 |

---

## 9. 参考链接

**工作区既有（可回读）**
- `research-nvfp4-alternative-runtimes-2026-08-13.md`（SGLang 第一候选、NVFP4 内核路径、SM120 动态检测）
- `research-cublaslt-grouped-gemm-nvfp4-sm120-2026-08-13.md`（cuBLASLt 支持矩阵、b12x 本集群实证、garbage 教训）
- `sglang-nvfp4-arch-design-2026-08-13.md`（含 V2-8 定稿：0.5.14 无 DSPARK、`flashinfer_cutlass` 首选、#26496 晚于 0.5.14）
- `sglang-nvfp4-test-plan-2026-08-13.md`（Phase-A/B、P2 矩阵、R9-A、判定阈值）
- `analysis-tp2-tp4-communication-2026-08-09.md`（DSV4 通信轻量、200G 余量、TP4 ring 延迟）
- `tp4-r12-final-report-2026-08-13.md` / `tp4-r8-final-report-2026-08-12.md` / `tp4-opt-execution-final-2026-08-13.md`（vLLM MXFP4 基线）
- `handoff-sglang-2607-four-node-2026-08-14.md`（26.07 容器 = 0.5.14 + flashinfer 0.6.14）

**外部（本次 WebSearch/WebFetch 补充）**
- mr.technology *DeepSeek-V4-Flash-0731's First Production Week*（2026-08-12）：NVFP4 ~1.4× MXFP4；SGLang/vLLM 1M ctx recipe：https://mr.technology/payloads/deepseek-v4-flash-0731-inference-stack-caught-up-august-2026
- SGLang PR #26496 *Changes for SM120 perf and usability for NVFP4*（~17% TPS、autoselect flashinfer_cutlass）：https://github.com/sgl-project/sglang/pull/26496
- NVIDIA forum *We unlocked NVFP4 on the DGX Spark: 20% faster than AWQ*（FlashInfer CUTLASS JIT SM120 157.9 / SM121 65.0 tok/s）：https://forums.developer.nvidia.com/t/361163
- NVIDIA forum *Marlin Fix: NVFP4 Actually Works on SM121*（vLLM flashinfer-cutlass SM121 静默降级、Marlin +16%）：https://forums.developer.nvidia.com/t/365119
- Avarok blog *We Unlocked NVFP4 on DGX Spark*（SM121 缺 cvt E2M1 指令、后端选路坑）：https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83
- Alfonso De Gregorio GB10 NVFP4 GEMM 实测（峰值 ~511 TFLOPS、tuned CUTLASS ~375 TFLOPS）：LinkedIn post 2026-08
- NVIDIA research *Pushing Intelligence to 4-bit*（NVFP4 vs MXFP4 精度、~4.5 vs 4.25 bits/value）：https://research.nvidia.com/labs/eai/post/pushing-intelligence-to-4-bit
- NVIDIA blog *Train Models Faster with JAX and MaxText Using NVFP4 on Blackwell*（格式定义、GB300 7× GEMM vs Hopper FP8）：https://developer.nvidia.com/blog/...
- vLLM PR #40082（FlashInfer b12x SM120/121 集成、decode 小 M 1.3–1.6×）：https://github.com/vllm-project/vllm/pull/40082
- GPUStack *Kimi-K3 8×B300 vLLM vs SGLang*（prefill 打平 ~19K–23K tok/s）：https://tech.shaoqun.com/a/1161286.html
- DeepWiki RTX 6000 Pro benchmarks（NVFP4 vs AWQ 长上下文 3.1×、vLLM vs SGLang 长 ctx 分叉）：https://deepwiki.com/local-inference-lab/rtx6kpro/6.1-throughput-benchmarks

---

> 时效性说明：2026-08-14 时点。SGLang/FlashInfer/NGC 容器迭代极快；落地前须复核 0.5.14 的 `flashinfer_cutlass` 取值、PR #26496 合入 0.5.x 的具体版本、以及 NGC 26.08+ 容器内含 SGLang 版本。
