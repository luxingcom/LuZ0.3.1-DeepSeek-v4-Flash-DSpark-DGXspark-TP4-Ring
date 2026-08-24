# routeB FP8 算子取证分析：性能分解 / cutlass 优势机制 / 改进空间（2026-08-24）

- **执行人**：阿奇（Archi）· 系统架构师（architect-1）
- **任务**：任务二——调查取证 routeB FP8 的算子性能与瓶颈、cutlass FP8 的算子性能与优势、routeB 改进空间，为下一阶段开发做准备
- **纪律**：纯只读分析（本地文件 + SSH node01 只读 + 容器内源码级 read-only，未启动 GPU/未改生产）。涉及源码勘察经 `docker run --rm` 一次性容器 `cat/grep/sed`（CPU 侧，不触 GPU）
- **口径标注**：**【实测】**= F1 窗口/既有窗口实测；**【实测-源码】**= 容器/服务器源码直接验证；**【推算】**= 基于源码/形状/既有数据的推算；**【知识】**= 公开硬件/库事实（web 锚定）；**【待验证】**= 需 GPU 微基准/NCU 二次确认

---

## 0. 一页结论

1. **routeB FP8 慢的本质 = 硬件 2× 结构性差距（不可修）+ 内核浅流水 ≈1.8× 可修差距（可修）**。FP8 same-dtype 的 `MmaMXF8Op` 原子形状 (16,8,32) 在硬件上只有 FP4 `MmaMXF4Op` (16,8,64) 的一半 FLOPs/指令**【实测-源码】**——这是 Blackwell SM120 张量核的固有特性（GB10 实测 FP8 峰值 ≈248T vs FP4 ≈497T，**【知识/实测-外部】**），**任何指令选择都改不掉这 2×**。
2. **routeB FP8 的额外 ~1.8× 来自 GB10 仅 99KB SMEM 下的 2-stage 浅流水**【推算，源码级确认容量约束】**。128³ tile + tile_K=128 + BF16 C 的 SMEM 账：A+B=32KB/stage + C=32KB → **最多 2 个 AB stage**（FP4 同配置 3-4 stage）。2-stage 无法充分隐藏 TMA 延迟 → 实测 gate_up 156µs ≈ DRAM 83µs + MMA 峰值 69µs（**接近零重叠**）；lm_head 上 routeB 仅 42% MFU，cutlass 68% MFU。
3. **cutlass FP8 快在哪：同一块 99KB 上更深的流水 + 更小的每-stage 占用 + 形状级运行时选型 + 更粗的 block-128 scale（SF 流量 ÷4）**【实测 68% vs 42% MFU；机制为知识/推算】。生产形状 cutlass 154-169T vs routeB 105-110T（+35~60%）。
4. **改进空间：一条真实可行路径 + 两条辅助**。唯一能超越 cutlass 的是**内核级 SMEM/流水重构**（消除 sC epilogue staging、把 AB 流水从 2→3 stage）：预期 GEMM 侧 105T → **150-200T（60-80% MFU，1.4-2.0×）**，在 lm_head（MMA-bound）可超过 cutlass 169T。**"FP8 指令路径对齐 FP4 378T"是伪命题**——FP8 硬件上限 248T。辅助项：A-quant group32 融合（当前 E2E +24.6%）、decode M-trim。
5. **结论：若下阶段投入，先做 NCU 确认流水假设，再做 SMEM/流水重构（P1）；不投入则维持现役 cutlass（F1 判决维持）**。routeB FP8 的 GEMM 引擎置换叙事不因本次取证翻案，但**"可修 1.8×"若兑现可使 FP8 路径转正**——值得一次 P0 验证。

---

## 1. 证据与方法锚定

### 1.1 源码勘察清单（routeB 侧）【实测-源码】

| 项 | 证据 | 位置 |
|---|---|---|
| FP8 same-dtype dispatch | `(FP8, FP8, *, Float8E8M0FNU, 32) → MmaMXF8Op, use_mxf8f6f4=True, mma_K=32` | `routeb_official_v2/blockscaled_gemm_dispatch.py` |
| FP4 same-dtype dispatch | `(FP4, FP4, *, Float8E8M0FNU, 32) → MmaMXF4Op, use_mxf8f6f4=False, mma_K=64` | 同文件 |
| MMA 原子形状 | `MmaMXF8Op: shape (16,8,32), .kind::mxf8`；`MmaMXF4Op: shape (16,8,64), .kind::mxf4` | 容器 cutlass DSL `cute/nvgpu/warp/mma.py` |
| 排列粒度 | `perm_k = 32 if use_mxf8f6f4 else 64` | 容器 `utils/blackwell_helpers.py` |
| tile 约束 | FP8 仅允许 (128,128,128)/(128,128,256)；tile_K%128=0；cluster 强制 [1,1,1] | kernel argparse + validate + docstring |
| SMEM 容量 | `SMEM_CAPACITY_MAP: sm_121=101376, sm_120=101376`（**99KB**） | 容器 `utils/smem_allocator.py` |
| 流水 stage 计算 | `ab_stage = (smem_capacity - mbar - epi_bytes) // (ab_bytes_per_stage + sf_bytes_per_stage)` | kernel `_compute_stages` |
| epilogue | `sC` 独立 128×128×c_dtype（BF16=32KB）；`epi_stage = min(epi_stage_max,4)` | kernel `_setup_attributes`/SharedStorage |
| B 操作数 | `NxKxL, B can only be column-major("K")` = K-major；TMA G2S | kernel docstring + run_bs |
| SF 布局 | `[N, ceil(K/sf_vec)]` plain scale → `sf_plain_to_atom` swizzle | kernel `create_scale_factor_tensor` + F0 报告 |

### 1.2 F1 窗口实测锚定【实测】

| 项 | 值 |
|---|---|
| gate_up M=4096 | routeB 0.1562ms / 109.96T；cutlass 0.1116ms / 153.88T；ratio **0.715×** |
| down M=4096 | routeB 0.2406ms / 71.39T；cutlass 0.2118ms / 81.11T；ratio **0.880×** |
| lm_head M=4096 N=32384 | routeB 10.324ms / 105.25T；cutlass 6.441ms / 168.8T；ratio **1.60×（更慢）** |
| Msweep gate_up M=8/96 | routeB **47.1µs / 47.1µs 完全相同**（M pad 128） |
| Msweep down M=8/96/512 | routeB **胜出** 1.24×/1.87×/1.29×（K=512） |
| A-quant M=4096 | g32 0.2952ms vs g128 0.2369ms → **+58.3µs (+24.6%)** |
| FP4 大形状锚点 | routeB FP4 4096×14336×4096 = 349-379T（median 351.6-351.8）【实测，routeb_task12 p4 日志】 |

### 1.3 硬件锚定【知识/实测-外部】

- GB10（DGX Spark, sm_121, 48 SM, L2 24MB, DRAM 273GB/s）张量核 register-resident 实测：
  - **FP8 dense ≈ 248.4 TFLOPS**（mxf8f6f4 指令族）
  - **FP4 dense ≈ 496.6 TFLOPS**（mxf4nvf4，达到 500T spec 的 99.3%）
- 即 **FP4:FP8 = 2:1**（与 MMA 形状 (16,8,64):(16,8,32) 完全自洽）。
- 由此 routeB FP8 105T ≈ **42% 的 FP8 硬件峰值**；routeB FP4 378T ≈ **76% 的 FP4 峰值**；cutlass FP8 169T ≈ **68% 的 FP8 峰值**。

> **诚实标注**：FP8 峰值 248T/FP4 峰值 497T 来自公开 GB10 张量核实测（外部论坛/nvfp4bench + spec），与内部"FP4 峰值 500T【硬件标称】"一致。42%/68% MFU 为 实测 TFLOPS ÷ 该峰值的**推算**口径。

---

## 2. routeB FP8 慢在哪——瓶颈分解

### 2.1 瓶颈一（不可修）：FP8 MMA 指令 = ½ FP4 指令吞吐【实测-源码 + 知识】

- routeB FP8 same-dtype 走 `MmaMXF8Op`：原子 (16,8,32) → 每条 `mma.sync.kind::mxf8` 计算 16×8×32×2 = **8,192 FLOPs**。
- routeB FP4 same-dtype 走 `MmaMXF4Op`：原子 (16,8,64) → 每条 `mma.sync.kind::mxf4` 计算 16×8×64×2 = **16,384 FLOPs**。
- 同一 issue 速率下 FP8 吞吐只有 FP4 一半 → **这是 Blackwell SM120 张量核的硬件属性，不是 kernel 选错指令**（routeB 已命中正确的 `MmaMXF8Op`）。
- 贡献：FP8/FP4 的 **2.0×**（不可修）。这正是"同 kernel FP8 105T vs FP4 378T"中占比最大的部分。

### 2.2 瓶颈二（可修，主嫌疑）：GB10 99KB SMEM → FP8 被压到 2-stage 浅流水【源码推算 + 实测征兆】

**SMEM 账（tile 128×128×128，C=BF16，99KB 预算）**：

| 项 | FP8 | FP4 |
|---|---|---|
| A+B / stage | 128×128×1×2 = **32KB** | 128×128×0.5×2 = **16KB** |
| SF / stage | ~1KB（SFA+SFB [128,4] E8M0） | ~1KB |
| C epilogue | 128×128×2 = **32KB**（epi_stage=1） | 32KB |
| mbar | 1KB | 1KB |
| **ab_stage** | **(99-1-32)/(32+1) ≈ 2** | **(99-1-32)/(16+1) ≈ 3-4** |

- **FP8 锁定 2-stage**；`_compute_stages` 源码直接计算得出。tile_K=128 是 FP8 sf_vec=32 的硬约束（validate 拒绝 tile_K=64），A+B 32KB/stage 无法缩小；C 的 32KB 不可省（epi_stage 已=1）。
- 2-stage 意味着 producer（DMA warp）最多只能超前 1 个 buffer——TMA 延迟无法被充分隐藏。
- **实测征兆**：gate_up M=4096 routeB 156.2µs ≈ DRAM floor 82.7µs + MMA 峰值 69.3µs（**近乎零重叠**）；若充分重叠应为 max(82.7, 69.3) ≈ 83µs（→208T）。FP4 同配置 3-4 stage → 重叠明显更好（76% MFU）。
- **对 lm_head（MMA-bound）的直接影响**：routeB 42% MFU vs cutlass 68% MFU（见 §3）——**这是可修的 ~1.6×**。

### 2.3 瓶颈三（可修，次要）：小 M pad-128 浪费【实测】

- Msweep gate_up M=8 与 M=96 均为 **47.1µs**（完全相同）→ kernel 对 M<128 仍算完整 128 行 tile（M pad 128）→ M=8 时 **16× 计算浪费**，且 pad 行 epilogue 也照写。
- cutlass M=8 gate_up = 22.2µs → routeB 慢 2.1×。
- 对 lm_head decode（M=8/96）同理：M=8 pad 128 下 0.769ms vs M=96 0.706ms（都按 128 行算）。
- 注意：down（K=512）小 M 反而赢（§2.4），说明 pad 浪费在 K 短时被摊薄、且 routeB 固定开销低。

### 2.4 形状层：小 N / 小 K 的 arithmetic-intensity 特性【实测】

- **gate_up（N=512 小 N）**：per-CTA 128×512×4096 算术强度低、N-tile 仅 4 个 → 两 kernel 都远离峰值；routeB 因 2-stage 流水更差（154 vs 110T）。
- **down（K=512 小 K）**：仅 4 个 K-tile，per-tile 工作小 → routeB 的 persistent ping-pong 固定开销低，**小 M 反超 cutlass（M8 1.24×/M96 1.87×/M512 1.29×）**；但 M=4096 主形状仍 <1（C 输出 32MB 写入主导 → DRAM-bound，routeB epilogue 弱）。
- 结论：routeB 的胜场在"小 K + 小 M"（down decode 段），败场在"大 M + 大 K 或大 N"（shared prefill / lm_head）——恰好与生产形态（M=4096 prefill）相反。

### 2.5 E2E 层：A-quant group32 vs group128【实测】

- M=4096 时 g32 比 g128 多 **58.3µs（+24.6%）**；M=512 时 +114%、M=1024 +172%（相对差随 GEMM 变短而爆炸）。
- 这是**独立算子**（`per_token_group_quant_fp8`），不是 GEMM kernel 本身；但对 E2E 叙事，它吃掉了相当一部分 GEMM 增益（routeB 需要 group=32 的 A-quant，现役 cutlass 用 group=128）。

### 2.6 瓶颈贡献汇总表【推算】

| # | 瓶颈 | 贡献量级 | 可修性 | 证据 |
|---|---|---|---|---|
| B1 | mxf8 vs mxf4 指令 ½ 吞吐（硬件） | 2.0× | **不可修** | MMA 形状源码 + GB10 峰值 248/497T |
| B2 | GB10 99KB → FP8 2-stage 浅流水 | ~1.6-1.8×（42% vs 76% MFU） | **可修（内核重构）** | SMEM 账源码 + 156µs≈DRAM+MMA 征兆 + lm_head 42% MFU |
| B3 | M<128 pad 浪费 | M=8 时 16×（形状相关） | 可修（M-trim） | Msweep 47.1µs 恒等 |
| B4 | epilogue C 32KB SMEM 占用 | 间接（占 1/3 SMEM，压死 AB stage） | 可修（直接 TMA store） | SharedStorage + epi_stage 计算 |
| B5 | A-quant g32 vs g128 | +24.6%（E2E, M=4096） | 可修（融合/优化） | aquants JSON |

> 诚实标注：B2 的"≈1.8×"是**推算**——由 FP8 42% MFU vs FP4 76% MFU 反推；精确拆解（是 MMA pipe 空转还是 TMA wait）必须 NCU 确认（P0）。

---

## 3. cutlass FP8 快在哪——优势机制

### 3.1 实测事实【实测】

| 形状 | cutlass | routeB | cutlass 相对 |
|---|---|---|---|
| gate_up M=4096 | 153.9T（62% MFU） | 110.0T（44% MFU） | +40% |
| down M=4096 | 81.1T | 71.4T | +14% |
| lm_head M=4096 | **168.8T（68% MFU）** | 105.3T（42% MFU） | **+60%** |

- lm_head 是干净的 MMA-bound 形状（FLOPs 1.09T，DRAM 仅 1.53ms < MMA 4.38ms）→ **cutlass 68% MFU vs routeB 42% MFU 直接证明 cutlass 流水重叠能力 ~1.6× 更好**。

### 3.2 机制（分三层标注）【实测 + 知识/推算】

1. **运行时形状级选型【知识/推算】**：vLLM `cutlass_scaled_mm` 的 C++ 层（`scaled_mm_c3x` 族）按 (arch, M, N, K) 在多个预编译 tile/stage 配置间选择（grouped runner）。routeB 是**单一固定 128³+2-stage** 配置打天下。cutlass 对生产形状选到的配置在 99KB 内更优（每-stage 占用更小 → 更深流水）。
2. **更深的流水/更小的每-stage 足迹【推算】**：cutlass block-scaled FP8 用 tile_K=64 级 K 迭代 + 更紧凑 SMEM，同一 99KB 上可达 3-4+ stage；TMA producer 可跑远超前 → MMA 少空转。这解释了 68% vs 42% MFU。
3. **更粗的 scale 布局（SF 流量 ÷4）【知识/推算】**：现役 cutlass 走 DeepSeek 128×128 block-scale（`scale_b [K/128,N/128]`，在 epilogue/MMA 侧应用），SF 元素数 = routeB [N,K/32] 的 **1/4**，且无需 per-K-group atom swizzle。**代价是精度粒度粗（128 vs 32），这是 cutlass 用精度换带宽/指令的取舍**。
4. **工程成熟度【知识】**：生产路径 `Fp8LinearMethod → CutlassFp8BlockScaledMMKernel`，含 `process_weights_after_loading` 的权重布局预处理（B.T 视图）与多年调优；routeB 是研究示例级 kernel。

### 3.3 为什么小 M 也赢【实测】

- gate_up M=8：cutlass 22.2µs vs routeB 47.1µs（routeB M-pad 128 → 16× 浪费）。
- down M=8：cutlass 15.7µs vs routeB 12.7µs（**routeB 反赢**）——说明 routeB 的 persistent ping-pong 启动/固定开销其实不高，输在 pad 与流水，不是启动。

> **诚实标注**：cutlass C++ 侧精确 tile/stage 配置未能在编译产物（`.so`）中直接抽取，§3.2 的 1/2 为 vLLM 公开架构知识 + MFU 差距的推算，非源码级证据。核心结论（cutlass 重叠更好）由 68% vs 42% MFU 的实测支撑。

---

## 4. 改进空间评估表

> 前提口径：FP8 硬件峰值 ≈248T；"超越 cutlass" = GEMM 侧 > 154-169T。所有预期收益为【推算】区间，须经 P0/P1 微基准证实。

| 方向 | 证据 | 机制 | 预期收益（GEMM 侧） | 优先级 | 风险/成本 |
|---|---|---|---|---|---|
| **A. FP8 same-dtype 指令路径"对齐 FP4"** | dispatch 已用正确 MmaMXF8Op（16,8,32）【实测-源码】；GB10 FP8 峰值 248T【知识】 | **伪命题**：FP8 硬件上限 248T，永远到不了 FP4 378T。可修的是流水不是指令 | 指令层面 **0**；流水修复后 1.4-2.0× | **不做指令改动** | 低（无动作） |
| **B. SMEM/流水重构：消除 sC 直接 TMA store + 挤 3rd AB stage** | sC 32KB 占 99KB 的 1/3【源码】；2-stage 限制【源码推算】；156µs≈DRAM+MMA 征兆【实测】；lm_head 42% MFU【实测】 | 释放 C 占用的 32KB → 3×33KB+1KB ≈ 100KB（需再挤 SF/对齐 ~1KB）→ AB 3-stage → TMA 超前 2 buffer → MMA 空转大幅减少 | 105T → **150-200T（60-80% MFU，1.4-2.0×）**；lm_head 上可超 cutlass 169T | **P1（核心）** | 高：内核级改动 + SASS/数值回归；需 P0 先确认瓶颈确为流水 |
| **C. tile 尺寸/调度调整** | FP8 allowed tiles 仅 128³/128×128×256；cluster 强制 [1,1,1]【源码】 | 更大 tile 在 99KB 上更糟（128×256 → 1 stage）；更大 M tile 不受支持 | **≈0**（此方向是死路） | **不做** | — |
| **D. 内存布局消除 staging** | B K-major 已零拷贝【源码】；SF [N,K/32] 需 atom swizzle（小开销）；唯一大 staging 是 C epilogue | 与方向 B 重合：C staging 消除是核心；SF swizzle 仅 ~1KB 级 | 计入 B；独立看 **+3-5%** | P1（并入 B） | 低-中 |
| **E. A-quant group32 融合/优化** | aquants JSON：M=4096 +24.6%、M=1024 +172%【实测】 | 融合进 kernel 或优化 per_token_group_quant_fp8 的 group=32 路径 | E2E **-58µs（M=4096）~ -78µs（M=1024）** | **P1（必要条件）** | 中：量化算子侧，与 GEMM 正交 |
| **F. 小 M 段 M-trim** | down M8/M96 已胜（K=512）【实测】；gate_up M=8 pad 128 → 47µs vs cutlass 22µs【实测】；lm_head M=8 0.769ms | kernel 支持 M<128 尾 tile 裁剪（跳过 pad 行 MMA/epilogue） | gate_up M8: ~47→~25µs；lm_head decode 每步省 ~0.4-0.5ms | P2（若 lm_head 立项） | 中：需尾 tile 边界 mask 源码改造 |

---

## 5. 结论与路线图

### 5.1 核心裁决

1. **"FP8 无法超越 cutlass" 为时过早**。当前 F1 判据下（0.715×/0.880×）确实 FAIL；但取证显示 routeB FP8 只有 **42% MFU**，而同一 kernel FP4 有 76%、cutlass FP8 有 68%——**FP8 路径存在 ~1.6× 的可修流水差距**，修完 GEMM 侧可达 150-200T，**可以超过 cutlass 的 154-169T**。
2. **但修复不是"指令路径修正"**（那是伪命题：FP8 硬件上限 248T），而是**内核级 SMEM/流水重构**（方向 B），工作量大、确定性中。
3. **E2E 还需 A-quant group32 融合**（方向 E），否则 GEMM 修复的增益会被 +24.6% 的 A-quant delta 吃掉一半。

### 5.2 路线图

| 阶段 | 动作 | 判据（go/no-go） | 预计 |
|---|---|---|---|
| **P0（纯验证，1 窗口）** | NCU 微基准 routeB FP8 @ M=4096 三形状 + SASS 抽验：确认 stall 主因是 `tma_wait/pipe_wait`（流水假设）还是 MMA pipe 本身（不可修）；顺带验证 mxf8 指令计数 | 若 stall = pipeline → 方向 B 可投；若 = MMA pipe → **维持 cutlass，路线关闭** | 0.5 天窗口 |
| **P1（若 P0 为正）** | routeB FP8 内核重构：①epilogue C 直接 register→TMA store（去 sC 32KB）②挤 3rd AB stage（SF 布局/对齐再省 ~1KB）③（可选）M<128 trim。golden vs F1 同形状 | GEMM ≥ **150T（gate_up/lm_head）** 且 rel_err ≤1e-2 | 2-4 天 kernel + 1 天回归 |
| **P1（并行）** | A-quant group32 优化/融合 | A-quant delta 降到 ≤10µs（M=4096） | 1-2 天 |
| **P2（条件）** | lm_head 立项时：N=32384 pad + decode M-trim + 质量门（G1-G7） | 复用 F0 契约与 F1 数据 | 随 lm_head 排期 |
| **不做** | ①FP8→FP4 级效率（248T 上限）②更大 tile（99KB 死路）③b12x 方向（与 FP8 无关） | — | — |

### 5.3 对 F1 判决的更新建议

- **维持现役 `CutlassFp8BlockScaledMMKernel` 为生产默认**（F1 判决不变，直到 P0/P1 数据为正）。
- **routeB FP8 从"设计储备"升级为"P0 验证候选"**：不是因为它当前达标，而是因为取证表明**瓶颈可修且修复上限 > cutlass**。
- 若 P0 显示流水假设不成立（MMA pipe 已饱和），则 routeB FP8 正式关闭，lm_head/shared 的 FP8 叙事退回"显存收益 + 精度"，吞吐叙事放弃（与 F1 一致）。

---

## 6. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| **【实测】** | F1 shared/lm_head 全表（JSON + log：gate_up/down/lm_head/Msweep/aquants）；routeB FP4 大形状 349-379T（routeb_task12 p4 日志）；Msweep M=8/96 恒等 47.1µs；down 小 M 反超 |
| **【实测-源码】** | routeB dispatch 表（MmaMXF8Op mma_K=32 / MmaMXF4Op mma_K=64）；cutlass DSL MmaMXF8Op (16,8,32) / MmaMXF4Op (16,8,64) 定义；`get_permutation_mnk` perm_k=32/64；`SMEM_CAPACITY_MAP` sm_121=101376；`_compute_stages` 公式；FP8 tile 约束（128 倍数、cluster [1,1,1]、B K-major）；`run_bs` 计时路径 |
| **【知识/实测-外部】** | GB10 FP8 dense ≈248.4T / FP4 dense ≈496.6T（公开 GB10 张量核实测）；GB10 48SM/L2 24MB/273GB/s；vLLM cutlass_scaled_mm grouped runner 架构 |
| **【推算】** | SMEM 账（FP8=2 stage / FP4=3-4 stage）；gate_up 156µs≈DRAM+MMA 征兆；42%/68% MFU 分解；B2 的 1.6-1.8× 贡献；方向 B 收益 150-200T；方向 E 收益 -58µs |
| **【待验证】** | NCU 确认 stall 主因（P0）；方向 B 重构的 3rd stage 是否能在 99KB 内挤出（SMEM 账差 ~1KB，需实际编译验证）；cutlass 精确 tile/stage 配置（无法从 .so 抽取） |

**诚实声明**：
1. **B2 的量化（1.6-1.8×）是推算**：由"FP8 42% MFU vs FP4 76%/cutlass 68% MFU"反推，未经 NCU 逐 cycle 拆解；P0 前不可当作既成事实用于立项。
2. **"FP8 硬件峰值 248T"来自公开 GB10 张量核 register-resident 实测**，与内部 FP4 峰值 500T 口径一致；若实际运行时钟/功耗不同，MFU 数值会有 ±5-10% 漂移，但 2:1 的 FP4:FP8 关系由 MMA 形状在源码级固定。
3. **F1 的 cutlass 对照为 kernel-only**（`ops.cutlass_scaled_mm` 直接调用，block-128 scale），与 routeB kernel-only 对比公平；两者 A-quant 均未计入（A-quant delta 单列）。cutlass 的 block-128 与 routeB 的 sf_vec=32 是**不同量化粒度**的取舍——cutlass 以精度粒度换性能，routeB 反之。
4. 方向 B 的"3rd stage 挤入"在账面上差 ~1KB（3×33KB + 1KB mbar = 100KB > 99KB），必须通过消除 sC（-32KB）后重新分配 + 压缩 SF/对齐实现；**能否落地依赖实际编译与 SMEM 对齐，P1 前不可承诺**。

---

## 7. 引用索引

- F1：`fp8-f1-window-2026-08-24.md` + `/tmp/_fp8_f1/`（f1_shared_M4096.json / f1_shared_Msweep.json / f1_lmhead_32384.json / logs / run_f1_window_a.sh / f1_shared_fp8_bench.py / f1_lmhead_bench.py / verify_routeb_large.py / lmhead_cutlass_base.py）
- 前序：`opt-routeb-fp8-2026-08-23.md`、`upstream-check-perf-ceiling-2026-08-23.md`、`lmhead-fp8-f0-contract-2026-08-24.md`、`fp8-quality-impact-2026-08-23.md`、`bprime-window-2026-08-23.md`、`routeb-p4-ab-perf-2026-08-21.md`、`/tmp/routeb_task12/p4/*.log`（FP4 350-379T）
- 服务器源码：`node01:<INSTALL_DIR>/nvfp4/routeb_official_v2/`（dispatch + pingpong）；容器 `nvidia_cutlass_dsl`（mma.py、blackwell_helpers.py、smem_allocator.py、static_persistent_tile_scheduler.py）；容器 `vllm/model_executor/kernels/linear/scaled_mm/`
- 外部：GB10 张量核 FP8/FP4 实测（nvfp4bench / NVIDIA 论坛；GB10 spec）

*本报告由工程保障团队（系统架构师 architect-1）生成；纯只读分析，未启动 GPU/未改生产。P0/P1 立项与否请由人类工程负责人结合 NCU 微基准裁定。*
