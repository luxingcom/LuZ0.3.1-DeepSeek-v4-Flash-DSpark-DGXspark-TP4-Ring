# 原生 FP4 MMA 算子性能归因与提升路线（60~187 → 350 TFLOPS）

> 日期：2026-08-20 | 现状：生产实测 60~187 TFLOPS（预期 350 = GB10 FP4 dense 500 的 70%）
> 差距：利用率仅 **12~37%**——说明存在结构性开销或 shape/配置失配，非硬件瓶颈

---

## 一、差距归因（先定位再优化，P0）

### 1.1 60 vs 187 的跨度 = shape/开销依赖信号

| 假设 | 验证方法 | 判定依据 |
|---|---|---|
| **A. shape 依赖**（小 M 利用率低） | 按 M 扫描 | M=256 vs 4096 若差 3× → tile/占用问题 |
| **B. A 量化开销占比**（GEMM-only vs 全链路） | 拆测 | 全链路 - GEMM-only 若 >20% → 量化未融合/未优化 |
| **C. W/scale 预处理未缓存** | 首调 vs 稳态 | 每次调用重做 repack/swizzle → 拖累 |
| **D. tile 配置失配**（SM120 smem 99KB） | 换 tile 实例 | 128×128×128 vs 256×128×128 vs 128×256×64 |

### 1.2 理论边界（roofline）

| 项 | 值 |
|---|---|
| GB10 FP4 dense 峰值 | **500 TFLOPS**（nvfp4bench 实测 500.2） |
| 350 目标 = 70% | 需大 M + 最优 tile + 零多余开销 |
| 187 当前 = 37% | 大概率受 shape/量化开销限制 |

**MoE 真实 shape**：w1/w2 (M×4096×2048)、w3 (M×2048×4096)、M=256~4096。
→ N=2048 时 tile 覆盖（128 块 = 16 个）OK，但 K=2048 时 A 量化（M×2048 读 fp32）占比更高。

---

## 二、提升路线（按收益排序）

### P1. A 量化路径优化（最可能的主因，预期 +50~100 TFLOPS）

**现状假设**：A 量化独立 kernel（fp32 读 M×K + 量化 → fp4 打包），GEMM 再读 fp4 A。
- **P1a 量化 kernel 大 tile 化**：BLOCK_M 128/256 + 向量化（同 kernel② v17 的教训：减 block 增负载）
- **P1b A 输出 fp4 打包**（非 fp8/fp32）：GEMM 读 A 带宽减 8×（vs fp32）
- **P1c 量化与 GEMM 流水**：A 量化 kernel 与上一个专家 GEMM 重叠（CUDA stream / 双 buffer）
- **P1d 量化融合进 GEMM**（终极）：CUTLASS EVT 或 mainloop 内融合——需改内核源码（方案 B 内核若可 patch）

### P2. W/scale 预处理缓存（预期消除 5~15% 抖动）

- `w_repack + scale_swizzle` 每层一次（`process_weights_after_loading`），forward 零重复
- 确认当前集成是否已缓存（首调 vs 稳态）

### P3. Tile/配置调优（预期 +20~50 TFLOPS）

- CUTLASS 内核多实例化：`128×128×64 / 128×128×128 / 128×256×64 / 256×128×128`（SM120 smem 99KB 约束）
- GROUP_M swizzle（L2 友好）
- autotune key 含 (M,N,K)，按 MoE 真实 shape 预热

### P4. CUDA Graph 化（预期消除 launch 抖动）

- 整个 prefill MoE（quant + 3 专家 GEMM）graph 化（分 phase 捕获，R2 原则）
- decode/prefill 分别捕获

### P5. 内核替换对照（若 P1~P4 仍不足）

- **FlashInfer 0.6.8+ CUTLASS NVFP4 backend** 对照（社区实测 DGX Spark 65 tok/s OOTB，SM12x 优化 kernel 已集成）
- Luke Alonso cuteDSL/TileIR SM12x 内核（已入 FlashInfer TOT）
- 目标内核应 ≥350（70%）；若均不达 → 检查 A 量化是否成为瓶颈主因

---

## 三、分步执行

```bash
# ① 归因（5 分钟）：shape 扫描 + GEMM-only vs 全链路拆分
python perf_diag_fp4_gemm.py --backend cutlass   # 或 --backend flashinfer

# ② 按归因结果选择：
#    若 A 量化占比 >20%      → P1a/P1b/P1c（量化 kernel 优化 + 流水）
#    若 shape 依赖明显       → P3（tile 调优，按 MoE shape 预热）
#    若首调/稳态差距大       → P2（W 缓存）
#    都优化后               → P4（graph）+ P5（内核对照）
```

## 四、验收

| 阶段 | 目标 |
|---|---|
| 归因后基线 | 记录 60~187 的 shape 表（哪些 shape 187、哪些 60） |
| P1~P4 后 | **≥250 TFLOPS**（50%，中/大 M） |
| P5 对照后 | **≥350 TFLOPS**（70%，大 M） |
