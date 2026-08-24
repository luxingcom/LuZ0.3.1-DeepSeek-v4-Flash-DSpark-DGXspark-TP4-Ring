# NVFP4 双算子 SASS 诊断与真机复测报告（第七轮 / SASS 轮）

> 日期：2026-08-19 ｜ 交付包：nvfp4-kernels-delivery(2).zip（终版 v12.1 / v13）
> 环境：DGX Spark 生产 TP4 — node01 / `vllm-tp4-rank0` 容器
> torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a（GB10）
> 执行：EngineeringAssuranceTeam（生产容器内跑 pytest/benchmark，生产 vLLM 未受影响，GPU 0%）

---

## 〇、执行摘要（一句话结论）

**P0 SASS 诊断确认：`tl.dot_scaled(e2m1)` 在 Triton 3.6.0 / sm_121 被降级编译为 BF16 张量核心 MMA（`HMMA.16816.F32.BF16`），而非原生 Blackwell FP4 MMA（`tcgen05.mma` / `mma.sync...e2m1`）。** 这坐实了第六轮"强怀疑降级"的判断（细化：是 bf16 张量核心 MMA，不是 fp32 标量 FFMA），并解释了 kernel① 30~47 TFLOPS（GEMM-only）/ 7~20 TFLOPS（全算子）远低于 500 TFLOPS FP4 峰值的原因。两个算子正确性均全绿（kernel① 8/8、kernel② 7/7、paged 5/5）。

---

## 一、P0 SASS 诊断（kernel① prefill_gemm）

### 1.1 方法（修正了交付脚本的一个坑）

交付的 `sass_check_prefill_gemm.sh` 依赖 `cuobjdump --dump-sass`，而该工具内部需要 `nvdisasm`。
容器内 `cuobjdump` 存在，但 `nvdisasm` 不在 `PATH`，导致脚本 `cuobjdump --dump-sass` 静默 fatal
（`Could not find executable file 'nvdisasm'`），grep 拿到空结果（mma=0 且 FFMA=0，双零，属假阴性）。

**修正**：在容器内定位到 `nvdisasm`（`/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm`），
直接 `nvdisasm <cubin>` 拿到真实 SASS（11307 行）。同时 Triton 缓存里的 **`.ptx`** 也作为交叉验证。

- 触发编译：运行 `_nvfp4_gemm_kernel`（v12 文件，v13 的 GEMM kernel 与其逐字相同）于 M=256,K=4096,N=4096
- 编译缓存：`/root/.triton/cache/AZPILW6GB4IC4CSCEB5VFP2K2FXRXC7LBKLF2E6HTHM6B2SJX6ZQ/_nvfp4_gemm_kernel.{cubin,ptx}`

### 1.2 证据（grep 结果）

**SASS（`nvdisasm` 直出）：**
```
HMMA.16816.F32.BF16  ← Blackwell BF16 张量核心 MMA（出现 129 次）
TCGEN05              ← 0 次（新架构 FP4 张量核心路径未使用）
e2m1                 ← 0 次（无 FP4 MMA 编码）
FFMA / FADD / FMUL   ← 0 次（非 fp32 标量退化）
```

**PTX（Triton 生成意图，256 条 mma 全部为）：**
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   ← bf16 MMA
e2m1   ← 0 次；  tcgen05 ← 0 次
```

### 1.3 判定

| 假设 | 期望 SASS 特征 | 实测 | 结论 |
|---|---|---|---|
| 原生 FP4 MMA 生效 | `tcgen05.mma` / `mma.sync...e2m1` | 0 次 | ❌ 未生效 |
| fp32 标量退化（第六轮最初猜） | 大量 `FFMA` | 0 次 | ❌ 也非此 |
| **bf16 张量核心 MMA（e2m1→bf16 软件解包 + 逐组 scale）** | `HMMA.16816.F32.BF16` | **129 次** | ✅ **命中** |

**结论**：`dot_scaled(e2m1)` 在 Triton 3.6.0/sm_121 的 codegen 把 4-bit 操作数解包/转换为 bf16，
再走 bf16 张量核心 MMA。**未使用原生 Blackwell FP4 张量核心**——4-bit 的密度优势在计算端丢失，
且寄存器内的 e2m1→bf16 解包 + 逐组（32 元素）scale 处理开销巨大，这是性能天花板的根因。

> 注：v13 = v12 + W 预处理缓存（`_WEIGHT_CACHE`），二者 GEMM kernel（`tl.dot_scaled` 调用）逐字相同，
> 故本 SASS 结论对 v12 与 v13 同时成立。

---

## 二、kernel① prefill_gemm v13 复测

### 2.1 正确性（vs torch 参考，rtol/atol=5e-2）

| M | K | N | bias | 结果 | max_abs_err |
|---|---|---|---|---|---|
| 256 | 4096 | 4096 | F | ✅ | 0.0000 |
| 256 | 4096 | 4096 | T | ✅ | 0.0000 |
| 512 | 2048 | 4096 | F | ✅ | 0.0000 |
| 512 | 2048 | 4096 | T | ✅ | 0.0000 |
| 1024 | 4096 | 2048 | F | ✅ | 0.0000 |
| 1024 | 4096 | 2048 | T | ✅ | 0.0000 |
| 128 | 4096 | 4096 | F | ✅ | 0.0000 |
| 128 | 4096 | 4096 | T | ✅ | 0.0000 |

**CORRECTNESS: 8/8 PASSED**（全部逐字节精确，误差 0.0000）。

### 2.2 性能（TFLOPS，对比 round6 基线 20.5~45.8）

| M | K | N | 全算子(含A量化+W重打包) | GEMM-only(A预量化,W缓存) | round6 基线 |
|---|---|---|---|---|---|
| 256 | 4096 | 4096 | 7.1 | 29.6 | — |
| 512 | 4096 | 4096 | 7.8 | 38.8 | — |
| 1024 | 4096 | 4096 | 8.0 | 32.7 | — |
| 256 | 8192 | 8192 | 12.8 | 38.1 | — |
| 512 | 8192 | 8192 | 13.7 | 44.3 | — |
| 1024 | 8192 | 4096 | 8.2 | 46.9 | — |
| 256 | 4096 | 16384 | 19.6 | 37.6 | — |

**解读**：
- **GEMM-only 29.6~46.9 TFLOPS**，与 round6 基线（20.5~45.8）一致甚至略高 → v12/v13 GEMM kernel 本身健康、无回退。
- **全算子 7.1~19.6 TFLOPS**：A 量化 kernel + 主机侧 `preprocess_weights`（W 重打包 + scale 展开）开销
  把端到端吞吐拉低，小 M 尤甚。这是"分离量化架构"尚未把量化开销完全移出关键路径的体现。
- **两者都远低于 ~500 TFLOPS FP4 峰值** → 根因即 §一 SASS 结论：bf16 退化，未走原生 FP4 MMA。

---

## 三、kernel② kv_linear v12.1 复测

### 3.1 正确性（逐字节 vs torch 参考）

| T | shape | dtype | 结果 |
|---|---|---|---|
| 1 / 4 / 16 / 64 / 256 / 1024 / 4096 | (T,584) | uint8 | ✅×7 |

**CORRECTNESS: 7/7 PASSED（逐字节精确一致）** — grid 64 修复彻底，相比 v11 架构无精度回退。

### 3.2 带宽（GB/s，口径 bytes/token = 1024×4 + 584 = 4680；理论上限 273 GB/s）

| T | GB/s | 带宽利用率 |
|---|---|---|
| 1 | 0.4 | — |
| 4 | 1.5 | — |
| 16 | 5.2 | — |
| 64 | 12.2 | — |
| 256 | 16.6 | ~6% |
| 1024 | 18.3 | ~7% |
| 4096 | 18.7 | ~7% |
| 16384 | **18.8** | ~7% |

**解读**：大 T 带宽稳定在 ~18.8 GB/s（≈7% 理论极限），与 round6 结论一致。
对比历史：v11（生产 v4 架构）大 T ~58.4 GB/s（21%）→ **v12.1（MCP v2 架构）倒挂约 3×**。
kv_linear 是 memory-bound 算子，**验收主指标应为 GB/s 而非 speedup**；当前 7% 远未发挥硬件。

### 3.3 paged 变体（维持 v11）

`pytest test_nvfp4_ds_mla_kv_linear_paged.py` → **5 passed**（217s，编译慢但干净）。
paged 维持 v11，结论不变。

---

## 四、决策与下一步（回传给交付方）

基于 P0 SASS 结论，给出明确方向：

**kernel①（prefill_gemm）：bf16 退化已坐实 → 必须换原生 FP4 MMA 路径才能冲 400 TFLOPS**
- **方案 A（推荐先试）**：升级 Triton 至 3.7+ / nightly，修 sm_120/121 的 `dot_scaled` e2m1 codegen
  （含此前发现的 uint8 scale MLIR bug）。需复测 SASS 是否出现 `tcgen05.mma` / `mma.sync...e2m1`。
- **方案 B**：CUTLASS 手写 `mmaf_scaled`（m16n8k32 e2m1）tile kernel 替换 `dot_scaled`，绕过 Triton codegen。
- **方案 C**：DeepGEMM sm12x 分支（`DG_JIT_USE_NVRTC=off`），确认 sm_121 覆盖。
- 验证手段：本论 SASS 流程（清 Triton 缓存 → 跑 kernel → `nvdisasm` 直出 → grep `tcgen05`/`e2m1`）应成为
  每次 kernel① 改动的**准入门禁**——只要 SASS 仍只有 `HMMA.16816.F32.BF16`，就不算真 FP4。

**kernel②（kv_linear）：架构层面回退 → 带宽级重写**
- 目标：宽 tile 化（每 program 处理 8~16 组 × 多 token，连续 512B~1KB 读）、128-bit 向量化、scale 批量计算、
  减小 int64 开销 → 大 T 150~200 GB/s（55~73%）。
- 当前 v12.1 的 7% 是 MCP v2 窄向量化 + 标量串行循环导致，与 v11 的 21% 相比是架构倒挂，需重做。

**部署建议（不变）**：kernel① v12/v13、kernel② v12.1、paged v11 正确性全部全绿，
可先部署获得 14~30× 于 fp32 参考的收益；但 **kernel① 性能天花板（FP4 未生效）与 kernel② 带宽（7%）是已知未达目标项**，
需在下一迭代按上述方向解决。

---

## 五、交付物清单（对应交付包 README §六"回传给我"）

1. ✅ **SASS 诊断输出**：`sass_evidence/SASS_GREP_SUMMARY.txt` + 完整 `_nvfp4_gemm_kernel.sass.txt`（11307 行）+ `.ptx`
   → 判定：**`HMMA.16816.F32.BF16` ×129，无 `tcgen05`/`e2m1` → FP4 MMA 未生效（bf16 退化）**
2. ✅ **kernel① TFLOPS 表**：§二.2（全算子 7.1~19.6 / GEMM-only 29.6~46.9，对比 20~46 基线）
3. ✅ **kernel② GB/s**：§三.2（大 T ~18.8 GB/s，≈7% 理论极限）

附：复测运行日志 `run_logs.md`、复测脚本 `verify_v13.py` / `gemm_only_bench_v13.py` / `verify_v121.py`。

---

*Generated by EngineeringAssuranceTeam — NVFP4 kernel SASS verification round (2026-08-19)*
