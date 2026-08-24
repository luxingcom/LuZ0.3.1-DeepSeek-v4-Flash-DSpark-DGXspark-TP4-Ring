# NVFP4 delivery #5 复测报告（kernel② v17 + kernel① 双路径 A/B 对比）

> 日期：2026-08-20 ｜ 环境：DGX Spark 生产 TP4，`vllm-tp4-rank0`（torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a）
> 输入：`nvfp4-kernels-delivery(5).zip`（kernel② v17 手写优化版；kernel① 文件与 #4 完全一致）
> 用户要求：① 二号算子更新测试验证；② 一号算子两条实现路径完善修复后开展对比测试

---

## 〇、结论速览

| 项 | 结论 |
|---|---|
| **kernel② v17** | ✅ **8/8 逐字节 PASS + 大 T 194~262 GB/s（3.5~4.6× v11）**——达标（≥120 目标超额 1.6~2.2×），**建议替换 v11 部署** |
| kernel① **路径 A（Triton v16 fp8）** | 修复完成（F1+F2）：8/8 PASS 误差 0，但 **0.1~0.2 TFLOPS**；fp8 plain-dot 探针证 **Triton 3.6.0 sm_121 无任何原生 FP8 MMA codegen** → 思路在 3.6 上**不可救**，弃用 |
| kernel① **路径 A（Triton v15 bf16）** | ✅ 8/8 + GEMM-only **26.7~81.4 TFLOPS**——**Triton 3.6 可达最优，维持推荐** |
| kernel① **路径 B（CUTLASS mmaf）** | ❌ 环境三阻塞（B1 torch 头 / B2 CUTLASS 3.9 源码 / B3 layout_SFA·SFB TODO）+ round12 已判改道 **PR #42209**——本环境不可编译 |

---

## 一、kernel② kv_linear v17 —— 达标，建议部署 ✅

### 1.1 正确性：8/8 逐字节 PASS（shipped test，7 档 T + kv 入口，atol=0）

### 1.2 带宽：大 T 194~262 GB/s（3.5~4.6× v11），达标

统一 4680B/token 口径（读 1024×4 + 写 584），理论上限 273 GB/s：

| T | v11（旧冠军） | **v17** | v17/v11 | 理论占比 |
|---|---|---|---|---|
| 256 | 50.8 | **75.6** | 1.49× | 28% |
| 1024 | 56.8 | **262.3** | 4.62× | **96%** |
| 4096 | 61.0 | **248.9** | 4.08× | 91% |
| 16384 | 58.8 | **227.0** | 3.86× | 83% |
| 65536 | 55.2 | **194.3** | 3.52× | 71% |

**验收判据 ≥120 GB/s（2.3× v11）：达标（大 T 194~262，超额 1.6~2.2×）。**

### 1.3 配置归因（T=65536，kernel-only）

| BLOCK_G | TPP | warps | GB/s |
|---|---|---|---|
| **32** | **1** | 1 | **213.5（最优）** |
| 32 | 2 | 2 | 212.1 |
| 16 | 4 | 4 | 199.0 |
| 8 | 8 | 2 | 169.1 |

S1（多 token/多组负载）+ M1（16 元素连续 load + multiple_of）+ S3（zeros 内联 pad）组合有效，正是 v15 缺失的"减 block 增负载"方向。

### 1.4 建议
- **v17 替换 v11 部署**（大 T 3.5×+，正确性零漂移）；小 T（T=256）1.49× 也提升。
- 保留 v11 作回退；paged v11 维持。

---

## 二、kernel① 双路径 A/B 对比（完善修复后）

### 2.1 路径 A：Triton 两分支对比（本轮实测）

| 指标 | **v15（bf16 MMA）** | **v16-fixed（fp8 e4m3 scaled）** |
|---|---|---|
| 修复状态 | —（#4 轮 D1 修复后） | F1 位置参数 + F2 k_pack=True（两行，完成） |
| 正确性 | 8/8，误差 0 | 8/8，误差 0 |
| **性能** | **GEMM-only 26.7~81.4 TFLOPS** | **0.1~0.2 TFLOPS** |
| SASS | HMMA.16816.F32.BF16（原生 bf16 MMA） | 100,289 行；64× HMMA.BF16 + FFMA 264（**bf16 MMA + 标量 scale 软件模拟**） |
| 判定 | ✅ **推荐（Triton 3.6 最优）** | ❌ 弃用 |

### 2.2 路径 A 决定性探针：Triton 3.6.0 sm_121 无原生 FP8 MMA

为验证 v16 的 E2M1→e4m3 思路能否以"普通 fp8 dot"方式救活，本轮新增最小探针：
```
tl.dot(a_fp8(e4m3), b_fp8(e4m3), acc)  →  PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                                         SASS: HMMA.16816.F32（FP16 载体，无 E4M3 类型）
```
**普通 fp8 dot 也降级为 FP16 MMA**。结合上轮 dot_scaled(e4m3)→bf16+模拟的实锤：
**Triton 3.6.0 在 sm_121 上不存在任何原生 FP8 MMA codegen（scaled 与 plain 均无）**。
→ v16 的 fp8 思路在 3.6 上**无 kernel 级修复可救**（编译器能力边界，非算子 bug）。
→ 上轮"路径 A 完善修复"的结论就此固化：fp8 载体路线须等 Triton 3.7+ 或换 CUTLASS。

### 2.3 路径 B：CUTLASS mmaf_sm120 —— 环境阻塞（本环境不可编译）

- 本轮确认 `.cu` 与 #4 **逐字节相同**（无任何修复交付）；
- B1：容器 torch 缺 C++ 开发头（`torch/all.h` 不存在）→ torch extension 编译断；
- B2：CUTLASS 3.9 源码 GitHub 限速不可得（捆绑 4.3.4 为裁剪子集、PyPI 包无 C++ 头）；
- B3：blockscaled `Arguments` 必填 `layout_SFA/layout_SFB`（.cu 的 TODO 标注，捆绑头已实证）；
- round12 已判：**改道 vLLM PR #42209 `nvfp4_scaled_mm_sm120_kernels.cu`**（完整 vLLM 源码构建环境内集成 + SASS 门禁 `mmaf|e2m1|tcgen05`）。

### 2.4 A/B 对比结论（kernel①）

```
路径 A（Triton 3.6）：v15（bf16 MMA）为唯一可用实现，26.7~81.4 TFLOPS；v16（fp8）性能崩塌弃用
路径 B（CUTLASS）  ：本环境不可构建；走 PR #42209 后与 v15 对比（≥1.5× 才值得切换）
400 TFLOPS 目标   ：仅路径 B（PR #42209 / Triton 3.7+）可达；SASS 门禁 = grep mma.*e4m3|tcgen05
```

---

## 三、交付物

- 本报告 + `run_logs.md` + 脚本：`bench_k2_v17.py` / `probe_fp8_dot.py`（另含 #4 轮的 `patch_v16_fixed*.py` / `verify_v16_fixed.py` / `bench_v16_k1.py` / `sass_v16.py` 复用于路径 A）
- 证据：`evidence/`（k2_v17_result.txt + k1_ab_result.txt）
- 容器：`/vllm-workspace/nvfp4-delivery-v17/`

生产 4 rank 全程 healthy、GPU 0%、无残留进程，未做恢复操作。
