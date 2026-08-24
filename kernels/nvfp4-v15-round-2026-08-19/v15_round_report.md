# NVFP4 双算子 v15 轮复测报告（delivery #3）

> 日期：2026-08-19 ｜ 环境：DGX Spark 生产 TP4，`vllm-tp4-rank0`（torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a）
> 输入：`nvfp4-kernels-delivery(3).zip`（kernel① v15 Triton + CUTLASS mmaf_sm120 骨架；kernel② v14/v15）
> 范围：README §四/§六 + round10 报告 §六 —— ① v15 正确性+性能、① CUTLASS 构建+SASS 门禁、② v15 正确性+带宽、paged 复测

---

## 〇、结论速览

| 算子 | 版本 | 正确性 | 性能 | 判定 |
|---|---|---|---|---|
| ① prefill_gemm | **v15 Triton**（bf16 MMA） | ✅ **8/8**（repack 适配后，误差 0.000000） | GEMM-only **26.7~81.4 TFLOPS**；全算子 7.2~14.9 | ⚠️ **D1 布局缺陷：直喂生产格式即崩溃**；修复后可部署（3~5× 提升兑现一半），仍非 FP4 |
| ① prefill_gemm | **CUTLASS mmaf_sm120**（原生 FP4，400 目标） | — | — | ❌ **构建三连阻塞**（B1/B2/B3），骨架不可编译；建议改道 vLLM PR #42209 |
| ② kv_linear | **v15**（宽 tile） | ✅ **7/7 逐字节** | **10.6 GB/s**（4680B/token） | ❌ **比 v11(53.4) 慢 5×，比 v12.1(18.8) 慢 1.8×** —— 声称的 46.30× 未复现，**不建议采用** |
| ② kv_linear | v14 | 未测 | 未测 | ❌ **D2 编译即挂**（模块级 `_GROUP` 非 constexpr） |
| ② kv_linear | paged v11 | ✅ **5/5**（干净重跑） | — | ✅ 维持 |

---

## 一、kernel① prefill_gemm v15（bf16 MMA 路径）

### 1.1 正确性：8/8 PASS，且误差为 0

8 用例（4 shape × bias T/F，rtol/atol=5e-2）全部 PASS，**max_abs_err = 0.000000**。
原理：A/W 反量化值 = E2M1 码本 × 2^exp，bf16 可**精确**表示（E2M1 尾数 ≤2bit，×2 幂无舍入），fp32 累加与 torch fp32 参考仅累加序差异 → 误差 ~1e-7 量级，6 位小数显示为 0。

> ⚠️ **D1 布局缺陷（必改）**：v15 `_unpack_fp4_weights` 期望 **K 向打包 `[K//2, N]`**（lo=偶 K 行），
> 而生产转换器 `convert_mxfp4_to_nvfp4.py`、torch 参考、shipped 测试均为 **N 向打包 `[K, N//2]`**（lo=偶 N 列）。
> **as-shipped 直喂生产格式即崩溃**（已复现）：
> ```
> RuntimeError: shape '[4096, 2048]' is invalid for input of size 16777216
> ```
> v12/v13 的 wrapper 内置 `_repack_w_for_rhs_k_pack` 完成转换，**v15 丢失该逻辑**。本次复测在 harness 层做了等价 repack 才通过。
> 修复方向：`_unpack_fp4_weights` 增加 N→K 打包转换（或直接支持 `[K, N//2]` 输入），并修正 `N = W_packed.shape[1]` 取值为 2×N。

### 1.2 性能（GEMM-only vs 全算子）

| M,K,N | v15 GEMM-only TFLOPS | v13 基线 GEMM-only | v15 全算子 | 提升 |
|---|---|---|---|---|
| 256,4096,4096 | 43.2 | 29.6 | 9.6 | +1.46× |
| 512,4096,4096 | **73.5** | 38.8 | 7.2 | +1.89× |
| 1024,4096,4096 | **81.4** | 32.7 | 14.9 | +2.49× |
| 256,8192,8192 | 45.7 | 38.1 | 8.6 | +1.20× |
| 512,8192,8192 | **73.4** | 44.3 | 13.4 | +1.66× |
| 1024,8192,4096 | 54.5 | 46.9 | 12.2 | +1.16× |
| 256,4096,16384 | 26.7 | 37.6 | 8.5 | **0.71×（回退）** |

- GEMM-only 中/大 M 显著提升（最高 81.4，符合 round10 预期 60~100 的下沿偏上），N=16384 极端 shape 回退。
- **全算子仅 7.2~14.9**：每次调用重做 A 量化 kernel + **W→bf16 转换（[N,K] 全量，未缓存）** + A→bf16 转换，开销主导。
  **优化点：`_preprocess_weights` 应连 W_bf16 一起缓存**（v13 的 W 缓存只存了 fp32）。

### 1.3 SASS 复核：bf16 HMMA，非原生 FP4

```
HMMA.16816.F32.BF16: 64     TCGEN05: 0     e2m1: 0     FFMA: 0
```
与 round10 报告设计意图一致（bf16 MMA 是 Triton 3.6 下 dot_scaled 降级路径的可达最优）。
**400 TFLOPS 目标仍然必须走 CUTLASS 原生 FP4**（见下）。

---

## 二、kernel① CUTLASS mmaf_sm120（原生 FP4 路径）：构建阻塞实证

按 README 执行 `build_mmaf_sm120.sh` 前置要求逐一落实后的实测结论：

| 项 | 状态 | 证据 |
|---|---|---|
| nvcc | ✅ 13.0.88（V13.0.88） | 可编译 sm_121a |
| cmake | ✅ 4.4.2（pip 装） | 容器原本无 cmake |
| CUTLASS ≥3.9 源码 | ❌ **B2** | GitHub 直连限速 ~1MB/10s（容器与本地均如此），300MB tarball 不可得；容器内 vLLM 捆绑 4.3.4 为裁剪子集（缺 `cutlass/arch/mma_sm120.h`、无 CMake config）；PyPI `cutlass==0.9.0` 为 Python 绑定包，无 C++ 头 |
| torch C++ 开发头 | ❌ **B1** | `torch/extension.h` 依赖的 `torch/all.h` **不存在**（容器 torch 为运行时裁剪版）→ torch extension 绑定无法编译，与 CUTLASS 无关即可断 |
| .cu 自身 TODO | ❌ **B3** | 捆绑 4.3.4 头实证：blockscaled collective `Arguments` **必填 `layout_SFA`/`layout_SFB`** 成员（`sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp:356`），.cu 的 `args.mainloop={...}` 初始化列表缺这两个字段 → 编译不过；`args.epilogue.thread.alpha_ptr` 字段同样存疑 |

**结论：`.cu` 为骨架级代码（round10 报告自述"编译期核对点已用 TODO 标注"），当前环境 + 当前代码双重不可构建。**
**建议（round10 报告已自提）：改道 vLLM 生态现成 SM120 NVFP4 CUTLASS 内核 `nvfp4_scaled_mm_sm120_kernels.cu`（PR #42209 家族），在完整 vLLM 源码构建环境（含 torch 头 + CUTLASS）内集成，并保留 SASS 门禁 `mmaf|mma.sync.*e2m1|tcgen05` 作为验收标准。**

---

## 三、kernel② kv_linear v15（宽 tile）：正确但带宽回退

### 3.1 正确性：7/7 逐字节 PASS（T ∈ {1,4,16,64,256,1024,4096}）

### 3.2 带宽：全配置 ~10 GB/s，严重回退

T=65536，4680B/token（读 4KB+写 584B，round6 口径）：

| 版本 | 最佳 GB/s | 相对 |
|---|---|---|
| **v11**（round6 基线） | **53.4** | 1.00× |
| v12.1（round7） | 18.8 | 0.35× |
| **v15（本轮，全部 6 个 BLOCK_G/warps 配置）** | **10.6** | **0.20×** |

- v15 最佳配置 = BLOCK_G=32/warps=2（29.0ms），其余配置 30.8~62.1ms → **非 autotune 选型问题，是内核设计问题**。
- 根因：grid = (T, 64/BLOCK_G) = **131072 个微型 block**（T=65536），每 block 仅 2KB 读 + 288B 写，launch/占用开销主导；"宽 tile"（每 block 16 连续元素）反而放大 block 数。
- **round10 声称"46.30×"与"2~3 个数量级提升"未复现，实测为 5× 回退。**
- **建议：kernel② 不采用 v15，维持 v11（53.4 GB/s 仍是带宽冠军）**；若继续重写，方向是"每 block 处理整 token 或多 token"（减少 block 数、增大每 block 数据量），并做 128-bit 向量化负载实测。

### 3.3 v14：D2 缺陷，生产编译即挂

```
CompilationError: Cannot access global variable _GROUP from within @jit'ed function
```
模块级 `_GROUP` 未声明为 `tl.constexpr` → **v14 不可用**（shipped 即碎）。

### 3.4 paged v11：5/5 PASS

干净重跑 5/5（首跑曾 1 FAIL，系本机并发清理 Triton 缓存导致的文件竞争假阳性，非内核缺陷，单跑与重跑均通过）。

---

## 四、部署建议（回传交付方）

1. **kernel① v15 采纳前必须修 D1**（W 打包布局 repack + N 取值），并在 `_preprocess_weights` 中连 W_bf16 一并缓存；修复后 GEMM-only 可再提 ~1.7×（相对 v13），但 bf16 天花板 ~80 TFLOPS 不变。
2. **kernel① 400 目标**：不要继续在 Triton 3.6 上投入；改道 PR #42209 现成 CUTLASS 内核（构建环境需含 torch C++ 头）。SASS 门禁（`mmaf|e2m1|tcgen05`）继续作为每次改动的准入门禁。
3. **kernel② 建议维持 v11**，v15 带宽设计需推翻重来（减 block 增负载）；v14 直接弃用。
4. paged v11 无回归，可部署。

## 五、交付物

- 本报告 + `run_logs.md`（执行轨迹）
- 复测脚本：`verify_v15_k1.py` / `gemm_only_bench_v15.py` / `verify_v15_k2.py` / `probe_k2_bw.py` / `sass_v15.py`
- 证据：`evidence/`（v15 SASS 全量转储 370KB + 三个结论摘要）
- 打包：`nvfp4-v15-round-2026-08-19.zip`

生产 4 rank（rank0@01 / rank1@02 / rank2@04 / rank3@03）全程 healthy、GPU 0%、无残留进程，未做任何恢复操作。
