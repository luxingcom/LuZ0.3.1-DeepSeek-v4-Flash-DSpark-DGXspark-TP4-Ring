# cuBLASLt Grouped GEMM NVFP4 × SM120 支持调查

**日期**：2026-08-13 ｜ **状态**：社区调查完成 ｜ **结论等级**：官方文档 + 多源社区实证交叉验证

---

## TL;DR

| 问题 | 结论 |
|---|---|
| cuBLASLt **Grouped GEMM** + NVFP4 能否用于 SM120（GB10 / DGX Spark / RTX 50 系）？ | **不能**。官方支持矩阵截至 cuBLAS 13.4.0（CUDA 13.2 Update 1）仅覆盖 **Compute Capability 10.x（SM100/SM103 数据中心 Blackwell）与 11.0（SM110）**，**明确不含 12.x（SM120/SM121）**，且当前实现仅支持 K=64 MMA tile |
| 非 grouped 的 `cublasLtMatmul` + NVFP4 在 SM120？ | **支持**。CUDA 12.8 起启用（12.8 Update 1 明确加"Blackwell GeForce-class"），CUDA 13.2 对 DGX Spark 大 M×N 的 MXFP8/NVFP4 有最高 3× 性能提升 |
| SM120 上 NVFP4 MoE/grouped GEMM 的正路？ | **FlashInfer CUTLASS（compute_120f / CUDA 13.0+）或 FlashInfer b12x（CuteDSL SM12x fused MoE，本集群 vLLM 0.26.1.dev0 已集成）**；Marlin W4A16 为最稳兜底 |

---

## 1. 官方支持矩阵（权威证据）

**cuBLAS 13.4.0 release notes（CUDA 13.2 Update 1）原文**：

> Extended the experimental Grouped GEMM API in cuBLASLt to support NVFP4 inputs and bias epilogues **on Blackwell GPUs with Compute Capability 10.x and 11.0**. Current Grouped GEMM NVFP4 support uses only K equal to 64 MMA tile size.

翻译：cuBLASLt 的 **experimental Grouped GEMM API** 在 cuBLAS 13.4.0 才加入 NVFP4 输入与 bias epilogue 支持，但限定 **CC 10.x 与 11.0**；当前 NVFP4 Grouped GEMM 仅支持 K=64 的 MMA tile。

**同一份 notes 中关于 SM120 的正面内容**：

> Improved GEMM performance on **DGX Spark** systems for **MXFP8 and NVFP4** data types in large M and N problem sizes, with up to **3× performance improvement** for selected matrix shapes.

→ 这是**非 grouped** 的 `cublasLtMatmul` 在 DGX Spark（SM120/SM121）上的优化，不改变 Grouped GEMM 的支持边界。

**关键对照（CUDA 12.8 / 12.9）**：

- cuBLAS 12.8（CUDA 12.8）：首次为 **CC 10.0 及以上** 引入 micro-scaled 4-bit/8-bit 混合精度 GEMM（CUDA_R_4F_E2M1 + UE4M3 16 元素块缩放）。
- cuBLAS 12.8 Update 1：**Added support for block-scaled FP8 and FP4 datatypes on Blackwell GeForce-class GPUs** —— 这是 SM120 上普通 cublasLtMatmul FP4 可用的官方依据。
- cuBLAS 12.9：引入指针数组 batch（独立 batch pointers），但仅限低精度；无 Grouped GEMM NVFP4。

**结论**：官方 API 层面，"Grouped GEMM + NVFP4" 与 "SM120" 两个集合**不相交**——截止 cuBLAS 13.4.0。NVIDIA 论坛 NVIDIA 员工也在 CUDA 13.2 帖中复述同一支持范围（见 §3）。

---

## 2. SM120 ≠ SM100：为什么不能"顺手支持"

SM120（桌面/工作站 Blackwell：GB10 的 sm_121、GB202 的 sm_120）与 SM100（数据中心 Blackwell：B200/GB200 的 sm_100）是**两个 compute capability family**，不是同一芯片的降频版：

| 维度 | SM100（数据中心） | SM120/SM121（桌面/工作站/Spark） |
|---|---|---|
| CC family | 10.x（10.0/10.3） | 12.x（12.0/12.1） |
| SMEM/SM | 228 KB | **99 KB**（CUTLASS 按 228KB 设计的 tile 会溢出 → autotuner 全跳过） |
| TMEM / tcgen05 | 有 | GB10 **无 tcgen05**（NVIDIA 官方论坛确认，die space 让给 RT Core） |
| 特性后缀 | a | a / f（compute_120f = "full" 特性集，CUDA 13.0+ 才可用） |
| 预编译 cubin | — | SM100 cubin 在 SM120 上直接崩溃，互不兼容 |

FlashInfer issue #2723 给出了三种编译目标的实测：

| 编译目标 | CUDA | MMA 指令 | CUTLASS Grouped GEMM 结果 |
|---|---|---|---|
| `compute_120` | 12.8+ | 未启用 | "Arch conditional MMA" 错误 → 失败 |
| `compute_120a` | 12.8+ | 启用 | TMA WS tactics 失败，慢 fallback：14.6 tok/s |
| `compute_120f` | 13.0+ | 完整特性集 | 快速 TMA WS 可用：35–39 tok/s（接近 Marlin 46–49） |

---

## 3. 社区实证链

### 3.1 FlashInfer #2723「SM120 (RTX Blackwell) NVFP4 MoE: CUTLASS Grouped Block-Scaled GEMM Produces Invalid Output」
- **vLLM 原生 CUTLASS grouped GEMM 在 SM120 上产出垃圾结果**（silent garbage，正确性 bug），升级 CUTLASS 4.4.1 依旧（0 处 SM120 改动）。
- FlashInfer 路径在 SM120 不在 capability 检查内 → 需 patch 10+ 文件；patch 后正确但慢（autotuner fallback）。
- TRT-LLM 路径：C++ 硬编码 `SM == 10` + SM100-only cubin → SM120 直接崩溃。
- **突破**：CUDA 13.0 + `compute_120f`，DGX Spark（SM121）用户实测 35 tok/s（12.1f），单用户 14.6→39 tok/s（4 用户 6.9→18.2 tok/s/user）。
- 与 CUTLASS #3096（SM120 grouped GEMM 垃圾输出）为同一问题族；gunbark.dev SM120 feasibility ledger 亦引证。

### 3.2 NVIDIA 开发者论坛「CUDA 13.2 DGX Spark impact」
- 用户确认 cuBLASLt experimental Grouped GEMM 的 NVFP4/MXFP8 支持范围为 **CC 10.x 与 11.0**，并指出 SM121 不在其中。
- DGX Spark 上的收益是**非 grouped** 的 NVFP4/MXFP8 大 M×N 3× 优化。
- 附带修复：cublasLtMatmul 与 Tensor Memory 并发时结果错误（CUB，影响 CC 10.x/11.x since cuBLAS 12.8）——社区推测与用户观察到的质量退化相关。

### 3.3 vLLM 侧进展（SM120 NVFP4 生态）
| 项目 | 状态 |
|---|---|
| `nvfp4_scaled_mm_sm120_kernels.cu`（非 TMA dense NVFP4） | **confirmed**，prefill/TTFT 收益，Marlin W4A16 兜底 |
| `cutlass_scaled_fp4_mm_sm120a` device guard（vLLM **PR #29711**，hholtmann） | 已合入趋势，SM120 runtime dispatch |
| SM120 NVFP4 **MoE** 支持 | vLLM **Issue #31085**，进行中 |
| SM121 DGX Spark / Acer GN100 aarch64 | vLLM **Issue #36821**，进行中 |
| FP8 scaled_mm SM121 崩溃（guard 用错） | vLLM **PR #35568** 修复（enable_sm120_family） |
| vLLM `VLLM_FLASHINFER_MOE_BACKEND=latency` | SM121 上**必须**用 latency backend；throughput backend 报 "Failed to initialize cutlass TMA WS grouped gemm"（Avarok/eugr spark-vllm-docker 实证） |

### 3.4 本地既有线索（b12x）
本集群 vLLM 0.26.1.dev0 容器内已装 **FlashInfer CuteDSL b12x**（`/usr/local/lib/python3.12/dist-packages/b12x/`），其 SM120 支持显式声明：

- `_supports_current_device`：`p.is_cuda() and p.is_device_capability_family(120) and has_flashinfer_b12x_moe()`
- `_supports_quant_scheme`：`(kNvfp4Static, kNvfp4Dynamic)` 与 `(kNvfp4Static, None)`（W4A16；**in-kernel BF16→FP4 activation quant**，NVFP4 checkpoint 运行时兼容）
- 结构：`b12x/cute/fp4.py`（as_grouped_scale_view）、`b12x/distributed/pcie_oneshot.cu`、w4a16 MoE 绑定（modelopt_nvfp4 / fp4_e8m0_k32 两种 source format）

→ 这是**本集群唯一已就绪的 SM120 NVFP4 MoE 内核路径**，与 cuBLASLt Grouped GEMM 无关。

---

## 4. 可用性总表（2026-08-13 时点）

| 路径 | SM120/SM121 NVFP4 | 证据/备注 |
|---|---|---|
| `cublasLtMatmul`（dense，非 grouped） | ✅ 支持 | CUDA 12.8+（GeForce-class 12.8 U1）；13.2 大 M×N 3× |
| **cuBLASLt Grouped GEMM NVFP4** | ❌ **不支持** | 仅 CC 10.x / 11.0；K=64 限制；experimental |
| vLLM `cutlass_scaled_fp4_mm`（dense） | ✅ 支持 | sm120a kernel，PR #29711 起 runtime dispatch |
| vLLM 原生 CUTLASS **grouped** GEMM（MoE） | ❌ 垃圾输出 | CUTLASS #3096 / vLLM 论坛；勿用于生产 |
| FlashInfer CUTLASS MoE（compute_120f） | ✅ 可用 | CUDA 13.0+；35–39 tok/s 实测 |
| **FlashInfer b12x（CuteDSL SM12x fused MoE）** | ✅ **本集群已集成** | NVFP4 W4A16；is_device_capability_family(120) |
| Marlin W4A16 | ✅ 最稳兜底 | 46–49 tok/s 实测，高于 120a 的 14.6 |

---

## 5. 对本集群（DGX Spark TP4）的落地含义

1. **不要等 cuBLASLt Grouped GEMM 来加速 NVFP4 MoE**——它在 SM120 上不存在（且 experimental、K=64 限制）。官方路径在 SM120 只有 dense `cublasLtMatmul`。
2. **可用正路是 FlashInfer b12x / FlashInfer CUTLASS（compute_120f）**：本地 0.26.1.dev0 已含 b12x（W4A16，in-kernel 激活量化），理论上可直连 MJPansa/官方 NVFP4 checkpoint。
3. 与上一轮结论呼应：**即便切到 NVFP4 MoE 内核，本集群 latency-bound 的低并发单请求延迟不会变**；收益在带宽饱和档（大 batch / 长上下文 decode），测法仍按基准口径（per-request p50，禁用 agg_* 跨组对比）。
4. 若实验 b12x：先验证 `VLLM_FLASHINFER_MOE_BACKEND`（latency）与 CUDA 版本 ≥13.0（compute_120f），避免踩 SM120 grouped GEMM 垃圾输出坑。

---

## 6. 参考链接

- cuBLAS 13.4.0 release notes（CUDA 13.2 Update 1）：docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/
- cuBLAS 12.8 / 12.8 Update 1 / 12.9 release notes：docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/
- NVIDIA 官方博客：Boosting Matrix Multiplication Speed and Flexibility with NVIDIA cuBLAS 12.9
- NVIDIA 论坛：CUDA 13.2 DGX Spark impact（forums.developer.nvidia.com/t/cuda-13-2-dgx-spark-impact/363182）
- FlashInfer issue #2723：SM120 NVFP4 MoE grouped GEMM 无效输出
- vLLM PR #29711（cutlass_scaled_fp4_mm sm120 dispatch）、#35568（sm120 family guard）、Issue #31085/#36821
- Avarok/vllm-dgx-spark（HF）、eugr/spark-vllm-docker issue #143
- gunbark.dev SM120 feasibility ledger（个人工程账本，含 CUTLASS #3096 引证）
- 本地：`_b12x_supp.txt` / `_b12x_has.txt` / `_b12x_tp.txt`（b12x SM120 支持声明）

> 时效性说明：以上为 2026-08-13 时点结论。cuBLASLt Grouped GEMM NVFP4 为 NVIDIA 主动迭代方向，SM120 纳入支持矩阵取决于后续 release notes，建议 CUDA 13.3+ 发布后复查。
