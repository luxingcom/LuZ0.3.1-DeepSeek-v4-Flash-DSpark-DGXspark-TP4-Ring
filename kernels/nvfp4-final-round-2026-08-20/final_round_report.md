# NVFP4 最终交付包复测报告（kernel② v17 可替换确认 + kernel① 方案 B 实测）

> 日期：2026-08-20 ｜ 环境：DGX Spark 生产 TP4，`vllm-tp4-rank0`（torch 2.11.0+cu130 / triton 3.6.0 / flashinfer 0.6.15 / CC=(12,1)=sm_121a）
> 输入：`nvfp4-kernels-delivery-final.zip`（kernel② v17 安全可靠性文档+测试；kernel① 方案 B：PR #42209 / FlashInfer 集成）
> 用户指示：**按方案测试；kernel2 验证合格可替换**

---

## 〇、结论速览

| 项 | 结论 |
|---|---|
| **kernel② v17 替换** | ✅ **验证合格，同意替换**（8/8 逐字节 + 大 T 194~262 GB/s + 全边界与 torch 逐字节一致；安全套件 3 失败系测试脚本缺陷） |
| **kernel① 方案 B** | ⚠️ 本容器**不可直接跑通**：FlashInfer 0.6.15 wheel 缺 sm_121 原生 FP4 backend 编译扩展（b12x/cute-dsl 未编入）→ 需完整 vLLM 源码构建或重编 wheel；SASS 门禁已修正 |

---

## 一、kernel② kv_linear v17 —— 验证合格，可替换 ✅

### 1.1 证据链（完整）

| 维度 | 结果 |
|---|---|
| 正确性 | **8/8 逐字节 PASS**（shipped `test_v17.py`，7 档 T + kv 入口，atol=0） |
| 性能 | **大 T 194~262 GB/s**（4680B/token；T=1024 达 262.3=96% 理论；v11 的 3.5~4.6×；≥120 达标超额 1.6~2.2×） |
| 极端边界（本轮实测） | **zeros / +0 / -0 / 1e6 / 1e30 / 1e-30 / 6.0 与 torch ref 全部 byte_equal=True**（scale：zero=24、1e6=144、1e30=224、6.0=127） |
| 安全套件 | 确定性 / 内存无增长 / NaN 不崩溃 / 边界 T 等 **4 项通过** |
| 静态审计（交付方） | 数值 / 内存 / 并发 / 输入健壮性 / 长期稳定 全过 |

### 1.2 安全套件 3 失败的定性：测试脚本缺陷（非 v17 缺陷）

| 用例 | 脚本期望 | 实测正确值（v17=torch） | 定性 |
|---|---|---|---|
| test_saturation | scale=**255** | **144**（floor(log2(1e6/6))+127；255 需 max≈6×2^127） | 期望值写错 |
| test_sign_zero | scale=**1** | **24**（amax clamp 1e-30 → exp -103 → byte 24） | 期望值写错 |
| test_boundary_T | — | 比对无效 | **未 seed**：v17 与 ref 各吃一次不同 `randn` |

**修正建议**：saturation/sign_zero 期望值改为 144/24；boundary_T 共用同一输入。修正后全绿（本报告探针已证明）。

### 1.3 替换建议与注意事项

- **可替换 v11 部署**（linear 路径）；回退方案 = 保留 v11 文件即可。
- 部署落实加固：**R1** 首次调用可选 `isfinite` 断言（防 NaN/Inf 污染 KV）；**R2** CUDA Graph 捕获前 warmup 一次（触发 autotune）；**R3** paged v17 变体后续迭代（当前 paged 维持 v11 5/5）。
- 消费端兼容性（本轮新探针）：v17 信封（584B / E8M0 scale）面向 **vLLM PR#46329 linear reader**；FlashInfer `nvfp4_kv_quantize` 为独立语义（e4m3 位模式 scale，非 2 幂 floor-log2）→ 两者**不字节兼容属预期**，v17 服务 vLLM 路径，与 FlashInfer attention（如需）各自独立。

## 二、kernel① 方案 B（FlashInfer / vLLM CUTLASS FP4）—— 本容器实测

### 2.1 FlashInfer 0.6.15 API 面：完整

`mm_fp4`、`grouped_mm_fp4`、`mm_bf16_fp4`、`nvfp4_quantize`、`nvfp4_block_scale_interleave`、`nvfp4_attention_sm120*`、`nvfp4_kv_quantize` 全部存在——方案 B 的接口均已就位。

### 2.2 mm_fp4 四 backend 实测（4W4A GEMM）

| backend | 结果 |
|---|---|
| **b12x**（GB10 原生） | ❌ `RuntimeError: CuTe DSL is not available` —— **NGC vLLM wheel 未编入 CuTe DSL 扩展** |
| cute-dsl | ❌ 同上 + `capability 121 不支持` |
| trtllm | ❌ `BackendSupportedError: capability 121 不支持` |
| cudnn | ❌ `RuntimeError: aDesc 描述符错误`（nvfp4 matmul 描述符布局不符） |
| cutlass | ❌ `TypeError: TVM ffi 参数类型错误` |

**结论：该 wheel 无 sm_121 可用的原生 FP4 GEMM backend** —— 方案 B 首选（FlashInfer 0.6.8+ backend）在本容器内不可直接启用。

### 2.3 方案 B 落地路径（回传交付方）

按方案 §三/§六，三选一（均需**构建环境**，非运行时容器）：
1. **完整 vLLM 0.26 源码构建**：`git clone -b v0.26.x` + `setup.py build_ext --inplace`（FP4_ARCHS 含 12.1a）→ 用 `cutlass_scaled_fp4_mm_sm120a`；**注意 PR #26793 dispatch bug**（双架构编译可能错选 SM100 路径，只编 SM12x）
2. **重编 FlashInfer wheel**（含 cute-dsl/b12x 扩展）或升级 FlashInfer TOT（SM12x NVFP4 优化内核）
3. 采纳 **SASS 门禁修正**：SM12x 用 `grep -iE "mma.*e2m1|mmaf"`（NVIDIA 官方确认非 tcgen05）

验收标准（方案 §五）不变：正确性 ≤5e-2、SASS ≥1 条 `mma.*e2m1|mmaf`、**≥200 TFLOPS**、对照 v15 ≥1.5× 才切换。

## 三、交付物

- 本报告 + `run_logs.md` + 脚本：`probe_v17_edge.py` / `probe_kv_layout.py` / `test_planB_mm_fp4.py` / `probe_backends.py` / `probe_cudnn.py` / `probe_flashinfer.py` / `probe_fp4_api*.py`
- 证据：`evidence/`（k2_v17_final.txt + k1_planB_result.txt）
- 容器：`/vllm-workspace/nvfp4-delivery-final/`

生产 4 rank 全程 healthy、GPU 0%、无残留进程，未做恢复操作。
