# NVFP4 生产集群落地 —— 测试脚本统一库

本目录汇总 DGX Spark 生产集群 NVFP4 算子落地历轮的**正确性 / 性能 / SASS 门禁**测试脚本。
统一三份主机备份：容器 `/vllm-workspace/nvfp4-landing/tests/`、本地 `deliverables/engineering-assurance/nvfp4-landing/tests/`。

> `_work/` 为 zip 解压的临时提取区，仅供核对，不要作为运行工作区（勿修改生产路径）。

---

## 目录结构

| 子目录 | 内容 |
|--------|------|
| `kernel1/` | kernel① prefill GEMM（NVFP4 4W4A）测试：正确性 + 性能（Triton vs torch ref）+ 依赖实现 |
| `kernel2/` | kernel② KV-Linear（DS-MLA-KV-Linear）v17 测试：正确性 + 安全性 + 带宽基准 + 依赖实现 |
| `sass/`    | SASS 门禁脚本（验证是否编译出原生 FP4 MMA 指令） |

---

## 运行环境

- DGX Spark：GB10 / SM121(SM121a) / CUDA / Triton 3.6+ / torch ≥ 2.5
- 所有 kernel 测试需真实 CUDA（非 CUDA 环境会自动跳过或报错）
- 运行前：`pip install pytest triton`（内核依赖已在 vLLM 容器内自带）

---

## kernel1/ —— prefill GEMM 测试

### `test_nvfp4_4w4a_prefill_gemm.py`（正确性）
- **用途**：Triton 实现 vs torch 参考实现的数值一致性（4 组 M×K×N × bias 开/关）。
- **运行**：`python -m pytest test_nvfp4_4w4a_prefill_gemm.py -v`
- **期望**：8 个用例全部通过（`assert_close` rtol=atol=5e-2，4-bit 量化参与计算故容差放宽）。
- **依赖**：`nvfp4_4w4a_prefill_gemm_torch.py`（参考）、`nvfp4_4w4a_prefill_gemm_triton.py`（被测）。

### `benchmark_nvfp4_4w4a_prefill_gemm.py`（性能）
- **用途**：Triton vs torch 参考 7 种 shape 的延时/TFLOPS/加速比，实测目标 GB10 FP4 dense。
- **运行**：`python benchmark_nvfp4_4w4a_prefill_gemm.py`
- **期望**：打印各 shape 的 `ref_ms / triton_ms / speedup`、平均/最佳/最差加速比；
  注释注明当前为 fp32-dot 4W4A 语义内核，逼近 400 TFLOPS 需切 `tl.dot_scaled(e2m1)` 原生 FP4 MMA。
- **注意**：v15/v16 Triton 版本走原生 `dot_scaled` 路径（见下述实现文件）。

### 实现文件（供测试 import）
- `nvfp4_4w4a_prefill_gemm_torch.py` —— torch 纯参考实现（金标准语义）
- `nvfp4_4w4a_prefill_gemm_triton.py` —— 被测 Triton 实现
- `nvfp4_4w4a_prefill_gemm_v15_triton.py` —— v15：SASS 实锤后的 Triton 3.6 最优路径（实测 8.96×）
- `nvfp4_4w4a_prefill_gemm_v16_torch.py` / `v16_triton.py` —— v16 版本（dot_scaled 原生 FP4 MMA 尝试）

> **TODO（routeA 并入）**：kernel① 基于 vLLM 内置 `cutlass_scaled_fp4_mm` 的适配层测试
> （由 routeA 交付后并入本目录，验证 scale 语义 vs torch ref）。

---

## kernel2/ —— DS-MLA-KV-Linear v17 测试

### `test_nvfp4_ds_mla_kv_linear_v17.py`（正确性）
- **用途**：v17 逐字节正确性 —— 7 组 T（1/4/16/64/256/1024/4096）× atol=0 与 torch 参考完全一致；
  另含 `kv_entry` 形状/类型断言。
- **运行**：`python -m pytest test_nvfp4_ds_mla_kv_linear_v17.py -v`
- **期望**：8 个用例全部通过（`torch.equal` 逐字节相等，mismatch=0）。
- **依赖**：`nvfp4_ds_mla_kv_linear_torch.py`（参考）、`nvfp4_ds_mla_kv_linear_v17_triton.py`（被测）。

### `test_nvfp4_ds_mla_kv_linear_v17_safety.py`（安全性 / 可靠性，P0）
- **用途**：补充生产可靠性 —— 极端值/饱和/符号零/边界 T/确定性/长期运行显存泄漏/NaN 不崩。
- **运行**：`python -m pytest test_nvfp4_ds_mla_kv_linear_v17_safety.py -v`
- **期望**：7 个用例通过；尤其 `test_long_run_memory` 断言 1000 次调用后显存增长 < 64MB（无泄漏）。
- **生产建议**：CI / 发版门禁必跑。

### `benchmark_nvfp4_ds_mla_kv_linear_v17.py`（性能，带宽为主）
- **用途**：v17 vs v11 的 GB/s 带宽对比（读 4KB + 写 584B per token），主指标 GB/s。
- **运行**：`python benchmark_nvfp4_ds_mla_kv_linear_v17.py`
- **期望**：perf_report 打印不同 T 下 v11/v17 两线 GB/s；v11 是 53.4 带宽冠军，v17 目标为多 token + 向量化。
- **依赖**：`nvfp4_ds_mla_kv_linear_triton.py`（v11）、`nvfp4_ds_mla_kv_linear_v17_triton.py`（v17）。

### 实现文件
- `nvfp4_ds_mla_kv_linear_torch.py` —— torch 参考实现（逐字节金标准）
- `nvfp4_ds_mla_kv_linear_triton.py` —— v11 Triton 实现（带宽冠军）
- `nvfp4_ds_mla_kv_linear_v17_triton.py` —— v17 Triton 实现（被测，逐字节一致）

---

## sass/ —— SASS 门禁脚本

### `sass_check_prefill_gemm.sh`（完整诊断）
- **用途**：编译触发 → 定位 Triton cubin → dump SASS → 检索 FP4/MMA 指令，判定 dot_scaled 是否落地原生 FP4 MMA。
- **运行**：`bash sass_check_prefill_gemm.sh`（DGX Spark 上）
- **期望**：
  - ✅ 出现 `mma.sync.aligned.m16n8k32.f32.e2m1` / `tcgen05.mma` → FP4 MMA 生效；
  - ❌ 只有 `FFMA/FADD/FMUL` → dot_scaled 降级 fp32，需 Triton 3.7+ 或 CUTLASS mmaf_scaled。

### `sass_gate.sh`（轻量门禁，CI 用）
- **用途**：对已 dump 的 SASS 文本执行硬门禁：必须命中原生 FP4/张量核心指令，否则 exit 非 0。
- **用法**：`cuobjdump --dump-sass <x.cubin> > sass.txt && bash sass_gate.sh sass.txt`
- **期望**：命中 `mma` 且 `e2m1`（或 `mmaf`）时 PASS（exit 0）；仅 FFMA/FADD 时 FAIL（exit 1）。

---

## 使用建议

1. **发版前必跑**：kernel2 的 3 个脚本（正确性 + 安全性 + 性能）+ kernel1 正确性 + sass 门禁。
2. **routeA 交付 kernel① vLLM 适配层后**：将适配层测试并入 `kernel1/`，补齐 `cutlass_scaled_fp4_mm` 的 scale/内存/健壮性测试。
3. **容器发布**：将本 tests/ 同步到容器 `/vllm-workspace/nvfp4-landing/tests/`。