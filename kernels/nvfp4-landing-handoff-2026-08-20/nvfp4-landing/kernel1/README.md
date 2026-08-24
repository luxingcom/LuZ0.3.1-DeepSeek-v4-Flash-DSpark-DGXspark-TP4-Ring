# kernel① NVFP4 4W4A prefill GEMM —— 路线 A（原生 FP4 MMA）落地成果

> 版本 v2026-08-20 | 生产实测 | DGX Spark GB10 / sm_121a / vLLM 0.26

## 核心结论（生产实测，2026-08-20）

**路线 A 已打通并达成生产可用**：**原生 FP4 tensor-core MMA**（非 bf16 降级）。

| 项 | 结果 | 证据 |
|---|---|---|
| **正确性** | **8/8 PASS，rel=0.00141**（vs vLLM 官方 `dequantize_to_dtype`） | bench_mmaf_final.py |
| **性能** | **60~187 TFLOPS** 全 shape（大 shape 120~187） | bench_big.py |
| **SASS 门禁** | sm_120 cubin + 1349 个 FP4 符号（cutlass_scaled_fp4_mm/scaled_fp4_quant/e2m1） | sass_fp4_check.py |
| **对比 v15(bf16)** | 原生 FP4，数值精确（rel 0.00141 vs bf16 的近似） | — |

## 方案 B 落地路径（为何能零构建）

- **内核**：vLLM 0.26 内置 `vllm._custom_ops.cutlass_scaled_fp4_mm`（SM120a 原生 FP4 CUTLASS 内核，**已预编译进 vLLM**，无需源码构建）
- **量化**：`vllm._custom_ops.scaled_fp4_quant`（16-group e4m3，硬件量化）
- **关键洞察（本落地最大突破）**：`cutlass_scaled_fp4_mm` 与 vLLM 官方 NVFP4 dequant 数学**完全一致**（rel=0.00141），A/W 用 scaled_fp4_quant 量化即官方语义 → 无需自研 scale/alpha 语义
- 绕开了 Triton 3.6 无原生 FP4 MMA codegen 的限制（此前 v16 fp8 仅 0.1~0.2、v15 bf16 仅 26~81）

## 适配层

文件：`kernel1/nvfp4_4w4a_mmaf.py`

```python
from nvfp4_4w4a_mmaf import RouteA, nvfp4_4w4a_prefill_gemm

# 便捷入口（每次重做 W 预处理，适合验证/低频）
out = nvfp4_4w4a_prefill_gemm(A, W_packed, W_scale, bias=None)   # [M,N] fp32

# 生产高频入口（每层一次 preprocess 缓存）
impl = RouteA()
impl.preprocess_weights(W_packed, W_scale)        # 每层一次, 21ms
out = impl(A, use_cached_w=True)                  # 推理 [M,N] fp32
```

输入契约（既有格式）：
- `A` [M, K] fp32
- `W_packed` [K, N//2] uint8（NVFP4，N 打包，低半字节=偶 N 列）
- `W_scale` [K//32, N//128] uint8 E8M0
- 返回 [M, N] fp32

## 正确性验证（bench_mmaf_final.py）

8 组 shape 全部 rel=0.00141 vs 官方 dequantize_to_dtype：
```
M 256 K4096 N 4096:   rel=0.00141  82.7 TFLOPS
M 512 K2048 N 4096:   rel=0.00141  60.2 TFLOPS
M1024 K4096 N 2048:   rel=0.00141  63.8 TFLOPS
M 128 K4096 N 4096:   rel=0.00141  65.6 TFLOPS
M 256 K8192 N 8192:   rel=0.00141  95.6 TFLOPS
M 512 K8192 N 8192:   rel=0.00141 126.1 TFLOPS
M1024 K8192 N 4096:   rel=0.00141 113.2 TFLOPS
M 256 K4096 N16384:   rel=0.00141  85.7 TFLOPS
```

## 性能（bench_big.py，含大 shape）

```
M 2048 K 4096 N 4096:    90.2 TFLOPS
M 4096 K 4096 N 4096:    86.0
M 8192 K 4096 N 4096:    88.6
M 2048 K 8192 N 8192:   144.1
M 4096 K 8192 N 8192:   126.1
M 1024 K12288 N12288:   187.4   (最高)
M 2048 K 4096 N12288:   121.3
```

> 注：性能含 A 量化（scaled_fp4_quant 软件路径后端）+ workspace 分配。M 小时量化开销占比高；大 M/K 达 120~187 TFLOPS。若要冲 300+，需将 A 量化与 GEMM 融合进 CUDA Graph 或改用 cutlass backend。

## SASS 门禁

固化为 `tests/sass_fp4_check.py`。证据：
- `_C_stable_libtorch.abi3.so` 内嵌 **全部 sm_120 cubin**（`_C*.sm_120.cubin`）
- 1349 个 FP4 相关符号：`cutlass_scaled_fp4_mm`、`scaled_fp4_quant`、`torch_dtype_float4_e2m1fn_x2`、`cutlass::gemm` 模板
- 性能 60~187 TFLOPS（远超 bf16 66 TFLOPS 上限）佐证真原生 FP4 路径

## 部署建议

- **可直接部署**：正确性 8/8（rel 0.00141）、原生 FP4、性能优于 v15（26.7~81.4 且含 bf16 精度损失）
- **生产路径**：用 `RouteA` 类，每层 `preprocess_weights` 缓存（21ms/层一次），推理走 CUTLASS FP4 GEMM
- **回退**：保留 `nvfp4_4w4a_mmaf.py` 不动 vLLM 本体，删除即回退
- **进一步优化**（可选）：A 量化融合、CUDA Graph 捕获、多 shape 专属 autotune

## 参考

- vLLM 官方实现：`vllm/model_executor/kernels/linear/nvfp4/cutlass.py`
- 官方 dequant：`vllm/.../nvfp4_emulation_utils.py`（dequantize_to_dtype）
- 方案文档：交付包 `kernel1_planB_pr42209_integration.md`