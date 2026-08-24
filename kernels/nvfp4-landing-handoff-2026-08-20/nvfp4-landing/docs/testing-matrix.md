# NVFP4 算子测试对照矩阵

> 版本 v2026-08-20 | DGX Spark GB10 / sm_121a / vLLM 0.26

本表统一两个算子的正确性标准、性能指标、SASS 门禁判据与对比基线。脚本均在 `tests/`。

## kernel① prefill GEMM（4W4A）

| 实现 | 正确性基准 | 正确性结果 | 性能指标 | 性能实测 | SASS 门禁 | 判定 |
|---|---|---|---|---|---|---|
| **路线A**（cutlass_scaled_fp4_mm 适配层） | vs 官方 `dequantize_to_dtype` rel<0.02 | **8/8, rel=0.00141** | TFLOPS | 60~187 | sm_120 + FP4 符号 ✅ | ✅ **部署** |
| 路线B（FlashInfer mm_fp4） | — | 需升级 wheel | TFLOPS | 阻塞 | — | 🔶 备用 |
| v15（bf16 MMA） | vs torch ref ≤5e-2 | 8/8 | TFLOPS GEMM-only | 26.7~81.4 | bf16 HMMA（非FP4） | 回退 |
| v16（fp8 e4m3） | — | 8/8 | TFLOPS | 0.1~0.2 ❌ | bf16 模拟 | 弃用 |

**正确性脚本**：`bench_mmaf_final.py`（8 shape, rel vs 官方 dequant）
**性能脚本**：`bench_big.py`
**SASS 脚本**：`sass_fp4_check.py`
**对齐基线**：v15 = 26.7~81.4 TFLOPS（bf16）；torch fp32 = 17.1~18.4

## kernel② KV-Linear

| 实现 | 正确性基准 | 正确性 | 性能指标 | 实测 | 判定 |
|---|---|---|---|---|---|
| **v17** | 逐字节（shipped test） | **8/8** | GB/s(4680B/token) | 大T 194~262 | ✅ **替换 v11** |
| v11 | 逐字节 | 8/8 | GB/s | 53~61 | 回退/基线 |
| v12.1 | 逐字节 | 7/7 | GB/s | 17~19 | 弃用 |
| v15 | 逐字节 | 7/7 | GB/s | 9.3~10.5 | 弃用 |
| paged（v11） | 逐字节 | 5/5 | — | — | 维持 |

**正确性脚本**：`tests/kernel2/test_nvfp4_ds_mla_kv_linear_v17.py`（8/8）
**安全脚本**：`test_nvfp4_ds_mla_kv_linear_v17_safety.py`（注意 3 个测试脚本缺陷：期望值/seed）
**带宽脚本**：`benchmark_nvfp4_ds_mla_kv_linear_v17.py`
**理论带宽**：273 GB/s（DGX Spark HBM）

## SASS 门禁判据

| 架构 | 指令 | 判据 |
|---|---|---|
| SM12x（GB10 sm_121a） | `mma.*e2m1` / `mmaf` | 出现即原生 FP4 |
| ⚠️ 勿用 | `tcgen05` | 仅 SM10x |

## 版本矩阵（当前推荐）

| 算子 | 生产 | 备选 | 回退 |
|---|---|---|---|
| kernel① | 路线A（cutlass FP4） | 路线B（FlashInfer，待升级） | v15 |
| kernel② | **v17** | — | v11 |
| paged | v11 | — | — |