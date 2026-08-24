# NVFP4 双算子交付包（kernel① prefill_gemm + kernel② KV linear）— 精简版

> 环境：torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a（DGX Spark 生产 TP4，vLLM 0.26）
> 交付日期：2026-08-20 ｜ 本包仅含**当前有效**资料，过时版本与历史报告已清理

---

## 一、最终结论（真机多轮验证）

| 算子 | 推荐 | 正确性 | 真实性能 | 状态 |
|---|---|---|---|---|
| ① prefill_gemm（生产现役） | **routeA = vLLM 原生 `cutlass_scaled_fp4_mm_sm120a`** | ✅ | **80~180 TFLOPS** | ✅ 生产已部署 |
| ① 350 目标（在研） | **routeB（全新 350T 方案）** | — | 目标 ≥350（≥1.5× routeA） | 🔬 方案已定（见 §三） |
| ① 对照参考 | v15（Triton bf16） | ✅ 8/8 | 26.7~81.4 | 📎 仅 Triton 3.6 上限参考，**未进生产** |
| ② kv_linear | **v17** | ✅ **8/8 逐字节** | **194~262 GB/s（3.5~4.6× v11）** | ✅ 达标部署 |
| ② kv_linear 回退 | v11 | ✅ 生产验证 | 53.4 GB/s | 🔄 回退 |
| ② paged | v11 | ✅ 5/5 | — | ✅ 维持 |

## 二、版本演进（一句话）

| 版本 | kernel① | kernel② |
|---|---|---|
| v9~v11 | dot_scaled 降级 fp32/bf16，20~46 TFLOPS | 42× 优化（含 /6 修复） |
| v12~v13 | 分离量化 + W 缓存，20~46 TFLOPS | grid 64 修复（v12.1），7/7 |
| v14~v15 | — | 宽 tile 尝试 10.6 GB/s 回退（微型 block），弃用 |
| **v15（①）** | Triton bf16 上限（26.7~81.4），**未进生产** | — |
| v16/v16.1（①） | fp8 e4m3 scaled 无原生 codegen → 弃用 | — |
| **routeA（① 生产）** | **vLLM 原生 cutlass sm120a：80~180 TFLOPS** | — |
| **v17（②）** | — | **多 token/program + 向量化 + pad 内联：194~262 GB/s** |

## 三、kernel① routeB（350T 性能方案，在研）

- **现状**：routeA（vLLM 原生 cutlass，per-tensor scale）80~180 TFLOPS = 峰值 500 的 16~36%
- **routeB 三路线**：R3 调度调优（W 缓存 + A 量化分离 + tile 预热，180→230，零风险）→ R1 FlashInfer 0.6.8+ 评估（最快到 350）→ R2 自研 CUTLASS 3.9 blockscaled（`mmaf_scaled` m16n8k32 + per-block E8M0，tile 128×128×64，350~475，兜底主力）
- 关键事实：**SM12.x 原生 FP4 走 `mma.*` 指令族（非 tcgen05）** → SASS 门禁 `grep mma.*e2m1|mmaf`
- 详细文档：`kernel1-nvfp4_4w4a_prefill_gemm/kernel1_routeB_improvement.md`（方案定案）+ `kernel1_planB_pr42209_integration.md`（R1/R2 集成细节）

## 四、包结构（36 文件，已清理过时）

```
nvfp4-kernels-delivery/
├── README.md                                        ← 本文件
├── convert_high_precision_nvfp4.py                  ← MXFP4→NVFP4 高精度转换器（生产权重非 NVFP4 时先跑，--validate 验证）
├── perf_diag_fp4_gemm.py                            ← FP4 GEMM 性能归因诊断（shape 扫描 + 量化占比拆测）
├── entrypoint_registration.md                       ← vLLM 插件注册机制（quant/method/entry point 三要素）
├── entrypoint_activation_plan.md                    ← 停机窗口执行清单（Step0~5）
├── inference_workflow_integration.md                ← 推理工作流集成方案 v2（A/B 先行 + 开关 + 回滚）
├── fp4_gemm_perf_attribution.md                     ← 60~187 TFLOPS 归因 + P1~P5 提升路线
├── kernel_improvement_summary.md                    ← 新旧算子提升对比
├── kernel1-nvfp4_4w4a_prefill_gemm/   （6 文件）
│   ├── nvfp4_4w4a_prefill_gemm_v15_triton.py        ← 对照参考（Triton bf16，未进生产）
│   ├── nvfp4_4w4a_prefill_gemm_torch.py             ← 参考实现
│   ├── test_nvfp4_4w4a_prefill_gemm.py              ← pytest 8 用例
│   ├── benchmark_nvfp4_4w4a_prefill_gemm.py
│   ├── kernel1_routeB_improvement.md                ← 🆕 routeB 350T 方案定案
│   └── kernel1_planB_pr42209_integration.md         ← R1/R2 集成细节（FlashInfer/PR #42209）
└── kernel2-nvfp4_ds_mla_kv_linear/    （14 文件）
    ├── nvfp4_ds_mla_kv_linear_v17_triton.py         ← ✅ 推荐（194~262 GB/s）
    ├── nvfp4_ds_mla_kv_linear_triton.py             ← v11 回退（53.4 GB/s）
    ├── nvfp4_ds_mla_kv_linear_torch.py              ← 参考实现
    ├── nvfp4_ds_mla_kv_linear_paged_triton.py       ← paged v11（5/5 维持）
    ├── nvfp4_ds_mla_kv_linear_paged_torch.py
    ├── test_nvfp4_ds_mla_kv_linear.py / _v17.py / _paged.py / _v17_safety.py
    ├── benchmark_nvfp4_ds_mla_kv_linear.py / _v17.py / _paged.py
    ├── kernel2_v17_safety_reliability.md            ← 安全/可靠性检验报告
    └── kv_linear_optimization_space.md              ← 优化空间分析
└── nvfp4_vllm_plugin/                    （8 文件）
    ├── setup.py + verify_plugin_registered.py + ab_routeA_vs_b12x.py + ab_v17_semantics.py
    └── nvfp4_vllm_plugin/（__init__ / quant_config / moe_method / kv_writer）
```

## 五、生产验证指令（DGX Spark）

```bash
# kernel① v15（对照参考，非生产）
cd kernel1-nvfp4_4w4a_prefill_gemm
python -m pytest test_nvfp4_4w4a_prefill_gemm.py -v    # 8/8
python benchmark_nvfp4_4w4a_prefill_gemm.py            # Triton bf16 上限参考

# kernel① routeA（生产现役，归因诊断，需在包根目录跑）
cd ../.. && python perf_diag_fp4_gemm.py                   # 80~180 归因（shape/量化占比）

# kernel② v17（达标版本）
cd ../kernel2-nvfp4_ds_mla_kv_linear
python -m pytest test_nvfp4_ds_mla_kv_linear_v17.py -v          # 逐字节
python -m pytest test_nvfp4_ds_mla_kv_linear_v17_safety.py -v   # 6 组安全/可靠
python benchmark_nvfp4_ds_mla_kv_linear_v17.py                  # 大 T 194~262 GB/s
python -m pytest test_nvfp4_ds_mla_kv_linear_paged.py -v        # paged v11 5/5
```

## 六、下一步执行清单

- **A. kernel① routeB（350T）**：S0 归因（`perf_diag_fp4_gemm.py` 对 routeA）→ S1 R3 调度调优（W 缓存 + A 量化分离 + tile 预热，180→230）→ S2 R1 FlashInfer 0.6.8+ 评估 → S3 R2 自研 CUTLASS 3.9 blockscaled（350~475）；验收 ≥350 且 ≥1.5× routeA
- **B. 插件启用**：entrypoint 三选一（pip 非 -e 首选）→ 停机清单 Step0~5 → 灰度 K2→K1（A/B ≥1.5× 才启 K1）
- **C. 权重**：生产权重为 MXFP4 时先跑 `convert_high_precision_nvfp4.py --max-layers 1 --validate`
- **D. 回滚**：`pip uninstall nvfp4-vllm-plugin` / 关 env / 还原备份（≤5 min）
