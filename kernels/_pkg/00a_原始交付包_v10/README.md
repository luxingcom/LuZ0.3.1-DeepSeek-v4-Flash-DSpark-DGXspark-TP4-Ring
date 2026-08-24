# NVFP4 双算子交付包（kernel① prefill_gemm + kernel② KV linear）

> 环境：torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a（DGX Spark 生产 TP4，vLLM 0.26）
> 交付日期：2026-08-19

## 包结构

```
nvfp4-kernels-delivery/
├── README.md                          ← 本文件（交付清单 + 验证指令）
├── convert_mxfp4_to_nvfp4.py          ← 权重转换器（NV_K_BLOCK=32，已对齐 v10）
├── kernel1-nvfp4_4w4a_prefill_gemm/
│   ├── nvfp4_4w4a_prefill_gemm_triton.py    ← v10：原生 FP4 MMA（Triton 3.6 dot_scaled 实测签名）
│   ├── nvfp4_4w4a_prefill_gemm_torch.py     ← 参考实现（32 分组）
│   ├── test_nvfp4_4w4a_prefill_gemm.py      ← pytest（4 shapes × bias = 8 用例）
│   ├── benchmark_nvfp4_4w4a_prefill_gemm.py ← 吞吐基准
│   ├── report_nvfp4_4w4a_prefill_gemm_v10.md ← v10 交付报告
│   └── production_diag_v9_audit.md          ← 生产诊断 × v9 审查（缺陷 #7 根因与正解）
└── kernel2-nvfp4_ds_mla_kv_linear/
    ├── nvfp4_ds_mla_kv_linear_triton.py      ← v4 性能版（42.17×，scale 已修正 /6）
    ├── nvfp4_ds_mla_kv_linear_torch.py       ← 参考实现（已修正 /6）
    ├── test_nvfp4_ds_mla_kv_linear.py        ← 7 组 T 逐字节 (atol=0)
    ├── benchmark_nvfp4_ds_mla_kv_linear.py
    ├── nvfp4_ds_mla_kv_linear_paged_triton.py ← vLLM 分页版（scale 已修正 floor，无 libdevice 依赖）
    ├── nvfp4_ds_mla_kv_linear_paged_torch.py
    ├── test_nvfp4_ds_mla_kv_linear_paged.py
    ├── benchmark_nvfp4_ds_mla_kv_linear_paged.py
    ├── report_nvfp4_ds_mla_kv_linear.md       ← kernel② 报告（含 2.86×→42.17× 优化史）
    └── kv_linear_audit.md                     ← kernel② 兼容性核查（无 dot_scaled 缺陷；scale 统一）
```

## 两个算子的关键状态

| 算子 | 状态 | 说明 |
|---|---|---|
| ① prefill_gemm v10 | ✅ 缺陷 #7 已修复 | 原生 FP4 MMA（位置参数 + 32 分组 + rhs_k_pack）；**待生产 pytest 终审** |
| ② kv_linear v4 修正 | ✅ 核查通过 | 无 dot_scaled 缺陷；scale 语义已与生产统一；MCP 验证过 42.17× |
| ② kv_linear paged 修正 | ✅ 核查通过 | vLLM 分页语义；`ceil`→`floor` + 移除 libdevice；**待生产 pytest 终审** |

## 生产验证指令（DGX Spark）

```bash
# ── kernel① prefill_gemm（重点：dot_scaled 编译 + 32 分组数值）
cd kernel1-nvfp4_4w4a_prefill_gemm
python -m pytest test_nvfp4_4w4a_prefill_gemm.py -v     # 8 用例，rtol/atol=5e-2
python benchmark_nvfp4_4w4a_prefill_gemm.py             # 验收 ≥400 TFLOPS（最低 250 证明 FP4 MMA 生效）

# ── kernel② kv_linear（逐字节一致）
cd ../kernel2-nvfp4_ds_mla_kv_linear
python -m pytest test_nvfp4_ds_mla_kv_linear.py -v       # 7 组 T，atol=0
python -m pytest test_nvfp4_ds_mla_kv_linear_paged.py -v
python benchmark_nvfp4_ds_mla_kv_linear.py               # 参考 MCP 侧 42.17×

# ── 权重转换（T1.2 全量，顺延至内核验收后）
cd ..
python convert_mxfp4_to_nvfp4.py --input-dir <INSTALL_DIR>/models/deepseek-v4-flash-0731 \
                                 --output-dir <INSTALL_DIR>/models/dsv4f-0731-nvfp4 --with-mtp
```

## 回传给我

- kernel①：pytest 输出 + benchmark 各 shape TFLOPS 表
- kernel②：两个 pytest 输出（逐字节确认）+ benchmark 数字
- 若 kernel① 编译报错：贴完整 traceback，我带 v10 为基继续迭代
