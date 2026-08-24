# NVFP4 双算子交付包（kernel① prefill_gemm + kernel② KV linear）— v12

> 环境：torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a（DGX Spark 生产 TP4，vLLM 0.26）
> 交付日期：2026-08-19
> **v12 = v11（生产修复）基础上，采纳 MCP 第三轮开发验证的分离量化架构 + scale 语义修正**

---

## 一、版本演进（一句话）

| 版本 | 里程碑 | 状态 |
|---|---|---|
| v9 | MCP 生成，退回 fp32 `tl.dot`（性能自杀） | ❌ 弃用 |
| v10 | 人工构造 dot_scaled（位置参数 + 32 分组） | ⚠️ 生产暴露 3 硬约束不符 |
| **v11** | 合并生产修复（rhs `[K,N]` + uint8 scale + 不 trans）+ 舍入统一 | ✅ 生产编译跑通，**20 TFLOPS 瓶颈** |
| **v12** | 采纳 MCP 分离量化架构（A 预量化 kernel + 主机侧重打包）+ scale 修正 | 🆕 **本包**，待生产复测 |

## 二、v12 相对 v11 的变更

1. **kernel① 架构升级（性能关键）**：GEMM kernel 内不再做 A 量化/W 重打包（这是生产 20 TFLOPS 的瓶颈），改为：
   - `_quantize_fp32_to_nvfp4_packed`：A 量化独立 kernel（A fp32 → `A_packed [M,K//2]` + `A_scale [M,K//32]`）
   - 主机侧 `_repack_w_for_rhs_k_pack`：W `[K,N//2]` → `[K//2,N]`（每层一次，可缓存）
   - 主机侧 `_expand_w_scale`：W_scale `[K//32,N//128]` → `[N,K//32]`（uint8 e8m0）
   - GEMM kernel：纯 `tl.dot_scaled` MMA（大 BLOCK 128/256 + GROUP_M swizzle）
2. **scale 语义修正**：kernel① 量化 kernel 补 `/6`（`floor(log2(max/6))+127`，全码本归一化）；kernel② 去掉 `scale_factor` 误乘的 `×6`
3. **kernel② linear 架构**：采纳 MCP v2（服务端实测 **59.67×**），2D grid + TOKENS_PER_PROG autotune + 掩码求和打包
4. **paged 维持 v11**（MCP harness 不兼容 + 生成器信封布局错误，均与内核无关；生产 5/5 全精确）

## 三、包结构

```
nvfp4-kernels-delivery/
├── README.md                                   ← 本文件（v12）
├── convert_mxfp4_to_nvfp4.py                   ← 权重转换器（NV_K_BLOCK=32）
├── kernel1-nvfp4_4w4a_prefill_gemm/
│   ├── nvfp4_4w4a_prefill_gemm_triton.py       ← v11 生产修复版（dot_scaled 硬约束，已实测编译）
│   ├── nvfp4_4w4a_prefill_gemm_v12_triton.py   ← 🆕 v12 分离量化架构（建议优先部署）
│   ├── nvfp4_4w4a_prefill_gemm_torch.py        ← 参考实现（32 分组）
│   ├── test_nvfp4_4w4a_prefill_gemm.py         ← pytest 8 用例
│   ├── benchmark_nvfp4_4w4a_prefill_gemm.py    ← 吞吐基准
│   ├── report_mcp_validation_round3.md         ← 🆕 第三轮 MCP 开发验证报告
│   ├── report_nvfp4_v11_round2_analysis.md     ← 第二轮生产分析（性能根因）
│   ├── round2_production_report.md             ← 第二轮生产实测报告（原稿）
│   ├── report_nvfp4_4w4a_prefill_gemm_v10.md
│   └── production_diag_v9_audit.md
└── kernel2-nvfp4_ds_mla_kv_linear/
    ├── nvfp4_ds_mla_kv_linear_triton.py        ← v11 生产修复版（舍入统一，已验证）
    ├── nvfp4_ds_mla_kv_linear_v12_triton.py    ← 🆕 v12（MCP v2 架构 + scale 修正）
    ├── nvfp4_ds_mla_kv_linear_torch.py
    ├── nvfp4_ds_mla_kv_linear_paged_triton.py  ← v11 分页版（floor + 舍入统一）
    ├── nvfp4_ds_mla_kv_linear_paged_torch.py
    ├── test_nvfp4_ds_mla_kv_linear.py          ← 7 组 T 逐字节
    ├── test_nvfp4_ds_mla_kv_linear_paged.py
    ├── benchmark_nvfp4_ds_mla_kv_linear.py
    ├── benchmark_nvfp4_ds_mla_kv_linear_paged.py
    ├── kv_linear_audit.md
    └── report_nvfp4_ds_mla_kv_linear.md
```

## 四、两个算子的关键状态（MCP 第三轮实测）

| 算子 | MCP 服务端验证 | 性能（服务端 GPU 实测） | 生产实测 |
|---|---|---|---|
| ① prefill_gemm | ✅ passed（v3，3 轮） | **2.79×**（v11 架构 → v12 分离架构待复测） | v11：编译跑通、6/8 精确、20 TFLOPS |
| ② kv_linear linear | ✅ passed（v2，2 轮） | **59.67×**（目标 4.0 超标 14 倍） | 6/7 精确、speedup 10~41× avg 22.26× |
| ② kv_linear paged | ❌ harness 不兼容（8 轮 `torch not defined`） | 无（非内核问题） | **5/5 全精确** |

## 五、生产验证指令（DGX Spark）

```bash
# ── kernel① prefill_gemm（重点：v12 分离架构的 TFLOPS）
cd kernel1-nvfp4_4w4a_prefill_gemm
# 推荐先跑 v12（分离量化，性能目标 100~400 TFLOPS）
python -c "import nvfp4_4w4a_prefill_gemm_v12_triton as m; print(m.nvfp4_4w4a_prefill_gemm)"  # 冒烟
python benchmark_nvfp4_4w4a_prefill_gemm.py    # 或临时改 import 到 v12 版
python -m pytest test_nvfp4_4w4a_prefill_gemm.py -v   # 8 用例，rtol/atol=5e-2

# ── kernel② kv_linear（逐字节一致）
cd ../kernel2-nvfp4_ds_mla_kv_linear
python -m pytest test_nvfp4_ds_mla_kv_linear.py -v       # 7 组 T，atol=0
python -m pytest test_nvfp4_ds_mla_kv_linear_paged.py -v
python benchmark_nvfp4_ds_mla_kv_linear.py               # 复测（服务端基线 59.67×）

# ── 权重转换（T1.2 全量，顺延至内核验收后）
cd ..
python convert_mxfp4_to_nvfp4.py --input-dir <INSTALL_DIR>/models/deepseek-v4-flash-0731 \
                                 --output-dir <INSTALL_DIR>/models/dsv4f-0731-nvfp4 --with-mtp
```

> 💡 **v12 部署提示**：kernel① v12 的主机侧 `_repack_w_for_rhs_k_pack` / `_expand_w_scale` 每层只算一次，建议在 vLLM `process_weights_after_loading()` 中缓存 `W_packed_rhs` 与 `W_scale_rhs`，避免每次 forward 重复计算。

## 六、回传给我

- kernel①：v12 benchmark 各 shape TFLOPS 表（对比 v11 的 16.6~20.8）+ pytest 结果
- kernel②：两个 pytest 输出（逐字节确认）+ benchmark 数字
- 若编译报错：贴完整 traceback，我带 v12 为基继续迭代
