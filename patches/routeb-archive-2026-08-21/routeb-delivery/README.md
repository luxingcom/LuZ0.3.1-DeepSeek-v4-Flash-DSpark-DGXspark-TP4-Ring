# routeB 落地交付包（CUTLASS 4.4.0 Python DSL — 唯一路径）

> 环境：DGX Spark 4 节点 TP4 / CUDA 13.x（生产 13.2）/ CUTLASS 4.4.0 / torch 2.11 / vLLM 0.26 / sm_121a
> 目标：**routeB ≥350 TFLOPS（SM121 dense NVFP4）**，替代 routeA（vLLM 原生 cutlass，80~180 TFLOPS）
> 交付日期：2026-08-20 ｜ 社区实证基线：**356 TFLOPS**（baristankut，CUTLASS 4.4.0 + CUDA 13.1）

---

## 一、包结构

```
routeb-delivery/
├── README.md                                ← 本文件（执行顺序）
├── routeB_execution_plan.md                 ← 主执行计划（P0~P4 全流程 + 风险矩阵 + 参考资源）
├── kernel1_routeB_improvement.md            ← routeB 方案定案 v3（技术规格 + 路线收敛依据）
├── setup_routeb_env.sh                      ← P0 环境准备（cutlass-dsl 安装 + import 验证 + 备份）
├── patch_cutlass_dsl_sm121a.py              ← P1 sm_121a patch（mma.py 两处，自动 + 可 --revert）
├── routeb_bench_blockscaled.py              ← P2 复现 356 + tile sweep（完整实现：MmaMXF4Op kernel 构造 + SMEM 预算筛选 + 生产权重打包 + torch 正确性对照；host launch 段按官方 persistent_pingpong 示例补齐）
└── nvfp4_4w4a_prefill_gemm_v17_triton.py    ← MCP 语义对标内核（P3 A 量化 kernel 复用，已验 8 轮）
```

## 二、执行顺序（3~4 天）

```bash
# ── P0 环境（30 min）
bash setup_routeb_env.sh
#   输出：import cutlass 4.4.x 成功 + mma.py 已备份

# ── P1 patch（10 min）
python patch_cutlass_dsl_sm121a.py
#   验证：python -c "from cutlass.cute.nvgpu.warp.mma import BlockScaledMmaOp; print(BlockScaledMmaOp.admissible_archs)"  → 含 'sm_121a'
#   回滚：python patch_cutlass_dsl_sm121a.py --revert

# ── P2 复现 356 TFLOPS（0.5~1 天）
#   python routeb_bench_blockscaled.py --check            # 先验证正确性（torch 参考，无需 DSL）
#   python routeb_bench_blockscaled.py --shape 4096,14336,4096   # 复现 356 基线
#   python routeb_bench_blockscaled.py                     # 默认 shape 集 + tile sweep
#   说明：脚本已含 MmaMXF4Op kernel 构造、SMEM 预算筛选、生产权重打包、
#         torch 正确性对照；仅 host launch 段需按官方示例
#         （examples/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py）
#         补齐 TMA descriptor + grid launch（脚本内"生产替换点"标注处）。
#   验收：4096×14336×4096 ≥350 TFLOPS；SASS 门禁 nvdisasm | grep mma.*e2m1

# ── P3 语义对接（1 天）
#   MXF4 变体（E2M1 + UE8M0 + sf_vec_size=32）与生产 W_packed [K,N//2] / W_scale [K//32,N//128] 直配
#   A 量化复用 nvfp4_4w4a_prefill_gemm_v17_triton.py 的 _a_quant_kernel（已验证）

# ── P4 集成 A/B（1~2 天）
#   首选：插件独立 kernel（quant_config._nvfp4_prefill → routeB，M 阈值分派）
#   备选：flashinfer-b12x（CUTE_DSL_ARCH=sm_121a + VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x）
#   判据：≥1.5× routeA 且 ≥350 + pytest 8/8 + needle 128K → 灰度 K1
```

## 三、验收门槛（任一未达标 → 维持 routeA 现役，零风险）

| 阶段 | 验收 |
|---|---|
| P0 | import cutlass 4.4.x 成功 |
| P1 | 官方示例不再报 sm_121 not supported |
| P2 | **dense ≥350 TFLOPS**（基线 356）|
| P3 | pytest 8/8 + 生产权重直喂正确 |
| P4 | ≥1.5× routeA + 端到端全绿 |

## 四、关键约束提醒

1. **SMEM 99KB 硬约束**：tile 必须 ≤101,376 B；256×128×128（prefill）/ 128×128×128（decode）为已验证配置
2. **sf_vec_size=32**（MXF4 UE8M0）：与生产 32 分组直配；若示例默认 16 分组（NVF4 UE4M3），需改 sf_vec_size 或高精度转换器 `--block-k 16`
3. **锁版本**：`nvidia-cutlass-dsl-libs-cu13==4.4.2`（cu13 配套），勿追新（4.6 Operator API 与 4.4 有差异）
4. **driver ≥580.142**：sm121 ISA fallback 修复 + UMA 内存报告修复

## 五、参考资源

- 论坛：#359960（356 TFLOPS 实证）/ #364607 / #360142（SMEM 分析）
- CUTLASS issue #2800（sm_121a admissible_archs）
- BTankut/dgx-spark-sglang-moe-configs（MIT；Docker: ghcr.io/btankut/sglang-spark-glm47:latest）
- Jerry2423/cute_dsl_tutorials（04_gemm_blockscaled 整理版示例）
- CUTLASS 官方 examples/cute/blackwell_geforce/kernel/blockscaled_gemm/
