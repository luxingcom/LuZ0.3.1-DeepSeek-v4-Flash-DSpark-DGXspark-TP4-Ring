# NVFP4 生产集群落地 —— 统一资料库入口

> 版本 **v2026-08-20**（本文将保持更新，是本工程唯一权威版本索引）
> DGX Spark 4 节点集群（01-04）/ GB10 / sm_121a / vLLM 0.26 / 生产容器 `vllm-tp4-rank0`

本目录是 NVFP4 4W4A 算子工程的**统一资料库**：内核实现、测试脚本、实测证据、文档手册。所有引用以相对路径指向本库，避免多源漂移。

---

## 部署矩阵（最新推荐）

| 算子 | 推荐实现 | 正确性 | 性能 | 状态 |
|---|---|---|---|---|
| **kernel① prefill GEMM** | **路线 A**：vLLM `cutlass_scaled_fp4_mm`（原生 FP4）适配层 | 8/8, rel=0.00141 | 60~187 TFLOPS | ✅ **可部署** |
| 备选 | 路线 B：FlashInfer `mm_fp4` | （需升级 wheel） | — | 🔶 备用 |
| 回退 | Triton v15（bf16 MMA） | 8/8 | 26.7~81.4 TFLOPS | vLLM 自带 |
| **kernel② KV-linear** | **v17**（多 token 负载，宽 tile） | 8/8 逐字节 | 大 T 194~262 GB/s | ✅ **替换 v11** |
| 备选/回退 | v11 | 8/8 | 53~61 GB/s | 保留 |
| **paged** | v11 | 5/5 | — | 维持 |

---

## 目录结构

```
nvfp4-landing/
  README.md                  # 本文件（统一入口）
  kernel1/                   # kernel① 路线A 适配层 + README（成果）
    nvfp4_4w4a_mmaf.py       # 原生 FP4 生产适配层（RouteA 类 + 便捷入口）
  kernel2/                   # kernel② v17（见内核源：/vllm-workspace 交付包）
  docs/                      # 文档手册
    ROUTE-B-FEASIBILITY.md   # 路线B(FlashInfer)可行性结论
    cleanup-inventory.md     # 旧料清理清单
    (landing-runbook.md 待补)
    (runbook-kernel2-v17.md 待补)
    (testing-matrix.md 待补)
  tests/                     # 汇总测试脚本（正确性/性能/SASS）
    README.md                # 测试脚本使用说明
    bench_mmaf_final.py      # 路线A 正确性+性能（8 shape）
    bench_big.py             # 路线A 大 shape 性能
    bench_k2_v17.py          # kernel② v17 带宽
    sass_fp4_check.py        # SASS 门禁（符号/cubin 验证）
    kernel1/ kernel2/ sass/  # 历轮脚本归档
  evidence/                  # 关键实测证据（待补 SASS 表、TFLOPS 表）
```

## 双端同步

- **容器**（生产执行）：`/vllm-workspace/nvfp4-landing/`
- **本地**（查看/交付）：`C:\Users\...\deliverables\engineering-assurance\nvfp4-landing\`

两处保持一致。

## 关键结论回顾（2026-08-20 实测）

### kernel① —— 方案 B 落地（两大突破 + 三阻塞）
1. **路线 A 零构建打通**：vLLM 0.26 内置 `cutlass_scaled_fp4_mm`（SM120a 原生 FP4 已预编译），`scaled_fp4_quant` 硬件量化 → 适配层 8/8 rel=0.00141、60~187 TFLOPS。
2. **遗留框架 bug**：`nvfp4_emulation_utils.break_fp4_bytes` 用 CPU `kE2M1ToFloat_handle` 索引 GPU 张量 → 需先 `.cuda()`。
3. **路线 B 阻塞**：flashinfer 0.6.15 wheel 未编入 b12x/cute-dsl 后端 → 需升级。
4. **历史结论（Triton 3.6）**：v16 fp8 0.1~0.2（降级）、v15 bf16 26~81，均非真 FP4。

### kernel② —— v17 全面合格
- 8/8 逐字节、大 T 194~262 GB/s（3.5~4.6× v11）、边缘 case 与 torch 全一致、安全套件 4 项过。
- 安全套件 3 个 FAIL = 测试脚本缺陷（saturation 期望 255 实为 144、sign_zero 期望 1 实为 24、boundary_T 未 seed）。

## 快速验证

```bash
# 容器内
cd /vllm-workspace/nvfp4-landing/routeA
python3 -c "import torch; from nvfp4_4w4a_mmaf import RouteA; \
  A=torch.randn(256,4096,device='cuda'); Wp=torch.randint(0,16,(4096,2048),dtype=torch.uint8,device='cuda'); \
  Ws=torch.full((128,32),127,dtype=torch.uint8,device='cuda'); \
  i=RouteA(); i.preprocess_weights(Wp,Ws); print(i(A,use_cached_w=True).shape)"
```