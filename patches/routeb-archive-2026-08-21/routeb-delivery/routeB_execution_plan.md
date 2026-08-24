# routeB 落地执行计划（CUTLASS 4.4.0 Python DSL — 唯一路径）

> 版本 v2026-08-20 ｜ 依据：NVIDIA 论坛 #359960/#364607/#360142 + CUTLASS 官方文档 + BTankut/baristankut 开源实证
> 目标：**routeB ≥350 TFLOPS（SM121 dense NVFP4）**，替代 routeA（vLLM 原生 cutlass，80~180 TFLOPS）
> 环境：DGX Spark 4 节点 TP4 / CUDA 13.x / CUTLASS 4.4.0 / torch 2.11 / vLLM 0.26 / sm_121a

---

## 一、社区实证基线（必达数字）

| 指标 | 实测 | 来源 |
|---|---|---|
| NVFP4 dense GEMM（4096×14336×4096） | **356 TFLOPS（71% of 500 峰值）** | baristankut，CUTLASS 4.4.0 + CUDA 13.1 |
| MoE grouped（8/64 experts） | 120~154 TFLOPS | 同上 |
| Tile 256×128 | 154 TFLOPS（**prefill/大 batch 最优**） | 同上 |
| Tile 128×128 | ~147 TFLOPS（decode/小 batch） | 同上 |
| FP8 dense | 188 TFLOPS | 同上 |

**关键硬件约束**：SM121 SMEM = **101,376 B（99KB）**——所有 tile 配置必须在此预算内（SGLang 默认 ~147KB → OutOfResources，vLLM 同理需避开）。

## 二、技术路径总览

```
CUTLASS 4.4.0 CuTe Python DSL（nvidia-cutlass-dsl-libs-cu13==4.4.2）
  → patch：BlockScaledMmaOp.admissible_archs 加 sm_121a（issue #2800，两处）
  → kernel：官方 dense_blockscaled_gemm_persistent（CUDA DSL kernel 层）
           或 C++ collective builder（TN layout, cluster 1x1x1——baristankut 同款）
  → 语义：MXF4 变体（E2M1 × E2M1 + UE8M0 scale，sf_vec_size=32）——与生产 32 分组直配
  → tile：99KB 预算内 sweep（256×128 prefill / 128×128 decode）
  → 集成：独立 Python kernel 接入 vLLM 插件（首选）或 flashinfer-b12x（备选）
```

**为什么是唯一路径**：MCP 两轮验证（a769032a 等）确认 Triton 任何版本无原生 FP4 MMA codegen（bf16 回退 2.76×）；FlashInfer 生产不可用；手写 CUTLASS 3.9 C++ 三连编译阻塞（B1/B2/B3）。**只有 CUTLASS 4.4.0 Python DSL 有社区实证的 356 TFLOPS**。

## 三、Phase 0：环境准备（30 min）

```bash
# 1) CUDA runtime libs（pip 默认只有 libs-base）
pip install --no-deps nvidia-cutlass-dsl-libs-cu13==4.4.2

# 2) 验证 import
python -c "import cutlass; from cutlass.cute import *; print(cutlass.__version__)"
# 预期 4.4.x；若 import 报缺 CUDA lib → 检查 LD_LIBRARY_PATH 含 /usr/local/cuda/lib64

# 3) 确认 CUDA 版本 ≥13.0（生产 13.2 ✓）、driver ≥580.142（sm121 ISA 修复 + UMA 修复）
nvidia-smi | grep Driver

# 4) 备份将 patch 的文件
cp <site-packages>/cutlass/cute/nvgpu/warp/mma.py mma.py.bak
```

## 四、Phase 1：sm_121a patch（10 min，`patch_cutlass_dsl_sm121a.py` 自动执行）

**文件**：`<site-packages>/cutlass/cute/nvgpu/warp/mma.py`（两处）

```python
# ① admissible_archs（原：只认 sm_120a）
admissible_archs = ["sm_120a", "sm_121a"]

# ② base equality check（原：if not arch == Arch.sm_120a: raise OpError）
if arch not in (Arch.sm_120a, Arch.sm_121a):
    raise OpError(...)
```

**验证**：跑官方 dense_blockscaled_gemm 示例（小 shape），不再报 `sm_121 not supported`。

## 五、Phase 2：复现 356 TFLOPS（半天~1 天，`routeb_bench_blockscaled.py`）

1. **取官方示例**（CUTLASS 4.4 examples）：
   - `examples/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py`（官方）
   - 或 `Jerry2423/cute_dsl_tutorials/04_gemm_blockscaled/dense_blockscaled_gemm_persistent.py`（整理版，含 amax/prefetch 变体）
2. **配置对齐 baristankut**：
   - 数据类型：A/B = `Float4E2M1FN`（e2m1）；scale = `Float8E8M0FNU`（UE8M0）；acc = F32
   - **sf_vec_size = 32**（MXF4 变体——与生产 32 分组直配）
   - tile sweep：`256×128`（prefill）/ `128×128`（decode），tile_k 128（sf_vec_size=32 硬约束）
   - 集群：1×1×1（SM121 无 2-SM MMA）
3. **benchmark shape**：4096×14336×4096（复现 356）+ MoE 真实 shape（N=2048/4096, K=4096）
4. **验收**：dense ≥350 TFLOPS（SASS 门禁 `grep mma.*e2m1` 可选确认）

## 六、Phase 3：语义对接 MXF4（1 天）

| 项 | 处理 |
|---|---|
| W 权重 | 生产 `W_packed [K,N//2]`（N 向 e2m1 打包）+ `W_scale [K//32,N//128]`（E8M0 32 分组）——**MXF4 直配，主机侧零重排**（无需 16 分组 swizzle！） |
| A 激活 | 复用 MCP v17 量化 kernel（a769032a 已验 8 轮正确性）：fp32 → E2M1 打包 + E8M0 [M,K//32] |
| scale 布局 | MXF4 32 分组即 `ScaleMode.Blockwise1x32`（Operator API 命名）/ CuTe DSL sf_vec_size=32——**非 Swizzle32x4x4 的 NVF4 16 分组**，需在示例中改 sf_vec_size |
| 输出 | fp32（生产语义）｜ bf16 可探针 |
| 接口 | 4 参 `(A, W_packed, W_scale, bias)` 保持 vLLM 插件兼容 |

## 七、Phase 4：集成与 A/B（1~2 天）

**首选（独立 kernel 接入插件）**：
- `nvfp4_vllm_plugin/nvfp4_vllm_plugin/quant_config.py` 的 `_nvfp4_prefill` 骨架 → 调 routeB kernel
- M 阈值分派：prefill（M≥256）→ routeB；decode → B12X 原路径（零改动）

**备选（flashinfer-b12x backend）**——ai-muninn 四层配方：
```bash
# L3: patch flashinfer dense_blockscaled_gemm_sm120.py（sm_version 放宽 + CUTE_DSL_ARCH env）
# L4: env:
#   CUTE_DSL_ARCH=sm_121a
#   VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x
#   VLLM_FLASHINFER_MOE_BACKEND=latency
```

**A/B 判据**：routeB vs routeA 同 shape 同 harness，**≥1.5× 且 ≥350 TFLOPS** + pytest 8/8（误差 ≤1e-2）+ needle 128K 全绿 → 灰度 K1。

## 八、风险与回退

| 风险 | 对策 |
|---|---|
| 4.4.0 DSL 无 blockscaled 示例兼容性 | 以 4.4.0 examples 为准（4.6.0 Operator API 教程仅参考 API 名）；Jerry2423 整理版 4.4 兼容 |
| MXF4 32 分组 vs 示例默认 NVF4 16 分组 | 显式设 sf_vec_size=32；若 kernel 只支持 16 → 高精度转换器加 `--block-k 16`（scale 重算） |
| SMEM 超 99KB | tile 必须 ≤99KB 预算；256×128×128 已在 baristankut 验证 |
| CUDA 版本 API 漂移 | 锁定 `nvidia-cutlass-dsl-libs-cu13==4.4.2`（cu13 配套），不追新 |
| 任何阶段未达标 | 维持 routeA 现役（生产零改动） |

## 九、时间线与验收

| 阶段 | 时长 | 验收 |
|---|---|---|
| P0 环境 | 30 min | import cutlass 4.4.x 成功 |
| P1 patch | 10 min | 官方示例不报 sm_121 not supported |
| P2 复现 | 0.5~1 天 | **dense ≥350 TFLOPS**（356 基线） |
| P3 对接 | 1 天 | pytest 8/8 + 生产 W 直喂正确 |
| P4 集成 A/B | 1~2 天 | ≥1.5× routeA + 全绿 → 灰度 |
| **总计** | **3~4 天** | **routeB 上线，prefill 2×+** |

## 十、参考资源（已核实）

- 论坛：#359960（baristankut 356 实证）/ #364607 / #360142（BTankut SMEM 分析）
- CUTLASS issue #2800（sm_121a admissible_archs）
- 仓库：`BTankut/dgx-spark-sglang-moe-configs`（MIT，Docker 镜像 `ghcr.io/btankut/sglang-spark-glm47:latest`）
- 教程：`Jerry2423/cute_dsl_tutorials`（04_gemm_blockscaled 整理版）/ CUTLASS 官方 `examples/cute/blackwell_geforce/kernel/blockscaled_gemm/`
- CUTLASS 文档：Operator API 006_block_scaled_gemm（ScaleMode.Blockwise1x32 + Swizzle32x4x4 语义）
