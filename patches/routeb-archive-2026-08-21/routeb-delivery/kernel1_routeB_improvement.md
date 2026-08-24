# kernel① routeB 方案定案 v3（CUTLASS 4.4.0 Python DSL — 350T 实证路径）

> 版本 v2026-08-20（v3）｜ 状态：**路径定案，落地执行**
> 基线 routeA：vLLM 原生 `cutlass_scaled_fp4_mm_sm120a`（PR #42209/#21309），生产实测 80~180 TFLOPS
> 目标 routeB：**≥350 TFLOPS**（GB10 FP4 峰值 500 的 70%），相对 routeA ≥1.5× 才切换
> 环境：DGX Spark 4 节点 TP4 / torch 2.11 / triton 3.6 / sm_121a / vLLM 0.26

---

## 一、v3 重大更新（依据 NVIDIA 官方论坛实证）

**社区研究员 baristankut 已实证：SM121 原生 NVFP4 dense GEMM = 356 TFLOPS（71% 峰值）**

| 来源 | 关键事实 |
|---|---|
| 论坛 #364607 | baristankut 用 **CUTLASS 4.4.0**（CuTe Python DSL）+ **一行 patch**（`sm_121a` 加入 `BlockScaledMmaOp.admissible_archs`）→ dense NVFP4 GEMM **356 TFLOPS** |
| 论坛 #360142（BTankut 详细分析） | SM121 SMEM 仅 **99KB**（SM100 的 228KB 的 43%）→ tile 必须 sweep：**256×128 = 154 TFLOPS（prefill/大 batch 最优）**；128×128 = ~147（decode/小 batch）；MoE grouped（8/64 experts）120~154 |
| CUTLASS issue #2800 | Python DSL `BlockScaledMmaOp` 把 FP4 限制在 sm_100a，sm_120a/sm_121a 被拒——patch 已由社区验证可行 |
| Colfax blockscaled 教程 | Blackwell 两条路径：**MXF4（E2M1+UE8M0，sf_vec_size=32）** 与 **NVF4（E2M1+UE4M3，sf_vec_size=16）**——**MXF4 与我们生产 32 分组 E8M0 格式直配** |

## 二、路线收敛（前版 R1/R2/R3 处置）

| 路线 | 状态 | 说明 |
|---|---|---|
| R1 FlashInfer 0.6.8+ | ❌ **出局** | 生产确认不可用 |
| R2 手写 CUTLASS 3.9 C++ .cu | ❌ **弃用** | B1/B2/B3 编译阻塞 + 已被 R2' 取代 |
| **R2' CUTLASS 4.4.0 Python DSL** | ✅ **定案（routeB 主路径）** | baristankut 实证 356 TFLOPS；免 C++ 模板地狱；MXF4 32 分组直配生产权重 |
| R3 调度调优（routeA 不动内核） | 🔄 可选辅助 | W 缓存 + A 量化分离 + tile 预热，180→230，零风险 |

**MCP 验证结论（本轮 a769032a，8 轮，speedup 2.76×）**：MCP 服务端 Triton **同样无法产出原生 e2m1 MMA**（内核明确标注 `FALLBACK PATH - bf16 dequant`）——Triton 侧（任何版本）确认无解，**routeB 必须走 CUTLASS 4.4.0 Python DSL**。

## 三、routeB 技术规格（R2' 定案）

```
工具链:   CUTLASS 4.4.0 CuTe Python DSL（pip: nvidia-cutlass-dsl-libs-cu13==4.4.2）
指令:     MmaMXF4Op（warp-level blockscaled MMA, m16n8k64）
         E2M1 × E2M1 + UE8M0 scale（sf_vec_size=32）+ F32 acc
         编译目标 sm_121a（SASS 门禁 grep mma.*e2m1|mmaf）
Patch:    nvidia_cutlass_dsl/.../cutlass/cute/nvgpu/warp/mma.py 两处：
         ① admissible_archs = ["sm_120a", "sm_121a"]
         ② if arch not in (Arch.sm_120a, Arch.sm_121a): raise OpError(...)
Tile:     256×128×128（prefill 主力，99KB SMEM 预算内）
          128×128×128（decode/小 batch 备选）
          tile_k 被 128 整除（sf_vec_size=32 硬约束）
Scale:    MXF4 UE8M0 32 分组 —— 与生产 W_scale [K//32,N//128] **直配，零转换**
A:        fp32 → 复用 MCP v17 量化 kernel（已验 8 轮正确性）→ e2m1 打包 + E8M0
W:        [K,N//2] N 向打包直用（对齐 CUTLASS MXF4 布局，主机侧无需重排）
输出:     fp32（生产语义）｜ bf16 可探针（提吞吐，精度需评估）
```

## 四、落地执行计划

| 阶段 | 内容 | 产出/判据 |
|---|---|---|
| **S0 环境** | 生产安装 `nvidia-cutlass-dsl-libs-cu13==4.4.2` + CUDA 13 runtime libs | import cutlass 成功 |
| **S1 Patch** | 改 `warp/mma.py` 两处（admissible_archs + equality check）+ env `CUTE_DSL_ARCH=sm_121a` | 官方 69_blackwell_sm120_blockscaled_gemm 示例跑通 |
| **S2 复现 356** | 官方 dense blockscaled 示例 + tile 256×128 sweep → 实测 | ≥350 TFLOPS（社区基线 356） |
| **S3 接入** | MXF4 32 分组语义对接（生产 W 直用）+ A 量化 kernel 对接（MCP v17 已验证）+ 4 参接口 | pytest 8/8 + ≥350 |
| **S4 A/B 切换** | routeB vs routeA 同 harness ≥1.5× + 数值 ≤1% + needle 128K | 全绿后灰度 K1 |

## 五、风险与回滚

- S2 复现若 <350：tile sweep 扩展（128×256 / 256×256 在 99KB 预算内）+ num_warps 调优
- S3 若 MXF4 布局与生产 W 有出入：备选 NVF4（UE4M3 16 分组）需高精度转换器加 `--block-k 16`（scale 重算，非简单重复）
- 任何阶段未达标 → 维持 routeA 现役（生产零改动）
- SASS 门禁硬门槛：`nvdisasm | grep mma.*e2m1` 不出现即判定失败

## 六、本轮 MCP 交付（语义对标/bf16 参考）

- `nvfp4_4w4a_prefill_gemm_v17_triton.py`（388 行，分离量化架构：A 量化 kernel + bf16 回退 GEMM kernel，**FALLBACK PATH 明确标注**）
- speedup 2.76×（MCP 服务端 GPU，bf16 路径）；正确性 v2~v8 连续 7 轮通过
- 用途：① routeB CUTLASS 的数值对标基（同输入同输出）② prefill 现役的 Triton 侧参考
