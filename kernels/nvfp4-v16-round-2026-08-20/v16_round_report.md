# NVFP4 双算子 v16 轮复测报告（delivery #4）—— 两算子性能终值

> 日期：2026-08-20 ｜ 环境：DGX Spark 生产 TP4，`vllm-tp4-rank0`（torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a）
> 输入：`nvfp4-kernels-delivery(4).zip`（kernel① v16.1：E2M1→FP8 e4m3 展开路径 + D1/scale 修正；kernel② 维持 v11）
> 用户要求：**继续测试，给出两个算子的性能**

---

## 〇、结论速览（两算子性能）

| 算子 | 推荐版本 | 正确性 | **性能（生产实测）** |
|---|---|---|---|
| ① prefill_gemm | **v15（bf16 MMA）** | ✅ 8/8（误差 0） | **GEMM-only 26.7~81.4 TFLOPS**（全算子 7.2~14.9） |
| ① prefill_gemm | **v16.1（本轮，fp8 e4m3 路径）** | ✅ 8/8（**修复后**，误差 0） | ❌ **0.1~0.2 TFLOPS**（SASS/PTX 证实 bf16 降级+模拟，**比 torch fp32 慢 100×**） |
| ② kv_linear | **v11（维持）** | ✅ 5/5+ | **53.6~60.6 GB/s**（4680B/token，大 T 稳定 ~56-60） |
| ② kv_linear | v12.1 / v15 | ✅ 7/7 / 7/7 | 17.0~19.0 / 9.3~10.5 GB/s（均低于 v11，弃用） |

**一句话**：v16 的 fp8 e4m3 路径在 Triton 3.6.0/sm_121 **不可用（性能崩塌）**，kernel① 仍以 v15（bf16 MMA）为 Triton 3.6 可达最优；kernel② 维持 v11。

---

## 一、kernel① v16.1 —— 正确性过、性能崩塌（P0）

### 1.1 as-shipped 编译失败（D3：dot_scaled 调用契约）

v16 按 Triton 3.6.0 签名核对，**as-shipped 无法编译**，错误：
```
dot_scaled() got multiple values for argument 'lhs_format'
```
签名实证（容器内 `triton/language/core.py`）：`dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, ...)`，
v16 调用 `dot_scaled(a_fp8, lhs_scale, w_fp8, rhs_scale, acc, lhs_format='e4m3', ...)` 将 w_fp8 放进第 3 位（lhs_format 槽）→ 冲突。
**MCP 8 轮 passed（total_tests=0）再次未捕获**——round12 教训第 3 次复现。

### 1.2 修复链（生产探针逐层定位，最小修复集）

| # | 问题 | Triton 3.6 源码依据 | 修复 |
|---|---|---|---|
| F1 | 位置参数错位 | `semantic.py: dot_scaled(lhs, lhs_scale, lhs_format, ...)` | 重排为 `dot_scaled(a, lhs_scale, 'e4m3', w, rhs_scale, 'e4m3', acc, ...)` |
| F2 | e4m3 必须 k_pack=True | `assert lhs_k_pack or lhs_format=="e2m1"`（e4m3 非 e2m1 → 必须 True） | `lhs_k_pack=True, rhs_k_pack=True`（fp8 PACKED=1，数据布局不变） |
| — | scale 形状（误修复已回退） | `scale_factor = 16 if scale.dtype.is_fp8e4nv() else 32` → **uint8 e8m0 因子=32**，v16 原 [K//32] 正确 | 不动 |

### 1.3 修复后正确性：8/8 PASS，max_abs_err = 0.000000

8 用例（4 shape × bias T/F，rtol/atol=5e-2）全 PASS 且误差 0 —— **D1（N 向打包）、A 侧 /6、e4m3 无损展开三个语义点全部数值验证**。

### 1.4 性能：0.1~0.2 TFLOPS —— SASS/PTX 定论（P0）

| 指标 | 数值 |
|---|---|
| v16.1(修复) 全算子 | **0.1~0.2 TFLOPS**（7 shape 全部） |
| torch fp32 matmul 同环境对照 | 17.1~18.4 TFLOPS |
| 声称（round11） | ~200 TFLOPS（25.05×）→ **实测差 3 个数量级** |

**SASS（nvdisasm 直出 100,289 行 vs v15 的 3,816）+ PTX 交叉铁证**：
```
HMMA.16816.F32.BF16: 64  （全部 HMMA，无一条原生 FP8 MMA）
TCGEN05: 0     E4M3 MMA: 0     FFMA: 264     IMAD: 317
PTX: mma 指令全为 .bf16；e4m3 数据经 cvt.rn.f16x2.e4m3x2 → fp16x2 upcast
SASS: F2FP.F16.E4M3.UNPACK_B（e4m3→fp16 硬件转换，但 MMA 载体为 bf16）
```
**定论：Triton 3.6.0 的 e4m3 dot_scaled 在 sm_121 上 = e4m3→fp16 转换 + bf16 张量核心 MMA + 标量 scale 软件模拟**（10 万行代码膨胀即模拟痕迹），**无原生 FP8 scaled MMA**。与 v15 相比：同样 bf16 MMA 载体，但 v16 多了逐元素 scale 模拟（FFMA/IMAD）→ 比 v15 还慢 2 个数量级。

### 1.5 kernel① 决策

- **v16.1 不可部署**（性能崩塌）；其 E2M1→e4m3 思路正确（数值无损已验证），但 **Triton 3.6.0 不具备 sm_121 原生 FP8 scaled MMA codegen**，此路在 3.6 上走不通。
- **kernel① 推荐维持 v15**（bf16 MMA，GEMM-only 26.7~81.4 TFLOPS，正确 8/8，D1 修复后可用）。
- 400 TFLOPS 目标唯一可行路径仍为：**升 Triton 3.7+（若有 FP8 scaled MMA codegen）或 vLLM PR #42209 CUTLASS 现成内核**（round12 已定方向）。
- **SASS 门禁升级**：kernel① 任何新路径验收 = grep `mma.*e4m3|tcgen05`（原生 FP8 MMA）；当前 v15/v16 都是 bf16 HMMA，均不达标。

## 二、kernel② kv_linear —— 性能终值（v11 维持）

统一 4680B/token（读 1024×4 + 写 584），理论上限 273 GB/s：

| T | v11（维持） | v12.1（参考） | v15（弃用） |
|---|---|---|---|
| 256 | 53.6 | 17.0 | 9.3 |
| 1024 | 56.5 | 18.6 | 10.3 |
| 4096 | 60.0 | 18.9 | 10.5 |
| 16384 | 60.6 | 19.0 | 10.5 |
| 65536 | 56.2 | 18.9 | 10.5 |

**kernel② 维持 v11：53.6~60.6 GB/s（大 T ~56-60，≈理论 21~22%）**，v12.1/v15 均显著更低（19/10.5），弃用确认。
paged v11 维持 5/5（前轮已验证）。

## 三、交付物

- 本报告 + `run_logs.md`
- 复测脚本：`patch_v16_fixed.py` / `patch_v16_fixed2.py`（F1+F2 修复生成器）/ `verify_v16_k1.py` / `verify_v16_fixed.py` / `bench_v16_k1.py` / `bench_k2_v11.py` / `sass_v16.py` / `probe_v16_compile.py`
- 证据：`evidence/`（v16 SASS 8.5MB 全量 + PTX 841KB + 两摘要）
- 容器：`/vllm-workspace/nvfp4-delivery-v16/`（含修复版模块 `nvfp4_4w4a_prefill_gemm_v16_fixed_triton.py`）

生产 4 rank 全程 healthy、GPU 0%、无残留进程，未做恢复操作。
