# kernel① routeB 方案定案（350T 性能方案，从零推进）

> 版本 v2026-08-20 ｜ 状态：**定案**
> 基线 routeA：vLLM 原生 `cutlass_scaled_fp4_mm_sm120a`（PR #42209/#21309），生产实测 80~180 TFLOPS
> 目标 routeB：**全新实现，≥350 TFLOPS**（GB10 FP4 峰值 500 的 70%），相对 routeA ≥1.5× 才切换
> 环境：DGX Spark 4 节点 TP4 / torch 2.11 / triton 3.6 / sm_121a / vLLM 0.26

---

## 一、routeA 现状与归因假设（为何官方内核只到 80~180）

| # | 假设 | 依据 | 验证方法 |
|---|---|---|---|
| H1 | **per-tensor scale 精度/路径**：`cutlass_scaled_fp4_mm_sm120a` 走 per-tensor scale（非 per-block），A 需先量化到 fp8/fp4——A 量化开销若计入调用，占比可达 20~40% | 官方签名确认 A/B 各一个 scale | perf_diag 拆测 GEMM-only vs 全链路 |
| H2 | **shape 失配**：生产 MoE 专家 GEMM（M=256~2048, N=2048/4096, K=4096）相对官方内核默认 tile 非最优 | 80 vs 180 跨度 2.25× 是强 shape 信号 | perf_diag 按 shape 扫描 |
| H3 | **sm120a 非 sm121a 特化**：官方内核按 sm_120a 编译，GB10 为 sm_121a，可能未吃到 121a 特化调度 | 编译目标差异 | SASS 门禁 + 编译对照 |
| H4 | **tile 保守**：vLLM 通用内核 tile 小于 smem 上限（SM120 99KB），未充分展开 | 官方示例保守配置 | 自研 tile 扫描 |

## 二、routeB 技术选型（三条候选路线）

| 路线 | 技术 | 预期 | 集成成本 | 风险 |
|---|---|---|---|---|
| **R1 现成官方** | FlashInfer 0.6.8+ NVFP4 CUTLASS backend（SM12x 优化 kernel，双 arch JIT 硬件 E2M1） | 200~400 | 低（pip/源码 + backend 切换） | 版本需 ≥0.6.8；vLLM 0.26 兼容性待验 |
| **R2 自研 CUTLASS** | CUTLASS 3.9 `OpClassBlockScaledTensorOp` + `mmaf_scaled`，sm120a/sm121a 双 target，per-block scale（E8M0 16 分组），tile 128×128×64 / 128×256×64 + 集群 2×1 + GROUP_M swizzle | 350~475 | 高（.cu + CMake + torch 绑定 + SASS 门禁） | 前次 B1/B2/B3 编译阻塞教训需规避 |
| **R3 调度调优** | 不换内核：routeA 加 W 缓存 + A 量化分离 + tile 预热 + CUDA Graph 分 phase | 180→230 | 低（插件层） | 天花板 ~250，不足以到 350 |

**推荐组合**：**R3 先行（零风险白拿 30~50）+ R1 评估（FlashInfer 版本可用性，最快到 350 的路径）+ R2 作为最终兜底/冲刺**（R2 是唯一能确定性 >350 的路线）。

## 三、routeB 设计规格（R2 自研 CUTLASS 版，主力目标）

```
MMA:       mmaf_scaled m16n8k32 (e2m1 × e2m1, fp32 acc)   ← SM12x 原生 FP4 指令
OpClass:   OpClassBlockScaledTensorOp
ArchTag:   Sm120（编译 sm_121a，SASS 门禁 grep mma.*e2m1|mmaf）
Tile:      128×128×64（SM120 smem 99KB 约束）| 备选 128×256×64 / 256×128×64
Cluster:   2×1（可选，128 SM 粒度）
Scale:     per-block E8M0（16 沿 K），主机侧 swizzle（4 scales/int32）
A:         fp32 → 独立量化 kernel → e2m1 打包 + E8M0 scale（K 向，对齐 cutlass 打包）
W:         [K,N//2] → [N,K//2] K 向重打包 + scale swizzle，每层一次缓存
输出:      fp32（生产语义）| 可选 bf16 探针（提吞吐，精度需评估）
```

## 四、执行计划（分阶段，每阶段可回滚）

| 阶段 | 内容 | 产出 | 判据 |
|---|---|---|---|
| **S0 归因**（30 min） | 生产跑 `perf_diag_fp4_gemm.py`（对 routeA，按 MoE 真实 shape） | 量化占比 / shape 表 | 定位 H1/H2 主因 |
| **S1 R3 调优**（1 天） | 插件层：routeA W 缓存 + A 量化分离 + tile 预热 + graph | 180→230 TFLOPS | 无新内核，可灰度 |
| **S2 R1 评估**（1~2 天） | 生产验证 FlashInfer ≥0.6.8 可用性 + NVFP4 backend 同 shape 实测 | FlashInfer TFLOPS 表 | ≥350 则 routeB=FlashInfer |
| **S3 R2 自研**（3~5 天） | CUTLASS 3.9 blockscaled .cu（按 §三规格）+ 编译 + SASS 门禁 + pytest/benchmark | 350~475 TFLOPS | ≥350 且 ≥1.5× routeA |
| **S4 A/B 切换** | routeB vs routeA 同 shape 同 harness + 数值 ≤1% + needle 128K | 生产切换决策 | 全绿后灰度 K1 |

## 五、需要的输入（下一步）

1. **[生产] `perf_diag_fp4_gemm.py` 输出**（routeA 的 shape/量化占比）——S0 依赖，决定 R3 调优重点
2. **[生产] FlashInfer 版本号**（`python -c "import flashinfer; print(flashinfer.__version__)"`）——S2 依赖
3. [我] 据 S0 结果启动 S1（插件层调优）或 S3（CUTLASS 编写）

## 六、风险与回滚

- 任何阶段 routeB 未达标 → 维持 routeA 现役（生产零改动）
- R2 前次 B1/B2/B3 编译阻塞教训：S3 前先以官方 `69_blackwell_sm120_blockscaled_gemm` 示例最小可编译验证，再叠加自研参数
- SASS 门禁是硬门槛：`nvdisasm | grep mma.*e2m1` 不出现即判定失败，不进入基准
