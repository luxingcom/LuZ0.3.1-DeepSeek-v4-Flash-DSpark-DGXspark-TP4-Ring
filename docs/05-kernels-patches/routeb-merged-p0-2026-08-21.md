# P0 merged-shape kernel 基准报告（Task #24，2026-08-21）

**任务**: Task #24 · 停机 + P0 merged-GEMM 策略 kernel 基础验证（用户方案：专家 N 维合并大 GEMM）
**执行**: Archi（系统架构师）· node01 生产停机窗口（GPU 独占），routeB vendored 官方 SM120/121 DSL kernel（dense_blockscaled_gemm_persistent_pingpong）
**产出**: 本报告 + p0_full.log（全量日志）+ p0_bench.py/p0_run.sh（在 01:/tmp/routeb_task12/）

> **一页结论（供决策）**
> 1. **停机完成**（UTC 2026-08-21 05:58:13）：四节点 rank 容器已删并验证（无 vllm 容器/systemd inactive/GPU 无计算进程），monitor 维持 stop 无复活。
> 2. **merge-8 @ 300T 门禁 PASS**：`3072×12288×4096` **NVFP4 vec16（E4M3 16 组）+ tile 128³ = 332.1 TFLOPS（MFU 66.4%，峰值 500T 口径）**，正确性 ref check @ tol=1e-2 通过。vec16（我的适配器格式）是该 shape 最快路径，比 vec32（309.3）快 7.4%。
> 3. **★K-tile 256 vec16：可编译可运行（关键未知解除），但性能反而 -33%**（223.8 vs 332.1 TFLOPS）——"K-256 减半 scale 重载→ 150→300+"的杠杆 3 预判在 kernel 级**证伪**（推断：K-256 使 SMEM 阶段数下降的流水线损失 > scale 重载节省）。vec32 K-256 依旧编译失败（复现 Task#12 已知 SF congruence 错误，作对照）。
> 4. **merge-4 兜底路线不占优**：6144×24576×4096 最高 300.3（vec32）/295.3（vec16）——均低于 merge-8 vec16 的 332.1，未达 350。"松弛合并度"兜底依据不成立，**merge-8 是最优档**。
> 5. **kernel 基础判定：Go（以 vec16 + tile 128³ 为推荐配置）**——merged-GEMM 策略的 kernel 载体侧（用户方案杠杆 1/2/4 已备、杠杆 3 排除）成立，MoE per-expert M_e≈96 困境可经 N 维合并进入 332T 档。P1 集成的剩余风险在 host 侧（分桶/重排/scatter 开销），不在 kernel。

---

## 1. 停机记录

- 时间点：**UTC 2026-08-21 05:58:13**
- 操作：01 docker rm -f vllm-tp4-rank0；02/03/04 现查删除 vllm-tp4-rank{1,2,3}
- 验证：四节点 docker ps 无 vllm 容器（0 计数）、nvidia-smi 无计算进程（01 实查 0；03/04 常驻 anemll-embed 属无关项）、vllm-* systemd 维持 inactive、monitor 保持 stop（无自愈复活）
- 生产保持停机状态（用户已批准窗口继续）

## 2. 测试环境

- 容器：生产镜像 0.2.1-v026.0（--rm，GPU 独占，cutlass DSL 4.5.2 + cutlass.testing shim 同 P2/P3）
- kernel：routeb_task12/routeb_official/dense_blockscaled_gemm_persistent_pingpong.py（vendored 官方 SM120/121 warp-level MMA，`mma.sync kind::mxf4` 原生 FP4 路径，SASS 已在 Task#12 门禁验证）
- 编排：p0_bench.py（routeb_bench_blockscaled.py 拷贝，tolerance 改 env 可配 ROUTEB_TOL）+ p0_run.sh（全矩阵）
- 参数：warmup 5 / iterations 30 / epi 128,128 / c_dtype Float16（B-N1 修复语义）/ 官方 ref check **开启且容差收紧至 1e-2**（任务判据）
- 数据：run_bs 内部合成随机数据 + 官方参考实现校验（P3 已另证真实生产权重 rel=4e-4 级）

## 3. 结果总表

| # | 配置 | shape (M×N×K) | tile | sf_vec/dtype | TFLOPS | MFU¹ | 正确性 @1e-2 |
|---|---|---|---|---|---|---|---|
| 1 | 回归锚点 | 4096×14336×4096 | 128³ | 32 / E8M0 | **359.4** | 71.9% | ✅ PASS |
| 2 | **vec16 merge-8（主目标）** | **3072×12288×4096** | **128³** | **16 / E4M3** | **332.1** | **66.4%** | ✅ PASS |
| 3 | vec16 merge-4 | 6144×24576×4096 | 128³ | 16 / E4M3 | 295.3 | 59.1% | ✅ PASS |
| 4 | ★vec16 merge-8 **K-256** | 3072×12288×4096 | 128×128×256 | 16 / E4M3 | 223.8 | 44.8% | ✅ PASS |
| 5 | vec16 merge-4 K-256 | 6144×24576×4096 | 128×128×256 | 16 / E4M3 | 221.1 | 44.2% | ✅ PASS |
| 6 | vec32 merge-8 对照 | 3072×12288×4096 | 128³ | 32 / E8M0 | 309.3 | 61.9% | ✅ PASS |
| 7 | vec32 merge-4 对照 | 6144×24576×4096 | 128³ | 32 / E8M0 | 300.3 | 60.1% | ✅ PASS |
| 8 | vec32 K-256 对照 | 3072×12288×4096 | 128×128×256 | 32 / E8M0 | — | — | ❌ **编译失败**（复现 Task#12：DSL `ValueError: Operation creation failed`，SF congruence，栈见 p0_full.log） |

¹ MFU = TFLOPS / 500（GB10 SM121 块缩放 FP4 峰值口径）

锚点复核：359.4 vs 社区基线 356 / P2 记录 368 —— 复现合格（±3% 运行间带），测试环境与 Task#12 基线一致。

## 4. 判定分析

### 4.1 merge-8 300T 门禁：PASS
- **332.1 TFLOPS = 66.4% MFU** ≥ 300T/60% 目标。用户方案的核心主张（per-expert M_e≈96 的 15T 困境 → N 维合并后 M=3072 进入张量核心舒适区）**在 kernel 级成立**。
- 最快路径是 **NVFP4 vec16（E4M3 16 组）**——即 Task #20 适配器的输出格式（E8M0→E4M3 LUT 精确扩展 + swizzle），权重链已验证（probe5: rel=1.41e-3）；且 vec16 比 vec32 快 7.4%（scale 流量减半的真实收益在 vec16 vs vec32 的对比中体现，而非 K-256）。
- 对照锚点（359.4）：merge-8 shape 达到锚点的 92.4%——shape 效率损失可接受。

### 4.2 K-tile 256：可行但负收益（杠杆 3 证伪）
- **编译性**：vec16 + K-256 **编译通过、运行正确**（对照：vec32 + K-256 仍编译失败，复现 Task#12 已知问题）——文档中"vec16 待测"的未知解除。
- **性能**：223.8 vs 332.1（**-33%**）。K-256 使每 stage 的 SMEM 占用近乎翻倍 → 流水线 stage 数下降，损失大于 scale 重载减半的收益。"150→300+ 靠 K-256"的预判不成立；**tile 128³ 即为该 kernel 的最优档**。

### 4.3 merge-4 兜底：不占优
- 295-300 TFLOPS，低于 merge-8 vec16 的 332.1，且未达 350。"松弛合并度"路线无性能依据；若 merge-8 的分桶重叠度不达预期，merge-4 的 kernel 档位仍可用（300T 档）但不是提升路径。

### 4.4 kernel 基础判定：**Go**
- 载体：**routeB vendored DSL kernel（pingpong 调度）+ NVFP4 vec16 + tile 128³ + epi 128,128 + c_dtype fp16/bf16（16-bit 铁律）**。
- 权重侧：Task #20 适配器（E8M0→E4M3 精确 LUT + swizzle，-0731 直配）或 -nvfp4 checkpoint 原生。
- 排除项：K-tile 256（负收益）、vec32 K-256（编译失败）、merge-4 作为主档（不占优，保留为分桶重叠度不足时的退路）。
- **P1 集成的风险不在 kernel 侧**：剩余全部在 host 侧——专家分桶/组合重叠度（用户方案的桶算法）、token 重排 gather/scatter 开销（每层 2 次 [M,K] 重排的内存带宽成本）、与 vLLM 调度栈的集成。这些决定 332T 的 kernel 峰值能保留多少到端到端。

## 5. 对 P1 的输入（推荐配置汇总）

| 项 | 推荐 | 依据 |
|---|---|---|
| kernel 载体 | routeB DSL pingpong | Task#12 SASS 门禁 + 本 P0 |
| scale 格式 | NVFP4 vec16 (E4M3) | §4.1（比 vec32 +7.4%，且适配器已备） |
| tile | 128×128×128 | §4.2（K-256 负收益） |
| epilogue / C dtype | epi 128,128 / fp16 或 bf16 | B-N1 修复（16-bit 铁律） |
| 合并档 | merge-8（3072×12288×4096） | §4.3 |
| 权重来源 | -0731 + Task#20 适配器（E8M0→E4M3 精确） | Task#20/#21 闭环 |
| 下一步风险清单 | ① 分桶重叠度实测（真实路由分布）② gather/scatter 端到端开销 ③ 与 vLLM 前向集成（apply 挂点同 A′ 插件模式） | — |

## 6. 工件

| 文件 | 位置 |
|---|---|
| p0_full.log（全矩阵日志含 vec32 K-256 失败栈） | 本地 _routea_work/ + 01:/tmp/_routea_work/ |
| p0_bench.py（tolerance env 化）/ p0_run.sh（矩阵） | 01:/tmp/routeb_task12/ + 本地 |
| 停机验证记录 | shutdown_out.txt |

环境约束：一次性容器 --rm；生产停机保持（窗口已批准）；01 GPU 独占（bench 峰值显存 < 3GB）；未触碰生产脚本/数据。
