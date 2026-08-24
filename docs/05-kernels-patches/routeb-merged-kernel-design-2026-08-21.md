# 静态间接寻址 merged-GEMM kernel 设计文档（Task #26 Phase A，2026-08-21）

**任务**: Task #26 · 用户设计指示实现——"热组合混合形态 + 小区静态间接寻址（查表加速）"
**作者**: Archi（系统架构师）
**依据**: P0 kernel 基准（332.1T @ merge-8 vec16 128³）+ P1-1 开销量化（朴素 host 方案 900× 开销差）+ P1-1 路由分布（M_g≥3072 达成率 26.8% 数据上界）+ 本日 kernel 源码核实（间接寻址可行性）

---

## 0. 用户指示解读（一句话）

> "热组合混合形态 + 间接寻址，实现多个组合混合形态叠加小区静态间接寻址，充分利用查表实现加速"

= 三层混合分派（热组合 merged-GEMM / 动态温组合 / 长尾 B12X）× kernel 侧 B/SFB tile 间接寻址（权重零拷贝）× 热组合静态预计算 tile 查表（免每步建表 + CG 友好）。

## 1. 数据依据回顾（设计约束）

| 事实 | 来源 | 设计含义 |
|---|---|---|
| 332.1T @ 3072×12288×4096, vec16, tile 128³ | P0 | kernel 峰值锚点；K-256 证伪，128³ 定版 |
| 朴素 concat+gather+scatter = 270ms/step vs 计算收益 0.3ms | P1-1 ①② | **host 侧组装结构性不可行 → 必须 kernel 侧零拷贝** |
| M_g≥3072 达成率 26.8%（hash 与 dense 层同，数据上界）；M_g≥1024：hash 58.9% / dense 26.8% | P1-1 ③ | 多档混合必须；**诚实预期：端到端收益 5-15%（§9）** |
| hash 层 top-10 组合覆盖 62%（top-set 频率 8220）；dense 层长尾 7,937 组合平均 3 token | P1-1 ③ | 热组合静态表的价值集中在 hash 层 + dense 层 top 组合 |
| B 的 TMA tile 坐标是运行时动态值（`tBgB[(None, tile_coord_mnl[1], ...)]`，pingpong.py:748-755） | 本日源码核实 | 间接寻址 = 局部改造，无框架级障碍迹象 |
| tile-N=128 整除 N_e=2048（每 expert 恰 16 个 tile-n） | 几何 | tile→expert 映射无跨界歧义 |

## 2. 总体架构：三层混合分派

```
                    ┌─ 每 step 路由 topk_ids [M, 6]
                    ▼
        ┌─ 组合分组（按 expert-set 精确匹配，一个小 kernel ~10μs/层）─┐
        │                                                            │
   热组合桶（静态表命中）        温组合桶（动态表）              长尾
   combo ∈ Registry(每层 top-K)   M_g ≥ DYN_MIN 的其余组合      零散 token
        │                              │                        │
   merged GEMM v2 (间接寻址)      merged GEMM v2 (动态表)     B12X W4A16 原路径
   静态 tile 表 (预计算)          每步建表 (~μs)              (per-expert, 不动)
        │                              │                        │
        └──────────── 行内加权 combine（零原子）────────────────┘
                        （exact-set 桶: 每 token 恰属一桶）
```

**关键设计选择——exact-set 桶（零浪费）**：热组合桶 = 高频**精确专家集**（如 hash 层 top-set 频率 8220）。token 的 6 专家全部落在桶内 → merged GEMM [M_g, 6×2048] **计算浪费 0×**、w2 输出 combine 为**行内 6 列块加权求和（无跨 token 原子）**。避开 P1-1 模拟 A 的"并集桶浪费 1.7×+"与原子 scatter 123ms 两个坑。
- 不完全落在任何热桶的 token → 长尾 B12X。**不做子集/并集桶**（v1 从简；浪费率可控的并集聚类留 v2 评估）。

**分派多档（用户"多个组合混合形态叠加"）**：
| 档 | 条件 | kernel | 预期效率 |
|---|---|---|---|
| T1 热（静态表） | combo ∈ Registry（每层 top-K，K~16-64） | merged v2 + **静态 tile 表** | M_g≥3072 → 332T；1024-3072 → 150-300T（待 Phase C 实测插值） |
| T2 温（动态表） | 其余组合且 M_g ≥ DYN_MIN(默认 512) | merged v2 + 动态表 | 100-300T |
| T3 长尾 | M_g < DYN_MIN | B12X 原路径 | 生产现状 |

## 3. kernel B/SFB tile 间接寻址机制（核心改造）

### 3.1 机制（源码已核实可行）

现状（vendored pingpong.py:678-755）：B 为连续 3D 张量 [N, K, L]，`tma_partition` 切 tile 视图，warp 循环里 `tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]`——**tile 坐标本来就是运行时动态值**。

改造：B 描述为**堆叠权重张量本身** [E, N_e, K, L]（4D TMA，E=256 上界，零拷贝——不 concat、不重排，直接用 layer 的 stacked NVFP4 vec16 权重）：

```
warp 循环内:
  e, t = tile_table[tile_coord_n]      # int32×2 设备内存查表（每 tile 一次，L2 常驻）
  tBgB_nkl = tBgB[(None, e, t, None, tile_coord_l)]   # 4D 坐标切片
  SFB 同理（[E, N_e, K/16] 表同源）
```

- **tile→(e,t) 表**：桶的专家集 {e_1..e_6} 展开为 16×6 个 (expert, local_tile) 项——**这就是用户说的"小区静态间接寻址查表"**：热组合的表在组合注册时预计算（静态张量，指针恒定）；温组合每步构建（[96] int32×2，构建成本 ~μs 级）。
- 输出 C 无需间接：merged 输出缓冲 [M_g, N_g] 连续写（epilogue 原样），行内 combine 在 C 上做。
- 改造范围：kernel 参数（+tile_table 指针 +E 维 TMA 重描述）、warp 循环 4 处坐标计算（A 不动/B/SFB/SFB视图）、host 侧 run_bs 增表参数。**无框架级障碍迹象**；风险点：TMA 4D 描述符 box 对齐（N_e=2048、K/2 打包行 128 对齐 ✓ 预满足）。

### 3.2 静态表结构与生命周期（用户增强项）

```
ComboRegistry（每层一份，host 侧）:
  combos: [K] 个 expert-set（frozenset）
  tables: [K] × [16×|combo|] × int32×2  静态 tile 表（预计算，设备常驻）
  row_cap: [K] 行容量（CG 槽位预留，§6）
内存: K=64 × 96 项 × 8B × 43 层 ≈ 2.1 MB —— 可忽略 ✓
```

- **识别**：v1 = 离线校准（mini/真实 prompt 集跑 P1-1 的采集器 → 每层 top-K 组合）；v2 = 在线 EMA 计数 + 按期重建（重建成本一次表构建 ~ms 级，低频）。
- **LRU 淘汰**：K 上限固定（64），命中频率 EMA 低于阈值的表槽让位（表重建 = 重写静态张量，指针不变）。
- 命中/未命中路径都走同一 kernel（表不同），**未命中组合无惩罚**（动态表，正确性同一路径——Phase B 验证覆盖）。

## 4. gather/scatter 融合

| 数据流 | 方案 | 成本核算（P1-1 实测锚点） |
|---|---|---|
| A 侧（token 行 gather + NVFP4 量化） | **fused quant+gather prologue kernel**（读 A bf16 行表 → 量化 E2M1/E4M3 → 写 packed 输出）：一次读一次写，替代"先 gather 后量化"两遍 | 热流量 ~30%：~0.5ms/step（对比朴素 63.7ms 全量） |
| w13 输出 combine | exact-set 桶内**行内 6 列块加权求和**（每 token 一行，无原子）：小 kernel 或并入 w2 prologue | 输出 [M_g, 12288] fp16 ≈ 75MB/层读 → ~0.4ms/step（热 30%） |
| w2 输出 combine | 同上行内求和 → [M, hidden]，**桶间无重叠（exact-set）→ 无原子** | ~0.3ms/step（热 30%） |
| 合计 host/prologue 开销 | — | **~1.2ms/step @ 30% 热流量**（vs 计算 ~1ms 档）——预算内；Phase C 实测校准 |

（若 v2 引入并集聚类桶，跨桶 combine 需原子版兜底——默认关闭。）

## 5. 权重与内存预算（诚实账）

merged 路径权重 = **stacked NVFP4 vec16**（payload [256, N_e, K/2] 原样 + E4M3 k16 scale [256, N_e, K/16] swizzled——Task#20 适配器产出，payload 零拷贝）。**长尾 B12X 需要 b12x 打包格式**（就地重打包销毁原始 payload → 必须先克隆）：

| 方案 | 长尾路径 | 显存增量/rank | KV 影响（util 0.80） | 评价 |
|---|---|---|---|---|
| v1（推荐） | B12X 双表示 | **+~35GB**（payload 克隆 33 + scale 4.3+swizzled 4.3，共享 e4m3 派生链） | KV 6.08M → ~2.2M tokens（-64%） | 内存代价明确；生产部署需裁决，**mini 验证不受影响** |
| v2 备选 | Triton W4A16（消费同一 stacked 权重） | +~9GB | KV -15% | decode 走 Triton 的性能未验（生产弃用它选 B12X 有原因）——需补测 |
| 不可行 | 长尾走 merged per-expert | — | — | M_e≈96 下 kernel 6-43T，差于 B12X（P0/Task#20 数据） |

**生产部署的 KV-性能权衡交由用户/主理人裁决**（P2 教训：util 0.80 被 checker 硬校验不可调）。mini 单卡验证阶段无此约束。

## 6. CUDA Graph 兼容性分析（主理人维度）

- **生产 CG 现状**：cudagraph 捕获仅 decode 尺寸（≤96，VLLM_USE_BREAKABLE_CUDAGRAPH）；**prefill 走 eager**。merged 路径只挂 prefill（M_g≥512 档）→ **天然不进捕获，动态性无冲突** ✓
- **decode = B12X 原路径**，其捕获行为生产已验证，不动 ✓
- **静态表的 CG 深层价值（若未来 merged 进 decode 图）**：静态表 = 指针恒定的设备张量 + `row_cap` 行容量槽位（桶固定容量如 4096，token 表按槽位写入、空槽由 scheduler 跳过——通过静态 predicate mask 表达）→ kernel 启动参数完全静态 → **可捕获**。代价 = 容量 padding 浪费（M_g 波动 <2× 时可接受）。v1 不实现，设计留位（表结构已含 row_cap）。
- 结论：**v1 的 CG 兼容性由"merged 仅 prefill（eager）+ decode 不动"结构性保证**；静态表为 v2 CG 化预留了正确形态。

## 7. 集成形态（A′ 插件同款挂点）

```
plugin: routea_merged_plugin/（PYTHONPATH/pip + vllm.general_plugins entry point）
env:    VLLM_MOE_MERGED=0/1（总开关）  VLLM_MOE_MERGED_HOT_K=64（每层热组合数）
        VLLM_MOE_MERGED_DYN_MIN=512    VLLM_MOE_MERGED_MIN_M（T1/T2 的 M_g 下限）
类:     MergedB12xExperts(B12xExperts 子类)
  process_weights_after_loading:
    1. 先派生 stacked vec16 权重（Task#20 适配器; payload 克隆保护）
    2. super().process_weights_after_loading（B12X 打包, 供长尾/decode）
    3. 载入/构建 ComboRegistry（校准文件或首步在线）
  apply(M<MIN_M 或 decode): super()（B12X 原样）
  apply(prefill): 组合分组 → T1/T2 桶逐桶 run_bs_v2（间接寻址）+ fused prologue/
                  combine → 长尾 token 子批量 super()
  挂点: oracle.mxfp4.backend_to_kernel_cls + quantization.mxfp4.select_...（P1 已验证双注入）
```

## 8. Phase B/C 计划与验收判据

**Phase B（共享 GPU，只做正确性，shape ≤ 2048×4096×4096）**
1. `routeb_official_v2/`（独立副本，基线 128³ 版本不动）：B/SFB 间接寻址 + 表参数；先单 expert 表（应逐位等价基线 kernel）
2. 正确性矩阵：单 expert / 多组合桶 / 桶间不同 expert 顺序（表顺序无关性）/ 静态表与动态表同结果 / 跨桶边界 token / vs torch 参考 rel≤1e-2
3. mini logprob（若共享显存允许；否则 deferred 到 Phase C 前置）
4. 交付：v2 kernel + 测试脚本 + 正确性报告

**Phase C（干净 GPU，需协调窗口）**
- 间接寻址开销：v2 vs 基线 128³ 同 shape TFLOPS 差（判据：保留 ≥90% 峰值，即 ≥300T @ merge-8）
- 静态 vs 动态表增益（建表成本 + 查表 L2 命中）
- fused prologue/combine 端到端：热流量 30% 模拟负载的 merged 路径总开销占比（判据：≤ 计算时间的 1.5×）
- 多档 M_g 效率曲线（512/1024/2048/3072/6144）→ 校准 DYN_MIN 与 K

## 9. 诚实预期声明（必须向用户传达）

即使 kernel 侧完美（332T×90% 保留、零组装开销），**端到端收益受路由数据上界约束**：
- hash 层（3/43+MTP）：62% 流量可享 332T 档 → 该层 MoE 提速上限 ~2×
- dense 层（40/43）：27% 流量享 332T 档 → 该层 MoE 提速上限 ~1.2×
- **全模型 MoE 加权预期 ≈ 1.1-1.25×；端到端（MoE 占比折算）≈ +5-15%**。与"MFU 30%→60%+ 带来的数倍提升"愿景存在量级差距——差距根源是 dense 层长尾（73% 流量、平均 3 token/组合），非 kernel 能力。若用户目标为 >2× 端到端，需回到分桶算法/模型侧（如专家数削减、路由正则化）层面，本设计如实标注边界。

## 10. 风险清单

| 风险 | 概率 | 缓解 |
|---|---|---|
| DSL 4D TMA 描述符/E 维 tma_partition 兼容性 | 中 | Phase B 第一步单 expert 表等价测试早暴露；fallback：B 视作 [E×N_e] 2D + 表映射全局行号（tile 内 128 行不跨界保证下单 TMA 3D 即可——表存全局 tile 行号，E 维不进描述符） |
| 组合分组 kernel 引入 host 延迟 | 低 | ~10μs/层（哈希分组），Phase C 实测 |
| B12X 双表示显存（生产） | 确定 | §5 裁决项；mini 验证不受影响 |
| dense 层收益低于预期 | 确定（数据已证） | §9 预期管理；K/DYN_MIN 档位可按 Phase C 数据再调 |

---

# Phase B 执行记录（Task #26，2026-08-21，接班 Archi-2）

**环境**: 生产运行中共享 GPU（util 0.80→96% 波动，仅小 shape 正确性测试，无性能测量）；生产镜像 0.2.1-v026.0 一次性容器 `--rm`；NVFP4 vec16 E4M3 / tile 128³ / c_dtype Float16（16-bit 铁律）全程遵守。

## B.1 接管状态与前任结论消化

前任（服务器故障中断）工件：`patch_v2_indirect.py`（P1-P10 kernel+run_bs 改造已应用）+ `phaseb_test.log`（T1a identity PASS / **T1b 单 expert 非零偏移 FAIL 99.7% mismatch**）+ 三个已写未跑的诊断脚本（结果丢失）。本轮重跑诊断链还原真相：

1. `dsl_scalar_load_test` → LLVM ERROR：**该测试本身无效**（compiled @cute.kernel 裸调用缺 `.launch`，非标量 load 语义问题）。
2. `phaseb_diag` A/B/C（map=[0]*16 / [80]*16 / identity）三 case 输出完全相同 → 一度判为"map 静默失效（CoordTensor 坐标直通）"——**后证实是 run_bs 返回值管线 bug 的假象**（B.2）。
3. IR/PTX 级核实（CUTE_DSL_KEEP=ir-debug/ptx）：kernel 内 `memref.load(map)` → B tile 坐标 `map<<7`、SFB tile 坐标 `map`（原始 tile 单位）链路**完整正确**，`ld.global.b32` 真实存在。
4. `phaseb_min_repro`（绕过 run_bs 直调 kernel 类，同 compiled kernel 三次执行）：identity rel=2.99e-4 / **map=80 原地改 rel=3.14e-4** / **map=15 新指针 rel=3.27e-4** 全 PASS → **间接寻址 kernel 本体自始正确**。

## B.2 T1b 伪失败根因：run_bs 两处管线 bug（kernel 无罪）

| Bug | 机理 | 修复 |
|---|---|---|
| **P8 参考校验 gather 行索引错** | `tile_map.repeat_interleave(128)` 把 tile 索引直接当全局行索引（漏 `×128 + tile 内偏移`）→ T1b 的 99.7% "mismatch" 是**参考错**而非 kernel 错 | `tile_rows = tile_map*128 + arange(128)`（patch_v2_fix_runbs.py Fix1） |
| **P10 返回 c_torch 管线断链** | `skip_ref_check=True` 时 bench 走 `generate_tensors` 重生成张量，原 `c_tensor`/`c_torch` 从未被 kernel 写入 → 返回 `c_ref` 随机初值（同 seed 确定性）→ phaseb_diag 的 "identity 语义" 假象 | 返回前用原张量补一次执行（Fix2） |

**教训（DSL 工具链）**：cutlass DSL 4.5.2 中 `@cute.kernel` 内 `tensor[dynamic_idx]` 元素读在正确 launch 下有效（memref.load）；直接调用 compiled kernel 对象（无 `.launch`）报 "LLVM ERROR: unsupported operation"，易与元素访问问题混淆。from_dlpack 动态标记张量为 memref（非 CoordTensor），元素访问语义正常。

## B.3 设计偏差（有意简化，报备）

- **表格式**：设计 §3.1 为 `(expert, local_tile)` int32×2 二段查表；实现为**每 merged tile 一个 int32 全局 tile id**（= expert×16 + local_tile）。语义等价、每表减半（96 项 × 4B = 384B/表）、kernel 省一次乘加与第二次访存。若 Phase C 需要 per-expert 元数据可无损扩展（全局 id 可分解）。
- **静态表机制验证**：map 原地重写（指针恒定）→ 输出随之变化（min_repro exec2）= **静态表 LRU 原地重写语义成立**；新指针替换（exec3）= 动态表语义成立；T3/T4 逐位一致 = 表顺序无关 + 静态/动态同果。
- B 4D [E,N_e,K,L] TMA 重描述（§3.1 原案）未采用——fallback 路线（B 视作 [E×N_e] 2D 行空间 + 全局 tile 查表，§10 风险表预设）即已足够且更简，实测正确。

## B.4 正确性矩阵结果：**16/16 PASS**

| 组 | 项 | 结果 |
|---|---|---|
| T0（phaseb_bitwise，直调 kernel 类） | v2 identity vs **基线 kernel 逐位一致（maxdiff=0.0）** + torch 精确参考 rel=2.77e-4 | ✅ |
| T1a | identity 原语义（run_bs 内置参考 tol=1e-2） | ✅ |
| T1b | 单 expert 非零偏移 [5]（n_stacked=16384）——前任伪失败项 | ✅ |
| T2 | 多 expert 桶 [3,7,1]（N_merged=6144，n_stacked=32768） | ✅ |
| T3 | 表顺序无关：[7,1,3] vs [3,7,1] 同 expert 列块**逐位一致** | ✅ |
| T4 | 静态 vs 动态表（同内容不同分配）输出**逐位一致**（确定性） | ✅ |
| T5 | 跨桶边界：桶A[0,1]/桶B[2,3]/合并[0,1,2,3] 各自参考校验 PASS | ✅ |
| T6 | M=2048 / 4096 / **8240**（生产 max-num-batched-tokens 档，非 128 整除 M）merge-8 N=12288 | ✅×3 |
| viewlaunch T1-T5（零 kernel 改动桶管线：stacked 权重视图 + 行内加权 combine） | 单 expert / 桶[3,7,1] / 桶序无关 / 跨桶共享 stacked / M=2048+8192 merge-8，rel=2.35e-4~3.37e-4 | ✅ 6/6 |

- 注 1：T5 跨 launch 逐位对比在 run_bs 口径下不成立（run_bs 全局 RNG 生成数据，N_merged 不同 → 数据流不同源）——跨桶共享 stacked 语义由 viewlaunch T4（显式共享张量）覆盖。
- 注 2：**M=8240 直接通过**（无需 8192 回退）——生产 chunk 档位对 kernel 无 M 对齐障碍（partial M tile 正确处理）。
- 注 3：min_repro 中 map=80 原地改后 16 列块全同（= expert5 tile0 复制），交叉验证查表语义。

## B.5 mini logprob：deferred（报备）

Phase B 交付判据（§8）中的 mini logprob 推迟至 Phase C 前置：(1) merged 路径 vLLM 插件集成（MergedB12xExperts.apply，§7）尚属 Phase C 交付物，kernel 级正确性已由 T0-T6 + viewlaunch 完整覆盖；(2) 共享 GPU（生产 util 0.80 硬校验、11GB 空闲）下 mini 重建 + 双路径 logprob 风险收益比不佳。建议与插件集成合并到 Phase C 干净窗口执行。

## B.6 工件清单

| 位置 | 内容 |
|---|---|
| 01:/tmp/routeb_task12/routeb_official_v2/ + 01:/tmp/_routea_work/routeb_official_v2/（双份） | 改造后 kernel（patch_v2_indirect.py P1-P10 + patch_v2_fix_runbs.py Fix1/Fix2）；**基线 routeb_official/ 未动**（T0 逐位等价证明） |
| 01:/tmp/_routea_work/ + 本地 _routeb_extract/routeb-delivery/ | phaseb_test.py（矩阵 9 项）/ phaseb_bitwise.py（T0）/ phaseb_viewlaunch_test.py（桶管线 6 项）/ phaseb_min_repro.py（根因隔离）/ phaseb_diag*.py · phaseb_load*_test.py（诊断链）/ patch_v2_indirect.py · patch_v2_fix_runbs.py · patch_t5_rng.py / phaseb_final_run{,2}.log |

**Phase B 判定：PASS（16/16）。** 间接寻址改造对基线 kernel 零语义扰动（逐位等价），静态/动态表、桶序无关、跨桶边界、生产 M 档全部正确。剩余风险移交 Phase C：性能保留率（≥90% 峰值判据）、fused prologue/combine 端到端、插件集成 + mini logprob。
