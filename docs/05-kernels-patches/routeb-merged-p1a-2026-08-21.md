# P1 第一段：merged-GEMM host 侧开销量化报告（Task #3，2026-08-21）

**任务**: P1-1 · host 侧开销量化（① 权重 N-concat 组装 ② gather/scatter ③ 真实路由分布）——merged-GEMM 方案 Go/No-Go 依据
**执行**: Archi（系统架构师）· node01 GPU 独占（生产停机窗口），一次性容器
**产出**: 本报告 + p1_overhead.json（开销实测）+ routing_capture.jsonl（真实路由 30,717 tokens × 4 层）+ 三个分析脚本

> **一页结论（供裁决）**
> **判定：调整（朴素形态阻塞，存在唯一可行绕开方向，需设计转向 + 目标校准）**
> 1. **① 权重 concat 组装：朴素方案阻塞级**——torch.cat 48×12MB 实测 5.42ms/层 → **233ms/step**（D2D 实测带宽仅 **202GB/s**，比预估 273 低 26%）；rank 口径（60 命中专家×3.4MB）83ms/step。两者均超 50ms 止损线。
> 2. **② gather/scatter：朴素 torch 合计 187ms/step**（dup gather 63.7 + 原子 index_add_ scatter 123.0）。**①+② 朴素总开销 ≈ 270ms/step，而 merged GEMM 计算收益只有 ~0.3-1.2ms/step 量级——开销/收益 ≈ 900×（rank 口径），结构性不可行。**
> 3. **③ 真实路由分布（mini 真实权重，3 hash 层 + 1 dense 层）——两个关键校准**：
>    - **M_g≥3072 达成率仅 26.8%**（hash 与 dense 层相同；即使完美按专家组合分桶）。文档"3072×12288 主档"只覆盖 27% 流量；M_g≥1024 也只有 58.9%（hash）/26.8%（dense）。**长尾是主体**：dense 层 7,937 个组合、平均频率 3 token/组合。
>    - **幂律结构分层明显**：hash 层（路由表驱动，占生产 43 层中的 3 层+MTP）top-10 组合覆盖 62%、top-100 覆盖 95%；dense 层（gate 驱动，40 层主体）top-10 仅 31.5%。
> 4. **绕开方案评估：host 侧无解，唯一正解在 kernel 侧**——连续区间零拷贝（浪费 7.5×~25×，证伪）、组合缓存 LRU（hash 59%/dense 31% 覆盖 + 4.3-17GB/rank 显存，不完整）、grouped 免拷贝（run_bs 需连续 B，桶内专家不连续仍需拷贝）。**可行方向 = kernel 侧 B 间接寻址（tile 级 expert 指针表）**：tile-N=128 且每 expert N_e=2048 是 128 整倍数 → 每个 tile-n 唯一属于一个 expert → B tile 基址查表即可，零拷贝 + 桶组合自由 + 大 GEMM 三者兼得。属 vendored DSL kernel 中等改造（估 1-2 周）。
> 5. **建议设计转向（供用户/主理人裁决）**：merged-GEMM 不做全量替代，做**热组合加速层**（hash 层 top-10 组合 62% 流量、M_g 高达 8,220；长尾回落 B12X 原路径）+ kernel B 间接寻址消组装开销；M_g 目标降档至 1024-2048 档并按达成率曲线（§3.3）设定流量分流比。

---

## 1. ① 权重 N-concat 组装代价（实测，01 GPU）

| 操作 | 实测 | ×43 层/step |
|---|---|---|
| D2D copy 576MB（带宽基线） | 5.56ms（**202 GB/s** 双向） | — |
| torch.cat 48×12MB（任务口径） | 5.42ms/层 | **233.1 ms** |
| index_select stacked[256,12MB]×48 | 5.45ms/层 | 234.1 ms |
| index_select 60 expert×3.4MB（rank 口径） | 1.93ms/层 | **83.1 ms** |
| index_select 40 / 80 expert×3.4MB | 1.31 / 2.59ms/层 | 56.2 / 111.3 ms |

- GB10 实测 D2D 带宽 **202GB/s**（任务预估 273GB/s 偏乐观 26%）。
- cat 与 index_select 等价（同一带宽瓶颈）；无 host 侧技巧可绕。
- **50ms 止损线：rank 口径 83ms、任务口径 233ms，均超**。

## 2. ② token gather/scatter 开销（M=4096 chunk 级，实测）

| 操作 | 实测 | ×43 层/step |
|---|---|---|
| dup gather index_select [4096,4096]bf16→[24576,4096] | 1.48ms（253GB/s） | 63.7 ms |
| scatter-add index_add_ [24576,4096]→[4096,4096]（含零初始化） | 2.86ms | **123.0 ms** |
| 朴素 permute gather（无重复） | 0.31ms | 13.1 ms |

- **①+② 朴素合计 ≈ 270ms/step**。
- 端到端算账：merged 路径有效 FLOPs ≈ 24576 对 × 2048 × 4096 × 2 = 0.41 TFLOP（全模型）/step；TP4 rank 口径 ~0.1 TFLOP → **332T 下计算 ~0.3ms**。朴素 host 开销 270ms = 收益的 ~900×。**即便 gather/scatter 可融合进 kernel（量化+gather 入 prologue、加权 scatter 入 epilogue，需 DSL 开发），权重 concat 是结构性开销，host 侧无解**（组合缓存/区间分桶见 §4 评估）。

## 3. ③ 真实路由分布（mini 真实权重 4 层 × 39 prompts = 30,717 tokens）

采集方式：B12xExperts.apply 挂钩（路由终态 topk_ids，含 hash 层 tid2eid 表映射），12 个 prefill batch × 4 层 = 48 条记录。

### 3.1 基础统计

| 层 | distinct 组合 | top-10 覆盖 | top-100 覆盖 | 命中专家数/batch 中位 |
|---|---|---|---|---|
| layer0-2（hash，路由表驱动） | 466 | **62.2%** | 95.4% | 247-253 |
| layer3（dense，noaux_tc gate） | 7,937 | 31.5% | 44.1% | 189 |

### 3.2 分桶模拟（merged GEMM 正确浪费口径：计算 = 桶内 tokens × 桶专家并集列；有效 = 桶内对数）

| 方案 | 结果 | 结论 |
|---|---|---|
| A 严格共现聚类（桶并集≤12） | hash 层 G=466 桶（碎片化），浪费 1.0×；dense 层 G=1137，浪费 1.7× | 聚类可行但桶碎片——**桶数就是组合数，长尾桶 M_g 极小** |
| B 连续区间×8/16/32（零拷贝视图） | 计算浪费 **7.5× / 13.5× / 22.7×** | **证伪**：零拷贝（连续区间）与 N-merge 大 GEMM 数学互斥 |
| C 完美按组合分桶 | 浪费 0，但 M_g=组合频率 | 见 3.3 达成率 |
| D 组合缓存（LRU 预拼） | hash top-8/16/32 覆盖 59/67/78%（显存 4.3/8.7/17.3GB/rank）；dense 31/33/36% | 部分缓解，长尾仍需现场拼 |

### 3.3 M_g 阈值 vs 流量覆盖率（调档依据）

| M_g 阈值 | hash 层覆盖 | dense 层覆盖 |
|---|---|---|
| ≥256 | 62.2% | 28.0% |
| ≥1024 | 58.9% | 26.8% |
| ≥2048 | 34.9% | 26.8% |
| **≥3072（文档主档）** | **26.8%** | **26.8%** |

- hash 层 top-set 频率 8,220（占 26.8% 流量的单组合桶）；前 10 组合 62%。
- dense 层长尾平均频率 3 token/组合——**gate 路由本质分散，27% 是 M_g≥3072 的天花板**（该层无任何分桶算法能突破——数据上界）。

## 4. 绕开方案评估汇总

| 选项 | 评估 | 结论 |
|---|---|---|
| a) 组合缓存 LRU | hash 59%@top-8、dense 31%；显存 4.3-17GB/rank；剩余流量仍现场 concat | 不完整，仅辅助 |
| b) grouped-GEMM 描述符免拷贝 | run_bs 需单一连续 B；Task#20 grouped 慢因 per-expert 小 M（与 merged 不矛盾但 kernel 不支持 merged-group 形态） | 需 kernel 开发，与 c 等价 |
| **c) kernel 侧 B 间接寻址（tile 级 expert 指针表）** | **tile-N=128 整除 N_e=2048 → 每 tile-n 唯一 expert → B tile 基址查表零拷贝；桶组合自由；大 GEMM 保留** | **唯一正解**；DSL 中等改造（tile scheduler 传 per-tile expert offset，SFB 同理），估 1-2 周 |
| d) 连续子集分桶 | 浪费 7.5×+ | 证伪 |

## 5. 建议（第二段设计输入，待裁决）

1. **形态转向**：merged-GEMM 做**热组合加速层**而非全量替代——top 组合（M_g≥1024）走 merged（332T 档），长尾走 B12X 原路径。按 3.3 曲线：hash 层 ~59% 流量可享 merged，dense 层 ~27%；**加权全模型预期 ~30% MoE 流量走 332T 档**（43 层中 3 hash + 40 dense）。
2. **kernel 前置开发**：B 间接寻址（选项 c）是第二段的前置——没有它，任何 host 组装都吃掉全部收益（270ms vs 0.3ms）。gather/scatter 同步融入 kernel prologue/epilogue。
3. **目标校准**：M_g 主档建议 1024-2048（hash 58.9%/34.9% 覆盖）；3072 档作为 top 组合特例（26.8% 流量）。
4. 若用户坚持全量 merged + 3072 主档：数据不支持（27% 上限 + host 开销），需回报设计校准。

## 6. 工件

| 文件 | 说明 |
|---|---|
| p1_overhead_bench.py / p1_overhead.json / p1_overhead_out.txt | ①② 实测 |
| route_capture.py / routing_capture.jsonl | 路由采集（48 records，30,717 tokens） |
| analyze_routing{,2,3}.py / analyze*_out.txt | 三层分析（统计/分桶/达成率曲线） |

环境：一次性容器 --rm、GPU 独占、生产保持停机、mini 用后已清理。
