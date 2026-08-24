# 大 batch 性能根因定论 + 新架构 batch 策略分析 — 2026-08-22

**执行**: Archi（batch-analyst，系统架构师）· 纯数据/源码分析（零 GPU 占用；服务器只读取证 + 生产镜像源码 grep + Prometheus 历史指标）
**任务**: ① 层 1 A/B（batched 4096→8240）失败定论级归因 ② 新架构（merged-GEMM 插件 + Triton 长尾）下大 batch 的条件分析 ③ 组合测试矩阵与推荐执行序
**数据源**: layer1-ab-20260821.md / e2e-baseline-prod-2026-08-21.md / routeb-merged-{p1a,phasec,e2e-smoke}-2026-08-21.md / routeb-pr-2x-strategy-2026-08-21.md / results_8240 原始日志 / Prometheus（02:8191，A/B 双窗口 07:16-08:03 与 08:18-08:50 历史Series）/ 生产镜像 0.2.1-v026.0 源码 / routing_capture.jsonl（30,717 tokens）重算

---

## 0. 一页结论（供裁决）

1. **层 1 失败的根因定论：probe 回答了一个基准从未问的问题。** 本 fork 调度器把每个请求的 prefill 步进硬钳制在 `long_prefill_token_threshold=1024` tokens/步（scheduler.py 双循环钳制），`max_num_batched_tokens` 只决定"一步打包几个请求"，**不改变单请求 GEMM 的 M**。PR 基准是串行单请求 → 两种配置下每步 M 都是 1024、B12X 每 expert M_e≈24-40 → probe 的"M 4096→8192 边际 1.38×"前提在 e2e 中从未成立 → PR -3%~+0.8% 正是机制预测值（×1.00）。
2. **C12 -9.8% 不能归因于 8240：全程 0 抢占、KV 峰值占用仅 13.53%、prefix 命中率 89.0% 两轮相同、接受率反而更高（3.57 vs 3.37）；panorama C12 四档方向不一致（+21%/-4.5%/+4.4%/-17%）** → 单轮测量方差主导（dspark probabilistic 抽样使 decode 吞吐天然 ±10-20% 波动）。同理 **C1 +17% 是接受率运气**（acc_len 3.65→4.40，+20%，恰好解释 +17%），不是 batched 效应。
3. **64K 抖动轮（74.3s）非 KV 驱逐**（KV 占用峰值 13.53%，驱逐假设被证伪）；全程 4 节点 GPU 93-96% 占用无停顿、时钟稳定，两轮 64K 段温度同达 84-89°C 热极限 → 归因热极限下轮间随机波动（高置信分析级，单轮事件）。
4. **KV 缩水账本（定论）**：bytes/token = 5.78KB（源码推导 + 插件 +10.2GB ↔ -1.76M tokens 交叉验证）。batched 4096→8240 的 KV 代价 = **12.8GB（-2.22M tokens）**，其中仅 ~1.7GB 是朴素激活增量，**~11GB 属 batched 缩放的工作区/graph 池（未分解，头号嫌疑 breakable cudagraph 池，待实测）**——大 batch 的真实代价比激活模型贵 7.7×。
5. **新架构收益方向反转（分析级）**：merged 的相对优势在 **小 M**（单流 chunk=1024，B12X 长尾最弱区间 ~30-65T vs merged M_g≥256 即 157T+）而非大 M（B12X 尾部自身随 M 改善）。**大 batch 不是新架构 PR 收益的来源；单流/低并发 prefill 才是。** 且合成语料的 merged 覆盖/加速被填充文本高估（窗口中位加速 1.0×、均值 2.1×），真实流量路由重采集是任何 PR 预测的前置门。
6. **可行 batch 上限结论**：B12X 路线 8240 已证伪、12288 更差（KV 1.64M），维持 4096 定论。插件路线（瘦身后）：**4096 为主档（KV ≥5.15M）**；**8264 为上限档**（顺带消除 spec WARNING，需 scheduled≥8192；KV ~2.9M，可接受）；**12288 不可行**（KV ~0.77M < 12 流生产底线）。
7. DE 不减的机制条件成立（decode 步 batch = seqs×8，与 batched 无关；权重带宽型），**真实风险在 Triton per-expert kernel 的 launch 开销**（每步 1.7-3.4 千次 kernel 发射，未实测）——插件重跑必须含 C1/C12 DE 快测。

---

## 1. 根因定论表（现象 → 根因 → 证据）

| # | 现象 | 根因（口径） | 证据 |
|---|------|------------|------|
| R1 | PR 四档 -3.0%~+0.8%，probe 1.38× 落空 | **调度器 chunk 钳制**：单请求每步 prefill ≤1024 tokens，`max_num_batched_tokens` 只控制步内请求打包数，不改变 GEMM M。PR 基准串行单发 → 两配置每步 M 均=1024，B12X M_e≈24-40（全 256 expert 命中时 1024×6/256=24），probe 的 M 4096→8192 场景在基准中不存在 → 预期增益 ×1.00，实测 ×0.97-1.01 吻合（**定论**） | scheduler.py:519-521（running 循环 `if 0 < long_prefill_token_threshold < num_new_tokens: num_new_tokens = threshold`）+ :861-863（waiting 循环同钳制）；start 脚本 `--long-prefill-token-threshold 1024`；bench_panorama.py L84-95（prefill 轮串行单请求）；probe15 引自 routeb-pr-2x-strategy §2 |
| R2 | KV 缩水 -37%（6.02M→3.81M） | batched 缩放内存合计 **12.8GB**（=2.22M tokens × 5.78KB），其中朴素激活仅 1.63→3.3GB（Δ1.67GB），**其余 ~11GB 为随 max_num_batched_tokens 缩放的工作区/缓冲/graph 池——具体构成未分解，待实测**（头号嫌疑：VLLM_USE_BREAKABLE_CUDAGRAPH=1 的分片 graph 池按 max batched tokens 定尺寸）。经验代价率：**+1 batched token ≈ -536 KV tokens（≈3.1MB）**（**量值定论；构成归因部分待实测**） | KV 数值：restart_layer1_20260821.log:12（3,805,244）vs restart_prod_20260821.log:12（6,024,962）；bytes/token=5780.7B 由 kv_cache_interface.py:398-403（584B/token/层）× compress_ratios [5 层×1, 19 层×4, 19 层×128] 推导，与插件冒烟 KV -1.76M tokens ↔ "+9GB 派生超预算"（e2e-smoke §5）交叉吻合（10.2GB） |
| R3 | 64K prefill 抖动轮（74.3s vs 57.4/57.5s，prefill 1763 tok/s） | **非 KV 驱逐**（原报告假设证伪）：整轮 KV 占用峰值仅 13.53%、0 抢占。实际为**热极限下轮间随机波动**：4 节点 GPU 全程 93-96% 占用无停顿（+17s 是 GPU 忙时）、SM 时钟稳定 2385-2509MHz、两轮 64K 段温度同至 84-89°C（GB10 持续满带宽 prefill 的稳态热区），8240 轮 3 期间个别节点时钟微降至 2385MHz（**排除项定论；正向归因高置信分析级**） | Prometheus（02:8191）：`vllm:kv_cache_usage_perc` max=0.1353 @08:42（65536t C12 decode 段）；`vllm:num_preemptions_total` 07:16-09:00 全程 0；`DCGM_FI_DEV_GPU_UTIL` 08:24-08:28 四节点 93-96%；`DCGM_FI_DEV_GPU_TEMP` 基线 07:22-07:26 峰值 86-89°C vs 8240 08:24-08:26 峰值 85-88°C；XID 字段恒 31（陈旧值，非活动错误） |
| R4 | C12 decode -9.8%（408→368），C1 +17%（92.8→108.5） | **两件都是 dspark probabilistic 抽样的接受率波动，非 batched 效应**。C1 +17% ≈ acc_len 3.65→4.40（+20.5%）的运气（吞吐=acc_len/步时，步时不变）；C12 段 8240 轮接受率反而更高（conc 全程 3.57 vs 3.37），且 decode 步 batch=12×8=96 tokens 与 batched 无关（96≪4024）、cudagraph 尺寸集相同、无抢占无 KV 压力——**无任何 batched 相关机制可解释 -9.8%**；panorama C12 四档 ±4-21% 混合方向 + c1 十轮极差 2.1×（78.5-162.7）证明系统单轮方差 ±10-20%（**定论：不可归因于 8240；方差主导**） | Prometheus 窗口计算：c1_10 段（07:20:25-07:21:27 vs 08:43:12-08:44:04）accepted/drafts=3332/914=3.65 vs 3196/726=4.40；conc 段 3.37 vs 3.57；校验：4.40/41ms≈107 tok/s ≈ 实测 108.5 ✓，3.65/41ms≈89 ≈ 实测 92.8 ✓；speculative.py:1293-1305（slots=n-1=6）；config/vllm.py:1689-1740（scheduled=batched-72） |
| R5 | max_num_scheduled_tokens WARNING（4024/8168）不消失 | **结构性提示**：scheduled = batched − 6×12(=draft slots)；WARNING 触发条件是 scheduled < 8192 → 8240−72=8168 仍差 24 tokens。**batched ≥ 8264 才静默**。reserved 72 tokens 占预算 1.8%/0.9%，无实质性能影响（**定论**） | config/vllm.py:1703-1725（`if max_num_scheduled_tokens < 8192: warning_once`）；4024=4096−72、8168=8240−72 双配置吻合 |
| R6 | MTP/dspark 交互变化？ | **无变化**：全程接受率 3.76 vs 3.80（run 级），draft tokens/轮=7.0 两轮相同，drafts 计数正常推进。batched 对 spec 路径的唯一影响是 R5 的 72-slot 预留（**定论**） | Prometheus run 级：144493/38398=3.76 vs 138179/36396=3.80；268786/38398=7.0 vs 254772/36396=7.0 |
| R7 | probe 1.38× 本身错了吗？ | **没有错，但回答错了问题**：1.38× 是"并发多请求同 шаг prefill 打包翻倍"场景的 B12X 边际收益（M_e 96→192，仍低于 768 拐点但改善真实存在）——而 PR 基准是串行单发，测不到这个场景；基准中也没有任何指标测过并发 prefill 打包（**定论**） | probe15（routeb-pr-2x §2）；scheduler.py waiting 循环：多请求各取 1024 直到 budget 耗尽（4096→~3-4 请求/步，8240→~7-8 请求/步） |
| R8 | C1 波动区间扩大（82-98 → 78.5-162.7） | probabilistic draft 抽样的接受率轮间方差（温度 0 只固定 target 输出，draft 采样仍随机；RNG 流随请求历史变化）。轮 6 峰值 162.7 tok/s 需 acc_len≈7.3/步——接受率运气上界（**定论机制；方差量化见 R4**） | c1_10rounds 原始 10 轮数据（results_8240/03_c1_10rounds.log）；`draft_sample_method: probabilistic`（启动配置） |

**方法论教训（写入后续所有 A/B）**：
- **DE 对比必须用接受率归一**（acc_len 从 /metrics 取），否则 ±20% 的抽样噪声伪装成回归；
- **PR 基准需补并发 prefill 探针**（4/8 路并发唯一 nonce prompt）——这才是 batched tokens 真正改变的量；
- 单轮结果必须 ≥3 轮中位 + 极差报告（64K 抖动轮教训）；Prometheus 的 acc/preempt/KV%/GPU 温度四条曲线实时盯。

---

## 2. 新架构下大 batch 的条件分析

### 2.1 M 的真实语义（本节为后续一切的前提，源码定论）

| batched | scheduled（=token budget） | 单请求 chunk 上限 | 步内可打包并发 prefill 请求数 | 步级 M 上限 |
|---|---|---|---|---|
| 4096（现产） | 4024 | **1024**（threshold 钳制） | ~3-4 | ~4024 |
| 8240 | 8168 | 1024 | ~7-8 | ~8168 |
| **8264** | **8192** | 1024 | ~8 | 8168（WARNING 静默） |
| 12288 | 12216 | 1024 | 12（=max_num_seqs 封顶） | 12288 |

> 推论：batched 的收益上限受 `max_num_seqs(12) × threshold(1024) = 12288` 封顶；超过 12288 无任何打包收益。单流 PR 与 batched **完全解耦**（M 恒 1024）。

### 2.2 收益侧：merged 覆盖与加速重算（30,717 tokens 步级窗口模拟，分析级）

模型：T_merged(M_g) 用 Phase C 实测曲线（157T@256→351T@1536 平台）；T_B12X(M) 用 probe15 双点幂律外推（123T@4096, 170T@8192, 65T@1024——注意 65T 对单流场景偏乐观，e2e TTFT 反推约 30-40T，故下表 merged 侧偏保守）。MIN_M=256（Phase C 建议）。

步级覆盖率（窗口平均）：

| M_step | hash 层 cov@256 | hash cov@2048 | dense 层 cov@256 | MoE 加速 hash/dense | 43 层加权 MoE | PR（MoE 占 55%） |
|---|---|---|---|---|---|---|
| 1024（单流） | 26.8% | 0% | 26.8% | 2.10 / 2.10 | ×2.10 | **×1.40** |
| 4024（batched 4096 打包满） | 33.2% | 25.3% | 25.3% | 1.51 / 1.47 | ×1.47 | ×1.21 |
| 8168（batched 8240） | 55.4% | 25.1% | 25.1% | 1.37 / 1.26 | ×1.27 | ×1.13 |
| 12288 | 57.5% | 27.9% | 27.9% | 1.26 / 1.17 | ×1.17 | ×1.09 |

**三个关键读数**：
1. **方向反转**：M 越大，B12X 尾部自身越快（M_e 24→288），merged 的相对优势越小。merged 插件对 **单流 PR（M=1024，B12X 最弱区间）** 的分析级增益最大（MoE ×2.1 → PR ×1.4），对大 batch 打包场景增益反而最小（×1.13）。**"大 batch 叠 merged"不是收益放大器，是收益稀释器。**
2. **合成语料高估严重（本次新发现）**：hash 层语料前 ~8K tokens 是整段同 cohort 的填充文本（窗口内 100% 单一 expert 组合，加速 5.2×）；**窗口中位数加速仅 1.0×**（多数窗口无 ≥256 cohort）。若真实流量像中位数窗口，merged 单流增益≈0；若像均值窗口，×1.4。**真实流量路由重采集（P1-1 采集器现成）是一切 PR 预测的前置门，建议列为插件 A/B 前置项。**
3. **Triton 长尾 prefill 性能未知是双向风险**：插件把长尾从 B12X 换成 Triton W4A16 grouped——若 Triton 尾部 prefill 慢于 B12X，大 M 场景可能净负。需要"MIN_M=∞（纯尾部）"隔离臂实测。

### 2.3 成本侧：KV 预算账（bytes/token=5.78KB，定论基数）

| 配置 | KV tokens | 600K/请求并发上限 | 口径 |
|---|---|---|---|
| B12X 4096（现产基线） | 6.02M | 10.0× | 实测定论 |
| B12X 8240 | 3.81M（-37%） | 6.3× | 实测定论 |
| B12X 12288 | ~1.64M | 2.7× | 线性外推·分析级 |
| 插件（P1 瘦身达标 ≤-15%）4096 | ≥5.15M | 8.6× | 目标值·待实测（fix-engineer Task #2 交付） |
| 插件 8264 | ~2.9M | 4.8× | 分析级（沿用 B12X batched 代价率，插件下需实测重定标） |
| 插件 12288 | ~0.77M | **1.3×** | 分析级·**不可行**（12 流生产底线失守，单 600K 请求后近乎零余量） |
| util 0.80→0.82 附加 | **+443K tokens**（+2.56GB） | +0.7× | 算术定论；**需四节点 checker KEY_PARAMS 同步**（运维已知双维护点） |

- 功能底线：pool ≥ max_model_len(600K)。基准最坏负载（panorama 64K 档 12 流 × 131K real tokens ≈ 1.57M）在插件 8264 下仍有 1.85× 余量 ✓。
- **插件+8240/8264 的 KV 真值必须实测**（插件工作区 ≠ B12X 工作区；R2 的 12.8GB 中 ~11GB 未分解构成，插件下可能不同）——重跑窗口直接记录 `GPU KV cache size` 即得。

### 2.4 DE 不减的条件（用户核心关切）

1. **机制层（定论）**：decode 步 batch = seqs×8（C12=96 tokens），与 max_num_batched_tokens 无关（96 ≪ 4024/8168）；decode 步每轮都调度（running 循环 FCFS，decode 廉价不饥饿）。batched 4096→8240 对纯 decode **无机制影响**——层 1 实测的 ±波动均可用接受率/方差解释（R4）。
2. **带宽层（分析级）**：decode 是 expert 权重带宽型。校验：C1 步时 ~41ms ↔ 每层 ~10-15 个 distinct expert × 12.6MB × 43 层 ≈ 8GB @273GB/s ✓；C12 步时 ~100ms ↔ ~40+ distinct experts ≈ 21-43GB ✓。**步时 ∝ distinct expert 权重流量（随 batch 亚线性饱和）+ KV 读（线性但小）**——batched tokens 不进入该方程。
3. **真实风险（待实测，P2 已列）**：插件 decode 全量 Triton per-expert。若按 per-expert 网格发射，每步 1.7-3.4 千次 kernel launch（40-80 experts × 43 层）——launch 开销可能吃掉带宽优势。**C1/C12 DE 快测是插件 A/B 的必测项，不是可选项。**
4. **spec slots（定论）**：72-slot 预留无性能影响；batched ≥8264 静默 WARNING（用户"batched 大于 draft 需求"的解 = **8264**，不是 8240）。
5. **混合负载 ITL（分析级，需关注）**：8240 下 prefill+decode 同步的 engine step 步长 ~2×（最多 8168 prefill tokens/步）→ 混合期单步时延上升、ITL 尾部变差；纯 DE 吞吐不受影响。若生产 SLO 含 ITL p99，需在矩阵中加混合负载臂。

---

## 3. 组合测试矩阵与判据（实测口径）

**总原则**：每臂 ≥3 轮中位+极差；DE 一律接受率归一（acc_len 报出）；PR 加并发 prefill 探针（4/8 路并发唯一 nonce）；Prometheus 四曲线（acc/preempt/KV%/温度）全程记录。**全部插件臂前置依赖 fix-engineer Task #2（P0-A DSL 预编译 / P0-B 量化器 / P1 瘦身）落地。**

### Phase 0 — 插件复冒烟（batched 4096，主档）【依赖 Task #2 完成】

| 臂 | 配置 | 判据（保留门槛） |
|---|---|---|
| 0a | 插件 on（MIN_M=256）× batched 4096 | PR 四档 vs 2510/2500/2420/2270：**≥+10% 保留、0~-3% 复盘、<-3% 回退**；DE C1/C12（acc 归一后）**带内 ±5%**（基线 92.8/408）；KV ≥5.15M；mini logprob 差 ≤1% |
| 0b | MIN_M=128 臂（可选，+0.5h） | 单流 PR 相对 0a 增量 >0 则采用 128（真实流量 cohort 更小场景的兜底） |
| 0c | **尾部隔离臂**：MIN_M=∞（merged 永不触发=纯 Triton 尾） | 量化 Triton 尾 prefill/decode vs B12X 的差值——若 PR <-5% 且 DE <-8%，插件路线 No-Go 信号 |

### Phase 1 — batched 轴（Phase 0 判 Go 后）

| 臂 | 配置 | 判据 |
|---|---|---|
| 1a | 插件 on × **batched 8264**（非 8240：顺带静默 WARNING）× util 0.80 | KV ≥2.6M（实测记录真值）；**并发 prefill 探针**（8 路）PR vs 0a 的并发档 ≥+8%（这是 batched 唯一能兑现收益的场景）；串行 PR 四档不回退 >3%；DE C12 带内 |
| 1b | 1a 胜出时叠加 util 0.82（四节点 checker 同步改） | KV 回升至 ≥3.0M；全指标不回退 |
| 1c | （对照，零成本）B12X × 8264 | 不跑——8240 已证伪且机制（R1/R7）对 8264 同样成立；仅当需 WARNING 静默验证时跑启动确认即可 |

### Phase 2 — kernel2 v17 KV 轴【依赖：v17 生产调用点设计立项（当前 v17 仅落位可 import，未接入推理路径，见 nvfp4-dual-kernel-deploy §五/§六）】

| 臂 | 配置 | 判据 |
|---|---|---|
| 2a | Phase 1 胜者 × v17 KV-linear 接入 | KV-linear 算子级 189-278GB/s（已实测）能否转化为 e2e：PR/DE 增量 ≥+3% 保留；KV 格式回归（logprob 门 ≤1%） |
| 2b | 2a × batched {4096, 8264} 2×2 收口 | 组合最优点定档，全量回归（PR 四档+DE 四并发+c1_10+code agent） |

### 不测项（定论豁免）
- **12288**：插件下 KV ~0.77M 不可行（2.3 节）；B12X 下 8240 已证伪且 12288 更差；收益侧 M_e=288 时 merged 优势仅 ×1.17（2.2 节）。三方皆否。
- B12X × 8240 复测：已证伪（层 1），机制清楚，不重复。

---

## 4. 推荐执行序

1. **[前置，天级] 真实流量路由重采集**（P1-1 采集器复用）→ 重算 2.2 节表。**这是 Phase 0 PR 预期设定的唯一依据**——合成语料的中位窗口加速 1.0× 意味着"PR +40%"可能实为"+0%"，先知道再测。
2. **Phase 0**（依赖 fix-engineer Task #2）：插件复冒烟 0a/0c（0b 可选）→ PR/DE/KV 三判据。
3. **Phase 1**：8264 臂 + util 0.82（checker 同步）。重点测**并发 prefill 探针**（batched 的真收益场景）。
4. **Phase 2**：v17 调用点设计单独立项（高侵入，需灰度方案）→ 2×2 收口。
5. 长线（不改结论，仅记录）：若未来要突破 12288 打包上限，需先动 max_num_seqs（连带 KV 需求与 cudagraph 尺寸集）与 long_prefill_token_threshold（chunk 钳制是单流 M 的真正旋钮——**若目标是单流 PR，调 threshold(1024→2048) 比调 batched 更直接**，但需重估激活峰值与 ITL，列为后续单变量实验候选）。

---

## 5. 证据与工件索引

| 项 | 位置 |
|---|---|
| Prometheus 取证（本报告全部指标数字） | 02:8191 query_range；窗口：基线 07:16-08:03 / 8240 08:18-08:50 UTC 2026-08-21；指标：num_preemptions/prefix_cache_{hits,queries}/kv_cache_usage_perc/spec_decode_{num_accepted,num_draft_tokens,num_drafts}/generation_tokens/DCGM_{GPU_UTIL,SM_CLOCK,GPU_TEMP,POWER,XID} |
| 调度器钳制源码 | 镜像 0.2.1-v026.0 vllm/v1/core/sched/scheduler.py:519-521（running）、:861-863（waiting）、:445（token_budget=max_num_scheduled_tokens） |
| scheduled/WARNING 源码 | vllm/config/vllm.py:1689-1740；vllm/config/speculative.py:1293-1305（slots_per_req=n-1=6） |
| KV bytes/token | vllm/v1/kv_cache_interface.py:398-403（584B/token/层）+ 模型 config compress_ratios [0,0,4,128,...,0,0,0] |
| KV 实测值 | 01:/tmp/_routea_work/restart_layer1_20260821.log:12 / restart_prod_20260821.log:12 |
| 基准脚本语义 | 01:/tmp/bench_pkg/scripts/bench_panorama.py（prefill 串行单发 L84-95；decode 同 prompt 允许 prefix cache）/ conc_decode_only.py（同 prompt ×C 流） |
| 路由重算脚本/数据 | 本地 routing_capture.jsonl 步级窗口模拟（30,717 tokens；本文 §2.2 表源数据） |
| 层 1 原始日志 | e2e-baseline-results/results_8240/（本地+01:/tmp/bench_pkg/results_8240/） |
| kernel1/2 部署现状 | nvfp4-dual-kernel-deploy-2026-08-20.md（落位可 import、**未接调用点**）；sre-perf-twokernels-2026-08-20.md（算子级 v17 189-278GB/s；e2e 10 负载未执行） |

**口径声明**：R1-R8 表中标"定论"的条目均有源码行号或 Prometheus 双窗口数据直接支撑；§2.2 全部为分析级（模型假设已列明，含合成语料偏差）；§2.3 插件行与 §3 判据为待实测口径。本报告不做 e2e 外推声明——所有 PR/DE 预期值仅供测试设计，保留/回退以实测判据为准。
