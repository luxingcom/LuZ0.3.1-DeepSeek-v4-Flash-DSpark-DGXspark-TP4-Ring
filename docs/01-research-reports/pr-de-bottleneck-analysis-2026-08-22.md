# PR/DE 瓶颈深度归因报告：把"算力得不到发挥"分解到组件级（2026-08-22）

**执行**: prde-bottleneck（系统架构师）· Roofline 分析模型 + 生产在位低侵入取证（不重启、单请求小负载）
**任务**: 用户判断"当前有足够的算力但得不到发挥"（PR 2510 tok/s → 线性层平均算力 51.5T ≈ FP4 峰值 2 PFLOPS 聚合的 ~2.6%、bf16 峰值的 ~10%）——本报告把该缺口定量归因到组件级，并给出优化挂钩。
**口径标注**: 【实测】= 在位仪器/日志直接测量；【理论】= Roofline 下界计算；【推断】= 实测×理论约束推出的区间。

---

## 0. 一页结论（供裁决）

1. **"算力得不到发挥"的真相：PR 的每 token 时间里，~62% 的理论下界是"非算力资源"（UMA 带宽 + 环网通信）——在当前模型形态（W4 MoE + TP4 + bf16 ring AR）与负载形态（单流 prefill M=1024）下，FP4 算力物理上就吃不满**。要吃满 bf16 tensor core 需 MoE 每 expert M_e≥114（步进 M≥4.9K）；吃满 FP4 需 M_e≥458（步进 M≥19.5K），而当前调度钳制封顶 12×1024=12.3K。
2. **PR 步时瀑布（407ms/步，M=1024）**: MoE 185-204ms（45-50%）> **TP4 all-reduce ~127ms（31%，实测确认 bf16 全量、无压缩、串行全暴露）** > attention 40-85ms（10-21%）> bf16 稠密池 ~30-35ms（7-9%）> 杂项 5-15ms。PR 理论上限（kernel 全优、通信形态不变）≈ **3560-3700 tok/s（+42-47%）**；叠加 fp8 AR ≈ 4600-4800；再叠加 AR 重叠 ≈ 5500+。
3. **AR 是性价比最高的第一刀**：实测每步线上流量 tx=rx=1119MB/节点（= bf16 ring 理论 1084MB，差 3%），2 收口+2 发口环单向，突发带宽上限 ~8.8GB/s → **~127ms/步全暴露**（fork 禁了 pynccl 和 custom AR，prefill eager 无重叠；且 `fuse_allreduce_rms=False`、`fuse_gemm_comms=False` 两个现成开关未开）。fp8 AR 即 -63ms（PR +18%）。
4. **DE 不是算力问题**：C1 步 41ms 中 MoE 权重读 ~23ms（56%）+ 稠密池 ~7ms + AR 小消息延迟 ~5.5-7ms + MTP draft ~2-4ms；**C1 ≈ 权重带宽 Roofline 的 73-75%，C12（M=96，~229 distinct experts → ~31GB 权重/步）已顶到 273GB/s 带宽墙**。DE 上行空间 = 接受率（acc_len 3.65→6+ ≈ 1.6-1.9×）与 AR 延迟（one-shot ≈ +9%），不在算力。
5. **两个必须修正的既有口径**：① bench "4K/16K/32K/64K" 标签实际 prompt 为 8.2K/32.8K/65.5K/131K tokens（make_prompt 的 FOX 句≈2× label）——2510 tok/s 是真实值（逐轮行用真实 prompt_tokens），但上下文档位标称全部失真；② 08-12 报告"UMA 有效带宽 643GB/s"实为 **CPU 侧 PMU 计数**（armv8_pmuv3_0/1 只有 2 个 CPU 簇 PMU），不能作 GPU 带宽证据——GPU 侧仍应按 273GB/s spec + 业务反推。
6. **头号异常待查**：threshold 1024→2048 A/B 实测 PR 反降 22.5%，与本 Roofline 预测（MoE 199→~145μs/token，PR 应 +15-20%）**方向相反**——扩 M 是最大的单项杠杆（probe15: M=4096 时 B12X 带宽效率 320GB/s 级 vs M=1024 的 169GB/s），必须 root-cause 后重启该实验（候选机制见 §6.4）。

---

## 1. 数据源与在位取证清单（2026-08-22 08:25-08:33 UTC）

| # | 手段 | 结果 | 侵入性 |
|---|------|------|--------|
| E1 | vLLM profile API 探测（openapi.json 路由表） | **fork 未暴露 /start_profile**（404）→ torch profiler trace 需重启，放弃，列入窗口清单 | 零 |
| E2 | 4 轮单请求 prefill 复测（~8.2K tokens，唯一 nonce）+ nvidia-smi 200ms 采样 | clean 轮 TTFT 3.37-3.40s / **2411-2433 tok/s**（vs 基线 2510，-4%：同节点 aicad 服务 CPU 竞争）；**prefill 期间 SM util 93-96% 连续、功率仅 48-59W**；decode 95%/33-42W；首请求冷效应 +0.9s（+26%） | 单请求×9，零风险 |
| E3 | RDMA 计数器采样（/sys/class/infiniband/*/ports/1/counters，200ms，4 口） | **prefill 每步（1024 tok）每节点线上 tx=rx=1119MB（octets）**；2 口纯收 + 2 口纯发（环单向）；IDLE 时 0 流量 | 零 |
| E4 | perf mem_access PMU（-a，1s 间隔） | 只有 CPU 侧 PMU（armv8_pmuv3_0/1）→ 判定为 CPU 访问计数，不用于 GPU 带宽推断（修正 08-12 口径） | 零 |
| E5 | 生产日志/配置（docker inspect + logs） | 权重 **40.5GiB/rank**、KV 53.58GiB/6,042,089 tokens（5.78KB/tok）、NCCL_ALGO=RING + 4ch + 4HCA、`disable_custom_all_reduce=True`、`VLLM_DISABLE_PYNCCL=1`、`fuse_allreduce_rms=False`、`fuse_gemm_comms=False`、MTP dspark n=7 probabilistic、cudagraph 尺寸 1-96（decode only） | 零 |

历史数据复用：probe15（克隆环境单 GPU、per-rank 形状 E=256/I=512/H=4096，W4A16 每 token 边际 2.46→1.78μs）、panorama 基线、large-batch-analysis、routeb-pr-2x、b12x-tail-path-strategy、analysis-tp2-tp4-communication。
本地工件：`_prde_bn/prde_bn_evidence{,2,3,4}/`（GPU 时间线/RDMA 计数器/workload 标记），服务器留档 `/tmp/prde_bn_evidence*`。

---

## 2. Roofline 模型（账本）

### 2.1 硬件与模型常数

| 项 | 值 | 口径 |
|---|---|---|
| GB10 每 chip | FP4 dense 500T / bf16 ~125T / UMA 273GB/s | spec【理论】 |
| 互联 | 4×RoCE 200G（~4.4-5GB/s/口实测），ring 每节点 2 发口 + 2 收口 → 突发 busbw 上限 ~8.8-10GB/s | 【实测+推断】 |
| 权重/rank | 40.5GiB = MoE W4 34.6GB（805MB/层×43）+ bf16 稠密池 ~5.9GB（shared experts 0.54 + attn 投影 ~1.1 + lm_head 0.27 + embed/MTP） | 【实测】+【理论】分解 |
| 线性层 FLOPs/token | 全模型 20.5G（routed 13.0 + shared 2.16 + attn 投影 4.3 + lm_head 1.07），per-rank 5.14G | 【理论】（既有账） |
| 通信/token | 44 层×2 次 AR×4096×**2B(bf16)** = 736KB 算法量；ring wire/节点/步（1024 tok）= 1.5×8.4MB×86 = **1084MB** | 【理论】+【实测】印证 |
| MoE 算术强度 @M_e=24 | 96 FLOP/B vs 机器平衡点 1831（FP4）/458（bf16）→ **深带宽区** | 【理论】 |

### 2.2 关键校准：实测通信量反推 AR 精度与形态

E3 实测每步每节点 wire tx=rx=1119MB vs bf16 ring 理论 1084MB（**差 3%**，余量 = MTP 层/lm_head gather/杂项）；fp8 假设只预测 542MB（差 2 倍，排除）；bf16+2×流量假设排除。**定论：AR 为 bf16 全量、NCCL RING、无压缩、无重叠**（eager 前向 + pynccl/custom AR 均禁用 → 计算流串行阻塞）。
AR 墙钟【推断】= 1119MB ÷ 突发 busbw（8.8-10GB/s 双发口上限）≈ **112-127ms/步（28-31%）**。
交叉验证：功率模型 0.31×~25W(NCCL 等待) + 0.69×~65W(计算) ≈ 52W ≈ 实测 prefill 平均 48-52W ✓。

### 2.3 TTFT 2× 之谜（口径定论）

bench_panorama.py `make_prompt`: "4096t" 标签 = FOX×819 句 ≈ **8202 真实 tokens**（usage.prompt_tokens 实证 8201-8203）。逐轮行 `prefill=pt/ttft` 用真实 token 数 → **2510 tok/s 为真实 GPU 吞吐**；汇总行 `length/med`=1254 是 label 口径的假数。8.2K÷1024≈8.09 步×407ms=3.29s≈TTFT 3.27s ✓ 步间 host 间隙可忽略（E2 的 SM util 连续 96% 亦印证）。**上下文档位实际为 8.2K/32.8K/65.5K/131K**——"64K PR 衰减 -10%"实为 131K 时的衰减，attention 随 ctx 的增长比标称档位看起来温和得多。

---

## 3. PR 瀑布（407ms/步，M=1024，单流 ~8.2K ctx）

| # | 组件 | 时间/步 | 占比 | Roofline 下界 | 距离 | 口径 |
|---|------|--------|------|--------------|------|------|
| 1 | **MoE routed（B12X W4A16）** | **185-204ms** | **45-50%** | **127ms**（805MB/层×43 ÷ 273GB/s，带宽型） | 1.5-1.6×（带宽效率 62-69%，有效 169-186GB/s） | 【实测】probe15 M=1024=4.755ms/层（含 bind）×43；生产复用 binding → 下沿 185 |
| 2 | **TP4 all-reduce（bf16 ring）** | **~112-127ms** | **28-31%** | fp8=63ms / 重叠后≈0 | 全暴露 | 【实测】流量 1119MB/步【推断】墙钟 |
| 3 | Attention（sparse MLA topk512 + SWA128 + 3 hash 层） | 40-85ms | 10-21% | ~7ms（QK+PV 计算 bf16@100%） | 6-12× | 【推断】差额（probe 克隆 30% 上界 vs 减去 AR 后的余量下界） |
| 4 | bf16 稠密池（shared+attn 投影+lm_head） | ~30-35ms | 7-9% | 15.4ms（@100% MFU）/ 26ms（@50%） | 1.2-1.35×（MFU ~50-60%） | 【理论】 |
| 5 | elementwise/routing/host/launch | 5-15ms | 1-4% | — | — | 【推断】差额 |
| | **合计** | **407ms** | 100% | **270-281μs/token** | — | 实测 398μs/token |

**每 token 视角**: MoE 181-199μs + AR 109-124μs + attn 39-83μs + 稠密 29-34μs + 杂 5-15μs = 398μs。

### PR 理论上限（分档）

| 情形 | 步时 | PR | 增量 |
|------|------|-----|------|
| 现状 | 407ms | 2510 | — |
| 全组件 Roofline（AR 形态不变） | ~275-288ms | **3560-3720** | +42-48% |
| + fp8/量化 AR（-56-63ms） | ~220-230ms | **4450-4650** | +77-85% |
| + AR 重叠/异步 TP（-100ms 级） | ~150-180ms | **5700-6800** | +127-171% |
| （参考）MoE 吃满 bf16 TC 需 M_e≥114 → threshold≥4.9K；吃满 FP4 需 M_e≥458 → 不可达（seqs 12×1024 封顶） | | | |

### 算力缺口的物理分解（回答"90% 去哪了"）

- **~62%（结构性，非算力资源）**：MoE 权重流下界 124μs/token（UMA 带宽）+ AR 下界 124μs/token（环网带宽）——这部分**在任何 kernel 优化下都存在**，只能源头减量（扩 M 摊薄 MoE 权重读、量化/重叠 AR）。
- **~25%（kernel 层）**：B12X @M_e=24 带宽效率 62-69%（缺口 ~60-77ms/步）+ 稠密池 MFU ~50%（~15ms）+ attention 效率（未分解）。
- **~5-8%（系统层）**：AR 全暴露（若可重叠，最多回收 127ms 中的一部分——注意此条与结构性条目的 fp8 项不叠加计算）。
- SM util 96% + 功率 59W 的组合 = "SM 忙但都是访存/小 kernel + NCCL 等待"的直接物理证据【实测】。

---

## 4. DE 瀑布（C1，步 41ms，acc_len 3.65 → 10.8ms/token）

| # | 组件 | 时间/步 | 占比 | 依据 |
|---|------|--------|------|------|
| 1 | verify 前向 MoE 权重读 | ~23ms | 56% | 8 pos×6=48 routes → ~46 distinct experts×3.14MB×43=6.2GB ÷ 273GB/s=22.8ms【理论】（TP-on-I：每 rank 读全部活跃 expert 的 I 分片） |
| 2 | verify 稠密池权重读 | ~7ms | 17% | 1.9GB ÷ 273GB/s【理论】 |
| 3 | AR 小消息延迟 | ~5.5-7ms | 13-17% | 86 AR×64KB×2 跳×~64-80μs（延迟型，非带宽型）【推断】 |
| 4 | MTP draft 链（dspark n=7） | ~2-4ms | 5-10% | MTP 层+lm_head 权重流量级【推断，待 trace】 |
| 5 | 采样/KV 读/杂 | ~3-5ms | 8-12% | 差额（KV @4K ctx 仅 23.7MB/步，可忽略） |

- **C1 = 权重带宽 Roofline 的 73-75%**（30ms 权重地板 vs 41ms 实测）——"B12X decode 带宽最优"论断成立，剩余缺口是 AR 延迟 + draft + 串行杂项。
- **C12（408 tps agg）已在带宽墙上**：M=96 → ~229 distinct experts → ~31GB 权重/步 ÷ ~100ms 步时 ≈ 310GB/s（≈273 spec + L2 辅助）【推断】。C12 的每流 34 tps 不是调度失败，是物理极限——**12 并发 decode 的唯一上行是 acc_len**（每 accepted token 摊薄 31GB 权重读）。
- acc_len 口径警示：dspark 自报 acceptance 6.5-7.1/7（draft 链质量）vs Prometheus e2e acc_len 3.65-4.40（含拒绝/重滚）——两者相差近 2×，优化收益测算应以 e2e acc_len 为准。
- 注意 R4（large-batch）已证：C1/C12 波动 ±10-20% 是 probabilistic 采样的接受率运气，DE 对比必须接受率归一。

---

## 5. TOP 归因排序（"算力得不到发挥"）

| 排名 | 归因 | 量化 | 性质 |
|------|------|------|------|
| **1** | **TP4 all-reduce bf16 全量串行** | ~112-127ms/步 = 28-31%；每 token 736KB 算法量、109-124μs | 系统层，**最易回收**（fork 开关现成、fp8 现成方案） |
| **2** | **MoE @M_e=24 带宽型几何 + B12X 62-69% 带宽效率** | 185-204ms/步（下界 127）；缺口 60-77ms | 结构（几何）+ kernel（效率）双重；几何部分靠扩 M/merged 桶，效率部分靠 kernel |
| **3** | **稠密池 bf16 低 MFU** | ~30-35ms（@50-60% MFU）| kernel 层；注意 **W4A4 在 M=1024 是 0.85×（probe15 实测，反而更慢）**——量化路线必须与扩 M 联动，单独做是负收益 |
| **4** | Attention/index | 40-85ms（下界 ~7ms）；生产占比显著低于 probe 的 30% 估计 | kernel 层，优先级**下调**；构成未分解（hash 层 index 构建 vs topk 选择 vs gather） |
| **5** | host/launch/杂 | 5-15ms | 小头；prefill eager 无 cudagraph，但 SM 连续 96% 说明 launch 间隙不构成主缺口【实测】 |

**DE 侧**：带宽墙（C1 73-75% roofline、C12 顶墙）+ AR 延迟 + draft 开销。**不存在"算力闲置"可回收项**——DE 的所有上行都在"少读权重"（acc_len）与"少等延迟"（one-shot AR），与算力无关。

---

## 6. 优化挂钩（与既有路线衔接，按性价比排序）

### 6.1 立即可测（需窗口重启，低风险）
| 措施 | 预期 | 依据/衔接 |
|------|------|----------|
| **开 `fuse_allreduce_rms=True`**（fork pass_config 现成开关，现 False） | -15-40ms/步（86 次 norm+AR 往返融合）→ PR +4-10% | 本报告 §2.2；config/vllm.py:1180 区 |
| **fp8/量化 AR**（通信量减半：1119→560MB/步） | -56-63ms/步 → PR +15-18% | §2.2 实测流量；与 NCCL_CROSS_NIC/通道配置兼容性需验 |
| DE one-shot AR（64KB 小消息，4 节点 2 跳→1 跳） | decode -3-4ms/步 → C1 +9-10% | §4 #3 |
| 开 `enable_layerwise_nvtx_tracing` + torch profiler（单请求） | 取证 attention/misc 分解、B12X 生产 bind 开销、draft 链成本 | §7 清单 |

### 6.2 短中期（kernel/工程）
| 措施 | 预期 | 依据/衔接 |
|------|------|----------|
| **root-cause threshold 2048 A/B 反降 22.5% 异常**（§6.4）→ 重启扩 M 实验（1024→2048/4096） | MoE 199→145/97μs/token → PR +10-22%（若异常排除） | probe15 曲线 + 本 Roofline；扩 M 同时改善 merged 桶覆盖（b12x-tail §4.5 乘性叠加） |
| **B12X 小 M 带宽效率 62→85%**（块填充率 24/32=75%、不均匀路由双块、bind 残留） | -50-60ms/步 → PR +14-17% | probe15 M=1024 vs M=4096 外推；衔接 b12x-tail F1/F10 |
| **merged-GEMM 插件（RouteB/B-lite）** | 单流 PR ×1.4（M=1024 恰为 merged 最优区间） | large-batch §2.2；前置 = 真实流量路由重采集 |
| 稠密池 MFU 提升（融合/批布局）；W4A4 **仅当扩 M 落地后**再评估 | -10-15ms/步 | probe15 W4A4 0.85×@M=1024 caveat |
| AR 重叠/异步 TP（大工程） | 至多再 -80-100ms → PR +25-30% | §3 上限表 |

### 6.3 DE 侧
| 措施 | 预期 | 依据 |
|------|------|------|
| 接受率 3.65→6+（draft 温度/采样策略/dspark 调参） | C1 1.6-1.9×、C12 聚合同比例 | §4；以 e2e acc_len 为准 |
| MTP draft 链成本压缩（若 trace 证实 2-4ms 且可并行/合批） | +5-10% | 待 §7-4 |

### 6.4 头号异常：threshold 1024→2048 A/B 反降 22.5%（实测）vs Roofline 预测 +15-20%
候选机制（按嫌疑排序，需 trace/复测定论）：
1. **激活峰值**：M=2048 → 峰值激活 ~2×（2.03GiB→~4GiB），expandable_segments/碎片化可能触发 allocator 慢路径或 cudagraph 池重排；
2. **chunked prefill 与 prefix cache/KV 写交互**（每 chunk 步的 KV 写放大或 block 分配惩罚）；
3. **调度器步进逻辑**：threshold 改变单请求 chunk 数（8.2K: 8 步→5 步），若步间有固定开销应**变快**——反降说明存在随 M 超线性增长的项（激活重算？attention 二次项？）；
4. AR 消息 2×（8.4→16.8MB/AR）落在 NCCL 协议切换点（NCCL_TUNER_THRESHOLD=40960、NCCL_BUFFSIZE=8MB）——16.8MB>8MB buffsize 触发不同协议路径，若该路径效率低可解释反向。
**行动**：复测臂必须带 §7-1 的 trace + RDMA 计数器同采（AR 时间是否随 M 超线性）。

---

## 7. 需窗口的后续 profiling 清单（按价值排序）

1. **torch profiler trace（单请求 4K prefill + 短 decode）**：开启需重启暴露 /start_profile（或设 VLLM tracing env）。产出 top-20 kernel 组件归属表（B12X/attention/nccl/elementwise/其它）→ 钉死 attention 40-85ms 区间、B12X 生产 bind 残留、launch 间隙、draft 链成本。**这是当前最大的测量盲区**。
2. **threshold A/B 复测带全仪器**（§6.4）：trace + RDMA 同采 + 激活内存监控。
3. **NCCL busbw 突发实测**（nccl-tests 一次性容器 ×4 节点，可免重启做）：校准 8.8-10GB/s 上限假设 → AR 墙钟从【推断】转【实测】。
4. **dspark draft 链 trace**：draft 2-4ms 假设 + 接受率动力学（probabilistic 采样的方差结构）。
5. **层级 NVTX**（enable_layerwise_nvtx_tracing，现 False）：MoE/attention/AR 逐层时间分布。

---

## 8. 证据索引

| 项 | 位置 |
|---|---|
| 在位取证原始数据（GPU 时间线/RDMA/PMU/workload 标记） | 本地 `deliverables/engineering-assurance/_prde_bn/prde_bn_evidence{,2,3,4}/`；服务器 01:`/tmp/prde_bn_evidence*` |
| 取证脚本 | `_prde_bn/evidence_run{,2,3,4}.sh` |
| prefill 复测 | clean 轮 2411-2433 tok/s（TTFT 3.37-3.40s，prompt_tokens 8201-8203）；首请求冷效应 +0.9s |
| RDMA 实测 | run4：r1 窗口 3.24s，roceP2p1s0f0 rcv 4465MB / roceP2p1s0f1 xmit 4464MB / rocep1s0f0 rcv 4469MB / rocep1s0f1 xmit 4467MB（octets，×4 换算） |
| probe15 原始表 | `_routea_work/probe15_out.txt`（M=1024: W4A16 4.755ms/层）；脚本 `probe15_w4a4_vs_w4a16.py` |
| 生产配置/日志 | 01: docker inspect vllm-tp4-rank0；docker logs（权重 40.5GiB、KV 6.04M tokens、cudagraph 1-96） |
| 基线数字 | e2e-baseline-results/results/（panorama/c1_10/conc_decode）；large-batch-analysis-2026-08-22.md（acc_len 3.65/4.40、步时 41ms 校验链） |
| 通信理论账 | analysis-tp2-tp4-communication-2026-08-09.md（368KB/token 为 FP8 假设口径——**本报告实测修正为 bf16 736KB/token**） |

**局限声明**：attention/杂项的分解依赖差额法（区间而非点值）；AR 墙钟由实测流量÷带宽上限推断（±12%）；probe15 为克隆环境口径（随机均匀路由 vs 生产路由分布不同，M_e 方差更小）；DE draft 成本为量级推断。上述均已在表中标注，转实测依赖 §7 清单。

> 本报告由工程保障团队协作生成，关键决策请由人类工程负责人复核。
