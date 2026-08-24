# GSM8K 真实路由重采集 + 调度器 Threshold A/B 测试报告

**日期**: 2026-08-22（UTC）
**执行**: window-engineer（SRE, 工程保障团队）
**停机窗口**: 2026-08-22T06:21:22Z 停机完成 → 2026-08-22T07:33:21Z 生产恢复验证通过（约 72 分钟）
**集群**: DGX Spark 4 节点 TP4（GB10/sm_121a），vLLM 0.26 fork，DeepSeek V4 Flash，flashinfer_b12x W4A16 MoE + Dspark MTP n=7 投机解码，chunked prefill 4096，util 0.80

---

## 执行摘要

1. **GSM8K 真实路由重采集**：1319 题全量 8-shot 业务格式，捕获 1.02M tokens（每层 254,975，为合成语料 30,717 的 8.3 倍）。**合成语料既高估了路由集中度（top-10 减半），也高估了 merged 桶覆盖率（等样本量下高估 3.7~8×）**；按 merged 插件实际作用域（单步内合并）计，真实流量收益上限极低。
2. **调度器 threshold A/B**：`long_prefill_token_threshold` 为 CLI 显式参数（无需改源码）。1024 → 2048 实测：**PR 无收益**——4K 档 -22.5%、16K -5.2%、32K/64K 持平；DE 归一后持平（方差内）。**Route 1（M 扩展）判定否定**。
3. **生产已恢复**：threshold 1024 原值（md5 校验一致），四容器 healthy，自愈链 head/timer/3×worker 全部 active，health 200，B12X/dspark 指标在场。

---

## 一、停机开窗（06:20–06:21 UTC）

| 步骤 | 结果 |
|---|---|
| systemctl stop（01: head/healthcheck.timer/healthcheck/cluster；02-04: worker） | 全部 inactive |
| docker rm -f 四节点 rank 容器 | rank0/1/2/3 全部清除 |
| nvidia-smi 计算进程验证 | 01/02 = 0；03/04 各 1（anemll-embed 常驻，无关项） |
| 停机时刻 | **2026-08-22T06:21:22Z** |

---

## 二、GSM8K 真实路由重采集（第一优先任务）

### 2.1 方法

- **采集器**：复用 P1-1 采集器（patch `B12xExperts.apply` 抓路由终态 topk_ids），脚本 `route_capture_gsm8k.py`
- **模型**：mini 4 层模型（真实 checkpoint 前 4 层，`/tmp/_routea_work/mini0731`），01 单卡，flashinfer_b12x 后端，enforce_eager，max_num_batched_tokens=4096
- **数据**：GSM8K test 全量 1319 题（02:~/gsm8k_test.jsonl），**8-shot few-shot 提示构造与生产评测脚本 final_gsm8k.py 完全一致**（业务真实格式）
- **口径**：prefill-only（max_tokens=1），prefix caching 开启 → 捕获每题新增 tail tokens（8-shot 共享前缀被缓存，与生产行为一致）
- **产出**：`/tmp/_routea_work/routing_capture_gsm8k.jsonl`（896 records = 4 层 × 224 prefill batch，总计 1,019,900 tokens，每层 254,975）
- **层语义**：layer 0-2 = hash 路由层（按 token id 哈希，三层路由逐层完全一致）；layer 3 = dense/gate 层

### 2.2 结果（三口径，均与合成语料同法对照）

**口径 1：全语料聚合（与昨日 merged 评估报告同口径）**

| 指标 | 合成语料（30,717 tok/层） | GSM8K 真实（254,975 tok/层） |
|---|---|---|
| M_g≥1024 覆盖（hash） | 58.9% | **80.9%** |
| M_g≥1024 覆盖（dense） | 26.8% | **65.0%** |
| M_g≥256 覆盖（hash / dense） | 62.2% / 28.0% | 84.3% / 68.1% |
| M_g≥128 覆盖（hash / dense） | 90.4% / 30.1% | 86.6% / 68.2% |
| M_g≥64 覆盖（hash / dense） | 90.4% / 34.3% | 89.2% / 69.2% |
| top-10 组合集中度（hash） | 62.2% | **34.2%** |
| top-10 组合集中度（dense） | 31.5% | **11.5%** |
| distinct sets（hash / dense） | 466 / 7,937 | 5,798（12.4×）/ 59,171（7.5×） |
| hash top-32/64/100 | 77.7% / 91.7% / 95.4% | 62.3% / 82.4% / 85.9% |
| dense top-32/64/100 | 36.2% / 40.6% / 44.1% | 22.9% / 39.4% / 57.8% |

**口径 2：等样本量下采样对照（GSM8K 随机降至 30,717 tok/层，seed=42——分离样本量效应）**

| 指标 | 合成 | GSM8K 下采样 |
|---|---|---|
| hash ≥1024 / ≥256 / ≥64 | 58.9% / 62.2% / 90.4% | **16.1% / 69.7% / 82.4%** |
| dense ≥1024 / ≥256 / ≥64 | 26.8% / 28.0% / 34.3% | **3.3% / 9.7% / 67.0%** |
| top-10（hash / dense） | 62.2% / 31.5% | 34.1% / 11.6% |

→ 口径 1 的高覆盖率主要由 8.3× 样本量驱动；**剥离样本量后，真实流量的大桶覆盖率远低于合成语料估计（hash 3.7×、dense 8× 高估）**。集中度指标（top-K）对样本量不敏感，是稳定的分布形状差异。

**口径 3：单步内合并（per-forward——merged 插件的实际作用域，最决策相关）**

| 指标 | 合成 | GSM8K 真实 |
|---|---|---|
| hash ≥64 / ≥128 / ≥256 | 61.4% / 55.6% / 33.7% | **17.9% / 13.3% / 3.2%** |
| dense ≥64 | 27.2% | **3.2%** |
| 单步内 set 频率中位数（hash / dense） | 4 / 1 | 2 / 1 |
| sets/步（hash / dense） | 117.7 / 1107 | 178.5 / 462.5 |

（注：vLLM 启动 warmup dummy batch 注入约 1.6 个百分点的全同路由伪影记录，已核验并注明；扣除后真实 hash ≥64 ≈ 16%，dense ≥64 ≈ 0~2%。）

### 2.3 结论

1. **昨日"合成语料集中度失真"的怀疑证实，但方向与直觉相反**：重复文本既制造虚假热门组合（集中度虚高 2×），也制造虚假大桶（覆盖率虚高 3~8×）。两个失真同源。
2. **当前 merged 插件语义（forward 内合并）在真实流量下收益上限极低**：hash 层 M≥64 仅 ~16%，dense 层近乎为 0（dense/gate 路由逐 token 高度多样化，59K distinct sets，单步中位频率 1）。要达到口径 1 的 hash 80.9% / dense 65% 必须跨步累积 token（需暂存 hidden states/KV，实现代价高）。
3. **组合缓存（top-K 预拼）真实命中率减半**：hash top-10 = 34.2%（合成 62.2%），top-32 = 62.3% 需 17.3 GB/rank 显存；dense top-10 = 11.5%。吸引力显著下降。
4. hash 层 0-2 路由逐层完全一致（token id 哈希决定）——若做 hash 层专用优化可三层共享路由计算/缓存。

---

## 三、调度器 Threshold A/B（Route 1 主测试）

### 3.1 修改途径（前置查明）

`long_prefill_token_threshold` 是 **EngineArgs CLI 显式参数**，生产脚本 `<INSTALL_DIR>/scripts/start_tp4_head.sh:57` 直接写死 `--long-prefill-token-threshold 1024`。修改 = sed 替换 + `.bak` 留档，**无需改容器源码 / volume mount**。checker（check_vllm_script.sh）不校验该参数，无 KEY_PARAMS 冲突。max-model-len=600000 下 2048/4096 均合法。

### 3.2 A/B 设计与执行

- **Arm A（1024，现场基线）**：生产配置原样重启（restart_run.sh mode 0，含 dspark n=7），防冷启动条件漂移取当日锚点
- **Arm B（2048）**：仅改 threshold，其余全部不动；.bak 留档（start_tp4_head.sh.bak-thrab-20260822）
- 每臂均验证：`'long_prefill_token_threshold': N` 出现在 non-default args 日志、B12X_MXFP4 后端、dspark speculative_config 在场
- **4096 未测**：2048 剂量-反应已呈单调负向（短上下文回退、64K 持平示大 M 无增益空间），按止损纪律省 25 分钟窗口

### 3.3 PR 结果（panorama 单流口径，3 轮中位，tok/s）

| 档位 | Arm A（1024 现场） | Task#25 基线 | Arm B（2048） | **B vs A** | B vs 基线 |
|---|---|---|---|---|---|
| 4K（实 8.2K tok） | 2561 | 2510 | 1986 | **-22.5%** | -20.9% |
| 16K（实 32.8K） | 2527 | 2500 | 2396 | **-5.2%** | -4.2% |
| 32K（实 65.5K） | 2434 | 2420 | 2451 | +0.7% | +1.3% |
| 64K（实 131.1K） | 2256 | 2270 | 2256 | 0.0% | -0.6% |

Arm A 与 Task#25 基线偏差 ≤±2%（锚点可靠）。Arm B 4K 档轮次呈双峰（1888/2552/1986 tok/s），16K 档三轮一致偏慢——回退真实存在，非单轮离群。

### 3.4 DE 结果（dspark n=7，3 轮中位，接受率归一）

| 指标 | Arm A（1024） | Arm B（2048） |
|---|---|---|
| C1 tput_sum | 87.5 tok/s（参照 92.8） | 82.0 tok/s |
| C1 acc_len / tput/acc_len | 0.492 / 177.9 | 0.479 / 171.0（-3.9%） |
| C12 tput_sum | 369.1 tok/s（参照 408） | 370.0 tok/s |
| C12 acc_len / tput/acc_len | 0.420 / 879.3 | 0.483 / 766.6（-12.8%） |

dspark probabilistic 单轮方差大（C12 acc_len 轮间 0.388–0.510 vs 0.401–0.422，分布重叠），归一后差异在噪声带内，**不构成确定性回退结论，也不构成保留理由**（PR 已回退）。

### 3.5 长上下文 ITL 观察（64K prefill 期间单流 decode 穿插）

| 指标 | 1024（预热后复测） | 2048 |
|---|---|---|
| post-p50 ITL | 475 / 493 ms | 923 / 55 ms（双峰） |
| post-p99 ITL | 1844 / 541 ms | 1423 / 1861 ms |
| post-max ITL | 2619 / 541 ms | 1423 / 2058 ms |
| 64K prefill TTFT | 62–63 s | 58–60 s |

两臂方差均大（穿插时序运气主导）。机制上与观测一致：1024-chunk（≈430 ms/块）下 decode 步进节奏 ≈500 ms；2048-chunk（≈850 ms/块）下要么 ~900 ms 间隔要么突发快步。**无饿死、无系统性劣化定论；2048 最坏档停顿略长但未超 1024 侧最差轮**。注：1024 侧首轮测于冷集群（pre-p50 460 ms 异常），已用预热后复测数据。

### 3.6 激活峰值 / KV 确认

KV cache size：Arm A 6,070,522 / Arm B 6,016,007（-0.9%）/ 恢复后 6,042,089（-0.5%）——启动期 profiling 方差量级，无结构性变化。内存 profile 预算由 max_num_batched_tokens=4096 决定，chunk 大小不改峰值预算；util 0.80 不变。

### 3.7 Route 1 判定

**否定。** 判据"PR ≥+8% 保留价值"远未达到（最佳档 +0.7%）；短上下文显著回退。机制解读：单流 PR 的 B12X grouped GEMM 在 prefill M 维度并非瓶颈（M_e≈24-40 区间在 W4A16 路径已够用）；M 翻倍反而在短上下文受损（疑与 kernel tile 配置/chunk 内 attention 二次项/调度行为相关），64K 持平说明大 M 无增益空间。**三路线组合优化中可排除该变量。**

---

## 四、回滚与生产终态（07:11–07:33 UTC）

| 项 | 状态 |
|---|---|
| start_tp4_head.sh 还原 | cp .bak-thrab-20260822 覆盖，md5 = b22c91b6…（与改前一致），threshold 1024 确认 |
| .bak 留档 | start_tp4_head.sh.bak-thrab-20260822 保留（与现文件内容一致的审计副本） |
| worker 脚本 | 未改动（threshold 仅在 head serve 命令），无 md5 同步需求 |
| 集群恢复 | head-first：systemctl start vllm-tp4-head.service → TCPStore 就绪 → 3×worker unit → healthcheck.timer |
| 容器 | rank0/1/2/3 全部 Up (healthy) |
| 自愈链 | 01: vllm-tp4-head.service active + vllm-healthcheck.timer active；02/03/04: vllm-tp4-worker.service active ×3 |
| health | 200 |
| KV | 6,042,089 tokens（6.04M，启动 profiling 方差内） |
| B12X | "Using 'B12X_MXFP4' Mxfp4 MoE backend" 在场 |
| dspark | spec_decode drafts 计数器随流量增长（实测 811），acceptance/drafts 指标在场 |
| 验证时刻 | 2026-08-22T07:33:21Z |

附注：02-04 worker 单元存在历史遗留的 "unit file changed on disk, daemon-reload" 警告（非本次操作引入，本次未改单元文件），运行态正常，建议后续维护窗口统一 daemon-reload。

---

## 五、结论与建议

1. **Route 1（调度器 threshold M 扩展）：否定，建议从组合策略中排除。** 1024→2048 无 PR 收益（4K -22.5%、16K -5.2%、32K/64K 持平），DE 归一持平。单流 PR 的优化杠杆不在调度器 chunk 维度。
2. **merged 桶方向收益上限需按真实流量重估**：单步内合并 hash ~16% @M≥64、dense ~2%；跨步累积才可能达聚合口径覆盖，但需暂存设计。尾路径策略应以此为输入。
3. **组合缓存方向吸引力下降**：真实 top-10 命中减半（hash 34.2%、dense 11.5%）。
4. **后续可选验证**：若需进一步确认 4096 档（本次按剂量-反应跳过）；DE C12 步效率名义 -13% 可用 ≥5 轮复核排除噪声。
5. **方法论沉淀**：路由类评估必须用真实多样化语料（GSM8K 8-shot 业务格式是现成的合格样本源），并区分「全语料聚合 / 等样本量 / 单步内」三口径——三者结论可差 5 倍以上。

## 附：产物清单

- 01:/tmp/_routea_work/routing_capture_gsm8k.jsonl（原始路由采集，896 records / 1.02M tokens）
- 01:/tmp/_routea_work/route_capture_gsm8k.py（采集脚本）、analyze_routing4/5/6.py（三口径分析）、bench_mixed_itl.py（混合 ITL 基准）
- 01:/tmp/_routea_work/pr_armA/B.json、de_armA/B.json、mixed_itl_armB.json、mixed_itl_armA2.json（A/B 原始数字）
- 01:<INSTALL_DIR>/scripts/start_tp4_head.sh.bak-thrab-20260822（threshold 变更审计留档）
