# W4A4（LuZ0.3.1）vs W4A16（B1 检查点）— 改动差异核对 + 测试结果逐项对比 + 覆盖矩阵审计

- **执行人**：雷克斯（Rex）· SRE 工程师（sre-engineer-1）
- **日期**：2026-08-23（本地）
- **性质**：纯只读审计 + 数据整理（本地 deliverables + SSH 只读集群），**未启动容器、未触碰 GPU、未修改任何生产资产**
- **对象**：当前生产 W4A4 形态 **LuZ0.3.1** vs 两天前检查点 **W4A16 B1 基线**（vLLM 0.26.1 fork，DeepSeek V4 Flash 0731 ckpt）
- **审计证据链**：本地 `deliverables/engineering-assurance/` 报告 + 原始 JSON/日志 + SSH 只读读取集群 `<INSTALL_DIR>/scripts/` 当前启动脚本与 `.bak-*` 序列、`<INSTALL_DIR>/backup/quality-gate/`、`/tmp/_luz031/logs/` 启动日志、`/tmp/_bench_luz031/official/benchmark_package_20260819/data/测试数据汇总.md`（官方 8/19 参考）
- **审计时刻集群状态**：生产停机态（`docker ps` 无 vllm-tp4-*/tp4-bench-*、8001 空闲），镜像 LuZ0.3.1（85f2149fad0b）在位

---

## 0. 术语与两基座定义

| 术语 | 定义 | 数据来源 |
|---|---|---|
| **W4A16 B1 基线** | 2026-08-23 凌晨 w4a4-ext 窗口实测的 W4A16 生产基线（FI 0.6.16 + W4A16 + thr4096 + util 0.80 + MTP n7 + batched 4096 + seqs 12）。**注意**：本审计采用「同窗同工具可比」口径的 w4a4-ext B1 实测值作为主基线，而非 `runbook-tp4-v1.5`（08-12）记录的更早固化值（util 0.65→0.72 时代、MTP n5、batched 4096、seqs 6）。两处 B1 概念需区分（详见 §1.3） | w4a4-ext-2026-08-23.md + `_w4a4_ext_assets/logs/b1_*` + b1_mem.log 启动参数 |
| **W4A4 LuZ0.3.1** | 当前生产终态 = W4A4 full（VLLM_MOE_W4A4=2）+ 池补丁（VLLM_B12X_SHARED_WRAPPER=1）+ FI 0.6.16 + thr4096 + util 0.82 + MTP n7 | luz031-deployment-2026-08-23.md + 当前 `start_tp4_head.sh` 实测 + `/tmp/_luz031/logs/luz031_startup.log` 启动参数 |
| **官方 8/19 参考** | 官方基准包 benchmark_package_20260819 的 8/19 复测原基座（W4A16 时代，seqs=16、batched 8240、util 0.82、MTP n5），用于 decode-only 官方口径对比 | `/tmp/_bench_luz031/official/benchmark_package_20260819/data/测试数据汇总.md` |

**关键口径警示**：LuZ0.3.1 验收（luz031-deployment）中「vs B1」对比用的是 w4a4-ext B1 臂（同窗同工具）；「vs 官方 8/19」对比用的是官方原基座（跨窗、形态差异大）。**两套 B1 参考不能混用**——任务书所述「runbook B1 固化 util 0.72」与 w4a4-ext B1 实测（util 0.80）不是同一配置。本审计在每张表标注所用基线口径。

---

## 1. 任务一：改动差异核对表（逐项、来源文件标注）

### 1.0 启动脚本 .bak 序列核对（SSH 只读实测）

`start_tp4_head.sh` 当前版本与关键 `.bak-*` 留档的 diff 实测结果：

| .bak 文件 | 与当前脚本关系 | diff 实测摘要 |
|---|---|---|
| `.bak-luz031-20260823`（02:54 生成） | = W4A16 基线快照（luz031 部署前） | 当前 vs 该文件差异：util 0.80→0.82、SERVE_CMD 加 plugin_a1 安装前缀、新增 4 行 W4A4 env（W4A4=2/MIN_M/CG）+ SHARED 0→1、新增 FI 0.6.16 挂载 2 行 |
| `.bak-fi016-20260823`（00:56 生成，fi016 注入前） | 与 `.bak-luz031` **内容一致**（diff 为空） | luz031 §2 实锤：fi016 注入的挂载行在 w4a4-ext 恢复时被误删，`.bak-luz031` 与 `.bak-fi016` 均不含 FI 0.6.16 挂载行 |
| `.bak-thr4096-20260822`（08-22 14:16） | = threshold 4096 采纳后、wsdedup 前 | 与 `.bak-fi016` 差异：thr 1024→4096、无 overlay 挂载行、无 SHARED env |
| `.bak-bt4096-20260818_055518`（08-18） | = batched 4096 早期（util 0.80 首次） | MTP n7 已在此出现；batched 8264；thr 1024 |
| `.bak-util72-20260818_024439` | = 08-18 更早 | util 0.65、MTP n5、batched 4096、thr 1024、capture 1..64 |
| `.bak-ncclB1`（08-17） | NCCL MAX_CHANNELS 16 时代 | MAX_NCHANNELS=16（后被 8/18 调为 4） |

**结论：任务书所述「B1 固化 util 0.72、block_size=64」未在任一 .bak/runbook 中直接出现**——runbook v1.5 记录为 util 0.65（R11 固化）；w4a4-ext B1 实测 util 0.80。「0.72」可能来自中间某次未留档临时值或任务书笔误，本审计以实测为准并标注此差异。

### 1.1 env / 启动参数逐项核对

| # | 参数项 | 旧值（W4A16 B1） | 新值（LuZ0.3.1） | 变更载体/来源文件 | 影响面 | 已知代价/收益 |
|---|---|---|---|---|---|---|
| 1 | `VLLM_MOE_W4A4` | 无/0 | **2**（full） | 当前 head 脚本 L150；.bak-luz031 无此行 | MoE 权重路径切换到 W4A4 量化执行（MXFP4 payload） | 收益：prefill 吞吐 +6~15%；代价：decode C12 -8%（W4A4 full 已知带） |
| 2 | `VLLM_MOE_W4A4_MIN_M` | 无 | **3072** | 当前 head L151 | full 模式下不生效（hybrid 语义残留，B2 同款） | 无（文档标注残留） |
| 3 | `VLLM_MOE_W4A4_CG` | 无 | **1** | 当前 head L152 | 启用 cudagraph 下 W4A4 路径 | 与 static workspace 相关（P2 报告称 CG=1 static 路径慢，后被 thr4096 部分化解） |
| 4 | `VLLM_B12X_SHARED_WRAPPER` | **0**（.bak-luz031 L147） | **1** | 当前 head L153 | 几何键共享池，跨层 wrapper 去重 | 收益：weight 68.15→45.32 GiB（省 22.83 GiB）；代价：无（池性能零代价） |
| 5 | `--long-prefill-token-threshold` | 4096（B1 已是 4096；更早 1024） | **4096**（不变） | 当前 L57；.bak-thr4096 确认 | — | 无变更（threshold 4096 于 08-22 采纳，B1 时已在位） |
| 6 | `--max-num-batched-tokens` | 4096（B1 实测） | **4096**（不变） | b1_mem.log / luz031_startup.log | — | 无变更（官方 8/19 参考为 8240，但 B1 同窗为 4096） |
| 7 | `--gpu-memory-utilization` | **0.80**（B1 实测 / .bak-luz031） | **0.82** | 当前 L59；checker L86 KEY_PARAMS | KV 容量 + 显存预算 | 收益：KV 5.50→5.73M（回补 +0.23M）；**回补不达合成预期 +0.44M，记录不阻断**（luz031 §3.4） |
| 8 | `--max-num-seqs` | **12**（B1 实测 b1_mem.log） | **12**（实测确认，L53 + checker） | 当前 L53；luz031_startup.log | — | 无变更（任务书「16→当前值」：官方 8/19 参考是 16，但 B1 同窗已是 12） |
| 9 | `--speculative-config` | **dspark n=7**（B1 已是 n7） | **dspark n=7**（不变） | b1_mem.log；当前 L61 | — | 无变更（n7 自 .bak-bt4096 08-18 起；官方 8/19 参考是 n5） |
| 10 | flashinfer 版本 | **0.6.16**（B1 臂同窗；fi016 已注入） | **0.6.16**（luz031 补回挂载） | 当前 L176-177 挂载行；fi016-replacement | 内核树版本 | 收益：三门通过；**风险事件**：03:01-05:36 曾被误回滚至 0.6.15（luz031 §2），luz031 已补回 |
| 11 | plugin_a1 安装 | **无**（SERVE_CMD 直接 vllm serve） | **新增**（`pip install /tmp/plugin_a1_install` 前缀） | 当前 L50 SERVE_CMD | 注入 W4A4B12xExperts 插件 | 收益：W4A4 full 路径激活；代价：容器启动多一次 pip install（秒级） |
| 12 | 池补丁 overlay | 无（B1 时 overlay-wsdedup 挂载行在，env=0 零行为） | **保留**（flashinfer_b12x_moe.py bind-mount + SHARED=1） | 当前 L172 overlay 挂载；wsdedup-l3-combo | 池化 wrapper 路径 | 见 #4 |
| 13 | cudagraph capture | capture 1..96（sizes 16 档，B1 同款） | capture 1..96（sizes 16 档） | 当前 L96-97；b1_mem.log | — | 无变更 |
| 14 | NCCL env | `MIN=4/MAX=4`（B1 同款，.bak-bt4096 起） | **MIN=4/MAX=4**（不变） | 当前 L114/L119 | — | 无变更（MAX 16→4 于 08-18，非本次变更） |
| 15 | `kv_cache_dtype` | nvfp4_ds_mla | **nvfp4_ds_mla**（不变） | b1_mem.log；luz031_startup.log | — | 无变更 |
| 16 | 其他 R11 参数（max-model-len 600000 / breakable cudagraph / prefix caching / autotune / VLLM_USE_B12X_MOE 等） | 全部在位 | **全部在位**（逐字保留） | 当前 ENV_ARGS vs .bak-luz031 | — | 无变更（diff 仅上述差异） |

### 1.2 权重 / 检查点差异

| # | 项 | 旧值（W4A16 B1） | 新值（LuZ0.3.1） | 来源 | 影响面 | 已知代价/收益 |
|---|---|---|---|---|---|---|
| 17 | weight（MoE 权重显存） | **40.5 GiB** | **45.32 GiB**（B2 精确一致，池生效） | w4a4-ext §2.5；luz031 §3.4；startup log Actual usage 45.32 GiB | 权重驻显存体积 +4.82 GiB | **注意**：W4A4 并非省显存，反而 +4.82 GiB（E4M3 scales 4.3 + 1 共享 wrapper 0.54；相比未池化 68.15 已省 22.83）。收益体现在 prefill 计算效率而非权重体积 |
| 18 | KV tokens（准入容量） | **6.037M**（6,037,164） | **5,730,000**（5.73M） | b1_mem.log；luz031_startup.log；bprime-window 佐证（6.02M） | KV 容量 -5.1% | 已知代价：KV vs W4A16 -4.5~-5%；门 ≥5.7M 通过。回补 util 0.82 后 +0.23M 不达合成预期 |
| 19 | KV 内存预算 | 53.53 GiB | 50.81 GiB | b1_mem.log / luz031_startup.log | — | 与 util 0.82 + weight +4.82 联动 |

> 注：weight 45.32 vs 40.5 的直接比较有口径差异——w4a4-ext B1 臂 40.5 GiB 是 W4A16（nvfp4/MXFP4 权重）；LuZ0.3.1 45.32 GiB 是 W4A4 full 执行 + 池化。任务书「45.32 vs 40.5（-0731 MXFP4 vs nvfp4 差异）」表述需修正为「W4A4 full 执行下权重显存反而更大」。

### 1.3 镜像 / overlay / 脚本差异

| # | 项 | 旧值（W4A16 B1） | 新值（LuZ0.3.1） | 来源 | 影响面 |
|---|---|---|---|---|---|
| 20 | 镜像 | 0.2.1-v026.0（34.2GB，digest e100ddad568a） | **LuZ0.3.1**（34.4GB，digest 85f2149fad0b，自包含 bake：FI 0.6.16 树 + ws-dedup overlay + 池化插件）+ 基座锚点 LuZ0.3.1-base | luz031 §5；SSH docker images 实测 | 恢复/分发形态 |
| 21 | 启动脚本 | start_tp4_head.sh（.bak-luz031 快照，无插件前缀、无 W4A4 env、无 FI 挂载） | 当前 start_tp4_head.sh（+plugin_a1 前缀 + W4A4 env×4 + FI 0.6.16 挂载×2 + util 0.82） | 当前 vs .bak-luz031 diff 实测 | 生产配置 |
| 22 | checker KEY_PARAMS | util 0.80（.bak-luz031 checker） | util 0.82 + seqs 12 + batched 4096（当前 checker L86） | SSH 实测 | 启动自检 |

### 1.4 差异项汇总

- **逐项差异总计：22 项核对**，其中**实际变更 9 项**（#1、2、3、4、7、10、11、12、22），**无变更/已在 B1 在位 13 项**（#5、6、8、9、13、14、15、16、17/18/19 数值联动、20、21 结构性）。
- **任务书预判修正**：
  - 「max-num-seqs 16→当前值」→ 实测 B1 已为 12，官方 8/19 参考为 16（不同参考口径）。
  - 「spec n5→n7？B1 是否已是 n7」→ **B1 已是 n7**（08-18 起），非本次变更。
  - 「util 0.72→0.82」→ runbook/实测未见 0.72；B1 实测 0.80 → 0.82。
  - 「block_size=64」→ 未在任何启动脚本/runbook 直接出现（KV block 逻辑见 e4-kv-footprint 但无显式 64 参数），标注为「未见证据」。

### 1.5 影响性能的关键差异 Top 5（按影响排序）

| 排名 | 变更 | 影响方向 | 量级证据 |
|---|---|---|---|
| 1 | **W4A4 full（VLLM_MOE_W4A4=2）** | prefill +6~15%（PR 四档）、并发 +11~13%；decode C12 -8% | w4a4-ext / luz031-deployment 实测 |
| 2 | **池补丁（SHARED=1）** | 使 W4A4 full 可运行（weight 68.15→45.32），KV 恢复 | wsdedup-l3 / luz031 实测 |
| 3 | **FI 0.6.16**（补回挂载） | 内核版本一致性（0.6.15 误回滚风险），性能带内 | fi016-replacement；luz031 §2 |
| 4 | **util 0.80→0.82** | KV +0.23M 回补（不达合成预期），总容量 5.73M | luz031 §3.4 |
| 5 | **plugin_a1 安装前缀** | W4A4 执行路径激活（部署结构变更） | 脚本 diff 实测 |

---

## 2. 任务二：测试结果逐项对比总表

### 2.1 PR 单流 prefill 四档（3 轮中位，tok/s）

| 档位 | W4A16 B1 | LuZ0.3.1 | Δ vs B1 | 来源 |
|---|---|---|---|---|
| 4K（8.2K tok） | 2769（2768.5） | 2950.5 | **+6.6%** | b1_panorama.json / luz031 §3.1 |
| 16K（32.8K） | 2770（2769.9） | 2943.6 | **+6.3%** | 同上 |
| 32K（65.5K） | 2565（2565.0） | 2834.2 | **+10.5%** | 同上 |
| 64K（131K） | 2215（2215.0） | 2550.0 | **+15.1%** | 同上 |

> 任务书「+6.6~15.1%」验证：实测 +6.3%（16K）~ +15.1%（64K），方向一致，量级略修正。

### 2.2 并发聚合（4K 档，3 轮中位，tok/s）

| 并发 | W4A16 B1 | LuZ0.3.1 | Δ vs B1 | 来源 |
|---|---|---|---|---|
| C6 | 2744（b1_conc6 agg_tps_med） | 3057 | **+11.4%** | b1_conc6.json / luz031 §3.2 |
| C12 | 2737（b1_conc12 agg_tps_med） | 3056 | **+11.6%** | b1_conc12.json / luz031 §3.2 |
| C6 med TTFT | 11.65s | 10.47s | -10.1% | b1_conc6.json / luz031 |
| C12 med TTFT | 20.58s | 18.39s | -10.6% | b1_conc12.json / luz031 |

> **W4A16 B1 并发数据存在**（w4a4-ext B1 臂，非缺失）。

### 2.3 DE（接受率归一 step_eff，4 轮中位；含 tokens/step 与步时）

| 指标 | W4A16 B1 | LuZ0.3.1 | Δ | 来源 |
|---|---|---|---|---|
| C1 step_eff | 17.7（tput_sum med 79.6 / tokens/step med 4.491） | 18.2 | **+2.8%** | b1_de.json / luz031 §3.3 |
| C12 step_eff | 87.2（tput_sum med 360.4 / tokens/step med 4.135） | 80.2 | **-8.0%** | 同上 |
| C1 接受率（tokens/step） | 4.491 | ~4.7（step_eff 反推，luz031 未单列 raw） | 中性 | b1_de.json |
| C12 接受率 | 4.135 | 中性 | 中性 | b1_de.json |
| C12 步时（ms/step 派生） | ~（w4a4-ext B2 口径 85.1 → LuZ 80.2） | — | C12 落 W4A4 full 代价带（-6~-9% 口径并存） | luz031 §3.3 / bench-regression-attribution |

> 任务书「DE C1/C12：18.2/80.2 vs 17.7/87.2（含 tokens/step 接受率、步时）」验证成立。注意 **LuZ0.3.1 的 DE 报告中未单列 tokens/step 与 ms/step 原始值**（luz031-deployment 只报 step_eff），thr2048-retest 对 thr4096 臂有完整 DE 明细（C1 step_eff 18.95、tokens/step 5.225、ms/step 52.9；C12 83.6 / 4.495 / 12.0）。

### 2.4 官方口径 decode-only（Session A，vs 官方 8/19 参考）

| 场景 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 | 来源 |
|---|---|---|---|---|---|---|
| C1 | 97.1 \| 124.0 | 73.9 \| 136.5 | -23.9% | +10.1% | 🔴 回退 | luz031-bench §2.1 |
| C4 | 218.0 \| 233.7 | 186.7 \| 218.8 | -14.4% | -6.4% | 🔴 回退 | 同上 |
| C8 | 286.3 \| 302.9 | 274.5 \| 348.3 | -4.1% | +15.0% | ⚠ 持平 | 同上 |
| C12 | 342.8 \| 358.2 | 349.3 \| 397.5 | +1.9% | +11.0% | ✅ 提升 | 同上 |
| Agent 平均 | 81.3 \| 84.6 | 70.4 \| 76.5 | -13.4% | -9.6% | 🔴 回退 | 同上 |
| S1 fox p512 | 97.1 \| 124.0 | 77.8 \| 131.1 | -19.9% | +5.7% | 🔴 回退 | 同上 |

> 形态差异（官方 seqs16/batched8240/MTP n5 vs LuZ seqs12/batched4096/MTP n7/W4A4）已在对比表并列标注，Δ 为综合差异。归因详见 bench-regression-attribution（不稳定+接受率方差为主、C12 步时 -8% 为辅）。

### 2.5 质量门 / needle（两基座覆盖）

| 门 | W4A16 B1 | LuZ0.3.1 | 两基座是否有数据 |
|---|---|---|---|
| golden 质量门（稳定 4 prompt greedy 逐字一致） | **有参考快照**：`reference-b1w4a16-fi016-20260823.json`（=b1_greedy_ref.json，W4A16 B1 捕获）+ wsdedup M1 golden（m1_golden.json）。B1 自身以 golden 门被引用为参考基线；M1（W4A16）门测 4/4（wsdedup-l3 §2.4） | **4/4 exact match**（own_stable 4/4），vs B1 参考 | ✅ **两基座都有**（B1 为参考侧 + M1 门测 4/4；LuZ 为被测侧 4/4） |
| needle 64K | **有**：routea-p2（08-21，W4A16 基线）needle_A 64K×3+128K×2 = 3/5；fi016（08-23 W4A16）64K 3/3 + 128K 2/2；wsdedup M1 2/5（统计口径波动大） | **3/3 PASS**（mid/late/late；128K 加测 1/2，late 位已知抖动） | ✅ **两基座都有**（B1 时代多窗口有 needle 数据，但口径/次数不一） |

> **覆盖核对**：任务书「golden 4/4、needle 64K 3/3（两基座是否都有？B1 时代有无 golden/needle）」→ **B1 时代有 golden 参考与 needle 数据**，但 B1 没有「官方口径存档的质量门 4/4 门测记录」（B1 是参考侧而非被测侧；wsdedup M1 门测 4/4 是最接近的 B1 侧质量门）。

### 2.6 资源门（weight / KV / memory）

| 指标 | W4A16 B1 | LuZ0.3.1 | Δ | 来源 |
|---|---|---|---|---|
| weight | 40.5 GiB | 45.32 GiB | **+11.9%** | w4a4-ext §2.5 / luz031 §3.4 |
| KV tokens | 6,037,164 | 5,730,000 | **-5.1%** | b1_mem.log / luz031_startup.log |
| KV 内存 | 53.53 GiB | 50.81 GiB | -5.1% | b1_mem.log / luz031_startup.log |
| peak 激活 | 2.03 GiB | 2.03 GiB | 0 | w4a4-ext §2.5 |
| CUDAGraph | ~0.7 GiB | 1.4 GiB（luz031 startup log） | +0.7 | luz031_startup.log |

---

## 3. 覆盖矩阵与缺口清单

### 3.1 覆盖矩阵（哪些口径两基座都有 / 单边 / 完全没测）

| 测试口径 | W4A16 B1 | LuZ0.3.1 | 覆盖判定 |
|---|---|---|---|
| PR 单流四档（panorama） | ✅ w4a4-ext B1 实测 | ✅ luz031 验收实测 | ✅ **两基座都有（同工具同窗可比）** |
| 并发 C6/C12（prefill 聚合） | ✅ w4a4-ext B1 实测 | ✅ luz031 验收实测 | ✅ **两基座都有** |
| DE C1/C12（step_eff 接受率归一） | ✅ w4a4-ext B1 实测 | ✅ luz031 验收实测 | ✅ **两基座都有**（但 LuZ 侧缺 tokens/step 与 ms/step 原始明细，thr2048-retest 补了 thr4096 臂完整值） |
| 官方 decode-only（Session A） | ⚠️ **官方口径对 W4A16 B1 未测过**（w4a4-ext B1 用的是自研 PR/并发/DE 工具；08-21 e2e-baseline 用官方包但那是 W4A16 时代测的官方口径、与 LuZ 的 Session A 跨窗不同形态） | ✅ Session A 实测 | ⚠️ **单边**（LuZ 有官方 decode-only，W4A16 B1 无同窗官方 decode-only；仅有 08-21 跨窗官方包数据可参考） |
| golden 质量门 | ⚠️ B1 为**参考侧**（reference-b1w4a16 + m1_golden）；B1 自身门测 = wsdedup M1 4/4 | ✅ 4/4 | ✅ **两基座都有**（B1 是参考侧 + M1 门测） |
| needle 64K | ✅ 多窗口（routea-p2 3/5、fi016 3/3、wsdedup M1 2/5） | ✅ 3/3 | ✅ **两基座都有**（B1 侧口径不统一，统计波动大） |
| 资源门（weight/KV/mem） | ✅ b1_mem.log / w4a4-ext | ✅ luz031 startup log | ✅ **两基座都有** |
| P0 拆账（三池 µs） | ⚠️ 无 W4A16 直接实测（fi017 推算带） | ✅ Session B 实测 | ⚠️ **单边**（LuZ 有实测，W4A16 只有 fi017 推算带） |
| threshold 变体（2048/4096/8192） | ✅ B1/B2/B3（w4a4-ext）+ 08-22 复测 | ✅ thr2048-retest（A1/B1/A2/B2） | ✅ **两基座都有**（thr2048 仅对 LuZ 测，W4A16 无 thr2048 对照） |
| 回归日志门（error/traceback 0） | ⚠️ B1 侧未见明确归档 | ✅ luz031 0 条 | ⚠️ **单边** |

### 3.2 明确覆盖缺口（需标注不全项与补齐方案）

| # | 缺口 | 状态 | 补齐方案 |
|---|---|---|---|
| G1 | **官方 decode-only 对 W4A16 B1 未在同窗测过**：Session A 只有 LuZ0.3.1 一侧；W4A16 B1 与官方 8/19 的官方口径对比只能引用 08-21 e2e-baseline（跨窗、seqs16/8240、形态与 B1 不同） | 单边 | 如需官方口径 W4A16↔W4A4 同窗 A/B：在 LuZ0.3.1 克隆镜像旁路下，用 `VLLM_MOE_W4A4=0` + 官方包跑一次 W4A16 对照（预算 +2h），或接受 08-21 数据作为参考 |
| G2 | **B1 时代 golden 门缺少「官方口径存档记录」**：B1 是 golden 参考侧；wsdedup M1 门测 4/4 是最近的 B1 侧质量门，但不在官方基准包口径下 | 部分 | 将 `reference-b1w4a16-fi016-20260823.json` 固化为 golden 基线资产（已在 quality-gate backup），后续任何变更统一用 quality_gate.py 比对此参考 |
| G3 | **needle 两基座口径不统一**：B1 侧多窗口口径不一（routea 3/5、fi016 3/3、wsdedup M1 2/5 统计波动）；LuZ 侧 3/3 为 64K 抽验 | 部分 | 统一 needle 口径（64K×3 固定 mid/late/late，128K×2 记录项）入 runbook；A6 已建议降级为 smoke 口径 |
| G4 | **LuZ0.3.1 DE 缺 tokens/step 与 ms/step 原始明细**：luz031-deployment 只报 step_eff | 部分 | thr2048-retest 已对 thr4096 臂补测完整 DE 明细（tokens/step 5.225/4.495、ms/step 52.9/12.0）；正式报告中引用 thr2048-retest §2.2 补齐 |
| G5 | **P0 拆账无 W4A16 直接实测**（fi017 推算带 vs LuZ Session B 实测） | 单边 | 如需精确 W4A16 池拆账，可对 LuZ0.3.1-base 镜像跑 W4A16 形态 P0（+30-60min），或接受推算带对比 |
| G6 | **threshold 2048 无 W4A16 对照**：thr2048 只在 LuZ0.3.1 测（A1/A2）；W4A16 侧只有 1024/4096/8192 | 部分 | thr2048 判决已基于 LuZ 形态（PR 恶化 + 无 DE/并发收益），W4A16 侧可视为继承阈值结论（threshold 与 MoE 后端正交，风险低） |
| G7 | **回归日志门（0 error/traceback）无 B1 侧归档** | 单边 | B1 时代未归档此门；建议后续两基座对比时统一加日志门 |

### 3.3 覆盖总结

- **两基座都有的口径**：PR 四档、并发 C6/C12、DE step_eff、golden 质量门、needle、资源门。
- **单边口径**：官方 decode-only（仅 LuZ 同窗）、P0 拆账（仅 LuZ 实测）、回归日志门（仅 LuZ）。
- **完全没测**：无完全没测的核心性能口径；缺的主要是「官方 decode-only 的 W4A16 B1 同窗对照」与「W4A16 P0 实测」。

---

## 4. 结论汇报摘要

1. **差异项数**：逐项核对 22 项；实际变更 9 项（W4A4 full、MIN_M、CG、SHARED、util 0.82、FI 0.6.16 补回、plugin_a1 前缀、池 overlay 保留、checker 同步），其余 13 项为「B1 已在该值」无变更。
2. **关键差异（影响性能 Top5）**：① W4A4 full（prefill +6~15%/并发 +11~13%/decode C12 -8%）；② 池补丁 SHARED=1（weight 68.15→45.32 使形态可运行）；③ FI 0.6.16 补回（内核一致性，曾误回滚）；④ util 0.80→0.82（KV +0.23M 回补）；⑤ plugin_a1 安装前缀（W4A4 执行路径激活）。
3. **覆盖缺口清单**：G1 官方 decode-only 无 W4A16 同窗对照（最核心缺口）；G2 golden 门 B1 侧缺官方口径存档（参考侧存在）；G3 needle 口径不统一；G4 LuZ DE 缺 tokens/step+ms/step 明细（thr2048-retest 已补）；G5 P0 无 W4A16 实测；G6 thr2048 无 W4A16 对照（风险低）；G7 日志门无 B1 归档。

---

## 5. 证据索引

| 数据 | 来源 |
|---|---|
| 当前生产脚本与 .bak 序列 diff | SSH 只读 `node01:<INSTALL_DIR>/scripts/start_tp4_head.sh` + `.bak-*` + worker（02 节点） |
| LuZ0.3.1 启动参数/KV/weight | SSH `node01:/tmp/_luz031/logs/luz031_startup.log` |
| W4A16 B1 启动参数/KV | `_w4a4_ext_assets/logs/b1_mem.log`（non-default args + KV 6,037,164 + KV mem 53.53 GiB） |
| W4A16 B1 PR/并发/DE | `_w4a4_ext_assets/logs/b1_panorama.json` / `b1_conc6.json` / `b1_conc12.json` / `b1_de.json` |
| W4A4 B2 PR/并发/DE（参考带） | `_w4a4_ext_assets/logs/b2_panorama.json` / `b2_conc6/12.json` / `b2_de.json` |
| LuZ0.3.1 验收 | `luz031-deployment-2026-08-23.md` / `_luz031_official_bench/data/luz031_vs_official_*.md` |
| 官方 8/19 参考 | SSH `node01:/tmp/_bench_luz031/official/benchmark_package_20260819/data/测试数据汇总.md` |
| B1 golden 参考 | SSH `node01:<INSTALL_DIR>/backup/quality-gate/reference-b1w4a16-fi016-20260823.json` + `_w4a4_ext_assets/logs/b1_greedy_ref.json` |
| needle（B1 侧） | `routea-tp4-p2-2026-08-21/needle_A.json`（本地）+ fi016-replacement §（64K 3/3） |
| W4A16 官方 decode-only（08-21 跨窗） | `e2e-baseline-prod-2026-08-21.md` + `e2e-baseline-results/results/*.log` |
| thr2048 变体 | `luz031-thr2048-retest-2026-08-23.md` |
| wsdedup L3（池/W4A4 翻案） | `wsdedup-l3-combo-2026-08-23.md` + `_wsdedup_l3_assets/` |
| bprime（b′ No-Go、golden/needle 佐证） | `bprime-window-2026-08-23.md` |
| decode 归因 | `bench-regression-attribution-2026-08-23.md` |
| P0 拆账 | `_luz031_official_bench/data/p0/p0_accounting_data.md` |

---

*纪律遵守：纯只读审计（本地 + SSH 只读），未启动容器、未触碰 GPU、未修改生产资产；所有差异以实际脚本/文件/日志核对而非仅凭报告；两套 B1 口径明确区分；覆盖缺口如实标注。*

*本审计由工程保障团队（SRE）生成，供工程总监汇总复核。*
