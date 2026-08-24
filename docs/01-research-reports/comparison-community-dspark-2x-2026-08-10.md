# 社区 DSpark 2×DGX Spark 方案 vs 我方集群方案 — 逐项对比与可借鉴项分析

**日期**：2026-08-10
**作者**：Archi（系统架构师）· 工程保障团队
**性质**：架构分析 / 对比报告（只读，未执行任何变更）
**数据源**：
- 社区：`MiaAI-Lab/DeepSeek-V4-Flash-DSpark-2x-DGX-Spark`（81 commits，README + 配方已在线核实，2026-08-10）
- 我方：`deliverables/engineering-assurance/` 多份报告（runbook-dspark-vllm-2026-08-06、benchmark-A/B-group-2026-08-08、benchmark-tp2-noll-subset-2026-08-09、report-nccl-tcp-firewall-2026-08-09、research-vllm-chunked-prefill-priority-2026-08-09、analysis-tp2-tp4-communication-2026-08-09、roce-optimization-summary-2026-08-09、audit-cluster-4node-2026-08-07 等）+ 启动脚本（start_head_v026r.sh / start_*_groupB.sh）

---

## 📌 TL;DR（执行摘要）

1. **性能总评：我方在关键单流指标上略优或持平**——131072/c1 prefill：我方 1702~1821（TP2 去 LL 口径 1702）vs 社区 1665（+2%~+9%）；decode c1：我方 71-80 vs 社区 65-74（+5%~+10%）。双方差距不大，属同代同栈水平。
2. **启动顺序：worker-first vs head-first 不矛盾，是 vLLM 版本差异**——社区 0.25 用 worker-first 规避 mp init 竞态；我方 0.26.1 已实证 head-first + TCPStore 轮询 + EngineCore 存活校验（8/8 修复）为正确顺序，且我方 8-01（0.25 时代）也用过 worker-first。**结论：各自版本各自对，结论不可跨版本迁移。**
3. **社区两个最大借鉴点：Regular CUDA graph（+28.6%）与 Keys-Concurrency-Patch**——两者分别直击我方"未测 regular/breakable"与"无文档化 dspark 并发补丁"两个空白，优先级最高。
4. **社区运维项"禁 earlyoom"应直接采纳**——我方有四机 OOM 历史（Exited 137），GB10 统一内存下 earlyoom 误杀 vLLM 的风险真实存在。
5. **我方独特优势：4 节点扩展性、twin 双 HCA 带宽余量、priority+threshold 调度（32768 分界 0.77→1.05）、网络加固（限 TCP+PFC+isolcpus）**——社区均为 2 节点裸直连、无调度保护、无网络优化。

---

## 🎯 核心结论卡片

| 维度 | 社区（2×DGX） | 我方（A/B 组，4 节点） | 判定 |
|------|--------------|----------------------|------|
| 单流 decode c1 | 65-74 t/s | 71-80 t/s | 我方略优 |
| 131K c1 prefill | 1665 t/s | 1702-1821 t/s | 我方略优 |
| 短 ctx prefill | 256→447 t/s（口径异） | 512→1121-1178 t/s | 口径不可直比，我方不弱 |
| 并发聚合 | c3=134.6 / c6=340.5（decode-only，口径异） | c5 短 ctx ~80-158；长 ctx UMA 饱和 | 需同口径验证 |
| 启动顺序 | worker-first（0.25） | head-first（0.26.1，8/8） | 版本相关，各自正确 |
| CUDA graph | regular 最优（+28.6%） | breakable（未测 regular） | **社区可借鉴** |
| dspark 并发 | Keys Patch（c16=315 agg） | 无文档化补丁，实测 c≤5 err=0 | **需评估** |
| earlyoom | 明确禁用 | 未处理（有 OOM 历史） | **建议采纳** |
| 调度保护 | 无（priority/threshold 未提） | priority+threshold+4096 | 我方优势 |
| 网络 | 数据面直连（无防火墙） | 限 TCP + PFC + isolcpus | 我方优势 |
| 上下文 | 1M | 600K | 社区领先 |

---

## 1. 逐项对比总表

| # | 维度 | 社区（Anemll 0.1.1 / vLLM 0.25） | 我方（Anemll 0.2.1 / vLLM 0.26.1.dev0） | 差异性质 | 谁优 |
|---|------|-----------------------------------|------------------------------------------|---------|------|
| 1 | 拓扑 | 2 节点，head <NODE_IP> + worker <NODE_IP> | 4 节点；A 组(01/02) TP2 + B 组(03/04) 待成环 | 规模/冗余 | 我方（扩展性） |
| 2 | 互联 | 单 HCA `rocep1s0f1`（单逻辑口） | twin 双 HCA `rocep1s0f1,roceP2p1s0f1`（01↔02 module1，02↔04 module0） | 带宽余量 | 我方（188G vs ~111G） |
| 3 | 数据面 | NCCL_SOCKET_IFNAME=数据口，无防火墙 | NCCL 控制 TCP 限对端 IP + 25000；其余 TCP DROP | 安全/语义 | 我方（安全） |
| 4 | 管理口 | GLOO 与数据面一致 | GLOO_SOCKET_IFNAME=enP7s7（管理口） | 路径隔离 | 我方 |
| 5 | 启动顺序 | **worker-first**（README 明确，避免 mp init 竞态） | **head-first**（TCPStore 就绪→worker，EngineCore pgrep + worker 存活校验，8/8） | 版本差异 | 各自版本正确 |
| 6 | 镜像 | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（vLLM 0.25） | `anemll/dspark-vllm-gx10:0.2.1-v026.0`（vLLM 0.26.1.dev0） | 版本 | 我方（更新，功能更全） |
| 7 | tokenizer | `--tokenizer-mode deepseek_v4` + encoding_dsv4.py + 4 档 thinking | 未显式设置；用 `--tool-call-parser deepseek_v4` / `--reasoning-parser deepseek_v4` | 可能不等价 | 需核实 |
| 8 | spec 配置 | dspark 5-token probabilistic | dspark 5-token probabilistic + 动态K [[1,1,5],[2,4,4],[5,6,3]] | 增强 | 我方 |
| 9 | dspark 并发 | Keys-Concurrency-Patch（Patch 2b，ragged query_start_loc 不依赖 num_rejected_tokens_gpu） | 无文档化补丁；实测 c≤5 err=0（486 样本） | 正确性边界 | 需评估 |
| 10 | KV cache | nvfp4_ds_mla；18.08GiB/2,493,464 tok ≈7.6KB/tok（padded 584B） | nvfp4_ds_mla；13.61GiB/2.13M tok ≈6.7KB/tok | 布局/口径 | 基本同栈 |
| 11 | max_model_len | 1048576（1M，满池并发 2.38×） | 600000 | 上限 | 社区（1M） |
| 12 | max_num_seqs | 6 | 6 | 同 | 平 |
| 13 | batched tokens | 8192 | 4096 | 调度 trade-off | 分场景 |
| 14 | chunked prefill | 显式 `--enable-chunked-prefill` | V1 默认启用（未显式） | 等效 | 平 |
| 15 | async scheduling | 显式 `--async-scheduling` | V1 默认（未显式） | 等效 | 平 |
| 16 | 调度保护 | 未提及 priority / long-prefill-threshold | `--scheduling-policy priority` + `--long-prefill-token-threshold 2048` | **我方核心优势** | 我方 |
| 17 | partial prefill | 未提及 | 镜像不支持（arg_utils 硬抛 NotImplementedError，参数已移除） | 我方受限 | 社区（若有） |
| 18 | CUDA graph | `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（regular 最优，C1 +28.6%） | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（breakable），`--max-cudagraph-capture-size 24` | **社区可借鉴** | 社区（待我方验证） |
| 19 | flashinfer sampler | `VLLM_USE_FLASHINFER_SAMPLER=1` | 同 | 同 | 平 |
| 20 | b12x MoE | `VLLM_USE_B12X_MOE=1` + `flashinfer_b12x` | 同 + `VLLM_TRITON_MLA_SPARSE=1` + tilelang 两档 patch | 增强 | 我方 |
| 21 | 网络优化 | 无（MTU/GID 基础） | MTU 9000 + QoS PFC(P3/P5) + isolcpus=16-19 四台 + rp_filter | **我方优势** | 我方 |
| 22 | earlyoom | **明确禁用** | 未处理 | **社区可借鉴** | 社区 |
| 23 | 镜像一致性 | 双节点手动 pull 同一 tag | 双机同 digest（0.2.1-v026.0 9ea563a7）+ 本地 registry | 同 | 我方（更可复现） |
| 24 | HF cache | 离线 HF cache 完整性强调 | HF_HUB_OFFLINE=1 + 持久卷 + 本地 registry | 同 | 平 |
| 25 | 1M retrieval benchmark | 明确不做完整 1M retrieval | 未做完整 1M（600K 上限） | 同 | 平 |
| 26 | 监控 | 未强调 | Prometheus/Grafana/DCGM/alertmanager 全套 | 运维 | 我方 |

---

## 2. 拓扑与并行（重点）

### 2.1 2 节点 vs 4 节点（A/B 组）

| 维度 | 社区（2 节点） | 我方（4 节点） |
|------|--------------|---------------|
| 规模 | 固定 TP2 单副本 | A 组（01+02）TP2 生产 + B 组（03+04）TP2 独立/待成环；成环后可 TP4（512GB 统一内存池） |
| 冗余 | 单副本无故障域 | A/B 双副本，可互为 fallback |
| 扩展 | 需重新布线/买交换机 | 链式 04—02—01 已具备 200G 双链路；TP4 需补 03 接线 + ring 闭环 |
| 代价 | 简单 | 编排复杂度上升；TP4 ring 2 跳延迟（每层 all-reduce 58→116µs） |

**架构判断**：我方 4 节点形态是**超集**——单副本时与社区等价（TP2 双节点直连），多副本/TP4 提供额外内存池。社区方案是 2 节点的"最小可行配方"，我方是"多副本 + 可演进"形态。**TP4 的真实价值在 512GB 内存池而非速度**（analysis-tp2-tp4-communication 已定论：200G 带宽不是 TP4 瓶颈，2 跳 ring 延迟反而劣化）。

### 2.2 单 HCA vs twin 双 HCA

- 社区：`NCCL_IB_HCA=rocep1s0f1`（单逻辑口），`NCCL_SOCKET_IFNAME=enp1s0f1np1`（单口）。
- 我方：`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`（twin 双逻辑口，01↔02 走 module1 的 136/137 双子网；02↔04 走 module0），`NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1`。

**分析**：twin 把单物理链路的两个逻辑通道并行化（实测 twin 并行带宽 188G vs 单口 ~111G）。decode 阶段链路仅用 ~2%（MLA 压缩后通信极低），twin 收益集中在 **prefill**（131K 全量 prefill 单批 47GB 通信）。社区 decode 导向配方单口足够；我方有长 ctx prefill 与未来 TP4 需求，twin 是正确的冗余+带宽投资。

### 2.3 head-first vs worker-first —— 谁对？（核心差异，需讲透）

**结论：不是"谁对谁错"，而是 vLLM 版本差异导致的两种正确顺序。**

| 项 | 社区（0.25） | 我方（0.26.1） |
|----|-------------|---------------|
| 顺序 | worker 先起，head 后起 | head 先起（TCPStore 就绪→worker 再起） |
| README 理由 | "worker-first startup avoids a race during multi-node mp initialization" | head 就绪 = pgrep EngineCore + worker 存活校验（8/8 死锁修复） |
| 我方历史证据 | 8-01 时代（0.1.1/0.25）我方**也**用 worker-first 且"实证必要性"（research-dgx-community-cluster 08-02 §2.1） | 升级 0.26.1 后 worker-first 4 次 3 挂（NCCL 卡死 H1：rank0 才创建 TCPStore）；head-first + 轮询 25000 → 3/3 成功；再加固为 8/8 |

**根因链**：
- **0.25 mp 执行器**：head 的 coordinator 会去连接远端 worker 的 mp transport；若 worker 未先监听，head 侧会遇连接竞态 → **worker-first** 让远端 listener 先就位，规避该竞态。
- **0.26.1 mp 执行器 + torch.distributed NCCL**：rank0（head）在 `init_process_group` 时**创建 TCPStore(25000)**，worker 先启必因 store 未建而失败；head 先启 + 轮询 store 就绪 + 确认 EngineCore 进程存活再拉 worker，是确定性的正确顺序。我方还有 H2（NVIDIA #366127 双 Spark GB10 系统性死锁）与 H3（IPv6 污染）两个历史根因叠加，故 head-first + 存活校验是综合最优。

**结论与建议**：
1. **我方 8/8 修复与社区 worker-first 不矛盾**——两者针对不同版本的竞态源。若有人建议"照社区改 worker-first"，我方应拒绝（在 0.26.1 会复现 NCCL 卡死）。
2. **启动顺序结论不可跨版本迁移**（runbook 坑位 #9 同源）：升级 vLLM 或回退镜像后必须重验启动顺序。
3. 建议在 runbook 补一条：worker-first 是 0.25 语义，head-first 是 0.26.1 语义；若未来升级 0.27+ 需按新版本 mp 行为重测。

---

## 3. 镜像 / 版本

| 维度 | 社区 0.1.1（vLLM 0.25） | 我方 0.2.1-v026.0（vLLM 0.26.1.dev0） |
|------|-------------------------|---------------------------------------|
| 内核/特性基线 | DSpark/NVFP4 DS-MLA/b12x 内置；kill-switch 部分未注册（设置 = no-op 警告） | 更全；NCCL 2.30.7 前插；tilelang 两档 patch 入镜像；`VLLM_TRITON_MLA_SPARSE=1` |
| **并发 partial prefill** | 未提及（0.25 时代参数） | **0.26.1 镜像不支持**：`arg_utils._check_feature_supported()` 硬抛 NotImplementedError，`--max-num-partial-prefills`/`--max-long-partial-prefills` 参数已移除（08-09 实测） |
| spec 动态K | 无（社区固定 5-token） | 动态K [[1,1,5],[2,4,4],[5,6,3]]（c10 +36% 历史收益） |
| 可复现性 | 手动 pull 单 tag | 本地 registry + digest 语义 + 双机同 digest |

**版本差异影响**：
1. **并发 partial prefill 缺失**是我方 0.26.1 的硬约束——社区 0.25 若可用该特性（未证实），则我方在"长 prefill 并发推进"上少一个原生手段，只能靠 `long-prefill-token-threshold` + `priority` 间接补偿（已生效，见 §5）。
2. **vLLM 0.26.1.dev0 是 dev 构建**，相对社区 0.25 稳定版存在偏差风险；但我方已通过 486 样本 err=0 + GSM8K 95.0% 实证稳定。
3. 社区 kill-switch 大量为 no-op（0.1.1 不注册）——我方 0.2.1 对这些变量有实际语义，配置时需区分。

---

## 4. DSpark 并发：Keys-Concurrency-Patch 评估

### 4.1 社区补丁是什么

- 来源：`drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash`，commit `7e4d94bb...`（Patch 2b）。
- **核心修复**：ragged `query_start_loc` 检测不再依赖 `num_rejected_tokens_gpu`（旧路径在高并发混合 prefill/decode batch 下会算错 start 位置 → 结果错乱/崩溃）。
- 贡献面：①服务端并发正确性 ②稳定 main-KV slot 映射 ③混合 prefill/decode batch 的 ragged 路径 ④DGX Spark 早期 nvfp4_ds_mla 接线。
- 社区持有者断言：**"没有 Keys 的公开工作，本仓库在真实并发下无法正确运行。"**
- 并发收益（单 TP=2，Keys 后）：静态同时到达 c16=**315.1 agg**（每流 19.7）；错峰独立到达 c16=205.0，全部成功率 100%；确定性 victim 输出高负载逐字节一致。

### 4.2 我方是否有并发补丁 / 稳定性依据

- **文档化补丁：无**。我方交付物中没有任何等价于 Keys 的 dspark 并发补丁记录。
- **稳定性依据（实证）**：
  - A 组 45 组合、B 组 54 组合（486 样本）全部 err=0，覆盖 c1/c3/c5；
  - TP2 去 LL 子集（512/4096/16384/32768 × c1/c3/c5 + 131072 哨兵）err=0；
  - 32768/c5 在 priority+threshold 后 decode 并发恢复（ratio 1.05），说明 **c5 并发在调度层面是稳定的**。
- **局限**：我方并发上限 = `--max-num-seqs 6` 且实测只到 c5；**从未在 c8/c16 验证过 dspark 并发正确性**。若业务未来开更高并发，缺少 Keys 修复就是未知风险区。

### 4.3 是否应评估该补丁 —— 应，且分两步

| 步骤 | 动作 | 优先级 |
|------|------|--------|
| 1 | **核实 0.2.1 是否已并入 Keys 修复**：比对 0.2.1 镜像内 dspark kernel 源码与 drowzeys commit（`query_start_loc` / `num_rejected_tokens_gpu` 相关 diff）；若 Anemll 上游已 merge，则我方事实上已覆盖，只需文档化 | P1 |
| 2 | 若 0.2.1 **未**包含：评估 vendor 移植成本（补丁为 kernel + python 路径调整，社区已提供 `patches/keys-concurrency.patch`）；移植后在 c8/c16 做正确性回归（确定性 victim 对比 + acceptance） | P1 |

**架构建议**：即便 0.2.1 已覆盖，也建议**补一份我方 dspark 并发正确性的显式验证文档**（当前只有"实测 err=0"的隐式证据），把并发上限、ragged 路径、确定性校验固化，避免"换镜像后静默回退"。

---

## 5. 调度参数

| 参数 | 社区 | 我方 | 分析 |
|------|------|------|------|
| max_num_batched_tokens | **8192** | **4096** | 社区更大 → 单步吞吐上限高；我方 4096 已被 spec decoding 约束下验证有效（Chunked prefill 日志可见） |
| enable_chunked_prefill | 显式 | V1 默认 | 等效 |
| async_scheduling | 显式 | V1 默认 | 等效 |
| scheduling_policy | 未提及（fcfs） | **priority** | **我方核心优势**：交互优先于 batch |
| long_prefill_token_threshold | 未提及（0/禁用） | **2048** | **我方核心优势**：长 prefill 分块、decode 可插队 |
| 并发 partial prefill | 未提及 | 不支持（0.26.1 硬抛） | 我方少一个原生手段，靠 threshold/priority 补偿 |
| max_num_seqs | 6 | 6 | 平 |
| max_model_len | 1048576 | 600000 | 社区更大 |
| gpu_mem_util | 0.80~0.85（0.835 验证） | 0.80（A组）/0.88（B组） | 我方 B 组更高 → KV 预算略多 |

**调度优劣分析**：
1. **社区的 8192 + chunked + async 是"吞吐优先"配置**：最大化 prefill 聚合，但**没有 priority/threshold 保护**——在 vLLM V1 默认 `max_num_partial_prefills=1` 下，单长 prefill 会独占预算，后到 decode 请求被 head-of-line 阻塞（这正是我方 review-mla-compression-decode-collapse 定位的根因）。
2. **我方的 4096 + priority + threshold 2048 是"并发公平优先"配置**：以牺牲部分 c5 per-request prefill（TP2 去 LL 子集显示 c5 prefill 较 B 组 -14~-28%）换取 decode 不被饿死——**核心证据：32768 c5/c1 decode 比由 B 组 0.77 升至 1.05**（并发不再负收益），这是社区配置**不会达到**的结果。
3. **结论**：两者各为单边最优。**理想形态 = 社区批大小 + 我方调度保护**（8192 + priority + threshold）。建议在维护窗口做 4096→8192 的受控 A/B，观察 prefill 聚合是否提升且 32768 分界是否保持 ≥1.0。
4. `--enable-chunked-prefill`/`--async-scheduling` 在 V1 均为默认，社区显式写出属保守习惯，不构成差异。

---

## 6. 性能对比（含口径差异）

### 6.1 关键锚点对照

| 指标 | 社区 | 我方 | 相对 |
|------|------|------|------|
| 131072 / c1 prefill | 1665 | 1702（TP2 去LL）/ 1768（B组）/ 1821（A组） | **+2%~+9%** |
| decode c1（coding/json） | 65-74 | 71-80（A/B 组 coding/json 全 ctx 持平） | **+5%~+10%** |
| 512/c1 prefill | 447（256 prompt） | 1121-1178（512 ctx） | 口径异，见 6.2 |
| 4096/c1 prefill | 2563（2048 prompt） | 1929-2062 | 口径异，见 6.2 |
| 131072/c2 decode | 30.9 | 无同口径 | 需同口径验证 |
| 并发聚合 decode | x3=134.6 / C6=340.5（decode-only，自然结束探针） | c5 短 ctx 聚合 ~80-158（含 prefill 混合）；长 ctx UMA 饱和 | 口径异 |
| TTFT（131K） | — | 哨兵 ~67.76s | 无对照 |

### 6.2 口径差异说明（务必读，避免误判）

- **我方口径**：per-request p50（`bench_prefill_decode` 系），随机前缀防 prefix-cache 假象、温度统一、decode/prefill 解耦、max_tokens=128。这是被方法学规范固化的严格口径（runbook §3.5）。
- **社区口径**：decode bench 2048 完成 token（x1/x3/x4 聚合）；prefill sweep 为单请求 prefill tok/s（256/2048/8192/32768/131072 五档）；C6 为 1M ctx 自然结束探针（生成长度短、KV 已预热，聚合口径）。
- **512 prefill 447 vs 我方 1178 的"异常"**：社区 256-token 档把单请求固定开销（kernel launch、spec 初始化、chunk 边界）摊到极短 prompt 上 → tok/s 被稀释；社区自己的 2048 档到 2563、8192 档到 1713，说明**短 prompt 档不可用于横向对比**。我方 512=1121 反而是健康值（固定开销摊到 512 token 已充分）。
- **C6=340.5 的解读**：这是 decode-only + 短生成 + 高并发聚合，与"我方 c5 混合负载 80-96"不是同一物理量。**结论：社区峰值不可直接采信为全面领先，需同口径 head-to-head 才能定论。**

### 6.3 Regular vs Breakable CUDA graph —— 我方最值得测的一项

| 项 | 社区（0.25，1M ctx 自然结束探针） | 我方（0.26.1） |
|----|-----------------------------------|----------------|
| 配置 | `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（regular） | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（breakable） |
| C1 decode 热中位数 | 74.55 → **95.9**（**+28.6%**） | 71-80（breakable） |
| C2 聚合 decode | 134.2 → **151.8**（+13.1%） | 无同口径 |
| 单流自然 decode | ~96 t/s（regular） | 71-80（breakable） |

**判断**：
- 社区在 **dspark 5-token + max-cudagraph-capture-size 24** 完全相同的组合下测得 regular 显著优于 breakable，说明 dspark 路径下 regular graph 不必然有兼容问题。
- 我方 runbook 决策史"b12x+breakable+动态K 已是 GB10 最优"是在 0.25/早期 0.26 环境下得出的；**0.26.1 未做过 regular vs breakable A/B**。
- **若 +28.6% 在我方 0.26.1 复现**，decode c1 将从 71-80 升至 ~85-96，这是当前可预期收益最大的单项。
- **风险提示**：regular graph 对 ragged/dspark rejected-token 路径更敏感；需在 32768/131072 双 ctx 回归（尤其 c5 并发），防止"单流涨、并发崩"。

**动作**：P0，维护窗口做 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` A/B（保持 max-cudagraph-capture-size 24），验证 C1 提升与 c3/c5 并发正确性。

---

## 7. 网络 / 运维

### 7.1 我方限 TCP+防火墙收窄 vs 社区数据面直连（无防火墙）

| 项 | 社区 | 我方 | 分析 |
|----|------|------|------|
| 数据面 TCP | 无限制（直连裸跑） | 只放 NCCL 控制 TCP（对端 IP + 25000 + ESTABLISHED），其余 DROP | 我方安全加固，但属"语义削弱"（数据面不再是 0 TCP） |
| NCCL_SOCKET_IFNAME | 数据口（与 TP/GLOO 一致） | 数据口（enp1s0f1np1,enP2p1s0f1np1）+ GLOO 已改管理口 enP7s7 | 我方 GLOO 已隔离；NCCL 控制面仍走数据面 |
| 风险 | 无防火墙，内网任意访问 | ①漏 <NODE_IP>/24（NCCL 可能任选 136/137 逻辑口）②整 /24 放行偏宽 ③TCP 与 RoCE 同链路 → +128% 延迟（60% 背景实测） | 需补 137 子网或切管理口方案 |

**架构判断**：
- 我方防火墙收窄是 **audit-cluster-4node 安全整改的直接产出**（四机暴露在 Wi-Fi 段、无认证 registry），方向正确。
- 但当前实现是"**必要但偏宽**"：report-nccl-tcp-firewall 已给出更优方案——把 `NCCL_SOCKET_IFNAME` 也切到管理口 enP7s7，数据面恢复纯 RoCE（0 TCP），可撤销数据面放行。**该方案需在 staging 验证 vLLM ZMQ/mq 是否也走数据面后再落地**。
- 社区"无防火墙"是裸实验室配方；我方目标形态（管理口控制 + 数据面纯 RoCE + 防火墙 DROP）是生产级，不应为了对齐社区而放松。

### 7.2 我方独有：QoS PFC + isolcpus + MTU9000

- 社区无任何 QoS/CPU 隔离；我方已落地：MTU 9000、QoS PFC(P3/P5)（持久化待维护窗口）、isolcpus=16-19 四台（nozh_full + rcu_nocbs）、rp_filter=1。
- 收益：1 跳 RDMA 16B 延迟基线 3.27µs/p99 3.46µs；60% TCP 背景 +128% → 隔离后恢复基线。**这是社区无法直接对标的生产级延迟稳定性投入。**

### 7.3 earlyoom —— 应采纳社区建议

- 社区明确：`sudo systemctl stop earlyoom && sudo systemctl disable earlyoom`，原因是高 GPU 压力下可能误杀 vLLM worker/head（即使系统有 swap）。
- 我方现状：**未处理**。且 audit-cluster-4node 已记录 **.58 内存耗尽风险 + LLM worker Exited(137) 疑似 OOM**；GB10 统一内存使 GPU 工作集计入主机内存，earlyoom 恰好会把"GPU 压力"误判为"主机 OOM"。
- **建议**：四机执行禁用（需 sudo），或配置 earlyoom 白名单保护 vLLM/embed 进程；同时补内存告警阈值（audit 遗留 P0 项）。优先级 P1。

### 7.4 其余运维项

- 双节点镜像一致：社区手动 pull；我方双机同 digest + 本地 registry，已覆盖且更严格。
- 离线 HF cache 完整性：双方一致（我方 HF_HUB_OFFLINE=1 + 持久卷）。
- NVFP4 padded 584B vs 416 true-layout：社区用 padded 584B 路径（416 true-layout 在 ~411 token 后失败）；我方 nvfp4_ds_mla 属同族路径，6.7KB/tok 与社区 7.6KB/tok 的差异源于 util/布局，**需核对 0.26.1 是否同样走 padded 路径**（P3）。
- 1M retrieval benchmark：双方都未做完整 1M，平。

---

## 8. 可借鉴项清单（带优先级）

| # | 借鉴项 | 我方现状 | 优先级 | 预期收益 | 落地动作 |
|---|--------|----------|--------|----------|----------|
| 1 | **Regular CUDA graph**（`VLLM_USE_BREAKABLE_CUDAGRAPH=0`） | breakable=1，未测 regular | **P0** | C1 decode +28.6%（74.55→95.9，社区 0.25）；若在我方复现，decode 71-80 → ~85-96 | 维护窗口 A/B（保持 max-cudagraph-capture-size 24），32768/131K × c1/c3/c5 回归 |
| 2 | **Keys-Concurrency-Patch 核实/移植** | 无文档化补丁；c≤5 err=0 | **P1** | 解锁 c8/c16 并发正确性（社区 c16=315 agg）；消除"换镜像静默回退"风险 | ①比对 0.2.1 与 drowzeys commit ②缺失则 vendor `patches/keys-concurrency.patch` ③c8/c16 确定性回归 |
| 3 | **禁用 earlyoom** | 未处理，有 OOM 历史 | **P1** | 防高 GPU 压力误杀 vLLM（Exited 137 同源） | 四机 systemctl disable earlyoom（sudo）；或白名单保护 vllm/embed + 内存告警 |
| 4 | **1M 上下文可行性** | max_model_len=600000 | **P2** | 覆盖更长 RAG/agent（社区 1M 满池并发 2.38×） | 评估 600K→1M 的 KV 预算/seqs 权衡；验证 padded NVFP4 路径 |
| 5 | **tokenizer-mode deepseek_v4** | 未显式设置（用 tool/reasoning-parser） | **P2** | 0731 checkpoint 无 HF Jinja chat_template；社区专用 DSV4 编码器支持 4 档 thinking | 核实 0.26.1 是否支持该 tokenizer-mode / 镜像是否含 encoding_dsv4.py；对比 thinking 与 reasoning_effort 行为 |
| 6 | **启动顺序版本差异文档化** | head-first（0.26.1 正确） | **P2** | 防误按社区 0.25 做法改配置导致 NCCL 卡死 | runbook 补"worker-first=0.25 语义 / head-first=0.26.1 语义，跨版本重验" |
| 7 | **batched 4096→8192 受控 A/B** | 4096 + priority + threshold | **P2** | 提升 prefill 聚合（社区 8192 口径）且保持 32768 分界 ≥1.0 | 维护窗口 A/B，监控 c5 prefill trade-off 与 decode 稳定 |
| 8 | **KV 布局口径核对** | nvfp4_ds_mla 6.7KB/tok | **P3** | 确认 padded vs true-layout 路径一致 | 核对 0.26.1 nvfp4_ds_mla 布局；无需立即动作 |

---

## 9. 结论

### 9.1 我方相对社区的优势

1. **架构形态**：4 节点（A/B 组 + TP4 演进）vs 2 节点；多副本故障域 + 潜在 512GB 内存池。
2. **调度保护**：priority + long-prefill-token-threshold 2048 已实证解决长 prefill 阻塞 decode（32768 c5/c1 0.77→1.05），社区无此层。
3. **网络工程**：twin 双 HCA（188G）+ MTU9000 + QoS PFC + isolcpus + 限 TCP 防火墙 + GLOO 管理口隔离——社区裸直连不可对标。
4. **可复现性与可观测性**：本地 registry + 双机同 digest、NCCL_DEBUG 留证、Grafana/DCGM 全栈、编排脚本 + 回滚锚点、runbook（坑位 11 条）。
5. **核心单流性能**：131K prefill +2~9%、decode c1 +5~10%，同代同栈略优。
6. **启动确定性**：8/8 head-first 修复（vs 社区 worker-first 属 0.25 语义）。

### 9.2 我方相对社区的差距

1. **上下文上限**：600K vs 1M（社区满池 2.38×）。
2. **并发上限**：c≤5 且无文档化并发补丁（社区 Keys 后 c16=315 agg 且逐字节正确）。
3. **CUDA graph**：未验证 regular 路径，潜在 +28.6% decode 未兑现。
4. **tokenizer 等价性**：`--tokenizer-mode deepseek_v4` 未对齐，thinking/reasoning 细节存在未核实偏差。
5. **原生并发 partial prefill**：0.26.1 不支持（社区 0.25 未证实但可能可用）。
6. **调度批大小**：batched 4096 < 社区 8192（以 priority/threshold 补偿，但 raw prefill 聚合上限略低）。

### 9.3 我方风险（相对社区需注意）

1. **OOM/earlyoom 未处理**——有四机 Exited 137 前科，高 GPU 压力下 vLLM 被误杀风险真实（P1）。
2. **防火墙收窄语义**——数据面 TCP 放行范围偏宽 + 漏 137 子网；NCCL 控制面改管理口（更优方案）尚未验证，存在"时好时坏"隐患。
3. **0.26.1.dev0 为 dev 构建**——相对社区 0.25 稳定版有偏差风险；且不支持并发 partial prefill。
4. **dspark 并发正确性依赖隐式验证**——未显式覆盖 c8/c16 与 ragged 路径，换镜像后可能静默回退。
5. **TP4 若推进**：2 跳 ring 延迟劣化 + 03 补线/闭环成本，须与内存池收益权衡。
6. **社区 C6=340.5 峰值未同口径验证**——若真实存在，我方高并发 decode 聚合存在差距（但大概率口径差异）。

---

## 10. 行动清单（按优先级）

| # | 行动 | 负责 | 优先级 |
|---|------|------|--------|
| 1 | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` A/B（32768/131K × c1/c3/c5），验证 regular +28.6% 是否复现 | Tessa + 主理人 | P0 |
| 2 | 核实 0.2.1 是否含 Keys 修复（对比 drowzeys commit 7e4d94b）；缺失则评估移植 + c8/c16 确定性回归 | Archi + Cody | P1 |
| 3 | 四机禁用 earlyoom（或白名单保护 vllm/embed）+ 补内存告警 | Rex | P1 |
| 4 | 决策：NCCL_SOCKET_IFNAME 切管理口（更优方案） vs 补 137 子网收窄现状 | 用户 + Rex | P1 |
| 5 | 核实 tokenizer-mode deepseek_v4 在 0.26.1 的支持与我方 thinking 行为差异 | Archi | P2 |
| 6 | batched 4096→8192 受控 A/B（保持 priority/threshold） | Tessa | P2 |
| 7 | 600K→1M ctx 可行性评估（KV 预算 + padded 路径） | Archi | P2 |
| 8 | runbook 补启动顺序版本依赖说明 + dspark 并发正确性验证文档 | Docu | P2 |

---

## ⚠️ 局限与假设

- 社区性能数字源自该仓库 README/配方（2026-08-10 在线核实），未在本集群同口径复测；所有"相对"结论以我方严格口径（随机前缀/温度统一/per-request p50）为基准。
- 我方 0.2.1 是否含 Keys 修复、是否支持 `--tokenizer-mode deepseek_v4`，属**待核实**项，未在镜像内取证。
- 启动顺序分析基于我方 8-01（0.25 worker-first 可用）与 8-06 后（0.26.1 head-first 3/3、8/8）的观测；社区 0.25 内部细节以 README 声明为准。
- 数值中 TP2 去 LL 131072/c1=1702.51 与 A/B 组（1821/1768）口径同日不同，取三值范围表述。

---

> 本报告由工程保障团队架构师 Archi 产出，关键决策请由人类工程负责人复核。
