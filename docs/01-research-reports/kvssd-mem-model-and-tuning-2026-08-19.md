# TP4 vLLM UMA 内存模型剖析 + 参数优化矩阵

- **作者**：Cody（代码审查师，工程保障团队）
- **日期**：2026-08-19
- **范围**：4×DGX Spark（121GB UMA）TP4 vLLM 0.26（anemll/dspark-vllm-gx10:0.2.1-v026.0）deepseek-v4-flash-0731 内存模型理论分解 + 实机数据归因 + 可操作参数优化矩阵
- **数据口径**：Rex RCA（`kvssd-rca-memutil07-2026-08-19.md`）实机数据已引用；SRE 内存构成采集（`kvssd-mem-composition-data-2026-08-19.md`）**未就绪**，相关项标注 **「待实测」**
- **审查边界**：仅静态分析（本地文件/前次审查报告），不 SSH 节点

---

## 0. TL;DR

1. **`--gpu-memory-utilization` 在 UMA 上 = KV cache 池的「分配预算」，不是宿主 RSS 的硬上限**。vLLM 启动时按 `总内存 × util − 权重/激活峰值` 预分配 KV 池（CUDA 张量），但进程总 RSS 还会叠加页缓存（155GiB 模型 mmap）、宿主机进程、卸载框架、运行时超额——这就是 0.8 时 used 110-115GB > 理论 96.8GB 的结构性原因，**不是泄漏**。
2. **0.70 修正已实证有效**（conc3×65536 4/4 通过，谷底 01=7.9G/02=8.7G/03·04=2.5G），但 03/04 头寸仅 ~2.5GB，**仍偏薄**。
3. **最大未利用杠杆**：当前生产仍是 `max-num-seqs 12 + capture 96`，而 R11（08-13 已收敛的稳定配置）是 `seqs 6 + capture 64 + util 0.65`。事故只回撤了 util（0.8→0.7），**并发与 cudagraph 还没回撤**。降 `max-num-seqs 12→8` 是收益/风险比最高的下一步。
4. **CPU 主层 2GiB 实测利用率 0-1%**（≈0-20MB），`cpu_bytes_to_use` 从 2GiB→1GiB 几乎零代价、释放最坏情况 ~1GB，对 03/04 边际最友好。
5. **权重驻留估计 ~50-55GB/节点（TP4 + EP=4 分摊），而非前次审查的 ~79GB**；需启动日志 `gpu_memory_used` 实测钉死（见 §2 差异说明）。

---

## 1. vLLM 在 UMA 上的内存模型（0.26 源码/文档语义）

### 1.1 `gpu_memory_utilization` 的真实语义

标准 vLLM 流程（`vllm/v1/core.py` → `model_executor.determine_available_memory()`）：

```
可用 KV 内存 = torch.cuda.mem_get_info().total × gpu_memory_utilization
              − memory_profile_run 峰值（权重加载 + 一次前向激活峰值）
```

- 在 UMA（Grace Blackwell）上 `torch.cuda.total` 返回 **121GB 统一内存池**（GPU+CPU 同一物理池，无独立显存）。
- **是 KV 池的硬上限**：KV cache 张量在启动时按该预算**预分配**，`num_gpu_blocks` 由此定死；`expandable_segments` 因 OffloadingConnector 不兼容被置空，分配器不弹性扩展、不收缩 → KV 池是**一次性钉死**的（重启才生效）。
- **不是宿主 RSS 的硬上限**：进程实际占用 = KV 池 + 权重驻留 + 激活 + cudagraph + NCCL + 卸载框架 + 宿主机其它进程 + 页缓存。这些都在 121GB 同一物理池内互相挤占。**0.8 时 used 110-115GB 就是「预算 96.8 + 预算外 ~15GB」的结构性构成**（前次审查 §3.1 已定性）。
- **关键推论**：UMA 下 `util` 调低**会**立刻缩小 KV 池（重启生效，RCA 实测 0.7 释放 ~10-12GB/节点），但**不会**自动缩小页缓存/其它进程——「释放可用内存」的效果来自 KV 池收缩，与「可回收页缓存被压出」叠加。

### 1.2 121GB 去向理论分解（每节点）

| # | 项 | 量级估计 | 固定/动态 | 说明 |
|---|---|---|---|---|
| 1 | **模型权重驻留** | **~50-55GB**（见 §2） | 固定 | 155.4GiB / TP4 分摊；MMAP 文件映射，RSS 按需驻留，页缓存可回收 |
| 2 | **KV cache 池（预分配）** | 0.7: ~25-35GB / 0.8: ~35-45GB | 动态（池固定、占用按需） | = 预算 − 权重 − 激活预留；`60 万 × 12 seq × ~6.7KB/token ÷ TP4` 最坏 ~12GB/节点，池容量 > 最坏需用量以容余量 |
| 3 | **activations（prefill 峰值）** | 瞬态 2-8GB | 动态 | 长上下文 conc3 突发放大；被 `max-num-batched-tokens` 分块 |
| 4 | **cudagraph 池** | capture 1..96 档 × 每档批次缓冲 ≈ 1-4GB | 固定（启动捕获） | `--max-cudagraph-capture-size 96` 比 R11 的 64 多 ~50% 档位 |
| 5 | **NCCL 通信缓冲**（TP4 环网） | 0.5-2GB | 固定 | allreduce 中间缓冲 + communicator |
| 6 | **KV 卸载框架** | ~3-3.5GB | 动态（上限固定） | CPU 主层 2GiB + kv_buffer 1GB + 元数据（前审 §二） |
| 7 | **宿主机其它进程**（redis/pg/neo4j/minio/grafana/litellm/embed-8022/OS） | **5-10GB** | 固定 | 03/04 额外跑 embed → 更紧 |
| 8 | **页缓存**（155GiB 模型文件 mmap 读入） | 数 GB-40GB | 动态（可回收） | 计入 `free used`，压力下被内核回收 → cgroup current 下降（RCA 观察到 23-28→7-12GB） |
| 9 | **vLLM 运行时超额**（分配器碎片、JIT/autotune workspace、投机 draft） | 1-4GB | 固定 | expandable 关闭 + 默认分配器不回收 |

### 1.3 「数据交换 → 内存占用上升」的路径（回答 Q2）

| 路径 | 机理 | 相对贡献 |
|---|---|---|
| **KV cache 增长** | 每 token ~6.7KB（nvfp4 MLA 压缩后）；60 万 ctx × 12 seq 最坏 7.2M token → 48GB/集群 ≈ 12GB/节点 | ★★★ 长上下文主因 |
| **prefill activation 峰值** | conc3×65536 = 3 路 65K prefill 同时计算；虽然被 4096 分块，但并发聚合 + 卸载管线叠加 | ★★★ 突发主因（事故触发器） |
| **页缓存** | 155GiB 模型文件被 mmap/读取，未 `madvise DONTNEED` 时按需驻留 | ★★ 占用大但可回收 |
| **NCCL allreduce 中间缓冲** | TP4 环网 allreduce 12.5M 元素 3× 权重规模，buffer 驻留 | ★ |
| **卸载管线** | CPU 主层 LRU（2GiB 预算）+ fs 读写线程缓冲（MB 级）+ 压缩 | ★（确定性 ~3GB，已量化） |
| **页表 / slab / 内核元数据** | 数万 KV 文件 + mmap 页表 | ★ |

> **注意**：RCA 实测 CPU 主层利用率 0-1%（≈0-20MB），**fs 层才是卸载主通道**——`cpu_bytes_to_use=2GiB` 是「最坏情况预算」而非「当前占用」，降低它释放的是最坏情况头寸。

---

## 2. 真实占用构成表（121GB 估算分解）

> 权重分解按：routed experts 33024 投影 I8 打包 4bit + E8M0 scale（~137-145GB 总量）→ **TP4 + expert-parallel 分摊 ≈ 34-36GB/节点**；attn FP8 + 共享专家 + 路由 + norm ≈ 2-4GB；embed BF16（129280×4096 ≈1.06GB，brief 记 505MB 待核对 dims）≈ 0.3-1GB；MTP/dspark 投机模块（FP8）≈ 10-20GB（**若按 rank 复制**）。

| 项 | 估计（0.7 util，每节点） | 固定/动态 | 依据/来源 |
|---|---|---|---|
| 模型权重（routed experts EP 分摊） | 34-36GB | 固定 | 33024 × ~4.2MB(4bit+scale) ÷ 4 |
| 模型权重（attn/embed/norm/shared/MTP） | 10-20GB | 固定 | FP8/BF16，MTP 若复制则偏上限 |
| **小计：权重驻留** | **~45-55GB** | 固定 | **待实测**：head 启动日志 `gpu_memory_used` / `torch.cuda` summary |
| KV cache 池 | ~25-35GB | 动态 | 0.7 预算 84.7 − 权重 − 激活预留 |
| cudagraph 池 | 1-4GB | 固定 | capture 1..96 |
| NCCL 缓冲 | 0.5-2GB | 固定 | TP4 环网 |
| 卸载框架（CPU 层 2GiB + kv_buffer 1GB） | ≤3-3.5GB（实测占用 ~0-20MB 层 + MB 级） | 动态上限 | RCA 实测利用率 0-1% |
| vLLM 运行时超额 | 1-4GB | 固定 | 分配器碎片/JIT/投机 |
| 宿主机进程 | 5-10GB | 固定 | 03/04 含 embed-8022 |
| 页缓存（模型文件） | 数 GB-40GB | 动态可回收 | `free used` 计入；RCA：cgroup 23-28→7-12GB |
| **合计（宿主 used 口径）** | **~95-115GB** | — | 与 0.7/0.8 实测 used 110-115GB 同量级 ✓ |

**差异说明（相对前次审查）**：前次 `kvssd-memory-footprint-review` 取权重 ~79GB，本次按 EP=TP4 分摊修正为 ~45-55GB。若实机 `gpu_memory_used` 显著 >55GB，则可能是 (a) MTP 全量复制、(b) experts 未 EP 分摊（每 rank 全量 33024 投影 ≈137GB，UMA 下不可能）、(c) 权重加载未压实（I8 解包为 4bit 两倍）。**以启动日志为准。**

---

## 3. 参数优化矩阵

> 优先级：**P0**=建议尽快做（收益/风险比高）；**P1**=观察后做；**P2**=备选后手。当前生产 = 事故后 0.70，R11（08-13）稳定基线 = 0.65 / seqs 6 / capture 64。

| 参数 | 当前值 | 建议值/方向 | 内存影响 | 风险 | 优先级 |
|---|---|---|---|---|---|
| **max-num-seqs** | 12 | **8**（回 R11 方向 6-8） | 并发 KV 最坏量 7.2M→4.8M token（-33%）；scheduler 批压力降 | 吞吐/并发容量降；conc≤3 场景几乎无感 | **P0** |
| **gpu-memory-utilization** | 0.70 | **维持 0.70**（0.65 留作 03/04 后手） | 维持现状释放 10-12GB；再降 0.05 ≈ +6GB avail，但 KV 池缩小 → 卸载更激进 | KV 池缩小 → 更多 SSD 卸载、zstd CPU、写放大 | **P0（维持）** |
| **max-cudagraph-capture-size** | 96 | **64**（capture-sizes 1..64 含 36） | cudagraph 池档位 -33%，省 ~0.5-1.5GB | 捕获覆盖面下降；seqs=8 稳态批尺寸 ≤64 已覆盖 | **P0** |
| **cpu_bytes_to_use** | 2GiB | **1-1.5GiB**（实测利用率 0-1%） | 最坏情况省 0.5-1GB；当前占用几乎不变 | CPU 主层命中率略降（当前本就 ~0） | **P0** |
| **kv_buffer_size** | 1e9（默认） | 保持；**待实测**是否实际占用 | 若 OffloadingConnector 实际不占用可显式收紧 | 卸载传输吞吐 | **P1（先实测）** |
| **max-num-batched-tokens** | 4096 | **保持 4096**（必要时 2048） | 4096 已分块 prefill；降到 2048 减半单块激活峰值但块数翻倍 | 批大小/调度开销 | **P1** |
| **KV cache dtype** | nvfp4_ds_mla | **保持** | 已是最优压缩（~6.7KB/token 量级） | — | — |
| **enable-prefix-caching** | on | **保持** | 命中复用省 KV；长随机前缀（conc3 测试）不增负担 | 大量相异前缀会占住 KV 块 | — |
| **页缓存回收** | 无策略 | **posix_fadvise(DONTNEED)/vmtouch 于模型加载后**，或负载前 `drop_caches` 定时 | 释放 file-backed 数 GB-40GB | 首次访问/回读慢；需避开负载期 | **P1** |
| **swap / overcommit** | 15G disk swap | 评估 **zram 或降 swappiness**（事故期 swap thrash 拖慢 NCCL） | 兜底突发；zram 压缩内存交换更快 | zram 占 CPU；不能替代主内存预算 | **P1** |
| **NCCL 缓冲（MAX_CH）** | 4 | **保持** | 通信 buffer 规模与通道数相关 | — | — |
| **03/04 embed-8022** | 运行 | **关停/迁走**（03/04 头寸最紧） | 每节点省 1-2GB | embed 服务不可用 | **P0（03/04）** |

### 3.1 03/04 头寸仅 2.5GB 的专项组合建议

现状：0.70 下 conc3×65536 谷底 03/04 = 2.5-3.1GB（RCA 实测），margin 偏窄。**组合拳（按顺序生效）**：

1. **关 03/04 embed-8022**（省 1-2GB，最直接）；
2. **`cpu_bytes_to_use` 2GiB→1GiB**（最坏情况省 1GB，当前占用不变）；
3. **`max-num-seqs` 12→8**（KV 最坏量 -33%，长上下文并发下直接降低池内压力与卸载风暴概率）；
4. **`max-cudagraph-capture-size` 96→64**（省池 ~0.5-1.5GB）；
5. 上述生效后若 03/04 谷底仍 <2GB，**再降 util 0.70→0.65**（每步 +6GB avail，但 KV 池更小、卸载更激进——作为最后手段而非首选）。

> 预期：1-3 步合计把 03/04 谷底从 2.5GB 抬到 ~5-7GB，无需动 util，保住 KV 池容量。**不建议在 seqs 未降时先降 util**——KV 池缩小 + 并发不减会双倍放大卸载流量。

---

## 4. 用户四个问题（直接回答）

### Q1：为什么内存会耗尽？
0.8 util 把 96.8GB 预留给 GPU（权重+KV+激活），宿主侧名义只剩 ~24GB；03/04 还跑 embed，稳态 avail 只剩 4-6GB。conc3×65536（3 路 65K prefill 并发）的**激活尖峰 + KV 增长 + 卸载管线**把 4-6GB 打穿至 0 → 03 先 NVRM `NV_ERR_NO_MEMORY`（03:18）→ avail=0（03:20）→ 系统冻结 → 内核 oom-killer 杀 `VLLM::Worker_TP`（03:54）→ NCCL ALLREDUCE 因系统 stall 无法推进 → 300s watchdog 超时 → worker 死 → 集群宕。**内存耗尽在前，NCCL 超时是被动受害者**（RCA 时间线证实）。

### Q2：哪些数据交换增加了内存占用？
按贡献：① **KV cache 增长**（60 万 ctx×12 seq×~6.7KB/token，最坏 ~12GB/节点）；② **prefill activation 峰值**（conc3 突发，事故触发器）；③ **页缓存**（155GiB 模型文件 mmap 读入，可回收但计入 used）；④ **NCCL allreduce 中间缓冲**；⑤ **卸载管线**（CPU 主层 + 压缩缓冲，确定性 ~3GB）；⑥ **页表/slab**。io 补丁本身 MB 级可忽略（前审）。

### Q3：真实占用多少 / 构成？
0.7 下每节点宿主口径 ~95-105GB used / avail 18-25GB（空闲）；0.8 事故态 used 110-115GB / 03·04 avail 4-6G。构成：权重驻留 ~45-55GB（固定）+ KV 池 ~25-35GB（动态）+ 宿主机进程 5-10GB + 页缓存数 GB-40GB（可回收）+ 卸载 ≤3.5GB + cudagraph/NCCL/运行时 3-8GB。**详见 §2 表**；`gpu_memory_used`、KV 池大小、cgroup rss/anon 拆分待 SRE 采集补实。

### Q4：哪些参数可优化（按收益排序）？
| 排序 | 参数 | 动作 | 收益 | 优先级 |
|---|---|---|---|---|
| 1 | **max-num-seqs** | 12→8 | KV 最坏量 -33%，长上下文并发压力直降 | P0 |
| 2 | **03/04 embed** | 关停/迁走 | 每节点省 1-2GB，直接补 03/04 头寸 | P0 |
| 3 | **cpu_bytes_to_use** | 2GiB→1GiB | 最坏情况省 1GB，当前占用几乎不变 | P0 |
| 4 | **max-cudagraph-capture-size** | 96→64 | cudagraph 池省 ~0.5-1.5GB，回到 R11 覆盖面 | P0 |
| 5 | **页缓存回收** | fadvise/vmtouch/drop_caches | 释放 file-backed 数 GB | P1 |
| 6 | **gpu-memory-utilization** | 维持 0.70，0.65 作后手 | 已释放 10-12GB；再降以牺牲 KV 池为代价 | P1 |
| 7 | **swap** | zram/降 swappiness | 改善突发兜底，避免 swap thrash | P1 |
| 8 | **max-num-batched-tokens / kv_buffer_size / NCCL** | 保持 / 实测后定 | 边际 | P2 |

---

## 5. 数据缺口（待 SRE `kvssd-mem-composition-data` 补实）

1. head 启动日志 `gpu_memory_used`（权重驻留权威值，验证 ~45-55GB vs ~79GB 分歧）；
2. KV 池实际 `num_gpu_blocks` × block 字节（钉死 ~6.7KB/token 与池容量）；
3. cgroup `rss/anon/file` 拆分（区分权重 RSS vs 页缓存 vs KV 池）；
4. `kv_buffer_size` 在 OffloadingConnector 路径是否真实占用；
5. 空闲 vs 负载下 `free` used 的逐项归因（页缓存占比）。

---

## 6. 结论

- **模型已定位**：util 是 KV 池预算非 RSS 硬上限；0.8 事故 = 预算外结构占用 + conc3 突发打穿薄头寸；0.70 修正有效（4/4 通过）。
- **下一步最优**：不动 util，先降 `max-num-seqs 12→8` + `capture 96→64` + `cpu_bytes 2→1GiB` + 03/04 关 embed——把 03/04 谷底抬到 5-7GB，同时保住 KV 池容量、避免卸载放大。
- **验证**：组合落地后复测 conc3×65536（沿用 `/tmp/verify_conc3_65536.py`），盯 03/04 谷底 ≥5GB + 无 NVRM OOM + kv_load_failure=0，作为放行判据。
- **长期**：页缓存回收策略 + zram 兜底 + 03/04 embed 迁走为结构性缓解；权重驻留实测后固化内存台账。
