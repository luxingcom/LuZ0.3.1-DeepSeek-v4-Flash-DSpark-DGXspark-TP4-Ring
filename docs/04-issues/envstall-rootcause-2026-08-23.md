# P0 环境级随机 AR stall 根因调查报告（只读取证）

**日期:** 2026-08-23（本地）/ 取证窗口 2026-08-22 19:04–19:20 UTC
**作者:** Archi（系统架构师）· engineering-fullwindow 团队（envstall-investigator）
**对象:** W1 窗口发现的随机 AR stall（17–20ms、1ms 量子化、per-run 全有或全无、与库/通道数/pin/embed 无关）
**上游输入:** ringonly-w1-2026-08-23.md（§4 P0 章节）、slowround-rootcause-2026-08-22.md、expd-r123-2026-08-23.md
**口径标注:** 【实测】= 本轮直接取证；【实测旁证】= 数据支持但样本小/窗口重叠；【推断】= 证据链推论；【协议】= 需窗口判决实验（本报告只列不执行）
**纪律:** 全程纯只读（ethtool/proc/日志/容器元数据/采样器数据分析），零 GPU 负载、零网络主动探测、零生产改动，与并行 GPU 任务零冲突。

---

## 0. 执行摘要

| 项 | 结论 |
|---|---|
| **GDR 路径** | **GDR=0 四节点全部实锤**（NCCL 自检 `cuMemGdrSupport 0` + 无 gdrdrv/GDRCopy 内核模块 + nvidia-smi 无 GDR 字段）→ AR 数据面 = host staging（proxy 线程 + C2C + RDMA），stall 必然落在该路径上【实测】 |
| **NIC/织物侧** | **排除**：PFC 四节点全关、pause 帧零、ECN 计数零移动、无重传/RNR（W1 hw_counters 差分）、W1 窗口 mlx5 IRQ 近零（无中断风暴）、环网直连无交换机——"交换机 PFC/ECN"行动项不适用【实测】 |
| **头号根因候选** | **NCCL proxy 线程"低功耗定时轮询"模式抽签**：stall 运行期间 NCCL proxy 核（CPU 8/9）出现 ~1–2.6kHz arch_timer 信号，clean 运行无/弱——proxy 线程以 ~0.5–1ms 定时唤醒推进数据，解释 1ms 量子化、per-run 全有或全无、无重传无错误、GPU 自旋等待【实测旁证，n 小】 |
| **时间恶化（25%→5%）** | 未定论：候选= communicator 重复创建/销毁导致的驱动/资源状态累积（~60+ 次生命周期）；**统计警示：n=10 下 3/10→0/10 的差异未达判决强度**【推断】 |
| **生产 e2e（A2）** | **生产稳态未观测到灾难性 stall**：expD R1–R7 panorama 全部带内（2753–2853），稳态 TTFT 差仅 ~3%（慢簇 2.96–2.98 vs 中簇 2.88–2.90s）——若 17ms/AR 灾难性 stall 发生在生产 prefill，PR 将坍塌 >30%，未观测。首请求 1.2s 差为 warmup 类（与既有模式方差一致）。**但 18:53 重启后 communicator 是否中招未验证（生产空闲无流量）——需 E5 判决**【实测+推断】 |
| **附带发现** | ① Prometheus（aicad-prometheus-1）16:33:13Z 起 down（R2 dockerd 重启连带），监控盲区 2.5h+，需恢复；② W1 窗口"comp IRQ 尖峰"实为 **NFS 模型加载流量**（NFS 走 RoCE 口 10.100.140.x），非 NCCL；③ 前序分析中 14:30 "1.04 亿中断风暴"为**采样器首帧累计值伪影**（已核实并更正）【实测】 |

---

## 1. GDR 路径确认（任务 1）——实锤：GDR=0，AR 走 host staging

### 1.1 证据【实测】

| 证据 | 来源 | 内容 |
|---|---|---|
| NCCL 驱动级自检 | W1 日志（四节点各自 rank 日志） | `NCCL INFO Symmetric memory is not supported. cuMemEnable 1, globalGinSupport 0, cuMemGdrSupport 0`——01/02/03/04 四节点全部出现 |
| 环建立行 | 同上 | `Connected all rings, use ring PXN 0 GDR 0` |
| 内核模块 | 01 lsmod | 无 `gdrdrv`（GDRCopy）；`/dev/gdrdrv` 不存在 |
| nvidia-smi | 01 `nvidia-smi -q` | 无 GDR/GPUDirect 相关字段（GB10 驱动不暴露） |
| 拓扑 | 01 `nvidia-smi topo -m` | 单 GPU/节点（无 peer access 问题域）；GPU-NIC 关系 NODE（同 NUMA，经 PCIe） |

**结论：GPUDirect RDMA 在本平台（GB10 + CUDA 13.0 + 2.30.7+cuda13.0 驱动栈）不可用，NCCL IB 传输对 GPU 数据必须经 host staging：GPU kernel ↔（C2C/UMA）↔ host 缓冲 ↔（proxy 线程 + ibverbs）↔ NIC。AR 的每一次数据推进都依赖 proxy 线程活性。**

### 1.2 host staging 路径的 stall 机制分析

W1 已证（引用）：enqueue 0.02ms、GPU event 18ms → 延迟在 GPU 时间线内，NCCL kernel 自旋等数据；数据到达但慢（1ms 量子）。结合本轮新证据（§3）：

- **1ms 量子化的来源锁定在 host 侧 proxy 线程的定时唤醒**：stall 运行期间 proxy 核出现 ~1kHz 级 arch_timer（§3.3），clean 运行没有。即 stall 模式 = proxy 进度循环落入"定时睡眠轮询"（每 ~0.5–1ms 醒一次处理一段数据），而非忙轮询。
- 库二进制（/opt/nccl-ringonly/libnccl.so.2.30.7，`NCCL version 2.30.7+cuda13.0`）strings 取证：存在 `nanosleep`、`sched_yield`、`ncclProfilerProxyCtrlSleep` 符号及 `NCCL_IB_MQP_RETRY_SLEEP_MSEC`、`NCCL_SOCKET_POLL_TIMEOUT_MSEC` 等 env——2.30.7 存在 proxy 睡眠类机制【实测】。
- R1 IRQ 迁移（0-4,10-14）与 proxy 核（8-9，libncclpin 规则）**不重叠**：stall 不是"IRQ 打进 proxy 核"的直接竞争（W1 窗口 8/9 核中断量近零）；W1 的 NO_PIN 对照阴性也一致（详见 §5 RC2 弱点分析）。

---

## 2. NIC 侧只读取证（任务 2）——织物侧干净，全部排除

### 2.1 PFC / ECN / 拥塞【实测，四节点】

| 检查项 | 01 | 02 | 03 | 04 | 结论 |
|---|---|---|---|---|---|
| PFC（ethtool --show-pfc 不支持，用 ethtool -a） | RX/TX off | RX/TX off | RX/TX off | RX/TX off | PFC 全关（符合直连拓扑既定策略；未"意外开启"） |
| pause 帧（rx/tx_pause_ctrl_phy 等） | 0 | 0 | 0 | 0 | 零 pause |
| rx_prio0_buf_discard | 8239/7383（两 HCA 各自同值） | 2616/2319 | 1917/1564 | — | 静止（12s 双快照零移动）；开机累计量级，非窗口相关 |
| ECN/重传/RNR | 零 | 零 | 零 | 零 | ethtool -S 全量 2456 计数器中错误/拥塞类无非零移动项（叠加 W1 7 次 stall 运行 280 AR 的 hw_counters 差分零移动） |

### 2.2 R1 迁移实证【实测】

- 四节点 84×4 mlx5 IRQ 的 `smp_affinity_list` 全部 = **0-4,10-14**（本轮逐 IRQ 核验，01 全量+02/03/04 抽验）。
- R1 实际执行时间 = **16:33 UTC**（expD 报告；任务简报中 "~17:00" 为近似）。15:47 的 r1_cycle 首轮为失败尝试（构建容器抢 GPU 显存）。
- 历史落位（开机累计计数）：迁移前 comp 中断峰值在 CPU 5–9（comp5=30.1M、comp9=14.4M、comp13=10.1M 累计——与阶段 1 审计一致）。
- **迁移后生产流量中断热点 = CPU 11（约 35%）+ CPU 4 + CPU 0**（16:36–17:59 各次生产重启窗口实测分布）。

### 2.3 W1 窗口 NIC 活动 = 近零【实测】

W1 窗口（18:08–19:00）逐分钟 mlx5 IRQ 总量：**200–2,200/分钟**（对比生产重启期 80–100 万/分钟）——nccl-tests 期间（stall 或 clean）NIC 无中断风暴、无异常事件：
- 无 async0（NIC 事件队列 IRQ）异常（全天除首帧伪影外零记录）；
- NCCL 数据面零完成中断 = proxy 忙轮询 CQ（与阶段 2 结论一致，本轮再次确认）。

### 2.4 两项重要更正（避免后续误导）【实测】

1. **"14:30 中断风暴 1.04 亿/30s" 是采样器首帧伪影**：sampler2（irq_source.tsv）14:30:14 启动，首帧记录的是**开机累计值**（comp13 首帧 10.5M ≈ 阶段 1 审计累计 10.1M；async0 首帧 5–6M）。非真实风暴。后续任何基于该 TSV 的分析须跳过首帧。
2. **生产期 comp IRQ 尖峰 = NFS 模型加载流量，非 NCCL**：RoCE 口承载 NFS（`<NODE_IP>:nfs ← <NODE_IP>`，01 上多条 NFS 连接）。16:36/16:44/17:08/17:21/17:31/17:41/17:56/18:54–18:56 的 80–100 万/分钟尖峰时刻 = 各次生产重启的模型加载阶段（NFS TCP over RoCE 口）。NCCL AR（生产或测试）均不产生完成中断。
   - 含义：W1 报告 §4.4 建议 1"交换机侧 PFC/ECN/队列调度"**不适用**——环网为直连线缆（无交换机），且 NIC 侧 PFC/pause/ECN 全零。建议撤销该项，资源转投 §7 判决实验。
   - 观察项（低优先级）：模型加载时 NFS 流量 + 13–17K/s 中断落在 0-4,10-14（R1 后）——与 R2 守护进程带重叠，但重启期 vLLM 尚未服务，无实际冲突记录。

---

## 3. stall 机制核心证据链（本轮新增）

### 3.1 1ms 量子化的实证形态【实测，日志取证】

stall 运行逐迭代数值（probe_prod_4ch_1，tokens=96，40 迭代）：
```
19.96,21.98,19.99,21.00,23.97,19.01,22.97,18.01,23.98,19.00,22.98,17.00,24.91,18.07,23.98,21.00,...
```
全部落在**精确 1.000ms 网格**（整数毫秒，相位偏移约 -0.02ms = 测量开销）。这不是带宽受限的连续分布，是**固定周期事件驱动的推进**——每个周期推进一个数据量子。跨运行取值域 13–31ms（≈13–31 个周期）。

### 3.2 时间恶化数据（W1 §4.3 引用 + 本轮复核）

| 时段 | clean 率 | 备注 |
|---|---|---|
| 18:18–18:25 | ~25–30% | A2/scan 4ch/8ch 干净 |
| 18:31–18:45 | ~20–33% | probe 序列 |
| 18:47–18:52 | 0–10%（v5on8 0/10、v5off8 1/10、prod4 0/10、v5on16 1/6） | 大样本统计轮 |

**统计警示**【推断】：n=10 时 3/10→0/10 差异，Fisher 精确检验 p≈0.21——"恶化"趋势方向一致但未达判决强度；不能排除平稳随机（p≈15–25%）+ 小样本波动的合成。判定需 E4（大样本复现）。

### 3.3 头号证据：stall 运行期间 proxy 核出现 ~1kHz 定时器信号【实测旁证】

用 /tmp/_slowround/irq_source.tsv（2s 粒度 per-IRQ×per-CPU）对齐 stall/clean 运行窗口（±3s 窄窗 + [DONE-8s, DONE+1s] 运行窗双口径；注意相邻运行间隔 4–6s，宽窗有重叠污染，以下取窄窗口径为主）：

| 事件 | 类型 | CPU8 arch_timer（/2s） | CPU9（/2s） | 判读 |
|---|---|---|---|---|
| stall_A3@18:18:25 | stall | 0 | **3,469** | ~870Hz |
| stall_r2@18:24:43 | stall | 0 | **5,252** | ~1.3kHz |
| stall_scan48@18:24:57 | stall | 0 | **1,126** | 弱信号 |
| stall_probe2@18:31:28 | stall | 0 | **4,066** | ~1kHz |
| stall_stat1@18:47:36 | stall | **4,418** | 0 | ~1.1kHz |
| clean_A2@18:18:19 | clean | 0 | 0 | 无 |
| clean_scan@18:18:35 | clean | 0 | 0 | 无 |
| clean_r3@18:24:48 | clean | 0 | 728 | ~180Hz（与相邻 stall 窗重叠污染） |
| clean_probe1@18:31:23 | clean | 0 | 1,394 | ~350Hz |
| clean_stat3@18:48:39 | clean | 355 | 0 | 无 |

**判读**：NCCL proxy 线程（libncclpin 规则带 8-9，isolcpus 隔离核，W1 期间几乎无其他负载）在 stall 运行期间出现 ~0.9–1.3kHz 的 arch_timer 中断——即该核上有线程以 ~1ms 周期性定时唤醒；clean 运行期间 proxy 核定时器近零（忙轮询不碰 hrtimer）。**与 3.1 的 1ms 量子化直接互证：stall 模式 = proxy 进度循环在定时睡眠轮询态，数据每 ~1ms 醒来推进一个量子。**

口径与限度：n=5 stall / 5 clean，窗口重叠存在污染，信号强度跨越 0–5.2K 区间——标注【实测旁证】，判决需 E1/E2（§7）。

### 3.4 "per-run 抽签"的机制归位【推断】

- W1 已证：抽签在通信建立时刻决定，运行内 300 迭代不变；
- 本轮补充：与 IRQ 位置无关（R1 前后历史 stall 签名一致——W1 报告"签名与历史 17–20ms 一致"，历史数据全部为 R1 之前采集）；
- 推断：**抽签变量是 communicator 建立期 proxy 线程的初始化路径/资源分配**（如：初始化期间空闲时长触发低功耗睡眠逻辑并"闩锁"、CQ/comp-vector 分配、proxy 线程创建时序竞态——与 expD 证实的 libncclpin 按名 pin 竞态漏网 4/7 同类环境）；
- 通道数无关（4/8/16 同概率）与"proxy 每线程一档资源"的抽签一致（通道数改变的是线程/资源数量，不改变单线程落入睡眠态的概率）。

---

## 4. 时间恶化机制（任务 3）

| 候选 | 证据 | 评估 |
|---|---|---|
| 热/电源 | thermal zone 63–68°C（阶段1）；W1 dmon 2.5GHz 满频；本轮 62°C/14W（空闲） | 排除【实测】 |
| NIC 缓冲/表项老化 | 计数器零异常 | 排除（无证据）【实测】 |
| host 内存碎片 | 12s 快照：MemFree 108/127GB；buddyinfo 2MB 块 12,100 个、4MB 4,29 个、8MB 仅 1 个；THP=madvise、AnonHugePages=0；iommu.passthrough=0（SMMU 开启） | 高阶碎片存在但 2MB 充足；W1 期间内存压力低，35 分钟内快速劣化难以用碎片解释。**弱候选**【实测+推断】 |
| communicator 生命周期累积（驱动/NIC/UMA 资源 churn） | 今日 ~60+ 次创建/销毁（7 次 TP4 重启 + ~50 nccl-tests 运行 + 构建容器）；14:03 首次 nccl-tests 因生产占显存 OOM 失败（无早期基线可对照） | **首选候选**：每次 init 走 slow-path（资源分配变慢/竞态窗口变宽）→ 睡眠态闩锁概率上升。需 E4 判决【推断】 |
| 采样器自身负荷 | sampler2 CPU 5%（13:49s/4.5h），per-CPU 中断增量微小 | 排除【实测】 |
| 残留进程 | ps 无 nccl/ringopt 残留 | 排除【实测】 |
| 统计噪声 | §3.2 | **不可排除**——n 小，3/10→0/10 未达显著【推断】 |

补充：journalctl（since 14:00）无 SMMU/IOMMU 故障记录；PSI（cpu/mem/io）近零；无 kswapd 压力迹象【实测】。

---

## 5. 生产 e2e 影响评估（任务 4，A2 行动项）

### 5.1 可用证据【实测】

| 时段 | 生产状态 | 数据 | 判读 |
|---|---|---|---|
| 14:22–16:31（R1 前） | 运行（慢轮现象在案——采样器因此启动） | phase1/thr4096：verify probe TTFT 3.92/2.88/**11.98s**（14:24–14:31） | 11.98s 尖刺为当日最重单点，但恰逢生产 verify + 无仪器化慢轮对齐（阶段2结论），归因未定 |
| 16:02–16:49（expD R1–R7） | 7 次重启抽签 | 首请求 TTFT 慢簇 4.25–4.43s×4 / 中簇 3.07–3.48s×3；**稳态** TTFT 2.96–2.98 vs 2.88–2.90s；PR 2753–2768 vs 2832–2853 | 全部在历史带内；双峰为**既有**重启级模式方差（±8–13%，存在多日），非今日新增性能坍塌 |
| 18:09–18:53 | **生产停机**（W1 窗口） | 无生产数据 | stall 最严重时段无生产观测——盲区 |
| 18:53–至今 | 运行（18:59 healthy） | 容器日志仅启动路由/健康检查，**无推理流量** | 18:53 建立的 communicator 是否落入 stall 模式**未验证** |

### 5.2 判定

1. **生产稳态未观测到灾难性 stall**【实测+推断】：若 17–20ms/AR 的 stall 作用于生产 prefill（86 次 AR 量级），PR 将劣化 >30%、稳态 TTFT 将 +1.3s——实测稳态差仅 ~3%（且属既有模式方差）。首请求 +1.2s 差为 warmup 一次性成本（cudagraph/JIT/first-touch，expD 口径）。
2. **"生产同概率中招"的准确口径**：W1 证明的是生产**库+4ch 配置**在 nccl-tests **形态**下同概率中招（环境级，非库级）；**不等于生产 vLLM 形态正在中招**。
3. **形态差异点（回答"若生产未中招，差异在哪"）**【实测】：
   - communicator 生命周期：nccl-tests 每运行新建/销毁（抽签机会 ~60 次/日）vs 生产一次建立长期存活（抽签机会 = 重启次数）；
   - 生产有 CUDA graphs 捕获 + vLLM 完整初始化路径（NCCL 调用全部 graph 内捕获，warmup 期高频唤醒 proxy——若睡眠态会被强制"热身"退出，则生产天然免疫）【推断】；
   - NCCL env 差异：生产设置 NCCL_ALGO=RING/MAX_NCHANNELS=4/IB_RETRY_CNT=7/IB_TIMEOUT=1000/BUFFSIZE=8M/GID_INDEX=3 等（IB_TIMEOUT 非上游标准 env 名，是否被定制版消费未验证）。
4. **风险敞口**：18:53 生产重启发生在 stall 概率高峰期（18:47–18:52 实测 0–10% clean）——若抽签传染生产形态，当前生产 communicator 可能已中招，但生产空闲无法显现。**E5（对当前生产跑 panorama 3 轮）是 A2 的闭环判决**。
5. **监控盲区（附带事故）**【实测】：aicad-prometheus-1 于 16:33:13Z exit(0)（R2 dockerd/containerd 重启连带），未自动拉起；16:33 至今（19:20+）无 Prometheus 数据，Grafana 数据停滞。**建议 team-lead 安排 `docker start aicad-prometheus-1`**（并检查 restart policy）。本轮生产影响评估被迫依赖 expD 快照而非时序指标。

---

## 6. 根因候选排序

| 排名 | 候选 | 机制 | 证据强度 | 判决实验 |
|---|---|---|---|---|
| **RC1** | **NCCL proxy 线程睡眠轮询态抽签（GDR=0 host staging 放大）**：communicator 建立期 proxy 落入 ~1ms 定时唤醒的低功耗进度模式且不退出（或极慢退出）；AR 数据每周期推进一个量子 | 1ms 精确量子 + stall 期 proxy 核 1kHz arch_timer + clean 期无 + per-run 闩锁 + 无重传无错误无 IRQ 活动 + 库内存在 nanosleep/ProxyCtrlSleep 机制 + 与库/通道数/pin 无关 | 【实测旁证，中高】 | E1/E2/E3 |
| RC2 | proxy/NCCL 线程 pin 漏网落点（libncclpin 竞态，expD 实锤 4/7 重启漏网）→ 落入忙核被调度节流 | 与 expD 漏网率同量级；但 W1 NO_PIN 对照阴性 + R1 前后 stall 同签名 + proxy 核（8/9）W1 期间近零负载（无竞争对象） | 【弱】——除非 NO_PIN 下调度器仍把 proxy 放 8/9（隔离核最空，对照失效可能） | E3 |
| RC3 | communicator 生命周期累积劣化（驱动/SMMU/pgtable/UMA 资源 churn）→ init slow-path 概率上升（解释 25%→5%） | 时间相关方向一致；统计未达显著；无直接资源证据（slab/资源计数无历史快照可对照） | 【推断，弱-中】 | E4 |
| RC4 | NIC/织物（PFC/ECN/队列/中断风暴） | 全部计数器阴性 + 直连无交换机 + W1 窗口 IRQ 近零 | 【排除】 | — |
| RC5 | SMMU/页映射粒度抽签（THP=madvise、iommu 开启下 MR 页表 4KB vs 2MB） | 环境事实成立；但无法解释精确 1ms 量子与 proxy 核 1kHz 信号；碎片证据不支持快速劣化 | 【弱】 | E3（CUMEM_HOST 开关） |

**机制统一图（推断）**：
```
GB10 平台 GDR=0（驱动不支持）──→ AR 数据面 = NCCL proxy 线程（host staging，C2C+ibverbs）
                                        │
                    communicator 建立即抽签 ──→ 忙轮询态（clean，0.13ms/AR）
                                        └──→ ~1ms 定时睡眠轮询态（stall，13–31ms/AR，1ms 量子）
                                                ↑ init 期空闲/慢路径触发闩锁（RC1）
                                                ↑ 生命周期累积使抽签恶化（RC3，未定论）
生产形态：communicator 长寿命 + CUDA graph 高频唤醒 → 稳态未见灾难 stall（待 E5 确认）
```

---

## 7. 需窗口判决实验（只列清单，不执行）

> 全部为 nccl-tests 级微操作（无生产改动）或纯读操作；执行须与并行 GPU 任务排程协调（GPU/网络独占）。

| # | 实验 | 设计 | 判据 | 成本 |
|---|---|---|---|---|
| **E1** | **stall 运行在线抓取 proxy 线程状态（RC1 直接实锤）** | 跑 probe 系列（~6s/臂），每臂运行期间对 nccl-tests 进程所有线程抓：`/proc/PID/task/TID/{wchan,status,stat,sched}`（voluntary_ctxt_switches 增率）、`perf stat -t TID` 若可用 | stall 臂 proxy 线程 wchan=hrtimer_nanosleep 类 + 自愿切换 ~1000/s → RC1 实锤；忙轮询（wchan=0，0 切换）→ RC1 推翻转 RC2/RC5 | 低（probe 已有，纯读附加） |
| **E2** | NCCL proxy 轨迹日志 | probe 臂注入 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,PROXY`（或 TRACE），diff clean/stall 臂 proxy 建立与进度日志（sleep/wake/低功耗转换行） | 找到模式闩锁的具体条件行 | 低 |
| **E3** | env A/B 判决 | 10 臂 ×10 次统计 clean 率：(a) 基线；(b) `NCCL_CUMEM_HOST_ENABLE=0`；(c) `NCCL_IB_MQP_RETRY_SLEEP_MSEC=0`；(d) `taskset` 强制整个测试进程独占 8-9（proxy 与主线程同核）；(e) `NCCL_IB_QPS_PER_CONNECTION=1` | 任一臂 clean 率显著变化 → 定位抽签变量归属（proxy 睡眠/内存分配/QP 资源/pin） | 中（~10 分钟） |
| **E4** | 退化复现与复位 | 连续 50 次 probe，每 10 次记录 clean 率 + `/proc/slabinfo`、`/sys/class/infiniband/*/ports/*/counters` 快照；条件允许时对照重启后首 20 次 | 退化曲线复现 + 与某资源单调增长相关 → RC3 实锤；reboot 后 clean 率复位 → 泄漏/累积确认 | 中（~30 分钟 + 需重启窗口） |
| **E5** | **生产 A2 闭环判决** | 对当前生产（18:53 代 communicator）跑既有 panorama 3 轮（4K prompt） | TTFT/PR 带内（≈2.9s/2750–2850）→ 生产未中招，stall 为 nccl-tests 形态特有（comm 生命周期差异）；显著坍塌 → 生产中招，需立即处置（重启避让或修复） | 低（既有工具） |
| E6 | strace 佐证（可与 E1 合并） | stall 运行期 `strace -c -p <proxy TID>` 5s | syscall 直方图出现 ~5000 次/5s nanosleep/clock_nanosleep → RC1 实锤 | 低 |

**修复方向预研（供 team-lead 排程，不在本任务范围）**：若 E1/E2 实锤 RC1——(a) 排查 NCCL 2.30.7 proxy 睡眠机制的触发条件与退出条件（定制版可打补丁：proxy 建立后强制禁用睡眠态/缩短唤醒周期）；(b) 环境变量规避（E3 中有效的项）；(c) 升级评估：驱动/CUDA 栈若未来版本支持 cuMemGdrSupport=1（GPUDirect），整个 host staging 依赖消失——根治路径。

---

## 8. 附带发现与移交事项

1. **Prometheus down（监控事故）**：aicad-prometheus-1 exited 16:33:13Z（R2 重启 dockerd 连带），未自愈。建议 team-lead：`docker start aicad-prometheus-1` + 检查 restart policy（与 alertmanager/grafana 的 compose 期望对齐）。恢复后回补 16:33–now 的指标评估生产影响。
2. **W1 报告修订建议**：§4.4 行动项 1"交换机侧 PFC/ECN 配置"应撤销（直连拓扑无交换机；NIC 侧已证阴性）；资源转投 §7 E1–E5。
3. **采样器数据使用规范**：irq_source.tsv 首帧（14:30:14）为开机累计伪影，任何分析须跳过；两个采样器仍在运行（stop_irq/stop_irq2 控制），建议保持至 E1–E5 判决完成。
4. **NFS over RoCE 口**（<NODE_IP>/24 等四条直连子网）：模型加载产生 80–100 万中断/分钟（当前落 0-4,10-14）。与 NCCL 共享 NIC/中断带——重启期无冲突记录，但若未来在服务期发生大 NFS 流量（如模型热切换）会与 A725 带上的守护进程竞争，列入观察项。
5. **本轮取证脚本与数据**：本机 `_envstall/{storm.py,trend.py,quantum.py,quantum2.py}`（对应远端 /tmp/_envstall_*.py，纯读分析）；原始数据均取自四节点 /tmp/_slowround/（采样器）与 /tmp/_ringopt/v5/logs/（W1 资产），零改动。

## 9. 证据索引（行号/时间戳级）

- GDR：01/02/03/04 各 rank 日志 `Symmetric memory ... cuMemGdrSupport 0`（W1 日志，时间戳 56:56–58:58 各帧）；`Connected all rings, use ring PXN 0 GDR 0`。
- PFC：四节点 `ethtool -a` RX/TX off（19:05–19:15 UTC 本轮）；`ethtool -S` pause/buf_discard 数值见 §2.1。
- R1：01 `/proc/irq/346-429/smp_affinity_list` 全量 = 0-4,10-14（本轮 19:07）；r1_irq_migrate.sh 源码；expd-r123 报告 §1（16:33 执行）。
- 1ms 量子：`/tmp/_ringopt/v5/logs/probe_prod_4ch_1_r0.log` all_ms 行；probe_stats.log stat_v5on8_* 系列。
- proxy 核 1kHz 信号：`/tmp/_slowround/irq_source.tsv`（epoch 1787422705±、1787423083±、1787423496±、1787424456± 等，CPU 8/9 的 arch_timer 行）。
- NFS/comp IRQ：01 `ss -t state established`（<NODE_IP>:nfs）；irq_source.tsv 16:36–18:56 各尖峰分钟。
- 采样器伪影：sampler2 启动时刻 14:30:14（ps + TSV 首帧 epoch 1787409014）与首帧累计值比对。
- Prometheus：02 `docker inspect aicad-prometheus-1`（started 08-17, finished 2026-08-22T16:33:13Z）。
- 生产带内证据：expd-r123-2026-08-23.md §2.1/§2.2（R1–R7 全量数据）。
- 内存/内核：01 /proc/cmdline（isolcpus=8-9 rcu_nocbs=8-9 iommu.passthrough=0 init_on_alloc=0）、/proc/meminfo、/proc/buddyinfo、THP=madvise（19:12）。
- 库二进制：`strings /opt/nccl-ringonly/libnccl.so.2.30.7`——`NCCL version 2.30.7+cuda13.0`、`nanosleep`、`sched_yield`、`ncclProfilerProxyCtrlSleep`、`NCCL_IB_MQP_RETRY_SLEEP_MSEC`、`NCCL_SOCKET_POLL_TIMEOUT_MSEC`。

---

*本报告全部结论以【实测/实测旁证/推断/协议】四档口径标注；判决依赖 §7 窗口实验，未执行任何主动负载。*
