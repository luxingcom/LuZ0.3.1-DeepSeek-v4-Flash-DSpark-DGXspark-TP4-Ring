# 慢轮（间歇性慢请求/慢模式）根因调查报告

**日期:** 2026-08-22
**作者:** Archi（系统架构师）· engineering-advance 团队
**对象:** DGX Spark 4 节点 TP4 集群（GB10/sm_121a）重启级模式方差（臂间 ±8-13%）+ 轮内间歇性慢轮（~2000 vs ~2550 tok/s）
**口径标注:** 【实测】= 本审计直接取证；【推断】= 由证据链推论；【协议】= 需窗口的判决实验设计

---

## 0. 执行摘要

| 项 | 结论 |
|---|---|
| 重启级模式方差（±8-13%）头号假说 | **libncclpin 按名 pin 间歇漏网 + 线程落点抽签**（01 上代实测有 EngineCore 名线程越出 15-19 规则带；当前代四机全部正常——pin 生效按重启代次存在抽签性）【实测旁证+推断】 |
| 模式方差并列假说 | NCCL channel→NIC 口（环网 2 收 2 发）映射每重启抽签【推断】 |
| 轮内慢轮 | **未定论**：捕获到的唯一仪器化臂为快模式臂（0 慢轮），无慢轮样本可对齐；候选=间歇干扰类（守护进程活动/embedding 共租请求/API 路径），支持证据为快臂全程无干扰亦无慢轮【实测阴性】 |
| 已下修 | 原"NIC IRQ 风暴打断 NCCL"假说：NCCL 忙轮询 CQ，prefill 期 NIC 完成中断不形成风暴（阶段2实测）；但 mlx5 IRQ 绑 5-9 含隔离核 8-9 仍是设计缺陷，arch_timer ~375-725Hz/核常驻所有核（isolcpus 未配 nohz_full） |
| 关键诊断口径（新） | 慢轮判定必须看 **steps/s**（同 prompt 下每步速率），tput 方差可能只是 MTP 接受率随机性（阶段2实测：de C1 轮 tput 方差 36.6% 但 steps/s 恒定 19.6-20.1） |
| 已排除 | 电源/时钟/热降频、irqbalance、RPS、GPU 内存压力、JIT/SM/RDMA 字节差、libncclpin 部署不一致 |
| 建议下一步 | 判决实验 B（慢轮在线抓取，采样器已四节点就位）+ 实验 D（多臂重启抽签复现），见 §5 |
| 廉价修复（卫生级） | IRQ 迁离 8-9（可逆零停机）；守护进程 AllowedCPUs；慢轮抓取期间维持采样器运行 |

---

## 1. 阶段1：四节点系统级审计（只读）

审计时间：2026-08-22 13:50–14:20 UTC（生产重启窗口期间，采样避开了重启进行中的节点；当前代进程 14:04 起稳定后完成核验）
原始数据：本机 `_audit/audit{,2,3}-dgxspark0{1,2,3,4}.txt`

### 1.1 CPU 拓扑与隔离配置【实测】

- 四节点一致：`isolcpus=8-9 rcu_nocbs=8-9`（内核 cmdline + /sys/devices/system/cpu/isolated 双确认）
- CPU 分布：0-4/10-14 = A725 @2808MHz；5-9/15-19 = X925 @3900MHz；NUMA 单节点
- cpufreq governor：全部 `performance`【实测】
- 温度：四节点 thermal zone 63-68°C，无过热事件【实测】

### 1.2 NIC IRQ 亲和性 —— 头号发现【实测】

> ⚠️ 阶段2修订：本节配置冲突与累计落位均为事实，但"prefill 期 IRQ 风暴打断 NCCL"的机制推论已被阶段2实测下修（NCCL 忙轮询 CQ，完成中断不形成风暴，见 §3.3）。配置缺陷本身仍成立，作为卫生级修复项保留（R1）。

- **四节点全部 84 个 mlx5_comp/async IRQ 的 smp_affinity = 5-9**（含隔离核 8-9）
- 配置来源：`mlx-irq-pin.service`（四节点 active）执行 `/usr/local/sbin/mlx-irq-pin.sh`，注释自述依据 `plan-cpu-cluster0-nccl-pin-2026-08-10.md`："①中断不打隔离核 0-4(A725/NCCL轮询) ②中断处理与NCCL同集群0共享L3(12MB)"
- **设计冲突实锤**：该方案设计时 NCCL 轮询核是 0-4；现行 shim v8 已改为 NCCL→8-9，但 IRQ 绑定 5-9 未同步迁移 → **NIC 中断直接打进 NCCL 专用隔离核 8-9**
- 中断实际落位（01，开机 5 天累计）：
  - CPU5: comp0/4/5/14/19（comp5=30.1M 次，全队列最高）
  - CPU6: comp1/6/10/15（comp1=4.3M）
  - CPU7: comp7/11/16（comp7=2.2M）
  - **CPU8: comp2/8/12/17（comp17=7.3M、comp12=1.5M、comp2=1.4M）**
  - **CPU9: comp3/9/13/18（comp9=14.4M、comp13=10.1M、comp3=1.1M）**
- NET_RX softirq 累计（isolcpus 只隔离任务调度，不隔离 IRQ/softirq）：
  - 01: CPU5=37.9M, CPU13=33.1M(!), CPU9=26.2M, CPU6=17.4M, CPU8=12.0M
  - 02: CPU6=49.5M, CPU5=41.6M, CPU9=13.2M, CPU13=12.4M, CPU8=5.6M
  - 03: CPU5=9.1M, CPU6=31.3M, CPU9=16.1M, CPU7=16.2M, CPU8=7.0M, CPU13=4.9M
  - 04: 同 03 量级
- 空闲期 5s 增量采样（01）：**CPU8 +5213 为全核最高**（其余核 1-3K）——即使空闲，隔离核仍在持续吃中断
- RPS 全部关闭（rps_cpus=0）→ 软中断留在 IRQ 核处理，无扩散【实测】
- irqbalance：四节点 inactive → affinity 稳定不被覆盖【实测】

### 1.3 shim v8（libncclpin.so）生效核验【实测】

- 四节点 `<INSTALL_DIR>/lib/libncclpin.so` MD5 = ce43c688c5164ac7efd5105c94fdab77（v8）✅ 一致
- shim-deploy.sh check 输出 [MISMATCH] MISSING 为该脚本自身 SSH 免密/主机密钥问题（root@node01 Permission denied），**非实际部署缺失**
- 当前代（14:04 重启后）四节点 Worker 进程线程级 affinity 分布完全一致：
  - 主线程 → 5-9
  - NCCL 线程（pt_nccl_watchdg/heartbt、NCCL IbAsync 0-3、NCCL Progress、NCCL Service/RAS/UDS，共 25 线程）→ **8-9**（实测 PSR 全部落在 8/9）
  - 其余 35 线程（gloo、VLLM::Worker、CUDA driver 等）→ 5-19
  - EngineCore 进程 → 15-19；vllm 前端 → 5-9
- **上一代（01，13:33 启动代）异常**【实测】：名为 `VLLM::EngineCor` 的线程 TID 778833 实测 PSR=5（越出 15-19 规则带）→ libncclpin 按线程名 pin 存在间歇性漏网；该代线程散布 6,7,10,11,13,14,15-19
- 推论：pin 生效与否（及漏网线程落点）按重启代次存在抽签性【推断】

### 1.4 宿主守护进程 CPU 污染全景【实测】—— 确认 threshold-retester 头号线索

| 守护进程 | 线程数 | affinity | 实测落位 |
|---|---|---|---|
| dockerd | 56-58 | 无 pin（0-19） | 大量落 15-19（01: 15×7、16×3、17×2、18×3、19×3）|
| containerd | 23-28 | 无 pin | 落 5-19 含 15-19（01: 15×3、17×2、18×3）|
| dcgm-exporter | 34-40 | 无 pin | 落 15-19（01: 15×4、17×4、18×4、19×2）|
| docker-proxy | ×24+ 进程 | 无 pin | 全带游走 |
| containerd-shim | ×17 进程 | 无 pin | 全带游走 |
| postgres | ×18 进程 | 无 pin | 全带游走 |
| prometheus（02） | 26 | 0-19 | 落 5-19 含 15-19 |
| monitor_tp4_* | 1 | 无 pin | 全带 |

- **"CPU 15-19 非真隔离"确认**：EngineCore 专属带 15-19 上同时常驻 dockerd/containerd/dcgm-exporter/postgres 数十线程
- 当日干扰源实录：01 上 `docker pull eugr/spark-vllm-b12x`（fi-eugr-prep 活动）运行中，镜像解压为 CPU 密集操作，containerd/dockerd/docker-proxy 均无 pin

### 1.5 GPU 共租发现【实测】

- 03/04 各运行 Qwen3-Embedding-0.6B 服务（docker，port 8022，已运行 1.5h+）：
  - GPU 显存 5750 MiB；TP4 Worker 99.3-99.6GB → 合计 ~105/121GB（有裕量，暂非内存压力问题）
  - 其 EngineCore 102 线程 + pt_nccl 线程**全部无 pin**，实测游走 PSR 1,2,4,5,6,12,15,17（侵入 EngineCore 带 15-19 与 NCCL/前端带 5-9）
  - 若 bench 期间有 embedding 请求 → GPU SM 时间分片 + CPU 带内竞争 → 慢轮候选
- 01/02 无共租（仅 TP4 + postgres）

### 1.6 API 路径软中断集中【实测】

- bench 客户端 → 01:8000（docker bridge/veth/docker-proxy 链路）
- NET_RX 累计集中在 A725 慢核：01 CPU13=33.1M（2808MHz），02 CPU13=12.4M
- docker-proxy 无 pin 游走，与软中断同核竞争 → 端到端 tok/s（含 streaming）抖动候选

### 1.7 已排除/降级项【实测】

| 项 | 证据 | 结论 |
|---|---|---|
| 电源/时钟/热 | dmesg 无 throttling/power cap 事件；governor=performance；63-68°C | 排除（负载期需 dmon 复核，阶段2） |
| irqbalance 覆写 | 四节点 inactive | 排除 |
| RPS 扩散 | 全网口 rps_cpus=0 | 排除（软中断集中于 IRQ 核） |
| GPU 内存压力 | 03/04 105/121GB | 降级（裕量 ~15GB） |
| libncclpin 部署不一致 | 四机 MD5=v8 一致 | 排除 |
| GB10 时钟策略切换 | 无 dmesg 痕迹、无静默期（前序） | 排除 |

---

## 2. 假说排序表（阶段2修订版）

| 排名 | 假说 | 机制 | 证据强度 | 可证伪实验 |
|---|---|---|---|---|
| H2↑ | **libncclpin 按名 pin 间歇漏网 + 线程落点抽签**（重启级模式方差头号候选） | hook 竞态导致关键线程（调度循环/sampler/NCCL Progress）漏 pin 或落 5-19 拥挤核 → 本臂整体慢 ±8-13%；warmup 首 prefill TTFT 即暴露 | 01 上代越界实测【实测，单点】+ 当前代四机正常对照【实测】 | 实验 B/D：快/慢臂逐线程 Cpus_allowed_list 审计 + py-spy dump |
| H1↓ | **NCCL channel→NIC 口映射 / 通道资源抽签**（模式方差并列候选；原 IRQ 机制下修） | 环网 2 收 2 发，每重启 channel→端口/资源分配变化 → 有效带宽/延迟模式变化 → 臂级模式【推断】 | 环网拓扑事实【实测】+ H1 原 IRQ 风暴机制被忙轮询事实削弱【实测】 | 实验 C：NCCL_DEBUG=INFO 通道建立日志 × 臂模式 × RDMA 分口流量对齐 |
| H3 | **守护进程污染 5-19 全带**（慢轮候选，需干扰事件） | dockerd/containerd/dcgm/docker-proxy/postgres 无 pin 游走，间歇活动（scrape/checkpoint/docker pull 解压/镜像 GC）撞关键核 → 轮内慢轮 | 线程落位全景【实测】+ 快臂守护进程平静【实测】（未观察到慢轮，自洽但非判决） | 慢臂 host_daemon_cpu 对齐（采样器已就位）；docker pull 等 ops 活动日历 × 慢轮时刻 |
| H4 | **03/04 embedding 共租**（慢轮候选） | bench 期间 embedding 请求 → GPU SM 分片 + 无 pin 线程侵入 5-19 | 共租事实【实测】，请求时刻未知 | embedding 访问日志×慢轮时刻对齐；dmon 慢轮时刻他进程 SM 占用 |
| H5 | **API 路径 veth/软中断集中 A725 慢核 CPU13** | 客户端 token streaming 经 docker-proxy/veth，NET_RX 集中 CPU13（2808MHz）+ docker-proxy 无 pin + UVM BH 同核 → 端到端吞吐抖动 | 软中断分布【实测】 | 慢轮时刻 CPU13 softirq 增量 vs 快轮 |
| H6(新) | **arch_timer 常驻干扰（~375-725Hz/核，含隔离核）** | isolcpus 未配 nohz_full；隔离核/Engine 带持续吃 timer 中断，本底噪声叠加突发时可能放大尾延迟【推断】 | 中断源归因【实测】 | 实验 A 附带验证：nohz_full=8-9 需重启验证（高风险，暂缓） |

**机制统一图（阶段2修订，推断）:**

```
重启（TP4 restart）
  ├─ libncclpin 按名 pin 竞态 → 关键线程漏网/落点（H2）──────┐
  ├─ NCCL channel→NIC 口映射抽签（H1'）─────────────────────┤→ 本臂"模式"（快/中/慢，±8-13%）
  └─ （其他待排除：UMA 布局、通道数）───────────────────────┘    warmup 首 prefill TTFT 即可测出
臂内（15 轮）
  ├─ 守护进程/ops 间歇活动撞关键核（H3）──┐
  ├─ embedding 请求（H4，03/04）─────────┤→ 轮内慢轮（需干扰事件触发；快模式臂 0 慢轮佐证）
  └─ API 路径软中断（H5）────────────────┘   诊断口径：同 prompt 下 steps/s 变化（非 tput）
已下修：NIC IRQ 风暴打断 NCCL（忙轮询不触发）；de 轮 tput 方差=MTP 接受率（负载非设施）
```

---

## 3. 阶段2：thr4096 仪器化数据合成分析

### 3.1 数据现状与覆盖【实测】

thr4096-adopt 采纳窗口完整时间线（01:/tmp/_thr4096/）：
- 13:56-14:13 baseline_1024（旧 threshold，重启前代进程）
- ~14:04-14:10 threshold 切 4096 + 四节点服务重启（当前代，pin 全部正常）
- 14:24-14:31 verify：probe（TTFT 3.92/2.88/11.98s）+ PR 四档全景 + de 轮 + greedy 对比
- 14:37-14:39 stability 14 轮（4K/16K 交替）+ dmon/宿主CPU/RDMA 连续采样
- ~~重要事件：verify greedy 对比 FAIL（OUTPUT DRIFT，No-Go）~~ → **【终稿修正】假阳性，team-lead 独立核实裁定**：/tmp/_mtp_tune/greedy_ref_run1.json vs run2.json（**同 1024 配置连跑两次**）在完全相同的两个 prompt（reason/zh）上同样 DIFF，其余 4 个稳定 prompt 全部 IDENTICAL——漂移是 batched 推理近平票翻转 + probabilistic 投机解码的既有运行级非确定性，与 threshold 无关（两 prompt 均 <100 tok，低于两种 threshold，无影响通道）。**thr4096-adopt 的 PASS 裁定成立，生产维持 4096 不回滚**。复盘项：质量门判据应改用稳定 prompt 集或 KL 散度，避免朴素逐字比对假阳性（thr4096-adopt 已提）。

本组补充仪器化（填补 thr4096 采集缺口）：四节点部署 per-CPU IRQ/NET_RX 采样器（v1 聚合，2s）+ 中断源归因采样器（v2，2s，delta>100），自 14:16 起连续运行，完整覆盖 verify+stability 窗口。

### 3.2 stability 14 轮对齐分析【实测】

本轮为**快模式臂**：4K TTFT 2.87-2.91s（±0.7%）、16K TTFT 11.54-11.59s（±0.2%），**0 慢轮**；PR 2828-2859 tok/s（+13% vs baseline 2510）。

**逐轮 per-band 中断增量（4 节点合并，数值为总中断数/轮窗口）：**

| 轮 | ctx | TTFT | 8-9(NCCL) | 5-7(前端) | 15-19(Engine) | 13(veth) |
|---|---|---|---|---|---|---|
| r1 | 4K | 2.87 | 9,329 | 16,056 | 31,387 | 7,036 |
| r2 | 16K | 11.59 | 54,537 | 72,953 | 167,800 | 35,126 |
| r3-r13 | ... | ±0.02s | 同量级稳定 | 同量级稳定 | 同量级稳定 | 同量级稳定 |
| r14 | 16K | 11.55 | 53,959 | 75,299 | 154,754 | 33,491 |

**中断源归因（14 轮窗口合并 vs 空闲对照，4 节点）：**

| 中断源 | prefill 期间(127s) | 空闲(120s) | 折算每核速率 |
|---|---|---|---|
| arch_timer | 3.82M（15-19 带独占 1.33M） | 2.49M（15-19 带 1.08M） | **~375-725 Hz/核 常驻，含隔离核 8-9** |
| mem_timer | 253K | 165K | 低 |
| mlx5_comp(RoCE) | 未检出（v2 阈值>100/2s） | 未检出 | **prefill 期间 NIC IRQ 速率低（NCCL 忙轮询 CQ，不触发完成中断风暴）** |
| nvidia(GPU) | ~700 | 0 | 极低（GPU 完成走 busy-poll/kernel 轮询） |

**宿主守护进程 CPU（stability 窗口，01）：** dockerd 0.7%、containerd 1.9%、dcgm-exporter 2.0%、node_exporter 0.7%、containerd-shim 0%——全部平静无尖峰，30s 桶时间线平稳。
**dmon（4 节点）：** sm_avg 71-72%、sm_p95 96%、pwr_avg 49-52W、pwr_max 64-68W——快模式臂的 GPU 画像即"SM 高占用+低功率"（含通信 kernel 自旋），印证该签名不能区分快慢。
**RDMA：** 140s 窗口 xmit 312.6GB、0 错误/丢弃。

### 3.3 阴性发现与假说修订【实测→推断】

1. **H1 量级大幅下修**：NCCL 数据面走 CQ 忙轮询，prefill 期间 mlx5 完成中断不形成风暴——"IRQ 风暴打断 NCCL"不是慢轮主机制。mlx5 IRQ 绑 5-9（含 8-9）仍是**设计缺陷**（隔离核持续吃 timer/偶发中断，且 5-7 与前端共核），但证据不支持其为主因。
2. **arch_timer 常驻干扰浮出**：~375-725Hz/核的中断在所有核常驻（含隔离核 8-9、Engine 带 15-19）——isolcpus 未配 nohz_full；prefill 期 15-19 带每秒吃 ~500+ 次中断。量级小（每次 ~1-2μs）但构成持续本底噪声；对"隔离核纯净度"的既有认知是修正。
3. **de C1 轮间方差（36.6%）完全由 MTP 接受率解释**：tput/tokens_per_step = 19.6/19.7/19.9/20.1 steps/s **恒定**——本臂 decode"方差"是负载随机性（接受长度），非基础设施。这为慢轮判定给出关键诊断口径：**必须看 steps/s 而非 tput**（同 prompt 下 steps/s 变化才是基础设施问题）。
4. **快模式臂 0 慢轮**：慢轮需要干扰事件才触发——支持"间歇干扰"类假说（H3/H4/H5），削弱"稳态机制"类假说。
5. **重启级模式方差的机制收窄**：H1 的 CQ→IRQ 映射抽签机制被忙轮询事实大幅削弱；H2（pin 漏网/线程落点抽签）上升为模式方差头号候选；NCCL channel→NIC 口映射（环网 2 收 2 发，通道到端口的分配每次重启可能变化）为并列候选【推断】。

### 3.4 阶段2结论

- 慢轮根因**未能定论**：捕获到的唯一仪器化臂是快模式臂（0 慢轮），无慢轮样本可对齐。方法论（多源时间线对齐）已验证可用。
- 已建立快模式参考画像（IRQ/守护进程/dmon/RDMA 四源基线），供后续慢臂对比。
- 假说排序修订 + 判决实验协议（§5）为下一步路径；四节点采样器仍在运行（/tmp/_slowround/，stop 文件可控），下次慢臂出现时可立即对齐。

---

## 4. 修复建议（代价/风险评估，阶段2修订）

| # | 措施 | 实施代价 | 风险 | 预期收益 |
|---|---|---|---|---|
| R1 | **IRQ affinity 迁离 8-9**：mlx-irq-pin.sh TARGET 从 5-9 改 0-4（或 0-4,10-14），`echo 0-4 > /proc/irq/N/smp_affinity_list` + 改 service 持久化 | 低（单命令可逆、零停机） | 低（A725 核处理中断稍慢；建议迁 0-4 而非 10-14 避开 veth CPU13） | 卫生级：隔离核 8-9 真正纯净（消除 timer 外的 NIC 中断残余+未来风险）；对慢轮预期收益有限（H1 已下修） |
| R2 | **守护进程 cpuset 限带**：systemd drop-in `AllowedCPUs=0-4,10-14`（dockerd/containerd/dcgm-exporter/node-exporter/prometheus/postgres），docker-proxy/containerd-shim 走 cgroup 级 | 中（逐服务 reload；需验证 A725 峰值余量） | 中 | 消除 H3：15-19/5-9 带真正专属；顺带改善 CPU13 的 UVM BH/veth 拥挤 |
| R3 | **embedding 服务处置**：pin 到 0-4,10-14 或迁出 bench 节点；其 GPU 共租（5.75GB+SM 分片）单独评估 | 低-中 | 低（需确认服务 SLA 归属方） | 消除 H4 |
| R4 | **libncclpin v9**：修复按名 pin 竞态（fork 时全量 reapply / 监听 prctl / 启动后周期 reassert） | 高（开发+四机部署窗口） | 中 | 消除 H2 漏网抽签——模式方差头号候选的根治项 |
| R5 | **nohz_full=8-9 评估**（消除隔离核 timer tick） | 高（内核参数需重启验证） | 中（nohz_full 与 rcu_nocbs 交互、单核任务必须满载才省 tick） | 消除 H6 本底噪声（~375-725Hz/核）；收益待量化，建议排在实验 A 之后 |
| R6 | **API 路径优化**：docker-proxy 换 host network、评估 veth 软中断迁核 | 中 | 中 | 缓解 H5 |

**实施顺序（team-lead 排程裁定）：R1+R2+R3 打包进下一停机窗口搭车执行（不单独动生产）→ R4 与实验 D（5 次重启抽签复现）合并为"下一窗口慢轮专项"提请用户批准 → R5（nohz_full 评估）在实验 A 后评估**

---

## 5. 需窗口判决实验协议

### 实验 A：IRQ 迁移 A/B 判决（验证 IRQ 卫生修复收益，约 40 分钟窗口）
> 阶段2修订：H1 的"IRQ 风暴"机制已下修，本实验降级为卫生级验证（隔离核纯净度对尾延迟的影响），非慢轮主判决。
1. 前置：确认生产 stable，记录当前 /proc/interrupts 基线（四节点）
2. 基线臂：跑 3 轮同 prompt bench（记 TTFT + **steps/s** + 期间 per-CPU IRQ 增量，采样器已就位）
3. 干预：四节点 `for irq in $(grep mlx5_comp /proc/interrupts | cut -d: -f1); do echo 0-4 > /proc/irq/$irq/smp_affinity_list; done`（迁 A725，避开 5-9 全带）
4. 干预臂：同 prompt 再跑 3 轮；判据：TTFT/steps/s 变化 + CPU8/9 中断增量归零
5. 回滚：echo 5-9 恢复（或 systemctl restart mlx-irq-pin）

### 实验 B：慢轮在线抓取（验证 H2/H3/H4/H5，生产运行中只读附加，无需停机）
> 四节点采样器（/tmp/_slowround/irq_percpu.tsv + irq_source.tsv）已部署运行中，随取随用。
1. bench 期间后台循环（10s 周期）：`ps -eLo pid,tid,psr,comm` 存档 + 逐线程 Cpus_allowed_list 审计 EngineCore/Worker（统计越界线程数）
2. 慢轮出现时（**判据用 steps/s 掉档**，同 prompt；勿用 tput——MTP 接受率会假阳性）：`py-spy dump --pid $(pgrep -f EngineCore)` 抓调度循环栈
3. 同时抓：慢轮窗口 per-CPU IRQ 增量（采样器数据）、守护进程 CPU（thr4096 host_daemon_cpu.sh 可复用）、03/04 embedding 访问日志与 dmon 他进程 SM 占用
4. 对齐分析：慢轮 vs 快轮窗口四源对比（工具链已验证：_audit/slowround_align.py）

### 实验 C：NCCL 通道映射取证（验证 H1' 模式抽签机制，需计划内重启窗口）
1. 生产 env 注入 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV`（借用计划内维护窗口；注：原设想的"回滚免费窗口"已不成立——greedy FAIL 系假阳性，生产维持 4096 不回滚）
2. 解析 init 日志：channel 数、每 channel 的 NIC 口/资源分配
3. 重启 ×2-3 臂，同 prompt 各 3 轮；比对：通道映射 vs 臂模式（warmup TTFT）vs RDMA 分口流量（rdma_stab.tsv 格式可复用）
4. 附带：每臂重启后立即抓 EngineCore/Worker 逐线程 affinity（验证 H2 漏网率）——重启后 5 分钟内完成快照

### 实验 D：多臂重启抽签复现（模式方差判决，约 2-3 小时窗口）
> 排程裁定（team-lead）：与 R4（libncclpin v9 修 pin 竞态）合并为"下一窗口的慢轮专项"一并提请用户批准。
1. 固定 prompt 集，连续 5 次完整重启（计划内维护窗口串行做）
2. 每次：warmup 首 prefill TTFT（模式预测器）→ 同 prompt 3 轮 → 重启后 5min 内线程 affinity 全量快照
3. 判据：臂模式（TTFT/tput）与 (a) 越界线程数、(b) NCCL 通道映射、(c) 线程落点分布的相关性 → 定位模式方差的抽签变量
4. 产出：模式方差根因定论 + 修复项验证（libncclpin v9 或通道固定化）

### 采样器运行说明（运维交接）
- 位置：四节点 /tmp/_slowround/（irq_percpu.tsv 每 CPU 聚合、irq_source.tsv 中断源归因，2s 粒度）
- 停止：`touch /tmp/_slowround/stop_irq /tmp/_slowround/stop_irq2`
- 数据量：~5MB/小时/节点，重启自动清除（/tmp）；建议慢轮判决完成前保持运行

---

## 6. 附录：审计方法与数据清单

- 阶段1脚本：`_audit/slowround-audit.sh`（全景）、`slowround-audit2.sh`（sudo 深挖+IRQ 采样）、`slowround-audit3.sh`（NCCL 线程落位+softirq+RPS）、`worker-affinity.sh`（线程级 affinity）
- 阶段2脚本：`slowround-irq-sampler.sh`（v1 per-CPU 聚合）、`slowround-irq-sampler2.sh`（v2 中断源归因，四节点持续运行中）、`slowround_align.py`（多源对齐分析，已验证）
- 阶段1原始输出：`_audit/audit{,2,3}-dgxspark0{1,2,3,4}.txt`（12 份）
- 阶段2原始数据：`_audit/irq_percpu_0{1-4}.tsv`、`irq_source_0{1-4}.tsv`、`stab_4096.json`、`host_daemon_cpu.tsv`；服务器侧 01:/tmp/_thr4096/（thr4096-adopt 产出）
- 关键单点命令存档：taskset 逐 PID 核验、libncclpin.so MD5（四机=ce43c688 v8 ✓）、mlx-irq-pin.sh 源码、bind_irq_forward.sh 源码（legacy）
- 注意事项：节点 taskset 输出为中文本地化（全角冒号），自动化解析需用 /proc 的 Cpus_allowed_list 替代
- 服务器侧残留：四节点 /tmp/_slowround/ 采样器（含数据，停止方式见 §5 运行说明）、/tmp/ 审计脚本——均为 /tmp 临时文件，重启自清，无生产状态修改

## 7. 后记：与并行工作的交叉情报

- ~~thr4096-adopt verify greedy 对比 FAIL（OUTPUT DRIFT，No-Go）~~ → **【终稿修正·team-lead 独立核实裁定为假阳性】**：/tmp/_mtp_tune/greedy_ref_run1.json vs run2.json（同 1024 配置连跑两次）在完全相同的两个 prompt（reason/zh）上同样 DIFF，其余 4 个稳定 prompt 全部 IDENTICAL——漂移为 batched 推理近平票翻转 + probabilistic 投机解码的既有运行级非确定性，与 threshold 无关（两 prompt 均 <100 tok，低于两种 threshold）。**PASS 裁定成立，生产维持 4096 不回滚**（本报告初稿曾误读为正确性回归，已修正）。复盘项：质量门判据改稳定 prompt 集或 KL 散度（thr4096-adopt 已提）。
- 修复排程裁定（team-lead）：R1+R2+R3 打包进下一停机窗口搭车执行；R4（libncclpin v9）与实验 D（5 次重启抽签复现）合并为"下一窗口慢轮专项"提请用户批准；四节点采样器保持运行至窗口。
- baseline C1 decode（13:56-14:13，旧代进程）r0-r3 = 99.9/66.8/92.9/80.9 tok/s，方差 ±20%——注意该代恰为 01 上 pin 异常代（EngineCore 线程散布 5-19），与"漏网抽签→模式方差"假说方向一致（相关性，非因果，样本 n=1）。
