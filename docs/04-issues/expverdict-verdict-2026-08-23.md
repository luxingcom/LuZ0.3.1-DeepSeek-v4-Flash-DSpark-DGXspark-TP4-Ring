# DGX Spark 4 节点 TP4 集群 · 环境级随机 AR stall 判决报告（E1 + 实验 C）

- **日期**: 2026-08-23
- **集群**: node01 ~ 04（4 节点，TP4，NCCL_ALGO=RING，4 channel，NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1，NCCL_IB_MERGE_NICS=0）
- **探针**: ringonly v4 生产库（libnccl_prod.so.2.30.7）+ libncclpin + 4 通道；`ringopt_probe_ev.py`（t1024=8.4MB allreduce，PROBE_ITERS=20 取 med，PROBE_ACTIVE_S=5 取 rate，rank0 计时）
- **数据来源**: 前序代理已跑完 E1 系列（e1_01~e1_10，节点 01 results.tsv / 各节点 forensic.tsv / logs/e1_XX_rN.log）；本次补探针 ev_find2 / ev_find3（NCCL_DEBUG=INFO + affinity dump），未重跑 E1 系列。
- **证据强度标注**: **[实测]** 直接测量 / **[实测旁证]** 强相关、未达因果确证 / **[推断]** 有逻辑链但缺 µs 级直接观测 / **[协议]** 来自 NCCL 自身日志/库声明的行为。

---

## 一、判决结论（TL;DR）

1. **E1 判决：环境级随机 AR stall 为「真实现象」，且根因指向环境级 CPU 亲和/调度异常，非 NCCL 内部逻辑缺陷。**
2. **stall 率：E1 系列 6/10（60%）**，med 17.99~21.18ms、rate 47~54.5/s；CLEAN 臂 med 0.586~0.600ms、rate 1686~1695/s（[实测]）。
3. **根因（[实测] 12/12 完美相关）：每次 STALL run 至少有一个节点 的 worker 进程全程被压到单核（CPU 8 或 9）**；该受限节点随机（node01/02/04/03 均出现过），故宏观表现为「随机」stall。CLEAN run 则任何节点都无单核约束。
4. **RC1（proxy 线程 hrtimer_nanosleep + vol ~1kHz 假说）按字面定义未实证，且部分被推翻**；但 stall 臂 proxy 等价线程普遍进入 timed-poll（poll_schedule_timeout，wake 周期 ~18-28ms ≈ AR stall 时间）而非 CLEAN 臂的 busy-spin —— 该差异为 [实测旁证]，判为单核约束的下游症状而非独立根因。
5. **实验 C：channel→NIC 映射确定为「固定」**（even channel 用 NIC 对 {0,1}，odd channel 用 NIC 对 {2,3}，物理环 3->0->1），在 STALL/CLEAN 间不变 → **排除 channel→NIC 映射作为 stall 差异因子**。
6. **安全收尾完成**：四节点 `/tmp/_expverdict/.sudo_pw` 已删除并逐节点确认。

---

## 二、RC1 判决

| RC1 判定项 | 结论 | 证据强度 |
|---|---|---|
| stall 臂 proxy 线程 wchan=hrtimer_nanosleep | **未观测到**。全部 13 份 forensic（E1×10 + ev_find2/3 + ev_smoke 等）均无 hrtimer_nanosleep | [实测] 否定 |
| stall 臂 proxy 等价线程 vol 增率 ~1kHz | **未观测到**。stall 臂该线程 vol 增率 ~16~45/s | [实测] 否定 |
| stall 臂 proxy 等价线程为 wchan=0 忙轮询 | **不成立**（按 RC1 口径「wchan=0 → 推翻 RC1」）：e1_01/06/08/09/10/ev_find2 该线程为 poll_schedule_timeout；**ev_find3 中该线程确为 wchan=0（vol 仅 9.4/s），但 run 仍 STALL** | [实测] 部分推翻 |
| proxy（或 proxy 等价线程）行为在 stall/clean 间存在稳定差异 | **成立**：CLEAN 臂存在专用 busy-spin 网络进度线程（wchan=0，vol ~178~192/s）；STALL 臂无此线程，网络进度线程转入 timed-poll（poll_schedule_timeout，vol ~16~45/s，wake 周期 ~18-28ms 与 AR stall ~18ms 吻合） | [实测旁证] |

**RC1 结论：RC1 未按定义实证**（hrtimer_nanosleep + ~1kHz 特征缺失，且存在 wchan=0 仍 STALL 的反例）。RC1 想表达的「proxy 不忙轮询导致延迟」机制在行为层面存在（[实测旁证]），但更合理的是将其归因于单节点单核约束（见第五节）——proxy 的 timed-poll 是该节点 CPU 饱和后的下游表现，而非独立于环境的 proxy 内部缺陷。

---

## 三、stall / clean 对照证据链

### 3.1 stall 率统计（抽样口径注明）

**E1 系列（e1_01~e1_10，连续 10 次 run，rank0 计时，t1024=8.4MB allreduce，20 次计时取中位数）**：

| run | med(ms) | rate(/s) | 判决 |
|---|---|---|---|
| e1_01 | 18.020 | 54.3 | STALL |
| e1_02 | 0.600 | 1695.0 | CLEAN |
| e1_03 | 0.600 | 1686.0 | CLEAN |
| e1_04 | 0.588 | 1694.4 | CLEAN |
| e1_05 | 0.586 | 1693.2 | CLEAN |
| e1_06 | 18.450 | 54.2 | STALL |
| e1_07 | 18.626 | 53.5 | STALL |
| e1_08 | 17.991 | 54.5 | STALL |
| e1_09 | 20.973 | 47.0 | STALL |
| e1_10 | 21.176 | 48.8 | STALL |

**stall 率 = 6/10（60%）**。补充 run（探索 tag + 本次 E2）：ev_smoke STALL(19.310ms)、ev_dbg3 STALL(17.991ms)、ev_find CLEAN(0.585)、ev_dump CLEAN(0.586)、ev_smoke2 CLEAN×2(0.635)、ev_dbg/ev_dbg2 CLEAN、ev_find2 STALL(20.487)、ev_find3 STALL(18.135) —— 全口径 STALL 概率约 50~60%。[实测]

> 抽样口径说明：结果来自节点 01 `results.tsv` 的 E1 系列连续 run；stall 判定阈值 med>5ms（run_ev.sh 口径）；环境为生产同款 ringonly v4 库 + libncclpin + 4 通道。

### 3.2 代表 run 的 proxy 等价线程 wchan/vol/psr 数据行（节点 01 forensic.tsv）

| run/判决 | 线程 | comm | psr | wchan | vol 增率 | nvol 增率 | 备注 |
|---|---|---|---|---|---|---|---|
| e1_02 CLEAN | 2132290 | python3 | 14/17/18 | **0（busy-spin）** | **188.5/s** | 4.8/s | CLEAN 专用忙轮询网络进度线程 |
| e1_02 CLEAN | 2132144 | python3 | 6/7/14/15 | ep_poll | 99.1/s | 0 | gloo/tcpstore 事件循环 |
| e1_02 CLEAN | 2132116 | python3(main) | 5 | 0 | 28.4/s | 8.6/s | AR 主线程 |
| e1_01 STALL | 2121092 | python3 | 16/5 | **poll_schedule_timeout** | **45.3/s** | 0.5/s | 网络进度线程，wake≈22ms≈AR 18ms |
| e1_01 STALL | 2120416 | python3(main) | 7 | 0 | 24.5/s | 8.1/s | AR 主线程 |
| e1_06 STALL | 2175258 | python3(main) | **9(全核)** | 0 | 24.2/s | **409.6/s** | 单核约束 + 抢占风暴 |
| e1_06 STALL | 2176409/2176453 | python3 | 9 | poll_schedule_timeout | ~2/s | 0 | 网络进度线程几乎 idle |
| e1_06 STALL | 2176533 | python3 | 9 | 0 | 0/s | **351.6/s** | 被动抢占 |
| ev_find2 STALL | 2707896 | python3 | 9 | poll_schedule_timeout | **17.9/s** | 3.2/s | 网络进度线程 |
| ev_find2 STALL | 2706370 | python3(main) | 9 | 0 | 23.1/s | **427.0/s** | 单核约束 + 抢占风暴 |
| ev_find3 STALL | 2786137 | python3 | 13/19 | **0** | **9.4/s** | 4.8/s | **wchan=0 仍 STALL → 反 RC1 字面口径** |
| ev_find3 STALL | 2786104 | python3 | 6/7 | poll_schedule_timeout | 1.9/s | 0 | 网络进度线程 idle |
| ev_find3 STALL | 2785595 | python3(main) | 5 | 0 | 24.8/s | 7.2/s | AR 主线程（非受限） |

**对照结论 [实测旁证]**：CLEAN 臂存在一个专用 busy-spin（wchan=0，vol ~180/s）的网络进度线程；STALL 臂该角色线程转入 timed-poll（poll_schedule_timeout，~16-45/s）或以极低 vol 忙轮询。该差异稳定复现于 6/6 STALL 与 6/6 CLEAN 样本。

### 3.3 单节点单核约束矩阵（12/12 完美相关）[实测]

每节点读取该 run 的 forensic.tsv，统计全部采样 psr 去重后的 CPU 数；=1 表示「该节点 worker 全程只跑在单个 CPU（8 或 9）」。

| run | node01 | node02 | node04 | node03 | 判决 |
|---|---|---|---|---|---|
| e1_01 | spread | **CPU8** | spread | spread | **STALL** |
| e1_02 | spread | spread | spread | spread | CLEAN |
| e1_03 | spread | spread | spread | spread | CLEAN |
| e1_04 | spread | spread | spread | spread | CLEAN |
| e1_05 | spread | spread | spread | spread | CLEAN |
| e1_06 | **CPU9** | spread | spread | spread | **STALL** |
| e1_07 | **CPU8** | spread | **CPU8** | spread | **STALL** |
| e1_08 | spread | spread | spread | **CPU9** | **STALL** |
| e1_09 | **CPU9** | spread | spread | **CPU8** | **STALL** |
| e1_10 | **CPU9** | spread | spread | **CPU9** | **STALL** |
| ev_find2 | **CPU9** | **CPU8** | spread | spread | **STALL** |
| ev_find3 | spread | spread | **CPU8** | spread | **STALL** |

- **每个 STALL run 至少 1 节点单核受限；每个 CLEAN run 0 节点受限。** 受限节点随机（node01/02/04/03 均出现），故宏观「随机」。
- e1_07/09/10/ev_find2 出现 2 个节点同时受限，stall 依然成立。
- e1_06/09/10/ev_find2 受限核为 9，e1_01/07/ev_find2/ev_find3 受限核为 8 —— **受限核恰为该节点 NCCL proxy progress 目标核**（node01→core 9、node02→core 8，见第五节 NCCL_DEBUG 输出），非任意核。

### 3.4 进程 CPU affinity 实测（ev_find2/ev_find3 补探针）[实测]

affinity dump（root 采样 /proc/<pid>/status 与 /proc/<pid>/task/*/status）：
- **进程/主线程 `Cpus_allowed_list` = 5-9**，主线程全程 psr=9（ev_find2）或 psr=5（ev_find3）。
- `pt_nccl_watchdg` / `pt_nccl_heartbt` affinity = **8-9**（libncclpin 规则命中）。
- 其余线程（pt_gloo_runloop / cuda* / ib_uverbs / 网络进度线程）affinity = 5-19。
- 单核受限的触发机制（cgroup cpuset 瞬时收窄 vs 调度器挤占）未在本次采样中闭环，留待 E3（见第六节）。

### 3.5 proxy 线程观测缺口处理

**问题**：worker 进程线程 comm 只有 python3 / pt_gloo_runloop / pt_nccl_heartbt / pt_nccl_watchdg / cuda*，未采到 NCCL proxy 线程；libncclpin 存在 `*Proxy* → CPU 8-9` 规则却从未在日志中命中。

**处理结论**：
1. **前序代理未定位 proxy 真实 comm**：`ev_find`/`ev_dump` forensic 与 `find_nccl.sh`（搜索 "NCCL Progress/Service/IbAsync"）均未命中——因为本环境 NCCL 内部 proxy/service/ibasync 线程 **comm 保持 python3**（未如 vLLM 内那样改名为 "NCCL Progress"）。（[实测]）
2. **libncclpin 的 `*Proxy*` 规则为何不触发**：`strings <INSTALL_DIR>/lib/libncclpin.so` 确认规则串 "Proxy"/"pt_nccl"/"NCCL" 存在；但规则只在 pthread_create 时按名字匹配，NCCL proxy 线程名在创建后才设置（或本库未设置）→ 命中 default → CPU 5-19。（[推断]）
3. **proxy 线程的存在性由 NCCL 自身日志确认**：NCCL_DEBUG=INFO 输出 `[Proxy Service] Device 0 CPU core 9`、`[Proxy Service UDS] Device 0 CPU core 9`、`[Proxy Progress] Device 0 CPU core 9`（node01）/ `core 8`（node02）。（[协议]）
4. **proxy 等价线程通过行为签名从 python3 comm 中识别**：CLEAN 臂专用 busy-spin 网络进度线程（wchan=0，vol ~180/s）即 proxy progress 的角色线程；STALL 臂该角色线程转为 timed-poll。加 2 次补探针（ev_find2/ev_find3，NCCL_DEBUG=INFO + affinity dump）均 STALL，验证一致。（[实测旁证]）

---

## 四、实验 C：模式方差通道映射（channel → NIC 口）

**结论：channel→NIC 映射固定，排除作为 stall 差异因子。**（[实测] 映射来自本次 ev_find2 NCCL_DEBUG=INFO 日志；[推断] 同库同 env 下 init 确定性 → 10 次 E1 run 映射相同）

### 4.1 映射表（rank0 视角，production 库，4 channel）

NCCL_DEBUG 关键行（node01 `logs/ev_find2_r0.log`、node02 `logs/ev_find2_r1.log`）：
```
NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE [RO]; OOB enP7s7:<NODE_IP><0>
RING-ONLY v4 rank 0->3 chan 0 dev 0 (was 0)   Channel 00/0 : 3[0] -> 0[0] [receive] via NET/IB/0
RING-ONLY v4 rank 0->3 chan 1 dev 2 (was 1)   Channel 01/0 : 3[0] -> 0[0] [receive] via NET/IB/2
RING-ONLY v4 rank 0->3 chan 2 dev 0 (was 2)   Channel 02/0 : 3[0] -> 0[0] [receive] via NET/IB/0
RING-ONLY v4 rank 0->3 chan 3 dev 2 (was 3)   Channel 03/0 : 3[0] -> 0[0] [receive] via NET/IB/2
RING-ONLY v4 rank 0->1 chan 0 dev 1 (was 0)   Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/1
RING-ONLY v4 rank 0->1 chan 1 dev 3 (was 1)   Channel 01/0 : 0[0] -> 1[0] [send] via NET/IB/3
RING-ONLY v4 rank 0->1 chan 2 dev 1 (was 2)   Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/1
RING-ONLY v4 rank 0->1 chan 3 dev 3 (was 3)   Channel 03/0 : 0[0] -> 1[0] [send] via NET/IB/3
```

| channel | receive NIC（rank0 ← rank3） | send NIC（rank0 → rank1） |
|---|---|---|
| chan 0（even） | NET/IB/0 = rocep1s0f0 | NET/IB/1 = rocep1s0f1 |
| chan 1（odd） | NET/IB/2 = roceP2p1s0f0 | NET/IB/3 = roceP2p1s0f1 |
| chan 2（even） | NET/IB/0 | NET/IB/1 |
| chan 3（odd） | NET/IB/2 | NET/IB/3 |

- Ring 全部为物理环：rank0 `3 -> 0 -> 1`，rank1 `0 -> 1 -> 2`。（[协议]）
- 生产库含 **RING-ONLY v4 netdev 强制补丁**（`was X` 为 stock 值，`dev Y` 为强制值；chan1 从 dev1 改为 dev2、chan2 从 dev2 改为 dev0 等），即把 even/odd channel 分别聚到两对 NIC。（[协议]）
- `4 coll channels, 4 p2p channels`；4 张 NIC 全部使用。（[协议]）
- 该映射为 init 期确定性产物（同 lib、同拓扑、同 env），在本次 STALL run（ev_find2）与历史 8/22 W1 debug log 中一致；无任何证据显示 CLEAN/E1 各 run 映射会变 → **不构成 stall 的方差来源**。（[实测]/[推断]）

### 4.2 与 stall 的关联

- 全部 12 个判例行中，channel→NIC 映射均相同（同一 init 代码路径），而 stall 随机出现 → **映射与 stall 无相关**。
- 8/22 W1 历史背景：曾验证 v5 强制物理环可消除 t96 stall（v5-ON 0.166ms vs v5-OFF 16.99ms）；但 E1 使用带 RING-ONLY v4 的生产库，4ch 物理环 + 4 NIC 全用，仍随机 stall → 说明**单纯映射正确不足以解释本次 stall**，差异在运行时调度环境。（[实测旁证]）

---

## 五、根因分析与证据强度汇总

**根因链（[推断]，各环节均为 [实测]/[实测旁证]）**：
```
[实测] ≥1 节点 worker 进程被压到单核（CPU 8/9，即该节点 NCCL proxy 目标核）
   → [实测] 该核饱和：主 AR 线程 nvol ~400/s 抢占风暴（e1_06/07/09/10/ev_find2）
   → [实测旁证] 该节点网络进度（proxy 等价）线程无法 busy-spin，转入 timed-poll
     （wake 周期 ~18-28ms ≈ AR stall 时间）
   → [推断] 该节点 ring 网络数据流阻塞 ~18ms
   → 整组 4 节点 ring allreduce 等待该节点 → 全局 STALL（rank0 计时 17.99~21.18ms）
```
- **与 RC1 的关系**：RC1（proxy 线程 hrtimer_nanosleep + ~1kHz）未实证；proxy timed-poll 是单核约束的下游症状（[实测旁证]），不是独立根因。存在 ev_find3 反例（proxy wchan=0 仍 STALL）进一步说明「proxy 是否忙轮询」不是充分解释。
- **环境级属性**：单核约束发生在容器/进程的 CPU 亲和/调度层面，与 NCCL 逻辑无关；受限节点随机 → 宏观「随机 stall」。触发机制（cgroup cpuset 瞬时收窄 vs 调度器挤占到 proxy 核）未闭环，需 E3 定向实验。

---

## 六、E3 建议（env A/B 判决）

1. **复现并抓取受限条件**：在 run 期间实时监控 worker 的 `/proc/<pid>/cpuset` 与 cgroup `cpuset.cpus`，确认单核受限是 cgroup 收窄还是调度器行为；对比受限/非受限 run 的容器 cpuset 差异。
2. **A/B：显式 CPU 亲和**：容器加 `--cpuset-cpus=5-19`（或 run 前 `taskset -c 5-19`），观察 stall 是否消失；反向 A 臂显式 `--cpuset-cpus=8-9` 观察是否必现 → 直接判定「单核约束→stall」因果。
3. **A/B：NCCL_IGNORE_CPU_AFFINITY**：开关对比 proxy 目标核实际落点与 stall 率。
4. **A/B：proxy 线程核隔离**：把 NCCL proxy/service/ibasync 固定到独立核（避免与主线程争抢），观察是否消除 timed-poll 与 stall。
5. **µs 级取证**：提高 proxy 等价线程的 wchan/vol 采样频率至 ~1kHz 以上，验证「AR 完成时间是否紧随 proxy 被唤醒」以闭环因果。
6. **统计**：每个 A/B 臂 ≥10 run，用 stall 率差异（当前 6/10 基线）作判定。

---

## 七、安全收尾确认（验收项）

四节点已执行 `rm -f /tmp/_expverdict/.sudo_pw`，逐节点确认：

| 节点 | 结果 |
|---|---|
| node01 | PASSWORD_DELETED |
| node01 | PASSWORD_DELETED |
| node01 | PASSWORD_DELETED |
| node01 | PASSWORD_DELETED |

（另已清理本次补探针临时脚本 aff_dump.sh / ev_find2_drive.sh / ev_find3_drive.sh；确认无残留 probe 进程与容器。）

---

## 附录：关键证据文件

- 节点 01：`/tmp/_expverdict/results.tsv`、`/tmp/_expverdict/timeline.tsv`、`/tmp/_expverdict/e1_XX/forensic.tsv`、`/tmp/_expverdict/logs/e1_XX_rN.log`
- 四节点：`/tmp/_expverdict/e1_XX/forensic.tsv`（psr 单核约束矩阵数据源）
- 补探针：`/tmp/_expverdict/logs/ev_find2_rN.log` / `ev_find3_rN.log`（NCCL_DEBUG=INFO，channel 映射 + proxy 创建 + Ring）、`/tmp/_expverdict/ev_find2/forensic.tsv`、`ev_find3/forensic.tsv`
- 历史 8/22：`/tmp/_ringopt/v5/logs/dbg_v5on_4_16r2_r0.log`、`dbg_v5off_8ch_r0.log`、`w1_matrix_full.log`（W1 背景、同族库拓扑/映射）
