# AR stall 生产闭环（E5 + E3）窗口执行报告

- **执行**：雷克斯（Rex）· SRE 工程师（sre-engineer-2）
- **日期**：2026-08-23
- **状态**：**完成** — E5 生产形态闭环判决 + E3 单核约束触发机制定位 + 恢复确认全过
- **任务来源**：主理人（工程总监）下发 — AR stall 生产闭环（E5 + E3）窗口执行
- **形态**：生产停机态下克隆镜像旁路（`LuZ0.3.1-bench-20260823`，digest `85f2149f`），与生产同构；探针类实验复用 expverdict 资产，**免提权方案**（`.sudo_pw` 已按安全要求删除、未重建；非 root 可读 psr / Cpus_allowed_list / cpuset.cpus.effective / vol/nvol，仅 wchan 被 Yama 屏蔽记录 "-"）
- **证据强度标注**：**[实测]** 直接测量 / **[实测旁证]** 强相关 / **[推断]** 有逻辑链 / **[协议]** 库/日志声明

---

## 0. 一页结论

1. **E5 判决：生产形态未中招。** 克隆生产同构容器（tp4-bench-rank0..3，四机 healthy）panorama 四档 x3 轮全部带内（4K PR 2959.6 vs 采纳带 2950.5；16K 2984.1 vs 2943.6；32K 2872.2 vs 2834.2；64K 2642.9 vs 2550.0；4K TTFT 2.77s ≈ 历史带 2.9s）→ **无坍塌**。环境级 AR stall 是**探针形态（nccl-tests 短寿命 communicator）特有**，生产长寿命 communicator 稳态免疫。
2. **E3 判决：单核约束触发机制 = 调度器/系统级（非 cgroup cpuset 收窄），需 cpuset 显式方案。**
   - 全程 **STALL 34/34 run 至少一节点单核受限（CPU 8/9）、CLEAN 16/16 run 零受限** —— 单核约束与 stall **完美相关**（比 E1 的 12/12 更完整，五臂全口径）。
   - run 期 cpuset 监控：四臂自然运行（a/b/c/e）`cpuset.cpus.effective` **恒为 0-19**、`proc_cpus_allowed=5-9` 不变 → **排除 cgroup cpuset 瞬时收窄**；单核受限是进程/线程在既有 affinity 内 **psr 塌缩到 NCCL proxy 目标核（8/9）** 的调度器行为。
   - env A/B：arm(b) CUMEM_HOST_ENABLE=0 clean 60%、arm(c) MQP_RETRY_SLEEP=0 clean 50%、arm(e) QPS_PER_CONNECTION=1 clean 40%（基线 10%）——方向性改善但未消除 stall（n=10 未达显著性）；**arm(d) 显式 pin 8-9 → 0/10 clean + med 27-31ms（比基线 stall 18ms 更重）**，确定性复现并加重 stall → 单核塌缩到 proxy 目标核是**因果级触发**。
3. **恢复确认全过**：tp4-bench-* 四机无残留、无探针进程/容器、`vllm-healthcheck.timer` 保持基线 inactive+enabled、生产脚本 md5 与窗口前一致（head 2b66686b / worker 88d3fbe6）、8001 空闲、无 `.sudo_pw` 残留。

---

## 1. E5 判决：生产形态是否中招

### 1.1 执行形态（生产同构）

- 克隆镜像 `LuZ0.3.1-bench-20260823`（digest 85f2149f）四机在位；复用既有 bench 资产（`start_tp4_head.bench.sh` + 各 worker bench 脚本），diff 保真门 4/4 PASS（仅「容器名 tp4-bench-rankN」「镜像 tag」两处差异），check_vllm_script 4/4 PASS。
- 容器：`tp4-bench-rank0`(01)/`rank1`(02)/`rank2`(04)/`rank3`(03)，head-first 启动，四机 **Up (healthy)**。
- 启动核验（与生产逐项对齐）：`VLLM_MOE_W4A4=2`、`VLLM_B12X_SHARED_WRAPPER=1`、threshold 4096、**flashinfer 0.6.16**、util 0.82（weight 45.32 GiB）、MTP dspark n=7、KV 5,805,111 tokens、`/health` 200。

### 1.2 panorama 四档 x3 轮（唯一 nonce，输出 1 token，TTFT/PR）

| 档位 | LuZ0.3.1 采纳带（PR） | E5 实测 PR | Δ vs 采纳带 | TTFT 中位 | 判定 |
|---|---|---|---|---|---|
| 4K（8.2K tok） | 2950.5 | **2959.6** | +0.3% | **2.77s** | ✓ 带内 |
| 16K | 2943.6 | **2984.1** | +1.4% | 10.98s | ✓ 带内 |
| 32K | 2834.2 | **2872.2** | +1.3% | 22.82s | ✓ 带内 |
| 64K | 2550.0 | **2642.9** | +3.6% | 49.60s | ✓ 带内 |

- 4K 逐轮 TTFT：3.0s / 2.8s / 2.7s（PR 2757 / 2960 / 3003）——无任何一轮坍塌（历史 stall 若作用于生产 prefill，PR 将劣化 >30%）。
- **判决：生产形态未中招**（[实测]）。E5 直接闭环 A2 遗留——当前 LuZ0.3.1 生产长寿命 communicator 稳态免疫环境级随机 AR stall；stall 为 nccl-tests 探针形态（短寿命 communicator）特有。

---

## 2. E3 判决：单核约束触发机制定位

### 2.1 方法学

- **免提权方案**：`.sudo_pw` 已删、sudo 需密码 → E3 全部 no-root。实测验证非 root 可读 `/proc/<pid>/stat`(psr)、`task/*/status`(Cpus_allowed_list + vol/nvol)、cgroup `cpuset.cpus.effective`（r--r--r--）；仅 wchan 被 Yama(ptrace_scope=1) 屏蔽记录 "-"。
- 五臂 A/B，每臂 10 次（arm_d 首次因脚本部署路径问题 nolog，修复后重跑 10 次有效）：
  - (a) 基线（无附加 env）
  - (b) `NCCL_CUMEM_HOST_ENABLE=0`（host CUDA memory allocation 路径）
  - (c) `NCCL_IB_MQP_RETRY_SLEEP_MSEC=0`（proxy 睡眠/retry）
  - (d) `--cpuset-cpus=8-9` 显式 pin 整个进程到 proxy 目标核（docker cgroup cpuset）
  - (e) `NCCL_IB_QPS_PER_CONNECTION=1`（QP 资源）
- 每 run 每节点并行 no-root forensic（psr / cpuallowed / cpuset_eff / vol / nvol），逐线程 ~0.2s 采样。
- 注：arm_b_03 / arm_c_07 两 run 触发 300s watchdog（一节点 rendezvous/清理 hang），rank0 med 仍解析，verdict 保留为 STALL，标注于数据表。

### 2.2 clean 率 A/B 表

| 臂 | 变量 | clean | stall | clean% | med_ms 范围 | 判读 |
|---|---|---|---|---|---|---|
| a | 基线 | 1 | 9 | **10%** | 0.96 / 17.99–23.995 | 复现环境级 stall（E1 6/10 同族） |
| b | CUMEM_HOST_ENABLE=0 | 6 | 4 | **60%** | 0.585–0.603 / 17.99–23.99 | 方向性改善（n=10 未达显著，Fisher p≈0.06） |
| c | MQP_RETRY_SLEEP=0 | 5 | 5 | **50%** | 0.589–0.919 / 18.08–18.90 | 方向性改善 |
| d | **pin 8-9** | **0** | **10** | **0%** | **27.0–30.98（全部加重）** | **确定性复现并加重 stall** |
| e | QPS_PER_CONNECTION=1 | 4 | 6 | **40%** | 0.589–1.057 / 17.99–22.98 | 方向性改善（弱） |

- 统计口径：n=10/臂，clean 率差异为方向性证据（arm_b 接近显著）；arm_d 为受控对照（强制 pin → 0% clean + med 上移 27–31ms），**因果级**。
- 取舍说明：五臂全跑（未做三臂降级）；arm_b/c 各含 1 个 watchdog run（verdict 取自 rank0 med，如实标注）。

### 2.3 单核约束矩阵（no-root forensic，E1 方法学全口径）

`psr 80%+ 落在 8/9` 判定该节点单核受限；`.`=有采样无受限，`-`=无采样/失败。

| tag | verdict | n01 | n02 | n04 | n03 | | tag | verdict | n01 | n02 | n04 | n03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arm_a_01 | STALL | . | . | . | 8 | | arm_c_01 | CLEAN | . | . | . | . |
| arm_a_02 | STALL | . | . | . | 9 | | arm_c_02 | STALL | . | . | 8 | . |
| arm_a_03 | CLEAN | . | . | . | . | | arm_c_03 | CLEAN | . | . | . | . |
| arm_a_04 | STALL | 9 | . | . | . | | arm_c_04 | STALL | 9 | . | 8 | . |
| arm_a_05 | STALL | 9 | . | 8 | . | | arm_c_05 | STALL | . | . | . | 8 |
| arm_a_06 | STALL | . | 8 | . | . | | arm_c_06 | CLEAN | . | . | . | . |
| arm_a_07 | STALL | . | 9 | 8 | . | | arm_c_07 | STALL | 9 | . | . | . |
| arm_a_08 | STALL | 9 | 8 | 8 | . | | arm_c_08 | CLEAN | . | . | . | . |
| arm_a_09 | STALL | 9 | . | 9 | . | | arm_c_09 | CLEAN | . | . | . | . |
| arm_a_10 | STALL | . | 9 | 8 | . | | arm_c_10 | STALL | . | . | 8 | . |
| arm_b_01 | STALL | 9 | . | 9 | . | | arm_d_01 | STALL | 8 | 8 | 8 | 8 |
| arm_b_02 | STALL | . | . | . | 8 | | arm_d_02 | STALL | 8 | 8 | 8 | 8 |
| arm_b_03 | STALL | . | . | . | 9 | | arm_d_03 | STALL | 8 | 8 | 8 | 8 |
| arm_b_04 | CLEAN | . | . | . | . | | arm_d_04 | STALL | 8 | 8 | 8 | 8 |
| arm_b_05 | CLEAN | . | . | . | . | | arm_d_05 | STALL | 8 | 8 | 8 | 8 |
| arm_b_06 | CLEAN | . | . | . | . | | arm_d_06 | STALL | 8 | 8 | 8 | 8 |
| arm_b_07 | CLEAN | . | . | . | . | | arm_d_07 | STALL | 8 | 8 | 8 | 8 |
| arm_b_08 | CLEAN | . | . | . | . | | arm_d_08 | STALL | 8 | 8 | 8 | 8 |
| arm_b_09 | STALL | 9 | . | 8 | 8 | | arm_d_09 | STALL | 8 | 8 | 8 | 8 |
| arm_b_10 | CLEAN | . | . | . | . | | arm_d_10 | STALL | 8 | 8 | 8 | 8 |
| arm_e_01 | CLEAN | . | . | . | . | | | | | | | |
| arm_e_02 | STALL | 9 | . | . | . | | | | | | | |
| arm_e_03 | CLEAN | . | . | . | . | | | | | | | |
| arm_e_04 | STALL | 8 | 8 | 9 | . | | | | | | | |
| arm_e_05 | STALL | . | . | . | 8 | | | | | | | |
| arm_e_06 | STALL | . | 9 | . | . | | | | | | | |
| arm_e_07 | CLEAN | . | . | . | . | | | | | | | |
| arm_e_08 | STALL | . | . | . | 8 | | | | | | | |
| arm_e_09 | CLEAN | . | . | . | . | | | | | | | |
| arm_e_10 | STALL | . | . | . | 9 | | | | | | | |

**相关统计**：
- **STALL runs 中至少一节点单核受限：34/34（100%）**
- **CLEAN runs 中至少一节点单核受限：0/16（0%）**

### 2.4 cpuset 监控发现（触发条件定位）

| 项目 | 自然臂（a/b/c/e） | pin 臂（d） |
|---|---|---|
| `cgroup cpuset.cpus.effective` | **恒 0-19**（无收窄） | **8-9**（受控 pin） |
| `proc_cpus_allowed`（主线程） | **恒 5-9**（libncclpin 规则，不变） | 8-9 |
| 单核受限形态 | 进程/线程 psr 在 affinity(5-9) 内塌缩到 8 或 9 | 全进程强制 8-9 |
| 受限节点 | 随机（01/02/04/03 均出现） | 全部节点 |
| verdict | STALL（受限节点出现时） | 10/10 STALL、med 加重 |

- **E3 判决（机制归属）**：单核约束触发**非 cgroup cpuset 瞬时收窄**（[实测] cpuset.cpus.effective 全程 0-19）；为**调度器/系统级行为**——既有 affinity(5-9) 内 worker 线程 psr 塌缩到该节点 NCCL proxy 目标核（8/9），受限核即 proxy 目标核。env A/B 表明 proxy 睡眠/内存分配/QP 资源改变**抽签概率**（b/c/e 方向性改善）但均未消除 stall；arm(d) 显式 pin 证明**单核塌缩到 proxy 目标核是因果级触发**。结论走 E3 设计的「单核约束为调度器/系统级，需 cpuset 显式方案」分支。

### 2.5 修复方向（供 team-lead 排程，本窗口未执行）

1. **cpuset 显式方案**：将 NCCL proxy/service/ibasync 线程固定到独立核（避免与主 AR 线程争抢），或隔离 8-9 后禁止非 proxy 线程落入——直接消除「单核塌缩到 proxy 目标核」这一因果条件（E3 判决最直接落点）。
2. env 规避（方向性证据，需更大 n 确认）：`NCCL_CUMEM_HOST_ENABLE=0`（b 臂 60% clean）为最强候选；`NCCL_IB_MQP_RETRY_SLEEP_MSEC=0`、`NCCL_IB_QPS_PER_CONNECTION=1` 次之。
3. **根治**：驱动/CUDA 栈未来版本支持 GDR（cuMemGdrSupport=1）则 host-staging proxy 依赖消失。

---

## 3. 恢复确认（验收项）

| # | 验收项 | 结果 |
|---|---|---|
| R1 | tp4-bench-* 四机无残留（docker ps -a） | ✅ 四机 clean |
| R2 | 探针进程（ringopt_probe_ev / torch.distributed.run） | ✅ 四机无（核对 pgrep 仅自匹配） |
| R3 | vllm-healthcheck.timer 基线（inactive+enabled） | ✅ head inactive+enabled（与窗口前一致；自愈链关闭态） |
| R4 | 生产启动脚本 md5 与窗口前一致 | ✅ head `2b66686b…`（=窗口前）、worker `88d3fbe6…` 三机一致 |
| R5 | 8001 端口空闲 | ✅ free |
| R6 | `.sudo_pw` 残留 | ✅ 四机无（本窗口未创建，全程免提权） |
| R7 | 克隆 tag / bench 脚本留档（不删除） | ✅ 保留（供复核） |

---

## 4. 证据索引

- E5：`/tmp/_arstall_e5/e5_panorama.json` / `e5_panorama.log` / `e5_run_console.log`；`/tmp/_bench_luz031/logs/clone_head.log`、`clone_rank{1,2,3}.log`、`arstall_e5_launch.log`；`/tmp/_bench_luz031/logs/clone_startup.log`（启动核验）
- E3：`/tmp/_expverdict/results.tsv`（五臂 verdict）、`/tmp/_expverdict/arm_{a,b,c,d,e}_*/forensic.tsv`（四节点逐线程 psr/cpuallowed/cpuset_eff/vol/nvol）、`/tmp/_expverdict/logs/arm_*_rN.log`、`/tmp/_expverdict/e3_all_arms_console.log`、`arm_d_series_console.log`
- E3 资产（免提权）：`/tmp/_expverdict/proxy_forensic_nosudo.sh`、`ev_node_nosudo.sh`、`ev_node_nosudo_pin.sh`、`run_ev_nosudo.sh`、`run_ev_nosudo_pin.sh`、`ev_series_nosudo.sh`、`/tmp/_ringopt/v5/ringopt_node2_cpuset.sh`（四机）
- 本地副本：`arstall_work/e3_data/`（results.tsv + 各臂四节点 forensic 脱敏副本）、`arstall_work/e3_assets/`（脚本）

---

## 5. 安全收尾

- **key 日志不回传**：E5 panorama 客户端容器内注入 key（env），服务端日志 `/tmp/_bench_luz031/logs/`、`/tmp/_arstall_e5/e5_*.log` 不含明文 key；本报告/本地副本均不落 key。
- **服务器日志清理建议**：`/tmp/_bench_luz031/logs/clone_head.log` 含 serve 命令明文 key（既有惯例标注）——建议窗口后由管理员清理或重算 key（与前序 luz031 报告一致的处理）。
- **无新增提权资产**：本窗口未创建 `.sudo_pw`，全程免提权，四机确认无残留。

---

*纪律：E5/E3 排程 GPU 独占（tp4-bench 停删后才启动 E3 探针）；全程一次一个 GPU 任务；nvidia-smi/free 双查 OOM 防护；探针结束无残留；恢复后生产保持停机态可随时启动。*
