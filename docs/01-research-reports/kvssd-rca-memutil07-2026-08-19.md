# TP4 KVSSD 事故 RCA 与 mem-util 0.70 修复验证报告

- **日期**: 2026-08-19（UTC+8 时间线；节点均为 UTC）
- **作者**: Rex（SRE Engineer，engineering-assurance）
- **范围**: 现场接管与恢复 / KV 卸载内存核查 / 溢出核对 / NCCL 超时根因 / 0.70 复测验证
- **集群**: 4×DGX Spark UMA（121.6GiB 可用），TP4 vLLM 0.26（anemll/dspark-vllm-gx10:0.2.1-v026.0），deepseek-v4-flash-0731，max-model-len 600000

---

## 一、TL;DR

1. **现场**：接管时 head(01) 容器死、TCPStore 未监听；worker 02/03/04 均卡在 `TCPStore 300s connect timeout` 重试态（health 假阳性）。已完成 **head-first 全量重建**（01 head → 02 → 04 → 03），4 rank 归齐、/health=200、真实推理正常（0.2s）。worker monitor 已恢复。
2. **根因判定**：**UMA 内存耗尽是主因，NCCL ALLREDUCE 300s 超时是被动受害者**，非"先超时后内存耗尽"。证据链：03 在 03:18 UTC 已出现 NVRM `NV_ERR_NO_MEMORY`、avail 03:20 打到 **0.0GB**（均早于 03:23 的 NCCL 超时）；03:54 内核 oom-killer 直接杀掉 `VLLM::Worker_TP`(total-vm 279GB) 与 `VLLM::EngineCor`；03 系统级冻结约 50min（Prometheus 数据中断）。触发组合：**conc3×65536 长上下文并发 + gpu-memory-utilization 0.80 下宿主可用内存仅 4-6GB**。
3. **KV 卸载内存**：CPU 主层实测利用率仅 **0-1%（约 0-20MB / 2GiB）**，不是内存大户；fs 层为主（4 次复测累计 store 157.5GB，磁盘净增 01≈30GB）。0.8 时"超 ~15GB"超额来自 UMA 语义下 KV cache 增长 + 页缓存 + 容器 RSS 对宿主头寸的侵蚀。
4. **0.70 验证**：**受控 conc3×65536 连续 4 次全部通过**（status=200，约 101k prompt tokens/req，wall ~120s），无 NCCL 超时、无 OOM、无冻结。内存谷底 01=7.9-8.6G / 02=8.7-9.5G / 03=2.5-3.1G / 04=2.5-3.1G，**全程未归零，且第 2 次后平台化**（无持续蠕变）。**结论：0.70 修正有效**。
5. **风险提示**：03/04 在持续负载下头寸仅 ~2.5GB，margin 偏窄；若并发>conc3 或更长时间连续负载，仍建议监控并预留 0.65/降 max-num-seqs 的后手。

---

## 二、现场接管与恢复（05:06–05:12 UTC）

### 接管时状态
| 节点 | rank | 状态 |
|---|---|---|
| 01(<MGMT_OCTET>) | rank0 head | `vllm-tp4-rank0` **Exited(1)**，TCPStore :25999 未监听 |
| 02(<MGMT_OCTET>) | rank1 | `vllm-tp4-rank1` Up 4min(healthy)，但日志处于 TCPStore connect timeout 重试 |
| 03(<MGMT_OCTET>) | rank3 | `vllm-tp4-rank3` Up 4min(healthy)，同左（十分钟前刚重启） |
| 04(<MGMT_OCTET>) | rank2 | `vllm-tp4-rank2` Up 4min(healthy)，同左 |

- 3 个 worker 容器均在 `[c10d] The client socket has timed out after 300000ms while trying to connect to (<NODE_IP>, 25999)` 重试态——**health=200/healthy 为假阳性**（healthcheck 仅 `pgrep VLLM::EngineCore`）。
- head 05:01 启动失败根因：rank0 与 rank1(02) gloo `is_in_the_same_node` 建连时 `Connection closed by peer [<NODE_IP>]:44227`——02 当时容器正被重建，rank 集合不同步（冷启动互杀）。

### 处置（head-first 全量重建）
1. 停 worker monitor（`systemctl stop vllm-tp4-worker.service`，sudo）并 `docker rm -f` 旧 worker 容器（02/03/04）。
2. `VLLM_API_KEY=<key> NO_WAIT=1 bash start_tp4_head.sh` 起 head（前置自检通过）。
3. 按序起 worker：02(rank1) → 04(rank2) → 03(rank3)。
4. head 就绪 ~240s（模型 155.43GiB/48 shards）。
5. 恢复 worker monitor（`systemctl start vllm-tp4-worker.service`，3 节点 active）。

### 验证
- `/health` = 200，`/v1/models` 正常，真实推理 200 / 0.2s 返回 "OK"。
- 4 rank 均 Up(healthy)，TCPStore 5 连接。

---

## 三、根因分析

### 3.1 主因：UMA 内存耗尽 → NCCL 超时（被动受害者）

**时间线证据（Prometheus + 内核日志，UTC）：**
| UTC | 事件 | 证据 |
|---|---|---|
| 02:00–03:10 | 0.8 基线，avail 01/02=9-12G、03/04=4-6G | Prometheus node_memory_MemAvailable |
| 03:18:22 | **03 NVRM `NV_ERR_NO_MEMORY`**（GPU 驱动分配失败） | journalctl -b -2 -k |
| 03:20 | **03 avail=0.0GB** | Prometheus |
| 03:23 | **NCCL ALLREDUCE 12.5M 元素 300s 超时（首次事故）** | 主理人已确认 |
| 03:20–04:15 | **03 系统级冻结**（Prometheus 数据中断约 50min，仅 ICMP 通） | Prometheus 缺口 |
| 03:54:37 | **内核 oom-killer**：killed `VLLM::Worker_TP`(total-vm 279973836kB) + `VLLM::EngineCor` | journalctl -b -2 -k |
| 03:59–04:05 | 01/03/04 持续 NVRM OOM | journalctl |
| 04:05 / 04:32 / 04:55 | 03、04 多次重启 | journalctl --list-boots |

- **关键因果**：内存耗尽（03:18 NVRM OOM、03:20 avail=0）**先于** NCCL 超时（03:23）。worker 因无法分配内存/无进展 → 环网 allreduce 无法在 300s 内完成 → watchdog 杀 worker → TP4 宕机。NCCL 超时是表象，内存耗尽才是根。
- 01 同样出现 NVRM `NV_ERR_NO_MEMORY`（02:28/03:10/03:18/03:59/04:03/04:44/04:47），内存压力为**全集群**现象。
- swap 在耗尽点已被占用 6-10GB，不足以兜底（总 15GB）。

### 3.2 为什么 0.8 会耗尽
- UMA：`--gpu-memory-utilization 0.80` 预留 ~96.8GB（0.8×121）给 GPU，宿主侧仅剩 ~24GB 名义；03/04 还跑 embed-8022 等 → 稳态 avail 只剩 4-6GB。
- conc3×65536（3 个并发 65536 token）长上下文 prefill/激活 + KV cache 增长 + KV 卸载 fs 管线 + 页缓存，把 4-6GB 头寸打穿至 0。

### 3.3 次要因素
- **SSD 满盘（01 186G/200G=93%）**：KV offload 写满时 fs 写路径阻塞/抖动，放大延迟与压力；但复现时磁盘仅 3.4G 仍崩溃 → **非主因**。
- **KV 卸载 CPU 竞争**：fs 读写线程 4+4 + zstd 压缩在长上下文高负载下抢占 CPU（NCCL pin CPU 8-9），加剧 NCCL 进度停滞；cpu_cache 层本身只占 0-20MB，非内存大户。

---

## 四、KV 卸载内存核查表（0.7 vs 0.8）

| 项目 | 0.8（历史，Prometheus） | 0.7（本次实测） |
|---|---|---|
| 稳态 avail 01/02 | 9-12 GB | 24.0-24.8 GB |
| 稳态 avail 03/04 | 4-6 GB | 18.6-18.7 GB |
| conc3×65536 谷底 01/02 | 0.8-2.7 GB | 7.9-8.6 GB |
| conc3×65536 谷底 03/04 | **0.0 GB（03 冻结）** | **2.5-3.1 GB** |
| vLLM 容器 RSS（docker stats） | — | ~5 GiB/节点 |
| vLLM 容器 cgroup current（含页缓存） | — | 空闲 23-28 GiB；负载后 7-12 GiB |
| cpu 主层利用率（2GiB） | ~0.9%（~20MB） | 0.0%（负载下仍 ~0） |
| fs 层累计 store | 876 GB（事故窗口） | 157.5 GB（4 次复测，~39GB/次） |
| kvssd 磁盘净增 | 01 至 186G（93%） | 01 3.4G→34G（18%），02/03/04 28K |

**内存去向解释**（UMA 语义）：
- "used 110-115GB（理论 96.8GB，超 ~15GB）"的差额 = GPU KV cache 增长（长上下文）+ 容器页缓存（模型权重文件）+ 容器 RSS + fs 压缩/线程缓冲。**这些全在同一物理池内**，0.8 预留太少导致负载下无缓冲。
- cpu_bytes_to_use=2GiB 的 CPU 主层**实际几乎不驻留数据**（利用率 0-1%），fs 层才是卸载主通道。

---

## 五、0.70 复测验证（关键判定）

**脚本**: `/tmp/verify_conc3_65536.py`（01 宿主 → 拷入容器，3 并发 × 65536 ctx，max_tokens=8）

| 轮次 | 时间(UTC) | req0 | req1 | req2 | wall |
|---|---|---|---|---|---|
| #1 | 05:17:03 | 200 / 101481pt / 122.8s | 200 / 101383pt / 122.8s | 200 / 101536pt / 122.8s | 122.9s |
| #2 | 05:19:33 | 200 / 101336pt / 120.3s | 200 / 101658pt / 120.3s | 200 / 101369pt / 120.3s | 120.4s |
| #3 | 05:22:16 | 200 / 101412pt / 120.5s | 200 / 101625pt / 120.4s | 200 / 101497pt / 120.3s | 120.5s |
| #4 | 05:24:17 | 200 / 101542pt / 121.0s | 200 / 101590pt / 120.5s | 200 / 101674pt / 121.0s | 121.1s |

- **4/4 全部通过，未复现 NCCL 超时/崩溃/OOM/冻结**。
- 内存轨迹（15s 采样）：谷底稳定后 01≈7.9G、02≈8.7G、03≈2.6G、04≈2.5G，**第 2 次复测后平台化**，无持续蠕变。
- 测试全程集群健康（health=200，4 rank Up，TCPStore 5 连接）。
- **判定：0.70 修正有效。**

### 残留风险
- 03/04 在持续 conc3×65536 下头寸仅 ~2.5GB；若并发 >3 或连续更长时间负载，仍可能逼近 0。建议保持监控阈值（<2GB 告警），并评估 0.65 或 max-num-seqs 收敛作为后手。

---

## 六、建议（按优先级）

1. **保留 0.70 上线**，同时为 03/04 增加 `avail<2GB` 与 `NVRM NV_ERR_NO_MEMORY` 的 Prometheus 告警。
2. **清理 01 kvssd 历史文件**（现 34G，事故时曾达 186G），加 disk_avail 告警与定期清理策略，避免满盘再次成为放大器。
3. **修复 health 假阳性**：healthcheck 不应仅 `pgrep EngineCore`，应校验 rank 全齐（TCPStore 3 worker 连接）后再报 healthy。
4. **核对 `vllm-healthcheck.timer` 状态**：主理人描述"已 disable"，但 01 实测 `is-enabled=enabled`（当前无调度，`0 timers listed`）。建议显式 `disable`/`mask`，防误触发 rebuild。
5. **head 服务自愈缺口**：`vllm-tp4-head.service` 当前 inactive+disabled（事故期停用），head 容器为手动拉起，崩溃后不会自愈。建议验证完成后恢复该服务或明确由 worker monitor 兜底。
6. **KV 卸载 fs 层**：评估 4+4 线程 + zstd 压缩在高负载下的 CPU 抢占，必要时降线程或改异步压缩；当前 CPU 主层几乎闲置（0-20MB），可考虑调小 cpu_bytes_to_use 释放名义预留。
7. **持续负载压测**：事故发生在 benchmark 第 31 组，建议用 benchmark 全量（≥31 组）在 0.70 下再跑一轮，验证平台化后的长期稳定性。

---

## 七、时间线（UTC，=UTC+8-8h）

| UTC | 事件 |
|---|---|
| 08-19 02:00–03:10 | 0.8 配置下 benchmark 进行，avail 持续下滑（01/02 9-12G，03/04 4-6G） |
| 03:18:22 | 03 NVRM NV_ERR_NO_MEMORY |
| 03:20 | 03/04 avail 0.0GB |
| 03:23 | **首次事故：NCCL ALLREDUCE 300s 超时，watchdog 杀 worker，TP4 宕机** |
| 03:20–04:15 | 03 系统级冻结（Prometheus 中断约 50min） |
| 03:54:37 | 03 内核 oom-killer 杀 VLLM::Worker_TP / VLLM::EngineCor |
| 04:05–04:55 | 03/04 多次重启 |
| ~04:56 | 主理人 mem-util 0.80→0.70 修正落地（脚本+硬校验已同步，bak 留档） |
| 05:00–05:01 | head 手动启动失败（rank1 建连中断） |
| 05:06 | Rex 接管；head 死、worker 卡 TCPStore 重试 |
| 05:09–05:12 | head-first 全量重建完成，4 rank 归齐，推理正常 |
| 05:17–05:26 | **conc3×65536 连续 4 次复测全部通过（0.70）** |
