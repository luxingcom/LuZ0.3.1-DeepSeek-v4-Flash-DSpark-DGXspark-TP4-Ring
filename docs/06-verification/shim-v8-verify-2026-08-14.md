# shim v8 功能核实报告（含记忆纠错）

- 日期：2026-08-14 15:10
- 背景：执行"镜像生产环境三项测试"前核实 shim 现状，发现**本地记忆与服务器实际/文档多处冲突**，本文为核实结论。

## 1. 核实结论（TL;DR）

**生产 shim v8（md5 ce43c688）功能完整、符合设计**：
- NCCL 数据面线程（`pt_nccl_*`、`pt_tcpstore_*`、`NCCL*`、`*Proxy*`）→ **CPU 8-9（isolcpus 隔离核）** ✅
- EngineCore 线程（`EngineCore*`、`VLLM::EngineC*`，comm 截断为 `VLLM::EngineCor`）→ **CPU 15-19** ✅
- 其余线程默认 → CPU 5-19 ✅
- 线程 pin 竞态已由 v8 mark-then-pin 修复（runbook D6 记录，2026-08-14 pin_smoke 冒烟复现无异常）✅

## 2. 证据链

### 2.1 冒烟测试（v8 .so 直接 LD_PRELOAD）
```
[libncclpin] NCCLProxy   => CPU 8-9
[libncclpin] pt_nccl1    => CPU 8-9
[libncclpin] EngineCore0 => CPU 15-19
[libncclpin] VLLM::EngineC1 => CPU 15-19
[libncclpin] misc_thread / default => CPU 5-19
```

### 2.2 生产负载下 PSR 采样（01，ps -eLo pid,tid,psr,comm）
```
VLLM::EngineCor  → PSR 16（∈15-19）✅
pt_tcpstore_uv   → PSR 8  ✅
pt_nccl_watchdg/heartbt → PSR 8  ✅（多线程均 8）
vllm 主线程      → PSR 5/6/7/15/16（默认池 5-19）
```

### 2.3 关键认知（纠正）
- **NCCL 线程位于 engine worker 子进程**（本次 PID 2382562），**不在 API server 进程**（PID 1）——此前 14:50 采样 PID 1 未见 NCCL 线程是**方法错误**，非 v8 失效
- 采样须用 `ps -eLo pid,tid,psr,comm | grep -E "NCCL|EngineC"`（负载下，runbook 亦如此指示）

## 3. 记忆纠错（MEMORY.md 已更新）

| # | 旧记忆（错误/过时） | 实际（2026-08-14 实测） |
|---|---|---|
| 1 | 隔离核 isolcpus=0-4，实际 1-4 | **isolcpus=8-9**（2×X925 3900MHz） |
| 2 | A725 给 NCCL、X925 5-9 计算 | 0-4/10-14=A725 2808MHz；5-9/15-19=X925；**NCCL→8-9（隔离核）、EngineCore→15-19、0-4 放开** |
| 3 | shim v4 待办（NCCL 落隔离核；thread_entry 竞态） | **v8 已实现**（runbook v1.5：NCCL/PT→8-9、EngineCore→15-19、D6 竞态已修复） |
| 4 | 采样 vLLM 主进程即可见 NCCL 线程 | 须采样 **engine worker 子进程** |

## 4. 文档核对（md5 之谜已解开）

| 项 | 文档记录 | 服务器实际 | 状态 |
|---|---|---|---|
| shim v8 md5（01） | deployment-guide 行 242：**ce43c688...（v8，四机一致）**；shim-deploy.sh EXPECTED_V8=ce43c688 | ce43c688... | **一致 ✅** |
| runbook A.3 "shim v8 md5 06069400" | rollback-anchors 行 41：06069400... 实为 **start_tp4_head.sh 脚本** md5，runbook 表述歧义 | — | 澄清：非 shim md5 |
| ringonly md5 | b7784b49...（deployment-guide 行 241） | 待 agent 核对 | 待补 |
| PSR 设计 | NCCL 8-9 / Engine 15-19 | 实测一致 ✅ | 一致 |
| isolcpus | 8-9 | 8-9 ✅ | 一致 |

## 5. 对后续工作的影响

1. **① MoE 通信优化重心转移**：shim 线程 pin 已达标 → 优化转向 NCCL 通信参数（CHANNELS/QPS/PEER_HCA 多 HCA）而非线程 pin；shim v9 仅保留可选细分（tcpstore→5-7）
2. **② ③ 不受影响**（独立于 shim）
3. **停机方式**：TP4 由 systemd self-heal 托管，须按 SOP：worker(02/04/03)→head(01) systemctl stop；启动 `start_tp4_cluster.sh`
4. 四机 PSR 全量采样由调查 agent 佐证中（结果后补）

## 6. 待办
- [x] md5 差异核实：06069400 实为 head 脚本 md5（rollback-anchors 行 41），**非 shim**；v8 权威 md5=ce43c688（deployment-guide + shim-deploy.sh 双重确认）
- [x] Agent A 文档核对完成：runbook/rollback 记录性错误已修订（备份 .bak-shimverify-20260814）
- [x] Agent B 四机 PSR 采样完成（下表）

## 7. 四机 PSR 全量采样（Agent B，2026-08-14 15:2x）

| 机器 | EngineCore (affinity/psr) | NCCL 线程 (affinity/psr) | cpuset |
|---|---|---|---|
| 01 rank0 | 15-19 / 16 | Worker_TP0: 8×heartbt+8×watchdg+1×tcpstore / **8-9** / 8 | 1-19 |
| 02 rank1 | 15-19 / 15 | Worker_TP1: 8×heartbt+8×watchdg / **8-9** / 8 | 1-19 |
| 03 rank3 | 15-19 / 15 | Worker_TP3: 8×heartbt+8×watchdg / **8-9** / 8 | 1-19 |
| 04 rank2 | 15-19 / 19 | Worker_TP2: 8×heartbt+8×watchdg / **8-9** / 8 | 1-19 |

- 四机 isolcpus=8-9、cpuset=1-19、LD_PRELOAD shim 生效一致；TCPStore 仅 rank0（02 无 tcpstore 属正常）
- **结论：四机生产 NCCL 相关线程全部按要求隔离到 8-9，EngineCore 15-19，设计达标** ✅
- 附注：03/04 的 anemll-embed-8022 容器（Qwen3-Embedding-0.6B）进程名同为 VLLM::EngineCor 但 affinity 0-19 未受 shim（未挂 LD_PRELOAD），与 TP4 无关，可后续关注整体隔离
- 验证请求：1 个小推理 HTTP 200 / 1.53s 正常
