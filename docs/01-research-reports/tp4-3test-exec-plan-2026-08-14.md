# 镜像生产环境三项攻关测试：执行计划（修订版 v2）

- 日期：2026-08-14 15:10（v2 修订）
- 修订原因：shim v8 功能核实推翻"shim v4 待办"假设；记忆纠错（isolcpus=8-9 非 0-4）；发现 systemd 自愈停机方式
- 范围：① MoE 通信优化（v8 核实 + v9 增量评估 + NCCL 参数调优）→ ② c5 恢复 → ③ vLLM 0.27

---

## 0. 现状盘点（2026-08-14 15:00 实测更新）

| 项 | 状态 |
|---|---|
| 生产 | TP4 healthy（systemd 托管 + self-heal，**不可 docker stop**，须 systemctl 顺序停机） |
| **shim v8** | **功能已核实达标**：`pt_nccl_*/pt_tcpstore_* → PSR 8-9（隔离核）`、`VLLM::EngineCor → PSR 15-19`（负载下 `ps -eLo` 采样实证）；NCCL 线程在 **engine worker 子进程**（非 API server PID 1）；竞态已由 v8 mark-then-pin 修复（runbook D6） |
| 记忆纠错 | MEMORY.md 已更新：isolcpus=**8-9**（非旧记录 0-4/1-4）；0-4/10-14=A725 2808MHz、5-9/15-19=X925 3900MHz；"shim v4 待办"标记完成 |
| vLLM 0.27 | clone 后台进行中；**无 aarch64 官方 wheel** → NGC 26.07（torch 2.13.0a0+nv26.07 / flashinfer 0.6.14 / triton 3.6 需升 3.7.1）容器内源码编译 sm_121 |
| c5 基线 | 131072: 595.53/7.01；32768: 678.68/16.95（未变） |
| 停机 SOP | worker(02/04/03) → head(01) `systemctl stop vllm-tp4-{worker,head}.service`；启动 `bash start_tp4_cluster.sh`（head-first）；窗口前置：通知 litellm 下游 |

## 1. ① MoE 通信优化（重新定义）

**结论先行**：shim 线程 pin 已达标（v8），**"NCCL 数据面线程落隔离核"目标已实现**。剩余优化空间分两级：

### 1a. shim v9 增量（可选，小）
- v9 = v8 功能全保留（NCCL→8-9、EngineCore→15-19、默认 5-19）+ 可选细分：`pt_tcpstore_*`（低频心跳）→ 5-7，把隔离核 8-9 让给纯数据面 `pt_nccl_*`
- 风险：改 pin 布局需重新 PSR 验证；收益小（tcpstore 占用隔离核时间占比低）
- 结论倾向：**不做或仅做冒烟验证**，避免无谓风险

### 1b. NCCL 通信参数调优（主攻，直击 368KB all-reduce）
- 现状：ring-only busbw 4.4GB/s、ALGO=RING、NET_PLUGIN=none、MERGE_NICS=0、IB_PEER_HCA per-peer、CHANNELS=MIN2
- A/B 项：`NCCL_CHANNELS`（2→4/8）、`NCCL_IB_QPS_PER_CONNECTION`、`NCCL_BUFFSIZE`、`NCCL_IB_TIMEOUT`/`RETRY_CNT`（现 7）、多 HCA 并行（MERGE_NICS=0 时 PEER_HCA 双 HCA 并发）
- 验证：`nccl-tests allreduce`（4 rank 环）busbw + bench PR/DE（c1@131K）对比基线（1896.4/104.1）
- 载体：测试窗口内改容器 env（同镜像）→ 对比 → 恢复

### 1c. 生产 PSR 合规性
- 已实测（单机 01）：NCCL 类→8、EngineCore→16 ✅；**四机全量采样由调查 agent 佐证中**

## 2. ② c5 恢复（不变）
- 基线/归因/手段 A/B 见 v1；载体同 ① 窗口（同镜像改参数）

## 3. ③ vLLM 0.27（不变）
- NGC 26.07 容器编译；编译前需 `pip install triton==3.7.1`（NGC 自带 3.6.0）
- 关注：#4495 B12x Direct M=1、#48957 空 c128 skip、#49486 topk/router skip、#48047 q-head padding、#48993 MXFP4 indexer KV、#50004 adaptive topk
- 0.27 官方已支持 DeepSeek-V4 + DSpark（#50242）→ 可不依赖 anemll 定制 overlay

## 4. 执行顺序（修订）

| Phase | 内容 | GPU | 前置 |
|---|---|---|---|
| P0 | vLLM 0.27 clone/编译（后台）+ v8 核实报告 + agent 佐证 | 否 | 完成中 |
| P1（窗口 1） | ①b NCCL 参数 A/B + ② c5 复测（同镜像改 env，**无镜像构建**） | 是 | 用户窗口 + systemctl 停机 SOP |
| P2（窗口 2） | ③ v0.27 部署验证（唯一需新镜像） | 是 | P0 编译完成 |
| P3 | 恢复生产 + 汇总 + 生产切换建议 | - | - |

## 5. 团队调查佐证（已派）
- Agent A（Explore）：runbook/部署指南 shim md5（06069400 vs ce43c688）与 PSR 设计核对
- Agent B（general-purpose）：四机负载下 PSR 全量采样（NCCL/EngineCore affinity 合规性）

## 6. 风险
- systemd 自愈：窗口内须 systemctl stop 全套，且测试后 start_tp4_cluster.sh 恢复
- v0.27 编译 3-6h 失败风险（aarch64/triton 3.7.1/SM121）
- NCCL 参数 A/B 若劣化 → 立即回滚 env（容器重启成本 ~5min/轮）
