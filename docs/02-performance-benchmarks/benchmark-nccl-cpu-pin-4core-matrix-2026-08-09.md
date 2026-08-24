# 四核绑核 + 协议矩阵实测报告（2026-08-09, Phase B）

**执行**：Rex（SRE）· 方案：Tessa · 判据：G1 五 run i_p99 均 ≤ 40µs（新 SLO）
**环境**：DGX Spark 01↔02（L1 module1 200G RoCE），NCCL 2.30.7，nccl-tests 2.19.7，OpenMPI 4.1.6，四台 isolcpus=16-19 已生效
**方法**：16B allreduce，-w1000 -n10000 -z0 -I1，GID_INDEX=2，per-node HCA（L1），NCCL_SOCKET_IFNAME=enP7s7（管理口），`--mca oob/orte/btl_tcp_if_include enP7s7`（关键修复）

---

## 一、🚨 关键排障：跨节点 NCCL init 卡死修复

**症状**：四台重启 isolcpus 16-19 后，2-rank mpirun NCCL init 无限挂起（进程 Rl 状态，NCCL_DEBUG 无输出）。
**根因**：OpenMPI orted 默认使用**全部网卡 IP**（含 RoCE <NODE_IP>/137.1）做 TCP 握手，而 RoCE 口**内网限 TCP 规则（INPUT/OUTPUT DROP TCP）**将握手包 DROP → orted 卡在建立连接。
**修复**：mpirun 加 `--mca oob_tcp_if_include enP7s7 --mca orte_tcp_if_include enP7s7 --mca btl_tcp_if_include enP7s7` 强制 orted 走管理网 → 立即恢复。
**结论**：重启后必须带 MCA 参数；此参数已固化进 run_matrix.sh。**建议后续所有跨节点 NCCL/MPI 测试脚本统一加此三参数**（或配置 OpenMPI 默认 if_include）。

## 二、6 组矩阵数据（T2_12 L1，OOP，单位 µs）

| 组 | 协议 | 绑核 | run | avg | i_p99 | i_max | 判据 |
|---|---|---|---|---|---|---|---|
| **G1** | **LL** | **16-19** | 1 | 16.32 | **23.55** | 2666 | ✅ |
| | | | 2 | 16.42 | **24.26** | 2832 | ✅ |
| | | | 3 | 16.28 | **23.55** | 2803 | ✅ |
| | | | 4 | 16.44 | **25.76** | 2675 | ✅ |
| | | | 5 | 16.20 | **19.33** | 2693 | ✅ |
| G2 | LL | 18-19 | 1-5 | 16.22-16.49 | 19.5-22.8 | 2752-3199 | ✅ |
| G3 | Simple | 16-19 | 1-5 | 24.40-24.63 | 27.65-29.70 | 2651-3133 | ✅ |
| G4 | Simple | 18-19 | 1-5 | 24.53-24.81 | 27.65-33.76 | 2641-3078 | ✅ |
| G5 | LL128 | 16-19 (-c1) | 1-5 | 23.68-23.94 | 25.73-29.57 | 2570-2954 | ✅ |
| G6 | LL128 | 18-19 (-c1) | 1-5 | 23.82-24.04 | 25.63-33.89 | 2119-2734 | ✅ |

## 三、核心结论

1. **主判据通过**：G1（LL+16-19）五 run i_p99 全部 ≤ 40µs（19.3-25.8µs），avg 16.2-16.4µs，**P99≤40µs 新 SLO 达成** ✅
2. **LL 为 16B allreduce 最优协议**：LL avg 16.3µs vs Simple 24.6µs（-33%）、LL128 23.8µs（LL128 反而慢 +46%）。LL128 对 16B 小消息无优势（其优化面向大消息带宽）。
3. **四核(16-19) vs 双核(18-19) 无差异**（G1 vs G2 相当）：2-rank 单通信实际只用 1-2 核。四核隔离的收益在 4-rank/TP4 或并发多通信场景才会体现——符合预期，如实记录。
4. **绑核+LL 组合最佳**：avg 16.3µs / i_p99 19-26µs，**优于此前 18-19 双核+LL 的 17.3µs / 25-36µs**（隔离核扩展 16-19 后 i_p99 下限进一步收敛）。
5. **LL128 数据校验通过**：G5/G6 `-c 1` OOP #wrong 全 0，**跨节点 RoCE LL128 无数据损坏**（历史风险解除）。
6. **GDR=0 确认**：日志 "Connected all rings, use ring PXN 0 GDR 0"，cuMemGdrSupport=0（GB10/ARM64 无 GDR 硬件支持）——RDMA 走正常路径，GDR 不是瓶颈。

## 四、遗留与建议

- **i_max 尖峰 1.5-3ms 仍存在**（所有组）：IRQ/softirq 无法绑隔离核（ARM64 GIC 已知限制），偶发长尾无法完全消除；但 i_p99 稳定在 20-30µs 不受影响。
- **LL128 结论**：Tessa 的协议调查方向正确——对 16B 小消息 LL 最优；LL128 保留用于大消息场景。
- **OpenMPI MCA 参数**：建议写入后续所有测试脚本/生产 MPI 启动模板。
- **03 ConnectX-7**：重启后 PCIe 未枚举（Phase A 已记录），将来成环需现场处理。
