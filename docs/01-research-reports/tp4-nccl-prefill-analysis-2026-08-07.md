# TP4 组网 NCCL 收益分析 + prefill 100:1 场景路径重估（综合报告）

**日期**：2026-08-07
**工作流**：系统设计 / 技术选型（工作流 2 变体）
**参与成员**：Archi（收益建模与决策）/ Rex（组网执行清单）/ Tessa（prefill 基准计划）

---

## 📌 TL;DR（执行摘要）

- **结论**：TP4 收益远大于 sm_121a 重编——**P0 阻塞项是 .55/.59 物理接线**；TP4 bring-up 直接用 cu132 全栈（NCCL 2.30.7）；sm_121a SASS 重编**暂缓（Conditional No-Go）**，以 TP4 实测为门禁。
- **决策**：TP4 有线组网 + NCCL 2.30.7 = **Go**（预期收益 1.7-1.9×，门禁=物理接线 + 实测带宽 ≥8GB/s）；sm_121a 重编 = **Conditional No-Go**（TP4 实测满足 3 条件后再回来）。
- 严重度分布：🔴 阻塞 1 项（物理接线）/ 🟠 高 2 项（带宽实测、NCCL 版本）/ 🟡 中 3 项。
- 阻塞 / 非阻塞：**阻塞**——.55/.59 无直连口枚举，Wi-Fi 组网量化不可行（~200× 慢于计算），接线前 TP4 No-Go。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（条件 = 物理接线 + 实测带宽 ≥8GB/s） |
| 阻塞项数量 | 1 项（P0：.55/.59 ConnectX-7 直连口接线启用） |
| TP4 预期收益 | 理想 2.0× / 现实 **1.7-1.9×**（prefill_tps，需 comm 隐藏成立） |
| sm_121a 重编 | ⏸️ Conditional No-Go（3-5% 收益 vs 高工程风险） |
| 建议下一步 | P0 接线+带宽实测 → P1 cu132 全栈 TP4 bring-up → P2 profile 后定 sm_121a |

---

## 1. 关键前提修正：网络带宽存疑（最高优先级待核实）⚠️

- NVIDIA 官方规格：DGX Spark = **ConnectX-7 @ 200 Gbps**（非 25G）；但 ConnectX-7 走 **PCIe Gen5 x4 ≈ 96 Gbps** → 单卡有效带宽 **B_net ≈ 12 GB/s**（200G 被 PCIe 卡住，双口共享不叠加）。
- **必须实测**：`ethtool enp1s0f1np1 | grep Speed` + `iperf3 -u -b` / `ib_write_bw`——确认是 12GB/s 还是 3.1GB/s（25G）。
- **该数字决定 TP4 成败**：12GB/s → 收益 1.7-1.9×；3.1GB/s → TP4 ≈ TP2 甚至更差。

## 2. TP4 组网 NCCL 收益量化（Archi 模型）

### 2.1 显存收益
TP2 78G/机 → TP4 **39G/机**，每机释放 ≈39G 给 KV/激活（可用 39G→82G，约翻倍）→ 长上下文 prefill 防 OOM、可加大 chunk/batch。

### 2.2 算力收益
TP4 单机 FLOPs = N·T/2 → **prefill 吞吐上限 ×2**（相对 TP1 ×4）。

### 2.3 通信开销（公式）
每层 2 次 all-reduce（环算法每节点净传输 1.5×H×T）：
- 8K prefill 通信字节 ≈ 3·L·H·T ≈ 16GB
- **通信/计算时间比 = 6·L·H·P / (B_net·N) ≈ 0.8** @ 12GB/s → 可被计算重叠隐藏（<1）✅
- ⚠️ 若仅 25G（3.1GB/s）：比值 ≈ 3.1 → 通信暴露，TP4 无收益

### 2.4 Wi-Fi 组网量化不可行
Wi-Fi 有效带宽 12-37MB/s + 63-124ms RTT：单次 50MB all-reduce RoCE≈12ms vs Wi-Fi≈**3.7s**；8K prefill 累计 ≈600s vs 计算 2.6s → **~200× 慢于计算** → 接线前 TP4 直接 No-Go。

### 2.5 NCCL 2.28.9 → 2.30.7：建议升级（低风险）
2.30.x 相关改进：**PXN 死锁修复（2.30.7）、RoCE LAG 轮询 QP 负载均衡（多口直接受益）、Blackwell NVLSTree 调优、IB 端口自恢复、NCCL Inspector（4 机 bring-up 诊断）**。生产 2.28.9 留作回退。

## 3. 路径 1（sm_121a SASS 重编）在 prefill 100:1 下的重估

### 3.1 计算受限确认
GB10 临界算术强度 AI* = P/BW ≈ 250T/273G ≈ **915 FLOP/byte**；prefill 算术强度 ≈ 2T（FP8 1B）→ **T ≥ ~460 token 即进入计算受限区**（100:1 场景成立）→ GEMM 效率直接决定 wall-time，不再被 decode 稀释。

### 3.2 端到端收益重估
- 新估算：GEMM 提升 0-10% × GEMM 占 prefill 时间份额 f≈0.6-0.8 → **端到端 0-8%（中枢 3-5%）**（旧 decode 主导估算 0-2% → **放大 2-4×**）
- cuBLAS 13.4/cuDNN 9.24 走 cuBLAS 路径的 kernel **无需重编已含 Blackwell 原生实现**（≈0 成本）；vLLM 主 GEMM 走 cutlass 自研才需重编，且 ptxas/Triton sm_121a bug 风险未消
- 净判断：**3-8% 收益 × 高工程风险（nightly cu132 生产化 + SASS 重编）→ 不值得作为前置投入**

## 4. 综合决策（Archi 结论）

| 项 | 决策 | 门禁 |
|----|------|------|
| TP4 有线组网 + NCCL 2.30.7 | ✅ **Go**（收益 1.7-1.9×） | 物理接线 + 实测带宽 ≥8GB/s |
| sm_121a SASS 重编 | ⏸️ **Conditional No-Go** | TP4 实测满足：① comm 隐藏（加速比 >1.5×）② profile 显示 prefill GEMM ≥70% ③ 目标上下文已计算受限 |
| 零风险中间路径 | 先迁 **cu132 全栈（不重编 SASS）** | A/B 验证 vLLM 0.26.1.dev0 兼容性（自动吃到 cuBLAS 13.4/NCCL 2.30.7） |

## 5. 分阶段建议

- **P0（阻塞）**：.55/.59 接线 + 启用直连口 + MTU 9000 + **实测带宽 ≥8GB/s**（iperf3/ib_write_bw）
- **P1**：cu132 全栈 TP4 bring-up（NCCL 2.30.7、环拓扑、排除 Wi-Fi、分级 2→3→4 机 bring-up）；测 4K/16K/32K prefill 吞吐对 TP2 A/B
- **P2**：profile 定 GEMM 占比 → 再决定 sm_121a A/B（仅重编 vLLM/cutlass GEMM 路径）

## 6. 组网执行要点（Rex 清单摘要）

- **拓扑**：直连环 58→60→59→55→58（4 条线、每机 2 口全用）；IP 沿用 <NODE_IP>/24（A 链）+ <NODE_IP>/24（B 链），新增 10.100.138/139 按需
- **必须先补**：四机 SSH 互信（.58→.55 实测 Permission denied）
- **必须补静态路由 + ip_forward=1**（否则远端 rank 不可达）
- **NCCL**：`NCCL_IB_HCA=mlx5_x` 显式限定、`NCCL_SOCKET_IFNAME=en*`（排除 wlP9s9）、`NCCL_IB_TIMEOUT=1000`/`RETRY_CNT=7` 沿用、`NCCL_DEBUG=WARN`（首跑 INFO）
- **竞态预案**：head 先启 TCPStore（轮询 25000）→ rank1/2/3 间隔 2-5s **串行**启动（禁止并行 spawn）；降级 TP4→TP3→TP2；保留 TP2 脚本回滚
- **防火墙**：TCP 25000 + UDP 4791（RoCEv2）+ NCCL 动态口；监控：IB 计数器（port_rcv_errors）、blackbox ICMP、NCCL 日志告警

## 7. 基准测试计划（Tessa 摘要）

- 测例：8k/32k/128k/256k（**新增 128k/256k 校准文本**）× batch 1/4/8/16/32，max_tokens 32（8k@80 为精确 100:1）
- 主指标：prefill_tps、TTFT p50/p95、吞吐-延迟曲线；**随机前缀铁律（final_matrix 需修 prefix-cache 隐患）**
- 对比矩阵：A TP2 基线 → B TP4（隔离组网）→ C TP4+cu132（隔离软件栈）
- 验收：C2 prefill_tps ≥1.25×C1（256k 档 ≥1.1×）；TP4 理论上限 2.0×、下限 1.25×；cu132 设"不劣化"线 0.95×
- TP4 特有：C4 NCCL 正确性（world=4 all-reduce 0 误差 + 带宽实测）、C5 拉起 SOP（4/4 ready）、C6 断线恢复

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | .55/.59 ConnectX-7 直连口接线 + 启用 + MTU 9000 + **实测带宽**（ethtool/iperf3/ib_write_bw，确认 12GB/s vs 3.1GB/s） | Rex | P0 | 本周 |
| 2 | 补四机 SSH 互信（.58→.55/.59 公钥） | Rex | P0 | 今日 |
| 3 | cu132 全栈镜像铺到 4 机（.58/.60 从 registry 拉取）+ NCCL 2.30.7 bring-up（环拓扑、分级 2→3→4） | Rex/Archi | P1 | 1-2 周 |
| 4 | prefill_bench.py 开发（修 prefix-cache、128k/256k 校准）+ C1/C2 对比矩阵 | Tessa | P1 | 1-2 周 |
| 5 | TP4 实测后按 3 条件门禁评估 sm_121a 重编（comm 隐藏/GEMM≥70%/计算受限） | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- **最高优先级待核实**：实际链路速率与实测带宽（决定 TP4 成败）；DeepSeek-V4-Flash 的 H/L/激活 dtype/活跃参数数/KV 是否 MLA 压缩；.55/.59 无直连口枚举根因（线缆/驱动/BIOS）；vLLM 0.26.1.dev0 对 torch 2.14.0.dev+cu132 兼容 A/B。
- 收益模型基于公开规格与公式推算（H≈12K、P≈250TFLOPS），实测数据出来后需回填校准。
- 无交换机点对点 RoCE 无 PFC/DCB，靠重传，NCCL_IB_TIMEOUT=1000 已覆盖；第二口命名/IB 设备名以实测为准。

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：TP4 NCCL 收益模型（带宽前提修正/显存/算力/通信公式/加速比估算/NCCL 版本对比）、路径 1 重估（计算受限确认/端到端收益/条件门禁）、综合决策与分阶段建议
- **Rex（SRE）**：组网执行清单（物理层诊断/环拓扑 IP 规划/静态路由/NCCL 配置/竞态预案/降级回滚/监控）
- **Tessa（测试专家）**：prefill 主导基准计划（已落盘 TP4_prefill_bench_plan.md：口径/对比矩阵/用例表/脚本改造/验收标准）
- 主理人侦察：.55 仅 Wi-Fi 无直连口枚举（lspci 有 PCI bridge 但 infiniband 空）、Wi-Fi 延迟 63-124ms、生产 NCCL 2.28.9、cu132 全栈 NCCL 2.30.7

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。P0 接线与带宽实测完成前，TP4 相关投入请谨慎排期。
