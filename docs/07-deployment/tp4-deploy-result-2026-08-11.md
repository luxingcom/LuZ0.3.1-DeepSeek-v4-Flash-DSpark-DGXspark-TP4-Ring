# TP4 环网部署结果报告（ring-only 补丁路线）

**日期**：2026-08-11
**工作流**：工作流 4（部署执行）+ 技术攻坚（社区方案落地）
**参与成员**：Archi（机制分析+补丁设计实施）、Rex（构建/部署/验证）、Zhen（编排）
**状态**：✅ **TP4 已上线可用**（纯环网 4 机 tensor parallel）

---

## 📌 TL;DR（执行摘要）

- ✅ **TP4（4 机环网 TP=4）部署成功上线**：四机 vllm-tp4-rank0~3 全部 Up (healthy)，8001 /health=200，/v1/models 正常，Chat 冒烟通过（fingerprint `-tp4-` 确认），简单 bench 8 tok/s（冒烟级，首请求含预热）。
- 🔧 **攻坚核心**：纯环网（每机仅 2 OSFP）stock NCCL 无法 4 rank TP4（init 对全 rank 对建 QP + index 配对 → 非直连必 110）——通过社区 GLM-5.2 路线 **ring-only NCCL 补丁 + per-peer 对口映射**解决：
  - 补丁 v1：`ncclTransportP2pConnect` 过滤跨机非环邻对（PAT 无条件建 distance-2 对是纯环死结根源）
  - 补丁 v2：自定义 env `NCCL_IB_PEER_HCA`（peerRank=devName）在 send/recvSetup 强制环邻对用物理对口
- 📊 4 rank 验证：CONN OK sum=6、**零 110**、busbw 4.4GB/s（单口量级，2 rank 直连 6.4GB/s 佐证）
- ⚠️ 带宽现状：单口 IB 量级（4-6.4GB/s），双口/twin 优化为后续项（NCCL_IB_MERGE_NICS 或双 channel）
- 回滚：TP4 脚本备份 `.bak-tp4-patch2` 保留；TP2 锚点未动

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 TP4 上线可用（纯环网技术攻坚成功） |
| 部署状态 | 4 rank healthy、8001=200、冒烟通过 |
| 攻坚方案 | ring-only 补丁 v1+v2（过滤非环邻对 + per-peer 对口） |
| 4 rank 验证 | sum=6 ✓、零 110、busbw 4.4GB/s |
| 关键 env | NCCL_ALGO=RING、NCCL_IB_SUBNET_AWARE_ROUTING=1、NCCL_IB_PEER_HCA |
| 遗留 | 双口带宽优化、完整 bench 对比 TP2 |

---

## 🔍 攻坚过程（为什么 stock NCCL 无解）

| 层 | 机制 | 证据 |
|---|---|---|
| 现象 | 4 rank allreduce `ibv_modify_qp 110`（01 的 <RING_SUBNET> 连 03 的 <RING_SUBNET> 非直连） | 多次实测 |
| Why1 | NCCL init 对**全 rank 对**建 IB QP（非仅 RING 相邻）——PAT connect 无条件为 distance-2 pair 建连 | 源码 init.cc Phase6 + 日志 |
| Why2 | 建连选口按 **HCA index 统一配对**（与物理连通无关）；每机 index0 口只连 1 个邻居但有 2 个 | 对照实验：单口对口成功、多口 index 配对失败 |
| Why3 | 无交换机时 NCCL 无法感知 fabric 拓扑（隐式全互联假设） | NCCL 机制 + 社区实证 |
| 结论 | 纯环网 stock NCCL（IB/TCP 均）不可行 → 需补丁 | GLM-5.2 同结论 |

## 🔧 补丁方案（社区 GLM-5.2 路线，落地）

### v1：ring-only 过滤（src/transport.cc，ncclTransportP2pConnect）
```c
/* 跨机非环邻对跳过（recv/send 循环各一条） */
if (comm->peerInfo[peer].hostHash != comm->peerInfo[comm->rank].hostHash &&
    peer != ringPrev && peer != ringNext) continue;
```
- 关键发现：NCCL init 无条件执行 `ncclTransportPatConnect`（maxLocalRanks==1），PAT 对 n=4 建 distance-2 对（0↔2、1↔3）——正是纯环必挂的非相邻对
- 运行期必须 `NCCL_ALGO=RING` 强制（Tree/CollNet/PAT 连接被过滤）

### v2：per-peer 对口映射（src/transport/net.cc）
- `ncclIbPeerHcaOverride(comm, peerRank, &netDev)`：解析 env `NCCL_IB_PEER_HCA="peerRank=devName;..."`，在 sendSetup/recvSetup 覆盖 req.netDev（peer 口 IP 在建连时不可得，故走显式 env）
- 每机配置（rank 01=0/02=1/04=2/03=3）：
  | 机 | NCCL_IB_PEER_HCA |
  |---|---|
  | 01 | `1=rocep1s0f1;3=rocep1s0f0`（→02 f1、→03 f0） |
  | 02 | `0=rocep1s0f1;2=rocep1s0f0` |
  | 04 | `1=rocep1s0f0;3=rocep1s0f1` |
  | 03 | `0=rocep1s0f0;2=rocep1s0f1` |

### 构建与部署
- 源码：NVIDIA/nccl tag v2.30.7-1（commit 73cf112），容器内构建（anemll 0.2.1 --entrypoint bash --user root，CUDA_HOME=/usr/local/cuda，NVCC_GENCODE sm_121），产物 `/opt/nccl-ringonly/libnccl.so.2.30.7`（四机 MD5=4cc43e3b25ddf275701c11b3d566b686）
- 加载：docker run `-v /opt/nccl-ringonly:/opt/nccl-ringonly:ro -e LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`
- 版本区分：banner `2.30.7+cuda13.0`（补丁版）vs `2.30.7+cuda13.3`（pip 原版）

## 📊 验证与运行数据

| 项 | 结果 |
|---|---|
| 4 rank all_reduce | CONN OK sum=6、**零 110** |
| busbw | 4.4 GB/s（4 rank RING 单口）；2 rank 直连 6.4 GB/s |
| per-peer 对口日志 | rank0→3 用 <RING_SUBNET>、rank1→0 用 <RING_SUBNET>、rank2→1 用 <NODE_IP>、rank3→2 用 <RING_SUBNET> 全部物理对口 ✓ |
| TP4 容器 | 四机 vllm-tp4-rank0~3 Up (healthy) |
| 服务 | 8001 /health=200、/v1/models=deepseek-v4-flash-0731、Chat 冒烟 ✓（-tp4- fingerprint） |
| 简单 bench | 5 并发 avg 3.81s、p50 1.74s、8 tok/s（冒烟级） |

## ✅ 行动清单（后续）

| # | 行动 | 负责 | 紧急度 | 验收 |
|---|------|------|--------|------|
| 1 | Grafana 面板核对（用户任务 7） | Rex | P1 | ✅ **已完成**：四机 dcgm/node-exporter 全 up、RoCE 16 口数据在、TP4 head 8001 metrics 可抓，无数据面故障；差异项：188:8001 无效 target（worker 无 8001，建议移除）P2 |
| 2 | 双口带宽优化（NCCL_IB_MERGE_NICS 或双 channel/PEER_HCA 双 dev） | Archi+Rex | P2 | busbw 提升对比 |
| 3 | 完整 bench 对比 TP2（prefill_tps/TTFT/preemption） | Rex | P2 | 达预期 |
| 4 | 补丁源码/脚本/报告归档（<INSTALL_DIR>/backup/tp4-<date>/ + deliverables） | Docu | P2 | 归档完整 |
| 5 | 回填 Runbook（TP4 部署、隔离核 1-4、补丁方案） | Docu | P2 | runbook 更新 |

## ⚠️ 待完善 / 已知局限

- busbw 为单口量级（4-6.4GB/s），双口/twin 未启用；TP4 长 prefill 吞吐待完整 bench 确认
- nccl-tests 因容器缺 libmpi 改用 torch distributed 脚本验证（功能等价）
- 补丁对多 GPU/机场景有跨机过滤副作用（本场景每机 1 GPU 无影响）；NCCL 版本升级需重新验证
- 完整回退路径：还原 start_tp4 脚本 .bak-tp4-patch2 → 容器 stop → 可回 TP2（锚点未动）

---

## 📚 数据来源 & 成员产出索引

- Archi：机制分析（PAT/全对建连/index 配对）、补丁 v1/v2 设计与实施（transport.cc/net.cc diff、NCCL_IB_PEER_HCA）
- Rex：构建链打通（容器内构建 glibc 兼容）、四机分发部署、4 rank 验证、TP4 恢复与冒烟
- 社区参考：GLM-5.2 4 机环帖（SIRCL/ring-only 路线）、NCCL 官方 playbook（4 机须交换机）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
