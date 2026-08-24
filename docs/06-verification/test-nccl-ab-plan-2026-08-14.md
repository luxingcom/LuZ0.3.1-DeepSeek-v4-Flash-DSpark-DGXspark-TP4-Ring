# ① NCCL 参数 A/B 执行方案（v0.27 环网通信优化）

- **编制**：testing-expert-1（测试专家 Tessa）
- **日期**：2026-08-14（v2 修订：N1 测试环境 GID=3 / N2 MERGE_NICS 不执行 已拍板）
- **前置**：SRE 已部署 `test-0.2.1-v027` TP4 冒烟通过；**本方案在 0.27 测试服务上执行**（若 0.27 部署未就绪，降级为 0.26 测试窗口执行，见 §3 分支）
- **GPU**：TP4 全量占用
- **预估总耗时**：**2–2.5 h**（基线 A0 20min + A1–A5 每档 ~15–20min + 决策回滚余量）

---

## 0. 目标

针对 TP4 每 token 的 **368KB all-reduce**（MLA 激活）通信，用 NCCL 环境变量 A/B 寻找比生产基线更高的 busbw / 更低的延迟，同时确保端到端 PR/DE 不劣化。

**验证载体**：`nccl-tests all_reduce`（4 rank 环）+ `bench_prefill_decode_async.py`（c1@131K PR/DE）。

---

## 1. 生产 NCCL 基线 env（已从 start_tp4_worker.02.sh v1.5-r11 实测确认）

| env | 生产值 | 说明 |
|---|---|---|
| `NCCL_ALGO` | `RING` | 环网算法（ring-only 补丁强制） |
| `NCCL_MIN_NCHANNELS` | `2` | **主 A/B 对象** |
| `NCCL_NET` | `IB` | RoCE |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | 子网感知路由 |
| `NCCL_NET_PLUGIN` | `none` | 禁插件 |
| `NCCL_IB_MERGE_NICS` | `0` | **禁合并（保 per-peer 对口）** |
| `NCCL_IB_PEER_HCA` | per-peer（见下） | 双 HCA 对口 |
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` | 4 twin 口全暴露 |
| `NCCL_IB_GID_INDEX` | **生产 head=3 / worker=2（保持不动）；0.27 测试环境统一 `3`** | N1 已拍板：测试环境统一 GID=3（社区 GB10 铁律），生产 GID 不动 |
| `NCCL_IB_TIMEOUT` / `RETRY_CNT` | `1000` / `7` | 超时/重试 |
| `NCCL_IB_TOS` | `46` | 流量类别 |
| `NCCL_CROSS_NIC` | `1` | 跨 NIC |
| `NCCL_SOCKET_IFNAME` | `enP7s7` | socket 网卡 |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | 配合 shim PSR pin |
| `NCCL_DEBUG` / `DEBUG_FILE` | `INFO` / `/var/log/vllm/nccl-%h.log` | 调试点 |
| `LD_PRELOAD` | `/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2` | shim + ring-only 补丁 |

**PEER_HCA 现值（per rank，双 HCA 对口）**：
```
rank0: 1=rocep1s0f1,roceP2p1s0f1;3=rocep1s0f0,roceP2p1s0f0
rank1: 0=rocep1s0f1,roceP2p1s0f1;2=rocep1s0f0,roceP2p1s0f0
rank2: 1=rocep1s0f0,roceP2p1s0f0;3=rocep1s0f1,roceP2p1s0f1
rank3: 0=rocep1s0f0,roceP2p1s0f0;2=rocep1s0f1,roceP2p1s0f1
```
> 每个 peer 已暴露 2 个 HCA（物理口 + twin 口）；`MERGE_NICS=0` 时 NCCL 按 channel 分配 NIC，理论上 `MIN_NCHANNELS=2` 已能双 HCA 并行。A/B 重点是**验证是否真双 HCA 并发** + **通道数上限**。

---

## 2. 0.27 环网状态决策门（依赖 SRE 冒烟结果）

0.27 是否仍需 ring-only 补丁 / PEER_HCA / 双 HCA 并发，以 SRE 冒烟确认结果为准：

```bash
# a) 0.27 容器内 NCCL 版本与来源
docker exec vllm-tp4-rank0 sh -c \
  "python3 -c 'import torch; print(torch.cuda.nccl.version())'; ldd \$(which vllm) | grep -i nccl"
# b) LD_PRELOAD 是否仍能截获 torch_nccl（ABI 风险点）
docker exec vllm-tp4-rank0 sh -c "cat /proc/1/environ | tr '\\0' '\\n' | grep -E 'LD_PRELOAD|NCCL'" 2>/dev/null
# c) 原生环网是否可用：不带 LD_PRELOAD 起 1 个 4-rank all_reduce 小测，看 NCCL init 是否走 RING + IB
# d) PEER_HCA 行为：0.27 若原生支持 subnetwork/多 NIC 路由，可能不再需要 per-peer 对口
```

**分支**：
- **分支 A（0.27 原生环网可用）**：A/B 直接在 0.27 上做，`LD_PRELOAD` 置空（或仅保留 shim 若 PSR 需要），其余 env 同生产；A7 专门验证 0.27 原生双 NIC 行为。
- **分支 B（0.27 仍需 overlay）**：A/B 在 0.27 上叠加 `LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`，env 同生产。
- **分支 C（0.27 未就绪）**：A/B 在 0.26 测试窗口（生产同款镜像、改 env 无镜像构建）执行，结论供 0.27 复用（0.27 原生环网确认后再复测关键档）。

---

## 3. A/B 矩阵

> 执行方式：复制生产启动脚本为 `start_tp4_head_b12x_ab.sh`（及 worker 变体），改其中 `ENV_ARGS` 的 NCCL 行 → 每档先 `cp <脚本> .bak-ncclA<序号>` → 重启容器 → 验证 → 下一档。

| # | 参数 | 生产基线 | 试验值 | 依据 | 风险 | 回滚 |
|---|---|---|---|---|---|---|
| A0 | 基线复测 | 生产 env | 同生产 | 同窗对照 | 低 | — |
| A1 | `NCCL_MIN_NCHANNELS` | 2 | **4** | 双 HCA 多通道并行 | 低 | 还原 2 |
| A2 | `NCCL_MIN_NCHANNELS` | 2 | **8** | 通道上限探测 | 中（CPU/内存↑） | 还原 2 |
| A3 | `NCCL_IB_QPS_PER_CONNECTION` | 默认(8) | **4 / 16** | 每连接 QP 数调节小消息并发 | 低 | 还原默认 |
| A4 | `NCCL_BUFFSIZE` | 默认 4M | **8M / 16M** | 368KB×N 大消息聚合 | 中（内存/显存↑） | 还原默认 |
| A5 | `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | **1** | QP 数据拆分（与 A3 组合） | 中 | 还原 0 |
| A6 | `NCCL_IB_MERGE_NICS` | 0 | 1 | 双 NIC 合并 | **高：破坏 per-peer 对口，默认不做** | 仅记录不执行 |
| A7 | `NCCL_IB_PEER_HCA` 双 HCA 并发 | per-peer 双 HCA | 同 peer 单 HCA vs 双 HCA 对比；或 0.27 原生多 NIC 对照 | 验证双 HCA 是否真并发 | 中 | 还原 per-peer 双 HCA |

**每档标准验证流程（~15–20 min）**：

```bash
# 1) 改脚本 env → 备份 → 重启（SRE 执行容器切换）
cp start_tp4_head_b12x_ab.sh .bak-ncclA1
#   修改 NCCL_MIN_NCHANNELS=4（worker 变体同步改）
#   顺序停 0.27 → 用改后脚本起 → 等 health 200

# 2) nccl-tests all_reduce（4 rank，环，busbw）
#    节点上安装/复用 nccl-tests，hostfile 四机
mpirun --hostfile hosts.txt -np 4 -x LD_PRELOAD -x NCCL_ALGO=RING \
  -x NCCL_MIN_NCHANNELS=4 -x NCCL_IB_PEER_HCA="<per-rank>" \
  /usr/local/bin/all_reduce_perf -b 16M -e 512M -f 2 -g 1 -n 100 -w 10
#    记录 busbw GB/s（生产基线 ~4.4 GB/s ring-only，v3 双口 ~23.86 GB/s 为历史峰值参考）

# 3) 端到端 bench（c1@131K，coding）
python3 bench_prefill_decode_async.py --group NCCA1 \
  --endpoint http://<NODE_IP>:8001/v1 --key <KEY> \
  --model deepseek-v4-flash-0731 --concurrency 1 --ctx 131072 \
  --tasks coding --rounds 3 --engine asyncio --out ./results_nccl_a1

# 4) NCCL debug 日志佐证（确认双 HCA / channel 数）
grep -iE "channel|net.*device|IB.*dev|rocep|P2p" /var/log/vllm/nccl-<host>.log | tail -30

# 5) 判定与记录
#    PASS 条件：busbw ↑（相对 A0）且 c1@131K PR/DE 不劣化（PR≥A0×0.97，DE≥A0×0.97）
#    否则：还原 .bak，重启，进入下一档
```

**判定汇总模板**：

| # | busbw (GB/s) | c1@131K PR | c1@131K DE | 双 HCA 证据 | 判定 |
|---|---|---|---|---|---|
| A0 基线 | ~4.4 | 1896.4 | 104.1 | — | 基准 |
| A1 (ch=4) | | | | | |
| A2 (ch=8) | | | | | |
| A3 (qps) | | | | | |
| A4 (buff) | | | | | |
| A5 (split) | | | | | |
| A7 (peer) | | | | | |

**最终结论规则**：
- 选出「busbw 最高且 E2E 不劣化」的档位作为**候选参数集**（通常 ≤1 档，避免多变量耦合）。
- 候选参数集需在 c1@131K + c4@131K 复测一轮确认（+20min）。
- 任何一档 E2E 劣化 → 立即回滚（容器重启 ~5min/轮）；全部档位无收益 → 维持生产 env，记录「NCCL 已接近环网最优」。

---

## 4. 双 HCA 并发专项（A7，验证而非调参）

**问题**：生产 PEER_HCA 已给每 peer 2 个 HCA，但 `MIN_NCHANNELS=2` 是否真的让两个 HCA 并行工作，需日志证实。

```bash
# 每 rank 的 NCCL debug 日志中应出现 2 个 net 设备（rocep + roceP2p 各一，或 2 个物理口）
grep -iE "net.*dev|device.*\[|IB.*port|rocep" /var/log/vllm/nccl-<host>.log | head -40
# 期望：channel0 → HCA_A，channel1 → HCA_B（双 HCA 并行）
# 若日志显示 2 channel 都落在同一 HCA → PEER_HCA 双 HCA 未生效，A7 试验「同 peer 仅列 1 个 HCA」vs「列 2 个」对比
```
**A7 判定**：双 HCA 并行为「已生效」→ 记录即可；「未生效」→ 试验 PEER_HCA 配置变体并重测 A1。

---

## 5. 回滚与恢复

- 每档回滚：`.bak-ncclA<序号>` 还原 + 重启（~5min）。
- 全程不修改生产脚本（`<INSTALL_DIR>/scripts/start_tp4_worker.sh` 等原文件不动）。
- 窗口结束恢复生产：
```bash
ssh node01 "cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh"   # 约 8 min
# 验证：8001=200 + 四机 healthy + PSR（NCCL→8-9、Engine→15-19）
```

---

## 6. 决策点（督导/用户已拍板）

- **N1（已拍板）**：0.27 测试环境统一 `NCCL_IB_GID_INDEX=3`（社区 GB10 铁律）；**生产 GID（head=3/worker=2）保持不动**。A/B 在 0.27 测试环境执行时 GID 一律用 3。
- **N2（已拍板）**：A6（`MERGE_NICS=1`）**不执行**（高破坏风险，破坏 per-peer 对口）。
- **N3（待 SRE 冒烟结论）**：0.27 分支 A/B/C 由 SRE 冒烟结果决定；若冒烟确认 0.27 原生环网可用，自动走分支 A，A7 增加「0.27 原生多 NIC 路由」对照档。
- **N4**：若 0.27 尚未就绪（分支 C），本方案在 0.26 窗口执行；结论与 0.27 的映射关系（同 env 复测）需在汇总报告中说明。
- **执行顺序（team-lead 传达）**：③ v0.27 主矩阵 → 同窗 0.26 对照 → **① NCCL A/B** → ② c5 诊断。

> 本方案由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
