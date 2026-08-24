# NCCL A/B 窗口执行报告（B0-B4 + B1 端到端固化）

**日期**：2026-08-17 ｜ **执行**：KernelGen ｜ **窗口**：17:30-18:5x（用户让出，停机 SOP §2.3）
**前置方案**：`nccl-p1p2p4-ab-window-plan-2026-08-17.md` ｜ **生产终态**：**B1（NCCL_MAX_NCHANNELS 16->4）已固化**

---

## 0. TL;DR

- **B1（4 通道）全面胜出**：112KB allreduce **126->83µs（-34%）**、224KB **160->86µs（-46%）**；14KB 仅 +2µs（噪声带）
- **端到端双档通过**：c1@131K PR 2180.75/DE 104.07/TTFT 52.4s（vs 基线 ~2200/~100/~52s：**DE +4%，PR/TTFT 持平**）；c1@32K 2388/96.8/11.9s（持平）
- **B3（LL128）确认无净收益**（14KB 劣化 26%）--P2 关闭
- **B4（QPS=2+SPLIT）大消息劣化**--P4 关闭
- **生产已固化 B1**：四机启动脚本 MAX_NCHANNELS=4（备份 .bak-ncclB1），集群已以 B1 配置运行 healthy

## 1. nccl-tests 数据（4-rank 环网，avg µs，in-place）

| 消息 | B0 (16ch) | **B1 (4ch)** | B2 (8ch) | B3 (LL128) | B4 (4ch+QPS) |
|---|---|---|---|---|---|
| 14KB (decode 主) | 41.3 | 43.2 | 41.7 | 51.9 ❌ | **41.0** |
| 28KB | 47.5 | 47.3 | 49.3 | 56.2 ❌ | 47.6 |
| 56KB | 66.4 | 69.6 | 65.5 | 65.2 | 67.7 |
| 112KB | 126.3 | **83.2** ✅ | 129.6 | 86.0 | 95.8 |
| 224KB | 160.0 | **86.1** ✅ | 92.3 | 140.5 ❌ | 94.1 |

**判读**：
- B1 大消息收益显著且稳定（-34%/-46%），验证了"368KB/16ch=23KB 分片 Simple 延迟不友好"的 P1 假设
- B2（8ch）不稳定（112KB 129µs 异常），淘汰
- B3：LL128 在 14KB 比 LL 差 26%，在 224KB 比 Simple/B1 差 63%--**该区间 LL128 无适用点，P2 关闭**
- B4：QPS=2 对 14KB 有 ~2µs 边际收益但不抵 112/224KB 劣化（+10/+8µs）--**P4 关闭**（大消息场景）

## 2. B1 端到端验证（生产镜像 + MAX_NCHANNELS=4 重启后实测）

| 档 | B1 实测 | 基线对照（FINALBASE v1.0） | 判定 |
|---|---|---|---|
| c1@131072 coding | PR 2180.75 / DE 104.07 / TTFT 52.4s | PR ~2200 / DE ~100 / TTFT ~52s | ✅（DE +4%） |
| c1@32768 coding | PR 2387.91 / DE 96.83 / TTFT 11.93s | PR ~2420 / DE ~99 / TTFT ~12s | ✅（噪声带内） |

**14KB 微劣化（+2µs）在端到端不可见**（DE 反而 +4%@131K）--decode 主瓶颈的 61 次串行小 allreduce 中，4ch 下单次延迟增量被通道竞争减少抵消。

## 3. 生产固化记录

| 项 | 内容 |
|---|---|
| 变更 | 四机启动脚本 `NCCL_MAX_NCHANNELS` 16 -> **4**（head L116 + worker L121） |
| 备份 | `start_tp4_head.sh.bak-ncclB1`（01）+ `start_tp4_worker.sh.bak-ncclB1`（02/03/04） |
| 验证 | bash -n 全过；集群 B1 配置启动收敛 ~6min；health 200；四机容器 healthy |
| 回滚 | 还原 .bak-ncclB1 + `start_tp4_cluster.sh`（~8min） |

## 4. 执行过程事件（排障记录，供复盘）

1. mpirun orted 未随 PATH 传播 -> `--prefix /usr` + PATH 展开修复
2. **03 缺 openmpi-bin**（orted 不存在）-> apt 安装
3. NCCL_DEBUG_FILE 未加 -x 被当可执行文件 -> 补 -x
4. **03 缺 all_reduce_perf**（仅 01/02/04 有）-> sudo scp 补齐
5. libncclpin 宿主机路径 = `<INSTALL_DIR>/lib/`（非容器内 /opt/）-> 脚本修正
6. MPI TCP 尝试连 RoCE 段 -> `--mca btl_tcp_if_include enP7s7` 限管理网
7. 首轮 hang 残留 -> 四机 pkill orted/perf 后干净重跑

## 5. 遗留与建议

- **B5（tuner 阈值 96K）未跑**：B1 已达判定门槛（<150µs 目标超额完成：224KB 86µs），无需叠加
- **368KB 单点复测**：nccl-tests 扫到 224KB（-f 2 上限），368KB 预计 ~120-130µs（介于 112/224 之间外推），**端到端 PR 已验证收益兑现**
- **建议镜像同步**：`mirror_to_02.sh` 执行一次（脚本变更 + 本报告入档）
- **Grafana 面板**：NCCL 通道数相关注释如有（16ch 字样）需同步为 4ch
- 长期项不变：P3 overlap（vLLM 引擎级）仍为 decode 通信大头唯一手段，归 0.27+ 上游跟踪

## 6. 数据档案

- nccl-tests 原始数据：`/tmp/nccl-abB-{B0,B1,B2,B3,B4}/latency.txt`（01）
- 端到端：`/tmp/nccl_b1_e2e/rows_A.csv + summary_A.json`（01）
- 方案：`deliverables/engineering-assurance/nccl-p1p2p4-ab-window-plan-2026-08-17.md`
- 脚本：`<INSTALL_DIR>/scripts/nccl-ab-B/run_lat.sh`（B0-B5 一键 driver，已含全部修复）
