# NCCL P1+P2+P4 打包 A/B 窗口执行方案（备灾就位版）

**日期**：2026-08-17 ｜ **编制**：KernelGen ｜ **状态**：备灾中（等用户停机窗口）
**前置分析**：`nccl-operator-latency-analysis-2026-08-17.md`（P1/P2/P4 定义）+ `nccl-proto-threshold-scan-2026-08-16.md`（协议扫描依据）

---

## 0. 关键事实（源码核实结论）

### P1 是 env 级，无需重编
stageB tuner（enqueue.cc L2153-2176，PerSizeTuner 双带）**按 `nBytes`（消息总量）判协议**，阈值读 `NCCL_TUNER_THRESHOLD`：
```cpp
int wantProto = (nBytes <= th) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;  // th 默认 40960
```
**P1 的本质 = 把 368KB（prefill 主消息）也赶进 LL**--但 8/16 扫描实测 368KB 在 LL 下 **20× 爆炸**！
-> **P1 修正**：不是"368KB 走 LL"，而是"**368KB 档减通道**（16ch -> 4ch，每通道分片 92KB，Simple 合理区间）"。通道数对延迟的作用机制 = 减少每通道分片尺寸 + 减少 QP 竞争。
**实施 = 纯 env**：`NCCL_MAX_NCHANNELS=4`（或 tuner 逻辑不动，仅调 env）。

### P2 是 env 级（LL128 中间档）
`NCCL_PROTO=LL128` 是 NCCL 原生 env。**冲突点**：tuner 代码在 `NCCL_PROTO` 设置时会... 核实：tuner 无 `NCCL_PROTO != NULL` 守卫（该守卫只在 CTA_POLICY 分支）-> LL128 env 会与 tuner 覆盖冲突，**需临时清空 tuner**（`NCCL_TUNER_THRESHOLD` 置 0 会 fallback 到默认 40960？核实：tenv 解析 t>0 才覆盖，置 "0" 则保留默认 40960 --**无法用 env 关闭 tuner**）。
-> **P2 执行方式**：单独档位直接设 `NCCL_PROTO=LL128`，此时 cost table 仍被 tuner 改写... 
**决策**：P2 档临时改用 **vLLM bench 对比**（不依赖 tuner 开关）：a) 生产现配置（tuner 40KB）vs b) `NCCL_PROTO=LL128` 强制全局（观察 368KB 是否劣化 + 14KB 是否改善）。若 LL128 全局强制下 368KB 劣化 >5%，则 P2 需等 tuner 加 LL128 分支（代码级，下一轮）。

### P4 是 env 级
`NCCL_IB_QPS_PER_CONNECTION=2` + `NCCL_IB_SPLIT_DATA_ON_QPS=1`（T3 遗留）。

## 1. A/B 档位矩阵（每档 ~15min）

| # | 配置 | 目的 | 验证重点 |
|---|---|---|---|
| **B0** | 生产现配置（tuner 40KB/16ch/8M） | 同窗基线 | 173µs 复现 |
| **B1** | `NCCL_MAX_NCHANNELS=4`（其余同 B0） | **P1**：368KB->92KB/通道 | 368KB µs↓？14KB 不劣化？ |
| **B2** | `NCCL_MAX_NCHANNELS=8` | P1 备选粒度 | 同上 |
| **B3** | `NCCL_PROTO=LL128`（全局强制） | **P2**：LL128 空白验证 | 14KB µs？368KB 劣化幅度 |
| **B4** | B1 + `NCCL_IB_QPS_PER_CONNECTION=2 NCCL_IB_SPLIT_DATA_ON_QPS=1` | **P4**：深管道 | 368KB/14KB 双向 |
| **B5**（视 B3 结果） | tuner 阈值微调 `NCCL_TUNER_THRESHOLD=98304`（96KB，368KB/4ch） | P1 变体 | 92KB 分片协议翻转点 |

**执行载体**：`/opt/nccl-tests/all_reduce_perf`（4 rank mpirun 环网）+ `bench_prefill_decode_async.py`（c1@131K PR/DE + c1@32K）。

## 2. 标准测试命令（每档）

```bash
# ① nccl-tests 单尺寸延迟（关键尺寸各 3 档）
mpirun -np 4 --hostfile /opt/nccl-tests/hosts.txt \
  -x LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2" \
  -x NCCL_ALGO=RING -x NCCL_NET=IB -x NCCL_NET_PLUGIN=none \
  -x NCCL_IB_HCA=<4口> -x NCCL_IB_PEER_HCA=<per-rank> -x NCCL_IB_GID_INDEX=3 \
  -x NCCL_MIN_NCHANNELS=<档位值> -x NCCL_MAX_NCHANNELS=<档位值> \
  -x NCCL_BUFFSIZE=8388608 -x NCCL_TUNER_THRESHOLD=40960 \
  [-x NCCL_PROTO=LL128] [-x NCCL_IB_QPS_PER_CONNECTION=2 -x NCCL_IB_SPLIT_DATA_ON_QPS=1] \
  /opt/nccl-tests/build/all_reduce_perf -b 14K -e 368K -f 2 -g 1 -n 200 -w 10
# 记录: 14K/32K/64K/128K/368K 各档 avg latency µs（-o csv 或日志解析）

# ② 端到端（仅 B0/B1/B3/B4 跑，c1@131K + c1@32K 各 3 rounds）
python3 bench_prefill_decode_async.py --group NCCL_B<n> \
  --endpoint http://<NODE_IP>:8001/v1 --key <KEY> --concurrency 1 --ctx 131072 \
  --tasks coding --rounds 3 --engine asyncio --out ./results_nccl_b<n>

# ③ NCCL 调试日志佐证（通道数/协议实际选择）
NCCL_TUNER_DEBUG=1（容器 env 临时加）或 grep 日志
```

## 3. 判定门槛（定版）

| 档 | 通过条件 | 不通过处置 |
|---|---|---|
| B1/B2 | 368KB µs < **150**（vs B0 173）且 c1@131K PR/DE 劣化 <3% 且 14KB µs 不升 >5% | 还原 16ch |
| B3 | 14KB µs 降 ≥10% 且 368KB 劣化 <5% -> 记录 LL128 价值，下轮 tuner 加分支 | 若 368KB 爆炸 -> 确认 LL128 不适用全局，P2 关闭 |
| B4 | 368KB 或 14KB 任一 ↓≥5% 且端到端不劣化 | 还原默认 QPS |

**总判定**：选出 µs 最优且端到端不劣化的组合 -> c1@131K + c4@131K 复测一轮 -> 生产固化（改 start_tp4 脚本 env + 备份）。

## 4. 回滚预案

- 全程只动**测试命令 env**（mpirun -x）与容器临时 env，**不改生产脚本/不改 .so**
- 端到端档需重启容器换 env：每档前 `cp start_tp4_head.sh .bak-ncclB<n>`，档毕还原重启（~5min/次）
- 终态恢复：`bash <INSTALL_DIR>/scripts/start_tp4_cluster.sh`（~8min）+ 验证 8001=200 + 四机 healthy + PSR 正常
- **【SRE 检查点·2026-08-17 复盘新增】窗口结束必须恢复 monitor 自愈服务**（2026-08-17 B1 窗口曾遗漏：09:31 停 monitor 后未恢复，自愈链断裂，容器经 cluster 脚本拉起后无 docker wait 跟随）：
  1. 四机 `sudo systemctl start vllm-tp4-head.service`（01）/ `vllm-tp4-worker.service`（02/03/04）
  2. 确认 `systemctl is-active` = active 且 MainPID 有效（docker wait 附着已运行容器，容器零扰动）
  3. `systemctl is-active vllm-healthcheck.timer` = active（01）+ 跑一次 `bash <INSTALL_DIR>/scripts/healthcheck.sh` = exit 0

## 5. 窗口时间预算

| 步骤 | 耗时 |
|---|---|
| B0 基线（nccl-tests + e2e） | 20min |
| B1/B2/B4 各档（nccl-tests + 选档 e2e） | 3×15min |
| B3 LL128 | 15min |
| 复测 + 生产固化 + 恢复 | 30min |
| **合计** | **~2h**（含容器重启余量） |

## 6. 待窗口前确认项（已就位/待办）

- [x] tuner 源码逻辑已核实（env 级实现确认）
- [x] nccl-tests 四机就位（/opt/nccl-tests/build/all_reduce_perf）+ mpirun 可用
- [x] 8/16 阈值扫描数据（40KB 翻转点、LL 大消息爆炸边界）
- [x] **执行脚本已落库**：`<INSTALL_DIR>/scripts/nccl-ab-B/`（hosts.txt 环序 01-02-04-03 + run_lat.sh 一键 B0-B5 driver，bash -n 过）
- [ ] 窗口期间生产流量确认清零（running=0 检查）
- [ ] 端到端档 API key 执行时从 vllm.env 读取
- [x] **窗口结束恢复 monitor 自愈服务**（2026-08-17 B1 后复盘新增，SOP §4 终态恢复必做）

## 7. 重要设计修正（源码核实后）

1. **P1 原设想（368KB 走 LL）被 8/16 扫描数据否决**--368KB 在 LL 下 20× 爆炸。修正为"**减通道到 4**"（368KB/4ch=92KB/通道，Simple 合理区间）
2. **P2 的 LL128 与 tuner 存在覆盖冲突**（tuner 无法用 env 关闭：`NCCL_TUNER_THRESHOLD=0` 不生效，fallback 默认 40960）-> B3 档直接全局强制 LL128 观察双向影响；若 368KB 劣化 >5%，LL128 需 tuner 加第三分支（代码级，下一轮）
3. per-rank PEER_HCA 四份已从生产脚本提取（环序 01-02-04-03），内嵌于 run_lat.sh
