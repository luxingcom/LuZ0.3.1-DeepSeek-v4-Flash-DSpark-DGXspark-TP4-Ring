# Threshold 4096 生产采纳报告

**日期**：2026-08-22（服务器 UTC）
**执行**：SRE 工程师（thr4096-adopt）
**结论**：**采纳成功**——三门验证全过，生产以 `--long-prefill-token-threshold 4096` 运行，自愈链已恢复。PR 四档 +8.5% ~ +13.5%，DE 归一后无回退，质量门通过。

---

## 1. 变更记录（UTC 时间线）

| 时间 | 操作 |
|------|------|
| 14:09-14:10 | 停自愈链（01: `vllm-healthcheck.timer`+`vllm-tp4-head.service`；02/03/04: `vllm-tp4-worker.service`），停前查状态确认 active，停后容器存活 |
| 14:14 | 标准停机完成：四容器 `rm -f`（rank0~3），GPU 零残留进程，8001/25999 端口空闲 |
| 14:16:00 | 四机同步改 threshold 1024→4096（01 head 脚本 L57 + 02/03/04 worker 脚本 L56） |
| 14:16:10 | head-first 重启（restart_tp4.sh 手动编排，绕过 B12X 门禁死锁） |
| 14:21:39 | READY（Application startup complete） |
| 14:24 | 重启验证通过 |
| 14:24-14:32 | 采纳验证（PR panorama / DE / greedy 质量门） |
| 14:37-14:40 | 仪器化稳定性采集（14 轮 4K/16K 交替） |
| 14:45 | 自愈链恢复 + 最终验证 |

**留档（md5）**：

| 文件 | 备份（.bak-thr4096-20260822，1024 原态） | 改后（4096） |
|------|------|------|
| 01 start_tp4_head.sh | b22c91b601cdc0ed5096d5b59023b0d3 | 7ca88744324ff5fc8460be2d40b09d3c |
| 02/03/04 start_tp4_worker.sh | e610bef72c8b127d1766fd1eac35b103（×3 一致） | d314df39fe2b74bf4fa297decc62acce（×3 一致） |

备份 md5 与昨日 restore 后的 1024 原态完全一致，确认回滚点干净。回滚脚本：`/tmp/_thr4096/rollback_threshold.sh`。

## 2. 重启验证

- 4 rank healthy（rank0@01 / rank1@02 / rank2@04 / rank3@03）
- health 200
- **threshold=4096 live**（启动参数核验：`--long-prefill-token-threshold 4096`）
- B12X_MXFP4 MoE backend 在场；B12X route-pack 预热完成（1~4096 capacities）
- DSpark MTP n=7 在场（`num_spec_tokens=7`，acceptance 草稿模型 96 params 已加载）
- GPU KV cache size: **6,046,679 tokens**（1024 会话 6,013,432，+33K 属内存布局差异）
- dspark CUDA graphs (FULL) 11/11 捕获完成

## 3. 采纳验证

### 3.1 PR 四档 panorama（3 轮中位，模式探针 OK：首请求 TTFT 3.92s / 第二请求 2.88s）

| 档位（标签→实际 tokens） | 4096 实测 | 1024 基线 | Δ | 判据 |
|------|------|------|------|------|
| 4K（8.2K tok） | **2849 tok/s** | 2510 | **+13.5%** | ≥+8% ✓ |
| 16K（32.8K） | **2829 tok/s** | 2500 | **+13.2%** | ≥+8% ✓ |
| 32K（65.5K） | 2724 tok/s | 2420 | +12.6% | 无回退 ✓ |
| 64K（131K） | 2462 tok/s | 2270 | +8.5% | 无回退 ✓ |

与昨日探路（+12%/+13%）带内复现；逐轮方差极小（4K 三轮 2837/2856/2849）。per-token 392→351μs 量级与预期一致。

### 3.2 DE（C1/C12，各 4 轮取 r1-r3 中位，接受率归一 step_eff = tput / tokens_per_step）

| 指标 | 1024 基线* | 4096 实测 | Δ |
|------|------|------|------|
| C1 tput 中位 | 80.9 tok/s | 72.6 tok/s | -10%（接受率波动） |
| C1 step_eff 中位 | 18.9 | 19.9 | **+5.0%（改善方向，带内）** |
| C12 tput 中位 | 365.8 tok/s | 390.9 tok/s | +6.9% |
| C12 step_eff 中位 | 90.1 | 93.0 | **+3.2%（带内 ✓）** |

*基线采自 14:04 重启会话（慢模式），验证采自快模式会话；原始 tput 差异由概率草稿采样（`draft_sample_method=probabilistic`）接受率波动解释（acc/draft 轮间波动 2.1-3.5）。归一后无回退 >5%，**DE 判据通过**，"DE 与 threshold 无关"结论再次成立。

### 3.3 质量门（greedy 短 logprob 对照，6 prompt，temp=0）

- 4/6 稳定 prompt（fox_repeat / count / code / list）：1024 参考 vs 4096 输出**逐字一致** ✓
- 2/6（reason / zh）DIFF——**已证与 threshold 无关**：`/tmp/_mtp_tune/greedy_ref_run1.json` vs `run2.json`（同 1024 配置连跑两次）在这两个 prompt 上同样 DIFF。漂移属概率投机解码既有运行级非确定性；两 prompt 均 <100 token，低于两种 threshold，prefill 计算路径完全相同。
- **判定：PASS**。建议复盘项：greedy 质量门应固定用 4 个稳定 prompt，或改用 logprob KL 散度替代逐字比对。

## 4. 仪器化稳定性采集（慢轮根因调查供数）

**窗口**：14:37:34-14:39:54 UTC（140s），14 轮 4K/16K 交替（各 7 轮），threshold 4096 快模式会话。

**慢轮观测：0/14（0%）**——本次采集未出现慢轮（判据 TTFT > 同档中位×1.3）。频率如实记录为 0%；快模式下 4K/16K TTFT 极稳（4K 2.869-2.908s，16K 11.537-11.588s，极差 <1.4%/0.5%）。

**同步采集仪器**（全部落盘 01:/tmp/_thr4096/stability/，含每轮 epoch 时间戳可对齐）：

| 仪器 | 数据 | 摘要 |
|------|------|------|
| dmon ×4 节点（1s） | dmon_01/02/03/04_stab.log | SM avg 71-72% / max 96% / p95 96%；功率 avg 49-52W / max 64-68W；四节点均衡（±0.8%） |
| 宿主守护 CPU（2s，ps PSR 逐核） | host_daemon_cpu.tsv（1431 行） | containerd 1.9%、dcgm-exporter 2.0%、dockerd 0.7%、node_exporter 0.7%（生命周期均值口径）；load1 avg 2.5 / max 3.39 |
| RDMA 计数器（0.5s，4 口 ×5 计数） | rdma_stab.tsv（245 采样） | **零错误零丢弃**；窗口 xmit 总量 312.6 GB |
| 每轮 TTFT | stab_4096.json | 4K med 2.878s / pr med 2850.6；16K med 11.564s / pr med 2834.1 |

**已知局限**：ps pcpu 为进程生命周期均值而非瞬时值（瞬时占用需 pidstat 口径，后续可升级采样器）；load1 与 dmon 为瞬时口径。零慢轮下本轮数据作为快模式基线参考保留；慢轮事件数据本次未捕获到。

**慢轮相关旁证（本日观测）**：14:04 会话（1024）探针 4K 4.30s ≈ 1907 tok/s（vs 正常 ~2500），同日 14:16 会话（4096）探针 2.88s——重启级模式方差再次实证（±8-24% 档），与既有认知一致：模式由重启落点决定，与 threshold 取值无关（两日 4096/1024 均见快慢会话）。

## 5. 生产终态

- **threshold 4096 保持启用**（采纳态，非测试态）：四机脚本已固化 4096
- 自愈链全链 active：01 head.service + healthcheck.timer；02/03/04 worker.service
- health 200 / 4 rank healthy / B12X_MXFP4 / dspark n=7 / KV 6,046,679 tokens
- 回滚路径：`/tmp/_thr4096/rollback_threshold.sh`（恢复 .bak-thr4096-20260822）+ restart_tp4.sh，全程 <10 分钟

## 6. 事件与复盘项

1. **事件（已自愈）**：13:58-14:04 采集 DE 基线时，C12 高负载使 /health 探针超时，自愈链误判"服务不可用"触发 rank0 主动重建（healthcheck-rebuild 冷却机制正常拦截了后续连环重建），集群 5 分钟内自愈恢复。本变更窗口全程先停自愈链再操作，未再发生。
   - **复盘建议**：healthcheck-rebuild.sh 的 /health 探针在高负载下误判率高，建议加负载感知（如查 /metrics running_requests）或放宽探针超时；基准作业前应制度化"先停 healthcheck.timer"。
2. **复盘建议**：greedy 质量门改用稳定 prompt 子集或 logprob KL 口径（见 3.3）。
3. **复盘建议**：宿主守护 CPU 采样升级为 pidstat 瞬时口径（见 §4 局限）。

## 7. 数据位置

- 工作目录：`node01:/tmp/_thr4096/`
  - `baseline_1024/`：变更前基线（probe / greedy_ref_1024 / de_base_1024）
  - `verify/`：采纳验证（probe_4096 / panorama_4096.log / de_4096 / greedy_4096.log）
  - `stability/`：稳定性采集（stab_4096.json / analysis.json / stab_run.log + 仪器数据）
  - `logs/`：全程操作日志
  - `rollback_threshold.sh`：回滚脚本
- 脚本备份：四机 `<INSTALL_DIR>/scripts/*.bak-thr4096-20260822`
