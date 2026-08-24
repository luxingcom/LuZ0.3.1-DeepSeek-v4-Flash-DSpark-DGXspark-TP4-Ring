# G1 补齐 + 生产恢复报告（2026-08-24）

- **执行**: 雷克斯（Rex）· SRE 工程师
- **UTC 时间窗**: 2026-08-23 23:50 → 2026-08-24 00:50
- **集群**: node01~04（4×DGX Spark GB10, TP4, <NODE_IP>/187/188/189）
- **版本**: LuZ0.3.1（0.2.1-v026.0 基座 + overlay: W4A4=2/SHARED=1/thr4096/util0.82/MTP n7/FI 0.6.16/wsdedup）
- **数据资产**: 服务器 /tmp/_bench_luz031/ + 本地 deliverables/engineering-assurance/_luz031_official_bench/data/

---

## 0. 执行摘要

| 步骤 | 结果 | 说明 |
|---|---|---|
| ① G1 数据补齐（W4A16 同窗） | ✅ 完成 | 克隆集群 VLLM_MOE_W4A4=0 全矩阵，对照表生成，克隆已停删 |
| ② CUMEM=0 落地 | ✅ 完成 | 四机 start 脚本 env 单行变更 + 留档 + checker PASS |
| ③ 生产 LuZ0.3.1 启动 | ✅ 完成 | start_tp4_cluster.sh B12X 门禁死锁后改直启路径，核验全项 PASS |
| ④ embed 服务 | ⚠ 受阻 | vLLM worker 占 ~103GiB，embed 启动 util 检查失败（crash-loop 已止血），待决策 |
| ⑤ 守卫/自恢复/持久化 | ⚠ 需 root | systemd 自愈链四机未激活，需 root 执行（命令已列）；监控数据源确认 ✓ |
| ⑥ G2-G7 归档 | ✅ 完成 | G2 基线在位 / G3 已入 runbook §H / G4 引用 / G7 状态（cron 缺口已记录） |

---

## ① G1 对照表（W4A16 vs W4A4 同窗 decode-only）

> 同窗口径：均为 LuZ0.3.1-bench-20260823 克隆集群（util0.82/thr4096/batched4096/MTP n7/FI0.6.16），仅差 VLLM_MOE_W4A4（W4A16=0 vs W4A4=2）；预热 24；S1=10 轮 / 其余 5 轮；decode-only（自首 token 计时）。
> Δ = (W4A16 − W4A4) / W4A4；Δ>0 = W4A16 更快。Session A(W4A4) 于 08-23 08:57 UTC 采集，G1(W4A16) 于 08-24 00:02 UTC 采集。

### 1.1 单流 decode-only（t/s）

| 项 | W4A4（中位\|最优） | W4A16（G1 中位\|最优） | Δ中位 | Δ最优 | 解读 |
|---|---|---|---|---|---|
| S1_fox_p512 | 77.8 \| 131.1 | 74.7 \| 95.6 | -4.0% | -27.1% | 单流噪声档 |
| S2_fox_p256 | 85.0 \| 105.3 | 96.7 \| 123.8 | +13.8% | +17.6% | W4A16 更快 |
| S3_list | 98.0 \| 110.2 | 103.1 \| 106.4 | +5.2% | -3.4% | W4A16 更快 |
| S4_agent_tool | 84.2 \| 129.6 | 82.7 \| 90.7 | -1.8% | -30.0% | 单流噪声档 |

### 1.2 并发聚合 decode-only（t/s）

| 并发 | W4A4（中位\|最优） | W4A16（G1 中位\|最优） | Δ中位 | Δ最优 | 解读 |
|---|---|---|---|---|---|
| C1 | 73.9 \| 136.5 | 71.6 \| 93.9 | -3.1% | -31.2% | 单流噪声档 |
| C4 | 186.7 \| 218.8 | 203.5 \| 213.2 | **+9.0%** | -2.6% | W4A16 更快 |
| C8 | 274.5 \| 348.3 | 306.4 \| 320.3 | **+11.6%** | -8.0% | W4A16 更快 |
| C12 | 349.3 \| 397.5 | 411.8 \| 441.9 | **+17.9%** | +11.2% | W4A16 更快 |

### 1.3 Agent 5 场景 decode-only（t/s）

| 场景 | W4A4（中位\|最优） | W4A16（G1 中位\|最优） | Δ中位 | Δ最优 | 解读 |
|---|---|---|---|---|---|
| Math | 90.4 \| 97.3 | 87.5 \| 109.4 | -3.2% | +12.4% | 接近/噪声 |
| JSON | 78.3 \| 81.4 | 86.0 \| 90.1 | +9.8% | +10.7% | W4A16 更快 |
| Code | 90.2 \| 108.5 | 93.8 \| 101.0 | +4.0% | -6.9% | W4A16 更快 |
| Communication | 53.8 \| 54.9 | 55.0 \| 57.0 | +2.2% | +3.8% | W4A16 更快 |
| Narrative | 39.6 \| 40.7 | 41.5 \| 42.0 | +4.8% | +3.2% | W4A16 更快 |
| **平均** | **70.4 \| 76.5** | **72.8 \| 79.9** | **+3.4%** | **+4.4%** | |

### 1.4 附：KV / 形态

- KV tokens：W4A16 6,314,440 vs W4A4 5,796,156（+0.52M，W4A16 更高，与 release notes「KV vs W4A16 基线 -4.5%」同向）
- 数据文件：g1_w4a16_m1_single.json / m2_conc.json / m3_agent.json / g1_w4a16_vs_w4a4.md / regression log（服务器 + 本地均留档）

### 1.5 SRE 判定

- **W4A4 full 的 decode 代价在并发下显著且随并发放大**：C4 +9.0%、C8 +11.6%、C12 +17.9%（与 phase3b 代价带 -6~-9% 同向、偏大；w4a4-ext ±3% 口径下超带）。
- 单流噪声大（fox 40-124 已知波动）不足以判定单流差异；Agent 平均 W4A16 +3.4%。
- 结论：G1 量化补齐完成；业务以 prefill+并发为主、W4A4 的 prefill 收益见 G4，**不改变生产采纳 W4A4 的结论**。后续可关注 C12 档 W4A4 代价（+17.9%）是否影响高并发业务。

---

## ② CUMEM=0 落地确认

- **变更**: `start_tp4_head.sh`（01）+ `start_tp4_worker.sh`（02/03/04）ENV_ARGS 增加 `-e 'NCCL_CUMEM_HOST_ENABLE=0'`（插于 `NCCL_IB_TOS=46` 之后）。
- **留档**: `.bak-cumem0-20260824` 四机 OK。
- **diff 核对**: 每机仅新增一行 `>   -e 'NCCL_CUMEM_HOST_ENABLE=0'`，无其他差异。
- **checker**: check_vllm_script.sh 四机 PASS。
- **生效确认（双证据）**:
  - 容器 env：`docker exec vllm-tp4-rank0 sh -c 'echo $NCCL_CUMEM_HOST_ENABLE'` → `0`（worker 同）。
  - NCCL 日志：`NCCL_CUMEM_HOST_ENABLE set by environment to 0`。
  - ⚠ 提示：NCCL 日志另一处 `cuMemEnable 1` 是**对称内存导出（cuMemExport）**标志，与 NCCL_CUMEM_HOST_ENABLE（host 缓冲池）**无关**，勿误读为未生效。

---

## ③ 生产 LuZ0.3.1 启动核验清单（全项 PASS）

### 3.1 启动过程

- `start_tp4_cluster.sh` 因 **B12X 门禁死锁** 中止（详见 3.3）。经确认后采用直启路径：head-first + 5s 错峰 worker（= release notes §5 head-first 重建同款），四机 vllm-tp4-rank0..3 就绪耗时 ~500s，均 healthy。
- 注：systemd start 需 root（<USER> 仅 vllm.env NOPASSWD），故走 ssh 直启；生产 start 脚本未改动。

### 3.2 核验清单

| # | 项 | 实测 | 判定 |
|---|---|---|---|
| 1 | env W4A4 | `VLLM_MOE_W4A4=2`（head+worker） | ✅ |
| 2 | env SHARED/CG/MIN_M | `VLLM_B12X_SHARED_WRAPPER=1` / `VLLM_MOE_W4A4_CG=1` / `MIN_M=3072` | ✅ |
| 3 | CUMEM=0 生效 | env + NCCL 日志见 §② | ✅ |
| 4 | FlashInfer | 容器 import → **0.6.16** | ✅ |
| 5 | thr4096 | `max_num_batched_tokens=4096` + `long_prefill_token_threshold=4096` | ✅ |
| 6 | util | `gpu_memory_utilization=0.82` | ✅ |
| 7 | MTP | `num_spec_tokens=7`（dspark, probabilistic） | ✅ |
| 8 | KV dtype | `nvfp4_ds_mla` | ✅ |
| 9 | KV tokens | **5,796,156**（≥5.7M 门过） | ✅ |
| 10 | W4A4 后端 | `Using 'B12X_MXFP4' Mxfp4 MoE backend` + `Using W4A4B12xExperts` | ✅ |
| 11 | /health | **200** | ✅ |
| 12 | quality_gate | 首跑 3/4（code 函数名 fib↔fibonacci 运行级非确定），连跑 2 次 **4/4 PASS** | ✅（备注 flaky） |
| 13 | 模式探针 | 3×8K TTFT **2.77/2.80/3.15s**（<6s 阈值），prefill 2925 tok/s（基线 2510 带内） | ✅ |
| 14 | DE 抽验 | C1 step_eff **19.0**（基线 18.2，thr4096 实测带 18.5-19.2） | ✅ |
| 15 | cudagraph | PIECEWISE 16/16 + FULL 12/12 + DSpark 11/11（日志实证） | ✅ |

### 3.3 事故记录：start_tp4_cluster.sh B12X 门禁死锁

- **现象**: 00:23:41 编排中止，head `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`；B12X 门禁 300s 超时。
- **根因**: 死锁——① 编排 step 2.5 等 head 日志出现 `Using 'B12X_MXFP4'` 才启动 worker（防多 worker 并行撞 b12x JIT 竞态）；② 但 head 引擎核心初始化（parallel_state.py backend=nccl）需全部 4 rank 加入 NCCL 通讯域才会推进到 MoE/B12X 加载；③ head 单独启动阻塞在 NCCL peer 等待，300s `--distributed-timeout-seconds` 后引擎核心失败。→ worker 被门禁挡住、B12X 又依赖 worker，死锁。
- **证据**: head-only 复跑（00:24:46）阻塞 NCCL init，~300s 后复现同一 RuntimeError。
- **修复**: 采用直启路径（head + 5s 错峰 worker），无死锁。建议后续修复 start_tp4_cluster.sh 门禁逻辑或弃用该编排（走 systemd 自愈链，见 §⑤）。

---

## ④ embed 服务状态

- **当前**: 03/04 无 anemll-embed-8022 在运行（已停删 crash-loop 容器）。
- **尝试结果**: 按 start_embed_8022.sh 启动后 03/04 均 crash-loop（03 restarts=30，04 restarts=14）。
- **根因**: `ValueError: Free memory on device cuda:0 (7.78/121.63 GiB) < desired GPU memory utilization (0.92, 111.9 GiB)`。vLLM 生产 worker（util 0.82）占 ~103 GiB，剩余 ~7.78 GiB；embed 启动脚本未显式传 `--gpu-memory-utilization`，vLLM 回落默认 util 0.92 → 启动检查需 111.9 GiB → 失败。
- **基线说明**: 生产基线为 embed 先起（占 ~6-8GB）再起 vLLM worker；当前顺序反转导致 embed 起不来。
- **待决策选项**: A) 脚本加 `--gpu-memory-utilization 0.05` 尝试共存（改脚本+偏紧，OOM 风险）；B) 保持停机，下次 vLLM 重启窗口按 embed-first 恢复（推荐）；C) 短暂重启 03/04 vLLM worker 重排（有 TP4 抖动风险）。
- **建议**: 方案 B。SRE 已向主理人请示。

---

## ⑤ 守卫/自恢复/持久化恢复清单

### 5.1 现状（需 root 完成激活）

| 项 | 当前 | 目标 | 说明 |
|---|---|---|---|
| vllm-tp4-head.service | disabled/inactive | enabled+active | monitor 接管 head 自愈 |
| vllm-tp4-worker.service (02/03/04) | enabled/inactive | enabled+active | monitor 接管 worker 自愈 |
| vllm-healthcheck.service | disabled/inactive | enabled | timer 触发目标 |
| vllm-healthcheck.timer | enabled/inactive | enabled+active | 60s 探针+主动重建 |

- **阻塞**: systemd enable/start 需 root；<USER> sudo 需密码（NOPASSWD 仅 vllm.env）。已向主理人报告，待 root 操作员执行下列命令：
  - node01: `systemctl enable --now vllm-tp4-head.service`；`systemctl enable --now vllm-healthcheck.service`；`systemctl enable --now vllm-healthcheck.timer`
  - node01/03/04: `systemctl enable --now vllm-tp4-worker.service`
- **安全性**: monitor 检测容器已运行会 docker wait 跟随，正常容器零扰动；崩溃时触发 head-first 全链重建（已设计验证）。

### 5.2 已完成确认

- **Prometheus/告警/监控数据源**: aicad-prometheus-1 / alertmanager / grafana / dcgm-exporter / node-exporter 均运行；Prometheus `job_name=vllm` 已配置抓取 `<NODE_IP>:8001/metrics`（head /metrics 200，5s 间隔抓取已见日志）✅
- **自愈链设计组件**: monitor_tp4_head.sh / monitor_tp4_worker.sh / healthcheck.sh / healthcheck-rebuild.sh / systemd unit 文件均四机在位 ✅
- **持久化状态汇总**: 生产 vLLM 容器 --restart no（设计基线，systemd 接管）；监控栈 restart:always（compose 源）；embed 用 unless-stopped（有意设计）。主机重启持久化依赖 systemd 服务 enabled（当前 head/healthcheck 未启用，待 root 激活）。

### 5.3 待决策：生产容器 restart=always

- 与既有自愈设计冲突（self-recovery.md L19 明确 vLLM 容器 --restart no，生命周期由 systemd 掌控）。已向主理人请示；在 systemd 恢复前建议不追加 docker update --restart=always，待 root 激活 systemd 后保持 --restart no。

---

## ⑥ G2-G7 归档清单

| 项 | 内容 | 状态 |
|---|---|---|
| **G2** golden 基线固化 | `reference-b1w4a16-fi016-20260823.json` 已在 `<INSTALL_DIR>/backup/quality-gate/` 且 `reference-latest.json` 指向之（08-23 05:26，即 LuZ0.3.1 采纳验收快照） | ✅ 在位 |
| **G3** needle 口径统一 | 已新增 runbook §H「needle 抽验口径统一建议」（脚本集中 / 标准位置 mid/late / 判定标准 / 记录落点；备份 `.bak-g3needle-20260824`）；64K 3/3 PASS、128K late 抖动已记 | ✅ 已入 runbook |
| **G4** DE 明细引用 thr2048-retest | 引用 `/tmp/_thr2048_retest/`（4 臂 A1/B1/A2/B2，thr_analyze.py）：thr4096 DE C1 step_eff 18.5-19.2 / C12 83.2-83.9，PR 4K ~2960-2986，mode_probe_ttft 3.08-3.63s；生产抽验 C1 step_eff 19.0 在此带内 | ✅ 已引用 |
| **G7** 日志门归档 | logrotate `/etc/logrotate.d/vllm-tp4` 四机在位；journald `SystemMaxUse=500M` 四机在位；clean_tmp_logs.sh 四机在位；⚠ **每周 clean cron 未安装**（2026-08-17 文档称已部署，实测 root crontab 无 `clean_tmp_logs` 条目，需 root 补装） | ⚠ cron 缺口 |

---

## 7. 遗留事项与建议

1. **root 激活 systemd 自愈链**（§5.1 命令）— 步骤 5 关键阻塞，需 root 操作员执行后 SRE 复验。
2. **embed 恢复决策**（§4）— 建议方案 B（下次窗口 embed-first）。
3. **start_tp4_cluster.sh B12X 门禁死锁** — 建议修复门禁逻辑（TCPStore 就绪后错峰启动 worker，替代 B12X 等待）或标记弃用，避免下次误用。
4. **G7 clean_tmp_logs cron 缺口** — 建议 root 补装 `30 3 * * 0 <INSTALL_DIR>/scripts/maintenance/clean_tmp_logs.sh`。
5. **quality_gate `code` prompt flaky** — 运行级非确定（fib↔fibonacci），若频繁 3/4 可考虑与 reason/zh 同款除名或改用包络判据，待评估。
