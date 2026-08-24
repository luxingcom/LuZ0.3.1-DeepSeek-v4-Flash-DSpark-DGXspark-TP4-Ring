# Runbook：TP4 v1.5（R11 修复增量）2026-08-12

**版本**：v1.5-R11｜**维护**：Docu｜**前置**：先读 `README.md` + `ops/ops-discipline-quickref.md`
**关系**：本文为 `runbook-tp4-append-2026-08-12.md`（§A-D，TP4 上线基线）之后的 **R11 修复增量**；与旧 runbook 合并时以本文覆盖旧 §2/§4 中"隔离核 1-4/seqs=12/StartLimit 600/8"等过期值。

---

## §A. R11 修复内容（8/12 落地，四机重启验证 7 项全过）

### A.1 修复清单
| # | 修复 | 变更 | 落地证据 |
|---|---|---|---|
| 1 | **worker 就绪门禁** | worker monitor 等 head TCPStore:25999 就绪后再启容器，避免冷启动抢跑 | runbook append §A.5 |
| 2 | **monitor 退避** | systemd StartLimit **1800s/20 次**（原 600/8），防崩溃循环误入 failed | `vllm-tp4-*.service` 实测 |
| 3 | **head 快速失败 + worker 120s** | head 起容器后限时等待就绪，超时退出交 systemd；worker 容器缺失且集群成形才触发 head 全链重建 | monitor 脚本实测 |
| 4 | **NFS 启动竞态修复** | 03/04 改**本地 serving**（`.local-backup`），消除 NFS 挂载竞态；NFS 集中化恢复中 | fv ⑥ `readlink` 实测 |
| 5 | **互杀守卫** | worker 仅在 **head API 健康且集群已成形（rank 全连 TCPStore）但本 rank 缺失** 时才 `rm rank0` 触发全链重建；**冷启动未成形不动 head** | `monitor_tp4_worker.sh` 注释实测 |
| 6 | **shim v8** | `libncclpin.so` v6→v7→**v8**：NCCL/PT 线程绑 **8-9**（isolcpus=8-9），EngineCore 15-19 | PSR 实测 aff=8-9 / 15-19 |
| 7 | **fix72 capture** | `--max-cudagraph-capture-size 64` + sizes 1..64（原 80/1..80） | head 日志 serve 命令实测 |
| 8 | **seqs=6 / util 0.65** | `--max-num-seqs 12→6`、`--gpu-memory-utilization 0.60→0.65` | head 日志实测 |

### A.2 验证 7 项（final_verify.sh，8/12 23:41）
① isolcpus=8-9 四机 + nproc=18 ✅ ② head/worker 服务 active（互斥各 1）✅ ③ vllm-tp4-rank0~3 全 Up(healthy)、8001=200 ✅ ④ PSR：EngineCore aff=15-19、NCCL Progress/pt_nccl_*/pt_tcpstore aff=8-9 ✅ ⑤ 互杀 HEAD_KILL=0（02/03/04，当前 boot）✅ ⑥ 03/04 本地 serving OK（.local-backup）✅ ⑦ chat 冒烟 `"OK"` ✅

### A.3 重启恢复流程（R11 增量）
- 开机顺序 01→02→03→04；systemd 已 enable，正常重启**勿手动启**（head monitor 链式自愈）。
- 重启后复核：`isolcpus=8-9` / MTU 9000 / GID=3 / `/opt/nccl-ringonly` md5 `b7784b49885659c27765e648884e4edd` / shim v8 md5（01=`ce43c688c5164ac7efd5105c94fdab77`）→ 8001=200 即就绪。
- **GID 预案（2026-08-15 新增）**：当前统一 GID=3（idx3 与 idx2 同指向 IPv4 RoCEv2 GID，已验证有效）。若未来某机 `ibv_modify_qp failed 61` / GID3 空，预案为**整体切回 GID=2 并重启 TP4**（四机 head/worker 脚本 NCCL_IB_GID_INDEX 全部改 2，重建容器做 NCCL 重新握手），勿单机改。
- 故障定位沿用 runbook append §A.7 故障表 + quickref §4。

## §B. 512-131072 测试配置基线（TP4S，Tessa 全矩阵）

### B.1 矩阵规格（45 组合，seqs=6 生产配置）
- ctx ∈ {512, 2048, 8192, 32768, 131072} × task ∈ {coding, json, prose} × conc ∈ {1, 3, 5}
- rounds：小档（512/2048/8192）=3、大档（32768/131072）=2
- 执行器：`_tessa_tp4_bench/tp4s_matrix_run.py`（endpoint 8001/v1，uuid 随机前缀防 prefix 命中，投机指标 /metrics 差分）
- 输出：`_tessa_tp4_bench/TP4S/`

### B.2 基线口径（提交验收时引用）
- 主指标：p50×conc（prefill / decode / TTFT / TPOT / TPS）；峰值取 max；禁用 agg_*。
- 早期抽测（conc=1）：decode 100-122 tok/s（coding/json 档）、TTFT 512=0.3s / 131072=56.9s、spec accept≈0.75-0.86、prefix hit=0（预期）。
- 结果回填由 Tessa 完成（任务 #14/15）；本文仅固化**口径与基线规格**。

## §C. NFS 集中化恢复流程（进行中）

1. **当前态**：01/02 本地模型（NFS 导出源保留）；03/04 走本地 `.local-backup`；`mount | grep nfs` 无业务 NFS 挂载。
2. **恢复目标**：01 主源 export → 03 挂 <NODE_IP>；02 备源 → 04 挂 <NODE_IP>（原双源拓扑，见 fault-tolerance.md §2）。
3. **恢复步骤（窗口执行）**：
   - 01/02 确认 exportfs 在位（`exportfs -v`）→ 03/04 `mount -a` 后 `mount | grep nfs4` 复核。
   - 软链检查：03/04 `<INSTALL_DIR>/models/deepseek-v4-flash-0731` → `<MODELS_DIR>/...`（挂载版），`.local-backup` 保留作兜底。
   - 启动验证走 §A.3；**恢复后须更新 server-maintenance-handbook.md §1/§4**（NFS 双源状态）。
4. **回退**：若恢复后出现启动竞态 → 软链改回 `.local-backup` 重走 §A.3（即 R11 修复态）。

## §D. 遗留跟踪（P2）
双口 v3 补丁（预期 25GB/s，停线窗口）｜异常点复测（16384/coding/c1 等）｜mft/mlxlink FEC 基线｜QoS 持久化｜NFS 集中化恢复落地与文档回填。

---

## §E. TP4 重启姿势与自愈链（2026-08-23 SRE 新增，源自 phase3b/w4a4-ext/bprime 三窗口实证）

### E.1 正确重启姿势（唯一推荐路径）
- **`systemctl stop vllm-tp4-head.service` ≠ 容器停**：服务 ExecStart 是 monitor（`docker wait` 跟随），停服务不停容器；且服务停后 workers 不会自动跟随（phase3b 教训：rank0 独等 rendezvous 12 分钟后需手动拉三 worker）。
- **配置变更后重启（唯一正确姿势）**：四机脚本改好 + checker 过后，**服务保持 active**，`docker rm -f vllm-tp4-rank0` 触发 **head-first 全链重建**（monitor 清 worker 容器 → 各机 systemd 自愈重建，worker 带 head TCPStore 门禁防冷启动互杀，全程零人工干预）。冷启动约 16 分钟。
- **恢复自愈链三件套（缺一不可）**：① `vllm-tp4-head.service` active（01）+ ② `vllm-tp4-worker.service` active（02/03/04）+ ③ `vllm-healthcheck.timer` active（01）+ `/health` 200。恢复后按此清单逐项核验。

### E.2 监控栈自愈（Prometheus 边界 case 修复，2026-08-23）
- 症状：`aicad-prometheus-1` 曾 Exited(137) 2 天不自愈（exit + docker daemon 重启边界下 `unless-stopped` 不回拉）。
- 修复：`docker update --restart=always aicad-prometheus-1`（运行时）+ `/opt/aicad/docker-compose.yml` L125 `restart: always`（compose 源，`.bak-luz031-20260823` 留档）。
- 注意：Prometheus 是节点冻死信号探针（S1 分诊第一项，`<MGMT_OCTET>:8191`），它自身 down 时先怀疑节点级 UMA 耗尽，不要只当容器问题。

## §F. 基准作业纪律（2026-08-23 SRE 新增）

1. **基准前**：`systemctl stop vllm-healthcheck.timer`（防探针误判触发重建打断测量——healthcheck-rebuild 探针超时已放宽 10s→30s，`.bak-luz031-20260823` 留档四机，但 timer 停用才是基准期正解）。
2. **基准后**：`systemctl start vllm-healthcheck.timer` 恢复（勿忘，属于 §E.1 三件套）。
3. **测量口径**：每臂 stall 探针（重启后 3 短 4K，TTFT>6s 疑似重签 ≤2 次）+ 模式探针先行；≥3 轮取中位；DE 用接受率归一（step_eff = tput / tokens_per_step）。
4. **质量门**：一律用 `<INSTALL_DIR>/scripts/quality_gate.py`（稳定 4 prompt 集 fox_repeat/count/code/list + 包络判据 + 参考快照管理；reason/zh 已除名——运行级非确定已证）。历史 /tmp 散落脚本已加弃用指向。

## §G. LuZ0.3.1 生产形态（2026-08-23 采纳，详情见 LuZ0.3.1-release-notes.md）

- 构成：W4A4 full（`VLLM_MOE_W4A4=2`，plugin_a1 池化插件 md5 `e5ed0c85`）+ 池补丁（`VLLM_B12X_SHARED_WRAPPER=1`）+ FI 0.6.16 overlay + threshold 4096 + util 0.82。
- 回滚链：start 脚本/checker `.bak-luz031-20260823`（四机）+ 插件 `.bak-wsdedupl3-20260823`（原版 c2d1de3d）+ `restore_luz031.sh`（<INSTALL_DIR>/backup/luz031-checkpoint-20260823/）。

## §H. needle 抽验口径统一建议（2026-08-24 SRE 新增，源自 G3 归档）

1. **现状**：needle 测试脚本散落在各窗口临时目录（/tmp/_fi016/verify、/tmp/_wsdedup_l3/logs 等），口径不统一、脚本未固化。已有实测记录（fi016 窗口）：64K mid/late/late **3/3 PASS**；128K 加测 1/2（late 位已知抖动，已记入 LuZ0.3.1-release-notes.md）。
2. **统一口径建议**（下次 needle 窗口执行时采纳）：
   - 脚本集中：固化为 `<INSTALL_DIR>/scripts/needle_check.py`（当前无统一脚本；迁入时按脚本变更纪律 .bak + checker/文档回填）。
   - 标准位置：64K 必测 **mid / late** 两档；128K 可选加测（late 位已知抖动，结果须标注位置）。
   - 判定标准：响应中出现目标 access-code 字符串即 PASS（示例：X7Q-95-Bravo）；位置命中按目标 token 区间 ±容忍判定。
   - 记录：结果落 `<INSTALL_DIR>/verification-logs/needle_<UTC>.json`，报告引用须注明 ctx/pos/轮次。
3. **已知事项**：128K late 位抖动（1/2）属运行级已知，不作为阻断；128K 加测结果仅作参考。
