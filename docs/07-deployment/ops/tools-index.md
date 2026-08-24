# 维护工具清单（Tools Index）v1.5-R11

**日期**：2026-08-12（R11 修订）｜**维护**：Docu｜**依据**：生产脚本实测（四机 `ls <INSTALL_DIR>/scripts` + `/usr/local/sbin`）
**R11 变更**：脚本↔文档引用统一走 `scripts/REFERENCE.md`（权威索引，位于 `<INSTALL_DIR>/docs/`）；各脚本头部 `# DOCS:` 已约定引用。

---

## A. 生产核心（<INSTALL_DIR>/scripts/）

### A.1 start_tp4_cluster.sh（仅 01）
- **用途**：TP4 四机集群编排（head-first 幂等）。唯一权威启动入口。
- **调用**：`cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh`
- **内部**：GPU-gate≤180s → 依序 rank1(02)/rank2(04)/rank3(03) → 轮询 8001 就绪（MAX_WAIT=600s）→ ERR_PATTERNS 诊断。
- **输出**：`$HOME/start_tp4_cluster.log`。

### A.2 start_tp4_head.sh（仅 01）/ start_tp4_worker.sh（02/03/04 通用）
- **用途**：单 rank 容器启停（容器 `--restart no`，由 systemd 管生命周期）。
- **调用**：head `bash start_tp4_head.sh`；worker `NODE_RANK=1 VLLM_HOST_IP=<NODE_IP> bash start_tp4_worker.sh`（02/03/04 的 NODE_RANK=1/3/2）。
- **ENV**：NO_WAIT=1（systemd monitor 内部调用，跳过就绪轮询）。
- **副作用**：每次执行 `docker inspect` 覆写 `backup/rollback_tp4-rank{N}.json` 回滚锚点。

### A.3 monitor_tp4_head.sh（01）/ monitor_tp4_worker.sh（02/03/04）
- **用途**：systemd `vllm-tp4-head.service`/`vllm-tp4-worker.service` 的 ExecStart 主体，容器生命周期监护+自愈触发。
- **调用**：由 systemd 调用；**禁止手动直跑**。
- **行为**：容器在→`docker wait`；head 缺失→先清 worker→重建 head；worker 缺失→等 head TCPStore:25999→重建。
- **输出**：`journalctl -u vllm-tp4-head/-worker`。

### A.4 check_vllm_script.sh（四机）
- **用途**：vLLM 启动脚本自检（防三类事故：注释吞续行/关键参数缺失/sudo $HOME）。
- **调用**：`bash check_vllm_script.sh <script_path>`。
- **返回**：0=通过 1=失败 2=用法错误（编排前必跑）。
- **注意**：A 组关键参数已同步生产（`max-model-len 400000` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，8/12 14:48 更新），与 start_tp4_* 一致。

### A.5 start_embed_8022.sh（03/04 生产，01/02 备用）
- **用途**：embed 服务（anemll-embed-8022，KV 4GB，max-num-seqs 32 / len 8192）。
- **调用**：`bash start_embed_8022.sh`。**铁律**：必须 `--kv-cache-memory`（VLLM_GPU_MEMORY_UTILIZATION 已失效，不用会 OOM）；镜像 ENTRYPOINT 已含 serve。

### A.6 锚点/退役脚本（勿动）
- `start_v026r_cluster.sh`（01）：**TP2 降级唯一入口**，全程未动；`start_head_v026r.sh`/`start_worker_v026r.sh` 配套。
- `start_groupB_*.sh`（03/04）、`bind_irq_forward.sh`（02，IRQ 绑定，裸脚本需补帮助头）。

### A.7 restore_grafana_dashboards.sh（02）
- **用途**：Grafana dashboard 配置恢复（volume 误删/容器迁移后），从宿主机备份回填容器 + provisioning。
- **调用**：`bash restore_grafana_dashboards.sh [备份目录]`（默认 <INSTALL_DIR>/backups/grafana-20260813/）。
- **备份**：<INSTALL_DIR>/backups/grafana-20260813/（vllm-realtime.json 等 4 文件 + README）。
- **持久化要点**：Grafana 数据目录 = named volume `aicad_grafanadata`（容器重建不丢，仅 volume rm 会丢）；修改面板 = docker cp 出→改→回（provisioning 10s 加载，API 修改会被回滚）。

## B. 系统工具（/usr/local/sbin/，四机）

| 工具 | 用途 | 调用 | 输出 |
|---|---|---|---|
| `mlnx-qos-setup.sh` | RoCE QoS（trust=dscp + PFC P3/P5） | `sudo bash mlnx-qos-setup.sh` | 无回显；exit 0 |
| `iptables-save-custom.sh` | 白名单规则落盘 rules.v4（剔除 docker 动态链） | `sudo bash iptables-save-custom.sh` | `OK: custom rules saved...(N rules)` |

> ⚠️ 原任务清单中的 `gpuhealth.sh`/`fastfail.sh` **无独立脚本**：GPU 门禁内嵌于 start_tp4_cluster.sh（GPU_OK 循环）；fastfail 仅存于 TP2 时代 `.bak-20260811-fastfail` 备份。QoS/iptables 工具实际在 `/usr/local/sbin/`，**不在** `<INSTALL_DIR>/scripts/`。

## C. 远程运维辅助

- **模型数据（R11 本地 serving）**：01/02 本地 `/home/<USER>/models/...`；03/04 `<MODELS_DIR>/...local-backup`；NFS 双源集中化恢复中（恢复流程见 runbook v1.5 §C）。
- litellm 网关：`<NODE_IP>:4000`（并发 12，`/health` 401=有鉴权在线）；每日备份 cron（01/02）。
- 日志位置：`/var/log/vllm/nccl-*.log`（容器内）、`$HOME/vllm-logs/`、`journalctl -u vllm-tp4-*`。

## D. 文档引用（R11 新增）

- **权威文档根**：`<INSTALL_DIR>/docs/`（01/02 镜像，见 `docs/README.md`）。
- **脚本↔文档索引**：`docs/scripts/REFERENCE.md`——每个脚本的 DOCS 引用、帮助头标准、同步命令。查脚本先看它。
