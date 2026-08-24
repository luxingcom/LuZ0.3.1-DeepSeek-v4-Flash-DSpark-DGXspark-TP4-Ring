# 服务器维护手册（Server Maintenance Handbook）v1.5-R11

**日期**：2026-08-12｜**维护**：Docu｜**权威文档**：`README.md`、`rollback-anchors-2026-08-12.md`、`ops/ops-discipline-quickref.md`
**适用**：DGX Spark 4 机 TP4 生产（01=186/<MGMT_OCTET> 02=187/<MGMT_OCTET> 03=188/<MGMT_OCTET> 04=189/<MGMT_OCTET>）
**本版变更**：R11 修复后配置基线（seqs=6、isolcpus=8-9、EngineCore 15-19、capture 64、StartLimit 1800/20、互杀守卫、本地 serving）

---

## 1. 拓扑与角色基线

| rank | 主机 | 管理网 | RoCE 环邻 | 系统服务 |
|---|---|---|---|---|
| 0 | node01 | <NODE_IP> | 02(f1)/03(f0) | `vllm-tp4-head.service` |
| 1 | node01 | <NODE_IP> | 01(f1)/04(f0) | `vllm-tp4-worker.service`(rank1)、litellm:4000 |
| 2 | node01 | <NODE_IP> | 02(f0)/03(f1) | `vllm-tp4-worker.service`(rank2) |
| 3 | node01 | <NODE_IP> | 04(f1)/01(f0) | `vllm-tp4-worker.service`(rank3) |

- 控制面：MASTER_ADDR=<NODE_IP>、MASTER_PORT=25999（TP4 专用）；RoCE MTU 全 9000。
- 镜像：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（四机一致）。
- **模型数据（本地 serving，R11）**：软链 `<INSTALL_DIR>/models/deepseek-v4-flash-0731` → 01/02 本地 `/home/<USER>/models/...`；03/04 → `<MODELS_DIR>/deepseek-v4-flash-0731.local-backup`（本地兜底副本）。**NFS 集中化恢复中：当前无 NFS 挂载（`mount | grep nfs` 仅 nfsd 内核）**，恢复落地后本表更新为双源。

### 1.1 服务清单（常驻服务全览，2026-08-13 复核）

| 服务 | 主机 | 端口 | 类型 | 健康检查 |
|---|---|---|---|---|
| vllm-tp4-rank0 | 01 | 8001(host 网络) | 容器 | `curl 127.0.0.1:8001/health` |
| vllm-tp4-rank1 | 02 | 8002(host 网络) | 容器 | 同上 8002 |
| vllm-tp4-rank2 | 04 | 8003(host 网络) | 容器 | 同上 8003 |
| vllm-tp4-rank3 | 03 | 8004(host 网络) | 容器 | 同上 8004 |
| vllm-tp4-head.service | 01 | - | systemd(system) | `systemctl status vllm-tp4-head` |
| vllm-tp4-worker.service | 02/03/04 | - | systemd(system) | `systemctl status vllm-tp4-worker` |
| responses-gateway.service | 02 | 8003 | systemd(user) | `curl 127.0.0.1:8003/health` |
| litellm-proxy | 02 | 4000(host 网络) | 容器 | `curl 127.0.0.1:4000/health`(401=鉴权在线) |
| embed(anemll-embed-8022) | 03/04 | 8022 | 容器 | `curl 127.0.0.1:8022/v1/models` |
| registry | 02 | 5000 | 容器 | `curl 127.0.0.1:5000/v2/` |
| aicad-grafana-1 | 01/02 | 3000 | 容器 | `curl 127.0.0.1:3000/api/health` |
| aicad-prometheus-1 | 02 | 8191(9090) | 容器 | `curl 127.0.0.1:8191/api/v1/query?query=up` |
| aicad-alertmanager-1 | 02 | 9093 | 容器 | `curl 127.0.0.1:9093/-/healthy` |
| aicad-postgres-1 | 01/02 | 8082->5432 | 容器 | `docker exec aicad-postgres-1 pg_isready` |
| litellm-pg | 02 | - | 容器 | `docker exec litellm-pg pg_isready` |
| aicad-redis-1 | 01/02 | 6379 | 容器 | `docker exec aicad-redis-1 redis-cli ping` |
| aicad-neo4j | 01/02 | 7474/7687 | 容器 | `curl 127.0.0.1:7474` |
| aicad-minio-1 | 01 | 19000->9000 | 容器 | `curl 127.0.0.1:19000/minio/health/live` |
| aicad-fw-25000 | 01 | 25000 | 容器 | iptables 白名单入口 |
| node-exporter | 四机 | 9100 | 容器 | `curl 127.0.0.1:9100/metrics` |
| dcgm-exporter | 四机 | 9400 | 容器 | `curl 127.0.0.1:9400/metrics` |

> 端口以 2026-08-13 `docker ps` 实测为准；vLLM/litellm 为 host 网络（无端口映射）。网关 8003 上游：chat→<MGMT_OCTET>:8001，embed→<MGMT_OCTET>:8022。

## 2. 配置基线（R11 参数全集，生产脚本实测 8/12 23:33）

| 组 | 参数 | 值 | 说明 |
|---|---|---|---|
| 模型 | `--max-model-len` | 400000 | 400k |
| 服务 | `--max-num-seqs` | **6** | seqs=6（R11 由 12 下调，规避 OOM/长稳） |
| KV | `--kv-cache-dtype` | nvfp4_ds_mla | MLA 量化 |
| 显存 | `--gpu-memory-utilization` | **0.65** | R11 已落地 |
| 前缀 | `--enable-prefix-caching` | on | Prefix KV |
| CUDA Graph | `--max-cudagraph-capture-size` | **64** | **fix72**：`--cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64`（R11，由 80/1..80 下调） |
| 投机 | `--speculative-config` | dspark,5 tokens | 不变 |
| 分布式 | `--tensor-parallel-size 4 --nnodes 4` | mp 后端 | 不变 |
| 补丁 | `LD_PRELOAD` | `/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2` | **shim v8** + ring-only |
| CPU | `--cpuset-cpus` | 1-19 | **NCCL/PT 线程→8-9（isolcpus=8-9）；EngineCore→15-19**（R11） |
| NCCL | ALGO=RING MIN_NCHANNELS=2 IB_HCA=4口 TOS=46 GID_INDEX=3 | env 全集见 runbook §A.3 | 不变 |
| systemd | StartLimit | **IntervalSec=1800 / Burst=20** | R11（由 600/8 放大） |
| systemd | Restart=always RestartSec=15 TimeoutStartSec=1500 User=<USER> | | 不变 |
| 自愈 | 互杀守卫 | **集群未成形时不动 head** | R11 新增（防冷启动互杀） |
| 容器 | `--restart no` | 由 systemd 管 | 不变 |

## 3. 脚本索引（详见 tools-index.md + scripts/REFERENCE.md）

- 编排：`start_tp4_cluster.sh`(01)｜启停：`start_tp4_head.sh`/`start_tp4_worker.sh`
- 自愈：`monitor_tp4_head.sh`(01)/`monitor_tp4_worker.sh`(02/03/04) + systemd（含互杀守卫）
- 自检：`check_vllm_script.sh`(四机)｜embed：`start_embed_8022.sh`(03/04 生产)
- 系统工具：`/usr/local/sbin/mlnx-qos-setup.sh`、`iptables-save-custom.sh`
- 各脚本头部 `# DOCS:` 指向 `<INSTALL_DIR>/docs/` 对应文档（见 scripts/REFERENCE.md §1）

## 4. 日常巡检清单（R11 增量项已标 ◆）

| 项 | 命令 | 期望 |
|---|---|---|
| 服务 | `systemctl is-active vllm-tp4-head vllm-tp4-worker` | active |
| 端口 | `curl -o /dev/null -w '%{http_code}' http://<NODE_IP>:8001/health` | 200 |
| 网关 | `curl -o /dev/null -w '%{http_code}' http://<NODE_IP>:4000/health` | 401（有鉴权=在线） |
| GPU | `nvidia-smi`（01/02 内存余量） | 01/02 ≥10G 余量 |
| ◆ 隔离核 | `cat /proc/cmdline \| tr ' ' '\n' \| grep isolcpus` | `isolcpus=8-9`；nproc=18 |
| ◆ PSR 分布 | head 机 `/proc/<EngineCore>/status` Cpus_allowed_list | EngineCore=**15-19**；NCCL Progress/pt_nccl_*/pt_tcpstore=**8-9** |
| ◆ 互杀守卫 | `journalctl -b -u vllm-tp4-worker -n 50` | 无 `HEAD_KILL` 频发（期望 0-1） |
| 网络 | 四机 `ethtool -S` 16 口 | PHY/IP err 全 0；FEC corr 无增长 |
| ◆ 模型源 | `readlink <INSTALL_DIR>/models/deepseek-v4-flash-0731` | 01/02→本地；03/04→.local-backup；NFS 恢复后查 `mount \| grep nfs4` |
| systemd | `journalctl -u vllm-tp4-head -n 50` | 无 NCCL/Err 关键错误 |
| ◆ 长档并发红线 | 128K 长档（≥32768） | **并发 ≤c3**（c5 反噬，见部署指南 §B/D4） |
| 日志 | `/var/log/vllm/nccl-*.log`、`$HOME/vllm-logs/` | 无 110/错误堆积 |
| 补丁 | `docker logs vllm-tp4-rank0 \| grep "NCCL version"` | `2.30.7+cuda13.0` |

## 5. 故障排查流程（TL;DR）

1. **先判级**：8001 全挂=P0→立即重建；单点=P1→定位。
2. **读日志**：`journalctl -u vllm-tp4-*` → `docker logs vllm-tp4-rank{N}` → `/var/log/vllm/nccl-*.log`。
3. **对照速查**：错误码/现象 → `ops/ops-discipline-quickref.md` §4。
4. **回滚锚点**：任何补丁/网络/脚本异常 → `rollback-anchors-2026-08-12.md` 对应 §2 条目。
5. **重建**：正常情况 systemd 自愈；人工介入走启停纪律（maintenance-plans.md §1）。
6. **复盘**：事故处理完 24h 内补 incident-* 文档。

## 6. 已知遗留（P2 观察项）

- QoS（mlnx-qos-setup.sh）**无 cron/timer 持久化**——重启后需手动执行。
- **NFS 集中化恢复进行中**：当前 03/04 走本地 `.local-backup`（无 NFS 挂载）；恢复后需更新 §1/§4 并复验容错（fault-tolerance.md §2）。
- 01 上 `vllm-cluster.service`（TP2 legacy）处于 failed，建议 disable。
- 01/02 FEC 共享计数为设备级累计，需 mft/mlxlink 精确基线。
