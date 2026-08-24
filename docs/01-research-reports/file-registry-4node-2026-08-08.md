# DGX 四机生产文件调用索引（File Registry v1.0）

**日期**：2026-08-08
**维护**：工程保障团队 | 存放：<INSTALL_DIR>/docs/file-registry.md（四机同步）
**原则**：重要文件不存放于公共/临时目录（/tmp）；容器挂载源一律持久化；索引为恢复与故障排查的唯一依据

---

## 🏷️ 统一命名规范（2026-08-08，Archi 规范）

| DGXspark编号 | hostname | 管理IP | ssh别名(主) | 旧hostname | 旧ssh别名 | 角色 |
|------|----------|--------|-------------|------------|-----------|------|
| 01 | node01 | <NODE_IP> | node01 | spark-05cd | aicad-server60 | head（TP2 rank0） |
| 02 | node01 | <NODE_IP> | node01 | edgexpert-0c69 | aicad-server | worker/监控 |
| 03 | node01 | <NODE_IP> | node01 | gx10-3f4d | gx10-55 | embed/组B head |
| 04 | node01 | <NODE_IP> | node01 | gx10-31c4 | gx10-59 | embed/组B worker |

- Prometheus 标签：machine=node=dgxspark0N（小写统一）
- 旧别名保留兼容（deprecated，3 个月后移除）；NVIDIA sync 编号映射不变
- RoCE 内网：136/137（01↔02）、138/139（03↔04）

## 0. 目录规范（Archi 2026-08-08）

```
<INSTALL_DIR>/
├── models/    # 模型权重统一入口（软链，物理位置见索引）
├── envs/      # 构建环境（nvcc-wrapper、vllm-envc-cache 等，已迁出 /tmp）
├── scripts/   # 编排脚本（start_v026r_cluster.sh 等）
├── configs/   # 受管配置副本（netplan/litellm/prometheus/nvidia-sync）
├── cache/     # vllm-cache / tilelang-cache（预留迁移）
├── logs/      # vllm-logs / startup / sync
├── backup/    # 回滚锚点 + 索引快照
└── docs/      # 本索引 + Runbook
```

**重启存活标记**：✅=持久（重启保留）｜⚠️=/tmp（重启丢失）｜🔄=需重建/恢复

---

## 1. 核心文件索引（四机）

### 1.1 模型权重（生产只读挂载）

| # | 机 | 当前路径 | 统一入口(软链) | 用途 | 调用方 | 重启 | 大小 |
|---|----|---------|---------------|------|--------|------|------|
| 1 | .58 | /home/<USER>/models/deepseek-v4-flash-0731 | <INSTALL_DIR>/models/deepseek-v4-flash-0731 | LLM 权重(48分片) | vllm-envE-worker 挂载 /models:ro | ✅ | 156G |
| 2 | .60 | /home/<USER>/models/deepseek-v4-flash-0731 | 同上 | LLM 权重 | vllm-envE-node 挂载 /models:ro | ✅ | 156G |
| 3 | .55 | <MODELS_DIR>/deepseek-v4-flash-0731 | 同上 | LLM 权重 | (55+59 TP2 用) | ✅ | 156G |
| 4 | .59 | <MODELS_DIR>/deepseek-v4-flash-0731 | 同上 | LLM 权重 | (55+59 TP2 用) | ✅ | 156G |
| 5 | .55 | <MODELS_DIR>/Qwen3-Embedding-0.6B | <INSTALL_DIR>/models/Qwen3-Embedding-0.6B | embed 权重 | embed-qwen3-vllm 挂载 | ✅ | 1.2G |
| 6 | .59 | <MODELS_DIR>/Qwen3-Embedding-0.6B | 同上 | embed 权重 | embed-qwen3-vllm 挂载 | ✅ | 1.2G |
| 7 | .58 | <MODELS_DIR>/Qwen3-Embedding-0.6B | 同上 | embed 权重 | anemll-embed-8022 挂载 | ✅ | 1.2G |

### 1.2 构建环境（2026-08-08 已迁出 /tmp——本轮加固核心）

| # | 机 | 当前路径 | 容器内挂载点 | 用途 | 调用方 | 重启 |
|---|----|---------|-------------|------|--------|------|
| 8 | .60 | <INSTALL_DIR>/envs/nvcc_wrapper.py | /tmp/env-e-build/nvcc_wrapper.py:ro | deepgemm JIT nvcc wrapper(sm_121a) | vllm-envE-node (DG_JIT_NVCC_COMPILER) | ✅(原⚠️) |
| 9 | .58 | <INSTALL_DIR>/envs/nvcc_wrapper.py | 同上 | 同上 | vllm-envE-worker | ✅(原⚠️) |
| 10 | .60 | <INSTALL_DIR>/envs/vllm-envc-cache | /cache/huggingface | HF 缓存 | vllm-envE-node | ✅(原⚠️) |
| 11 | .58 | <INSTALL_DIR>/envs/vllm-envc-cache | /cache/huggingface | HF 缓存 | vllm-envE-worker | ✅(原⚠️) |
| 12 | .55 | /tmp/env-e-build（NVIDIA Sync 残留） | — | sync 构建产物（非生产） | nvidia-sync | ⚠️（忽略） |
| 13 | .59 | /tmp/env-e-build（NVIDIA Sync 残留） | — | 同上 | nvidia-sync | ⚠️（忽略） |

> ⚠️ 坑位记录：.55/.59 曾发现 /tmp/env-e-build/nvcc_wrapper.py 被误建为**目录**（.58 重启后同样发生），导致容器挂载失败——已迁出 /tmp 根治；若未来仍有进程重建 /tmp 下文件，不影响生产（生产挂载源已指向 <INSTALL_DIR>/envs/）。

### 1.3 JIT 编译缓存（/home 持久，重启保留）

| # | 机 | 路径 | 容器内 | 用途 | 重启 |
|---|----|------|--------|------|------|
| 14 | .58/.60 | ~/vllm-cache | /root/.cache/vllm:rw | deepgemm/flashinfer JIT 缓存（重启免重编） | ✅ |
| 15 | .58/.60 | ~/tilelang-cache | /root/.tilelang/cache:rw | TileLang JIT 缓存 | ✅ |
| 16 | .58/.60 | ~/patch-v026/ | /usr/local/lib/python3.12/dist-packages/vllm/...:ro | vLLM patch（tilelang.py 等） | ✅ |

### 1.4 编排脚本与日志（统一入口）

| # | 机 | 规范路径 | ~/ 兼容软链 | 用途 | 重启 |
|---|----|---------|------------|------|------|
| 17 | .60 | <INSTALL_DIR>/scripts/start_v026r_cluster.sh | ~/start_v026r_cluster.sh | 集群编排 v2.0（幂等+诊断） | ✅ |
| 18 | .60 | <INSTALL_DIR>/scripts/start_head_v026r.sh | ~/start_head_v026r.sh | head 启动 v3.3 加固（GID_INDEX=2） | ✅ |
| 19 | .58 | <INSTALL_DIR>/scripts/start_worker_v026r.sh | ~/start_worker_v026r.sh | worker 启动 v1.0（2026-08-08 新建） | ✅ |
| 20 | .60 | <INSTALL_DIR>/backup/rollback_*.json | — | 容器回滚锚点（原 /tmp 已迁出） | ✅ |
| 21 | .58/.60 | ~/vllm-logs → <INSTALL_DIR>/logs/vllm（规划） | — | vLLM/NCCL 日志（NCCL_DEBUG_FILE 落此） | ✅ |

### 1.5 配置文件

| # | 机 | 路径 | 用途 | 备注 |
|---|----|------|------|------|
| 22 | 四机 | /etc/netplan/99-nvidia-sync-cluster.yaml | RoCE overlay IP（55↔59: 138/139；58↔60: 136/137） | NVIDIA Sync 生成 |
| 23 | .58 | /opt/aicad/monitoring/prometheus.yml | Prometheus 采集配置（4 台管理网直采） | 备份 .bak-* |
| 24 | .58 | /home/<USER>/litellm/config.yaml | litellm 3 端点 embed 池 + LLM 路由 | 备份 .bak-* |
| 25 | .58 | /opt/aicad/monitoring/grafana/provisioning/dashboards/aicad.yml | Grafana provisioning | 卷 aicad_grafanadata |
| 26 | .58 | /opt/aicad/monitoring/grafana/vllm-realtime.json | vLLM 面板（provisioning 版，2026-08-08 修订） | 卷 dashboards 目录 |
| 27 | .60 | /etc/systemd/system/vllm-cluster.service | TP2 集群自启动（oneshot, 2026-08-08 新建） | systemctl enable 已设 |

### 1.6 密钥与凭据（敏感）

| # | 机 | 路径 | 用途 | 备注 |
|---|----|------|------|------|
| 28 | .60(node01) | ~/.ssh/config（node01 别名，原 DGXspark02 兼容保留）+ id_ed25519_nvsync_cluster_assistant | head→worker 编排 ssh | 勿落 /tmp |
| 29 | .55(node01) | ~/.ssh/config（node01 别名，原 gx10-59 兼容保留） | 55→59 免密中转（经 RoCE） | 2026-08-08 建立 |
| 30 | 四机 | 统一 sudo 密码 <PASSWORD>（⚠️ P0 待轮换） | 运维 | 用户知悉风险 |

---

## 2. 重启恢复流程（故障排查 SOP）

```
系统重启后:
1. 检查容器: docker ps (registry/prometheus/grafana/litellm 等 restart policy 自动恢复)
2. TP2 容器: vllm-envE-node/worker 若 Exited → 不要 docker start!
   (TP2 必须 head-first 重建: worker 单独 start 无法重新 join 已断开的 TP 对)
3. 正确恢复: 在 .60 执行 → bash <INSTALL_DIR>/scripts/start_v026r_cluster.sh
   (幂等: 自动清理残留 → head 先起 → TCPStore :25000 就绪 → worker 起 → API 双阶段健康)
4. 或系统重启后自动: systemctl status vllm-cluster.service (已 enable, 开机自动编排)
5. embed 检查: anemll-embed-8022 (.58) / embed-qwen3-vllm (.55/.59) 自动恢复后 curl /health
6. 关键挂载预检: ls <INSTALL_DIR>/envs/nvcc_wrapper.py (若缺失: 从对端拷贝 1710B 文件)
```

**常见故障速查**（错误模式 → 根因 → 修复）：
| 错误 | 根因 | 修复 |
|------|------|------|
| HFValidationError '/models' | head 容器缺模型挂载 | start_head_v026r.sh 已含挂载，重跑编排 |
| `std::filesystem::exists(nvcc_path)` | 缺 nvcc_wrapper 挂载 | 检查 <INSTALL_DIR>/envs/nvcc_wrapper.py |
| `ibv_modify_qp failed 61` | .60 重启后 GID3 空 | NCCL_IB_GID_INDEX=2（脚本已固定） |
| `Connection closed by peer [<NODE_IP>]` | worker 残留连旧 master | 编排脚本幂等清理后 head-first 重建 |
| `not a directory` mount error | /tmp/env-e-build 被误建目录 | 生产挂载已迁 <INSTALL_DIR>/envs/（不受影响） |

---

## 3. 待办（下一窗口）

- [ ] ~/vllm-cache、~/tilelang-cache、~/vllm-logs 迁移 <INSTALL_DIR>/{cache,logs}（低风险，避免 JIT 重编）
- [ ] 55+59 LLM TP2 部署脚本（start_head_5599.sh / start_worker_5599.sh）——benchmark 前
- [ ] internal API key 明文 → env 文件 chmod 600 + --env-file（Cody Medium）
- [ ] nvidia-sync 产物重定向 <INSTALL_DIR>/envs/env-e-build（或 systemd mount 单元）
- [ ] 口令轮换（P0）
