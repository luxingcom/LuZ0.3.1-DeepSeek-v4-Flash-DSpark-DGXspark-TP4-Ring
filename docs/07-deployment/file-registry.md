# DGX 四机生产文件调用索引（File Registry v1.0）

# File Registry — 已过时标注

> ⚠️ **【2026-08-26 标注】本文档描述 2026-08-08 的 TP2/embed/组B 旧架构，当前生产已迁移至 TP4 环网（LuZ0.3.1-R12）。
> 本文档保留仅供历史追溯，不作为现行部署依据。TP4 版索引待重写。


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
- RoCE 内网：<RING_SUBNET>（01↔02）、<RING_SUBNET>（03↔04）

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
| 1 | <MGMT_OCTET> | /home/<USER>/models/deepseek-v4-flash-0731 | <INSTALL_DIR>/models/deepseek-v4-flash-0731 | LLM 权重(48分片) | vllm-envE-worker 挂载 /models:ro | ✅ | 156G |
| 2 | <MGMT_OCTET> | /home/<USER>/models/deepseek-v4-flash-0731 | 同上 | LLM 权重 | vllm-envE-node 挂载 /models:ro | ✅ | 156G |
| 3 | <MGMT_OCTET> | <MODELS_DIR>/deepseek-v4-flash-0731 | 同上 | LLM 权重 | (55+59 TP2 用) | ✅ | 156G |
| 4 | <MGMT_OCTET> | <MODELS_DIR>/deepseek-v4-flash-0731 | 同上 | LLM 权重 | (55+59 TP2 用) | ✅ | 156G |
| 5 | <MGMT_OCTET> | <MODELS_DIR>/Qwen3-Embedding-0.6B | <INSTALL_DIR>/models/Qwen3-Embedding-0.6B | embed 权重 | embed-qwen3-vllm 挂载 | ✅ | 1.2G |
| 6 | <MGMT_OCTET> | <MODELS_DIR>/Qwen3-Embedding-0.6B | 同上 | embed 权重 | embed-qwen3-vllm 挂载 | ✅ | 1.2G |
| 7 | <MGMT_OCTET> | <MODELS_DIR>/Qwen3-Embedding-0.6B | 同上 | embed 权重 | anemll-embed-8022 挂载 | ✅ | 1.2G |

### 1.2 构建环境（2026-08-08 已迁出 /tmp——本轮加固核心）

| # | 机 | 当前路径 | 容器内挂载点 | 用途 | 调用方 | 重启 |
|---|----|---------|-------------|------|--------|------|
| 8 | <MGMT_OCTET> | <INSTALL_DIR>/envs/nvcc_wrapper.py | /tmp/env-e-build/nvcc_wrapper.py:ro | deepgemm JIT nvcc wrapper(sm_121a) | vllm-envE-node (DG_JIT_NVCC_COMPILER) | ✅(原⚠️) |
| 9 | <MGMT_OCTET> | <INSTALL_DIR>/envs/nvcc_wrapper.py | 同上 | 同上 | vllm-envE-worker | ✅(原⚠️) |
| 10 | <MGMT_OCTET> | <INSTALL_DIR>/envs/vllm-envc-cache | /cache/huggingface | HF 缓存 | vllm-envE-node | ✅(原⚠️) |
| 11 | <MGMT_OCTET> | <INSTALL_DIR>/envs/vllm-envc-cache | /cache/huggingface | HF 缓存 | vllm-envE-worker | ✅(原⚠️) |
| 12 | <MGMT_OCTET> | /tmp/env-e-build（NVIDIA Sync 残留） | — | sync 构建产物（非生产） | nvidia-sync | ⚠️（忽略） |
| 13 | <MGMT_OCTET> | /tmp/env-e-build（NVIDIA Sync 残留） | — | 同上 | nvidia-sync | ⚠️（忽略） |

> ⚠️ 坑位记录：<MGMT_OCTET>/<MGMT_OCTET> 曾发现 /tmp/env-e-build/nvcc_wrapper.py 被误建为**目录**（<MGMT_OCTET> 重启后同样发生），导致容器挂载失败——已迁出 /tmp 根治；若未来仍有进程重建 /tmp 下文件，不影响生产（生产挂载源已指向 <INSTALL_DIR>/envs/）。

### 1.3 JIT 编译缓存（/home 持久，重启保留）

| # | 机 | 路径 | 容器内 | 用途 | 重启 |
|---|----|------|--------|------|------|
| 14 | <MGMT_OCTET>/<MGMT_OCTET> | ~/vllm-cache | /root/.cache/vllm:rw | deepgemm/flashinfer JIT 缓存（重启免重编） | ✅ |
| 15 | <MGMT_OCTET>/<MGMT_OCTET> | ~/tilelang-cache | /root/.tilelang/cache:rw | TileLang JIT 缓存 | ✅ |
| 16 | <MGMT_OCTET>/<MGMT_OCTET> | ~/patch-v026/ | /usr/local/lib/python3.12/dist-packages/vllm/...:ro | vLLM patch（tilelang.py 等） | ✅ |

### 1.4 编排脚本与日志（统一入口）

| # | 机 | 规范路径 | ~/ 兼容软链 | 用途 | 重启 |
|---|----|---------|------------|------|------|
| 17 | <MGMT_OCTET> | <INSTALL_DIR>/scripts/start_v026r_cluster.sh | ~/start_v026r_cluster.sh | 集群编排 v2.0（幂等+诊断） | ✅ |
| 18 | <MGMT_OCTET> | <INSTALL_DIR>/scripts/start_head_v026r.sh | ~/start_head_v026r.sh | head 启动 v3.3 加固（GID_INDEX=2，历史 2026-08-15 前；现=3，见 deployment-guide:167） | ✅ |
| 19 | <MGMT_OCTET> | <INSTALL_DIR>/scripts/start_worker_v026r.sh | ~/start_worker_v026r.sh | worker 启动 v1.0（2026-08-08 新建） | ✅ |
| 20 | <MGMT_OCTET> | <INSTALL_DIR>/backup/rollback_*.json | — | 容器回滚锚点（原 /tmp 已迁出） | ✅ |
| 21 | <MGMT_OCTET>/<MGMT_OCTET> | ~/vllm-logs → <INSTALL_DIR>/logs/vllm（规划） | — | vLLM/NCCL 日志（NCCL_DEBUG_FILE 落此） | ✅ |

### 1.5 配置文件

| # | 机 | 路径 | 用途 | 备注 |
|---|----|------|------|------|
| 22 | 四机 | /etc/netplan/99-nvidia-sync-cluster.yaml | RoCE overlay IP（55↔59: <RING_SUBNET>；58↔60: <RING_SUBNET>） | NVIDIA Sync 生成 |
| 23 | <MGMT_OCTET> | /opt/aicad/monitoring/prometheus.yml | Prometheus 采集配置（4 台管理网直采） | 备份 .bak-* |
| 24 | <MGMT_OCTET> | /home/<USER>/litellm/config.yaml | litellm 3 端点 embed 池 + LLM 路由 | 备份 .bak-* |
| 25 | <MGMT_OCTET> | /opt/aicad/monitoring/grafana/provisioning/dashboards/aicad.yml | Grafana provisioning | 卷 aicad_grafanadata |
| 26 | <MGMT_OCTET> | /opt/aicad/monitoring/grafana/vllm-realtime.json | vLLM 面板（provisioning 版，2026-08-08 修订） | 卷 dashboards 目录 |
| 27 | <MGMT_OCTET> | /etc/systemd/system/vllm-cluster.service | TP2 集群自启动（oneshot, 2026-08-08 新建） | systemctl enable 已设 |

### 1.6 密钥与凭据（敏感）

| # | 机 | 路径 | 用途 | 备注 |
|---|----|------|------|------|
| 28 | <MGMT_OCTET>(node01) | ~/.ssh/config（node01 别名，原 node0X 兼容保留）+ id_ed25519_nvsync_cluster_assistant | head→worker 编排 ssh | 勿落 /tmp |
| 29 | <MGMT_OCTET>(node01) | ~/.ssh/config（node01 别名，原 gx10-59 兼容保留） | 55→59 免密中转（经 RoCE） | 2026-08-08 建立 |
| 30 | 四机 | 统一 sudo 密码 <见 secrets 管理指引>（⚠️ P0 待轮换） | 运维 | 用户知悉风险 |

---

## 2. 重启恢复流程（故障排查 SOP）

```
系统重启后:
1. 检查容器: docker ps (registry/prometheus/grafana/litellm 等 restart policy 自动恢复)
2. TP2 容器: vllm-envE-node/worker 若 Exited → 不要 docker start!
   (TP2 必须 head-first 重建: worker 单独 start 无法重新 join 已断开的 TP 对)
3. 正确恢复: 在 <MGMT_OCTET> 执行 → bash <INSTALL_DIR>/scripts/start_v026r_cluster.sh
   (幂等: 自动清理残留 → head 先起 → TCPStore :25000 就绪 → worker 起 → API 双阶段健康)
4. 或系统重启后自动: systemctl status vllm-cluster.service (已 enable, 开机自动编排)
5. embed 检查: anemll-embed-8022 (<MGMT_OCTET>) / embed-qwen3-vllm (<MGMT_OCTET>/<MGMT_OCTET>) 自动恢复后 curl /health
6. 关键挂载预检: ls <INSTALL_DIR>/envs/nvcc_wrapper.py (若缺失: 从对端拷贝 1710B 文件)
```

**常见故障速查**（错误模式 → 根因 → 修复）：
| 错误 | 根因 | 修复 |
|------|------|------|
| HFValidationError '/models' | head 容器缺模型挂载 | start_head_v026r.sh 已含挂载，重跑编排 |
| `std::filesystem::exists(nvcc_path)` | 缺 nvcc_wrapper 挂载 | 检查 <INSTALL_DIR>/envs/nvcc_wrapper.py |
| `ibv_modify_qp failed 61` | <MGMT_OCTET> 重启后 GID3 空 | NCCL_IB_GID_INDEX=3（脚本已固定） |
| `Connection closed by peer [<NODE_IP>]` | worker 残留连旧 master | 编排脚本幂等清理后 head-first 重建 |
| `not a directory` mount error | /tmp/env-e-build 被误建目录 | 生产挂载已迁 <INSTALL_DIR>/envs/（不受影响） |

---

## 3. 待办（下一窗口）

- [ ] ~/vllm-cache、~/tilelang-cache、~/vllm-logs 迁移 <INSTALL_DIR>/{cache,logs}（低风险，避免 JIT 重编）
- [ ] 55+59 LLM TP2 部署脚本（start_head_5599.sh / start_worker_5599.sh）——benchmark 前
- [ ] internal API key 明文 → env 文件 chmod 600 + --env-file（Cody Medium）
- [ ] nvidia-sync 产物重定向 <INSTALL_DIR>/envs/env-e-build（或 systemd mount 单元）
- [ ] 口令轮换（P0）

---

## 4. TP4 时代新增文件/功能索引（QA 审计 2026-08-17 补充）

> 本目录索引 v1.0（2026-08-08）发布于 TP2（vllm-envE）时代；2026-08-12 起生产已切换为 **TP4 vLLM（vllm-tp4）**。以下为 TP4/治理阶段新增的关键文件与功能映射，由 QA 审计（Tessa，2026-08-17）核对落盘。旧版 §1.4/§2 中 TP2 条目保留为历史参考，不用于 TP4 恢复。

### 4.1 TP4 编排与自愈（生产运行）

| 文件 | 用途 | 关联文档 |
|---|---|---|
| <INSTALL_DIR>/scripts/start_tp4_head.sh | head(rank0) 启动（systemd 调用） | docs/scripts/REFERENCE.md；docs/runbook-tp4-v1.5-2026-08-12.md |
| <INSTALL_DIR>/scripts/start_tp4_worker_deepgemm.sh | worker(rank1/2/3) 启动 | docs/scripts/REFERENCE.md；docs/rollback-anchors-2026-08-12.md |
| <INSTALL_DIR>/scripts/start_tp4_cluster.sh | TP4 四机编排（head-first 幂等） | docs/scripts/REFERENCE.md；docs/ops/maintenance-plans.md |
| <INSTALL_DIR>/scripts/check_vllm_script.sh | 启动脚本完整性自检 | docs/scripts/REFERENCE.md；docs/ops/tools-index.md |
| <INSTALL_DIR>/scripts/monitor_tp4_head.sh | systemd 自愈 monitor | docs/scripts/REFERENCE.md；docs/ops/self-recovery.md |
| /etc/systemd/system/vllm-tp4-head.service | systemd 单元（EnvironmentFile=secrets/vllm.env） | deliverables/engineering-assurance/systemd-recovery-2026-08-15.md |
| <INSTALL_DIR>/secrets/vllm.env | **密钥文件（600 root:root）**，VLLM_API_KEY | 权限已核验；禁止落盘明文 |

### 4.2 NCCL 定制库与补丁（2be94172 生产链）

| 文件 | 用途 | 关联文档 |
|---|---|---|
| /opt/nccl-ringonly/libnccl.so.2.30.7 | 生产 NCCL 定制库（md5 **2be94172**） | deliverables/engineering-assurance/nccl-tuner-netdev-hardcode-adr-architect-2026-08-16.md（ADR-015）；nccl-final-performance-baseline-2026-08-17.md |
| <INSTALL_DIR>/lib/libncclpin.so | shim v8 线程绑定（NCCL 8-9 / EngineCore 15-19） | deliverables/engineering-assurance/nccl-ringonly-optimization-2026-08-15.md |
| backup/nccl-official-2307-hardened-20260816/patches/ | v1-ring-only / v4-netdev-hardcode / stageB-tuner-two-band / stageB-hardened-two-branch 补丁 | ADR-015（S1.7/S1.11/S1.12）；docs/tp4-service-deployment-guide 附录 |
| /opt/nccl-2307/ | 官方 2.30.7 源码（干净重建基线） | ADR-015 S1.7 |

### 4.3 基准测试工具（QA 判读配套）

| 文件 | 用途 | 关联文档 |
|---|---|---|
| <INSTALL_DIR>/bench_v2.py | vLLM v2 基准体系（DE/PR/接受率/monitor 隔离） | deliverables/engineering-assurance/nccl-benchmark-v2-report-qa-2026-08-16.md；nccl-final-performance-baseline-2026-08-17.md |
| <INSTALL_DIR>/bench_prefill_decode_async.py | 纯 prefill/纯 decode 分离基准（asyncio 真并发） | deliverables/engineering-assurance/nccl-followup-tests-qa-2026-08-16.md；test-nccl-ab-plan-2026-08-14.md |

### 4.4 测试/归档目录（非生产）

| 目录 | 内容 | 状态 |
|---|---|---|
| <INSTALL_DIR>/scripts/v027-test/ | vLLM 0.27 flashinfer patch 测试脚本（21 个） | 测试用；归档 backup/v027-nvfp4-archive-20260815/ |
| <INSTALL_DIR>/scripts/nccl-ab-B/ | NCCL A/B 单档延迟测试 | 待清（P0-5，见 governance-final-report） |
| /opt/2hop-s1/ | 2-hop 项目全部测试脚本/补丁/数据 | **D 收尾归档**：deliverables/engineering-assurance/nccl-2hop-archive-manifest-v1-architect-2026-08-17.md |
| <INSTALL_DIR>/backup/nccl-2hop-proto-archive-20260817/ | 2-hop proto 补丁与库归档 | 同上 |
| <INSTALL_DIR>/backup/nccl-official-2307-*/ | 官方源码+补丁各阶段归档 | ADR-015 S1.7-S1.12 |

### 4.5 治理交付物（deliverables/engineering-assurance/）

- 8/16-8/17 全量报告与 ADR 均在 <INSTALL_DIR>/deliverables/engineering-assurance/（约 40 份 md），总索引见 governance-overview-table-2026-08-17.md 与 governance-final-report-2026-08-17.md。
- 本机同步副本：四机 deliverables/ 目录由 mirror_to_02.sh 定期镜像（见 scripts/mirror_to_02.sh）。

### 4.6 已知待办（审计追加）

- [ ] file-registry 旧 §1.4/§2 TP2 条目与 TP4 现状并存，建议下窗口整篇刷新（QA 审计 2026-08-17 标记）
- [ ] 部署指南 §5.1（8/13 P1-P5 补丁表）与附录（8/16 Stage B 链）建议加交叉引用
- [ ] secrets/vllm.env 仅存 01 已核验；02/03/04 同构文件建议一并核验权限
