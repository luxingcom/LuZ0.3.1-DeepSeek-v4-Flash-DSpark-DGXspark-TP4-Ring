# 回滚锚点与降级引用手册 v1.5-R11（Rollback Anchors & Fallback Reference）

**日期**：2026-08-12（R11 修订）｜**维护**：Docu｜**权威路径**：`<INSTALL_DIR>/docs/rollback-anchors-2026-08-12.md`（01/02 镜像）
**适用范围**：DGX Spark 4 机 TP4 生产集群（01=node01/<MGMT_OCTET>/186，02=node01/<MGMT_OCTET>/187，03=node01/<MGMT_OCTET>/188，04=node01/<MGMT_OCTET>/189）
**当前生产栈（R11）**：shim **v8** + util **0.65** + **seqs=6** + fix72 capture(64/1..64) + Prefix KV + 400k + 本地 serving（NFS 集中化恢复中）+ systemd 自愈（StartLimit **1800/20** + 互杀守卫 + 门禁）+ TP4 环网（start_tp4_* 系列），8001=200

> 本文件为 R11 修订版，**取代**旧版同文件（旧版在本地 `deliverables/engineering-assurance/rollback-anchors-2026-08-12.md` 保留）。变更点：shim 回滚锚点升级为 v8 链、新增 R11 脚本 `.bak-r11` 体系、StartLimit 值更新、互杀守卫说明。

---

## 0. 快速导航（TL;DR）

| 需要 | 动作 | 详见 |
|---|---|---|
| TP4 起不来，回 TP2 | 四机 rm vllm-tp4-rank0~3 → 01 `start_v026r_cluster.sh` | §1.2 / §4 |
| R11 脚本改动想还原 | 用 `.bak-r11-20260812-2222xx` 覆盖当前 start_tp4_* | §1.1 |
| NCCL 补丁库回退 | 还原 `/opt/nccl-ringonly/libnccl.so.2.30.7.bak-v2` | §2.1 |
| 隔离核 shim 库回退 | 还原 `libncclpin.so.bak-v7`（v8 前最后版本） | §2.2 |
| 网络配置回退 | 还原 `backup/ring-fix-20260811/<host>/` | §2.3 |
| R11 修复态回退 | 软链改回 `.local-backup` + 旧 util/seqs 脚本 | §2.8 |

---

## 1. 脚本状态清单（<INSTALL_DIR>/scripts/）

### 1.1 生产在用（✅ R11 版本，勿动）

| 机 | 脚本 | 说明 |
|---|---|---|
| 01 | `start_tp4_cluster.sh` | TP4 编排 v2（head-first 幂等） |
| 01 | `start_tp4_head.sh` | head(rank0) 启动（shim v8 + util 0.65 + seqs=6 + capture 64） |
| 02/03/04 | `start_tp4_worker.sh` | worker 启动（rank1/2/3） |
| 01 | `monitor_tp4_head.sh` | head 自愈 monitor（容器退出→清 worker→重建） |
| 02/03/04 | `monitor_tp4_worker.sh` | worker 自愈 monitor（**含互杀守卫**） |
| 四机 | `check_vllm_script.sh` | vLLM 参数完整性自检 |
| 四机 | `start_embed_8022.sh` | embed 生产启动（03/04 主） |

**R11 备份链**（8/12 22:22-22:24 应用前快照）：`start_tp4_head.sh.bak-r11-20260812-222213`、`start_tp4_worker.sh.bak-r11-20260812-222213`（02/03/04 为 `-142214/-142215`）、`monitor_tp4_head.sh.bak-r11-20260812-222213`、`monitor_tp4_worker.sh.bak-r11-20260812-142214`、`check_vllm_script.sh.bak-r11-20260812-222213`。

**脚本 MD5（R11 最终，8/12 23:41 复核）**：
- head `06069400908504bc35e73d9144218b9f`｜worker `83cdc3c9668c0adc7fc634a898411b5b`｜mon_head `284b7b147676ae2fd6e64c60c73ca78a`｜mon_worker `3dfe1af7fab071d93da9cfb4c7c2affa`｜check `60496107ed59e28d233f87b08348c69b`

### 1.2 退役但保留为回滚锚点（⚠️ 锚点）
- 01 `start_v026r_cluster.sh`（+ `.bak-*` 共 5 份）：**TP2 降级唯一入口**，全程未动。
- 01 `start_head_v026r.sh`（+ `.bak-*`）；02 `start_worker_v026r.sh`（+ `.bak-*`）。
- 03/04 `start_groupB_*.sh`：组 B 双机锚点。

### 1.3 临时/测试（archi 系列——待阿奇测试完成后确认）
- 01 `<INSTALL_DIR>/archi-test/scripts/`：🔴 在用不动。
- 本地 `deliverables/engineering-assurance/scripts/`、`sre-tmp-20260811/`：文档引用快照。

---

## 2. 回滚锚点详表

### 2.1 NCCL ring-only 补丁库（`.bak-v2`）
- 位置：四机 `/opt/nccl-ringonly/libnccl.so.2.30.7.bak-v2`（60.5M）
- 当前生产：`libnccl.so.2.30.7`（MD5 `b7784b49885659c27765e648884e4edd`，banner `2.30.7+cuda13.0`）
- 注记：v3 双口已上线（2026-08-12 生效）
- 回滚：`sudo cp libnccl.so.2.30.7.bak-v2 libnccl.so.2.30.7` → 重启 TP4（head-first）｜优先级 P1

### 2.2 隔离核 shim 库（v8 回滚链）
- **当前生产 = v8**：`<INSTALL_DIR>/lib/libncclpin.so`（70920B，8/12 15:03，绑定 **8-9**）
- 备份链：`.bak-v7`（70696B，v8 前）→ `.bak-v6-20260812` → `.bak-v5-18-19` → `.bak-v4` → `.bak-v3`（各版本见 `_r11_fix/ncclpin_v*.c` 源码）
- 回滚：`sudo cp <INSTALL_DIR>/lib/libncclpin.so.bak-v7 <INSTALL_DIR>/lib/libncclpin.so` → 重启 TP4 容器（head-first）｜优先级 P2

### 2.3 网络配置锚点（`ring-fix-20260811`）
- 位置：四机 `<INSTALL_DIR>/backup/ring-fix-20260811/<dgxspark0N>/`（netplan/hosts/rules.v4/daemon.json）
- 用途：8/11 环补闭环前快照；回滚会断 03↔01 直连｜优先级 P1

### 2.4 补丁源码归档（`tp4-20260812`）
- 位置：01 `<INSTALL_DIR>/backup/tp4-20260812/`（79M，src+diff+artifacts+scripts+logs+README）
- 用途：NCCL 补丁源码级重建/审计唯一依据｜优先级 P0 禁删

### 2.5 TP2 容器配置快照
- 01 `backup/tp2-node.json`、02 `backup/tp2-worker.json`（docker inspect 快照，TP2 还原依据）｜P1

### 2.6 镜像 tag 锚点
- `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（TP4/TP2 生产，四机一致）✅
- `dspark-vllm-gx10:0.2.1-archi-test`（阿奇 A/B 在用）🔴 不动
- 历史 tag（<MGMT_OCTET>:5000 0.1.1 / 0.2.1-v026.0 副本）🟡 可保留作回退

### 2.7 其他 backup/ 归档
- `rollback_tp4-rank{0-3}.json`（四机 backup/，每次启停覆写）
- `nccl-pin-archive-20260809`、`scripts-archive/`、`tmp-residue-*`、`iptables_*_before_*.txt`（详见本地旧版 §2.7）

### 2.8 R11 修复态回退（新增）
- 场景：本地 serving 后 03/04 推理异常需回 NFS：软链改回 `<MODELS_DIR>/deepseek-v4-flash-0731`（挂载版）+ `mount -a` 复核（流程见 runbook v1.5 §C）。
- 场景：seqs=6/capture 64 性能异常需回旧档：还原 `.bak-tp4-seq6`（6→12 前）/`.bak-tp4-util065`（util 0.65 前）/`check_vllm_script.sh.bak-400k-20260812-144825`。

---

## 3. TP4 ↔ TP2 切换速查（不变）

TP4→TP2：四机 `docker rm -f vllm-tp4-rank0~3` → 01 `bash <INSTALL_DIR>/scripts/start_v026r_cluster.sh` → 验证 8001/8003/4000=200。
TP2→TP4：01 `bash start_tp4_cluster.sh`（head-first）→ 验证 8001=200 + 四机 healthy + banner `2.30.7+cuda13.0`。

---

## 4. 交付物索引（R11 系列）

| 文档 | 要点 |
|---|---|
| `README.md` | 权威文档入口 + R11 基线速记 |
| `runbook-tp4-v1.5-2026-08-12.md` | **R11 修复 8 项 + 验证 7 项 + TP4S 测试基线 + NFS 恢复流程** |
| `ops/server-maintenance-handbook.md` v1.5-R11 | 维护手册（R11 参数全集/巡检增量） |
| `scripts/REFERENCE.md` | 脚本↔文档引用索引 + 帮助头标准 |
| `rollback-anchors-2026-08-12.md` | **本文档** |
| 本地 deliverables/…/tp4-*-report-2026-08-12.md | 8/12 系列报告（r8/solidify/v3-deepen/bottleneck） |

## 5. 待办 / 风险提示
1. 阿奇 A/B 测试窗口结束后处置 archi-test 资产。
2. NFS 集中化恢复落地后更新本手册与维护手册（模型源状态）。
3. `rollback_tp4-rank*.json` 建议固化容器 inspect 快照（对齐 tp2-*.json）。
4. `start_v026r_cluster.sh` 为 TP2 唯一权威入口，**任何维护不得覆盖**。
