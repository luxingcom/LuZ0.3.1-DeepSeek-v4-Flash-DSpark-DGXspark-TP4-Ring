# DEPLOY_ASSIST · 预检内嵌位置对照表 + 克隆不一致→首次部署重建操作步骤

**工作流**: 工程保障 · 克隆环境 RoCE/GID 预检内嵌 (SRE Rex)
**日期**: 2026-08-26
**依据**: `incident-clone-roce-prevention-2026-08-25.md` 检查点1-5 + §分层预防 (P0-配置/观测)
**交付目录**: `deliverables/engineering-assurance/prod-hardening-2026-08-26/integrated/`

> 本文档配套 3 个内嵌版启动脚本，说明「预检插在哪几行」「遇到克隆环境不一致如何重建」。
> 所有改动仅新增预检序列 / 替换写死 index，原脚本 ENV_ARGS / BINDS / 参数 / 变量名全部保留。

---

## 一、预检内嵌位置对照表

### 1. `start_tp4_head.sh` (v1.5-r11 integrated)

| 行区段 | 插入/替换内容 | 对应检查点 |
|--------|--------------|-----------|
| 文件头 (VERSION 区) | 加 EXITCODES 3=GID布局No-Go / 4=env失配；注释 GID_SUGGEST_INDEX/PROBE_BIN 等 | 检查点1-5 汇总 |
| 变量定义区 (~L47) | 注入路径 `TP4_HARDEN_DIR/PROBE_BIN/PREFLIGHT_BIN/GID_ENV_BIN` + `GID_SUGGEST_INDEX` | 检查点2/3 |
| `ENV_ARGS` 组装之前 | `source gid_index_env.sh` 决定 `NCCL_IB_GID_INDEX`（缺失→降级 -1 并告警） | 检查点2 |
| `ENV_ARGS` 内原 L102 | **替换** `-e 'NCCL_IB_GID_INDEX=3'` → `-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"` | 检查点2 铁律 |
| `check_vllm_script.sh` 通过后、`docker rm` 前 | 新增 `probe_gid_index.sh` 探测建议 index + `preflight_roce_gid.sh --degrade [--expect-index N]`；exit3/4 → 打印 `[preflight-FAIL]` + 重建指引，**在 docker run 前返回 3** | 检查点1/3 |
| 预检异常分支 | `_preflight_fail()` 输出「克隆环境不一致→首次部署重建 6 步」指引 | 检查点1a/3 |

### 2. `start_tp4_worker.sh` (v1.5-r12 integrated)

| 行区段 | 插入/替换内容 | 对应检查点 |
|--------|--------------|-----------|
| 文件头 (VERSION 区) | 加 EXITCODES 3/4；注入路径 + `PEER_HCA` 前置 | 检查点1-5 |
| `ENV_ARGS` 组装之前 | `source gid_index_env.sh` 决定 `NCCL_IB_GID_INDEX` | 检查点2 |
| `ENV_ARGS` 内原 L107 | **替换** `-e 'NCCL_IB_GID_INDEX=3'` → `${NCCL_IB_GID_INDEX}` | 检查点2 铁律 |
| D1 门禁后、`check_vllm_script.sh` 后、`docker rm` 前 | 新增 **口名/HCA 一致性预检**（校验 `rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1` 存在且无预期外口名；失配→`[preflight-FAIL]` exit4 env失配） | 检查点1a |
| 同一处 | `probe_gid_index.sh` + `preflight_roce_gid.sh --degrade [--expect-index N]`；exit3/4 → 重建指引 exit3 | 检查点1/3 |

### 3. `start_tp4_cluster.sh` (v1.5-r12 integrated)

| 行区段 | 插入/替换内容 | 对应检查点 |
|--------|--------------|-----------|
| 变量区 | 注入 `PREFLIGHT_BIN/PROBE_BIN`、四机映射 `CLUSTER_HOST`、`RINGONLY_LIB` | 检查点1/3 |
| **step1 起 head 之前**（原 step0/挂载/GPU/自检/清理之后、`trap`/step1 之前） | 新增 **STEP 0 集群级一致性核验**：四机逐个 (a) GID 空洞枚举 (b) `preflight_roce_gid.sh --degrade` exit3/4 判定 (c) `RINGONLY libnccl.so.2` MD5；(d) 四机 MD5 唯一性汇总。任一空洞/No-Go/MD5 不一致 → 集群停启，`_cluster_rebuild_hint()` 输出重建指引并 **exit 3** | 检查点1/3 + RING 全集一致性 |
| 后续 step1-4 | **保持原逻辑完全不变**（head-first 幂等 + TCPStore 门禁 + B12X 错峰） | — |

---

## 二、克隆环境不一致 → 首次部署重建操作步骤

> 触发场景：`start_tp4_*.sh` 在 docker run 前打印
> `[preflight-FAIL]` / `[cluster-FAIL]` / `[rebuild-guidance]` 即代表克隆 RoCE/GID 布局与生产不一致。
> **绝不带病启动**（RCA 最大教训：等容器起来再卡死诊断 = 代价极高）。

### Step 1 · dump 全机 GID 表（检查点 1a）
四机各跑（确认口名是否 `rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1`，**非此名 → 直接 NO-GO**）：
```bash
for p in /sys/class/infiniband/*/ports/*/gids/[0-9]*; do
  idx=${p##*/}; dev=${p%/*}
  gid=$(cat "$p" 2>/dev/null)
  [ "$gid" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && st=HOLE || st=ok
  printf "%-24s idx=%2s %-45s %s\n" "$dev" "$idx" "$gid" "$st"
done | sort
```

### Step 2 · 修复 GID 空洞
- 任一 `HOLE` / `::` / 全零 → 执行 `fix_gid_holes`（重建该 index 的 IPv4 RoCEv2 GID，使 index 与克隆实际 RoCE 网段 L2 前缀对齐）。
- 修复后重跑 Step 1 dump 复检无洞。

### Step 3 · 重建网络配置（重算 HCA / PEER_HCA）
- 依据实际口名重算 `NCCL_IB_HCA`（head/worker 内嵌脚本读取的 4 twin 口集合）。
- worker 的 `PEER_HCA`（rank1/2/3 的 RING 邻接口映射）须按实际口名重写，否则报 `[preflight-FAIL] exit4`。

### Step 4 · preflight 通过（检查点 1+3 硬门）
每机独立 + 集群协调两级别都须通过：
```bash
# 每机本机布局门
bash preflight_roce_gid.sh --degrade                 # exit0 才放行
# 期望 index 校验（index 为数字时）
bash preflight_roce_gid.sh --expect-index <N>        # N=首 RoCEv2/IPv4 非洞 index
# 四机交叉核对（由 head 协调层统一做，见 cluster STEP 0）
# QA-fix I4/I8: `--peers <host:idx>` 判据3 交叉核对由调用方 start_tp4_cluster.sh 在 STEP 0
#   编排层以四机邻接表触发 (preflight/probe 均支持 "host:idx" 与 "host=idx" 两种写法)。
```
全部 exit0 后，`NCCL_IB_GID_INDEX` 由 `gid_index_env.sh` 按『人工覆写 → probe → 洞栅栏 → -1 动态兜底』决策注入。

### Step 5 · 再启动
```bash
bash /opt/aicad-prod/scripts/start_tp4_cluster.sh    # 集群编排（内含 STEP 0 四机复核）
# 或单机调试：
bash /opt/aicad-prod/scripts/start_tp4_head.sh
NODE_RANK=N VLLM_HOST_IP=<ip> bash /opt/aicad-prod/scripts/start_tp4_worker.sh
```

### Step 6 · 验证
- `NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log` 且 `-v ~/vllm-logs:/var/log/vllm` → 日志落持久卷，`nccl-log` 中核对实际生效的 GID index。
- `watchdog_hardened.sh` / `healthcheck_hardened.sh` 确认运行期无 `ibv_modify_qp` 22/61、无 `shm_broadcast` hang。
- carrier flap 探针（检查点5）观察 link_down_events 增量，必要时治物理根因。

---

## 三、多层分层防护点映射（对齐事故报告 T0/L1/L2/观测）

| 层 | 故障 | 本交付防护点 |
|----|------|-------------|
| T0 | carrier flap（物理，克隆放大） | cluster STEP 0 口名/HCA 核验 + worker 口名门 + 检查点5 采集指引（watchdog 探针配合） |
| L1 | QP 断 → IBV_WC_RETRY_EXC_ERR → worker 静默死 | 日志落持久卷（保留原有 `-v ~/vllm-logs`）+ 预检 fail-fast 减少带病 QP |
| L2 | GID 空洞 → 写死 index → shm_broadcast hang（**本方案主防**） | 移除写死 `=3` → 动态注入；`preflight --expect-index` + 空洞栅栏在 docker run 前阻断 |
| 观测 | 日志不落卷/守卫盲区/看门狗失效 | 保留持久卷 + crash_dump.sh（ExecStopPost）+ healthcheck/watchdog 加固（独立脚本配套） |

> 关键铁律重申：**克隆严禁盲目复用生产 `NCCL_IB_GID_INDEX=3`**，必须「实测当前 GID 表 → 选型 → 注入」，本内嵌序列已强制该路径；任一不一致显式告警，绝不静默 pass。