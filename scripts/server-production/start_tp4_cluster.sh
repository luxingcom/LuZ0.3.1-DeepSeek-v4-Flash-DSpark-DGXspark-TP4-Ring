#!/bin/bash
# =============================================================
# SCRIPT: start_tp4_cluster.sh
# VERSION: v1.5-r12 (integrated)
# ROLE: TP4 四机编排 (head-first, 幂等) — 在 dgxspark01 运行
# HOST: dgxspark01
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/maintenance-plans.md
# DOCS: file:///opt/aicad-prod/docs/runbook-tp4-v1.5-2026-08-12.md
# EXITCODES: 0=成功 1=业务失败 2=用法错误 3=集群级 GID/一致性 No-Go 4=env失配 130=被signal
# CHANGE: 改脚本须 check_vllm_script.sh 通过 + .bak-<tag> 留档 + 更新 REFERENCE.md
# FIX(2026-08-24, r12): step 2.5 B12X 门禁死锁修复 — 原门禁等 head 日志 "Using 'B12X_MXFP4'"
#   才启 workers, 但 head 引擎核心初始化需 4 rank NCCL 域 → 冷启动互等 300s 超时。
#   改为 TCPStore:25999 就绪(step 2 信号) + B12X_JIT_STAGGER 固定错峰启 workers
#   (保留防多 worker 并行撞 b12x JIT 竞态意图)。留档 .bak-b12xgate-fix-20260824。
# =============================================================
# [内嵌 2026-08-26] 集群级 RoCE/GID 一致性核验 (incident-clone-roce-prevention-2026-08-25.md
#   检查点1+3): 在 step2(启 workers) 之前, 四机 GID 布局 / 口名 / RINGONLY MD5 逐一核验,
#   任一 exit3/4 或有洞 / MD5 不一致 → 集群停启 + 提示【首次部署重建】, 不带病启 workers。
# FIX(2026-08-26, QA gidrange): STEP0 0.1 段 GID 空洞枚举原遍历 /sys/class/infiniband/*/ports/*/gids/[0-9]*
#   全部索引槽 (MLNX 网卡含大量未配置空槽, 四机实测 1027 文件仅 index0-3 有效=16),
#   空槽被误判 HOLE → 生产 4 口 ring 必被 exit3 拦截。改为仅枚举 index0-3
#   (与 preflight_roce_gid.sh 判据1 对齐, 只查真实布局窗口), 空槽不再判洞。
#   留档 .bak-cluster-gidrange-20260826。
# =============================================================
# R11 KEY_PARAMS: seqs=6 | util=0.65 | capture=1..64(max64) | PSR: NCCL=8-9 EngineCore=15-19
# rank 映射: 01=0(186) 02=1(187) 04=2(189) 03=3(188) | MASTER_PORT=25999(管理网)
# =============================================================
set -uo pipefail

# -h/--help (无位置参数)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
start_tp4_cluster.sh — TP4 四机编排 (head-first, 幂等, dgxspark01)
  R11 关键参数: seqs=6 | util=0.65 | capture=1..64(max64) | PSR: NCCL=8-9 EngineCore=15-19
  内嵌一致性核验(2026-08-26): 四机 GID/口名/RINGONLY MD5 一致性, 异常→集群停启+重建提示
  用法: bash start_tp4_cluster.sh [--help]
参考: /opt/aicad-prod/docs/scripts/REFERENCE.md（脚本↔文档索引）| /opt/aicad-prod/docs/README.md（入口）
EOF
  exit 0
fi

HEAD_HOST="dgxspark01"
MASTER_ADDR="192.168.5.186"
MASTER_PORT=25999
API_PORT=8001
API_KEY="${VLLM_API_KEY:?VLLM_API_KEY is not set}"
MAX_WAIT_S=600
# r12: 门禁语义替换 — TCPSTORE_GATE_WAIT=step 2.5 TCPStore 复核门禁上限;
# B12X_JIT_STAGGER=错峰启 worker 间隔, 保留"防多 worker 并行撞 b12x JIT 竞态"意图 (8/20 SEV1 复盘)
TCPSTORE_GATE_WAIT="${TCPSTORE_GATE_WAIT:-300}"
B12X_JIT_STAGGER="${B12X_JIT_STAGGER:-20}"
HEAD_LOG="$HOME/start_tp4_cluster.log"
PROD=/opt/aicad-prod

# [内嵌 2026-08-26] 生产加固脚本路径 (检查点1-3)
PREFLIGHT_BIN="${TP4_PREFLIGHT_BIN:-${PROD}/scripts/preflight_roce_gid.sh}"
PROBE_BIN="${TP4_PROBE_BIN:-${PROD}/scripts/probe_gid_index.sh}"
# 四机核验主机映射 (rank -> hostname:ip) — 依 runbook + head 注释; 检查点1a 需要真实机名/口名
#   head=01(186) | worker: 02=1(187) 04=2(189) 03=3(188)
declare -A CLUSTER_HOST=(
  [0]="dgxspark01:192.168.5.186"
  [1]="dgxspark02:192.168.5.187"
  [2]="dgxspark04:192.168.5.189"
  [3]="dgxspark03:192.168.5.188"
)
# RINGONLY 库可选路径 (检查 RING 全集一致性; MD5 不一致 → WARN/或 No-Go)
RINGONLY_LIB="/opt/nccl-ringonly/libnccl.so.2"

# rank->host 映射 (环序) — QA-fix I7: 原全指向 dgxspark01:192.168.5.186 为基线继承的错误,
#   导致 step3 启 worker 全 ssh 到 head。现修正为真实四机映射, 与上方 CLUSTER_HOST 保持一致:
#   rank1=dgxspark02:192.168.5.187(187) | rank2=dgxspark04:192.168.5.189(189) | rank3=dgxspark03:192.168.5.188(188)
#   拓扑: 01=rank0(186) 02=rank1(187) 04=rank2(189) 03=rank3(188)
declare -A RANK_HOST=(
  [1]="dgxspark02:192.168.5.187"
  [2]="dgxspark04:192.168.5.189"
  [3]="dgxspark03:192.168.5.188"
)

ERR_PATTERNS="HFValidationError|nvcc_wrapper|ibv_modify_qp|GLOO|Gloo|connectFullMesh|collective_rpc|NCCL error|OutOfMemory|CUDA out of memory|No such file|not a directory|Engine core initialization failed|ncclSystemError|DistStoreError|NV_ERR_NO_MEMORY|Broken pipe"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

diagnose() {
  log "[诊断] 关键错误扫描 (head 日志 $HEAD_LOG):"
  grep -E "$ERR_PATTERNS" "$HEAD_LOG" 2>/dev/null | tail -6 || echo "  (无已知错误模式)"
  for r in 0 1 2 3; do
    log "[诊断] rank${r} 容器最近日志:"
    docker logs --tail 8 "vllm-tp4-rank${r}" 2>&1 | tail -6 || true
  done
}

# ---- 守卫: 必须在本机(head)执行 ----
[ "$(hostname)" = "dgxspark01" ] || [ "$(hostname)" = "spark-05cd" ] || {
  echo "ERROR: 本脚本仅可在 head(dgxspark01) 上运行" >&2
  exit 1
}

# ---- 前置: 挂载源 ----
if ! [ -f "$PROD/envs/nvcc_wrapper.py" ]; then
  log "[error] 挂载源缺失: $PROD/envs/nvcc_wrapper.py" >&2
  exit 1
fi
if ! [ -e "$PROD/models/deepseek-v4-flash-0731/config.json" ]; then
  log "[error] 模型挂载源缺失: $PROD/models/deepseek-v4-flash-0731" >&2
  exit 1
fi
log "[ok] 挂载源就绪"

# ---- GPU-gate: nvidia-smi 就绪 (≤180s) ----
log "[pre] GPU 就绪探测 (nvidia-smi, 最多 180s)..."
GPU_OK=0
for ((i=0; i<36; i++)); do
  if nvidia-smi -L >/dev/null 2>&1; then GPU_OK=1; break; fi
  sleep 5
done
if [ "$GPU_OK" -ne 1 ]; then
  log "[error] 180s 内 nvidia-smi 不可用 — GPU 驱动未就绪, 中止" >&2
  exit 1
fi
log "[ok] GPU 就绪 (第 $((i+1)) 次探测)"

# ---- 前置自检: 4 个脚本 ----
log "[pre] 脚本自检: head + 3 worker"
if ! HOME=/home/liuxiaoya "$PROD/scripts/check_vllm_script.sh" "$PROD/scripts/start_tp4_head.sh"; then
  log "[error] head 脚本自检失败" >&2; exit 1
fi
for r in 1 2 3; do
  host="${RANK_HOST[$r]%%:*}"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "HOME=/home/liuxiaoya $PROD/scripts/check_vllm_script.sh $PROD/scripts/start_tp4_worker.sh"; then
    log "[error] rank${r}($host) 脚本自检失败" >&2; exit 1
  fi
done
log "[ok] 4 脚本自检通过"

# ---- 清理残留容器 (幂等) ----
docker rm -f vllm-tp4-rank0 >/dev/null 2>&1 || true
for r in 1 2 3; do
  host="${RANK_HOST[$r]%%:*}"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "docker rm -f vllm-tp4-rank${r} >/dev/null 2>&1 || true"
done
log "[ok] 四机容器已清理"

# ---- 8001 释放验证 (head) ----
if ss -tln 2>/dev/null | grep -q ":$API_PORT"; then
  log "[error] :${API_PORT} 仍被占用, 中止" >&2
  ss -tlnp 2>/dev/null | grep ":$API_PORT" | head -2
  exit 1
fi

# =============================================================
# [内嵌 2026-08-26] STEP 0: 集群级 RoCE/GID/口名/RINGONLY 一致性核验 (检查点1+3)
#   在 step1(起 head) / step2(启 workers) 之前 强制四机核验, 杜绝克隆不一致带病启动。
#   分层: T0 carrier flap(物理) / L1 QP 断(运行时) / L2 GID 空洞(重启卡死, 本门主防)
#   任一机 exit3(No-Go)/exit4(env失配) 或有洞 / 口名失配 / RINGONLY MD5 不一致
#   → 集群停启 + 提示【首次部署重建】, 不继续启任何容器。
# =============================================================
_cluster_rebuild_hint() {
  echo "------------------------------------------------------------" >&2
  echo "[cluster-FAIL] 克隆环境 RoCE/GID 布局跨机不一致 → 集群停启, 拒绝带病启动" >&2
  echo "[rebuild-guidance] 请执行【首次部署重建】(检查点1a/检查点3) 后再整集群启动:" >&2
  echo "  1) 四机各 dump GID 表:  for p in /sys/class/infiniband/*/ports/*/gids/[0-9]*; do ... done" >&2
  echo "  2) 每机跑 preflight: bash ${PREFLIGHT_BIN} --degrade   (有洞 → 修)" >&2
  echo "  3) fix_gid_holes 重建缺失 GID;  重算 HCA/PEER_HCA 到克隆实际口集合" >&2
  echo "  4) 对齐 RINGONLY 库: 各机 ${RINGONLY_LIB} MD5 须一致, 否则重拷同一镜像内实现" >&2
  echo "  5) 四机各自 preflight --expect-index N 全部 exit0 后再运行本编排脚本" >&2
  echo "  6) 通过后重启: bash $0" >&2
  echo "------------------------------------------------------------" >&2
}

cluster_gate_fail=0
# 用 run-remote 抽象: rank0 本机直跑, rank1-3 ssh
declare -A NODE_GID_HAS_HOLE=()
declare -A NODE_GID_OK=()
declare -A NODE_MD5=()
declare -A NODE_RC=()
EMPTY_ALLZERO='^0000:0000:0000:0000:0000:0000:0000:0000$'

# 辅助: 在指定节点执行一段检查命令, 返回 (stdout 捕获于 var)
run_node_check() {
  local r="$1" script="$2"
  local host="${CLUSTER_HOST[$r]%%:*}"
  if [ "$r" = "0" ]; then
    eval "$script"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "bash -s" <<EOF
$script
EOF
  fi
}

log "[step 0/4] 集群级 RoCE/GID/口名/RINGONLY 一致性核验 (4 机) — 检查点1+3"
for r in 0 1 2 3; do
  host="${CLUSTER_HOST[$r]%%:*}"
  ip="${CLUSTER_HOST[$r]##*:}"

  # --- 0.1 GID 空洞/布局自检 (每机跑 preflight 本机布局门) ---
  log "  [node0x] 核验 ${host} (${ip}) GID 布局..."
  node_script='
    set -uo pipefail
    ALLZ="^0000:0000:0000:0000:0000:0000:0000:0000$"
    hole=0; n=0
    # QA-fix(2026-08-26, gidrange): 仅枚举 index0-3 (与 preflight 判据1 对齐);
    #   原遍历 [0-9]* 把未配置空槽(index>3)误判 HOLE → 4 口 ring 必误拦。空槽不判洞。
    for idx in 0 1 2 3; do
      for gf in /sys/class/infiniband/*/ports/*/gids/${idx}; do
        [ -f "$gf" ] || continue
        n=$((n+1))
        g=$(cat "$gf" 2>/dev/null | tr -d "\n")
        if [ -z "$g" ] || [ "$g" = "::" ] || [[ "$g" =~ ^0000:0000:0000:0000:0000:0000:0000:0000$ ]]; then
          echo "HOLE ${gf##*/}"
          hole=1
        fi
      done
    done
    echo "NODEGIDS=$n HOLE=$hole"
    echo "NODEHOST=$(hostname)"
  '
  if OUT=$(run_node_check "$r" "$node_script" 2>&1); then
    echo "$OUT" | sed 's/^/    [gid] /'
    hole=$(echo "$OUT" | grep -oE 'HOLE=[01]' | cut -d= -f2 | head -1)
    if [ "$hole" = "1" ]; then NODE_GID_HAS_HOLE["$r"]=1; NODE_GID_OK["$r"]=0; cluster_gate_fail=1; else NODE_GID_OK["$r"]=1; fi
  else
    echo "    [gid] ${host} GID 枚举失败" >&2
    NODE_GID_OK["$r"]=0; cluster_gate_fail=1
  fi

  # --- 0.2 preflight 硬门 (exit3=No-Go / exit4=env失配) ---
  log "  [node0x] ${host} preflight 硬门 (exit3/4 判定)..."
  pre_script=$'set -uo pipefail\nif [ -x '"${PREFLIGHT_BIN}"$' ]; then\n  bash '"${PREFLIGHT_BIN}"$' --degrade >/tmp/preflight.out 2>&1; rc=$?; cat /tmp/preflight.out; echo "PREFLIGHT_RC=$rc"; exit 0\nelse\n  echo "PREFLIGHT_RC=255 no-bin"; exit 0\nfi'
  if PRE_OUT=$(run_node_check "$r" "$pre_script" 2>&1); then
    prc=$(echo "$PRE_OUT" | grep -oE 'PREFLIGHT_RC=[0-9]+' | cut -d= -f2 | head -1)
    echo "$PRE_OUT" | sed 's/^/    [pre] /' | grep -E '\[preflight\]|No-Go|HOLE|PREFLIGHT_RC' || true
    NODE_RC["$r"]="$prc"
    if [ -n "$prc" ] && { [ "$prc" = "3" ] || [ "$prc" = "4" ]; }; then
      log "  [cluster-FAIL] ${host} preflight=No-Go(rc=$prc) — 布局不一致" >&2
      cluster_gate_fail=1
    fi
  else
    log "  [warn] ${host} preflight ssh/执行异常, 记 WARN (避免静默)" >&2
    NODE_RC["$r"]="-1"; cluster_gate_fail=1
  fi

  # --- 0.3 RINGONLY MD5 一致性 (全克隆对同一 RING 实现; 跨机不一致 → 需重维护) ---
  log "  [node0x] ${host} RINGONLY MD5 (${RINGONLY_LIB})..."
  # FIX(2026-08-26, md5fix): 原 ANSI-C quoting+eval 的 awk "{print \\\$1}" 在 set -u 下被 shell
  #   提前展开为未绑定变量 $1 → MD5_OUT 捕获失败 → 4 机全判 UNREACH 误报失配。改为纯单/双引号
  #   拼接, awk 收到字面 \$1 → 经 eval/ssh 两路径均正确输出 hash (bash -n + 实测通过)。
  md5_script='set -uo pipefail; if [ -f '"${RINGONLY_LIB}"' ]; then md5sum '"${RINGONLY_LIB}"' | awk "{print \$1}"; else echo NOFILE; fi'
  if MD5_OUT=$(run_node_check "$r" "$md5_script" 2>&1); then
    md5v=$(echo "$MD5_OUT" | grep -Eo '^[0-9a-f]{32}' | head -1)
    [ -z "$md5v" ] && md5v="NOFILE"
    NODE_MD5["$r"]="$md5v"
    echo "    [md5] ${host} ${RINGONLY_LIB} = ${md5v}"
  else
    NODE_MD5["$r"]="UNREACH"
    echo "    [md5] ${host} RINGONLY 不可达" >&2
  fi
done

# --- 汇总: RINGONLY MD5 唯一性 (四机须一致) ---
# QA-fix I10: 原实现用 grep -v 过滤 UNREACH/NOFILE, 导致"某节点缺库"不被后续 MD5 唯一性
#   判据拦截 → 集群门静默放过。现补: 任一节点 NOFILE(缺库)/UNREACH(不可达) 一律视为失配,
#   集群门 fail (四机 RING 全集一致性: 缺实现即不一致)。
for r in 0 1 2 3; do
  case "${NODE_MD5[$r]}" in
    NOFILE|UNREACH)
      log "[cluster-FAIL] rank${r} (${CLUSTER_HOST[$r]%%:*}) RINGONLY = ${NODE_MD5[$r]} (缺库/不可达) → 视为失配" >&2
      cluster_gate_fail=1
      ;;
  esac
done
MD5_UNIQ=$(for r in 0 1 2 3; do echo "${NODE_MD5[$r]}"; done | sort -u | grep -vE '^(UNREACH|NOFILE)$')
MD5_NUNIQ=$(echo "$MD5_UNIQ" | grep -c .)
if [ "$MD5_NUNIQ" -gt 1 ]; then
  log "[cluster-FAIL] RINGONLY 库跨机 MD5 不一致 (${MD5_NUNIQ} 个版本):" >&2
  for r in 0 1 2 3; do echo "    rank${r} (${CLUSTER_HOST[$r]%%:*}): ${NODE_MD5[$r]}" >&2; done
  cluster_gate_fail=1
else
  log "[ok] RINGONLY 库四机 MD5 一致: ${MD5_UNIQ}"
fi

# --- 汇总判定: 任一机有洞 / preflight No-Go / MD5 不一致 → 集群停启 ---
if [ "$cluster_gate_fail" = "1" ]; then
  log "[FAIL] 集群级一致性核验未通过 (见上方 [cluster-FAIL]/[gid HOLE]/preflight No-Go)" >&2
  _cluster_rebuild_hint
  exit 3
fi
log "[ok] 4 机 GID 布局/口名/preflight/RINGONLY MD5 一致性核验通过"

trap 'if [ $? -ne 0 ]; then log "[FAIL] 编排中止, 执行诊断:"; diagnose; fi' EXIT

# ---- 1. 启动 head(rank0) ----
log "[step 1/4] 启动 head 容器 (rank0)..."
nohup bash "$PROD/scripts/start_tp4_head.sh" > "$HEAD_LOG" 2>&1 &
HEAD_PID=$!
log "[i] head 启动脚本后台 PID=$HEAD_PID, 日志: $HEAD_LOG"

# ---- 2. 轮询 head TCPStore :25999 ----
log "[step 2/4] 轮询 head TCPStore :${MASTER_PORT} 就绪 (最多 ${MAX_WAIT_S}s)..."
HEAD_READY=0
CONSEC=0
for ((i=0; i<MAX_WAIT_S/5; i++)); do
  if ss -tln 2>/dev/null | grep -q ":$MASTER_PORT"; then
    CONSEC=$((CONSEC+1))
    if [ "$CONSEC" -ge 2 ]; then HEAD_READY=1; log "[ok] TCPStore :${MASTER_PORT} 就绪 (连续 2 次)"; break; fi
  else
    CONSEC=0
  fi
  sleep 5
done
if [ "$HEAD_READY" -ne 1 ]; then
  log "[error] head TCPStore 未就绪, 中止" >&2
  diagnose
  exit 1
fi

# ---- 2.5 B12X JIT 竞态错峰门禁 (修复版 2026-08-24, 替代 8/20 的 B12X 日志门禁) ----
# 原门禁死锁根因: head 引擎核心初始化 (parallel_state backend=nccl) 需全部 4 rank 加入
#   NCCL 通讯域才推进到 MoE/B12X 加载; 原门禁等 head 日志 "Using 'B12X_MXFP4'" 才启 workers,
#   head 单独启动阻塞在 NCCL peer 等待 → B12X_MXFP4 日志永不出现 → 300s 超时
#   "Engine core initialization failed" → 冷启动死锁 (见 g1-production-restore-2026-08-24 §3.3)。
# 修复: 门禁目标改为 TCPStore:25999 就绪 (head 容器启动即开, 与 step 2 同一信号) +
#   固定错峰 (B12X_JIT_STAGGER) 逐个启动 workers, 保留"防多 worker 并行撞 b12x JIT 竞态"
#   原始意图, 同时让 4 rank 尽快进入 NCCL rendezvous, 消除互等死锁。
log "[step 2.5/4] B12X 错峰门禁: 复核 TCPStore :${MASTER_PORT} 就绪 (最多 ${TCPSTORE_GATE_WAIT:-300}s), 随后按 ${B12X_JIT_STAGGER:-20}s 错峰启 workers"
TCPSTORE_READY=0
for ((i=0; i<${TCPSTORE_GATE_WAIT:-300}/5; i++)); do
  if ss -tln 2>/dev/null | grep -q ":$MASTER_PORT"; then
    TCPSTORE_READY=1
    log "[ok] TCPStore :${MASTER_PORT} 就绪 (第 $((i+1)) 次, $((i*5))s)"
    break
  fi
  if grep -qE "Engine core initialization failed" "$HEAD_LOG"; then
    log "[error] head 引擎核心初始化失败, 提前中止" >&2
    diagnose
    exit 1
  fi
  sleep 5
done
if [ "$TCPSTORE_READY" -ne 1 ]; then
  log "[error] TCPStore :${MASTER_PORT} ${TCPSTORE_GATE_WAIT:-300}s 内未就绪, 中止" >&2
  diagnose
  exit 1
fi

# ---- 3. 错峰启动 3 个 worker (rank1=02, rank2=04, rank3=03) ----
# r12: 错峰启动 — 每启一个 worker 间隔 B12X_JIT_STAGGER 秒, 防多 worker 并行撞 b12x JIT 竞态
#   (8/20 SEV1 事故根因的原始预防意图保留); 不再等 B12X_MXFP4 日志(死锁), 仅限错峰时序。
# KEYFIX(2026-08-24, r12+keyfix): worker ssh 启动命令须带 VLLM_API_KEY=${VLLM_API_KEY}
#   (worker 叶子脚本 ${VLLM_API_KEY:?} 强依赖; r11 有此传递, r12 初版改写 step 3 时遗漏,
#    导致 worker 启动即 "VLLM_API_KEY is not set" — 本次已恢复)
# 方案A: per-rank 邻接口 HCA (各 rank 只暴露与两 RING 邻居直连的口)
declare -A RANK_HCA=(
  [1]="rocep1s0f1,rocep1s0f0"   # 02: 136->01, 192.168.5.186->04
  [2]="rocep1s0f1,rocep1s0f0"   # 04: 138->03, 192.168.5.186->02
  [3]="rocep1s0f1,rocep1s0f0"   # 03: 138->04, 140->01
)
for r in 1 2 3; do
  host="${RANK_HOST[$r]%%:*}"
  ip="${RANK_HOST[$r]##*:}"
  log "[step 3/4] 启动 rank${r} worker (${host}, ${ip}, HCA=${RANK_HCA[$r]})..."
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
    "VLLM_API_KEY=${VLLM_API_KEY} NODE_RANK=${r} VLLM_HOST_IP=${ip} NCCL_IB_HCA=${RANK_HCA[$r]} nohup bash $PROD/scripts/start_tp4_worker.sh > $HOME/start_tp4_rank${r}.log 2>&1 &"
  log "[i] rank${r} 启动命令已下发 (${host})"
  # B12X JIT 错峰: 除最后一个 worker 外, 间隔 ${B12X_JIT_STAGGER}s
  if [ "$r" -lt 3 ]; then
    log "[i] 错峰等待 ${B12X_JIT_STAGGER:-20}s (防多 worker 并行撞 b12x JIT 竞态)..."
    sleep "${B12X_JIT_STAGGER:-20}"
  fi
done

# ---- 4. worker 存活与对端门禁 (120s) ----
log "[step 4/4] 等待 worker 容器出现 (对端门禁 120s)..."
for r in 1 2 3; do
  host="${RANK_HOST[$r]%%:*}"
  OK=0
  for ((w=0; w<24; w++)); do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "docker ps --format '{{.Names}}' | grep -q '^vllm-tp4-rank${r}$'"; then
      OK=1; log "[ok] rank${r}(${host}) 容器已起 (第 $((w+1)) 次, ${w}*5s)"
      break
    fi
    sleep 5
  done
  if [ "$OK" -ne 1 ]; then
    log "[error] rank${r}(${host}) 容器未在 120s 内出现, 中止" >&2
    diagnose
    exit 1
  fi
done

log "[ok] 四机 TP4 容器均已启动. head 日志: $HEAD_LOG"
log "[i] 等待全部就绪: 轮询 head 'Application startup complete' (最长 15min)"
for i in $(seq 1 180); do
  if docker logs vllm-tp4-rank0 2>&1 | grep -q "Application startup complete"; then
    log "[ok] TP4 READY (${i}0s)"
    exit 0
  fi
  sleep 10
done
log "[warn] 未就绪, 观察 docker logs vllm-tp4-rank0" >&2
exit 1