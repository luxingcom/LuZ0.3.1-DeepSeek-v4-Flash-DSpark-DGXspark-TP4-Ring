#!/bin/bash
# =============================================================
# SCRIPT: start_tp4_cluster.sh
# VERSION: v1.5-r12
# USAGE: bash start_tp4_cluster.sh [--help]
# ROLE: TP4 四机编排 (head-first, 幂等) — 在 node01 运行
# HOST: node01
# DOCS: file://<INSTALL_DIR>/docs/scripts/REFERENCE.md
# DOCS: file://<INSTALL_DIR>/docs/ops/maintenance-plans.md
# DOCS: file://<INSTALL_DIR>/docs/runbook-tp4-v1.5-2026-08-12.md
# EXITCODES: 0=成功 1=业务失败 2=用法错误 130=被signal
# CHANGE: 改脚本须 check_vllm_script.sh 通过 + .bak-<tag> 留档 + 更新 REFERENCE.md
# FIX(2026-08-24, r12): step 2.5 B12X 门禁死锁修复 — 原门禁等 head 日志 "Using 'B12X_MXFP4'"
#   才启 workers, 但 head 引擎核心初始化需 4 rank NCCL 域 → 冷启动互等 300s 超时。
#   改为 TCPStore:25999 就绪(step 2 信号) + B12X_JIT_STAGGER 固定错峰启 workers
#   (保留防多 worker 并行撞 b12x JIT 竞态意图)。留档 .bak-b12xgate-fix-20260824。
# =============================================================
# R11 KEY_PARAMS: seqs=6 | util=0.65 | capture=1..64(max64) | PSR: NCCL=8-9 EngineCore=15-19
# rank 映射: 01=0(186) 02=1(187) 04=2(189) 03=3(188) | MASTER_PORT=25999(管理网)
# =============================================================
set -uo pipefail

# -h/--help (无位置参数)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
start_tp4_cluster.sh — TP4 四机编排 (head-first, 幂等, node01)
  R11 关键参数: seqs=6 | util=0.65 | capture=1..64(max64) | PSR: NCCL=8-9 EngineCore=15-19
  用法: bash start_tp4_cluster.sh [--help]
参考: <INSTALL_DIR>/docs/scripts/REFERENCE.md（脚本↔文档索引）| <INSTALL_DIR>/docs/README.md（入口）
EOF
  exit 0
fi

HEAD_HOST="node01"
MASTER_ADDR="<NODE_IP>"
MASTER_PORT=25999
API_PORT=8001
API_KEY="${VLLM_API_KEY:?VLLM_API_KEY is not set}"
MAX_WAIT_S=600
# r12: 门禁语义替换 — TCPSTORE_GATE_WAIT=step 2.5 TCPStore 复核门禁上限;
# B12X_JIT_STAGGER=错峰启 worker 间隔, 保留"防多 worker 并行撞 b12x JIT 竞态"意图 (8/20 SEV1 复盘)
TCPSTORE_GATE_WAIT="${TCPSTORE_GATE_WAIT:-300}"
B12X_JIT_STAGGER="${B12X_JIT_STAGGER:-20}"
HEAD_LOG="$HOME/start_tp4_cluster.log"
PROD=<INSTALL_DIR>

# rank->host 映射 (环序)
declare -A RANK_HOST=(
  [1]="node01:<NODE_IP>"
  [2]="node01:<NODE_IP>"
  [3]="node01:<NODE_IP>"
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
[ "$(hostname)" = "node01" ] || [ "$(hostname)" = "spark-05cd" ] || {
  echo "ERROR: 本脚本仅可在 head(node01) 上运行" >&2
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
if ! HOME=/home/<USER> "$PROD/scripts/check_vllm_script.sh" "$PROD/scripts/start_tp4_head.sh"; then
  log "[error] head 脚本自检失败" >&2; exit 1
fi
for r in 1 2 3; do
  host="${RANK_HOST[$r]%%:*}"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "HOME=/home/<USER> $PROD/scripts/check_vllm_script.sh $PROD/scripts/start_tp4_worker.sh"; then
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
  [1]="rocep1s0f1,rocep1s0f0"   # 02: 136->01, <NODE_IP>->04
  [2]="rocep1s0f1,rocep1s0f0"   # 04: 138->03, <NODE_IP>->02
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
