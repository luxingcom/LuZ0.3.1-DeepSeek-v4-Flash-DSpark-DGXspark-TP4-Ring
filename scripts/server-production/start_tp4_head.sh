#!/bin/bash
# =============================================================
# SCRIPT: start_tp4_head.sh
# VERSION: v1.5-r12 (integrated, SRE 修正)
#  内嵌生产加固 (2026-08-26):
#   · incident-clone-roce-prevention-2026-08-25.md 检查点1-5 → 启动前强制预检
#   · 强制序列: probe_gid_index.sh → preflight_roce_gid.sh(--expect-index) → gid_index_env.sh
#   · 写死 -e 'NCCL_IB_GID_INDEX=3' → 经 env 决策后变量注入 (克隆严禁照抄生产 index)
#   · exit2/3(No-Go) 在 docker run 前 fail-fast, 提示克隆不一致 + 首次部署重建流程
#   · 检查点引用: 检查点1(布局/空洞) 检查点2(index适配) 检查点3(fail-fast) 检查点4(日志) 检查点5(carrier flap)
# USAGE: bash start_tp4_head.sh [--help]  (systemd 由 monitor_tp4_head.sh 调用)
# ROLE: head(rank0) 启动 — dgxspark01 (192.168.5.186)
# HOST: dgxspark01 | spark-05cd  (dgxspark01~04 为 SSH 别名; 真机名 head=spark-05cd)
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/server-maintenance-handbook.md
# DOCS: file:///opt/aicad-prod/docs/runbook-tp4-v1.5-2026-08-12.md
# EXITCODES: 0=成功 1=业务失败 2=用法错误 3=GID布局No-Go(预检阻断) 4=env失配 130=被signal
# CHANGE: 改脚本须 check_vllm_script.sh 通过 + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
# R12 KEY_PARAMS: --max-num-seqs 12 | --gpu-memory-utilization 0.78 |
#                 --max-cudagraph-capture-size 96 (capture-sizes 1..96) |
#                 PSR: NCCL=8-9 (isolcpus) | EngineCore=15-19 (shim v8)
# CONC(R12, 2026-08-26): 入口信号量代理 concurrency_proxy.py 占 0.0.0.0:8001
#                  → 转发 127.0.0.1:8002 (本容器 vLLM 新端口); --max-num-batched-tokens 8192 (全 rank 一致)
#                  → 在飞≤12 (与 --max-num-seqs 12 对齐, 消除 > shm 环6 结构性不匹配)
# TP4 环网: 01=rank0(186) 02=rank1(187) 04=rank2(189) 03=rank3(188)
# 控制面: MASTER_ADDR/VLLM_HOST_IP=192.168.5.186 MASTER_PORT=25999
# 容器: vllm-tp4-rank0 --restart no
# =============================================================
set -euo pipefail

# =============================================================
# 生产加固注入路径 (检查点1-5) — 可用 env 覆写, 默认 /opt/aicad-prod/scripts
# =============================================================
TP4_HARDEN_DIR="${TP4_HARDEN_DIR:-/opt/aicad-prod/scripts}"
PROBE_BIN="${TP4_PROBE_BIN:-${TP4_HARDEN_DIR}/probe_gid_index.sh}"
PREFLIGHT_BIN="${TP4_PREFLIGHT_BIN:-${TP4_HARDEN_DIR}/preflight_roce_gid.sh}"
GID_ENV_BIN="${TP4_GID_ENV_BIN:-${TP4_HARDEN_DIR}/gid_index_env.sh}"
# 本机建议 index (由 probe 实测填充; 供 preflight --expect-index / cluster 汇总)
GID_SUGGEST_INDEX=""

# -h/--help (无位置参数, 不干扰 systemd/monitor 调用)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
start_tp4_head.sh — TP4 head(rank0) 启动 (dgxspark01 / spark-05cd)
  R12 关键参数: seqs=12 | util=0.78 | capture=1..96(max96) | PSR: NCCL=8-9 EngineCore=15-19
  内嵌预检(2026-08-26): probe → preflight(--expect-index) → gid_index_env 注入
  用法: systemd(monitor_tp4_head.sh) 调用, 或手动 bash start_tp4_head.sh [--help]
参考: /opt/aicad-prod/docs/scripts/REFERENCE.md（脚本↔文档索引）| /opt/aicad-prod/docs/README.md（入口）
EOF
  exit 0
fi

# QA-fix HR: head hostname 守卫对齐 cluster L90 (dgxspark01|spark-05cd)。真实机名 head=spark-05cd
#   (192.168.5.60), dgxspark01~04 仅为 SSH 别名; 只认 dgxspark01 会致真机 exit1 → cluster step1 起不来。
([ "$(hostname)" = "dgxspark01" ] || [ "$(hostname)" = "spark-05cd" ]) && [ "${NODE_RANK:-0}" = "0" ] || {
  echo "ERROR: 本脚本仅可在 head 机 (dgxspark01 / spark-05cd, NODE_RANK=0) 运行" >&2
  exit 1
}

IMG="192.168.5.187:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"
NAME="vllm-tp4-rank0"
NEW_SERVED_NAME="deepseek-v4-flash-0731"
# CONC(R12): 对外入口 8001 已由 concurrency_proxy.py 占用; 本容器 vLLM 监听 8002
PORT="8002"
MASTER_ADDR="192.168.5.186"
MASTER_PORT="25999"
NODE_RANK=0

# ---- 回滚锚点 ----
docker inspect "$NAME" > "/opt/aicad-prod/backup/rollback_tp4-rank0.json" 2>/dev/null || true

SERVE_CMD="rm -rf /tmp/plugin_a1_install; cp -r /opt/aicad-prod/nvfp4/plugin_a1 /tmp/plugin_a1_install 2>/dev/null; pip install --no-deps -q /tmp/plugin_a1_install >/dev/null 2>&1; vllm serve\
  --model /models\
  --served-model-name deepseek-v4-flash-0731\
  --kv-cache-dtype nvfp4_ds_mla\
  --max-model-len 600000\
  --max-num-seqs 12\
  --max-num-batched-tokens 8192\
  --long-prefill-token-threshold 4096\
  --scheduling-policy priority\
  --gpu-memory-utilization 0.78\
  --enable-auto-tool-choice\
  --tool-call-parser deepseek_v4\
  --reasoning-parser deepseek_v4\
  --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'\\
  --moe-backend flashinfer_b12x\
  --distributed-executor-backend mp\
  --distributed-timeout-seconds 300\
  --port 8002\
  --api-key \"${VLLM_API_KEY:?VLLM_API_KEY is not set}\"\
  --enable-flashinfer-autotune\
  --max-cudagraph-capture-size 96\
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64 72 80 88 96\
  --enable-prefix-caching\
  --enable-prompt-tokens-details\
  --generation-config vllm\
  --tensor-parallel-size 4 --nnodes 4 --node-rank 0\
  --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}"
# 掩码 api-key 值, 防密钥落入操作日志 (P1-2 / 对齐上游 PR#89 脱敏, 2026-08-24)
MASKED_SERVE_CMD=$(printf '%s' "$SERVE_CMD" | sed -E 's/(--api-key[= ]+)[^ ]+/\1********/g' || printf '%s' "$SERVE_CMD")
echo "[i] serve 命令: $MASKED_SERVE_CMD"

# =============================================================
# [内嵌] GID index 环境决策 (检查点2) — 在 ENV_ARGS 组装之前 source
#   gid_index_env.sh 调 probe 实测 → 设 NCCL_IB_GID_INDEX (动态-1 或实际值)
#   source 失败不阻断 (降级 -1), 见函数内告警
# =============================================================
if [ -f "$GID_ENV_BIN" ]; then
  # gid_index_env.sh 内含 return, 需在函数/source 上下文外执行
  # shellcheck disable=SC1090
  source "$GID_ENV_BIN" || true
else
  echo "[preflight-WARN] 未找到 gid_index_env.sh ($GID_ENV_BIN), 降级 NCCL_IB_GID_INDEX=-1 (动态) (检查点2)" >&2
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:--1}"
  export GID_INDEX_DECIDED="manual-fallback"
fi
# 记录建议值供后续 preflight --expect-index (probe 已输出; 此处沿用决策值)
GID_SUGGEST_INDEX="${NCCL_IB_GID_INDEX:--1}"
echo "[preflight] 决策 NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} (decided=${GID_INDEX_DECIDED:-unknown})"

ENV_ARGS=(
  -e 'CUDA_DEVICE_ORDER=PCI_BUS_ID'
  -e 'CUDA_VISIBLE_DEVICES=0'
  -e 'DG_JIT_NVCC_COMPILER=/tmp/env-e-build/nvcc_wrapper.py'
  -e 'DG_JIT_USE_NVRTC=0'
  -e 'DSPARK_SLOT_CLAMP=1'
  -e 'FLASHINFER_DISABLE_VERSION_CHECK=1'
  -e 'GLOO_SOCKET_IFNAME=enP7s7'
  -e 'HEADLESS='
  -e 'HF_HOME=/cache/huggingface'
  -e 'HF_HUB_OFFLINE=1'
  -e 'LANG=C.UTF-8'
  -e 'LC_ALL=C.UTF-8'
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu'
  -e 'LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2'
  -e "MASTER_ADDR=${MASTER_ADDR}"
  -e "MASTER_PORT=${MASTER_PORT}"
  -e 'NCCL_ALGO=RING'
  -e 'NCCL_CROSS_NIC=1'
  -e 'NCCL_DEBUG=INFO'
  -e 'NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log'
  # === [内嵌 2026-08-26] 移除写死 index=3, 改经 gid_index_env 决策的动态值 (检查点2铁律) ===
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
  -e 'NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1'   # 4 twin 口全暴露
  -e 'NCCL_IB_TIMEOUT=1000'
  -e 'NCCL_IB_RETRY_CNT=7'
  -e 'NCCL_IB_TOS=46'
  -e 'NCCL_CUMEM_HOST_ENABLE=0'
  -e 'NCCL_IGNORE_CPU_AFFINITY=1'
  # === NCCL 延迟优化 T1aM4+MAX_CH16 (2026-08-16 生产生效) ===
  # 来源: deliverables/engineering-assurance/nccl-p0-scan-results-2026-08-16.md
  #       + nccl-maxch16-e2e-verification-2026-08-16.md
  # MIN_NCHANNELS 2->4: 通道下限提升; PROTO=Simple: 368KB档去LL128开销
  # BUFFSIZE 4M->8M: 加深小消息管道; MAX_NCHANNELS=16: 修正64通道协议误判
  #   (368KB/16=23KB/通道选对协议, 368KB allreduce 923->173us, -81%)
  # 端到端: 32K PR+5.5%/DE+6.5%/TTFT-4%; 131K PR+21.7%/TTFT-17.2%
  -e 'NCCL_MIN_NCHANNELS=4'
  -e 'NCCL_TUNER_THRESHOLD=40960'
  -e 'NCCL_BUFFSIZE=8388608'
  -e 'NCCL_MAX_NCHANNELS=4'
  -e 'NCCL_SET_THREAD_NAME=1'
  -e 'NCCL_NET=IB'
  -e 'NCCL_IB_SUBNET_AWARE_ROUTING=1'
  -e 'NCCL_NET_PLUGIN=none'
  -e 'NCCL_IB_MERGE_NICS=0'
  -e 'NCCL_SOCKET_IFNAME=enP7s7'
  -e "NODE_RANK=${NODE_RANK}"
  -e 'NVIDIA_VISIBLE_DEVICES=all'
  -e 'PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
  -e 'PIP_DISABLE_PIP_VERSION_CHECK=1'
  -e "PORT=${PORT}"
  -e 'PYTHONDONTWRITEBYTECODE=1'
  -e 'PYTHONUNBUFFERED=1'
  -e 'PYTHONFAULTHANDLER=1'
  -e 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800'
  -e 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
  -e 'SERVED_MODEL_NAME=deepseek-v4-flash-0731'
  -e 'TILELANG_CLEANUP_TEMP_FILES=1'
  -e 'TORCH_CUDA_ARCH_LIST=12.1a'
  -e 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1'
  -e 'VLLM_DISABLE_PYNCCL=1'
  -e 'VLLM_ENGINE_READY_TIMEOUT_S=600'
  -e "VLLM_HOST_IP=${MASTER_ADDR}"
  -e "VLLM_DP_MASTER_IP=${MASTER_ADDR}"
  -e 'VLLM_USE_B12X_MOE=1'
  -e 'VLLM_USE_BREAKABLE_CUDAGRAPH=1'
  -e 'VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096'
  -e 'VLLM_USE_FLASHINFER_SAMPLER=1'
  -e 'VLLM_DSPARK_LOCAL_ARGMAX=1'
  -e 'PYTHONPATH=/opt/aicad-prod/nvfp4/kernel1:/opt/aicad-prod/nvfp4/kernel2'
  -e 'VLLM_TRITON_MLA_SPARSE=1'
  # === W4A4 arm (ws-dedup L3, 2026-08-23) ===
  -e 'VLLM_MOE_W4A4=2'
  -e 'VLLM_MOE_W4A4_MIN_M=3072'
  -e 'VLLM_MOE_W4A4_CG=1'
  -e 'VLLM_B12X_SHARED_WRAPPER=1'
)

if ! /opt/aicad-prod/scripts/check_vllm_script.sh "$0"; then
  echo "[i] 前置自检失败, 终止启动 (修复后重试)" >&2
  exit 1
fi

# =============================================================
# [内嵌] P0 硬门: RoCE GID 布局预检 fail-fast (检查点1+3) — 在 docker run 之前
#   多层分层: T0 carrier flap / L1 QP 断 / L2 GID 空洞 (connect.cc:317-321 写死踩空)
#   本门阻断 L2 重启卡死 (shm_broadcast hang) 的最大诱因: 空洞/子网不一致/期望 index 失配
#   任一异常 ⇒ exit 3 (No-Go), 绝不创建容器后才诊断 (RCA 最大教训)
# =============================================================
_preflight_fail() {
  echo "------------------------------------------------------------" >&2
  echo "[preflight-FAIL] RoCE GID 布局 No-Go — 禁止带病启动 (检查点3 fail-fast)" >&2
  echo "[preflight-FAIL] 检测到克隆环境 RoCE/GID 布局与生产不一致 (空洞/子网/口名/HCA 失配)" >&2
  echo "------------------------------------------------------------" >&2
  echo "[rebuild-guidance] 遇到【克隆环境不一致】→ 请执行【首次部署重建】后再启动:" >&2
  echo "  1) dump 全机 GID 表:  for p in /sys/class/infiniband/*/ports/*/gids/[0-9]*; do ... done  (检查点1a)" >&2
  echo "  2) 修复 GID 空洞:     rerun fix_gid_holes (重建 IPv4 RoCEv2 GID, index 对齐 RoCE 网段)" >&2
  echo "  3) 重建网络配置:      重算 HCA/PEER_HCA 到克隆实际口集合 (rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1)" >&2
  echo "  4) 重跑 preflight:    bash ${PREFLIGHT_BIN} --expect-index ${GID_SUGGEST_INDEX:-N}  须 exit0" >&2
  echo "  5) 再启动本脚本:      bash $0  → 通过后才 docker run" >&2
  echo "  6) 验证:              NCCL_DEBUG_FILE 落持久卷 / dmesg 与 vllm.log 核对 index 实际生效" >&2
  echo "------------------------------------------------------------" >&2
}

# 若依赖脚本缺失 → 无法做预检不是静默放过; 按"避免静默故障"原则显式告警并阻断
if [ ! -x "$PREFLIGHT_BIN" ] || [ ! -x "$PROBE_BIN" ]; then
  echo "[preflight-WARN] 预检依赖缺失: preflight=${PREFLIGHT_BIN} probe=${PROBE_BIN}" >&2
  echo "[preflight-FAIL] 缺失 GID 预检脚本无法保证安全启动, fail-fast (检查点3)" >&2
  _preflight_fail
  exit 3
fi

# --- 1) probe: 显式探测建议 index 并打印到操作日志 (检查点2) ---
echo "[preflight] 探测本机建议 GID index (probe_gid_index.sh)..."
if PROBE_OUT=$(bash "$PROBE_BIN" 2>&1); then
  echo "$PROBE_OUT" | sed 's/^/[probe]   /'
else
  echo "[preflight-WARN] probe 执行异常, 沿用决策值 ${NCCL_IB_GID_INDEX} (降级路径)" >&2
fi
PROBE_SUGG=$(echo "$PROBE_OUT" | grep -oE '建议 NCCL_IB_GID_INDEX=[^ ]+' | head -1 | cut -d= -f2)
[ -n "$PROBE_SUGG" ] && GID_SUGGEST_INDEX="$PROBE_SUGG"
# 决策值若为 -1 且 probe 给数字, 仍以 -1(动态)为准 — 已在 env 决策层收敛

# --- 2) preflight: 判空洞/子网/期望 index (检查点1+检查点3) ---
#     --expect-index <决策值>: index 为数字且决策非 -1 时严格校验; -1(动态) 时仍跑布局/hole/子网
EXPECT_ARG=""
if [ "$GID_SUGGEST_INDEX" != "-1" ] && [ "$GID_SUGGEST_INDEX" != "REMOVE" ]; then
  EXPECT_ARG="--expect-index ${GID_SUGGEST_INDEX}"
fi
echo "[preflight] 执行 GID 布局预检: ${PREFLIGHT_BIN} --degrade ${EXPECT_ARG}"
# 注意: --peers 交叉核对由 start_tp4_cluster.sh 在协调层统一做; head 独立跑本机布局门
if ! bash "$PREFLIGHT_BIN" --degrade ${EXPECT_ARG}; then
  RC=$?
  echo "[preflight-WARN] preflight 返回 $RC (期望 0=OK / 3=No-Go)"
  if [ "$RC" = "3" ] || [ "$RC" = "4" ]; then
    _preflight_fail
    exit 3
  fi
  # 其它非0 (1=运行错误/2=用法/工具缺失): 不静默, 降级但告警
  echo "[preflight-WARN] preflight 异常返回 $RC, 不阻断但记录 (避免静默故障)" >&2
fi

docker rm -f "$NAME" 2>/dev/null || true
mkdir -p ~/vllm-logs /tmp/vllm-envc-cache

BINDS=(
  -v /opt/aicad-prod/models/deepseek-v4-flash-0731:/models:ro
  -v /opt/aicad-prod/envs/nvcc_wrapper.py:/tmp/env-e-build/nvcc_wrapper.py:ro
  -v /opt/aicad-prod/envs/vllm-envc-cache:/cache/huggingface
  -v /opt/aicad-prod/nvfp4:/opt/aicad-prod/nvfp4:ro
  -v "$HOME/vllm-cache:/root/.cache/vllm:rw"
  -v "$HOME/tilelang-cache:/root/.tilelang/cache:rw"
  -v "$HOME/b12x-cache:/root/.cache/b12x:rw"
  -v "$HOME/patch-v026/model_executor/kernels/mhc/tilelang.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro"
  -v /opt/aicad-prod/lib/libncclpin.so:/opt/libncclpin.so:ro
  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro
  -v /opt/aicad-prod/overlay-wsdedup/flashinfer_b12x_moe.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro
  # === FI 0.6.16 overlay (fi016 窗口注入; 2026-08-23 03:01 被 w4a4-ext 恢复误覆盖, luz031 补回) ===
  -v /opt/aicad-prod/nvfp4/flashinfer-0.6.16/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer:ro
  -v /opt/aicad-prod/overlay-mask/api_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py:ro
  # === shm_broadcast 环容量 6→24 overlay (CONC 12, 消除 max-num-seqs=12 > 环6 结构性不匹配) ===
  -v /opt/aicad-prod/overlay-shm/parallel_state.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py:ro
  -v "$HOME/flashinfer-cache:/root/.cache/flashinfer:rw"
)

docker run -d --name "$NAME" \
  --restart no \
  --network host --ipc=host --privileged --gpus all \
  --cpuset-cpus=1-19 \
  --memory 112g --memory-swap 112g \
  --log-opt max-size=100m --log-opt max-file=3 \
  --shm-size=64gb --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --health-cmd "curl -sf -o /dev/null -m 5 http://127.0.0.1:8002/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 5 --health-start-period 900s \
  "${BINDS[@]}" \
  -v ~/vllm-logs:/var/log/vllm \
  "${ENV_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMG" -lc "$SERVE_CMD"

echo "[i] 容器已启动: ${NAME}"
echo "[i] 等待就绪 (≤15min cold start): 轮询 docker logs 'Application startup complete'"
if [ "${NO_WAIT:-0}" = "1" ]; then echo "[i] NO_WAIT 模式: 容器已启动, 交回 monitor 跟随"; exit 0; fi
for i in $(seq 1 180); do
  if docker logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "[ok] READY (${i}0s)"
    # [warmup] 容器就绪后优先预捕获 CUDA graph（conc=6,8,12），避免运行期即时捕获停顿
    # [warmup-guard] 仅 head(rank0) 触发（误被拷贝到 worker 时 NODE_RANK != 0 直接跳过）
    if [ "${NODE_RANK:-}" != "0" ]; then
      echo "[warmup] 非 head(rank=${NODE_RANK:-?})，跳过预热"; exit 0
    fi
    if [ -f /tmp/bench_v2_mc.py ] && [ -n "$(cat /tmp/key_test.txt 2>/dev/null)" ]; then
      KEY="$(cat /tmp/key_test.txt)"
      WUOUT=/tmp/warmup_$(date +%Y%m%d_%H%M%S)
      setsid nohup /tmp/benchy/bin/python /tmp/bench_v2_mc.py         --endpoint http://127.0.0.1:8001/v1 --key "$KEY"         --run-type both --concurrency 6,8,12 --warmup-conc 6,8,12 --warmup-only         --out "${WUOUT}" > "${WUOUT}.log" 2>&1 < /dev/null &
      echo "[warmup] 已启动预捕获 PID=$! (conc=6,8,12) log=${WUOUT}.log"
    else
      echo "[warmup] 跳过：缺 /tmp/bench_v2_mc.py 或 key_test.txt"
    fi
    exit 0
  fi
  sleep 10
done
echo "[warn] 未就绪（可能仍在加载权重）; 观察 docker logs ${NAME}" >&2
exit 1