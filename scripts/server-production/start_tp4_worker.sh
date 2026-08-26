#!/bin/bash
# =============================================================
# SCRIPT: start_tp4_worker.sh
# VERSION: v1.5-r12 (integrated)
#  内嵌生产加固 (2026-08-26):
#   · incident-clone-roce-prevention-2026-08-25.md 检查点1-5 → 启动前强制预检
#   · 强制序列: probe_gid_index.sh → preflight_roce_gid.sh(--expect-index) → gid_index_env.sh
#   · 写死 -e 'NCCL_IB_GID_INDEX=3' → 经 env 决策后变量注入 (克隆严禁照抄生产 index)
#   · exit2/3(No-Go) 在 docker run 前 fail-fast, 提示克隆不一致 + 首次部署重建流程
#   · 检查点引用: 检查点1(布局/空洞) 检查点2(index适配) 检查点3(fail-fast) 检查点4(日志) 检查点5(carrier flap)
# USAGE: NODE_RANK=N VLLM_HOST_IP=<ip> bash start_tp4_worker.sh [--help]
# ROLE: worker(rank1/2/3) 启动 (02/03/04)
# HOST: dgxspark02/03/04
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/server-maintenance-handbook.md
# DOCS: file:///opt/aicad-prod/docs/rollback-anchors-2026-08-12.md
# EXITCODES: 0=成功 1=业务失败 2=用法错误 3=GID布局No-Go(预检阻断) 4=env失配(口名/HCA) 130=被signal
# CHANGE: 改脚本须 check_vllm_script.sh 通过 + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
# R12 KEY_PARAMS: --max-num-seqs 12 | --gpu-memory-utilization 0.78 |
#                 --max-cudagraph-capture-size 96 (capture-sizes 1..96) |
#                 PSR: NCCL=8-9 (isolcpus) | EngineCore=15-19 (shim v8)
# CONC(R12, 2026-08-26): --max-num-batched-tokens 8192 (与 head 全 rank 一致, 门禁 B1 对齐);
#                 worker --port 8001 保留 (MP 模式 worker 不对外服务, 端口仅内部绑定)
# 容器名自动: vllm-tp4-rank${NODE_RANK} | 模型源: 03/04 本地 .local-backup(临时) / NFS(恢复后)
# =============================================================
set -euo pipefail

# =============================================================
# 生产加固注入路径 (检查点1-5) — 可用 env 覆写, 默认 /opt/aicad-prod/scripts
# =============================================================
TP4_HARDEN_DIR="${TP4_HARDEN_DIR:-/opt/aicad-prod/scripts}"
PROBE_BIN="${TP4_PROBE_BIN:-${TP4_HARDEN_DIR}/probe_gid_index.sh}"
PREFLIGHT_BIN="${TP4_PREFLIGHT_BIN:-${TP4_HARDEN_DIR}/preflight_roce_gid.sh}"
GID_ENV_BIN="${TP4_GID_ENV_BIN:-${TP4_HARDEN_DIR}/gid_index_env.sh}"
GID_SUGGEST_INDEX=""

# -h/--help (无位置参数, 不干扰 monitor 调用)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
start_tp4_worker.sh — TP4 worker(rank1/2/3) 启动 (02/03/04)
  R12 关键参数: seqs=12 | util=0.82 | capture=1..96 | PSR: NCCL=8-9 EngineCore=15-19
  内嵌预检(2026-08-26): probe → preflight(--expect-index) → gid_index_env 注入
  用法: NODE_RANK=N VLLM_HOST_IP=<ip> bash start_tp4_worker.sh [--help]
参考: /opt/aicad-prod/docs/scripts/REFERENCE.md（脚本↔文档索引）| /opt/aicad-prod/docs/README.md（入口）
EOF
  exit 0
fi

# 必填参数
NODE_RANK="${NODE_RANK:-}"
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
[ -n "$NODE_RANK" ] && [ -n "$VLLM_HOST_IP" ] || {
  echo "ERROR: 需指定 NODE_RANK 与 VLLM_HOST_IP (如 NODE_RANK=1 VLLM_HOST_IP=192.168.5.187)" >&2
  exit 1
}

IMG="192.168.5.187:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"
NAME="vllm-tp4-rank${NODE_RANK}"
PORT="8001"
MASTER_ADDR="192.168.5.186"
MASTER_PORT="25999"

# ---- 回滚锚点 ----
docker inspect "$NAME" > "/opt/aicad-prod/backup/rollback_tp4-rank${NODE_RANK}.json" 2>/dev/null || true

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
  --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'\
  --moe-backend flashinfer_b12x\
  --distributed-executor-backend mp\
  --distributed-timeout-seconds 300\
  --port 8001\
  --api-key \"${VLLM_API_KEY:?VLLM_API_KEY is not set}\"\
  --enable-flashinfer-autotune\
  --max-cudagraph-capture-size 96\
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64 72 80 88 96\
  --enable-prefix-caching\
  --enable-prompt-tokens-details\
  --generation-config vllm\
  --tensor-parallel-size 4 --nnodes 4 --node-rank ${NODE_RANK}\
  --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}"
# 掩码 api-key 值, 防密钥落入操作日志 (P1-2 / 对齐上游 PR#89 脱敏, 2026-08-24)
MASKED_SERVE_CMD=$(printf '%s' "$SERVE_CMD" | sed -E 's/(--api-key[= ]+)[^ ]+/\1********/g' || printf '%s' "$SERVE_CMD")
echo "[i] serve 命令: $MASKED_SERVE_CMD"

PEER_HCA=""
case "${NODE_RANK}" in
  1) PEER_HCA="0=rocep1s0f1,roceP2p1s0f1;2=rocep1s0f0,roceP2p1s0f0" ;;
  2) PEER_HCA="1=rocep1s0f0,roceP2p1s0f0;3=rocep1s0f1,roceP2p1s0f1" ;;
  3) PEER_HCA="0=rocep1s0f0,roceP2p1s0f0;2=rocep1s0f1,roceP2p1s0f1" ;;
esac

# =============================================================
# [内嵌] GID index 环境决策 (检查点2) — 在 ENV_ARGS 组装之前 source
# =============================================================
if [ -f "$GID_ENV_BIN" ]; then
  # shellcheck disable=SC1090
  source "$GID_ENV_BIN" || true
else
  echo "[preflight-WARN] 未找到 gid_index_env.sh ($GID_ENV_BIN), 降级 NCCL_IB_GID_INDEX=-1 (动态) (检查点2)" >&2
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:--1}"
  export GID_INDEX_DECIDED="manual-fallback"
fi
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
  -e 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800'
  -e 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
  -e 'SERVED_MODEL_NAME=deepseek-v4-flash-0731'
  -e 'TILELANG_CLEANUP_TEMP_FILES=1'
  -e 'TORCH_CUDA_ARCH_LIST=12.1a'
  -e 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1'
  -e 'VLLM_DISABLE_PYNCCL=1'
  -e 'VLLM_ENGINE_READY_TIMEOUT_S=600'
  -e "VLLM_HOST_IP=${VLLM_HOST_IP}"
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

# =============================================================
# D1 模型就绪门禁 (NFS/权重依赖): ≤300s, 每 5s 检查
#   检查点: config.json 可见; 若模型软链落 NFS(/mnt/models-nfs) 须确认挂载
#   超时返回 2 (可重试类) — monitor_tp4_worker.sh 据此做指数退避
# =============================================================
MODEL_DIR="/opt/aicad-prod/models/deepseek-v4-flash-0731"
GATE_OK=0
for _g in $(seq 1 60); do
  if [ -f "${MODEL_DIR}/config.json" ]; then
    MODEL_REAL=$(readlink -f "$MODEL_DIR" 2>/dev/null || echo "$MODEL_DIR")
    case "$MODEL_REAL" in
      /mnt/models-nfs*)
        if mountpoint -q /mnt/models-nfs 2>/dev/null; then GATE_OK=1; fi ;;
      *) GATE_OK=1 ;;
    esac
    [ "$GATE_OK" = "1" ] && break
  fi
  sleep 5
done
if [ "$GATE_OK" != "1" ]; then
  MPT="NFS 未挂载"
  mountpoint -q /mnt/models-nfs 2>/dev/null && MPT="NFS 已挂载"
  echo "[fail] 模型就绪门禁超时(300s): $MODEL_DIR 无可用 config.json ($MPT)" >&2
  ls -la "$MODEL_DIR" 2>&1 | head -3 >&2
  exit 2
fi
echo "[i] 模型就绪: $(readlink -f "$MODEL_DIR" 2>/dev/null || echo "$MODEL_DIR")"

if ! /opt/aicad-prod/scripts/check_vllm_script.sh "$0"; then
  echo "[i] 前置自检失败, 终止启动 (修复后重试)" >&2
  exit 1
fi

# =============================================================
# [内嵌] worker 口名/HCA 一致性预检 (检查点1a) — docker run 之前 fail-fast
#   worker 的 PEER_HCA 依赖口名布局; 口名非预期 (非 rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1)
#   → 直接 No-Go (exit 4), 避免克隆口名/HCA 失配带病启动 (检查点1a NO-GO 判据)
# =============================================================
EXPECTED_PORTS="rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1"
HCA_OK=1
HCA_FOUND=0
# QA-fix I5: 原实现遍历 /sys/class/infiniband/*/ 取到的是 IB 设备名(如 mlx5_0),
#   却用 netdev 口名(rocep1s0f0...)比对 → 恒不匹配 → 真实机正常 worker 也误 exit4。
#   现改为遍历 /sys/class/net/* 枚举实际 RoCE 网卡口名 (netdev), 据此判定 HCA/口名布局。
# FIX(2026-08-26, QA check1a revert): I5 改错方向 — 本集群 DGX 的 IB 设备名(/sys/class/infiniband/*)
#   恰为 RoCE 口名 roceP2p1s0f0/rocep1s0f0/rocep1s0f1/roceP2p1s0f1 (与 EXPECTED_PORTS 完全一致),
#   而 /sys/class/net/* 下为 en* netdev (enP2p1s0f0np0 等) 无 roce* → I5 后真实机恒 HCA_FOUND=0 误 exit4。
#   改回遍历 /sys/class/infiniband/* (与 preflight_roce_gid.sh GID_SYS 对齐)。留档 .bak-worker-check1a-20260826。
ACTUAL_PORTS=""
for d in /sys/class/infiniband/*; do
  [ -d "$d" ] || continue
  nic=${d##*/}
  # 只看 RoCE 口名 (rocep/roceP 前缀); 忽略管理网/enP7s7/lo/veth/docker 等
  case "$nic" in
    roce*) HCA_FOUND=1; ACTUAL_PORTS="$ACTUAL_PORTS $nic" ;;
  esac
done
for nic in $ACTUAL_PORTS; do
  echo "[preflight] 检测到 RoCE 网卡口名: $nic"
done
if [ "$HCA_FOUND" = "1" ]; then
  for wish in $EXPECTED_PORTS; do
    if [ ! -e "/sys/class/infiniband/$wish" ]; then
      echo "[preflight-WARN] 期望口名缺 $wish (RoCE 口/HCA 布局与生产不一致)" >&2
    fi
  done
  # 以实际 netdev 口名集合为准判定"预期外口名"; 与生产 4 twin 口集合不一致 → No-Go
  for nic in $ACTUAL_PORTS; do
    case " $EXPECTED_PORTS " in
      *" $nic "*) ;;
      *) echo "[preflight-FAIL] 发现预期外 RoCE 口名: $nic (非生产布局, 检查点1a No-Go)" >&2; HCA_OK=0 ;;
    esac
  done
else
  echo "[preflight-WARN] 未检出任何 RoCE 口名 (无 /sys/class/infiniband/roce*), 检查物理层/驱动 (检查点1/5)" >&2
  HCA_OK=0
fi
if [ "$HCA_OK" != "1" ]; then
  echo "------------------------------------------------------------" >&2
  echo "[preflight-FAIL] 口名/HCA 与生产不一致 (检查点1a NO-GO) — 克隆布局差异" >&2
  echo "[rebuild-guidance] 请执行【首次部署重建】: dump GID → fix_gid_holes → 重算 HCA/PEER_HCA 到克隆实际口 → preflight 通过 → 再启动" >&2
  echo "------------------------------------------------------------" >&2
  exit 4
fi

# =============================================================
# [内嵌] P0 硬门: RoCE GID 布局预检 fail-fast (检查点1+3) — 在 docker run 之前
# =============================================================
_preflight_fail() {
  echo "------------------------------------------------------------" >&2
  echo "[preflight-FAIL] RoCE GID 布局 No-Go — 禁止带病启动 (检查点3 fail-fast)" >&2
  echo "[preflight-FAIL] 检测到克隆环境 RoCE/GID 布局与生产不一致 (空洞/子网/口名/HCA 失配)" >&2
  echo "------------------------------------------------------------" >&2
  echo "[rebuild-guidance] 遇到【克隆环境不一致】→ 请执行【首次部署重建】后再启动:" >&2
  echo "  1) dump 全机 GID 表:  for p in /sys/class/infiniband/*/ports/*/gids/[0-9]*; do ... done  (检查点1a)" >&2
  echo "  2) 修复 GID 空洞:     rerun fix_gid_holes (重建 IPv4 RoCEv2 GID, index 对齐 RoCE 网段)" >&2
  echo "  3) 重建网络配置:      重算 HCA/PEER_HCA 到克隆实际口集合" >&2
  echo "  4) 重跑 preflight:    bash ${PREFLIGHT_BIN} --expect-index ${GID_SUGGEST_INDEX:-N}  须 exit0" >&2
  echo "  5) 再启动本脚本:      bash $0  → 通过后才 docker run" >&2
  echo "  6) 验证:              NCCL_DEBUG_FILE 落持久卷 / vllm.log 核对 index 实际生效" >&2
  echo "------------------------------------------------------------" >&2
}

if [ ! -x "$PREFLIGHT_BIN" ] || [ ! -x "$PROBE_BIN" ]; then
  echo "[preflight-WARN] 预检依赖缺失: preflight=${PREFLIGHT_BIN} probe=${PROBE_BIN}" >&2
  echo "[preflight-FAIL] 缺失 GID 预检脚本无法保证安全启动, fail-fast (检查点3)" >&2
  _preflight_fail
  exit 3
fi

echo "[preflight] 探测本机建议 GID index (probe_gid_index.sh)..."
if PROBE_OUT=$(bash "$PROBE_BIN" 2>&1); then
  echo "$PROBE_OUT" | sed 's/^/[probe]   /'
else
  echo "[preflight-WARN] probe 执行异常, 沿用决策值 ${NCCL_IB_GID_INDEX} (降级路径)" >&2
fi
PROBE_SUGG=$(echo "$PROBE_OUT" | grep -oE '建议 NCCL_IB_GID_INDEX=[^ ]+' | head -1 | cut -d= -f2)
[ -n "$PROBE_SUGG" ] && GID_SUGGEST_INDEX="$PROBE_SUGG"

EXPECT_ARG=""
if [ "$GID_SUGGEST_INDEX" != "-1" ] && [ "$GID_SUGGEST_INDEX" != "REMOVE" ]; then
  EXPECT_ARG="--expect-index ${GID_SUGGEST_INDEX}"
fi
echo "[preflight] 执行 GID 布局预检: ${PREFLIGHT_BIN} --degrade ${EXPECT_ARG}"
if ! bash "$PREFLIGHT_BIN" --degrade ${EXPECT_ARG}; then
  RC=$?
  echo "[preflight-WARN] preflight 返回 $RC (期望 0=OK / 3=No-Go)"
  if [ "$RC" = "3" ] || [ "$RC" = "4" ]; then
    _preflight_fail
    exit 3
  fi
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
  # === FI 0.6.16 overlay (fi016 窗口注入) ===
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
  --health-cmd "pgrep -f VLLM::EngineCore >/dev/null 2>&1 || exit 1" \
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
    echo "[ok] READY (${i}0s)"; exit 0
  fi
  sleep 10
done
echo "[warn] 未就绪（可能仍在加载权重）; 观察 docker logs ${NAME}" >&2
exit 1