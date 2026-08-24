#!/bin/bash
# =============================================================
# 组B head(03/node01/<MGMT_OCTET>) — anemll 0.2.1 LLM TP2 启动脚本 v1.0
# 基线: A组 start_head_v026r.sh (v3.3, GID_INDEX=2) 改造
# 变更: master <NODE_IP>:25055 | 挂载 <MODELS_DIR> | 端口 8001
#       | NCCL 2.30.7 LD_LIBRARY_PATH 前插 | hostname 校验 node01
# 负责人: SRE Rex | 2026-08-08
# =============================================================
set -euo pipefail

# ---- 守卫 ----
[ "$(hostname)" = "node01" ] && [ "${NODE_RANK:-0}" = "0" ] || {
  echo "ERROR: 本脚本仅可在 node01(NODE_RANK=0) 运行 (当前 hostname=$(hostname))" >&2
  exit 1
}

IMG="<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"   # anemll 0.2.1 21.6G (9ea563a724d4), 双机一致
# 如需 digest pin: IMG="<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0@sha256:9ea563a724d4..."
NAME="vllm-groupb-head"
MASTER_ADDR="<NODE_IP>"
MASTER_PORT="25055"
PORT="8001"
GID_INDEX="${NCCL_IB_GID_INDEX:-2}"   # 按 03/04 GID 检查结果覆盖; 默认 2(对照 A组 <MGMT_OCTET> 修复)

# ---- 回滚锚点 ----
mkdir -p <INSTALL_DIR>/backup
docker inspect "$NAME" > "<INSTALL_DIR>/backup/rollback_${NAME}.json" 2>/dev/null || true
echo "[i] 回滚锚点: <INSTALL_DIR>/backup/rollback_${NAME}.json"

# ---- vLLM serve 命令 (固定 heredoc, 防二次解析 brace expansion) ----
SERVE_CMD="vllm serve \
  --model /models \
  --served-model-name deepseek-v4-flash-0731 \
  --kv-cache-dtype nvfp4_ds_mla \
  --max-model-len 131072 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --scheduling-policy priority \
  --gpu-memory-utilization 0.88 \
  --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\"}' \
  --moe-backend flashinfer_b12x \
  --distributed-executor-backend mp \
  --port ${PORT} \
  --api-key <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa \
  --enable-flashinfer-autotune \
  --max-cudagraph-capture-size 24 \
  --enable-prompt-tokens-details \
  --generation-config vllm \
  --distributed-timeout-seconds 300 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
  --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}"
echo "[i] serve 命令: $SERVE_CMD"

# ---- 静态 env 基线 (Anemll + NCCL 2.30.7 + GID 按检查) ----
ENV_ARGS=(
  -e 'CUDA_DEVICE_ORDER=PCI_BUS_ID'
  -e 'CUDA_VISIBLE_DEVICES=0'
  -e 'DG_JIT_NVCC_COMPILER=/tmp/env-e-build/nvcc_wrapper.py'
  -e 'DG_JIT_USE_NVRTC=0'
  -e 'DSPARK_SLOT_CLAMP=1'
  -e 'FLASHINFER_DISABLE_VERSION_CHECK=1'
  -e 'GLOO_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1'
  -e 'HEADLESS='
  -e 'HF_HOME=/cache/huggingface'
  -e 'HF_HUB_OFFLINE=1'
  -e 'LANG=C.UTF-8'
  -e 'LC_ALL=C.UTF-8'
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu'   # NCCL 2.30.7 前插
  -e "MASTER_ADDR=${MASTER_ADDR}"
  -e "MASTER_PORT=${MASTER_PORT}"
  -e 'NCCL_CROSS_NIC=1'
  -e 'NCCL_DEBUG=INFO'
  -e 'NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h-%p.log'
  -e "NCCL_IB_GID_INDEX=${GID_INDEX}"     # 关键坑位1: 03/04 GID 检查后覆盖
  -e 'NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1'
  -e 'NCCL_IB_RETRY_CNT=7'
  -e 'NCCL_IB_TIMEOUT=1000'
  -e 'NCCL_IGNORE_CPU_AFFINITY=1'
  -e 'NCCL_NET=IB'
  -e 'NCCL_PROTO=LL,LL128,Simple'
  -e 'NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1'
  -e 'NODE_RANK=0'
  -e 'NVIDIA_VISIBLE_DEVICES=all'
  -e 'PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
  -e 'PIP_DISABLE_PIP_VERSION_CHECK=1'
  -e 'PORT=8001'
  -e 'PYTHONDONTWRITEBYTECODE=1'
  -e 'PYTHONUNBUFFERED=1'
  -e 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
  -e 'SERVED_MODEL_NAME=deepseek-v4-flash-0731'
  -e 'TILELANG_CLEANUP_TEMP_FILES=1'
  -e 'TORCH_CUDA_ARCH_LIST=12.1a'
  -e 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1'
  -e 'VLLM_DISABLE_PYNCCL=1'
  -e 'VLLM_ENGINE_READY_TIMEOUT_S=7200'
  -e "VLLM_HOST_IP=${MASTER_ADDR}"        # <NODE_IP> 数据面
  -e "VLLM_DP_MASTER_IP=${MASTER_ADDR}"
  -e 'VLLM_USE_B12X_MOE=1'
  -e 'VLLM_USE_BREAKABLE_CUDAGRAPH=1'
  -e 'VLLM_USE_FLASHINFER_SAMPLER=1'
  -e 'VLLM_DSPARK_LOCAL_ARGMAX=1'
  -e 'VLLM_TRITON_MLA_SPARSE=1'
)

# ---- 幂等清理 + 持久目录 ----
docker rm -f "$NAME" 2>/dev/null || true
mkdir -p <INSTALL_DIR>/envs/vllm-envc-cache \
         <INSTALL_DIR>/cache/vllm-cache \
         <INSTALL_DIR>/cache/tilelang-cache \
         <INSTALL_DIR>/logs/vllm \
         <INSTALL_DIR>/backup

BINDS=(
  -v <MODELS_DIR>/deepseek-v4-flash-0731:/models:ro          # 03/04 挂载路径=<MODELS_DIR> (非 /home/<USER>)
  -v <INSTALL_DIR>/envs/nvcc_wrapper.py:/tmp/env-e-build/nvcc_wrapper.py:ro   # 部署前先拷贝(1710B)
  -v <INSTALL_DIR>/envs/vllm-envc-cache:/cache/huggingface
  -v <INSTALL_DIR>/cache/vllm-cache:/root/.cache/vllm:rw   # deepgemm JIT 持久, 重启免重编
  -v <INSTALL_DIR>/cache/tilelang-cache:/root/.tilelang/cache:rw
  # 可选(如需与 A组 patch-v026 对齐, 需先验证 21.6G 镜像差异):
  # -v <INSTALL_DIR>/patch-v026:/usr/local/lib/python3.12/dist-packages/vllm/v1/...:ro
)

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --network host --ipc=host --privileged --gpus all \
  --shm-size=64gb --ulimit memlock=-1 --ulimit stack=67108864 \
  --health-cmd "pgrep -f VLLM::EngineCore >/dev/null 2>&1 || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 5 --health-start-period 900s \
  "${BINDS[@]}" \
  -v <INSTALL_DIR>/logs/vllm:/var/log/vllm \
  "${ENV_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMG" -lc "$SERVE_CMD"

echo "[i] 容器已启动: ${NAME}"
echo "[i] 等待 head EngineCore 进程 (无需 worker 配对, ≤10min; 端到端就绪由编排层 :8001 判定)"
for i in $(seq 1 60); do
  if docker exec "$NAME" pgrep -f 'VLLM::EngineCore' >/dev/null 2>&1; then
    echo "[ok] HEAD ENGINE READY (${i}0s)"; exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" != "running" ]; then
    echo "[err] head 容器已退出, 最近日志:" >&2
    docker logs --tail 100 "$NAME" >&2
    exit 1
  fi
  sleep 10
done
echo "[err] head EngineCore 未在 10min 内就绪, 最近日志:" >&2
docker logs --tail 100 "$NAME" >&2
exit 1
