#!/bin/bash
# =============================================================
# head(60) — 环境 E Anemll 升级版启动脚本 (v3.0-live, Anemll)
# 目标: 环境E 0731 + dspark-vllm-gx10:0.1.1 (vLLM 0.25, SM121a kernels (wrapper))
#       1M-ctx seqs6(0.85) nvfp4_ds_mla KV DSpark 5-token
# 变更 vs hybrid-1.6 版: 固定 heredoc 生成 vLLM 0.25 serve 命令
#       + shm/memlock/stack + GID 5->3 + 日志轮询就绪判定
# 负责人: 工程保障团队 SRE 雷克斯 | 2026-08-02
# 变更: E= D' + nvcc wrapper(sm_120f->sm_121a) + DG_JIT_NVCC_COMPILER=wrapper + deep_gemm cache 预清
# v3.1 加固 (SRE 雷克斯 2026-08-03): + --api-key <API_KEY>-* (8001 认证)
#       + deep_gemm JIT cache 持久卷 ~/vllm-cache/deep_gemm (重启免重编 sm_121a)
#       + 移除启动时 rm -rf deep_gemm cache
# v3.2 加固 (SRE 雷克斯 2026-08-03): vllm-cache 整卷持久化 ~/vllm-cache:/root/.cache/vllm:rw
#       + flashinfer autotune + modelinfos 一并持久化, 冷启动再省 ~1min
# v3.3 nccl-fix (SRE 雷克斯 2026-08-06): + --distributed-timeout-seconds 300 (卡死5min快速失败)
#       + VLLM_ENGINE_READY_TIMEOUT_S 7200->600 + NCCL_DEBUG INFO + NCCL_DEBUG_FILE (/var/log/vllm=nccl-%h.log)
# =============================================================
set -euo pipefail

# ---- 守卫 ----
[ "$(hostname)" = "node01" ] && [ "${NODE_RANK:-0}" = "0" ] || {
  echo "ERROR: 本脚本仅可在 spark-05cd(NODE_RANK=0) 运行" >&2
  exit 1
}

IMG="<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"  # TP2恢复 2026-08-07: 改指向本地 registry 34.2G (ghcr 直连不稳定)  # digest pin 2026-08-02 (deploy 前 resolve 填入)
NAME="vllm-envE-node"
NEW_SERVED_NAME="deepseek-v4-flash-0731"
PORT="8001"

# ---- 回滚锚点 ----
docker inspect "$NAME" > "<INSTALL_DIR>/backup/rollback_${NAME}.json" 2>/dev/null || true
echo "[i] 回滚锚点: /tmp/rollback_${NAME}.json"

# ---- vLLM 0.25 serve 命令（固定 heredoc，不再从镜像 CMD 抽取改写）----
# 注意: 双引号定义 + JSON 单引号包裹, 避免 bash -lc 二次解析触发 brace expansion
#        vLLM 0.26.1.dev0 (dspark-vllm-gx10:0.2.1-v026.0) 未实现 Concurrent Partial Prefill, 非默认值(默认=1)必然在
#        arg_utils._check_feature_supported() 抛 NotImplementedError 崩溃(2026-08-09 实测); 保留 --max-num-batched-tokens 4096
#        与 --scheduling-policy priority(标准参数), 与 .bak-sched-20260809162358 已知配置对齐
SERVE_CMD="vllm serve \
  --model /models \
  --served-model-name deepseek-v4-flash-0731 \
  --kv-cache-dtype nvfp4_ds_mla \
  --max-model-len 600000 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --scheduling-policy priority \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"num_speculative_tokens_per_batch_size\":[[1,1,5],[2,4,4],[5,6,3]]}' \
  --moe-backend flashinfer_b12x \
  --distributed-executor-backend mp \
  --distributed-timeout-seconds 300 \
  --port 8001 \
  --api-key <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa \
  --enable-flashinfer-autotune \
  --max-cudagraph-capture-size 24 \
  --enable-prompt-tokens-details \
  --generation-config vllm \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
  --master-addr <NODE_IP> --master-port 25000"
echo "[i] serve 命令: $SERVE_CMD"

# ---- 静态完整 env 基线（Anemll 版；剔除 hybrid-1.6 特有项）----
ENV_ARGS=(
  -e 'CUDA_DEVICE_ORDER=PCI_BUS_ID'
  -e 'CUDA_VISIBLE_DEVICES=0'
  -e 'DG_JIT_NVCC_COMPILER=/tmp/env-e-build/nvcc_wrapper.py'  # E: sm_120f->sm_121a wrapper
  -e 'DG_JIT_USE_NVRTC=0'
  -e 'DSPARK_SLOT_CLAMP=1'
  -e 'FLASHINFER_DISABLE_VERSION_CHECK=1'
  -e 'GLOO_SOCKET_IFNAME=enP7s7'
  -e 'HEADLESS='
  -e 'HF_HOME=/cache/huggingface'
  -e 'HF_HUB_OFFLINE=1'
  -e 'LANG=C.UTF-8'
  -e 'LC_ALL=C.UTF-8'
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu'  # NCCL2.30.7 前插 2026-08-07
  -e 'MASTER_ADDR=<NODE_IP>'
  -e 'MASTER_PORT=25000'
  -e 'NCCL_CROSS_NIC=1'
  -e 'NCCL_DEBUG=INFO'   # nccl-fix 2026-08-06: WARN->INFO 留证据
  -e 'NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log'   # nccl-fix: 落盘到 vllm-logs 卷(host ~/vllm-logs)
  -e 'NCCL_IB_GID_INDEX=2'   # 2026-08-08 加固: .60 重启后 GID3 变空(GID2=<NODE_IP> 有效)   # 修复: 5->3 (idx5 GID 为空, idx3 为 RoCEv2 IPv4)
  -e 'NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1'
  -e 'NCCL_IGNORE_CPU_AFFINITY=1'   # reason: 绑核评估 2026-08-09 (ncclpin): vllm serve 命令无显式 taskset/亲和设置; isolcpus=16-19 已由内核自动排除用户线程, vllm 主进程天然落 0-15; 若要显式绑 NCCL 线程到 16-19 需 cgroup/shim 方案(4-rank 场景落地); 当前 2-rank TP2 先用 NCCL_PROTO=LL 降低 CPU 代理参与
  -e 'NCCL_NET=IB'
  -e 'NCCL_PROTO=LL'   # reason: 2026-08-09 (ncclpin): 实测 16B allreduce LL vs Simple -33% (16.3 vs 24.6µs), i_p99 19-26µs 满足 P99≤40µs SLO; LL 降低 CPU 代理参与契合绑核; 大消息场景可后续用 tuner plugin 或 '^' 排除法; 由原 LL,LL128,Simple 自动选择改为强制 LL
  -e 'NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1'
  -e 'NCCL_IB_TIMEOUT=1000'   # RoCE hang 防护
  -e 'NCCL_IB_RETRY_CNT=7'    # 重试次数
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
  -e 'VLLM_DISABLE_PYNCCL=1'  # 修复: pynccl ncclCommInitRank 跨节点 invalid usage, 改用 torch.distributed NCCL
  -e 'VLLM_ENGINE_READY_TIMEOUT_S=600'   # nccl-fix 2026-08-06: 7200->600, API server 等引擎最多 10min 不静默挂 2h
  -e 'VLLM_HOST_IP=<NODE_IP>'  # 修复: get_ip() 默认走管理网 192.168.5.x, 强制数据面 ZMQ/mq 通信
  -e 'VLLM_DP_MASTER_IP=<NODE_IP>'  # 修复: coord_store 默认 127.0.0.1, 多节点 worker 连 loopback 卡死; 强制指向 head
  -e 'VLLM_USE_B12X_MOE=1'
  -e 'VLLM_USE_BREAKABLE_CUDAGRAPH=1'
  -e 'VLLM_USE_FLASHINFER_SAMPLER=1'
  -e 'VLLM_DSPARK_LOCAL_ARGMAX=1'
  -e 'VLLM_TRITON_MLA_SPARSE=1'
)

docker rm -f "$NAME" 2>/dev/null || true
mkdir -p ~/vllm-logs /tmp/vllm-envc-cache

BINDS=(
  -v <INSTALL_DIR>/models/deepseek-v4-flash-0731:/models:ro  # 2026-08-08 加固: 统一入口(软链)
  -v <INSTALL_DIR>/envs/nvcc_wrapper.py:/tmp/env-e-build/nvcc_wrapper.py:ro  # 2026-08-08 加固: 宿主源迁出 /tmp(重启丢失)  # E: nvcc wrapper (sm_120f->sm_121a)
  -v <INSTALL_DIR>/envs/vllm-envc-cache:/cache/huggingface  # 2026-08-08 加固: 宿主源迁出 /tmp
  -v "$HOME/vllm-cache:/root/.cache/vllm:rw"
  -v "$HOME/tilelang-cache:/root/.tilelang/cache:rw"  # v3.3 TILELANG_CACHE_DIR 持久卷: TileLang JIT cache (cache+tmp)
  -v "$HOME/patch-v026/model_executor/kernels/mhc/tilelang.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro"  # Plan A+D: two-tier fixed tile_n/n_splits (kill per-request JIT recompile)
)

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --network host --ipc=host --privileged --gpus all \
  --shm-size=64gb --ulimit memlock=-1 --ulimit stack=67108864 \
  --health-cmd "pgrep -f VLLM::EngineCore >/dev/null 2>&1 || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 5 --health-start-period 900s \
  "${BINDS[@]}" \
  -v ~/vllm-logs:/var/log/vllm \
  "${ENV_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMG" -lc "$SERVE_CMD"

echo "[i] 容器已启动: ${NAME}"
echo "[i] 等待就绪 (≤15min cold start): 轮询 docker logs 'Application startup complete'"
for i in $(seq 1 90); do
  if docker logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "[ok] READY (${i}0s)"; exit 0
  fi
  sleep 10
done
echo "[warn] 未就绪（可能仍在加载权重）; 观察 docker logs ${NAME}" >&2
exit 1
