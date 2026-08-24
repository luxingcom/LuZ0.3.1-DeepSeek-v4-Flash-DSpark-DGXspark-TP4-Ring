#!/bin/bash
# restart_run.sh <mode> — P2 运行切换: 四节点 MODE 设置 + 容器清理 + 手动 head-first 重启
# (跳过 start_tp4_cluster.sh 的 B12X 门禁——与当前 vLLM rendezvous 语义死锁, 2026-08-21 实证)
set -u
MODE="${1:?usage: restart_run.sh <mode 0|1|2>}"
PROD=<INSTALL_DIR>

echo "[restart] MODE=$MODE 写入四节点"
echo "$MODE" > <INSTALL_DIR>/nvfp4/plugin_a1/MODE
for h in node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "echo $MODE > <INSTALL_DIR>/nvfp4/plugin_a1/MODE"
done

echo "[restart] 清理四机容器"
docker rm -f vllm-tp4-rank0 >/dev/null 2>&1 || true
for h in node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    "docker rm -f vllm-tp4-rank1 vllm-tp4-rank2 vllm-tp4-rank3 >/dev/null 2>&1" || true
done
sleep 5
if ss -tln 2>/dev/null | grep -q ':8001'; then echo "[error] :8001 仍占用"; exit 1; fi

export VLLM_API_KEY="$(echo '<PASSWORD>' | sudo -S cat $PROD/secrets/vllm.env 2>/dev/null | grep -oP '^VLLM_API_KEY=\K.*' | head -1)"
[ -n "$VLLM_API_KEY" ] || { echo "[error] API key 获取失败"; exit 1; }

echo "[restart] 启动 head (rank0)"
nohup bash $PROD/scripts/start_tp4_head.sh > /tmp/_routea_work/head_run.log 2>&1 &

echo "[restart] 轮询 TCPStore :25999 (≤600s)"
READY=0
for i in $(seq 1 120); do
  if ss -tln 2>/dev/null | grep -q ':25999'; then READY=1; break; fi
  sleep 5
done
[ "$READY" = "1" ] || { echo "[error] TCPStore 未就绪"; exit 1; }
echo "[restart] TCPStore 就绪, 错峰启动 workers (15s 间隔)"

declare -A RH=(
  [1]="node01:<NODE_IP>"
  [2]="node01:<NODE_IP>"
  [3]="node01:<NODE_IP>"
)
for r in 1 2 3; do
  h="${RH[$r]%%:*}"; ip="${RH[$r]##*:}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    "VLLM_API_KEY=${VLLM_API_KEY} NODE_RANK=${r} VLLM_HOST_IP=${ip} NCCL_IB_HCA=rocep1s0f1,rocep1s0f0 nohup bash $PROD/scripts/start_tp4_worker.sh > \$HOME/start_tp4_rank${r}.log 2>&1 &"
  echo "[restart] rank${r} (${h}) 已下发"
  sleep 15
done

echo "[restart] 等待就绪: 'Application startup complete' (≤20min)"
for i in $(seq 1 120); do
  if docker logs vllm-tp4-rank0 2>&1 | grep -q "Application startup complete"; then
    echo "[restart] RUN READY (mode=$MODE)"
    docker logs vllm-tp4-rank0 2>&1 | grep -E "Using .*Mxfp4|Using W4A4|KV cache size|routea" | tail -6
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q vllm-tp4-rank0; then
    echo "[error] head 容器已退出"; docker logs --tail 20 vllm-tp4-rank0 2>&1 | tail -10; exit 1
  fi
  sleep 10
done
echo "[error] 20min 未就绪"
docker logs --tail 30 vllm-tp4-rank0 2>&1 | tail -15
exit 1
