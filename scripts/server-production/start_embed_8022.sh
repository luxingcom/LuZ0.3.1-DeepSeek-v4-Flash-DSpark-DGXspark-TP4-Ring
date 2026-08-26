#!/bin/bash
# =============================================================
# SCRIPT: start_embed_8022.sh
# VERSION: v1.5-r11
# USAGE: bash start_embed_8022.sh
# ROLE: embed 生产启动 (03/04 生产, 01/02 备用)
# HOST: dgxspark01~04
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/tools-index.md
# EXITCODES: 0=成功 1=业务失败
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# ============================================================
# DGX Spark embed 生产启动脚本（统一标准，2026-08-09 定稿）
# 用法：bash start_embed_8022.sh
# 适用：任意节点切换 embed 模式时使用（03/04 生产，01/02 备用）
#
# ⚠️ 铁律（anemll 0.2.1，2026-08-09 OOM 事故教训）：
#   1. VLLM_GPU_MEMORY_UTILIZATION 环境变量已失效（日志 "Unknown vLLM
#      environment variable detected"）→ 回落默认 util 0.92 导致 KV cache
#      预分配 110GB 吃光统一内存快 OOM → 必须用 --kv-cache-memory=<bytes>
#   2. 镜像 ENTRYPOINT=["vllm","serve"] → docker run 的 CMD 不能带 serve
#      （直接给模型路径），否则 vllm serve serve /models/... 报 unrecognized
#   3. 必须 --gpus all（否则 "Failed to infer device type"）
# ============================================================
set -e

IMG="192.168.5.187:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"
NAME="anemll-embed-8022"
PORT=8022
MODEL_DIR="/data/models"
# 4GB KV cache：embed 场景（max-num-seqs 32 / max-model-len 8192）充足；
# 若后续并发需求升高再上调（观察实际水位）
KV_CACHE_BYTES=4294967296

echo "=== 检查镜像就绪 ==="
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: $IMG 未就绪"; exit 1; }
echo "✅ 镜像就绪"

echo "=== 清理旧容器（若存在）==="
docker rm -f "$NAME" 2>/dev/null || true

echo "=== 启动 embed（port $PORT, kv-cache-memory=${KV_CACHE_BYTES}）==="
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all \
  -v "$MODEL_DIR":/models \
  -p $PORT:$PORT \
  "$IMG" \
  /models/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 --port $PORT \
  --served-model-name Qwen3-Embedding-0.6B \
  --max-model-len 8192 --max-num-seqs 32 \
  --enforce-eager --trust-remote-code \
  --kv-cache-memory=${KV_CACHE_BYTES}

echo "=== 等待就绪（最多 180s）==="
READY=0
for i in $(seq 1 36); do
  CODE=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  if [ "$CODE" = "200" ]; then READY=1; echo "✅ embed $PORT ready（第 ${i} 次检查，$((i*5))s）"; break; fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "⚠️ 180s 内未就绪，容器状态与日志："
  docker ps -a --format "{{.Names}}|{{.Status}}" | grep "$NAME" || true
  docker logs --tail 15 "$NAME" 2>&1 | tail -15
  exit 1
fi

echo "=== 容器状态 ==="
docker ps --format "{{.Names}}|{{.Status}}" | grep "$NAME"
echo "DONE"
