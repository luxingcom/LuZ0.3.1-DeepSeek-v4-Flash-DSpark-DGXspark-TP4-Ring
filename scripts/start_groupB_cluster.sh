#!/bin/bash
# =============================================================
# 组B LLM TP2 编排脚本 (在 node01 执行) v1.0
# 时序铁律: head 先起 → TCPStore :25055 就绪 → worker 后起 → API 轮询
# 参照: A组 start_v026r_cluster.sh (v2.0)
# =============================================================
set -euo pipefail

H_IP="<NODE_IP>"          # 03 head
API_KEY="<API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa"
HEAD_SCRIPT="<INSTALL_DIR>/scripts/start_head_groupB.sh"
WORKER_SCRIPT="<INSTALL_DIR>/scripts/start_worker_groupB.sh"

# 0) 幂等清理(双机) —— 禁止只重建单边
echo "== 清理残留 =="
docker rm -f vllm-groupb-head 2>/dev/null || true
ssh node01 "docker rm -f vllm-groupb-worker 2>/dev/null || true"

# 1) head 先起
echo "== 启动 head =="
bash "$HEAD_SCRIPT"

# 2) 轮询 TCPStore :25055 (head 机宿主探测; 容器内 ss 无权限)
echo "== 等待 TCPStore ${H_IP}:25055 =="
READY=0
for i in $(seq 1 60); do
  if nc -z "$H_IP" 25055 2>/dev/null; then echo "[ok] TCPStore OK (${i}0s)"; READY=1; break; fi
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "TCPStore 未就绪, 看 docker logs vllm-groupb-head"; exit 1; }

# 3) worker 后起 + 存活校验 (12x5s, 启动即崩溃立即失败)
echo "== 启动 worker =="
ssh node01 "bash $WORKER_SCRIPT"
echo "== 校验 worker 容器存活 =="
ssh node01 "for i in \$(seq 1 12); do
  st=\$(docker inspect -f '{{.State.Status}}' vllm-groupb-worker 2>/dev/null)
  [ "\$st" = "running" ] && { echo '[ok] worker running'; exit 0; }
  sleep 5
done
echo '[err] worker 启动失败, 最近日志:' >&2
docker logs --tail 100 vllm-groupb-worker 2>&1 >&2
exit 1"

# 4) 轮询 8001 /v1/models (120x10s, 冷启动余量)
echo "== 等待 LLM API :8001 =="
READY=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
        -H "Authorization: Bearer ${API_KEY}" \
        http://127.0.0.1:8001/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "[ok] LLM READY (${i}0s)"; READY=1; break; fi
  sleep 10
done
if [ "$READY" -ne 1 ]; then
  echo "[err] LLM 未就绪, 双机最近日志:" >&2
  echo "--- node01 head ---" >&2; docker logs --tail 100 vllm-groupb-head >&2 2>&1
  echo "--- node01 worker ---" >&2; ssh node01 "docker logs --tail 100 vllm-groupb-worker" >&2
  exit 1
fi

echo "== 组B TP2 集群就绪: http://<NODE_IP>:8001 =="
