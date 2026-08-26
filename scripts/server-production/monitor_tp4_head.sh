#!/bin/bash
# =============================================================
# SCRIPT: monitor_tp4_head.sh
# VERSION: v1.5-r11
# ROLE: head(rank0) systemd 自愈 monitor — dgxspark01 (systemd ExecStart, 无 -h)
# HOST: dgxspark01
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/self-recovery.md
# EXITCODES: 恒非零退出触发 systemd Restart (自愈循环)
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
# TP4 head(rank0) systemd 自愈 monitor — dgxspark01
# 生命周期: 拉起容器 -> 前台跟随 docker wait -> 容器退出恒返回非零 -> systemd Restart
# head 重建前先清 worker 容器 -> 触发各机 systemd 自愈重建 (vLLM rank 须全部重新注册 TCPStore)
# D3 快速失败: TCPStore 就绪后等待 4 rank 接入; 60s 无新 rank 进展(未齐) => exit(1)
#             触发 head-first 全链重建, 替代旧版 300s 空转 (--distributed-timeout 兜底 NCCL 挂起)
set -uo pipefail
export HOME=/home/liuxiaoya
NAME=vllm-tp4-rank0
MASTER_PORT=25999
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker wait "$NAME" || true
  exit 1
fi
for host in dgxspark02 dgxspark04 dgxspark03; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
    "docker rm -f \$(docker ps -aq --filter name=vllm-tp4-rank) 2>/dev/null" >/dev/null 2>&1 || true
done
NO_WAIT=1 bash /opt/aicad-prod/scripts/start_tp4_head.sh || exit 1
# ---- D3 rank 就绪门禁 ----
echo "[i] 等待 head TCPStore :${MASTER_PORT} 就绪..."
for i in $(seq 1 60); do
  if ss -ltn 2>/dev/null | grep -q ":${MASTER_PORT} "; then break; fi
  if [ "$i" -eq 60 ]; then echo "[fail] TCPStore 60s 未监听, 快速失败"; exit 1; fi
  sleep 5
done
echo "[i] TCPStore 就绪, 等待 4 rank 接入 (60s 无新进展即放弃)"
LAST=0
NO_PROGRESS=0
while :; do
  N=$(ss -tn state established 2>/dev/null | grep -c ":${MASTER_PORT} ")
  if [ "$N" -ge 3 ]; then
    echo "[ok] rank 全齐 (TCPStore ${N} 连接)"
    break
  fi
  if [ "$N" -gt "$LAST" ]; then
    LAST="$N"; NO_PROGRESS=0
    echo "[i] rank 进展: ${N}/3 worker 接入"
  else
    NO_PROGRESS=$((NO_PROGRESS+1))
  fi
  if [ "$NO_PROGRESS" -ge 12 ]; then
    echo "[fail] 60s 无新 rank 接入 (当前 ${N}/3), 触发 head-first 全链重建" >&2
    exit 1
  fi
  sleep 5
done
docker wait "$NAME" || true
exit 1
