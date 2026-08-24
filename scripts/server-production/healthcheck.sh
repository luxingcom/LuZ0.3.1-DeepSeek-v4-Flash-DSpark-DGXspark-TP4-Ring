#!/bin/bash
# =============================================================
# SCRIPT: healthcheck.sh
# VERSION: v1.0-p1
# ROLE: vLLM TP4 只读健康探针 (P0) — 四机通用
# USAGE: bash healthcheck.sh [--role head|worker] [--timeout SEC]
#   --role 可省略: 自动探测本机容器名 (rank0=>head, rank1-3=>worker)
# EXITCODES: 0=healthy 1=unhealthy 2=用法错误
# 只读: 仅探测并输出状态, 不执行任何恢复动作
#       (主动重建由 healthcheck-rebuild.sh / systemd 自愈负责)
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
set -uo pipefail
export HOME=/home/<USER>

ROLE=""
TIMEOUT=10

while [ $# -gt 0 ]; do
  case "$1" in
    --role) ROLE="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-10}"; shift 2 ;;
    -h|--help) grep -E '^# (USAGE|EXITCODES|ROLE)' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

NAME=""
if [ -z "$ROLE" ]; then
  # 自动探测本机角色
  if docker ps --format '{{.Names}}' | grep -qx 'vllm-tp4-rank0'; then
    ROLE=head; NAME=vllm-tp4-rank0
  elif docker ps --format '{{.Names}}' | grep -qE '^vllm-tp4-rank[1-3]$'; then
    ROLE=worker; NAME=$(docker ps --format '{{.Names}}' | grep -E '^vllm-tp4-rank[1-3]$' | head -1)
  else
    echo "[healthcheck] 未找到 vllm-tp4-rank* 容器 (本机非 TP4 成员或服务未拉起)"
    exit 1
  fi
else
  case "$ROLE" in
    head)   NAME=vllm-tp4-rank0 ;;
    worker) NAME=$(docker ps --format '{{.Names}}' | grep -E '^vllm-tp4-rank[1-3]$' | head -1) ;;
    *) echo "role 必须是 head|worker" >&2; exit 2 ;;
  esac
fi

FAIL=0

# 1. 容器存在且 running
if ! docker ps --filter "name=^${NAME}$" --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[healthcheck][${ROLE}] x 容器 ${NAME} 不存在或未运行"
  FAIL=1
else
  echo "[healthcheck][${ROLE}] ok 容器 ${NAME} 运行中: $(docker ps --filter "name=^${NAME}$" --format '{{.Status}}')"
fi

# 2. head 额外 HTTP 探针 (workers 无对外 HTTP 端口)
if [ "$ROLE" = "head" ] && [ "$FAIL" = "0" ]; then
  if curl -sf -m "$TIMEOUT" http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "[healthcheck][head] ok 8001 /health 正常"
  else
    echo "[healthcheck][head] x 8001 /health 不可用"
    FAIL=1
  fi
fi

exit "$FAIL"
