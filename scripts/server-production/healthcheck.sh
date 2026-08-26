#!/bin/bash
# =============================================================
# SCRIPT: healthcheck.sh
# VERSION: v1.1-p3
# ROLE: vLLM TP4 只读健康探针 (P0) — 四机通用
# USAGE: bash healthcheck.sh [--role head|worker] [--timeout SEC] [--grace-sec SEC]
#   --role 可省略: 自动探测本机容器名 (rank0=>head, rank1-3=>worker)
# EXITCODES: 0=healthy 1=unhealthy 2=用法错误
# 只读: 仅探测并输出状态, 不执行任何恢复动作
#       (主动重建由 healthcheck-rebuild.sh / systemd 自愈负责)
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# P0-3 (2026-08-25): 冷启动宽限 — 引擎冷启动 + 4并发×600K首次prefill 期间
#       /health 未就绪不应误判失败 (防 04:46 式冷启动误杀/重建风暴)。
#       宽限窗口默认 900s (= docker health --health-start-period 900s):
#         * 容器运行时长 < 宽限窗口 且 引擎就绪标记 ("Application startup complete")
#           尚未出现 → 判 healthy(skip HTTP 探针), 不触发重建。
#         * 就绪标记出现 或 运行时长 ≥ 宽限窗口 → 恢复正常 HTTP 探针。
#       容器缺失始终为硬故障 (不受宽限保护)。
# =============================================================
set -uo pipefail
export HOME=/home/liuxiaoya

ROLE=""
TIMEOUT=10
GRACE_SECONDS="${GRACE_SECONDS:-900}"

while [ $# -gt 0 ]; do
  case "$1" in
    --role) ROLE="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-10}"; shift 2 ;;
    --grace-sec) GRACE_SECONDS="${2:-900}"; shift 2 ;;
    -h|--help) grep -E '^# (USAGE|EXITCODES|ROLE|P0-3)' "$0"; exit 0 ;;
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

# 2. 冷启动宽限 (P0-3, 2026-08-25): 仅 head 适用 (workers 无 HTTP 探针)。
#    容器运行时长 < 宽限窗口 且 引擎就绪标记未出现 → skip HTTP 探针, 判 healthy。
#    容器缺失不在此列 (硬故障仍需快速重建)。
if [ "$ROLE" = "head" ] && [ "$FAIL" = "0" ]; then
  GRACE_ACTIVE=0
  STARTED_AT=$(docker inspect -f '{{.State.StartedAt}}' "$NAME" 2>/dev/null)
  if [ -n "$STARTED_AT" ]; then
    START_EPOCH=$(date -d "$STARTED_AT" +%s 2>/dev/null || echo 0)
    NOW_EPOCH=$(date +%s)
    UPTIME=$(( NOW_EPOCH - START_EPOCH ))
    if [ "$START_EPOCH" -gt 0 ] && [ "$UPTIME" -lt "$GRACE_SECONDS" ]; then
      # 宽限窗口内: 就绪标记出现则提前结束宽限, 否则 skip 探针
      if docker logs "$NAME" 2>/dev/null | grep -q "Application startup complete"; then
        echo "[healthcheck][head] ok 引擎就绪标记已出现 (uptime ${UPTIME}s), 提前结束冷启动宽限"
      else
        echo "[healthcheck][head] ok 冷启动宽限中 (uptime ${UPTIME}s < ${GRACE_SECONDS}s), skip /health 探针 (P0-3)"
        GRACE_ACTIVE=1
      fi
    fi
  fi

  if [ "$GRACE_ACTIVE" = "0" ]; then
    # 3. head 正常 HTTP 探针 (workers 无对外 HTTP 端口)
    # CONC(R12): 8001 为并发代理入口, 探针直连后端 8002 反映 vLLM 真实健康
    if curl -sf -m "$TIMEOUT" http://127.0.0.1:8002/health >/dev/null 2>&1; then
      echo "[healthcheck][head] ok 8002 /health 正常"
    else
      echo "[healthcheck][head] x 8002 /health 不可用"
      FAIL=1
    fi
  fi
fi

exit "$FAIL"
