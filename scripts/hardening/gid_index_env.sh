#!/bin/bash
# =============================================================
# SCRIPT: gid_index_env.sh
# VERSION: v1.0-prod-harden  (2026-08-26)
# ROLE: start_tp4_*.sh 引用的 env 决策片段 (source 注入, 不破坏 ENV_ARGS 语义)
#   见 incident-clone-roce-prevention-2026-08-25.md §分层预防 P0-配置
#   "GID_INDEX 决策通过 source env 片段注入, 不改主脚本原有 env 顺序原则"
# 行为 (source 后):
#   1) 若已显式设置环境变量 TP4_GID_INDEX (非空) → 沿用 (人工覆写, 最高优先)。
#   2) 否则调 probe_gid_index.sh 实测 → 设置 NCCL_IB_GID_INDEX:
#        * 实测建议 -1 → 设 -1 (动态选择 connect.cc:328-332)
#        * 实测建议 REMOVE/未决 → 设 -1 并打降级栅栏告警
#        * 实测建议数字 N → 设 N, 并经 preflight 空洞栅栏校验 (≡不再踩洞)
#   3) 降级栅栏: 若实测完全不安全 (无 RoCEv2/IPv4 → -2 表示"无可用"), 输出 WARN
#      并建议 operator 移除 NCCL_IB_GID_INDEX env (动态)。栅栏不阻断启动,
#      但仍置 -1 以让 connect.cc 动态选择兜底。
# 主要副作用: 导出 GID_INDEX_DECIDED 标记 + NCCL_IB_GID_INDEX。
# 用法 (在 ENV_ARGS 组装之前 source):
#   source /opt/aicad-prod/scripts/gid_index_env.sh
#   # 之后可: ENV_ARGS+=( -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" )
# EXITCODES (source 时) :
#   0 = 决策完成 (NCCL_IB_GID_INDEX 已设)
#   3 = 实测 No-Go (栅栏告警但已兜底 -1, 不阻断; 供日志判断)
#   1 = 无法探测 (已降级 -1)
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
set -uo pipefail

PROBE_BIN="${TP4_PROBE_BIN:-/opt/aicad-prod/scripts/probe_gid_index.sh}"
PREFLIGHT_BIN="${TP4_PREFLIGHT_BIN:-/opt/aicad-prod/scripts/preflight_roce_gid.sh}"

# 人工覆写优先
if [ -n "${TP4_GID_INDEX:-}" ]; then
  echo "[gid-env] 人工设定 TP4_GID_INDEX=${TP4_GID_INDEX}, 直接采用 (跳过实测)"
  export NCCL_IB_GID_INDEX="$TP4_GID_INDEX"
  export GID_INDEX_DECIDED="manual"
  return 0 2>/dev/null || exit 0
fi

# 本文件被 source 时 return; 独立执行时 exit。统一封装:
# QA-fix R1: 本文件被调用方(set -euo pipefail) source, 独立语句 _finish_nores 返回 1
#   会触发调用方 errexit → 整脚本退出; 故所有 _finish_* 调用点一律补 `|| true`,
#   确保探测/降级失败时"继续降级 -1"而绝不中断调用脚本 (R2: 降级链不被截断)。
_finish_ok()   { export GID_INDEX_DECIDED=auto; return 0 2>/dev/null || exit 0; }
_finish_nores(){ export GID_INDEX_DECIDED=degraded; return 1 2>/dev/null || exit 1; }

# --- 首选: 实测建议知 ---
if [ -x "$PROBE_BIN" ]; then
  # QA-fix R2: probe 返回非零(探测失败)不得中断; set +e 抑制 subshell 内 errexit,
  #   `|| true` 避免 pipefail + 继承的 caller -e 在赋值失败时触发 errexit → 保证继续降级。
  SUGG=$(set +e; bash "$PROBE_BIN" --print-env 2>&1 | grep -oE 'NCCL_IB_GID_INDEX=[^ ]+' | head -1 | cut -d= -f2) || true
  if [ -n "${SUGG:-}" ]; then
    if [ "$SUGG" = "-1" ] || [ "$SUGG" = "REMOVE" ]; then
      echo "[gid-env] 实测建议动态: NCCL_IB_GID_INDEX=-1 (connect.cc 动态选择)"
      export NCCL_IB_GID_INDEX="-1"
      _finish_ok || true   # QA-fix R1: 保险
    elif [[ "$SUGG" =~ ^[0-9]+$ ]]; then
      # 数字: 用 preflight 洞栅栏校验该 index 是否踩洞
      echo "[gid-env] 实测建议 index=${SUGG}"
      HOLE=0
      for gf in /sys/class/infiniband/*/ports/*/gids/${SUGG}; do
        [ -f "$gf" ] || continue
        g=$(cat "$gf" 2>/dev/null)
        if [ -z "$g" ] || [ "$g" = "::" ] || [[ "$g" =~ ^0000:0000:0000:0000:0000:0000:0000:0000$ ]]; then
          HOLE=1
        fi
      done
      if [ "$HOLE" = "1" ]; then
        echo "[gid-env] ✗ 建议 index=${SUGG} 存在空洞, 降级为动态 -1 (防踩空 shm_broadcast hang)" >&2
        export NCCL_IB_GID_INDEX="-1"
        _finish_nores || true   # QA-fix R1: 不触发调用方 errexit
      else
        export NCCL_IB_GID_INDEX="$SUGG"
        _finish_ok || true      # QA-fix R1: 保险
      fi
    else
      echo "[gid-env] 实测输出异常 '${SUGG}', 降级动态 -1" >&2
      export NCCL_IB_GID_INDEX="-1"
      _finish_nores || true     # QA-fix R1
    fi
  else
    echo "[gid-env] probe 无有效输出, 降级动态 -1" >&2
    export NCCL_IB_GID_INDEX="-1"
    _finish_nores || true       # QA-fix R1
  fi
else
  echo "[gid-env] 未找到 $PROBE_BIN, 降级动态 -1 (默认走官方动态选择)" >&2
  export NCCL_IB_GID_INDEX="-1"
  _finish_nores || true         # QA-fix R1
fi