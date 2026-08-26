#!/bin/bash
# =============================================================
# SCRIPT: probe_gid_index.sh
# VERSION: v1.0-prod-harden  (2026-08-26)
# ROLE: 探测当前 RoCEv2/IPv4 GID 实际 index, 输出建议的 NCCL_IB_GID_INDEX
#   见 incident-clone-roce-prevention-2026-08-25.md 检查点2 / §分层预防 P0-配置
#   "probe_gid_index.sh: 枚举各口 index, 输出建议值 = 首个 type 为 RoCEv2/IPv4、非 HOLE
#   的 index; 若多机该 index 子网前缀不一致 → 输出 REMOVE/-1 (动态), 否则输出该 index"
# USAGE:
#   bash probe_gid_index.sh                       # 输出建议 index 值 (单机视角)
#   bash probe_gid_index.sh --expect N            # 校验期望 index 有效性
#   bash probe_gid_index.sh --peers h:idx ...     # 多机交叉核对, 输出最终可注入值
#       # 邻接表 "<host>:<index>" (兼容 "host=idx"); 由调用方 cluster(start_tp4_cluster.sh)
#       # 在编排层以四机邻接表调用 (检查点1+3)
#   bash probe_gid_index.sh --print-env           # 输出 "NCCL_IB_GID_INDEX=<val>" 可 source
# 输出: 建议 index 数值 (0-255) 或 -1 (动态选择) 或 REMOVE (建议移除该 env, 即不注入)
# EXITCODES:
#   0 = 有明确建议值 (数值 / -1 / REMOVE)
#   3 = No-Go (无法可靠判定, 应手工介入)
#   1 = 运行错误
# CHANGE: 改脚本须 bash -n + check_vllm_script.sh 通过 + .bak-<tag> 留档
# =============================================================
set -uo pipefail

GID_SYS="/sys/class/infiniband"
ALLZERO='^0000:0000:0000:0000:0000:0000:0000:0000$'
EXPECT=""
PRINT_ENV=0
declare -A PEER_IDX=()

while [ $# -gt 0 ]; do
  case "$1" in
    --print-env) PRINT_ENV=1; shift ;;
    --expect) EXPECT="${2:-}"; shift 2 ;;
    --peers)
      shift
      while [ $# -gt 0 ]; do
        case "$1" in --*) break ;; esac
        # QA-fix P2: 兼容文档 `host:idx` 与解析 `host=idx`; 都不匹配则显式告警 (不静默跳过判据3)
        if [[ "$1" == *=* ]] || [[ "$1" == *:* ]]; then
          h="${1%%[=:]*}"; i="${1#*[=:]}"; PEER_IDX["$h"]="$i"
        else
          echo "[probe-WARN] --peers 参数 '$1' 缺少 host:idx 分隔符, 忽略该邻接表项" >&2
        fi
        shift
      done
      ;;
    -h|--help) grep -E '^# (USAGE|EXITCODES|ROLE|VERSION)' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

[ -d "$GID_SYS" ] || { echo "[fail] 无 $GID_SYS" >&2; exit 3; }

norm_gid() { echo "$1" | tr -d ':'; }
subnet_of() { echo "$1" | cut -c1-16; }

# --- 收集: 每口 index -> {gid, type, subnet} ; 首个 IPv4 RoCEv2 非洞 index ---
FIRST_IPV4=""
declare -A IDX_SUBNET=()       # index -> 汇总的子网(首个)
declare -A IDX_TYPE=()

for dev in "$GID_SYS"/*/; do
  dev=${dev%/};
  for portdir in "$dev"/ports/*/; do
    gidsdir="${portdir}/gids"
    [ -d "$gidsdir" ] || continue
    for gf in "$gidsdir"/[0-9]*; do
      [ -f "$gf" ] || continue
      idx=${gf##*/}
      gid=$(cat "$gf" 2>/dev/null | tr -d '\n')
      [ -z "$gid" ] && continue
      gid_norm=$(norm_gid "$gid")
      # 空洞跳过
      if [ "$gid" = "::" ] || [[ "$gid" =~ $ALLZERO ]] || [ -z "$gid_norm" ]; then
        continue
      fi
      typef="${portdir}/gid_attrs/types/${idx}"
      gtype="unknown"
      [ -f "$typef" ] && gtype=$(cat "$typef" 2>/dev/null)
      IDX_TYPE["$idx"]="$gtype"
      # 记录子网 (同一 index 不同口可能不同, 逐口追加比对)
      IDX_SUBNET["$idx"]="${IDX_SUBNET[$idx]:-} $(subnet_of "$gid_norm")"
      # 找首个 IPv4 RoCEv2
      case "$gtype" in
        *"RoCE v2"*|*"RoCEv2"*|*"rocev2"*)
          if [ -z "$FIRST_IPV4" ]; then FIRST_IPV4="$idx"; fi
          ;;
      esac
    done
  done
done

# --- 判定: 多个 index 被 RoCEv2 占用时, index 数值越高越可能不稳定(生产 index3) ---
# QA-fix Q1: 原实现仅当"index3 是首个 IPv4"才选 3; 现改为: 若 index3 段为 IPv4 非洞 → 优先建议 3,
#   否则取首个 RoCEv2/IPv4 index。
SUGGEST=""
if [ -n "$FIRST_IPV4" ]; then
  # index3 段若为 RoCEv2/IPv4 且非洞 (IDX_TYPE[3] 为 RoCEv2 且 IDX_SUBNET[3] 非空非洞) → 优先 3
  IDX3_VALID=0
  if [ -n "${IDX_SUBNET[3]:-}" ]; then
    case "${IDX_TYPE[3]:-}" in
      *"RoCE v2"*|*"RoCEv2"*|*"rocev2"*) IDX3_VALID=1 ;;
    esac
  fi
  if [ "$IDX3_VALID" = "1" ]; then
    SUGGEST="3"
  else
    SUGGEST="$FIRST_IPV4"
  fi
else
  SUGGEST="-1"
fi

# --- 多机交叉核对 ---
if [ "${#PEER_IDX[@]}" -gt 0 ]; then
  ALLV="$SUGGEST"
  for h in "${!PEER_IDX[@]}"; do ALLV="$ALLV ${PEER_IDX[$h]}"; done
  N_UNIQ=$(echo $ALLV | tr ' ' '\n' | sort -u | grep -v '^$' | wc -l)
  if [ "$N_UNIQ" -gt 1 ]; then
    echo "[probe] 多机 index 不一致 ($ALLV) → 写死单一 index 会跨网段/入洞"
    SUGGEST="-1"
  fi
fi

# --- 期望 index 校验 ---
if [ -n "$EXPECT" ]; then
  if [ -n "${IDX_SUBNET[$EXPECT]:-}" ] && [ "${IDX_SUBNET[$EXPECT]:-}" != "None" ] && [[ "${IDX_SUBNET[$EXPECT]:-}" != " " ]]; then
    echo "[probe] 期望 index=$EXPECT 有效 (subnet:${IDX_SUBNET[$EXPECT]})"
  else
    echo "[probe] 期望 index=$EXPECT 是空洞或缺失 → 建议改动态" >&2
    SUGGEST="-1"
  fi
fi

# --- 输出 ---
echo "[probe] 建议 NCCL_IB_GID_INDEX=$SUGGEST  (首个 RoCEv2/IPv4 非洞 index)"
echo "[probe] 各 index type:"
for i in 0 1 2 3; do echo "  idx=$i type=${IDX_TYPE[$i]:-NONE} subnet=${IDX_SUBNET[$i]:-NONE}"; done

if [ "$PRINT_ENV" = "1" ]; then
  if [ "$SUGGEST" = "-1" ] || [ "$SUGGEST" = "REMOVE" ]; then
    echo "NCCL_IB_GID_INDEX=-1"
  else
    echo "NCCL_IB_GID_INDEX=$SUGGEST"
  fi
fi

echo "[probe] 备注: 若生产曾写死 index3 而此处建议不同, 遵循实时实测 (铁律: 实测→选型, 非照抄)"
exit 0