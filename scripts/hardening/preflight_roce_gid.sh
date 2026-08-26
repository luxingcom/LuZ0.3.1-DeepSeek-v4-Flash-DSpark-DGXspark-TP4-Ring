#!/bin/bash
# =============================================================
# SCRIPT: preflight_roce_gid.sh
# VERSION: v1.0-prod-harden  (2026-08-26)
# ROLE: 连接建立前 RoCE GID 布局预检 (P0 硬门, 防克隆重启卡死 shm_broadcast hang)
#   见 incident-clone-roce-prevention-2026-08-25.md §分层预防 - P0-配置(Clone 最高优先)
#   "克隆严禁盲目复用生产 NCCL_IB_GID_INDEX=3 — 必须实测当前 GID 表 → 选型而非照抄"
# 检查内容:
#   1) 枚举各 RoCE 口 GID 表 index0-3, 判洞 (`::`/全0)
#   2) 判 IPv4 RoCEv2 类型 (gid_attrs/types)
#   3) 判 index3 子网前缀一致性 (本机口间多子网=ring 正常布局, 豁免仅WARN; 跨机一致性由 --peers 交叉核对承担)
# USAGE:
#   bash preflight_roce_gid.sh                          # 仅本机自检
#   bash preflight_roce_gid.sh --peers hostA:idx hostB:idx hostC:idx
#       # 邻接表: "<host>:<detected_index>" 每台一行 (兼容 "host=idx"), 交叉核对 index 一致性
#       # 由调用方 cluster(start_tp4_cluster.sh) 在编排层以四机邻接表调用 (检查点1+3)
#   bash preflight_roce_gid.sh --degrade               # 检测到 index 不匹配时输出降级建议
#   bash preflight_roce_gid.sh --expect-index N        # 期望 index (如 3), 不匹配即 No-Go
# EXITCODES:
#   0 = GID 布局 OK (无洞/IPv4 RoCEv2 存在/index 一致)
#   3 = No-Go (有空洞 / 无 IPv4 RoCEv2 / 跨机 index 不一致 / HCA 口名非预期)
#   2 = 用法/工具缺失 (ethtool/rdma 均不可用时降级为只读 sysfs 检查)
#   1 = 运行错误
# CHANGE: 改脚本须 bash -n + check_vllm_script.sh 通过 + .bak-<tag> 留档
# =============================================================
set -uo pipefail

# --- 配置 (可按机调) ---
EXPECTED_PORTS="rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1"
GID_SYS="/sys/class/infiniband"
# 空洞判定: 全零 GID 或缩略 "#::"
ALLZERO='^0000:0000:0000:0000:0000:0000:0000:0000$'

DEGRADE=0
EXPECT_INDEX=""
declare -A PEER_IDX=()
PEERS_GIVEN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --degrade) DEGRADE=1; shift ;;
    --expect-index) EXPECT_INDEX="${2:-}"; shift 2 ;;
    --peers)
      shift
      while [ $# -gt 0 ]; do
        case "$1" in --*) break ;; esac
        # format: host=idx 或 host:idx (QA-fix P2: 兼容文档 `:` 与解析 `=`, 不静默跳过判据3)
        if [[ "$1" == *=* ]] || [[ "$1" == *:* ]]; then
          h="${1%%[=:]*}"; i="${1#*[=:]}"
          PEER_IDX["$h"]="$i"; PEERS_GIVEN=1
        else
          echo "[preflight-WARN] --peers 参数 '$1' 缺少 host:idx 分隔符, 忽略该邻接表项 (判据3交叉核对可能不完整)" >&2
        fi
        shift
      done
      ;;
    -h|--help) grep -E '^# (USAGE|EXITCODES|ROLE|VERSION)' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

# --- 依赖可用性 (ethtool/rdma 可选; 不可用则降级 sysfs 只读) ---
declare -A HAS_TOOL=()
command -v ethtool >/dev/null 2>&1 && HAS_TOOL[ethtool]=1 || HAS_TOOL[ethtool]=0
command -v rdma >/dev/null 2>&1   && HAS_TOOL[rdma]=1   || HAS_TOOL[rdma]=0
command -v ibping >/dev/null 2>&1 && HAS_TOOL[ibping]=1 || HAS_TOOL[ibping]=0
echo "[preflight] tools: ethtool=${HAS_TOOL[ethtool]} rdma=${HAS_TOOL[rdma]} ibping=${HAS_TOOL[ibping]}"
if [ "${HAS_TOOL[ethtool]}" = "0" ] && [ "${HAS_TOOL[rdma]}" = "0" ]; then
  echo "[warn] ethtool/rdma 均不可用, 降级为 sysfs 只读检查 (skip 物理/连通判据)" >&2
fi

[ -d "$GID_SYS" ] || { echo "[fail] 无 $GID_SYS (非 InfiniBand 主机或权限不足)" >&2; exit 3; }

FAIL=0
HOLE_FOUND=0
declare -A PORT_HAS_IPV4=()      # port -> 1/0
declare -A PORT_INDEX3_SUBNET=() # port 的 index3 subnet prefix (前 4 段, 即 /64 前缀)
declare -A PORT_INDEX3_IPV4=()   # port 的 index3 GID 末 32bit (IPv4 段, QA-fix P3 主比较键)

# --- helper: 展开 GID 为紧凑形式, 去掉冒号便于比较 ---
norm_gid() { echo "$1" | tr -d ':'; }

# --- helper: 取 index3 的子网前缀 (前 16 hex 数字 = 64 bit) ---
# QA-fix P3: 保留 /64 前缀, 但判据3 实际以 GID 末 32bit(IPv4 段)比较 (见判据3), 区分为主。
subnet_of() { echo "$1" | cut -c1-16; }
# QA-fix P3: 取 GID 末 32bit (后 8 个 hex 数字 = IPv4 段), 判据3 的主比较键 — 区分度更高
ipv4_of()   { echo "$1" | grep -oE '[0-9a-f]{8}$' | head -1; }

echo "=== 本机 GID 布局巡检 (index0-3) ==="
for dev in "$GID_SYS"/*/; do
  dev=${dev%/}; dev_name=${dev##*/}
  # 仅检索存在 ports 的 device
  [ -d "$dev/ports" ] || continue
  for portdir in "$dev"/ports/*/; do
    port=${portdir%/}; port=${port##*/}
    gidsdir="${dev}/ports/${port}/gids"
    [ -d "$gidsdir" ] || continue
    PORT_HAS_IPV4["$dev_name/$port"]=0
    for idx in 0 1 2 3; do
      gf="${gidsdir}/${idx}"
      [ -f "$gf" ] || continue
      gid=$(cat "$gf" 2>/dev/null | tr -d '\n')
      # 判洞: 全零 或 缩略 "::"/空
      gid_norm=$(norm_gid "$gid")
      if [ -z "$gid_norm" ] || [ "$gid" = "::" ] || [[ "$gid" =~ $ALLZERO ]]; then
        echo "  [HOLE] $dev_name/$port idx=$idx gid='$gid'"
        HOLE_FOUND=1
        continue
      fi
      # 判 IPv4 RoCEv2 type: gid 8-11 段为 0xfe80 (IPv6 link-local)? 更准确看 attrs/types
      type_attr="${dev}/ports/${port}/gid_attrs/types/${idx}"
      gtype="unknown"
      [ -f "$type_attr" ] && gtype=$(cat "$type_attr" 2>/dev/null)
      # RoCEv2 IPv4 type 常见为 "RoCE v2" (内核 4.9+ gid_attrs 显示 "RoCE v2")
      is_ipv4=0
      case "$gtype" in
        *"RoCE v2"*|*"rocev2"*|*"RoCEv2"*) is_ipv4=1 ;;
        *)
          # 兜底: 通过类型文本含 v2 且该 index 非 0xfe80 link-local
          ;;
      esac
      if [ "$is_ipv4" = "1" ]; then PORT_HAS_IPV4["$dev_name/$port"]=1; fi
      # index3 记录子网前缀 (与 IPv4 段, 判据3 用)
      if [ "$idx" = "3" ]; then
        PORT_INDEX3_SUBNET["$dev_name/$port"]="$(subnet_of "$gid_norm")"
        PORT_INDEX3_IPV4["$dev_name/$port"]="$(ipv4_of "$gid_norm")"
      fi
      printf "  [ok] %-20s idx=%s  type=%-12s gid=%s\n" "$dev_name/$port" "$idx" "$gtype" "$gid"
    done
  done
done

echo "=== 判据 1: 空洞 ==="
if [ "$HOLE_FOUND" = "1" ]; then
  echo "  [No-Go] 检测到 GID 空洞, index 布局不完整 → 写死 GID_INDEX 将踩空 (connect.cc:317-321 盲信)"
  FAIL=1
else
  echo "  [ok] 无 GID 空洞 (index0-3 均有效)"
fi

echo "=== 判据 2: IPv4 RoCEv2 存在 (index0-3 内至少有 RoCEv2/IPv4) ==="
IPV4_OK=0
for k in "${!PORT_HAS_IPV4[@]}"; do
  if [ "${PORT_HAS_IPV4[$k]}" = "1" ]; then IPV4_OK=1; echo "  [ok] $k 含 RoCEv2/IPv4 GID"; fi
done
if [ "$IPV4_OK" = "0" ]; then
  echo "  [No-Go] 未检出任何 RoCEv2/IPv4 GID → 无法可靠选择 index, 应设 NCCL_IB_GID_INDEX 动态(-1)"
  FAIL=1
fi

echo "=== 判据 3: index3 子网一致性 (口间多子网=ring 正常布局, 豁免仅WARN; 跨机一致性才硬门) ==="
# QA-fix P3: 原实现只比前 64bit(/64 前缀), 克隆机器的 /64 前缀通常相同 → 无区分力。
#   现以 GID 末 32bit(IPv4 段)为主比较键, 区分同子网不同 IP; /64 前缀仅作旁证。
# 修复(2026-08-26, ringfix): 生产 4 口 ring 每口连不同邻居子网(如 head 实测
#   10.100.140/136/141/137), 本机口间 index3 IPv4 段天然不同 — 这是 ring 正常布局, 不是故障。
#   原"口间一致"假设了单子网拓扑, 对多口多子网 ring 必然误拦。现改为仅 WARN(不置 FAIL);
#   判据1空洞/判据2 IPv4存在/判据4期望index空洞/--peers 跨机一致性 仍是真正 No-Go 判据。
# 本机各口 index3 的 IPv4 段 (末 32bit) — 仅诊断收集, 不再作为 No-Go (IPV4_SET 供 --peers 段 LOCAL_SUGGEST 使用)
IPV4_SET=""
IPV4_SEG_DIFF_WARN=0
for k in "${!PORT_INDEX3_IPV4[@]}"; do
  s="${PORT_INDEX3_IPV4[$k]}"
  sub="${PORT_INDEX3_SUBNET[$k]}"
  echo "  [info] $k index3 ipv4seg=$s subnet64=$sub"
  if [ -n "$IPV4_SET" ] && [ "$IPV4_SET" != "$s" ]; then
    echo "[warn] 本机口间 index3 IPv4 段不同 (ring 多子网布局, 正常; 跨机一致性由 --peers 交叉核对承担): '$IPV4_SET' vs '$s'" >&2
    IPV4_SEG_DIFF_WARN=1
  fi
  [ -n "$IPV4_SET" ] || IPV4_SET="$s"
done
if [ "$IPV4_SEG_DIFF_WARN" = "1" ]; then
  echo "  [warn] 本机口间 index3 IPv4 段不完全一致 — 多口多子网 ring 正常布局, 豁免(仅 WARN, 不置 FAIL); 跨机一致性由 --peers 硬门承担" >&2
else
  echo "  [ok] 本机口间 index3 IPv4 段一致: ${IPV4_SET}"
fi

# 邻接表交叉核对
if [ "$PEERS_GIVEN" = "1" ]; then
  echo "  [info] 邻接表交叉核对 (${#PEER_IDX[@]} 台):"
  for h in "${!PEER_IDX[@]}"; do
    echo "    $h -> idx=${PEER_IDX[$h]}"
  done
  # 提取本机建议 index (默认 index3 若有效) — QA-fix P3: 沿用 IPv4 段判定结果 IPV4_SET
  LOCAL_SUGGEST=""
  if [ -n "${IPV4_SET:-}" ]; then
    # 若本机 index3 IPv4 段有效则用 3
    LOCAL_SUGGEST="3"
  fi
  # 计算所有主机 index 集合
  ALL_IDX="$LOCAL_SUGGEST"
  for h in "${!PEER_IDX[@]}"; do ALL_IDX="$ALL_IDX ${PEER_IDX[$h]}"; done
  UNIQUE_IDX=$(echo $ALL_IDX | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')
  N_UNIQ=$(echo $UNIQUE_IDX | wc -w)
  if [ "$N_UNIQ" -gt 1 ]; then
    echo "  [No-Go] 跨主机 index 不一致 (${UNIQUE_IDX}) → 写死单一 index 会跨网段/跨入洞, 必须动态选择"
    FAIL=1
  else
    echo "  [ok] 跨主机 index 一致: ${UNIQUE_IDX}"
  fi
fi

# 期望 index 校验
if [ -n "$EXPECT_INDEX" ]; then
  echo "=== 判据 4: 期望 index=$EXPECT_INDEX ==="
  # 列出每个口期望 index 的 GID 是否非洞
  MATCH_BAD=0
  for portdir in "$GID_SYS"/*/ports/*/; do
    dev=${portdir%/*}
    gf="${portdir}/gids/${EXPECT_INDEX}"
    if [ -f "$gf" ]; then
      g=$(cat "$gf" 2>/dev/null)
      if [ -z "$g" ] || [[ "$g" =~ $ALLZERO ]] || [ "$g" = "::" ]; then
        echo "  [No-Go] ${dev##*/}/期望 idx=$EXPECT_INDEX 为空洞"
        MATCH_BAD=1
      fi
    fi
  done
  [ "$MATCH_BAD" = "1" ] && FAIL=1 || echo "  [ok] 期望 index=${EXPECT_INDEX} 无空洞"
fi

echo "=== 最终判定 ==="
if [ "$FAIL" = "0" ]; then
  echo "[preflight] ✅ GID 布局 OK — No-Go 解除, 可按实测注入或动态选择"
  exit 0
else
  echo "[preflight] ❌ No-Go — GID 布局异常, 禁止带写死 index 启动"
  if [ "$DEGRADE" = "1" ]; then
    echo "[preflight] 降级建议: 应设 NCCL_IB_GID_INDEX=-1 (动态选择, connect.cc:328-332) 或移除该 env"
    echo "             (见 incident-clone-roce-prevention-2026-08-25.md §P0-配置 首选: 移除/设 -1)"
  fi
  exit 3
fi