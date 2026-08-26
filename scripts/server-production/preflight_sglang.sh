#!/bin/bash
# =============================================================
# SCRIPT: preflight_sglang.sh
# VERSION: v0.1 (SGLang 测试环境准备阶段 — 只读一键核验)
# ROLE: 四机一键核验（镜像/权重/端口/NCCL/内存），供正式运行前使用
# USAGE: bash preflight_sglang.sh [--local] [node_ip ...]
#   --local      仅核验当前节点
#   node_ip...   指定核验节点（默认全部 186/187/188/189）
#   可用 env: NODES="ip1 ip2 ..." 覆盖默认节点列表
# =============================================================
# 核验项:
#   1) 镜像存在 + RootFS layer md5 四机一致 (期望 96e467d43b59c2362246545bacb4c9fe)
#   2) 权重: 48 shards + config.json + hf_quant_config.json + tokenizer.json
#   3) 端口 8010/8011/26000 空闲
#   4) NCCL 补丁文件存在 + 容器内 /proc/self/maps 实际加载 2.30.7
#   5) 内存可用量
#   6) 生产 vLLM 容器状态（信息项: 正式运行前需停止）
# =============================================================
# DOCS: file:///opt/aicad-prod/docs/ops/tools-index.md
set -uo pipefail

IMG="nvcr.io/nvidia/sglang:26.07-py3"
EXPECT_LAYER_MD5="96e467d43b59c2362246545bacb4c9fe"
MODEL_DIR="/opt/aicad-prod/models/deepseek-v4-flash-0731-nvfp4"
SSH_USER="${SSH_USER:-liuxiaoya}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"

# 节点列表
LOCAL_ONLY=0
POS=()
for a in "$@"; do
  case "$a" in
    --local) LOCAL_ONLY=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) POS+=("$a") ;;
  esac
done
if [ "$LOCAL_ONLY" = "1" ]; then
  NODES=( "127.0.0.1" )
elif [ ${#POS[@]} -gt 0 ]; then
  NODES=( "${POS[@]}" )
else
  NODES=( 192.168.5.186 192.168.5.187 192.168.5.188 192.168.5.189 )
fi
[ -n "${NODES:-}" ] && : || NODES=( "${NODES[@]}" )

# 远程执行函数: run_node <ip> <script_str>
run_node() {
  local ip="$1"; shift
  if [ "$ip" = "127.0.0.1" ] || [ "$ip" = "$(hostname -I 2>/dev/null | awk '{print $1}')" ]; then
    bash -c "$*"
  else
    ssh $SSH_OPTS "${SSH_USER}@${ip}" 'bash -s' <<EOF
$*
EOF
  fi
}

declare -A LAYER_MD5 WEIGHT_SHK PORTS_OK NCCL_OK MEM_AVAIL VLLM_STATE

for ip in "${NODES[@]}"; do
  echo "==================== [preflight] ${ip} ===================="
  OUT=$(run_node "$ip" '
    set -uo pipefail
    echo "=== 1. IMAGE+LAYER_MD5 ==="
    docker image inspect --format "{{json .RootFS.Layers}}" '"${IMG}"' 2>/dev/null | md5sum | awk "{print \$1}" | tee /dev/stderr || echo "IMG_MISSING"
    echo "=== 2. WEIGHT ==="
    SHK=$(ls '"${MODEL_DIR}"'/model-*.safetensors 2>/dev/null | wc -l)
    echo "shards=${SHK}"
    for f in config.json hf_quant_config.json tokenizer.json generation_config.json; do
      [ -f "'"${MODEL_DIR}"'/$f" ] && echo "$f OK" || echo "$f MISSING"
    done
    echo "=== 3. PORTS ==="
    for p in 8010 8011 26000; do
      if ss -tln 2>/dev/null | grep -q ":$p "; then echo "$p BUSY"; else echo "$p FREE"; fi
    done
    echo "=== 4. NCCL ==="
    ls -la /opt/nccl-ringonly/libnccl.so.2.30.7 >/dev/null 2>&1 && echo "ringonly OK" || echo "ringonly MISSING"
    ls -la /opt/aicad-prod/lib/libncclpin.so >/dev/null 2>&1 && echo "shim OK" || echo "shim MISSING"
    INMAP=$(docker run --rm \
      -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro \
      -v /opt/aicad-prod/lib/libncclpin.so:/opt/libncclpin.so:ro \
      -e LD_LIBRARY_PATH=/opt/nccl-ringonly \
      -e LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2" \
      '"${IMG}"' bash -c "grep libnccl /proc/self/maps 2>/dev/null | grep -oE \"libnccl\\.so\\.[0-9.]+|ncclringonly\" | sort -u | tr \"\\n\" \" \"" 2>/dev/null)
    echo "in-container-nccl: ${INMAP:-NO_NCCL_MAPPED}"
    echo "=== 5. MEM ==="
    free -h | awk "/^Mem:/{print \"avail=\"\$7}"
    echo "=== 6. VLLM ==="
    docker ps --format "{{.Names}} {{.Status}}" 2>/dev/null | grep "^vllm-tp4-" || echo "vllm-tp4 NOT RUNNING"
  ' 2>&1)
  echo "$OUT"
  echo
  # 汇总
  LAYER_MD5["$ip"]=$(echo "$OUT" | grep -A1 "IMAGE+LAYER_MD5" | grep -E '^[0-9a-f]{32}' | head -1)
  WEIGHT_SHK["$ip"]=$(echo "$OUT" | grep -oE 'shards=[0-9]+' | cut -d= -f2)
  PORTS_OK["$ip"]=$(echo "$OUT" | grep -cE ' (8010|8011|26000) FREE')
  NCCL_OK["$ip"]=$(echo "$OUT" | grep -cE 'in-container-nccl:.*2\.30\.7')
  MEM_AVAIL["$ip"]=$(echo "$OUT" | grep -oE 'avail=[0-9.]+[A-Z]?' | head -1)
  VLLM_STATE["$ip"]=$(echo "$OUT" | grep -E '^(vllm-tp4|vllm-tp4 NOT)' | head -1)
done

echo
echo "========================== [preflight] 汇总 =========================="
printf "%-16s %-10s %-8s %-12s %-8s %-10s %s\n" "NODE" "LAYER_MD5" "SHARDS" "PORTS_FREE" "NCCL230" "MEM" "VLLM"
ALL_OK=1
for ip in "${NODES[@]}"; do
  lm="${LAYER_MD5[$ip]:-NA}"
  lm_ok="NG"; [ "$lm" = "$EXPECT_LAYER_MD5" ] && lm_ok="OK" || ALL_OK=0
  sh="${WEIGHT_SHK[$ip]:-0}"; sh_ok="NG"; [ "$sh" = "48" ] && sh_ok="OK" || ALL_OK=0
  pf="${PORTS_OK[$ip]:-0}"; pf_ok="NG"; [ "$pf" = "3" ] && pf_ok="OK" || ALL_OK=0
  nc="${NCCL_OK[$ip]:-0}"; nc_ok="NG"; [ "$nc" -ge 1 ] && nc_ok="OK" || ALL_OK=0
  printf "%-16s %-10s %-8s %-12s %-8s %-10s %s\n" "$ip" "${lm_ok}" "$sh" "${pf}/3" "$nc_ok" "${MEM_AVAIL[$ip]:-NA}" "${VLLM_STATE[$ip]:-NA}"
done
echo "---------------------------------------------------------------------"
if [ "$ALL_OK" = "1" ]; then
  echo "[preflight] 全部通过 ✔ （正式运行前仍需按互斥流程 stop vLLM + 门禁）"
else
  echo "[preflight] 存在 NG 项 — 请逐项核查后再正式运行" >&2
  exit 1
fi
