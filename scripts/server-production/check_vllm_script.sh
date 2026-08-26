#!/bin/bash
# =============================================================
# SCRIPT: check_vllm_script.sh
# VERSION: v1.5-r12
# USAGE: bash check_vllm_script.sh <script_path>  (已带 -h/--help + exit 2)
# ROLE: vLLM 启动参数完整性自检 (四机, 启动脚本前置调用与 CI)
# HOST: dgxspark01~04
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md
# DOCS: file:///opt/aicad-prod/docs/ops/tools-index.md
# EXITCODES: 0=通过 1=失败 2=用法错误
# R12 (2026-08-26): B1 对齐 live head/worker - max-num-batched-tokens 4096->8192,
#                 gpu-memory-utilization 0.82->0.78 (与叶子脚本一致, 门禁放行);
#                 修正 C 段输出逻辑 (FAIL=1 时误报 '检查完成')
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
# check_vllm_script.sh - vLLM 启动脚本自检工具 (2026-08-10 加固)
# 目的: 防止三类启动失败复发 (2026-08-10 重启事故复盘):
#   A. 行尾注释吞掉续行符 (docker run / SERVE_CMD 区 # ... \)
#   B. 关键参数/挂载缺失 (max-model-len / LD_PRELOAD / libncclpin)
#   C. sudo 下 $HOME 重置导致挂载源失效
# 用法: check_vllm_script.sh <script_path>
# 返回: 0=通过 1=失败 (供启动脚本前置调用与 CI)
# =============================================================
set -u
SCRIPT="${1:-}"
if [ -z "$SCRIPT" ]; then
  # 2026-08-23 luz031: 无参自动发现本机 start 脚本（此前无参静默 exit 2 曾误报 FAIL）
  if [ "$(hostname)" = "dgxspark01" ]; then
    _cands="/opt/aicad-prod/scripts/start_tp4_head.sh /opt/aicad-prod/scripts/start_tp4_worker.sh"
  else
    _cands="/opt/aicad-prod/scripts/start_tp4_worker.sh /opt/aicad-prod/scripts/start_tp4_head.sh"
  fi
  for _c in $_cands; do
    [ -f "$_c" ] && SCRIPT="$_c" && break
  done
fi
[ -n "$SCRIPT" ] || { echo "用法: $0 <script_path>  (无参时自动发现本机 start_tp4_head.sh / start_tp4_worker.sh)"; exit 2; }
[ -f "$SCRIPT" ] || { echo "✗ 脚本不存在: $SCRIPT"; exit 1; }

FAIL=0

echo "[check] $SCRIPT"

# A1. bash 语法
if ! bash -n "$SCRIPT" 2>/tmp/check_vllm_err.$$; then
  echo "  ✗ bash -n 语法错误: $(cat /tmp/check_vllm_err.$$)"; FAIL=1
else
  echo "  ✓ 语法通过"
fi

# A2. 注释吞续行扫描 (行内 # 注释在行尾 \ 之前 => 续行失效)
BAD=$(grep -nE "^[[:space:]]*(-[edv]|--)[^#]*#[^\\\\]*\\\\[[:space:]]*$" "$SCRIPT" 2>/dev/null | head -5)
if [ -n "$BAD" ]; then
  echo "  ✗ 发现 注释吞续行 行 (需移除行尾注释或拆分):"
  echo "$BAD"
  FAIL=1
else
  echo "  ✓ 无注释吞续行"
fi

# A3. 行尾反斜杠尾随空格检查 (\  换行 => 续行失效)
TRAIL=$(awk '/\\[ ]+$/ {print NR": "$0}' "$SCRIPT" 2>/dev/null | head -3)
if [ -n "$TRAIL" ]; then
  echo "  ✗ 行尾反斜杠后带尾随空格 (续行将失效):"
  echo "$TRAIL"
  FAIL=1
else
  echo "  ✓ 无尾随空格续行"
fi

# A4. 编排器识别 (start_*_cluster.sh / ssh 调度叶子脚本 => 编排器, 非直接 serve 脚本)
#    2026-08-15 SRE: 修复对 start_tp4_cluster.sh 误报缺关键参数 (编排器不含直接 serve 参数)
#    对 head/worker 叶子脚本仍严格检查 B1
ORCH=0
ORCH_REASON=""
if echo "$SCRIPT" | grep -qE 'start_.*_cluster\.sh$'; then
  ORCH=1; ORCH_REASON="文件名含 cluster"
elif grep -qE 'start_tp4_(head|worker)\.sh' "$SCRIPT" && grep -qw ssh "$SCRIPT"; then
  ORCH=1; ORCH_REASON="ssh 调度叶子 head/worker 脚本"
fi
if [ "$ORCH" = "1" ]; then
  echo "  ℹ 编排器脚本 ($ORCH_REASON): B1 关键参数由叶子 head/worker 脚本负责, 此处跳过参数完整性 (标注 OK)"
fi

# B1. 关键参数 (2026-08-26 r12 与 live 对齐: bt 8192 + gmu 0.78 + seqs12 + capture 96;
#     2026-08-12 r11 初版: seqs6 + capture 1..64/max64)
if [ "$ORCH" = "1" ]; then
  echo "  ✓ 关键参数完整性 (编排器: 由叶子 head/worker 脚本负责, 跳过)"
else
for key in "max-model-len 600000" "gpu-memory-utilization 0.78" "max-num-seqs 12" "max-num-batched-tokens 8192" "VLLM_USE_BREAKABLE_CUDAGRAPH=1" "LD_PRELOAD=/opt/libncclpin.so" "max-cudagraph-capture-size 96" "cudagraph-capture-sizes 1 2 4 8 16 24 32 36 40 48 56 64 72 80 88 96"; do
  if ! grep -qF "$key" "$SCRIPT"; then
    echo "  ✗ 缺关键参数: $key"; FAIL=1
  fi
done
fi
[ $FAIL -eq 0 ] && echo "  ✓ 关键参数检查完成"

# B2. 依赖文件
for f in /opt/aicad-prod/lib/libncclpin.so /opt/aicad-prod/models/deepseek-v4-flash-0731/config.json; do
  if [ ! -e "$f" ]; then
    echo "  ✗ 缺依赖: $f"; FAIL=1
  fi
done
echo "  ✓ 依赖文件检查完成"

# C1. SERVE_CMD 展开长度 (短于 300 字符 => 续行断裂)
if grep -q "^SERVE_CMD=" "$SCRIPT"; then
  LEN=$(sed -n '/^SERVE_CMD=/,/--master-port [0-9]*"/p' "$SCRIPT" | wc -c)
  if [ "$LEN" -lt 300 ]; then
    echo "  ✗ SERVE_CMD 展开异常 (长度 $LEN < 300, 续行断裂?)"; FAIL=1
  else
    echo "  ✓ SERVE_CMD 完整 (长度 $LEN)"
  fi
fi

# C2. $HOME 引用检查 (脚本使用 $HOME 挂载 => 禁止在 sudo 下运行)
if grep -q '\$HOME' "$SCRIPT"; then
  if [ "$HOME" = "/root" ]; then
    echo "  ✗ 当前 HOME=/root (sudo?) 但脚本使用 \$HOME 挂载, 必须 HOME=/home/liuxiaoya 运行"; FAIL=1
  else
    echo "  ✓ HOME=$HOME (非 root, \$HOME 挂载安全)"
  fi
fi

# D1. docker run 重启策略防回退 (2026-08-11): head/worker 叶子脚本必须显式 --restart no
#     防止回退 unless-stopped (容器崩溃被 docker 拉起会与编排清理/自检逻辑冲突, 见重启事故复盘)
#     作用域仅限 vLLM head/worker 叶子脚本 (vllm-envE-node/worker);
#     embed 用 unless-stopped 是有意设计 (03/04 生产 embed 池靠 docker restart policy 开机自启, 见 start_embed_8022.sh)
if grep -qE "vllm-envE-(node|worker)" "$SCRIPT" && grep -q "docker run" "$SCRIPT"; then
  if grep -qE -- "--restart[[:space:]]+no" "$SCRIPT"; then
    echo "  ✓ docker run 含 --restart no (防回退)"
  elif grep -qE -- "--restart[[:space:]]+(unless-stopped|always)" "$SCRIPT"; then
    echo "  ✗ docker run 使用 unless-stopped/always — 必须改为 --restart no (防回退)"; FAIL=1
  else
    echo "  ✗ docker run 缺少 --restart no (head/worker 必须显式声明)"; FAIL=1
  fi
fi

if [ $FAIL -eq 0 ]; then
  echo "[check] ✅ 全部通过"
  exit 0
else
  echo "[check] ❌ 发现 $FAIL 类问题, 请修复后重试"
  exit 1
fi
