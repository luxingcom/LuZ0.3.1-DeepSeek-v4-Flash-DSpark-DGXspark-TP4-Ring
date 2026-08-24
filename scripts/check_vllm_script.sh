#!/bin/bash
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
[ -n "$SCRIPT" ] || { echo "用法: $0 <script_path>"; exit 2; }
[ -f "$SCRIPT" ] || { echo "✗ 脚本不存在: $SCRIPT"; exit 1; }

FAIL=0

echo "[check] $SCRIPT"

# A1. bash 语法
if ! bash -n "$SCRIPT" 2>/tmp/check_vllm_err; then
  echo "  ✗ bash -n 语法错误: $(cat /tmp/check_vllm_err)"; FAIL=1
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

# B1. 关键参数 (A 组生产配置, 2026-08-10 用户批准)
for key in "max-model-len 768000" "max-num-seqs 12" "VLLM_USE_BREAKABLE_CUDAGRAPH=0" "LD_PRELOAD=/opt/libncclpin.so"; do
  if ! grep -qF "$key" "$SCRIPT"; then
    echo "  ✗ 缺关键参数: $key"; FAIL=1
  fi
done
[ $FAIL -eq 0 ] || echo "  ✓ 关键参数检查完成"

# B2. 依赖文件
for f in <INSTALL_DIR>/lib/libncclpin.so <INSTALL_DIR>/models/deepseek-v4-flash-0731/config.json; do
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
    echo "  ✗ 当前 HOME=/root (sudo?) 但脚本使用 \$HOME 挂载, 必须 HOME=/home/<USER> 运行"; FAIL=1
  else
    echo "  ✓ HOME=$HOME (非 root, \$HOME 挂载安全)"
  fi
fi

if [ $FAIL -eq 0 ]; then
  echo "[check] ✅ 全部通过"
  exit 0
else
  echo "[check] ❌ 发现 $FAIL 类问题, 请修复后重试"
  exit 1
fi
