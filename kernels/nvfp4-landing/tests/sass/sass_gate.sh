#!/usr/bin/env bash
# ============================================================================
# sass_gate.sh —— NVFP4 SASS 硬门禁（CI/发版用）
# 用途：对已 dump 的 SASS 文本做门禁判定，确认编译出原生 FP4 / 张量核心 MMA
#       指令，而非降级为 fp32 (FFMA)。用于验证 dot_scaled / mmaf 路径真正落地。
# 运行：
#   cuobjdump --dump-sass <triton_kernel.cubin> > /tmp/nvfp4_sass.txt
#   bash sass_gate.sh /tmp/nvfp4_sass.txt
# 期望输出：
#   PASS  → 命中原生 FP4/tensor-core MMA（mma + e2m1 / tcgen05 / mmaf），exit 0
#   FAIL  → 只有 FFMA/FADD/FMUL（降级 fp32），exit 1
# ============================================================================
set -u

SASS_FILE="${1:?用法: bash sass_gate.sh <sass.txt>}"
if [ ! -f "$SASS_FILE" ]; then
  echo "FAIL: 找不到 SASS 文件 $SASS_FILE"
  exit 1
fi

MMA=$(grep -icE "mma" "$SASS_FILE" || true)
E2M1=$(grep -icE "e2m1" "$SASS_FILE" || true)
MMAF=$(grep -icE "mmaf" "$SASS_FILE" || true)
TCGEN=$(grep -icE "tcgen05|tcgen" "$SASS_FILE" || true)
FFMA=$(  grep -icE "FFMA|FADD|FMUL|FMUL3" "$SASS_FILE" || true)

echo "=== NVFP4 SASS 门禁 ==="
echo "mma 指令数      : $MMA"
echo "e2m1 特征数     : $E2M1"
echo "mmaf 特征数     : $MMAF"
echo "tcgen 指令数    : $TCGEN"
echo "ffma/fadd/fmul  : $FFMA"

# 判定：命中原生 FP4 张量核心路径
if [ "$MMA" -gt 0 ] && { [ "$E2M1" -gt 0 ] || [ "$MMAF" -gt 0 ] || [ "$TCGEN" -gt 0 ]; }; then
  echo "PASS: 命中原生 FP4/tensor-core MMA（e2m1/mmaf/tcgen 任一成立）→ FP4 MMA 生效"
  exit 0
else
  echo "FAIL: 未命中原生 FP4 MMA（仅 FFMA=$FFMA 或 mma 数=$MMA）→ dot_scaled 可能降级为 fp32/bf16"
  exit 1
fi