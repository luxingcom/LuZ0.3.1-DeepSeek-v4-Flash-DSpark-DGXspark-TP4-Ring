#!/usr/bin/env bash
# ============================================================================
# routeB Phase 0：环境准备（CUTLASS Python DSL）
# 用法：bash setup_routeb_env.sh
# 环境：DGX Spark GB10（sm_121a）/ CUDA 13.x / vLLM 0.26 生产容器
#
# 修复记录（Task #13，2026-08-20，precheck 行动清单）：
#   A10 driver 版本仅 echo → 加 ≥580.142 比较校验 + exit 1 fail-fast
#   A11 备份 cp 前加存在性检查（.bak 已存在则跳过，防覆盖干净备份）
#   A16 --no-deps 假设容器预装 CUDA13 runtime → 显式安装 nvidia-cuda-runtime-cu13
#   A21 备份 cp 加 -p（保留权限/时间戳）
#
# ⚠ 版本提示（2026-08-20 现场核验，供 architect 决策）：
#   生产镜像 dspark-vllm-gx10:0.2.1-v026.0 内已装 nvidia-cutlass-dsl==4.5.2
#   （libs-base/cu13=4.5.2，libs-core/cu12=4.6.0 混装），且 4.5.2 的
#   warp/mma.py admissible_archs 已含 Arch.sm_121a——patch 在 4.5.2 上是 no-op。
#   下列钉版 4.4.2 安装为"降级"，会重新引入 patch 需求；是否保留钉版
#   由 architect 版本决策定，本脚本仅提示不拦截。
# ============================================================================
set -euo pipefail

echo "===== [1/4] 检查 CUDA / driver ====="
if ! nvidia-smi > /dev/null 2>&1; then
  echo "❌ nvidia-smi 不可用，fail-fast 退出" >&2
  exit 1
fi
nvidia-smi | grep -E "Driver|CUDA" || true
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver: $DRIVER_VER  (要求 ≥580.142：sm121 ISA fallback 修复 + UMA 内存报告修复)"

# ---- A10：driver ≥580.142 fail-fast 校验 ----
MIN_DRIVER="580.142"
if [ -z "${DRIVER_VER:-}" ] || [ "$DRIVER_VER" = "?" ]; then
  echo "❌ 无法获取 driver 版本，fail-fast 退出" >&2
  exit 1
fi
if printf '%s\n%s\n' "$MIN_DRIVER" "$DRIVER_VER" | sort -V | head -1 | grep -qx "$MIN_DRIVER"; then
  echo "✅ Driver $DRIVER_VER ≥ $MIN_DRIVER，校验通过"
else
  echo "❌ Driver $DRIVER_VER < 要求 $MIN_DRIVER，fail-fast 退出" >&2
  exit 1
fi

echo ""
echo "===== [2/4] 安装 CUTLASS Python DSL（cu13 配套）====="
# 版本现状提示（只读，不拦截）
INSTALLED_CUTLASS=$(pip show nvidia-cutlass-dsl 2>/dev/null | awk '/^Version:/{print $2}' || true)
if [ -n "$INSTALLED_CUTLASS" ]; then
  echo "ℹ️  当前已装 nvidia-cutlass-dsl=$INSTALLED_CUTLASS"
  if [ "$INSTALLED_CUTLASS" != "4.4.2" ]; then
    echo "⚠️  钉版 4.4.2 将覆盖/降级当前 $INSTALLED_CUTLASS；若 ≥4.5 则 mma.py patch 不再必要（见文件头注释）"
  fi
fi
pip install --no-deps nvidia-cutlass-dsl-libs-cu13==4.4.2
# ---- A16：显式安装 CUDA13 runtime（原 --no-deps 假设容器预装，不保证成立）----
# 注：--no-deps 避免污染生产依赖树（ai-muninn 配方第 1 层）
pip install --no-deps nvidia-cuda-runtime-cu13

echo ""
echo "===== [3/4] 验证 import ====="
python - <<'PY'
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
try:
    import cutlass
    print(f"✅ cutlass DSL 版本: {cutlass.__version__}")
    from cutlass.cute import *
    print("✅ cutlass.cute 导入成功")
except Exception as e:
    print(f"❌ import 失败: {e}")
    print("   排查: LD_LIBRARY_PATH 是否含 /usr/local/cuda/lib64")
    raise
PY

echo ""
echo "===== [4/4] 备份待 patch 文件 ====="
MMA_PY=$(python -c "import cutlass, os; print(os.path.join(os.path.dirname(cutlass.__file__), 'cute/nvgpu/warp/mma.py'))" 2>/dev/null || echo "")
if [ -n "$MMA_PY" ] && [ -f "$MMA_PY" ]; then
  # ---- A11 + A21：备份存在则跳过（防覆盖干净备份）；cp -p 保留权限/时间戳 ----
  if [ -f "${MMA_PY}.bak-routeb" ]; then
    echo "⏭️  备份已存在，跳过（保留最早的干净备份）: ${MMA_PY}.bak-routeb"
  else
    cp -p "$MMA_PY" "${MMA_PY}.bak-routeb"
    echo "✅ 已备份: ${MMA_PY}.bak-routeb"
  fi
else
  echo "⚠️ 未找到 mma.py（待 Phase 1 patch 脚本定位）"
fi

echo ""
echo "===== Phase 0 完成 ====="
echo "下一步：python patch_cutlass_dsl_sm121a.py（Phase 1）"
echo "（若 DSL ≥4.5：admissible_archs 已含 sm_121a，Phase 1 将报'无变更'，属预期）"
