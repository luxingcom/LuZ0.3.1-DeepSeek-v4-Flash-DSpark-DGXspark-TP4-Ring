#!/usr/bin/env bash
# ============================================================================
# routeB Phase 0：环境准备（CUTLASS 4.4.0 Python DSL）
# 用法：bash setup_routeb_env.sh
# 环境：DGX Spark GB10（sm_121a）/ CUDA 13.x / vLLM 0.26 生产容器
# ============================================================================
set -euo pipefail

echo "===== [1/4] 检查 CUDA / driver ====="
nvidia-smi | grep -E "Driver|CUDA" || echo "⚠️ nvidia-smi 不可用"
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "?")
echo "Driver: $DRIVER_VER  (要求 ≥580.142：sm121 ISA fallback 修复 + UMA 内存报告修复)"

echo ""
echo "===== [2/4] 安装 CUTLASS 4.4.0 Python DSL（cu13 配套）====="
pip install --no-deps nvidia-cutlass-dsl-libs-cu13==4.4.2
# 注：--no-deps 避免污染生产依赖树；CUDA 13 runtime libs 若缺需单独装：
#   pip install --no-deps nvidia-cuda-runtime-cu13  （ai-muninn 配方第 1 层备注）

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
  cp "$MMA_PY" "${MMA_PY}.bak-routeb"
  echo "✅ 已备份: ${MMA_PY}.bak-routeb"
else
  echo "⚠️ 未找到 mma.py（待 Phase 1 patch 脚本定位）"
fi

echo ""
echo "===== Phase 0 完成 ====="
echo "下一步：python patch_cutlass_dsl_sm121a.py（Phase 1）"
