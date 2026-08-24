#!/usr/bin/env bash
# ============================================================================
# SASS 诊断脚本：确认 nvfp4_4w4a_prefill_gemm 的 dot_scaled 是否编译为原生 FP4 MMA
# 环境：DGX Spark（GB10 / sm121a / Triton 3.6.0 / torch 2.11）
# 用法：bash sass_check_prefill_gemm.sh
# ============================================================================
set -e

echo "=== [1/4] 触发内核编译（生成 Triton 缓存 cubin）==="
python3 - <<'PY'
import torch, sys, os
sys.path.insert(0, ".")
import nvfp4_4w4a_prefill_gemm_v12_triton as m
from test_nvfp4_4w4a_prefill_gemm import make_weights

A = torch.randn(256, 4096, device="cuda", dtype=torch.float32)
W, ws = make_weights(4096, 4096, scale=0.5, device="cuda")
m.nvfp4_4w4a_prefill_gemm(A, W, ws)
torch.cuda.synchronize()
print("内核已运行，编译缓存已生成")
PY

echo ""
echo "=== [2/4] 定位最新 Triton cubin ==="
CUBIN=""
for d in "$HOME/.triton/cache" "${TRITON_CACHE_DIR:-}"; do
    if [ -d "$d" ]; then
        CUBIN=$(ls -t "$d"/*/*.cubin 2>/dev/null | head -1)
        [ -n "$CUBIN" ] && break
    fi
done
if [ -z "$CUBIN" ]; then
    echo "未找到 cubin，尝试 python 定位："
    CUBIN=$(python3 - <<'PY'
import glob, os
for pat in [os.path.expanduser("~/.triton/cache/*/*.cubin"), os.path.join(os.environ.get("TRITON_CACHE_DIR","/tmp"), "**/*.cubin")]:
    fs = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)
    if fs:
        print(fs[-1]); break
PY
)
fi
echo "cubin: $CUBIN"

echo ""
echo "=== [3/4] dump SASS 并检索 MMA 指令 ==="
if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "$CUBIN" > /tmp/nvfp4_sass.txt 2>&1 || true
    echo "--- 原生 FP4/张量核心指令（mma / tcgen / e2m1）---"
    grep -iE "mma|tcgen|e2m1" /tmp/nvfp4_sass.txt | head -40 || echo "（无 mma/tcgen 指令）"
    echo ""
    echo "--- 浮点指令统计（FFMA 等）---"
    grep -cE "FFMA|FADD|FMUL" /tmp/nvfp4_sass.txt || true
    echo ""
    echo "--- mma 指令总条数 ---"
    grep -ciE "mma" /tmp/nvfp4_sass.txt || true
else
    echo "cuobjdump 不可用，改用 python (pynvml/nvdisasm) 检查："
    python3 - <<'PY'
import re
try:
    from cuda import cudart  # noqa
except Exception:
    pass
# 兜底：直接打印 cubin 二进制中的指令串特征
with open(r"CUBIN_PLACEHOLDER", "rb") as f:
    data = f.read()
for pat in [b"mma.sync", b"tcgen05", b"e2m1", b"MMA"]:
    print(pat, "->", data.count(pat), "处")
PY
fi

echo ""
echo "=== [4/4] 判定 ==="
echo "✅ 出现 mma.sync.aligned.m16n8k32.f32.e2m1 / tcgen05.mma 系列 → FP4 MMA 生效"
echo "   → 性能问题是调度/BLOCK（调 BLOCK_M/N、GROUP_M、num_warps、num_stages）"
echo "❌ 只有 FFMA/FADD/FMUL → dot_scaled 降级为 fp32 MMA"
echo "   → 升级 Triton 3.7+（修 sm121 e2m1 codegen）或 CUTLASS mmaf_scaled 手写"
