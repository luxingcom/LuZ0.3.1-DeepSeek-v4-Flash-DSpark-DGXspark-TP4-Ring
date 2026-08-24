#!/bin/bash
# p0_run.sh — Task#24 P0 merged-shape kernel 基准全矩阵（routeB vendored 官方 kernel）
# 运行于生产镜像一次性容器（GPU 独占，生产已停）
set -u
cd /routeb
TOL="${ROUTEB_TOL:-1e-2}"
ITER="${ROUTEB_ITER:-30}"

run() {  # run <tag> <shape> <tile> <sfvec>
  local tag="$1" shape="$2" tile="$3" sfvec="$4"
  echo ""
  echo "########## [$tag] shape=$shape tile=$tile sf_vec=$sfvec tol=$TOL iter=$ITER ##########"
  ROUTEB_TOL="$TOL" python3 p0_bench.py --shape "$shape" --tile "$tile" \
    --sf-vec "$sfvec" --c-dtype Float16 --epi 128,128 \
    --warmup 5 --iterations "$ITER" 2>&1 | tail -12
  echo "########## [$tag] EXIT=$? ##########"
}

# 1. 回归锚点（应复现 ~368）
run anchor_368 "4096,14336,4096" "128,128,128" 32
# 2. NVFP4 vec16 @128³（适配器格式）
run vec16_merge8 "3072,12288,4096" "128,128,128" 16
run vec16_merge4 "6144,24576,4096" "128,128,128" 16
# 3. ★K-tile 256 vec16（本任务最关键未知）
run vec16_merge8_K256 "3072,12288,4096" "128,128,256" 16
run vec16_merge4_K256 "6144,24576,4096" "128,128,256" 16
# 4. MXF4 vec32 对照
run vec32_merge8 "3072,12288,4096" "128,128,128" 32
run vec32_merge4 "6144,24576,4096" "128,128,128" 32
# 5. vec32 K-256（Task#12 已知失败复现，对照用）
run vec32_merge8_K256 "3072,12288,4096" "128,128,256" 32

echo ""
echo "===== P0 ALL DONE ====="
