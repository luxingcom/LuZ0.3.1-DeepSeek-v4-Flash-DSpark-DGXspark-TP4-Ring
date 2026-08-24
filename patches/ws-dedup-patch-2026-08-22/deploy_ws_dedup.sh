#!/bin/bash
# deploy_ws_dedup.sh — 部署 b12x wrapper 跨层去重补丁到四节点
# (overlay bind mount 模式, 参照 deploy_fp8ar.sh 先例; .bak 留档 + checker 校验)
#
# 前置: /tmp/_ws_dedup/flashinfer_b12x_moe.py 已就位 (补丁文件, env 门控默认 off)
#
# !! 本脚本只在维护/实验窗口执行 —— 部署本身会修改 start 脚本 (需重启才生效)。
# !! 补丁默认零行为变化: VLLM_B12X_SHARED_WRAPPER 未设置/0 时 off 路径与原文件
# !! AST 级等价 (L1 T2d/T3c 已验证)。生产 W4A16 路径 (flashinfer_b12x → B12xExperts)
# !! 不经过此文件, 挂载本身对生产无影响。
#
# 回滚: cp $SCRIPT.bak-wsdedup-20260822 $SCRIPT (或删掉注入的 env + bind 行)
set -u
TAG="wsdedup-20260822"
W=/tmp/_ws_dedup
PROD=<INSTALL_DIR>
OVERLAY_DIR=$PROD/overlay-wsdedup
TARGET_REL=vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
CONT_PATH=/usr/local/lib/python3.12/dist-packages/$TARGET_REL

echo "[wsdedup] 1/4 分发 overlay 文件到四节点"
for h in node01 node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mkdir -p $OVERLAY_DIR" || exit 1
done
for h in node01 node01 node01 node01; do
  scp -q $W/flashinfer_b12x_moe.py "$h:$OVERLAY_DIR/flashinfer_b12x_moe.py" || exit 1
done
echo "-- md5 校验 (四节点应一致):"
md5sum $W/flashinfer_b12x_moe.py
for h in node01 node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "md5sum $OVERLAY_DIR/flashinfer_b12x_moe.py" || exit 1
done

inject() {  # $1=script path, $2=host (for log)
  local S="$1"
  cp -n "$S" "$S.bak-$TAG"
  # 幂等: 已注入则跳过
  if grep -q "overlay-wsdedup" "$S"; then
    echo "  [skip] $2 已注入"; return 0
  fi
  # 1) env 注入 (显式 off; 窗口实验时改 0→1)
  sed -i "s|  -e 'VLLM_TRITON_MLA_SPARSE=1'|  -e 'VLLM_TRITON_MLA_SPARSE=1'\n  -e 'VLLM_B12X_SHARED_WRAPPER=0'|" "$S"
  # 2) bind mount 注入
  sed -i "s|  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro|  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro\n  -v $OVERLAY_DIR/flashinfer_b12x_moe.py:$CONT_PATH:ro|" "$S"
  bash -n "$S" || { echo "  [error] $2 语法错误"; return 1; }
  grep -n "VLLM_B12X_SHARED_WRAPPER\|overlay-wsdedup" "$S"
}

echo "[wsdedup] 2/4 注入 head 脚本 (01)"
inject $PROD/scripts/start_tp4_head.sh node01 || exit 1

echo "[wsdedup] 3/4 注入 worker 脚本 (02-04)"
for h in node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "bash -s" <<'REMOTE' || exit 1
    set -e
    TAG=wsdedup-20260822
    PROD=<INSTALL_DIR>
    OVERLAY_DIR=$PROD/overlay-wsdedup
    CONT_PATH=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
    S=$PROD/scripts/start_tp4_worker.sh
    cp -n "$S" "$S.bak-$TAG"
    if grep -q "overlay-wsdedup" "$S"; then echo "  [skip] 已注入"; exit 0; fi
    sed -i "s|  -e 'VLLM_TRITON_MLA_SPARSE=1'|  -e 'VLLM_TRITON_MLA_SPARSE=1'\n  -e 'VLLM_B12X_SHARED_WRAPPER=0'|" "$S"
    sed -i "s|  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro|  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro\n  -v $OVERLAY_DIR/flashinfer_b12x_moe.py:$CONT_PATH:ro|" "$S"
    bash -n "$S"
    grep -n "VLLM_B12X_SHARED_WRAPPER\|overlay-wsdedup" "$S"
REMOTE
done

echo "[wsdedup] 4/4 checker 校验 (01 head)"
bash $PROD/scripts/check_vllm_script.sh $PROD/scripts/start_tp4_head.sh \
  || echo "[warn] head checker FAIL (需检查)"
for h in node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "bash $PROD/scripts/check_vllm_script.sh $PROD/scripts/start_tp4_worker.sh" \
    || echo "[warn] $h worker checker FAIL (需检查)"
done

echo "[wsdedup] DONE"
echo "NOTE: 生效需重启容器; 窗口实验开启共享池: start 脚本中 VLLM_B12X_SHARED_WRAPPER=0 → 1"
echo "NOTE: W4A4 实验容器 (非标准 start 脚本启动的) 需自行加同样 -v 挂载"
