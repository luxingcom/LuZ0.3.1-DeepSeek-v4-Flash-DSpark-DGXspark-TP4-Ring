#!/bin/bash
# =============================================================
# SCRIPT: shim-deploy.sh  v1.0-r12
# ROLE: 四机 libncclpin.so (shim) 部署/校验/回滚
# HOST: dgxspark01 (本地运行; 免密 ssh -> 02/03/04)
# USAGE:
#   bash shim-deploy.sh check                    # 四机核验当前 MD5 是否 = v8
#   bash shim-deploy.sh deploy <src.so>          # 分发+校验+部署前自动备份(回滚锚点)
#   bash shim-deploy.sh rollback                 # 用 .bak-v7 恢复四机
#   bash shim-deploy.sh diff                     # 打印四机当前 MD5 与锚点列表
# REQUIRE: 远端 sudo 需密码 -> 设环境变量 SHIM_SUDO_PW, 否则交互输入
# EXPECTED_V8_MD5: ce43c688c5164ac7efd5105c94fdab77
# V8 部署: bash shim-deploy.sh deploy <v8.so>  (部署前自动生成 .bak-pre-deploy-<ts>)
# CHANGE: 改脚本须 bash -n + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
# DOCS: file:///opt/aicad-prod/docs/scripts/REFERENCE.md | file:///opt/aicad-prod/deliverables/engineering-assurance/nccl-ringonly-optimization-2026-08-15.md
set -uo pipefail

NODES=(dgxspark01 dgxspark02 dgxspark03 dgxspark04)
DEST=/opt/aicad-prod/lib/libncclpin.so
LIBDIR=/opt/aicad-prod/lib
V7_ANCHOR=/opt/aicad-prod/lib/libncclpin.so.bak-v7
EXPECTED_V8=ce43c688c5164ac7efd5105c94fdab77

SUDO_PW="${SHIM_SUDO_PW:-}"
[ -n "$SUDO_PW" ] || { printf 'sudo 密码: '; read -rs SUDO_PW; printf '\n'; }

runsudo() { # $1=host  $2=cmd
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "echo '$SUDO_PW' | sudo -S -p '' $2"
}

md5_of() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "md5sum $DEST | cut -d' ' -f1"
}

check_all() {
  echo "== 四机 shim MD5 核验 (期望 v8=$EXPECTED_V8)"
  local bad=0
  for h in "${NODES[@]}"; do
    m=$(md5_of "$h")
    if [ "$m" = "$EXPECTED_V8" ]; then echo "  [OK]   $h  $m (v8)"; else echo "  [MISMATCH] $h  ${m:-MISSING}"; bad=1; fi
  done
  [ "$bad" = "0" ] && echo "== 四机一致 = v8 ✅" || echo "== 存在不一致 ⚠️"
  return $bad
}

do_diff() {
  echo "== 四机当前 shim MD5 与锚点"
  for h in "${NODES[@]}"; do
    m=$(md5_of "$h")
    echo "  [$h] 当前=$m"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "ls $LIBDIR/libncclpin.so.bak-v* 2>/dev/null | head -3" 2>/dev/null | sed 's/^/       锚点: /'
  done
}

do_deploy() {
  local src="${1:-}"
  [ -n "$src" ] && [ -f "$src" ] || { echo "✗ 源文件不存在: $src"; exit 2; }
  local srcmd5; srcmd5=$(md5sum "$src" | cut -d' ' -f1)
  echo "== 源: $src  MD5=$srcmd5"
  for h in "${NODES[@]}"; do
    echo "--- [$h] 部署 ---"
    # 1. 部署前备份当前版本为回滚锚点
    runsudo "$h" "cp -a $DEST $LIBDIR/libncclpin.so.bak-pre-deploy-\$(date +%Y%m%d-%H%M%S)" || { echo "FAIL: $h 备份失败, 中止部署"; return 1; }
    # 2. 分发到远端临时目录 (本地 scp 用当前用户免密)
    scp -o BatchMode=yes -o ConnectTimeout=8 "$src" "$h:/tmp/shim-deploy-tmp.so" || { echo "FAIL: $h scp 分发失败, 中止部署"; return 1; }
    # 3. 安装 + 权限 + 校验
    runsudo "$h" "install -m 0755 /tmp/shim-deploy-tmp.so $DEST && rm -f /tmp/shim-deploy-tmp.so" || { echo "FAIL: $h 安装失败, 中止部署"; return 1; }
    local got; got=$(md5_of "$h") || { echo "FAIL: $h MD5 校验失败(ssh 不可达), 中止部署"; return 1; }
    if [ "$got" = "$srcmd5" ]; then echo "  [OK] $h  MD5=$got"; else echo "FAIL: $h MD5 不匹配 got=$got 期望=$srcmd5"; return 1; fi
  done
  echo "== 四机部署成功 =="
  return 0
}

do_rollback() {
  echo "== 用 $V7_ANCHOR 回滚四机"
  local fail=0
  for h in "${NODES[@]}"; do
    echo "--- [$h] ---"
    if ! runsudo "$h" "test -f $V7_ANCHOR && cp -a $V7_ANCHOR $DEST && chmod 0755 $DEST && echo ROLLBACK_OK"; then
      echo "✗ $h 回滚失败 (无 v7 锚点?)"; fail=1; continue
    fi
    echo "  [OK] $h MD5=$(md5_of "$h")"
  done
  [ "$fail" = "0" ] && echo "== 回滚完成 (v7) ==" || echo "== 回滚存在失败 ⚠️ =="
  return $fail
}

case "${1:-}" in
  check)   check_all ;;
  diff)    do_diff ;;
  deploy)  do_deploy "${2:-}" ;;
  rollback) do_rollback ;;
  -h|--help|*)
    sed -n '3,12p' "$0"; exit 0 ;;
esac
