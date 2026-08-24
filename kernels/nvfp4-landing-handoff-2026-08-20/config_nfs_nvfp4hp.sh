#!/bin/bash
# NVFP4-HP 权重 NFS 体系配置（01/02 导出 + 03/04 挂载 + symlink）
# 对齐现有 -nvfp4 模式
set -e
SUDO="echo '<PASSWORD>' | sudo -S"

echo "========== [01] 导出 -nvfp4-hp =========="
ssh -o BatchMode=yes node01 "$SUDO bash -c \"
  if ! grep -q 'deepseek-v4-flash-0731-nvfp4-hp' /etc/exports; then
    echo '/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp <NODE_IP>/24(ro,sync,no_subtree_check) <NODE_IP>/24(ro,sync,no_subtree_check)' >> /etc/exports
    echo '  appended 01 exports'
  else
    echo '  01 exports already has nvfp4-hp'
  fi
  exportfs -ra
  exportfs | grep nvfp4-hp
\""

echo "========== [02] 导出 -nvfp4-hp =========="
ssh -o BatchMode=yes node01 "$SUDO bash -c \"
  if ! grep -q 'deepseek-v4-flash-0731-nvfp4-hp' /etc/exports; then
    echo '/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp <NODE_IP>/30(ro,sync,no_subtree_check)' >> /etc/exports
    echo '  appended 02 exports'
  else
    echo '  02 exports already has nvfp4-hp'
  fi
  exportfs -ra
  exportfs | grep nvfp4-hp
\""

echo "========== [03] 从 01 挂载 -nvfp4-hp =========="
ssh -o BatchMode=yes node01 "$SUDO bash -c \"
  mkdir -p <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp
  if ! grep -q 'deepseek-v4-flash-0731-nvfp4-hp' /etc/fstab; then
    echo '<NODE_IP>:/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp nfs4 ro,vers=4.2,hard,timeo=600,nconnect=4 0 0' >> /etc/fstab
    echo '  appended 03 fstab'
  fi
  mount -a || echo '  mount -a (03) warning'
  ln -sfn <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp <INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4-hp
  echo '  symlink done'
  ls -la <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp/ | head -3
  mount | grep nvfp4-hp
\""

echo "========== [04] 从 02 挂载 -nvfp4-hp =========="
ssh -o BatchMode=yes node01 "$SUDO bash -c \"
  mkdir -p <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp
  if ! grep -q 'deepseek-v4-flash-0731-nvfp4-hp' /etc/fstab; then
    echo '<NODE_IP>:/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp nfs4 ro,vers=4.2,hard,timeo=600,nconnect=4 0 0' >> /etc/fstab
    echo '  appended 04 fstab'
  fi
  mount -a || echo '  mount -a (04) warning'
  ln -sfn <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp <INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4-hp
  echo '  symlink done'
  ls -la <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp/ | head -3
  mount | grep nvfp4-hp
\""

echo "========== NFS 配置完成 =========="