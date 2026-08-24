#!/bin/bash
# head(01) -> 02 分发 NVFP4 高精度权重（148G）
# 限速 ~400Mbps，避免冲击生产 200G 网络；断点续传 rsync
set -e
SRC=/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp
DST=node01:/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp
# 确保目标父目录存在
ssh -o BatchMode=yes node01 "mkdir -p /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp"
echo "=== 开始 rsync 分发 (限速 --bwlimit=50000 KB/s ≈ 400Mbps) ==="
rsync -avh --progress --bwlimit=50000 --partial \
  "$SRC/" "$DST/" 2>&1 | tail -30
echo "=== rsync 完成，目标端校验 ==="
ssh -o BatchMode=yes node01 "du -sh /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp; ls /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp/ | wc -l"
echo "=== DIST_TO_02_DONE ==="