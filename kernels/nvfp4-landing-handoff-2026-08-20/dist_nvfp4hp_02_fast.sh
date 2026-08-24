#!/bin/bash
# head(01) -> 02 无限速续传发送 NVFP4-HP 权重（148G，--partial 续传，200G 内部网络）
set -e
SRC=/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp
ssh -o BatchMode=yes node01 "mkdir -p /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp"
echo "=== 开始无限速 rsync 续传 ==="
rsync -avh --progress --partial --inplace \
  "$SRC/" "node01:/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp/" 2>&1 | tail -25
echo "=== rsync 完成，目标端校验 ==="
ssh -o BatchMode=yes node01 "du -sh /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp; ls /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp/*.safetensors | wc -l"
echo "=== DIST_TO_02_DONE ==="