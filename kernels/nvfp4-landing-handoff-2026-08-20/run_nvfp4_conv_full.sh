#!/bin/bash
# 全量 43 层 MXFP4 -> NVFP4 高精度转换（流式逐层，纯 CPU，一次性容器，不放 GPU）
set -e
IMG='<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0'
OUT='<INSTALL_DIR>/nvfp4/models/dsv4f-0731-nvfp4-hp'
rm -rf "$OUT"
mkdir -p "$OUT"
# 后台日志
LOG=<INSTALL_DIR>/nvfp4/models/full_convert.log
nohup docker run --rm --name nvfp4-conv-full \
  --cpus=16 \
  --entrypoint python3 \
  -v <INSTALL_DIR>/nvfp4/convert_high_precision_nvfp4_stream.py:/cv.py:ro \
  -v /home/<USER>/models/deepseek-v4-flash-0731:/models:ro \
  -v "$OUT":/out \
  "$IMG" /cv.py --input-dir /models --output-dir /out \
    --mode high > "$LOG" 2>&1 &
echo "全量转换已启动，日志: $LOG PID: $!"