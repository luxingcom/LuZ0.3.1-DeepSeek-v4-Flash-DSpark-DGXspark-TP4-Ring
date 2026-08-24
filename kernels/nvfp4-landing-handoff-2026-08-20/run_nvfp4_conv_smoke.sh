#!/bin/bash
# 一次性容器跑转换（纯CPU，不分配GPU），先单层试跑
set -e
IMG='<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0'
OUT='<INSTALL_DIR>/nvfp4/models'
mkdir -p "$OUT/smoke-nvfp4-hp"
docker run --rm --name nvfp4-conv-smoke \
  --cpus=12 \
  --entrypoint python3 \
  -v <INSTALL_DIR>/nvfp4/convert_high_precision_nvfp4_stream.py:/cv.py:ro \
  -v /home/<USER>/models/deepseek-v4-flash-0731:/models:ro \
  -v "$OUT/smoke-nvfp4-hp":/out \
  "$IMG" /cv.py --input-dir /models --output-dir /out \
    --mode high --max-layers 1 --validate