#!/bin/bash
# 隔离验证容器：fork 加载 NVFP4-HP 权重（B 路线第一道验证）
# 用法: bash verify_fork_nvfp4_load.sh [--moe-backend <val>]
# TP1 单节点, 不联网, 仅测加载+启动, 观察 fork 解析 NVFP4 权重行为
set -euo pipefail
IMG='<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0'
MODEL='<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4-hp'
MOE_BACKEND="${1:-auto}"
NAME="nvfp4-verify"

docker rm -f "$NAME" >/dev/null 2>&1 || true
LOG="/tmp/nvfp4_verify_${MOE_BACKEND//\//_}.log"
echo "=== 日志: $LOG ==="

# 覆盖镜像 ENTRYPOINT(含 vllm serve), 直接用 python 调 vllm
docker run --name "$NAME" \
  --entrypoint python3 \
  --gpus all \
  --shm-size=16g \
  -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/cache/hf-verify \
  -v "$MODEL":/models:ro \
  "$IMG" -c "
import os, sys, time
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
from vllm import LLM
print('=== fork 尝试加载 NVFP4-HP 权重 (moe_backend=$MOE_BACKEND) ===', flush=True)
t0=time.time()
try:
    llm = LLM(
        model='/models',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.60,
        max_model_len=4096,
        enforce_eager=True,
        moe_backend='$MOE_BACKEND',
        kv_cache_dtype='nvfp4_ds_mla',
    )
    print(f'=== 加载成功, 耗时 {time.time()-t0:.1f}s ===', flush=True)
    # 打印模型量化信息
    cfg = llm.llm_engine.model_config
    print('quantization:', cfg.quantization, flush=True)
    print('moe_backend:', getattr(cfg, 'moe_backend', '?'), flush=True)
    # 简单前向
    out = llm.generate(['hello world'], sampling_params={'max_tokens': 8})
    print('=== 前向 OK ===', out[0].outputs[0].text[:20] if out else 'EMPTY', flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print('=== LOAD_FAILED ===', flush=True)
" > "$LOG" 2>&1
echo "=== 容器退出, 查看日志尾部 ==="
tail -45 "$LOG"