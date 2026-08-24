#!/bin/bash
# ringopt_node_v5.sh <node_rank> <min_ch> <max_ch> <tag> [extra_env ...]
# W1 v5 test runner (per node). Differences vs ringopt_node.sh:
#   * LD_PRELOAD points at the v5 library staged in /tmp/_ringopt/v5 (mounted read-only
#     as /v5lib). Production /opt/nccl-ringonly is NEVER touched or mounted.
#   * NO_PIN=1 drops libncclpin from LD_PRELOAD (default: pin ON, matching the 21.4GB/s
#     baseline methodology; only run pinned arms inside a real maintenance window).
#   * scan script = /work/v5/ringopt_scan_v5.py (sizes to 128MB, CHECK / RINGOPT_SIZES env).
# Extra env examples:
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH   (取证 arm)
#   NCCL_RINGONLY_V5=0                             (v5-off A/B arm)
#   NCCL_IB_QPS_PER_CONNECTION=2                   (QPS2 arm)
set -u
NODE_RANK="$1"; MINCH="$2"; MAXCH="$3"; TAG="$4"; shift 4
IMG="<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0-b12x-recovered-20260820"
W=/tmp/_ringopt
V5=$W/v5
EXTRA=""
for e in "$@"; do EXTRA="$EXTRA -e $e"; done
PRE="/v5lib/libnccl.so.2.30.7"
[ "${NO_PIN:-0}" != "1" ] && PRE="/opt/libncclpin.so $PRE"
docker run --rm --network host --privileged --gpus all --ipc=host \
  -v $W:/work \
  -v $V5:/v5lib:ro \
  -v <INSTALL_DIR>/lib/libncclpin.so:/opt/libncclpin.so:ro \
  -e 'NCCL_ALGO=RING' \
  -e 'NCCL_CROSS_NIC=1' \
  -e 'NCCL_IB_GID_INDEX=3' \
  -e 'NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1' \
  -e 'NCCL_IB_TIMEOUT=1000' \
  -e 'NCCL_IB_RETRY_CNT=7' \
  -e 'NCCL_IB_TOS=46' \
  -e 'NCCL_IGNORE_CPU_AFFINITY=1' \
  -e "NCCL_MIN_NCHANNELS=$MINCH" \
  -e "NCCL_MAX_NCHANNELS=$MAXCH" \
  -e 'NCCL_BUFFSIZE=8388608' \
  -e 'NCCL_NET=IB' \
  -e 'NCCL_IB_SUBNET_AWARE_ROUTING=1' \
  -e 'NCCL_NET_PLUGIN=none' \
  -e 'NCCL_IB_MERGE_NICS=0' \
  -e 'NCCL_SOCKET_IFNAME=enP7s7' \
  -e 'GLOO_SOCKET_IFNAME=enP7s7' \
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu' \
  -e "LD_PRELOAD=$PRE" \
  -e "RINGOPT_TAG=$TAG" \
  $EXTRA \
  --entrypoint python3 "$IMG" -m torch.distributed.run \
  --nnodes 4 --nproc_per_node 1 --node_rank "$NODE_RANK" \
  --master_addr <NODE_IP> --master_port 26189 /work/v5/ringopt_scan_v5.py
