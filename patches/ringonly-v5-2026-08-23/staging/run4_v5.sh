#!/bin/bash
# run4_v5.sh <tag> <minch> <maxch> [extra_env...]
# Four-node launcher for v5 arms (run on node01). Node order = physical ring:
# 01(rank0)-02(rank1)-04(rank2)-03(rank3). Per-node logs -> /tmp/_ringopt/v5/logs/<tag>_r<i>.log
TAG="$1"; MC="$2"; XC="$3"; shift 3
W=/tmp/_ringopt/v5
mkdir -p $W/logs
for spec in "node01 0" "node01 1" "node01 2" "node01 3"; do
  set -- $spec
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" \
    "bash $W/ringopt_node_v5.sh $2 $MC $XC $TAG $* > $W/logs/${TAG}_r$2.log 2>&1" &
done
wait
echo "===== RUN4_V5 $TAG DONE $(date -u +%T) ====="
grep -h "\[ringopt-v5" $W/logs/${TAG}_r0.log | head -20
