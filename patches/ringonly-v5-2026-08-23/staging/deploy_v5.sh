#!/bin/bash
# deploy_v5.sh — stage v5 artifacts to ALL 4 nodes at /tmp/_ringopt/v5/ (staging only).
# NEVER touches /opt/nccl-ringonly (production) or any production config.
# Run on node01 AFTER a successful build + artifact generation (MD5-RECORD-v5.txt
# must exist). Idempotent.
set -eu
SRC=/tmp/nccl-v5-src
STG=/tmp/_ringopt/v5
NODES="node01 node01 node01 node01"   # physical ring order

[ -f $SRC/build/lib/libnccl.so.2.30.7 ] || { echo "FATAL: built lib not found"; exit 1; }
[ -f $STG/MD5-RECORD-v5.txt ] || { echo "FATAL: MD5-RECORD-v5.txt missing (run artifact gen first)"; exit 1; }

mkdir -p $STG/logs $STG/results
cp $SRC/build/lib/libnccl.so.2.30.7 $STG/
cp $SRC/v5-incremental.patch $SRC/v5-full-chain.patch $STG/

for h in $NODES; do
  echo "== staging $h =="
  ssh -o BatchMode=yes -o ConnectTimeout=8 $h "mkdir -p $STG/logs $STG/results"
  scp -q $STG/libnccl.so.2.30.7 $STG/MD5-RECORD-v5.txt \
        $STG/ringopt_scan_v5.py $STG/ringopt_node_v5.sh $STG/run4_v5.sh $STG/w1_matrix.sh \
        $STG/v5-incremental.patch $STG/v5-full-chain.patch $h:$STG/
  ssh -o BatchMode=yes $h "chmod +x $STG/*.sh && md5sum $STG/libnccl.so.2.30.7"
done
echo "== deploy done; verify md5 against MD5-RECORD-v5.txt above =="
