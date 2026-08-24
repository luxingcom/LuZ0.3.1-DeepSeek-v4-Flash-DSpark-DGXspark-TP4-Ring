#!/bin/bash
# w1_matrix.sh — W1 maintenance-window test matrix for the v5 ring-forced library.
# Run ON node01 DURING THE WINDOW (needs GPUs on all 4 nodes; production must be down).
#
# Arms (sequential, ~35-45 min total incl. container startup):
#   A1 取证 v5-OFF 8ch   (NCCL_RINGONLY_V5=0, DEBUG=INFO)  -> reproduce stall + capture
#        NCCL's own multi-ring search / rail alternation ring orders ("Ring %02d" lines)
#   A2 取证 v5-ON  8ch   (DEBUG=INFO)                       -> "RING-ONLY v5 ... forced ring" lines,
#        all channels physical 0-1-2-3-0, stall must be gone
#   A3 取证 v5-ON  MIN=4/MAX=16 (DEBUG=INFO)                -> effective channel count at 16
#   B  correctness gate: v5 16ch, CHECK=1, all sizes 64KB-128MB
#   C1 channel scan v5 4ch  (full curve, busbw vs 21.4GB/s @8.4MB baseline)
#   C2 channel scan v5 8ch
#   C3 channel scan v5 16ch
#   D  (optional, uncomment) QPS2@4ch zero-patch arm: NCCL_IB_QPS_PER_CONNECTION=2
#
# JUDGEMENT CRITERIA (W1 gate):
#   J1 stall gone    : A2/C2/C3 AR time at t1024 (8.4MB) is bandwidth-scale (<= ~1.5ms),
#                      NOT the constant 17-20ms stall; small sizes (t24/t96) drop back to
#                      ~100-400us scale.
#   J2 busbw win     : best of {8ch,16ch} busbw @t1024 >= 25 GB/s (target band 25-28).
#   J3 no regression : C1 (v5 4ch) busbw @t1024 within 21.4 +/- 5%.
#   J4 correctness   : arm B reports PASS on every size.
#   J5 rings physical: A2 log shows every channel prev/next = rank-1/rank+1 (mod 4).
# If J1 fails with v5 on -> hypothesis wrong, escalate; do NOT proceed to e2e.
set -u
W=/tmp/_ringopt/v5
cd $W

echo "===== W1 MATRIX START $(date -u) ====="
echo "--- preflight: md5 must match MD5-RECORD-v5.txt on all 4 nodes ---"
for h in node01 node01 node01 node01; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 $h "md5sum $W/libnccl.so.2.30.7" &
done; wait

echo "--- A1 取证: v5 OFF, 8ch (stall reproduction + stock ring orders) ---"
bash run4_v5.sh dbg_v5off_8ch 8 8 RINGOPT_SIZES=t96 NCCL_RINGONLY_V5=0 \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH
echo "--- A2 取证: v5 ON, 8ch (forced physical rings) ---"
bash run4_v5.sh dbg_v5on_8ch 8 8 RINGOPT_SIZES=t96 \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH
echo "--- A3 取证: v5 ON, MIN=4/MAX=16 (effective channel count) ---"
bash run4_v5.sh dbg_v5on_4_16 4 16 RINGOPT_SIZES=t96 \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH

echo "--- B correctness gate: v5 16ch, CHECK=1, 64KB-128MB ---"
bash run4_v5.sh chk_v5_16ch 16 16 CHECK=1

echo "--- C1 channel scan: v5 4ch ---"
bash run4_v5.sh scan_v5_4ch 4 4
echo "--- C2 channel scan: v5 8ch ---"
bash run4_v5.sh scan_v5_8ch 8 8
echo "--- C3 channel scan: v5 16ch ---"
bash run4_v5.sh scan_v5_16ch 16 16

# echo "--- D optional: QPS2 @ 4ch (zero-patch shortcut arm) ---"
# bash run4_v5.sh scan_qps2_4ch 4 4 NCCL_IB_QPS_PER_CONNECTION=2

echo "===== SUMMARY $(date -u) ====="
echo "--- ring orders (A1 stock vs A2 forced) ---"
grep -h "Ring [0-9][0-9] :" logs/dbg_v5off_8ch_r0.log | head -8
grep -h "RING-ONLY v5" logs/dbg_v5on_8ch_r0.log | head -8
echo "--- per-arm key numbers (t1024 = 8.4MB, t4096 = 33.5MB, t16384 = 128MB) ---"
for t in dbg_v5off_8ch dbg_v5on_8ch chk_v5_16ch scan_v5_4ch scan_v5_8ch scan_v5_16ch; do
  echo "[$t]"; grep -h "\[ringopt-v5" logs/${t}_r0.log
done
echo "--- J1 stall check: any t1024/t4096/t16384 ms > 5 in v5-on arms = FAIL ---"
grep -h "\[ringopt-v5" logs/scan_v5_8ch_r0.log logs/scan_v5_16ch_r0.log logs/dbg_v5on_8ch_r0.log \
  | grep -E "t1024|t4096|t16384" || true
echo "===== W1 MATRIX DONE $(date -u) — apply judgement J1-J5 above ====="
