#!/bin/bash
# SASS gate v4 runner: explicit attribute probing + PATH fix for cuobjdump
LOG=/work/sass_gate_log2.txt
: > $LOG
export PATH=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin:$PATH
echo '===== 1. intercept compile =====' >> $LOG
python sass_gate2.py >> $LOG 2>&1
echo 'INTERCEPT_RC='$? >> $LOG
ls -la /work/cubins >> $LOG 2>&1

echo '===== 2. disassemble =====' >> $LOG
CUOBJ=/usr/local/cuda/bin/cuobjdump
NVD=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin/nvdisasm
for f in /work/cubins/*.cubin; do
  [ -e "$f" ] || continue
  echo "--- $f" >> $LOG
  "$CUOBJ" -sass "$f" > "/work/cubins/$(basename $f).sass" 2>> $LOG
  RC=$?
  LINES=$(wc -l < "/work/cubins/$(basename $f).sass" 2>/dev/null)
  echo "cuobjdump rc=$RC lines=$LINES" >> $LOG
  if [ "$RC" -ne 0 ] || [ "$LINES" -lt 5 ]; then
    "$NVD" -c "$f" > "/work/cubins/$(basename $f).sass" 2>> $LOG || true
    echo "nvdisasm retry rc=$? lines=$(wc -l < /work/cubins/$(basename $f).sass 2>/dev/null)" >> $LOG
  fi
done

echo '===== 3. grep gate (SASS) =====' >> $LOG
for s in /work/cubins/*.sass; do
  [ -e "$s" ] || continue
  total=$(grep -ic 'mma' "$s" 2>/dev/null)
  e2m1=$(grep -icE 'mma.*e2m1|e2m1.*mma' "$s" 2>/dev/null)
  mxf=$(grep -ic 'mxf4' "$s" 2>/dev/null)
  echo "SASS_FILE=$s mma_total=$total mma_e2m1=$e2m1 mxf4=$mxf" >> $LOG
  echo '-- sample mma lines:' >> $LOG
  grep -iE 'mma' "$s" | head -10 >> $LOG
done

echo '===== 4. grep gate (PTX, secondary evidence) =====' >> $LOG
for p in /work/cubins/*.ptx; do
  [ -e "$p" ] || continue
  total=$(grep -ic 'mma\.sync' "$p" 2>/dev/null)
  e2m1=$(grep -ic 'e2m1' "$p" 2>/dev/null)
  both=$(grep -icE 'mma\.sync.*e2m1|e2m1.*mma\.sync' "$p" 2>/dev/null)
  echo "PTX_FILE=$p mma_sync=$total e2m1=$e2m1 mma_sync_e2m1=$both" >> $LOG
  grep -iE 'mma\.sync.*e2m1|e2m1.*mma' "$p" | head -5 >> $LOG
done
echo SASS_DONE >> $LOG
