#!/bin/bash
# SASS gate FINAL: CUTE_DSL_KEEP=cubin,ptx -> dump artifacts -> cuobjdump -sass -> grep gate
LOG=/work/sass_gate_final.txt
: > $LOG
export PATH=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin:$PATH
export CUTE_DSL_KEEP=ptx,cubin
export CUTLASS_KEEP=ptx,cubin
export CUTE_DSL_DUMP_DIR=/work/dump
export CUTLASS_DUMP_DIR=/work/dump
export CUTE_DSL_CACHE_DIR=/work/cache2
export CUTLASS_CACHE_DIR=/work/cache2
rm -rf /work/dump /work/cache2
mkdir -p /work/dump

echo '===== 1. compile with KEEP=ptx,cubin =====' >> $LOG
python sass_gate2.py >> $LOG 2>&1
echo 'COMPILE_RC='$? >> $LOG

echo '===== 2. dumped artifacts =====' >> $LOG
find /work/dump -type f -exec ls -la {} \; >> $LOG 2>&1
find /work/cache2 -type f 2>/dev/null | head -20 >> $LOG

echo '===== 3. disassemble all cubins =====' >> $LOG
CUOBJ=/usr/local/cuda/bin/cuobjdump
NVD=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin/nvdisasm
mkdir -p /work/sass
for f in $(find /work/dump /work/cache2 -name '*.cubin' -type f 2>/dev/null); do
  base=$(basename "$f")
  echo "--- $f ($(stat -c%s "$f") bytes)" >> $LOG
  $CUOBJ -sass "$f" > "/work/sass/${base}.sass" 2>> $LOG
  RC=$?
  LINES=$(wc -l < "/work/sass/${base}.sass" 2>/dev/null)
  echo "cuobjdump rc=$RC lines=$LINES" >> $LOG
  if [ "$RC" -ne 0 ] || [ "$LINES" -lt 5 ]; then
    "$NVD" -c "$f" > "/work/sass/${base}.sass" 2>> $LOG || true
    echo "nvdisasm retry rc=$? lines=$(wc -l < /work/sass/${base}.sass 2>/dev/null)" >> $LOG
  fi
done
ls -la /work/sass >> $LOG

echo '===== 4. grep gate =====' >> $LOG
for s in /work/sass/*.sass; do
  [ -e "$s" ] || continue
  total=$(grep -ic 'mma' "$s" 2>/dev/null)
  e2m1=$(grep -icE 'mma.*e2m1|e2m1.*mma' "$s" 2>/dev/null)
  mxf=$(grep -ic 'mxf4' "$s" 2>/dev/null)
  echo "SASS_FILE=$s mma_total=$total mma_e2m1=$e2m1 mxf4=$mxf" >> $LOG
done

echo '===== 5. sample mma lines from biggest sass =====' >> $LOG
BIG=$(ls -S /work/sass/*.sass 2>/dev/null | head -1)
echo "biggest: $BIG" >> $LOG
grep -iE 'mma' "$BIG" | head -20 >> $LOG
echo '-- header:' >> $LOG
head -8 "$BIG" >> $LOG

echo '===== 6. PTX grep (secondary) =====' >> $LOG
for p in $(find /work/dump /work/cache2 -name '*.ptx' -type f 2>/dev/null); do
  total=$(grep -ic 'mma\.sync' "$p" 2>/dev/null)
  e2m1=$(grep -icE 'mma\.sync.*e2m1|e2m1' "$p" 2>/dev/null)
  echo "PTX_FILE=$p mma_sync=$total e2m1=$e2m1" >> $LOG
  grep -iE 'mma\.sync' "$p" | head -8 >> $LOG
done
echo SASS_DONE >> $LOG
