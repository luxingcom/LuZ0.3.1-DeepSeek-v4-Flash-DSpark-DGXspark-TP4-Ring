#!/bin/bash
# SASS gate v6: disassemble the GEMM object file dump + grep gate
LOG=/work/sass_gate_log3.txt
: > $LOG
export PATH=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin:$PATH
CUOBJ=/usr/local/cuda/bin/cuobjdump
NVD=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin/nvdisasm

echo '===== file info =====' >> $LOG
file /work/cubins/obj1_dump.o >> $LOG 2>&1 || true
ls -la /work/cubins >> $LOG

echo '===== cuobjdump -sass =====' >> $LOG
$CUOBJ -sass /work/cubins/obj1_dump.o > /work/cubins/gemm.sass 2>> $LOG
echo "cuobjdump rc=$? lines=$(wc -l < /work/cubins/gemm.sass 2>/dev/null)" >> $LOG

echo '===== grep gate (SASS) =====' >> $LOG
S=/work/cubins/gemm.sass
total=$(grep -ic 'mma' $S 2>/dev/null)
e2m1=$(grep -icE 'mma.*e2m1|e2m1.*mma' $S 2>/dev/null)
mxf=$(grep -ic 'mxf4' $S 2>/dev/null)
lds=$(grep -ic 'lds' $S 2>/dev/null)
echo "SASS_FILE=$S mma_total=$total mma_e2m1=$e2m1 mxf4=$mxf lds=$lds" >> $LOG
echo '-- sample mma lines:' >> $LOG
grep -iE 'mma' $S | head -20 >> $LOG
echo '-- arch/function header:' >> $LOG
head -20 $S >> $LOG

echo '===== cuobjdump ELF header (arch check) =====' >> $LOG
$CUOBJ /work/cubins/obj1_dump.o 2>&1 | head -20 >> $LOG
echo SASS_DONE >> $LOG
