#!/bin/bash
# SASS gate v2: find default DSL cache location -> copy db out -> extract cubin -> disassemble
LOG=/work/sass_gate_log.txt
: > $LOG
echo '===== 1. compile (default cache location) =====' >> $LOG
ROUTEB_DIAG=ones ROUTEB_DUMP=0 python bn1_run.py 256,256,512 128,128 128,128,128 Float16 > /work/sass_compile_log.txt 2>&1
echo 'COMPILE_RC='$? >> $LOG

echo '===== 2. locate cache =====' >> $LOG
grep -rn "compiled_cache\|\.cutlass" /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/ --include='*.py' 2>/dev/null | grep -iv test | head -20 >> $LOG
find / -name 'compiled_cache*' -o -name '.cutlass' -type d 2>/dev/null | grep -v proc | head -10 >> $LOG
ls -la /root/.cutlass/cache 2>&1 >> $LOG
ls -la $HOME/.cutlass/cache 2>&1 >> $LOG

echo '===== 3. copy cache out =====' >> $LOG
rm -rf /work/cache_export
mkdir -p /work/cache_export
CAND=$(find / -name 'compiled_cache*' 2>/dev/null | grep -v proc | grep -v /work | head -3)
echo "candidates: $CAND" >> $LOG
for f in $CAND; do
  cp "$f" /work/cache_export/$(echo "$f" | tr '/' '_') 2>> $LOG
done
find /root/.cutlass -type f 2>/dev/null | head -10 >> $LOG
cp -r /root/.cutlass/cache/* /work/cache_export/ 2>> $LOG || true
ls -la /work/cache_export >> $LOG 2>&1

echo '===== 4. sqlite explore =====' >> $LOG
python - >> $LOG 2>&1 <<'PY'
import sqlite3, os, glob
dbs = glob.glob('/work/cache_export/**/*.db', recursive=True) + glob.glob('/work/cache_export/*.db')
print('dbs found:', dbs)
for db in dbs:
    print('=== DB', db, os.path.getsize(db), 'bytes')
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','index')")
    for name, sql in cur.fetchall():
        print('  OBJECT', name)
        print('   ', (sql or '').replace(chr(10), ' ')[:400])
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            cur.execute(f'SELECT COUNT(*) FROM "{t}"')
            n = cur.fetchone()[0]
            print('  TABLE', t, 'rows=', n)
    except Exception as e:
        print('  count err', e)
    con.close()
PY

echo '===== 5. extract ELF blobs =====' >> $LOG
python - >> $LOG 2>&1 <<'PY'
import sqlite3, glob, zlib, os
os.makedirs('/work/cubins', exist_ok=True)
dbs = glob.glob('/work/cache_export/**/*.db', recursive=True) + glob.glob('/work/cache_export/*.db')
idx = 0
for db in dbs:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        cur.execute(f'SELECT * FROM "{t}"')
        cols = [d[0] for d in cur.description]
        print('table', t, 'cols', cols)
        for row in cur.fetchall():
            for ci, val in enumerate(row):
                if isinstance(val, (bytes, bytearray)) and len(val) > 1024:
                    blob = bytes(val)
                    print(f'  row blob col={cols[ci]} len={len(blob)} head={blob[:16].hex()}')
                    cands = [blob]
                    try:
                        cands.append(zlib.decompress(blob))
                    except Exception:
                        pass
                    for k, data in enumerate(cands):
                        off = data.find(b'\x7fELF')
                        if off >= 0:
                            out = f'/work/cubins/k{idx}_t{t}_c{ci}_z{k}.cubin'
                            open(out, 'wb').write(data[off:])
                            print('    ELF at offset', off, '-> wrote', out, len(data) - off, 'bytes')
                    idx += 1
    con.close()
PY
ls -la /work/cubins >> $LOG 2>&1

echo '===== 6. disassemble =====' >> $LOG
NVD=/usr/local/lib/python3.12/dist-packages/tokenspeed_triton/backends/nvidia/bin/nvdisasm
CUOBJ=/usr/local/cuda/bin/cuobjdump
for f in /work/cubins/*.cubin; do
  [ -e "$f" ] || continue
  echo "--- $f" >> $LOG
  "$CUOBJ" -sass "$f" > "/work/cubins/$(basename $f).sass" 2>> $LOG
  RC=$?
  echo "cuobjdump rc=$RC lines=$(wc -l < /work/cubins/$(basename $f).sass 2>/dev/null)" >> $LOG
  if [ $RC -ne 0 ] || [ ! -s "/work/cubins/$(basename $f).sass" ]; then
    "$NVD" -c "$f" > "/work/cubins/$(basename $f).sass" 2>> $LOG || true
    echo "nvdisasm retry rc=$? lines=$(wc -l < /work/cubins/$(basename $f).sass 2>/dev/null)" >> $LOG
  fi
done

echo '===== 7. grep gate =====' >> $LOG
for s in /work/cubins/*.sass; do
  [ -e "$s" ] || continue
  total=$(grep -ic 'mma' "$s" 2>/dev/null)
  e2m1=$(grep -icE 'mma.*e2m1|e2m1.*mma' "$s" 2>/dev/null)
  mxf=$(grep -ic 'mxf4' "$s" 2>/dev/null)
  echo "SASS_FILE=$s mma_total=$total mma_e2m1=$e2m1 mxf4=$mxf" >> $LOG
  grep -iE 'mma' "$s" | head -8 >> $LOG
done
echo SASS_DONE >> $LOG
