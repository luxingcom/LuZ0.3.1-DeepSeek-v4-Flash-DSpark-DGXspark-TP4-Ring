import subprocess, os, re, tempfile

SO = "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
NVD = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm"
out_dir = "/tmp/sass_fp4"
os.makedirs(out_dir, exist_ok=True)

# extract all sm_120 cubins to files, then disassemble and grep FP4 MMA
# cuobjdump --extract-elf=dir
r = subprocess.run(["cuobjdump", "--extract-elf=" + out_dir, SO], capture_output=True, text=True)
print("extract exit:", r.returncode, "err:", r.stderr.splitlines()[-1][:120] if r.stderr else "")

cubins = sorted([f for f in os.listdir(out_dir) if f.endswith(".cubin")])
print("extracted cubins:", len(cubins))

mma_e2m1 = 0; mmaf = 0; total_mma = 0; tcgen = 0
files_with_fp4 = []
for c in cubins:
    p = os.path.join(out_dir, c)
    try:
        dd = subprocess.run([NVD, p], capture_output=True, text=True, timeout=120)
        sass = dd.stdout
        e2 = len(re.findall(r'MMA[^\n]*(E2M1|FP4|Q4)', sass, re.I))
        mf = len(re.findall(r'\bMMA[F]\b|mma\.|mmaf', sass, re.I))
        tm = len(re.findall(r'\bMMA\b', sass, re.I))
        tg = len(re.findall(r'TCGEN', sass, re.I))
        mma_e2m1 += e2; mmaf += mf; total_mma += tm; tcgen += tg
        if e2 > 0 or 'mmaf' in sass.lower() or 'e2m1' in sass.lower():
            files_with_fp4.append(c)
    except Exception as ex:
        print("  skip", c, str(ex)[:80])

print("=== SASS 门禁统计 (sm_120 cubins) ===")
print(f"  MMA 指令总数(E2M1/FP4/Q4): {mma_e2m1}")
print(f"  mmaf/FP4系: {mmaf}")
print(f"  TCGEN(TCGEN05): {tcgen}")
print(f"  MMA 总数: {total_mma}")
print(f"  含FP4 MMA 的 cubin 文件数: {len(files_with_fp4)}")
# fp4 MMA 关键指令样例：抽取一条
if files_with_fp4:
    p = os.path.join(out_dir, files_with_fp4[0])
    dd = subprocess.run([NVD, p], capture_output=True, text=True, timeout=120)
    lines = [l for l in dd.stdout.splitlines() if re.search(r'MMA[^\n]*(E2M1|FP4|Q4)|mmaf', l, re.I)]
    print("=== FP4 MMA 指令样例 ===")
    for l in lines[:8]:
        print("  ", l.strip()[:120])