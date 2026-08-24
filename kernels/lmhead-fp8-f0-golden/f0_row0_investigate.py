import struct, math
PATH = "<INSTALL_DIR>/models/deepseek-v4-flash-0731/model-00045-of-00048.safetensors"
OFF = 262164
K = 4096
DT = 2

def read_row(r):
    with open(PATH, "rb") as f:
        f.seek(OFF + r * K * DT)
        raw = f.read(K * DT)
    vals = []
    for i in range(K):
        u = struct.unpack_from("<H", raw, i * 2)[0]
        f32 = struct.unpack("<f", struct.pack("<I", u << 16))[0]
        vals.append(f32)
    return vals

v = read_row(0)
# split into 32-groups, show maxabs per group
print("row 0: total maxabs", max(abs(x) for x in v))
big_idx = [(i, x) for i, x in enumerate(v) if abs(x) > 10.0]
print("count of |x|>10:", len(big_idx))
print("first 30 big:", big_idx[:30])
print("last 10 big:", big_idx[-10:])
print("\n=== per-32-group maxabs (row 0) ===")
for g in range(0, K, 32):
    grp = v[g:g + 32]
    ma = max(abs(x) for x in grp)
    print("group %3d [%4d:%4d] maxabs=%g nz=%d" % (g // 32, g, g + 32, ma, sum(1 for x in grp if x != 0)))

# Also scan a few more rows that might be outliers: check every 1000th row maxabs quickly
print("\n=== quick outlier scan (every 100th row, maxabs) ===")
def read_row_maxabs(r):
    with open(PATH, "rb") as f:
        f.seek(OFF + r * K * DT)
        raw = f.read(K * DT)
    ma = 0.0
    for i in range(K):
        u = struct.unpack_from("<H", raw, i * 2)[0]
        f32 = struct.unpack("<f", struct.pack("<I", u << 16))[0]
        a = abs(f32)
        if a > ma: ma = a
    return ma

outliers = []
for r in range(0, 129280, 100):
    ma = read_row_maxabs(r)
    if ma > 5.0:
        outliers.append((r, ma))
print("rows with maxabs>5 (every-100 scan):", outliers[:40])
