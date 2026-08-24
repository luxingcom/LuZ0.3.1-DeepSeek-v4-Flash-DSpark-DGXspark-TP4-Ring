import struct, math, statistics
from collections import Counter
PATH = "<INSTALL_DIR>/models/deepseek-v4-flash-0731/model-00045-of-00048.safetensors"
OFF = 262164
N, K = 129280, 4096
DT = 2

def read_rows(rows):
    out = {}
    with open(PATH, "rb") as f:
        for r in rows:
            f.seek(OFF + r * K * DT)
            raw = f.read(K * DT)
            vals = []
            for i in range(K):
                u = struct.unpack_from("<H", raw, i * 2)[0]
                f32 = struct.unpack("<f", struct.pack("<I", u << 16))[0]
                vals.append(f32)
            out[r] = vals
    return out

rows = [0, 1, 2, 127, 128, 129, 1000, 5000, 12345, 30000, 64639, 64640, 90000, 129278, 129279]
data = read_rows(rows)

print("=== per-row stats (K=4096) ===")
allv = []
for r, v in data.items():
    mx = max(abs(x) for x in v)
    mn = min(v); mxx = max(v)
    mean = sum(v) / len(v)
    allv.extend(v)
    print("row %6d: min=%.6f max=%.6f maxabs=%.6f mean=%.6f nz=%d" % (r, mn, mxx, mx, mean, sum(1 for x in v if x != 0)))

print("\n=== global sample stats ===")
n = len(allv)
print("sampled elements:", n)
print("global min", min(allv), "max", max(allv), "maxabs", max(abs(x) for x in allv))
print("mean", sum(allv) / n)
print("zero count:", sum(1 for x in allv if x == 0), "of", n)

def e8m0_exp(maxabs):
    if maxabs == 0:
        return None
    e = math.ceil(math.log2(maxabs))
    return e

print("\n=== E8M0 per-32-group scale exponent distribution (sampled rows) ===")
cnt = Counter()
for r, v in data.items():
    for g in range(0, K, 32):
        grp = v[g:g + 32]
        ma = max(abs(x) for x in grp)
        e = e8m0_exp(ma)
        if e is not None:
            cnt[e] += 1
print("total 32-groups:", sum(cnt.values()))
print("exp range:", min(cnt), max(cnt))
print("top exps:", cnt.most_common(12))

print("\n=== block dynamic range (128x128 vs 32) ===")
for r in rows[:4]:
    v = data[r]
    dr128 = []; dr32 = []
    for g in range(0, K, 128):
        grp = v[g:g + 128]; ma = max(abs(x) for x in grp)
        mnz = min((abs(x) for x in grp if x != 0), default=None)
        dr128.append((ma / mnz) if mnz else float("inf"))
    for g in range(0, K, 32):
        grp = v[g:g + 32]; ma = max(abs(x) for x in grp)
        mnz = min((abs(x) for x in grp if x != 0), default=None)
        dr32.append((ma / mnz) if mnz else float("inf"))
    med128 = statistics.median([d for d in dr128 if d != float("inf")])
    med32 = statistics.median([d for d in dr32 if d != float("inf")])
    print("row %d: 128-block median dyn-range=%.1f | 32-block median dyn-range=%.1f" % (r, med128, med32))
