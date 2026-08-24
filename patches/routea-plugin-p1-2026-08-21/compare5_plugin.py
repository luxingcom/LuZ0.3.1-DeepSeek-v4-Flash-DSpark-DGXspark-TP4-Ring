#!/usr/bin/env python3
"""compare5_plugin.py — P1 插件验证判定:
V0 基线(off) / V1 hybrid全回落 / V2 full / V3 hybrid MIN_M=64
1) V1 vs V0 逐位一致（子类透传零污染）
2) V2 vs V0 质量（总 lp 差、mean|Δ|）
3) V3 每个 prompt 的 prefill logprobs 判定走的是 W4A4(=V2) 还是 W4A16(=V0)
"""
import json
import sys
sys.path.insert(0, "/tmp/_routea_work")
from compare_lp import load, actual_prompt_lp, gen_top1


def plp_vector(entry):
    v = {}
    for pos in range(1, len(entry["prompt_token_ids"])):
        lp = actual_prompt_lp(entry, pos)
        if lp is not None:
            v[pos] = lp
    return v


def bitwise_plp(a, b):
    va, vb = plp_vector(a), plp_vector(b)
    if va.keys() != vb.keys():
        return False
    return all(va[k] == vb[k] for k in va)


def total_lp(d):
    tot = 0.0
    for e in d:
        for pos in range(1, len(e["prompt_token_ids"])):
            lp = actual_prompt_lp(e, pos)
            if lp is not None:
                tot += lp
        for g in e["gen_logprobs"]:
            _, lp = gen_top1(e, g["pos"])
            tot += lp
    return tot


v0 = load("/tmp/_routea_work/lp_v0.json")
v1 = load("/tmp/_routea_work/lp_v1.json")
v2 = load("/tmp/_routea_work/lp_v2.json")
v3 = load("/tmp/_routea_work/lp_v3.json")

print("[1] V1(hybrid 全回落 W4A16) vs V0(基线):")
allsame = all(bitwise_plp(a, b) for a, b in zip(v1, v0))
gensame = all(a["gen_text"] == b["gen_text"] for a, b in zip(v1, v0))
print(f"    prefill logprobs 逐位一致: {allsame} | gen 文本一致: {gensame}")

print("\n[2] V2(full W4A4) vs V0(基线):")
t0, t2 = total_lp(v0), total_lp(v2)
n = 0
d_sum = 0.0
for a, b in zip(v2, v0):
    va, vb = plp_vector(a), plp_vector(b)
    for k in va:
        d_sum += abs(va[k] - vb[k])
        n += 1
print(f"    总 logprob: V0={t0:.1f} V2={t2:.1f} 相对差={(t2-t0)/abs(t0)*100:+.2f}% "
      f"(判据 ≤1%)")
print(f"    prefill mean|Δlp|={d_sum/n:.4f} (n={n})")

print("\n[3] V3(hybrid MIN_M=64) 分派判定（每 prompt prefill）:")
for i, (e3, e0, e2) in enumerate(zip(v3, v0, v2)):
    m = len(e3["prompt_token_ids"])
    if bitwise_plp(e3, e2):
        which = "W4A4 (=V2 逐位一致)"
    elif bitwise_plp(e3, e0):
        which = "W4A16 (=V0 逐位一致)"
    else:
        which = "混合/其他 —— 需检查"
    print(f"    prompt {i} (M={m}): {which}")
print("\n== DONE ==")
