#!/usr/bin/env python3
"""compare6_dispatch.py — 量化 V3 分派: 每 prompt 的 V3-vs-V0 / V3-vs-V2 prefill mean|Δlp|。"""
import sys
sys.path.insert(0, "/tmp/_routea_work")
from compare_lp import load, actual_prompt_lp


def plp_vector(entry):
    v = {}
    for pos in range(1, len(entry["prompt_token_ids"])):
        lp = actual_prompt_lp(entry, pos)
        if lp is not None:
            v[pos] = lp
    return v


v0 = load("/tmp/_routea_work/lp_v0.json")
v2 = load("/tmp/_routea_work/lp_v2.json")
v3 = load("/tmp/_routea_work/lp_v3.json")

print(f"{'prompt':>7} {'M':>4} | {'V3-V0 mean|Δ|':>13} | {'V3-V2 mean|Δ|':>13} | 判定")
for i, (e3, e0, e2) in enumerate(zip(v3, v0, v2)):
    m = len(e3["prompt_token_ids"])
    v3v, v0v, v2v = plp_vector(e3), plp_vector(e0), plp_vector(e2)
    keys = v3v.keys() & v0v.keys() & v2v.keys()
    d0 = sum(abs(v3v[k] - v0v[k]) for k in keys) / max(len(keys), 1)
    d2 = sum(abs(v3v[k] - v2v[k]) for k in keys) / max(len(keys), 1)
    verdict = "W4A16" if d0 < d2 / 10 else ("W4A4" if d2 < d0 / 10 else "?混合")
    print(f"{i:>7} {m:>4} | {d0:13.5f} | {d2:13.5f} | {verdict}")
