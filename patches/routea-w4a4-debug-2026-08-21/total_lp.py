#!/usr/bin/env python3
"""total_lp.py — 五方总 logprob（质量保持率）视角: B1/B2/E/T/C。"""
import json
import sys
sys.path.insert(0, "/tmp/_routea_work")
from compare_lp import load, actual_prompt_lp, gen_top1

runs = {
    "B1 W4A16+clamp ": "/tmp/_routea_work/lp_0731.json",
    "B2 W4A16 noclmp": "/tmp/_routea_work/lp_w4a16_noclamp.json",
    "E  EMU   W4A4  ": "/tmp/_routea_work/lp_emu.json",
    "T  b12x  W4A4  ": "/tmp/_routea_work/lp_w4a4.json",
    "C  cutl  W4A4  ": "/tmp/_routea_work/lp_cutlass.json",
}
print(f"{'run':18s} | {'Σ prompt lp':>12s} | {'Σ gen top1 lp':>13s} | {'合计':>10s} | 相对B1")
base = None
for name, p in runs.items():
    d = load(p)
    tot_p, tot_g = 0.0, 0.0
    for e in d:
        for pos in range(1, len(e["prompt_token_ids"])):
            lp = actual_prompt_lp(e, pos)
            if lp is not None:
                tot_p += lp
        for g in e["gen_logprobs"]:
            _, lp = gen_top1(e, g["pos"])
            tot_g += lp
    tot = tot_p + tot_g
    if base is None:
        base = tot
    print(f"{name:18s} | {tot_p:12.1f} | {tot_g:13.1f} | {tot:10.1f} | {(tot-base)/abs(base)*100:+.2f}%")
