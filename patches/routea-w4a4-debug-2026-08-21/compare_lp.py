#!/usr/bin/env python3
"""compare_lp.py — mini 模型 logprob 三方对照: T(W4A4 noclamp) vs B2(W4A16 noclamp) vs B1(W4A16 clamp)。"""
import json
import math
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def actual_prompt_lp(entry, pos):
    """position pos 的 actual token logprob (prompt_token_ids[pos])"""
    tid = str(entry["prompt_token_ids"][pos])
    for e in entry["prompt_logprobs"]:
        if e["pos"] == pos:
            return e["lp"].get(tid)
    return None


def gen_top1(entry, pos):
    g = entry["gen_logprobs"][pos]
    top = sorted(g["top5"].items(), key=lambda kv: -kv[1])
    return int(top[0][0]), top[0][1]


def compare(A, B, name_a, name_b):
    tot_d, n_d = 0.0, 0
    max_d = 0.0
    agree, n_g = 0, 0
    gen_d, n_gd = 0.0, 0
    for ea, eb in zip(A, B):
        assert ea["prompt_id"] == eb["prompt_id"]
        # prompt logprobs (actual token)
        for pos in range(1, min(len(ea["prompt_token_ids"]), len(eb["prompt_token_ids"]))):
            la = actual_prompt_lp(ea, pos)
            lb = actual_prompt_lp(eb, pos)
            if la is None or lb is None:
                continue
            d = abs(la - lb)
            tot_d += d
            n_d += 1
            max_d = max(max_d, d)
        # generation top-1
        for ga, gb in zip(ea["gen_logprobs"], eb["gen_logprobs"]):
            ta, la = gen_top1(ea, ga["pos"])
            tb, lb = gen_top1(eb, gb["pos"])
            n_g += 1
            if ta == tb:
                agree += 1
                gen_d += abs(la - lb)
                n_gd += 1
            else:
                gen_d += abs(la - lb)
                n_gd += 1
    print(f"--- {name_a} vs {name_b} ---")
    print(f"  prompt-actual-token logprob: n={n_d}, mean|Δ|={tot_d/max(n_d,1):.4f}, max|Δ|={max_d:.4f}")
    print(f"  gen top-1: n={n_g}, 一致率={agree/max(n_g,1)*100:.1f}%, "
          f"mean|Δlp(top1)|={gen_d/max(n_gd,1):.4f}")
    return tot_d / max(n_d, 1), max_d, agree / max(n_g, 1)


def main():
    b1 = load(sys.argv[1])   # W4A16 + clamp (生产基线)
    b2 = load(sys.argv[2])   # W4A16 no clamp
    t = load(sys.argv[3])    # W4A4 no clamp
    print(f"prompts: B1={len(b1)} B2={len(b2)} T={len(t)}\n")
    compare(t, b2, "T: W4A4(noclamp)", "B2: W4A16(noclamp)")   # 纯量化语义差
    print()
    compare(b2, b1, "B2: W4A16(noclamp)", "B1: W4A16(clamp)")   # clamp 效应
    print()
    compare(t, b1, "T: W4A4(noclamp)", "B1: W4A16(clamp)")      # 端到端总差


if __name__ == "__main__":
    main()
