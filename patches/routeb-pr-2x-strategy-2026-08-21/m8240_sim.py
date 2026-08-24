#!/usr/bin/env python3
"""m8240_sim.py — 杠杆栈层2叠加效应: 同一路由数据在 step M=4096 vs 8240 下的 M_g 覆盖率。
模型: 每 set 在 token 总体中的占比 p_s → 步 M 下 M_g = p_s × M。
覆盖@T = sum(p_s for M_g(s) >= T)。"""
import json
import sys
from collections import Counter

records = [json.loads(l) for l in open(sys.argv[1])]
layer_sets = {}
for i, r in enumerate(records):
    for t in r["topk"]:
        layer_sets.setdefault(i % 4, Counter())[frozenset(t)] += 1

THRESHOLDS = (512, 1024, 2048, 3072, 6144)
print(f"{'M':>6} | " + " | ".join(f"≥{T:>4}" for T in THRESHOLDS) + "  （各层覆盖@阈值）")
for M in (4096, 8240, 12288):
    for layer in (0, 3):
        sets = layer_sets[layer]
        n = sum(sets.values())
        row = []
        for T in THRESHOLDS:
            cov = sum(c for c in sets.values() if c / n * M >= T) / n
            row.append(f"{cov*100:4.0f}%")
        print(f"{M:>6} L{layer} | " + " | ".join(row))

print("\n=== MoE 加速估算（merged kernel 效率按 M_g 档位折算） ===")
# M_g→TFLOPS 插值（P0 实测锚点: M=3072@332T; 估算: 1024~200T, 512~120T, <256~60T）
def eff(M_g):
    if M_g >= 3072: return 332
    if M_g >= 2048: return 280
    if M_g >= 1024: return 200
    if M_g >= 512: return 120
    return 60
BASE_T = 100  # B12X per-expert 现状有效 TFLOPS 估
for M in (4096, 8240):
    for layer in (0, 3):
        sets = layer_sets[layer]
        n = sum(sets.values())
        # merged 桶流量加权效率（全部组合都走 merged, exact-set 零浪费）
        num = sum(c * eff(c / n * M) for c in sets.values())
        # 混合: M_g>=512 走 merged, 其余 B12X
        mix_num = sum(c * (eff(c / n * M) if c / n * M >= 512 else BASE_T)
                      for c in sets.values())
        print(f"  M={M} L{layer}: 全merged 有效TFLOPS={num/n:.0f}, "
              f"混合(≥512 merged)={mix_num/n:.0f} vs B12X {BASE_T}T "
              f"→ MoE 加速 {mix_num/n/BASE_T:.2f}x")
