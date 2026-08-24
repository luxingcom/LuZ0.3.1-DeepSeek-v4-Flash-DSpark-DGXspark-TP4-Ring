#!/usr/bin/env python3
"""analyze_routing3.py — M_g 阈值 vs token 覆盖率曲线（调档依据）。"""
import json
import sys
from collections import Counter

records = [json.loads(l) for l in open(sys.argv[1])]
layer_sets = {}
for i, r in enumerate(records):
    for t in r['topk']:
        layer_sets.setdefault(i % 4, Counter())[frozenset(t)] += 1

print("=== M_g 阈值 vs 覆盖率（两种口径） ===")
print(f"{'阈值':>6} | {'完美按set分桶(hash层)':>22} | {'完美按set分桶(dense层)':>23} | {'严格聚类桶(hash)':>18}")
for th in (256, 512, 1024, 1536, 2048, 3072, 6144):
    row = [f"{th:>6}"]
    for layer in (0, 3):
        sets = layer_sets[layer]
        n = sum(sets.values())
        cov = sum(c for _, c in sets.most_common() if c >= th) / n
        row.append(f"{cov*100:>21.1f}%")
    # 严格聚类桶口径（桶并集<=12）：hash 层桶 = 466 sets → 桶大小=set 频率（浪费0）
    row.append("（同完美口径, hash 桶=set）")
    print(" | ".join(row))

# 长尾结构
print("\n=== set 频率分布（hash 层=路由表 / dense 层=gate） ===")
for layer in (0, 3):
    sets = layer_sets[layer]
    n = sum(sets.values())
    freqs = sorted((c for c in sets.values()), reverse=True)
    cum = 0
    marks = {}
    for i, c in enumerate(freqs):
        cum += c
        for K in (10, 50, 100, 200, 466 if layer == 0 else 1000):
            if i + 1 == K:
                marks[K] = cum / n
    print(f"  layer{layer}: top-set 频率 {freqs[0]}, 前10 累计 {marks.get(10,0)*100:.0f}%, "
          f"前100 累计 {marks.get(100,0)*100:.0f}%, 长尾 set 数={len(freqs)} "
          f"(平均频率 {n//len(freqs)})")
