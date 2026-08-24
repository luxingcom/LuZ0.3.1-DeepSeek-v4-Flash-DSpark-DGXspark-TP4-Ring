#!/usr/bin/env python3
"""analyze_routing2.py — 修正版分桶模拟（merged GEMM 正确浪费口径）。
模型: 桶 g 的 merged GEMM = [M_g tokens(去重,桶内≥1专家) × N_g 桶专家并集×N_e 列]。
有效 = 桶内 (token,expert) 对数 × N_e。浪费 = 计算/有效。"""
import json
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_routea_work/routing_capture.jsonl"
records = [json.loads(l) for l in open(PATH)]
N_EXP, TOPK = 256, 6
NE = 2048  # 每 expert w13 行（全模型口径; rank 口径同理等比）

layer_sets = {}
for i, r in enumerate(records):
    layer = i % 4
    c = layer_sets.setdefault(layer, Counter())
    for t in r["topk"]:
        c[frozenset(t)] += 1

# ---------- A: 组合重叠分桶（严格版: set 整体入桶, 并集上限 12, jaccard>=0.34, rest 兜底） ----------
print("=== A: 组合重叠分桶（严格贪心, 桶并集<=12, rest 桶兜底） ===")
for layer in sorted(layer_sets):
    sets = layer_sets[layer]
    n_tok = sum(sets.values())
    buckets = []   # (expert_union_set, token_count)
    for s, c in sets.most_common():
        placed = False
        for idx in range(len(buckets)):
            bes, bc = buckets[idx]
            union = bes | s
            if len(union) <= 12 and len(bes & s) / len(union) >= 0.34:
                buckets[idx] = (union, bc + c)
                placed = True
                break
        if not placed:
            buckets.append((set(s), c))
    # rest 桶（所有没被良好聚类的落最后一个桶）已在 buckets 里（小桶）
    # merged GEMM 逐桶: tokens_in_bucket ~= token_count（近似, 去重效应忽略）
    compute = sum(bc * len(bes) * NE for bes, bc in buckets)
    useful = n_tok * TOPK * NE
    # 达成: M_g>=3072 且 N_g<=12*NE(24576) 的桶
    ok_tokens = sum(bc for bes, bc in buckets if bc >= 3072 and len(bes) <= 12)
    print(f"  layer{layer}: G={len(buckets)}, 浪费率={100*(compute/useful-1):.0f}% "
          f"(计算/有效={compute/useful:.1f}x), M_g>=3072&N_g<=12 桶覆盖 token="
          f"{ok_tokens/n_tok*100:.0f}%, "
          f"top8桶 M_g={[bc for _, bc in sorted(buckets, key=lambda b:-b[1])[:8]]}")

# ---------- B: 连续区间（正确浪费口径） ----------
print("\n=== B: 连续区间分桶（零拷贝视图, 正确浪费口径） ===")
for group_size in (8, 16, 32):
    for layer in sorted(layer_sets):
        sets = layer_sets[layer]
        n_tok = sum(sets.values())
        # 桶 token 集合 = 触碰该区间的 token 数; 对数 = 区间内 pair 数
        tok_in = [0] * (N_EXP // group_size)
        pair_in = [0] * (N_EXP // group_size)
        for s, c in sets.items():
            touched = set()
            for e in s:
                g = e // group_size
                pair_in[g] += c
                touched.add(g)
            for g in touched:
                tok_in[g] += c
        compute = sum(tok_in[g] * group_size * NE for g in range(N_EXP // group_size))
        useful = n_tok * TOPK * NE
        mg_ok = sum(1 for g in range(N_EXP // group_size) if pair_in[g] >= 3072)
        print(f"  layer{layer} 区间×{group_size}: 计算/有效={compute/useful:.1f}x, "
              f"M_g(对)>=3072 组数={mg_ok}/{N_EXP//group_size}")

# ---------- C: 理想聚类上界（若每 token 完美落桶） ----------
print("\n=== C: 完美聚类下界（token 6 专家全落一桶, N_g=6） ===")
for layer in sorted(layer_sets):
    sets = layer_sets[layer]
    n_tok = sum(sets.values())
    # 完美情形: 每桶只含同 set 的 token → 计算=有效, 浪费 0, 但 M_g=set 频率
    freqs = sorted((c for _, c in sets.most_common()), reverse=True)
    big = sum(c for c in freqs if c >= 3072)
    print(f"  layer{layer}: set 频率>=3072 的 token 占比={big/n_tok*100:.1f}% "
          f"(若完美按 set 分桶); top set 频率={freqs[0]}")

# ---------- D: hash 层组合缓存可行性 ----------
print("\n=== D: 组合缓存显存账（hash 层 top-K 组合预拼） ===")
for layer in (0, 3):
    sets = layer_sets[layer]
    n_tok = sum(sets.values())
    for K in (8, 16, 32):
        cov = sum(c for _, c in sets.most_common(K)) / n_tok
        mem = K * 12 * 1024 * 1024 * 43 / 1e9  # K 组合×12MB×43 层
        print(f"  layer{layer}: top-{K} 覆盖 {cov*100:.0f}% (缓存显存 {mem:.1f} GB/rank)")
