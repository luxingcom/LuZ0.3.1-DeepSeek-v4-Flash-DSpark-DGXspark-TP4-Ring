#!/usr/bin/env python3
"""analyze_routing.py — 路由分布统计 + 分桶模拟（P1-1 ③）。
输入: routing_capture.jsonl（每条 = 一次 MoE forward 的 topk_ids [n_tok][6]）
按 record index % 4 分层（4 层 mini: 0-2 hash 层, 3 dense 层）。
输出: 基础统计 + 共现聚类分桶模拟 + 连续区间分桶模拟（零拷贝选项）。"""
import json
import sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_routea_work/routing_capture.jsonl"

records = []
with open(PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"records: {len(records)} (4 层 × {len(records)//4} prefill batch)")

N_EXP = 256
TOPK = 6

# ---------------- 基础统计 ----------------
print("\n=== 基础统计（按层） ===")
all_sets = defaultdict(Counter)      # layer -> Counter(frozenset)
hit_experts = defaultdict(list)      # layer -> [distinct expert count per record]
expert_mass = defaultdict(Counter)   # layer -> expert -> pair count
for i, r in enumerate(records):
    layer = i % 4
    sets = [frozenset(t) for t in r["topk"]]
    for s in sets:
        all_sets[layer][s] += 1
    pairs = [e for t in r["topk"] for e in t]
    hit_experts[layer].append(len(set(pairs)))
    expert_mass[layer].update(pairs)

for layer in sorted(all_sets):
    n_tok = sum(all_sets[layer].values())
    sets = all_sets[layer]
    top10 = sets.most_common(10)
    top10_cov = sum(c for _, c in top10) / n_tok
    top100_cov = sum(c for _, c in sets.most_common(100)) / n_tok
    he = hit_experts[layer]
    print(f"  layer{layer}: {n_tok} tokens, distinct sets={len(sets)}, "
          f"top10-set 覆盖={top10_cov*100:.1f}%, top100={top100_cov*100:.1f}%, "
          f"命中专家数/step min/med/max={min(he)}/{sorted(he)[len(he)//2]}/{max(he)}")

# ---------------- 分桶模拟 A: 共现聚类（用户方案近似） ----------------
def bucket_by_cooccur(layer_sets, n_tok, target_G=8, min_bucket_tokens=256):
    """贪心共现分桶: 按专家集频率降序; set 归入重叠率最高且专家并集不超限的桶,
    否则新桶; 桶专家并集上限 ~12(=24576/2048 的 N 预算), 桶数上限 target_G*2。
    返回桶列表 [(experts_frozenset, n_tokens)]"""
    buckets = []  # [ (expert_set, token_count) ]
    for s, c in layer_sets.most_common():
        best_j, best_idx = 0.0, -1
        for idx, (bes, bc) in enumerate(buckets):
            union = bes | s
            if len(union) > 12:      # N 预算: 12 experts × 2048 = 24576
                continue
            j = len(bes & s) / len(union)   # Jaccard
            if j > best_j:
                best_j, best_idx = j, idx
        if best_idx >= 0 and best_j >= 0.25:
            bes, bc = buckets[best_idx]
            buckets[best_idx] = (bes | s, bc + c)
        elif len(buckets) < target_G * 2:
            buckets.append((set(s), c))
        else:
            # 挂到重叠最高的桶（允许略超 N 预算时选最小代价）
            bes, bc = max(buckets, key=lambda b: len(b[0] & s))
            i2 = buckets.index((bes, bc))
            buckets[i2] = (bes | s, bc + c)
    return buckets

print("\n=== 分桶模拟 A: 共现聚类（用户方案近似, N 预算≤12 expert/桶） ===")
for layer in sorted(all_sets):
    n_tok = sum(all_sets[layer].values())
    buckets = bucket_by_cooccur(all_sets[layer], n_tok)
    buckets.sort(key=lambda b: -b[1])
    useful = n_tok * TOPK                     # 有效 (token,expert) 对
    effective = sum(bc * len(bes) for bes, bc in buckets)  # 实际计算列数
    mg = [bc for _, bc in buckets]
    ng = [len(bes) for bes, _ in buckets]
    print(f"  layer{layer}: G={len(buckets)}, M_g dist={sorted(mg, reverse=True)[:8]}..."
          f"（tokens/桶）, N_g(expert数) dist={sorted(ng, reverse=True)[:8]}, "
          f"浪费率={100*(1-useful/effective):.1f}%")

# ---------------- 分桶模拟 B: 连续区间分桶（零拷贝选项 d） ----------------
print("\n=== 分桶模拟 B: 连续区间分桶（零拷贝, stacked 权重连续切片视图） ===")
for group_size in (8, 16, 32):
    n_groups = N_EXP // group_size
    for layer in sorted(all_sets):
        pairs = Counter()
        for s, c in all_sets[layer].items():
            for e in s:
                pairs[e] += c
        # 每组 M_g = 落入该组 expert 区间的 (token,expert) 对数
        mg = [sum(pairs[e] for e in range(g * group_size, (g + 1) * group_size))
              for g in range(n_groups)]
        n_tok = sum(all_sets[layer].values())
        # 该方案下每对恰好计算一次（无浪费），GEMM 形状 = [M_g, group_size*N_e]
        big = sum(1 for m in mg if m >= 3072)
        print(f"  layer{layer} 区间×{group_size}: G={n_groups}, M_g(对/组) "
              f"med={sorted(mg)[n_groups//2]}, max={max(mg)}, "
              f"≥3072 的组数={big}/{n_groups}")

# ---------------- 组合缓存命中率（选项 a） ----------------
print("\n=== 组合缓存（LRU, 选项 a）: 每层 top-K 组合覆盖率 ===")
for layer in sorted(all_sets):
    n_tok = sum(all_sets[layer].values())
    sets = all_sets[layer]
    for K in (32, 64, 128):
        cov = sum(c for _, c in sets.most_common(K)) / n_tok
        print(f"  layer{layer}: top-{K} set 覆盖 {cov*100:.1f}%")
