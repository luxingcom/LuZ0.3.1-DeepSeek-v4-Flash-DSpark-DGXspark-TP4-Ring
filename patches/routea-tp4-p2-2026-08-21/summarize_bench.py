#!/usr/bin/env python3
"""summarize_bench.py — bench_tp4.py 输出汇总。用法: python summarize_bench.py bench_A.json [bench_B.json ...]"""
import json
import sys


def med(xs):
    v = sorted(x for x in xs if x is not None)
    return v[len(v) // 2] if v else None


for path in sys.argv[1:]:
    d = json.load(open(path, encoding='utf-8'))
    print(f"===== {path} =====")
    for k in ("ttft_4k", "ttft_16k", "ttft_2k"):
        if k not in d:
            continue
        tt = [r["ttft_s"] for r in d[k]]
        print(f"  {k}: median={med(tt)}s  all={tt}")
    for k in ("decode_1x256",):
        for r in d.get(k, []):
            print(f"  {k}: ttft={r['ttft_s']}s total={r['total_s']}s decode_tps={r['decode_tps']}")
    for k in ("decode_12x128",):
        for r in d.get(k, []):
            print(f"  {k}: total={r['total_s']}s agg_tps={r['agg_tps']}")
    lp = d.get("logprob", [])
    tot = sum(x.get("sum_lp", 0) for x in lp if "sum_lp" in x)
    n = sum(x.get("n_tok", 0) for x in lp if "n_tok" in x)
    errs = [x for x in lp if "error" in x]
    print(f"  logprob: sum={tot:.1f} over {n} tokens ({n/max(len(lp),1):.0f}/prompt), errors={len(errs)}")
    for k in ("needle_64k", "needle_128k"):
        for r in d.get(k, []):
            print(f"  {k}: pass={r.get('pass')} pos={r.get('pos')} "
                  f"latency={r.get('latency_s')}s resp={r.get('resp', '')!r}")
