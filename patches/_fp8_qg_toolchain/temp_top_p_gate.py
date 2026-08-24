#!/usr/bin/env python3
"""temp_top_p_gate.py — 温度采样 / top-p 抽验（G4/G5）
================================================================================
用途: temp∈{0.6,0.9} × top-p∈{0.9,1.0} 下，对比 BF16 参考与 FP8 候选的
      (1) logprob 漂移 (2) 候选集重叠率 (3) distinct-n 多样性不降。

流程:
  1) collect（GPU 窗口）: 在目标形态上按 config 采集 N 次（--reps 默认 5），
     每次一个 run 文件（samples[].outputs 含 N 个输出）：
       VLLM_API_KEY=... python3 temp_top_p_gate.py collect \
           --quant bf16 --tag bf16_t06 --temperature 0.6 --top-p 1.0 \
           --out runs/run_bf16_t06_p10.json
       （对 fp8 同样采集 candidate run）
  2) analyze（零 GPU）:
       python3 temp_top_p_gate.py analyze \
           --baseline runs/run_bf16_t06_p10.json \
           --candidate runs/run_fp8_t06_p10.json \
           --out assets/temp_top_p_verdict.json

判据（参数化，--overlap-min 默认 0.90，--drift-pct-max 默认 1.0，
      --distinct-ratio-min 默认 0.90）:
  - logprob sum drift %（逐输出对）均值 |.| ≤ drift-pct-max
  - top-k 候选集重叠率（逐位置交集/并集，k=采集的 top_logprobs）≥ overlap-min
  - distinct-n（token 级 distinct 数）ratio = cand/base ≥ distinct-ratio-min
  全部满足 => PASS。

说明: temp/top-p 采样本身随机，重叠率比较的是"分布支持"而非逐字一致性；
      reps 越多统计越稳（GPU 窗口建议 ≥5）。
================================================================================
"""
import argparse
import json
import os
import sys

import run_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))


def _topk_overlap(row_a, row_b, k=None):
    """单位置 top-k 候选集重叠率 = |A∩B| / max(1, min(|A|,|B|))。"""
    ta = {str(x["token"]) for x in row_a[:k]} if row_a else set()
    tb = {str(x["token"]) for x in row_b[:k]} if row_b else set()
    if not ta and not tb:
        return None
    denom = max(1, min(len(ta), len(tb)))
    return len(ta & tb) / denom if denom else None


def _distinct_tokens(outputs):
    s = set()
    for o in outputs:
        for t in o.get("tokens") or []:
            s.add(str(t))
    return len(s)


def cmd_collect(args):
    if not rc.need_key():
        return 2
    doc = json.load(open(args.ref, encoding="utf-8"))
    samples = []
    for s in doc["samples"]:
        if s.get("tier") != "B":
            continue
        if args.subset and s["id"] not in args.subset.split(","):
            continue
        prompt = s["prompt"]
        print(f"  collect {s['id']} temp={args.temperature} "
              f"top_p={args.top_p} reps={args.reps}", flush=True)
        outputs = [rc.gen(prompt, temperature=args.temperature, top_p=args.top_p,
                          max_tokens=s.get("max_tokens", 256),
                          top_logprobs=args.top_logprobs)
                   for _ in range(args.reps)]
        samples.append({"id": s["id"], "tier": s["tier"],
                        "category": s["category"], "prompt": prompt,
                        "outputs": outputs})
    meta = {
        "quant": args.quant, "model": rc.MODEL,
        "config": {"temperature": args.temperature, "top_p": args.top_p,
                   "top_logprobs": args.top_logprobs, "reps": args.reps},
        "collected_at": __import__("time").strftime("%Y%m%d-%H%M%SZ",
                                                    __import__("time").gmtime()),
        "notes": args.tag or "",
    }
    path = args.out or os.path.join(HERE, "runs",
                                    f"run_{args.quant}_t{args.temperature}_p{args.top_p}.json")
    rc.save_run(path, meta, samples)
    print(f"[collect] {len(samples)} samples x {args.reps} -> {path}")
    return 0


def cmd_analyze(args):
    run_b = rc.load_run(args.baseline)
    run_c = rc.load_run(args.candidate)
    common, by_b, by_c = rc.align_samples(run_b, run_c)
    if not common:
        print("[error] 无公共样本", file=sys.stderr)
        return 1

    overlap_all, drift_pcts, distinct_ratio = [], [], []
    per = {}
    for sid in common:
        ob = by_b[sid]["outputs"]
        oc = by_c[sid]["outputs"]
        if not ob or not oc:
            continue
        n_pairs = min(len(ob), len(oc))
        overlaps, drifts = [], []
        for i in range(n_pairs):
            tl_b = ob[i].get("top_logprobs") or []
            tl_c = oc[i].get("top_logprobs") or []
            n_pos = min(len(tl_b), len(tl_c))
            if n_pos == 0:
                continue
            ov = [ov2 for ov2 in (_topk_overlap(tl_b[j], tl_c[j],
                                                k=args.top_k) for j in range(n_pos))
                  if ov2 is not None]
            if ov:
                overlaps.append(sum(ov) / len(ov))
            d = rc.logprob_drift(ob[i].get("logprobs") or [],
                                 oc[i].get("logprobs") or [])
            if d and d.get("sum_drift_pct") is not None:
                drifts.append(abs(d["sum_drift_pct"]))
        if overlaps:
            overlap_all.extend(overlaps)
        drift_pcts.extend(drifts)
        db = _distinct_tokens(ob)
        dc = _distinct_tokens(oc)
        per[sid] = {"overlap_mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
                    "drift_pct_mean": round(sum(drifts) / len(drifts), 4) if drifts else None,
                    "distinct_base": db, "distinct_cand": dc,
                    "distinct_ratio": round(dc / db, 4) if db else None}

    if not overlap_all and not drift_pcts:
        print("[error] 无有效配对输出", file=sys.stderr)
        return 1
    agg_overlap = sum(overlap_all) / len(overlap_all) if overlap_all else None
    agg_drift = sum(drift_pcts) / len(drift_pcts) if drift_pcts else None
    ratios = [v["distinct_ratio"] for v in per.values() if v["distinct_ratio"] is not None]
    agg_distinct_ratio = sum(ratios) / len(ratios) if ratios else None

    overlap_pass = (agg_overlap is not None) and (agg_overlap >= args.overlap_min)
    drift_pass = (agg_drift is not None) and (agg_drift <= args.drift_pct_max)
    distinct_pass = (agg_distinct_ratio is not None) and \
        (agg_distinct_ratio >= args.distinct_ratio_min)
    overall = overlap_pass and drift_pass and distinct_pass

    verdict = {
        "schema": "fp8-qg-temp-top-p/1",
        "baseline": args.baseline, "candidate": args.candidate,
        "aggregate": {"overlap_rate": round(agg_overlap, 4) if agg_overlap else None,
                      "drift_pct_mean": round(agg_drift, 4) if agg_drift else None,
                      "distinct_ratio": round(agg_distinct_ratio, 4) if agg_distinct_ratio else None,
                      "n_samples": len(per), "n_pairs": len(overlap_all)},
        "gates": {"overlap_min": args.overlap_min, "drift_pct_max": args.drift_pct_max,
                  "distinct_ratio_min": args.distinct_ratio_min},
        "verdicts": {"overlap_pass": overlap_pass, "drift_pass": drift_pass,
                     "distinct_pass": distinct_pass, "overall_pass": overall},
        "per_sample": per,
    }
    rc.write_json(args.out, verdict)
    print(f"[temp_top_p] overlap={verdict['aggregate']['overlap_rate']} "
          f"drift={verdict['aggregate']['drift_pct_mean']} "
          f"distinct_ratio={verdict['aggregate']['distinct_ratio']}")
    print(f"[temp_top_p] overall={'PASS' if overall else 'FAIL'}")
    print(f"[temp_top_p] -> {args.out}")
    return 0 if overall else 1


def main():
    ap = argparse.ArgumentParser(description="温度采样 / top-p 抽验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("collect")
    pc.add_argument("--ref", default=os.path.join(HERE, "reference_set.json"))
    pc.add_argument("--quant", required=True, choices=["bf16", "fp8"])
    pc.add_argument("--temperature", type=float, default=0.6)
    pc.add_argument("--top-p", type=float, default=1.0)
    pc.add_argument("--top-logprobs", type=int, default=10)
    pc.add_argument("--reps", type=int, default=5)
    pc.add_argument("--subset", default="")
    pc.add_argument("--tag", default="")
    pc.add_argument("--out", default=None)
    pa = sub.add_parser("analyze")
    pa.add_argument("--baseline", required=True)
    pa.add_argument("--candidate", required=True)
    pa.add_argument("--top-k", type=int, default=None)
    pa.add_argument("--overlap-min", type=float, default=0.90)
    pa.add_argument("--drift-pct-max", type=float, default=1.0)
    pa.add_argument("--distinct-ratio-min", type=float, default=0.90)
    pa.add_argument("--out", default="assets/temp_top_p_verdict.json")
    args = ap.parse_args()
    return cmd_collect(args) if args.cmd == "collect" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
