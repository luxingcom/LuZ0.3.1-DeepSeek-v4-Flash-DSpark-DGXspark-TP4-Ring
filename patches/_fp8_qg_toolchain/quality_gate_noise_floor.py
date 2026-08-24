#!/usr/bin/env python3
"""quality_gate_noise_floor.py — BF16 噪声底测量（离线分析，零 GPU）
================================================================================
用途: 对 BF16 同配置背靠背两次采集（run1/run2），计算每样本与聚合指标，产出
      噪声底表 + 推荐门限（KL 2-3×、困惑度 Δ 与重标定逻辑）。

指标（fp8-qg-run/1 输入）:
  每样本:
    kl_ab / kl_ba : 受限 top-k 支持 KL（nats/token），A‖B 与 B‖A 两方向
    ppl_a / ppl_b : 困惑度（top-1 采样 token logprob 口径）
    ppl_delta_abs : |ppl_b - ppl_a|
    lp_drift      : logprob_drift 统计（n/mean_abs_diff/max_abs_diff/sum_drift_pct）
  聚合:
    kl_floor      : max(加权 mean kl_ab, 加权 mean kl_ba) —— 噪声底（保守取大）
    ppl_floor     : 加权 |ppl_delta|
    lp_mean_abs   : 加权 mean_abs_diff
  推荐门限（--multiplier，默认 3.0）:
    kl_gate       = multiplier × kl_floor
    ppl_delta_gate = max(ppl_delta_max, multiplier × ppl_floor)   # 重标定逻辑
    recalibrated   = (multiplier × ppl_floor > ppl_delta_max) 即噪声底超 0.05
  退出码: 0=成功（门限已产出）；1=分析错误。

用法:
  python3 quality_gate_noise_floor.py analyze run1.json run2.json \
      --out assets/noise_floor.json [--multiplier 3.0] [--ppl-delta-max 0.05]
  python3 quality_gate_noise_floor.py show noise_floor.json
================================================================================
"""
import argparse
import json
import os
import sys

import run_common as rc


def _per_sample(run_a, run_b, common, by_a, by_b):
    per = {}
    weights = {}
    for sid in common:
        sa, sb = by_a[sid], by_b[sid]
        if not sa["outputs"] or not sb["outputs"]:
            continue
        oa, ob = sa["outputs"][0], sb["outputs"][0]
        lps_a = oa.get("logprobs") or []
        lps_b = ob.get("logprobs") or []
        tl_a = oa.get("top_logprobs") or []
        tl_b = ob.get("top_logprobs") or []
        n = min(len(lps_a), len(lps_b))
        if n == 0:
            continue
        kl_ab, kl_ba = [], []
        for i in range(min(len(tl_a), len(tl_b))):
            k1 = rc.kl_common_support(tl_a[i], tl_b[i])
            k2 = rc.kl_common_support(tl_b[i], tl_a[i])
            if k1 is not None:
                kl_ab.append(k1)
            if k2 is not None:
                kl_ba.append(k2)
        ppl_a = rc.ppl_from_logprobs(lps_a)
        ppl_b = rc.ppl_from_logprobs(lps_b)
        entry = {
            "tier": sa.get("tier"),
            "category": sa.get("category"),
            "n_tokens": n,
            "kl_ab": round(sum(kl_ab) / len(kl_ab), 6) if kl_ab else None,
            "kl_ba": round(sum(kl_ba) / len(kl_ba), 6) if kl_ba else None,
            "ppl_a": round(ppl_a, 6) if ppl_a else None,
            "ppl_b": round(ppl_b, 6) if ppl_b else None,
            "ppl_delta_abs": round(abs(ppl_b - ppl_a), 6) if (ppl_a and ppl_b) else None,
            "lp_drift": rc.logprob_drift(lps_a, lps_b),
        }
        per[sid] = entry
        weights[sid] = n
    return per, weights


def cmd_analyze(args):
    run_a = rc.load_run(args.run1)
    run_b = rc.load_run(args.run2)
    common, by_a, by_b = rc.align_samples(run_a, run_b)
    if not common:
        print("[error] 两 run 无公共样本", file=sys.stderr)
        return 1
    per, weights = _per_sample(run_a, run_b, common, by_a, by_b)
    if not per:
        print("[error] 无可计算样本（缺 logprobs/top_logprobs）", file=sys.stderr)
        return 1

    kl_ab, _ = rc.mean_wt(per, "kl_ab", weights)
    kl_ba, _ = rc.mean_wt(per, "kl_ba", weights)
    ppl_floor, _ = rc.mean_wt(per, "ppl_delta_abs", weights)

    # lp_drift 为嵌套 dict，单独聚合
    lp_vals = [p["lp_drift"]["mean_abs_diff"] for p in per.values()
               if p.get("lp_drift")]
    lp_mean = round(sum(lp_vals) / len(lp_vals), 6) if lp_vals else None

    kl_floor = max(kl_ab or 0.0, kl_ba or 0.0)
    mult = args.multiplier
    ppl_gate_raw = mult * (ppl_floor or 0.0)
    ppl_delta_gate = max(args.ppl_delta_max, ppl_gate_raw)
    recalibrated = ppl_gate_raw > args.ppl_delta_max
    # 零底保护：实测 KL 底为 0（确定性一致）时也不允许门限为 0（否则任何非零
    # KL 都误报）。取 1e-4 为最小门限下限（约 log(1.0001)，远小于可测漂移）。
    KL_EPS = 1e-4
    kl_gate = max(mult * kl_floor, KL_EPS)

    agg = {
        "run_a": args.run1, "run_b": args.run2,
        "n_samples": len(per),
        "kl_ab_mean": kl_ab, "kl_ba_mean": kl_ba,
        "kl_floor": round(kl_floor, 6),
        "ppl_floor": ppl_floor,
        "lp_mean_abs_diff": lp_mean,
    }
    thresholds = {
        "kl_multiplier": mult,
        "kl_gate": round(kl_gate, 6),
        "ppl_delta_max": args.ppl_delta_max,
        "ppl_delta_gate": round(ppl_delta_gate, 6),
        "recalibrated": recalibrated,
        "note": ("困惑度噪声底超过固定门限 0.05，门限已按 3×噪声底重标定；"
                 "请在设计文档 §10 重标定决策表登记" if recalibrated else
                 "困惑度噪声底在 0.05 内，保持固定门限"),
    }
    doc = {"schema": "fp8-qg-noise-floor/1",
           "aggregate": agg, "thresholds": thresholds,
           "per_sample": per,
           "meta": {"run_a_meta": run_a["meta"], "run_b_meta": run_b["meta"]}}
    rc.write_json(args.out, doc)
    print(f"[noise_floor] n={agg['n_samples']} "
          f"kl_floor={agg['kl_floor']} ppl_floor={agg['ppl_floor']}")
    print(f"[thresholds] kl_gate={thresholds['kl_gate']} "
          f"ppl_delta_gate={thresholds['ppl_delta_gate']} "
          f"recalibrated={thresholds['recalibrated']}")
    print(f"[noise_floor] -> {args.out}")
    return 0


def cmd_show(args):
    doc = json.load(open(args.file, encoding="utf-8"))
    print(json.dumps({"aggregate": doc["aggregate"],
                      "thresholds": doc["thresholds"]},
                     ensure_ascii=False, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description="BF16 噪声底测量")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("run1")
    pa.add_argument("run2")
    pa.add_argument("--out", default="assets/noise_floor.json")
    pa.add_argument("--multiplier", type=float, default=3.0)
    pa.add_argument("--ppl-delta-max", type=float, default=0.05)
    ps = sub.add_parser("show")
    ps.add_argument("file")
    args = ap.parse_args()
    return cmd_analyze(args) if args.cmd == "analyze" else cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
