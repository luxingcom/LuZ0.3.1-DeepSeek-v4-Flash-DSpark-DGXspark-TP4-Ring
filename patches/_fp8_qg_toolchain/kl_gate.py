#!/usr/bin/env python3
"""kl_gate.py — KL 门 + 困惑度门（BF16 参考 vs FP8 候选，离线分析，零 GPU）
================================================================================
用途: 在参考集上计算 KL(p_bf16 ‖ p_fp8) 与困惑度 Δ，对照噪声底门限判定。

判据（设计文档 §6，与 fp8-quality-impact §3.3 一致）:
  G2 KL 门: 加权 KL(p_bf16‖p_fp8) < kl_multiplier × noise_floor.kl_floor
            （默认 3×；--kl-multiplier 可调）
  G3 困惑度: 加权 ΔPPL = PPL_fp8 − PPL_bf16 ≤ ppl_delta_gate
            （ppl_delta_gate 优先取 noise_floor 的推荐值——含重标定；否则固定 0.05）
  综合: G2 且 G3 通过 => PASS；否则 FAIL 并列出超限样本（定位热点）。

用法:
  python3 kl_gate.py gate \
      --baseline runs/run_bf16_dist_run1.json \
      --candidate runs/run_fp8_dist.json \
      --noise-floor assets/noise_floor.json \
      --out assets/kl_verdict.json
  python3 kl_gate.py ppl  --baseline ... --candidate ...   # 仅看困惑度
================================================================================
"""
import argparse
import json
import os
import sys

import run_common as rc


def _pair_metrics(run_ref, run_cand):
    """每样本: KL(ref‖cand), ppl_ref, ppl_cand, ppl_delta。返回 (per, weights)。"""
    common, by_r, by_c = rc.align_samples(run_ref, run_cand)
    per, weights = {}, {}
    for sid in common:
        sr, sc = by_r[sid], by_c[sid]
        if not sr["outputs"] or not sc["outputs"]:
            continue
        or_, oc = sr["outputs"][0], sc["outputs"][0]
        lps_r = or_.get("logprobs") or []
        lps_c = oc.get("logprobs") or []
        tl_r = or_.get("top_logprobs") or []
        tl_c = oc.get("top_logprobs") or []
        n = min(len(lps_r), len(lps_c))
        if n == 0:
            continue
        kls = []
        for i in range(min(len(tl_r), len(tl_c))):
            k = rc.kl_common_support(tl_r[i], tl_c[i])
            if k is not None:
                kls.append(k)
        ppl_r = rc.ppl_from_logprobs(lps_r)
        ppl_c = rc.ppl_from_logprobs(lps_c)
        per[sid] = {
            "tier": sr.get("tier"),
            "category": sr.get("category"),
            "n_tokens": n,
            "kl_ref_cand": round(sum(kls) / len(kls), 6) if kls else None,
            "ppl_ref": round(ppl_r, 6) if ppl_r else None,
            "ppl_cand": round(ppl_c, 6) if ppl_c else None,
            "ppl_delta": round(ppl_c - ppl_r, 6) if (ppl_r and ppl_c) else None,
        }
        weights[sid] = n
    return per, weights


def _load_noise_floor(path):
    if not path or not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def cmd_gate(args):
    run_ref = rc.load_run(args.baseline)
    run_cand = rc.load_run(args.candidate)
    per, weights = _pair_metrics(run_ref, run_cand)
    if not per:
        print("[error] 无可计算样本", file=sys.stderr)
        return 1
    kl_mean, _ = rc.mean_wt(per, "kl_ref_cand", weights)
    ppl_delta_mean, _ = rc.mean_wt(per, "ppl_delta", weights)

    nf = _load_noise_floor(args.noise_floor)
    KL_EPS = 1e-4  # 与 noise_floor 一致的最小门限下限
    if nf:
        th = nf["thresholds"]
        if args.kl_multiplier is None:
            kl_gate = th["kl_gate"]
        else:
            kl_gate = max(args.kl_multiplier * nf["aggregate"]["kl_floor"], KL_EPS)
        ppl_gate = th["ppl_delta_gate"] if args.ppl_delta_max is None else \
            args.ppl_delta_max
        recalibrated = th.get("recalibrated", False)
    else:
        kl_gate = max(args.kl_multiplier * 1e-4, KL_EPS) if args.kl_multiplier else 0.0003
        ppl_gate = args.ppl_delta_max if args.ppl_delta_max else 0.05
        recalibrated = False
        print("[warn] 无 noise_floor 文件：KL 门用占位底（3e-4），"
              "仅适合自检/占位，正式判定必须先测噪声底", file=sys.stderr)

    kl_pass = (kl_mean if kl_mean is not None else float("inf")) < kl_gate
    ppl_pass = (ppl_delta_mean if ppl_delta_mean is not None else float("inf")) <= ppl_gate
    overall = kl_pass and ppl_pass

    outliers = [{"id": sid, **v} for sid, v in per.items()
                if (v.get("kl_ref_cand") is not None and v["kl_ref_cand"] >= kl_gate)
                or (v.get("ppl_delta") is not None and v["ppl_delta"] > ppl_gate)]
    outliers.sort(key=lambda x: -(x.get("kl_ref_cand") or 0))

    verdict = {
        "schema": "fp8-qg-kl-gate/1",
        "baseline": args.baseline, "candidate": args.candidate,
        "noise_floor": args.noise_floor,
        "aggregate": {"kl_ref_cand": kl_mean, "ppl_delta": ppl_delta_mean,
                      "ppl_ref": rc.mean_wt(per, "ppl_ref", weights)[0],
                      "ppl_cand": rc.mean_wt(per, "ppl_cand", weights)[0],
                      "n_samples": len(per)},
        "gates": {"kl_gate": round(kl_gate, 6), "ppl_delta_gate": round(ppl_gate, 6),
                  "recalibrated": recalibrated},
        "verdicts": {"kl_pass": kl_pass, "ppl_pass": ppl_pass, "overall_pass": overall},
        "n_outliers": len(outliers),
        "outliers": outliers[:args.top_outliers],
        "per_sample": per,
    }
    rc.write_json(args.out, verdict)
    print(f"[kl_gate] kl={kl_mean} gate={round(kl_gate,6)} "
          f"pass={kl_pass}")
    print(f"[ppl_gate] delta={ppl_delta_mean} gate={round(ppl_gate,6)} "
          f"pass={ppl_pass} recalibrated={recalibrated}")
    print(f"[kl_gate] overall={'PASS' if overall else 'FAIL'} "
          f"outliers={len(outliers)}")
    print(f"[kl_gate] -> {args.out}")
    return 0 if overall else 1


def cmd_ppl(args):
    run_ref = rc.load_run(args.baseline)
    run_cand = rc.load_run(args.candidate)
    per, weights = _pair_metrics(run_ref, run_cand)
    ppl_delta, _ = rc.mean_wt(per, "ppl_delta", weights)
    print(f"[ppl] n={len(per)} delta={ppl_delta}")
    for sid, v in sorted(per.items(), key=lambda kv: -(kv[1]["ppl_delta"] or 0))[:10]:
        print(f"  {sid}: ppl_ref={v['ppl_ref']} ppl_cand={v['ppl_cand']} "
              f"delta={v['ppl_delta']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="KL 门 + 困惑度门")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("gate")
    pg.add_argument("--baseline", required=True)
    pg.add_argument("--candidate", required=True)
    pg.add_argument("--noise-floor", default=None)
    pg.add_argument("--kl-multiplier", type=float, default=None)
    pg.add_argument("--ppl-delta-max", type=float, default=None)
    pg.add_argument("--out", default="assets/kl_verdict.json")
    pg.add_argument("--top-outliers", type=int, default=10)
    pp = sub.add_parser("ppl")
    pp.add_argument("--baseline", required=True)
    pp.add_argument("--candidate", required=True)
    args = ap.parse_args()
    return cmd_gate(args) if args.cmd == "gate" else cmd_ppl(args)


if __name__ == "__main__":
    sys.exit(main())
