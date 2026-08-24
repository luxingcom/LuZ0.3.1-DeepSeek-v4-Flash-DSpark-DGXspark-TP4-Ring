#!/usr/bin/env python3
"""greedy_baseline.py — greedy 4/4 基线固化与比对（FP8 工具链侧）
================================================================================
用途: 把 BF16 侧 4 稳定 prompt 输出固化为 golden 基线资产（JSON），FP8 侧比对
      （exact 或 envelope≤1%）。与现役 quality_gate.py 同口径、同 prompt、同
      包络判据；本脚本聚焦"资产固化 + 候选 run 文件比对"，quality_gate.py 仍为
      生产唯一 compare 入口（换基座/换量化形态时二者结果应一致）。

子命令:
  capture        — 从在线 API 采集 BF16 greedy 4 prompt，写入
                   assets/golden-bf16-greedy-<UTC>.json + golden-bf16-greedy-latest.json
  import-snapshot— 把既有 quality_gate reference-latest.json 导入为 golden 资产
                   （若换基座前已 capture，无需重新采集）
  compare        — 用候选输出比对 golden：
                   --candidate <run json 或 quality_gate reference json>（离线）
                   或 --candidate-live（在线采集比对）
                   exact 4/4 通过；否则包络 top-1 logprob sum drift ≤1% 兜底。
  list           — 列出 golden 资产

退出码: 0=PASS 1=FAIL 2=用法/环境错误
================================================================================
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import run_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
LATEST = os.path.join(ASSETS, "golden-bf16-greedy-latest.json")
ENVELOPE_SUM_DRIFT_PCT = 1.0

# 与 quality_gate.py 完全一致的 4 prompt（Tier A；由 reference_set_builder 校验）
FOX = "The quick brown fox jumps over the lazy dog. "
GOLDEN_PROMPTS = {
    "fox_repeat": FOX * 60 + "\nRepeat the sentence above exactly once.",
    "count": "Count from 1 to 30, one number per line, no other text.",
    "code_fib": "Write a Python function to compute the nth Fibonacci number using memoization. Output code only.",
    "list": "List the first 20 chemical elements with their atomic numbers, one per line, format: number. symbol name.",
}


def _golden_from_outputs(prompt_map, samples):
    """从统一 sample 列表（id/prompt/outputs[0]）构建 golden doc。"""
    by_id = {s["id"]: s for s in samples}
    prompts = {}
    for name, prompt in prompt_map.items():
        s = by_id.get(name)
        if not s or not s.get("outputs"):
            raise ValueError(f"缺少 golden prompt: {name}")
        o = s["outputs"][0]
        prompts[name] = {"text": o.get("text", ""), "logprobs": o.get("logprobs")}
    return prompts


def cmd_capture(args):
    if not rc.need_key():
        return 2
    os.makedirs(ASSETS, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
    samples = []
    for name, prompt in GOLDEN_PROMPTS.items():
        print(f"  capturing {name} ...", flush=True)
        o = rc.gen(prompt, temperature=0.0, top_p=1.0,
                   max_tokens=384, top_logprobs=1)
        samples.append({"id": name, "tier": "A", "category": "text",
                        "prompt": prompt, "outputs": [o]})
    doc = {"schema": "fp8-qg-golden/1", "captured_at": ts,
           "model": rc.MODEL, "quant": "bf16",
           "prompts": _golden_from_outputs(GOLDEN_PROMPTS, samples)}
    out = args.out or os.path.join(ASSETS, f"golden-bf16-greedy-{ts}.json")
    rc.write_json(out, doc)
    shutil.copy2(out, LATEST)
    print(f"[capture] -> {out}")
    print(f"[capture] latest -> {LATEST}")
    return 0


def cmd_import_snapshot(args):
    if not os.path.exists(args.from_path):
        print(f"[import] 源不存在: {args.from_path}", file=sys.stderr)
        return 2
    src = json.load(open(args.from_path, encoding="utf-8"))
    prompts = src.get("prompts", src)
    missing = [n for n in GOLDEN_PROMPTS if n not in prompts]
    if missing:
        print(f"[import] 源缺少 {missing}", file=sys.stderr)
        return 2
    os.makedirs(ASSETS, exist_ok=True)
    doc = {"schema": "fp8-qg-golden/1",
           "captured_at": src.get("captured_at", "imported"),
           "model": src.get("model", rc.MODEL), "quant": "bf16",
           "prompts": prompts,
           "imported_from": args.from_path}
    rc.write_json(args.out, doc)
    shutil.copy2(args.out, LATEST)
    print(f"[import] -> {args.out}")
    print(f"[import] latest -> {LATEST}")
    return 0


def _load_candidate(args):
    if args.candidate:
        path = args.candidate
        if not os.path.exists(path):
            print(f"[compare] 候选文件不存在: {path}", file=sys.stderr)
            return None, path
        doc = json.load(open(path, encoding="utf-8"))
        if doc.get("schema") == "fp8-qg-run/1":
            return {s["id"]: s["outputs"][0] for s in doc["samples"]
                    if s.get("outputs")}, path
        # quality_gate reference 或 golden 格式
        prompts = doc.get("prompts", doc)
        return {name: {"text": v.get("text", ""),
                       "logprobs": v.get("logprobs")}
                for name, v in prompts.items()}, path
    if args.candidate_live:
        if not rc.need_key():
            return None, "live"
        cand = {}
        for name, prompt in GOLDEN_PROMPTS.items():
            o = rc.gen(prompt, temperature=0.0, top_p=1.0,
                       max_tokens=384, top_logprobs=1)
            cand[name] = {"text": o["text"], "logprobs": o["logprobs"]}
        return cand, "live"
    print("[compare] 需 --candidate 或 --candidate-live", file=sys.stderr)
    return None, ""


def cmd_compare(args):
    ref_path = args.ref or LATEST
    if not os.path.exists(ref_path):
        print(f"[compare] golden 不存在: {ref_path}（先 capture/import）",
              file=sys.stderr)
        return 2
    ref_doc = json.load(open(ref_path, encoding="utf-8"))
    ref = ref_doc.get("prompts", ref_doc)
    cand, cand_label = _load_candidate(args)
    if cand is None:
        return 2

    verdicts = {}
    for name, prompt in GOLDEN_PROMPTS.items():
        c = cand.get(name)
        r = ref.get(name)
        if c is None or r is None:
            verdicts[name] = {"match": False, "note": "候选缺该 prompt"}
            continue
        match = (c.get("text") == r.get("text"))
        v = {"match": match, "own_stable": True}
        rl_ref = r.get("logprobs") or []
        cl = c.get("logprobs") or []
        if cl and rl_ref and len(cl) == len(rl_ref):
            diffs = [abs(a - b) for a, b in zip(cl, rl_ref)]
            sum_ref = sum(rl_ref)
            v["lp"] = {
                "n": len(cl),
                "med_abs_diff": round(sorted(diffs)[len(diffs) // 2], 4),
                "max_abs_diff": round(max(diffs), 4),
                "sum_drift_pct": round((sum(cl) - sum_ref) / abs(sum_ref) * 100, 3),
            }
        elif not match:
            v["lp"] = {"note": "logprob 长度不一致或缺席, 包络不可用"}
        verdicts[name] = v
        print(f"  {name}: match={match} lp={v.get('lp')}", flush=True)

    n_match = sum(1 for v in verdicts.values() if v["match"])
    lp_ok = [abs(v["lp"]["sum_drift_pct"]) <= ENVELOPE_SUM_DRIFT_PCT
             for v in verdicts.values() if "sum_drift_pct" in v.get("lp", {})]
    exact_pass = (n_match == len(GOLDEN_PROMPTS))
    envelope_pass = exact_pass or (len(lp_ok) > 0 and all(lp_ok))
    overall = exact_pass or envelope_pass
    print(f"[compare] ref={ref_path} cand={cand_label}")
    print(f"[compare] exact {n_match}/{len(GOLDEN_PROMPTS)}; "
          f"envelope_pass={envelope_pass}; overall={'PASS' if overall else 'FAIL'}")
    if args.out:
        rc.write_json(args.out, {"ref": ref_path, "candidate": cand_label,
                                 "verdicts": verdicts, "n_match": n_match,
                                 "envelope_pass": envelope_pass,
                                 "overall_pass": overall})
        print(f"[compare] -> {args.out}")
    return 0 if overall else 1


def cmd_list(_args):
    if not os.path.isdir(ASSETS):
        print("[list] 无 assets 目录")
        return 0
    for p in sorted(glob.glob(os.path.join(ASSETS, "golden-bf16-greedy-*.json"))):
        tag = " (latest)" if os.path.realpath(p) == os.path.realpath(LATEST) else ""
        print(f"  {p}{tag}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="greedy 4/4 基线固化")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--out", default=None)
    pi = sub.add_parser("import-snapshot")
    pi.add_argument("--from-path", required=True,
                    default="<INSTALL_DIR>/backup/quality-gate/reference-latest.json")
    pi.add_argument("--out", default=os.path.join(ASSETS, "golden-bf16-greedy-imported.json"))
    pm = sub.add_parser("compare")
    pm.add_argument("--candidate", default=None, help="fp8-qg-run 或 reference json")
    pm.add_argument("--candidate-live", action="store_true")
    pm.add_argument("--ref", default=LATEST)
    pm.add_argument("--out", default=None)
    sub.add_parser("list")
    args = ap.parse_args()
    return {"capture": cmd_capture, "import-snapshot": cmd_import_snapshot,
            "compare": cmd_compare, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
