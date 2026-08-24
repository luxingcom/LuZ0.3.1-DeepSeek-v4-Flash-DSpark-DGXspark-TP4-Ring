#!/usr/bin/env python3
"""quality_gate.py — 生产贪心质量门（固化版，LuZ0.3.1 起为唯一口径）
========================================================================
来源: phase3b golden_env.py (W-3 包络判据) + w4a4-ext greedy prompt 集清洗。
固化原因（2026-08-23 用户批准）:
  - prompt 集修正: reason/zh 两个 prompt 已除名 —— 2026-08-23 w4a4-ext 窗口
    同配置背靠背三次捕获证明其运行级非确定（概率投机解码 draft 采样非确定，
    与被测配置无关）。不稳定 prompt 会产生假 No-Go 信号（B2 初判即被触发）。
  - 历史散落脚本 (/tmp/_mtp_tune/greedy_check.py, /tmp/_wsdedup_l3/golden_env.py
    等) 均为窗口临时资产，一律指向本脚本；/tmp 可能被清理，勿再依赖。

口径（与 phase3b W-3 / w4a4-ext / bprime-window 一致）:
  1) 主判据: 4 个稳定 prompt (fox_repeat/count/code/list) temp=0 输出
     与参考快照逐字一致 4/4 => PASS
  2) 包络判据（不一致时兜底）: token 级 top-1 logprob 总和漂移
     |sum drift| <= 1% => envelope PASS（参照 P2 系统噪声带 ±0.6% 与
     内核 2 ULP 级非确定）；逐字不一致且包络超界 => FAIL 待查
  3) own_stable 复跑: 同窗口自复跑一次，若自身不稳定则该 prompt 结果
     标注 own_stable=False（运行级非确定提示，不改变判定）

参考快照管理:
  - 快照目录: <INSTALL_DIR>/backup/quality-gate/
  - capture => reference-<UTC>.json 并更新 reference-latest.json
  - compare 默认用 reference-latest.json；--ref 可指定历史快照
  - 换基座（FI 版本/量化形态/kernel 变更）时应重新 capture 并保留旧快照

用法:
  VLLM_API_KEY=... python3 quality_gate.py capture  [--out PATH]
  VLLM_API_KEY=... python3 quality_gate.py compare  [--ref PATH] [--out RESULT.json]
  python3 quality_gate.py list     # 列出快照
退出码: 0=PASS 1=FAIL 2=用法/环境错误
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import requests

API = "http://127.0.0.1:8001/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")
HDR = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
MODEL = "deepseek-v4-flash-0731"
FOX = "The quick brown fox jumps over the lazy dog. "
REF_DIR = "<INSTALL_DIR>/backup/quality-gate"
REF_LATEST = os.path.join(REF_DIR, "reference-latest.json")

# 稳定 prompt 集（reason/zh 已除名，见模块 docstring）
PROMPTS = [
    ("fox_repeat", FOX * 60 + "\nRepeat the sentence above exactly once."),
    ("count", "Count from 1 to 30, one number per line, no other text."),
    ("code", "Write a Python function to compute the nth Fibonacci number using memoization. Output code only."),
    ("list", "List the first 20 chemical elements with their atomic numbers, one per line, format: number. symbol name."),
]

ENVELOPE_SUM_DRIFT_PCT = 1.0  # 包络: top-1 logprob 总和漂移 <= 1%


def gen(prompt, max_tokens=384, want_lp=False):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }
    if want_lp:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    r = requests.post(API, headers=HDR, json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    txt = j["choices"][0]["message"].get("content", "")
    lps = None
    if want_lp and j["choices"][0].get("logprobs"):
        lps = [c["logprob"] for c in j["choices"][0]["logprobs"]["content"]]
    return txt, lps


def cmd_capture(args):
    os.makedirs(REF_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
    out = args.out or os.path.join(REF_DIR, f"reference-{ts}.json")
    res = {"captured_at": ts, "model": MODEL, "prompts": {}}
    for name, p in PROMPTS:
        txt, lps = gen(p, want_lp=True)
        res["prompts"][name] = {"text": txt, "logprobs": lps}
        print(f"  captured {name}: {len(txt)} chars lp={len(lps) if lps else '-'}",
              flush=True)
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    shutil.copy2(out, REF_LATEST)
    print(f"[capture] saved -> {out}")
    print(f"[capture] latest updated -> {REF_LATEST}")
    return 0


def cmd_compare(args):
    ref_path = args.ref or REF_LATEST
    if not os.path.exists(ref_path):
        print(f"[compare] 参考快照不存在: {ref_path}（先 capture）", file=sys.stderr)
        return 2
    ref_doc = json.load(open(ref_path))
    ref = ref_doc.get("prompts", ref_doc)
    verdicts = {}
    for name, p in PROMPTS:
        txt, lps = gen(p, want_lp=True)
        txt2, _ = gen(p)
        own_stable = (txt == txt2)
        match = (txt == ref[name]["text"])
        v = {"match": match, "own_stable": own_stable}
        rl_ref = ref[name].get("logprobs") or []
        if lps and rl_ref and len(lps) == len(rl_ref):
            diffs = [abs(a - b) for a, b in zip(lps, rl_ref)]
            sum_ref = sum(rl_ref)
            v["lp"] = {
                "n": len(lps),
                "med_abs_diff": round(sorted(diffs)[len(diffs) // 2], 4),
                "max_abs_diff": round(max(diffs), 4),
                "sum_drift_pct": round((sum(lps) - sum_ref) / abs(sum_ref) * 100, 3),
            }
        elif not match:
            v["lp"] = {"note": "logprob 长度不一致或缺席, 包络不可用"}
        verdicts[name] = v
        print(f"  {name}: match={match} own_stable={own_stable} "
              f"lp={v.get('lp')}", flush=True)
    n_match = sum(1 for v in verdicts.values() if v["match"])
    lp_ok = [abs(v["lp"]["sum_drift_pct"]) <= ENVELOPE_SUM_DRIFT_PCT
             for v in verdicts.values() if "sum_drift_pct" in v.get("lp", {})]
    exact_pass = (n_match == len(PROMPTS))
    envelope_pass = exact_pass or (len(lp_ok) > 0 and all(lp_ok))
    overall = exact_pass or envelope_pass
    print(f"[compare] ref={ref_path}")
    print(f"[compare] exact_match {n_match}/{len(PROMPTS)}; "
          f"logprob_envelope_pass={envelope_pass}; "
          f"overall={'PASS' if overall else 'FAIL'}")
    if args.out:
        json.dump({"ref": ref_path, "verdicts": verdicts, "n_match": n_match,
                   "envelope_pass": envelope_pass, "overall_pass": overall},
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"[compare] result -> {args.out}")
    return 0 if overall else 1


def cmd_list(_args):
    if not os.path.isdir(REF_DIR):
        print(f"[list] 无快照目录 {REF_DIR}")
        return 0
    for p in sorted(glob.glob(os.path.join(REF_DIR, "reference-*.json"))):
        tag = " (latest)" if os.path.realpath(p) == os.path.realpath(REF_LATEST) \
            else ""
        print(f"  {p}{tag}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="生产贪心质量门（稳定 4 prompt + 包络判据）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cap = sub.add_parser("capture", help="抓取参考快照（换基座时执行）")
    p_cap.add_argument("--out", default=None)
    p_cmp = sub.add_parser("compare", help="与参考快照比对（质量门）")
    p_cmp.add_argument("--ref", default=None)
    p_cmp.add_argument("--out", default=None)
    sub.add_parser("list", help="列出参考快照")
    args = ap.parse_args()
    if not KEY and args.cmd != "list":
        print("[quality_gate] 缺 VLLM_API_KEY 环境变量", file=sys.stderr)
        return 2
    return {"capture": cmd_capture, "compare": cmd_compare,
            "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
