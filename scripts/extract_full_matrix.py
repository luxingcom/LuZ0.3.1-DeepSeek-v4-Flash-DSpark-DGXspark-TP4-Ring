#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整提取多个 bench summary 的 DE / PR 矩阵。
用法: python extract_full_matrix.py <summary1.json> [summary2.json ...]
按 (mode,label,conc) 合并，输出 DE(coding/json/prose) 与 PR(512..131076) 各并发的 p50 指标。"""
import json
import sys

def get_rows(p):
    d = json.load(open(p, encoding="utf-8"))
    return d.get("summary", [])

def g(row, k, default=""):
    v = row.get(k)
    if v is None:
        return default
    if isinstance(v, float):
        return round(v, 4)
    return v

def main():
    if len(sys.argv) < 2:
        sys.exit("用法: python extract_full_matrix.py <summary1.json> [summary2.json ...]")
    allr = []
    for p in sys.argv[1:]:
        allr += get_rows(p)

    # key = (mode, label, conc)  ; label 为 task_type 或 prefix
    de = {}   # (task, conc) -> row   ; task in {coding,json,prose}
    pr = {}   # (prefix, conc) -> row ; prefix in {512,2048,8192,32768,131076}
    for row in allr:
        conc = int(row["concurrency"])
        if row["mode"] == "de":
            task = row["task_type"]
            key = ("de", task, conc)
            if key not in de or row.get("requests_ok",0) >= de[key].get("requests_ok",0):
                de[key] = row
        elif row["mode"] == "pr":
            pref = str(row.get("prefix_len"))
            key = ("pr", pref, conc)
            if key not in pr or row.get("requests_ok",0) >= pr[key].get("requests_ok",0):
                pr[key] = row

    concs = [1,2,4,6,8,12]

    # ---------- DE ----------
    print("# DE 性能矩阵（input 512 → output 4096，每并发3波）\n")
    for task in ["coding","json","prose"]:
        print(f"\n## DE · {task}\n")
        print("| 并发 | ok/total | p50 prefill_tps | p50 decode_tps | p50 TTFT(s) | p50 total(s) | accept | 实际并发 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for conc in concs:
            row = de.get(("de", task, conc))
            if not row:
                print(f"| {conc} | 缺失 | - | - | - | - | - | - |")
                continue
            ok = g(row,"requests_ok"); tot = g(row,"requests_total")
            mon = row.get("monitor") or {}
            reached = mon.get("reached_full_concurrency")
            print(f"| {conc} | {ok}/{tot} | {g(row,'p50_prefill_tps')} | {g(row,'p50_decode_tps')} | {g(row,'p50_ttft_s')} | {g(row,'p50_total_s')} | {g(row,'acceptance_rate')} | {'打满' if reached else '未打满⚠️'} |")

    # ---------- PR ----------
    print("\n\n# PR 性能矩阵（纯 prefill，output=1 token，每并发3波）\n")
    for pref in ["512","2048","8192","32768","131076"]:
        print(f"\n## PR · prefix={pref}\n")
        print("| 并发 | ok/total | p50 prefill_tps | p50 TTFT(s) | p50 total(s) | accept | 实际并发 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for conc in concs:
            row = pr.get(("pr", pref, conc))
            if not row:
                print(f"| {conc} | 缺失 | - | - | - | - | - |")
                continue
            ok = g(row,"requests_ok"); tot = g(row,"requests_total")
            mon = row.get("monitor") or {}
            reached = mon.get("reached_full_concurrency")
            print(f"| {conc} | {ok}/{tot} | {g(row,'p50_prefill_tps')} | {g(row,'p50_ttft_s')} | {g(row,'p50_total_s')} | {g(row,'acceptance_rate')} | {'打满' if reached else '未打满⚠️'} |")

if __name__ == "__main__":
    main()
