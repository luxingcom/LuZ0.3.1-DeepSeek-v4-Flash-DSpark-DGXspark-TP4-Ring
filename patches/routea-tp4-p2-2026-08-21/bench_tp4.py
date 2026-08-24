#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_tp4.py — P2 TP4 三配置基准（TTFT/decode/logprob 统计/needle）。
运行: 容器 --network host, env VLLM_API_KEY。输出 JSON 到 --out。"""
import argparse
import json
import random
import time

import requests

BASE = "http://127.0.0.1:8001/v1"
MODEL = "deepseek-v4-flash-0731"


def api_key():
    import os
    return os.environ["VLLM_API_KEY"]


def hdr():
    return {"Authorization": f"Bearer {api_key()}"}


def wait_health(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if r.status_code == 200:
                print(f"[health] OK ({time.time()-t0:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


# ---------- filler / prompts（确定性, 三运行完全一致）----------
def make_filler(n_sent, seed=7):
    rng = random.Random(seed)
    subjects = ["the survey team", "a quiet customs officer", "the lighthouse keeper",
                "an aging tram conductor", "the night-shift engineer", "a cartographer",
                "the harbor master", "an apprentice baker", "the radio operator"]
    verbs = ["catalogued", "measured", "repaired", "photographed", "archived",
             "calibrated", "sketched", "inspected", "logged"]
    objs = ["seventeen brass instruments", "the tidal ledger", "a crate of spare valves",
            "the coastal fog charts", "four crates of citrus", "the signal lamp lenses",
            "a bundle of telegraph forms", "the auxiliary generator", "the tide gauge"]
    out = []
    for i in range(n_sent):
        out.append(f"Record {i}: {rng.choice(subjects)} {rng.choice(verbs)} "
                   f"{rng.choice(objs)} on day {100 + i % 800} of the expedition.")
    return " ".join(out)


def ttft_test(prompt, max_tokens=8, reps=5):
    res = []
    for _ in range(reps):
        t0 = time.time()
        first = None
        with requests.post(f"{BASE}/completions", headers=hdr(), stream=True, timeout=600,
                           json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                                 "temperature": 0.0, "stream": True}) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    try:
                        j = json.loads(payload)
                        ch = j["choices"][0]
                        if ch.get("text"):
                            if first is None:
                                first = time.time() - t0
                    except Exception:
                        pass
        res.append({"ttft_s": round(first, 3) if first else None})
        time.sleep(2)
    return res


def decode_test(prompt, max_tokens=256, reps=3):
    res = []
    for _ in range(reps):
        t0 = time.time()
        first = None
        ntok = 0
        with requests.post(f"{BASE}/completions", headers=hdr(), stream=True, timeout=600,
                           json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                                 "temperature": 0.0, "stream": True}) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    try:
                        j = json.loads(payload)
                        ch = j["choices"][0]
                        if ch.get("text"):
                            ntok += 1
                            if first is None:
                                first = time.time() - t0
                    except Exception:
                        pass
        total = time.time() - t0
        decode_s = total - (first or 0)
        res.append({"ttft_s": round(first, 3) if first else None,
                    "total_s": round(total, 3),
                    "decode_tps": round((ntok - 1) / decode_s, 2) if decode_s > 0 and ntok > 1 else None})
        time.sleep(2)
    return res


def concurrent_decode(prompt, n=12, max_tokens=128, reps=2):
    import concurrent.futures as cf
    res = []
    for _ in range(reps):
        t0 = time.time()
        def one(_):
            with requests.post(f"{BASE}/completions", headers=hdr(), stream=True, timeout=600,
                               json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                                     "temperature": 0.0, "stream": True}) as r:
                n = 0
                for line in r.iter_lines():
                    if line and line.startswith(b"data: ") and line[6:] != b"[DONE]":
                        try:
                            if json.loads(line[6:])["choices"][0].get("text"):
                                n += 1
                        except Exception:
                            pass
                return n
        with cf.ThreadPoolExecutor(n) as ex:
            counts = list(ex.map(one, range(n)))
        dt = time.time() - t0
        res.append({"n_req": n, "total_s": round(dt, 3),
                    "agg_tps": round(sum(counts) / dt, 2)})
        time.sleep(3)
    return res


LOGPROB_PROMPTS = [
    "The history of computing hardware spans from mechanical calculators to modern "
    "quantum processors. Trace the major milestones from 1940 to 2020, focusing on "
    "the transition from vacuum tubes to transistors to integrated circuits.",
    "计算圆周率的历史方法有很多。请解释蒙特卡洛方法估算 π 的原理, 给出具体的伪代码实现。",
    "In 2019, a company reported revenue of $4,872,300 and expenses of $3,191,450. "
    "In 2020, revenue grew by 23.7% while expenses grew by 11.2%.",
    "全球气候系统的反馈机制是理解变暖预测的关键。请分别说明水蒸气反馈、冰盖-反照率反馈。",
    "量子纠缠与经典关联的本质区别是什么? 请从 Bell 不等式的数学表述出发。",
] + [make_filler(60, seed=11), make_filler(120, seed=13)]


def logprob_stats():
    totals = []
    for i, p in enumerate(LOGPROB_PROMPTS):
        try:
            r = requests.post(f"{BASE}/completions", headers=hdr(), timeout=300, json={
                "model": MODEL, "prompt": p, "max_tokens": 1, "temperature": 0.0,
                "echo": True, "logprobs": 0})
            j = r.json()
            lp = j["choices"][0]["logprobs"]["token_logprobs"]
            tot = sum(x for x in lp if x is not None)
            totals.append({"prompt_id": i, "n_tok": len([x for x in lp if x is not None]),
                           "sum_lp": round(tot, 3)})
        except Exception as e:
            totals.append({"prompt_id": i, "error": str(e)[:100]})
        time.sleep(1)
    return totals


def needle_test(ctx_tokens_target, secret, reps=1, seed=3):
    # 句子≈17 token, 估算句数
    n_sent = int(ctx_tokens_target / 17)
    rng = random.Random(seed)
    res = []
    for rep in range(reps):
        filler1 = make_filler(n_sent // 2, seed=seed + rep)
        filler2 = make_filler(n_sent - n_sent // 2, seed=seed + 100 + rep)
        pos = rng.choice(["mid", "early", "late"])
        if pos == "early":
            prompt = (f"MEMO: the access code for this shift is {secret}.\n"
                      f"{filler1} {filler2}\n\n"
                      f"Question: What is the access code? Reply with the code only.")
        elif pos == "late":
            prompt = (f"{filler1} {filler2}\n"
                      f"MEMO: the access code for this shift is {secret}.\n\n"
                      f"Question: What is the access code? Reply with the code only.")
        else:
            prompt = (f"{filler1}\nMEMO: the access code for this shift is {secret}.\n"
                      f"{filler2}\n\nQuestion: What is the access code? Reply with the code only.")
        try:
            t0 = time.time()
            r = requests.post(f"{BASE}/completions", headers=hdr(), timeout=900, json={
                "model": MODEL, "prompt": prompt, "max_tokens": 48, "temperature": 0.0})
            txt = r.json()["choices"][0]["text"]
            res.append({"target_ctx_tok": ctx_tokens_target, "pos": pos,
                        "pass": secret in txt, "resp": txt[:60],
                        "latency_s": round(time.time() - t0, 1)})
        except Exception as e:
            res.append({"target_ctx_tok": ctx_tokens_target, "error": str(e)[:100]})
        time.sleep(2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-needle", action="store_true")
    ap.add_argument("--needle-only", action="store_true")
    args = ap.parse_args()
    if not wait_health():
        print("[fatal] health timeout")
        json.dump({"fatal": "health timeout"}, open(args.out, "w"))
        return
    out = {"ts": time.time()}
    if args.needle_only:
        out["needle_64k"] = needle_test(64000, "X7Q-95-Bravo", reps=3)
        out["needle_128k"] = needle_test(128000, "Zulu-31-Kilo", reps=2)
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"[done] wrote {args.out}")
        return
    # TTFT（chunked prefill 4096 → 每 chunk M=4096 = W4A4 目标档）
    out["ttft_4k"] = ttft_test(make_filler(240, seed=21), reps=5)     # ≈4.1K tok
    out["ttft_16k"] = ttft_test(make_filler(960, seed=22), reps=3)    # ≈16K tok
    out["ttft_2k"] = ttft_test(make_filler(120, seed=23), reps=5)     # ≈2K tok (中段)
    # decode
    short = "Briefly explain why the sea appears blue to a sailor on a clear day."
    out["decode_1x256"] = decode_test(short, 256, reps=3)
    out["decode_12x128"] = concurrent_decode(short, 12, 128, reps=2)
    # logprob 统计
    out["logprob"] = logprob_stats()
    # needle
    if not args.skip_needle:
        out["needle_64k"] = needle_test(64000, "X7Q-95-Bravo", reps=2)
        out["needle_128k"] = needle_test(128000, "Zulu-31-Kilo", reps=1)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
