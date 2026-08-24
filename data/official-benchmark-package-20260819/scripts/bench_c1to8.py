#!/usr/bin/env python3
"""C1-C8 并发测试: decode-only 口径, 每场景 3 轮取中位数"""
import requests, time, concurrent.futures, json, statistics, sys

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
MODEL = "deepseek-v4-flash-0731"
p = "The quick brown fox jumps over the lazy dog. " * 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/seqs_result.json"

def decode_only_req(i):
    t0 = time.time(); t_first = None; t_last = None; usage = None
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 512, "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}
    }, stream=True, timeout=600)
    for line in r.iter_lines():
        if not line: continue
        line = line.decode()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try: ev = json.loads(data)
        except Exception: continue
        now = time.time()
        if t_first is None and ev.get("choices"):
            ch = ev["choices"][0]
            if ch.get("delta", {}).get("content"): t_first = now
        if ev.get("usage"):
            t_last = now; usage = ev["usage"]
    ct = usage.get("completion_tokens", 0) if usage else 0
    if t_first and t_last and t_last > t_first:
        return ct, (ct - 1) / (t_last - t_first), t_last - t0
    return ct, 0, time.time() - t0

results = {}
for C in range(1, 9):
    per_stream = []
    for rnd in range(3):
        with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
            futs = [pool.submit(decode_only_req, i) for i in range(C)]
            for f in concurrent.futures.as_completed(futs):
                ct, d_tps, el = f.result()
                if d_tps > 0:
                    per_stream.append(d_tps)
        time.sleep(0.3)
    med = statistics.median(per_stream) if per_stream else 0
    results[C] = {"per_stream": round(med, 1), "agg": round(med * C, 1)}
    print(f"C{C}: 每流 {med:.1f} | 聚合 {med*C:.1f}", flush=True)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"结果已存 {OUT}")
