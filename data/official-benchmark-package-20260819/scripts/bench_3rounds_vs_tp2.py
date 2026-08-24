#!/usr/bin/env python3
"""
TP4 vs TP2 三轮对比测试（对照 7/17 bench_comprehensive.py 方案）
每项测 3 轮，去掉第一轮冷启动，取第 2/3 轮
- 单流 4 项: p256/g64, p256/g256, p512/g64, p512/g256
- 并发 C1-C8 (decode-only 聚合)
- Agent 5 场景
输出对比 TP2 (7/17 最优) 数据
"""
import requests, time, concurrent.futures, statistics, json

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
MODEL = "deepseek-v4-flash-0731"
TIMEOUT = 600

def api_call(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False}
    }, timeout=TIMEOUT)
    d = r.json()
    return d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], time.time() - t0

def decode_only(prompt, gen=512):
    t0 = time.time(); t_first = None; t_last = None; usage = None
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}
    }, stream=True, timeout=TIMEOUT)
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
        return (ct - 1) / (t_last - t_first)
    return 0

# 单流场景（3 轮去 1）
def bench_single(name, prompt, gen, runs=3):
    results = []
    for i in range(runs):
        pt, ct, el = api_call(prompt, gen)
        results.append(ct / el)
        time.sleep(0.3)
    return statistics.mean(results[1:]) if len(results) > 1 else results[0]

# 并发（3 轮去 1，decode-only 每流中位 × C）
def bench_conc(C, prompt, gen=512):
    aggs = []
    for rnd in range(3):
        streams = []
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
            futs = [pool.submit(decode_only, prompt, gen) for i in range(C)]
            for f in concurrent.futures.as_completed(futs):
                d = f.result()
                if d > 0: streams.append(d)
        wall = time.time() - t0
        if streams:
            med = statistics.median(streams)
            aggs.append(med * C)
        time.sleep(0.3)
    return statistics.median(aggs[1:]) if len(aggs) > 1 else (aggs[0] if aggs else 0)

P256 = "The quick brown fox jumps over the lazy dog. " * 30
P512 = "The quick brown fox jumps over the lazy dog. " * 60
P4K  = "The quick brown fox jumps over the lazy dog. " * 480

MATH = ("Solve the following problem step by step. Show all your work.\n\n"
        "A rectangular field has a perimeter of 240 meters. The length is twice the width. "
        "If a farmer wants to plant crops in 75% of the field, how many square meters will be planted? "
        "Also, if each square meter yields 3.5 kg of crop, what is the total yield in metric tons?\n\nAnswer:")
JSON = ("Generate a comprehensive JSON document describing a fictional e-commerce platform "
        "with: company info, product catalog (at least 8 products with specs and pricing), "
        "customer reviews, shipping policies, payment methods, and API documentation. "
        "Make it realistic. Output only valid JSON:")
CODE = ("Write a complete Rust implementation of a concurrent, async HTTP rate limiter using "
        "tokio. Support per-IP rate limiting with sliding window, implement as tower middleware, "
        "production-ready with doc comments. Provide the complete code with no explanations:")
COMM = ("Write a formal business email to a potential client introducing our AI consulting "
        "services, referencing a previous conversation, proposing a 3-phase engagement model "
        "with timelines, and including a call to action. Professional but warm tone. "
        "Write the complete email:")
NARR = ("Write a creative short story about a robot who discovers it can dream. "
        "Open with its first dream experience, explore how it changes its understanding of "
        "consciousness, include dialogue with its human creator, and end with a poignant "
        "reflection. Use vivid sensory descriptions. Write the complete story:")

print("=" * 72)
print("  TP4 vs TP2 三轮对比（去冷启动，取 2/3 轮）")
print("  配置: NVFP4 TP4, seqs=16, batched=8240, util=0.835, MTP=5")
print("=" * 72)

# 1. 单流
print("\n[1/3] 单流（3 轮去 1，均值）")
ss = {}
for name, prompt, gen in [("p256/g64", P256, 64), ("p256/g256", P256, 256),
                          ("p512/g64", P512, 64), ("p512/g256", P512, 256)]:
    ss[name] = bench_single(name, prompt, gen)
    print(f"  {name}: {ss[name]:.1f} t/s")

# 2. 并发（3 轮去 1）
print("\n[2/3] 并发 C1-C8（3 轮去 1，decode-only 聚合中位）")
conc = {}
for c in range(1, 9):
    conc[c] = bench_conc(c, P256)
    print(f"  C{c}: {conc[c]:.1f} agg")

# 3. Agent
print("\n[3/3] Agent 场景（3 轮去 1，均值）")
agent = {}
for name, prompt in [("Math", MATH), ("JSON", JSON), ("Code", CODE),
                     ("Communication", COMM), ("Narrative", NARR)]:
    agent[name] = bench_single(name, prompt, 512)
    print(f"  {name}: {agent[name]:.1f} t/s")

# 汇总对比 TP2 (7/17)
print("\n" + "=" * 72)
print("  对比 TP2（7/17 最优方案）")
print("=" * 72)
TP2 = {"C1": 62.2, "C2": 126.3, "C4": 145.6, "C6": 139.9,
       "Math": 67.4, "JSON": 62.7, "Code": 64.0, "Communication": 50.3, "Narrative": 38.8}
print(f"  {'指标':<16} {'TP4':>10} {'TP2':>10} {'Δ':>8}")
for k in ["C1", "C2", "C4", "C6", "Math", "JSON", "Code", "Communication", "Narrative"]:
    cur = conc.get(int(k[1:])) if k.startswith("C") and k[1:].isdigit() else agent.get(k)
    prev = TP2.get(k)
    if cur and prev:
        delta = (cur / prev - 1) * 100
        flag = "✅" if delta >= -1 else ("⚠️" if delta >= -5 else "🔴")
        print(f"  {k:<16} {cur:>8.1f}  {prev:>8.1f}  {delta:>+6.1f}% {flag}")
# Agent 平均
tp4_avg = statistics.mean(agent.values())
print(f"  {'Agent平均':<16} {tp4_avg:>8.1f}  {55.8:>8.1f}  {(tp4_avg/55.8-1)*100:>+6.1f}%")
print("=" * 72)
