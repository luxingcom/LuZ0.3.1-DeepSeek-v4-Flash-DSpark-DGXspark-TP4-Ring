#!/usr/bin/env python3
"""
TP4 性能全景测试: prefill 速度 + 纯 decode 速度 × 上下文长度 × 并发
- prefill: 唯一 nonce 前缀, 输出 1 token, 测 TTFT → prefill tok/s (3 轮中位数)
- decode: 固定上下文长度, 并发请求, decode-only 口径 (3 轮中位数)
"""
import requests, time, json, statistics, uuid, concurrent.futures

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
MODEL = "deepseek-v4-flash-0731"

FOX = "The quick brown fox jumps over the lazy dog. "

def make_prompt(length, nonce=""):
    n = length // 5
    body = FOX * n
    return f"nonce-{nonce} " + body if nonce else body

def measure_prefill_once(length):
    """唯一 nonce, 流式, 输出 1 token, 返回 TTFT 和 prompt_tokens"""
    nonce = uuid.uuid4().hex[:8]
    prompt = make_prompt(length, nonce)
    t0 = time.time()
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}
    }, stream=True, timeout=900)
    ttft = None
    usage = None
    for line in r.iter_lines():
        if not line: continue
        line = line.decode()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try: ev = json.loads(data)
        except Exception: continue
        now = time.time()
        if ttft is None and ev.get("choices"):
            ch = ev["choices"][0]
            if ch.get("delta", {}).get("content"):
                ttft = now - t0
        if ev.get("usage"):
            usage = ev["usage"]
    if usage is None or ttft is None:
        return None
    return usage.get("prompt_tokens", 0), ttft

def measure_decode_once(prompt, gen=512):
    """decode-only 口径: (ct-1)/(t_last-t_first)"""
    t0 = time.time(); t_first = None; t_last = None; usage = None
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}
    }, stream=True, timeout=900)
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
    if usage is None or t_first is None or t_last <= t_first:
        return None
    ct = usage.get("completion_tokens", 0)
    return ct, t_last - t0, (ct - 1) / (t_last - t_first)

# ═══════════════════════════════════════════════════════
print("=" * 70)
print("  TP4 性能全景: prefill + decode × 上下文 × 并发")
print("  配置: seqs=12 batched=8240 util=0.835 MTP=5")
print("=" * 70)

# ── 1. Prefill 速度（唯一 nonce，3 轮中位数）──
print("\n[1/2] Prefill 速度 (唯一 nonce, 输出 1 token, 3 轮中位数)")
print("-" * 60)
print(f"  {'上下文':>8} {'TTFT(中位)':>12} {'Prefill tok/s':>16}")
PREFILL_LENS = [4096, 16384, 32768, 65536]
prefill_results = {}
for length in PREFILL_LENS:
    ttfts = []
    for rnd in range(3):
        r = measure_prefill_once(length)
        if r:
            pt, ttft = r
            ttfts.append(ttft)
            print(f"     {length:>6}t 轮{rnd+1}: TTFT={ttft:.1f}s prefill={pt/ttft:.0f} tok/s", flush=True)
        time.sleep(0.3)
    if ttfts:
        med = statistics.median(ttfts)
        # prompt tokens 取中位 TTFT 对应轮次的
        prefill_results[length] = (med, length / med)
        print(f"  => {length:>6}t | {med:>10.2f}s | {length/med:>14.0f} tok/s")

# ── 2. Decode 速度（上下文 × 并发，3 轮中位数）──
print("\n[2/2] 纯 Decode 速度 (decode-only 口径, 输出 512, 3 轮中位数)")
print("-" * 60)
CTX_LENS = [256, 4096, 16384, 65536]
CONCURRENCY = [1, 4, 8, 12]

def bench_decode_conc(ctx_len, conc):
    prompt = make_prompt(ctx_len)  # 同 prompt 允许 prefix cache（只测 decode）
    aggs = []
    for rnd in range(3):
        results = []
        def worker(i):
            r = measure_decode_once(prompt, 512)
            return r
        t_wall0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(worker, i) for i in range(conc)]
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r:
                    results.append(r)
        wall = time.time() - t_wall0
        if results:
            total_ct = sum(r[0] for r in results)
            agg = total_ct / wall
            # 每流 decode 中位数
            streams = [r[2] for r in results]
            aggs.append(agg)
        time.sleep(0.3)
    if aggs:
        return statistics.median(aggs)
    return None

decode_results = {}
for ctx in CTX_LENS:
    print(f"\n  --- 上下文 {ctx}t ---")
    for conc in CONCURRENCY:
        agg = bench_decode_conc(ctx, conc)
        if agg:
            per = agg / conc
            decode_results[(ctx, conc)] = agg
            print(f"    C{conc:>2}: {agg:>7.1f} agg | {per:>6.1f}/stream", flush=True)

# ── 汇总 ──
print("\n" + "=" * 70)
print("  汇总")
print("=" * 70)
print("\n── Prefill ──")
for length, (ttft, tps) in prefill_results.items():
    print(f"  {length:>6}t → TTFT {ttft:.1f}s → {tps:.0f} tok/s")
print("\n── Decode (agg tok/s) ──")
print(f"  {'ctx':>8} " + "".join(f"C{c:>10}" for c in CONCURRENCY))
for ctx in CTX_LENS:
    row = f"  {ctx:>7}t "
    for c in CONCURRENCY:
        v = decode_results.get((ctx, c))
        row += f"{v:>10.1f}" if v else f"{'-':>10}"
    print(row)
print("\n完成")
