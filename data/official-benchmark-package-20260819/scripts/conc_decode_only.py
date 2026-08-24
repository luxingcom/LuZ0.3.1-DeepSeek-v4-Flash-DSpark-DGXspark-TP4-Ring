import requests, time, concurrent.futures, json, statistics

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
p = "The quick brown fox jumps over the lazy dog. " * 60

def decode_only_req(i):
    t0 = time.time(); t_first = None; t_last = None; usage = None
    r = requests.post(API, headers=H, json={
        "model": "deepseek-v4-flash-0731",
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

for C in [1, 4, 8, 12]:
    streams = []
    wall0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
        futs = [pool.submit(decode_only_req, i) for i in range(C)]
        for f in concurrent.futures.as_completed(futs):
            ct, d_tps, el = f.result()
            if d_tps > 0:
                streams.append(d_tps)
    wall = time.time() - wall0
    if streams:
        print(f"C{C}: 每流decode中位 {statistics.median(streams):.1f} | 聚合(中位×C) ~{statistics.median(streams)*C:.0f} | 总token/墙钟 {sum([ct for ct,_,_ in []] ) or ''}{'':2s} wall={wall:.1f}s", flush=True)
