import requests, time, json, statistics

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
p = "The quick brown fox jumps over the lazy dog. " * 60

def decode_only_req():
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

print("=== C1 单流 10 轮 (seqs=16, 当前状态) ===")
results = []
for i in range(10):
    ct, d, el = decode_only_req()
    results.append(d)
    print(f"  轮{i+1}: decode={d:.1f} t/s ({ct}t)", flush=True)
    time.sleep(0.3)
print(f"  中位数: {statistics.median(results):.1f} | 均值: {statistics.mean(results):.1f} | 范围: {min(results):.1f}-{max(results):.1f}")
