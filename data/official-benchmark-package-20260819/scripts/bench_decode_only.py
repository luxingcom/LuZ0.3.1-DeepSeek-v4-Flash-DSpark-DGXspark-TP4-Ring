import requests, time, json

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Authorization": "Bearer <BEARER>", "Content-Type": "application/json"}
MODEL = "deepseek-v4-flash-0731"

# 上游标准 decode-only: (completion_tokens - 1) / (t_last - t_first)
def decode_only(prompt, gen=512, nonce=""):
    t0 = time.time()
    t_first = None
    t_last = None
    usage = None
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}
    }, stream=True, timeout=600)
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except Exception:
            continue
        now = time.time()
        if t_first is None and ev.get("choices"):
            ch = ev["choices"][0]
            if ch.get("delta", {}).get("content"):
                t_first = now
        if ev.get("usage"):
            t_last = now
            usage = ev["usage"]
    el_total = t_last - t0 if t_last else time.time() - t0
    ct = usage.get("completion_tokens", 0)
    if t_first is None or t_last is None or t_last <= t_first:
        return None
    decode_tps = (ct - 1) / (t_last - t_first)
    return ct, el_total, decode_tps

P256 = "The quick brown fox jumps over the lazy dog. " * 30
# 官方编号列表指令（规律文本，MTP 接受率高）
LIST = ("Return exactly 128 numbered lowercase English words, one per line. "
        "Each word must be lowercase and different from the others. "
        "Start with 1. and end with 128.")

for name, prompt in [("fox p256", P256), ("编号列表", LIST)]:
    print(f"\n===== {name} /g512 decode-only 3 轮 =====")
    results = []
    for i in range(1, 4):
        time.sleep(0.5)
        r = decode_only(prompt, 512, nonce=f"-{i}")
        if r:
            ct, el, d = r
            results.append(d)
            print(f"  轮{i}: {ct}t 端到端{el:.1f}s decode={d:.1f} t/s")
        else:
            print(f"  轮{i}: 测量失败")
    if results:
        import statistics
        print(f"  => decode-only 中位数: {statistics.median(results):.1f} t/s | 均值: {statistics.mean(results):.1f} t/s")
