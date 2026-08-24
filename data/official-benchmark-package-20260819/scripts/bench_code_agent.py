import requests, time, json, statistics

API = "http://YOUR_API_URL/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <BEARER>"}
MODEL = "deepseek-v4-flash-0731"

def decode_only(prompt, gen=2048):
    t0 = time.time(); t_first = None; t_last = None; usage = None
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0,
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
    return ct, t_last - t0, (ct - 1) / (t_last - t_first), t_first - t0

# 代码生成（红黑树，对应帖子的 124.5 tok/s 场景）
CODE = ("Write a complete implementation of a Red-Black Tree in C++ with insert, delete, "
        "and search operations. Include all helper functions, rotations, and color handling. "
        "Provide the complete code with no explanations:")
# Agent 工具调用（短输出，对应帖子的 146.5 tok/s 场景）
AGENT = ("You are an assistant that must call tools to answer. Use get_weather for any weather "
         "question and get_stock_price for any stock question. Return tool calls in JSON format. "
         "What is the weather in Beijing and Shanghai and Guangzhou?")

for name, prompt, gen in [("代码生成(红黑树)", CODE, 2048), ("Agent工具调用", AGENT, 310)]:
    print(f"\n===== {name} decode-only 3 轮 (g{gen}) =====")
    results = []
    for i in range(1, 4):
        time.sleep(0.5)
        r = decode_only(prompt, gen)
        if r:
            ct, el, d, ttft = r
            results.append(d)
            print(f"  轮{i}: {ct}t TTFT={ttft:.1f}s decode={d:.1f} t/s")
        else:
            print(f"  轮{i}: 失败")
    if results:
        print(f"  => decode-only 中位数: {statistics.median(results):.1f} t/s")
