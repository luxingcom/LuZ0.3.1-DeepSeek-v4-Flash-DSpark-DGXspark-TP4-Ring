import requests, time, subprocess, threading

API = "http://YOUR_MASTER_IP:8000/v1/chat/completions"
H = {"Content-Type": "application/json"}
MODEL = "deepseek-v4-flash-0731"
p = "The quick brown fox jumps over the lazy dog. " * 60

NODES = ["NODE0", "NODE1", "NODE2", "NODE3"]

def sample_gpu():
    for name in NODES:
        try:
            if name == "NODE0":
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
            else:
                out = subprocess.run(["ssh", "YOUR_USER@" + name,
                                      "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"],
                                     capture_output=True, text=True, timeout=8).stdout.strip()
            print(f"  {name}: {out}%", flush=True)
        except Exception as e:
            print(f"  {name}: ERR", flush=True)

def do_req():
    t0 = time.time()
    r = requests.post(API, headers=H, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 512, "temperature": 0.0, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False}
    }, timeout=600)
    d = r.json()
    el = time.time() - t0
    print(f"  [请求] {d['usage']['completion_tokens']}t {el:.1f}s = {d['usage']['completion_tokens']/el:.1f} t/s", flush=True)

t = threading.Thread(target=do_req); t.start()
time.sleep(1.5)
for i in range(3):
    time.sleep(2)
    print(f"--- GPU sample {i+1} (请求进行中) ---", flush=True)
    sample_gpu()
t.join()
