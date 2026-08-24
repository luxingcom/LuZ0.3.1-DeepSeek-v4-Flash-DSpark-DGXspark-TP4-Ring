import subprocess, os, struct

SO = "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
print("=== vLLM C ext FP4 符号/架构检查 ===")
print("file exists:", os.path.exists(SO), "size:", os.path.getsize(SO)//1024//1024, "MB")

# 1) strings for fp4 symbols
r = subprocess.run(["strings", SO], capture_output=True, text=True)
hits = [l for l in r.stdout.splitlines() if any(k in l.lower() for k in ["nvfp4", "scaled_fp4", "mmaf", "e2m1"])]
print("fp4 symbol-like strings:", len(hits))
for h in hits[:12]:
    print("  ", h[:100])

# 2) cuobjdump list elf/sass arch segments
for cmd in [["cuobjdump", "--list-elf", SO], ["cuobjdump", "--list-text", SO]]:
    print("\n$", " ".join(cmd[:2]))
    rr = subprocess.run(cmd, capture_output=True, text=True)
    print(rr.stdout[:1500])
    if rr.stderr.strip():
        print("err:", rr.stderr.splitlines()[-1][:150])