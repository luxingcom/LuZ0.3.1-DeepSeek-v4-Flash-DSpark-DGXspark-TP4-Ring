#!/usr/bin/env python3
"""inspect ckpt structure for mini-model extraction: classify tensor names,
map layers->shards, list global/mtp/hc tensors."""
import json, struct, os, re, sys
from collections import Counter, defaultdict

def scan(model):
    inv = {}
    for fn in sorted(os.listdir(model)):
        if not fn.endswith(".safetensors"): continue
        p = os.path.join(model, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__": continue
            inv[name] = (fn, meta["dtype"], meta["shape"], meta["data_offsets"])
    return inv

for model in sys.argv[1:]:
    print(f"\n########## {model} ##########")
    inv = scan(model)
    print(f"total tensors: {len(inv)}")
    # classify
    cats = Counter()
    layers = defaultdict(list)
    for name in inv:
        m = re.match(r"layers\.(\d+)\.", name)
        if m:
            layers[int(m.group(1))].append(name)
            cats["layers.N"] += 1
        elif name.startswith("mtp."):
            cats["mtp"] += 1
        elif name.startswith("hc_"):
            cats["hc_*"] += 1
        elif name.startswith("compressor"):
            cats["compressor"] += 1
        else:
            cats["global"] += 1
    print("categories:", dict(cats))
    nl = sorted(layers)
    print(f"layers present: {nl[0]}..{nl[-1]} count={len(nl)}")
    if nl:
        l0 = layers[nl[0]]
        # per-layer tensor name patterns (strip indices)
        pats = Counter(re.sub(r"\.\d+\.", ".N.", n) for n in l0)
        print(f"layer {nl[0]} tensor patterns ({len(l0)} tensors):")
        for pat, c in sorted(pats.items()):
            print(f"   {pat} x{c}")
    # globals
    print("global tensors:")
    for name in sorted(inv):
        if not re.match(r"layers\.\d+\.", name) and not name.startswith(("mtp.", "hc_", "compressor")):
            print(f"   {name} {inv[name][1]} {inv[name][2]}")
    print("hc_/compressor tensors:")
    for name in sorted(inv):
        if name.startswith(("hc_", "compressor")):
            print(f"   {name} {inv[name][1]} {inv[name][2]}")
    print("mtp tensors (first 12):")
    for name in sorted(inv):
        if name.startswith("mtp."):
            print(f"   {name} {inv[name][1]} {inv[name][2]}")
            if sum(1 for _ in [0]) and len([n for n in inv if n.startswith('mtp.')]) > 0: pass
    # shards containing layers 0..4
    print("layer->shard map (layers 0..4):")
    for l in range(5):
        shards = sorted({inv[n][0] for n in layers.get(l, [])})
        print(f"   layer {l}: {shards}")
    # mtp count
    mtps = sorted({n.split('.')[1] for n in inv if n.startswith('mtp.')})
    print("mtp indices:", mtps)
