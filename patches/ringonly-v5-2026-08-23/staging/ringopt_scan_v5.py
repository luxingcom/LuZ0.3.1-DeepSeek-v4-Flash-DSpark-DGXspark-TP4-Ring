#!/usr/bin/env python3
"""ringopt_scan_v5.py — W1 window microbenchmark for the v5 ring-forced library.

Differences vs ringopt_scan.py:
  * extended sizes up to 128MB (t8192=67MB, t16384=128MB) for the channel A/B matrix
  * RINGOPT_SIZES env selects a subset of labels (comma separated, e.g. "t1024,t4096,t16384")
  * CHECK=1 enables a correctness gate: each rank deterministically regenerates all 4
    per-rank tensors on CPU (seeds 42+rank), computes the exact expected sum locally and
    compares against the NCCL allreduce result (bf16 tolerance). W1 judgement requires
    PASS on every size before any e2e work.
  * result JSON written to /work/v5/results/<TAG>_result.json (no overwrite between arms)

Run via ringopt_node_v5.sh (torchrun 4x1, LD_PRELOAD = v5 lib in /v5lib).
"""
import os, statistics, torch, torch.distributed as dist

ALL_SIZES = [
    # label, tokens(×4096 hidden ×bf16): 8=64K 24=192K 48=384K 96=768K(decode C6) 192=1.5M
    # 512=4M 1024=8.4M(th1024) 2048=16.8M 4096=33.5M(th4096) 8192=67M 16384=128M
    ("t8", 8), ("t24", 24), ("t48", 48), ("t96", 96), ("t192", 192),
    ("t512", 512), ("t1024", 1024), ("t2048", 2048), ("t4096", 4096),
    ("t8192", 8192), ("t16384", 16384),
]

def bench(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)

def main():
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank(); ws = dist.get_world_size()
    dev = torch.device("cuda:%d" % torch.cuda.current_device())
    torch.cuda.set_device(dev)

    sel = os.environ.get("RINGOPT_SIZES", "")
    sizes = ALL_SIZES if not sel else [s for s in ALL_SIZES if s[0] in set(sel.split(","))]
    do_check = os.environ.get("CHECK", "0") == "1"
    tag = os.environ.get("RINGOPT_TAG", "run")

    results = {}
    all_ok = True
    for label, tokens in sizes:
        torch.manual_seed(42 + rank)
        x = torch.randn(tokens, 4096, device=dev, dtype=torch.bfloat16)
        def bf16_c10d():
            t = x.clone(); dist.all_reduce(t, group=dist.group.WORLD); return t
        t_ms = bench(bf16_c10d)
        nbytes = x.numel() * 2
        busbw = nbytes * 2 * (ws - 1) / ws / t_ms / 1e9 * 1e3
        entry = {"bytes": nbytes, "ms": round(t_ms, 4), "us": round(t_ms * 1000, 1),
                 "busbw_GBps": round(busbw, 2)}
        if do_check:
            # deterministic expected sum: regenerate every rank's tensor locally (CPU, same seeds)
            exp = torch.zeros(tokens, 4096, dtype=torch.float32)
            for r in range(ws):
                exp += torch.randn(tokens, 4096, generator=torch.Generator().manual_seed(42 + r))
            got = bf16_c10d()
            exp_bf = exp.to(torch.bfloat16)
            diff = (got.float() - exp_bf.float()).abs()
            rel = (diff / exp_bf.float().abs().clamp_min(1e-3)).max().item()
            ok = bool(torch.isfinite(got).all().item()) and rel < 0.05
            entry["check_max_rel_err"] = round(rel, 6); entry["check"] = "PASS" if ok else "FAIL"
            all_ok = all_ok and ok
        results[label] = entry
        if rank == 0:
            chk = ("  check=%s(maxrel=%.2e)" % (entry["check"], entry["check_max_rel_err"])) if do_check else ""
            print(f"[ringopt-v5:{tag}] {label} ({nbytes/1e6:.2f}MB): {t_ms:.3f} ms  "
                  f"busbw={busbw:.2f} GB/s{chk}", flush=True)
        del x; torch.cuda.empty_cache()

    if rank == 0:
        import json
        os.makedirs("/work/v5/results", exist_ok=True)
        with open(f"/work/v5/results/{tag}_result.json", "w") as f:
            json.dump({"tag": tag, "all_check_pass": all_ok if do_check else None, "sizes": results},
                      f, indent=1)
        if do_check:
            print(f"[ringopt-v5:{tag}] CORRECTNESS GATE: {'PASS' if all_ok else 'FAIL'}", flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
