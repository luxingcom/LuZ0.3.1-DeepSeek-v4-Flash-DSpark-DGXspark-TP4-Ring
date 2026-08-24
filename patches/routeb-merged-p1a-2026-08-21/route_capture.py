#!/usr/bin/env python3
"""route_capture.py v2 — patch B12xExperts.apply 抓路由终态 topk_ids。"""
import json
import sys

sys.path.insert(0, "/work")

import vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe as _b12x
import torch

_orig_apply = _b12x.B12xExperts.apply

def _apply_capture(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                   activation, *a, **kw):
    try:
        with open("/work/routing_capture.jsonl", "a") as f:
            f.write(json.dumps({
                "n_tok": int(topk_ids.shape[0]),
                "topk": topk_ids.detach().to(torch.int32).cpu().tolist(),
            }) + "\n")
    except Exception as e:
        print(f"[capture] err {e}", file=sys.stderr)
    return _orig_apply(self, output, hidden_states, w1, w2, topk_weights,
                       topk_ids, activation, *a, **kw)

_b12x.B12xExperts.apply = _apply_capture
print("[capture] B12xExperts.apply patched")

from vllm import LLM, SamplingParams  # noqa: E402
from run_mini import PROMPTS as BASE_PROMPTS  # noqa: E402


def make_filler(n_sent, seed):
    import random
    rng = random.Random(seed)
    S = ["the survey team", "a quiet customs officer", "the lighthouse keeper",
         "an aging tram conductor", "the night-shift engineer", "a cartographer"]
    V = ["catalogued", "measured", "repaired", "photographed", "archived", "logged"]
    O = ["seventeen brass instruments", "the tidal ledger", "a crate of spare valves",
         "the coastal fog charts", "four crates of citrus", "the signal lamp lenses"]
    return " ".join(
        f"Record {i}: {rng.choice(S)} {rng.choice(V)} {rng.choice(O)} on day {100+i%800}."
        for i in range(n_sent))


def main():
    prompts = list(BASE_PROMPTS)
    for s in range(32):
        prompts.append(make_filler(20 + (s * 7) % 40, seed=100 + s))
    print(f"[capture] {len(prompts)} prompts")
    llm = LLM(model="/work/mini0731", max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True, dtype="bfloat16",
              trust_remote_code=True, kv_cache_dtype="fp8", max_num_seqs=12,
              moe_backend="flashinfer_b12x",
              max_num_batched_tokens=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(prompts, sp)
    print("[capture] done -> /work/routing_capture.jsonl")


if __name__ == "__main__":
    main()
