#!/usr/bin/env python3
"""run_mini.py — mini 模型单卡 vLLM 离线推理, 抓 prompt logprobs + 生成 logprobs。
用法: python3 run_mini.py --model /work/mini0731 --out /work/lp_0731.json [--moe-backend flashinfer_b12x]"""
import argparse
import json
import sys

from vllm import LLM, SamplingParams

PROMPTS = [
    "The history of computing hardware spans from mechanical calculators to modern "
    "quantum processors. Trace the major milestones from 1940 to 2020, focusing on "
    "the transition from vacuum tubes to transistors to integrated circuits, and "
    "explain how Moore's law drove the industry forward. Include specific years, "
    "companies, and processor names where relevant, and discuss the social impact "
    "of each generational shift.",
    "计算圆周率的历史方法有很多。请解释蒙特卡洛方法估算 π 的原理, 给出具体的 "
    "伪代码实现, 并分析其收敛速度与随机数质量的关系。然后对比刘徽割圆术、 "
    "莱布尼茨级数和 Chudnovsky 算法的复杂度差异。",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n\n"
    "Review this code for bugs, edge cases, and performance improvements. Consider "
    "what happens with duplicate elements, already-sorted input, and very large arrays.",
    "In 2019, a company reported revenue of $4,872,300 and expenses of $3,191,450. "
    "In 2020, revenue grew by 23.7% while expenses grew by 11.2%. In 2021, revenue "
    "declined by 8.3% but expenses declined by 15.9%. Calculate the profit margins "
    "for each year, the compound annual growth rate of revenue, and explain which "
    "year had the best operational efficiency and why.",
    "全球气候系统的反馈机制是理解变暖预测的关键。请分别说明水蒸气反馈、冰盖- "
    "反照率反馈、云反馈和碳循环反馈的正负属性及其不确定性来源, 并解释为什么 "
    "平衡气候敏感度(ECS)的概率分布至今仍有较宽的范围。",
    "量子纠缠与经典关联的本质区别是什么? 请从 Bell 不等式的数学表述出发, 解释 "
    "CHSH 游戏中量子策略如何超越经典上界 2, 并说明 2√2 上界的来源及其与 Tsirelson "
    "界的关系。这对量子密钥分发和量子隐形传态分别意味着什么?",
    "Write a short story about a lighthouse keeper who discovers that the light has "
    "been guiding something other than ships for the past thirty years. The story "
    "should have a melancholic yet hopeful tone.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-backend", default="flashinfer_b12x")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--no-eager", action="store_true")
    args = ap.parse_args()

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=not args.no_eager,
        dtype="bfloat16",
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        max_num_seqs=12,
        max_num_batched_tokens=4096,
    )
    # moe_backend 传递方式探测
    from vllm.engine.arg_utils import EngineArgs
    import dataclasses
    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    if "moe_backend" in fields:
        kwargs["moe_backend"] = args.moe_backend
        print(f"[run] EngineArgs.moe_backend = {args.moe_backend}")
    else:
        import vllm.envs as envs
        cand = [a for a in dir(envs) if "MOE" in a]
        print(f"[run] EngineArgs lacks moe_backend; env candidates: {cand}")
        raise SystemExit("cannot pass moe backend")

    print(f"[run] loading {args.model} ...", flush=True)
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        logprobs=20, prompt_logprobs=1)
    outs = llm.generate(PROMPTS, sp)

    result = []
    for i, o in enumerate(outs):
        entry = {"prompt_id": i, "prompt": o.prompt[:120]}
        # prompt logprobs (actual token logprob per prompt position)
        plp = []
        if o.prompt_logprobs is not None:
            for pos, d in enumerate(o.prompt_logprobs):
                if d is None:
                    continue
                plp.append({"pos": pos,
                            "lp": {str(tid): lpp.logprob for tid, lpp in d.items()}})
        entry["prompt_logprobs"] = plp
        # generation logprobs
        gen = []
        g = o.outputs[0]
        for pos, d in enumerate(g.logprobs or []):
            topk = {str(tid): lpp.logprob for tid, lpp in
                    sorted(d.items(), key=lambda kv: -kv[1].logprob)[:5]}
            gen.append({"pos": pos, "tok": g.token_ids[pos], "top5": topk})
        entry["gen_text"] = g.text
        entry["gen_logprobs"] = gen
        entry["prompt_token_ids"] = list(o.prompt_token_ids)
        result.append(entry)
        print(f"[run] prompt {i}: {len(plp)} prompt-lp, {len(gen)} gen-lp, "
              f"text={g.text[:60]!r}")
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[run] wrote {args.out}")


if __name__ == "__main__":
    main()
