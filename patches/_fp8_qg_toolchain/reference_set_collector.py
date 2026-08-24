#!/usr/bin/env python3
"""reference_set_collector.py — 参考集在线采集（GPU 窗口执行）
================================================================================
用途: 在目标部署（LuZ0.3.1 或 FP8 克隆镜像）上按统一配置采集参考集 logprobs，
      产出 fp8-qg-run/1 文件供离线分析（noise_floor / kl_gate / greedy / temp-top-p）。

用法:
  VLLM_API_KEY=... python3 reference_set_collector.py collect \
      --ref reference_set.json \
      --quant bf16 --tag run1 \
      --config greedy \
      --out runs/run_bf16_greedy_run1.json
  # --config greedy => temp=0, top_p=1, top_logprobs=1（G1/包络；与 quality_gate 同口径）
  # --config dist    => temp=0, top_p=1, top_logprobs=10（KL/困惑度噪声底）
  # --config temp    => 配合 --temperature 0.6/0.9 与 --top-p 0.9/1.0、--reps N
  # --tiers A,B / --subset id,id,...  可限定子集（省窗口时间）

GPU 窗口协议（设计文档 §4.2）:
  - BF16 噪声底: 同配置背靠背两遍（run1/run2），draft 配置不变。
  - 采集前确认 CUMEM_HOST_ENABLE=0 环境生效（避免环境噪声污染测量）。
  - Tier C（64K）逐样本 max_tokens 较小，但 prompt 很长，耗时显著；建议单独
    一个 collect 调用（--tiers C），勿与 A/B 混跑导致超时。
================================================================================
"""
import argparse
import json
import os
import sys
import time

import run_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(HERE, "reference_set.json")

CONFIGS = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_logprobs": 1},
    "dist": {"temperature": 0.0, "top_p": 1.0, "top_logprobs": 10},
}


def resolve_prompt(s):
    """普通 prompt 直出；long_ctx 从 _generated/<id>.txt 读构建好的全文。"""
    if "long_ctx" in s:
        p = os.path.join(HERE, "_generated", s["id"] + ".txt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} 不存在：请先运行 reference_set_builder.py build")
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return s["prompt"]


def cmd_collect(args):
    if not rc.need_key():
        return 2
    doc = json.load(open(args.ref, encoding="utf-8"))
    if args.config in CONFIGS:
        cfg = dict(CONFIGS[args.config])
    elif args.config == "temp":
        if args.temperature is None or args.top_p is None:
            print("[error] --config temp 需要 --temperature 与 --top-p",
                  file=sys.stderr)
            return 2
        cfg = {"temperature": args.temperature, "top_p": args.top_p,
               "top_logprobs": 10}
    else:
        print(f"[error] 未知 --config {args.config}（greedy|dist|temp）",
              file=sys.stderr)
        return 2
    max_tokens = args.max_tokens
    reps = args.reps if args.config == "temp" else 1

    samples = []
    for s in doc["samples"]:
        if args.tiers and s["tier"] not in args.tiers.split(","):
            continue
        if args.subset and s["id"] not in args.subset.split(","):
            continue
        prompt = resolve_prompt(s)
        mt = s.get("max_tokens") or max_tokens
        print(f"  collecting {s['id']} (tier={s['tier']} cat={s['category']} "
              f"max_tokens={mt} reps={reps})", flush=True)
        outputs = []
        for _ in range(reps):
            out = rc.gen(prompt, temperature=cfg["temperature"],
                         top_p=cfg["top_p"], max_tokens=mt,
                         top_logprobs=cfg["top_logprobs"])
            outputs.append(out)
        samples.append({"id": s["id"], "tier": s["tier"],
                        "category": s["category"], "prompt": prompt,
                        "outputs": outputs})
    meta = {
        "quant": args.quant,
        "model": rc.MODEL,
        "config": {"temperature": cfg["temperature"], "top_p": cfg["top_p"],
                   "top_logprobs": cfg["top_logprobs"],
                   "max_tokens": max_tokens},
        "collected_at": time.strftime("%Y%m%d-%H%M%SZ", time.gmtime()),
        "notes": args.tag or "",
    }
    path = args.out or os.path.join(HERE, "runs",
                                    f"run_{args.quant}_{args.tag or args.config}.json")
    rc.save_run(path, meta, samples)
    print(f"[collect] {len(samples)} samples -> {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="FP8 质量门参考集在线采集")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--quant", required=True, choices=["bf16", "fp8"])
    p.add_argument("--tag", default="")
    p.add_argument("--config", required=True, choices=["greedy", "dist", "temp"])
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=rc.DEFAULT_MAX_TOKENS)
    p.add_argument("--reps", type=int, default=5, help="temp 配置采样次数")
    p.add_argument("--tiers", default="A,B", help="如 'A,B' 或 'C'")
    p.add_argument("--subset", default="")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
