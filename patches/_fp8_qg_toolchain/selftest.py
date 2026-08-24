#!/usr/bin/env python3
"""selftest.py — 零 GPU 逻辑自检（无需 VLLM_API_KEY / GPU）
================================================================================
用合成 run 数据驱动各分析脚本，验证:
  1) reference_set_builder validate/build/stats
  2) quality_gate_noise_floor analyze（噪声底 + 门限 + 重标定逻辑）
  3) kl_gate gate（PASS 场景与 FAIL 场景）
  4) greedy_baseline compare（exact 与 envelope 场景）
  5) temp_top_p_gate analyze（重叠率/漂移/distinct 判据）

用法:
  python3 selftest.py          # 全量自检，0=全部通过
  python3 selftest.py --keep   # 保留临时文件（默认清理）
================================================================================
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TMP = None
FAILS = []


def run(cmd, expect=0):
    r = subprocess.run([PY] + cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != expect:
        FAILS.append(f"{' '.join(cmd)}: rc={r.returncode} (expect {expect})\n"
                     f"  stdout: {r.stdout[-800:]}\n  stderr: {r.stderr[-800:]}")
        print(f"  [FAIL] {' '.join(cmd)}")
    else:
        print(f"  [ok] {' '.join(cmd)}")
    return r


SELFTEST_TOKENS = ["tok%02d" % m for m in range(10)]
SELFTEST_PROBS = [0.50, 0.24, 0.12, 0.06, 0.035, 0.02, 0.012, 0.008, 0.004, 0.001]


def make_top_logprobs(tokens, probs):
    """构造一行 top_logprobs（tokens 的 logprob，按 probs 概率）。"""
    return [{"token": t, "logprob": math.log(p)} for t, p in zip(tokens, probs)]


def make_run(path, quant, n_samples=6, seed=0, perturb=0.0, topk=10):
    """合成 run 文件。perturb=0 时分布确定（KL≈0）；>0 时随机交换 top-2 标签
    模拟候选分布漂移（KL 显著 >0）。"""
    rng = random.Random(seed)
    samples = []
    for i in range(n_samples):
        toks = [f"t{i}_{j}" for j in range(5)]
        lps, tl = [], []
        for j in range(5):
            row_tokens = list(SELFTEST_TOKENS[:topk])
            row_probs = list(SELFTEST_PROBS[:topk])
            if perturb > 0 and rng.random() < perturb:
                row_tokens[0], row_tokens[1] = row_tokens[1], row_tokens[0]
            tl.append(make_top_logprobs(row_tokens, row_probs))
            lps.append(math.log(row_probs[0]))  # top-1 logprob
        samples.append({
            "id": f"s{i}", "tier": "B", "category": "code",
            "prompt": f"prompt {i}",
            "outputs": [{"text": " ".join(toks), "tokens": toks,
                         "logprobs": lps, "top_logprobs": tl}],
        })
    rc_meta = {"quant": quant, "model": "test", "config": {"temperature": 0.0,
               "top_p": 1.0, "top_logprobs": topk, "max_tokens": 64},
               "collected_at": "2026-08-24T00:00:00Z", "notes": "selftest"}
    from run_common import save_run
    save_run(path, rc_meta, samples)
    return path


def main():
    global TMP
    keep = "--keep" in sys.argv
    TMP = tempfile.mkdtemp(prefix="fp8qg_selftest_")
    print(f"[selftest] tmp={TMP}")
    try:
        # 1) 参考集
        run(["reference_set_builder.py", "validate"])
        run(["reference_set_builder.py", "stats"])
        run(["reference_set_builder.py", "build"])
        # 2) 噪声底
        r1 = os.path.join(TMP, "bf16_run1.json")
        r2 = os.path.join(TMP, "bf16_run2.json")
        make_run(r1, "bf16", seed=1, perturb=0.0)
        make_run(r2, "bf16", seed=2, perturb=0.0)
        nf = os.path.join(TMP, "noise_floor.json")
        run(["quality_gate_noise_floor.py", "analyze", r1, r2, "--out", nf])
        # 3) KL 门 PASS（候选与参考同分布）
        cand = os.path.join(TMP, "fp8_good.json")
        make_run(cand, "fp8", seed=3, perturb=0.0)
        run(["kl_gate.py", "gate", "--baseline", r1, "--candidate", cand,
             "--noise-floor", nf, "--out", os.path.join(TMP, "kl_ok.json")], expect=0)
        # 3b) KL 门 FAIL（候选明显漂移）
        cand_bad = os.path.join(TMP, "fp8_bad.json")
        make_run(cand_bad, "fp8", seed=4, perturb=0.9)
        run(["kl_gate.py", "gate", "--baseline", r1, "--candidate", cand_bad,
             "--noise-floor", nf, "--out", os.path.join(TMP, "kl_bad.json")], expect=1)
        # 4) greedy compare（exact 与 envelope 场景）
        gold = os.path.join(TMP, "golden.json")
        gold_doc = {"schema": "fp8-qg-golden/1", "captured_at": "t", "model": "test",
                    "quant": "bf16", "prompts": {
                        "fox_repeat": {"text": "X", "logprobs": [-1.0, -2.0]},
                        "count": {"text": "Y", "logprobs": [-1.0, -2.0]},
                        "code_fib": {"text": "Z", "logprobs": [-1.0, -2.0]},
                        "list": {"text": "W", "logprobs": [-1.0, -2.0]}}}
        json.dump(gold_doc, open(gold, "w"))
        run(["greedy_baseline.py", "compare", "--candidate", gold,
             "--ref", gold], expect=0)
        # 4b) envelope 兜底场景：文本不同但 logprob 漂移 ≤1%
        env_doc = json.loads(json.dumps(gold_doc))
        for k in env_doc["prompts"]:
            env_doc["prompts"][k]["text"] = "DIFFERENT"
            env_doc["prompts"][k]["logprobs"] = [-1.01, -2.01]  # 漂移极小
        envf = os.path.join(TMP, "envelope.json")
        json.dump(env_doc, open(envf, "w"))
        run(["greedy_baseline.py", "compare", "--candidate", envf,
             "--ref", gold], expect=0)
        # 5) temp/top-p analyze
        tb = os.path.join(TMP, "tt_b.json")
        tc = os.path.join(TMP, "tt_c.json")
        make_run(tb, "bf16", seed=5, perturb=0.0)
        make_run(tc, "fp8", seed=6, perturb=0.0)
        run(["temp_top_p_gate.py", "analyze", "--baseline", tb,
             "--candidate", tc, "--out", os.path.join(TMP, "tt.json")], expect=0)
    finally:
        if not keep:
            shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f"\n[selftest] {len(FAILS)} FAILED:")
        for f in FAILS:
            print(f)
        return 1
    print("\n[selftest] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
