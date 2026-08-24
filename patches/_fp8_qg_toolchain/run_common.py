#!/usr/bin/env python3
"""run_common.py — FP8 质量门工具链共享模块
================================================================================
用途: 统一 run 文件 schema、API logprobs 采集、数学工具（KL/PPL/漂移）、对齐。
设计目标: 所有分析脚本（noise_floor / kl_gate / greedy compare / temp-top-p
analyze）零 GPU 可运行；GPU 仅用于 reference_set_collector 的在线采集。

Run 文件 schema（v1）:
{
  "schema": "fp8-qg-run/1",
  "meta": {
    "quant": "bf16" | "fp8",           # 被测量化形态
    "model": "...",                    # API model 名
    "config": {"temperature": 0.0, "top_p": 1.0, "top_logprobs": 10,
               "max_tokens": 256},     # 该文件统一生成配置
    "collected_at": "UTC 时间",
    "notes": "自由备注（如 run1/run2 标签）"
  },
  "samples": [
    {
      "id": "code_fib", "tier": "A|B|C", "category": "code",
      "prompt": "...",
      "outputs": [                       # 通常 1 个（greedy/KL）；temp/top-p 可为 N 个（reps）
        {"text": "...", "tokens": [...], "logprobs": [...],
         "top_logprobs": [[{"token": "...", "logprob": -0.1}, ...], ...]}
      ]
    }, ...
  ]
}

口径（与 quality_gate.py 保持一致）:
  - API: http://127.0.0.1:8001/v1/chat/completions（可用 QGT_API 覆盖）
  - 认证: VLLM_API_KEY
  - 模型: deepseek-v4-flash-0731（可用 QGT_MODEL 覆盖）
================================================================================
"""
import json
import math
import os
import sys
import time

try:
    import requests
except ImportError:  # 分析路径可无 requests；采集路径需要
    requests = None

API = os.environ.get("QGT_API", "http://127.0.0.1:8001/v1/chat/completions")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("QGT_MODEL", "deepseek-v4-flash-0731")
HDR = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

DEFAULT_MAX_TOKENS = 384
DEFAULT_TOP_LOGPROBS = 10
DEFAULT_TIMEOUT = 600


# --------------------------------------------------------------------------
# API 采集（GPU 窗口用）
# --------------------------------------------------------------------------
def gen(prompt, temperature=0.0, top_p=1.0, max_tokens=DEFAULT_MAX_TOKENS,
        top_logprobs=DEFAULT_TOP_LOGPROBS, thinking=False, timeout=DEFAULT_TIMEOUT):
    """单次 chat/completions 调用，返回含 logprobs 的输出 dict。"""
    if requests is None:
        raise RuntimeError("采集需要 requests 库（pip install requests）")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"thinking": thinking},
    }
    r = requests.post(API, headers=HDR, json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    txt = choice["message"].get("content", "")
    tokens, logprobs, top_logprobs_out = [], [], None
    if choice.get("logprobs") and choice["logprobs"].get("content"):
        content = choice["logprobs"]["content"]
        tokens = [c.get("token", "") for c in content]
        logprobs = [c.get("logprob") for c in content]
        top_logprobs_out = []
        for c in content:
            tl = c.get("top_logprobs") or []
            top_logprobs_out.append(
                [{"token": str(x.get("token", "")), "logprob": float(x.get("logprob", 0.0))}
                 for x in tl])
    return {"text": txt, "tokens": tokens, "logprobs": logprobs,
            "top_logprobs": top_logprobs_out}


# --------------------------------------------------------------------------
# Run 文件 I/O
# --------------------------------------------------------------------------
def save_run(path, meta, samples):
    doc = {"schema": "fp8-qg-run/1", "meta": meta, "samples": samples}
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return path


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "fp8-qg-run/1":
        raise ValueError(f"{path}: 不是 fp8-qg-run/1 schema 文件")
    return doc


def align_samples(run_a, run_b):
    """按 sample.id 对齐两个 run，返回 (common_ids, by_id_a, by_id_b)。"""
    by_a = {s["id"]: s for s in run_a["samples"]}
    by_b = {s["id"]: s for s in run_b["samples"]}
    common = [i for i in by_a if i in by_b]
    return common, by_a, by_b


def write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


# --------------------------------------------------------------------------
# 数学工具
# --------------------------------------------------------------------------
def _logsumexp(lps):
    m = max(lps)
    return m + math.log(sum(math.exp(x - m) for x in lps))


def probs_from_top_logprobs(row):
    """把单 token 位置的 top_logprobs 行归一化为 {token: prob}（top-k 内部归一）。"""
    lps = [float(x["logprob"]) for x in row]
    lse = _logsumexp(lps)
    return {str(x["token"]): math.exp(float(x["logprob"]) - lse) for x in row}


def kl_common_support(row_a, row_b):
    """受限支持 KL：仅对两 run 公共 top-k token 计算（各自重归一）。
    返回 nats/token；无公共 token 时返回 None。
    注: 这是 top-k 受限 KL 估计量，低估全词表 KL；k 越大越接近真值。
        参考集 Tier B 采集用 top_logprobs=10，够统计分布门。
    """
    if not row_a or not row_b:
        return None
    pa = probs_from_top_logprobs(row_a)
    pb = probs_from_top_logprobs(row_b)
    common = set(pa) & set(pb)
    if not common:
        return None
    za = sum(pa[t] for t in common)
    zb = sum(pb[t] for t in common)
    kl = 0.0
    for t in common:
        qa = pa[t] / za
        qb = pb[t] / zb
        if qa > 0 and qb > 0:
            kl += qa * math.log(qa / qb)
    return kl


def ppl_from_logprobs(logprobs):
    """困惑度 = exp(-mean(logprob))；基于 top-1 采样 token 的 logprob。"""
    valid = [lp for lp in logprobs if lp is not None]
    if not valid:
        return None
    return math.exp(-sum(valid) / len(valid))


def logprob_drift(lps_a, lps_b):
    """逐 token top-1 logprob 漂移统计（与 quality_gate.py 包络同口径）。
    返回 None（长度不齐/缺席）或 dict: n, mean_abs_diff, max_abs_diff,
    sum_drift_pct（= (sumA-sumB)/|sumB|*100，符号以 B=参考）。"""
    if not lps_a or not lps_b or len(lps_a) != len(lps_b):
        return None
    diffs = [abs(a - b) for a, b in zip(lps_a, lps_b) if a is not None and b is not None]
    if not diffs:
        return None
    sum_b = sum(b for b in lps_b if b is not None)
    return {
        "n": len(diffs),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
        "max_abs_diff": round(max(diffs), 6),
        "sum_drift_pct": round((sum(a for a in lps_a if a is not None) - sum_b)
                               / abs(sum_b) * 100, 4) if sum_b else None,
    }


def mean_wt(per_sample, key, weights):
    """按 token 数加权的样本级指标聚合。per_sample: dict[id]=dict。"""
    tot = 0.0
    wsum = 0.0
    n = 0
    for sid, v in per_sample.items():
        w = weights.get(sid, 0) or 0
        val = v.get(key)
        if val is None or w <= 0:
            continue
        tot += val * w
        wsum += w
        n += 1
    return round(tot / wsum, 6) if wsum > 0 else None, n


def format_verdict(out_path, verdict):
    write_json(out_path, verdict)
    print(f"[verdict] -> {out_path}")
    return verdict


def need_key():
    if not KEY:
        print("[error] 缺 VLLM_API_KEY 环境变量（采集需要；纯分析不需要）",
              file=sys.stderr)
        return False
    return True
