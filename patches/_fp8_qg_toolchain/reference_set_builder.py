#!/usr/bin/env python3
"""reference_set_builder.py — 参考集构建/校验/长上下文 prompt 生成
================================================================================
零 GPU 工具：
  validate  — 校验 reference_set.json 结构、id 唯一、Tier A 与 quality_gate.py
              4 prompt 逐字一致（防漂移）。
  build     — 生成"可采集清单"（普通 prompt 直出；长上下文展开为完整 prompt
              文本写入 _generated/<id>.txt，供采集脚本与人工复核）。
  stats     — 打印参考集数量/分布统计。

长上下文生成规则（G7）:
  - needle: 用无重复的 filler 句子填充至 target_len_tokens，在 depth 位置插入
    needle 句，末尾附 question。filler 句子数量按字符/token 近似（~4 char/token
    中文或 ~0.75 token/word；本实现用保守英文 filler，见 FILLER_SENTENCE）。
  - tail:   filler 填满 target_len_tokens 后，尾部直接附 tail_instruction。
  token 估算为近似值；真实 token 数以采集脚本实际 prompt token 数为准（附录中
  记录估算误差，最终规模以 64K 抽验窗口实测校准）。
================================================================================
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference_set.json")
GEN_DIR = os.path.join(HERE, "_generated")

# 与 quality_gate.py 一致的 FOX 串（Tier A fox_repeat 校验用）
FOX = "The quick brown fox jumps over the lazy dog. "
QUALITY_GATE_A = {
    "fox_repeat": FOX * 60 + "\nRepeat the sentence above exactly once.",
    "count": "Count from 1 to 30, one number per line, no other text.",
    "code_fib": "Write a Python function to compute the nth Fibonacci number using memoization. Output code only.",
    "list": "List the first 20 chemical elements with their atomic numbers, one per line, format: number. symbol name.",
}

# 长上下文 filler：无重复的独立句子（每句约 12 token）
FILLER_SENTENCES = [
    "The harbor was quiet under a pale morning sky.",
    "A gardener watered the roses beside the stone wall.",
    "The old clock in the tower struck nine times.",
    "She read a page from the leather-bound journal.",
    "The ferry crossed the bay carrying a few passengers.",
    "Rain tapped softly on the tin roof of the shed.",
    "Two cyclists climbed the hill along the coast road.",
    "The baker arranged fresh bread in the window.",
    "Clouds gathered slowly above the distant mountains.",
    "A cat slept on the windowsill in the afternoon sun.",
    "The librarian shelved returns near the reading room.",
    "Workers painted the fence with white enamel.",
    "A violin played a slow melody in the plaza.",
    "The fishermen mended nets by the wooden pier.",
    "Leaves drifted across the empty courtyard.",
    "A child flew a kite on the windy beach.",
    "The chemist measured a clear liquid into the flask.",
    "Cars lined up at the crossing as the light turned red.",
    "The astronomer pointed the telescope at the eastern sky.",
    "A letter arrived in the morning mail with no return address.",
]


def load_ref(path=None):
    with open(path or REF, "r", encoding="utf-8") as f:
        return json.load(f)


def _id_map(doc):
    ids = {}
    for s in doc["samples"]:
        if s["id"] in ids:
            raise ValueError(f"重复 sample id: {s['id']}")
        ids[s["id"]] = s
    return ids


def cmd_validate(args):
    doc = load_ref(args.ref)
    ids = _id_map(doc)
    errors = []
    n = {"A": 0, "B": 0, "C": 0}
    for s in doc["samples"]:
        t = s.get("tier")
        if t not in doc["tiers"]:
            errors.append(f"{s['id']}: 未知 tier {t}")
        else:
            n[t] += 1
        if not s.get("prompt") and "long_ctx" not in s:
            errors.append(f"{s['id']}: 缺少 prompt 或 long_ctx")
        if s.get("tier") == "C" and "long_ctx" not in s:
            errors.append(f"{s['id']}: Tier C 必须带 long_ctx")
    # Tier A 与 quality_gate.py 对齐校验
    for aid, expected in QUALITY_GATE_A.items():
        s = ids.get(aid)
        if not s:
            errors.append(f"Tier A 缺 {aid}")
            continue
        if s["tier"] != "A":
            errors.append(f"{aid}: 应为 Tier A")
        if s.get("prompt") != expected:
            errors.append(f"{aid}: prompt 与 quality_gate.py 不一致")
    print(f"[validate] samples={sum(n.values())} A={n['A']} B={n['B']} C={n['C']}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("[validate] OK: 结构与 quality_gate.py 对齐")
    return 0


def build_longctx_prompt(s, meta):
    lc = s["long_ctx"]
    target = int(lc.get("target_len_tokens", 65536))
    # 估算每 filler 句 token（~12），保守循环直至达到目标
    per_sent = 12
    n_sent = max(1, int(target / per_sent))
    filler = ""
    i = 0
    while len(filler.split()) < target * 0.9:  # 词数下限近似
        filler += FILLER_SENTENCES[i % len(FILLER_SENTENCES)] + " "
        i += 1
    words = filler.split()
    kind = lc.get("type")
    if kind == "needle":
        depth = float(lc.get("depth", 0.5))
        n = len(words)
        insert_at = max(1, int(n * depth))
        body = " ".join(words[:insert_at]) + " " + lc["needle"] + " " + " ".join(words[insert_at:])
        prompt = body + "\n\n" + lc["question"]
    elif kind == "tail":
        body = " ".join(words)
        prompt = body + "\n\n" + lc["tail_instruction"]
    else:
        raise ValueError(f"{s['id']}: 未知 long_ctx.type {kind}")
    est_tokens = int(len(prompt) / 4.0)  # 字符/token 粗估
    return prompt, est_tokens


def cmd_build(args):
    doc = load_ref(args.ref)
    os.makedirs(GEN_DIR, exist_ok=True)
    manifest = []
    for s in doc["samples"]:
        entry = dict(s)
        if "long_ctx" in s:
            prompt, est = build_longctx_prompt(s, doc.get("meta", {}))
            entry["generated_prompt"] = prompt
            entry["est_tokens"] = est
            fn = os.path.join(GEN_DIR, s["id"] + ".txt")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(prompt)
            entry["prompt_file"] = fn
        manifest.append(entry)
    out = os.path.join(HERE, "collect_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"schema": "fp8-qg-manifest/1", "created": "2026-08-24",
                   "samples": manifest}, f, ensure_ascii=False, indent=1)
    print(f"[build] -> {out} ({len(manifest)} samples)")
    for s in manifest:
        if s.get("est_tokens"):
            print(f"  {s['id']}: est_tokens={s['est_tokens']} -> {s.get('prompt_file')}")
    return 0


def cmd_stats(args):
    doc = load_ref(args.ref)
    by_tier = {}
    by_cat = {}
    for s in doc["samples"]:
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    print(f"[stats] total={len(doc['samples'])}")
    print(f"  by_tier: {by_tier}")
    print(f"  by_cat : {by_cat}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="FP8 质量门参考集构建/校验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "build", "stats"):
        p = sub.add_parser(name)
        p.add_argument("--ref", default=REF, help="参考集 JSON 路径")
    args = ap.parse_args()
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
