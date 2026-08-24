#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task7: Redaction batch-replace tool for the release copy (open-source repo).
APPLIES TO THE RELEASE COPY ONLY (never the source).
Patterns are loaded from ./redact-patterns.json which is NOT committed
(see .gitignore); this file only ships the tool + placeholder semantics.

Usage:
    python redact.py            # apply replacements (idempotent)
    python redact.py --map-only # regenerate REDACTION-MAP.csv without rewriting files
"""
import os, re, csv, json, sys

DST = os.path.dirname(os.path.abspath(__file__))
PATTERNS_FILE = os.path.join(DST, "redact-patterns.json")

# semantic order: more specific first
SEMANTIC = [
    ("sudo password", "<PASSWORD>"),
    ("old API key prefix", "<KEY_PREFIX_OLD>"),
    ("new API key prefix", "<KEY_PREFIX_NEW>"),
    ("openai-style api key", "<API_KEY>"),
    ("api_key value", "<API_KEY>"),
    ("VLLM_API_KEY", "<API_KEY>"),
    ("bearer token", "<BEARER>"),
    ("internal username", "<USER>"),
    ("install dir", "<INSTALL_DIR>"),
    ("models dir", "<MODELS_DIR>"),
    ("home dir", "<HOME_DIR>"),
    ("hostname node", "node0X"),
    ("internal ip", "<NODE_IP>"),
]

EXCLUDE_NAMES = {"redact.py", "REDACTION-MAP.csv", ".gitignore", "README.md",
                 "redact-patterns.json"}

def load_rules():
    if not os.path.exists(PATTERNS_FILE):
        sys.exit("redact-patterns.json not found — patterns are kept out of the release repo.")
    with open(PATTERNS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [(r["regex"], r["placeholder"], r["description"]) for r in data["rules"]]

def apply_rules(map_only=False):
    rules = load_rules()
    rows = []
    total = 0
    for root_dir, _, files in os.walk(DST):
        if "_server_fetch" in root_dir or ".git" in root_dir:
            continue
        for fn in files:
            if fn in EXCLUDE_NAMES:
                continue
            fp = os.path.join(root_dir, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".pyc", ".zip", ".tar", ".gz", ".so", ".bin", ".png", ".jpg", ".svg", ".csv"):
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
            except Exception:
                continue
            new = txt
            for pattern, placeholder, desc in rules:
                new, n = re.subn(pattern, placeholder, new)
                if n:
                    rows.append((pattern, placeholder, desc, n))
                    total += n
            if (not map_only) and new != txt:
                with open(fp, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
    agg = {}
    for a, b, c, n in rows:
        agg[(a, b, c)] = agg.get((a, b, c), 0) + n
    with open(os.path.join(DST, "REDACTION-MAP.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regex", "placeholder", "description", "count_replaced"])
        for (a, b, c), n in sorted(agg.items(), key=lambda x: -x[1]):
            w.writerow([a, b, c, n])
    print("REDACTION DONE total replacements:", total)

if __name__ == "__main__":
    apply_rules(map_only="--map-only" in sys.argv)
