#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task7: Redaction mapping table + batch replace for the release copy.
APPLIES TO THE RELEASE COPY ONLY (never the source). 
Self-excludes this script, build helpers, and the REDACTION-MAP output.
Re-runnable: placeholders are idempotent.
"""
import os, re, csv

DST = r"C:\Users\novAI\WorkBuddy\集群部署\deliverables\engineering-assurance\release\LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring"

EXCLUDE_NAMES = {"redact.py", "REDACTION-MAP.csv", "_build_manifest.txt", "_build_run.txt",
                 "_org.txt", "_fetch.txt", "_fetch2.txt", "_fetch3.txt", "_fetch_list.txt",
                 "_fix.txt", "_scan1.txt", "_verify.txt", "_verify2.txt", "_skeleton.txt",
                 "_docs_sub.txt", "pyver_check.txt", "README.md", ".gitignore",
                 "build_release_copy.py"}

# ---- redaction mapping: (regex, placeholder, description) ----
RULES = [
    # secrets / passwords (explicit)
    (r"AS1217hf", "<PASSWORD>", "sudo password plaintext"),
    (r"c3b4de54", "<KEY_PREFIX_OLD>", "old API key prefix"),
    (r"0bec83af", "<KEY_PREFIX_NEW>", "new API key prefix"),
    # api keys
    (r"sk-[A-Za-z0-9]{8,}", "<API_KEY>", "openai-style api key"),
    (r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9_\-]{12,})", r"\1<API_KEY>", "api_key= value"),
    (r"(?i)(VLLM_API_KEY['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9_\-]{12,})", r"\1<API_KEY>", "VLLM_API_KEY"),
    (r"(?i)(Bearer\s+)([A-Za-z0-9_\-\.]{12,})", r"\1<BEARER>", "bearer token"),
    # internal username
    (r"\bliuxiaoya\b", "<USER>", "internal username"),
    # internal paths (specific first)
    (r"/opt/aicad-prod", "<INSTALL_DIR>", "internal install dir"),
    (r"/home/liuxiaoya/models", "<MODELS_DIR>", "models dir"),
    (r"/home/liuxiaoya", "<HOME_DIR>", "home dir"),
    (r"/data/models", "<MODELS_DIR>", "models dir alt"),
    # hostnames
    (r"dgxspark0[1-4]", "node01", "hostname -> node01..04"),
    # internal IPs with optional port -> placeholder + port preserved
    (r"(192\.168\.\d{1,3}\.\d{1,3})(:\d{1,5})?", r"<NODE_IP>\2", "internal ip"),
    (r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<NODE_IP>", "internal ip 10.x"),
]

def build_map_csv():
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
            for pattern, placeholder, desc in RULES:
                new, n = re.subn(pattern, placeholder, new)
                if n:
                    rows.append((pattern, placeholder, desc, n))
                    total += n
            if new != txt:
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
    build_map_csv()
