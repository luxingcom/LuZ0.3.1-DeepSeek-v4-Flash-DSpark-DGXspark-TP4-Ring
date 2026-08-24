#!/usr/bin/env python3
"""patch_start_scripts.py — 给 start_tp4_head.sh / start_tp4_worker.sh 打 plugin_a1 插桩。
改动: (1) SERVE_CMD 前置 pip install 插件 (2) ENV_ARGS 追加 3 个 env(MODE 文件驱动)。
幂等: 已打过补丁则跳过。先做 .bak-plugin-20260821 留档。"""
import shutil
import sys

ENV_ANCHOR = "  -e 'VLLM_TRITON_MLA_SPARSE=1'\n"
ENV_INSERT = (
    "  # === plugin_a1 W4A4 prefill (A' route, P2 2026-08-21): MODE 文件驱动; 0=off 生产原样 ===\n"
    "  -e \"VLLM_MOE_W4A4=$(cat <INSTALL_DIR>/nvfp4/plugin_a1/MODE 2>/dev/null | tr -d '[:space:]')\"\n"
    "  -e 'VLLM_MOE_W4A4_MIN_M=3072'\n"
    "  -e 'VLLM_MOE_W4A4_CG=1'\n"
)
SERVE_OLD = 'SERVE_CMD="vllm serve\\'
SERVE_NEW = ('SERVE_CMD="rm -rf /tmp/plugin_a1_install; '
             'cp -r <INSTALL_DIR>/nvfp4/plugin_a1 /tmp/plugin_a1_install 2>/dev/null; '
             'pip install --no-deps -q /tmp/plugin_a1_install >/dev/null 2>&1; '
             'vllm serve\\')


def patch(path):
    with open(path) as f:
        src = f.read()
    if "plugin_a1" in src:
        print(f"[skip] {path}: 已含 plugin_a1 补丁")
        return
    bak = path + ".bak-plugin-20260821"
    shutil.copy2(path, bak)
    assert src.count(ENV_ANCHOR) == 1, f"ENV anchor count={src.count(ENV_ANCHOR)} in {path}"
    assert src.count(SERVE_OLD) == 1, f"SERVE anchor count={src.count(SERVE_OLD)} in {path}"
    src = src.replace(ENV_ANCHOR, ENV_ANCHOR + ENV_INSERT)
    src = src.replace(SERVE_OLD, SERVE_NEW)
    with open(path, "w") as f:
        f.write(src)
    print(f"[ok] {path} patched (bak={bak})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
