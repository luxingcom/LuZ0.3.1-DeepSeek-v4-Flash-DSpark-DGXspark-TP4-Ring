#!/usr/bin/env python3
"""
routeB Phase 1：CUTLASS 4.4.0 Python DSL sm_121a patch（自动执行 + 可回滚）

背景：CUTLASS issue #2800 —— BlockScaledMmaOp / MmaMXF4Op 把 FP4 MMA 限制在
      sm_100a，sm_120a/sm_121a 被 admissible_archs 拒收。baristankut 实证：
      加 sm_121a 后 SM121 dense NVFP4 达 356 TFLOPS。

本脚本自动完成两处 patch（论坛 #364607 + ai-muninn 配方第 2 层）：
  ① warp/mma.py admissible_archs 加 sm_121a
  ② warp/mma.py base equality check 放宽到 (sm_120a, sm_121a)

用法：
  python patch_cutlass_dsl_sm121a.py          # 应用 patch
  python patch_cutlass_dsl_sm121a.py --revert # 回滚（恢复 .bak）
"""
import os
import re
import shutil
import sys

ADMISSIBLE_PATTERNS = [
    # ① admissible_archs 列表（兼容多种写法）
    r'admissible_archs\s*=\s*\[[^\]]*\]',
    # ② equality check（兼容多种写法）
    r'if\s+arch\s*[!=]=\s*Arch\.sm_120a\s*:',
    r'if\s+not\s+arch\s*==\s*Arch\.sm_120a\s*:',
]

def locate_mma_py() -> str:
    """定位 cutlass DSL 的 warp/mma.py。"""
    import cutlass
    base = os.path.dirname(cutlass.__file__)
    # 常见路径（v4.4.x 安装布局）
    candidates = [
        os.path.join(base, "cute", "nvgpu", "warp", "mma.py"),
        os.path.join(base, "python_packages", "cutlass", "cute", "nvgpu", "warp", "mma.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "未找到 mma.py；请手工定位：python -c \"import cutlass, os; "
        "print(os.path.dirname(cutlass.__file__))\" 后 find 该目录 -name mma.py"
    )

def apply_patch(path: str) -> None:
    src = open(path, encoding="utf-8").read()
    bak = path + ".bak-routeb"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"备份 → {bak}")

    changed = False

    # ① admissible_archs：确保含 sm_121a
    m = re.search(ADMISSIBLE_PATTERNS[0], src)
    if m:
        old = m.group(0)
        if "sm_121a" not in old:
            # 在列表内插入 sm_121a（保持 sm_120a 存在）
            if "sm_120a" in old:
                new = old.replace("sm_120a", "sm_120a\", \"sm_121a", 1)
            else:
                new = re.sub(r'\[', '["sm_120a", "sm_121a"', old, count=1)
            src = src.replace(old, new, 1)
            changed = True
            print("✅ ① admissible_archs 已加 sm_121a")
        else:
            print("⏭️  ① admissible_archs 已含 sm_121a")
    else:
        print("⚠️  未匹配 admissible_archs 模式（需人工核对）")

    # ② equality check：放宽为 (sm_120a, sm_121a)
    for pat in ADMISSIBLE_PATTERNS[1:]:
        m = re.search(pat, src)
        if m and "sm_121a" not in m.group(0):
            old = m.group(0)
            new = old.replace(
                "Arch.sm_120a", "Arch.sm_120a, Arch.sm_121a"
            )
            # 处理 `!=` 语义：arch not in (...) 形式
            src = src.replace(old, new, 1)
            changed = True
            print(f"✅ ② equality check 已放宽: {new.strip()[:80]}")
            break
    else:
        if changed:
            print("⏭️  ② equality check 模式未单独匹配（可能已随①处理，或需人工核对）")

    if changed:
        open(path, "w", encoding="utf-8").write(src)
        print(f"\n✅ patch 已写入 {path}")
    else:
        print("\n✅ 无变更（可能已 patch 过）")

def revert(path: str) -> None:
    bak = path + ".bak-routeb"
    if os.path.exists(bak):
        shutil.copy2(bak, path)
        print(f"✅ 已回滚 {path}")
    else:
        print("❌ 无备份文件，无法回滚")

def main() -> None:
    path = locate_mma_py()
    print(f"目标文件: {path}")
    if "--revert" in sys.argv:
        revert(path)
    else:
        apply_patch(path)
    print("\n验证：python -c \"from cutlass.cute.nvgpu.warp.mma import BlockScaledMmaOp; "
          "print(BlockScaledMmaOp.admissible_archs)\"")
    print("预期输出含 'sm_121a'")

if __name__ == "__main__":
    main()
