#!/usr/bin/env python3
"""
routeB Phase 1：CUTLASS Python DSL sm_121a patch（自动执行 + 可回滚 + 原子写 + 参数化）

背景：CUTLASS issue #2800 —— BlockScaledMmaOp / MmaMXF4Op 把 FP4 MMA 限制在
      sm_100a，sm_120a/sm_121a 被 admissible_archs 拒收。baristankut 实证：
      加 sm_121a 后 SM121 dense NVFP4 达 356 TFLOPS。

本脚本自动完成两处 patch（论坛 #364607 + ai-muninn 配方第 2 层）：
  ① warp/mma.py 所有 admissible_archs 列表加 sm_121a（原版只改第一处，已修复）
  ② warp/mma.py base equality check 放宽到 (sm_120a, sm_121a)

用法：
  python patch_cutlass_dsl_sm121a.py                        # 自动定位并应用 patch
  python patch_cutlass_dsl_sm121a.py --target /path/mma.py  # 指定目标（本地副本测试用）
  python patch_cutlass_dsl_sm121a.py --revert               # 回滚（恢复 .bak）
  python patch_cutlass_dsl_sm121a.py --target X --revert    # 对指定目标回滚

修复记录（Task #13，2026-08-20，precheck 行动清单）：
  A2  equality-check 变换改为产出 `if arch not in (Arch.sm_120a, Arch.sm_121a):`
      / `if arch in (...)` 形式。原版把 `Arch.sm_120a` 替换为
      `Arch.sm_120a, Arch.sm_121a`，对 `!=` / `not ==` 源会产生
      `if not arch == Arch.sm_120a, Arch.sm_121a:` —— SyntaxError。
  A12 admissible_archs 插入改为引号风格自适应正则，兼容单引号 / 双引号 /
      裸 token / Arch.xxx 枚举写法；重复执行不叠加（幂等）。
  A13 写入改为临时文件 + os.replace 原子替换（原版直接 open(w) 非原子）。
  A19 revert() 无备份时 sys.exit(1) 非静默失败（原版仅 print）。
  附带修复：admissible_archs 多处出现时全部处理（原版 re.search 只取第一处）。
  新增 --target 参数，便于对本地副本实测，不依赖生产环境 import cutlass。

⚠ 重要事实（2026-08-20 现场核验，供版本决策）：
  生产镜像 dspark-vllm-gx10:0.2.1-v026.0 内 nvidia-cutlass-dsl==4.5.2 的
  warp/mma.py 中 MmaSM120BlockScaledOp.admissible_archs 已含 Arch.sm_121a
  （两处列表均已包含，且无 equality check，检查已为 `not in` 成员式）——
  DSL 4.5.2 上本 patch 为天然 no-op。本 patch 仅对 ≤4.4.x（如 setup 钉
  4.4.2 降级安装后）必要。另：BlockScaledMmaOp 在 4.5.2 中已不存在，
  验证提示改为 MmaMXF4Op（其继承 MmaSM120BlockScaledOp 的 admissible_archs）。
"""
import os
import re
import shutil
import sys

# ① admissible_archs 列表（兼容单/双引号、枚举多种写法）
ADMISSIBLE_PATTERN = re.compile(r'admissible_archs\s*=\s*\[[^\]]*\]')

# ② equality check（兼容 == / != / not == 写法）
#    捕获组 1: 可选 not 前缀；捕获组 2: == 或 !=
EQ_PATTERN = re.compile(r'if\s+(not\s+)?arch\s*([!=]=)\s*Arch\.sm_120a\s*:')

BAK_SUFFIX = ".bak-routeb"


def locate_mma_py() -> str:
    """定位 cutlass DSL 的 warp/mma.py。"""
    import cutlass
    base = os.path.dirname(cutlass.__file__)
    candidates = [
        # v4.5.x 实测布局（python_packages 为实际 import 根）
        os.path.join(base, "cute", "nvgpu", "warp", "mma.py"),
        # 备选布局（dsl_packages 树）
        os.path.join(os.path.dirname(base), "..", "dsl_packages",
                     "cutlass", "cute", "nvgpu", "warp", "mma.py"),
        # v4.4.x 安装布局兜底
        os.path.join(base, "python_packages", "cutlass", "cute",
                     "nvgpu", "warp", "mma.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    raise FileNotFoundError(
        "未找到 mma.py；请手工定位：python -c \"import cutlass, os; "
        "print(os.path.dirname(cutlass.__file__))\" 后 find 该目录 -name mma.py，"
        "或用 --target 显式指定路径"
    )


def _insert_121a_into_list(list_text: str) -> str | None:
    """在 admissible_archs 列表文本中插入 sm_121a（引号风格自适应，幂等）。

    返回新列表文本；无法安全插入时返回 None（调用方跳过并告警）。
    """
    if "sm_121a" in list_text:
        return list_text  # 已含，幂等跳过
    # 优先：枚举写法 Arch.sm_120a → Arch.sm_120a, Arch.sm_121a
    if re.search(r'Arch\.sm_120a\b', list_text):
        return list_text.replace(
            "Arch.sm_120a", "Arch.sm_120a, Arch.sm_121a", 1)
    # 其次：带引号写法（单或双），保持原引号风格
    m = re.search(r"(['\"])(sm_120a)\1", list_text)
    if m:
        q = m.group(1)
        return list_text.replace(
            m.group(0), f"{q}sm_120a{q}, {q}sm_121a{q}", 1)
    # 最后：裸 token 写法
    if re.search(r'(?<![.\w])sm_120a\b', list_text):
        return list_text.replace("sm_120a", "sm_120a, sm_121a", 1)
    return None


def _atomic_write(path: str, content: str) -> None:
    """A13：临时文件 + os.replace 原子替换。"""
    tmp = path + ".tmp-routeb"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    os.replace(tmp, path)


def apply_patch(path: str) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    bak = path + BAK_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"备份 → {bak}")

    changed = False

    # ① admissible_archs：所有含 sm_120a 而不含 sm_121a 的列表均加 sm_121a
    def _sub_admissible(m: re.Match) -> str:
        nonlocal changed
        old = m.group(0)
        new = _insert_121a_into_list(old)
        if new is None:
            print("⚠️  admissible_archs 列表不含 sm_120a，跳过（需人工核对）")
            return old
        if new != old:
            changed = True
            print("✅ ① admissible_archs 已加 sm_121a")
            return new
        print("⏭️  ① admissible_archs 已含 sm_121a")
        return old

    src = ADMISSIBLE_PATTERN.sub(_sub_admissible, src)

    # ② equality check：放宽为 in / not in (sm_120a, sm_121a) 形式（A2 修复）
    eq_hits = EQ_PATTERN.findall(src)

    def _sub_eq(m: re.Match) -> str:
        nonlocal changed
        negated = (m.group(1) is not None) or (m.group(2) == "!=")
        kw = "not in" if negated else "in"
        changed = True
        return f"if arch {kw} (Arch.sm_120a, Arch.sm_121a):"

    src = EQ_PATTERN.sub(_sub_eq, src)
    if eq_hits:
        print(f"✅ ② equality check 已放宽 x{len(eq_hits)} → "
              f"'if arch not in (Arch.sm_120a, Arch.sm_121a):' 形式")
    else:
        print("⏭️  ② equality check 模式未匹配（可能已 patch 过 / 4.5.2 天然无此检查）")

    if changed:
        _atomic_write(path, src)  # A13：原子写
        print(f"\n✅ patch 已写入 {path}")
    else:
        print("\n✅ 无变更（可能已 patch 过，或 DSL ≥4.5 天然支持 sm_121a）")


def revert(path: str) -> None:
    bak = path + BAK_SUFFIX
    if os.path.exists(bak):
        shutil.copy2(bak, path)
        print(f"✅ 已回滚 {path}")
    else:
        # A19：非静默失败
        print(f"❌ 无备份文件 {bak}，无法回滚", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    do_revert = "--revert" in args
    target = None
    if "--target" in args:
        try:
            target = args[args.index("--target") + 1]
        except IndexError:
            print("❌ --target 需要一个路径参数", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(target):
            print(f"❌ 目标文件不存在: {target}", file=sys.stderr)
            sys.exit(2)

    path = target if target else locate_mma_py()
    print(f"目标文件: {path}")
    if do_revert:
        revert(path)
    else:
        apply_patch(path)
    print("\n验证：python -c \"from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op; "
          "print(MmaMXF4Op.admissible_archs)\"")
    print("预期输出含 'sm_121a'")


if __name__ == "__main__":
    main()
