#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b′ L1 CPU 验证（一次性容器内运行, 无 GPU）:

T1  off 路径零行为变化 — VLLM_MOE_W4A4_NATIVE 未设/0 时, bprime 插件与池化版
    plugin_a1（/tmp/_wsdedup_l3/w4a4_experts_pooled.py, phase3b L1/L3 已验证资产）
    在 _derive_w4a4 / 分派决策 / wrapper 构造 kwargs / _w13_layout 上逐项一致。
T2  native prepare 契约 — prepare_w4a16_e8m0_native_weights 输出
    weight_layout/modelopt、payload 零拷贝（data_ptr 断言）、scale grid 形状、
    micro 共享、row_rotation 等变性（pack(roll(s)) == roll(pack(s))）。
T3  monkeypatch — _install_native_layout_policy_patch 后
    _w4a16_weight_layout_for_source(fp4_e8m0_k32)=="modelopt"、其他格式透传
    "packed"、幂等、native off 时不安装（子进程核验）。

运行: python3 test_l1_cpu.py  (容器内, PYTHONPATH 含插件父目录)
"""
import importlib.util
import os
import subprocess
import sys
import types

import torch

PASS, FAIL = [], []

# CPU 容器纪律: swizzle_blockscale 内部 .cuda() 迁移在无驱动容器不可用 ——
# 置 no-op（数学路径保持原实现, 仅去设备迁移; 报告口径注明）
torch.Tensor.cuda = lambda self, *a, **kw: self


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PLUGIN_DIR = os.environ.get("BPRIME_PLUGIN_DIR", "/opt/plugin_bprime")
POOLED_PATH = os.environ.get("POOLED_BASE_PATH", "/assets/w4a4_experts_pooled_base.py")
BPRIME_PATH = os.path.join(PLUGIN_DIR, "routea_plugin_a1_bprime", "w4a4_experts.py")

# ---------------------------------------------------------------- common fakes
E, HIDDEN, INTER = 32, 512, 256
N13 = 2 * INTER  # w13 rows
TOPK = 4


class FakeMoeConfig:
    experts_per_token = TOPK
    hidden_dim = HIDDEN
    intermediate_size_per_partition = INTER
    max_num_tokens = 4096
    activation = None
    in_dtype = torch.bfloat16


class FakeParam:
    def __init__(self, t):
        self.data = t
        self.device = t.device
        self.dtype = t.dtype


class FakeLayer:
    def __init__(self, w13, w2, s13, s2):
        self.w13_weight = FakeParam(w13)
        self.w2_weight = FakeParam(w2)
        self.w13_weight_scale = FakeParam(s13)
        self.w2_weight_scale = FakeParam(s2)
        self.activation = None


def make_weights(seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(0, 256, (E, N13, HIDDEN // 2), dtype=torch.uint8,
                        generator=g, device=device)
    w2 = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8,
                       generator=g, device=device)
    s13 = torch.randint(118, 127, (E, N13, HIDDEN // 32), dtype=torch.uint8,
                        generator=g, device=device)
    s2 = torch.randint(118, 127, (E, HIDDEN, INTER // 32), dtype=torch.uint8,
                       generator=g, device=device)
    return w13, w2, s13, s2


def make_instance(cls, mode, extra=None):
    """绕过 B12xExperts.__init__（重构造, 与被测逻辑无关）, 手工置插件状态。"""
    obj = cls.__new__(cls)
    obj._w4a4_mode = mode
    if hasattr(cls, "_w4a4_native") or "_w4a4_native" in cls.__init__.__code__.co_names:
        obj._w4a4_native = (extra or {}).get("native", False)
    obj._w4a4_min_m = 3072
    obj._w4a4_debug = False
    obj._w4a4_ready = False
    obj._native_prepared = None
    obj._wrapper = None
    obj._w13 = None
    obj._w2 = None
    obj._w13_sf_mma = None
    obj._w13_sf_store = None
    obj._w2_sf_mma = None
    obj._w2_sf_store = None
    obj._unit = None
    obj._fc2_scale = None
    obj._prepared_fp4_moe_by_dtype = {}
    obj._unit_scale_by_device = {}
    obj._released_w4a16_source_scales = False
    obj.moe_config = FakeMoeConfig()
    obj.quant_config = types.SimpleNamespace()
    return obj


WRAPPER_CALLS = []


class RecordingWrapper:
    def __init__(self, **kwargs):
        WRAPPER_CALLS.append(kwargs)
        self.kwargs = kwargs

    def run(self, **kwargs):
        raise RuntimeError("not used in L1")


def install_wrapper_stub():
    import flashinfer.fused_moe as ff
    ff.B12xMoEWrapper = RecordingWrapper


def maybe_stub_sf_conversion(*mods):
    """flashinfer_convert_sf_to_mma_layout 若非 CPU 安全, 以确定性 stub 替换
    （两模块同 stub —— 对比有效性保持; 记录在报告口径中）。"""
    stubbed = []
    for mod in mods:
        fn = getattr(mod, "flashinfer_convert_sf_to_mma_layout", None)
        if fn is None:
            continue
        try:
            t = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
            fn(t.reshape(2, 16), m=4, k=16, num_groups=2)
        except Exception:
            mod.flashinfer_convert_sf_to_mma_layout = (
                lambda x, m, k, num_groups: x.reshape(num_groups, -1))
            stubbed.append(mod.__name__)
    return stubbed


# ================================================================ T1
def t1_off_equivalence():
    print("\n=== T1: off 路径零行为变化（native 未设/0 ⇒ 与池化版一致）===")
    pooled = load_module_by_path("w4a4_experts_pooled_base", POOLED_PATH)
    bprime = load_module_by_path("w4a4_experts_bprime", BPRIME_PATH)
    stubbed = maybe_stub_sf_conversion(pooled, bprime)
    if stubbed:
        print(f"[note] sf mma conversion stubbed on CPU for: {stubbed}")

    install_wrapper_stub()

    for native_env in (None, "0"):
        if native_env is None:
            os.environ.pop("VLLM_MOE_W4A4_NATIVE", None)
        else:
            os.environ["VLLM_MOE_W4A4_NATIVE"] = native_env
        for mode in (0, 1, 2):
            WRAPPER_CALLS.clear()
            w = make_weights(seed=1234)
            layer_p = FakeLayer(*[t.clone() for t in w])
            layer_b = FakeLayer(*[t.clone() for t in w])
            inst_p = make_instance(pooled.W4A4B12xExperts, mode)
            inst_b = make_instance(bprime.W4A4B12xExperts, mode,
                                   {"native": False})
            if mode in (1, 2):
                inst_p._derive_w4a4(layer_p)
                n_p = len(WRAPPER_CALLS)
                inst_b._derive_w4a4(layer_b)
                n_b = len(WRAPPER_CALLS) - n_p
                tag = f"native={native_env!r} mode={mode}"
                check(f"T1/{tag}/wrapper-kwargs",
                      n_p == 1 and n_b == 1
                      and WRAPPER_CALLS[0] == WRAPPER_CALLS[1],
                      f"kwargs equal={WRAPPER_CALLS[0] == WRAPPER_CALLS[1]}")
                check(f"T1/{tag}/w13-values",
                      torch.equal(inst_p._w13, inst_b._w13))
                check(f"T1/{tag}/w2-values",
                      torch.equal(inst_p._w2, inst_b._w2))
                check(f"T1/{tag}/sf13-values",
                      torch.equal(inst_p._w13_sf_store, inst_b._w13_sf_store))
                check(f"T1/{tag}/sf2-values",
                      torch.equal(inst_p._w2_sf_store, inst_b._w2_sf_store))
                # 别名关系一致（packed hybrid: 副本; full: 就地）
                check(f"T1/{tag}/aliasing",
                      (inst_p._w13.data_ptr() == layer_p.w13_weight.data.data_ptr())
                      == (inst_b._w13.data_ptr() == layer_b.w13_weight.data.data_ptr()))
            # 分派决策
            decisions_p = [inst_p._use_w4a4(m) for m in (1, 8, 96, 3071, 3072, 4096)]
            decisions_b = [inst_b._use_w4a4(m) for m in (1, 8, 96, 3071, 3072, 4096)]
            check(f"T1/native={native_env!r} mode={mode}/dispatch",
                  decisions_p == decisions_b, f"{decisions_p} vs {decisions_b}")
            # _w13_layout: off 时都为 "w31"（fork 基座语义）
            check(f"T1/native={native_env!r} mode={mode}/w13-layout",
                  inst_p._w13_layout() == inst_b._w13_layout() == "w31",
                  f"{inst_p._w13_layout()} / {inst_b._w13_layout()}")

    # native=1 仅在 mode=1 改变行为（对照确认）
    os.environ["VLLM_MOE_W4A4_NATIVE"] = "1"
    WRAPPER_CALLS.clear()
    w = make_weights(seed=1234)
    w_ref = [t.clone() for t in w]  # 原始序参考（derive 会就地半交换）
    layer = FakeLayer(*w)
    inst = make_instance(bprime.W4A4B12xExperts, 1, {"native": True})
    inst._derive_w4a4(layer)
    check("T1/native=1 mode=1/payload-alias（零拷贝共享）",
          inst._w13.data_ptr() == layer.w13_weight.data.data_ptr()
          and inst._w2.data_ptr() == layer.w2_weight.data.data_ptr())
    check("T1/native=1 mode=1/values-equal-packed-derive",
          torch.equal(
              inst._w13,
              torch.cat([w_ref[0][:, INTER:], w_ref[0][:, :INTER]], dim=1).contiguous()))
    check("T1/native=1 mode=1/w13-layout=w13", inst._w13_layout() == "w13")
    check("T1/native=1 mode=1/layer-payload-swapped-inplace",
          torch.equal(layer.w13_weight.data[:, :INTER], w_ref[0][:, INTER:]))
    os.environ.pop("VLLM_MOE_W4A4_NATIVE", None)


# ================================================================ T2
def t2_native_prepare():
    print("\n=== T2: native prepare 契约（CPU, _make_workspace stub）===")
    import b12x.moe.fused.w4a16.prepare as prep

    orig_mkws = prep._make_workspace
    prep._make_workspace = (
        lambda device, *, max_blocks_per_sm=4: torch.zeros(
            (8,), dtype=torch.int32))
    try:
        from b12x.moe.fused.w4a16.prepare import (
            prepare_w4a16_e8m0_native_weights)

        w13, w2, s13, s2 = make_weights(seed=7)
        unit = torch.ones(E, dtype=torch.float32)

        # 就地半交换（b′ derive 路径产生的物理序）
        n = N13 // 2
        tmp = w13[:, :n].clone()
        w13[:, :n].copy_(w13[:, n:])
        w13[:, n:].copy_(tmp)
        tmp = s13[:, :n].clone()
        s13[:, :n].copy_(s13[:, n:])
        s13[:, n:].copy_(tmp)

        native = prepare_w4a16_e8m0_native_weights(
            w13, s13, unit, w2, s2, unit,
            activation="silu", params_dtype=torch.bfloat16,
            w13_layout="w13")

        check("T2/weight_layout==modelopt",
              native.weight_layout == "modelopt", native.weight_layout)
        check("T2/source_format==fp4_e8m0_k32",
              native.source_format == "fp4_e8m0_k32")
        check("T2/w13-layout-preserved", native.w13_layout == "w13")
        check("T2/payload-zero-copy/w13",
              native.w13.data_ptr() == w13.data_ptr())
        check("T2/payload-zero-copy/w2",
              native.w2.data_ptr() == w2.data_ptr())
        check("T2/scale-grid-new-alloc/w13",
              native.w13_scale.data_ptr() != s13.data_ptr()
              and native.w13_scale.dtype in (torch.uint8, torch.float8_e8m0fnu),
              f"dtype={native.w13_scale.dtype}")
        check("T2/scale-grid-new-alloc/w2",
              native.w2_scale.data_ptr() != s2.data_ptr())
        exp_shape = (E, HIDDEN // 32, N13)
        check("T2/scale-grid-shape/w13",
              tuple(native.w13_scale.shape) == exp_shape,
              f"{tuple(native.w13_scale.shape)} vs {exp_shape}")
        exp_shape2 = (E, INTER // 32, HIDDEN)
        check("T2/scale-grid-shape/w2",
              tuple(native.w2_scale.shape) == exp_shape2,
              f"{tuple(native.w2_scale.shape)} vs {exp_shape2}")
        check("T2/micro-shares-packed-grid",
              native.micro_w13_scale is native.w13_scale
              and native.micro_w2_scale is native.w2_scale)
        check("T2/global-scales-unit-vals",
              torch.all(native.w13_global_scale == 1.0)
              and torch.all(native.w2_global_scale == 1.0))

        # row_rotation 等变性: pack(roll(s)) == roll(pack(s), dim=N-axis)
        s_raw = torch.randint(118, 127, (E, N13, HIDDEN // 32),
                              dtype=torch.uint8)
        g_rot = prep._pack_e8m0_k32_scales(
            s_raw, size_k=HIDDEN, size_n=N13, row_rotation=INTER)
        g_plain = prep._pack_e8m0_k32_scales(
            s_raw, size_k=HIDDEN, size_n=N13, row_rotation=None)
        rolled = torch.cat([g_plain[:, :, INTER:], g_plain[:, :, :INTER]],
                           dim=2)
        check("T2/pack-row-rotation-equivariance",
              torch.equal(g_rot, rolled),
              "pack(roll(s)) == roll(pack(s)) — scale grid 行随 payload 半交换"
              "一致旋转（kernel source_n_rotation 的配套不变量）")

        # 布局标签数学一致性: "w13"+swapped ≡ "w31"+unswapped（同一逻辑权重）。
        # pack(swap(s), rot=INTER) 中两次半交换相消 ⇒ 两场景 packed scale grid
        # 逐位相等; 物理序差异完全由 kernel 的 source_n_rotation 补偿。
        w13b, w2b, s13b, s2b = make_weights(seed=7)
        unitb = torch.ones(E, dtype=torch.float32)
        nat_w31 = prepare_w4a16_e8m0_native_weights(
            w13b, s13b, unitb, w2b, s2b, unitb,
            activation="silu", params_dtype=torch.bfloat16,
            w13_layout="w31")
        grid_w13 = native.w13_scale
        grid_w31 = nat_w31.w13_scale
        check("T2/w13-label-math-consistency",
              torch.equal(grid_w13.view(torch.uint8),
                          grid_w31.view(torch.uint8)),
              "native(swapped, w13).scale == native(unswapped, w31).scale"
              " —— 旋转在 prepare 内相消, 两布局声明产出同一 logical grid")
    finally:
        prep._make_workspace = orig_mkws


# ================================================================ T3
T3_SUBPROCESS = r"""
import os, sys
mode = os.environ.get("VLLM_MOE_W4A4", "0")
native = os.environ.get("VLLM_MOE_W4A4_NATIVE", "0")
sys.path.insert(0, os.environ["BPRIME_PLUGIN_DIR"])
import routea_plugin_a1_bprime  # noqa: F401  (import 即 install)
from b12x.integration import tp_moe as b12x_tp_moe
fn = b12x_tp_moe._w4a16_weight_layout_for_source
print("POLICY_E8M0", fn("fp4_e8m0_k32"))
print("POLICY_NVFP4", fn("modelopt_nvfp4"))
print("PATCHED", getattr(fn, "_bprime_patched", False))
print("INSTALLED", routea_plugin_a1_bprime._installed)
print("SMALLM_DIRECT", os.environ.get("B12X_W4A16_SMALL_M_DIRECT", "<unset>"))
"""


def t3_monkeypatch():
    print("\n=== T3: 布局政策 monkeypatch ===")
    env = dict(os.environ)
    env.update({
        "VLLM_MOE_W4A4": "1",
        "VLLM_MOE_W4A4_NATIVE": "1",
        "BPRIME_PLUGIN_DIR": PLUGIN_DIR,
    })
    out = subprocess.run([sys.executable, "-c", T3_SUBPROCESS], env=env,
                         capture_output=True, text=True, timeout=600)
    got = dict(line.split(" ", 1) for line in out.stdout.strip().splitlines()
               if " " in line)
    check("T3/native-on/policy-e8m0==modelopt", got.get("POLICY_E8M0") == "modelopt",
          f"stdout={out.stdout.strip()!r} stderr={out.stderr[-400:]!r}")
    check("T3/native-on/policy-nvfp4==packed(透传)",
          got.get("POLICY_NVFP4") == "packed")
    check("T3/native-on/patch-flag", got.get("PATCHED") == "True")
    check("T3/native-on/plugin-installed", got.get("INSTALLED") == "True")
    # e8m0×micro direct 上游缺陷（L1 GPU 实证）⇒ native 模式强制关闭
    check("T3/native-on/smallm-direct-forced-off",
          got.get("SMALLM_DIRECT") == "0")

    env2 = dict(env)
    env2["VLLM_MOE_W4A4_NATIVE"] = "0"
    out2 = subprocess.run([sys.executable, "-c", T3_SUBPROCESS], env=env2,
                          capture_output=True, text=True, timeout=600)
    got2 = dict(line.split(" ", 1) for line in out2.stdout.strip().splitlines()
                if " " in line)
    check("T3/native-off/policy-e8m0==packed(不安装)",
          got2.get("POLICY_E8M0") == "packed",
          f"stdout={out2.stdout.strip()!r} stderr={out2.stderr[-400:]!r}")
    check("T3/native-off/patch-flag-false", got2.get("PATCHED") == "False")
    check("T3/native-off/smallm-direct-untouched",
          got2.get("SMALLM_DIRECT") == "<unset>")

    env3 = dict(env)
    env3["VLLM_MOE_W4A4_NATIVE"] = "1"
    env3["VLLM_MOE_W4A4"] = "2"  # full: native 不适用, 不安装 patch
    out3 = subprocess.run([sys.executable, "-c", T3_SUBPROCESS], env=env3,
                          capture_output=True, text=True, timeout=600)
    got3 = dict(line.split(" ", 1) for line in out3.stdout.strip().splitlines()
                if " " in line)
    check("T3/full+native/policy-e8m0==packed(native 忽略)",
          got3.get("POLICY_E8M0") == "packed",
          f"stdout={out3.stdout.strip()!r} stderr={out3.stderr[-400:]!r}")


# ================================================================ T1b process_weights_after_loading (native 编排)
def t1b_native_orchestration():
    print("\n=== T1b: process_weights_after_loading native 编排顺序 ===")
    bprime = load_module_by_path("w4a4_experts_bprime2", BPRIME_PATH)
    maybe_stub_sf_conversion(bprime)
    install_wrapper_stub()

    calls = []
    orig_prewarm = bprime._prewarm_b12x_route_pack
    orig_cache = bprime._maybe_release_cuda_cache
    bprime._prewarm_b12x_route_pack = lambda **kw: calls.append(("prewarm", kw))
    bprime._maybe_release_cuda_cache = lambda dev: calls.append(("cache", dev))

    import b12x.moe.fused.w4a16.prepare as prep
    orig_mkws = prep._make_workspace
    prep._make_workspace = lambda device, *, max_blocks_per_sm=4: torch.zeros(
        (8,), dtype=torch.int32)

    from b12x.moe.fused.w4a16.prepare import prepare_w4a16_e8m0_native_weights
    real_prepare = prepare_w4a16_e8m0_native_weights
    prep.prepare_w4a16_e8m0_native_weights = (
        lambda *a, **kw: calls.append(("prepare", "called")) or real_prepare(*a, **kw))

    try:
        w13, w2, s13, s2 = make_weights(seed=99)
        layer = FakeLayer(w13, w2, s13, s2)
        inst = make_instance(bprime.W4A4B12xExperts, 1, {"native": True})
        inst._release_w4a16_source_scales = (
            lambda l: calls.append(("release_scales", l)))
        inst._release_w4a16_source_weights = (
            lambda l: calls.append(("release_weights", l)))

        inst.process_weights_after_loading(layer)

        order = [c[0] for c in calls]
        check("T1b/call-order",
              order == ["prepare", "prewarm", "release_scales",
                        "release_weights", "cache"],
              f"{order}")
        check("T1b/prepared-stored",
              torch.bfloat16 in inst._prepared_fp4_moe_by_dtype
              and inst._prepared_fp4_moe_by_dtype[torch.bfloat16].w4a16
              is inst._native_prepared)
        prep_obj = inst._prepared_fp4_moe_by_dtype[torch.bfloat16]
        check("T1b/prepared-contract",
              prep_obj.source_format == "fp4_e8m0_k32"
              and prep_obj.w13_layout == "w13"
              and prep_obj.w4a16.weight_layout == "modelopt")
        check("T1b/prewarm-kwargs",
              calls[1][1].get("num_experts") == E
              and calls[1][1].get("topk") == TOPK
              and calls[1][1].get("max_tokens") == 4096,
              str(calls[1][1]))
        check("T1b/payload-survives-as-plugin-refs",
              inst._w13.data_ptr() == w13.data_ptr()
              and inst._native_prepared.w13.data_ptr() == w13.data_ptr())

        # off 路径（native=False, mode=1）应走 super()（此处以 super 调用被记录）
        super_called = []
        inst2 = make_instance(bprime.W4A4B12xExperts, 1, {"native": False})
        bprime.B12xExperts_orig = bprime.B12xExperts

        class FakeSuper:
            def process_weights_after_loading(self, l):
                super_called.append(True)

        # 直接替换 MRO 上父类方法不可行 ⇒ 用未绑定方式验证分支选择:
        # mode=1+native=False 时不产生 native prepare 调用
        calls.clear()
        try:
            inst2.process_weights_after_loading(FakeLayer(*make_weights(seed=1)))
            # 若无 stub 则会进入真实 super 路径（CPU 上可能失败）—— 仅断言未走 native
        except Exception:
            pass
        check("T1b/off-path-not-native",
              all(c[0] != "prepare" for c in calls),
              f"native prepare 未被调用: {[c[0] for c in calls]}")
    finally:
        prep._make_workspace = orig_mkws
        prep.prepare_w4a16_e8m0_native_weights = real_prepare
        bprime._prewarm_b12x_route_pack = orig_prewarm
        bprime._maybe_release_cuda_cache = orig_cache


if __name__ == "__main__":
    t1_off_equivalence()
    t2_native_prepare()
    t1b_native_orchestration()
    t3_monkeypatch()
    print(f"\n===== L1 CPU RESULT: {len(PASS)} PASS / {len(FAIL)} FAIL =====")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
        sys.exit(1)
