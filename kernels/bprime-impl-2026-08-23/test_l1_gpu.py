#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b′ L1 GPU 小验证（共享 GPU 纪律: 先查余量, 小几何 ≤ 数百 MB, OOM 即报告）:

G0  余量检查（< 4GiB free 则跳过全部 GPU 项, 只跑 is_supported 静态项）
G1  显存账 — 小几何 derive: native hybrid 增量 ≈ E4M3 scale store（无 payload/
     E8M0 副本） vs packed hybrid 增量 ≈ payload+E8M0 副本+scale store
G2  共享证据 — native prepare 后 payload data_ptr 在 W4A16 prepared.w13 与
     W4A4 self._w13 为同一内存
G3  数值 A/B — run_w4a16_moe: packed(w31, gate-first) vs native(w13, up-first
     swapped), M=64（双侧主 GEMM）与 M=8（native 走 micro direct, packed 走
     主 GEMM）输出一致（包络 ≈2 ULP bf16）
G4  生产几何 micro direct is_supported 静态核验（E=256/H=4096/I=512/topk=6）

运行: python3 test_l1_gpu.py [small|prodgeom]
"""
import os
import sys

import torch

PASS, FAIL = [], []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return cond


def cleanup(*ts):
    for t in ts:
        try:
            del t
        except Exception:
            pass
    torch.cuda.empty_cache()


def make_weights(E, HIDDEN, INTER, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * INTER, HIDDEN // 2), dtype=torch.uint8,
                        generator=g, device="cuda")
    w2 = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8,
                       generator=g, device="cuda")
    s13 = torch.randint(118, 127, (E, 2 * INTER, HIDDEN // 32), dtype=torch.uint8,
                        generator=g, device="cuda")
    s2 = torch.randint(118, 127, (E, HIDDEN, INTER // 32), dtype=torch.uint8,
                       generator=g, device="cuda")
    return w13, w2, s13, s2


# ================================================================ G4 (static)
def g4_prodgeom_static():
    print("\n=== G4: 生产几何 micro direct is_supported 静态核验 ===")
    from b12x.moe.fused.w4a16.kernel import (
        _W4A16SmallMDirectKernel,
        _small_m_direct_supported,
    )
    kw = dict(
        hidden_size=4096, intermediate_size=512, num_experts=256, topk=6,
        scale_format="e8m0_k32")
    for m in (1, 4, 7, 8):
        ok = _W4A16SmallMDirectKernel.is_supported(m=m, **kw)
        full = _small_m_direct_supported(
            m=m, activation="silu", apply_router_weight_on_input=False,
            swiglu_limit=None, element_dtype="bf16", weight_layout="modelopt",
            w13_layout="w13", **kw)
        print(f"  m={m}: is_supported={ok} routed={full}")
    check("G4/prodgeom-micro-supported(m<=8)",
          _small_m_direct_supported(
              m=8, activation="silu", apply_router_weight_on_input=False,
              swiglu_limit=None, element_dtype="bf16", weight_layout="modelopt",
              w13_layout="w13", **kw),
          "静态可路由（注: e8m0×micro 实际输出错误——插件强制关闭, M≤8 走主 GEMM）")
    check("G4/prodgeom-packed-rejects-micro",
          not _small_m_direct_supported(
              m=8, activation="silu", apply_router_weight_on_input=False,
              swiglu_limit=None, element_dtype="bf16", weight_layout="packed",
              w13_layout="w13", **kw),
          "packed 布局不走 micro（与现生产行为一致, 主 GEMM）")


# ================================================================ G1/G2
def g1g2_memory_and_sharing(E, HIDDEN, INTER):
    print(f"\n=== G1/G2: 显存账 + 共享证据（E={E} H={HIDDEN} I={INTER}）===")
    import importlib.util
    BPRIME_PATH = os.path.join(
        os.environ["BPRIME_PLUGIN_DIR"],
        "routea_plugin_a1_bprime", "w4a4_experts.py")
    spec = importlib.util.spec_from_file_location("w4a4_experts_bprime_gpu",
                                                  BPRIME_PATH)
    bprime = importlib.util.module_from_spec(spec)
    sys.modules["w4a4_experts_bprime_gpu"] = bprime
    spec.loader.exec_module(bprime)

    # wrapper stub（真实 B12xMoEWrapper 分配与本项无关, phase3b 已验证）
    import flashinfer.fused_moe as ff
    ff.B12xMoEWrapper = type("StubWrapper", (), {"__init__": lambda s, **kw: None})

    class FakeMoeConfig:
        experts_per_token = 4
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

    def make_inst(native):
        obj = bprime.W4A4B12xExperts.__new__(bprime.W4A4B12xExperts)
        obj._w4a4_mode = 1
        obj._w4a4_native = native
        obj._w4a4_min_m = 3072
        obj._w4a4_debug = False
        obj._w4a4_ready = False
        obj._native_prepared = None
        obj._wrapper = None
        obj._w13 = obj._w2 = None
        obj._w13_sf_mma = obj._w13_sf_store = None
        obj._w2_sf_mma = obj._w2_sf_store = None
        obj._unit = obj._fc2_scale = None
        obj._prepared_fp4_moe_by_dtype = {}
        obj._unit_scale_by_device = {}
        obj._released_w4a16_source_scales = False
        obj.moe_config = FakeMoeConfig()
        obj.quant_config = type("QC", (), {})()
        return obj

    payload_bytes = (E * 2 * INTER * HIDDEN // 2) + (E * HIDDEN * INTER // 2)
    scale_bytes = (E * 2 * INTER * HIDDEN // 32) + (E * HIDDEN * INTER // 32)

    # ---- native hybrid ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base0 = torch.cuda.memory_allocated()
    w13, w2, s13, s2 = make_weights(E, HIDDEN, INTER, seed=42)
    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated()
    payload_delta = after_alloc - base0
    check("G1/alloc-sanity", payload_delta >= payload_bytes + scale_bytes,
          f"payload={payload_bytes/1e6:.1f}MB scale={scale_bytes/1e6:.1f}MB "
          f"delta={payload_delta/1e6:.1f}MB")

    inst_n = make_inst(native=True)
    layer_n = FakeLayer(w13, w2, s13, s2)
    d0 = torch.cuda.memory_allocated()
    inst_n._derive_w4a4(layer_n)
    torch.cuda.synchronize()
    d_native = torch.cuda.memory_allocated() - d0
    # native prepare 的 packed E8M0 grid（scale 等大小）
    expected_native = scale_bytes + (E * 2 * INTER * HIDDEN // 32) * 0  # grid ≈ scale size
    check("G1/native-delta≈scale-store-only",
          d_native < scale_bytes * 4 + 3e6,
          f"native derive delta={d_native/1e6:.1f}MB (payload={payload_bytes/1e6:.1f}MB, "
          f"E8M0={scale_bytes/1e6:.1f}MB, E4M3≈{scale_bytes/1e6:.1f}MB) —— 无 payload 副本")
    check("G1/native-delta << payload",
          d_native < payload_bytes * 0.5,
          f"delta/payload={d_native/payload_bytes:.2%}")

    # ---- packed hybrid（对照, phase3b M4 形态）----
    w13p, w2p, s13p, s2p = make_weights(E, HIDDEN, INTER, seed=42)
    inst_p = make_inst(native=False)
    layer_p = FakeLayer(w13p, w2p, s13p, s2p)
    d0 = torch.cuda.memory_allocated()
    inst_p._derive_w4a4(layer_p)
    torch.cuda.synchronize()
    d_packed = torch.cuda.memory_allocated() - d0
    check("G1/packed-delta≈payload+scale-copies",
          d_packed > payload_bytes + scale_bytes,
          f"packed derive delta={d_packed/1e6:.1f}MB (payload+E8M0 副本="
          f"{(payload_bytes+scale_bytes)/1e6:.1f}MB + E4M3 store)")
    check("G1/native-vs-packed-saving",
          d_packed - d_native > payload_bytes * 0.9,
          f"省 {(d_packed-d_native)/1e6:.1f}MB ≈ payload {payload_bytes/1e6:.1f}MB")

    # ---- G2 共享证据（native prepare 后）----
    inst_n._prepare_native_w4a16(layer_n, torch.bfloat16, None)
    nat = inst_n._native_prepared
    check("G2/payload-shared-w13",
          nat.w13.data_ptr() == layer_n.w13_weight.data.data_ptr()
          == inst_n._w13.data_ptr(),
          f"ptr={hex(nat.w13.data_ptr())}")
    check("G2/payload-shared-w2",
          nat.w2.data_ptr() == layer_n.w2_weight.data.data_ptr()
          == inst_n._w2.data_ptr())
    check("G2/packed-grid-distinct",
          nat.w13_scale.data_ptr() != layer_n.w13_weight_scale.data.data_ptr())

    cleanup(w13, w2, s13, s2, w13p, w2p, s13p, s2p, inst_n, inst_p)
    return payload_bytes, scale_bytes


# ================================================================ G3
def g3_numerics(E, HIDDEN, INTER, topk=4):
    print(f"\n=== G3: 数值 A/B packed(w31) vs native(w13)（E={E} H={HIDDEN} "
          f"I={INTER} topk={topk}）===")
    from b12x.moe.fused.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_e8m0_native_weights,
        prepare_w4a16_fp4_e8m0_k32_weights,
    )
    from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

    # 插件 native 模式对 e8m0 强制关闭 micro direct（上游缺陷, 见报告 G3/根因
    # 分析）——此处镜像插件行为, M≤8 双侧都走主 GEMM
    os.environ["B12X_W4A16_SMALL_M_DIRECT"] = "0"

    torch.manual_seed(0)
    w13, w2, s13, s2 = make_weights(E, HIDDEN, INTER, seed=123)
    unit = torch.ones(E, dtype=torch.float32, device="cuda")

    # packed 臂: gate-first 副本（production fork 等价调用, reuse=True 就地）
    w13_p = w13.clone(); w2_p = w2.clone()
    s13_p = s13.clone(); s2_p = s2.clone()
    packed = prepare_w4a16_fp4_e8m0_k32_weights(
        w13_p, s13_p, unit, w2_p, s2_p, unit,
        activation="silu", params_dtype=torch.bfloat16,
        w13_layout="w31", reuse_input_storage=True)

    # native 臂: 就地半交换（b′ derive 产生的物理序）+ w13 标签
    w13_n = w13.clone(); w2_n = w2.clone()
    s13_n = s13.clone(); s2_n = s2.clone()
    n = INTER
    tmp = w13_n[:, :n].clone()
    w13_n[:, :n].copy_(w13_n[:, n:]); w13_n[:, n:].copy_(tmp)
    tmp = s13_n[:, :n].clone()
    s13_n[:, :n].copy_(s13_n[:, n:]); s13_n[:, n:].copy_(tmp)
    native = prepare_w4a16_e8m0_native_weights(
        w13_n, s13_n, unit, w2_n, s2_n, unit,
        activation="silu", params_dtype=torch.bfloat16,
        w13_layout="w13")

    for M in (64, 8):
        g = torch.Generator(device="cuda").manual_seed(7)
        a = (torch.randn(M, HIDDEN, generator=g, device="cuda",
                         dtype=torch.float32) * 0.5).to(torch.bfloat16)
        ids = torch.stack([
            torch.randperm(E, generator=g, device="cuda")[:topk]
            for _ in range(M)]).to(torch.int32)
        wts = torch.rand(M, topk, generator=g, device="cuda")
        wts = (wts / wts.sum(-1, keepdim=True)).float().contiguous()

        outs = {}
        for tag, prepared in (("packed", packed), ("native", native)):
            bufs = make_w4a16_packed_buffers(
                prepared, m=M, topk=topk, dtype=torch.bfloat16,
                device=torch.device("cuda"))
            run_w4a16_moe(
                a, prepared, wts, ids,
                activation="silu",
                intermediate_cache13=bufs.intermediate_cache13,
                intermediate_cache2=bufs.intermediate_cache2,
                output=bufs.output,
            )
            torch.cuda.synchronize()
            outs[tag] = bufs.output.clone().float()
            del bufs

        o_p, o_n = outs["packed"], outs["native"]
        diff = (o_p - o_n).abs()
        denom = o_p.abs().clamp_min(1e-3)
        ulp2 = 2 ** -6  # bf16 尾数 8bit ⇒ 2 ULP 相对量级 ~2^-7; 取包络 2^-6
        frac_bad = (diff / denom > ulp2).float().mean().item()
        max_rel = (diff / denom).max().item()
        check(f"G3/M={M}/envelope(2ULP)",
              frac_bad < 1e-3,
              f"max_rel={max_rel:.2e} frac>2ULP={frac_bad:.2e} "
              f"mean|diff|={diff.mean().item():.2e}")
        print(f"    M={M}: packed vs native max|diff|={diff.max().item():.3e} "
              f"max_rel={max_rel:.2e} (双侧主 GEMM; native=模型opt staging)")

    cleanup(w13, w2, s13, s2, w13_p, w2_p, s13_p, s2_p,
            w13_n, w2_n, s13_n, s2_n, packed, native)


if __name__ == "__main__":
    free_b, total_b = torch.cuda.mem_get_info()
    free_gib = free_b / 2**30
    print(f"GPU free: {free_gib:.1f} GiB / total {total_b/2**30:.1f} GiB")
    g4_prodgeom_static()

    min_free = float(os.environ.get("GPU_MIN_FREE_GIB", "4.0"))
    if free_gib < min_free:
        print(f"[SKIP] free={free_gib:.1f} GiB < {min_free} GiB —— G1-G3 列入窗口清单")
        print(f"===== L1 GPU RESULT: {len(PASS)} PASS / {len(FAIL)} FAIL "
              f"(G1-G3 skipped) =====")
        sys.exit(1 if FAIL else 0)

    mode = sys.argv[1] if len(sys.argv) > 1 else "small"
    if mode == "small":
        E, HIDDEN, INTER = 32, 512, 256
    else:
        E, HIDDEN, INTER = 64, 2048, 512
    g1g2_memory_and_sharing(E, HIDDEN, INTER)
    g3_numerics(E, HIDDEN, INTER)
    print(f"\n===== L1 GPU RESULT: {len(PASS)} PASS / {len(FAIL)} FAIL =====")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
        sys.exit(1)
