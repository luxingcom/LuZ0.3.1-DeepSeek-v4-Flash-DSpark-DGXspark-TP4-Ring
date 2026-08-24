#!/usr/bin/env python3
"""P4 性能 A/B 全矩阵：routeB（CUTLASS DSL MXF4 blockscaled）vs routeA（vLLM cutlass_scaled_fp4_mm）。

口径：
- kernel-only：纯 GEMM 时间（cuda event 批内计时，warm-L2，多轮取中位）
- 端到端：含 A 量化（routeB = v17 triton 量化+打包+SF swizzle；routeA = scaled_fp4_quant）
- 判据：routeB ≥1.5× routeA（同 shape）且大 shape ≥350 TFLOPS

Stage:
  ENV      环境信息
  SWEEP    routeB tile/epi 选优（每 shape 配置）
  MATRIX   生产 MoE shapes × M 扫描（w1/w3: N=2048,K=4096；w2: N=4096,K=2048）
  BIG      大 shape 回归（4096×14336×4096 ×3、8192×14336×4096、14336 相关 dense）
  SMALL_M  M=64/128/256 端到端延迟 + 分解（M 阈值分派依据）
  DUMP     results.json + markdown 表

输出：/work/p4/results.json（结构化）+ stdout（人读表）
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cutlass
import routeb_pipe as rp
pp = rp.pp

dev = "cuda"
torch.manual_seed(20260821)
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

MOE_CONFIGS = [
    ("w1", 2048, 4096),   # A[M,4096] × W[4096→2048]
    ("w3", 2048, 4096),   # 同 w1（重复测量，验证方差）
    ("w2", 4096, 2048),   # A[M,2048] × W[2048→4096]
]
M_SWEEP = [256, 512, 1024, 2048, 4096, 8192, 16384]
BIG_SHAPES = [
    (4096, 14336, 4096),   # 主回归 shape（368.1 基线）
    (8192, 14336, 4096),
    (4096, 4096, 14336),   # 14336 相关 dense 投影（补充）
    (8192, 4096, 14336),
]
SMALL_M = [64, 128, 256]
TILES = [(128, 128, 128), (128, 128, 256)]
EPIS = [(128, 128), (64, 32)]

# 冒烟模式：P4_SMOKE=1 缩减矩阵（全链路验证用）
if os.environ.get("P4_SMOKE") == "1":
    M_SWEEP = [256, 4096]
    MOE_CONFIGS = [("w1", 2048, 4096), ("w2", 4096, 2048)]
    BIG_SHAPES = [(4096, 14336, 4096)]
    SMALL_M = [64, 256]
    TILES = [(128, 128, 128)]
    EPIS = [(128, 128)]
    print("[SMOKE MODE]", flush=True)

R = {"env": {}, "sweep": [], "matrix": [], "big": [], "small_m": [], "notes": []}


def log(msg):
    print(msg, flush=True)


def note(msg):
    R["notes"].append(msg)
    log(f"  [note] {msg}")


# ---------------------------------------------------------------------------
# routeA 侧
# ---------------------------------------------------------------------------
sys.path.insert(0, "<INSTALL_DIR>/nvfp4/kernel1")
from nvfp4_4w4a_mmaf import RouteA  # noqa: E402
import vllm._custom_ops as _co  # noqa: E402


class RouteABench:
    """routeA 稳态：W 预处理一次（RouteA 类缓存），量化/GEMM 计时。"""

    def __init__(self, K, N):
        self.K, self.N = K, N
        W_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device=dev)
        W_scale = torch.randint(117, 138, (K // 32, N // 128), device=dev).to(torch.uint8)
        t0 = time.time()
        self.ra = RouteA()
        self.ra.preprocess_weights(W_packed, W_scale)
        self.prep_s = time.time() - t0
        # 稳态资源预建（生产会缓存；对 routeA 有利方向，保守判定我方门槛）
        self.gs = torch.tensor([1.0], dtype=torch.float32, device=dev)
        self.alpha = torch.tensor([1.0], dtype=torch.float32, device=dev)
        self.pad = None  # 输出 N padding 差值

    def quant(self, A16):
        return _co.scaled_fp4_quant(A16, self.gs, True, "none", None)

    def mm(self, a_q, a_sf):
        out = _co.cutlass_scaled_fp4_mm(a_q, self.ra._wq, a_sf, self.ra._wsf,
                                        self.alpha, torch.bfloat16)
        if out.shape[-1] != self.N:
            out = out[..., : self.N]
        return out

    def e2e(self, A16):
        a_q, a_sf = self.quant(A16)
        return self.mm(a_q, a_sf)


# ---------------------------------------------------------------------------
# Stage ENV
# ---------------------------------------------------------------------------
log("========== Stage ENV ==========")
R["env"] = {
    "cutlass": str(getattr(cutlass, "version", None) or cutlass.__version__),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "vllm": __import__("vllm").__version__,
    "gpu": torch.cuda.get_device_name(0),
    "cc": list(torch.cuda.get_device_capability(0)),
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
}
log(json.dumps(R["env"], indent=2))

routeA_benches = {}

# ---------------------------------------------------------------------------
# Stage SWEEP：routeB tile/epi 选优
# ---------------------------------------------------------------------------
log("\n========== Stage SWEEP（routeB tile/epi 选优） ==========")
SWEEP_CONFIGS = [("w1_w3", 2048, 4096), ("w2", 4096, 2048),
                 ("bigN", 14336, 4096), ("bigK", 4096, 14336)]
SWEEP_M = {"w1_w3": [256, 1024, 4096], "w2": [256, 1024, 4096],
           "bigN": [4096], "bigK": [4096]}
best_combo = {}   # cfg_name -> (tile, epi)
compiled_holder = {}  # cfg_key -> compiled object

for cfg_name, N, K in SWEEP_CONFIGS:
    log(f"\n--- sweep {cfg_name}: N={N}, K={K} ---")
    Wt = (torch.randn(N, K, device=dev) * 0.3)
    results = []
    for tile in TILES:
        for epi in EPIS:
            key = f"{cfg_name}_t{tile[0]}x{tile[1]}x{tile[2]}_e{epi[0]}x{epi[1]}"
            try:
                g0 = rp.RouteBGemm(SWEEP_M[cfg_name][0], N, K, tile, epi)
                compiled_holder[key] = (g0, tile, epi, N, K, Wt)
                g0.set_W(Wt)
                for M in SWEEP_M[cfg_name]:
                    gM = rp.RouteBGemm(M, N, K, tile, epi, _compile=False)
                    gM.compiled = g0.compiled
                    A16 = (torch.randn(M, K, device=dev) * 0.5).half()
                    gM.set_A(A16)
                    ms = rp.time_ms(gM.run, warmup=10, iters=50, rounds=3)
                    tf = rp.tflops(M, N, K, ms)
                    results.append({"cfg": cfg_name, "M": M, "tile": list(tile),
                                    "epi": list(epi), "ko_ms": ms, "ko_tflops": tf})
                    log(f"  {key} M={M}: {ms*1e3:8.1f} us  {tf:7.1f} TFLOPS")
                    del gM
                del g0
                torch.cuda.empty_cache()
            except Exception as ex:
                log(f"  {key}: FAILED {repr(ex)[:120]}")
                results.append({"cfg": cfg_name, "tile": list(tile), "epi": list(epi),
                                "error": repr(ex)[:200]})
    R["sweep"].extend(results)
    # 以最大 sweep M 的 TFLOPS 排序选优
    rank_m = max(SWEEP_M[cfg_name])
    cands = [r for r in results if r.get("M") == rank_m and "ko_tflops" in r]
    if cands:
        best = max(cands, key=lambda r: r["ko_tflops"])
        best_combo[cfg_name] = (tuple(best["tile"]), tuple(best["epi"]))
        log(f"  ★ {cfg_name} 最优: tile {best['tile']} epi {best['epi']} "
            f"@M={rank_m} {best['ko_tflops']:.1f} TFLOPS")

CFG2BEST = {"w1": best_combo.get("w1_w3"), "w3": best_combo.get("w1_w3"),
            "w2": best_combo.get("w2")}
BIG2BEST = {(4096, 14336, 4096): best_combo.get("bigN"),
            (8192, 14336, 4096): best_combo.get("bigN"),
            (4096, 4096, 14336): best_combo.get("bigK"),
            (8192, 4096, 14336): best_combo.get("bigK")}

# ---------------------------------------------------------------------------
# Stage MATRIX：MoE shapes × M 扫描（A/B 双口径）
# ---------------------------------------------------------------------------
log("\n========== Stage MATRIX（MoE shapes × M 扫描） ==========")

for cfg_name, N, K in MOE_CONFIGS:
    tile, epi = CFG2BEST[cfg_name]
    log(f"\n--- {cfg_name}: N={N}, K={K}, tile {tile}, epi {epi} ---")
    if (N, K) not in routeA_benches:
        routeA_benches[(N, K)] = RouteABench(K, N)
    rab = routeA_benches[(N, K)]
    log(f"  routeA W 预处理: {rab.prep_s:.2f}s")

    g0 = rp.RouteBGemm(M_SWEEP[0], N, K, tile, epi)
    Wt = (torch.randn(N, K, device=dev) * 0.3)
    wq, ws = g0.set_W(Wt)
    checked = False

    for M in M_SWEEP:
        try:
            g = rp.RouteBGemm(M, N, K, tile, epi, _compile=False)
            g.compiled = g0.compiled
            g.set_W_prepacked(wq, ws)  # [fix] 每 M 实例需写入 W（否则 B 未初始化）
            A16 = (torch.randn(M, K, device=dev) * 0.5).half()
            aq, asf = g.set_A(A16)

            rel = rel_fused = None
            if not checked and M <= 4096:
                ref = rp.dequant_ref(aq, asf, wq, ws)
                g.run(); torch.cuda.synchronize()
                rel = ((g.out().float() - ref).abs().max() / ref.abs().max()).item()
                # 融合量化路径（E2E 计时所用）单独校验
                g.e2e_call(A16); torch.cuda.synchronize()
                rel_fused = ((g.out().float() - ref).abs().max() / ref.abs().max()).item()
                checked = True
                if rel > 0.02:
                    note(f"{cfg_name} M={M} 数值校验 rel_err={rel:.4f} 超 0.02！")
                if rel_fused > 0.02:
                    note(f"{cfg_name} M={M} 融合量化路径 rel_err={rel_fused:.4f} 超 0.02！")

            iters = 50 if M <= 8192 else 30
            rb_ko = rp.time_ms(g.run, warmup=10, iters=iters, rounds=3)
            rb_e2e = rp.time_ms(lambda: g.e2e_call(A16), warmup=10, iters=iters, rounds=3)

            a_q, a_sf = rab.quant(A16)
            ra_ko = rp.time_ms(lambda: rab.mm(a_q, a_sf), warmup=10, iters=iters, rounds=3)
            ra_e2e = rp.time_ms(lambda: rab.e2e(A16), warmup=10, iters=iters, rounds=3)

            cell = {
                "cfg": cfg_name, "M": M, "N": N, "K": K,
                "tile": list(tile), "epi": list(epi), "rel_err": rel,
                "rel_err_fused": rel_fused,
                "rb_ko_ms": rb_ko, "rb_ko_tflops": rp.tflops(M, N, K, rb_ko),
                "rb_e2e_ms": rb_e2e, "rb_e2e_tflops": rp.tflops(M, N, K, rb_e2e),
                "ra_ko_ms": ra_ko, "ra_ko_tflops": rp.tflops(M, N, K, ra_ko),
                "ra_e2e_ms": ra_e2e, "ra_e2e_tflops": rp.tflops(M, N, K, ra_e2e),
            }
            cell["sp_ko"] = cell["rb_ko_tflops"] / cell["ra_ko_tflops"]
            cell["sp_e2e"] = cell["rb_e2e_tflops"] / cell["ra_e2e_tflops"]
            cell["pass15"] = cell["sp_e2e"] >= 1.5 and cell["sp_ko"] >= 1.5
            R["matrix"].append(cell)
            log(f"  M={M:6d}  KO: B {cell['rb_ko_tflops']:6.1f} / A {cell['ra_ko_tflops']:6.1f}"
                f"  = {cell['sp_ko']:5.2f}x   E2E: B {cell['rb_e2e_tflops']:6.1f} /"
                f" A {cell['ra_e2e_tflops']:6.1f} = {cell['sp_e2e']:5.2f}x"
                f"   {'✓' if cell['pass15'] else '✗'}"
                + (f"  rel={rel:.4f}" if rel is not None else ""))
            del g
            torch.cuda.empty_cache()
        except Exception as ex:
            traceback.print_exc()
            R["matrix"].append({"cfg": cfg_name, "M": M, "N": N, "K": K,
                                "error": repr(ex)[:200]})
    del g0
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Stage BIG：大 shape 回归 + 扩展
# ---------------------------------------------------------------------------
log("\n========== Stage BIG（大 shape） ==========")

for (M, N, K) in BIG_SHAPES:
    combo = BIG2BEST.get((M, N, K))
    if not combo:
        note(f"({M},{N},{K}) 无可用 tile 组合，跳过")
        continue
    tile, epi = combo
    log(f"\n--- ({M},{N},{K}) tile {tile} epi {epi} ---")
    try:
        if (N, K) not in routeA_benches:
            routeA_benches[(N, K)] = RouteABench(K, N)
        rab = routeA_benches[(N, K)]

        g = rp.RouteBGemm(M, N, K, tile, epi)
        Wt = (torch.randn(N, K, device=dev) * 0.3)
        wq, ws = g.set_W(Wt)
        A16 = (torch.randn(M, K, device=dev) * 0.5).half()
        aq, asf = g.set_A(A16)
        if M <= 4096:
            g.run(); torch.cuda.synchronize()
            ref = rp.dequant_ref(aq, asf, wq, ws)
            rel = ((g.out().float() - ref).abs().max() / ref.abs().max()).item()
        else:
            rel = None
        iters = 20
        rb_ko = rp.time_ms(g.run, warmup=10, iters=iters, rounds=5)
        rb_e2e = rp.time_ms(lambda: g.e2e_call(A16), warmup=10, iters=iters, rounds=3)
        a_q, a_sf = rab.quant(A16)
        ra_ko = rp.time_ms(lambda: rab.mm(a_q, a_sf), warmup=10, iters=iters, rounds=5)
        ra_e2e = rp.time_ms(lambda: rab.e2e(A16), warmup=10, iters=iters, rounds=3)
        cell = {
            "M": M, "N": N, "K": K, "tile": list(tile), "epi": list(epi), "rel_err": rel,
            "rb_ko_ms": rb_ko, "rb_ko_tflops": rp.tflops(M, N, K, rb_ko),
            "rb_e2e_ms": rb_e2e, "rb_e2e_tflops": rp.tflops(M, N, K, rb_e2e),
            "ra_ko_ms": ra_ko, "ra_ko_tflops": rp.tflops(M, N, K, ra_ko),
            "ra_e2e_ms": ra_e2e, "ra_e2e_tflops": rp.tflops(M, N, K, ra_e2e),
        }
        cell["sp_ko"] = cell["rb_ko_tflops"] / cell["ra_ko_tflops"]
        cell["sp_e2e"] = cell["rb_e2e_tflops"] / cell["ra_e2e_tflops"]
        cell["pass350"] = cell["rb_ko_tflops"] >= 350.0
        R["big"].append(cell)
        log(f"  KO: B {cell['rb_ko_tflops']:6.1f} / A {cell['ra_ko_tflops']:6.1f}"
            f" = {cell['sp_ko']:5.2f}x   E2E: B {cell['rb_e2e_tflops']:6.1f} /"
            f" A {cell['ra_e2e_tflops']:6.1f} = {cell['sp_e2e']:5.2f}x"
            f"   ≥350: {'✓' if cell['pass350'] else '✗'}"
            + (f"  rel={rel:.4f}" if rel is not None else ""))
        del g
        torch.cuda.empty_cache()
    except Exception as ex:
        traceback.print_exc()
        R["big"].append({"M": M, "N": N, "K": K, "error": repr(ex)[:200]})

# 主 shape：run_bs 官方口径 3 次取中位（368.1 复现口径）
log("\n--- 主 shape run_bs 官方口径 ×3（368.1 复现） ---")
try:
    tile, epi = BIG2BEST[(4096, 14336, 4096)]
    vals = []
    for i in range(3):
        t_us = pp.run_bs((4096, 14336, 4096, 1), cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
                         cutlass.Float8E8M0FNU, 32, cutlass.Float16, cutlass.Float32,
                         "k", "k", "n", tile, epi, 1e-1, 5, 10, True)
        tf = rp.tflops(4096, 14336, 4096, t_us / 1e3)
        vals.append(tf)
        log(f"  run_bs round {i+1}: {t_us:.1f} us -> {tf:.1f} TFLOPS")
    vals.sort()
    R["big_regression_runbs"] = {"median_tflops": vals[1], "rounds": vals}
    log(f"  run_bs 中位: {vals[1]:.1f} TFLOPS")
except Exception as ex:
    traceback.print_exc()
    R["big_regression_runbs"] = {"error": repr(ex)[:200]}

# ---------------------------------------------------------------------------
# Stage SMALL_M：M=64/128/256 端到端延迟 + 分解
# ---------------------------------------------------------------------------
log("\n========== Stage SMALL_M（M 阈值分派依据） ==========")

for cfg_name, N, K in MOE_CONFIGS:
    tile, epi = CFG2BEST[cfg_name]
    log(f"\n--- {cfg_name}: N={N}, K={K} ---")
    rab = routeA_benches[(N, K)]
    g0 = rp.RouteBGemm(256, N, K, tile, epi)
    Wt = (torch.randn(N, K, device=dev) * 0.3)
    wq, ws = g0.set_W(Wt)
    for M in SMALL_M:
        try:
            g = rp.RouteBGemm(M, N, K, tile, epi, _compile=False)
            g.compiled = g0.compiled
            g.set_W_prepacked(wq, ws)  # [fix] 每 M 实例需写入 W
            A16 = (torch.randn(M, K, device=dev) * 0.5).half()
            g.set_A(A16)

            def rb_quant_only():
                rp.triton_a_quant(A16, g._aq, g._asf)
                g._pack_a(g._aq)
                rp.sf_scatter(g._asf, g.sfa_u8, g._sf_idx)

            rb_q = rp.time_ms(rb_quant_only, warmup=10, iters=100, rounds=3)
            rb_qf = rp.time_ms(lambda: g.quant_2pass(A16), warmup=10, iters=100, rounds=3)
            rb_k = rp.time_ms(g.run, warmup=10, iters=100, rounds=3)
            rb_e2e = rp.time_ms(lambda: g.e2e_call(A16), warmup=10, iters=100, rounds=3)
            rb_e2e_u = rp.time_ms(lambda: g.e2e_call_unfused(A16), warmup=10, iters=100, rounds=3)

            ra_q = rp.time_ms(lambda: rab.quant(A16), warmup=10, iters=100, rounds=3)
            a_q, a_sf = rab.quant(A16)
            ra_k = rp.time_ms(lambda: rab.mm(a_q, a_sf), warmup=10, iters=100, rounds=3)
            ra_e2e = rp.time_ms(lambda: rab.e2e(A16), warmup=10, iters=100, rounds=3)

            cell = {
                "cfg": cfg_name, "M": M, "N": N, "K": K,
                "rb_quant_us": rb_q * 1e3, "rb_quant_2pass_us": rb_qf * 1e3,
                "rb_kernel_us": rb_k * 1e3,
                "rb_e2e_us": rb_e2e * 1e3, "rb_e2e_unfused_us": rb_e2e_u * 1e3,
                "ra_quant_us": ra_q * 1e3, "ra_kernel_us": ra_k * 1e3,
                "ra_e2e_us": ra_e2e * 1e3,
            }
            cell["ra_e2e_tflops"] = rp.tflops(M, N, K, ra_e2e)
            cell["rb_e2e_tflops"] = rp.tflops(M, N, K, rb_e2e)
            cell["sp_e2e"] = cell["rb_e2e_us"] and ra_e2e / rb_e2e or 0
            R["small_m"].append(cell)
            log(f"  M={M:4d}  B: quant2p {rb_qf*1e3:6.1f} (unfused {rb_q*1e3:6.1f})"
                f" + gemm {rb_k*1e3:6.1f} = e2e {rb_e2e*1e3:6.1f} us ({cell['rb_e2e_tflops']:5.1f} TF,"
                f" unfused e2e {rb_e2e_u*1e3:6.1f})   "
                f"A: quant {ra_q*1e3:6.1f} + gemm {ra_k*1e3:6.1f} = e2e {ra_e2e*1e3:6.1f} us"
                f" ({cell['ra_e2e_tflops']:5.1f} TF)   B/A 延迟比 {ra_e2e/rb_e2e:5.2f}x")
            del g
            torch.cuda.empty_cache()
        except Exception as ex:
            traceback.print_exc()
            R["small_m"].append({"cfg": cfg_name, "M": M, "error": repr(ex)[:200]})
    del g0
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Stage DUMP
# ---------------------------------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
log(f"\nresults.json 已写入 {OUT_JSON}")
log("P4_MATRIX_DONE")
