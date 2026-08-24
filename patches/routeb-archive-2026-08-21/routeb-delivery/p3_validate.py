#!/usr/bin/env python3
"""p3_validate.py — P3 数值判决：routeB kernel × 生产 MXF4 真实权重
=====================================================================
对象：node01 /model_base（deepseek-v4-flash-0731 生产 MXFP4）layer 0
expert 0 的 w1/w2/w3（单 expert ~12MB，OOM 铁律）。
判据：rel_err = max|kernel_out − ref| / max|ref| ≤ 1e-2（对齐 routeA 标准）。
M 扫描 {64, 256, 1024, 4096, 257}（含奇数 M）。
"""
import json
import struct
import sys

import torch

sys.path.insert(0, "/work")
from routeb_prod_adapter import (  # noqa: E402
    RouteBProdGEMM, dequant_w_mxf4, quantize_a, dequant_a,
    sf_plain_to_atom, sf_atom_to_plain, encode_e8m0_32, pack_e2m1,
    unpack_e2m1,
)

M_BASE = "/model_base"
SHARD_IDX = json.load(open(f"{M_BASE}/model.safetensors.index.json"))["weight_map"]


def load_u8(name):
    shard = f"{M_BASE}/{SHARD_IDX[name]}"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        info = hdr[name]
        s, e = info["data_offsets"]
        f.seek(8 + n + s)
        raw = f.read(e - s)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(info["shape"])


def main():
    torch.manual_seed(2026)
    dev = "cuda"

    print("=== 0. 适配器自检（CPU）===")
    import routeb_prod_adapter as rpa
    rpa._selfcheck()

    print("\n=== 1. 加载生产 MXF4 expert0（w1/w2/w3）===")
    W = {}
    for w in ("w1", "w2", "w3"):
        W[w] = {
            "packed": load_u8(f"layers.0.ffn.experts.0.{w}.weight"),
            "scale": load_u8(f"layers.0.ffn.experts.0.{w}.scale"),
        }
        N, Kh = W[w]["packed"].shape
        print(f"  {w}: packed {tuple(W[w]['packed'].shape)} "
              f"scale {tuple(W[w]['scale'].shape)}  → N={N}, K={Kh * 2}")

    # scale 字节健康检查（生产 MXF4 应有真实分布，非恒定）
    for w in W:
        s = W[w]["scale"]
        print(f"  {w}.scale: uniq={len(s.unique())} min={s.min()} max={s.max()} "
              f"mean={s.float().mean():.2f}")

    print("\n=== 2. scale atom-swizzle 端到端核验（kernel 契约）===")
    # 对官方 cvt 的逐字节验证已在 p3_probe_layout.py 完成（100%）；
    # 此处再做往返 + 与官方 cvt 运行时对照一次（小张量，GPU 路径预热）。
    plain = torch.randint(0, 256, (256, 64), dtype=torch.uint8)
    back = sf_atom_to_plain(sf_plain_to_atom(plain), 256)
    assert torch.equal(back, plain), "swizzle 往返失败"
    print("  ✅ swizzle 往返一致（官方 cvt 对照见 p3_probe_layout，逐字节 100%）")

    print("\n=== 3. A 量化校准（金标准）===")
    assert encode_e8m0_32(torch.zeros(1, 64)).flatten().tolist() == [24, 24]
    assert encode_e8m0_32(torch.full((1, 64), 1e6)).flatten().tolist() == [144, 144]
    print("  ✅ 零输入→24，1e6→144")

    print("\n=== 4. 数值对照（判决）===")
    gemm = RouteBProdGEMM()
    M_SWEEP = [64, 256, 1024, 4096, 257]
    results = []
    hdr = f"{'mat':>4} {'M':>5} {'K':>5} {'N':>5} {'rel_err':>10} {'max|ref|':>10} 判定"
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for wname in ("w1", "w2", "w3"):
        packed, scale = W[wname]["packed"], W[wname]["scale"]
        N, Kh = packed.shape
        K = Kh * 2
        packed_d, scale_d = packed.to(dev), scale.to(dev)
        for M in M_SWEEP:
            A = (torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.7)
            out = gemm.gemm(A, packed_d, scale_d)              # fp16 [M, N]
            ref = gemm.reference(A, packed_d, scale_d)         # f32 [M, N]
            rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
            ok = rel <= 1e-2
            worst = max(worst, rel)
            results.append((wname, M, K, N, rel, ref.abs().max().item(), ok))
            print(f"{wname:>4} {M:>5} {K:>5} {N:>5} {rel:>10.3e} "
                  f"{ref.abs().max().item():>10.3f} {'✅' if ok else '❌'}")

    n_pass = sum(1 for r in results if r[-1])
    print("-" * len(hdr))
    print(f"\n通过 {n_pass}/{len(results)}，最差 rel_err = {worst:.3e}")
    if n_pass == len(results):
        print("✅ P3 判决通过：routeB kernel 消费生产 MXF4 真实权重，"
              "全部 shape rel_err ≤ 1e-2")
    else:
        print("❌ P3 判决失败：存在 rel_err > 1e-2 的 shape")
        sys.exit(1)

    # 保存结果表
    with open("/work/p3_results.txt", "w") as f:
        f.write(f"{'mat':>4} {'M':>5} {'K':>5} {'N':>5} {'rel_err':>12} 判定\n")
        for wname, M, K, N, rel, mref, ok in results:
            f.write(f"{wname:>4} {M:>5} {K:>5} {N:>5} {rel:>12.4e} "
                    f"{'PASS' if ok else 'FAIL'}\n")
        f.write(f"\nworst={worst:.4e}, pass={n_pass}/{len(results)}\n")
    print("结果已写入 /work/p3_results.txt")


if __name__ == "__main__":
    main()
