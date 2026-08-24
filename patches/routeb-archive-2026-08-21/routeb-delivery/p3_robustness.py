#!/usr/bin/env python3
"""p3_robustness.py — 跨层/跨 expert 稳健性抽查（生产 MXF4 真实权重）"""
import json
import struct
import sys

import torch

sys.path.insert(0, "/work")
from routeb_prod_adapter import RouteBProdGEMM  # noqa: E402

M_BASE = "/model_base"
IDX = json.load(open(f"{M_BASE}/model.safetensors.index.json"))["weight_map"]


def load_u8(name):
    shard = f"{M_BASE}/{IDX[name]}"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        info = hdr[name]
        s, e = info["data_offsets"]
        f.seek(8 + n + s)
        raw = f.read(e - s)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(info["shape"])


def main():
    torch.manual_seed(7)
    gemm = RouteBProdGEMM()
    dev = "cuda"
    cases = [
        ("layers.0.ffn.experts.7.w1", 256),
        ("layers.21.ffn.experts.5.w2", 256),
        ("layers.42.ffn.experts.200.w3", 1024),
        ("layers.0.ffn.experts.255.w1", 129),   # 又一个奇数 M
    ]
    print(f"{'tensor':>36} {'M':>5} {'rel_err':>10} 判定")
    all_ok = True
    for name, M in cases:
        w = name.rsplit(".", 1)[1]
        packed = load_u8(f"{name}.weight").to(dev)
        scale = load_u8(f"{name}.scale").to(dev)
        N, Kh = packed.shape
        K = Kh * 2
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.7
        out = gemm.gemm(A, packed, scale)
        ref = gemm.reference(A, packed, scale)
        rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        ok = rel <= 1e-2
        all_ok &= ok
        print(f"{name:>36} {M:>5} {rel:>10.3e} {'✅' if ok else '❌'}")
    print(("\n✅ 稳健性抽查全部通过" if all_ok else "\n❌ 存在失败"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
