"""kernel② kv_linear 性能基准（用户要求的"两算子性能"）：
- v11（维持部署版，round6 带宽冠军 53.4 GB/s）
- v12.1（round7 修复版 18.8 GB/s，参考）
- v15（round 弃用版 10.6，参考，已确认不采用）
统一 4680B/token 口径（读 1024*4 + 写 584）。
"""
import torch
import time

from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as v11_impl
from nvfp4_ds_mla_kv_linear_v12_triton import nvfp4_ds_mla_kv_linear as v121_impl
from nvfp4_ds_mla_kv_linear_v15_triton import nvfp4_ds_mla_kv_linear as v15_impl

DEVICE = "cuda"
BIG_TS = [256, 1024, 4096, 16384, 65536]


def bench(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("=" * 66)
    print("kernel② kv_linear GB/s（4680B/token 口径，理论上限 273 GB/s）")
    print("=" * 66)
    print(f"{'T':>7} | {'v11(维持)':>10} | {'v12.1(参考)':>10} | {'v15(弃用)':>10}")
    print("-" * 50)
    for T in BIG_TS:
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        BYTES = T * 4680
        row = f"{T:>7d} |"
        for name, impl in [("v11", v11_impl), ("v121", v121_impl), ("v15", v15_impl)]:
            try:
                impl(kv)  # warm + autotune
                t = bench(lambda: impl(kv))
                row += f" {BYTES/t/1e9:>8.1f} |"
            except Exception as e:
                row += f" {'ERR':>8} |"
        print(row)


if __name__ == "__main__":
    main()
