"""kernel② v17 vs v11 带宽对照（4680B/token）+ v17 配置归因。
"""
import torch
import time

from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as v11_impl
from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl

DEVICE = "cuda"
BIG_TS = [256, 1024, 4096, 16384, 65536]


def bench(fn, iters=30):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("=" * 60)
    print("kernel② v17 vs v11 GB/s（4680B/token，理论 273）")
    print("=" * 60)
    print(f"{'T':>7} | {'v11':>8} | {'v17':>8} | v17/v11")
    print("-" * 42)
    for T in BIG_TS:
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        BYTES = T * 4680
        v11_impl(kv)
        t11 = bench(lambda: v11_impl(kv))
        v17_impl(kv)
        t17 = bench(lambda: v17_impl(kv))
        g11, g17 = BYTES / t11 / 1e9, BYTES / t17 / 1e9
        print(f"{T:>7d} | {g11:>6.1f} | {g17:>6.1f} | {g17/g11:>5.2f}x")

    # v17 配置归因（大 T 用 .fn 固定配置）
    print("\n--- v17 配置归因 T=65536（.fn 固定 BLOCK_G/TPP）---")
    from nvfp4_ds_mla_kv_linear_v17_triton import _nvfp4_kv_linear_v17_kernel as k17
    T = 65536
    torch.manual_seed(0)
    kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
    out = torch.zeros(T, 584, dtype=torch.uint8, device=DEVICE)
    BYTES = T * 4680
    for bg, tpp, nw in [(32, 1, 1), (32, 2, 2), (32, 4, 4), (16, 4, 4), (16, 8, 4), (8, 8, 2)]:
        def run(bg=bg, tpp=tpp, nw=nw):
            grid = (triton.cdiv(T, tpp) * (64 // bg),)
            k17.fn[grid](kv, out, T, kv.stride(0), out.stride(0),
                         BLOCK_G=bg, TOKENS_PER_PROG=tpp, num_warps=nw, num_stages=3)
        import triton
        t = bench(run, iters=20)
        print(f"BLOCK_G={bg:>2} TPP={tpp} warps={nw} | {t*1e3:8.3f} ms | {BYTES/t/1e9:6.1f} GB/s")


if __name__ == "__main__":
    main()
