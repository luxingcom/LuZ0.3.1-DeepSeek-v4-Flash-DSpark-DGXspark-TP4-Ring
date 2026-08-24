"""kernel② v15 带宽探针：剔除 torch.zeros 分配开销 + 固定 BLOCK_G/num_warps 对比 +
v14 宽 tile 版对比 + 真实流量口径（4680B/token）跨版本对齐。"""
import torch
import time

from nvfp4_ds_mla_kv_linear_v15_triton import _nvfp4_ds_mla_kv_linear_kernel as k15
from nvfp4_ds_mla_kv_linear_v14_triton import nvfp4_ds_mla_kv_linear as v14_impl
from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as v11_impl
import triton

DEVICE = "cuda"


def timeit(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    T = 65536
    torch.manual_seed(0)
    kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
    kv_fp32 = kv.float().contiguous()
    out = torch.zeros(T, 584, dtype=torch.uint8, device=DEVICE)
    BYTES = T * 4680  # 读 1024*4 + 写 584（对齐 round6 口径）

    # 0) 纯分配/清零开销
    t_zero = timeit(lambda: torch.zeros(T, 584, dtype=torch.uint8, device=DEVICE))
    print(f"torch.zeros({T},584) 开销: {t_zero*1e3:.3f} ms")

    # 1) v15 固定配置（用 .fn 原始 JIT 绕过 autotune）
    for bg, nw in [(32, 2), (32, 4), (16, 2), (16, 4), (8, 1), (8, 2)]:
        def run_v15(bg=bg, nw=nw):
            grid = (T, 64 // bg)
            k15.fn[grid](kv_fp32, out, T=T, BLOCK_G=bg, num_warps=nw, num_stages=2)
        t = timeit(run_v15)
        gbps = BYTES / t / 1e9
        print(f"v15 BLOCK_G={bg:>2} warps={nw} | {t*1e3:8.3f} ms | {gbps:7.1f} GB/s(4680) | {(T*584)/t/1e9:6.1f} GB/s(584)")

    # 2) v15 autotune 整体（含 alloc）
    from nvfp4_ds_mla_kv_linear_v15_triton import nvfp4_ds_mla_kv_linear as v15_impl
    v15_impl(kv)
    t = timeit(lambda: v15_impl(kv), iters=20)
    print(f"v15 autotune(含alloc) | {t*1e3:8.3f} ms | {BYTES/t/1e9:7.1f} GB/s(4680)")

    # 3) v14 整体（含 alloc）——已知缺陷：模块级 _GROUP 非 constexpr，生产编译即挂
    try:
        v14_impl(kv)
        t = timeit(lambda: v14_impl(kv), iters=20)
        print(f"v14 整体(含alloc)    | {t*1e3:8.3f} ms | {BYTES/t/1e9:7.1f} GB/s(4680)")
    except Exception as e:
        print(f"v14 编译失败: {type(e).__name__}: {str(e)[:100]}")

    # 4) v11 整体（含 alloc，round6 基线 58.4 GB/s 的版本）
    v11_impl(kv)
    t = timeit(lambda: v11_impl(kv), iters=20)
    print(f"v11 整体(含alloc)    | {t*1e3:8.3f} ms | {BYTES/t/1e9:7.1f} GB/s(4680)")


if __name__ == "__main__":
    main()
