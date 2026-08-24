"""kernel② v15 忠实复测：kv[T,1024] → [T,584] 逐字节 vs torch ref + GB/s 基准。
T 档位与 shipped test 一致 [1,4,16,64,256,1024,4096]，外加大数据档位看带宽。
"""
import torch
import time

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_v15_triton import nvfp4_ds_mla_kv_linear as v15_impl

DEVICE = "cuda"
TS = [1, 4, 16, 64, 256, 1024, 4096]
BIG_TS = [256, 1024, 4096, 16384, 65536]


def main():
    print("=" * 70)
    print("kernel② v15 正确性复测（逐字节 rtol=0 atol=0）")
    print("=" * 70)
    n_pass = 0
    for T in TS:
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        ref = ref_impl(kv)
        out = v15_impl(kv)
        ok_shape = out.shape == (T, 584)
        ok_dtype = out.dtype == torch.uint8
        try:
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            byte_exact = True
        except Exception as e:
            byte_exact = False
            err = str(e).splitlines()[0] if str(e) else "?"
        if ok_shape and ok_dtype and byte_exact:
            n_pass += 1
            print(f"[PASS] T={T:>6d} shape={tuple(out.shape)} dtype={out.dtype} 逐字节一致")
        else:
            detail = "byte-exact FAIL" if not byte_exact else ("shape/dtype FAIL")
            print(f"[FAIL] T={T:>6d} {detail} | {err if not byte_exact else ''}")
    print(f"正确性: {n_pass}/{len(TS)} PASS")

    print("\n" + "=" * 70)
    print("kernel② v15 带宽基准（bytes/token=584B，理论上限 273GB/s 为参考）")
    print("=" * 70)
    print(f"{'T':>8} | {'耗时 ms':>10} | {'GB/s':>10}")
    print("-" * 42)
    for T in BIG_TS:
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        v15_impl(kv)  # autotune warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            v15_impl(kv)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        gbps = (T * 584) / dt / 1e9
        print(f"{T:>8d} | {dt*1e3:>10.3f} | {gbps:>9.1f}")


if __name__ == "__main__":
    main()
