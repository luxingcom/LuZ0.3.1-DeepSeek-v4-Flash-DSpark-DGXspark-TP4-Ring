"""kernel② v12.1 复测：正确性（逐字节 vs torch ref）+ 带宽（GB/s）

环境：DGX Spark GB10 / sm_121a / torch 2.11 / triton 3.6
口径：每 token 读 K/V fp32 1024×4B + 写 584B 信封 ≈ 4680 B/token
基线（round6）：v12.1 大 T ~18.9 GB/s（7%）｜v11 58.4 GB/s（21%）｜理论 273 GB/s
"""
import sys
import torch
import triton

sys.path.insert(0, ".")

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_v12_triton import nvfp4_ds_mla_kv_linear as triton_impl

DEVICE = "cuda"
BYTES_PER_TOKEN = 1024 * 4 + 584  # K/V read + 584B envelope write


def main():
    print("=" * 90)
    print("KERNEL2 v12.1 CORRECTNESS  (byte-exact vs torch ref)")
    print("=" * 90)
    TS = [1, 4, 16, 64, 256, 1024, 4096]
    pass_n = 0
    for T in TS:
        total = 1
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        ref = ref_impl(kv)
        out = triton_impl(kv)
        ok = (out.shape == (T, 584)) and (out.dtype == torch.uint8) and torch.equal(out, ref)
        pass_n += ok
        print(f"[{'PASS' if ok else 'FAIL'}] T={T:5d} shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"CORRECTNESS: {pass_n}/{len(TS)} passed\n")

    print("=" * 90)
    print(f"KERNEL2 v12.1 PERF  (GB/s, bytes/token={BYTES_PER_TOKEN})")
    print("=" * 90)
    SHAPES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    print(f"{'T':>6} | {'time_ms':>9} {'GB/s':>8}")
    for T in SHAPES:
        torch.manual_seed(0)
        kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5
        _ = triton_impl(kv)
        torch.cuda.synchronize()
        t = triton.testing.do_bench(lambda: triton_impl(kv), warmup=25, rep=100)
        gb = (T * BYTES_PER_TOKEN) / (t * 1e-3) / 1e9
        print(f"{T:>6} | {t * 1e3:9.2f} {gb:8.1f}")


if __name__ == "__main__":
    main()
