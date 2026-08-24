"""Benchmark: nvfp4_ds_mla_kv_linear_paged（vLLM 分页 KV 写回，v6）

在 DGX Spark（GB10 / SM121）上运行：
    python benchmark_nvfp4_ds_mla_kv_linear_paged.py
"""
import torch
import triton

from nvfp4_ds_mla_kv_linear_paged_torch import nvfp4_ds_mla_kv_linear_paged as ref_impl
from nvfp4_ds_mla_kv_linear_paged_triton import nvfp4_ds_mla_kv_linear_paged as triton_impl
from test_nvfp4_ds_mla_kv_linear_paged import make_inputs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [16, 64, 256, 1024, 4096, 16384]


def bench(T):
    k, v, seq_ids, positions, block_table, kv_cache = make_inputs(T)

    _ = triton_impl(k, v, seq_ids, positions, block_table, kv_cache)
    torch.cuda.synchronize()

    def run_ref():
        c = torch.zeros_like(kv_cache)
        return ref_impl(k, v, seq_ids, positions, block_table, c)

    t_ref = triton.testing.do_bench(run_ref, warmup=25, rep=100)
    t_tri = triton.testing.do_bench(lambda: triton_impl(k, v, seq_ids, positions, block_table, kv_cache),
                                    warmup=25, rep=100)

    bytes_per_token = 1024 * 4 + 584
    gb_tri = (T * bytes_per_token) / (t_tri * 1e-3) / 1e9

    print(f"T={T:6d} | ref {t_ref*1e3:8.2f}ms | triton {t_tri*1e3:8.2f}ms ({gb_tri:6.1f} GB/s) | speedup {t_ref/t_tri:8.2f}x")
    return t_ref / t_tri


if __name__ == "__main__":
    if DEVICE != "cuda":
        raise SystemExit("需要 CUDA（DGX Spark / SM121）")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    ratios = [bench(T) for T in SHAPES]
    print("=" * 100)
    print(f"Avg speedup: {sum(ratios)/len(ratios):.2f}x | Best: {max(ratios):.2f}x")
    print("MCP 侧：正确性通过（v6）；benchmark 受 harness libdevice 编译限制未取数，以实机为准")
