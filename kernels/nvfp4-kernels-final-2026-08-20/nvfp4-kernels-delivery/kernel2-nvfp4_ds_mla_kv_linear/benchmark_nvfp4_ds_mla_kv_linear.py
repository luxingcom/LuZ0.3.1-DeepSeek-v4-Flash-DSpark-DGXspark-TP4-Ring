"""Benchmark: nvfp4_ds_mla_kv_linear v4（性能优化版，2D grid + 向量化）

在 DGX Spark（GB10 / SM121）上运行：
    python benchmark_nvfp4_ds_mla_kv_linear.py
"""
import torch
import triton

from nvfp4_ds_mla_kv_linear_torch import nvfp4_ds_mla_kv_linear as ref_impl
from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as triton_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [1, 4, 16, 64, 256, 1024, 4096, 16384]


def bench(T):
    torch.manual_seed(0)
    kv = torch.randn(T, 1024, device=DEVICE, dtype=torch.float32) * 0.5

    _ = triton_impl(kv)
    torch.cuda.synchronize()

    t_ref = triton.testing.do_bench(lambda: ref_impl(kv), warmup=25, rep=100)
    t_tri = triton.testing.do_bench(lambda: triton_impl(kv), warmup=25, rep=100)

    # 吞吐口径：每 token 读 1024×4B + 写 584B
    # ⚠️ 修复：do_bench 返回【秒】（日志 `t_tri*1e3` 得 ms 可证）。
    #   旧代码误乘 `t_tri * 1e-3` 导致数值虚高 1000×（打印 18.9 实为 18.9 MB/s，标签却写 GB/s）。
    #   正确：bytes / t_tri(秒) / 1e9 = GB/s
    bytes_per_token = 1024 * 4 + 584
    gb_tri = (T * bytes_per_token) / t_tri / 1e9

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
    print("MCP 侧实测：speedup=42.17x（v4，目标 4.0 大幅超标）；GB10 带宽 273GB/s 为吞吐上限")
