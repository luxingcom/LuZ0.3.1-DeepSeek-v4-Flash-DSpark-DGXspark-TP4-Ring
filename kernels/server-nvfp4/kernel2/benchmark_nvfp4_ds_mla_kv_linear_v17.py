# ============================================================================
# benchmark_nvfp4_ds_mla_kv_linear_v17.py —— v17 带宽基准（修正口径 GB/s）
# 主指标：GB/s（读 4KB + 写 584B per token）；对比 v11（53.4 带宽冠军）
# ============================================================================
import torch
import triton
import triton.testing

from nvfp4_ds_mla_kv_linear_triton import nvfp4_ds_mla_kv_linear as v11_impl
from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear as v17_impl

T_SIZES = [256, 1024, 4096, 16384, 65536]
BYTES_PER_TOKEN = 1024 * 4 + 584  # 读 4KB fp32 + 写 584B 信封


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=T_SIZES,
        line_arg='impl',
        line_vals=['v11', 'v17'],
        line_names=['v11 (带宽冠军)', 'v17 (多token+向量化)'],
        styles=[('red', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='kv_linear_bw',
        args={},
    )
)
def bench(T, impl):
    torch.manual_seed(0)
    kv = torch.randn(T, 1024, device='cuda', dtype=torch.float32)
    fn = v11_impl if impl == 'v11' else v17_impl
    # warmup（含 autotune 首调）
    fn(kv)
    torch.cuda.synchronize()
    t = triton.testing.do_bench(lambda: fn(kv), warmup=25, rep=100)
    gbps = (T * BYTES_PER_TOKEN) / t / 1e9  # do_bench 返回秒 → 真实 GB/s
    return gbps


def main():
    bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
