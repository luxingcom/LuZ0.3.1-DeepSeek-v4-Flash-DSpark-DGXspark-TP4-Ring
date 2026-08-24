# ============================================================================
# ab_routeA_vs_b12x.py —— kernel① routeA(cutlass 原生FP4) vs 生产 B12X(w4a16) A/B
# 校准（2026-08-20, zip-final 复核）：
#   * 统一到 routeA（cutlass_scaled_fp4_mm_sm120a），非旧 v15(Triton bf16)
#   * 用 **RouteA 类缓存 W**（preprocess_weights 一次，do_bench 复用），
#     修复便捷入口每次调用重做 W 反量化+scaled_fp4_quant(≈7s) 导致测量失真
#   * W 用真实 NVFP4 打包([K,N//2]+[K//32,N//128])；B12X 侧反量化+fp16 matmul 作下界
# 用法：python ab_routeA_vs_b12x.py   （DGX Spark 干净 GPU，非共享）
# ============================================================================
import torch
import triton
import triton.testing

# ---- routeA（cutlass 原生 FP4，生产现役）----
from nvfp4_4w4a_mmaf import RouteA

# ---- B12X 对照（生产 MoE w4a16）：反量化 + fp16 matmul 作为可比下界 ----
def b12x_sim_gemm(A_fp16, W_packed_mx, W_scale_mx):
    M, K = A_fp16.shape
    N = W_packed_mx.shape[1] * 2
    lo = (W_packed_mx & 0x0F).float()
    hi = ((W_packed_mx >> 4) & 0x0F).float()
    vals = torch.stack([lo, hi], dim=-1).reshape(-1)
    FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                             device=A_fp16.device)
    w = FP4_TABLE[vals.long()].reshape(K, N)
    ws = W_scale_mx.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)[:K, :N]
    w = (w * ws).to(torch.float16)
    return torch.matmul(A_fp16, w.t())


def make_weights(K, N, device="cuda"):
    """同权重两种打包：NVFP4 [K,N//2]+scale[K//32,N//128] + MX [K//2,N]+scale。"""
    torch.manual_seed(0)
    w = torch.randn(K, N, device=device) * 0.1
    amax = w.abs().amax(dim=0).clamp(min=1e-9)
    # NVFP4 scale e = floor(log2(amax/6))
    exp = torch.floor(torch.log2(amax / 6.0)).clamp(-126, 127)
    w_scale = exp.unsqueeze(0).repeat(K // 32, 1).reshape(K // 32, N // 128)
    w_scale_f = torch.pow(2.0, w_scale.float())           # 已是 2^exp（未 bias，RouteA 用原始）
    w_norm = torch.clamp(w / w_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)[:K, :N], -6, 6)
    idx = ((w_norm.abs().unsqueeze(-1) - torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)).abs().argmin(-1))
    mag = idx.to(torch.uint8) | (torch.sign(w_norm) < 0).to(torch.uint8) * 8
    packed = (mag[:, 0::2] | (mag[:, 1::2] << 4)).to(torch.uint8)  # [K, N//2]
    scale_u8 = (exp + 127.0).clamp(0, 255).to(torch.uint8).contiguous()  # E8M0 带 bias
    # MX scale（per-32, K 向）
    mx_scale = torch.floor(torch.log2(amax.clamp(min=1e-30))).clamp(-126, 127)
    mx_scale_u8 = (mx_scale + 127.0).clamp(0, 255).to(torch.uint8).unsqueeze(0).repeat(K // 32, 1)
    return packed, scale_u8, mx_scale_u8


SHAPES = [(256, 4096, 2048), (512, 4096, 2048), (1024, 4096, 2048),
          (2048, 4096, 2048), (1024, 2048, 4096), (4096, 4096, 2048)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['shape'], x_vals=[str(s) for s in SHAPES],
        line_arg='impl', line_vals=['routeA_cutlass', 'b12x_w4a16_sim'],
        line_names=['routeA (cutlass FP4)', 'B12X 模拟 (w4a16)'],
        styles=[('green', '-'), ('red', '-')], ylabel='TFLOPS', plot_name='routeA_vs_b12x',
        args={}))
def bench(shape, impl):
    M, K, N = eval(shape)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    w_packed, w_scale, mx_scale = make_weights(K, N)
    if impl == 'routeA_cutlass':
        # RouteA 缓存 W（一次 preprocess，do_bench 复用）——正确测量 GEMM 本身
        ra = RouteA()
        ra.preprocess_weights(w_packed, w_scale)
        fn = lambda: ra(A, use_cached_w=True)
    else:
        fn = lambda: b12x_sim_gemm(A, w_packed, mx_scale)
    fn()
    torch.cuda.synchronize()
    t = triton.testing.do_bench(fn, warmup=25, rep=100)
    return 2 * M * N * K / t / 1e12


def main():
    print("=== kernel① routeA(cutlass 原生 FP4) vs B12X(w4a16 模拟) prefill A/B（干净 GPU）===")
    print("口径：routeA 用 RouteA 类缓存 W（GEMM-only）；B12X 用反量化+fp16 matmul（下界）")
    bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()