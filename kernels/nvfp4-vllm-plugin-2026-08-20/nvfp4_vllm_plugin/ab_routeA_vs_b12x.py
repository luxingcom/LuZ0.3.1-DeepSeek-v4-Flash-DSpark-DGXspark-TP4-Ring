# ============================================================================
# ab_routeA_vs_b12x.py —— kernel① v15 vs 生产 B12X prefill 段 A/B 对照
# 目的（生产报告 §五 遗留项）：证明 routeA（4W4A）相对 B12X（w4a16）的 prefill 增量价值
# 用法：python ab_routeA_vs_b12x.py   （DGX Spark，需 torch/triton）
# ============================================================================
import os
import torch
import triton
import triton.testing

# ---- routeA（4W4A，已验证 cutlass 内核，本包） ----
from nvfp4_4w4a_mmaf import nvfp4_4w4a_prefill_gemm as routeA_gemm

# ---- B12X 对照（生产 MoE 段）：等价口径 = 同 shape 的 w4a16 反量化 + fp16 GEMM ----
# 生产 B12X 为 w4a16（权重 4-bit、激活 fp16）——此处用"反量化权重 + fp16 matmul"作为
# 可比基线（真实 B12X kernel 加速比以生产 harness 为准，本脚本给下界）
def b12x_sim_gemm(A_fp16, W_packed_mx, W_scale_mx):
    """MXFP4 w4a16 模拟：解包 → 反量化 → fp16 matmul（生产 B12X 数值等价口径）。"""
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
    """构造同输入：NVFP4 打包（v15 格式）+ MX 打包（B12X 模拟格式）。"""
    torch.manual_seed(0)
    w = torch.randn(K, N, device=device) * 0.1
    # NVFP4：N 向打包 [K, N//2]
    amax = w.abs().amax(dim=0).clamp(min=1e-9)
    exp = torch.floor(torch.log2(amax / 6.0)) + 127
    exp = exp.clamp(0, 255).to(torch.uint8)
    w_scale = exp.unsqueeze(0).repeat(K // 32, 1).reshape(K // 32, N // 128)
    w_scale_f = torch.pow(2.0, w_scale.float() - 127.0)
    w_norm = torch.clamp(w / w_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)[:K, :N], -6, 6)
    idx = ((w_norm.abs().unsqueeze(-1) - torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)).abs().argmin(-1))
    mag = idx.to(torch.uint8) | (torch.sign(w_norm) < 0).to(torch.uint8) * 8
    packed = (mag[:, 0::2] | (mag[:, 1::2] << 4)).to(torch.uint8)  # [K, N//2]
    # MX：K 向打包 [K//2, N]（B12X 风格）
    mx_scale = torch.floor(torch.log2(amax.clamp(min=1e-30))) + 127
    mx_scale = mx_scale.clamp(0, 255).to(torch.uint8).unsqueeze(0).repeat(K // 32, 1)
    return packed, w_scale, mx_scale


SHAPES = [(256, 4096, 4096), (512, 4096, 4096), (1024, 4096, 4096),
          (512, 8192, 8192), (1024, 8192, 4096)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['shape'], x_vals=[str(s) for s in SHAPES],
        line_arg='impl', line_vals=['v15_4w4a', 'b12x_w4a16_sim'],
        line_names=['v15 (4W4A)', 'B12X 模拟 (w4a16)'],
        styles=[('green', '-'), ('red', '-')], ylabel='TFLOPS', plot_name='routeA_vs_b12x',
        args={}))
def bench(shape, impl):
    M, K, N = eval(shape)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    w_packed, w_scale, mx_scale = make_weights(K, N)
    if impl == 'v15_4w4a':
        fn = lambda: routeA_gemm(A, w_packed, w_scale)
    else:
        fn = lambda: b12x_sim_gemm(A, w_packed, mx_scale)
    fn()
    torch.cuda.synchronize()
    t = triton.testing.do_bench(fn, warmup=25, rep=100)
    return 2 * M * N * K / t / 1e12


def main():
    print("=== kernel① routeA(v15 4W4A) vs B12X(w4a16 模拟) prefill A/B ===")
    print("注：B12X 侧为反量化+fp16 matmul 模拟（生产真实 B12X kernel 加速比以生产 harness 为准）")
    bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
