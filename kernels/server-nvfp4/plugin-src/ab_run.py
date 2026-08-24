"""routeA vs fp16 matmul prefill A/B v5 —— RouteA 缓存权重后测 __call__（避免便捷入口重复量化）。"""
import torch
import triton
import triton.testing
from nvfp4_4w4a_mmaf import RouteA


def make_packed_from_W(w):
    K, N = w.shape
    wb = w.reshape(K // 32, 32, N // 128, 128)
    amax_b = wb.abs().amax(dim=(1, 3)).clamp(min=1e-9)
    w_scale = (torch.floor(torch.log2(amax_b / 6.0)) + 127).clamp(0, 255).to(torch.uint8)
    sf = torch.pow(2.0, w_scale.float() - 127.0)
    w_scale_full = sf.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)
    w_norm = torch.clamp(w / w_scale_full, -6, 6)
    idx = ((w_norm.abs().unsqueeze(-1) - torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=w.device)).abs().argmin(-1))
    mag = idx.to(torch.uint8) | ((w_norm < 0).to(torch.uint8) * 8)
    packed = (mag[:, 0::2] | (mag[:, 1::2] << 4)).to(torch.uint8)
    return packed, w_scale


def tf(m, n, k, t):
    return 2 * m * n * k / t / 1e12


SHAPES = [(256, 4096, 4096), (512, 4096, 4096), (1024, 4096, 4096), (512, 8192, 8192)]


if __name__ == "__main__":
    print("=== routeA(RouteA缓存) vs fp16 matmul prefill A/B ===")
    print(f"{'shape(M,K,N)':<22}{'routeA':>10}{'fp16mm':>10}{'routeA/fp16':>12}")
    for (M, K, N) in SHAPES:
        torch.manual_seed(0)
        W = torch.randn(K, N, device='cuda') * 0.1
        packed, w_scale = make_packed_from_W(W)
        # RouteA 缓存
        impl = RouteA(); impl.preprocess_weights(packed, w_scale)
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        rA = lambda: impl(A, use_cached_w=True)
        o = rA(); torch.cuda.synchronize()
        tA = triton.testing.do_bench(rA, warmup=10, rep=50)
        tflA = tf(M, N, K, tA)
        # fp16 基线
        Wf = W.to(torch.float16)
        bm = lambda: torch.matmul(A, Wf)
        bm(); torch.cuda.synchronize()
        tB = triton.testing.do_bench(bm, warmup=10, rep=50)
        tflB = tf(M, N, K, tB)
        print(f"{str((M,K,N)):<22}{tflA:>10.1f}{tflB:>10.1f}{tflA/max(tflB,1e-9):>12.2f}x", flush=True)
    print("=== done ===")