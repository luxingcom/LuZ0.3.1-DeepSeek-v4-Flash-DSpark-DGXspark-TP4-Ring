# ============================================================================
# perf_diag_fp4_gemm.py —— 原生 FP4 MMA 性能归因诊断（生产 DGX Spark）
# 目的：定位 60~187 TFLOPS 与 350 目标的差距（shape 依赖 / A 量化开销 / W 缓存 / tile）
# 用法：python perf_diag_fp4_gemm.py [--backend cutlass|flashinfer] [--gemm-only]
# ============================================================================
import argparse
import time
import torch

# ── 被测入口（按生产集成方式 import；骨架以 vLLM 原生 cutlass_scaled_fp4_mm 为例）──
try:
    from vllm._custom_ops import cutlass_scaled_fp4_mm as _fp4_mm
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

try:
    from flashinfer.jit import has_prebuilt_ops  # 占位：FlashInfer NVFP4 backend 入口按实际 API
    HAS_FLASHINFER = False  # 骨架：接入实际 backend 后置 True
except ImportError:
    HAS_FLASHINFER = False


def quant_a_fp4(A: torch.Tensor):
    """A 量化（对齐方案 B 的 A 量化 kernel 输出：fp4 打包 + swizzled scale）。

    骨架：此处用简单实现占位——生产应替换为实际 A 量化 kernel。
    返回 (A_q [M, K//2] uint8, A_sf [M, K//16/4] fp8e4m3)。
    """
    M, K = A.shape
    # 简化占位（真实实现为 Triton/CUDA kernel）：
    A_q = torch.zeros(M, K // 2, dtype=torch.uint8, device=A.device)
    A_sf = torch.zeros(M, K // 64, dtype=torch.float8_e4m3fn, device=A.device)
    return A_q, A_sf


def preprocess_w(W_packed: torch.Tensor, W_scale: torch.Tensor, N: int, K: int):
    """W 预处理（骨架：生产为 repack + scale swizzle，可缓存）。"""
    B_q = W_packed.t().contiguous() if W_packed.shape[0] == K else W_packed  # [N, K//2]
    B_sf = torch.zeros(N, K // 64, dtype=torch.float8_e4m3fn, device=W_packed.device)
    return B_q, B_sf


def fp4_gemm(A_q, B_q, A_sf, B_sf, M, N, K, backend):
    D = torch.empty(M, N, dtype=torch.bfloat16, device=A_q.device)
    alpha = torch.tensor(1.0, dtype=torch.float32, device=A_q.device)
    if backend == "cutlass" and HAS_CUTLASS:
        _fp4_mm(D, A_q, B_q, A_sf, B_sf, alpha)
        return D
    # FlashInfer 或其他 backend 骨架
    raise NotImplementedError(f"backend {backend} 未接入；按生产集成 API 补全")


def bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cutlass", choices=["cutlass", "flashinfer"])
    ap.add_argument("--gemm-only", action="store_true",
                    help="仅测纯 GEMM（跳过 A 量化；用于拆分量化开销）")
    ap.add_argument("--shapes", default="256x4096x2048,512x4096x2048,1024x4096x2048,"
                                         "2048x4096x2048,4096x4096x2048,1024x2048x4096")
    args = ap.parse_args()

    print(f"=== FP4 MMA 性能归因（backend={args.backend}, gemm_only={args.gemm_only}）===")
    print(f"{'MxNxK':<18}{'GEMM-only':>12}{'全链路':>12}{'量化占比':>10}{'TFLOPS':>10}")

    for s in args.shapes.split(","):
        M, K, N = map(int, s.split("x"))
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device="cuda")
        W_scale = torch.randint(0, 255, (K // 32, N // 128), dtype=torch.uint8, device="cuda")

        # W 预处理（稳态：缓存后）
        B_q, B_sf = preprocess_w(W_packed, W_scale, N, K)

        flops = 2.0 * M * N * K

        def run_full():
            A_q, A_sf = quant_a_fp4(A)
            return fp4_gemm(A_q, B_q, A_sf, B_sf, M, N, K, args.backend)

        if args.gemm_only:
            A_q, A_sf = quant_a_fp4(A)   # 量化成本不计入 GEMM-only
            t = bench(lambda: fp4_gemm(A_q, B_q, A_sf, B_sf, M, N, K, args.backend))
            print(f"{s:<18}{t*1e3:>10.2f}ms{'':>12}{'':>10}{flops/t/1e12:>8.1f}")
            continue

        t_full = bench(run_full)
        # 量化部分 = 全链路 - GEMM-only（单独测）
        A_q, A_sf = quant_a_fp4(A)
        t_gemm = bench(lambda: fp4_gemm(A_q, B_q, A_sf, B_sf, M, N, K, args.backend))
        t_quant = max(t_full - t_gemm, 0.0)
        print(f"{s:<18}{t_gemm*1e3:>10.2f}ms{t_full*1e3:>12.2f}ms{t_quant/t_full*100:>9.1f}%{flops/t_full/1e12:>8.1f}")

    print("\n判据：")
    print("  量化占比 >20% → P1（量化 kernel 大 tile/向量化/流水）")
    print("  M 依赖明显（M=256 vs 4096 差 3×）→ P3（tile 调优）")
    print("  首调 vs 稳态差距大 → P2（W 缓存）")
    print("  均达标后仍 <350 → P5（FlashInfer 0.6.8+ 内核对照）")


if __name__ == "__main__":
    main()
