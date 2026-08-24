"""探针：Triton 3.6.0 tl.dot 配 fp8 e4m3 输入 → SASS 是否原生 E4M3 MMA？
若原生 → v16b（预缩放 + 普通 fp8 dot）可行；若降级 bf16 → 3.6 无 FP8 MMA codegen。
"""
import subprocess, glob, os, torch
import triton
import triton.language as tl


@triton.jit
def fp8_dot_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b, acc=acc)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def main():
    M, N, K = 256, 256, 256
    A = torch.randint(0, 255, (M, K), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    B = torch.randint(0, 255, (K, N), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    C = torch.empty(M, N, dtype=torch.float32, device="cuda")
    grid = (1,)
    fp8_dot_kernel[grid](A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
                         num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    print("FP8_DOT_COMPILED_OK", C.sum().item())

    cubins = glob.glob(os.path.expanduser("~/.triton/cache/**/*.cubin"), recursive=True)
    nvd = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm"
    sass = subprocess.run([nvd, cubins[0]], capture_output=True, text=True).stdout
    counts = {
        "HMMA.E4M3 (原生FP8 MMA)": sum(1 for ln in sass.splitlines() if "E4M3" in ln and "HMMA" in ln),
        "HMMA.BF16": sass.count("HMMA.16816.F32.BF16"),
        "HMMA (any)": sum(1 for ln in sass.splitlines() if "HMMA" in ln),
        "TCGEN05": sum(1 for ln in sass.splitlines() if "TCGEN" in ln),
        "FFMA": sum(1 for ln in sass.splitlines() if "FFMA" in ln),
        "SASS lines": len(sass.splitlines()),
    }
    for k, v in counts.items():
        print(f"{k}: {v}")

    # 性能对照
    import time
    for _ in range(10):
        fp8_dot_kernel[grid](A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        fp8_dot_kernel[grid](A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100
    print(f"fp8 dot 256x256x256: {t*1e3:.3f} ms = {2*256*256*256/t/1e12:.1f} TFLOPS")


if __name__ == "__main__":
    main()
