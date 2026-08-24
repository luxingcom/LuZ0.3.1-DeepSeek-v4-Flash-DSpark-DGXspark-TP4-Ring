"""v16 fixed 内核 SASS 复核：确认 e4m3 scaled MMA 是原生张量核心还是软件模拟。"""
import subprocess, glob, os, torch, sys
sys.path.insert(0, ".")
from nvfp4_4w4a_prefill_gemm_v16_fixed_triton import _nvfp4_gemm_kernel

torch.manual_seed(0)
A = torch.randn(256, 4096, device="cuda", dtype=torch.float32)
W = (torch.rand(4096, 2048, device="cuda") * 2 - 1).to(torch.uint8)
ws = torch.full((128, 32), 130, dtype=torch.uint8, device="cuda")
C = torch.empty(256, 4096, dtype=torch.float32, device="cuda")
bias = A  # unused placeholder

# 固定配置编译（.fn 绕过 autotune）
grid = (16,)
_nvfp4_gemm_kernel.fn[grid](
    A, W, ws, C, bias,
    256, 4096, 4096,
    A.stride(0), A.stride(1),
    W.stride(0), W.stride(1),
    ws.stride(0), ws.stride(1),
    C.stride(0), C.stride(1),
    HAS_BIAS=False, BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8,
    num_warps=4, num_stages=3,
)
torch.cuda.synchronize()
print("COMPILED_OK", C.sum().item())

cubins = glob.glob(os.path.expanduser("~/.triton/cache/**/*.cubin"), recursive=True)
cubin = cubins[0]
nvd = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm"
sass = subprocess.run([nvd, cubin], capture_output=True, text=True).stdout
open("/tmp/v16_sass.txt", "w").write(sass)

counts = {
    "HMMA.16816.F32.E4M3": sass.count("E4M3"),
    "HMMA.16816.F32.BF16": sass.count("HMMA.16816.F32.BF16"),
    "HMMA (any)": sum(1 for ln in sass.splitlines() if "HMMA" in ln),
    "TCGEN05": sum(1 for ln in sass.splitlines() if "TCGEN" in ln),
    "FFMA": sum(1 for ln in sass.splitlines() if "FFMA" in ln),
    "FADD": sum(1 for ln in sass.splitlines() if "FADD" in ln),
    "FMUL": sum(1 for ln in sass.splitlines() if "FMUL" in ln),
    "HFMA2": sum(1 for ln in sass.splitlines() if "HFMA2" in ln),
    "IMAD": sum(1 for ln in sass.splitlines() if "IMAD" in ln),
}
for k, v in counts.items():
    print(f"{k}: {v}")
print("SASS lines:", len(sass.splitlines()))
