"""kernel① v15 bf16 GEMM SASS 复核：确认 HMMA.16816.F32.BF16（预期）无 TCGEN05/e2m1。"""
import subprocess, glob, os, torch
from nvfp4_4w4a_prefill_gemm_v15_triton import _bf16_gemm_kernel

A = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
W = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
C = torch.empty(256, 4096, dtype=torch.float32, device="cuda")
grid = (16,)
_bf16_gemm_kernel.fn[grid](
    A, W, C, 256, 4096, 4096,
    A.stride(0), A.stride(1), W.stride(0), W.stride(1),
    C.stride(0), C.stride(1),
    BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8,
    num_warps=4, num_stages=3,
)
torch.cuda.synchronize()
print("COMPILED_OK", C.sum().item())

cubins = glob.glob(os.path.expanduser("~/.triton/cache/**/*.cubin"), recursive=True)
print("cubins:", len(cubins))
cubin = cubins[0]
nvd = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvdisasm"
sass = subprocess.run([nvd, cubin], capture_output=True, text=True).stdout
open("/tmp/v15_sass.txt", "w").write(sass)

counts = {
    "HMMA.16816.F32.BF16": sass.count("HMMA.16816.F32.BF16"),
    "HMMA (any)": sum(1 for ln in sass.splitlines() if "HMMA" in ln),
    "TCGEN05": sum(1 for ln in sass.splitlines() if "TCGEN" in ln),
    "e2m1": sum(1 for ln in sass.splitlines() if "e2m1" in ln.lower()),
    "FFMA": sum(1 for ln in sass.splitlines() if "FFMA" in ln),
}
for k, v in counts.items():
    print(f"{k}: {v}")
print("SASS lines:", len(sass.splitlines()))
