"""最小编译探针：抓 v16 kernel 完整编译错误（不截断）。"""
import torch
import sys
sys.path.insert(0, ".")
from nvfp4_4w4a_prefill_gemm_v16_triton import nvfp4_4w4a_prefill_gemm as v16_impl

torch.manual_seed(0)
A = torch.randn(256, 4096, device="cuda", dtype=torch.float32)
W = (torch.rand(4096, 2048, device="cuda") * 2 - 1) * 0.5
ws = torch.full((128, 32), 130, dtype=torch.uint8, device="cuda")
try:
    out = v16_impl(A, W.to(torch.uint8), ws, None)
    print("COMPILED_OK", tuple(out.shape))
except Exception as e:
    msg = str(e)
    # CompilationError 结构：前半是源码上下文，错误在 "error:" 之后或末尾
    lines = msg.splitlines()
    print("=== FULL ERROR (last 25 lines) ===")
    for ln in lines[-25:]:
        print(ln)
