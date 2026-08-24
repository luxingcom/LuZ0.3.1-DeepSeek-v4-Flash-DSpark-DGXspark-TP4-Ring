import torch

x0 = torch.randn(1000, 1000, device="cuda")
torch.cuda.synchronize()
print("ctx ok, reserved=", torch.cuda.memory_reserved())
import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe as fib  # noqa
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout  # noqa
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # noqa
    swizzle_blockscale,
)
print("imports ok, reserved=", torch.cuda.memory_reserved())
x1 = torch.randn(1000, 1000, device="cuda")
torch.cuda.synchronize()
print("alloc after imports ok, reserved=", torch.cuda.memory_reserved())
