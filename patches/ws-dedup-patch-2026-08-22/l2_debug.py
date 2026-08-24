"""Debug: (1) on-path delta=0 anomaly; (2) NaN in output. Focused, small."""
import gc
import os

import torch

import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe as fib
from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

dev = "cuda"
E, TOPK, K, N, MAXT = 8, 2, 512, 256, 256


def mem():
    return torch.cuda.memory_allocated()


def mk(**ov):
    o = fib.FlashInferB12xExperts.__new__(fib.FlashInferB12xExperts)
    a = dict(global_num_experts=E, topk=TOPK, hidden_dim=K,
             intermediate_size_per_partition=N, max_num_tokens=MAXT,
             num_local_experts=E, _activation_str="silu", _wrapper=None)
    a.update(ov)
    for k, v in a.items():
        setattr(o, k, v)
    return o


print("=== D1: direct ctor delta (no pool involvement) ===")
b = mem()
wr_direct = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=K,
                           intermediate_size=N, use_cuda_graph=True,
                           max_num_tokens=MAXT, num_local_experts=E,
                           activation="silu")
print(f"direct ctor delta = {mem()-b} bytes")
print(f"  _moe_output: {wr_direct._moe_output is not None} "
      f"{tuple(wr_direct._moe_output.shape) if wr_direct._moe_output is not None else ''}")
print(f"  _static: {wr_direct._static_workspace is not None}, "
      f"_dynamic: {wr_direct._dynamic_workspace is not None}")
del wr_direct
gc.collect()

print("=== D2: pool on-path delta ===")
os.environ["VLLM_B12X_SHARED_WRAPPER"] = "1"
fib._B12X_WRAPPER_POOL.clear()
b = mem()
f = mk()
print(f"before ensure: {b}")
f._ensure_wrapper()
a = mem()
print(f"after ensure: {a}  delta = {a-b} bytes")
w = f._wrapper
print(f"  wrapper id={id(w)}, pool size={len(fib._B12X_WRAPPER_POOL)}")
print(f"  _moe_output: {w._moe_output is not None}, "
      f"_static: {w._static_workspace is not None}, "
      f"_dynamic: {w._dynamic_workspace is not None}")

print("=== D3: second same-geometry instance (pool hit, delta must be 0) ===")
b = mem()
f2 = mk()
f2._ensure_wrapper()
print(f"pool-hit delta = {mem()-b} bytes; same obj: {f2._wrapper is w}")

print("=== D4: NaN hunt ===")
os.environ.pop("VLLM_B12X_SHARED_WRAPPER", None)
fib._B12X_WRAPPER_POOL.clear()

torch.manual_seed(1234)
g = torch.Generator(device=dev).manual_seed(22)


def make_layer(seed, zero_weights=False):
    gg = torch.Generator(device=dev).manual_seed(seed)
    if zero_weights:
        w1 = torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8, device=dev)
        w2 = torch.zeros(E, K, N // 2, dtype=torch.uint8, device=dev)
    else:
        w1 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
        w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
    w1_sf = torch.rand(E * (2 * N) * K // 16, device=dev, dtype=torch.float32,
                       generator=gg) * 0.02 + 0.002
    w2_sf = torch.rand(E * K * N // 16, device=dev, dtype=torch.float32,
                       generator=gg) * 0.02 + 0.002
    return dict(
        w1=w1, w2=w2,
        w1_sf=convert_sf_to_mma_layout(w1_sf, m=2 * N, k=K, num_groups=E),
        w2_sf=convert_sf_to_mma_layout(w2_sf, m=K, k=N, num_groups=E),
        w1a=torch.ones(E, device=dev, dtype=torch.float32),
        w2a=torch.ones(E, device=dev, dtype=torch.float32),
        fc2=torch.ones(E, device=dev, dtype=torch.float32),
    )


M = 128
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32, generator=g)
tw = torch.rand(M, TOPK, device=dev, dtype=torch.float32, generator=g)
tw = tw / tw.sum(-1, keepdim=True)

wr = f._wrapper  # reuse existing wrapper (workspace already allocated)

for label, layer in [("random-weights", make_layer(22)),
                     ("zero-weights", make_layer(22, zero_weights=True))]:
    out = wr.run(x=x, w1_weight=layer["w1"], w1_weight_sf=layer["w1_sf"],
                 w2_weight=layer["w2"], w2_weight_sf=layer["w2_sf"],
                 token_selected_experts=ids, token_final_scales=tw,
                 w1_alpha=layer["w1a"], w2_alpha=layer["w2a"],
                 fc2_input_scale=layer["fc2"])
    torch.cuda.synchronize()
    of = out.float()
    nan_cnt = torch.isnan(of).sum().item()
    print(f"[{label}] shape={tuple(out.shape)} nan={nan_cnt}/{of.numel()} "
          f"maxabs_nonnan={of.nan_to_num(0).abs().max().item():.6f} "
          f"min={of.nan_to_num(0).min().item():.6f}")

# NaN structure: which positions?
layer = make_layer(22)
out = wr.run(x=x, w1_weight=layer["w1"], w1_weight_sf=layer["w1_sf"],
             w2_weight=layer["w2"], w2_weight_sf=layer["w2_sf"],
             token_selected_experts=ids, token_final_scales=tw,
             w1_alpha=layer["w1a"], w2_alpha=layer["w2a"],
             fc2_input_scale=layer["fc2"])
torch.cuda.synchronize()
of = out.float()
nan_mask = torch.isnan(of)
print("nan per-row (first 16 rows):", nan_mask.sum(-1)[:16].tolist())
print("unique expert pairs in first rows:", ids[:4].tolist())
# try unique experts per token
ids_uniq = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
out2 = wr.run(x=x, w1_weight=layer["w1"], w1_weight_sf=layer["w1_sf"],
              w2_weight=layer["w2"], w2_weight_sf=layer["w2_sf"],
              token_selected_experts=ids_uniq, token_final_scales=tw,
              w1_alpha=layer["w1a"], w2_alpha=layer["w2a"],
              fc2_input_scale=layer["fc2"])
torch.cuda.synchronize()
of2 = out2.float()
print(f"[unique-experts] nan={torch.isnan(of2).sum().item()}/{of2.numel()} "
      f"maxabs_nonnan={of2.nan_to_num(0).abs().max().item():.6f}")
