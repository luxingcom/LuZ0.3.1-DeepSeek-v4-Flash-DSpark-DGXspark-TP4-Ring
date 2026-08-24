#!/usr/bin/env python3
"""debug T2 failures: scale grid aliasing / dtype / label-consistency"""
import torch
from b12x.moe.fused.w4a16 import prepare as prep

prep._make_workspace = lambda device, *, max_blocks_per_sm=4: torch.zeros(
    (8,), dtype=torch.int32)

E, HIDDEN, INTER = 32, 512, 256
N13 = 2 * INTER

g = torch.Generator().manual_seed(7)
w13 = torch.randint(0, 256, (E, N13, HIDDEN // 2), dtype=torch.uint8, generator=g)
w2 = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8, generator=g)
s13 = torch.randint(118, 127, (E, N13, HIDDEN // 32), dtype=torch.uint8, generator=g)
s2 = torch.randint(118, 127, (E, HIDDEN, INTER // 32), dtype=torch.uint8, generator=g)
unit = torch.ones(E, dtype=torch.float32)

n = N13 // 2
tmp = w13[:, :n].clone(); w13[:, :n].copy_(w13[:, n:]); w13[:, n:].copy_(tmp)
tmp = s13[:, :n].clone(); s13[:, :n].copy_(s13[:, n:]); s13[:, n:].copy_(tmp)

nat13 = prep.prepare_w4a16_e8m0_native_weights(
    w13, s13, unit, w2, s2, unit, activation="silu",
    params_dtype=torch.bfloat16, w13_layout="w13")
print("w13_scale dtype:", nat13.w13_scale.dtype,
      "shape:", tuple(nat13.w13_scale.shape))
print("ptrs: grid", nat13.w13_scale.data_ptr(), "s13", s13.data_ptr(),
      "equal:", nat13.w13_scale.data_ptr() == s13.data_ptr())
print("storage sizes: grid", nat13.w13_scale.untyped_storage().nbytes(),
      "s13", s13.untyped_storage().nbytes())
print("clamp const:", prep._E8M0_K32_BF16_MAX_SCALE_BYTE)

# label-consistency debug
g2 = torch.Generator().manual_seed(7)
w13b = torch.randint(0, 256, (E, N13, HIDDEN // 2), dtype=torch.uint8, generator=g2)
w2b = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8, generator=g2)
s13b = torch.randint(118, 127, (E, N13, HIDDEN // 32), dtype=torch.uint8, generator=g2)
s2b = torch.randint(118, 127, (E, HIDDEN, INTER // 32), dtype=torch.uint8, generator=g2)
unitb = torch.ones(E, dtype=torch.float32)
assert torch.equal(s13b, torch.cat([s13[:, n:], s13[:, :n]], dim=1)), "swap check"

nat31 = prep.prepare_w4a16_e8m0_native_weights(
    w13b, s13b, unitb, w2b, s2b, unitb, activation="silu",
    params_dtype=torch.bfloat16, w13_layout="w31")

ga, gb = nat13.w13_scale, nat31.w13_scale
rolled = torch.cat([gb[:, :, INTER:], gb[:, :, :INTER]], dim=2)
print("label-consistency equal:", torch.equal(ga, rolled))
diff = (ga != rolled)
print("num diff:", int(diff.sum()), "of", ga.numel())
if diff.any():
    idx = diff.nonzero()[0]
    print("first diff at", idx.tolist(), "ga=", ga[tuple(idx.tolist())].item(),
          "rolled=", rolled[tuple(idx.tolist())].item())
    e, kk, nn = idx.tolist()
    print("around: ga[e,kk,:6]=", ga[e, kk, :6].tolist())
    print("        rolled[e,kk,:6]=", rolled[e, kk, :6].tolist())
