"""kernel① v16.1 完整修复版（F1+F2+F3）正确性：8 用例 + max_abs_err 对照旧 torch ref。
"""
import torch
import sys
sys.path.insert(0, ".")
from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_v16_fixed_triton import (
    nvfp4_4w4a_prefill_gemm as fixed_impl,
    preprocess_weights_clear,
)

DEVICE = "cuda"
E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def make_weights(K, N, scale, device):
    w = (torch.rand(K, N, device=device) * 2 - 1) * scale
    w_scale_raw = w.abs().amax(dim=0).clamp(min=1e-9)
    w_scale_blocks = w_scale_raw.view(N // 128, 128).amax(dim=1)
    exp = torch.floor(torch.log2(w_scale_blocks.clamp(min=1e-30) / 6.0)) + 127.0
    exp = exp.clamp(0, 255).to(torch.uint8)
    w_scale = exp.unsqueeze(0).repeat(K // 32, 1)
    w_scale_f = torch.pow(2.0, w_scale.float() - 127.0)
    w_scale_expanded = w_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)
    w_scaled = w / w_scale_expanded
    signs = torch.sign(w_scaled)
    w_abs = w_scaled.abs()
    pos = E2M1_VALUES[:8].to(device)
    idx = (w_abs.unsqueeze(-1) - pos).abs().argmin(dim=-1)
    w_q = (signs * pos[idx]).nan_to_num(0.0)
    nib = torch.zeros(K, N, dtype=torch.uint8, device=device)
    mag = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    mag_val = torch.where(w_q.abs().unsqueeze(-1) == pos.unsqueeze(0).unsqueeze(0),
                          mag, torch.zeros_like(mag)).sum(dim=-1).to(torch.uint8)
    sign_bit = (w_q < 0).to(torch.uint8) * 8
    nib = (mag_val | sign_bit).to(torch.uint8)
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = lo | (hi << 4)
    return packed.contiguous(), w_scale.contiguous()


CASES = [
    (256, 4096, 4096),
    (512, 2048, 4096),
    (1024, 4096, 2048),
    (128, 4096, 4096),
]


def main():
    print("kernel① v16.1 完整修复版 正确性复测（rtol/atol=5e-2，对照旧 torch ref）")
    n_pass = 0
    for (M, K, N) in CASES:
        for use_bias in (False, True):
            torch.manual_seed(0)
            A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
            W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
            bias = torch.randn(N, device=DEVICE, dtype=torch.float32) if use_bias else None
            preprocess_weights_clear()
            ref = ref_impl(A, W_n, W_scale, bias)
            try:
                out = fixed_impl(A, W_n, W_scale, bias)
                max_abs = (out - ref).abs().max().item()
                rel = max_abs / (ref.abs().max() + 1e-9)
                try:
                    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
                    status = "PASS"
                    n_pass += 1
                except Exception:
                    status = "FAIL"
                print(f"[{status}] M={M:5d} K={K:5d} N={N:5d} bias={int(use_bias)} | "
                      f"max_abs_err={max_abs:.6f} rel={rel.item():.6f}")
            except Exception as e:
                print(f"[ERROR] M={M:5d} K={K:5d} N={N:5d} bias={int(use_bias)} | "
                      f"{type(e).__name__}: {str(e)[-160:]}")
    print(f"\n结论: {n_pass}/{len(CASES)*2} PASS")


if __name__ == "__main__":
    main()
