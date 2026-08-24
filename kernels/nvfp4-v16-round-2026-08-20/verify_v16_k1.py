"""kernel① v16 生产复测：
Case A: as-shipped 直调（预期 dot_scaled 位置参数冲突编译失败）
Case B: 最小修复版（dot_scaled 位置参数按 3.6 签名重排：lhs,lhs_scale,'e4m3',rhs,rhs_scale,'e4m3',acc）
        → 正确性 8 用例 + max_abs_err（对照生产 oracle：旧 torch ref）
"""
import torch
import sys

sys.path.insert(0, ".")
from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_v16_triton import (
    nvfp4_4w4a_prefill_gemm as v16_impl,
    preprocess_weights_clear,
)

DEVICE = "cuda"
E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def make_weights(K, N, scale, device):
    """与 shipped test 一致：N 打包 [K, N//2]，lo=偶 N 列；scale [K//32, N//128]"""
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
    # ---- Case A：as-shipped ----
    print("=" * 70)
    print("Case A: v16 as-shipped 直调（不修改）")
    print("=" * 70)
    torch.manual_seed(0)
    M, K, N = 256, 4096, 4096
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
    preprocess_weights_clear()
    try:
        out = v16_impl(A, W_n, W_scale, None)
        print(f"[NO-ERROR] shape={tuple(out.shape)}")
        # 快速数值 sanity
        ref = ref_impl(A, W_n, W_scale, None)
        err = (out - ref).abs().max().item()
        print(f"[sanity] max_abs_err={err:.6f}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)[:400]}")

    # ---- Case B：最小修复（dot_scaled 位置参数重排）----
    print("\n" + "=" * 70)
    print("Case B: 最小修复版（dot_scaled 参数按 Triton 3.6 签名重排）")
    print("=" * 70)
    import nvfp4_4w4a_prefill_gemm_v16_triton as v16mod
    import re

    src = open("nvfp4_4w4a_prefill_gemm_v16_triton.py").read()
    old = "acc = tl.dot_scaled(\n            a_fp8_int, lhs_scale,\n            w_fp8,    rhs_scale,\n            acc,\n            lhs_format='e4m3', rhs_format='e4m3',\n            rhs_k_pack=False,\n        )"
    new = "acc = tl.dot_scaled(\n            a_fp8_int, lhs_scale, 'e4m3',\n            w_fp8,    rhs_scale, 'e4m3',\n            acc,\n            lhs_k_pack=False, rhs_k_pack=False,\n        )"
    assert old in src, "patch anchor not found!"
    patched = src.replace(old, new)
    open("nvfp4_4w4a_prefill_gemm_v16_fixed_triton.py", "w").write(patched)
    import importlib
    mod = importlib.import_module("nvfp4_4w4a_prefill_gemm_v16_fixed_triton")
    fixed_impl = mod.nvfp4_4w4a_prefill_gemm

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
                      f"{type(e).__name__}: {str(e)[:150]}")
    print(f"\nCase B 结论: {n_pass}/{len(CASES)*2} PASS（rtol/atol=5e-2，对照旧 torch ref）")


if __name__ == "__main__":
    main()
