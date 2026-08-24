"""kernel① v15 忠实复测：make_weights(N打包) → repack K打包 → v15，对比 torch ref。
同时复现 as-shipped 直喂（无 repack）的崩溃，判定 v15 布局契约缺陷。
"""
import torch

from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
from nvfp4_4w4a_prefill_gemm_v15_triton import (
    nvfp4_4w4a_prefill_gemm as v15_impl,
    preprocess_weights_clear,
)

DEVICE = "cuda"
E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def make_weights(K, N, scale, device):
    """与 shipped test 完全一致：E2M1 低半字节优先 N 打包 [K, N//2] + scale [K//32, N//128]"""
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


def repack_n_to_k(packed_n, K, N):
    """N 打包 [K, N//2]（lo=偶列）→ K 打包 [K//2, N]（lo=偶 K 行），v15 _unpack 期望的布局。"""
    lo = packed_n & 0x0F
    hi = (packed_n >> 4) & 0x0F
    nib = torch.empty(K, N, dtype=torch.uint8, device=packed_n.device)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    nib_t = nib.t().contiguous()          # [N, K]
    lo_k = nib_t[:, 0::2]                 # [N, K//2] 偶 K 行
    hi_k = nib_t[:, 1::2]                 # [N, K//2] 奇 K 行
    packed_k = (lo_k | (hi_k << 4)).t().contiguous()  # [K//2, N]
    return packed_k


CASES = [
    (256, 4096, 4096),
    (512, 2048, 4096),
    (1024, 4096, 2048),
    (128, 4096, 4096),
]


def main():
    print("=" * 70)
    print("kernel① v15 正确性复测（repack 适配后 vs torch ref，rtol/atol=5e-2）")
    print("=" * 70)
    total_pass = 0
    total_cases = 0
    for (M, K, N) in CASES:
        for use_bias in (False, True):
            total_cases += 1
            torch.manual_seed(0)
            A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
            W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
            bias = torch.randn(N, device=DEVICE, dtype=torch.float32) if use_bias else None
            preprocess_weights_clear()

            ref = ref_impl(A, W_n, W_scale, bias)
            try:
                W_k = repack_n_to_k(W_n, K, N)
                out = v15_impl(A, W_k, W_scale, bias)
                max_abs = (out - ref).abs().max().item()
                rel = (out - ref).abs().max() / (ref.abs().max() + 1e-9)
                try:
                    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
                    status = "PASS"
                    total_pass += 1
                except Exception:
                    status = "FAIL"
                print(f"[{status}] M={M:5d} K={K:5d} N={N:5d} bias={int(use_bias)} | "
                      f"max_abs_err={max_abs:.6f} max_rel={rel.item():.6f}")
            except Exception as e:
                print(f"[ERROR] M={M:5d} K={K:5d} N={N:5d} bias={int(use_bias)} | "
                      f"{type(e).__name__}: {str(e)[:120]}")

    # ---- 复现 as-shipped 直喂崩溃（无 repack）----
    print("\n" + "=" * 70)
    print("缺陷复现：v15 直喂 shipped 格式 [K, N//2]（生产转换器输出，无 repack）")
    print("=" * 70)
    torch.manual_seed(0)
    M, K, N = 256, 4096, 4096
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)
    preprocess_weights_clear()
    try:
        out = v15_impl(A, W_n, W_scale, None)
        print(f"[NO-CRASH] shape={tuple(out.shape)}  —— 意外没崩，需人工核对数值")
    except Exception as e:
        print(f"[CRASH] {type(e).__name__}: {str(e)[:200]}")

    print(f"\n结论: {total_pass}/{total_cases} PASS（rtol/atol=5e-2）")


if __name__ == "__main__":
    main()
