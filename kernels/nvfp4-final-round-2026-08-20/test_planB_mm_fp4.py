"""方案 B 实测：FlashInfer mm_fp4（4W4A 原生 FP4 GEMM）
Stage 1: FlashInfer 原生 nvfp4_quantize → mm_fp4（冒烟 + SASS 门禁 + TFLOPS）
Stage 2: 本语义量化（32 组 E8M0 阈值链 + N 向→K 向 repack + E8M0→e4m3→block_scale_interleave）
         → mm_fp4 → 对照 torch 参考（rtol/atol 5e-2）
"""
import torch
import time
import flashinfer

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


def quant_a_ours(A):
    """A [M,K] → E2M1 K 向打包 [M,K//2] + E8M0 scale [M, K//32]（32 组阈值链）。"""
    M, K = A.shape
    g = 32
    Ag = A.reshape(M, K // g, g)
    amax = Ag.abs().amax(dim=2, keepdim=True)
    av = torch.clamp(amax / 6.0, min=1e-38)
    ae = torch.floor(torch.log2(av)).to(torch.int32)
    aec = torch.clamp(ae + 127, 0, 255)
    ascale = torch.pow(2.0, (aec - 127).to(torch.float32))
    An = torch.clamp(Ag / (ascale + 1e-38), -6.0, 6.0)
    sgn = torch.sign(An)
    ab = An.abs()
    pos = E2M1_VALUES[:8].to(DEVICE)
    idx = (ab.unsqueeze(-1) - pos.view(1, 1, 1, 8)).abs().argmin(dim=-1)
    Aq = sgn * pos[idx]
    # e2m1 nibble
    mag_val = torch.zeros_like(ab, dtype=torch.int32)
    for i, p in enumerate(pos):
        mag_val = torch.where(ab == p, i, mag_val)
    sign_bit = (Aq < 0).to(torch.int32) * 8
    nib = (mag_val + sign_bit).to(torch.uint8)  # [M, K//32, 32]
    # K 向打包：低半字节=偶 K 元素
    nib_pairs = nib.reshape(M, K // 2, 2)
    packed = (nib_pairs[:, :, 0] | (nib_pairs[:, :, 1] << 4)).contiguous()  # [M, K//2]
    a_sf = aec.squeeze(-1).to(torch.uint8).contiguous()  # [M, K//32]
    return packed, a_sf


def e8m0_to_e4m3_bits(sf_u8):
    """E8M0 uint8 → e4m3 位模式 uint8（2^exp 在 e4m3 精确：bit = (exp+7)<<3，exp=byte-127）。"""
    exp = sf_u8.int() - 127
    field = (exp + 7).clamp(0, 15)
    return (field << 3).to(torch.uint8)


def repack_w_n2k(W_n, K, N):
    """[K, N//2] N 向 → [N, K//2] K 向。"""
    lo = W_n & 0x0F
    hi = (W_n >> 4) & 0x0F
    nib = torch.empty(K, N, dtype=torch.uint8, device=W_n.device)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    nib_t = nib.t().contiguous()          # [N, K]
    lo_k = nib_t[:, 0::2]                 # 偶 K
    hi_k = nib_t[:, 1::2]                 # 奇 K
    return (lo_k | (hi_k << 4)).contiguous()  # [N, K//2]


def main():
    M, K, N = 256, 4096, 4096
    torch.manual_seed(0)
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W_n, W_scale = make_weights(K, N, scale=0.5, device=DEVICE)

    from nvfp4_4w4a_prefill_gemm_torch import nvfp4_4w4a_prefill_gemm as ref_impl
    ref = ref_impl(A, W_n, W_scale, None)

    # ---- Stage 1: FlashInfer 原生量化 ----
    print("=" * 60)
    print("Stage 1: FlashInfer 原生 nvfp4_quantize + mm_fp4")
    print("=" * 60)
    a_gsf = torch.ones(1, device=DEVICE)  # 全局 scale
    try:
        a_q, a_sf = flashinfer.nvfp4_quantize(A.bfloat16(), a_gsf, sf_vec_size=16)
        print("a_q", tuple(a_q.shape), "a_sf", tuple(a_sf.shape), a_sf.dtype)
        w_k = repack_w_n2k(W_n, K, N).t().contiguous()  # [K//2, N]（mm_fp4 契约：b 行=K 对）
        # W scale [K//32,N//128] → 按 N 行 [N, K//32] → e4m3 → swizzle
        ws_rhs = W_scale.repeat_interleave(128, dim=1).t().contiguous()  # [N, K//32]
        b_sf_raw = e8m0_to_e4m3_bits(ws_rhs)
        b_sf = flashinfer.block_scale_interleave(b_sf_raw)
        a_sf_swz = flashinfer.block_scale_interleave(a_sf)
        print("b_sf swizzled", tuple(b_sf.shape), "a_sf swizzled", tuple(a_sf_swz.shape))
        out = flashinfer.mm_fp4(a_q, w_k, a_sf_swz, b_sf, alpha=None, block_size=16, backend="b12x")
        print("mm_fp4 out", tuple(out.shape), out.dtype, "sum", out.float().sum().item())
        err = (out.float() - ref).abs().max().item()
        print(f"Stage1 max_abs_err vs torch ref: {err:.6f}")
    except Exception as e:
        print(f"Stage1 FAIL: {type(e).__name__}: {str(e)[:300]}")

    # ---- Stage 2: 本语义量化 + mm_fp4 ----
    print("\n" + "=" * 60)
    print("Stage 2: 本语义量化（32 组阈值链）+ mm_fp4 vs torch ref")
    print("=" * 60)
    try:
        a_q2, a_sf2 = quant_a_ours(A)          # [M,K//2], [M,K//32]
        w_k2 = repack_w_n2k(W_n, K, N).t().contiguous()
        a_sf2_e4 = e8m0_to_e4m3_bits(a_sf2)          # [M, K//32]
        ws_rhs2 = W_scale.repeat_interleave(128, dim=1).t().contiguous()
        b_sf2_e4 = e8m0_to_e4m3_bits(ws_rhs2)
        a_sf2_swz = flashinfer.block_scale_interleave(a_sf2_e4)
        b_sf2_swz = flashinfer.block_scale_interleave(b_sf2_e4)
        out2 = flashinfer.mm_fp4(a_q2, w_k2, a_sf2_swz, b_sf2_swz, alpha=None, block_size=16, backend="b12x")
        err2 = (out2.float() - ref).abs().max().item()
        try:
            torch.testing.assert_close(out2.float(), ref, rtol=5e-2, atol=5e-2)
            verdict = "PASS"
        except Exception:
            verdict = "FAIL"
        print(f"Stage2 {verdict} | max_abs_err={err2:.6f} | shape={tuple(out2.shape)}")

        # ---- 性能（bf16 输出，W 预打包）----
        for _ in range(5):
            flashinfer.mm_fp4(a_q2, w_k2, a_sf2_swz, b_sf2_swz, block_size=16, backend="b12x")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            flashinfer.mm_fp4(a_q2, w_k2, a_sf2_swz, b_sf2_swz, block_size=16, backend="b12x")
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 30
        print(f"mm_fp4 {M}x{K}x{N}: {t*1e3:.3f} ms = {2*M*K*N/t/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"Stage2 FAIL: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
