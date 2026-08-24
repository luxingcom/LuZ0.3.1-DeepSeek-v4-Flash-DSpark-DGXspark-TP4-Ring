import triton
import triton.language as tl
import torch


@triton.jit
def _scalar_to_e2m1(val):
    ax = tl.abs(val)
    sign_bit = (val < 0.0).to(tl.int32)
    idx = tl.zeros_like(ax).to(tl.int32)
    idx = tl.where(ax >= 0.25, 1, idx)
    idx = tl.where(ax >= 0.75, 2, idx)
    idx = tl.where(ax >= 1.25, 3, idx)
    idx = tl.where(ax >= 1.75, 4, idx)
    idx = tl.where(ax >= 2.5,  5, idx)
    idx = tl.where(ax >= 3.5,  6, idx)
    idx = tl.where(ax >= 5.0,  7, idx)
    return idx | (sign_bit << 3)


@triton.jit
def _quantize_block_16(
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
):
    a0  = tl.abs(x0);  a1  = tl.abs(x1);  a2  = tl.abs(x2);  a3  = tl.abs(x3)
    a4  = tl.abs(x4);  a5  = tl.abs(x5);  a6  = tl.abs(x6);  a7  = tl.abs(x7)
    a8  = tl.abs(x8);  a9  = tl.abs(x9);  a10 = tl.abs(x10); a11 = tl.abs(x11)
    a12 = tl.abs(x12); a13 = tl.abs(x13); a14 = tl.abs(x14); a15 = tl.abs(x15)

    m01   = tl.maximum(a0,  a1);   m23   = tl.maximum(a2,  a3)
    m45   = tl.maximum(a4,  a5);   m67   = tl.maximum(a6,  a7)
    m89   = tl.maximum(a8,  a9);   m1011 = tl.maximum(a10, a11)
    m1213 = tl.maximum(a12, a13);  m1415 = tl.maximum(a14, a15)
    m0123    = tl.maximum(m01,   m23);   m4567     = tl.maximum(m45,   m67)
    m891011  = tl.maximum(m89,   m1011); m12131415 = tl.maximum(m1213, m1415)
    m07  = tl.maximum(m0123, m4567)
    m815 = tl.maximum(m891011, m12131415)
    abs_max = tl.maximum(m07, m815)

    safe_max = tl.maximum(abs_max, 1e-30)
    log2_val = tl.floor(tl.log2(safe_max / 6.0))
    e8m0_exp = log2_val + 127.0
    e8m0_exp = tl.minimum(tl.maximum(e8m0_exp, 0.0), 255.0)
    e8m0_uint = e8m0_exp.to(tl.uint8)

    scale_val = tl.exp2(log2_val)
    inv_scale = 1.0 / scale_val

    s0  = x0  * inv_scale; s1  = x1  * inv_scale; s2  = x2  * inv_scale; s3  = x3  * inv_scale
    s4  = x4  * inv_scale; s5  = x5  * inv_scale; s6  = x6  * inv_scale; s7  = x7  * inv_scale
    s8  = x8  * inv_scale; s9  = x9  * inv_scale; s10 = x10 * inv_scale; s11 = x11 * inv_scale
    s12 = x12 * inv_scale; s13 = x13 * inv_scale; s14 = x14 * inv_scale; s15 = x15 * inv_scale

    c0  = _scalar_to_e2m1(s0);  c1  = _scalar_to_e2m1(s1)
    c2  = _scalar_to_e2m1(s2);  c3  = _scalar_to_e2m1(s3)
    c4  = _scalar_to_e2m1(s4);  c5  = _scalar_to_e2m1(s5)
    c6  = _scalar_to_e2m1(s6);  c7  = _scalar_to_e2m1(s7)
    c8  = _scalar_to_e2m1(s8);  c9  = _scalar_to_e2m1(s9)
    c10 = _scalar_to_e2m1(s10); c11 = _scalar_to_e2m1(s11)
    c12 = _scalar_to_e2m1(s12); c13 = _scalar_to_e2m1(s13)
    c14 = _scalar_to_e2m1(s14); c15 = _scalar_to_e2m1(s15)

    p0 = (c0  & 0xF) | ((c1  & 0xF) << 4)
    p1 = (c2  & 0xF) | ((c3  & 0xF) << 4)
    p2 = (c4  & 0xF) | ((c5  & 0xF) << 4)
    p3 = (c6  & 0xF) | ((c7  & 0xF) << 4)
    p4 = (c8  & 0xF) | ((c9  & 0xF) << 4)
    p5 = (c10 & 0xF) | ((c11 & 0xF) << 4)
    p6 = (c12 & 0xF) | ((c13 & 0xF) << 4)
    p7 = (c14 & 0xF) | ((c15 & 0xF) << 4)

    return p0, p1, p2, p3, p4, p5, p6, p7, e8m0_uint


@triton.jit
def _nvfp4_ds_mla_kv_linear_kernel(
    k_ptr,
    v_ptr,
    out_ptr,
    T,
    k_stride_t,
    v_stride_t,
    out_stride_t,
):
    t = tl.program_id(0)
    t_valid = t < T

    k_base = k_ptr + t.to(tl.int64) * k_stride_t
    v_base = v_ptr + t.to(tl.int64) * v_stride_t
    out_base = out_ptr + t.to(tl.int64) * out_stride_t

    for blk in tl.static_range(0, 32):
        elem_off = blk * 16

        x0  = tl.load(k_base + elem_off +  0, mask=t_valid, other=0.0)
        x1  = tl.load(k_base + elem_off +  1, mask=t_valid, other=0.0)
        x2  = tl.load(k_base + elem_off +  2, mask=t_valid, other=0.0)
        x3  = tl.load(k_base + elem_off +  3, mask=t_valid, other=0.0)
        x4  = tl.load(k_base + elem_off +  4, mask=t_valid, other=0.0)
        x5  = tl.load(k_base + elem_off +  5, mask=t_valid, other=0.0)
        x6  = tl.load(k_base + elem_off +  6, mask=t_valid, other=0.0)
        x7  = tl.load(k_base + elem_off +  7, mask=t_valid, other=0.0)
        x8  = tl.load(k_base + elem_off +  8, mask=t_valid, other=0.0)
        x9  = tl.load(k_base + elem_off +  9, mask=t_valid, other=0.0)
        x10 = tl.load(k_base + elem_off + 10, mask=t_valid, other=0.0)
        x11 = tl.load(k_base + elem_off + 11, mask=t_valid, other=0.0)
        x12 = tl.load(k_base + elem_off + 12, mask=t_valid, other=0.0)
        x13 = tl.load(k_base + elem_off + 13, mask=t_valid, other=0.0)
        x14 = tl.load(k_base + elem_off + 14, mask=t_valid, other=0.0)
        x15 = tl.load(k_base + elem_off + 15, mask=t_valid, other=0.0)

        p0, p1, p2, p3, p4, p5, p6, p7, scale = _quantize_block_16(
            x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15,
        )

        pb = blk * 8
        tl.store(out_base + pb + 0, p0.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 1, p1.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 2, p2.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 3, p3.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 4, p4.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 5, p5.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 6, p6.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 7, p7.to(tl.uint8), mask=t_valid)
        tl.store(out_base + 512 + blk, scale, mask=t_valid)

    for blk in tl.static_range(0, 32):
        elem_off = blk * 16

        x0  = tl.load(v_base + elem_off +  0, mask=t_valid, other=0.0)
        x1  = tl.load(v_base + elem_off +  1, mask=t_valid, other=0.0)
        x2  = tl.load(v_base + elem_off +  2, mask=t_valid, other=0.0)
        x3  = tl.load(v_base + elem_off +  3, mask=t_valid, other=0.0)
        x4  = tl.load(v_base + elem_off +  4, mask=t_valid, other=0.0)
        x5  = tl.load(v_base + elem_off +  5, mask=t_valid, other=0.0)
        x6  = tl.load(v_base + elem_off +  6, mask=t_valid, other=0.0)
        x7  = tl.load(v_base + elem_off +  7, mask=t_valid, other=0.0)
        x8  = tl.load(v_base + elem_off +  8, mask=t_valid, other=0.0)
        x9  = tl.load(v_base + elem_off +  9, mask=t_valid, other=0.0)
        x10 = tl.load(v_base + elem_off + 10, mask=t_valid, other=0.0)
        x11 = tl.load(v_base + elem_off + 11, mask=t_valid, other=0.0)
        x12 = tl.load(v_base + elem_off + 12, mask=t_valid, other=0.0)
        x13 = tl.load(v_base + elem_off + 13, mask=t_valid, other=0.0)
        x14 = tl.load(v_base + elem_off + 14, mask=t_valid, other=0.0)
        x15 = tl.load(v_base + elem_off + 15, mask=t_valid, other=0.0)

        p0, p1, p2, p3, p4, p5, p6, p7, scale = _quantize_block_16(
            x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15,
        )

        pb = 256 + blk * 8
        tl.store(out_base + pb + 0, p0.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 1, p1.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 2, p2.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 3, p3.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 4, p4.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 5, p5.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 6, p6.to(tl.uint8), mask=t_valid)
        tl.store(out_base + pb + 7, p7.to(tl.uint8), mask=t_valid)
        tl.store(out_base + 544 + blk, scale, mask=t_valid)


def nvfp4_ds_mla_kv_linear(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()

    T = k.shape[0]
    assert k.shape == (T, 512), f"Expected k shape (T, 512), got {k.shape}"
    assert v.shape == (T, 512), f"Expected v shape (T, 512), got {v.shape}"

    output = torch.zeros(T, 584, dtype=torch.uint8, device=k.device)

    if T == 0:
        return output

    _nvfp4_ds_mla_kv_linear_kernel[(T,)](
        k,
        v,
        output,
        T,
        k.stride(0),
        v.stride(0),
        output.stride(0),
        num_warps=4,
        num_stages=2,
    )

    return output
