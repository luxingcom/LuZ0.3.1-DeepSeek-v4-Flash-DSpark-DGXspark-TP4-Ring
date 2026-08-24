import torch


def nvfp4_ds_mla_kv_linear(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    NVFP4 DS-MLA KV cache linear layout writer.

    Args:
        k: K activations, shape [T, 512], any dtype (converted to fp32)
        v: V activations, shape [T, 512], any dtype (converted to fp32)

    Returns:
        output: uint8 tensor of shape [T, 584]
                bytes[0:256]   = K packed NVFP4 E2M1 (2 values per byte)
                bytes[256:512] = V packed NVFP4 E2M1 (2 values per byte)
                bytes[512:544] = K E8M0 block scales (32 blocks of 16)
                bytes[544:576] = V E8M0 block scales (32 blocks of 16)
                bytes[576:584] = zero pad
    """
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    T = k.shape[0]
    assert k.shape == (T, 512)
    assert v.shape == (T, 512)

    def quantize_nvfp4(x: torch.Tensor):
        """
        Quantize fp32 tensor [T, 512] to NVFP4 E2M1.

        Returns:
            packed: uint8 [T, 256]  (two 4-bit values per byte, low nibble first)
            scales: uint8 [T, 32]   (one E8M0 scale per 16-element block)
        """
        T, D = x.shape
        num_blocks = D // 16

        x_blocks = x.reshape(T, num_blocks, 16)

        abs_max = x_blocks.abs().amax(dim=-1)

        safe_max = abs_max.clamp(min=1e-30)

        log2_val = torch.log2(safe_max / 6.0).floor()

        e8m0_exp = log2_val + 127.0
        e8m0_exp = e8m0_exp.clamp(0.0, 255.0).to(torch.uint8)

        scale_val = torch.pow(2.0, log2_val)

        x_scaled = x_blocks / scale_val.unsqueeze(-1)

        e2m1_table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32, device=x.device
        )

        x_abs = x_scaled.abs()
        x_sign = x_scaled.sign()

        diff = (x_abs.unsqueeze(-1) - e2m1_table.view(1, 1, 1, 8)).abs()
        nearest_idx = diff.argmin(dim=-1).to(torch.int32)

        sign_bit = (x_sign < 0).to(torch.int32)
        nvfp4_codes = nearest_idx | (sign_bit << 3)

        nvfp4_codes = nvfp4_codes.reshape(T, D).to(torch.int32)

        lo = nvfp4_codes[:, 0::2] & 0xF
        hi = nvfp4_codes[:, 1::2] & 0xF
        packed = (lo | (hi << 4)).to(torch.uint8)

        return packed, e8m0_exp

    k_packed, k_scales = quantize_nvfp4(k)
    v_packed, v_scales = quantize_nvfp4(v)

    output = torch.zeros(T, 584, dtype=torch.uint8, device=k.device)

    output[:, 0:256] = k_packed
    output[:, 256:512] = v_packed
    output[:, 512:544] = k_scales
    output[:, 544:576] = v_scales

    return output
