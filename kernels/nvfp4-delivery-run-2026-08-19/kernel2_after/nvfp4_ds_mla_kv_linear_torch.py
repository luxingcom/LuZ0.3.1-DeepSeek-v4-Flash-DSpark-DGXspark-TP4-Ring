import torch


def nvfp4_ds_mla_kv_linear(kv: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch reference implementation of NVFP4 DS-MLA KV linear layout writer.

    Input:  kv [T, 1024] fp32 (or any dtype, converted to fp32)
            columns [0:512] = K, columns [512:1024] = V
    Output: [T, 584] uint8
            bytes [0:256]   = K packed NVFP4 E2M1 (2 values per byte)
            bytes [256:512] = V packed NVFP4 E2M1 (2 values per byte)
            bytes [512:544] = K scales E8M0 (1 per group of 16)
            bytes [544:576] = V scales E8M0 (1 per group of 16)
            bytes [576:584] = pad (zeros)
    """
    kv = kv.to(torch.float32)
    T = kv.shape[0]

    k = kv[:, :512]
    v = kv[:, 512:]

    def quantize_nvfp4(x: torch.Tensor):
        """
        Quantize [T, 512] fp32 -> NVFP4 E2M1 with per-16 E8M0 scale.
        Returns packed uint8 [T, 256] and scale uint8 [T, 32].
        """
        T, N = x.shape
        num_groups = N // 16

        groups = x.reshape(T, num_groups, 16)

        abs_max = groups.abs().amax(dim=-1, keepdim=True)
        abs_max = abs_max.clamp(min=1e-30)

        # NVFP4 E8M0：scale = 2^floor(log2(max/6))（与生产 v5/内核 floor 语义逐字节一致）
        log2_abs_max = torch.log2(abs_max / 6.0)
        scale_exp = torch.floor(log2_abs_max).to(torch.int32)
        scale_exp = scale_exp.clamp(-126, 127)

        scale_float = torch.pow(2.0, scale_exp.to(torch.float32))

        x_scaled = groups / scale_float

        x_clamped = x_scaled.clamp(-6.0, 6.0)

        def float_to_e2m1(val: torch.Tensor) -> torch.Tensor:
            sign = (val < 0).to(torch.int32)
            abs_val = val.abs()

            e2m1_table_abs = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                dtype=torch.float32, device=val.device
            )

            abs_expanded = abs_val.unsqueeze(-1)
            table_expanded = e2m1_table_abs.reshape(*([1] * len(abs_val.shape)), 8)
            diffs = (abs_expanded - table_expanded).abs()
            mantissa_exp = diffs.argmin(dim=-1).to(torch.int32)

            nibble = sign * 8 + mantissa_exp
            nibble = nibble.to(torch.uint8)
            return nibble

        nibbles = float_to_e2m1(x_clamped)

        nibbles_flat = nibbles.reshape(T, N)
        lo = nibbles_flat[:, 0::2]
        hi = nibbles_flat[:, 1::2]
        packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
        packed = packed.to(torch.uint8)

        scale_exp_clamped = (scale_exp.squeeze(-1) + 127).clamp(0, 255)
        scale_bytes = scale_exp_clamped.to(torch.uint8)

        return packed, scale_bytes

    k_packed, k_scales = quantize_nvfp4(k)
    v_packed, v_scales = quantize_nvfp4(v)

    pad = torch.zeros(T, 8, dtype=torch.uint8, device=kv.device)

    out = torch.cat([k_packed, v_packed, k_scales, v_scales, pad], dim=1)

    return out


# ---- 兼容入口：K/V 分离输入 ----
def nvfp4_ds_mla_kv_linear_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()
    assert k.shape == v.shape and k.ndim == 2 and k.shape[1] == 512
    return nvfp4_ds_mla_kv_linear(torch.cat([k, v], dim=1))
