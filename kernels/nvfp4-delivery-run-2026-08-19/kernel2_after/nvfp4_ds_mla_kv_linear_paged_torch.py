import torch


def nvfp4_ds_mla_kv_linear_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    kv_cache: torch.Tensor,
) -> torch.Tensor:
    """
    PAGED variant of NVFP4 DS-MLA KV writer.

    Scatter-write NVFP4-quantized K/V into paged KV cache.

    Args:
        k: [T, 512] fp32 - key tensor
        v: [T, 512] fp32 - value tensor
        seq_ids: [T] int32 - sequence IDs for each token
        positions: [T] int32 - positions for each token
        block_table: [num_seqs, max_blocks] int32 - block table mapping
        kv_cache: [num_blocks, block_size, 584] uint8 - preallocated KV cache

    Returns:
        kv_cache: modified in-place and returned
    """
    BLOCK_SIZE = 256
    T = k.shape[0]

    k_f32 = k.float()  # [T, 512]
    v_f32 = v.float()  # [T, 512]

    def quantize_to_nvfp4_e2m1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        N = orig_shape[-1]
        batch_dims = orig_shape[:-1]

        num_groups = N // 16
        x_grouped = x.reshape(*batch_dims, num_groups, 16)

        max_abs = x_grouped.abs().amax(dim=-1)  # [..., num_groups]

        safe_max_abs = max_abs.clamp(min=1e-38)

        log2_6 = torch.log2(torch.tensor(6.0, dtype=torch.float32))
        log2_max = torch.log2(safe_max_abs)
        scale_exp_f = log2_max - log2_6

        scale_exp = torch.floor(scale_exp_f).to(torch.int32)  # 与 triton/paged v5 统一：floor 语义
        scale_exp = scale_exp.clamp(-127, 128)

        e8m0_bytes = (scale_exp + 127).to(torch.uint8)  # [..., num_groups]

        scale_f32 = torch.pow(2.0, scale_exp.float())  # [..., num_groups]

        scale_broadcast = scale_f32.unsqueeze(-1)
        x_scaled = x_grouped / scale_broadcast.clamp(min=1e-38)

        x_clamped = x_scaled.clamp(-6.0, 6.0)

        e2m1_pos_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=x.device
        )

        signs = (x_clamped < 0).to(torch.int32)
        abs_vals = x_clamped.abs()

        abs_expanded = abs_vals.unsqueeze(-1)
        diffs = (abs_expanded - e2m1_pos_values).abs()
        nearest_idx = diffs.argmin(dim=-1)

        nibbles = (signs * 8 + nearest_idx).to(torch.uint8)

        nibbles_reshaped = nibbles.reshape(*batch_dims, num_groups, 8, 2)
        low_nibbles = nibbles_reshaped[..., 0]
        high_nibbles = nibbles_reshaped[..., 1]

        packed_bytes = (low_nibbles | (high_nibbles << 4)).to(torch.uint8)

        packed = packed_bytes.reshape(*batch_dims, N // 2)
        scales = e8m0_bytes

        return packed, scales

    k_packed, k_scales = quantize_to_nvfp4_e2m1(k_f32)  # [T, 256], [T, 32]
    v_packed, v_scales = quantize_to_nvfp4_e2m1(v_f32)  # [T, 256], [T, 32]

    T_count = k_packed.shape[0]

    pad = torch.zeros(T_count, 8, dtype=torch.uint8, device=k.device)
    envelope = torch.cat([k_packed, v_packed, k_scales, v_scales, pad], dim=1)  # [T, 584]

    seq_ids_long = seq_ids.long()
    positions_long = positions.long()

    block_indices = positions_long // BLOCK_SIZE
    slot_indices = positions_long % BLOCK_SIZE

    bid = block_table[seq_ids_long, block_indices].long()  # [T]

    kv_cache[bid, slot_indices, :] = envelope

    return kv_cache
