# ============================================================================
# nvfp4_vllm_plugin/nvfp4_vllm_plugin/kv_writer.py
# ============================================================================
"""kernel② v17 KV 写回路径（新建，非替换 fused_compress_quant_cache）。

依据生产报告：生产 KV-linear 走 `fused_compress_quant_cache`（fp8_ds_mla paged
[64,584]），**v17 是 NVFP4 信封（584B）量化工具**——仅在 `--kv-cache-dtype
nvfp4_ds_mla`（NVFP4 KV）启用时使用，fp8 路径保持 fused_compress 原样。

接入方式（二选一，均零侵入）：
  A. writer hook：由上层在 NVFP4 KV dtype 的模型前向中调用
  B. paged 写回：v17_paged（R3 变体）对接 kv_cache[bid, slot, :]
"""

import os
import torch


def write_nvfp4_kv(
    kv_latent: torch.Tensor,          # [T, 1024] fp32/bf16（K/V latent 拼接）
    kv_cache: torch.Tensor,           # [num_blocks, block_size, 584] uint8（paged）
    seq_ids: torch.Tensor,            # [T] int32
    positions: torch.Tensor,          # [T] int32
    block_table: torch.Tensor,        # [num_seqs, max_blocks] int32
    block_size: int = 256,
) -> torch.Tensor:
    """NVFP4 KV 写回（linear 信封 → paged 槽位）。

    ① nvfp4_ds_mla_kv_linear_v17(kv_latent) → env [T, 584]
    ② 按 block_table 散写（R3 paged 变体落地后替换为单 kernel）
    ③ 返回 kv_cache（in-place）
    """
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear
    env = nvfp4_ds_mla_kv_linear(kv_latent)            # [T, 584]
    bid = block_table[seq_ids.long(), (positions.long() // block_size)]
    slot = positions.long() % block_size
    kv_cache[bid, slot, :] = env                        # 散写（R3 前为 torch 索引，生产换 kernel）
    return kv_cache


def warmup_k2(T: int = 8, device: str = "cuda") -> None:
    """服务启动 warmup（autotune 稳态选型）——CUDA Graph 捕获前必调（R2）。"""
    from nvfp4_ds_mla_kv_linear_v17_triton import nvfp4_ds_mla_kv_linear
    kv = torch.randn(T, 1024, device=device)
    nvfp4_ds_mla_kv_linear(kv)
    torch.cuda.synchronize()
