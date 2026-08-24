import torch
import pytest

from nvfp4_ds_mla_kv_linear_paged_torch import nvfp4_ds_mla_kv_linear_paged as ref_impl
from nvfp4_ds_mla_kv_linear_paged_triton import nvfp4_ds_mla_kv_linear_paged as triton_impl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 256


def make_inputs(T, num_seqs=2, max_blocks=8):
    """构造 paged 输入：token 按 seq 轮转分配 position，block_table 指向不同 block 区间。"""
    torch.manual_seed(0)
    k = torch.randn(T, 512, device=DEVICE, dtype=torch.float32) * 0.5
    v = torch.randn(T, 512, device=DEVICE, dtype=torch.float32) * 0.5

    seq_ids = torch.arange(T, device=DEVICE) % num_seqs
    positions = torch.arange(T, device=DEVICE)  # 各 seq 内部连续：seq0 占 0.., seq1 占 0..
    # 修正：每个 seq 的 position 从 0 开始独立计数
    per_seq_count = torch.zeros(num_seqs, dtype=torch.long, device=DEVICE)
    positions = torch.empty(T, dtype=torch.long, device=DEVICE)
    for t in range(T):
        s = int(seq_ids[t])
        positions[t] = per_seq_count[s]
        per_seq_count[s] += 1

    need_blocks = (int(per_seq_count.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert need_blocks <= max_blocks, f"max_blocks={max_blocks} 不够（需要 {need_blocks}）"

    # block_table：seq s 的第 b 块 -> block_id = s*max_blocks + b
    block_table = torch.arange(num_seqs * max_blocks, device=DEVICE).reshape(num_seqs, max_blocks)

    num_blocks = num_seqs * max_blocks
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, 584, dtype=torch.uint8, device=DEVICE)
    return k, v, seq_ids, positions, block_table, kv_cache


@pytest.mark.parametrize("T", [16, 64, 256, 512, 1024])
def test_paged(T):
    if DEVICE != "cuda":
        pytest.skip("需要 CUDA（DGX Spark / SM121）")
    k, v, seq_ids, positions, block_table, kv_cache = make_inputs(T)

    ref_cache = torch.zeros_like(kv_cache)
    ref = ref_impl(k, v, seq_ids, positions, block_table, ref_cache)
    out = triton_impl(k, v, seq_ids, positions, block_table, kv_cache)

    # 逐字节精确一致（量化 + 分页写入纯确定性）
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
