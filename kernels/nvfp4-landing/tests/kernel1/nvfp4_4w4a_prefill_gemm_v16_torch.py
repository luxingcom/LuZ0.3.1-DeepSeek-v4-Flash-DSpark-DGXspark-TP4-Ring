# ============================================================================
# nvfp4_4w4a_prefill_gemm_v16_torch.py —— 参考实现（E2M1→FP8 e4m3 无损展开）
# v16 思路（借鉴 danielwoz/vllm-dspark-nvfp4）：
#   E2M1 幅值 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 在 fp8 e4m3 中全部精确可表示
#   → 4W4A 语义完整保留，计算载体换为 FP8（Triton 3.6 原生支持 fp8 scaled MMA，
#     避免 e2m1 dot_scaled 的 BF16 HMMA 降级）
# 语义：C[M,N] = dot_scaled(A_fp8, A_scale, W_fp8, W_scale) + bias（fp32 累加）
# ============================================================================
import torch
from typing import Optional

# E2M1 FP4 码本（16 值，索引 0..15）
E2M1_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def nvfp4_4w4a_prefill_gemm(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 4W4A prefill GEMM（v16，FP8 计算载体）。

    A:        [M, K] fp32/bf16 激活
    W_packed: [K//2, N] uint8（e2m1 低半字节=第 1 个 K 元素）
    W_scale:  [K//32, N//128] uint8（E8M0，K 方向 32 分组、N 方向 128 块）
    bias:     [N] fp32 可选
    → C:      [M, N] fp32
    """
    A_fp32 = A.to(torch.float32)
    M, K = A_fp32.shape
    K2, N_half = W_packed.shape
    N = N_half * 2
    assert K2 == K, "W K 与 A K 不匹配"
    dev = W_packed.device

    e2m1_table = torch.tensor(E2M1_TABLE, dtype=torch.float32, device=dev)

    # ---- W：解包 E2M1 nibble → fp32 幅值（含符号）[K, N] ----
    W_int = W_packed.to(torch.int32)
    W_lo = e2m1_table[(W_int & 0x0F).reshape(-1)].reshape(K, N_half)
    W_hi = e2m1_table[((W_int >> 4) & 0x0F).reshape(-1)].reshape(K, N_half)
    W_fp32 = torch.stack([W_lo, W_hi], dim=2).reshape(K, N)  # [K, N]

    # ---- W scale：[K//32, N//128] → [K, N]（K 方向 32 组、N 方向 128 块共享）----
    W_scale_f = torch.pow(2.0, W_scale.to(torch.float32) - 127.0)
    W_scale_expanded = W_scale_f.repeat_interleave(32, dim=0).repeat_interleave(128, dim=1)
    W_scaled = W_fp32 * W_scale_expanded                       # [K, N] 反量化

    # ---- A：32 分组量化到 E2M1 → 反量化 fp32 ----
    g = 32
    Ag = A_fp32.reshape(M, K // g, g)
    amax = Ag.abs().amax(dim=2, keepdim=True)
    av = torch.clamp(amax / 6.0, min=1e-38)
    ae = torch.floor(torch.log2(av)).to(torch.int32)
    aec = torch.clamp(ae + 127, 0, 255)                        # E8M0 字节
    ascale = torch.pow(2.0, (aec - 127).to(torch.float32))
    An = torch.clamp(Ag / (ascale + 1e-38), -6.0, 6.0)
    e2m1_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            dtype=torch.float32, device=dev)
    sgn = torch.sign(An)
    ab = An.abs()
    diffs = (ab.unsqueeze(-1) - e2m1_pos.view(1, 1, 1, 8)).abs()
    idx = diffs.argmin(dim=-1)
    Aq = sgn * e2m1_pos.view(1, 1, 1, 8)[0, 0, 0][idx]
    A_deq = (Aq * ascale).reshape(M, K)                        # [M, K] 反量化

    # ---- FP8 e4m3 载体（无损：E2M1 幅值 ⊂ e4m3）----
    A_fp8 = A_deq.to(torch.float8_e4m3fn)                      # [M, K]
    W_fp8 = W_scaled.t().contiguous().to(torch.float8_e4m3fn)  # [N, K]

    # fp32 累加（模拟 dot_scaled(e4m3,e4m3) 语义）
    out = torch.matmul(A_fp8.float(), W_fp8.float().t())       # [M, N]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out
