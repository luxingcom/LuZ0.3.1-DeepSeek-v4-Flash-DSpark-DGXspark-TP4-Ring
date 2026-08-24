#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TP2/TP4 互联带宽与延时精确分析 v2
修正带宽口径：按 per-layer 通信字节 / per-layer 计算时间（流水线并行视角）
"""
import math

H = 4096; L = 43; HEADS = 64; HEAD_DIM = 512
Q_LO = 1024; MOE_N = 256; MOE_K = 6; MOE_FF = 2048
COMM_B = 1.0  # FP8
COMPRESS = [0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
            4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]

def per_layer_ar_bytes(tokens):
    """每层 2 次 all-reduce（QKV 投影 + O/FFN），每次 hidden×tokens×FP8"""
    return 2 * H * tokens * COMM_B

def per_layer_ag_bytes(tokens, compress_ratio):
    """每层 KV all-gather（MLA 压缩后）"""
    kv_dim = (Q_LO + HEAD_DIM) / max(1, compress_ratio) if compress_ratio else (Q_LO + HEAD_DIM)
    return kv_dim * tokens * COMM_B

# ── 网络物理参数（实测）──
NET_200G = 200.0  # Gbps
NET_100G = 100.0
LAT_NCCL_1HOP = 29.0   # µs，NCCL 16B all-reduce 单跳
LAT_WIRE_1HOP = 3.27   # µs，纯 wire

def fmt_gbps(v):
    return f"{v:.1f} Gbps ({v/25*100:.0f}% of 200G, {v/12.5*100:.0f}% of 100G)"

print("="*78)
print("A. 每 token 通信字节（全模型）")
print("="*78)
for sq in [1, 512, 4096, 32768, 131072]:
    ar = sum(per_layer_ar_bytes(sq) for _ in range(L+1))
    ag = sum(per_layer_ag_bytes(sq, cr) for cr in COMPRESS)
    print(f"  seq={sq:>7}: all-reduce={ar/1e6:>7.2f}MB  KV-ag={ag/1e6:>6.2f}MB  合计={ (ar+ag)/1e6:>7.2f}MB"
          f"  每token={(ar+ag)/sq/1024:.1f}KB")

print()
print("="*78)
print("B. 带宽需求（per-layer 通信 / per-layer 计算时间）")
print("="*78)
print("   用 vLLM 实测 prefill/decode 速率估算单层计算时间，从而得 per-layer 峰值带宽")
print()
# vLLM 实测（B 组 benchmark 2026-08-08）:
#   prefill: 512=1118 t/s, 4096=1900 t/s, 32768=2000 t/s, 131072=1660 t/s
#   decode: 短ctx ~70-80 t/s, 长ctx(131K) 2-7 t/s
prefill_rates = {512:1118, 4096:1900, 32768:2000, 131072:1660}
decode_rates  = {512:75, 4096:75, 32768:50, 131072:5}

for sq, rate in prefill_rates.items():
    layer_time = sq / rate / L  # s per layer
    ar = per_layer_ar_bytes(sq)
    ag = per_layer_ag_bytes(sq, 128)
    bw_ar = ar / layer_time / 1e9  # GB/s
    bw_ag = ag / layer_time / 1e9
    print(f"  prefill seq={sq:>6} ({rate:>4} t/s): 单层AR {ar/1e6:>6.1f}MB → 需 {bw_ar:>5.1f}GB/s "
          f"= {fmt_gbps(bw_ar*8)}")

print()
for sq, rate in decode_rates.items():
    # decode: 每步 1 token，单层计算时间 = 1/rate/L
    layer_time = 1.0 / rate / L
    ar = per_layer_ar_bytes(1)   # 1 token
    bw_ar = ar / layer_time / 1e9
    print(f"  decode  seq={sq:>6} ({rate:>4} t/s): 单层AR {ar/1e6*1000:>5.0f}KB → 需 {bw_ar:>6.3f}GB/s "
          f"= {fmt_gbps(bw_ar*8)}")

print()
print("="*78)
print("C. 延时影响（TP2 1跳 vs TP4 2跳）")
print("="*78)
print(f"  实测: NCCL 16B all-reduce 单跳={LAT_NCCL_1HOP}µs, wire 1跳={LAT_WIRE_1HOP}µs")
print(f"  每层 2 次 all-reduce:")
for tp, hops, label in [(2,1,"TP2"), (4,2,"TP4")]:
    per_layer = 2 * hops * LAT_NCCL_1HOP
    whole = per_layer * (L+1) / 1000
    print(f"  {label}: 每层 {per_layer:.0f}µs, 全模型 {whole:.2f}ms (纯通信, 流水未掩蔽)")

print()
print("  但大消息时带宽起主导: 单次 all-reduce 时间 = 延迟 + 字节/带宽")
print(f"  200G 链路单次 AR 传输时间 (seq=4096, 16MB): {16/25*1000:.0f}µs (200G) vs {16/12.5*1000:.0f}µs (100G)")
print(f"  延迟占比: 小消息(16B) 29µs 全延迟; 大消息(16MB) 带宽主导 (200G 传输33µs)")

print()
print("="*78)
print("D. TP2 vs TP4 汇总")
print("="*78)
print("""
| 维度                | TP2 (1跳)                 | TP4 (2跳 ring)            |
|---------------------|---------------------------|---------------------------|
| 每层 all-reduce     | 2 次 × 1 跳               | 2 次 × 2 跳               |
| 每层延迟 (小消息)   | ~58µs                     | ~116µs                    |
| 全模型延迟          | ~2.6ms                    | ~5.1ms                    |
| 带宽需求 (prefill)  | <10% of 200G              | <10% of 200G (同消息量)   |
| KV 通信             | ~11KB/token (MLA 128×)    | 同左 (压缩后极低)         |
| 200G 是否够用       | 充裕 (带宽余量 >10×)      | 充裕 (每边仍 <20%)        |
| 关键瓶颈            | 延迟 (GPU launch 主导)    | 延迟×2 + ring 2跳         |
""")
