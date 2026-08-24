# kernel① 方案 B：vLLM 现成 CUTLASS NVFP4 内核集成方案

> 日期：2026-08-20 | 背景：v17 轮实锤 Triton 3.6.0 sm_121 **无任何原生 FP8/FP4 MMA codegen**（v16 fp8 0.1~0.2 TFLOPS 弃用；v15 bf16 26.7~81.4 为 Triton 可达最优）→ 400 TFLOPS 仅方案 B 可达
> 路径：**vLLM 官方 `nvfp4_scaled_mm_sm120_kernels.cu`（SM120 专属）**，完整 vLLM 源码构建环境内集成

---

## 一、内核定位（已核实源码签名）

vLLM `csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`（PR #21309 LopezCastroRoberto 首入，后续 #42209 系列演进；同类参考 SubSir/nvfp_kernel）：

```cpp
void cutlass_scaled_fp4_mm_sm120a(
    torch::Tensor& D,            // 输出 [m, n]（bf16 或 fp16）
    torch::Tensor const& A,      // [m, k/2] uint8（FP4 E2M1 **K 向打包**，2 元素/字节）
    torch::Tensor const& B,      // [n, k/2] uint8（FP4 E2M1 K 向打包）
    torch::Tensor const& A_sf,   // [round_up(m,128), round_up(k/16,4)] fp8 e4m3（**swizzled**）
    torch::Tensor const& B_sf,   // [round_up(n,128), round_up(k/16,4)] fp8 e4m3（swizzled）
    torch::Tensor const& alpha); // fp32 全局 scale
```

约束：`k = A.size(1)*2`，`k%32==0`、`n%32==0`；需要 `CUTLASS_ARCH_MMA_SM120_SUPPORTED`（宏前置）；CUDA ≥12.8；`FP4_ARCHS "12.0a;12.1a"`。

**block-scaled 语义确认**：scale 每 **16 K 元素**一个（`rounded_k = k/16/4`，4 scales/int32 swizzle）——与我们的 32 分组兼容（32 组重复为 2×16 组，值相同）。

**⚠️ SASS 门禁修正（NVIDIA 官方确认）**：SM12.x 的 5 代 Tensor Core NVFP4 用 **`mma.*` 指令族**（ISA 兼容命名），**不是 tcgen05**（那是 SM10.x）。
→ 门禁从 `grep tcgen05` 改为 **`grep -iE "mma.*e2m1|mmaf|mma.*fp4"`**。

---

## 二、接口差异与适配点（我们的格式 → 内核期望）

| 项 | 我们（转换器/内核现状） | 内核期望 | 适配动作 |
|---|---|---|---|
| W 打包 | `W_packed [K, N//2]`（N 向，低半字节=偶 N 列） | `B [n, k/2]`（**K 向**，每字节 2 e2m1） | 主机侧 repack：`[K,N//2] → [N,K//2]`（解包→重打包，每层一次缓存） |
| W scale | `W_scale [K//32, N//128]` uint8 **E8M0** | `B_sf [n, k/16/4]` fp8 **e4m3** swizzled | ① E8M0→e4m3：`e4m3(2^(b-127))`（2 的幂在 e4m3 精确，无损）② 16 分组重复 ③ 4/int32 swizzle + 128 行 pad |
| A 打包 | v15 输出 dequant fp32 / v16 输出 fp8 e4m3 | `A [m, k/2]` K 向打包 e2m1 | **新 A 量化 kernel**：输出 K 向打包 nibble + A_sf swizzled（对齐 nvfp.ops.scaled_fp4_quant 格式） |
| A scale | 32 分组 E8M0 uint8 | A_sf e4m3 swizzled（每 16 K 元素） | 同上转换 |
| 输出 | fp32 | bf16/fp16 | `D.float()`（fp32 累加 → bf16 存储 → float 还原，精度 ~0.4% 可接受；或评估直接 bf16 喂下一层） |
| alpha | 1.0（无全局 scale） | fp32 标量 | 传 1.0（input_global_scale 预留） |

---

## 三、集成步骤（完整 vLLM 源码环境）

```bash
# 1) 完整 vLLM 源码（0.26 版本线）+ CUTLASS（vLLM 捆绑 3.x 即可，非裁剪子集）
git clone -b v0.26.x https://github.com/vllm-project/vllm.git
# 2) 构建 fp4 csrc（确认 FP4_ARCHS 含 12.1a）
cd vllm && python setup.py build_ext --inplace   # 或 cmake 构建
# 3) SASS 门禁（改 mma.*e2m1，非 tcgen05）
nvdisasm $(find . -name "*nvfp4*.so" | head -1) | grep -iE "mma.*e2m1|mmaf" | head
# 4) 验证
python - <<'PY'
from vllm._custom_ops import cutlass_scaled_fp4_mm
# ... 按 §四 适配层喂入
PY
```

**风险提示**：
- **编译期 dispatch bug**（PR #26793 实证）：`nvfp4_scaled_mm_entry.cu` 用预处理器选 SM100/SM120——**双架构编译时可能错选 SM100 路径**。集成时确认只编 SM12x 或改用 runtime dispatch（参考 `scaled_mm_entry.cu` 模式）
- FlashInfer 0.6.8+ 的 CUTLASS NVFP4 backend 已大幅改进（SM12x 优化 kernel 已入 TOT，vLLM 集成 PR 开放中）——**更新、更快，可作为方案 B 备选**（`FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"` JIT 双 arch，E2M1 走硬件，DGX Spark 实测 65 tok/s OOTB）

---

## 四、适配层设计（vLLM 集成骨架）

```python
# ① A 量化 kernel（Triton，输出内核期望格式）—— 对齐 nvfp.ops.scaled_fp4_quant
#    A [m,k] fp32 → A_q [m, k/2] uint8（K 向打包，低半字节=偶 K）+ A_sf [m, k/16/4] fp8e4m3 swizzled
#    复用 v16.1 量化逻辑（32 组 E8M0 → strict> 阈值链）→ 输出格式改造

# ② W 主机侧预处理（每层一次，缓存）—— 对齐内核期望
def preprocess_weights(W_packed, W_scale):  # W_packed [K,N//2], W_scale [K//32,N//128]
    # repack: [K,N//2] N 向 → [N,K//2] K 向
    B_q = repack_n2k(W_packed)              # [N, K//2] uint8
    # scale: E8M0 → e4m3, 32→16 分组, swizzle 4/int32, pad 128
    B_sf = scale_to_swizzle_e4m3(W_scale, N, K)  # [round_up(N,128), K/16/4] fp8e4m3
    return B_q, B_sf

# ③ GEMM 调用
def nvfp4_mmaf_gemm(A, W_packed, W_scale, bias=None):
    A_q, A_sf = quant_a(A)                          # Triton kernel
    B_q, B_sf = preprocess_weights(W_packed, W_scale)
    D = cutlass_scaled_fp4_mm_sm120a(A_q, B_q, A_sf, B_sf, alpha=1.0)  # bf16
    out = D.float()
    if bias is not None: out += bias
    return out
```

**vLLM 挂载**：`QuantizationConfig("nvfp4_4w4a_sm121")` + `FusedMoEMethodBase`，M 阈值分派（decode→b12x / prefill→mmaf）；A 量化 kernel 与 GEMM 均 CUDA Graph 可捕获。

---

## 五、验收标准

| 项 | 标准 |
|---|---|
| 正确性 | 8/8（与 v11/v15 torch 参考数值 ≤5e-2） |
| SASS | `grep -iE "mma.*e2m1|mmaf"` ≥1（原生 FP4，杜绝 bf16 降级） |
| 性能 | **≥ 200 TFLOPS**（v15 的 2.5×）；冲 400（GB10 FP4 500 的 80%） |
| 对照 | 与 v15（bf16）同 harness A/B；≥1.5× 才值得切换 |

## 六、决策建议

1. **首选 FlashInfer 0.6.8+ NVFP4 CUTLASS backend**（NVIDIA 官方背书、SM12x 优化 kernel 已集成、双 arch JIT 走硬件 E2M1）——若 vLLM 版本支持直接启用
2. 备选 vLLM 原生 `cutlass_scaled_fp4_mm_sm120a`（PR #21309/#42209，需完整源码 + 注意 dispatch bug）
3. 自研 `.cu`（round12 B1/B2/B3 阻塞）**放弃**
