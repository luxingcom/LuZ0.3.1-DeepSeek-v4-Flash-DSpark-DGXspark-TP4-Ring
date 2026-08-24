# 路线 B：FlashInfer FP4 —— 可行性结论（生产实测）

> 版本 v2026-08-20 | DGX Spark GB10 / sm_121a / flashinfer 0.6.15 / vLLM 0.26

## 结论

**路线 B（FlashInfer FP4）在当前生产容器的 flashinfer 0.6.15 wheel 上不可直接运行**，作为**备用路径**记录；**主路径采用路线 A（vLLM 内置 cutlass_scaled_fp4_mm）**，已打通（见 kernel1/README.md）。

## 已核实事实（生产实测，2026-08-20）

### FlashInfer 0.6.15 API 面（完整但 backend 阻塞）
```python
import flashinfer
# API 齐全：
flashinfer.mm_fp4(a, b, a_descale, b_descale, alpha, block_size=16, backend=...)
flashinfer.grouped_mm_fp4(...)
flashinfer.mm_bf16_fp4(...)          # W4A16（bf16 激活 × fp4 权重）
flashinfer.nvfp4_quantize(a, a_gsf, sf_vec_size=16)
flashinfer.nvfp4_block_scale_interleave(sf)
flashinfer.nvfp4_attention_sm120 / nvfp4_kv_quantize   # kernel② 消费端
```

### `mm_fp4` 四 backend 全阻塞（逐一实测）
| backend | 结果 |
|---|---|
| `b12x` / `cute-dsl` | `RuntimeError: CuTe DSL is not available`（NGC vLLM wheel 未编入 CuTe DSL 扩展） |
| `trtllm` | `BackendSupportedError: capability 121 不支持` |
| `cudnn` | `RuntimeError: aDesc 描述符错误`（布局不符） |
| `cutlass` | `TypeError: TVM ffi 参数类型错误` |

→ **该 wheel 未编译入 sm_121 可用的 FP4 原生 backend**。

### 已厘清的契约（供后续启用时参考）
- `nvfp4_quantize` 输入需 **fp16/bf16**（非 fp32）
- `mm_fp4` b 布局：`a.shape[1]==b.shape[0]` → **b 为 [K//2, N]**（K 打包，行=K 对）
- descale 需 uint8（e4m3 位模式）或 bf16

## 路线 B 可用化路径（如需启用）

1. **升级 FlashInfer** 到含 sm12x NVFP4 优化 kernel 的 TOT/最新版（NVIDIA 官方背书，双 arch JIT `FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"` 走硬件 E2M1）
2. **源码构建**（需完整 CUTLASS ≥3.9 + cmake；注意 github 限速与容器缺 torch C++ 头）
3. 重编 wheel 含 `cute-dsl`/`b12x` 扩展

## 决策

- **主路径**：**路线 A（vLLM 内置 cutlass_scaled_fp4_mm）** —— 已打通，8/8 正确性 rel 0.00141、60~187 TFLOPS、原生 FP4
- **路线 B**：记录为**备选**，待 FlashInfer 升级后可作为对照/冗余路径评估

## 参考

- 方案文档：交付包 `kernel1_planB_pr42209_integration.md`（§决策建议已指出 FlashInfer 为首选，但需升级 wheel）
- 本地实现：`flashinfer.mm_fp4` 官方 API / vLLM `nvfp4/flashinfer.py` kernel