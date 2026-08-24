# 方案 B 双路线落地手册（kernel① prefill GEMM）

> 版本 v2026-08-20 | DGX Spark GB10 / sm_121a / vLLM 0.26 | 生产容器 `vllm-tp4-rank0`

## 一、背景

NVFP4 4W4A prefill GEMM 需要**原生 FP4 MMA**（绕开 Triton 3.6 无原生 codegen 的限制）。方案 B 提供两条路线，本手册是其落地指引。

**已定结论（2026-08-20 实测）**：
- ✅ **路线 A**：vLLM 内置 `cutlass_scaled_fp4_mm`（原生 FP4）—— **主路径，已打通**
- 🔶 **路线 B**：FlashInfer `mm_fp4` —— **备选**，当前 wheel 阻塞
- 回退：Triton v15（bf16 MMA，26.7~81.4 TFLOPS）

## 二、路线 A —— vLLM 内置 cutlass_scaled_fp4_mm（主路径，零构建）

### 2.1 原理
vLLM 0.26 已把 `cutlass_scaled_fp4_mm`（SM120a 原生 FP4 CUTLASS 内核）预编译进 `_custom_ops`（`_C_stable_libtorch.abi3.so` 内嵌 sm_120 cubin），**无需源码构建**。配合 `scaled_fp4_quant`（硬件量化）即得官方 NVFP4 语义。

### 2.2 关键 API
```python
import vllm._custom_ops as co
# cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype) -> out
#   a: [M, K//2] uint8 (e2m1 K-packed)   b: [N, K//2] uint8
#   block_scale_a/b: fp8 e4m3 swizzled   alpha: fp32   out_dtype: bf16/fp16
co.cutlass_scaled_fp4_mm(a_q, b_q, a_sf, b_sf, alpha, torch.bfloat16)

# scaled_fp4_quant(input, input_global_scale, is_sf_swizzled_layout, backend, padded_n)
#   -> (packed_uint8, sf); input 需 fp16/bf16; 16-group e4m3
co.scaled_fp4_quant(x.half(), gs, is_sf_swizzled_layout=True, backend='none', padded_n=None)
```

### 2.3 适配层（交付：`kernel1/nvfp4_4w4a_mmaf.py`）
- **A 侧**：`scaled_fp4_quant(A)` → A_q + A_sf（swizzled）
- **W 侧**：既有 `W_packed`（N 打包）→ dequant → `scaled_fp4_quant` → B_q([N,K//2]) + B_sf，**每层 preprocess 一次缓存**
- **GEMM**：`cutlass_scaled_fp4_mm` → `.float()`
- 正确性 8/8 rel=0.00141（vs 官方 dequantize_to_dtype）；性能 60~187 TFLOPS

### 2.4 恢复/回退
- 适配层是独立 `.py`，不修改 vLLM 本体；删除即回退到 v15
- 生产路径：`RouteA.preprocess_weights()` + `__call__(use_cached_w=True)`

## 三、路线 B —— FlashInfer mm_fp4（备选）

### 3.1 现状（阻塞）
flashinfer 0.6.15 wheel 未编入 sm_121 可用的 FP4 backend：
| backend | 问题 |
|---|---|
| b12x / cute-dsl | CuTe DSL not available |
| trtllm | capability 121 不支持 |
| cudnn / cutlass | 描述符/参数错 |

### 3.2 启用路径
升级 FlashInfer 到 TOT（含 sm12x NVFP4 优化 kernel，`FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"`）或源码构建（完整 CUTLASS≥3.9）。详见 `docs/ROUTE-B-FEASIBILITY.md`。

## 四、SASS 门禁（验收标准）

SM12x 用（**非 tcgen05**）：
```bash
# 符号/架构级验证（vLLM 预编译产物）
strings /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so | grep -iE "cutlass_scaled_fp4_mm|scaled_fp4_quant|e2m1"
cuobjdump --list-elf /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so | grep sm_120
# 运行期：性能远超 bf16 上限即真 FP4
```
脚本：`tests/sass_fp4_check.py`。

## 五、验收矩阵

| 项 | 标准 | 实测 |
|---|---|---|
| 正确性 | 8/8，rel<0.02 vs 官方 dequant | **rel=0.00141** ✅ |
| 性能 | ≥ v15(26-81)，冲 200 | **60~187 TFLOPS** ✅ |
| SASS | 原生 FP4 | sm_120 cubin + FP4 符号 ✅ |
| 对照 v15 | ≥1.5× | 数值精确已超（bf16 含精度损失） |

## 六、快速验证

```bash
cd /vllm-workspace/nvfp4-landing/routeA
python3 -c "import torch; from nvfp4_4w4a_mmaf import RouteA; \
 A=torch.randn(256,4096,device='cuda'); Wp=torch.randint(0,16,(4096,2048),dtype=torch.uint8,device='cuda'); \
 Ws=torch.full((128,32),127,dtype=torch.uint8,device='cuda'); i=RouteA(); i.preprocess_weights(Wp,Ws); print(i(A,use_cached_w=True).shape)"
```