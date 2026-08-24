# NVFP4 权重 × 推理运行时 × SM120/SM121 支持调查（vLLM 之外）

**日期**：2026-08-13 ｜ **状态**：社区调查完成 ｜ **前提**：vLLM 侧 NVFP4 在 SM121 路径已否（CUTLASS grouped GEMM 垃圾输出 + Marlin 兜底性能不足）

---

## TL;DR

| 运行时 | NVFP4 支持 | SM120/SM121 可用 | DeepSeek-V4-Flash 适配 | 结论 |
|---|---|---|---|---|
| **SGLang** | ✅ 原生（NVIDIA 官方容器明示支持 DGX Spark） | ✅ | ✅（主线有 DSV4 NVFP4 MoE 方法；官方 NVFP4 版需 PR #25820） | **第一候选 ★★★** |
| **TensorRT-LLM** | ✅ 原生（官方 Spark 矩阵大量 NVFP4 ✅） | ✅（PR #11997 放开 SM120/121） | ❓ 不在官方 Spark 矩阵 | 支持但性能口碑差 ★★ |
| **llama.cpp** | ✅（GGML_TYPE_NVFP4=40，sm_120/121a 原生 FP4 MMQ） | ✅ | ❓ 架构支持需核实；需 NVFP4→GGUF 转换 | 单机/基线可用 ★★ |
| **DeepGEMM** | ✅（DSV4 官方推荐后端） | ❌ **SM100-only**（tcgen05/TMEM 依赖） | — | **不可用 ✗** |
| **HF Transformers** | ⚠️ 可加载（grouped_mm/batched_mm 消费 float32 scales） | ⚠️ 无 FP4 硬件加速 | ✅ | 正确性基准用 |
| Ollama / LM Studio | ✅（继承 llama.cpp NVFP4 GGUF） | ✅ | 同 llama.cpp | 慢，非生产 |

**核心结论**：vLLM 之外，**SGLang 是唯一在 SM120/SM121 上同时满足"原生 NVFP4 内核 + 官方支持声明 + DeepSeek-V4 架构适配"的运行时**。TensorRT-LLM 官方支持但 Spark 上实测性能垫底；llama.cpp 是可行的单机基线但 decode 无加速；DeepGEMM（DeepSeek 自家推荐后端）因依赖 SM100 独有特性在 Spark 上完全不可用。

---

## 1. SGLang（第一候选 ★★★）

### 1.1 NVIDIA 官方声明（最强证据）
NVIDIA SGLang 容器 **26.02 release notes**（CUDA 13.1.1.006 / SGLang 0.5.8）：

> Support NVIDIA innovative 4-bit floating point **NVFP4 format on Blackwell GPUs (including Jetson Thor and DGX Spark)**, which provides better training and inference performance with lower memory utilization. Supported for DeepSeek-R1, Llama-3.1-8B-Instruct.

→ NVIDIA 官方容器层面明确 NVFP4 × DGX Spark 支持，DeepSeek-R1 已验证。

### 1.2 实现路径（SGLang 主线源码实证，DeepWiki）
- DeepSeek-V4 MoE 专家量化走 **ModelOptNvFp4FusedMoEMethod**（NVFP4 = NVIDIA E2M1 + FP8 E4M3 块缩放，16 元素块）
- NVFP4 GEMM 经 **flashinfer_fp4_gemm**（CUTLASS + TRTLLM 双 kernel 后端），FlashInfer 不可用时 fallback 自定义 fp4_gemm
- **SM100 与 SM120 特性动态检测**（is_sm100_supported 等）——不是 SM100-only 的硬编码
- **cast_e2m1fn_to_e4m3fn 无损转换**：DeepSeek-V4 FP4 专家可无损转 FP8 走高效 FP8 内核（数值保持 + 性能双收）
- DeepEP token dispatch + FP4 输入；silu_and_mul_clamp 融合
- 权重 padding：CUTLASS 32 对齐 / TRTLLM 128 对齐（pad_nvfp4_weight）

### 1.3 SM120 专项支持
- **PR #24692**（SM120 Blackwell Desktop support for DeepSeek-V4）：新增 `mxfp4_moe_sm120_triton.py`（融合 FP4 去量化+GEMM，E2M1 半字节 LUT 解码 + 按组缩放，适配 99KB SMEM）+ `flash_mla_sm120_triton.py`（稀疏 MLA 解码）；PR 前 SM120 完全跑不了 DSV4（DeepGEMM JIT 崩溃）
- DeepGEMM 在 SM120 需禁用：`SGLANG_DISABLE_DEEP_GEMM=1` / `SGLANG_ENABLE_DEEP_GEMM=0` → CUTLASS fallback

### 1.4 DeepSeek-V4-Flash 具体适配
- NVIDIA 官方 **nvidia/DeepSeek-V4-Flash-NVFP4**（ModelOpt 0.44.0 量化，`moe_quant_algo: NVFP4`，权重以 FP8 存储）：官方文档声明 **SGLang 与 vLLM 均可推理**，SGLang 侧需 **PR #25820**（是否已合入主线需核实）
- 官方 0731 原版（MXFP4）SGLang 参考配置：`--moe-runner-backend flashinfer_mxfp4 --speculative-algorithm DSPARK`
- NVIDIA Spark playbook（build.nvidia.com/spark/sglang，2026-07-31 更新）：Spark 已验证矩阵含多个 NVFP4 模型（启动加 `--quantization modelopt_fp4`）；**DeepSeek-V4-Flash 列在 DGX Station 页而非 Spark 页**——官方未在 Spark 矩阵中验证 DSV4-Flash，需自行实测

### 1.5 已知坑
- gpt-oss 家族在 Spark/Jetson 因 OpenAI Triton issue 无法运行（与我们无关）
- Nemotron 系列需预装 flashinfer-jit-cache
- DeepEP 主要为 NVLink 优化：四机无 NVLink，跨机 EP 走 NCCL fallback，**TP/EP 通信行为需在环网上实测**（与 vLLM 的 NCCL 环境不同栈）

---

## 2. TensorRT-LLM（官方支持，性能口碑差 ★★）

### 2.1 支持证据
- **官方 Spark playbook**（build.nvidia.com/spark/trt-llm）：DGX Spark 支持矩阵大量 NVFP4 模型 ✅——Nemotron-3-Super-120B NVFP4、Nemotron-3-Nano-Omni-30B-A3B NVFP4、Llama-3.3-70B NVFP4、Qwen3-32B/14B/8B NVFP4、Phi-4 系列 NVFP4、Qwen3-235B-A22B（双 Spark）NVFP4 等
- **PR #11997**（2026-03-07）：Ungate fused MoE for SM120/SM121——NVFP4 fused MoE 的 SM 检查从 {100,103} 扩到 {100,103,120,121}；底层 CUTLASS 已有 `nvfp4_nvfp4_gemm_template_sm120.h`；CuTE DSL NVFP4 dense GEMM 同步放开
- 官方 NVFP4 量化 playbook（2026-07-21 更新）：Model Optimizer 0.45.0 recipe 量化 MoE → NVFP4 checkpoint（W4A16 交互 / W4A4 高并发）

### 2.2 社区实测（负面）
- NVIDIA 论坛实锤：**TRT-LLM + Llama-3.3-70B-NVFP4 = 5 tok/s**（单 Spark decode）；另一报告 2.5 tok/s；同机 LM Studio GGUF Q4_K_M 4.6-4.9 tok/s——"NVIDIA 自家 NVFP4 模型在自家 TRT-LLM 上比非 NVIDIA 量化慢"
- Nemotron-3-Super-120B NVFP4（官方明确可部署 Spark）：19-22 tok/s = 带宽上限 45 tok/s 的 42-48%，社区定性"kernel/软件栈未利用 FP4 tensor core"
- DeepSeek-V4-Flash **不在**官方 Spark 矩阵

---

## 3. llama.cpp（单机/基线可用 ★★）

### 3.1 支持证据
- **GGML_TYPE_NVFP4 = 40** 合入 llama.cpp（2026-03~04 PR 序列：#20644 dp4a kernel、#21074 generic NVFP4 MMQ、#21227 SYCL、#21455/#21539 Vulkan、**#22196 Blackwell-native NVFP4**）
- **build b8967（2026-04-29）**：原生 NVFP4 MMQ 启用 sm_120；RTX 5090 实测 prefill +43~68%（avg ~57%），decode 无变化
- **DGX Spark playbook**（vlaicu.io）：sm_121a 构建（`BLACKWELL_NATIVE_FP4=1`）跑 NVFP4/MXFP4 GGUF；decode 带宽 bound 无增益，prefill compute-bound 有加速——与我们的 memory-latency-bound 结论一致
- 转换：compressed-tensors/modelopt NVFP4 safetensors → GGUF（convert script PR #21095）

### 3.2 限制
- 需先转 GGUF（safetensors 不能直跑）
- **DeepSeek-V4-Flash 架构支持未确认**：llama.cpp 支持 DSV3.2，V4-Flash（33K 专家、MLA、DSPARK MTP）是 2026 新架构，社区提到"值得在 DSV4-Flash 上尝试"但无官方支持声明
- 四机分布式：llama.cpp 无跨机 TP/EP（单机多卡 rpc 仅雏形）→ 只能单 Spark，无法利用四机内存池

---

## 4. DeepGEMM（✗ 不可用——SM100-only）

- DeepSeek V3/V4 release notes 推荐后端；依赖 **tcgen05 + TMEM**（SM100/103 数据中心特性）
- **SM120 无 kernel image**（JIT 只出 100a/ 目录）；报 "no kernel image" 或静默失败
- 0xsero.github.io Blackwell GPU Wiki：移植需把 tcgen05.mma 改写为 mma.sync 链 + TMEM→寄存器/SMEM + tile 缩小适配 99KB + 禁 cluster_dim>1——"substantial rewrite"，社区移植进行中未完成
- HF Transformers `experts_implementation="deepgemm"` 同样要求 SM100+；SM120 上只能选 `grouped_mm`/`batched_mm`（消费 float32 块缩放，无 FP4 硬件路径）
- 陷阱：NVFP4（block-16，FP8 E4M3 scale）与 MXFP4（OCP 标准，block-32，FP6 E3M2 scale）布局不同，混用产出 silent garbage

---

## 5. 其他

- **HF Transformers**：可加载 NVFP4 权重（grouped_mm/batched_mm 消费 float32 scales）做正确性/精度基准，但无 SM120 FP4 硬件加速——**非性能路径**
- **Ollama / LM Studio**：继承 llama.cpp NVFP4 GGUF 支持；LM Studio 在 Spark 实测 70B ~4.6-4.9 tok/s——慢，仅桌面演示
- **MLC-LLM / TVM**：未发现 SM120 NVFP4 明确支持证据，不推荐
- **vLLM**（对照，已被否）：SM121 上 CUTLASS grouped GEMM 垃圾输出（CUTLASS #3096）、Marlin W4A16 兜底、FlashInfer b12x 已集成但用户评估不可行

---

## 6. 对本集群（DGX Spark TP4）的落地建议

1. **首选验证 SGLang**：主线 NVFP4 + SM120 动态检测 + DSV4 架构方法齐全。落地序列：
   - 核实 **PR #25820** 是否已合入主线（NVIDIA 官方 DSV4-Flash-NVFP4 支持的前提）
   - NGC 26.02 容器（CUDA 13.1.1 + SGLang 0.5.8 + flashinfer 0.6.1）或主线自建镜像
   - 四机 TP4/EP：注意 DeepEP 跨机走 NCCL（无 NVLink），环网通信行为需实测；沿用现有 NCCL 环境变量经验
   - MoE 后端：NVFP4 → flashinfer_fp4_gemm 路径；若走官方 0731 MXFP4 → flashinfer_mxfp4 + SM120 Triton kernels（PR #24692）
2. **llama.cpp 作单机对比基线**：sm_121a 构建 + NVFP4→GGUF，先确认 DSV4-Flash 架构可加载；只测单机吞吐/延迟，不做四机
3. **TensorRT-LLM 不建议**（官方支持但 Spark 性能垫底；DSV4 不在矩阵）
4. **DeepGEMM 排除**（SM100-only）
5. 收益预期与 vLLM 结论一致：SGLang NVFP4 也主要改善 **prefill/带宽饱和档**；decode latency-bound 单请求提升有限——A/B 口径不变（per-request p50，禁用 agg_* 跨组对比）

---

## 7. 参考链接

- NVIDIA SGLang 容器 26.02 Release Notes（PDF，docs.nvidia.com/deeplearning/frameworks）
- build.nvidia.com/spark/sglang（Spark 模型矩阵，2026-07-31 更新）
- build.nvidia.com/spark/trt-llm（TensorRT-LLM Spark 模型矩阵）
- build.nvidia.com/spark/nvfp4-quantization（Model Optimizer 0.45.0 NVFP4 量化 playbook）
- SGLang PR #24692（SM120 DSV4 支持）、PR #25820（DSV4-Flash-NVFP4，待核实合入状态）
- TensorRT-LLM PR #11997（SM120/121 fused MoE ungated）
- llama.cpp PR #20644/#21074/#22196（NVFP4：dp4a/MMQ/Blackwell-native）；b8967 release
- 0xsero.github.io/blackwell-gpu-wiki/kernels/deepgemm（SM120 移植状态）
- HF Transformers experts_interface 文档（deepgemm/grouped_mm/batched_mm）
- NVIDIA 论坛：NVFP4 on DGX Spark broken（367082）、Llama-3.3-70B TRT-LLM 5 tok/s
- insiderllm.com FP4 in llama.cpp guide（2026-04-29 更新）
- vlaicu.io DGX Spark + LlamaCPP Playbook（sm_121a / BLACKWELL_NATIVE_FP4）

> 时效性说明：2026-08-13 时点。SGLang/llama.cpp/TensorRT-LLM 均在快速迭代（NVFP4×SM120 是社区最活跃方向之一），建议落地前复核：SGLang PR #25820 合入状态、llama.cpp DSV4-Flash 架构支持、CUDA 13.3+ 后 cuBLASLt grouped GEMM 支持矩阵变化。
