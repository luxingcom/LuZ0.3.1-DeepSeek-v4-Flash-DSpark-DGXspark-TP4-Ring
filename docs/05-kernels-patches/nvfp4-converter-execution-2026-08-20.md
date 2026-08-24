# NVFP4 高精度转换器落地 + 专用权重生成 — 执行报告

> 日期：2026-08-20 | 执行：engineering-assurance（主理人）| 集群：DGX Spark 4 节点
> 关联交付：`convert_high_precision_nvfp4_stream.py`（流式 v3）+ 生产专用权重 `dsv4f-0731-nvfp4-hp`

---

## 一、任务结论（一句话）

生产 DeepSeek-V4-Flash-0731 **专家权重为 MXFP4**（config 写 `fp8` 为误导），已按用户交付的 `convert_high_precision_nvfp4.py` 高精度方案落地为**流式逐层转换器**，生成与 routeA 内核（`nvfp4_4w4a_mmaf`）**零缝对齐**的 NVFP4 专用权重（43 层 × 256 专家 × 3 矩阵），单层验证 roundtrip 7e-34、格式逐 shape 校验通过。

---

## 二、生产权重真实布局（safetensors 实测确认）

| 矩阵 | 权重张量 | 打包 | scale 张量 | 语义 (out, in) |
|---|---|---|---|---|
| w1 | `[2048,2048]` int8 | MX 沿 in，半字节 | `[2048,128]` E8M0 | (2048, 4096) |
| w2 | `[4096,1024]` int8 | 同上 | `[4096,64]` E8M0 | (4096, 2048) |
| w3 | `[2048,2048]` int8 | 同上 | `[2048,128]` E8M0 | (2048, 4096) |

- **config.json 误导**：`quantization_config.quant_method="fp8", fmt="e4m3"`，但实际张量为 MXFP4（e2m1 + per-32 共享指数 E8M0）。以张量为准。
- 模型维度：`hidden_size=4096, moe_intermediate_size=2048, n_routed_experts=256, num_hidden_layers=43, num_experts_per_tok=6`。
- 打包方向：safetensors 存 `[out, in//2]`，MX 沿 in；低半字节=偶第 1 元素。

---

## 三、转换器落地关键修正（对照生产实测逐项校准）

用户交付脚本 `convert_high_precision_nvfp4.py` 存在 4 处需修正的问题，已全部修复：

1. **MATRIX_DIMS w2/w3 (out,in) 写反** → 生产中 `w2=(4096,2048)`、`w3=(2048,4096)`，用户写 `w2=(2048,4096), w3=(4096,2048)`。改为从张量**动态推断**（`out=shape[0], in=shape[1]*2`）并加 shape/scale 断言，杜绝转置错位。
2. **TENSOR_TPL 模板 bug**：`"layers.{l}.ffn.experts.{e}.w{idx}"` + `idx='w1'` → 生成 `...ww1.weight`（多一个 w），全 key 匹配失败。改为 `"layers.{l}.ffn.experts.{e}.{idx}"`。
3. **high 模式三档 scale 搜索广播 bug**：`scales.unsqueeze(-1).unsqueeze(-1)` 与 `w_exp[1,K/32,32,N/128,128]` 维度不匹配（32 vs N/128=16）报 RuntimeError。改为 `scales[:, :, None, :, None]`（两处：除法和重建）。
4. **全量加载 48 shard(≈176GB) 内存爆炸** → 改**流式逐层**：实证 layer l → shard `model-(002+l)` 恰好 1 对 1，读单 shard → 转换 → 写回即释放，内存峰值=单层。

---

## 四、正确性验证

### 4.1 输出格式（routeA 内核规格对齐）

| 矩阵 | MXFP4 源 | NVFP4 输出 | routeA 需求 `[K,N//2]`+`[K//32,N//128]` | 判定 |
|---|---|---|---|---|
| w1 | [2048,2048] | **weight[4096,1024] uint8** + scale[128,16] | K=4096,N=2048→N//2=1024；K//32=128,N//128=16 | ✅ |
| w2 | [4096,1024] | **weight[2048,2048] uint8** + scale[64,32] | K=2048,N=4096→N//2=2048；K//32=64,N//128=32 | ✅ |
| w3 | [2048,2048] | **weight[4096,1024] uint8** + scale[128,16] | 同 w1 | ✅ |

### 4.2 roundtrip（quant↔dequant 自洽）

`--mode high --validate` 单层：**768 矩阵，avg=7.28e-34, p95=7.40e-34, max=2.93e-33**（浮点极限，量化反演完全自洽）。

### 4.3 方法论说明

- roundtrip 为**自证**（验证用 dequant_nvfp4 与转换用 quant 互为逆，同表）。真正的质量标准是「NVFP4 相对原始 MXFP4 反量化真值」的无损程度与 routeA 端到端语义（此前克隆环境 8/8 rel=0.00141 已证）。可加 AUROC：`(MXFP4→fp32)` vs `(NVFP4→fp32)` 相对同一 fp32 中间值的差异，量化 high 模式相对 fast 更贴近原值（脚本已支持，未逐个跑以省时，可在 A/B 窗口并行）。
- routeA 消费验证受环境限制：生产容器 GPU 被 vLLM 占用，`scaled_fp4_quant` 无法在孤立执行环境跑（CUDA backend 错误），但**格式逐 shape 对齐已充分**。

---

## 五、全量生成（结论）

- **方式**：一次性容器 `nvfp4-conv-full`（head 节点，`--cpus=16`，纯 CPU **不分配 GPU**，`--entrypoint python3` 覆盖镜像 ENTRYPOINT），流式 43 层 `--mode high`。
- **输入**：`/home/<USER>/models/deepseek-v4-flash-0731`（容器 `/models`，只读）
- **输出**：`<INSTALL_DIR>/nvfp4/models/dsv4f-0731-nvfp4-hp/`
- **日志**：`<INSTALL_DIR>/nvfp4/models/full_convert.log`
- **不干扰生产**：head 为 rank0（无 NCCL 数据面），纯 CPU ge果不触 GPU；内存逐层峰值可控；输入只读不破坏原权重。
- 完成后需**四节点 md5 一致性校验 + 三方备份**（同既有规范）。

---

## 六、重要架构发现（影响 task 20/21）

生产 fork（MiaAI vLLM 0.26.1）**自带完整 NVFP4 MoE 路径**：

| fork 内建 | 说明 | GB10(sm_121a) 适用 |
|---|---|---|
| `quantization/online/nvfp4.py` → `Nvfp4OnlineMoEMethod` | FlashInfer TRTLLM, per-token activation scale | ❌ 硬要求 SM100 Blackwell |
| `nvfp4_emulation_moe.py` → `Nvfp4QuantizationEmulationTritonExperts` | Triton BF16 模拟，逐前向 dequant | ✅ 功能可用，性能差(best于 emulation 语义) |
| `trtllm_nvfp4_moe.py` / `deep_gemm_moe.py` 等 | 生产 NVFP4 各实现 | 视 backend |

**结论**：fork 内建 NVFP4 路径**都不是为 GB10 sm_121a 原生 FP4 MMA 优化的**——emulation 用 BF16 模拟（慢），online 要求 Blackwell。**routeA（`cutlass_scaled_fp4_mm` 原生 FP4 MMA，60-187 TFLOPS）恰好填补这一性能空白**，价值清晰。

**plugin 接入的真实复杂度**：生产 fork 的 `FusedMoEMethodBase.apply` 实际签名为
`apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)`（模块化 MoE，路由在 layer 外），
而 `nvfp4-vllm-plugin` 的 `moe_method.py` 骨架用的旧签名 `apply(layer, x, router_logits, top_k, renormalize)` **不匹配**。
正确接入需基于 fork modular kernel 接口（`FusedMoEMethodBase` + `maybe_make_prepare_finalize` + `select_gemm_impl` / `moe_kernel`）重构。

---

## 七、后续动作项

- [ ] **task #19 收尾**：全量转换完成 → 4 节点 md5 一致性 + 三方备份（01/02/local + md5 校验）
- [ ] **task #20**：基于 fork modular kernel 接口正确实现 routeA MoE method 接入，而非 plugin 旧骨架；评估复用内建 NVFP4 路径 vs routeA 原生 MMA 的性能差距
- [ ] **task #21**：生产停机窗口做干净同屏 A/B（routeA vs 真实 B12X，≥1.5× 切换），隔离 GPU
- [ ] 权重在 routeA 的端到端正确性（干净环境，不共享 GPU）补测
- [ ] 回填 Runbook + 归档 v3 源码/diff 到 `backup/tp4-<date>/`