# routeA 生产导入：权重适配裁定 + 转换器验证 + 集成设计（2026-08-21）

**任务**: Task #20 · routeA 权重适配裁定 + 转换器构建 + 集成设计
**执行**: Archi（系统架构师）· node01 一次性容器（`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`，生产镜像，torch 2.11/cu13.0，GB10 sm_121）
**产出**: 本报告 + `routea_weight_adapter.py`（适配器，已验证）+ `routea_validate_layer0.py`（layer-0 判决脚本）+ 12 个 probe 脚本与原始输出（`_routea_work/`，本地与 01:/tmp/_routea_work/）

> **一页结论（供决策）**
> 1. **权重契约已闭环（好消息）**：生产 -0731 MXFP4 expert 权重对 routeA（cutlass NVFP4）**payload 零拷贝直配**——E2M1 打包字节完全兼容，仅 scale 需 E8M0→E4M3 扩展，且该转换对全模型 9.26GB scale（35,328 张量，字节范围 [118,126]）**逐值精确**。适配器在真实 layer-0 权重上验证 **rel=1.41e-03**（判据 1e-2），比 routeA 原 preprocess（requant，rel=7.84e-02）精确 56×。派生成本 ~0.2ms/expert、+4.33GB/rank（43 层）。
> 2. **性能裁定（决定性坏消息）**：routeA 的唯一可执行形态——per-expert cutlass GEMM 循环——在真实 MoE 调度几何下（topk=6 / 256 experts / `--max-num-batched-tokens 4096` → 平均 M_e≈96）**全程慢于生产 B12X W4A16 2.4–5.2×**（M=1024..16384 实测 0.19–0.42×）。克隆基准的 155–313 TFLOPS 只在 M_e≥1536（即单 chunk ≥6.5 万 token）出现，生产 prefill 上限 4096 根本到不了。**fork 内已有的 cutlass 分组 W4A4 MoE（cutlass_fp4_group_mm）在 sm_121 实测可用但同样慢 1.2–1.9×。**
> 3. **路径裁定**：**否决路径 A（运行时派生 + per-expert routeA 循环）、路径 B（离线转换）、路径 C（-nvfp4 直用）**——三者都服务于已被实测否决的 per-expert 形态。**唯一在 prefill 档跑赢的是 fork 内置的 b12x 原生 W4A4 融合 MoE（B12xMoEWrapper，M=4096 时 1.32×、16K 时 1.52×、M=1 时 6.2×；但 M=64–2048 中段 0.79–0.95×）**。该路径的权重构造链已从 fork 源码逐行核实，但在本构建上的**端到端语义验证未通过**（详见 §4.3），必须先单卡排障，不建议直接占 TP4 窗口。
> 4. **建议**：TP4 停机窗口不做 routeA 接入；把窗口留给 b12x-W4A4 的**单卡离线正确性排障**（1–2 天量级），通过后再谈 TP4 灰度。

---

## 0. 对既有设计资产的两处重要修正

| # | routeA-integration-review-2026-08-20 的论断 | 本次实测修正 |
|---|---|---|
| C1 | 接入基类为 `FlashInferB12xExperts`（NVFP4 oracle 路径），模型为 -nvfp4（ModelOpt NVFP4 checkpoint） | **生产实际加载 -0731（MXFP4）**：`start_tp4_head.sh` 挂载 `<INSTALL_DIR>/models/deepseek-v4-flash-0731:/models:ro`（config 无 moe_quant_algo → `Mxfp4MoEMethod`），`--moe-backend flashinfer_b12x` 在 MXFP4 oracle 映射到 **`B12xExperts`**（`experts/b12x_mxfp4_moe.py`），不是 `FlashInferB12xExperts`。两者是不同类、不同权重链。注意 01 上 `/models` 符号链接指向 -nvfp4（`/home/<USER>/models/deepseek-v4-flash-0731-nvfp4`），与生产容器内挂载无关，但属易混淆隐患 |
| C2 | "group_size=16 冲突"（routeA 假定 32×128 block vs ModelOpt group-16）是 P0 数学风险 | **不成立/不相关**：生产 -0731 scale 是逐行 E8M0 K/32，**不经反量化直接 LUT 扩展**到 E4M3 K/16 即精确（§1.2）。requant 路径（原 review V1 担心的）才是错误来源，已被直配路径取代 |

---

## 1. 权重布局契约实测（任务 §1）

### 1.1 生产 -0731（对接对象，权威复核）

layer-0 expert-0 实测（safetensors header 手工解析，与 P3 报告一致）：

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | [2048, 2048] | U8 | N=2048(inter), K=4096(hidden)，E2M1 打包 **[N, K//2]，低半字节=偶数 K** |
| `layers.0.ffn.experts.0.w1.scale` | [2048, 128] | F8_E8M0 | [N, K//32] 逐行 E8M0：`W[n,k]=e2m1(码)×2^(scale[n,k//32]−127)` |
| w2 | [4096, 1024] / [4096, 64] | U8/E8M0 | N=4096(hidden), K=2048(inter) |
| w3 | 同 w1 | | |

全模型 scale 扫描（48 shards、35,328 张量、9.26GB）：**字节范围 [118, 126]**，即 2⁻⁹…2⁻¹。

### 1.2 routeA/cutlass 消费契约（probe5 实证）

- **payload 零拷贝**：`cutlass_scaled_fp4_mm(a, b, sfa, sfb, alpha, out)` 的 `b` 侧要求 [N, K//2] uint8、沿 K 打包——vLLM `scaled_fp4_quant` 的打包约定实测为**低半字节=偶数 K，与生产字节完全一致**（probe5[C]：构造 even=0.5/odd=6.0 矩阵，量化后字节 0x71 → lo=even）。**生产 payload 无需任何重排/重打包，直接作为 b 传入。**
- **scale 链**：E8M0 [N, K//32] →(LUT 精确转换 + K 维 ×2 扩展)→ E4M3 plain [N, K//16] →(`swizzle_blockscale`)→ 128×4 swizzled。swizzle 公式与 `scaled_fp4_quant` 的 swizzled 输出**逐字节一致**（probe5[B]，经官方 `convert_swizzled_to_linear` 反解交叉验证）。
- **E8M0→E4M3 精确域**：E4M3 可精确表示 2⁻⁹…2⁸（E8M0 字节 118–135）。全模型范围 [118,126] ⊂ 精确域 → **零信息损失**。
- **routeA 原 `preprocess_weights` 否决**：其"反量化→32×128 E8M0 重打包→scaled_fp4_quant 重量化"链在真实权重上 rel=7.84e-02（双重量化损失）；直配路径 1.41e-03。**适配器取代 routeA 的 preprocess，只保留其 GEMM 调用面。**

### 1.3 -nvfp4（modelopt）实测（路径 C 评估用）

`w1.weight [N,K//2] U8` + `w1.weight_scale [N, K//16] F8_E4M3`（**plain，非 swizzled**）+ `weight_scale_2 / input_scale` 标量 F32。即 -nvfp4 就是"routeA 原生格式"的 modelopt 版（P3 已证其反量化与 -0731 逐值一致——正是本适配器所做的精确转换）。**格式上可直接喂 FlashInferB12xExperts**（fork 的 NVFP4 oracle 路径，§4），但换模型源会把量化方法切到 `ModelOptNvFp4Config`，且该路径正确性未验证（§4.3）。

### 1.4 数值判决（任务 §3，deliverable）

`routea_weight_adapter.py` + `routea_validate_layer0.py`（真实 layer-0 expert-0 权重，GPU 容器实测）：

| 用例 | rel | 判据 1e-2 |
|---|---|---|
| w1 全量 N=2048 K=4096 M=1024 | 1.41e-03 | PASS |
| w2 全量 N=4096 K=2048 M=1024 | 1.41e-03 | PASS |
| w3 全量 M=257（奇数 M） | 1.41e-03 | PASS |
| w13 TP4 分片 N=1024 K=4096 M=2048 | 1.41e-03 | PASS |
| w2 TP4 分片 N=4096 K=512 M=2048 | 1.41e-03 | PASS |

误差量级 = routeA 克隆基线（0.00141）本身（fp4 激活量化 + bf16 出舍入），权重侧零损失。CPU 自检（LUT 精确域 / K32→K16 扩展数值等价 / swizzle）全过。适配器纯 torch、无 vLLM 依赖（GEMM 验证除外），CPU 可跑。

---

## 2. 性能裁定（本次最重要的产出）

**基准环境**：生产 TP4 分片形状（w13 [256, 1024, 2048]、w2 [256, 4096, 256]，topk=6，256 experts），同一随机路由/激活，b12x W4A16 = 生产路径原样调用（plan/bind/b12x_moe_fp4，swiglu_limit=10）。

### 2.1 形态一：per-expert routeA 循环（原设想形态）—— **否决**

kernel-only per-expert GEMM（probe5[G]，权重缓存、无 launch 间隙）：

| M_e | gemm1 (N=1024,K=4096) | gemm2 (N=4096,K=512) |
|---|---|---|
| 48 | 18.0 TF | 15.8 TF |
| 96 | 43.0 TF | 31.5 TF |
| 384 | 97.4 TF | 86.4 TF |
| 1536 | 366.4 TF | 82.5 TF |

端到端整层 MoE（probe6b，含分桶/量化/激活/归约）：

| M | 平均 M_e | b12x W4A16 | routeA 循环 | 加速比 |
|---|---|---|---|---|
| 1024 | 24 | 4.88 ms | 25.05 ms | **0.19×** |
| 2048 | 48 | 5.97 ms | 25.17 ms | **0.24×** |
| 4096 | 96 | 10.03 ms | 26.00 ms | **0.39×** |
| 8192 | 192 | 14.64 ms | 37.63 ms | **0.39×** |
| 16384 | 384 | 27.94 ms | 65.81 ms | **0.42×** |

即使零调度开销，纯 GEMM 时间（M=2048：256×(0.022+0.013)=8.96ms）也超过 b12x 整层（5.97ms）。**结构性原因**：topk=6/256 experts 把 M 摊薄到 M_e≈M/43，而 routeA kernel 的效率拐点在 M_e≥768+。生产 `--max-num-batched-tokens 4096` 下不可能到达。

### 2.2 形态二：cutlass 分组 W4A4 MoE（fork 内置 `cutlass_fp4_group_mm`）—— **否决**

oracle 守卫（`cutlass_group_gemm_supported()=False`、`mxfp4_experts_quant_supported(sm121)=False`）之下实测：**NVFP4 分组 GEMM 链（`cutlass_fp4_group_mm` + `scaled_fp4_experts_quant` + fused silu-quant）在 sm_121 可执行**（MXFP4 分组 `cutlass_mxfp4_moe_mm` 未编译，NotImplementedError）。但性能（probe11）：

| M | b12x W4A16 | cutlass W4A4 分组 | 加速比 |
|---|---|---|---|
| 256 | 3.93 ms | 4.81 ms | 0.82× |
| 2048 | 5.93 ms | 9.88 ms | 0.60× |
| 8192 | 14.67 ms | 27.76 ms | 0.53× |

### 2.3 形态三：b12x 原生 W4A4 融合 MoE（`B12xMoEWrapper`，FlashInferB12xExperts 底座）—— **prefill 档唯一赢家，待正确性排障**

b12x 包自带 `quant_mode="nvfp4"`（W4A4，且为其默认值；生产 vLLM 侧硬编码 w4a16）。实测（probe15/16，独立张量、生产分片形状）：

| M | b12x W4A16 | b12x W4A4 | 加速比 |
|---|---|---|---|
| 1 | 0.360 ms | 0.058 ms | **6.17×** |
| 8 | 0.669 | 0.728 | 0.92× |
| 64 | 2.969 | 3.578 | 0.83× |
| 256 | 3.976 | 5.023 | 0.79× |
| 1024 | 4.755 | 5.574 | 0.85× |
| 2048 | 5.945 | 6.241 | 0.95× |
| **4096（生产 prefill 上限）** | **10.065** | **7.614** | **1.32×** |
| 8192 | 14.611 | 11.087 | 1.32× |
| 16384 | 27.855 | 18.286 | **1.52×** |

---

## 3. 权重路径裁定（任务 §2）

| 路径 | 内容 | 裁定 | 理由 |
|---|---|---|---|
| **A** 运行时派生（-0731 保持 + 层加载时派生 routeA 张量） | payload 零拷贝 + E4M3 scale 派生（+4.33GB/rank） | **否决**（服务于 per-expert 形态） | 形态已被 §2.1 实测否决；且若与 B12X W4A16 双驻留，b12x prepare **会就地重打包 payload**（probe6[A] 实证：字节变更、指针保留、multiset 相同），routeA 需再留一份原始 payload → +33GB/rank，不可行 |
| **B** 离线转换 -0731 → routeA 格式副本（~138GB 磁盘） | | **否决** | 同上，转换产物只被否决形态消费；且运行时仍需 B12X decode → 双表示驻留问题相同 |
| **C** -nvfp4 直用 | 格式即 routeA 原生（E4M3 K/16 plain） | **否决（作为 routeA 载体）** | 换模型源 → ModelOptNvFp4Config → NVFP4 oracle；其 flashinfer_b12x 分支 = `FlashInferB12xExperts`（W4A4 wrapper）——**这正是 §2.3 的赢家**，但该路径本构建语义验证未通过（§4.3），且换生产模型源是重大运维变更 |
| **A′（新提）** -0731 保持 + 插件派生 NVFP4 张量 → b12x W4A4 wrapper | E8M0→E4M3 精确派生 + swizzle + mma（本适配器已验证到 swizzle 层） | **唯一值得推进，前置条件=§4.3 排障通过** | M≥4096 prefill 1.32×、M=1 6.2×；中段（M=64–2048）0.79–0.95× 需 M-分派或接受小幅回退 |

**内存预算（A′，TP4 per-rank）**：
- 派生 E4M3 swizzled scale：+101MB/层 × 43 = **+4.33GB**
- wrapper 内部 `convert_sf_from_mma_layout(...).contiguous()` 再持一份同尺寸拷贝：**+4.33GB**
- E8M0 原 scale 可释放（若 W4A16 完全替换）：−2.16GB
- **净增 ≈ +6.5GB/rank**（W4A4 全替换形态；payload 零拷贝——wrapper 不重打包，与 B12xExperts 不同）。
- 若做 M-分派混合（W4A4 大 M + W4A16 中段/decode）：W4A16 侧需要 b12x 重打包副本 → **+33GB/rank，不可行**；替代方案是两个 wrapper 实例（`quant_mode="nvfp4"` + `quant_mode="w4a16", source_format="modelopt"`）共享同一原始 payload——**可行性未验证，列为 A′ 排障的必查项**。

---

## 4. 插件集成设计（任务 §4，基于实测修正）

### 4.1 接口事实（容器内源码核实）

- `Mxfp4MoEMethod.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)` —— 与任务给定签名一致 ✓。内部 `moe_kernel.apply(...)` → experts.apply（routing/prepare/finalize 全在框架内）。
- 注入点：`oracle/mxfp4.py::backend_to_kernel_cls` 的 `"flashinfer_b12x" → [B12xExperts]` 映射（Mxfp4MoEMethod.__init__ 经 `select_deepseek_v4_mxfp4_moe_backend` 取 experts_cls）。插件以 `vllm.general_plugins` entry point（沿用现有 plugin-src 机制）在 import 期 monkey-patch 该映射为包装类，env `VLLM_NVFP4_ROUTEA` 门控，启动脚本零改动。
- B12xExperts 关键行为：`process_weights_after_loading` 调 `prepare_b12x_fp4_moe_weights(reuse_input_storage=True)` **就地重打包 payload 并释放 layer 权重**（w4a16.packed），随后 apply 走 plan/bind/b12x_moe_fp4(quant_mode="w4a16")。
- W4A4 wrapper 侧（fork 源码逐行核实的构造链）：payload **[w3(up); w1(gate)] 行序**（`reorder_w1w3_to_w3w1`）→ scale 先 `swizzle_blockscale` 再 `flashinfer_convert_sf_to_mma_layout`（输入必须已是 swizzled，probe22/24 实证）→ `B12xMoEWrapper(quant_mode 默认 nvfp4).run(x, w1_weight, w1_weight_sf, w2_weight, w2_weight_sf, ids, scales, w1_alpha, w2_alpha, fc2_input_scale)`。swiglu_limit 仅 swigluoai 激活支持，DSV4 的 SILU 路径**不施加 clamp**（语义缺口，见 4.3）。

### 4.2 插件规格（A′ 形态，`plugin-src/` 重写）

```
nvfp4_vllm_plugin/
  __init__.py        # env 门控 VLLM_NVFP4_ROUTEA=1 时安装 patch
  routea_moe.py      # RouteAW4A4Experts(B12xExperts 子类):
                     #   process_weights_after_loading(layer):
                     #     1. 从 layer.w13/w2 (原始 MXFP4, [w1;w3] 行序) 取引用
                     #     2. 重排为 [w3;w1] (零拷贝 permute+contiguous 一次性, 或加载侧切片)
                     #     3. routea_weight_adapter.derive_routea_weights → E4M3 swizzled
                     #     4. convert_sf_to_mma_layout → w1/w2_sf_mma (缓存于实例)
                     #     5. 若混合形态: 喂 clone 给 super() 做 W4A16 prepare
                     #   apply(...): M ≥ VLLM_ROUTEA_M_MIN(默认 3072) → wrapper.run(W4A4)
                     #               否则 → super().apply (W4A16)
                     #   workspace_shapes: 两路径取大
  weight_adapter.py  # = 本次交付 routea_weight_adapter.py
```
预计 ~250 行。落地前两个必答前置：(a) §4.3 正确性排障；(b) 混合形态的 payload 共享方案（§3）。

### 4.3 ⚠ W4A4 路径语义验证**未通过**（本轮遗留的 P0）

按 fork 源码逐行复刻的构造链（[w3;w1] + swizzled→mma）在真实 layer-0 权重上：
- vs 精确 torch 参考：rel=0.51（期望 ~1e-2；probe22；nvfp4 激活量化假设比 mxfp4 假设更接近：0.51 vs 0.71）
- vs b12x W4A16（同权重、无 clamp）：rel=1.55（probe26）

已排除的原因：swizzle 公式（probe5[B] 字节级验证）、E8M0→E4M3 值（精确）、gate/up 行序（两种都试，probe22 A/B 相近）、topk 形状错误（已修）、SF plain-vs-swizzled 输入（swizzled 明显更优 0.51 vs 1.4）。**怀疑方向**：(i) 该路径在本构建从未被生产/测试真正执行过（oracle 明确把 flashinfer_b12x 排除在 auto 外、生产走 MXFP4→B12xExperts），存在真实缺陷的可能性不低；(ii) 激活量化语义（input_gs 与 w1_alpha 的耦合、fc2_input_scale 的静态/动态语义）与我参考实现的差异；(iii) kernel 对 M<某阈值走 dynamic 后端的边界行为（probe17/22 均为小 M）。
**排障路径（建议 1–2 天，单卡离线，无需 TP4）**：用 -nvfp4 checkpoint + `--moe-backend flashinfer_b12x` 起单卡 vLLM 离线推理（走 fork 自己的 FlashInferB12xExperts 全链）——若该链输出也不对，是 fork/flashinfer 缺陷（上报/绕开）；若对，则是我的 wrapper 直调契约仍有偏差，对照其 run 时 `_get_weight_views` 逐项 diff。**此项通过前，任何 W4A4 TP4 测试都无意义。**

另注意：DSV4 `swiglu_limit=10.0` 在 W4A4 路径不生效（生产 W4A16 生效）——即使排障通过，还需单层 logit 级 A/B 量化该缺口的影响（真实激活下 clamp 触发率应远低于随机权重场景；probe12b 中随机权重下 clamp 影响巨大 rel=1.19，不可外推）。

---

## 5. TP4 真实推理测试计划（修订）

**前提**：§4.3 排障通过。否则本节不启动，维持 B12X-only 现状（这也是本次裁定的默认建议）。

1. **P0 单卡离线**（node01，一次性容器，--rm）：
   - -nvfp4 + `--moe-backend flashinfer_b12x` 单卡 vLLM 加载 + 固定 seed 8 条 prompt 的 logprob 快照；
   - 同 prompt 在 -0731 + 生产配置（B12X W4A16）下取基线 logprob；对比逐 token logprob 偏差（阈值：mean |Δ| ≤ 0.02、无发散；附带 needle 抽验）。
2. **P1 插件形态单卡**：A′ 插件（-0731 + 派生）M≥3072 分派，重跑 P0 对比（隔离"派生 vs modelopt 原生 scale"差异）。
3. **P2 TP4 窗口**（生产 systemd 链 + 插件 + `VLLM_NVFP4_ROUTEA=1`）：
   - 稳定性：4 rank healthy、无 OOM（对照 +6.5GB/rank 预算，确认 KV 余量，必要时降 max-len/concurrency 跑测）；
   - 性能 A/B：chunked prefill（4096 token chunk）延迟/吞吐、首 token 延迟、decode 带宽（M=1 档应 6×、中段可能小幅回退——按 §2.3 表设定预期）；
   - 正确性：needle/长上下文抽验 + logprob 对比基线（P0 快照）；
   - 回退：unset env 重启即回 B12X 原样。
   - **判据**：prefill 档（M=4096 chunk）端到端加速 ≥ 15%（kernel 档 1.32× 折算整层 MoE 占比后）；decode 中段回退 ≤ 5%；不满足则维持 B12X-only 并归档。
4. **禁止事项**：不在 TP4 窗口同时验证多个变量（W4A4 与 KV K2 不联合灰度，首次只动 MoE）。

---

## 6. 工时与建议

| 事项 | 工时 | 前置 |
|---|---|---|
| W4A4 语义排障（§4.3，单卡离线，含 -nvfp4 全链 logprob 快照） | 1–2 天 | 无 |
| A′ 插件实现（routea_moe.py + adapter 集成 + 混合形态 payload 方案验证） | 2 天 | 排障通过 |
| P2 TP4 窗口测试（含 A/B、稳定性、回退演练） | 1 天（窗口内） | 插件就绪 |
| ~~per-expert routeA 集成~~ | ~~2–3 天~~ | **否决——实测 2.4–5.2× 回退，不做** |

**给 team-lead 的裁定一句话**：routeA 原始形态（per-expert cutlass 循环 + 权重转换 A/B/C 三路径）经生产形状实测全部否决；权重侧适配本身已完美闭环（零拷贝 + 精确 scale + rel=1.41e-03，适配器已交付）；唯一值得继续的是 fork 内置 b12x W4A4 融合 MoE（prefill 上限档 1.32×、M=1 档 6.2×），但其端到端语义在本构建未验证通过，建议窗口先做 1–2 天单卡排障再决定是否进 TP4。

## 附：环境与复现

- 容器：`docker run --rm --gpus all -v /tmp/_routea_work:/work -v <INSTALL_DIR>/models/deepseek-v4-flash-0731:/model0731:ro -w /work <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`
- 复现顺序：`python3 routea_weight_adapter.py`（CPU 自检）→ `python3 routea_validate_layer0.py`（判决）→ probes（5=契约+微基准, 6b=per-expert A/B, 8=cutlass mxfp4, 10=op 可用性+scale 扫描, 11=nvfp4 分组 A/B, 15/16=W4A4 vs W4A16, 17/22/26=W4A4 语义诊断）。
- 全程一次性容器（--rm）、<INSTALL_DIR> 与模型只读挂载、未启动生产、仅用 01 GPU（02 归 SRE）、单次峰值显存 < 8GB。
