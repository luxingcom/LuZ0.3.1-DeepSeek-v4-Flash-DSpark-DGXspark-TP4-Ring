# 调研修正决策落实：shared experts 改走 routeB FP8 路径（2026-08-23）

- **执行人**：阿奇（Archi）· 系统架构师（architect-2）
- **任务**：落实主理人决策"**改走 routeB FP8 路径**"（shared experts 优化方向从"FP4 新量化"修正为"routeB FP8 路径"），产出实施前研究：①FP8 payload 零拷贝可行性（数据布局/payload/scale/staging）；②收益重估（vs FP4 +0.8~1.5%、lm_head 竞争、A 量化前置判定）；③分阶段实施路径 + 验证门 + 风险；④优化对象排序更新。纯只读+设计，不碰 GPU/生产。
- **纪律**：本地报告 + SSH node01 只读勘察（checkpoint 元数据解析、生产 start 脚本、vLLM fork 源码一次性容器抽取），未启动生产、未触碰 GPU、无网络发布。
- **口径标注**：**[实测-源码]** = 服务器 checkpoint/容器源码直接验证；**[实测]** = 本团队既有实测数据（引用内部报告）；**[推算]** = 基于 Roofline/形状/FLOPs 推算，需 P0 profiler 或窗口微基准二次确认；**[待窗口验证]** = 本环境不可验证，必须由窗口执行。

---

## 0. 一页结论

1. **payload 零拷贝：源码级成立，运行级待 golden 实证**。shared experts 权重 = F8_E4M3 [N,K] K-contiguous + F8_E8M0 128×128 块 scale [实测-源码]；routeB kernel 的 B 操作数契约 = F8_E4M3 [N,K] K-major、SF=E8M0 sf_vec=32 [实测-源码]，且 dispatch 表明确支持 `(FP8, FP8, E8M0, 32) → MmaMXF8Op`。**权重 payload 字节布局与 routeB B 操作数一致 → 零拷贝直配可行**；唯一 staging = **scale 布局转换**（128×128 块 → 每行/每 32-K 组 E8M0，无损复制扩展，~11MB/rank 总量）[推算]，非有损量化。
2. **字节削减叙事需要修正：shared 走 FP8 无 decode 带宽收益**。shared experts **当前 compute 路径已经是 FP8 block-scaled w8a8**（`Fp8LinearMethod` block_quant → `CutlassFp8BlockScaledMMKernel`，`cutlass_scaled_mm`）[实测-源码]，权重已是 FP8 0.27GB/rank（0.54GB 是 bf16 误账口径）。routeB FP8 字节仍为 0.27GB → **shared 的 FP8 路径不削减 decode 字节**；÷2（0.54→0.27）只在"假设 bf16 基线"下成立，该基线不存在。**decode 字节削减属于 lm_head（head.weight=BF16 唯一 BF16 节点 [实测-源码]，routeB FP8 可 ÷2、W4A4 可 ÷4）**。
3. **收益重估（修正后）**：shared→routeB FP8 是"把已存在的 FP8 GEMM 换成更快的 FP8 GEMM"（GEMM 引擎置换），**不是引入新量化**——A 量化在当前路径已存在（动态 FP8 per-token-group），routeB FP8 仅需把 A scale 从 group=128 细化为 group=32 并适配 swizzle。预估 **PR +0.5~1.5%** [推算]，显著低于既有 FP4 口径的 +1.5~2.5%（上限），与修正后 FP4 路径 +0.8~1.5% 同量级；风险结构优于 FP4（无损 payload + 无损 scale，无校准/质量门前置），但 **GEMM 侧增益依赖 N=1024 小形状效率（未测）与 A-quant 不劣化（P4 教训）**。
4. **A 量化前置判定改变**：FP4 路径的"A 量化修复前置"（P4 短板）**不构成 FP8 路径的硬前置**——因为当前路径已付 A-quant 成本，routeB FP8 只是替换；但**必须设"routeB FP8 A-quant ≤ 当前 cutlass A-quant"的微基准门**，否则 E2E 仍可能被量化吃光（P4 教训）。"先修 A 量化再谈 routeB"的表述修正为"A-quant 适配器质量是 FP8 路径的一个验证门，不是前置工程"。
5. **排序更新（FP8 修正后）**：**P0 拆账实测 → shared（routeB FP8，首发，但收益下调至 +0.5~1.5% 且 decode 无带宽收益）→ lm_head（第二；FP8 路径可打 decode ÷2、W4A4 ÷4，且是唯一 BF16 节点，质量门最高）→ attn（第三，prefill 半场）**。shared 与 lm_head 的差距**进一步收窄**：FP8 修正后 shared 失去 decode 带宽叙事，而 lm_head 的 decode 字节 ÷2 完全成立。

---

## 1. 调研任务 1：routeB FP8 路径可行性

### 1.1 权重 payload 零拷贝判定 [实测-源码 + 推算]

**Checkpoint 实证（node01 safetensors 元数据直接解析，-0731）**：

| 张量 | dtype | shape | 语义 |
|---|---|---|---|
| `layers.X.ffn.shared_experts.w1.weight` | **F8_E4M3** | [2048, 4096] | gate 投影（N=2048, K=4096, K-contiguous） |
| `w1.scale` | **F8_E8M0** | [16, 32] | 128×128 块 scale（N/128 × K/128） |
| `w2.weight` | F8_E4M3 | [4096, 2048] | down 投影（N=4096, K=2048） |
| `w2.scale` | F8_E8M0 | [32, 16] | 128×128 块 scale |
| `w3.weight` | F8_E4M3 | [2048, 4096] | up 投影（与 w1 同构） |
| `w3.scale` | F8_E8M0 | [16, 32] | — |

config：`quant_method: fp8, fmt: e4m3, scale_fmt: ue8m0, weight_block_size: [128,128], activation_scheme: dynamic, n_shared_experts: 1` [实测-源码]。

**routeB kernel 契约（`routeb_official_v2/dense_blockscaled_gemm_persistent_pingpong.py` + `blockscaled_gemm_dispatch.py`）[实测-源码]**：
- dispatch 表：`(FP8, FP8, *, Float8E8M0FNU, 32) → MmaMXF8Op, use_mxf8f6f4=True, mma_K=32`——**FP8 same-dtype 路径在源码级已实现**。
- B 操作数：`NxKxL, B can only be column-major("K")` = K-contiguous；A 操作数 `MxKxL, row-major("K")` 亦 K-contiguous。
- SF：`tile_atom_to_shape_SF(b.shape, sf_vec_size)` → plain scale 布局 = **[N, ceil(K/sf_vec)]（每行 N、每 32 K 一组）**，经 `sf_plain_to_atom` 转 tcgen05 MMA atom 布局（`routeb_prod_adapter.py` 已实现 FP4 版本，FP8 用同一 atom 族，sf_vec=32）。
- 约束：FP8 路径 sf_vec 必须 32、sf_dtype 必须 E8M0、tile_K 必须 128 的倍数（routeB tile_k=128 锁定，满足）；tile 128³ + epi 128×128 在位。

**零拷贝判定**：
| 维度 | 结论 | 依据 |
|---|---|---|
| payload dtype | ✅ F8_E4M3 与 routeB FP8 B 操作数一致 | checkpoint + dispatch 表 [实测-源码] |
| payload 布局 | ✅ [N,K] K-contiguous = routeB B 的"column-major(K)" | torch 连续 [N,K] 即 K-major；routeB 经 from_dlpack 可 view [N,K,K-连续] [实测-源码] |
| 生产内存形态 | ✅ vLLM `Fp8LinearMethod` 的 layer.weight 为 [K,N]（N-contiguous），其转置视图恰为 [N,K] K-major → routeB 可零拷贝 view（不复制字节） | fp8.py + cutlass kernel `apply_block_scaled_mm` 用 `B.T` [实测-源码] |
| **scale** | ⚠️ **需布局转换**：checkpoint 是 128×128 块（[N/128,K/128]），routeB 期望每行/每 32-K 组（[N,K/32]）。转换 = **无损复制扩展**（每个 128×128 块 scale 复制到 128 行 × 4 个 K-组）→ [N,K/32] + atom swizzle。量级：每 rank/层 ~256KB，×43 ≈ **11MB/rank** [推算]，内存门可忽略 | 布局差异 [实测-源码] + 转换性质 [推算] |
| staging | ✅ 仅 scale（~11MB/rank 一次性，CPU/GPU 加载时）；payload 无 staging | — |

**诚实标注**：字节布局匹配是源码级实证，但"运行级零拷贝"（from_dlpack 对齐/divisibility/编译路径）仍需**一次性容器 golden 实证**（阶段 2 验证门）。存在两处已知风险点：①routeB FP8 的 M 对齐约束未在 P4 覆盖（P4 只测 FP4 m-major M%32）；②生产内存权重可能被 `process_weights_after_loading` 重排（Cutlass 路径的 `B.T` 处理），需逐层核对实际内存布局 [待窗口验证]。

### 1.2 字节削减路径修正（重要）[实测-源码]

**shared 当前 compute 路径实证**：vLLM fork 中 shared experts = `DeepseekV4MLP`（gate_up MergedColumnParallelLinear + down RowParallelLinear），quant_config = 模型 FP8（block_quant, weight_block_size [128,128]）→ `Fp8LinearMethod.apply` → `init_fp8_linear_kernel` → `_POSSIBLE_FP8_BLOCK_KERNELS` 选 `CutlassFp8BlockScaledMMKernel`（生产未设 `VLLM_USE_DEEP_GEMM`/`VLLM_BATCH_INVARIANT`，DeepGEMM/FlashInfer 分支关闭）[实测-源码]。

| 口径 | bf16（fi017 误账） | **实际（FP8，本报告实证）** | routeB FP8 |
|---|---|---|---|
| shared 权重/rank | 0.54GB | **0.27GB**（FP8 E4M3） | 0.27GB（不变） |
| decode 字节/step @273GB/s | 0.54GB → 1.98ms | **0.27GB → 0.99ms** | 0.99ms（**无收益**） |
| W4A4（FP4）后 | 0.135GB → ~0.5ms | 0.135GB → ~0.5ms | 0.5ms（**÷2 收益属于 FP4，不属于 FP8**） |

**结论**：
1. **"每步权重读 0.54G→0.27G/rank" 对 shared 不成立**——0.54G 是 bf16 误账口径，shared 当前已读 0.27G（FP8）。routeB FP8 不改变 shared decode 字节。
2. **decode 带宽削减的叙事必须整体修正**：池内字节账（fi017 §2.3）基于"全部 bf16"假设；实际池是 **FP8（shared/attn）+ BF16（lm_head）混合**。**lm_head 是池内唯一 BF16 节点**（`head.weight` = BF16 [129280,4096] [实测-源码]）→ routeB FP8 对 lm_head 可 ÷2、W4A4 可 ÷4。
3. 池内 decode 字节审计（attn 投影是否 FP8、MTP/embed bf16 权重占比）需在 P0 做 dtype 级重核 [待窗口验证]。

### 1.3 与 routed MoE 现役路径（B12X-only W4A4）接口衔接

- 现役 routed MoE：`FlashInferB12xExperts`（overlay `flashinfer_b12x_moe.py`）**仅支持 NVFP4（FP4）量化**，通过 `B12xMoEWrapper` 几何键池（`VLLM_B12X_SHARED_WRAPPER=1`）跨层去重，由 `VLLM_MOE_W4A4=2` 分派（MIN_M=3072 残留，实际全 M 走 W4A4）[实测-源码]。
- **shared 不在任何 W4A4 分档内**：它是独立 MLP 线性路径（`DeepseekV4MLP.forward`），未接入 wrapper [实测-源码]。直接塞进 B12x wrapper 不可行（wrapper 断言 NVFP4；shared 是 FP8 E4M3）。
- **衔接设计（FP8 路径）**：不经过 B12x wrapper。建议在 **`Fp8LinearMethod` 的 kernel 选择层**新增 `RouteBFp8BlockScaledMMKernel`（并入 `_POSSIBLE_FP8_BLOCK_KERNELS` 候选，按形状/env 门控 `VLLM_SHARED_ROUTEB_FP8=1`），复用现有 linear-method 管线（`process_weights_after_loading` + `apply_block_scaled_mm` 契约）。这样**与 routed MoE 现役路径正交**：routed MoE 继续走 B12X W4A4，shared 走 routeB FP8，互不干扰。
- vLLM 已有 `SharedExperts` runner 抽象（fused_moe/runner，含 aux-stream 重叠），但 DSV4 模型用的是独立 `DeepseekV4MLP`，未走该 runner [实测-源码]——**迁移到 fused-MoE 表示是可选后续**，与本次 routeB FP8 GEMM 置换正交（见 §4.3）。

---

## 2. 调研任务 2：收益重估

### 2.1 prefill/GEMM 侧 [推算，待微基准]

- **机制**：routeB FP8 替换当前 `CutlassFp8BlockScaledMMKernel`（FP8 w8a8 block-scaled）。当前路径池级有效 ~55-65T（M=1024 口径，fi017 §2.0；含 A-quant 与 launch 开销）；routeB FP8 在共享形状（gate_up [M,4096]×[4096,1024]、down [M,1024]×[1024,4096] per rank，M=4096）预估 **200-300T**（P4 FP4 同 N 域 w1 N=2048 为 300-330T；N=1024 小 N 折扣）[推算]。GEMM 侧节点级 3-4×。
- **扣 A-quant delta**：当前路径已付 FP8 动态 A-quant（per-token group=128）；routeB FP8 需 group=32（更细、4× scale 数）+ E8M0 + swizzle。若 A-quant 适配器沿用 `per_token_group_quant_fp8`（group_size=32）+ `sf_plain_to_atom`，成本与当前同族、略增（4× scale 写入）；**风险是 routeB A-quant 若走未融合 triton 双 pass 会重演 P4 短板（94-136GB/s）**——这是本路径最大的可执行性风险。
- **净收益**：shared 池份额 9-12µs/token（M=1024 口径）→ GEMM 部分 ~60-70% → routeB FP8 压缩 GEMM 3-4× → 省 ~4-6µs/token（M=1024 口径）；M=4096 生产形态份额更低，PR 折算 **+0.5~1.5%** [推算]。**显著低于既有 FP4 路径 +1.5~2.5% 的上限口径**（upstream-check 未扣 A-quant），与修正后 FP4 路径 +0.8~1.5% 同量级。

### 2.2 decode 侧（修正后）[实测-源码 + 推算]

- shared routeB FP8：**无 decode 字节收益**（§1.2）。shared 的 decode 带宽削减只有 W4A4 能兑现（÷2，省 ~0.5ms/step）——**已不在本决策范围**。
- lm_head routeB FP8：当前 BF16 0.27GB/rank → FP8 0.135GB → **省 ~0.5ms/step（C12 口径 ~4-5%）**；W4A4 → 0.07GB → 省 ~0.7ms/step（~5-6%）。lm_head 的 FP8 路径**同时给 decode 字节 ÷2 + prefill GEMM 加速**，且是池内唯一 BF16 节点。
- 池 decode 字节的 dtype 级重核（attn/MTP/embed）列为 P0 项目 [待窗口验证]。

### 2.3 A 量化前置判定（修正 P4 结论）[实测-源码 + 推算]

| 路径 | A-quant 前置 |
|---|---|
| FP4 路径（原方向） | **硬前置**：需新 FP4 A-quant（E2M1+pack），P4 实测 E2E 全 M 落后 14-47%，必须先修量化融合 |
| **FP8 路径（本决策）** | **非硬前置**：当前路径已做 FP8 动态 A-quant；routeB FP8 只需把 group 128→32 + E8M0 + swizzle。**门槛 = "routeB FP8 A-quant ≤ 当前 cutlass A-quant"的微基准门**，不是"先修 A 量化再集成" |

**含义**：主理人决策"改走 routeB FP8"在工程上确实绕开了 FP4 路径的"FP8→FP4 有损量化 + 校准 + 质量门"重负，但**没有绕开 A-quant 效率问题本身**（P4 教训迁移：A-quant 在 E2E 中可吃光 GEMM 增益）。A-quant 适配器必须按"与当前 cutlass A-quant 同性能或更优"设计，不可重走未融合 triton 双 pass。

### 2.4 与 lm_head 竞争关系（修正后）[推算]

- FP8 修正前：shared +0.8~1.5%（已扣 A-quant）vs lm_head +0.85% prefill + decode 带宽；差距已收窄。
- FP8 修正后：shared +0.5~1.5%（GEMM 置换、无 decode 收益）vs lm_head（BF16→FP8：prefill GEMM 加速 + decode ÷2 ~0.5ms/step + 显存省 ~135MB/rank；BF16→W4A4：decode ÷4 ~0.7ms/step）。
- **排序脆弱性上升**：shared 首发的理由收窄为"M=4096 甜点 + 全 token + GEMM 引擎置换风险低 + 质量门先例可迁移"；lm_head 因"唯一 BF16 节点 + decode 字节 ÷2 成立 + 每步仅 1 次 A-quant"反而更接近首发。**P0 拆账后 lm_head 反超 shared 的概率高于此前任何报告**。

---

## 3. 调研任务 3：实施路径（分阶段 + 验证门 + 风险）

### 3.1 阶段划分

| 阶段 | 内容 | 预计 | 验证门（go/no-go） | 风险 |
|---|---|---|---|---|
| **P0 前置** | P0 profiler 拆账（方案 A/B，同 fi017 §2.5）+ 池 dtype 级字节审计（shared/attn/lm_head/MTP 实际 FP8 vs BF16） | 0.5-1 天 | 实测 shared/lm_head/attn µs 份额 + 字节口径 → 校准排序 | — |
| **F0 契约核对**（纯离线） | scale 扩展脚本（128×128→[N,K/32] 无损复制）+ CPU golden；确认生产内存 weight/weight_scale_inv 实际布局（[K,N] vs [N,K]） | 0.5 天 | CPU golden：扩展后 dequant == 原始 dequant（逐位/rel<1e-6）；内存布局清单 | payload 布局与假设不符（需回退 view 方案） |
| **F1 FP8 kernel 微基准**（GPU 窗口，一次性容器） | routeB FP8 same-dtype（FP8×FP8+E8M0+sf_vec=32）× 共享三形状 × M∈{8,96,512,1024,4096}；KO + E2E（含 FP8 A-quant 适配器）vs 当前 cutlass FP8 | 0.5-1 天 | **E2E(M=4096) ≥ 1.1× 当前 cutlass**；A-quant delta 占比 < GEMM 增益；M=512-768 数据门 | N=1024 小 N 效率不足（P4 未测）；FP8 A-quant 重演 P4 短板 |
| **F2 零拷贝适配器**（GPU 容器） | `RouteBFp8BlockScaledMMKernel`：A-quant(group=32)+scale 扩展+swizzle+routeB FP8 GEMM；golden vs 当前 Fp8LinearMethod | 1-2 天 | golden rel_err ≤ 1e-2；零拷贝路径（from_dlpack 直配，无 payload 复制）实测确认 | 编译/JIT 缓存（121a arch）、对齐约束 |
| **F3 集成 L1**（无生产） | 并入 `_POSSIBLE_FP8_BLOCK_KERNELS`（env `VLLM_SHARED_ROUTEB_FP8=1` 门控）；off 路径逐字等价；env checker 四机同步；cudagraph 捕获 | 2 天 | L1 全过（off 路径 byte-equivalent；on 路径数值带内；0 ERROR） | Fp8LinearMethod 契约偏差；graph 捕获兼容 |
| **F4 窗口 A/B**（生产窗口） | 对照 LuZ0.3.1（routed W4A4 full 基线）：PR 四档 / DE C1 C12 / 内存 / 质量门 | 0.5-1 天 | PR 四档 ≥ -3%（目标 +0.5~1.5%）；DE ≥ -3%；质量门 golden 4/4 + logprob + 困惑度 Δ；内存门 scale +11MB/rank | 生产合成（autotune/版本）拖尾 |
| **F5 质量门 + 发布** | 扩展质量（golden 4 逐字 + logprob 对齐 + 困惑度 + 接受率不降 + needle 抽验）；回滚锚点 | 1-2 天 | 全门 PASS；`.bak` 快照 + head-first 重建核验 | — |

**关键 go/no-go 决策点**（F1）：routeB FP8 E2E(M=4096) 不达 1.1× 当前 cutlass，或 A-quant delta 吃光 GEMM 增益 → **FP8 路径降级为设计储备**，回到 FP4/W4A4 或维持现状（沉没成本 = F0/F1 半天窗口）。

### 3.2 与 FI 0.6.17 共享专家融合 API 的协同/替代关系 [上游实证 + 推算]

- **FI 0.6.17 共享专家融合（#4159/#4088 等）是 FP4/NVFP4 路线的 MoE 调用级融合**（把 shared 融进 fused MoE 调用，消独立 GEMM 调用开销）[上游实证]。它与本次 routeB FP8 GEMM 置换**方向不同、相互独立**：
  - routeB FP8 = **GEMM 引擎级**优化（替换 FP8 GEMM 内核，不改变调用结构）；
  - FI 0.6.17 shared fusion = **调用结构级**优化（shared 迁入 fused MoE 表示，需 FP4 权重，属 W4A4 方向）。
- **协同**：二者正交，可并行评估。若未来把 DSV4 shared 迁入 fused-MoE 表示（`SharedExperts` runner 已在 fork 内 [实测-源码]），其内部线性仍可用 routeB FP8 引擎 → 两级叠加。
- **替代关系**：若 FP8 路径 F1 微基准不达标，FI 0.6.17 的 shared fusion（W4A4 方向）可作为替代通道（需 FP8→FP4 有损量化 + 质量门，回到原 FP4 成本结构）。
- **建议**：本决策聚焦 routeB FP8；FI 0.6.17 shared fusion 维持 W8（U1 探路）观察位，不并行投入实现。

---

## 4. 调研任务 4：排序更新（FP8 修正后）

> 前提：P0 拆账实测未完成；以下基于既有实测 + [推算]。P0 后按实测份额 ±1 位浮动。

| 排序 | 对象 | 收益 [推算] | 就绪度（FP8 修正后） | 关键前置/门 | 排序依据 |
|---|---|---|---|---|---|
| **0** | **P0 拆账实测**（方案 A/B + 池 dtype 字节审计） | 锁定份额、消除 ±1 位浮动、修正池字节账 | 高 | 无 | 一切排序前提 |
| **1** | **shared → routeB FP8** | PR +0.5~1.5% [推算]（GEMM 引擎置换）；**decode 字节收益 = 0** | 中-高（payload 零拷贝源码级成立、无损 scale；**FP8 kernel 未测、N=1024 效率未知、A-quant 门待验**） | F0 契约 → F1 微基准（E2E≥1.1×）→ F2 golden → F3-F5 | M=4096 甜点 + 全 token + **无损/无校准**（相对 FP4 最大优势）+ 质量门先例可迁移；但收益下调、无 decode 带宽叙事 |
| **2** | **lm_head → FP8（先）/W4A4（后）** | FP8：prefill 加速 + **decode ÷2 ~0.5ms/step** + 显存 -135MB/rank；W4A4：decode ÷4 ~0.7ms/step [推算] | 中（BF16 唯一节点、routeB kernel 有、质量门流程待建） | 校准 + KL 门先行；F1 复用 | **唯一 BF16 节点 + decode 字节削减成立 + A-quant 每步 1 次**；FP8 修正后与 shared 差距显著收窄，**P0 后反超概率最高** |
| **3** | **attn 投影 → 分档量化** | PR +2~4%（仅 prefill 半场）[推算] | 中-高（FP8 现状已明确、形状多、数值风险分层） | o_proj 灰度 → q/kv/lora 逐层门 | FLOPs 份额最大但仅半场；decode 明确不做 |
| **4** | **routeB M 下探 / decode 稠密带宽** | 前提依赖型（先解决 A-quant 通用效率） | 低-中 | 与 shared/lm_head 共用 A-quant 适配器成果 | 伴随研究轨道；本报告修正后 decode 字节空间主要在 lm_head，非 shared |

**条件触发**（维持不变）：
- P0 实测 shared prefill µs 份额 <6µs/token（当前推算 9-12µs）或 lm_head >25% → **shared 让位 lm_head 首发**。
- F1 微基准 routeB FP8 E2E <1.1× 当前 cutlass → **FP8 路径降级**，shared 排序退至设计储备，lm_head 顺势前移。

**排序说明**：
1. shared 首发的理由从"零拷贝 + 快赢 + 资产全齐"修正为"**无损 payload/scale（无校准门）+ M=4096 甜点 + 全 token + GEMM 引擎置换风险结构优于 FP4**"；收益 +0.5~1.5%、无 decode 带宽收益。
2. lm_head 在 FP8 修正后成为**唯一同时满足 decode 字节 ÷2 + prefill + 显存**的对象（shared 不再是），排序脆弱性最高。
3. A-quant 适配器是 shared-FP8 与 routeB-M 下探的**共用前置工程**（非 FP4 意义的"修复"），建议与 F1/F2 绑定投入。

---

## 5. 下一步窗口任务清单

| # | 任务 | 归属 | 前置 | 预计 | 产出/判据 |
|---|---|---|---|---|---|
| W1 | **P0 profiler 拆账 + 池 dtype 字节审计**（shared/attn 是否 FP8、lm_head BF16、MTP/embed 权重字节） | SRE/架构 | 无 | 0.5-1 天 | 实测份额 + 字节口径 → 校准排序；确认本报告字节修正 |
| W2 | **F0 契约核对**（scale 扩展 CPU golden + 生产内存 weight 布局清单） | 架构 | 无（纯离线） | 0.5 天 | golden + 布局清单 |
| W3 | **F1 routeB FP8 微基准**（FP8 same-dtype × 共享三形状 × M 扫描，KO+E2E vs cutlass） | 架构/kernel | W2 | 0.5-1 天 | **E2E(M=4096)≥1.1× 当前 cutlass 为 go/no-go** |
| W4 | **F2 零拷贝适配器 + golden** | kernel 工程 | W3 数据为正 | 1-2 天 | rel_err ≤1e-2；payload 零拷贝实证 |
| W5 | **F3 集成 L1**（env 门控 + off 逐字等价 + checker 同步） | 架构 | W4 | 2 天 | L1 全过 |
| W6 | **F4/F5 窗口 A/B + 质量门** | SRE | W5 | 1-2 天窗口 | PR/DE/质量三门 + 回滚锚点 |
| W7 | **lm_head FP8 立项准备**（per-channel 校准 + KL 门；复用 routeB FP8 微基准形状 N=32320） | 架构 | W1 | 1 天 | 校准方法 + 质量门方案定稿 |
| W8 | **U1：FI 0.6.17**（维持 Go 有条件结论）——shared fusion API 观察位，不并行投入 | SRE/架构 | 无 | 1-2 天 | 上游 shared fusion 可用性结论（FP4 通道备选） |
| Watch | 池 decode 字节 dtype 重核结果反馈 fi017 §2.3 账目 | — | W1 | — | 修正后续字节/带宽估算 |

**窗口编排建议**：W2 纯离线可立即；W3 与 W7 可共享一次 GPU 微基准窗口（一次性容器 <1GB 显存纪律）；W4/W5 串行；W6 需生产窗口（对照 LuZ0.3.1）。

---

## 6. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| **[实测-源码]** | checkpoint shared 权重 F8_E4M3 [2048,4096]/[4096,2048] + F8_E8M0 [16,32]/[32,16]；config（quant fp8/e4m3/ue8m0/block 128×128/dynamic/n_shared=1）；**head.weight = BF16 [129280,4096]（lm_head 唯一 BF16）**；**shared 当前走 Fp8LinearMethod block_quant → CutlassFp8BlockScaledMMKernel（生产未设 VLLM_USE_DEEP_GEMM/BATCH_INVARIANT）**；`DeepseekV4MLP` 独立 MLP（未走 fused_moe SharedExperts runner）；`FlashInferB12xExperts` 仅 NVFP4；routeB kernel dispatch `(FP8,FP8,E8M0,32)→MmaMXF8Op` + SF 布局 [N,K/sf_vec] + tile_k%128=0；生产 env（VLLM_MOE_W4A4=2/MIN_M=3072/B12X_SHARED_WRAPPER=1） |
| **[实测]** | P4 全 FP4 A/B 矩阵（w1/w3/w2 × M，KO/E2E）；P4 small_m 数据（rb_quant_unfused 108-188µs vs 2pass 14.7-26.7µs vs kernel 18-21µs）；routeB 350T 平台（M≥1536）、N=2048 300-330T、N=12288 M_g 曲线；池 29-34µs/token / 55-65T（M=1024 口径）；LuZ0.3.1 三门 |
| **[推算]** | scale 扩展无损 + ~11MB/rank；shared→routeB FP8 **PR +0.5~1.5%**；lm_head FP8 decode ÷2 ~0.5ms/step、W4A4 ÷4 ~0.7ms/step；routeB FP8 共享形状 200-300T（N=1024 小 N 折扣）；A-quant delta 风险 |
| **[上游实证]** | FI 0.6.17 共享专家融合（#4159/#4088，FP4 方向）；FI 0.6.17 wheel/依赖（fi017 已核）；b12x 1.2.6 后零提交（upstream-check 已核） |
| **[待窗口验证]** | P0 拆账 + 池 dtype 字节审计；**routeB FP8 微基准（P4 未测 FP8）**；运行级 payload 零拷贝 golden；生产内存 weight 布局；A-quant 适配器性能门；F3-F5 集成与窗口 A/B |

**诚实声明**：
1. **本报告最重要的两条修正**：① shared 当前已是 FP8 block-scaled w8a8（"0.54G→0.27G/rank"字节削减对 shared 不成立，decode 带宽叙事整体需按 dtype 混合重核）；② **lm_head 是池内唯一 BF16 节点**（decode 字节削减真正属于 lm_head）。这两条共同把 shared 与 lm_head 的收益差进一步收窄。
2. **"payload 零拷贝"为源码级判定**（字节布局匹配），非运行级实证；运行级需 F2 golden 门。
3. routeB FP8 的 GEMM 侧增益（200-300T vs 当前 55-65T）基于 P4 FP4 数据外推，**N=1024/K=1024 小形状在 P4 未测**；当前路径在 M=4096 生产形态下的实际有效吞吐也需 P0 实证。
4. A-quant 效率（P4 短板）虽不再是"前置修复"，但仍是 FP8 路径最大执行风险——F1 的"E2E≥1.1× + A-quant delta < GEMM 增益"是硬 go/no-go。
5. 全部服务器勘察为只读（safetensors header 解析、config/脚本 cat、一次性容器 CPU 抽取源码），未启动生产、未触碰 GPU。
6. 排序在 P0 实测前有 ±1 位浮动（尤其 shared 与 lm_head 之间）；P0 后按实测份额重排，条件触发不变。

---

## 7. 引用索引

- 本报告关键实测源：fi017-p0-accounting-2026-08-23.md（池拆账）、opt-objective-research-2026-08-23.md（前序调研，含 shared FP8 修正）、upstream-check-perf-ceiling-2026-08-23.md（P0-P3 + 拐点）、routeb-p4-ab-perf-2026-08-21.md（P4 FP4 矩阵 + A-quant 短板）、routeb-merged-phasec-2026-08-21.md（M_g 曲线）、luz031-deployment-2026-08-23.md（生产形态）、engineering-assets-report-2026-08-22.md（routeB standby/适配器）
- 服务器勘察：node01 `/home/<USER>/models/deepseek-v4-flash-0731/`（safetensors header 解析）、`<INSTALL_DIR>/scripts/start_tp4_head.sh`（env）、`<INSTALL_DIR>/nvfp4/routeb_official_v2/`（FP8 dispatch 源码）、`<INSTALL_DIR>/nvfp4/routeB-delivery-2026-08-20/` + `/tmp/routeb_task12/`（bench/adapter/prod_adapter/P4 results）、`<INSTALL_DIR>/overlay-wsdedup/flashinfer_b12x_moe.py`、镜像 `anemll/dspark-vllm-gx10:LuZ0.3.1` 一次性容器（`vllm/models/deepseek_v4/nvidia/model.py`、`vllm/model_executor/layers/quantization/fp8.py`、`vllm/model_executor/kernels/linear/__init__.py`、`scaled_mm/cutlass.py`、`scaled_mm/deep_gemm.py`、`scaled_mm/flashinfer.py`、`fused_moe/runner/shared_experts.py`、`cutlass/utils/blockscaled_layout.py`）
- 上游：github.com/flashinfer-ai/flashinfer（0.6.17 共享融合）

*本报告由工程保障团队（系统架构师 architect-2）生成；排序与立项请由人类工程负责人结合 P0 拆账实测与 F1 微基准裁定。*
