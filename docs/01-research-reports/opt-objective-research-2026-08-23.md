# 自研优化方向调研：shared experts 优先测试对象 + routeB M 下探（2026-08-23）

- **执行人**：阿奇（Archi）· 系统架构师（architect-1）
- **任务**：①调研 shared experts（MoE 共享专家，M=4096 全量）在 routeB 平台的 kernel 就绪度 / 零拷贝适配器衔接 / MIN_M 分档现状 + 收益/部署面/风险/验证门；②调研 routeB M=1024→512-768 下探的 kernel 支持度 / 收益曲线 / 实施路径 / 风险；③产出优化对象推荐排序（标注"待 P0 拆账实测校准"）+ 下一步窗口任务清单
- **纪律**：纯只读分析（本地报告 + SSH node01 只读勘察 <INSTALL_DIR>/ + 一次性 CPU 容器读镜像源码），**不占 GPU、不动生产、无网络发布**
- **口径标注**：**[实测]** = 本团队既有实测数据（引用内部报告/服务器勘察）；**[实测-源码]** = 服务器 checkpoint/容器源码直接验证；**[推算]** = 基于 Roofline/形状/FLOPs 的推算，需 P0 profiler 或窗口微基准二次确认；**[上游实证]** = 上游仓库/Release 页面核验；**[待窗口验证]** = 本环境不可验证，必须由窗口执行

---

## 0. 一页结论

1. **方向 1（shared experts → routeB/W4A4）"资产全齐"的说法需要修正，但作为优先测试对象的排序仍成立**：直接勘察服务器 checkpoint 发现 **shared experts 权重是 FP8 E4M3 block-scaled（128×128 块 + E8M0 scale），不是 FP4 E2M1 payload** [实测-源码]——既有报告 P1 行"零拷贝适配器（E2M1 直配 + E8M0→E4M3 LUT）已验证"是**把 routed MoE 的 FP4 适配器误套到了 shared experts 节点上**。shared→W4A4 需要 **FP8→FP4 新量化（有损）**，或走 **routeB FP8 路径（payload 可能零拷贝，字节 ÷2 而非 ÷4）**；且 TP4 下 shared FFN 形状（gate_up N=1024/rank、down K=1024/rank）不是 routeB 甜点形状（350T 平台在 N=12288-14336）。收益推算**扣掉 A 量化开销后为 PR +0.8~1.5%**（upstream-check 的 +1.5~2.5% 是零拷贝/未计量化的上限口径）[推算]，**量化开销（A 侧）+ 形状适配是隐藏工程量**，快赢属性弱于既有报告表述；且该修正收窄了 shared 与 lm_head 的收益差。
2. **方向 2（routeB M=1024→512-768 下探）面临既有实测的硬约束**：routeB P4 实测 **E2E（含 A 量化）在全部 M 档无一处胜出**（M<256 落后 14-47%，M∈[256,4096] 也仅 0.53-0.97× vs routeA）[实测]，瓶颈在 A 量化（双 pass triton 94-136GB/s vs vLLM C++ ~200GB/s）不在 GEMM；**kernel 侧无 M≥1024 硬约束**（persistent tile scheduler 处理 partial tile，M%32 对齐即可）[实测-源码]，"MIN_M=1024" 是分派策略而非 kernel 限制。因此"下探 MIN_M"本身廉价，**真正的工作是 (a) 修 A 量化到单 pass/融合 prologue，(b) 验证解码段（M=8-96）稠密节点的带宽削减假设**（该假设未证，P4 只测了 routed 形状）。
3. **推荐排序（P0 拆账实测前，±1 位浮动）**：**P0 拆账实测 → shared experts（首发，但以"FP8→FP4 量化 + routeB 形状微基准"为前置门）→ lm_head（第二，唯一打 decode 带宽墙，质量门最高）→ attn 投影（第三，仅 prefill 半场）→ routeB M 下探（独立研究轨道，先修 A 量化再谈分派）**。shared 首发仍成立的理由不是"零拷贝"，而是**路由/质量门先例可迁移 + M=4096 甜点 + 全 token 覆盖**；但**扣掉 A 量化开销后 shared 相对 lm_head 的收益差显著收窄**（shared +0.8~1.5% vs lm_head 约 +0.85% prefill + decode 带宽，详见 §1.2），排序脆弱性高于既有报告表述——P0 后 lm_head 可能反超。
4. **叙事校准**："填满甜点区 + 削减墙下流量"大方向不变，但甜点区的"填"需要先解决 **A 侧量化的融合问题**（这是 routeB 路线 P4 就已暴露、且被 merged 管线 prologue 部分解决的已知短板），否则甜点区数字（368T kernel-only）在 E2E 兑现不出来。

---

## 1. 调研任务 1：shared experts 优先测试对象

### 1.1 资产盘点（含关键修正）

#### 1.1.1 节点定义与当前运行形态 [实测-源码 + 推算]

- **节点**：`DeepseekV4MLP`（`vllm/models/deepseek_v4/nvidia/model.py:82-159`），gate_up = `MergedColumnParallelLinear` + down = `RowParallelLinear`，全 token 经过，每层一次，43 层。
- **生产几何**（TP4，每 rank）：gate_up `[M,4096]×[4096,1024]`、down `[M,1024]×[1024,4096]`；prefill M=4096（chunk 甜点）。
- **当前 compute 路径**：quant_config = 模型 FP8 config（`quant_method: fp8`、block 128×128）；**bf16 池口径**（fi017 §2.0：shared 0.54GB/rank、55-65T 效率区）——是否实际走 FP8 GEMM 还是 bf16 upcast **需 P0 profiler 定论** [待窗口验证]，影响当前基准效率与转换增益。

#### 1.1.2 关键修正：shared experts 权重格式 [实测-源码]

服务器直接解析 `deepseek-v4-flash-0731` checkpoint（-nvfp4 同构）：
- `layers.X.ffn.shared_experts.w1.weight` = **F8_E4M3 [2048, 4096]**（gate/up 融合按 MLP 拆为 w1/w3 两 FP8 张量）
- `w1.scale` = **F8_E8M0 [16, 32]**（= N/128 × K/128 二维块，128×128 block-scaled）
- `w2.weight` = F8_E4M3 [4096, 2048]，`w2.scale` = F8_E8M0 [32, 16]
- 对照：routed experts payload = **U8（FP4 E2M1 打包）** + E8M0 [N, K//32] 逐行 scale [实测-源码]

**结论**：shared experts **不是 FP4 E2M1 payload**，既有 P1 行"零拷贝适配器（E2M1 直配）"不适用。shared→W4A4 需要新派生链：
- **选项 S1（W4A4，FP8→FP4）**：FP8 E4M3 block-scaled → FP4 E2M1 + E4M3/E8M0 scale，**有损量化，需校准 + 质量门**；"零拷贝"不成立。
- **选项 S2（routeB FP8 路径，MmaMXF8Op，sf_vec=32 E8M0）**：payload F8_E4M3 可能零拷贝直配，但 checkpoint scale 是 128×128 二维块、routeB FP8 期望行/基本块 E8M0（K/32 或 MKL 布局）→ **scale 布局需一次性转换**（廉价）；收益为字节 ÷2（vs bf16），非 ÷4。
- **选项 S3（暂不量化，仅分派调优）**：当前 bf16/FP8 路径微调，无 kernel 变更。

#### 1.1.3 routeB 平台可用资产 [实测]

| 资产 | 状态 | 与本节点关系 |
|---|---|---|
| routeB DSL dense kernel（routeb_official，368T @ 4096×14336×4096，tile 128³） | **standby 存档**（`<INSTALL_DIR>/nvfp4/routeb-archive-20260821.tar.gz`，A2） | 目标 GEMM 引擎；**但共享形状（N=1024/K=1024/rank）远小于其甜点 N=12288-14336**，效率需微基准 [推算：可能 250-300T KO] |
| W4A4 wrapper 池补丁（`VLLM_B12X_SHARED_WRAPPER=1`，flashinfer `B12xMoEWrapper` 跨层几何键池） | **生产在位**（LuZ0.3.1 实测） | 该池针对 **routed MoE**（E=256/topk6）；shared 若走 "E=1/topk=1 稠密当 MoE" 复用池，需新几何键（num_experts=1）[推算] |
| routea_weight_adapter（E2M1 直配 + E8M0→E4M3 LUT，rel=1.41e-3） | **已验证**（A4） | 只对 **FP4 payload** 有效；**不适用于 shared FP8** |
| fused prologue/combine Triton 管线（phasec：0.416ms @ M=8240） | 原型级验证 | shared 的 A 量化 [M,4096] 若走此融合可压低量化开销 [推算] |
| routeB P4 A/B 矩阵（w1/w2/w3 形状 × M 扫描） | 实测 | **N=1024/K=1024 形状未测**，是 shared 方向的直接空缺 [待窗口验证] |

#### 1.1.4 MIN_M 分档现状 [实测]

- 生产 start 脚本（node01 勘察）：`VLLM_MOE_W4A4=2`（routed MoE **全 M 走 W4A4**，MIN_M 不 gate）、`VLLM_MOE_W4A4_MIN_M=3072`（残留默认值）、`VLLM_B12X_SHARED_WRAPPER=1`。
- **shared experts 目前不在任何 W4A4 分档内**——它是独立 MLP 线性路径，未接入 wrapper [实测-源码]。
- "MIN_M=1024 分档"是**未来稠密池转化（shared/lm_head/attn）的分派设计阈值**（upstream-check P1-P3 行），尚未实现。

### 1.2 收益推算（修正后，含 A 量化扣减）[推算]

- 池内 shared 份额：9-12µs/token（M=1024 口径，fi017 §2.2）。
- **GEMM 侧**：shared per-rank FLOPs ≈ 12.6M/token（50.3M/4 TP），M=4096 时 ≈ 51.5G/层；当前 bf16 60T → 0.86ms/层；routeB 200-250T（N=1024 小 N 折扣，P4 的 w1 N=2048 为 300-330T）→ 0.21-0.26ms/层 → **43 层共省 ≈ 26-28ms/step**。
- **A 量化扣减（upstream-check 未计的关键项）**：shared FFN 每层需量化 gate_up A [M,4096]（33.5MB/rank）+ down A [M,1024]（8.4MB）；@94-136GB/s（双 pass triton 实测下限）≈ 0.31-0.45ms/层 × 43 ≈ **13-19ms/step**；若 fused prologue（phasec 0.416ms @ M=8240 → 推算 0.21ms @ M=4096）≈ **9ms/step**。
- **净收益**：GEMM 省 26-28ms − 量化 9-19ms ≈ **7-19ms/step ≈ 1.7-4.6µs/token ≈ PR +0.8~1.5%**（当前架构各自独立量化）——若未来与 routed MoE 的 W4A4 A 量化共享（A 已量化一次，routed/shared 复用）可再提，但属后续优化。
- **decode 侧带宽削减**：shared bf16 0.54GB/step → W4A4 0.135GB/step → **~1.5ms/step** [推算，fi017 §2.3]；FP8 路径 → ~0.99ms/step。前提是 decode M=8-96 也走 W4A4/FP8（需专门分派，MIN_M 下探不覆盖此段）。
- **对照修正说明**：既有 upstream-check "PR +1.5~2.5%" 是按零拷贝适配器 + 纯 GEMM 置换的**上限口径**（未扣 A 量化）；本报告扣减后为 **+0.8~1.5%**。该修正同时**收窄了 shared 与 lm_head 的收益差**（lm_head 每步仅 1 次 A 量化 [M,4096] ≈ 0.34ms，非 ×43，量化占比远低）。

### 1.3 部署改动面与风险

**改动面**（相对"资产全齐"预期的新增工作量）：
1. **权重派生新链**：FP8 E4M3 block-scaled → W4A4（FP4 + scale）或 routeB FP8 布局；需校验 scale 布局契约（128×128 块 vs routeB E8M0 行块）。
2. **集成挂点**：shared experts 走 `DeepseekV4MLP.forward`（MergedColumnParallelLinear/RowParallelLinear），与 routed MoE 的 `backend_to_kernel_cls` 挂点不同——需新 hook 或改用 B12xMoEWrapper(E=1, topk=1) 表示稠密 FFN（后者复用 wrapper 池但引入路由语义适配）。
3. **env 开关**：新 `VLLM_SHARED_W4A4=0/1`（或并入现有插件），需四机脚本 + checker 同步。

**风险分层**：
| 风险 | 级别 | 缓解 |
|---|---|---|
| FP8→FP4 新量化有损（非零拷贝） | 中 | 校准 + golden 4/4 + 困惑度 Δ + 接受率不降（routed MoE 质量门先例可迁移）|
| shared 形状（N=1024/K=1024）routeB 效率低于甜点 | 中-高 | P0 方案 B 微基准先行（M=8/96/512/1024/4096 × 三形状）|
| A 量化 E2E 开销吃掉 GEMM 增益（P4 教训） | 中-高 | fused prologue / C++ 单 pass（P4 A1 行动项）；不可跳过 |
| decode M=8-96 稠密带宽假设未证 | 中 | 专用微基准（routeB W4A4 @ M=8/96 vs bf16）|
| 模型 MLP 路径改动比 routed MoE 挂点更侵入 | 中 | 与 FI 0.6.17 共享专家融合 API 并行评估（上游把 shared 融进 MoE 调用，可能更省）|

### 1.4 测试验证门设计（对照 W4A16 基线 + LuZ0.3.1）

> 注意：**对照基线 = LuZ0.3.1（routed MoE 已 W4A4 full）**，不是 W4A16 基线——shared 改动是叠加在 LuZ0.3.1 之上的增量。

| 门 | 判据 | 参考值（LuZ0.3.1 实测） | 判定 |
|---|---|---|---|
| 内存门 | weight 增量 = 派生 scale/布局转换（目标 < +2GB/rank）；KV ≥ 5.3M | 45.32 GiB / 5.73M | PASS |
| 质量门 | golden 4 稳定 prompt 逐字一致 + logprob 对齐 LuZ0.3.1（mean Δ ≤ 0.02、无发散） | fox_repeat/count/code/list | PASS |
| 质量门 | 困惑度 Δ 带内（W4A4 先例：routed MoE golden 4/4 + 0.36-0.41% logprob 级） | 与 LuZ0.3.1 参考 | PASS |
| 性能门 | PR 四档 ≥ LuZ0.3.1 -3%（无回退）；目标 +1~2% | 2950.5/2943.6/2834.2/2550.0 | PASS / 收益确认 |
| 性能门 | DE C1/C12 step_eff ≥ LuZ0.3.1 -3%（18.2/80.2 参考；**decode 带宽削减若兑现应更好**） | 18.2 / 80.2 | PASS |
| 微基准 | shared 三形状 × M=8/96/512/1024/4096：routeB KO/E2E vs bf16 基线；量化开销占比 < GEMM 节省 | P0 方案 B 扩展 | 数据门（非 go/no-go）|
| 回归 | cudagraph 三档捕获完整；日志 0 ERROR；needle 64K 3/3 | — | PASS |

**关键决策点**：DE C1/C12 门 + 微基准中"量化开销 < GEMM 节省"是 go/no-go；任一不满足则 shared 方向退化为设计储备（沉没成本可控）。

---

## 2. 调研任务 2：routeB M=1024 → 512-768 下探

### 2.1 kernel 支持度 [实测-源码 + 实测]

- **routeB DSL kernel 无 M≥1024 硬约束**：`dense_blockscaled_gemm_persistent_pingpong.py` 用 PersistentTileScheduler（`_compute_grid` 对 C 做 `zipped_divide` 于 128×128 epi tile，partial tile 由 persistent scheduler 覆盖）；对齐约束 `is_valid_tensor_alignment` 对 FP4 A（m-major）要求 **M%32==0**（16B/4bit=32 元素），M=512/768/256 均满足。block 形态/尾数处理无障碍 [实测-源码]。
- **"MIN_M=1024"是分派策略**（wrapper `VLLM_MOE_W4A4_MIN_M` 默认 3072；稠密池设计阈值 1024），非 kernel 限制。
- **小 M 效率实测（kernel-only）**：
  - routeB P4 矩阵（dense 形状，KO）：w1/w3（N=2048,K=4096）M=256→215T、M=512→240T、M=1024→322T；w2（N=4096,K=2048）M=256→209T、M=512→254T、M=1024→300T [实测]
  - merged v2 M_g 曲线（N=12288）：256→157T、512→254T、768→314T、1024→338T、1536→351T、4096→367T [实测]
  - **平台 350T 从 M≥1536 开始**；M=512-768 处于 215-314T（形状相关），仍为 bf16 池（55-65T）的 3-5×。
- **routeB E2E（含 A 量化）硬约束**：P4 实测**全部 M 档无一处 ≥1.0× vs routeA**（M<256 落后 14-47%；M=256-4096 仅 0.53-0.97×）[实测]——瓶颈是 A 量化（双 pass triton 94-136GB/s vs vLLM C++ ~200GB/s；病理单 pass 仅 20GB/s），非 GEMM。**这条结论直接约束"下探"的收益兑现**。
- routeB 已知禁区：M=16384 崩塌 0.35×、K=14336 dense 投影 0.42-0.60× [实测]。

### 2.2 收益评估（预期收益曲线）[推算，待微基准]

把"下探 MIN_M"拆成两段收益机制：

**(a) prefill 中 M 段（M=512-1023）搬进 routeB**：
- 当前：该段（并发混批/尾 chunk）落在 bf16 池 55-65T。
- 下探后：routeB KO 215-314T（形状相关）→ 该段节点级 3-5×。
- **扣 A 量化**：P4 已证 E2E 无一处胜出，故**必须先修 A 量化（fused prologue / C++ 单 pass）**；否则该段收益为负。
- 负载占比：并发 C6/C12 下 4096 chunk 被拆，M∈[512,1024) 段占比依赖实际调度分布 [推算：低个位数 %]，折算 PR 增量 **+0.2~0.6%**（乐观，量化修复后）。

**(b) decode 小 M 段（M=8-96 带宽受限区）搬进 routeB/W4A4**：
- 机制不同：稠密节点每步全量读权重，带宽主导（shared 0.54GB→W4A4 0.135GB→省 ~1.5ms/step；lm_head 0.27GB→0.07GB→省 ~0.7ms/step）[推算]。
- **该段不依赖"下探到 512-768"**——需要的是稠密节点 decode 分派（MIN_M≈0 或单独规则）；P4 对 routed 形状测的"M<256 routeB 落后"**不能直接迁移到稠密带宽场景**（routed 只读路由专家子集、字节节省小）[推算]。
- 风险：routeB/W4A4 在 M=8-96 的 launch/prologue 开销 + A 量化 [M=96,4096] 虽小但 kernel 占用不足；**需专用微基准**。

**曲线结论**：真正搬进"甜点区"（M≥1536）的只有 prefill 大 chunk（已由 W4A4 full 捕获）；M 下探是把"次甜点"（215-314T）也纳入 routeB，**前提是先解决 A 量化 E2E 短板**。就既有证据看，"M=1024→512-768 下探"作为**独立自研方向**的投入产出弱于 shared/lm_head（它是**前提依赖型**：收益被 A 量化卡住）。

### 2.3 实施路径

1. **前置（不可跳）**：修 routeB A 量化——C++/CUDA 单 pass（P4 行动项 A1）或 fused prologue 全面接入（phasec 原型生产化）。
2. **P0 方案 B 扩展微基准**：routeB dense 形状（shared N=1024/K=1024、lm_head N=32320、attn N=4096）× M=256/512/768/1024/4096，KO 与 E2E 双口径。
3. **decode 稠密带宽专项**：routeB W4A4 @ M=8/96 稠密节点 vs bf16——验证"带宽削减 > 小 M 开销"假设。
4. **分派实验**：若 2、3 数据为正，改 MIN_M（稠密池分派 1024→512 或 768）→ 回归门（PR 四档/DE/质量）。
5. **kernel 级（可选，若 2 数据不足）**：小 M tile/调度调优（routeB README 已列 128×128 tile 作 decode 备选；P4 实测 tile 128³ + epi 128×128 已定版，小 M 增益空间有限 [实测]）。

### 2.4 风险

| 风险 | 级别 | 说明 |
|---|---|---|
| A 量化 E2E 吃掉全部 GEMM 收益（P4 已实测） | **高（当前状态）** | 不修 A 量化，"下探"无意义；这是 P4 的既有结论 |
| 小 M kernel 启动开销/占用不足 | 中 | M=256-512 网格小（4-8 M-tiles × 8 N-tiles）；persistent kernel 在小网格下效率有限 |
| 量化精度损失（小 M 更敏感） | 中 | A 侧 FP4 量化 + W 侧 FP8→FP4（shared）叠加；质量门覆盖 |
| 稠密 decode 带宽假设不成立 | 中 | P4 routed 数据相反（0.83-0.92×）；需专用测试 |
| routeB 形状敏感（N 小/K 小/K 大/M 超大） | 中 | P4 已证 N=2048/4096 OK、K=14336 崩、M=16384 崩；shared N=1024/K=1024 未测 |

---

## 3. 优化对象推荐排序（⚠ 待 P0 拆账实测校准）

> 前提说明：P0 拆账实测（fi017 方案 A/B，窗口待排）尚未完成。以下排序基于既有实测 + [推算]，P0 后按实测份额 ±1 位浮动。

| 排序 | 对象 | 收益 [推算] | 就绪度（修正后） | 关键前置 | 排序依据 |
|---|---|---|---|---|---|
| **0** | **P0 拆账实测**（方案 A 归因 + 方案 B 三节点微基准） | 锁定份额、消除 ±1 位浮动 | 高（半天级、不占生产 GPU） | 无 | **一切排序的前提** |
| **1** | **shared experts → routeB/W4A4（或 FP8）** | PR +0.8~1.5%（已扣 A 量化）[推算] + decode -1.5ms/step | 中-高（kernel 有、质量门先例有；**权重派生需新建、形状效率未测、A 量化扣减**） | ①shared 形状微基准；②A 量化融合；③FP8→FP4 校准门 | M=4096 甜点 + 全 token + 质量门可迁移；仍是 bf16 池最干净首发；但**非零拷贝、快赢弱化、与 lm_head 收益差收窄** |
| **2** | **lm_head → W4A4** | decode 步时 -5~6%（唯一打带宽墙）+ prefill 增量 + 显存 -190MB/rank [推算]；**A 量化每步仅 1 次（占比远低于 shared 的 ×43）** | 中（routeB kernel 有；质量门流程待建） | 校准 + KL 门先行 | ROI 高但质量门槛最高；**扣 A 量化后与 shared 的收益差显著收窄，P0 后可能反超为第一** |
| **3** | **attn 投影 → 分档量化（o_proj 灰度先行）** | PR +2~4%（仅 prefill 半场）[推算] | 中-高（形状多、数值风险分层） | o_proj 灰度 → q/kv/lora 逐层门 | FLOPs 份额最大（57%）但仅半场可转化；decode 明确不做 |
| **4** | **routeB M=1024→512-768 下探** | +0.2~0.6%（prefill 中 M 段）[推算]；decode 稠密带宽待验 | 低-中（**前提依赖型**） | 先修 A 量化（P4 A1）→ 微基准 → 分派实验 | 作为 shared/lm_head 的**伴随研究轨道**而非独立立项；收益被 A 量化卡住 |

**排序说明**：
1. **shared 首发仍成立但理由改变**：从"零拷贝、资产全齐、快赢"修正为"甜点形状 + 质量门迁移 + 全 token 覆盖"，且**必须设微基准/量化门为前置**（§1.4 决策点）；扣 A 量化后收益降至 +0.8~1.5%，与 lm_head 差收窄。
2. **lm_head 的"第二"位置是结构性稳健的**（唯一同时打 decode 带宽墙 + prefill + 显存），且 **A 量化每步仅 1 次**（相对 shared 的 ×43 量化开销小一个量级）——P0 后 lm_head 反超 shared 的概率**高于既有报告表述**。
3. **routeB M 下探不单独排高**：它服务的是"把更多段搬进甜点"，但 P4 已证 E2E 短板在 A 量化——先修量化，下探自然顺带获益；把 A 量化修复列为 P0 级前置工程与 shared/lm_head 共用。
4. **条件触发**：若 P0 实测 shared 份额 <6µs/token（当前推算 9-12µs）或 lm_head 份额 >25%，shared 首发位置让给 lm_head。

---

## 4. 下一步窗口任务清单

| # | 任务 | 归属 | 前置 | 预计 | 产出/判据 |
|---|---|---|---|---|---|
| W1 | **P0 profiler 拆账**（方案 A：fork 暴露 /start_profile 或 VLLM_TORCH_PROFILER_DIR 重启；方案 B：一次性容器三节点微基准） | SRE/架构 | 无 | 0.5-1 天 | shared/lm_head/attn µs 实测份额（M=4096 当前形态）→ 校准 §3 排序 |
| W2 | **P0 方案 B 扩展：shared 三形状 routeB 微基准**（[M,4096]×[4096,1024] + [M,1024]×[1024,4096]，M=8/96/512/1024/4096，KO + E2E vs bf16） | 架构/kernel | W1 | 0.5-1 天 | shared 方向 go/no-go 数据门（量化开销 < GEMM 节省）|
| W3 | **routeB A 量化修复**（C++/CUDA 单 pass 或 fused prologue 生产化） | kernel 工程 | 无（可并行） | 2-3 天 | 消除 P4 "E2E 全 M 落后" 短板；量化 ≥ 150GB/s |
| W4 | **shared experts W4A4 集成设计 + L1**（FP8→FP4 派生链 + DeepseekV4MLP hook 或 B12xMoEWrapper(E=1,topk=1) + env 开关 + 质量门计划） | 架构 | W2 数据为正 | 2 天 | L1 全过（含 off 路径逐字等价）|
| W5 | **shared 窗口 A/B**（对照 LuZ0.3.1，§1.4 全门） | SRE | W4 | 0.5-1 天窗口 | 内存/质量/性能三门 + DE C1/C12 |
| W6 | **decode 稠密带宽专项微基准**（routeB W4A4 @ M=8/96 稠密节点 vs bf16） | 架构/kernel | 可与 W2 合并 | 0.5 天 | "带宽削减 > 小 M 开销" 假设成立/证伪 |
| W7 | **lm_head W4A4 立项准备**（per-channel 校准 + KL 门先行） | 架构 | W1 | 1-2 天 | 校准方法 + 质量门方案定稿 |
| W8 | **U1：FI 0.6.17 升级**（Go 有条件，fi017 §1）——含**共享专家融合 API** 探路（可能替代/叠加 shared 自研集成） | SRE/架构 | 无 | 1-2 天 | W4A4 decode 中性度定论 + 上游共享融合可用性 |
| W9 | attn 投影 o_proj 灰度方案设计（分档量化 + 质量门） | 架构 | W1/W4 | 1 天 | 灰度顺序 + 门判据 |
| Watch | b12x 上游 PR #227/#222、#234 skinny-M tile、FI 0.6.18、vLLM #41834 对账 | — | — | 每周 | 上游修复/吸收动态 |

**窗口编排建议**：W1 与 W3 可并行（W3 不依赖拆账）；W2/W6 可合并为一次 GPU 微基准窗口（共享 GPU 纪律，一次性容器 <1GB 显存）；W4 与 W5 串行（L1 通过才开生产窗口）；W8 与 W4 并行（FI 0.6.17 的共享融合 API 若验证可用，shared 的集成形态可能切换）。

---

## 5. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| **[实测]** | routeB P4 A/B 矩阵（w1/w3/w2 × M=256-16384，KO/E2E；§2.1-2.4，routeb-p4-ab-perf-2026-08-21.md）；phasec M_g 曲线（256→157T…4096→367T，routeb-merged-phasec §4）；bprime mid-M staging（M=8/96/512/2048/3071，bprime-window §6）；W4A4 wrapper 小 M（routea-integration §2.3：M=8 0.92×/M=64 0.83×/M=4096 1.32×）；LuZ0.3.1 三门（luz031 §3）；生产 env（node01 start_tp4_head.sh 勘察）；bf16 池 55-65T/29-34µs（fi017 §2.0）|
| **[实测-源码]** | **shared experts 权重 = F8_E4M3 block-scaled + F8_E8M0 [16,32]/[32,16]（checkpoint 直接解析，-0731 与 -nvfp4 同构）**；routed experts payload = U8 FP4 + E8M0 [N,K//32]；config `n_shared_experts:1`、`quant_method: fp8`；`DeepseekV4MLP`（MergedColumnParallelLinear/RowParallelLinear）；routeB kernel PersistentTileScheduler + `is_valid_tensor_alignment`（M%32 for FP4 m-major）；routeB tile 128³ + epi 128×128、tile_k=128 锁定 |
| **[推算]** | shared 份额 9-12µs/token（fi017 效率调整带）；shared→W4A4 **PR +0.8~1.5%（已扣 A 量化；upstream-check +1.5~2.5% 为未计量化的上限口径）**；decode -1.5ms/step（shared）/ -0.7ms/step（lm_head）；lm_head prefill +0.85% 量级 + decode 步时 -5~6%；routeB M=512-768 收益 +0.2~0.6%；N=1024 小 N 效率折扣（250-300T→200-250T）；稠密 decode 带宽假设 |
| **[上游实证]** | FI 0.6.17 发布 + 共享专家融合 API（#4159/#4088 等，fi017 §1.1）；b12x 1.2.6 后零提交（upstream-check §1.1）|
| **[待窗口验证]** | P0 拆账（W1）；shared 形状微基准（W2）；A 量化修复（W3）；shared 窗口 A/B（W5）；decode 稠密带宽专项（W6）；shared 当前 compute 路径（bf16 vs FP8 GEMM）定论 |

**诚实声明**：
1. **shared experts 零拷贝适配器是本次最重要的修正**——既有 P1 行把 routed MoE 的 E2M1 直配适配器误套到 shared 节点；shared 权重实际为 FP8 E4M3 [实测-源码]。该修正不影响"shared 是池内可转化节点"的定性，但改变了就绪度、收益下限与风险结构。
2. routeB M 下探的收益估算受 P4 "E2E 全 M 落后"硬约束支配——本报告未把该方向按"已证可行"处理，而是标注为"前提依赖型"（先修 A 量化）。
3. shared/decode 的带宽收益按"每步全量读权重"假设（同 fi017 §2.3），未计 page cache/重叠；实际以 profiler/微基准为准。
4. 排序在 P0 实测前有 ±1 位浮动（尤其 shared 与 lm_head 之间）；P0 后按实测份额重排。
5. 全部服务器勘察为只读（checkpoint 解析、config/脚本 cat、一次性 CPU 容器读镜像源码），未启动生产、未触碰 GPU。

---

## 6. 引用索引

- 本报告关键实测源：routeb-p4-ab-perf-2026-08-21.md（A 量化 E2E 短板 + M 矩阵）、routeb-merged-phasec-2026-08-21.md（M_g 曲线）、bprime-window-2026-08-23.md（mid-M staging）、fi017-p0-accounting-2026-08-23.md（池拆账 + P1 顺序）、upstream-check-perf-ceiling-2026-08-23.md（P0-P3 + 拐点判定）、luz031-deployment-2026-08-23.md（生产形态）、routea-integration-design-2026-08-21.md（W4A4 wrapper 小 M + 适配器契约）、a3-hybrid-slim-design-2026-08-23.md（内存账 + b′）、engineering-assets-report-2026-08-22.md（A2 routeB standby / A4 适配器）
- 服务器勘察：node01 `<INSTALL_DIR>/models/deepseek-v4-flash-0731/`（checkpoint 解析）、`<INSTALL_DIR>/scripts/start_tp4_head.sh`（env）、`<INSTALL_DIR>/nvfp4/`（routeB 存档/插件）、一次性 CPU 容器镜像源码（`vllm/models/deepseek_v4/nvidia/model.py`）
- 上游：github.com/flashinfer-ai/flashinfer（0.6.17 + 共享专家融合）、github.com/local-inference-lab/b12x（PR #227/#222、#234）

*本报告由工程保障团队（系统架构师 architect-1）生成；排序与立项请由人类工程负责人结合 P0 拆账实测与共享窗口数据裁定。*
