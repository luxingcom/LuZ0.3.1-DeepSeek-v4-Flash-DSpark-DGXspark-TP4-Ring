# FI 0.6.17 升级评估 + bf16 稠密池 P0 拆账（2026-08-23）

- **执行**：阿奇（Archi）· 系统架构师（architect-2）
- **任务**：①FlashInfer 0.6.16 → 0.6.17 升级评估（U1 路径，Go/No-Go + 升级步骤 + 验证门）；②bf16 稠密池 37% 线性 FLOPs 的 P0 profiler 拆账（shared experts / lm_head / attn 投影 各自 FLOPs 与带宽占比）+ P1 立项顺序建议
- **纪律**：只读分析（web 核验 + 既有报告复用），不占 GPU、不动生产、无网络发布操作
- **口径标注**：**[上游实证]** = 上游仓库/Release/PyPI/文档页面直接核验；**[既有账-实测]** = 本团队既有报告实测口径；**[推算]** = 基于 Roofline/形状/FLOPs 比例的推算，需 P0 profiler 实测后采信；**[待窗口验证]** = 本环境无法验证，必须由窗口执行

---

## 0. 一页结论

1. **U1（FI 0.6.17）判决：Go（有条件）**。wheel 可用性已在线核验 [上游实证]：PyPI 发布 `flashinfer_python-0.6.17-py3-none-any.whl`（纯 Python + 首用 JIT/下载 kernel，与 0.6.16 部署同机制），Python 3.12 支持（`requires_python <4.0,>=3.10`），CUDA 13.0 在官方支持带（12.6/12.8/13.0/13.1），aarch64/GB10 适用（DGX Spark SM120/SM121 明确支持），CuTe-DSL 运行时下限**降回 4.5.2**（与生产 cutlass 4.5.2 精确匹配，这是 0.6.17 相对 0.6.16 的重要顺风项）。**条件**：① 5 个 fork 补丁需在 0.6.17 树上重放（其中 cuDNN bf16 ban / SM100 早退两项可能被上游 SM12x 同步部分吸收，需 diff 确认）；② 58 个 fork 新增文件（moe_ep/mega EP 树）需与 0.6.17 上游已落地的 MegaMoE `flashinfer.moe_ep` 对账；③ 测试容器三门全过才动生产。①②待窗口验证，③为既定纪律。
2. **P0 拆账（推算 → 实测）**：池总量 29-34µs/token [既有账-实测，M=1024 口径]，按 FLOPs 比例（attn 投影 57% / shared 29% / lm_head 14%）的均匀效率模型拆分 ≈ attn 17-19 / shared 8-10 / lm_head 4-5µs；考虑各节点形状 MFU 差异后带加宽为 attn 15-19 / shared 9-12 / lm_head 3-5µs [推算]。decode 带宽墙侧：C1 池权重读 1.9GB/步 [既有账-实测]，拆为 attn 1.1GB（57.6%）/ shared 0.54GB（28.3%）/ lm_head 0.27GB（14.1%）。
3. **P1 顺序建议（P0 实测前维持，实测后 ±1 位浮动可能）**：**shared experts 首发**（资产全齐、M=4096 甜点、数值风险低、PR +1.5~2.5% [推算] 且顺带 decode 带宽受益 ~1.4ms/步）→ **lm_head 第二**（唯一同时打 decode 带宽墙，C12 步时 -5~6% [推算]，质量门槛最高需校准+KL 门先行）→ **attn 投影第三**（FLOPs 份额最大但仅 prefill 半场可转化，o_proj 灰度先行，PR +2~4% [推算]）。

---

## 1. FI 0.6.17 升级评估（任务 1）

### 1.1 wheel 可用性核验 [上游实证]

| 项 | 核验结果 | 口径 |
|---|---|---|
| 版本存在性 | PyPI `flashinfer-python` 0.6.17 存在（2026-08-11 发布，GitHub Release 标记 Latest）；0.6.18 nightly 至 08-19 | [上游实证] |
| wheel 形态 | `flashinfer_python-0.6.17-py3-none-any.whl`（16.0MB，sha256 `ab7ce3ae…`）——**纯 Python wheel**，kernel 首用 JIT 编译/下载；另有 sdist | [上游实证] |
| Python | `requires_python = <4.0,>=3.10` → **Python 3.12 支持**（生产 `/usr/bin/python3` = 3.12） | [上游实证] |
| CUDA | 官方支持 12.6 / 12.8 / **13.0** / 13.1（生产 torch 2.11.0+cu130） | [上游实证] |
| aarch64/GB10 | 官方文档 DGX Spark/GB10 专项说明存在（SM120/SM121、`12.1a` arch target）；纯 Python wheel + 设备端 JIT 与 0.6.16 部署机制完全一致（0.6.16 已在 GB10 四节点生产运行） | [上游实证] |
| 依赖 | `apache-tvm-ffi / click / cuda-python>=12.0 / cuda-tile>=1.4.0 / einops / nccl4py>=0.3.1 / ninja / numpy / nvidia-cudnn-frontend>=1.25 / nvidia-cutlass-dsl>=4.5.0 / nvidia-ml-py / packaging / requests / tabulate / torch / tqdm`——生产镜像已全数在位（fi016 冒烟同基镜像）；cu13 extra = `nvidia-cutlass-dsl[cu13]>=4.5.0` | [上游实证] |
| 可选预编译包 | `flashinfer-cubin`（全架构 cubin）/ `flashinfer-jit-cache`（按 CUDA 版本，cu129/cu130）从 `https://flashinfer.ai/whl` 提供——冷启动加速用，非必需（生产已管 autotune 缓存 + JIT 磁盘缓存） | [上游实证] |

**0.6.17 关键增量与本集群相关度 [上游实证]**：

| 增量 | 与我们相关的点 |
|---|---|
| **NVFP4 W4A4 decode/prefill kernel parity on SM120/SM121**（#4285 同步 SM12x fused-MoE 到 b12x HEAD）+ **两个 NVFP4 量化精度修复 + `input_global_scale`**（#3932） | 直接作用于 LuZ0.3.1 的 W4A4 full 形态；是**把 W4A4 decode 中性度两口径悬案（phase3b -6~-9% vs w4a4-ext ±3%）拉回中性/正值**的头号候选 |
| **W4A16 小 batch 张量核 decode 路径 + cooperative persistent launches + shape-stable route packing**（#4255） | 作用于现生产 W4A16 decode 段（C1 M=8 / C12 M≈96）；潜在 decode 带宽墙左侧曲线抬高 |
| **共享专家融合（FP4）+ 统一 MoE API MXFP4 W4A8/W4A16 + SiTU**（#4159/#4088/#4104/#4239/#4180） | 与我们 bf16 池 shared experts 节点直接相关——若 shared experts 迁入 FP4 MoE 调用可消掉独立 bf16 shared-expert GEMM 调用开销（P1 探路项） |
| **CuTe-DSL 运行时下限 4.5.2**（#4101，drop 4.6.1 floor） | 与生产 cutlass 4.5.2 精确匹配——0.6.16 时代的 DSL 版本摩擦风险下降 |
| Kimi K3 MLA decode（spec q-len 8，同 DSpark MTP n=7 射程）+ no-rotary-tail 形状（#4178/#4108）+ SM121 WY decode 修复（#4117）+ MiniMax-M3 sparse MLA | P5 attention 方向（FI 侧唯一现实通道）的远期弹药，非 U1 主目标 |
| MegaMoE `flashinfer.moe_ep` 生产就绪（CUDA graph 捕获、融合 quant+stage、pre-quantized weight pack、跨层对称缓冲池化、持久 knob 缓存） | **上游已有跨层池化/预量化包**——与我们自做 wsdedup/池化补丁方向一致，且是 58 个 fork 文件对账的主体 |

### 1.2 与当前 vLLM fork（b12x 树）+ FI 0.6.16 挂载方式兼容性

**挂载机制（fi016 先例）**：目录级 bind-mount 覆盖镜像内 dist-packages，零 pip 变更。0.6.17 沿用同一机制，只需新树目录 `flashinfer-0.6.17/` + start 脚本两行挂载改动（`<INSTALL_DIR>/nvfp4/flashinfer-0.6.17/flashinfer` → `dist-packages/flashinfer` + `~/flashinfer-cache` 不变）。`FLASHINFER_DISABLE_VERSION_CHECK=1` 继续屏蔽 dist-info 版本滞后噪音。

**风险点（均需窗口 diff/冒烟验证）**：

| # | 风险 | 说明 | 处置 |
|---|---|---|---|
| R1 | **5 个 fork 补丁重放**：cuDNN bf16 ban / SM100 早退 / 2 trivial / artifact hash | 0.6.17 同步了 SM12x fused-MoE 到 b12x HEAD——**SM100 早退与部分 dispatch 修复可能已被上游吸收**；cuDNN bf16 ban 属 fork 特有防护，大概率仍需 | 窗口 diff 逐项确认：能省则省（少 patch 更优），不能省则 rebase 后 compileall 0 错误 |
| R2 | **58 个 fork 新增文件（moe_ep/mega EP 树）对账** | 0.6.17 上游已含 `flashinfer.moe_ep`（MegaMoE），与 fork 树同源但版本可能不同；vLLM fork 依赖 `B12xMoEWrapper / b12x_fused_moe / moe_ep.backends.mega` 符号 | 优先**复用上游 0.6.17 moe_ep 树**，只保留 fork 独有符号的 shim；对账清单 + import 面验证（fi016 §2.4 同款 11 项） |
| R3 | **b12x 树回归**（fork `utils/flashinfer.py` 调用面） | 0.6.17 的 MLA/fused_moe 变更总体**加法**，但 autotuner cache 路径版本目录变化（`autotune_cache/0.6.16/121a/` → `0.6.17/121a/`）→ 启动期 24 configs 重新 autotune（一次性冷启动代价） | 启动核验接受一次性 autotune 耗时；观察是否影响 rendezvous 超时（`--distributed-timeout-seconds 900` 已就位） |
| R4 | 依赖版本漂移 | 若走 pip 安装 0.6.17 会重解析 deps 可能升级 cutlass-dsl/torch——**必须 `--no-deps` 或 bind-mount 方式**（fi016 已规避，沿用） | 纪律约束 |

### 1.3 Go/No-Go 判决

> **Go（有条件）** —— 与 upstream-check §1.5 U1 定论一致，wheel 路径成立。前置条件（按序）：① fork 补丁重放 diff 干净（R1）；② moe_ep 树对账 + 容器 CPU import 冒烟（R2）；③ GPU 冒烟 5/5（mirror windowA 矩阵）；④ 测试容器三门全过；⑤ 生产窗口三门全过才置生产。任一前置不满足即回退到 0.6.16 终态（回滚链 §1.5）。

**不满足 No-Go 的条件**（列出以便窗口对照）：fork 补丁无法在 0.6.17 上 clean apply 且 rebase 后 compileall 报错、或 B12xMoEWrapper 符号缺失导致 vLLM 调用面不可用、或测试容器三门任何一门硬 FAIL（PR 出 ±3% 带外、质量门逐字 DIFF、回归日志 ERROR）——任一发生即 No-Go，维持 0.6.16。

### 1.4 升级步骤（U1，含回滚锚点）

| 阶段 | 步骤 | 预计 | 产出 |
|---|---|---|---|
| 0 预取 | 有网机器 `pip download flashinfer-python==0.6.17 --no-deps`，核 sha256 `ab7ce3ae…`；可选下载 `flashinfer-cubin` + `flashinfer-jit-cache`（cu130） | 0.5h | wheel + md5 |
| 1 组树 | 解包 wheel → 重放 5 fork 补丁（逐项 diff，吸收上游已修项）→ moe_ep 树对账 → `python3.12 -m compileall` 0 错误 → 打 tarball + md5 | 2-4h | `flashinfer-0.6.17-rebased.tar.gz`（含 rebase 清单） |
| 2 CPU 冒烟 | 一次性容器（基镜像 LuZ0.3.1）CPU import 全链（fi016 口径 22/23，TVM FFI 伪缺陷除外）+ vLLM 调用面 11 项 | 0.5h | import 清单 |
| 3 GPU 冒烟 | mirror windowA 5/5：B12xMoEWrapper 小 GEMM 输出**与 0.6.16 逐位一致**、b12x 关键符号、trtllm_ragged_attention、CuTe-DSL JIT 磁盘缓存、NVFP4 W4A4 小 GEMM（0.6.17 精度修复后与 bf16 参考误差带） | 1h | res_*.json |
| 4 测试容器三门 | 克隆环境或测试容器：PR 四档 ±3%（vs LuZ0.3.1 2950/2943/2834/2550）、C6/C12 ±5%、DE C1/C12 ±5%、质量门 4/4、needle 64K 3/3、日志 0 ERROR | 2-3h | 三门判定 |
| 5 生产替换 | 新树分发四机 → start 脚本 2 行挂载改 0.6.17 → `.bak-fi017-20260823` 四机留档 → checker 4/4 → head-first 重建 → 启动核验（**含 flashinfer 版本项**，luz031 教训）→ 生产三门 | 1-2h | 生产终态 |
| 6 E2 专项 | W4A4 decode 中性度长轮次定论（0.6.17 + W4A4 臂，DE 8 轮 C1/C12） | 0.5-1h | E2 关闭/维持 |

**回滚锚点（全部 <10 分钟）**：
1. 四机 `start_tp4_{head,worker}.sh.bak-fi017-20260823`（=0.6.16 注入态快照）恢复 + head-first 重建 → 容器内 `flashinfer.__version__` 回归 0.6.16。
2. **CRITICAL（luz031 §2 教训）**：恢复必须用 **fi017 注入前的快照**，不得误用旧 .bak；重建后必须核 flashinfer 版本项，杜绝"误回滚 2.5h 未发现"复发。
3. 兜底：LuZ0.3.1 自包含恢复镜像（FI 0.6.16 已 bake）`restore_luz031.sh`。
4. 0.6.16 树目录不删（未被挂载即不生效）；新树保留留档。

### 1.5 验证门（三门 + 专项）

| 门 | 判据 | 参考值（LuZ0.3.1 实测） | 判定 |
|---|---|---|---|
| 性能门 | PR 四档 ±3% | 2950.5 / 2943.6 / 2834.2 / 2550.0 | PASS |
| 性能门 | C6 / C12 ±5% | 3057 / 3056 | PASS |
| 性能门 | DE C1 / C12 step_eff ±5% | 18.2 / 80.2（接受率归一） | PASS |
| 质量门 | greedy 稳定 4 prompt 逐字一致（fox_repeat/count/code/list） | vs LuZ0.3.1 参考 | PASS |
| 质量门 | GPU 冒烟 B12X GEMM 与 0.6.16 逐位一致 | torch.equal=True | PASS |
| 回归观察门 | 四机日志 ERROR/Traceback = 0；autotune 24 configs 正常；needle 64K 3/3；KV ≈5.73M | — | PASS |
| **E2 专项** | W4A4 decode 中性度：0.6.17 + W4A4 臂 vs 0.6.16 + W4A16，DE 8 轮 C1/C12，step_eff ±3% 带内视为中性；若 >+3% 即为正收益（精度修复 + 小 batch 路径兑现） | 现两口径并存（-6~-9% vs ±3%） | 定论 |

### 1.6 待窗口验证清单（本环境不可验证，明确移交）

1. R1：5 fork 补丁在 0.6.17 树的 apply/diff 结果（尤其 cuDNN bf16 ban / SM100 早退是否被上游吸收）。
2. R2：58 fork 文件与 0.6.17 上游 moe_ep 对账结果 + `B12xMoEWrapper`/`b12x_fused_moe`/`moe_ep.backends.mega` 符号在位。
3. R3：vLLM fork `utils/flashinfer.py` 调用面在 0.6.17 下 CPU import + GPU 冒烟。
4. wheel 完整性（sha256）+ 可选 cubin/jit-cache 冷启动对比（非阻断）。

---

## 2. bf16 稠密池 P0 拆账（任务 2）

### 2.0 池总量账（既有账）

- **线性层 FLOPs 账 [既有账-实测]**：全模型 20.5G/token = routed MoE 13.0G（63.4%）+ **shared 2.16G（10.5%）+ attn 投影 4.3G（21.0%）+ lm_head 1.07G（5.2%）**；bf16 稠密池 = 7.53G = **36.7% ≈ 37%** ✓
- **池时间 [既有账-实测]**：PR 瀑布 M=1024 口径，池 30-35ms/步（7-9% of 407ms）≈ **29-34µs/token**；per-rank 池 FLOPs = 1.88G/token → 有效 55-65T（bf16 峰值 125T → MFU ~44-52%）
- **池权重/rank [既有账-实测]**：~5.9GB = shared 0.54 + attn ~1.1 + lm_head 0.27 + embed/MTP ~4.0
- **注**：29-34µs 为 M=1024 口径（pr-de 08-22）；LuZ0.3.1 已 M=4096，池 per-token 时间应更低（MFU 上升）。**P0 必须在 M=4096 当前生产形态下重测**，否则 +5~7% 总账绝对值需缩放。

### 2.1 三节点几何与 FLOPs 账

模型几何（config 实证）：L=43 全 MoE 同构层、hidden=4096、E=256/topk6/intermediate=2048、n_shared=1、MLA（64 头 × 512、q_lora=o_lora=1024、1 KV 头）、vocab=129280、MTP n=7。TP4：intermediate 512/rank、vocab 32320/rank。

| 节点 | 每层全模型 FLOPs（推算几何） | 全模型/token | per-rank/token | 权重/rank | 生产形状（prefill M=4096） |
|---|---|---|---|---|---|
| shared experts | 3×2·4096·2048 = 50.3M/层（gate+up+down） | 2.16G | 0.54G | 0.54GB（bf16） | 43 层 × [4096×4096]×[4096×512]×2 + [4096×512]×[512×4096] |
| lm_head | 2·4096·129280 = 1.059G | 1.07G | 0.27G | 0.27GB（bf16） | 1 次 × [4096×4096]×[4096×32320] |
| attn 投影（q/kv/o+lora） | 差额：4.3G/43 ≈ 100M/层 | 4.3G | 1.08G | ~1.1GB（bf16） | 43 层 × MLA q/kv/o（lora 1024 通路） |
| **合计** | — | **7.53G（37%）** | **1.88G** | **~1.91GB** | — |

（shared 几何与既有账 2.16G 自洽：2.16/43=50.2M ✓；lm_head 几何 1.059G 与 1.07G 自洽 ✓；attn 为差额项 [推算]——**这正是 P0 profiler 要测的**。）

### 2.2 µs 拆账表（推算，P0 前采信口径为带区间）

**模型 A：FLOPs 均匀效率**（null model，[推算]）：池 29-34µs × FLOPs 占比。

| 节点 | FLOPs 占比 | µs/token（均匀） | 效率调整带 [推算] | 与 upstream-check 一致 |
|---|---|---|---|---|
| attn 投影 | 57.1% | 16.6-19.4 | **15-19** | 12-18 ✓ |
| shared experts | 28.7% | 8.3-9.8 | **9-12**（N=512 小 N，MFU 偏低） | 8-12 ✓ |
| lm_head | 14.2% | 4.1-4.8 | **3-5**（N=32320 大 N，MFU 偏高） | 4-8 ✓ |

**不确定性标注**：三节点份额为形状/FLOPs 比例推算，非实测；排序（attn > shared > lm_head 的 µs 份额）在效率调整后仍稳健，但**立项顺序不依赖该份额精度**（见 §2.4）。

### 2.3 decode 带宽拆账（C1/C12 口径）

- C1 步池权重读 1.9GB/步 → ~7ms/步（17% of 41ms）[既有账-实测 pr-de §4]
- 字节拆账（权重/rank 比例，[推算]）：

| 节点 | 权重/rank/步 | @273GB/s | 占比 | W4A4 后字节 | 步时节省（C1 41ms 口径） |
|---|---|---|---|---|---|
| attn 投影 | 1.1GB | 4.03ms | 57.6% | 维持 bf16（decode M<1024，routeB 不可用） | 0（P3 明确 decode 不做） |
| shared experts | 0.54GB | 1.98ms | 28.3% | 0.135GB | **~1.5ms（+3.6%）** ← P1 顺带收益 |
| lm_head | 0.27GB | 0.99ms | 14.1% | 0.07GB | **~0.7ms**（≈5-6% of C12 ~12ms 口径） |

**lm_head 唯一性**：它是池内**唯一每 decode 步全量读取、且 W4A4 后字节 ÷4 无数值级联风险（logits 后置采样）**的节点；shared 也可转化但 decode 字节绝对值小（0.54GB），attn 因小 M FP4 效率反转明确不做——"lm_head 唯一同时作用于 decode 带宽墙"的定性成立 [推算-与 upstream-check 一致]。

### 2.4 可转化性（routeB 域内）与 P1 顺序建议

**routeB 平台 [既有账-实测]**：M≥1536 → 350T 平台，M=1024 → 338T，M≥4096 甜点（367T）；bf16 池当前 55-65T → 节点级 3.5-5×。

| 节点 | prefill 可转化（M=4096） | decode 可转化（M=8-96） | 预期收益 [推算] | 数值风险 | 资产齐备度 |
|---|---|---|---|---|---|
| **P1 shared** | ✅ M=4096 恰甜点 | 部分（W4A4 顺带 -1.5ms/步） | **PR +1.5~2.5%** | 低（稠密静态权重、质量门先例 golden 4/4） | **全齐**（routeB kernel + 零拷贝适配器 + W4A4 wrapper MIN_M + 池补丁在位） |
| **P1 lm_head** | ✅ M=4096 | ✅（唯一） | **decode -5~6% + prefill 增量 + 显存 -190MB/rank** | **最高**（logits 直驱采样分布，需校准 + KL 门 + 贪心逐字） | 中（routeB kernel 有、质量门流程待建） |
| **P2 attn 投影** | ✅（仅 prefill 半场） | ❌（M<1024 效率反转） | **PR +2~4%** | 中-高（q/kv 影响 attention score，o_proj 最低） | 中-高（o_proj 灰度先行） |

**P1 立项顺序建议（P0 实测后 ±1 位浮动可能）**：

1. **shared experts 首发**：资产零缺口（kernel/适配器/wrapper/池化全在位）、M=4096 恰在甜点、数值风险最低、PR +1.5~2.5% 且顺带 decode 带宽受益；**FI 0.6.17 共享专家融合 API 可在 U1 后作为替代/叠加路径评估**（若 shared 迁入 FP4 MoE 调用可消独立 GEMM 调用开销）。
2. **lm_head 第二**：ROI 最高（唯一打带宽墙 + prefill + 显存），但质量门槛最高——**先校准（per-channel）后集成，KL(量化‖bf16) 分布门 + 困惑度 + 贪心逐字 + 温度采样抽验**；FI 0.6.17 W4A16 小 batch TC 路径若验证可用，可作为 decode 侧 W4A16 折中（更低风险）选项。
3. **attn 投影第三**：FLOPs 份额最大但仅 prefill 半场可转化、数值风险分层——o_proj 灰度先行，q/kv/lora 逐层 perplexity + 长上下文 needle 门。

**条件触发**：若 P0 实测 lm_head prefill µs 份额 >25% 或 decode 墙份额 >20%（当前推算 14%），lm_head 可上提为首发；但因其质量门工作量更高，仍建议 shared 首发做快赢 + 资产复用。

### 2.5 P0 profiler 最小测量设计（待窗口执行，预计半天）

**目标**：把 §2.2/§2.3 的 [推算] 拆账转 [实测]；测量点在 **M=4096（当前生产 prefill 形态）** + decode C1（M=8）/C12（M≈96）。

**方案 A（主，真实负载归因，~30-60min）**：
- fork 暴露 `/start_profile`（当前 404，pr-de §1 E1）或设 `VLLM_TORCH_PROFILER_DIR` 重启单节点；`enable_layerwise_nvtx_tracing=True`（fork pass_config 现 False）。
- 单请求 4K（8.2K tok）prefill trace + 短 decode（C1/C12 各 30s）。产出 top-kernel 归属：shared experts / lm_head / attn q / attn kv / attn o /（lora）。
- 注意：trace 开启需重启（窗口任务）；测量纪律沿用双探针 + 停 healthcheck.timer + ≥3 轮中位。

**方案 B（次，一次性容器微基准，~30-60min）**：
- 一次性容器（基镜像 LuZ0.3.1，`--gpus all`，<1GB 显存共享纪律，mirror windowA）按生产形状直接跑三节点 bf16 GEMM 并记录 TFLOPs + 有效 GB/s：
  - shared：`[M×4096]×[4096×512]×2 + [M×512]×[512×4096]`，M∈{8,96,1024,4096}，×43
  - lm_head：`[M×4096]×[4096×32320]`
  - attn q/kv/o（lora 1024 通路）生产形状 ×43
- 产出每节点 µs/token（M=4096）与 ms/step（M=8/96）+ 带宽；与方案 A 交叉验证。

**方案 C（零 GPU，纯账）**：§2.2/§2.3 推算表本身即为 C，供 P0 前决策；P0 后替换。

**P0 判定口径**：若实测份额与推算带（attn 15-19 / shared 9-12 / lm_head 3-5µs）偏差 >±30%，以实测重排 P1；若偏差 <±30%，维持 §2.4 顺序。

---

## 3. 证据与假设分离清单

| 证据类型 | 内容 |
|---|---|
| **[上游实证]** | FI 0.6.17 发布（08-11）+ PyPI wheel 元数据（py3-none-any / requires_python / deps / sha256）+ 官方 docs（CUDA 12.6-13.1、DGX Spark SM121）+ Release 增量（#3932/#4285/#4255/#4253/#4130/#4101/#4178/#4108/#4117 + 共享专家融合）；b12x 1.2.6 后零提交、PR #227 Open（引 upstream-check） |
| **[既有账-实测]** | 线性 FLOPs 账（pr-de §2.1：20.5G 分解）；池 29-34µs/token（pr-de §3）；池权重 5.9GB 分解（pr-de §2.1）；C1 池权重读 1.9GB→7ms（pr-de §4）；routeB 332-368T / M≥1536 350T 平台（engineering-assets A1/A2 + routeb-merged）；LuZ0.3.1 三门数字（luz031 §3）；fi016 三门 + 挂载/回滚先例（fi016 §2/§3/§5）；windowA GPU 冒烟 5/5（fi016 同款流程） |
| **[推算]** | 池内三节点 µs 拆分（§2.2）；decode 带宽拆分（§2.3）；P1/P2/P3 收益（+1.5~2.5% / -5~6% / +2~4%）；lm_head 262MB/rank/步全量读假设；shared decode 顺带 -1.5ms/步 |
| **[待窗口验证]** | 5 fork 补丁 apply（R1）；58 fork 文件 vs 0.6.17 moe_ep 对账（R2）；vLLM 调用面 import + GPU 冒烟（R3）；P0 profiler 方案 A/B 执行 |

**诚实声明**：① attn 投影 4.3G 为既有账差额项，其内部 q/kv/o/lora 拆分几何未逐项验证（正是 P0 目标）；② lm_head decode 字节账按"每步全量读"假设，未计入 page cache/重叠；③ 池 29-34µs 为 M=1024 口径，M=4096 下应更低，+5~7% 总账在 P0 后需按当前形态重标定；④ P1 顺序在 P0 实测前有 ±1 位浮动可能。

---

## 4. 引用索引

- 上游：github.com/flashinfer-ai/flashinfer/releases（v0.6.17 08-11）、flashinfer.ai/releases、docs.flashinfer.ai/installation（CUDA 支持带 + DGX Spark SM121）、pypi.org/pypi/flashinfer-python/0.6.17/json（wheel 元数据）
- 内部：upstream-check-perf-ceiling-2026-08-23.md、fi016-replacement-2026-08-23.md、luz031-deployment-2026-08-23.md、windowA-fi-cg-budget-2026-08-23.md、w4a4-ext-2026-08-23.md、pr-de-bottleneck-analysis-2026-08-22.md、engineering-assets-report-2026-08-22.md、analysis-tp2-tp4-communication-2026-08-09.md、a3-hybrid-slim-design-2026-08-23.md、b12x-tail-path-strategy-2026-08-22.md

*本报告由工程保障团队（系统架构师 architect-2）生成；U1 生产落地与 P0 profiler 执行均须由窗口按纪律执行，P1 立项与否请由人类工程负责人结合 P0 实测裁定。*
