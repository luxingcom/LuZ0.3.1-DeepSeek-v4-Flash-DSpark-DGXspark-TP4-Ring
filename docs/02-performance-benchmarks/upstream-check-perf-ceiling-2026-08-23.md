# 上游核对 + 性能释放上限盘点报告（2026-08-23）

- **执行人**：阿奇（Archi）· 系统架构师（upstream-ceiling）
- **日期**：2026-08-23
- **任务**：①核对 b12x 上游有无修复我们遇到的问题/适用的新性能路径 ②盘点仍存在性能释放空间的线性算子节点，按硬件立项性能释放上限，评估性能拐点可优化性
- **纪律**：纯 web + 源码/既有报告分析，不占 GPU、不动生产（LuZ0.3.1 窗口由另一代理执行）
- **口径标注**：**[上游实证]** = 上游仓库/PR/issue 页面直接验证；**[实测]** = 本团队既有实测数据（引用内部报告）；**[推算]** = 基于 Roofline/形状的推算，需落地前二次确认

---

## 0. 一页结论

1. **上游一句话**：b12x 1.2.6（2026-08-20）之后**零新提交**（仅 2 个 commit：df58946 W4A16 resource introspection 移除 + 36bce2c 即 1.2.6 本身）——**我们的 6 项问题没有任何一项在 master 已修**；但 **PR #227（Open，未合并）修复了与我们的 e8m0×micro direct 数值缺陷同族的 native micro 路径越界/int64 收窄 bug（含 e8m0 native micro 的 M=1/2/4/8 测试）**，是最接近的开放修复。FlashInfer 已有 **0.6.17（08-11）**：NVFP4 W4A4 在 GB10 SM120/SM121 上 decode/prefill kernel 对等、两个 NVFP4 量化精度修复、W4A16 小 batch 张量核 decode 路径、**共享专家融合 API**——与我们的性能释放方向高度对症。
2. **最大性能释放空间一块**：**bf16 稠密池（37% 线性 FLOPs，~29-34µs/token of 398µs）→ W4A4/routeB**。其中 **shared experts 是最佳首发节点**（全 token、M=chunk 4096 恰在 routeB 350T 平台、权重静态零拷贝适配器已验证）；**lm_head 是唯一同时作用于 decode 带宽墙的节点**（W4A4 使权重读字节 ÷4，推算为 decode 步时省 ~5-6%）。池整体转化推算 PR +5~7%（3000-3100 → ~3200-3300，理论阶梯 3560-3720 的 1/3~1/2 缺口）。
3. **拐点判定一句话**：三个拐点中，**W4A4 甜点 M≥4096 是"已捕获且应扩大覆盖面"（工程可改）；M_e 768 与 decode 273GB/s 带宽墙是物理边界**（模型几何 topk6/E256 与 LPDDR5x 硬件决定）——但物理边界的**左侧曲线仍可抬高**（FlashInfer 0.6.17 小 batch 路径、lm_head 字节数削减、draft 接受率），这是"优化拐点"的正确姿势：不是移动拐点，而是提高拐点左侧的效率、把更多负载搬进拐点右侧。

---

## 1. 上游核对结果

### 1.1 b12x：1.2.6 之后的状态 [上游实证]

- 仓库 `github.com/local-inference-lab/b12x`（原 lukealonso/b12x，曾更名 sparkinfer，2026-08-06 改回）。
- **1.2.6 = commit 36bce2c（2026-08-20，"fix: support CUTLASS DSL 4.6.2"）之后没有任何新提交**。API 查询 `commits?since=2026-08-18` 仅返回 df58946（08-20 17:01，"fix(moe): remove W4A16 resource introspection"，随 1.2.6 一起发布）与 36bce2c 本身。
- 近期时间线（供背景）：08-16 TP16 size-routed all-reduce qualification；08-06 改名回 b12x；07-27 MX-FP6 (W6A8) 栈（内含 "MoE plan/scratch **deduplicated by geometry across layers**"——即我们 workspace 去重补丁的上游对应物）。
- 依赖现状不变：**torch ≥2.12 + nvidia-cutlass-dsl 4.6.x**（1.2.6 兼容 4.6.2）——与我们 torch 2.11.0+cu130 的冲突依旧，整体升级仍需先解 torch。

### 1.2 我们的问题清单 × 上游对照表

| # | 我们的问题 | 上游状态 | 证据 | 判定 |
|---|-----------|---------|------|------|
| 1 | **e8m0_k32 × micro direct（M≤8）数值缺陷**（bprime-impl 实锤：多几何 100% NaN/全错，is_supported 误放行；生产防护 = 强制 `B12X_W4A16_SMALL_M_DIRECT=0`，见 bprime-window §3） | **master 未修；PR #227（Open, 08-17, voipmonitor）是最接近的开放修复** | #227 "Honor inactive routes in native W4A16 microkernels"：修复 native ModelOpt W4A16 micro direct 路径三类 bug——① vLLM `-1` padding 哨兵直接索引 expert 存储（**越界读 → 错误输出/可能 NaN**）；② int64→int32 **先收窄后校验**（`2**32` 截断映射到 expert 0）；③ FC2-only 编译容量桶 < 运行时 M。测试覆盖 **M=1/2/4/8**，且明确含 `test_w4a16_e8m0_native_micro_ignores_inactive_routes_during_graph_replay`（**e8m0 native micro 路径**）。Pre-merge 检查未过（缩写定义 + docstring 27%），CodeRabbit 暂停，**未合并**；qualification 在 torch 2.13.0 上做。另有一个未回复的 Major review：wide `m>=2` 前瞻预取仍绕过 `_fc2_route` 验证 | **同族未修**。我们的"多几何 100% NaN"与 #227 的越界/收窄族症状吻合但根因未逐字对上（我们的复现是否走 inactive route 路径需对照 bprime-impl B.4 复现脚本确认）[推算]。另有参照系：PR #222 揭示上游存在"can_implement 接受但构建路径错"的 bug 类（MXFP4 E8M0 被 NVF4 op 无条件构建）——与我们 "is_supported 误放行" 同模式，提示 e8m0×micro 可能是 **dispatch 层接受 + micro kernel 构建层不支持** 的同构缺陷 |
| 2 | **native 路径 staging 小 M 性能**（`_stage_b_tile_modelopt_native` 逐 tile 索引 vs packed 扁平 cp_async：M=8 1.61×/M=96 1.74×/M=512 1.60×；导致 b′ hybrid N1 门 FAIL -25%/-34%，bprime-window §6） | **无任何上游动态** | 1.2.6 后零提交；issue/PR 检索无 "native staging"/`_stage_b_tile_modelopt_native` 相关条目；最接近的是 #234（**dense** NVFP4 GEMM 的 16-row skinny-M tile 特性请求，非 MoE grouped）与 #233（fused_quant_a 扩展到 NVFP4 dense，非 staging） | **未修、无人做**。b′ No-Go 维持；窗口数据（midm_E64/E128 曲线）保留为未来上游修复后的直接复验基线 |
| 3 | **workspace 跨层去重**（我们已自做补丁：wsdedup-l3 + 池化，+28GB×43 层 → 释放 22.8 GiB） | **上游已在 07-27 FP6 提交（77d1cfe）内做掉**："MoE plan/scratch deduplicated by geometry across layers"，plan（host 分配）→bind（仅 view，graph 安全）→run 三段式 | 仓库 README/FP6 提交 [上游实证] | **方向一致，无需 backport**。我们的池化补丁与上游思路等价（几何去重/共享池）；将来随整体升级自然获得上游版本 |
| 4 | **FlashInfer 0.6.16 之上还有新版吗** | **有：0.6.17（2026-08-11，stable latest）+ 0.6.18 nightly（至 08-19）** | Releases 页 [上游实证]：0.6.17 核心内容见 §1.3 | **升级候选成立**，wheel 升级成本低（我们 0.6.15→0.6.16 已走过同款路径） |
| 5 | **Eugr/SparkInfer attention 栈**（b12x 化 C12 坍塌 -49% + WO_PROJECTION 高并发不稳） | **无修复动态；且上游有两条新的反向证据** | ① issue #195（Open，无维护者回应，08-14）：fused NSA indexer 在 **48-SM GB10 上结构性落入 4× 慢的精确重扫回退**——DSV4（64 heads/topk 512）在 GB10 上 `ctas_per_group>=56` 不可达，**所有行数/所有上下文长度都被迫走 last-CTA 臂**，crossover 模型是在 188-SM 上拟合后未重拟合直接外推到 48-SM；② issue #182（Open）：FC2 'ultra' tile 自动选择→重固定校验往返失败，**DSV4 on GB10 初始化即崩**（两个开放 PR #39/#146 均未合并）；③ MLA absorbed projections 的 weight-only MXFP8 batched GEMM PR #61/#62 **abandoned**（WO_PROJECTION 方向无人在做）；④ b12x 上游自述 FP6 栈 "SM12.1 expected compatible but **unvalidated**"，主力验证硬件是 188-SM RTX PRO 6000——**GB10 48-SM 是上游的系统性盲区** | **attention b12x 化维持 No-Go**。现实路径转向 FlashInfer 侧（§1.3/§2.4）：vLLM PR #41834 的 stock-deps DSV4 SM12x 路径默认开启 FlashInfer SM120 sparse-MLA decode+prefill |
| 6 | **b12x 快照 0.15.3（05-30）vs 上游 1.2.6（08-20）→ 今天再核对 1.2.6 之后** | **1.2.6 之后零提交**（§1.1） | API [上游实证] | **无新东西可抄**；行动全部集中在开放 PR（#227/#222）与 FlashInfer 0.6.17 |

### 1.3 FlashInfer 0.6.17 与我们相关的增量 [上游实证]

1. **NVFP4 W4A4 kernel parity（decode/prefill）on DGX Spark GB10 + RTX PRO（SM120/SM121）** + **两个 NVFP4 量化精度修复 + `input_global_scale`**——直接作用于我们 B2（full W4A4）形态：phase3b 曾见 W4A4 decode 归一 -6~-9%（w4a4-ext §2.4 两口径并存），0.6.17 的精度修复与小 batch 路径是把它拉回中性/正值的头号候选 [推算]。
2. **W4A16 协作式持久启动 + 小 batch 张量核 decode 路径**——作用于现生产 W4A16 decode 段（C1 M=8/C12 M≈96）。
3. **统一 MoE API：MXFP4 W4A8/W4A16、共享专家融合（shared expert fusion）、SiTU 激活**——**共享专家融合与我们 bf16 池的 shared experts 节点直接相关**（上游把 shared expert 融进 MoE 调用，可能消掉独立 bf16 shared-expert GEMM 调用开销）。
4. Kimi K3 MLA decode（96 q 头/1 KV 头、TP-local 头数低至 6、**推测解码 query 长度至 8**——与我们 DSpark MTP n=7 形状同射程；#4178/#4108 no-rotary-tail 形状支持）+ **fix(gdn): support WY decode on SM121（#4117）**。
5. cute **SM120 groupwise GEMM**（#4130）。
6. 无任何 DeepSeek V4 专项条目——DSV4 的 SM12x 承载在 vLLM PR #41834（§1.4）。

### 1.4 vLLM 侧：DSV4/GB10/SM121 专项 [上游实证]

- **PR #41834（Open，288 commits，活跃维护）**："Add SM12x support for DeepSeek V4 Flash with essential fixes"——stock 依赖（不依赖未发布的 DeepGEMM #324/FlashInfer #3395）即可在 SM120/SM121 服务 DSV4-Flash-0731 + DSpark。与我们相关的内容：**FlashInfer SM120 sparse-MLA decode/prefill 默认开启**（`VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE/PREFILL`，可用性门控 + 优雅回退）、prefix-cache 幽灵块竞态根因修复（prefix caching + 推测解码同开时默认启用 guard）、DSpark 融合 Markov 采样器越界 token id 修复（全 TP rank 非法内存访问）、E8M0 block-scale upcast 移出 FP8 GEMM 热路径（每 25 个 decode 步减少 13,561 次内核启动）、eager scratch pool 默认关闭（并发混合负载 7/7 输出损坏）。依赖 pin：torch 2.13.0/triton 3.7.1/FlashInfer 0.6.16.post3/cutlass-dsl 4.6.0。**该 PR 是我们 fork（0.26.1-dev）中 DSV4 自定义实现的一大批正确性/性能修复的上游对账清单**——尤其 prefix-cache 竞态与 DSpark 采样器越界两项，建议对 fork 做逐项排查 [推算→待源码 diff]。
- #40082（05-20 已合并）引入的 `flashinfer-b12x` dense GEMM 后端（`FlashInferB12xNvFp4LinearKernel`，SM120/121 自动选择）：DGX Spark 全模型 +1.8%(1P)/+6.0%(8P) vs flashinfer-cutlass——注意这是 **NVFP4 权重的 dense linear** 路径，我们 fork 未启用（我们的 dense linear 走 FP8/bf16）。
- 社区参照（Aiden/Eugr 配方，非上游本体）：`VLLM_USE_B12X_MOE=1` 双机 DSV4 配方帖 prefill ~2×；1M 上下文配方。已在 upstream-tracking-2026-08-22 §4.2 覆盖，无新增。

### 1.5 升级路径/backport 评估（按投入产出排序）

| 路径 | 内容 | 工作量 | 风险 | 收益预期 |
|------|------|--------|------|---------|
| U1 | **FlashInfer 0.6.16 → 0.6.17**（wheel，测试容器先行） | 低（1-3 天，复刻 0.6.15→0.6.16 流程） | 低-中（fork `utils/flashinfer.py` API 门槛 + b12x 树回归） | W4A4 decode 中性度定论（E2 行动项）+ W4A16 小 batch decode + 共享专家融合 API 探路 [推算] |
| U2 | **backport PR #227 的路由验证逻辑到我们 b12x 0.15.3 的 micro.py**（防御纵深 + 未来解锁 micro direct M≤8） | 中（源码已在镜像，`micro.py` 模块结构存在；#227 基于 1.2.x 需手工映射；含 wide `m>=2` 预取遗漏需一并处理） | 中（改动 micro 路径需全套 e8m0 数值回归） | 解锁 `B12X_W4A16_SMALL_M_DIRECT=1` 的 M≤8 段——但 native staging 问题（#2）不修，micro direct 收益仍被 staging 盖住 [推算]。**优先级低于 U3-U5，建议作为 issue 素材先行** |
| U3 | **对外 issue 提交素材定稿**（用户未批，只备不交）：①e8m0×micro direct 数值缺陷（bprime-impl B.4 复现脚本 + 多几何 NaN 矩阵）；②`_stage_b_tile_modelopt_native` 小 M staging 曲线（bprime-window §6，含 M=8/96/512/2048/3071 两几何全表）；③（可选）GB10 48-SM attention 盲区补充证据给 #195/#182 | 极低 | 零 | 上游修复后我们免费受益；#195 的 crossover 重拟合是"零 SMEM 代价的最便宜真修法"（评审第 7 节），一个高质量的 GB10 实测跟帖可能推动它 |
| U4 | b12x 0.15.3 → 1.2.6 整体升级 | 高（torch 2.11→2.12+，连锁 Triton/vLLM fork 重编） | 高 | 收益已被我们自做补丁（workspace 去重、W4A4 wrapper、routeB）覆盖大半；**暂不建议**，与 vLLM 整体升级（#41834 路线，torch 2.13）合并考虑 |
| U5 | vLLM fork 对账 #41834 的正确性修复（prefix-cache 竞态、DSpark 采样器越界、eager scratch pool） | 中（逐项 diff + backport 小 PR） | 中 | 正确性保险（并发混合负载输出损坏类风险）；无直接性能收益但属生产稳健性必查项 |

---

## 2. 性能释放空间盘点（核心）

### 2.0 基线口径 [实测]

- 当前地图：PR 398µs/token（4K 档）；LuZ0.3.1 后 PR ≈ 3000-3100 tok/s（W4A4 B2 臂实测 2994/3060/3092，w4a4-ext §2）；理论阶梯 3560-3720。剩余缺口 ≈ 15-18%。
- bf16 稠密池：37% 线性 FLOPs @ 60-100T 效率区，≈ 29-34µs/token。
- AR 12.9% 步时已关闭（21.4GB/s = 本构建上限）；attention 10-21% 步时；decode 带宽墙 C12 已顶 273GB/s。

### 2.1 性能释放空间总表

> 硬件上限口径：routeB DSL dense NVFP4 = 332-368T（MFU 66%，tile 128³，M≥1536 即 350T 平台，M=1024 已 338T）[实测]；节点 FP4 峰值 500T [硬件标称]；HBM 273GB/s/节点 [硬件标称]。

| # | 节点 | 当前利用率/效率 [口径] | 硬件上限 | 释放空间 [口径] | 路径（用哪个已验证资产） | 工作量 | 拐点可优化性 |
|---|------|----------------------|---------|----------------|------------------------|--------|-------------|
| P1 | **shared experts（1×2048，全 token，prefill M=chunk 4096）** | bf16，60-100T 效率区，池内份额推算 ~8-12µs/token [推算：池 29-34µs 按形状比例分摊，**需 profiler 定论**] | routeB 350T 平台（M=4096 恰在甜点；M=1024 已 338T） | 节点级 3.5-5×（60-100T→350T）→ 池级省 ~6-9µs/token → **PR +1.5~2.5%** [推算] | **routeB dense NVFP4 kernel + 权重零拷贝适配器**（E2M1 直配 + E8M0→E4M3 LUT，rel=1.41e-3 已验证）+ W4A4 wrapper（MIN_M 分档：M≥1024 走 routeB，小 M 回退 bf16） | 中（fork shared expert 路径集成 + 校准 + 质量门；适配器/wrapper 均已有） | **甜点拐点已捕获**——本节点是把"未量化负载"搬进已验证甜点区的最干净案例 |
| P2 | **lm_head（vocab ~128K）** | bf16。prefill：M=4096 计算态；**decode：带宽态**——权重读 4096×128K×2B ÷4（TP4）≈ **262MB/rank/步**，@273GB/s ≈ 0.96ms/步，占 C12 步时（~12ms）≈ 8% [推算] | 计算态：routeB 350T；带宽态：W4A4+scales ≈ 70-80MB/rank/步 → **0.26ms/步** | **decode 步时省 ~0.7ms ≈ 5-6%（直接缓解带宽墙）+ prefill 节点级 3.5-5×**；池内份额推算 ~4-8µs/token（prefill 口径）[推算] | routeB kernel（M=4096）+ decode 侧 W4A4 grouped/dense 小 M 路径（或 FI 0.6.17 小 batch 路径）；权重内存 4096×128K：bf16 1.05GB → W4A4 ~0.3GB（TP4 分片后每 rank 省 ~190MB） | 中-高（**质量门槛最高**：logits 直接作用 token 分布，需 KL/困惑度门 + 校准；解码贪心门敏感） | **decode 带宽墙的唯一工程可改侧**：墙不动，每步字节数 ÷4 |
| P3 | **attn 投影（MLA q/kv/o + lora）** | bf16，60-100T；池内最大份额推算 ~12-18µs/token [推算，同 P1 待 profiling] | prefill（M=4096）：routeB 350T；**decode（M=8~96）：低于 routeB 可用域（M<1024）** | 仅 prefill 侧可转化：**PR +2~4%** [推算]；decode 侧维持 bf16（小 M FP4 效率反转，参照 #233/#234：上游 dense NVFP4 小 M 也是开放问题） | **分档量化**：prefill M≥阈值走 routeB（零拷贝适配器），decode 回退 bf16——复用 W4A4 wrapper MIN_M 机制。风险分层：o_proj 最低 → q_proj 中 → kv/lora（影响 attention score 本身）最高，**逐层灰度** | 中-高（形状多：q/kv/o/lora 各异；数值风险分层需逐算子质量门） | 甜点拐点覆盖面扩展，但只覆盖 prefill 半场 |
| P4 | **MoE prefill 残余（grouped W4A4，M_e≈96）** | W4A4 落地后 +8~13% 已捕获；M_e 96 vs B12X 效率拐点 768（几何差 8×） | merged kernel（routeB 派生 fused 管线）332T @ M≥1536 | 残余空间小：**估 +1~3%** [推算]。M_e 提升路径全部关闭：threshold 8192 已证伪（性能无增益 + KV 塌缩 -36.7%，w4a4-ext §2.5）；M_e=768 需 M=32768 不可行 | ①FlashInfer 0.6.17 W4A4 parity 重测（U1）；②上游 #234 skinny-M tile 若落地可跟进；③43% 开销比的 prologue/combine 继续压缩（参照 #233：上游实测激活量化支撑占 decode 墙 20.6%，fused_quant_a 融合上限 ~15% decode——对我们是 W4A4 激活量化融合的远期方向） | 低（U1 重测）/高（自研 grouped 小 M tile） | **M_e 768 拐点 = 物理边界**（模型几何 topk6/E256 决定）；可做的是抬高拐点左侧曲线（小 M kernel 效率），非移动拐点 |
| P5 | **attention（10-21% 步时）** | Eugr b12x 化失败（C12 -49%）；上游 GB10 48-SM 系统性盲区（#195 结构性 4× 回退、#182 初始化崩、SM12.1 unvalidated）[上游实证] | FlashInfer SM120 sparse-MLA（#41834 默认开启路径） | **+1~3% 步时** [推算]：FI 0.6.17 Kimi K3 MLA decode（spec q-len 8 同射程）+ SM121 WY fix #4117 + 0.6.16→0.6.17 升级 | U1（FI 0.6.17）A/B attention 后端；不在 b12x attention 上再投入；跟踪 #195 crossover 重拟合 | 低-中 | b12x attention 路线关闭（上游盲区实证）；FI 路线为唯一现实通道 |
| P6 | **decode 其余（带宽墙）** | C12 已顶 273GB/s [实测] | 物理带宽 | 结构性封顶确认。工程侧仅剩：**P2 lm_head 字节削减**（唯一大项）+ W4A4 decode 中性度定论（U1）+ draft 接受率（训练侧，#41834 参照系 DSpark 接受率均值 2.08-2.19，我们 tokens/step 4.1-4.4 已在其上 [实测]） | — | — | **物理边界**（LPDDR5x） |

### 2.2 bf16 稠密池转化的总量账 [推算]

- 池总量 29-34µs/token。routeB 平台效率 350T vs 当前 60-100T → 节点级 3.5-5×。
- **全池转化**（P1+P2+P3 prefill 侧 + P2 decode 侧）：池时间 29-34µs → ~9-13µs，**PR 省 ~20µs/token ≈ +5~7%（2994 → ~3160-3210 tok/s）**；加 P4/P5 残余与 FI 0.6.17 增量，**现实上限 ~3250-3350**——理论阶梯 3560-3720 的剩余部分由 M_e 拐点（P4 物理边界）与 AR（已关闭）锁定。
- **诚实标注**：池内三节点的 µs 拆分目前是形状比例推算，**立项第一步应是 profiler 拆账**（一次性容器可做，不占生产 GPU）；P1/P2/P3 的优先级排序可能因实测拆账而变，但"池是最大工程可改块"的结论不依赖拆账精度。

### 2.3 数值风险分层（池转化质量门设计）

| 层级 | 节点 | 风险 | 门 |
|------|------|------|-----|
| 低 | shared experts（P1） | 全 token 经过但为稠密静态权重；W4A4 wrapper 在 MoE routed 权重上已有质量门先例（golden 4/4） | golden 4 稳定 prompt + 困惑度 Δ + 接受率不降 |
| 中 | lm_head（P2） | logits 直接决定采样分布；贪心输出最敏感 | **KL(量化‖bf16) 分布门 + 困惑度 + 贪心逐字 + 温度采样抽验**；校准必须 per-channel |
| 中-高 | attn 投影（P3） | q/kv 影响 attention score（softmax 放大）；o_proj 仅线性变换 | o_proj 先行灰度；q/kv/lora 逐层 perplexity + 长上下文 needle 门 |

---

## 3. 性能拐点可优化性判定（正面回答"能否优化性能拐点"）

| 拐点 | 性质 | 判定 | 依据 |
|------|------|------|------|
| **W4A4 甜点 M≥4096** | **工程可改（已捕获）** | 拐点本身无需也不应移动（threshold 8192 已双重证伪：性能无增益 + KV 每token足迹 +58% 塌缩 [实测]）。**正确动作是扩大甜点覆盖面**：P1（shared experts 恰在 M=4096）、P2/P3（把 prefill M=4096 的 dense 算子搬进来），以及把 routeB 可用域下探（M=1024 已 338T [实测]——若 tile/调度再优化把 338T 平台下探到 M=512-768，P3 的 decode 半场和 P4 的 M_e 96 都能受益，**这是唯一值得投入的"移拐点"方向**，属自研 kernel 工作量级） | w4a4-ext §2.5、routeB 实测 |
| **B12X M_e 效率拐点 768** | **物理边界（模型几何）** | M_e = 6M/256：threshold 4096 → M_e 96；到 768 需 M=32768（KV/graph/时延全破）；budget 8192 已证伪。**不可通过调度改变**。可做的是抬高拐点左侧：FI 0.6.17 W4A4 parity（U1 重测）、上游 #234 skinny-M tile、自研 grouped 小 M tile（高投入低确定性） | w4a4-ext §3 判定 5/6、bprime-window |
| **decode 带宽墙 273GB/s** | **物理边界（LPDDR5x 硬件）** | C12 实测顶墙，调度已优。**可改的是每步字节数**：lm_head W4A4（P2，-180MB/rank/步 [推算]）是剩余最大单项；其后只剩 draft 接受率（训练侧）。AR 21.4GB/s 为本构建上限已关闭，b12x PCIe DMA/TP16 AR 均不适用跨节点以太拓扑 [上游实证-不适用] | 实测 + 硬件标称 |

**一句话总判定**：拐点不能被"移动"（除 routeB 平台下探这一自研方向外），但**三个拐点的经济含义不同**——W4A4 甜点是我们的资产（扩大覆盖）、M_e 768 是模型几何锁死的墙（只抬左侧曲线）、带宽墙是硬件墙（只削字节数）。LuZ0.3.1 之后的性能释放叙事应当从"移动拐点"切换为"**填满甜点区 + 削减墙下流量**"。

---

## 4. 优先级路线图

| 优先级 | 项 | 前置 | 预期 | 备注 |
|--------|----|------|------|------|
| **P0** | **profiler 拆账 bf16 池三节点 µs/token**（一次性容器，复刻 bprime mid-M 微基准纪律） | 无 | 把 P1/P2/P3 的推算份额变成实测份额，锁定立项顺序 | 半天级工作量；不占生产 GPU |
| **P0** | **U1：FlashInfer 0.6.16 → 0.6.17**（测试容器 →三门→ 深测） | 无 | W4A4 decode 中性度定论（关闭 E2 两口径悬案）+ W4A16 小 batch decode + 共享专家融合 API 探路 | 低风险，收益面横跨 P1/P4/P5/P6 |
| **P1** | **P1 shared experts → W4A4（routeB + 零拷贝适配器）** | P0 拆账确认份额 | PR +1.5~2.5% [推算] | bf16 池首发节点：资产全齐（kernel/适配器/wrapper MIN_M 机制/质量门先例） |
| **P1** | **P2 lm_head W4A4 立项**（校准 + KL 门先行） | P0 | decode -5~6% 步时 [推算] + prefill 增量 + 显存 -190MB/rank | 质量门最高，先校准后集成 |
| **P2** | **P3 attn 投影 prefill 侧分档量化**（o_proj 灰度先行） | P1 经验迁移 | PR +2~4% [推算] | decode 半场明确不做（小 M 效率反转） |
| **P2** | **U3：上游 issue 素材定稿**（e8m0×micro / staging 曲线 / GB10 盲区） | 用户批准对外提交 | 上游修复后免费受益 | 素材已备，只等授权 |
| **P2** | **U5：vLLM fork 对账 #41834 正确性修复**（prefix-cache 竞态 / DSpark 采样器越界 / eager scratch pool） | 无 | 生产稳健性 | 逐项 diff，小 PR backport |
| **P3** | **P5 attention：FI 0.6.17 SM120 sparse-MLA A/B** | U1 | +1~3% 步时 [推算] | b12x attention 维持 No-Go |
| **P3** | U2：backport #227 路由验证到 0.15.3 micro.py | — | 解锁 micro direct M≤8（被 staging 问题压住，收益有限） | 建议排在 U3 之后，让上游先修 |
| Watch | b12x 1.2.7+（torch≥2.12 路线）、vLLM #41834 合并动态、FI 0.6.18、#234 skinny-M tile、#195 crossover 重拟合 | — | — | 每周核对一次即可（1.2.6 后上游节奏已放缓） |

---

## 5. 引用索引（上游实证）

- b12x 仓库/releases/issues/PR：github.com/local-inference-lab/b12x
  - 1.2.6 = commit 36bce2c（08-20）；post-1.2.6 零提交（api.github.com commits?since=2026-08-18）
  - PR #227 Honor inactive routes in native W4A16 microkernels（Open，08-17）——micro direct 路由验证修复
  - PR #222 enable native MXFP4 in DenseGemmKernel（Open，08-16）——E8M0 dispatch 误构建 bug 类
  - Issue #182 FC2 ultra tile 破坏 DSV4 on GB10（Open，08-13）；Issue #195 NSA indexer GB10 4× 回退（Open，08-14）
  - Issue #233 fused_quant_a → NVFP4 dense（Open，08-19，含 decode 墙 20.6% 量化支撑实测）；Issue #234 16-row skinny-M tile（Open，08-19）；Issue #232 W4A8 dense（Open）
  - FP6 提交 77d1cfe（07-27）"MoE plan/scratch deduplicated by geometry across layers"
- FlashInfer：github.com/flashinfer-ai/flashinfer/releases——v0.6.17（08-11，latest）、0.6.16.post4（08-10）、0.6.18 nightly 至 08-19；#4117 SM121 WY decode、#4138 NVFP4 量化、#4130 SM120 groupwise GEMM、#4178/#4108 no-rotary-tail MLA
- vLLM：PR #40082（已合并 05-20，flashinfer-b12x MoE+FP4 dense GEMM SM120/121）；PR #41834（Open，stock-deps DSV4 SM12x，stable preview tag 20260809，含 DSpark 采样器修复与 prefix-cache 竞态修复）
- 内部实测引用：w4a4-ext-2026-08-23.md（B1/B2/B3 全矩阵）、bprime-window-2026-08-23.md（N1 门 + mid-M staging 曲线）、upstream-tracking-2026-08-22.md（fork 基线 0.26.1-dev @ d3d3b2cca、b12x 0.15.3 @ 7dc6fb8、依赖 pin）

---

## 6. 遗留与诚实声明

1. 池内三节点 µs 拆分（P1/P2/P3 份额）为形状比例推算，P0 拆账前立项顺序有 ±1 位浮动可能；池总量与效率区为团队既有实测口径。
2. PR #227 与我们 e8m0×micro 缺陷的根因同一性未逐字验证（需拿 bprime-impl B.4 复现脚本对照 #227 的三类修复逐条排除/确认）——报告按"同族未修"保守判定。
3. lm_head decode 带宽账（262MB/rank/步）按 bf16 权重 × TP4 均分 + 每步全量读假设推算，未计入 page cache/重叠效应；实际可省字节以 profiler 为准。
4. 所有 [推算] 数字在立项文档中须替换为实测后方可在采信口径下使用。

*本报告由工程保障团队（系统架构师）生成；P1/P2 立项与否请由人类工程负责人结合 P0 拆账结果裁定。*
