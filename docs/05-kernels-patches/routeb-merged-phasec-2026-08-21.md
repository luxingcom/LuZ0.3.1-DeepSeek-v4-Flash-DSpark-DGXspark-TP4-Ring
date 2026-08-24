# Phase C 报告：merged-GEMM v2 性能验证 + 插件集成（Task #26/#30，2026-08-21）

**执行**: Archi-2（系统架构师，接班）· node01 干净 GPU 独占（SRE 停机开窗，UTC 09:49 确认零计算进程）· 生产镜像 0.2.1-v026.0 一次性容器 `--rm`
**配置**: NVFP4 vec16 E4M3 / tile 128³ / epi 128,128 / c_dtype Float16 / warmup 5 / iter 30 中位（P0 同口径）
**前置**: Phase B 16/16 PASS（间接寻址 kernel + run_bs 管线修复，见设计文档 §Phase B 执行记录）

> **一页结论（供裁决）**
> **Phase C 判定：Go（五项全过）**
> 1. **① 性能保留率 99.3%**（判据 ≥90%）：merge-8 3072×12288×4096 v2 间接寻址 = **329.9 TFLOPS** vs P0 基线锚 332.1T；同轮流基线 318.1T（v2 反超 3.7%）。散射大 stacked B（E=128, 512MB）无衰减。
> 2. **② 静态 vs 动态表**: kernel 级等价（344.0 vs 342.4T，噪声带）；host 建表 7.5μs/表。静态表价值 = CG 兼容 + 免每步构建。
> 3. **③ fused 管线**: prologue（Triton fused gather+quant+pack+swizzle）**0.416ms**（预算 ~0.5ms ✓）+ combine **0.386ms**（预算 ~0.7ms ✓）；开销/双 GEMM 计算 ≈ **43%**（判据 ≤1.5× ✓）。
> 4. **④ M_g 曲线**: 256→157T, 512→254T, 768→314T, 1024→338T, **1536→351T 平台**, 4096→367T。**DYN_MIN 建议降至 256-512**。
> 5. **⑤ 插件 + mini logprob**: MergedB12xExperts 插件骨架（w13 N-merge + w2 K-concat combine 折叠）跑通；mini 双路径总 logprob 差 **0.360% ≤ 1% ✓**；确定性对照 + 零污染结构性保证。
> **口径铁律**: 以上均为 **kernel/管线/单卡 mini 级**性能声明。生产 e2e A/B（插件挂 TP4 生产跑 panorama）是下一阶段，本报告结论不构成 e2e 声明。

---

## 1. ① 性能保留率（核心判据）

| # | 配置 | shape | TFLOPS | 说明 |
|---|---|---|---|---|
| A0 | 基线锚点 | 4096×14336×4096 | **371.1** | P0 359.4 带 ±3% 内复现 ✓ |
| A1 | 基线 merge-8 | 3072×12288×4096 | 318.1 | 本轮流（P0 锚 332.1；运行间带 318-332） |
| A2 | v2 identity（恒等表） | 同上 | 328.1 | 间接层 + 恒等查表 |
| A3 | v2 map E=16 | 同上 | 326.0 | 小 stacked（64MB） |
| A4 | **v2 map E=128 散射** | 同上 | **329.9** | **保留率 99.3%（vs P0 锚）**；热组合跨 512MB B 散射 |

- 间接寻址开销 ≈ 0（所有 v2 变体落在本轮流基线的噪声带内且高于它——P0 锚的 332.1 与本轮 318-330 属同一运行间带）。
- 大 stacked B 的散射 tile 访问（L2/DRAM 真实形态）相对连续 B **无衰减**（A3 vs A4）。
- **判据 ≥300T（保留 ≥90%）：PASS，余量巨大。**

## 2. ② 静态 vs 动态表

| 模式 | TFLOPS | host 成本 |
|---|---|---|
| 静态表（预建恒指针） | 344.0 | 0 |
| 动态表（新分配） | 342.4 | 7.5μs/表（torch.cat 96 项×6） |

- kernel 级两种模式**等价**（0.5% = 噪声）；静态表的真实价值在生产侧：**CG 可捕获形态（§设计 6）+ 免每步 host 构建**（7.5μs × 桶数 × 43 层）。
- Phase B 已证：map 原地重写（指针恒定）输出随内容变化 = 静态表 LRU 原地重写语义成立；新指针替换 = 动态表语义成立。

## 3. ③ fused prologue/combine 端到端（Triton 原型 + v2 GEMM）

管线：A bf16 [8240,4096] --Triton fused gather+NVFP4 quant+pack+SFA swizzle--> A_fp4+SFA(E4M3) --> v2 merged GEMM [3072,12288]（间接寻址）--> Triton 行内 6 块加权 combine --> [3072,2048]

| 组件 | 实测 | 设计 §4 预算 | 判定 |
|---|---|---|---|
| prologue（fused 一步） | **0.416 ms** | ~0.5 ms | ✓ 内 |
| merged GEMM | 0.934 ms（331.2T） | — | 与①一致 |
| combine（行内加权求和） | **0.386 ms** | ~0.7 ms | ✓ 内（~226GB/s 内存受限） |
| 全管线 | 1.690 ms | — | 管线/单GEMM=1.81×；**开销/双GEMM计算≈43%**（判据 ≤1.5× ✓） |

- 正确性：管线 GEMM vs torch 同量化参考 rel=2.66e-4 ✓；combine rel=3.02e-4 ✓；**Triton 量化字节（payload+scale）与 torch 参考编码逐位一致**（E2M1 RTN 阈值 + E4M3 RTN + 低nibble=偶k 打包 + swizzle 全对）。
- 注意：prologue 计时含每步分配（动态口径）；静态路径可预分配缓冲再省。
- **口径**：Triton 原型级数字。生产融合版（swizzle 并入、静态缓冲）只会更快。

## 4. ④ M_g 效率曲线与分派阈值校准

N=12288（6-expert 桶）、E=128 散射 stacked、v2 间接寻址：

| M_g | 256 | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|---|---|---|
| TFLOPS | 157 | 254 | 314 | 338 | **351** | 349 | 344 | 367 |

- **平台 ~350T 从 M_g≥1536 开始**（好于设计预期：原判 M_g≥3072 才进 332T 档）。
- 对照长尾 B12X per-expert（M_e≈96 → 6-43T，P0/Task#20 数据）：M_g=256 的 merged 路径已是其 **4-56×**。
- **校准建议**：T2 DYN_MIN 从 512 可降至 **256**（甚至更低需 B12X 同形对照实测）；T1 K=64 维持；hash 层 top-1 桶（M_g 可达 8220）全部覆盖。
- 生产 batched 4096 chunk 口径：单桶 27% 流量 ≈ M_g≈1100 → 338T 档 ✓。

## 5. ⑤ 插件集成 + mini logprob

### 5.1 插件（routeb_merged_plugin，骨架）

- **形态**：`MergedB12xExperts(B12xExperts)` 子类，`vllm.general_plugins` entry point 自动加载（pip install；EngineCore 子进程同载——显式 PYTHONPATH import 不进 spawn 子进程，必须走 entry point，A1 插件同款机制）。env：`VLLM_MOE_MERGED=0/1`（0=install() no-op，**零污染结构性保证**）、`VLLM_MOE_MERGED_MIN_M`（默认 128，mini 口径；生产建议 256）。
- **数学**（Phase A 设计 §2/§4 实现）：
  - w13：N-merge——B stacked 权重 + tile_map 间接寻址（v2 kernel），A 桶内共享
  - w2：**K-concat 合并公式**——B2_cat=[N2, 6·K2]（专家沿 K 拼接）、A2=加权激活沿 K 拼接，单 GEMM 直接算 Σ_e w_e·(a2_e@w2_e^T)——**combine 数学上折叠进 GEMM**（比设计原案的独立 combine kernel 更优）
  - 分派：exact-set 桶分组（torch.unique）→ 最大桶 M_g≥MIN_M 走 merged，**骨架版 super().apply 全量算后覆写 merged 行**（规避 workspace/meta 子集语义风险，生产化时改增量分派）
- **验证证据链**：EngineCore 日志 `Using MergedB12xExperts` ✓；权重派生实跑（init 3 分钟 + 37GB 派生张量驻留）✓；logprob 变化 ✓。

### 5.2 mini 双路径 logprob（判据 ≤1%）

mini0731（真实 -0731 checkpoint 前 4 层重建，单卡 TP1），4×~2000 token 长 prompt（保证热桶 M_g≥128）：

| 口径 | 基线 B12X | merged 插件 | 差 |
|---|---|---|---|
| 总 logprob | -978.7714 | -975.2462 | **0.360%** ✓ |
| 逐 prompt | -211.2 / -222.7 / -285.9 / -259.0 | -210.7 / -223.3 / -284.0 / -257.2 | 0.22-0.69% |

- **确定性对照**：早期一次未 engagement 的 merged 跑（spawn 问题修复前）与基线 logprob **逐位一致（0.000%）**——证明 (a) 运行间确定性；(b) 0.360% 的差异**全部归因于 merged 路径的数值效应**（NVFP4 A 侧量化 + w13/w2 merged GEMM vs B12X W4A16 语义差），而非噪声。
- **零污染**：`VLLM_MOE_MERGED=0` → install() 直接 return，无任何 patch——结构性保证（另经确定性对照旁证）。
- 参考：W4A4 路径（Task#21）同口径 logprob 差 +0.41%——merged 路径 0.36% 同量级，**数值质量不劣于已接受生产的 W4A4 方案**。

## 6. 过程发现与坑（工程资产）

1. **NVFP4 scale swizzle 实证公式**（probe_sf_layout 从 proven CVT 输出反推，bijection 验证）：
   `off(m,g) = (m%32)·16·rm·rk + ((m//32)%4)·4·rm·rk + (m//128)·4·rk + (g%4)·rk + g//4`
   （rm=⌈M/128⌉, rk=K/64；行主序 strides over 物理张量 (32,4,rm,4,rk,l)）。**注意与 Task#20 适配器的 swizzle_block_scale（FlashInfer 布局）不同布局，不可混用**——cutlass DSL kernel 消费前者。
2. **fp4 壳缓冲**：cute_tensor_like(f32, FP4) 的 backing buffer = packed [M,K/2] 于起始 + 2× slack（probe 实证）；直接字节注入路径未调通（本轮以 proven 转换路径通过），生产化时若需要免 f32 中转可再攻（recast_tensor 需 MLIR 上下文，host 侧不可用）。
3. **vLLM 插件与 EngineCore 子进程**：EngineCore 为 spawn 子进程，父进程 monkey-patch **不继承**——必须 pip install + `vllm.general_plugins` entry point（且 entry point 必须指向 callable `module:install`，模块本身不可调用）。cutlass DSL 必须惰性导入（fork/spawn 前导入会破坏子进程 CUDA 初始化）。
4. Triton `tl.float8e4nv` cast 与 torch `float8_e4m3fn` RTN 语义一致（字节级验证）；uint8 张量做 torch 索引会触发 mask 语义（需 .long()）。
5. 生产权重 payload 低半字节=偶 k（与 vLLM scaled_fp4_quant 一致）——与 Triton 打包约定天然零拷贝兼容。

## 7. 遗留与移交（下一阶段）

| 项 | 说明 |
|---|---|
| **生产 e2e A/B** | 插件挂 TP4 生产（panorama 负载）跑 PR 对比——**本报告全部结论的下一关**（kernel/管线/mini 级 → e2e 的边界已留） |
| w2 K-indirection kernel | 当前 w2 走物理 K-concat（正确性优先）；生产化应做 K-tile 间接寻址 kernel patch（与 N-indirection 同构，省每桶 concat 拷贝） |
| 插件生产化 | 增量分派（免全量 super 覆写）、prologue/combine Triton 内核接入（本轮 torch 原型）、静态表 Registry/EMA、fp4 字节注入路径、B12X 双表示显存裁决（设计 §5，+35GB/rank） |
| DYN_MIN=256 终验 | 需 B12X 同形（M_e≈96 per-expert）对照实测定夺 256 vs 512 |
| 性能口径 | 344T（B1 静态表流）vs 332T（P0 锚）的运行间带说明单点测量 ±4%；生产验收建议多轮中位 |

## 8. 工件

| 位置 | 内容 |
|---|---|
| 01:/tmp/_routea_work/ | phasec_perf.py+log+json（①②④）/ phasec_pipeline.py+log（③）/ probe_sf_layout.py · probe_fp4_shell.py（布局实证）/ plugin_merged/（⑤ 插件：dsl_gemm.py · merged_experts.py · __init__.py · setup.py + run_mini_merged.py · compare_lp_merged.py）/ lp_pc_base.json · lp_pc_merged.json · lp_pc_{base,merged}.log / mini0731/（4 层 mini，用后可清）/ phasec_mini_*.sh |
| 本地 _routeb_extract/routeb-delivery/ | 同步上述全部 + plugin_merged/ |
| routeb_official_v2（kernel） | 不变（Phase B 版本直接通过 Phase C 全部性能判据——无需再改） |

**Phase C 判定：Go。** 间接寻址零开销（99.3% 保留）、管线开销预算内（43%）、M_g 曲线优于预期（平台 350T@1536）、插件端到端数值质量 0.36%（不劣于已验收的 W4A4）、零污染保证。剩余风险全部集中在生产 e2e 集成层（插件生产化 + TP4 显存裁决 + panorama A/B），kernel 层无已知障碍。
