# kernel② kv_linear 优化空间分析

> 日期：2026-08-20 | 现状：v11 为生产带宽冠军（验证团队实测 53.4，v12.1=18.8，v15=10.6，相对 5× 回退）
> 已知教训：① grid 微型 block（131072 个）是性能杀手；② "宽 tile"必须"减 block 增负载"；
> ③ 带宽算子验收以 **GB/s 实测**为准（MCP speedup 对标慢 torch 参考无意义）

---

## 〇、性能画像（确认瓶颈性质）

每 token 访存：**读 4KB（kv [T,1024] fp32）+ 写 584B（信封）≈ 4.68KB**
GB10 HBM 273 GB/s → 理论极限 T=65536: 306.7MB / 273GB/s ≈ **1.12ms**

| 版本 | 实测（验证团队口径） | 相对 v11 | HBM 利用率 | 瓶颈 |
|---|---|---|---|---|
| v11 | 53.4 | 1.00× | ~19.6% | 独立 pad kernel 巨 grid + 窄向量化 |
| v12.1 | 18.8 | 0.35× | ~6.9% | 16 元素/block 微型 block + 标量串行 |
| v15 | 10.6 | 0.20× | ~3.9% | **131072 微型 block**（grid=(T, 64/BLOCK_G)） |
| 理论 | 273 | 5.1× | 100% | — |

**判定：v11/v12.1/v15 全部未打满 HBM，优化空间 = 5×（到理论极限）以上。**

---

## 一、结构化优化（最大收益：减 block 增负载）★ 首要

| # | 手段 | 说明 | 预期收益 |
|---|---|---|---|
| S1 | **TOKENS_PER_PROG（多 token/program）** | 每个 program 处理 4~16 个 token 的连续段，grid 从 (T, ·) 降为 (T/TPP, ·) | block 数 131072→8192~16384（-12~16×）→ **大 T 带宽 2~4×** |
| S2 | **1D grid 合并** | (T, 64/BLOCK_G) 2D → 1D 扁平（T×段数），减少调度维度开销 | 辅助 |
| S3 | **pad 内联** | 信封 [576:584] pad 与 data/scale 同一 store 段（当前零填充独立处理） | 消除额外 pass |

## 二、访存优化（打满 HBM）

| # | 手段 | 说明 | 预期收益 |
|---|---|---|---|
| M1 | **128-bit 向量化** | load 用 1D 连续 arange + `tl.max_contiguous`/`multiple_of` 提示 → 16B 向量化；store 对齐 16B（584B 分 512+64+8 三段，或整行 584 按 8B 对齐） | 访存粒度 64B→512B+，**配合 S1 是关键** |
| M2 | **读侧大 tile** | kv [T,1024] 连续 → 每 program 读 TPP×1024×4B 连续块（16KB+），一次 load | 减少 load 指令数 |
| M3 | **写侧合并** | data[0:512] 一次 store（512B 连续）+ scale[512:576] 一次（64B） | 写 584B 分 2 段而非 3 段 |
| M4 | eviction 分级 | 输入 `evict_first`（一次性）、输出 `evict_last`（留 L2） | 已有，保持 |
| M5 | **int64→int32 地址** | T×584 信封索引用 int64 必要时才转；packed 偏移纯 int32 | 减 ALU 压力 |

## 三、计算侧优化（量化开销）

| # | 手段 | 说明 |
|---|---|---|
| C1 | scale 批量 | [BLOCK_G,16] 一次 max/log2/exp2（已有，保持） |
| C2 | 阈值链替代 | 7 次比较+求和 → 可尝试 `tl.floor(a_abs*2)/2` 型位技巧（**精度需逐字节验证**，风险项） |
| C3 | 打包并行 | nibble→packed 用 `tl.split`/移位（已有），BLOCK_G 大时打包全向量化 |

## 四、架构级（对齐社区最佳）

| # | 手段 | 说明 | 适配性 |
|---|---|---|---|
| A1 | **TMA 批量拷贝**（danielwoz） | flashinfer TMA-packed IO（smem 双缓冲） | Triton 写回场景 TMA 收益有限（小写），**暂不采用** |
| A2 | 独立量化 kernel + store kernel | 量化（compute-bound 段）与 store（memory-bound 段）分离，各自最优调度 | 可选，多一次 kernel launch，T 大时划算 |
| A3 | **CUDA Graph 兼容** | 确认无主机同步/动态形状（TPP 固定 constexpr）→ vLLM graph capture 友好 | 生产必需 |

## 五、场景级（整体软件运行预期）

| # | 手段 | 说明 |
|---|---|---|
| E1 | **小 T 与大 T 双配置** | decode 每步 T=6~8：launch 开销主导 → 用"每 program 整 token"（block≈T）；prefill 大 T：多 token/program——autotune key=['T'] 已支持，需保证小 T 配置存在 |
| E2 | **paged 变体同步优化** | 生产 paged v11 5/5；优化架构应同时落地 linear+paged 两版 |
| E3 | vLLM 集成 | 与 attention reader 同一 stream；`kv_scale_format="nvfp4"` 语义对齐（danielwoz 02-python 补丁） |

---

## 六、量化预期（GB/s 主指标）

| 方案 | 预期带宽 | 相对 v11 | HBM 利用率 |
|---|---|---|---|
| v11（现状） | 53.4 | 1.00× | ~19.6% |
| S1+M1（多 token + 128-bit 向量化） | **120~180** | **2.3~3.4×** | 44~66% |
| S1+M1+M2+A2（+大 tile + 分离 store） | **180~230** | **3.4~4.3×** | 66~84% |
| 理论极限 | 273 | 5.1× | 100% |

**建议目标：S1+M1 组合（v17 重写）→ 150+ GB/s，即 v11 的 3× 以上。**

---

## 七、实施路径建议

1. **首选：本地手写 v17**（基于 v11 正确语义 + S1 多 token + M1 128-bit 向量化 + S3 pad 内联）→ 与 v11 做 GB/s A/B（probe_k2_bw.py）——本地可控、避免生成器微型 block 复发
2. 备选：MCP autotune（约束必须写明"TOKENS_PER_PROG ≥4、grid ≤ T×16、128-bit 向量化、禁止 2D 微型 grid"）
3. 正确性：7 组 T 逐字节（atol=0）+ paged 5 组；性能：修正口径 benchmark（GB/s）
4. 达标后：linear + paged 双版同步落地 vLLM 集成

> 风险提示：C2 位技巧、A2 分离 kernel 属可选增强，若逐字节失配即回退（v11 语义为金标准）。
