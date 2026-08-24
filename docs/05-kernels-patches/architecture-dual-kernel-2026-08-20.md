# 双算子架构报告：kernel1 routeA→routeB 替换路径 + kernel2 生产集成验证

**日期**：2026-08-20
**作者**：Archi（系统架构师）
**对象**：DGX Spark 4 节点 TP4 vLLM 集群 / DeepSeek V4 Flash / MiaAI vL0.26.1 / B12X MoE backend + nvfp4_ds_mla KV
**前置输入**：`routeb-deploy-precheck-2026-08-20.md`（precheck 权威清单，本文直接引用 A1–A21/R1–R20 编号）；routeB 交付包 7 文件；kernel2 交付包 14 文件（本人已逐行核读 README、安全可靠性报告、paged/linear v17 Triton 源码、bench 脚本关键段）
**当前窗口状态**：生产容器停止、GPU 空闲、4 节点已恢复、所有操作 SEV4

---

## 📌 TL;DR

| 决策 | 结论 |
|------|------|
| kernel1 文件层 | **routeA 原位冻结不动**（零扰动回退锚点），routeB 以 `kernel1/routeB/` 新子树平行落位，MANIFEST md5 登记 |
| kernel1 集成层 | **推荐 Option(a) 独立 kernel 插件**（M≥256 prefill 分派 routeB，decode/MoE grouped 回落 B12X），Option(b) flashinfer-b12x 仅作备选 |
| kernel1 修正工期 | **4.5–6 天（中值 5.5 天）**，host launch 移植（A1）仍是根阻塞，占 P2 主导 1–1.5 天 |
| kernel2 paged | **BLOCK_SIZE=256 硬编码（wrapper:158）与生产疑似 block=64 不匹配 → paged 变体当前禁部署**；SRE 确认前挂起，确认后参数化 + 复验（合计 ~1 天） |
| kernel2 语义漂移 | paged v11 与 linear v17 存在**真实但低危**的 clamp/floor 漂移（详见 §3.2）——建议统一至 v11 金标准语义并文档化，2 人时 |
| kernel2 linear v17 | 维持"持久化 + 可调用"裁决不变（零调用点、零生产行为变化），其生产价值 = 数值金标准 + PR#46329 reader 写端验证器；**真正的生产写路径是 paged 变体**，linear 不是 |
| 双轨排期 | kernel1 为关键路径（~5.5 天），kernel2 各项均不占关键路径；总窗口建议 **6–7 天**，不足则 kernel1 P4 灰度顺延下一窗口（routeA 现役零风险） |
| 最关键 3 风险 | ① paged 块大小错配（若 block=64 且误部署 → KV cache 错位写坏，正确性灾难非性能问题）② 356 TFLOPS 单源不可复现 ③ host launch 移植工期溢出 |

---

## 一、ADR-1：kernel1 routeA→routeB 替换路径（文件层 + 集成层 + 验证链 + 回滚）

**状态**：Proposed（建议工程负责人裁定后转 Accepted）
**日期**：2026-08-20

### 1.1 背景

- 生产现状：`<INSTALL_DIR>/nvfp4/` 挂载 vLLM 容器（ro + PYTHONPATH），两算子"可 import 零调用点"接入；用户已裁决采用"持久化+可调用(安全)"模式。
- routeA（`cutlass_scaled_fp4_mm`，80–187 TFLOPS，克隆实测 rel=0.00141 全过）现役但**无生产调用点**——这大幅降低了替换风险等级：替换的是"潜在路径"而非"在跑路径"，回滚锚点天然存在。
- routeB = CUTLASS 4.4.0 Python DSL（MXF4 E2M1×E2M1 + UE8M0，sf_vec_size=32），社区单源实证 356 TFLOPS，交付包存在 3 项 P2 硬阻塞（B1 host launch NotImplementedError / B2 SMEM 误计 acc / B3 MmaMXF4Op FP8 dtype，本人已亲核 bench:188/60-66/126-131 属实）+ 19 项代码发现（A1–A21 行动项）。

### 1.2 文件层设计（ADR-1a）

**决策：routeA 原位冻结、routeB 新子树平行落位、开关在插件层不在文件层。**

```
<INSTALL_DIR>/nvfp4/
├── kernel1/
│   ├── nvfp4_4w4a_mmaf.py            # ← routeA：原路径原文件，冻结不动（回退锚点）
│   ├── routeB/                        # ← 新子树（修复验证全绿后才落位）
│   │   ├── nvfp4_4w4a_routeb.py      #    生产入口：4 参 (A, W_packed, W_scale, bias)
│   │   ├── routeb_kernel.py          #    CUTLASS DSL kernel 构造 + host launch（A1 移植产物）
│   │   ├── routeb_bench_blockscaled.py  # 修复后基准（仅供验证，不进生产 PYTHONPATH）
│   │   ├── patch_cutlass_dsl_sm121a.py  # 修复后（A2/A12/A14）
│   │   └── setup_routeb_env.sh          # 修复后（A10/A11/A16/A21）
│   └── v17_ref/
│       └── nvfp4_4w4a_prefill_gemm_v17_triton.py  # 数值真值基 + A 量化复用（A9 修复后）
├── kernel2/                            # 按交付包版本子目录化（v17/v11/paged），见 §3
├── plugin-src/nvfp4-vllm-plugin/       # 已落地默认禁用；routeB 达标后注册 entrypoint
└── MANIFEST.md                         # 全文件 md5 + 版本 + 用户裁决记录 + 落位日期
```

**设计理由与权衡**：

| 决策点 | 选择 | 理由 | 权衡 |
|--------|------|------|------|
| routeA 是否移入子目录 | **否，原位冻结** | 零扰动原则：routeA 是三级回滚的最终锚点，任何路径/import 变更都引入新风险面；生产"零调用点"意味着不动它没有任何代价 | 目录不够"整洁"——可读性让位于回滚确定性 |
| routeB 命名 | `nvfp4_4w4a_routeb.py`，**暴露与 routeA 完全一致的 4 参签名** | 插件层切换 = 改一个符号引用，回滚 = 改回来；接口契约稳定使 A/B 对照 harness 可复用 | 无（接口对齐是纯收益） |
| 基准/patch 脚本是否进生产挂载 | 落位但**明确不注册进生产调用路径** | 停机窗口内要在生产机上跑 P2/P3，文件在场便于复现；但以 MANIFEST 标注"工具件，非生产件" | 需纪律约束（MANIFEST + code review 双保险） |
| 版本管理 | MANIFEST.md（md5 清单）+ 落位 git tag（若 nvfp4/ 纳入 git） | md5 已是本团队既有实践（v17 md5 a795b2b4 比对先例）；git tag 提供时间轴 | ro 挂载下生产容器无法写入，落位操作须在宿主机进行 |
| routeB 验证未全绿时 | **不落位生产目录**，仅在验证机（单节点）工作目录迭代 | 防止"半成品进生产挂载"——P4 Go 之前 routeB 对生产必须是不可见状态 | 单节点验证→全节点落位有一次同步拷贝动作（纳入 P4 清单） |

### 1.3 集成层：Option(a) vs Option(b)（ADR-1b）

| 维度 | Option(a) 独立 kernel 插件（M≥256 分派） | Option(b) flashinfer-b12x 四层配方 |
|------|------|------|
| 复杂度 | **Low-Med**：只写插件层分派逻辑（quant_config `_nvfp4_prefill` 骨架已落地） | **Med-High**：需 patch flashinfer `dense_blockscaled_gemm_sm120.py`（第三层依赖）+ 2 个 env var |
| 回滚 | **L1 删插件引用即回退**（0 代价，不动 vLLM 本体） | env 回退（快），但 flashinfer patch 状态需另行追踪 |
| 调用点侵入 | 符合既有裁决"持久化+可调用"：routeB 先以可调用形态入插件，entrypoint 注册是最后一步 | 换 backend 本质上是改 vLLM 行为开关，侵入度高于插件形态 |
| 依赖面 | 仅 CUTLASS DSL 4.4.2 锁版本 | CUTLASS DSL + flashinfer 版本双重耦合（flashinfer 发布节奏不可控） |
| 分派粒度 | **显式 M 阈值分派**：prefill（M≥256）→routeB，decode→B12X；MoE grouped 可精确排除 | backend 级替换，粒度粗，MoE grouped 路径行为需额外验证 |
| 团队熟悉度 | 高（plugin-src 已落地、骨架已写） | 中（社区配方 ai-muninn，团队无一手经验） |
| 性能上限 | dense GEMM 直连，无中间层 | flashinfer 封装可能有额外开销/优化，未知 |

**决策：Option(a)。** 与 precheck 结论一致，本文补充两点架构理由：

1. **Option(a) 与"用户裁决模式"同构**——整个替换推进路线是"可 import 零调用点 → 修复验证 → 注册 entrypoint 灰度"，Option(a) 的插件形态让每一步都是增量开关；Option(b) 是一次性 backend 切换，无法分步灰度。
2. **范围界定**（本报告新增）：routeB 交付聚焦 **dense GEMM**（4096×14336×4096 基线）；MoE grouped（8/64 experts）社区仅 120–154 TFLOPS 且交付包未覆盖。DeepSeek V4 Flash 是 MoE 模型——**routeB 集成范围必须显式限定为 dense prefill GEMM（M≥256），MoE grouped 与 decode 全部维持 B12X 原路径**。Option(a) 的 M 阈值分派天然实现这一隔离；Option(b) 的 backend 级替换则需额外论证 grouped 路径不受影响。

**Option(a) 分派契约（建议写入插件）**：

```python
def nvfp4_prefill_gemm(A, W_packed, W_scale, bias=None):
    if A.shape[0] >= 256 and _routeb_ok():      # dense prefill
        return routeB(A, W_packed, W_scale, bias)
    return b12x_original_path(A, W_packed, W_scale, bias)  # decode / MoE grouped / fallback
```

`_routeb_ok()` 含运行时降级开关（env `VLLM_NVFP4_K1_ROUTEB=0` 强制回退），为 L1 回滚提供不重启容器的快速通道（若 env 热读可行；不可热读则回退仍是删插件引用 + 重启）。

### 1.4 验证链与修正工期（ADR-1c）

在 precheck 修正（4.5–5.5 天）基础上，按已知阻塞逐段重估：

| 阶段 | 内容 | 前置修复 | 工期 | Go 判据（No-Go 即回 routeA） |
|------|------|---------|------|------|
| P0 环境 | 4 节点 DSL 安装 + import + mma.py 备份 | A10/A11/A16/A21（setup 加固，~1h，先行修） | **0.5 天** | import cutlass 4.4.x 成功 + 备份 md5 登记 |
| P1 patch | 4 节点 sm_121a patch + 官方示例验证 | A2（SyntaxError）/A12（幂等替换）/A14（原子写），~2h | **0.5 天** | admissible_archs 含 sm_121a + 示例不报错 + --revert 演练通过 |
| P2 基准 | 单节点 356 复现 + tile sweep + SASS 门禁 | **A1 host launch 移植（1–1.5 天，根阻塞）** + A3/A5/A6/A7/A8/A13/A15/A20（量化修复包 ~4h） | **1.5–2 天** | ≥350 TFLOPS + SASS `mma.*e2m1` 命中 + rel_err<1e-2 + SMEM≤99KB |
| P3 语义 | 生产权重直配 + A 量化复用 + MXF4 N 向 | A9（v17 grid）/A17（sm121 复测）+ **A18 MXF4 N 向实测（0.5–1 天，本阶段最大不确定项）** | **1–1.5 天** | pytest 8/8 + W 直喂正确 + sf_vec=32 分组匹配 |
| P4 集成 | 4 节点 A/B + entrypoint 注册 + needle + 回滚演练 | 插件分派逻辑 + 4 节点 routeB 落位同步 | **1–1.5 天** | ≥1.5× routeA + needle 128K ≥95% + 4 节点一致 + L1 演练 <10min |
| **合计** | | | **4.5–6 天（中值 5.5）** | |

**与 precheck 4.5–5.5 的差异说明**：上限从 5.5 放宽到 6 天，原因有二——① P3 的 MXF4 N 向分组不匹配（R13）一旦坐实，需切 NVF4 16 分组 + 高精度转换器，该分支单独 0.5–1 天；② P4 entrypoint 注册是从"可调用"升级到"真调用点"的关键一跳（此前从未做过），首次注册可能暴露插件层未知问题。建议停机窗口按 **6–7 天**申报（含 20% buffer）；窗口不足时 kernel1 P4 灰度顺延下一窗口，routeA 现役期间零风险。

**A1（host launch 移植）执行建议**：以官方 `dense_blockscaled_gemm_persistent_pingpong.py` 为唯一母本整段移植（TMA descriptor + grid launch ~150 行），不做创造性改写；移植后先跑 `--check`（此时应已修复 A3/A13/A15，否则自洽性校验失效），再上 SASS 门禁。**注意 A3 修复时同步修正 encode 的 clamp 顺序**（当前 `clamp(0,255)` 作用在指数上且缺 +127，修复应为 `floor(log2(max/6))+127` 后再 clamp [0,255]，对照 v17:51 写法）。

### 1.5 回滚设计（ADR-1d）

三级回滚沿用 precheck 方案，本文按文件层设计补强：

| 级别 | 动作 | 文件层支撑 | 触发条件 |
|------|------|-----------|---------|
| **L0（新增）** | env 开关 `VLLM_NVFP4_K1_ROUTEB=0`（若实现热降级） | 分派契约内建 | 生产监控任何指标异常趋势（早于阈值） |
| L1 | 删插件 `_nvfp4_prefill` 引用 | routeA 原位冻结 = 回退目标**从未离开** | P4 任一指标未达 / 生产 prefill P99 >1.5× routeA |
| L2 | `patch --revert`（恢复 .bak） | A11 保证 .bak 不被覆盖、A14 保证 patch 不会写坏 mma.py、A19 保证 revert 失败不静默 | P2/P3 数值错误或 crash |
| L3 | `pip uninstall nvidia-cutlass-dsl-libs-cu13` | MANIFEST 记录安装清单 | DSL 污染依赖 |

**关键架构性质**：由于 routeA 文件原位冻结 + routeB 在 P4 Go 之前不进生产挂载，**整条验证链期间生产可用状态与今天完全一致**——回滚不是"恢复"，而是"保持未变"。这是零调用点现状送的最大红利，排期上应充分利用（验证可以大胆做，不需要为"部分完成状态"设计中态）。

---

## 二、ADR-2：kernel2 生产集成验证点

**状态**：Proposed
**日期**：2026-08-20

### 2.1 paged 块大小不匹配的架构影响（K 系列问题 K1）

**事实（本人亲核源码）**：`nvfp4_ds_mla_kv_linear_paged_triton.py` wrapper 第 158 行 `BLOCK_SIZE = 256` **硬编码**；kernel 以 `tl.constexpr` 接收该值并计算 `block_idx = position // BLOCK_SIZE`、`slot = position % BLOCK_SIZE`。寻址其余部分（`stride_cache_b/s`）已用 stride 泛化，唯独块大小写死。

**若生产 block=64（SRE 验证中）**：

- `position // 256` 与 `position % 256` 全部错位 → **写入错误 cache line**——这不是性能退化，是**静默正确性灾难**：KV cache 内容错乱，decode 读回错误上下文，且无 crash 无告警。
- **架构结论：SRE 确认 block size 之前，paged 变体禁部署（连"可调用"形态都不建议挂进生产 PYTHONPATH）**。README 宣称的"完全对齐 vLLM 分页语义"仅在 block=256 时成立；"paged 5/5 逐字节"测试同样是在 BLOCK_SIZE=256 假设下跑的——**5/5 通过不能外推到 block=64**。
- **修复路径（一旦确认 block=64）**：将 wrapper 改为 `BLOCK_SIZE = kv_cache.shape[1]`（参数化 + `assert kv_cache.shape[1] in (64, 256)` 防呆），改动量 <10 行；但**必须复跑 5/5 逐字节测试（按生产 block 值重参数化）+ 与 vLLM csrc `reshape_and_cache_nvfp4` 的写点对拍**。估 0.5 天（含复验）。由于 kernel 本体寻址已 stride 泛化，参数化后 block=64 大概率直接可用，但 grid/autotune 在小 block 下的负载分布变化需 bench 确认。
- **若 SRE 确认 block=256**：则硬编码"恰好正确"，但仍建议参数化（消除脆弱假设），降级为非阻塞加固。

**连带问题 K3（本人新增）**：paged kernel 的 block_table 寻址 `block_table_ptr + seq_id * max_blocks + block_idx` 假设 block_table 为 `[num_seqs, max_blocks]` row-major int 张量。**需与 vLLM 实际 block_table 布局/dtype（int32 vs int64）核对**——dtype 不符时 `.to(tl.int64)` 掩盖问题但 stride 语义可能错。建议列入 SRE 现场验证清单（与 block size 一并确认）。1 人时。

### 2.2 paged 与 linear v17 语义漂移评估（K2）

**事实（本人逐行比对两 kernel 源码）**，paged v11 与 linear v17 在三处量化语义存在真实差异：

| 语义点 | paged v11（paged_triton:90-101） | linear v17（v17_triton:78-83） | 漂移后果 |
|--------|------|------|---------|
| amax 下限 | `maximum(max_abs, 1e-38)` | `maximum(amax, 1e-30)` | amax∈(0, ~1e-30) 或 =0 时，safe floor 不同 → scale 指数不同 |
| 指数 clamp | `[-127, 128]` | `[-126, 127]` | 下界差 1（byte 0 vs byte 1）；上界 128 需 amax > 6·2¹²⁷≈3.6e38（超出 fp32 范围，实际不可达）——**上界漂移理论存在、实践不可达** |
| 舍入 tie | `abs_x <= 0.25 → 0`（tie 取低档） | `abs2 > 0.25 → 1`（等价：tie 取低档） | **无漂移**（两种写法数学等价，均与 torch argmin 一致） |

**对 reader 的影响评估**：

1. **零/极小值组**（amax=0 或 <1e-30）：两变体产出**不同的 scale 字节**（paged 路径 clamp 到 -127 → byte 0；v17 路径由 1e-30 floor 决定 → byte ~24 或 clamp -126 → byte 1）。但此时 packed data 的 nibble 均为 0，**反量化值 = 0 × 2^x = 0，两路径数值结果仍一致**——漂移只存在于 scale 元数据字节，不影响 reader 数值读出。**前提**：reader 对 scale=byte 0（2^-127）无非法值假设——需在 reader（flashinfer/b12x）侧确认 scale 字节 0 是否合法路径（多数实现按 2^(b-127) 纯查表，应无问题，但值得一条测试覆盖）。
2. **真实生产数据**（O(0.01–10)）：三处漂移全部不触发，**逐字节等价**——这解释了为何 v17 能对 v11 8/8 逐字节通过（测试数据无退化输入）。
3. **风险定级：低**。但架构上不可接受的状态是：**同一信封 584B 存在两份未统一的量化 spec**——未来 paged v17 移植（R3）时必然要对齐一次，晚对齐不如早对齐。

**建议（K2 行动项，2 人时）**：以 v11 金标准（生产已验证语义）为唯一 spec——即统一为 floor 1e-38 / clamp [-127,128]（或干脆统一到 v17 的 [-126,127]，二选一，倾向 v11 因为生产验证在先）——改 linear v17 两行 + 跑退化输入补充测试（全零/±1e-30/±1e-38 用例，安全报告 §八已设计）+ 在 MANIFEST 写入《584B 信封量化 spec v1.0》唯一权威定义。**注意**：若统一方向是改 v17，需重跑 8/8 逐字节回归确认真实数据无感。

### 2.3 linear v17 的生产价值定位（ADR-2 核心裁定）

**问题本质**：linear v17 的输出是新分配的 `[T,584]` 张量；而生产 vLLM 的 KV 写回路径是 paged in-place（reshape_and_cache 语义）。**linear 变体在生产调用图中没有自然调用点**——这不是缺陷，是它的形态决定的。

**定位裁定（三条，按价值排序）**：

1. **数值金标准 / 回归基线**：v17 已过 8/8 逐字节 + 6 组安全测试，是 kernel2 家族中验证最充分的实现。任何后续变体（paged v17 / R3 / block 参数化修复后的 paged v11 复验）都应以 v17+v11 双参考做逐字节对拍。**它存在的本身就是测试基础设施。**
2. **PR#46329 reader 写端验证器**：584B 线性信封的 reader（flashinfer/b12x 路径）读侧语义需要已知内容写端来验证——用 v17 生成构造性输入（全零/饱和/边界幅值）的确定性 KV 内容，喂给 reader 做读出对拍，是唯一不需要动生产 reader 就能验证"写读闭环"的手段。**建议在 P4 集成阶段增加一项：kernel2 写端（v17 linear）→ reader 读出闭环测试**（这直接服务"服务 PR#46329 reader"的目标，且零生产侵入）。
3. **未来 linear 缓冲路径预留**：若 vLLM 后续版本引入 prefill 线性 KV 缓冲（PR#46329 演进方向），linear v17 届时直接获得调用点。当前只需保持"持久化+可调用"。

**结论**：linear v17 **维持既有裁决（持久化+可调用），即刻可部署**（R1 isfinite 开关、R2 warmup 说明随包落地文档化），不需要也不应该为它制造调用点。kernel2 真正的"生产化第一候选"是 **paged 变体修复后**（block size 参数化 + K3 block_table 核对 + K2 语义统一 + 5/5 复验），届时经 `--kv-cache-dtype nvfp4_ds_mla` + `VLLM_NVFP4_K2=1` 灰度。**R2 warmup 的落实位置在插件层**（CUDA Graph 捕获前首调触发 autotune），这是启用 K2 的前置条件（K5 行动项）。

---

## 三、两算子统一问题清单与修复优先级

分级口径：**阻塞部署**=不修不能进生产挂载/不能注册调用点；**阻塞测试**=不修验证链跑不通或结论无效；**非阻塞**=加固/收尾。负责人角色：DEV=开发（kernel 实现者）、SRE=Rex、QA=Tessa。

### 3.1 阻塞部署（Deploy-Blocking，共 9 项）

| # | 来源 | 问题 | 工作量 | 负责人 | 说明 |
|---|------|------|--------|--------|------|
| D1 | A18 | MXF4 N 向分组(128) vs 标准(1×32) 匹配性实测；不匹配 → NVF4 16 分组 + 转换器 | 4–8h | DEV | P3 最大不确定项，影响 routeB 能否直配生产权重 |
| D2 | K1 | paged BLOCK_SIZE=256 硬编码 → 参数化 + 按生产 block 复验 5/5 | 0.5h 修 + 2h 复验 | DEV | **依赖 SRE block size 确认**；确认前 paged 禁部署 |
| D3 | K3 | paged block_table 布局/dtype 与 vLLM 实际核对 | 1h | SRE+DEV | 与 D2 同批处理 |
| D4 | K2 | paged/v17 量化语义统一（floor/clamp 两处）+ spec 文档化 + 退化输入回归 | 2h | DEV | 统一后双变体回归各跑一次 |
| D5 | K5 | R2 warmup 落实于插件层（K2 启用前置） | 1h | DEV | 仅在启用 `VLLM_NVFP4_K2=1` 时升级为硬阻塞 |
| D6 | A11 | setup 备份存在性检查（.bak 防覆盖） | 10min | SRE | 回滚链完整性（L2 依赖 .bak 干净） |
| D7 | A14 | patch 原子写（临时文件 + os.replace） | 20min | DEV | 防 patch 中断写坏 site-packages mma.py |
| D8 | A16 | setup 显式装 nvidia-cuda-runtime-cu13（去 --no-deps 脆弱假设） | 15min | SRE | P0 环境可靠性 |
| D9 | — | routeB 生产落位同步 + MANIFEST md5 登记 + entrypoint 注册演练 | 2h | SRE | P4 Go 后执行；含 4 节点一致性 md5 比对 |

### 3.2 阻塞测试（Test-Blocking，共 12 项）

| # | 来源 | 问题 | 工作量 | 负责人 | 说明 |
|---|------|------|--------|--------|------|
| T1 | A1 | **host launch 移植**（根阻塞，官方 persistent_pingpong ~150 行） | 8–12h | DEV | 整条 P2 链的解锁点 |
| T2 | A2 | patch equality-check SyntaxError（`arch not in (...)`） | 30min | DEV | 与 T1 并行可做 |
| T3 | A3 | encode_e8m0_32 补 +127（含 clamp 顺序修正） | 15min | DEV | 对照 v17:51 写法 |
| T4 | A5 | SMEM 估算删除 acc_bytes 项 | 10min | DEV | 否则 tile sweep 系统性跳过最优配置 |
| T5 | A6 | W_packed 布局重写（直出 [K,N//2]） | 1h | DEV | |
| T6 | A7 | MmaMXF4Op dtype FP8→Float4E2M1FN（现场核对 4.4.x 签名） | 30min | DEV | 需在目标机 `help(MmaMXF4Op)` 后定案 |
| T7 | A8 | W_scale 重写为 32(K)×128(N) block-max | 1h | DEV | |
| T8 | A12 | patch 字符串拼接改 AST/正则幂等替换 | 1h | DEV | |
| T9 | A13 | --check 加数值 assert（非仅 print） | 30min | DEV | 消除虚假正确性信心 |
| T10 | A15 | 误差度量绝对→相对（rel_err = max_err/ref.abs().max()） | 15min | DEV | 与生产语义（routeA rel<0.02）对齐 |
| T11 | A9+A17 | v17 `_a_quant_kernel` grid 修复 + autotune sm_121a 复测 | 2h | DEV | P3 A 量化复用的前置 |
| T12 | A20 | bench 计时循环外构造 kernel 一次 | 15min | DEV | 性能数字是 Go 门禁，测量必须干净 |

### 3.3 非阻塞（共 8 项）

| # | 来源 | 问题 | 工作量 | 负责人 |
|---|------|------|--------|--------|
| N1 | A10 | setup driver 比较加 fail-fast | 15min | SRE |
| N2 | A19 | patch revert 无备份时 sys.exit(1) | 5min | DEV |
| N3 | A21 | setup cp 加 -p | 2min | SRE |
| N4 | K4 | paged wrapper 加 shape/dtype 防呆断言（随 D2 一并做更优） | 15min | DEV |
| N5 | K6 | R3 paged v17 变体移植（后续迭代，非本窗口） | 1–2d | DEV |
| N6 | K8 | paged 5/5 测试按生产 block 值重参数化（实际并入 D2 复验） | 含 D2 | QA |
| N7 | — | kernel2 写端→reader 闭环测试用例设计（ADR-2 §2.3 第 2 条） | 2h | QA |
| N8 | — | 《584B 信封量化 spec v1.0》文档（并入 D4 产出） | 含 D4 | DEV |

**工作量汇总**：阻塞部署 9 项 ≈ 11–16 人时；阻塞测试 12 项 ≈ 15–20 人时（T1 占 8–12）；非阻塞 ≈ 3 人时 + R3 后续。**建议执行序**：T1 立即启动（根阻塞、工期最长）→ T2–T12 修复包（1 天内可清）→ D6–D8（setup/patch 加固，随 P0/P1 执行）→ D2–D5（kernel2 批，依赖 SRE 确认，独立于 kernel1 关键路径）→ D1 随 P3 → D9 随 P4。

---

## 四、风险矩阵（替换全过程）

| # | 风险 | 概率 | 影响 | 阶段 | 缓解 | 残余风险 |
|---|------|------|------|------|------|---------|
| RK1 | **routeB 不达 350 TFLOPS**（356 单源，无第二复现） | 中 | 高（routeB 放弃，维持 routeA） | P2 | tile sweep 扩展（128×256/256×256 ≤99KB）+ num_warps 调优；No-Go 判据明确 | 低——routeA 现役，零损失退场 |
| RK2 | **kernel2 paged 块大小错配**（若 block=64 且未参数化误部署） | 待定（SRE 验证中） | **致命**（静默写坏 KV cache，无告警） | kernel2 部署 | D2 参数化 + 防呆断言 + 确认前禁部署；5/5 复验不外推 | 低——只要纪律执行"确认前不部署" |
| RK3 | **host launch 移植工期溢出**（A1 超 1.5 天） | 中 | 中（P2 顺延，窗口挤压） | P2 | 官方示例整段移植不做改写；Jerry2423 整理版作二母本对照 | 中——DSL 4.4 API 漂移可能使移植超估 |
| RK4 | **MXF4 N 向分组不匹配**（R13/D1 坐实） | 中 | 高（需 NVF4 16 分组 + scale 重算转换器，+0.5–1 天） | P3 | D1 提前实测（可在 P2 期间用 --check 部分预验）；备选路径明确 | 中 |
| RK5 | **SASS bf16 静默回退** | 中 | 致命（假 4W4A，违反红线） | P2 | `nvdisasm \| grep mma.*e2m1` 硬门禁，不可绕过 | 低——门禁在位 |
| RK6 | **patch 损坏 site-packages**（非原子写/备份覆盖） | 中 | 高（DSL 环境不可用，回滚链断裂） | P1 | D6/D7 加固 + revert 演练纳入 P1 Go 判据 | 低 |
| RK7 | **paged/linear 语义漂移污染 KV cache** | 低 | 中（仅退化输入触发，数值影响趋零） | kernel2 | D4 统一 spec + 退化输入回归 + reader 对 scale=byte0 合法性确认 | 低 |
| RK8 | **K2 启用与 CUDA Graph 冲突**（autotune 在 graph 内触发） | 低 | 高（capture 失败/性能尖刺） | kernel2 灰度 | D5 warmup 前置 + 灰度期 graph capture 日志监控 | 低 |
| RK9 | **双轨并行窗口溢出**（kernel1 5.5d + kernel2 独立项） | 中 | 中（生产重启延期） | 全程 | kernel2 全部不占 kernel1 关键路径；窗口按 6–7 天申报；不足则 kernel1 P4 顺延 | 低——routeA 现役，延期无生产代价 |
| RK10 | **routeB 插件注册暴露未知问题**（首次真调用点） | 中 | 中 | P4 | 分派契约内 env 降级开关（L0）+ L1 删引用演练 <10min 纳入 Go 判据 | 中 |

---

## 五、给主理人的架构裁定请求（需人拍板项）

1. **ADR-1b 集成层 Option(a)**：本文与 precheck 一致推荐 Option(a)，并新增"dense-only 范围界定"（MoE grouped 永不切 routeB）——请确认范围界定。
2. **ADR-2 §2.3 linear v17 定位**：维持"持久化+可调用"、不制造调用点、增加"写端→reader 闭环测试"——请确认闭环测试纳入 P4。
3. **K2 语义统一方向**：统一到 v11 金标准（[-127,128] / 1e-38）还是 v17（[-126,127] / 1e-30）？本文倾向 v11（生产验证在先），但两者对真实数据等价，属低风险决策。
4. **窗口申报**：按 6–7 天双轨申报还是维持 routeA 现役把 kernel1 P4 灰度顺延下一窗口？取决于本次停机窗口剩余时长（主理人掌握）。

---

## 附：本报告与并行审查的关系

kernel2 交付包的代码级审查由 code-reviewer 并行进行；本文 kernel2 部分为**架构侧独立阅读产物**（K1–K8 编号体系独立于 Cody 的编号），交叉验证与去重由主理人汇总。K1（BLOCK_SIZE 硬编码）与任务简报中"SRE 现场验证生产 block 疑为 64"互为印证；K2（语义漂移）与安全报告"v17 与 v11 逐字节一致"**不矛盾**——一致性结论基于真实幅值测试数据，漂移只在退化输入触发，详见 §2.2。

> 免责声明：本报告基于交付包源码逐行静态审查 + precheck 权威清单交叉引用，未在目标硬件实跑。runtime 行为（SASS 发射、TFLOPS、block size 实际值）以现场执行为准。关键裁定请人类工程负责人复核。
