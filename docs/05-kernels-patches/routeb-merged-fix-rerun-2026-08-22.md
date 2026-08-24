# 修复重跑报告：merged-GEMM 插件 fix-rerun 窗口（2026-08-22）

**执行**: Archi（fix-engineer）· node01-04 TP4 生产集群 · 停机窗口 00:18–02:0x UTC
**结果**: **PR 四档 No-Go（新发现的架构级数值阻塞）**；P0-A 编译归因证伪；P0-B 调查升级为 w1 权重 scale 表示问题（阻塞根因）；Triton 路径两个独立真 bug 修复并验证；生产基线已恢复。

> **一页结论（供裁决）**
> 1. **P0-A 证伪**：cute.compile 对照实验（全核/钉 5 核 × 生产 shape b=262144 / 独立口径 b=32768 × M_pad 512/8224）全部 **~2.5s**——编译与 shape/CPU 无关。"DSL 冷编译 ≥45min" 的归因错误（45min 挂起真因未明，但已无必要：数值阻塞在前）。CUTE_DSL 文件缓存实测不命中（只落 .mlir dump），AOT 烘焙不可行亦无必要。
> 2. **P0-B 升级为架构阻塞（本窗口最重要发现）**：生产 checkpoint 的 **w1.scale 为宽分布**（35.4% 字节 < 2⁻⁹、28% > 512、行内 exponent 跨度中位 249——反量化有效权重跨 1e-39~3e38），而 w3/w2 scale 全部聚集 [121,123] 正常。E8M0→E4M3 LUT 对 byte≥136 产生 **NaN**（torch e4m3fn cast 无饱和）→ merged 路径 SFB 半数为 NaN/0 → **mini e2e logprob 38-41.5% 的主因**（两级 scale A 量化修复前后无差，实证排除 A 侧主因假设）。E4M3 kernel 表示范围 [2⁻⁹, 448] 无法承载 w1 分布，**无标量折叠可修**（跨度在行内沿 K 方向）。
> 3. **B12X 的 w1 真实语义未逆向**：B12X 以 fp4_e8m0_k32 原生消费这些字节且基线正常，但朴素 2^(b-127) 解释会 inf/NaN——b12x 库内部有"E2M1×2⁻¹²⁶ + scale×2⁷ + epilogue 补偿"因子化，未完全逆向。**逆向之前任何替代路径（merged/Triton）都无法在 w1 上数值正确**。
> 4. **Phase C 的 mini 0.36% 是假信号**：该 run 逐 prompt prompt-lp 与基线**逐位一致（0.00%）**——merged 未 engage，0.36% 全部来自 gen-lp 漂移。mini 全暴露口径的真实数值质量从未被验证过。
> 5. **Triton 长尾/decode 路径两个真 bug（v2.1 已修复，11/11 单测）**：E4M3 scale 整数直转（56× 系统误差）+ 排序分块跨界专家（随机路由大面积错值）。修复版 Triton 全链 vs torch 参考 rel=5.3e-3。
> 6. **v2.1 插件资产**（未部署生产）：两级 scale A 量化（flashinfer 实证语义 + fp16 溢出 cap16）、M_pad 档位化、SF/c 壳复用（修 v2.0 每调用 67MB 泄漏）、DSL 预编译 warmup、Triton 重写（专家对齐 + 静态容量哨兵 grid，CG 安全）。
> 7. **生产恢复基线**：start 脚本本窗口零改动（从未部署 v2.1）；集群已重启回基线（恢复验证见 §5）。

---

## 0. 执行时间线（UTC）

| 时间 | 事件 |
|---|---|
| 00:18:16-28 | 停机开窗：四节点服务停 + rank0-3 容器移除 + GPU 空闲验证（03/04 残留 EngineCore 属 anemll-embed 常驻，无关） |
| 00:20-00:45 | P0-A 编译对照实验（结论见 §1） |
| 00:40 | 中期汇报 team-lead（编译证伪 + 复现计划） |
| 00:45-01:30 | v2.1 插件实现 + 单元验证 11/11；期间发现 Triton 双 bug 并修复 |
| 01:00-01:15 | mini logprob 重跑（首次误跑 v2.0——sys.path 抢占；v2.1 重跑 38.4%） |
| 01:15-01:55 | 数值根因深挖：E4M0 宽分布发现 → LUT NaN → B12X 语义未明（§2） |
| 02:0x | 阻塞汇报 team-lead + 恢复生产基线 |

## 1. P0-A：DSL 编译归因证伪

| 条件（一次性容器，生产镜像） | cute.compile 耗时（含首执行） |
|---|---|
| 全核（20 cpu），M_pad=512，w13 生产 shape（B=262144, N=6144, K=4096） | 2.4s |
| 钉 5 核（cpuset 15-19），同上 | 2.4s |
| 独立测试口径（B=32768） | 2.6s |
| w2 combo shape（K=12288, B=1024） | 2.3s |
| M_pad=8224（hash 层最大桶档）全三 shape | 2.2-4.0s |

- 编译与 b_rows（8×差）、M_pad（16×差）、CPU 核数全部无关。冒烟"独立测试 2-4min"口径 = 脚本整体时间（含 import/torch/CUDA init）。
- 45min 挂起真因未复现（容器日志已随 rm 丢失）。mini（同 EngineCore 环境）当时跑通 → 环境差异在 TP4/43 层/MTP/cudagraph。**挂起调查已无必要**：数值阻塞（§2）使 merged 路径本就无法上线；若未来重启该项目，建议现场 py-spy 抓栈（host 已装 py-spy 0.4.2，脚本已备 p0a_pyspy_watch.sh）。
- 预编译退化为保险项：v2.1 已实现 _derive 期 M_pad 档 warmup（7 档 × 2 GEMM ≈ 35s 启动吸收）。

## 2. P0-B → 架构阻塞：w1 权重 scale 表示问题（证据链）

### 2.1 生产 checkpoint 实测（layers.0.ffn.experts.0，mini 同构）

| 张量 | 分布 | 行内 exponent 跨度 |
|---|---|---|
| **w1.scale** [2048,128] E8M0 | 35.4% < 2⁻⁹ · 36.7% 区间内 · 28% > 512 | **中位 249** |
| w3.scale [2048,128] | 100% 聚集 [~121,123] | 1-2 ✓ |
| w2.scale [4096,64] | 100% 聚集 [~121,123] | 1-2 ✓ |

反量化 w1 有效权重（payload×2^(s-127)）跨 **1e-39 ~ 3e38**；MTP 层同病（33.9%/28%）。E4M3 kernel 表示范围仅 [2⁻⁹, 448]（span 18）——行内沿 K 方向的 249 跨度**无任何标量/行级折叠可修**（scale 在 K-reduction 内混合）。

### 2.2 LUT NaN

`E8M0_TO_E4M3_LUT`（Task#20 同源构造）：byte ∈ [136, 255]（120 个值）→ torch e4m3fn cast 产生 **NaN**（无饱和）→ merged SFB 含 NaN scale 字节。

### 2.3 mini e2e 数值排查记录

| 配置 | mini 总 logprob 差 | prompt-lp 逐 prompt 差（确定性口径） |
|---|---|---|
| Phase C v1（lp_pc_merged.json） | 0.36% | **0.00%（逐位=基线——merged 未 engage，假信号）** |
| 冒烟 v2.0 重跑 | 38.4% | 54-63% |
| v2.1（两级 scale+Triton 修复+壳复用） | NaN×3 + 40.6% | 62-67%（NaN = fp16 溢出级联，A 侧 cap16 仍不够 mini 退化路由的 h 幅值） |

两级 scale 修复前后无差 → **A 侧量化器不是主因**（原假设排除）；selfcheck per-bucket rel 2.5-13.6%（同参考口径下正常 W4A4 应 ~1e-3）→ W 侧解释错误实锤。

### 2.4 B12X 语义未明（阻塞核心）

- B12X 集成链实证：`_source_format()="fp4_e8m0_k32"`、scale_format 原生 e8m0、blockscale 原样传入 `b12x.integration.prepare_b12x_fp4_moe_weights`。
- b12x 库 `cute/fp4.py: packed_dequant_e8m0x4_to_{half,bfloat}2x2`：docstring 表明存在"E2M1×2⁻¹²⁶ 表示 + scale×2⁷ 物化进 MMA 输入 + epilogue 剩余补偿"的因子化拆分（inline PTX 构造 half 位型）——完整语义未逆向。
- 朴素 2^(b-127) 解释 → h1=inf（实测）→ 与基线可运行矛盾 → **B12X 内部必有额外语义**（拆分/补偿/钳制）。
- 结论：**在逆向 B12X w1 有效权重语义之前，merged 与 Triton 替代路径都无法在 w1 上数值正确**。注：v2.1 的 Triton w13 分支同样消费 LUT 化 scale（w1 行同样损坏）——Triton 路径的 w1 修复同样被此阻塞。

## 3. Triton 路径双 bug（已修复 + 验证）

1. **E4M3 scale 整数直转**：v2.0 `tl.load(sf).to(tl.float32)` 把字节值当标量（scale 字节 56 → 56.0 而非 1.0）→ 确定性微测试实证（w13 输出 114688 vs 期望 2048 = 56×）。修复：`.to(tl.float8e4nv, bitcast=True).to(tl.float32)`。
2. **排序分块跨界专家**：16-row 块跨界 pair 用块首专家的权重（随机路由下大面积错值；块内专家纯一性破坏）。修复：按专家对齐 padding（searchsorted 构造，静态形状）+ 哨兵块（expert_id=E）早退 + 静态容量 grid `ceil((M·T+E·15)/16)`（CG 捕获安全，M/T/E 静态）。
- 验证：确定性微测试位级精确；随机 A/B vs torch 参考 rel=5.3e-3（v2.0 同输入 173× 误差）；单元 11/11。
- 注意：w2 分支改 E8M0 逻辑直寻址（exp2(u8-127) in-kernel 精确，零拷贝视图）——**w2 的数值语义已正确**（w2 scale 分布正常）；w13 分支的 w1 部分仍被 §2 阻塞。

## 4. v2.1 插件资产（/tmp/_routea_work/plugin_merged_v21，未部署生产）

| 组件 | 修复内容 | 验证 |
|---|---|---|
| dsl_gemm.py | 两级 scale A 量化（flashinfer 实证语义：gs=2688/amax, sf=(blk/6)×gs, payload=x×gs/sf, 输出÷gs；fp16 溢出保护 GS_CAP=16 可调）；M_TIERS 档位化 + pick_tier；SF/c 壳复用（v2.0 每次 GEMM 调用泄漏 SF 壳 67MB + 268MB f32 瞬态——KV -29% 主嫌疑）；make_sf_cute 壳背衬即物理存储 | T1-T4b PASS |
| triton_moe.py | 双 bug 修复 + E8M0 w2 寻址 + 专家对齐 + 哨兵容量 grid | T5 PASS, 微测试位级 |
| merged_experts.py | DSL 预编译 warmup（进程级一次，~35s）；w2 combo SF 壳随 combo 缓存；_s2_logical 驻留删除；selfcheck 修正 | mini 实跑 |
| 单测 | p21_unit_test.py 11/11（两级 scale/档位 pad/壳复用无泄漏/数值/E8M0 Triton） | 容器内 PASS |

## 5. 生产恢复验证

（恢复中——集群 head-first 冷启动 ~12min；终局数字：4 rank healthy + "Using B12X_MXFP4" + KV tokens + /health 200 + 零 Merged 日志）

## 6. 遗留移交（下一窗口裁决项）

| 项 | 优先级 | 说明 |
|---|---|---|
| **B12X w1 语义逆向** | P0（前置） | 用 b12x prepare+kernel 对 mini w1 做数值反解（已知有效输出 → 解 effective W）；预计 0.5-1 天。完成前 merged/Triton 的 w1 均无数值正确性可 |
| **merged 架构裁决** | P0 | 若 w1 不可进 merged：上界从 +5-15% 砍至 +3-8%（w13 收益减半），ROI 重评；备选 w3+w2-only merge（需 B12X 调用面改造取 h1）或搁置 |
| w1 宽分布成因 | P1 | 向模型/checkpoint 侧确认为何 w1.scale 宽分布而 w3/w2 正常（量化器 bug？有意为之？）——若是 checkpoint 侧 bug 且可修复重转换，全问题消失 |
| 45min 挂起真因 | P2 | 已无紧迫性；现场 py-spy 抓栈脚本已备 |
| mini 验证方法论 | P2 | prompt-lp 逐位对照必须 engage 验证（本次教训：0.36% 假信号）；比较脚本应校验 merged 确实执行（bucket 计数>0） |

## 7. 工件

| 位置 | 内容 |
|---|---|
| 01:/tmp/_routea_work/ | plugin_merged_v21/（全套代码）· p21_unit_test.py（11/11）· p0a_compile_probe.py + p0a_{full,pin}.log（编译对照）· p0b_probe.py/p0b_probe2.py（flashinfer 语义实证）· p0b_scale_blocker.py · p0b_w1_probe.py · p0b_lut_check.py（阻塞证据链）· p21_t5_{micro,micro2,ab}.py（Triton bug 微测试）· p21_lp_compare.py（四 run prompt-lp 对照）· lp_e2e_merged_v21.{json,log} · e2e_mini_v21.sh · p0a_pyspy_watch.sh |
| 本地 _routea_work/ | 同步上述全部 |
| 生产 | **零改动**：start 脚本未碰；/opt/.../plugin_merged 保持 v2.0（无 env 不激活）；基线运行 |
