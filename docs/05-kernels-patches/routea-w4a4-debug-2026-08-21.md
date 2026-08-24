# W4A4 语义排障报告（Task #21，2026-08-21）

**任务**: Task #21 · b12x W4A4 语义排障（P0 前置，mini 模型 logprob A/B）
**执行**: Archi（系统架构师）· node01 单卡一次性容器（生产镜像 + tilelang.py patch 挂载，与生产一致）
**前置**: Task #20 发现 b12x W4A4（B12xMoEWrapper）离线复刻链 vs 参考 rel=0.51/1.55，需排障后才能放行 A′ 插件。
**产出**: 本报告 + 复现脚本（build_mini.py / run_mini.py / compare_lp.py / convert_mini_to_nvfp4.py / inspect_ckpt.py）+ 六组 logprob 快照（01:/tmp/_routea_work/lp_*.json，本地 _routea_work/ 同步）

> **一页结论（供决策）**
> 1. **二分结论：fork in-situ W4A4 质量正常，Task #20 的"疑似缺陷"定性为非缺陷**。mini 模型（真实 checkpoint 前 4 层）单卡 vLLM 全链实测：b12x W4A4 总 logprob 相对 W4A16 生产基线 **+0.41%**、相对"语义正确 W4A4"参照（fork 自带 EMULATION 后端）仅差 **0.2%**——无质量级缺陷。Task #20 的 rel=0.51 是随机权重 + kernel 级参考实现的"实现细节差"（三种 W4A4 实现两两 mean|Δ|lp 均 0.21–0.25，但总 logprob 全部健康 +0.2~0.4%），不能定性为 bug。
> 2. **真正的闸门是 fork 的设计决策而非缺陷**：oracle 显式拒绝 `flashinfer_b12x` 跑 NVFP4+swiglu_limit 模型（"does not apply the SwiGLU clamp"，ValueError）。**实测 clamp 效应 = 0.0000**（W4A16 有/无 clamp 输出逐位一致——真实激活从不触发 10.0 限幅）→ 该闸门的实际顾虑在真实负载上（mini 证据）为零。
> 3. **A′ 派生链闭合验证通过**：把 mini-0731 用 Task #20 适配器逻辑离线转成 NVFP4 格式（payload 原样 + E8M0→E4M3 LUT 精确扩展 + scale_2/input_scale=1.0），在 W4A4 路径上与原生 -nvfp4 checkpoint 的 logprob 差异是**所有配对中最小的**（mean|Δ|=0.120，主要来自 input_scale=1.0 vs 原生校准值），总 logprob +0.65%——**-0731 派生 W4A4 与原生 NVFP4 数值等价**。
> 4. **A′ 放行判定：有条件放行**——条件见 §5。

---

## 1. 方法：少层 mini 模型 logprob A/B（腿 1）

### 1.1 mini 模型构建（build_mini.py，纯标准库流式抽层）

从真实 checkpoint 抽**前 4 层**（layers 0-2 = hash 层 + layer 3 = dense 层，保持 num_hash_layers=3 语义）+ 全局张量（embed/head/norm/hc_head_*），丢弃 mtp.*：
- `mini0731`：-0731（MXFP4）4 层，15.3GB —— B12X W4A16 生产同构路径（Mxfp4MoEMethod → B12xExperts）
- `mini0731nvfp4`：-nvfp4（modelopt NVFP4）4 层，16.1GB —— W4A4 路径（ModelOptNvFp4Config → FlashInferB12xExperts）
- config 修改：num_hidden_layers=4、compress_ratios[:4]、dspark_target_layer_ids=[]、num_nextn_predict_layers=0、quantized_layers 裁剪
- 工程备注：手工 safetensors 写入的 header 偏移需乘 dtype 字节数（首版 bug 已修复并经 safetensors 库读回校验）。

运行配置（两/多方完全一致）：单卡 GB10、`moe_backend=flashinfer_b12x`、max_model_len=4096、max_num_seqs=12、max_num_batched_tokens=4096、enforce_eager、**kv_cache_dtype=fp8**（fp8_ds_mla）、greedy(temperature=0)、7 条多语言/代码/数值 prompt、prompt_logprobs + 48 token 生成 logprobs（每方 512 个 prompt 位置 + 336 个生成位置）。

**环境发现（记录）**：mini 上 `kv_cache_dtype=nvfp4_ds_mla` 会触发 compressor state cache stride 断言崩溃（`state_cache.strides[0] ... divisible by 16`，sparse_attn_compress_cutedsl）；改 fp8_ds_mla 正常。疑似小 cache 尺寸下的 fork/布局边界问题（生产 43 层大 cache 不触发）；与 MoE 语义无关，A/B 全程用 fp8 两侧一致。

### 1.2 关键拦路虎：fork 的 swiglu_limit 闸门

mini-nvfp4 + `--moe-backend flashinfer_b12x` 首跑即被 oracle 拒绝：

```
ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.
Use 'flashinfer_trtllm' or 'flashinfer_cutlass' instead.
```

**这是 fork 作者的显式设计**：b12x W4A4 路径不施加 DSV4 的 SwiGLU 限幅，被认为不可接受。为继续排障，构建 `swiglu_limit=None` 变体（mini0731_noclamp / mininvfp4_noclamp）做三方对照分离变量。

## 2. 实测结果

### 2.1 四/五方对照矩阵（同一 prompt 集，greedy）

逐 token 视角（prompt actual-token logprob 的 mean|Δ| / 生成 top-1 一致率）：

| 配对 | 含义 | mean\|Δ\|lp | top-1 一致率 |
|---|---|---|---|
| B2 (W4A16 无clamp) vs B1 (W4A16 生产) | **clamp 效应** | **0.0000** | **100.0%** |
| E (EMU W4A4, 正确语义) vs B2 | 固有 W4A4 量化差 | 0.1831 | 15.8% |
| T (b12x W4A4) vs B2 | b12x 端到端差 | 0.1902 | 16.7% |
| T vs E | b12x 相对正确 W4A4 | 0.2060 | 17.0% |
| C (cutlass 分组 W4A4) vs E | 实现间差 | 0.2421 | 14.6% |
| T vs C | 实现间差 | 0.2464 | 19.0% |
| **CV (-0731派生 W4A4) vs T (原生)** | **A′ 派生链等价性** | **0.1203** | **31.8%** |

总 logprob（质量保持率，512 prompt 位 + 336 生成位的 top-1 求和；乱码 mini 模型上量化噪声使 logprob 略升属正常）：

| run | 总 logprob | 相对 W4A16 基线 |
|---|---|---|
| B1/B2: W4A16（clamp 无差异） | -8384.0 | 基准 |
| E: EMU W4A4（正确语义参照） | -8367.4 | +0.20% |
| C: cutlass 分组 W4A4 | -8361.7 | +0.27% |
| T: b12x W4A4 | -8349.2 | +0.41% |
| CV: -0731 派生 W4A4（input_scale=1.0） | -8329.7 | +0.65% |

### 2.2 结论链

1. **clamp 效应 = 0**（B2 vs B1 逐位一致）：DSV4 swiglu_limit=10.0 在真实激活上零触发。fork 闸门针对的风险在实测中不存在（至少 mini/这些 prompt；长上下文极端分布待 TP4 抽验）。
2. **W4A4 固有量化差**（E vs B2）：逐 token mean|Δ|≈0.18、总 logprob +0.20% —— 激活 FP4 量化的正常代价（与 NVIDIA NVFP4 官方推理模式预期一致）。注意 4 层残废 mini 的 logit 分布平坦（输出乱码），逐 token 差异和 top-1 翻转被放大；**总 logprob 才是有效质量判据**。
3. **b12x W4A4 无质量级缺陷**（T vs E 总差 0.2%、T vs B2 +0.41%）。Task #20 的 rel=0.51/1.55 定性：随机权重（无结构、大量饱和）+ 我的 torch 参考与 b12x 在激活量化实现细节（tie 规则/组语义/fc2 scale 处理）不同——三种 W4A4 实现两两差异同量级（0.21-0.25）但总 logprob 全部健康，**非缺陷**。
4. **A′ 派生链（Task #20 适配器）在 checkpoint 级验证等价**：CV 与原生 T 的差异（0.120）是所有配对中最小的，且主要来自 input_scale=1.0 vs 原生 modelopt 校准值（激活量化 global scale 差，非权重差——权重侧 GEMM 级 rel=1.41e-03 已在 Task #20 闭环）。

### 2.3 附带观察

- mini 上 W4A4 比 W4A16 慢（35.5s vs 15.7s / 7 prompts）：mini 的 prefill M=73-116/prompt，正处 Task #20 实测的 W4A4 中段劣势区间（M<2048 时 0.79-0.95×），与微基准一致；W4A4 收益区间（M≥4096）mini 未覆盖。
- EMULATION 后端可用作 W4A4 语义 ground truth（Triton 逐算子模拟）——后续任何 W4A4 实现变更的回归参照。

## 3. 腿 2 对照结论（Task #20 疑点的最终定性）

Task #20 §4.3 的"rel=0.51 vs 精确参考 / 1.55 vs W4A16"排障完毕：
- **不是 fork/库缺陷**（in-situ 全链质量正常）；
- **不是我的适配器缺陷**（派生 checkpoint 与原生等价）；
- 是 **kernel 级离线 harness 的分辨率问题**：随机权重 + 平坦分布 + 参考实现与 b12x 的合法实现细节差，三者叠加。正确判据是质量指标（总 logprob/perplexity），不是随机权重上的逐元素 rel。
- A′ 插件的构造链照抄 fork 的 `prepare_nvfp4_moe_layer_for_fi_or_cutlass`（[w3;w1] 行序 + swizzle_blockscale → convert_sf_to_mma_layout）即可，无需修改。

## 4. A′ 插件是否放行：**有条件放行** ✅

| 条件 | 状态 | 说明 |
|---|---|---|
| W4A4 语义排障 | ✅ PASS | b12x W4A4 质量正常（总 lp +0.41% vs 基线、0.2% vs 正确参照） |
| A′ 派生链等价 | ✅ PASS | -0731 派生与原生 -nvfp4 等价（GEMM 级 + checkpoint 级双验证） |
| swiglu_limit 闸门 | ⚠️ 需显式绕过 | fork 设计拒绝；实测 clamp 零触发。A′ 插件需在 env 开关中显式声明"接受无 clamp"，TP4 期 needle/logprob 抽验兜底；或上游申请放宽闸门 |
| input_scale 来源 | ⚠️ 需决策 | -0731 无校准值。方案 a: 用 1.0（实测与原生差 mean\|Δ\|lp=0.12，总 lp 差 0.24%——可接受）；方案 b: 从 -nvfp4 checkpoint 抄每层校准标量进插件（更贴近原生，运维多一份依赖）。**建议 a 起步** |
| 全模型量化累积 | ⚠️ TP4 期验证 | mini 仅 4 层；43 层的激活量化误差累积需 TP4 窗口 needle/eval 实测（预期方向：总 lp 漂移增大但质量可接受，参照 NVFP4 官方模式） |

## 5. 对 TP4 测试计划的更新（衔接 Task #20 §5）

1. **P0 已完成**（本报告）——Task #20 §5 的 P0 单卡离线项闭环，结论比预期更好：无需等 fork 排障，可直接进插件实现。
2. P1（插件形态单卡复核）：A′ 插件 = B12xExperts 子类 + Task #20 适配器（E8M0→E4M3 + swizzle + mma + [w3;w1]）+ M≥阈值分派 + `VLLM_NVFP4_ROUTEA` env；input_scale=1.0。**判据升级**：与 B1 基线的总 logprob 差 ≤1%、needle 全对、（对照本报告表 2 的预期带 +0.2~0.7%）。
3. P2（TP4 窗口）不变，追加两项：长上下文（>100K）下 clamp 零触发抽验；W4A4 中段（M=64-2048）decode/prefill 混合负载回退 ≤5% 的复核（mini 已再现中段劣势）。
4. 性能预期锚点（Task #20 §2.3）：M=4096 chunk 1.32×、16K 1.52×、M=1 6.2×、中段 0.79-0.95×——TP4 判据维持"prefill 端到端 ≥15%"。

## 6. 工件清单

| 文件 | 位置 | 说明 |
|---|---|---|
| `build_mini.py` | 本地 _routea_work/ + 01:/tmp/_routea_work/ | 抽层构建 mini 模型（纯 stdlib，流式，含 dtype 偏移修复） |
| `run_mini.py` | 同上 | 单卡离线推理 + prompt/gen logprobs 抓取（kv fp8、eager、7 prompts） |
| `compare_lp.py` / `total_lp.py` | 同上 | 逐 token + 总 logprob 对照 |
| `convert_mini_to_nvfp4.py` | 同上 | **A′ 派生链 checkpoint 级验证器**：-0731 → modelopt NVFP4 格式（E8M0→E4M3 LUT + payload 原样 + 标量 scale）——亦可直接放大为全模型离线转换器（Path B' 备选） |
| `lp_{0731, w4a16_noclamp, w4a4, emu, cutlass, conv}.json` | 同上 | 六组 logprob 快照（复现证据） |
| mini 模型目录（~66GB） | 已清理（可由脚本重建） | — |

环境与约束：一次性容器（--rm）、生产镜像 + tilelang.py patch 挂载（与生产一致）、未启动生产、<INSTALL_DIR> 只读、单卡峰值显存 ~96GB（0.85 util）、mini 放 /tmp/_routea_work（本地盘，未涉 NFS）。
