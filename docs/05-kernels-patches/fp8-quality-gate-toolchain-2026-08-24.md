# FP8 质量门工具链设计（2026-08-24）

- **作者**: 阿奇（Archi）· 系统架构师（architect-1）
- **任务**: FP8 质量门工具链（G2-G5 + 参考集 + 噪声底协议）设计 + 零 GPU 脚本落地；BF16 噪声底实测排入 GPU 窗口
- **上游输入**: fp8-quality-impact-2026-08-23.md（G1-G8 门设计）、lmhead-fp8-project-2026-08-23.md（工具链先行是 lm_head 硬前置）、quality_gate.py（现役生产贪心质量门）、bench-regression-attribution-2026-08-23.md（KL 门建议/假阳性史）、upstream-check-perf-ceiling-2026-08-23.md（P2 门）
- **口径标注**: 【实测-源码】= 既有源码/快照验证；【实测】= 既有实测数据；【推算】= 本工具链未实测前的量级估计；【待窗口验证】= 必须由 GPU 窗口执行
- **纪律**: 零 GPU 只做设计 + 脚本 + 逻辑自检；噪声底实测/参考集采集全部排入 GPU 窗口

---

## 0. 一页结论

1. **工具链构成**：1 个参考集（`reference_set.json`，31 条 = Tier A 4 + Tier B 24 + Tier C 3）+ 7 个脚本（builder / collector / noise_floor / kl_gate / greedy_baseline / temp_top_p_gate / selftest）。
2. **零 GPU 已就绪**：参考集定义、参考集校验/长上下文生成、噪声底分析、KL+困惑度门、greedy 基线比对（离线）、temp/top-p 分析、全套逻辑自检——**全部零 GPU 可验证，`selftest.py` 全绿**。
3. **GPU 窗口待办**：BF16 背靠背两次采集（噪声底实测）→ 门限定标；FP8 采集 → G2/G3 判定；greedy golden 采集（或 import 既有快照）；temp/top-p 采集；64K 长上下文采集。
4. **核心机制**：噪声底先测再定门（KL 2-3×，困惑度 Δ≤0.05 或按噪声底重标定）；受限 top-k 支持 KL 为估计量（top_logprobs=10）；greedy 基线资产与现役 `quality_gate.py` 同 prompt、同包络，生产 compare 仍以 `quality_gate.py` 为唯一入口。
5. **集成方式**：新增独立脚本（`_fp8_qg_toolchain/`），不改 `quality_gate.py`；两者共享 prompt 与 logprob 口径，由 `reference_set_builder validate` 强制对齐防漂移。

---

## 1. 背景与目标

### 1.1 为什么工具链先行

- fp8-quality-impact 裁定：**lm_head-FP8 是池内质量门最高节点**（唯一 BF16 无损节点 + 最终层 logits 直驱采样），G1-G7 全门是 F2/F4 硬前置；shared-FP8 F4 复用 G1-G3。
- KL/困惑度门需要的是 **"参考数据集 + BF16 噪声底"而非"校准"**（FP8 E4M3 权重变换数据无关，无校准门）。
- 本工具链 = G2（KL 门）+ G3（困惑度门）+ G4（温度采样）+ G5（top-p/top-k）+ 参考集 + 噪声底协议；G1（greedy）由现役 `quality_gate.py` 承担、本工具链做资产固化。

### 1.2 门矩阵覆盖（对应 fp8-quality-impact §4.1）

| # | 门 | 判据 | 工具链载体 | 阶段 |
|---|---|---|---|---|
| G1 | greedy 硬门 | 4 稳定 prompt 4/4；包络 ≤1% 兜底 | `quality_gate.py`（现役）+ `greedy_baseline.py`（资产固化） | F2/F3/F4/F5 |
| G2 | KL 门 | `KL(p_bf16‖p_fp8) < 3×噪声底` | `kl_gate.py` | F4/F5 |
| G3 | 困惑度门 | `PPL Δ ≤ 0.05`（噪声底超标则重标定） | `kl_gate.py` | F4/F5 |
| G4 | 温度采样 | temp∈{0.6,0.9}：logprob drift ≤1% + 候选集重叠率 ≥0.90 + distinct-n 不降 | `temp_top_p_gate.py` | F4/F5 |
| G5 | top-p/top-k | top-p=0.9 / top-k 抽验：候选集重叠率 ≥0.90 | `temp_top_p_gate.py`（top-p） | F4/F5 |
| G6 | MTP 接受率 | DE C1/C12 step_eff ±3% | `de_bench.py`（现成，不在本工具链） | F4 |
| G7 | 长上下文 64K | needle 3/3 + 尾部指令 | 参考集 Tier C + 既有 needle 流程 | F4/F5 |
| G8 | off 等价 | env off byte-equivalent | L1（不在本工具链） | F2/F3 |

---

## 2. 工具链架构

### 2.1 组件图

```
                    ┌──────────────────────────────────────────────┐
                    │  reference_set.json  (31 条 = A4+B24+C3)      │
                    │  tiers: A greedy-exact / B distribution / C   │
                    │         longctx(64K)                          │
                    └───────────────┬──────────────────────────────┘
                                    │
                 reference_set_builder.py  (零 GPU)
                   validate / build(长上下文) / stats
                                    │
                                    v
        ┌───────────────────────────┴───────────────────────────┐
        │  reference_set_collector.py  (GPU 窗口，VLLM_API_KEY) │
        │  --config greedy|dist|temp  → runs/run_<quant>_<tag>  │
        └───────────────────────────┬───────────────────────────┘
                                    │ fp8-qg-run/1 JSON
        ┌──────────┬──────────────┬─┴─────────┬────────────────┐
        v          v              v           v                v
  noise_floor  kl_gate      greedy_baseline temp_top_p     (G6/G7/G8
  (BF16 run1   (BF16 vs     (golden 资产 +   (BF16 vs FP8   现成门)
   vs run2)    FP8)         compare)         temp/top-p)
        │          │              │              │
        v          v              v              v
   assets/noise_floor.json  assets/kl_verdict.json  assets/golden-*.json  assets/temp_top_p_verdict.json
```

### 2.2 目录结构

```
deliverables/engineering-assurance/_fp8_qg_toolchain/
├── reference_set.json            # 参考集清单（v1，31 条）
├── run_common.py                 # 共享：API 采集 / run schema / KL/PPL/漂移数学
├── reference_set_builder.py      # 参考集校验 / 长上下文生成 / 统计（零 GPU）
├── reference_set_collector.py    # 在线采集（GPU 窗口）
├── quality_gate_noise_floor.py   # BF16 噪声底分析（零 GPU，输入两次 run）
├── kl_gate.py                    # KL 门 + 困惑度门（零 GPU）
├── greedy_baseline.py            # greedy 4/4 基线固化 + 比对
├── temp_top_p_gate.py            # 温度/top-p 抽验（collect GPU / analyze 零 GPU）
├── selftest.py                   # 零 GPU 逻辑自检（全绿）
├── _generated/                   # build 产出的 64K 长上下文 prompt（needle/tail）
├── assets/                       # golden 基线 / 噪声底 / 门限判定（GPU 窗口填充）
└── runs/                         # 采集 run 文件（GPU 窗口填充）
```

### 2.3 Run 文件 schema（fp8-qg-run/1）

```json
{
  "schema": "fp8-qg-run/1",
  "meta": {"quant": "bf16|fp8", "model": "...", "config": {"temperature": 0.0,
           "top_p": 1.0, "top_logprobs": 10, "max_tokens": 256},
           "collected_at": "UTC", "notes": "run1/run2 标签"},
  "samples": [{"id": "...", "tier": "A|B|C", "category": "code", "prompt": "...",
    "outputs": [{"text": "...", "tokens": [...], "logprobs": [...],
                 "top_logprobs": [[{"token": "...", "logprob": -0.1}, ...], ...]}]}]
}
```

- `outputs` 通常 1 个（greedy/KL）；temp/top-p 可为 N 个（`--reps`）。
- 采集配置：greedy 用 `top_logprobs=1`（与 quality_gate 包络同口径）；分布门用 `top_logprobs=10`（KL 支持）。

---

## 3. 参考集定义

### 3.1 Tier 结构与数量

| Tier | 用途 | 数量 | 采集配置 | 门 |
|---|---|---|---|---|
| **A** | greedy 硬门（与 `quality_gate.py` 4 prompt 逐字一致） | 4 | temp=0, top_lp=1 | G1 |
| **B** | 分布度量（KL / 困惑度 / 温度 / top-p） | 24 | temp=0, top_lp=10 | G2-G5 |
| **C** | 长上下文 64K（needle ×2 + 尾部指令 ×1） | 3 | temp=0, top_lp=10 | G7 |
| 合计 | | **31** | | |

### 3.2 Tier B 分布（24 条）

| 类别 | 数量 | 样本 id | 设计意图 |
|---|---|---|---|
| code | 5 | code_quick, code_sql, code_async, code_regex, code_heap | 结构化输出对 logits 排序敏感（另 Tier A 的 code_fib 复用为分布样本） |
| summary | 5 | sum_climate, sum_news, sum_meeting, sum_rewrite, sum_spec | 长文本分布 |
| multiturn | 4 | mt_trip, mt_code, mt_tool, mt_math | 多轮/tool-call 形态 |
| json | 5 | json_profile, json_products, json_entities, json_toolcall, json_config | JSON 结构化，格式敏感 |
| reason | 3 | re_word, re_logic, re_arith | 推理（仅分布度量，不进逐字门） |
| zh | 2 | zh_sum, zh_json | 中文（仅分布度量；reason/zh 已知运行级非确定，不用于逐字硬门） |

> 注：code_fib 同时是 Tier A 的 `code_fib`（greedy 门）与 Tier B 的分布样本（同一 prompt 两种采集配置），跨 Tier 复用。

### 3.3 Tier C 长上下文（3 条，64K 档）

| 样本 | 类型 | 设计 |
|---|---|---|
| needle_25 | needle | 64K filler，needle 植入 25% 深度，问句取回 |
| needle_75 | needle | 同上，75% 深度 |
| tail_embed | tail | 64K filler 后直接尾部指令（要求输出 ORANGE）——测"长上下文末端指令跟随" |

生成规则（`reference_set_builder.py build`）：无重复 filler 句循环至 ~59K 词（≈64K token 目标），needle 句插入指定深度，尾部附问句/指令。token 数为估算值，实际以窗口采集为准【待窗口验证】。

### 3.4 扩展至 200-500 条（KL 定标最终规模）

fp8-quality-impact §3.2 建议参考集 200-500 条以获得稳定 KL 统计。v1 核心集 31 条用于**门限定标阶段**（噪声底 + 首轮 FP8 判定）；若首轮 KL 统计方差过大（样本级 KL 离散度高），按模板变体扩展（不同数字/实体/主题，`variants` 字段），不新增采集形态负担。**先测 31 条噪声底，再决定是否扩样**——避免无谓窗口占用。

---

## 4. BF16 噪声底测量协议

### 4.1 协议步骤（GPU 窗口）

1. 确认目标部署 = LuZ0.3.1 生产形态（W4A4=2 / SHARED=1 / thr4096 / util 0.82 / MTP n7 / FI 0.6.16 / CUMEM=0）或克隆镜像 `LuZ0.3.1-bench-20260823`。
2. `reference_set_collector.py collect --quant bf16 --config dist --tag run1`（Tier A+B，top_lp=10）。
3. **背靠背立即**再跑 `--tag run2`（同配置、同 draft 配置，中间不做任何环境改动）。
4. `quality_gate_noise_floor.py analyze run1 run2 --out assets/noise_floor.json`。
5. 产出噪声底表 + 推荐门限（KL 2-3× / PPL Δ 与重标定标记）。

> 前提（与 fp8-quality-impact §5.4 一致）：A/B 窗口必须先纳入 CUMEM_HOST_ENABLE=0 + 重测协议，否则测量噪声 ±8-13% 淹没 FP8 收益；噪声底测量同样遵守。

### 4.2 指标定义

- **受限 top-k 支持 KL**：每 token 位置对两 run 的 top-k logprob 行取公共 token 集，各自重归一后算 `KL(p1‖p2)`，样本内平均 → 样本 KL；两方向（A‖B、B‖A）都算，**噪声底取大方向**。
- **困惑度**：`PPL = exp(-mean(logprob))`（top-1 采样 token 口径）；样本 `Δ = |PPL_b − PPL_a|`。
- **logprob 漂移**：与 quality_gate 包络同口径（sum_drift_pct / mean_abs_diff / max_abs_diff）。
- **聚合**：按 token 数加权平均。

> 诚实声明：受限 top-k 支持 KL 低估全词表 KL（只覆盖 top-k 质量）。`top_logprobs=10` 覆盖绝大多数概率质量，对门判定的**相对**比较（FP8 vs BF16）仍有效——门限建立在同一估计量上。若实测分布尾部长尾严重，可提 top_logprobs=20 并重测噪声底【待窗口验证】。

### 4.3 脚本

```bash
python3 quality_gate_noise_floor.py analyze runs/run_bf16_dist_run1.json \
    runs/run_bf16_dist_run2.json --out assets/noise_floor.json
# 输出 aggregate{kl_floor, ppl_floor, lp_mean_abs_diff} + thresholds{kl_gate, ppl_delta_gate, recalibrated}
```

---

## 5. KL 门实现（G2/G3）

### 5.1 判据

- **G2 KL 门**：`加权 KL(p_bf16‖p_fp8) < 3×noise_floor.kl_floor`（`--kl-multiplier` 可调，默认 3）。
- **G3 困惑度门**：`加权 ΔPPL = PPL_fp8 − PPL_bf16 ≤ ppl_delta_gate`，其中 `ppl_delta_gate = max(0.05, 3×noise_floor.ppl_floor)`。
- 零底保护：KL 底实测为 0（确定性一致）时门限取最小下限 1e-4，避免任何非零 KL 误报。
- 综合 PASS = G2 且 G3；FAIL 时列出超限样本（outliers）供热点定位。

### 5.2 重标定逻辑（写入协议）

```
if 3 × noise_floor.ppl_floor > 0.05:
    ppl_delta_gate = 3 × noise_floor.ppl_floor   # 按噪声底重标定
    recalibrated = True                            # 判定报告标记 + 设计文档 §10 登记
else:
    ppl_delta_gate = 0.05                          # 固定门限
```

- 若噪声底本身严重超预期（KL > 0.5 nats/token 或困惑度跨次 Δ > 0.05），按 fp8-quality-impact §5.5 触发**门设计重审**（降级到 JS 散度或 token 集重叠度量）——先于 lm_head F2 解决。
- 重标定决策表见 §10。

### 5.3 脚本

```bash
python3 kl_gate.py gate --baseline runs/run_bf16_dist_run1.json \
    --candidate runs/run_fp8_dist.json \
    --noise-floor assets/noise_floor.json --out assets/kl_verdict.json
# 退出码 0=PASS 1=FAIL；输出 aggregate + gates + outliers
python3 kl_gate.py ppl --baseline ... --candidate ...   # 仅困惑度
```

---

## 6. greedy 4/4 基线固化（G1 资产）

### 6.1 资产路径

`_fp8_qg_toolchain/assets/golden-bf16-greedy-<UTC>.json` + `golden-bf16-greedy-latest.json`（schema `fp8-qg-golden/1`，与 quality_gate reference 格式兼容）。

### 6.2 子命令

| 子命令 | GPU? | 用途 |
|---|---|---|
| `capture` | 需要 | 在线采集 BF16 greedy 4 prompt → golden 资产 |
| `import-snapshot --from <quality_gate reference-latest.json>` | 否 | **换基座前已 capture 过则直接导入既有快照**，免重复采集 |
| `compare --candidate <fp8 run 或 reference json>` | 否 | 候选 vs golden：exact 4/4；不一致时包络 top-1 sum drift ≤1% 兜底 |
| `list` | 否 | 列出 golden 资产 |

### 6.3 与 quality_gate.py 的关系

- 同 prompt（`reference_set_builder validate` 强制逐字对齐）、同包络判据。
- **生产 compare 仍以 `quality_gate.py` 为唯一入口**；`greedy_baseline.py` 是工具链侧的资产固化/离线比对（候选可以是 run 文件而非在线 API），两者结果应一致。

---

## 7. 温度采样 / top-p 抽验（G4/G5）

### 7.1 判据（参数化，默认值）

| 指标 | 判据 | 默认 |
|---|---|---|
| logprob sum drift % | 配对输出均值 \|.\| ≤ drift_pct_max | 1.0 |
| top-k 候选集重叠率 | ≥ overlap_min（k=采集 top_logprobs） | 0.90 |
| distinct-n 多样性 | ratio = cand/base ≥ distinct_ratio_min | 0.90 |

- 覆盖配置：temp∈{0.6,0.9} × top-p∈{1.0,0.9}（top-p 0.9 即 G5 抽验）。
- 重叠率比较的是**分布支持**而非逐字一致性（采样随机），reps ≥5 统计才稳。

### 7.2 脚本

```bash
# GPU 窗口：BF16 与 FP8 各采一遍（每 config 一个 run 文件）
VLLM_API_KEY=... python3 temp_top_p_gate.py collect --quant bf16 --temperature 0.6 --top-p 1.0 --reps 5
VLLM_API_KEY=... python3 temp_top_p_gate.py collect --quant fp8  --temperature 0.6 --top-p 1.0 --reps 5
# 零 GPU 分析
python3 temp_top_p_gate.py analyze --baseline runs/run_bf16_t0.6_p1.0.json \
    --candidate runs/run_fp8_t0.6_p1.0.json --out assets/temp_top_p_verdict.json
```

---

## 8. 工具链使用说明（F2/F4 调用顺序与判定流程）

### 8.1 前置（零 GPU，已就绪）

```bash
cd deliverables/engineering-assurance/_fp8_qg_toolchain
python3 reference_set_builder.py validate   # 对齐校验
python3 reference_set_builder.py build      # 生成 64K prompt（needle/tail）
python3 selftest.py                        # 全绿（零 GPU 逻辑自检）
```

### 8.2 F2（适配器 + golden，GPU 容器）

1. `greedy_baseline.py import-snapshot --from <INSTALL_DIR>/backup/quality-gate/reference-latest.json`（或 `capture`）。
2. BF16 噪声底采集（§4.1，run1/run2）→ `noise_floor.json`（**门限定标产出**）。
3. F2 阶段可选先跑 **CPU 参考集 KL 初值**（BF16 vs FP8 离线 dequant 对比，不占 GPU）——fp8-quality-impact §4.2 建议项。

### 8.3 F4（窗口 A/B）

按序执行，任一 FAIL 即停：

```
G1  greedy 4/4       quality_gate.py compare（生产入口）/ greedy_baseline.py compare（资产侧）
G2  KL 门            kl_gate.py gate --noise-floor assets/noise_floor.json
G3  困惑度门          （同 kl_gate.py，输出内）
G4  温度采样          temp_top_p_gate.py collect/analyze（temp 0.6/0.9）
G5  top-p 抽验        temp_top_p_gate.py collect/analyze（top-p 0.9）
G6  MTP 接受率        de_bench.py（现成，±3%）
G7  长上下文 64K      Tier C 采集 + needle/tail 判定（3/3）
G8  off 等价          L1（不在本工具链）
```

### 8.4 判定决策树

```
noise_floor 产出（BF16 run1 vs run2）
  ├─ recalibrated=False → G2 门限=3×kl_floor，G3 门限=0.05
  ├─ recalibrated=True  → G2 门限=3×kl_floor，G3 门限=3×ppl_floor（登记 §10）
  └─ 噪声底严重超预期（KL>0.5 或 ΔPPL>0.05）→ 门设计重审，先于 lm_head F2
FP8 candidate
  ├─ G1 exact 4/4 或 envelope≤1% → 过
  ├─ G2 kl<门限 且 G3 ΔPPL≤门限 → 过
  ├─ G4/G5 overlap≥0.90 且 drift≤1% 且 distinct 不降 → 过
  └─ 全过 → F4 质量门 PASS
```

### 8.5 集成方式（新增还是独立）

**独立新增**：全部脚本独立于 `quality_gate.py`，不修改现役代码。共享点仅 prompt 与 logprob 口径（由 builder validate 强制）。生产 compare 唯一入口仍是 `quality_gate.py`；工具链侧脚本用于（a）资产固化（b）离线比对（c）分布门（quality_gate 不含 KL/PPL/temp/top-p）。

---

## 9. 零 GPU 已就绪 vs GPU 窗口待办

### 9.1 零 GPU 已就绪（已自检通过）

| 项 | 验证状态 |
|---|---|
| `reference_set.json`（31 条参考集定义） | validate OK |
| `reference_set_builder.py` validate / build / stats | selftest OK |
| `run_common.py`（run schema / KL / PPL / 漂移数学） | selftest OK |
| `quality_gate_noise_floor.py` analyze（噪声底 + 门限 + 重标定） | selftest OK（含 PASS/FAIL 场景） |
| `kl_gate.py` gate / ppl（KL 门 + 困惑度门 + 重标定） | selftest OK（含 PASS/FAIL 场景） |
| `greedy_baseline.py` compare / import-snapshot / list | selftest OK（exact + envelope 场景） |
| `temp_top_p_gate.py` analyze | selftest OK |
| `selftest.py` 全量自检 | **ALL PASS** |

### 9.2 GPU 窗口待办

| 项 | 说明 | 前置 |
|---|---|---|
| BF16 噪声底采集（run1/run2） | `reference_set_collector.py collect --config dist` | 窗口 + CUMEM=0 |
| 噪声底门限定标 | `noise_floor.json` 实际值；若 ΔPPL 噪声底 >0.05 触发重标定 | 上项 |
| greedy golden 采集/导入 | `greedy_baseline.py capture` 或 import 既有快照 | 窗口（import 可不占） |
| FP8 采集 | `collector collect --quant fp8` | FP8 镜像就位 |
| G2/G3/G4/G5 判定 | `kl_gate.py` / `temp_top_p_gate.py` 对 FP8 run | FP8 采集 |
| Tier C 64K 采集 | 3 条长上下文（耗时长，单独 collect 调用） | 窗口 |
| 参考集规模定标 | 31 条统计是否足够，不足则扩展 200-500 条 | 首轮噪声底 |

---

## 10. 阈值与重标定决策表

| 门 | 门限 | 默认 | 重标定触发 | 重标定后 |
|---|---|---|---|---|
| G2 KL | `3 × noise_floor.kl_floor` | 3 | 无 | —（参数化 `--kl-multiplier`） |
| G3 困惑度 | `max(0.05, 3 × noise_floor.ppl_floor)` | 0.05 | `3×ppl_floor > 0.05` | 门限 = `3×ppl_floor`，`recalibrated=True` |
| G4 logprob drift | `drift_pct_max` | 1.0% | 无 | — |
| G4/G5 候选集重叠 | `overlap_min` | 0.90 | 无 | — |
| G4 distinct-n | `distinct_ratio_min` | 0.90 | 无 | — |

> 门限定标记录应回写 `assets/noise_floor.json`（阈值即该文件内 `thresholds`），并在此表登记人工裁定（date / 值 / 裁定人）。

---

## 11. ADR 摘要

```markdown
# ADR-2026-08-24-001: FP8 质量门工具链形态
**状态**: Accepted（零 GPU 脚本已落地，自检全绿；门限待 GPU 窗口定标）
**背景**: lm_head-FP8 需要 G1-G7 质量门；KL/困惑度门需要参考集 + BF16 噪声底，无需校准。
**选项**:
  A. 扩展 quality_gate.py 内联实现   → 复杂度 Med；污染生产唯一入口；失败
  B. 独立脚本 + 共享 prompt/口径      → 复杂度 Low；quality_gate.py 不动；builder validate 防漂移 ✅
  C. 先 GPU 再补脚本                → 违反"工具链先行是硬前置"；失败
**决策**: B。独立 `_fp8_qg_toolchain/`，7 脚本 + 1 参考集；门限先测噪声底再定。
**影响**: 变容易 = 零 GPU 落地、可离线复现、shared-FP8 F4 复用；需重新审视 = 参考集规模、
top_logprobs 是否需提至 20（视分布尾长）、噪声底严重超预期时的门设计重审。
```

---

## 12. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| 【实测-源码】 | quality_gate.py 4 prompt + 包络 ≤1% + own_stable；reason/zh 已除名；LuZ0.3.1 形态（W4A4=2/SHARED=1/thr4096/util0.82/MTP n7/FI0.6.16/CUMEM=0）；克隆镜像 LuZ0.3.1-bench-20260823 在位 |
| 【实测】 | reason/zh 近平票翻转（运行级非确定）；routed W4A4 logprob 级 0.36-0.41%；测量噪声 ±8-13%；quality_gate 参考快照存于 <INSTALL_DIR>/backup/quality-gate/ |
| 【推算】 | BF16 vs BF16 KL 噪声底量级 0.01-0.1 nats/token；困惑度 Δ 噪声底大概率 <0.05；参考集 31 条首轮统计或需扩展 200-500 条 |
| 【待窗口验证】 | BF16 噪声底实测值；KL/困惑度门限定标；参考集规模；64K 长上下文采集；G2-G5 对 FP8 的正式判定 |

**诚实声明**：
1. 受限 top-k 支持 KL 是估计量（低估全词表 KL），门限建立在同一估计量上的相对比较有效；若需绝对口径需 top_logprobs 提升 + 重测。
2. 64K 长上下文 prompt 的 token 数为估算值（~59K 词 ≈ 64K token），实际以窗口采集为准。
3. 本工具链只做质量门；性能门（PR/DE/KV）、off 等价、MTP 接受率走既有门，不在本交付范围。
4. 所有门限定标值以 GPU 窗口实测 `noise_floor.json` 为准，本设计文档给出的是协议与默认值。

---

## 13. 引用索引

- `fp8-quality-impact-2026-08-23.md`（G1-G8 门设计、噪声底协议、重标定逻辑）
- `lmhead-fp8-project-2026-08-23.md`（F0-F5、工具链先行硬前置）
- `tmp/quality_gate.py`（现役生产贪心质量门，固化版 LuZ0.3.1 口径）
- `bench-regression-attribution-2026-08-23.md`（KL 门建议、reason/zh 假阳性史）
- `upstream-check-perf-ceiling-2026-08-23.md`（P2 门、KL 门阈值先例）
- 工具链落地：`_fp8_qg_toolchain/`（本交付）

*本报告由工程保障团队（系统架构师 architect-1）生成；零 GPU 设计 + 脚本已就绪并自检全绿；噪声底实测与门限定标排入 GPU 窗口。*
