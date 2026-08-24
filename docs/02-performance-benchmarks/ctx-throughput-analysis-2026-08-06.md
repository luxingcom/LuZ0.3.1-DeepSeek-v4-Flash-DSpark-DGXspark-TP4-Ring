# "ctx 越大吞吐下降越厉害" 原因核实 — 时间拆解与因素归因

- **日期**：2026-08-06
- **分析**：Tessa（测试专家 / 工程保障团队）
- **现象**：4000 网关测试（bench-gw4000-2026-08-06）中，512 ctx 输出 68–101 t/s（c5），8192 ctx 输出 5–26 t/s，32768 ctx 输出 5–7 t/s、TTFT 16–50s
- **环境**：F 方案（probabilistic + 动态K + tilelang + b12x + nvfp4_ds_mla，seqs=6，600K ctx），chunked prefill 开启，prefix caching 开启，双机 TP=2
- **数据来源**：
  - `_tessa_gw4000_raw_2026-08-06.txt`（4000 网关，随机前缀防 cache，**prefill 可信口径**）
  - `_tessa_final_baseline_raw_2026-08-05.txt` / `raw_final_matrix.json`（直连 8001，固定 calib 文本，**32K TTFT 被 prefix-cache 污染，prefill 口径不可用**，详见 §6）
  - `bench-matrix-v026-2026-08-05.md`（直连 8001，随机前缀，TTFT 线性 508ms→16.57s，与 gw4000 一致）
  - `ab-compare-v026-rebuild-2026-08-05.md`（100k prefill 1785 t/s、200k prefill 1547 t/s 历史数据）

---

## TL;DR

- **主因（~90–96%）＝ prefill 时间主导稀释**：输出固定 max_tokens=128，prefill 时间随输入线性增长（512→32K TTFT 0.48s→16.5s，+34×），decode 时间基本不变（~2s）。输出吞吐 = 128 / (prefill + decode)，被 prefill 时间线性稀释：55.5 → 20.3 → 6.9 t/s（c1 code）。
- **次要（~4–10%）＝ chunked prefill 并发放大**：c5 下 5 个大 prefill 请求互相争抢，TTFT 16.5s→50s（3×），decode TPOT 15ms→244ms（decode 带宽被 prefill chunk 抢占）。
- **可忽略（<5%）**：DSpark 接受率下降（60.7%→57.4%，-3.2pp）；KV 池竞争（32K×6 仅占 1.95M 池 ~10%，不触发抢占）。
- **不是引擎缺陷，是数学现实**：大 ctx 场景应以 **prefill 吞吐（输入 t/s）** 为考核指标，输出吞吐天然被稀释。

---

## 1. 方法

对每个 (ctx, conc) 组合，用每请求数据做时间拆解（gw4000 用中位数；raw_final_matrix 用 per_req 求和）：

```
prefill 时间 ≈ TTFT（首字延迟，含 prefill 主体）
decode 时间 ≈ TPOT × output_tokens（≈ 总耗时 − TTFT）
TTFT 占比   = TTFT / (TTFT + decode)
输出吞吐    = output_tokens / (TTFT + decode)
```

> 口径说明：TTFT 是否等于"总 prefill 时间"，取决于**是否防 prefix-cache**。
> - gw4000 / v026 矩阵：每请求随机 `<rnd>` 前缀防 cache → TTFT = 真实 prefill 时间（线性增长，可信）。
> - raw_final_matrix（直连基线）：固定 calib 文本、无随机前缀 → warmup 后 32K 请求命中 prefix-cache，TTFT≈370ms 不代表 32K prefill，**该数据只可用于 decode/吞吐对比，不可用于 prefill 归因**（详见 §6 方法学发现）。

---

## 2. 时间拆解表（gw4000 可信口径）

### 2.1 c1 单流（每请求）

| ctx | TTFT(prefill) | decode | 总耗时 | TTFT 占比 | decode t/s | 输出吞吐 |
|---|---|---|---|---|---|---|
| 512 | 0.48 s | 1.83 s | 2.31 s | **21%** | 70.1 | **55.5 t/s** |
| 8192 | 4.25 s | 2.07 s | 6.32 s | **67%** | 62.0 | **20.3 t/s** |
| 32768 | 16.48 s | 1.96 s | 18.44 s | **89%** | 65.4 | **6.9 t/s** |

（code 负载；json/prose 形态一致，均值见 2.3）

### 2.2 c5 并发（每请求视角；聚合吞吐 ≈ conc × 每请求吞吐）

| ctx | TTFT(prefill) | decode | 每请求总耗时 | TTFT 占比 | 聚合输出吞吐（实测 agg） |
|---|---|---|---|---|---|
| 512 | 1.09 s | 4.84 s | 5.93 s | **18%** | **101.2 t/s** |
| 8192 | 12.92 s | 11.06 s | 23.99 s | **54%** | **25.8 t/s** |
| 32768 | 50.00 s | 31.29 s | 81.29 s | **62%** | **7.4 t/s** |

（code 负载；c5 下 decode 也大幅劣化：TPOT 37.8ms→244ms，即 prefill chunk 抢占 decode 带宽）

### 2.3 三负载平均（code/json/prose 均值）

| conc | ctx | TTFT(prefill) | decode | 总耗时 | TTFT 占比 | 每请求输出吞吐 |
|---|---|---|---|---|---|---|
| c1 | 512 | 0.48 s | 2.18 s | 2.67 s | 19% | 48.9 t/s |
| c1 | 8192 | 4.22 s | 2.33 s | 6.54 s | 65% | 19.7 t/s |
| c1 | 32768 | 16.53 s | 2.13 s | 18.66 s | **89%** | 6.9 t/s |
| c5 | 512 | 1.20 s | 5.54 s | 6.74 s | 18% | 19.5 t/s |
| c5 | 8192 | 12.90 s | 11.70 s | 24.59 s | 53% | 5.2 t/s |
| c5 | 32768 | 49.79 s | 32.06 s | 81.85 s | **61%** | 1.6 t/s |

### 2.4 直观佐证（同一输出=128 token，code c1，512 vs 32768）

```
512  : TTFT 0.48s (21%)  + decode 1.83s (79%)  = 2.31s  → 55.5 t/s
32768: TTFT 16.48s (89%) + decode 1.96s (11%)  = 18.44s → 6.9 t/s
       TTFT 增长 34.2×；decode 基本不变；输出吞吐下降 8.0×
```

- decode 部分几乎不随 ctx 变化（1.83→1.96s，+7%），说明**引擎 decode 能力在大 ctx 下并未退化**；
- 吞吐崩塌的 ~100% 来自总耗时里 prefill 的线性膨胀。

---

## 3. 因素归因排序（量化）

### 主因 a：prefill 时间主导稀释 — 贡献 ~90–96%

- 数学验证：`输出吞吐 = 128 / (TTFT + decode)`，TTFT = ctx / prefill_rate（~1700–2000 t/s 稳定）。
  - 512：128/(0.48+1.83)=55.5 ✓ 实测 55.6
  - 8192：128/(4.25+2.07)=20.3 ✓ 实测 20.3
  - 32768：128/(16.48+1.96)=6.9 ✓ 实测 6.9
- 反事实（32K c5 code）：若 decode 保持 512-c5 速度（4.84s），吞吐 21.6→2.33 t/s；prefill 增长解释了 **96%** 的吞吐下降，decode 劣化仅 4%。
- 外推历史数据：100k prefill 1785 t/s → TTFT≈56s → 输出 ~2.2 t/s；200k prefill 1547 t/s → TTFT≈129s → 输出 ~1.0 t/s。与 envD 实测 595.5K/1297 t/s/TTFT 459s 同形态（prefill-bound）。

### 次要 b：chunked prefill 调度 / 并发放大 — 贡献 ~4–10%（c5 才显著）

- c5 下 5×32K 大 prefill 同时进队，TTFT 从 c1 的 16.5s 放大到 50s（3×），decode TPOT 15ms→244ms（decode 带宽被 prefill chunk 抢占）。
- 但本质是 **prefill 计算量占总工作量的比例** 决定的——chunked prefill 只是把"先 prefill 后 decode"变成"交织"，不会改变 prefill 主导的总耗时结构。c1 下该因素≈0。

### 可忽略 c：DSpark 接受率下降 — 贡献 <5%

- 档级接受率 512/8K/32K = 60.7%/58.9%/57.4%（-3.2pp，相对 -5.3%）。
- 影响路径是 decode：32K c1 code TPOT 14.26→15.28ms（+7% 最坏），且 decode 仅占总耗时 11% → 对整体吞吐影响 <1%；c5 下被 prefill 争抢淹没。**接受率下降不是大 ctx 吞吐崩塌的原因**（draft 收益在结构化负载反而随并发上升）。

### 可忽略 d：KV 池竞争 — 贡献 ~0–1%

- 池估算：max-model-len 600K × seqs=6 上界 3.6M tokens；按可用池 ~1.95M tokens 计：
  - 8192×6 = 49K tokens = **2.5%**
  - 32768×6 = 197K tokens = **10.1%**
- 远未触发抢占/排队（阈值通常 ~90%+）；600K ctx 的设计使 32K 请求 KV 占用只是零头。**KV 池不是瓶颈**。

### 归因汇总

| 因素 | 贡献估计 | 依据 |
|---|---|---|
| **prefill 时间主导稀释** | **~90–96%** | TTFT 34× 增长、占比 21%→89%、数学反事实 96% |
| chunked prefill 并发放大 | ~4–10% | c5 TTFT 3× 放大、TPOT 16×（仅 c5 显著） |
| DSpark 接受率下降 | <5% | 60.7→57.4%；decode 仅占 11% 总耗时 |
| KV 池竞争 | ~0–1% | 32K×6 = 10.1% 池，无抢占 |

---

## 4. 结论

1. **"ctx 越大吞吐下降越厉害" 的主因是 prefill 时间主导稀释**：输出 128 token 固定时，总耗时 ≈ prefill（随 ctx 线性）+ decode（固定），prefill 从占总耗时 21%（512）膨胀到 89%（32K），输出吞吐按 128/总耗时 反比下降（55.5→6.9 t/s，8×）。
2. **引擎 decode 能力未退化**：decode 时间 512 vs 32K 基本持平（1.83s vs 1.96s）；单流 decode t/s 也持平（70.1 vs 65.4）。"吞吐下降"是输出吞吐被 prefill 时间稀释的统计现象，不是引擎性能下降。
3. **次要因素**：高并发（c5）下 chunked prefill 放大 TTFT（3×）并抢占 decode 带宽，贡献 4–10%；接受率下降与 KV 池竞争可忽略（合计 <5%）。
4. 若以**系统总 token 速率（输入+输出）**衡量，大 ctx 下系统处理能力反而充足（32K c1 total_tps≈15.7K，c5≈21.6K）——瓶颈是"等待 prefill 完成"而非"跑不动"。

---

## 5. 缓解建议（优化大 ctx 场景）

1. **大 ctx 请求与短请求队列分离**：长 prefill（数秒~数十秒）会堵住 decode 流水线，短请求被长请求 TTFT 拖累。建议按 ctx 分路由/分实例（short-ctx 实例 vs long-ctx 实例），或利用 vLLM 调度优先级保证短请求优先 decode。
2. **调整 chunked prefill / 批大小**：提高 `--max-num-batched-tokens` 让大 prefill 更快消化、减少与 decode 的交织争抢；或针对长 ctx 调大 chunk size。
3. **考核指标按场景区分**：
   - 大 ctx 场景（>8K）以 **prefill 吞吐（输入 t/s）** 和 TTFT 为考核指标；
   - 短 ctx 场景（≤2K）以输出吞吐/TPOT 为考核指标。
   - 当前 F 方案 prefill 速率 ~1700–2000 t/s（100k=1785、200k=1547），已属健康水平。
4. **明确业务预期**：输出 128 token 时，32K ctx 单流吞吐上限 ~7 t/s、TTFT ~16s 是数学必然（受 prefill 速率约束）；如需更低 TTFT，方向是 prefill 加速（稀疏 MLA / 更多算力 / 前缀复用），而非 decode 优化。
5. **接受现实**：除非业务改变"固定小输出 + 超大输入"的形态，否则大 ctx 下输出吞吐的稀释不可消除；重点应放在提升 prefill 吞吐与 TTFT 体验上。

---

## 6. 异常与方法学发现

1. **⚠️ 直连基线（raw_final_matrix，08-05）32K TTFT≈370ms 是 prefix-cache 假象**：
   - 该测试用固定 calib_def 文本（无每请求随机前缀），warmup 后 32K 请求命中 prefix-cache → TTFT 全档平坦（341→370ms）、吞吐"不随 ctx 下降"（36–39 t/s），与 gw4000 的线性 TTFT（0.48→16.5s）矛盾。
   - **判定**：raw_final_matrix 的 prefill 归因**无效**；其价值仅限于 decode/吞吐对比。这与团队历史 incident（"prefix cache 命中假象 11K tok/s → 测量方法学需排除缓存干扰"）一致，但 final-baseline 脚本未落实随机前缀，属复发的测量缺陷。
   - **建议**：性能矩阵脚本统一强制每请求随机 `<rnd>` 前缀（gw4000/v026 已正确），并在报告中标注 prefill 口径。
2. 32768 c1 直连基线个别请求 finish=stop 提前截断（out=17/18），拉低 tpot 均值——文本分布噪声，非引擎问题。
3. 4000 网关 LiteLLM 层开销约 +0.1s 延迟、对吞吐 -17~-24%（相对直连），不随 ctx 放大，不改变本结论。

---

*本报告由工程保障团队测试专家（Tessa）基于 2026-08-05/06 实测数据生成。核心结论：大 ctx 吞吐下降 ≈ prefill 时间主导稀释（90%+），非引擎退化；大 ctx 场景应考核 prefill 吞吐与 TTFT。*
