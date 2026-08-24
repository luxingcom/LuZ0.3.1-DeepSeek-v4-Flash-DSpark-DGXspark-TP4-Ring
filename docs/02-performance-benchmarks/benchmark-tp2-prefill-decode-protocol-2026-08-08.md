# TP2 双组 LLM 基准测试协议定稿（纯 prefill / 纯 decode 解耦）

**日期**：2026-08-08
**负责**：Tessa（测试专家）· 工程保障团队
**适用**：组 A（58+60，已就绪）/ 组 B（55+59，LLM TP2 部署后）
**脚本**：`bench_prefill_decode_async.py`（asyncio wave 真并发版，本协议配套）

---

## 1. 目标与口径（用户要求）

- **不做 CTX 聚合指标**：不得产出 `total_tokens / total_time` 这类把 prefill 与 decode 混在一个分母的混合 t/s（那会随上下文长度稀释、无法回答"纯解码多快"）。
- **只测两个解耦速度**：
  - **纯预填充速度 prefill t/s**（吃 prompt 阶段）
  - **纯解码速度 decode t/s**（出 token 阶段）
- 两者单请求同出、互不污染（见 §3 分离测量法）。
- 监控面板同步改为纯 decode / 纯 prefill 两个面板（另见 Grafana 修订交付）。

## 2. 测试矩阵

| 维度 | 取值 | 说明 |
|---|---|---|
| 组 | A（58+60）、B（55+59） | 各跑一遍同矩阵 |
| 并发 | 1 / 3 / 5 | asyncio 真并发，每波 conc 个在飞 |
| 上下文 | 512 / 4k / 16k / 64k / 128k | 128k < 600k max-model-len 上限 ✓ |
| 任务 | coding / json / prose | 三类代表性结构 |
| 轮数 | 3 波 × conc 并发 | p50 取全部成功请求 |
| 温度 | 0.6 | 与既有基线口径一致 |

- 可选极限档单点：ctx=524288（600k 附近），`--limit-check`，仅 conc=1。
- **执行顺序**：先小 ctx 后大 ctx（脚本已 `sorted(ctxs)`），先 conc=1 探底再上并发，便于早期发现错误、控制风险。

## 3. 分离测量法（解耦核心）

单请求同时测得两值（同一流式响应）：

```
prefill_tps = prompt_tokens / TTFT
decode_tps  = (completion_tokens - 1) / (total - TTFT)
```

- `TTFT` = 首个含 content 的 SSE chunk 到达时刻（= 该请求 prefill 完成时刻）。
- decode 分子减 1：排除 TTFT 对应的那个首 token，避免首 token 被 prefill/decode 双重计数。
- **不做 agg t/s 主指标**；批内聚合视角（wave_agg：Σprompt/max(TTFT)、Σ(ct-1)/(max(total)-max(ttft))）仅作辅助字段 `agg_*_p50`，用于与监控面板 rate() 交叉验证。

## 4. 任务模板（3 类，固定）

| 任务 | 结构特征 | 输出 max_tokens |
|---|---|---|
| coding | 带函数签名/约束的 Python 实现 + pytest | 512 |
| json | 固定 schema（8 名员工，nested arrays + enum + ISO date） | 512 |
| prose | 散文续写（~300 字），禁止总结 | 256 |

模板常量在脚本 `TASK_TEMPLATES` / `TASK_MAX_TOKENS`，三任务生成长度不同属设计（表征三类负载）。

## 5. 随机前缀强制（防 prefix-cache）

- 每请求填充文本全部 `uuid4` hex 随机（`rand_unit()`），**同档请求不共享前缀**，杜绝 prefix-cache 假象。
- 填充长度经校准探针（tokens/unit，`CALIB_UNITS=100`）逼近目标 prompt_tokens（系数 0.88），**实际值以 usage.prompt_tokens 为准**（±5% 内即可；t/s 已按实际归一化）。
- 指令收尾：填充噪声在前、真实任务指令在末尾，输出体现任务类型。

## 6. 并发语义（wave 模型，修正旧脚本缺陷）

- 旧版 `bench_prefill_decode.py` 只提交 rounds 个任务（max_workers=conc），rounds(3) < conc(5) 时在飞仅 3 个 → **并发档位测不满**。
- 新版：**每组合 = 3 波 × conc 并发在飞**；每波 `asyncio.gather(conc 个请求)` 同时发出。concurrency=5 即真 5 并发。
- 引擎：`--engine asyncio`（aiohttp，默认）/ `--engine threads`（requests，经 asyncio.to_thread 编排，回退）。
- 单组合请求量 = 3 波 × conc：conc=1→3、conc=3→9、conc=5→15。

## 7. 输出格式

- **CSV 逐轮明细**：`rows_{group}.csv`（group/ctx/task/concurrency/wave/ok/prompt_tokens/completion_tokens/ttft_s/total_s/prefill_tps/decode_tps/err）。
- **JSON summary**：`summary_{group}.json`，每组合含：
  - `p50_prefill_tps` / `p50_decode_tps`（主指标）+ `p50_ttft_s` / `p50_total_s` + min/max（≥3 请求时）
  - `agg_prefill_tps_p50` / `agg_decode_tps_p50`（批内聚合视角，辅助）
  - `rounds_ok` / `requests_total` / `errors` / `err_samples`
- **控制台表**：ctx × task × conc | prefill_tps(p50) | decode_tps(p50) | ttft | total。
- 汇报口径：每组组合 prefill_tps(p50) + decode_tps(p50)，另给每任务维度拆分（task 是组合字段，天然可分）。

## 8. 组 B（55+59）前置检查清单

1. **LLM TP2 部署**：anemll + mp executor，master `<NODE_IP>:25055`（= .55，rank0），node-rank 0/1；**head 端点 = .55:8001**。
2. **权重**：.55 156G ✓、.59 156G ✓（已确认）。
3. **互联**：55→59 免密已建（别名 gx10-59）；RoCE 200G RTT ~0.8ms；启动须 head-first 严格时序 + `NCCL_IB_GID_INDEX=2`（如遇 GID3 空，参照 .60 坑位）。
4. **NCCL 层 sanity**：`--sanity-log <head启动日志>` 判定 rank0/1 真正互连（防 HTTP 200 但 TP 未配对）；FAIL 即中止。
5. **/v1/models** 确认 served model 名（脚本自动核对，不在列表自动选第一个）。
6. **Grafana 抓取**：当前只抓 `<NODE_IP>:8001`，组 B 期间需 SRE 加 scrape target `<NODE_IP>:8001`（`node=head-55`），否则面板无组 B 数据。

## 9. Embed 卸载规则（用户要求：测哪组卸哪组）

- **测组 A（58+60）**：卸载 .58/.60 的 embed（anemll 8022 等，.60 曾因 LLM head 107G + embed 12G CUDA OOM），由对侧组 .55/.59 提供（litellm 池自动 failover）。
- **测组 B（55+59）**：卸载 .55/.59 的 embed，由 .58/.60 提供。
- 目的：释放 GPU 内存、保证 LLM 纯负载无 embed 争抢，得到可对比的纯 LLM prefill/decode 速度。
- 验证：`docker ps` 确认 embed 容器停 + litellm health 通过（对侧提供 200）。

## 10. 时长估算与验收

- 单组矩阵：3 conc × 5 ctx × 3 task × 3 波 × conc 请求；128k/64k 大 ctx 主导耗时，**单组约 1.5~3h**（A 组先跑，B 组部署后跑）。
- 验收判定：
  1. 随机前缀保证（无 cache 污染）
  2. usage.prompt_tokens 与目标偏差 ≤5%（偏差只影响执行时长，不影响归一化 t/s）
  3. 每组合 `rounds_ok ≥ 2/3` 波全部请求成功（conc=5 波内 ≥10/15 ok），错误样本入 CSV
  4. TP2 sanity：NCCL init 判定 PASS
  5. decode_tps 与监控面板 `rate(vllm:generation_tokens_total)` 交叉验证偏差 <20%（组 A 期间）
  6. 不产出任何 CTX 聚合混合指标

## 11. 命令块

```bash
# 组 A（58+60，head .60:8001；先卸 .58/.60 embed）
python3 bench_prefill_decode_async.py --group A \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <API_KEY>-<KEY> --model deepseek-v4-flash-0731 \
  --concurrency 1,3,5 --ctx 512,4096,16384,65536,131072 \
  --tasks coding,json,prose --rounds 3 --engine asyncio --out ./results_A

# 组 B（55+59，head .55:8001；先卸 .55/.59 embed；LLM TP2 部署后）
python3 bench_prefill_decode_async.py --group B \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <API_KEY>-<KEY> --model deepseek-v4-flash-0731 \
  --concurrency 1,3,5 --ctx 512,4096,16384,65536,131072 \
  --tasks coding,json,prose --rounds 3 --engine asyncio --out ./results_B \
  --sanity-log /home/<USER>/<head启动日志>.log

# 最小快验（5 分钟冒烟，embed 卸载后先跑）
python3 bench_prefill_decode_async.py --group A --endpoint http://<NODE_IP>:8001/v1 \
  --key <API_KEY>-<KEY> --concurrency 1,5 --ctx 512,131072 --tasks coding \
  --rounds 3 --out ./results_smoke

# 依赖
pip install aiohttp   # --engine asyncio（默认）
```
