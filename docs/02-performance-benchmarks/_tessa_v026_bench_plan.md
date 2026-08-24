# v0.26 定制升级 · 双基准测试方案（社区基准 + Anemll 方案原基准）

**作者**：Tessa（测试专家）
**日期**：2026-08-04（设计阶段，未执行）
**被测对象**：vLLM v0.26.0 定制镜像 `ghcr.io/anemll/dspark-vllm-gx10:0.2.0-v026.0`（保留 7 项定制：sm_121a wrapper / nvfp4_ds_mla KV / dspark 投机 / flashinfer_b12x MoE / deep_gemm JIT / D-patch core.py / VLLM_* env）
**基线**：vLLM 0.25.2.dev0（Anemll v0.1.1），E-600k/0.80，双 DGX Spark TP=2
**对比口径**：与 v0.25 生产实测（2026-08-04 `_tessa_bench_raw_2026-08-04.txt`）**同脚本、同参数、同指标字段**逐项横向对比
**v0.26 方案更新（2026-08-04）**：新增「3.3 Greedy A/B 专项」——灰度验证 `draft_sample_method=greedy`（官方推荐）vs 生产 `probabilistic`。v0.26 容器默认以 greedy 启动；probabilistic 对照复用 v0.25 历史数据。

---

## 0. 前置检查（每组测试前必做）

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 引擎就绪 | `curl -s http://<head>:8001/v1/models` | HTTP 200，`max_model_len=600000`，served-model-name=deepseek-v4-flash-0731 |
| 网关就绪 | `curl -s http://<worker>:8003/v1/models` | HTTP 200，含 local-v4-flash / deepseek-v4-flash-0731 / deepseek-v4-flash |
| 7 项定制生效 | 引擎启动日志 grep：nvfp4_ds_mla / flashinfer_b12x / dspark / nvcc_wrapper / patch core.py | 关键项在日志中可见 |
| 双机一致性 | head/worker 两侧 `nvidia-smi --query-gpu=name,memory.used --format=csv` + vLLM 进程启动参数 diff | 两机 GPU 状态一致、无异常进程 |
| thinking 默认态 | 普通请求（不带 enable_thinking）检查 `reasoning` 字段 | 记录默认态（历史 08-04 为默认关） |
| API key | 直连用 `<API_KEY>-*`；网关用 `<API_KEY>-*` | 401 视为失败 |

**拓扑速查**：直连 = head <NODE_IP>:8001（内部 key）；网关 = worker <NODE_IP>:8003（客户端 key）。

---

## 1. 组 1：社区基准（GSM8K 200 题同子集）

### 1.1 子集定义（关键同口径保证）
- 数据源：`openai/gsm8k` test split，本地文件 `C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\gsm8k_test.jsonl`（SRE 服务器侧下载，1319 题，字段 `question`/`answer`）。
- **子集 = idx 0-199**（即文件前 200 条）。由 `gsm8k_eval_E800k.py` 的 `load_data(path, limit=200)` 天然实现（`items[:200]`）。历史 08-04 与历史 E-600k 同子集均为此 200 条。
- 复测时**禁止改行顺序**，`--limit 200` 即自动同子集。

### 1.2 Prompt 模板（8-shot CoT，脚本内 EXAMPLES 常量）
```
Question: <示例Q1>
Answer: <示例A1>

Question: <示例Q2>
Answer: <示例A2>
...
Question: <被测题>
Answer:
```
- 8 条示例固定（脚本 `EXAMPLES` 列表，Wei et al. 2022 标准 GSM8K few-shot），**不得改动**。

### 1.3 评估口径
- 请求参数：`temperature=0`、`max_tokens=1024`、`concurrency=1`（顺序执行，避免相互干扰）、8-shot。
- 判定：正则优先取最后一个 `#### 数字`，否则取文末最后一个数字；归一化（去逗号/空格/$）后数值匹配（整数精确 + 浮点容差 1e-6）。
- 指标：accuracy、err 数、median_ct、median_elapsed_s。

### 1.4 测试命令（经网关 8003，客户端 key）
```bash
python gsm8k_eval_E800k.py \
  --host <NODE_IP> --port 8003 \
  --model deepseek-v4-flash-0731 \
  --api-key <API_KEY>-<实际key> \
  --data C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\gsm8k_test.jsonl \
  --limit 200 --concurrency 1 \
  --out results_gsm8k_v026.jsonl \
  --label v026 --env E-600k-v026 \
  --baseline-acc 0.9850
```
> 注：脚本路径在 `C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\gsm8k_eval_E800k.py`。`--baseline-acc 0.9850` 触发 z 检验。

### 1.5 与历史对比表模板

| 指标 | v0.25 历史（同子集 0-199） | v0.26 实测 | 判定 |
|---|---|---|---|
| accuracy | 98.50%（197/200） | 待填 | delta ≥ -1.5pp 且 z 检验不显著 → 通过 |
| err | 0 | 待填 | 必须 0 |
| median_elapsed_s | 1.9 | 待填 | 记录（不设硬阈值） |
| 参考：历史全量 1319 | 96.66% | — | 仅参考，不强制复测全量 |

**判定**：acc ≥ 97.0%（-1.5pp 容差）且 err=0 → PASS；否则 FAIL 需复测确认。

---

## 2. 组 2：Anemll 方案原基准

### 2.1 单流 S1/S3/S5（直连 8001，内部 key）

**脚本**：`bench_smoke_E600k.py`（流式，`stream_options.include_usage`，随机 24 字符前缀防 prefix-cache）
**Prompt 构造**：`<rnd:xxxxxxxx>\n` + 中文段落重复（`CHARS_PER_TOKEN=1.85` 字符/token 校准）
**请求参数**：`temperature=0.7`、`max_tokens=<output>`

| 场景 | 命令参数（--input / --output） | v0.25 历史实测（08-04） | 期望输出字段 |
|---|---|---|---|
| S1 | 200 / 512 | total_tps=108.95（TTFT 362.3ms, decode 40.9） | TTFT_ms / decode_tps / total_tps |
| S3 | 2048 / 512 | total_tps=482.79（TTFT 1303.7ms, decode 28.05） | 同上 |
| S5 | 200 / 2048 | total_tps=93.5（TTFT 366.8ms, decode 41.7） | 同上 |

> ⚠️ 历史 08-04 数值为 **total_tps（t/s）**，非秒；team-lead 摘要中的 "108.95s" 实为 total_tps=108.95。对比表以 total_tps 为主指标，TTFT/decode 为次指标。
> 历史 08-04 测量：S1 5 次取中位、S3/S5 3 次取中位。**复测沿用同样 attempts**。

```bash
# S1
python bench_smoke_E600k.py --host <NODE_IP> --port 8001 \
  --model deepseek-v4-flash-0731 --api-key <API_KEY>-<实际key> \
  --input 200 --output 512 --retries 5
# S3
python bench_smoke_E600k.py --host <NODE_IP> --port 8001 \
  --model deepseek-v4-flash-0731 --api-key <API_KEY>-<实际key> \
  --input 2048 --output 512 --retries 3
# S5
python bench_smoke_E600k.py --host <NODE_IP> --port 8001 \
  --model deepseek-v4-flash-0731 --api-key <API_KEY>-<实际key> \
  --input 200 --output 2048 --retries 3
```

**判定**：error=0 硬约束；total_tps / TTFT / decode 三指标相对 v0.25 变化在 ±15% 内 → PASS；超过 ±15% 复核并标注。

### 2.2 并发 c1/c3/c5（2048→256，直连 8001）

**脚本**：`bench_concurrency_smoke_E600k.py`（N 并发 × M 轮，取中位 agg_tps）
**请求参数**：`--input 2048 --output 256 --rounds 3 --temperature 0.7`

| 场景 | --concurrency | v0.25 历史（08-04） | 期望输出 |
|---|---|---|---|
| c1 | 1 | agg_tps=26.88（历史 28.96） | agg_tps / rounds / errors |
| c3 | 3 | agg_tps=39.57（历史 42.16） | 同上 |
| c5 | 5 | agg_tps=32.82（rounds 波动大） | 同上 |

```bash
for c in 1 3 5; do
  python bench_concurrency_smoke_E600k.py --host <NODE_IP> --port 8001 \
    --model deepseek-v4-flash-0731 --api-key <API_KEY>-<实际key> \
    --concurrency $c --input 2048 --output 256 --rounds 3
done
```

**判定**：errors=0；agg_tps 中位与 v0.25 相比 **≥ -20% 容差**（c5 历史波动大，要求放宽到 -25%）；c1/c3 复测一次取中位消除首轮噪声（历史 08-04 即首测 13.45 噪声后复测 26.88）。

### 2.3 短 prompt 高并发 c10/c20（200→64，经网关 8003）

**脚本**：`bench_concurrency_smoke_E600k.py`（同脚本，参数不同）
**请求参数**：`--input 200 --output 64`、rounds=2（历史为 2 轮取中位）

| 场景 | --concurrency | v0.25 历史（08-04） | 期望输出 |
|---|---|---|---|
| c10 | 10 | agg=26.79 t/s（rounds 17.69/35.89） | agg_tps / errors |
| c20 | 20 | agg=31.99 t/s | agg_tps / errors |

```bash
for c in 10 20; do
  python bench_concurrency_smoke_E600k.py --host <NODE_IP> --port 8003 \
    --model deepseek-v4-flash-0731 --api-key <API_KEY>-<实际key> \
    --concurrency $c --input 200 --output 64 --rounds 2
done
```

**判定**：errors=0 硬约束；agg_tps ≥ v0.25 的 -20% 容差 → PASS。

### 2.4 长序列 200k / 100k（经网关 8003）

**脚本**：`tessa_longseq_20260804.py`
**输入构造**：`<rnd:24字符>\n` + 中文段落重复至 `input_k*1000` token（CHARS_PER_TOKEN=1.85），尾部加 `请用一句话总结以上内容。`
**请求参数**：`--max-tokens 64 --temperature 0.7`、`stream: true`、timeout=900s

| 场景 | --input-k | v0.25 历史（08-04） | 期望输出字段 |
|---|---|---|---|
| 200k | 200 | prompt_tokens≈200,893，prefill=1547.5 t/s，TTFT=129.8s，err=0 | status / prompt_tokens / prefill_tps / ttft_ms / total_tps |
| 100k | 100 | prompt_tokens≈100,462，prefill=1785.2 t/s，TTFT=56.3s，err=0 | 同上 |

```bash
python tessa_longseq_20260804.py --host <NODE_IP> --port 8003 \
  --model local-v4-flash --api-key <API_KEY>-<实际key> --input-k 200
python tessa_longseq_20260804.py --host <NODE_IP> --port 8003 \
  --model local-v4-flash --api-key <API_KEY>-<实际key> --input-k 100
```

**判定**：status=200 且 err=None 硬约束；prefill_tps 与 v0.25 相比 ≥ -10% 容差（prefill 是 v0.26 宣称增强点，若退化需重点分析）；TTFT 记录并对比。

### 2.5 工具调用（经网关 8003）

**脚本**：`tessa_toolcall_20260804.py`
**工具定义**（固定，不得改动）：
```json
{"type":"function","function":{"name":"get_weather","description":"查询指定城市的天气信息",
  "parameters":{"type":"object","properties":{"city":{"type":"string","description":"城市名"}},"required":["city"]}}}
```
**请求参数**：user content=`北京今天天气怎么样？请查询天气工具。`、`tool_choice=auto`、`max_tokens=256`、`temperature=0`

```bash
python tessa_toolcall_20260804.py --host <NODE_IP> --port 8003 \
  --model local-v4-flash --api-key <API_KEY>-<实际key>
```

**判定**：status=200、finish_reason=tool_calls、tool_calls[0].function.name=get_weather 且参数含 city=北京 → PASS。

### 2.6 思考字段（enable_thinking 触发，经网关 8003）

**方法**：发起带 `chat_template_kwargs: {"enable_thinking": true}` 的请求（参考 `.tessa_test/run_thinking.sh` th4），校验响应 message：
- `reasoning` 非空
- `reasoning_content` 非空
- `reasoning == reasoning_content`（方案 A 镜像在网关层保证）
- 对照组：直连 8001 同请求 → 仅 `reasoning` 非空（无镜像，预期）

```bash
curl -s --max-time 180 -H "Authorization: Bearer <API_KEY>-<key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Please solve step by step: a farmer has 17 sheep, all but 9 run away. How many are left?"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":true}}' \
  http://<NODE_IP>:8003/v1/chat/completions
```

**判定**：网关响应中 reasoning 与 reasoning_content 均非空且相等 → PASS；直连侧仅 reasoning 非空 → 符合预期。

---

## 3. 新增对比项（v0.26 增强验证）

### 3.1 fused_topk_bias 效果（MoE router 密集负载）
- **目标**：验证 v0.26 fused_topk_bias 优化（官方宣称 1.5-2x topk 内核）在定制路径下的收益。
- **方法**：构造 **router 密集 prompt**（内容多样、覆盖多领域的长 prompt，促使 MoE 激活更多 expert），测解码 TPOT 与 TTFT。
- **Prompt 模板**：`<rnd:24字符>\n` + 多领域中文段落混合（数学/代码/常识/翻译各 1/4），长度 2048 token；输出 512 token。**与 S3 的差异仅在 prompt 内容多样性**。
- **执行**：用 `bench_smoke_E600k.py` 同参数（--input 2048 --output 512 --retries 3）但需准备 router-dense 专用 prompt 变体（执行阶段写 `tessa_routerdense_2026xxxx.py`，复用 bench_smoke 的 SSE 计时逻辑）。
- **指标**：TTFT_ms、decode_tps（TPOT 逆）、total_tps。
- **对比**：v0.26（fused_topk_bias 开）vs v0.25 同 prompt（无 fused_topk_bias）。预期 TTFT/decode 提升；若 v0.26 退化 ≥ -10% 视为异常。

### 3.2 DSpark 接受率观测（若可行）
- **目标**：观测 v0.26 dspark 投机解码的 draft 接受率（acceptance rate），判断投机收益是否保持。
- **方法**：
  1. 优先走 vLLM `/metrics`（Prometheus）查询 spec_decode 相关指标（`vllm:spec_decode_*`，需在 v0.26 确认指标名；历史未采到）。
  2. 若指标不可用：在 c5 并发 3 轮期间采样引擎日志 /metrics 增量，或按社区参考值 0.673 做软对比。
- **判定**：接受率可观测 → 记录并对比社区参考（0.673）；不可观测 → 标注 N/A，降级为吞吐对比（2.2 节 c1/c3/c5 已覆盖）。

### 3.3 Greedy A/B 专项（draft_sample_method=greedy vs probabilistic）★ v0.26 灰度决策项
- **背景**：v0.26 灰度拟采用官方推荐 greedy 采样（GB300 配方 `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}`）；生产 v0.25 为 `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`。架构师判断 greedy 接受率通常更高，需实测验证差异后决策。
- **口径决策（隔离变量）**：A/B **仅切换 `draft_sample_method` 字段**，`num_speculative_tokens` 保持 5（==block_size，与生产一致），避免同时改两个变量污染归因；GB300 的 num_spec=7 属另一项优化，若 greedy 验证通过可另行评估（本次范围外）。
- **执行注意（关键）**：v0.26 容器启动参数需含 `"draft_sample_method":"greedy"`，**以 greedy 为主验证**；probabilistic 对照需改容器参数并重启引擎（成本高）——**probabilistic 对照优先复用 v0.25 历史数据**（`_tessa_bench_raw_2026-08-04.txt`）。仅当历史数据缺失/口径不可比时，才安排重启容器补测 probabilistic（每轮重启约 5-10 min，需 SRE 配合，成本另计）。
- **测试目标**：验证 greedy 相对 probabilistic 在**接受率**与**吞吐**上的差异，支撑 v0.26 灰度采样策略决策。
- **测试项**：
  a. **接受率对比（thinking off）**：同一 prompt 集 = short monologue（~200 token 单轮对话）×3 + 长推理/agentic 轨迹（~2k token 多轮/长推理）×3，采集 speculative accept rate（vLLM `/metrics` `vllm:spec_decode_*`，不可用则引擎日志增量，与 3.2 同路径）；对照 v0.25 参考 ~40%。
  b. **接受率对比（thinking on）**：同 prompt 集，`enable_thinking=true` 再测一轮；对照 v0.25 参考 ~24%。
  c. **吞吐对比（thinking on）**：S1（200→512，--retries 5 取中位）+ c10/c20（200→64，--rounds 2），各跑一轮。注：thinking off 的 S1/c10/c20 greedy 数据直接取自 2.1/2.3 主基准（容器即 greedy），无需重跑。
- **对比表模板**：

| 指标 | v0.25 probabilistic（历史） | v0.26 probabilistic（可选） | v0.26 greedy | 结论 |
|---|---|---|---|---|
| accept rate · thinking off | ~40%（参考） | 待填（可选） | 待填 | |
| accept rate · thinking on | ~24%（参考） | 待填（可选） | 待填 | |
| S1 total_tps · thinking off | 108.95 | 待填（可选） | 待填（=2.1 实测） | |
| S1 total_tps · thinking on | —（未测） | 待填（可选） | 待填 | |
| c10 agg_tps · thinking off | 26.79 | 待填（可选） | 待填（=2.3 实测） | |
| c20 agg_tps · thinking off | 31.99 | 待填（可选） | 待填（=2.3 实测） | |

- **判定标准**（阈值由测试专家设定）：
  - greedy 相对 probabilistic **接受率或吞吐提升 ≥ 5%**，且组1 GSM8K 无质量回退 → **值得切换**，建议灰度采用 greedy；
  - 提升 < 5% 或持平 → 收益不足以抵消口径/风险差异，**维持 probabilistic**，记录数据备查；
  - 退化 ≥ -5% → **不切换**，回退 probabilistic 并报架构师复核。

---

## 4. 执行顺序与总时长预估

| 阶段 | 内容 | 预估耗时 |
|---|---|---|
| P0 | 前置检查（引擎/网关/定制/双机/thinking 态） | 5 min |
| P1 | 组2.6 思考字段 + 组2.5 工具调用（快速功能项） | 3 min |
| P2 | 组1 GSM8K 200 题（顺序） | 8 min（中位 1.9s×200 + 开销） |
| P3 | 组2.1 单流 S1/S3/S5 | 8 min |
| P4 | 组2.2 并发 c1/c3/c5 | 4 min |
| P5 | 组2.3 高并发 c10/c20 | 3 min |
| P6 | 组2.4 长序列 200k + 100k | 8 min（含输入生成与 prefill） |
| P7 | 新增 3.1 router-dense + 3.2 acceptance | 6 min |
| P8 | 新增 3.3 Greedy A/B（接受率 2 态 + thinking on 吞吐） | 12 min |
| P9 | 数据汇总 + 对比表 + 结论 | 5 min |
| **合计** | | **约 62 min（含间隔与重试缓冲）** |

> 若组1需全量 1319 复测（仅当 200 题出现精度回退时触发）：另加约 40 min。

---

## 5. 风险评估与注意事项（v0.26 灰度环境）

### 5.1 双机一致性
- **风险**：灰度部署中 head/worker 镜像或权重不一致 → 测试数据无效。
- **措施**：测试前对双机执行：`docker ps` 容器名/镜像 tag 对比、vLLM 进程启动参数 diff、`/v1/models` 的 `system_fingerprint` 应含 vllm-0.26 版本串。任一不一致 → 暂停测试，通知 SRE。

### 5.2 内存与 KV 池
- **风险**：v0.26 新内核（fused_topk_bias、b12x 变体）可能改变显存占用；长序列 200k 是最高压场景。
- **措施**：
  - 测试前记录 `nvidia-smi` 显存基线；200k 长序列前后对比，异常峰值（> 历史 2× 或触发 OOM）→ 停止后续用例。
  - 前置检查确认 `max_model_len=600000` 且 KV 池富余（历史 2.56M tokens）。
  - 长序列用例安排在所有短场景之后（P6），避免引擎内存压力污染前序数据。

### 5.3 超时与失败判定标准
| 失败类型 | 判定 | 处理 |
|---|---|---|
| HTTP 4xx/5xx / 连接失败 | 单次错误 | 重试 1 次；持续失败 → 记 FAIL 并检查引擎日志 |
| 单流某 attempt 异常（无首 token / 无 usage / 超时） | 该 attempt 失败 | 按脚本重试机制跳过；attempts 全部失败 → 场景 FAIL |
| 并发 errors > 0 | 场景 FAIL | 记录错误明细，检查是否资源耗尽（KV 溢出/并发限制） |
| GSM8K err > 0 | 组1 FAIL | 复测该题 1 次，仍 err → 标注 |
| 长序列 status != 200 或 timeout | 场景 FAIL | 检查是否 max-model-len / KV 不足，报 SRE |
| 双机一致性检查失败 | 全部暂停 | 通知 SRE 修复后重测 |

### 5.4 口径注意（防误判）
- **thinking 默认态**：历史 08-04 为「默认关闭」；若 v0.26 恢复为 thinking=max（或网关注入方案 C），S1/S3/S5 与 c1/c3/c5 的 total/decode 会系统性变化。**测试开始前记录 thinking 默认态，若与 v0.25 不一致 → 对比表中标注「口径差异」，不直接判 FAIL**。
- **model ID**：v0.26 镜像 served-model-name 若变化，命令中 `--model` 需同步；测试前以 `/v1/models` 实际为准。
- **S5/c5 波动**：历史即存在轮间波动（S5 attempt3=143.63、c5 rounds 27.69/32.82/49.52），对比时看中位数与容差，不因单轮波动判 FAIL。
- **随机前缀**：所有性能脚本每次请求生成随机 `<rnd>` 前缀防 prefix-cache 命中，保证真实 prefill 口径（历史一致）。
- **Greedy 口径（v0.26 新增）**：v0.26 容器以 `draft_sample_method=greedy` 启动，2.1/2.2/2.3 各项对比 v0.25 历史（probabilistic）时，差异同时含内核升级与采样策略变化——归因须结合 3.3 专项结论拆分，不得仅凭单组数据判 FAIL。

---

## 6. 脚本/资产清单（执行时复用，无需重写）

| 脚本 | 路径 | 用途 |
|---|---|---|
| gsm8k_eval_E800k.py | `C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\` | 组1 GSM8K |
| bench_smoke_E600k.py | 同上 | 组2.1 单流 + 新增 3.1（需 router-dense prompt 变体） |
| bench_concurrency_smoke_E600k.py | 同上 | 组2.2 / 2.3 并发 |
| tessa_longseq_20260804.py | 同上 | 组2.4 长序列 |
| tessa_toolcall_20260804.py | 同上 | 组2.5 工具调用 |
| run_thinking.sh（th4 参考） | `C:\Users\novAI\WorkBuddy\集群部署\.tessa_test\` | 组2.6 思考字段 |
| gsm8k_test.jsonl | `C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\` | GSM8K 数据源（1319，idx0-199 子集） |
| 历史基线 | `deliverables\engineering-assurance\_tessa_bench_raw_2026-08-04.txt` | 对比基准 |
| tessa_greedy_ab_20260804.py（执行阶段新增） | `C:\Users\novAI\AppData\Local\Temp\vllm_bench\E800k\` | 3.3 Greedy A/B：接受率采集 + agentic 轨迹 prompt |

---

## 7. 验收汇总模板（执行后填写，回填综合报告）

| 测试组 | 项数 | PASS/FAIL | 关键结论 |
|---|---|---|---|
| 组1 社区基准 GSM8K | 1 | ☐ | acc=__（vs 98.5%） |
| 组2 单流 S1/S3/S5 | 3 | ☐ | total_tps 对比 |
| 组2 并发 c1/c3/c5 | 3 | ☐ | agg_tps 对比 |
| 组2 高并发 c10/c20 | 2 | ☐ | agg_tps / errors |
| 组2 长序列 200k/100k | 2 | ☐ | prefill / TTFT |
| 组2 工具调用 | 1 | ☐ | tool_calls 正确 |
| 组2 思考字段 | 1 | ☐ | reasoning==reasoning_content |
| 新增 fused_topk_bias | 1 | ☐ | TTFT/decode 收益 |
| 新增 DSpark acceptance | 1 | ☐（可观测时） | 接受率 |
| Greedy A/B 接受率（thinking off） | 1 | ☐ | accept rate vs 历史 ~40% |
| Greedy A/B 接受率（thinking on） | 1 | ☐ | accept rate vs 历史 ~24% |
| Greedy A/B 吞吐（thinking on S1 + c10/c20） | 2 | ☐ | total_tps / agg_tps |
| **合计** | **19** | | |

> 本方案仅设计，不执行。待 v0.26 镜像 `:0.2.0-v026.0` 灰度部署就绪后，按本方案调度执行。
