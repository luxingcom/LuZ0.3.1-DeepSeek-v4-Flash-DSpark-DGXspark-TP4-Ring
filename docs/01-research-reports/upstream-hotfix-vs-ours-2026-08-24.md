# 上游 MiaAI-Lab hotfix 对照分析（16 项）— TP4 集群 vs 上游 2x DGX Spark

- **执行人**：阿奇（Archi）· 系统架构师（architect-1，纯只读）
- **日期**：2026-08-24
- **上游仓库**：`github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`（2× DGX Spark TP2 部署，基线 vLLM 0.25.2.dev0 / anemll 0.1.1）
- **我们集群**：4× DGX Spark GB10（sm_121）TP4，LuZ0.3.1（anemll 0.2.1-v026.0 基座 = vLLM 0.26.1 fork）+ overlay（W4A4=2 / SHARED=1 / thr4096 / util0.82 / MTP n7 / FI 0.6.16 / wsdedup / nvfp4_ds_mla / cudagraph 1-96）
- **纪律**：WebFetch（上游 docs/PATCHES.md、docs/vllm-027-new-patches.md、CHANGELOG、PR #89/#116/#121）+ SSH 只读 node01 容器内源码/启动脚本。未改任何生产配置。
- **口径**：**[上游实证]** = 上游仓库页面/PR/CHANGELOG 直接验证；**[容器实证]** = node01 容器内源码/日志直接验证；**[推断]** = 需落地前二次确认。

---

## 0. 一页结论

1. **最严重一项未解决**：**Issue #22 nvfp4_ds_mla 长上下文解码修复未引入**。[容器实证] 我们 `vllm/v1/attention/backends/mla/flashmla_sparse.py` L861 仍是 `use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"`，上游补丁是把该行改为 `in ("fp8_ds_mla", "nvfp4_ds_mla")`。**我们生产正用 `--kv-cache-dtype nvfp4_ds_mla`，即长上下文解码走慢速 `_forward_bf16_kv` 路径**。上游实测 600K+ 上下文 nvfp4 ≈1 tok/s vs fp8 ≈17 tok/s（短上下文不受影响）。这是 16 项中唯一可能直接压低我们"超长上下文"能力的项，建议列为 P1 验证/移植。
2. **第二项未解决**：**Issue #55 工具调用截断 finish_reason 语义**。[容器实证] `entrypoints/openai/chat_completion/serving.py` L706 仍报 `"tool_calls"`（上游改为 `"length"` + 丢弃非法 JSON args），且未修 FinishReason IntEnum 比较 bug。对 agent/tool 业务有正确性影响。
3. **第三项未解决（安全）**：**密钥日志脱敏 fail-closed（PR #89 对应物）未落地**。[容器实证] `docker logs` 中 vLLM `non-default args` 打印 `'api_key': ['<明文>']`；且我们 `start_tp4_head.sh` L77 `echo "[i] serve 命令: $SERVE_CMD"` 会把含 `--api-key` 的完整命令打印到操作日志。建议 P1 整改。
4. **六项 v0.27 backport 中我们天然占优**：fork 是 0.26.1-dev（晚于上游 0.25.2 基座），#48957（skip empty c128，compressor.py `save_partial_states` 在）[容器实证] 已含；#50004 我们从没有（上游已移除，我们无该损坏源）；#48407 不适用（无 dense-MHA 路由）；#49486/#50298/#50312 未确认/未发现签名。
5. **多项上游 TP2 形态修复对我们"不适用或天然安全"**：#79 自旋等待（我们 fork 默认 busy_loop_s=1）、#31/#34 thinking budget（我们默认不启用，与上游 opt-in 默认一致）、幻影 encoder（纯文本模型）、脚本权限、HF hub 超时（生产 HF_HUB_OFFLINE=1）。
6. **与我们既有结论的关联**：Issue #26（SWA prefix-cache retention）我们**已配置** `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`（start_tp4_head.sh）且 fork 支持该机制（kv_cache_coordinator.py）——与 E4 KV 足迹调查同源；Issue #27 我们用 `--long-prefill-token-threshold 4096`（上游 1024）是**有记录的主动决策**（thr4096 采纳，PR +8.5~13.5%）；cudagraph 损坏源（#50004/#49486）我们均无 → capture 1-96 无该风险。

---

## 1. 16 项 hotfix 对照总表

> 判定：✅已解决 / 🟡部分（机制类似但实现/值不同）/ ⚪不适用 / ❌未解决·需处置

| # | 上游 hotfix | 上游机制（证据） | 我们现状 | 判定 | 证据来源 |
|---|---|---|---|---|---|
| 1 | **suppress-stops-in-reasoning**（`DSPARK_SUPPRESS_STOPS_IN_REASONING=1`） | 客户端 stop 字符串在 `<think>` 内保持休眠，spec-decode 跳过 `</think>`；`patches/hotfix-dsv4-suppress-stops-in-reasoning.py`（f2305b1/027b51a5）[上游实证 PATCHES.md] | 容器全树 grep `SUPPRESS_STOPS` 无任何命中；启动脚本无该 env | ❌ 未解决（可选采纳，低-中优先：仅当 agent harness 使用 stop 字符串时才有收益；我们网关走标准 messages） | [容器实证] |
| 2 | **Issue #22 nvfp4_ds_mla 长上下文解码**（`DSPARK_SKIP_ISSUE22_HOTFIX`） | `flashmla_sparse.py` L880 的 `use_fp8_cache` 检查改为 `in ("fp8_ds_mla","nvfp4_ds_mla")`；否则 nvfp4 走 `_forward_bf16_kv` 慢路径，600K+ ≈1 tok/s vs fp8 ≈17 tok/s；584B 布局两 dtype 相同[上游实证 PATCHES.md] | 容器 `flashmla_sparse.py` **L861 仍是 `== "fp8_ds_mla"`**，补丁未引入；生产正用 nvfp4_ds_mla | ❌ **未解决·P1** | [容器实证] L861 |
| 3 | **Issue #26 SWA prefix-cache retention**（`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`） | SWA 组按设计释放窗口外 block 会拖垮混合前缀命中（warm 32K 全重 prefill 421s）；用 retention interval=4096 稀疏化 checkpoint 保 warm hit（1f9765e2 + #36 7048daf3 v2）[上游实证 CHANGELOG/PATCHES] | **我们已设 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`**（start_tp4_head.sh ENV_ARGS）；fork `kv_cache_coordinator.py` 支持该机制（_validate/retention_interval）[容器实证] | ✅ 已配置（值与上游一致 4096）；⚠ SWA hit 语义与 #36 v2 需 A/B 验证 | [容器实证] envs.py L301/1129、kv_cache_coordinator.py L30-126 |
| 4 | **Issue #27 长 prefill 分块上限**（`LONG_PREFILL_TOKEN_THRESHOLD=1024`） | ① 强制执行 `max_num_partial_prefills` 准入；② `LONG_PREFILL_TOKEN_THRESHOLD=1024` 限制单 chunk（2f180e7）[上游实证 PATCHES.md] | ① 我们 scheduler.py 无 `max_num_partial_prefills` 读取[容器实证]；② 我们用 `--long-prefill-token-threshold 4096`（fork 支持该参数，scheduler.py L507/L861 强制） | 🟡 部分：分块上限机制在但值为 4096（≠1024），且是**有记录主动决策**（thr4096 采纳）；准入并发公平性部分缺 | [容器实证] scheduler.py |
| 5 | **Issue #43 scheduler 诊断**（`DSPARK_ISSUE43_SCHED_DIAG=1`） | 每步每请求调度 token 摘要 + 零 token decode-skip 记录 + 纯 Python 模拟器；默认关零开销（f2d28bf）[上游实证 PATCHES.md] | 无该 env/诊断代码[容器实证]；但我们已有自研 diag 脚本（diag_rank0、sre_diag 系列） | ⚪ 不适用/可选：方法论可借鉴，我们已有自己的诊断工具；默认关、不影响正确性 | [容器实证] |
| 6 | **Issue #31/#34/#48 GPU thinking_token_budget**（`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`，默认关） | GPU-resident Triton kernel 硬顶 reasoning token；因 issue #66 omit-field 路径复现 ~4.5 tok/s 断崖，已改为 **opt-in 默认关**（2689b1f + b6b91c08）[上游实证 CHANGELOG 08-20/#66] | 我们**未设置**该 env → 处于与上游一致的"默认关闭安全态"；我们另有"≤c3 认知上限"方案 [既有报告 miaai-latest-2026-08-15] | ✅ 已解决（不启用即安全，与上游 opt-in 默认一致）；不照搬 hotfix | [既有报告] |
| 7 | **Issue #55 工具调用截断**（finish_reason tool_calls→length） | max_tokens 截断工具调用时流式/非流式都报 `length`（was `tool_calls`）+ 丢弃非法 JSON args；顺带修 FinishReason IntEnum≠StrEnum 比较 bug（db9b9ce）[上游实证 PATCHES.md] | 容器 `chat_completion/serving.py` L706 仍 `finish_reason_ = "tool_calls"`；无截断→length 逻辑[容器实证] | ❌ **未解决·P2**（正确性：agent 工具调用被截断时客户端收到非法 JSON + tool_calls 语义） | [容器实证] serving.py L706 |
| 8 | **Issue #79 P-core 自旋等待**（TP2 vLLM shm busy_loop_s=1s） | 双节点 shm broadcast 的 busy_loop 等待；`hotfix-gb10-spin-wait.sh` 设 busy_loop_s=1s[上游实证 patches/ 目录] | 容器 `shm_broadcast.py` L126 **默认 `busy_loop_s: float = 1`**，即已是 1s[容器实证]；我们 TP4 用同一 shm 机制 | ✅ 已解决（fork 默认即上游 hotfix 值；TP4 下无额外动作） | [容器实证] shm_broadcast.py L126 |
| 9 | **六项 v0.27 性能 backports** | 上游在 0.25.2 上 backport：#49486 skip topk（live）、#50312 MTP buffer（live）、#48957 skip empty c128（verified）、#50298 FlashMLA workspace（verified）、#48407 dense indexer（Stage A dormant）、#50004 adaptive topk（**已移除**，上游 #51318 回滚）[上游实证 vllm-027-new-patches.md] | 我们的 0.26.1 fork 天然不同：#48957 **已含**（compressor.py `save_partial_states`）[容器实证]；#50004 **从没有**（无损坏源，无需回滚）[容器实证]；#48407 不适用（无 dense-MHA 路由）[既有报告]；#49486/#50298/#50312 **未发现签名**（需逐项 diff 确认）[容器实证] | 🟡 部分：#48957 已含；#50004/#48407 不适用；#49486/#50298/#50312 未确认 | [容器实证] + [既有报告 upstream-tracking] |
| 10 | **DRAFT_SAMPLE_METHOD 门禁**（只允许 probabilistic/greedy，2026-08-21） | env 门禁：`.env.dspark` 只接受 probabilistic/greedy，其他值在 JSON 构建前非零退出（CHANGELOG 08-20）[上游实证] | 我们硬编码 `--speculative-config '{"draft_sample_method":"probabilistic"}'`，值在允许集内[容器实证 start_tp4_head.sh]；fork 无该 env 门禁 | ✅ 已解决（probabilistic 在门禁允许集；无门禁但硬编码不构成风险） | [容器实证] |
| 11 | **Graph-capture 损坏防护（PR #116/#121，2026-08-23 合入）** | 移除 #50004 adaptive-topk backport（上游 #51318 已回滚：capture 时 stride 不匹配 → 间歇静默输出损坏）；给 #49486 加 #52492 CUDA-capture 防护（`not is_current_stream_capturing()`）[上游实证 PR #116/#121] | 我们**从未含 #50004**、也**无 #49486 backport** → 没有上游损坏源；cudagraph capture 1-96 无该风险[容器实证 + 既有报告] | ✅ 已解决/不适用（损坏源在本 fork 不存在） | [上游实证 PR] + [容器实证] |
| 12 | **密钥日志脱敏 fail-closed（PR #89，2026-08-21 合入）** | 多密钥 `DSPARK_API_KEYS`（默认关）+ 启动日志 `non-default args` 中 key 替换为 `<redacted:N>`；配置错误 exit 2 fail-closed[上游实证 PR #89] | **未落地**：容器启动日志明文打印 `'api_key': ['c3b4de...']`[容器实证 docker logs]；start 脚本 L77 echo 含 `--api-key` 完整命令；我们仍用单 key `VLLM_API_KEY` | ❌ **未解决·P1（安全）** | [容器实证] docker logs + start_tp4_head.sh |
| 13 | **幻影 encoder 输出 token 移除（2026-08-22，Issue #109）** | 纯编码器 EC 生产者不再产生幻影 token ID 0 / 编码后循环（`make_empty_encoder_model_runner_output`）[上游实证 CHANGELOG] | 我们为**纯文本 decoder-only** 模型（DeepSeek V4 Flash），无 encoder-decoder/多模态分支；MTP 不受影响 | ⚪ 不适用 | [容器实证] |
| 14 | **partial-prefill cap 缓存校验（2026-08-22，Issue #105）** | 畸形 `DSPARK_MAX_INFLIGHT_PREFILLS` 不再启动后崩溃——构造时解析缓存一次，畸形值回退 `max_num_partial_prefills`[上游实证 CHANGELOG 08-21] | 我们 fork **无 `DSPARK_MAX_INFLIGHT_PREFILLS`** 该 env/代码[容器实证]；调度用 stock vLLM 字段 | ⚪ 不适用（我们无该 env 旋钮，无同类崩溃面） | [容器实证] |
| 15 | **HF hub 超时（HF_HUB_DOWNLOAD_TIMEOUT 120s/30s，2026-08-20）** | `prepare-dspark-model-cache.sh` 两个 docker run 块传 `HF_HUB_DOWNLOAD_TIMEOUT=120` + `HF_HUB_ETAG_TIMEOUT=30`，防慢链路杀死多 GB 分片[上游实证 CHANGELOG 08-20] | 生产容器 `HF_HUB_OFFLINE=1`（权重本地缓存，不联网下载）[容器实证 start_tp4_head.sh]；该问题仅存在于一次性模型准备阶段 | ⚪ 不适用（生产离线；若未来重新下载模型可采纳该超时值） | [容器实证] |
| 16 | **脚本可执行权限（2026-07-31）** | 修正仓库脚本 exec bit[上游实证 CHANGELOG 07-29/07-31 段] | 我们脚本有自己的权限纪律（check_vllm_script.sh 校验 + 0755 保留）；与运行正确性无关 | ⚪ 不适用 | [容器实证] |

---

## 2. 未解决·需处置项清单（按优先级）

### P1-1：Issue #22 nvfp4_ds_mla 长上下文解码走慢路径（正确性/性能）
- **现状**：`flashmla_sparse.py` L861 `use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"`；我们生产 `nvfp4_ds_mla` → 长上下文解码走 `_forward_bf16_kv` 慢路径。
- **上游机制**：一行改 `in ("fp8_ds_mla", "nvfp4_ds_mla")`；584B KV 布局两 dtype 相同，仅 kernel 分派不同。
- **处置建议**：
  1. 先在一次性测试容器（不占生产 GPU）确认我们 fork 的 `_forward_fp8_kv` 对 nvfp4_ds_mla 数据可正确消费（对照上游 fix 语义）；
  2. 长上下文 A/B（131K/256K/600K decode t/s）：改前 vs 改后，量化影响；
  3. 若验证通过，作为镜像 overlay 单行注入（或等 vLLM #41834 路线整体升级时自然获得）。
- **风险提示**：此发现**可能部分推翻**我们"131K decode ~8 tok/s 是 GB10 物理极限"的结论——上游 fp8 路径在 600K 仍 ~17 tok/s（TP2 单流）。若 nvfp4 进快速路径，超长上下文 decode 上限可能显著高于当前认知。需 A/B 实证后再定论。

### P1-2：API 密钥日志脱敏（安全）
- **现状**：容器启动日志 `non-default args` 明文打印 `api_key`；start 脚本 L77 echo 完整 SERVE_CMD（含 `--api-key` 明文）到操作日志。
- **处置建议**：
  1. vLLM 侧：启动脚本在 `vllm serve` 前对日志层做 key 掩码（或容器内 grep 阻断），对齐上游 PR #89 的 `<redacted:N>`；
  2. 脚本侧：`echo "[i] serve 命令..."` 改为打印脱敏版本（替换 `--api-key <val>` 为 `--api-key <redacted>`）；
  3. 已泄漏 key 轮换：此 key 已出现在日志，按我们 api-key-rotation 流程轮换；
  4. 可评估引入多 key `DSPARK_API_KEYS`（上游 PR #89）或维持单 key 但保证日志脱敏。

### P2-1：Issue #55 工具调用截断 finish_reason 语义（正确性）
- **现状**：`chat_completion/serving.py` L706 截断仍报 `tool_calls`，且 FinishReason IntEnum 字符串比较 bug 未修。
- **处置建议**：
  1. 对照上游 db9b9ce 的 serving 层逻辑，评估在我们 serving.py 上做等价修正（截断时 `finish_reason="length"` + 丢弃不可解析 JSON 的 tool_calls）；
  2. 若 agent 业务对"工具调用被截断后客户端拿到非法 JSON"敏感，优先级可升 P1；
  3. 顺带验证我们 gateway（8003 responses）是否在入口把 tool_calls 语义二次处理。

### P2-2：suppress-stops-in-reasoning（可选）
- **现状**：无该机制。
- **处置建议**：仅当 agent harness 使用客户端 stop 字符串且出现 `<think>` 内被截断症状时移植；否则记录为 watch 项（与 upstream Issue #52/#120 类 agent 会话韧性同族）。

### P3：六项 v0.27 backport 中 #49486/#50298/#50312 逐项 diff 确认
- **现状**：#49486（skip topk/router，TTFT -3.4%）、#50298（FlashMLA workspace reuse，kernel 1.88x）、#50312（MTP buffer，256MiB/rank）在我们 fork 未发现签名。
- **处置建议**：与 vLLM 上游 #41834 对账清单合并处理；小 PR 单挑 backport 到我们的 deepseek_v4 实现，性能收益预期小-中（TTFT 侧）。非紧急。

---

## 3. 与我们既有发现的关联

### 3.1 Issue #22 ↔ nvfp4_ds_mla / E4 KV 足迹
- E4 KV 足迹调查（e4-kv-footprint-2026-08-23）已确认 **nvfp4_ds_mla 每 token 584B 物理包络不变**（448+128+8），且指出"物理 per-token 字节两档相同"——这**与上游 issue22 的"584B 布局两 dtype 相同"断言一致**，进一步坐实我们的 dispatch 问题。
- 但 E4 调查的结论"短上下文影响小"仍需注意：上游 issue22 明确"短上下文 ~66 tok/s 不受影响、600K+ 才塌到 1 tok/s"。我们的 131K decode 数据可能在慢路径上**尚未到塌陷区**，因此未被既有测量暴露。**这解释了为什么我们的物理极限结论与上游 fp8 17 tok/s@600K 存在潜在矛盾**——需 A/B 澄清。

### 3.2 Issue #26 ↔ SWA / partial-prefill / E4 准入预留
- 我们已设 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`，与上游 #26 值一致；fork 支持该机制。
- E4 调查揭示了 SWA 组 `max_admission_blocks_per_request` 随 `max_in_flight_tokens` 上涨的预留机制（131→259 块）——**这是"SWA 每请求准入预留"而非"SWA 前缀命中拖垮"**，与上游 #26 的"SWA 释放窗口外 block 拖垮混合前缀命中"是两个不同现象。两者都指向 SWA 组是长上下文行为的核心变量。建议在长上下文 warm-hit A/B 中同时观测 retention interval 与 admission reserve。
- 上游 #36 v2 语义（"SWA 可收缩但用 retention 保 warm hit"）建议在 warm 命中复测中确认我们 coordinator 行为一致。

### 3.3 Issue #27 ↔ thr4096 决策
- 我们用 4096（上游 1024）是**有记录主动决策**（threshold-4096-adoption-2026-08-22：PR 四档 +8.5~13.5%，DE 归一无回退）。差异在于并发公平性：上游 #27 ①部分（`max_num_partial_prefills` 准入）我们没有，②分块上限我们有但值不同。
- 我们的 128K 长 ctx 并发断崖（≤c3 认知）与上游 #27/#43 同根；MiaAI 已确认 0.25.2/0.26.1 均不支持并发 partial prefill >1 → 根治需升 base image（上游 #45 方向），与我们 M2 结论一致。

### 3.4 cudagraph 损坏 ↔ capture 1-96
- 上游损坏源（#50004 adaptive topk capture-unsafe backport）在我们 fork **不存在**；我们也没有 #49486（不需要 #52492 防护）。
- 我们的 capture 1-96 + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 无该损坏风险；注意上游 PR #116 A/B 显示移除 #50004 无回归（8K 67.35→68.90 t/s），印证 capture 安全与 adaptive topk 无关。

### 3.5 DRAFT_SAMPLE_METHOD ↔ probabilistic
- 我们硬编码 probabilistic（允许集内），无门禁本身但无风险；若未来把 `draft_sample_method` 提到 env 可配置，需同步加门禁（防 typo 导致非法采样方法）。

### 3.6 幻影 encoder ↔ MTP n7
- 不适用：纯文本 decoder-only 模型，无 encoder 输出路径；MTP n7 不受 Issue #109 影响。

### 3.7 额外发现（非 16 项但重要）
- **启动脚本遗留 inert env**：`DSPARK_SLOT_CLAMP=1`、`VLLM_DSPARK_LOCAL_ARGMAX=1` 在容器 vLLM 全树无消费点[容器实证]——疑似旧 overlay 残留。建议清理或确认消费方（防误导排障）。
- **DSpark 多请求 slot/ragged（上游 Patch 1/2/2b）**：我们 fork 无 `dspark_proposer.py`，DSpark 实现走 `vllm/v1/spec_decode/dflash.py`（不同架构）。生产 max_num_seqs=12 + DSpark n7 的 DE 接受率健康（step_eff 18-19 / tokens_per_step 4.1-4.4）[既有报告]，无接受率塌缩迹象 → 不视为风险；但若未来出现多请求接受率下降，优先对照上游 Patch1 的 request-stable slot 思路。

---

## 4. 判定汇总（每项一行）

| # | hotfix | 判定 |
|---|---|---|
| 1 | suppress-stops-in-reasoning | ❌ 未解决（可选，低-中优先） |
| 2 | Issue #22 nvfp4_ds_mla 长上下文 | ❌ **未解决·P1（最严重）** |
| 3 | Issue #26 SWA prefix-cache retention | ✅ 已配置（4096 一致；#36 v2 语义待 A/B） |
| 4 | Issue #27 长 prefill 分块 | 🟡 部分（4096 vs 1024，主动决策；准入公平性缺） |
| 5 | Issue #43 scheduler 诊断 | ⚪ 不适用/可选（自有诊断工具） |
| 6 | Issue #31/#34/#48 thinking budget | ✅ 已解决（默认关=安全态，不照搬） |
| 7 | Issue #55 工具调用截断 | ❌ 未解决·P2（正确性） |
| 8 | Issue #79 P-core 自旋等待 | ✅ 已解决（fork 默认 busy_loop_s=1） |
| 9 | 六项 v0.27 backports | 🟡 部分（#48957 已含；#50004/#48407 不适用；#49486/#50298/#50312 未确认） |
| 10 | DRAFT_SAMPLE_METHOD 门禁 | ✅ 已解决（probabilistic 在允许集） |
| 11 | Graph-capture 损坏防护 | ✅ 已解决/不适用（损坏源不存在） |
| 12 | 密钥日志脱敏 | ❌ **未解决·P1（安全，日志明文泄漏）** |
| 13 | 幻影 encoder 输出 | ⚪ 不适用（纯文本模型） |
| 14 | partial-prefill cap 校验 | ⚪ 不适用（无该 env 旋钮） |
| 15 | HF hub 超时 | ⚪ 不适用（生产离线；下载脚本可采纳） |
| 16 | 脚本可执行权限 | ⚪ 不适用 |

---

## 5. 证据索引

### 上游（WebFetch 实证）
- `docs/PATCHES.md`（raw）：Patch1/2/2b、Issue #21/#22/#52、suppress-stops、编码器 fix 详情
- `docs/vllm-027-new-patches.md`（raw）：6 项 0.27 backport 状态表（#49486 live、#50312 live、#48957 verified、#50298 verified、#48407 dormant、#50004 removed + #52492 guard）
- `CHANGELOG.md`（raw）：2026-08-20（DRAFT_SAMPLE_METHOD 门禁、HF hub timeout、#66 opt-in）、08-21（DSPARK_API_KEYS、Issue #105）、08-22（#50004 移除 + #52492 guard、Issue #109 幻影 encoder）、08-23（boot-shape-warmup、Issue #120）
- PR #116（closed/unmerged，#50004 移除 + #52492 guard 内容）、#121（merged，current-main 整合）、#89（merged，多密钥 + 日志脱敏）

### 我们（SSH 只读 node01 容器实证）
- `flashmla_sparse.py` L861：`use_fp8_cache == "fp8_ds_mla"`（issue22 未修）
- `kv_cache_coordinator.py` L30-126 + `envs.py` L301/1129：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 机制存在
- `config/scheduler.py` L76-311：`long_prefill_token_threshold` 支持
- `v1/core/sched/scheduler.py` L507/L861：threshold 强制执行；无 `max_num_partial_prefills`
- `chat_completion/serving.py` L706/L972-978：截断仍报 tool_calls（issue55 未修）
- `compressor.py` L20-21/L349：`save_partial_states` 在（#48957 已含）
- `shm_broadcast.py` L126：`busy_loop_s=1` 默认（#79 已覆盖）
- `docker logs`：`non-default args` 明文 api_key（#89 对应物缺失）
- `start_tp4_head.sh`（生产脚本）：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`、`--long-prefill-token-threshold 4096`、`draft_sample_method=probabilistic`、`nvfp4_ds_mla`、capture 1-96、`HF_HUB_OFFLINE=1`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`（issue65 已覆盖，非 16 项）

### 既有报告引用
- upstream-tracking-2026-08-22（fork 基线 0.26.1-dev @ d3d3b2cca；#48957 已含、#49486 未找到签名）
- miaai-latest-2026-08-15（#66 thinking budget 断崖 → 不照搬；A1-A6 借鉴清单）
- research-miaai-dspark-comparison-2026-08-14（128K 断崖跨 TP 规模一致）
- e4-kv-footprint-2026-08-23（nvfp4_ds_mla 584B 包络、SWA 准入预留机制）
- threshold-4096-adoption-2026-08-22（4096 采纳证据：PR +8.5~13.5%）
- g1-production-restore-2026-08-24（LuZ0.3.1 生产核验：FI 0.6.16 / thr4096 / nvfp4_ds_mla / capture 全 PASS）

---

## 6. 局限与诚实声明

1. **Issue #22 的实测影响未在本集群量化**：容器源码证明走了慢路径，但"慢到什么程度"需 A/B（131K/256K/600K）才能定论；上游数字（600K ≈1 tok/s）是 TP2 口径。
2. **#49486/#50298/#50312 判定为"未确认"**：以 grep 签名 + 既有报告为准，未做逐字 diff；建议与 vLLM #41834 对账时补 diff。
3. **Issue #26 #36 v2 语义**：确认了机制与配置存在，但"SWA 命中不覆盖 curr_hit_length"的具体语义在 warm-hit A/B 前不视为已证明等价。
4. 容器内验证基于 node01 rank0 容器（LuZ0.3.1 当前运行态）；worker（02/03/04）代码同镜像，假定一致（未逐一开容器验证）。
5. 所有 SSH 操作只读（cat/grep/docker logs/docker exec sh -c 只读命令），未改任何配置、未重启任何服务。

---

*本报告由工程保障团队（系统架构师 architect-1）生成；所有处置建议需经工程负责人裁定后再落地。*
