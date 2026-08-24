# MiaAI-Lab 仓库最新提交（bug 修复）调研报告

**日期**：2026-08-15 ｜ **作者**：architect-1（架构师，只读调研，未改动任何生产配置）
**调研对象**：https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
**数据来源**：GitHub REST API（commits/issues/tags/branches/contents），所有日期/编号以 API 实际返回为准
**对照基线**：本项目 DGX Spark GB10（sm_121）四机 TP4，vLLM 0.27 + NVFP4 部署测试经验（2026-08-15）

---

## 0. 执行摘要

仓库近两周（2026-08-01 → 08-15）活动高度集中，**最近 30 个提交全部落在 08-12 ~ 08-14**，主题为调度公平性、KV 前缀缓存正确性、finish_reason 语义、thinking 预算、以及 0.27 性能 backport 系列。**仓库无 GitHub Release、无 tag**（版本管理靠 commit + 镜像 digest 固定）。

与本项目经验交叉后，得到 **3 个高价值可借鉴项**（均与我们的痛点直接同源）：
1. **A 类（生产可直接借鉴）**：`fix/worker-nccl-ib-gid-index` 分支 — 按节点注入 `NCCL_IB_GID_INDEX`，直接命中我们「head=3/worker=2 的 NCCL GID 不一致」整改项。
2. **A 类（生产稳定性）**：issue #65 的 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` + `TILELANG_CACHE_DIR` 持久化 — 防 CuTeDSL/TileLang 推理期 JIT 触发 `sample_tokens` RPC 超时导致引擎死亡。我们在 GB10 同样依赖 CuTeDSL JIT，风险同构。
3. **B 类（0.27 测试借鉴）**：issue #27/#43 的调度公平性修复（`max_num_partial_prefills` 强制执行 + `LONG_PREFILL_TOKEN_THRESHOLD=1024` + decode floor guardrail）— 与我们 128K 长 ctx 并发断崖认知同根；#45 指明根治需新 base image（并发 partial prefill）。

**重要警示**：issue #66（2026-08-15 open）证明 MiaAI 的 **thinking_token_budget GPU 版 hotfix 在 omit-field 路径仍复现 ~4.5 tok/s decode 断崖**（间歇性、非均匀、A/B 时漏检）。**结论：不要照搬该 hotfix；我们「≤c3」的认知性上限是更稳的方案。**

---

## 1. 仓库状态快照（GitHub API 证据）

| 项 | 值 |
|---|---|
| 默认分支 | `main` @ `f752cd04`（2026-08-14T14:54:43Z） |
| Release/Tag | 无（0 个 release，0 个 tag） |
| 分支 | main、fix/issue43-decode-fairness、fix/issue-4-env-prepare-rough-edges、fix/restore-prompt-tokens-details、fix/suppress-stops-in-reasoning、**fix/worker-nccl-ib-gid-index**、publish-dspark |
| 基础镜像 | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（vLLM 0.25.2.dev0+g752a3a504）— **注意与我们的 0.27 基线不同** |
| docs/vllm-027-new-patches.md | 已更新：0.27 backport 表（6 个脚本：#49486 live / #48407 Stage A dormant / #50312 live / #50004 live / #48957 script-verified / #50298 script-verified）+ 未 backport 列表（#49236 需 C++ 重编 / #46789 / #48993 / #48047 需 FlashInfer≥0.6.14） |

---

## 2. 最新提交清单（GitHub API commits?per_page=30，按日期倒序）

| 日期 | SHA | 类型 | 标题 |
|---|---|---|---|
| 08-14 | f752cd04 | docs | CHANGELOG: #55 no-regression audit + FinishReason IntEnum 发现 |
| 08-14 | db9b9cef | fix | **#55 工具调用截断报 finish_reason=length（was tool_calls）+ 丢弃非法 JSON args** |
| 08-14 | 1969774a | docs | README: max 推理量级 ~50k chars/~12.5k tok |
| 08-14 | e3659adb | docs | README: 客户端如何启用 thinking_token_budget |
| 08-14 | 2689b1fe | feat | V2 GPU-resident thinking_token_budget（PR #48 本地合并） |
| 08-14 | c3c89919 | docs | README: .env.dspark 有效开关清单 |
| 08-14 | 655d4fc1 | docs | README: 顶部快速开始 |
| 08-13 | 103af68a | merge | PR #44（#43 decode 公平性） |
| 08-13 | f2d28bff | fix | **#43 有界 decode 服务 + 每步调度诊断** |
| 08-13 | 3c9576c5 | merge | suppress-stops-in-reasoning 分支 |
| 08-13 | 3d11d0c2 | docs | changelog: stop-in-think 实测确认 |
| 08-13 | 027b51a5 | fix | **suppress-stops hotfix 同步到 worker**（scp + 清 root 残留目录） |
| 08-13 | f2305b17 | fix | **stop 字符串在 </think> 前保持休眠**（Capicua25x Patch 5 移植） |
| 08-13 | 5cdb9aa9 | ci | **CPU recipe 校验脚本**（push gate） |
| 08-13 | fb6cbd1e | docs | .github 占位 |
| 08-13 | d903470d | docs | changelog: thinking-budget 回退；无 tok/s 断崖 |
| 08-13 | 27a55695 | fix | **#38 启动脚本在 exec vllm 前应用 .sh hotfix；stop 用 docker rm -f 不挂起** |
| 08-13 | b6b91c08 | revert | 撤回 V2 thinking_token_budget hook 出启动路径 |
| 08-13 | 76ca4e58 | revert | 撤回 #31/#34 thinking_token_budget hotfix |
| 08-13 | 7048daf3 | fix | **#36 SWA 可再次收缩混合前缀命中长度** |
| 08-13 | 437e61fe | fix | V2 decode 增量扫描 thinking budget（随后撤回） |
| 08-13 | 524054d2 | fix | #34 默认 thinking budget 32k / max_tokens 128k（随后撤回） |
| 08-13 | e026ecf0 | docs | 建议 max_tokens > thinking_token_budget |
| 08-13 | 018c6bc3 | fix | #31 在 DSpark 启用 thinking_token_budget（随后撤回） |
| 08-13 | 2d872386 | docs | changelog #26/#27 |
| 08-13 | 1f9765e2 | fix | **#26 混合前缀缓存命中不从 SWA 组取（SWA min-hit 问题）** |
| 08-12 | 2f180e71 | fix | **#27 强制执行 max_num_partial_prefills；限制 prefill chunk** |
| 08-12 | cbd719fe | ops | **digest-pin DSPARK_VLLM_IMAGE（不可变清单）** |
| 08-12 | 831b1db4 | fix | **#24 backport 上游 #44993 grammar-boundary（json_schema+thinking 损坏）** |
| 08-12 | ff093bd1 | merge | #25 本地 HF 缓存缺 pinned revision 时告警 |

**另：分支 `fix/worker-nccl-ib-gid-index`（2026-07-24，eac6b56）** 不在 main 上，但与本项目 NCCL 问题直接相关（见 §4）。

---

## 3. 最近 Issues 状态（GitHub API，非 PR，按创建时间倒序）

| # | 状态 | 创建 | 标题 | 备注 |
|---|---|---|---|---|
| 66 | **open** | 08-15 | issue31 GPU thinking-budget hotfix：**omit-field 路径复现 ~4.5 tok/s decode 断崖** | ⚠️ 与我们的 decode 性能直接相关 |
| 65 | **open** | 08-15 | infra：**推理期 CuTeDSL/TileLang JIT 超过 sample_tokens RPC 超时 → 引擎死亡** | 修复：`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` + `TILELANG_CACHE_DIR` 持久化 |
| 58 | closed | 08-14 | Low Benchmark Question | 问答 |
| 57 | closed | 08-14 | **DSpark 投机解码在混合长度并发时崩溃引擎**（ValueError: uniform effective per-request target context lengths） | Dynamo standalone 部署；注意点见 §4 |
| 56 | closed | 08-14 | DEFAULT_THINKING=max 常不终止 | 由 #48 解决 |
| 55 | closed | 08-14 | **工具调用被 max_tokens 截断报 tool_calls 且返回非法 JSON** | 已修（db9b9ce） |
| 52 | **open** | 08-14 | **以 assistant 消息结尾的请求无 generation header → 空回合自续循环** | encoding 层渲染 bug，未修 |
| 45 | **open** | 08-13 | **升级基础镜像到支持 Concurrent Partial Prefill（解锁 #43 整请求公平性）** | 根治 #43 的方向 |
| 43 | closed | 08-13 | #27 在精确 32K/62K x2/x4/x8 冷 lane 仍不公平 | 已修（f2d28bf） |
| 40 | closed | 08-13 | 需要多少节点 | 问答 |
| 39 | **open** | 08-13 | 08-13 commits（#26/#27/#31/#34）：32k ctx 下 4.7x decode 回退 + 工具名损坏 | 部分由撤回解决 |
| 37 | **open** | 08-13 | **reasoning_effort=high + tools 退化（泄漏 DSML、无 tool_calls）** | 未修 |
| 36 | **open** | 08-13 | #26/#27 后：warm ~21k 共享前缀命中开始泄漏 DSML / token salad | 已修（7048daf v2） |
| 35 | **open** | 08-13 | #31/#34 budget hotfix 破坏 high+tools | 随 #31 撤回 |
| 34 | closed | 08-13 | #31 后续：标准 OpenAI 客户端空白回合 | 随 #31 撤回 |
| 33 | closed | 08-13 | 用户测试情况 | — |
| 32 | **open** | 08-13 | **默认配置在单个冷 256K prefill 时硬复位 GB10 head** | 与我们长 ctx 关注点同源 |
| 31 | closed | 08-13 | DEFAULT_THINKING=max 截断返回 content:null | 经 #48 重新实现 |
| 27 | closed | 08-12 | **x8 并发严重 decode lane 饥饿（1.10-21.69 tok/s，零抢占）** | 已修（2f180e7） |

---

## 4. 逐条对比分析（MiaAI 修复 ↔ 本项目经验）

### 4.1 A 类：生产（0.26）可直接借鉴的低风险正确性/稳定性修复

| # | MiaAI 修复 | 内容 | 与本项目关联 | 可借鉴性 |
|---|---|---|---|---|
| A1 | **NCCL GID 按节点注入**（分支 fix/worker-nccl-ib-gid-index，eac6b56） | RoCEv2 GID index 可在双 Spark 节点间不同、且**重启后漂移**；head .env 设 `WORKER_NCCL_IB_GID_INDEX`，start 脚本用 `REMOTE_NCCL_ENV="NCCL_IB_GID_INDEX=..."` 在**每次远程 compose 调用**注入（不依赖共享 .env）；未设时默认同 head；注释明确「单个共享 IPv4 index 同时用于两 rank 会以 `unhandled system error` 楔死 NCCL」。 | **直接命中我们 NCCL GID 不一致（head=3/worker=2，待整改）**。我们是 TP4 四机，GID 不一致风险更高。我们的 start_tp4_cluster.sh 目前是共享 .env 广播，正缺乏 per-node 覆盖。 | **A 类首推**。把「per-node GID override + remote compose 每次注入 env」移植到四机 start 脚本；`show_gids` 先各机确认。低风险（纯 env 注入）。 |
| A2 | **推理期 JIT RPC 超时防护**（issue #65，修复已在 issue 内给出） | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300` 默认值太短；NVFP4 MoE + MHC 混合 kernel 在 Blackwell sm_121a **按 shape 触发 JIT**（CuTeDSL W4A16FusedMoeKernel + TileLang mhc_pre_big_fuse_with_norm），新 shape 推理期编译可达分钟级 → `sample_tokens` RPC 超时 → vLLM v1 视为 worker 死亡 → **整引擎关闭**。修复：`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` + `TILELANG_CACHE_DIR` 绑定持久卷（TileLang cache 原来在容器可写层、重建即丢；CuTeDSL cache 已在持久 /tmp）。 | 我们在 GB10（sm_121）同样依赖 CuTeDSL JIT（NVFP4 cutlass MoE 通用内核），**推理期新 shape 编译风险同构**；且我们已有 decode autotuner fallback 的 JIT 类问题。生产若遇 worker 超时被误判死亡，`restart: unless-stopped` 只能自愈不能防。 | **A 类**：生产 compose 增加 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`（可覆盖）+ TileLang/CuTeDSL cache 目录持久化绑定。同时把「扩展启动 warmup 覆盖更多 shape」作为根治方向。 |
| A3 | **finish_reason 语义修复**（#55，db9b9ce） | max_tokens 截断工具调用时：流式/非流式都报 `finish_reason="length"`（was "tool_calls"），并丢弃不可解析 JSON 的 tool_calls。附带发现：**`FinishReason` 是 IntEnum 而非 StrEnum**，stock vLLM 的 `output.finish_reason == "stop"` 字符串比较从未命中（`tool_choice=required` 自然停止分支在上游是死代码），此 patch 顺带修了 ABORT/REPETITION 序列化为 '2'/'4' 的潜在 bug。 | 我们的网关/工具调用（8003 responses）存在相同截断语义风险；「poisons the conversation with 400s」链条与我们的 agent 会话韧性相关。0.26 生产若是同一 serving 层，此修复低风险高正确性收益。 | **A 类（若 0.26 有同类代码路径）**：monkey-patch 或等上游；注意 patch 是 serving 层、无上游改动。 |
| A4 | **stop 字符串在推理期休眠**（#55 之外，f2305b1 + 027b51a5） | 客户端 stop（如 harness 的 `Question:`）不再在 `<think>` 内部触发截断；spec-decode 跳过 </think>；默认开、`DSPARK_SUPPRESS_STOPS_IN_REASONING=0` 可关；同时**同步 hotfix 到 worker 并清 root 残留目录**（027b51a5 的运维细节）。 | 我们若用 harness stop 字符串 + thinking 提示词，存在相同「think 被截断 → content:null」风险。 | **A 类（按需）**：如生产 agent 用 stop 字符串则移植；否则低优先。 |
| A5 | **启动/停止编排纪律**（#38，27a55695） | 所有 .sh hotfix 在 compose entrypoint 内、**exec vllm 之前**应用（不在启动脚本里 up-d 后再 patch，避免在 weights 加载中重启造成 TCPStore 撕裂、worker exit 1、operator 卡 Stopping）；stop 用 `docker rm -f` 优先；compose `restart: unless-stopped` + `stop_grace_period=10s`。 | 我们四机 systemd 自愈 + 启动编排可对照：hotfix/配置变更应避免「先启动后热改」的竞态窗口。 | **A 类**：把「变更在 entrypoint 内先于 vllm 应用、stop 先 rm -f」写入四机启动/重部署 runbook。 |
| A6 | **镜像 digest-pin**（cbd719fe） | `.env.dspark.example` 固定镜像 digest（不可变清单），避免 tag 漂移。 | 我们多镜像环境（0.26/0.27/NG C26.07）依赖 tag，存在漂移风险。 | **A 类**：生产脚本一律 digest-pin。 |

### 4.2 B 类：0.27 测试环境借鉴（与我们 0.27 修复链同类/互补）

| # | MiaAI 修复 | 内容 | 与本项目关联 | 可借鉴性 |
|---|---|---|---|---|
| B1 | **decode lane 饥饿修复**（#27，2f180e7） | stock vLLM v1 scheduler 定义了 `max_num_partial_prefills` 却从未在 admission loop 读取；chunked prefill + async scheduling + max_num_seqs≥8 时，多个仍在 prefill 的请求各吃满 `max_num_batched_tokens`，decode 请求 `num_new_tokens==0` 被 `continue` **跳过（非抢占）** → 冷启动 x8 decode 0.36-2.07 tok/s。修复：① 等 inflight prefill 达到 `max_num_partial_prefills` 即停 admission；② `LONG_PREFILL_TOKEN_THRESHOLD=1024` 限制单 chunk。实测 x8 8K/16K/32K worst 从 2.07/0.47/0.36 → ~15 tok/s，0 抢占，MTP 96-99%。 | 与我们的**128K 长 ctx 并发断崖（≤c3 认知）同一调度机制范畴**。我们的 0.27 可能已有部分上游修复，但「decode 被 prefill 长 chunk 排挤」的机制在长 ctx 高并发下仍可能以不同形态存在。 | **B 类**：在 0.27 测试环境验证 `max_num_partial_prefills` 是否被正确读取 + `--long-prefill-token-threshold` 行为；与我们的 chunked-prefill 参数对照。 |
| B2 | **decode 公平性 floor + 每步调度诊断**（#43，f2d28bf） | 在 #27 之上增加：decode floor（每个 decode-active lane 保底 ≥1 步 token budget，排在后面前置请求需让路）、零 token decode-skip 记录（request_id, running_pos, num_computed_tokens）、每请求调度 token 记录、`DSPARK_ISSUE43_SCHED_DIAG=1` 每步一行摘要日志（默认关、零开销）。附带**纯 Python scheduler 模拟器**（无 GPU/vllm import）扫描 cap=1/2/3/8。 | 我们的 decode autotuner fallback 与 bench 数据缺诊断；「per-step 调度摘要」思路可移植到 0.27 测试环境定位 decode 断崖。 | **B 类（方法论）**：采纳「调度诊断日志 + 模拟器预验证」两个工具化做法。 |
| B3 | **混合前缀缓存 SWA min-hit 修复**（#26/#36，1f9765e2 + 7048daf3） | DSV4-Flash+DSpark 有 1×MLAAttentionSpec + 3×SlidingWindowMLASpec 四组 KV；`find_longest_cache_hit` 取各组 min，SWA 组按设计释放窗口外 block → 32K+ 命中塌到 0 → warm 全量重 prefill（warm wall 32K ~421s）。修复：SWA 命中长度不覆盖 curr_hit_length（v1，被 #36 纠正为「SWA 可收缩但用 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 稀疏化 checkpoint 保 warm hit」）。实测 x8 22.8K/44.7K/88.4K warm 命中 8/8。 | 我们 0.27 长 ctx 并发断崖的一部分可能与混合 KV 组前缀命中有关；「SWA 尾 evict MLA 前缀」的机制是值得验证的方向。 | **B 类**：0.27 测试环境检查 `HybridKVCacheCoordinator` 前缀命中是否受 SWA 组拖累；如适用可配 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`。 |
| B4 | **grammar-boundary 修复**（#24，backport 上游 #44993） | json_schema + thinking 组合下 grammar 损坏；修复 `should_advance` 用精确 delta token 窗口 + 记录 `reasoning_end_token_index`。实测 10/10 合法 JSON。 | 我们若用 structured output + thinking 组合，存在同类风险。 | **B 类**：0.27 已含 #44993 则验证即可；否则移植。 |
| B5 | **thinking_token_budget GPU 版（#31/#48）** | V2 GPU-resident Triton kernel 硬顶 reasoning token 数，无 host scan。 | 产品化能力；但**见 #66 警示**。 | **B/D（暂缓）**：`2689b1f` 已合入 main 但 **issue #66 证明 omit-field 路径仍复现 ~4.5 tok/s decode 断崖（间歇性）**。我们不应照搬；如确需该能力，需等 #66 修复并做 omit-field 全路径 A/B。 |

### 4.3 C 类：认知 / 方法论

| # | 项 | 内容 | 借鉴价值 |
|---|---|---|---|
| C1 | **CPU-only recipe gate**（5cdb9aa9） | push/PR 时跑 `scripts/ci-validate.sh`：shell 语法、patch 编译、关键单测、**拒绝重新上架已撤回的 hotfix**（#31/#34）、拒绝 #26 v1 版。 | 我们四机部署的 hotfix 应加「撤回黑名单 + 锚点校验」gate；防止旧补丁被无意回滚重放。 |
| C2 | **reproduce-issue*.py live harness + 纯 Python scheduler 模拟器**（#27/#43 配套） | 每个调度类 issue 带可复现协议脚本 + `tests/sim/` 无 GPU 模拟器，先模拟再上线。 | 我们已有 bench 协议，可补「调度级模拟器」对 128K 断崖做参数扫描（cap=1/2/3/8）再决定是否提并发。 |
| C3 | **no-regression audit 的双向教训**（#55 审计 + #66 反例） | #55 做 10-shape live regression sweep（严谨）；但 #48 的 A/B 只测了 budgeted 路径、漏了 omit-field 路径 → #66 复现断崖。 | **工程纪律**：任何性能相关 hotfix 必须覆盖「默认/未启用路径」做 A/B，不能只测开启路径；「没有 server-side default 注入」不代表 omit-field 无性能影响。 |
| C4 | **hotfix 生命周期规范** | idempotent、锚点校验、`--before/--after` 宿主侧校验、`--status`、entrypoint 内先应用、scp 同步 worker、env opt-out、镜像 digest-pin、CHANGELOG 逐日记录。 | 我们工程保障流程可对齐这套 hotfix 纪律（已部分采用 rollback-anchors 思路）。 |
| C5 | **per-step 调度诊断默认关、零开销**（#43 D） | 诊断日志默认关闭，不影响生产。 | 我们的诊断工具应默认零开销、按需开启。 |

### 4.4 D 类：不适用 / 暂缓

| # | 项 | 原因 |
|---|---|---|
| D1 | #48407 dense-prefill indexer（Stage A dormant） | 官方 0.27 main 线无 fork 锚点问题；且上游前提（有 dense-MHA 路由）与我们基线不同，照搬无意义。 |
| D2 | #46789 sequence parallelism / #48047（需 FlashInfer≥0.6.14）/ #49236（需 C++ op 重编）/ #48993（compact MXFP4 KV，unassessed） | feature 级或需镜像级改动；非本次借鉴范围。 |
| D3 | #57 DSpark 投机解码混合长度崩溃（ValueError uniform context lengths） | Dynamo standalone 部署、vLLM 0.21.1 基线，与我们 docker-compose TP4 不同；但**注意点**：若我们启用 `--speculative-config '{"method":"dspark",...}'` 且并发请求剩余上下文长度差异大，存在同类 proposer 限制崩溃风险，需先确认 0.27 是否已支持 ragged context（PATCHES.md 表明 fork 已修，上游待查）。 |
| D4 | thinking_token_budget hotfix | 见 B5/#66 — 暂缓，不照搬。 |

---

## 5. 借鉴清单（汇总，按 A/B/C/D）

### A 类 — 生产（0.26）可直接借鉴（低风险）
1. **A1 per-node NCCL GID 注入**（`WORKER_NCCL_IB_GID_INDEX` + remote compose env override）→ 整改我们 head=3/worker=2 的 NCCL GID 不一致。**首推。**
2. **A2 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` + TileLang/CuTeDSL JIT cache 持久化** → 防推理期 JIT 误杀引擎。
3. **A3 finish_reason=length 语义修复**（含 IntEnum≠StrEnum 潜在 bug）→ 工具调用截断正确性。
4. **A4 suppress-stops-in-reasoning**（按需，若用 harness stop 字符串）。
5. **A5 启动/停止编排纪律**（hotfix 先于 exec vllm、stop 先 rm -f、stop_grace_period）。
6. **A6 镜像 digest-pin**。

### B 类 — 0.27 测试环境借鉴
1. **B1 `max_num_partial_prefills` 强制执行 + `LONG_PREFILL_TOKEN_THRESHOLD=1024`**（与 128K 断崖同根，验证是否已含于 0.27）。
2. **B2 decode floor guardrail + per-step 调度诊断日志**。
3. **B3 混合 KV 组前缀命中（SWA min-hit）+ `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`**（长 ctx warm 命中验证）。
4. **B4 grammar-boundary #44993 验证**（structured output + thinking 组合）。
5. **B5 thinking_token_budget — 暂缓**（#66 复现断崖，等修复）。

### C 类 — 认知 / 方法论
1. **C1 hotfix 撤回黑名单 + 锚点校验 gate**（防旧补丁重放）。
2. **C2 调度级纯 Python 模拟器 + 可复现 live harness**（128K 断崖参数扫描）。
3. **C3 no-regression A/B 必须覆盖「默认/未启用路径」**（#66 反例）。
4. **C4 hotfix 生命周期纪律**（idempotent/锚点/before-after/entrypoint 内应用）。
5. **C5 诊断日志默认关、零开销**。

### D 类 — 不适用 / 暂缓
- #48407（dormant）、#46789/#48047/#49236/#48993（feature/镜像级）、#57（Dynamo 基线）、thinking_token_budget（#66 未修）。

---

## 6. 对生产修复计划的建议输入

1. **P0：NCCL GID 整改**（对应我们「待整改」项）
   - 采纳 A1：四机 start 脚本增加 per-node `WORKER_NCCL_IB_GID_INDEX`（默认同 head），远程 compose 每次注入 env override，不做 rank-unaware 的共享 .env 覆盖。
   - 先行 `show_gids` 确认四机实际 GID；参考 MiaAI 注释「index 重启后会漂移」，自愈脚本/systemd 应在每次 start 时重新注入，而非写死一次。

2. **P0：JIT 超时防护**（对应我们 decode autotuner fallback 之外的稳定性缺口）
   - 生产 compose 设置 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`；将 TileLang（`~/.tilelang/cache`）与 CuTeDSL cache 目录绑定到持久卷，避免容器重建丢 JIT 缓存。
   - 中长期：扩展启动 warmup 覆盖更多 MoE/batch shape（Anemll 侧建议），减少推理期 JIT 概率。

3. **P1：finish_reason / stop 语义**
   - 若 0.26 生产 serving 层有 #55 同类路径，移植 finish_reason=length + 丢弃非法 JSON args（A3）。
   - 若 agent harness 使用 stop 字符串，评估 A4 suppress-stops-in-reasoning。

4. **P1：128K 长 ctx 断崖的调度根因验证**（将「≤c3 认知」从经验升级为机制理解）
   - 0.27 测试环境验证 `max_num_partial_prefills` 读取与 `--long-prefill-token-threshold` 实际行为（B1）；
   - 用 C2 模拟器对四机参数做 cap=1/2/3/8 扫描，评估是否值得升级 base image 以支持并发 partial prefill（MiaAI #45 数据：cap=2 → 62Kx8 min/max 0.42→0.63；cap=3 → 0.72）。

5. **P2：工程纪律落地**
   - hotfix 撤回黑名单 + 锚点校验 gate（C1）；
   - 所有性能相关 hotfix 的 A/B 覆盖「默认/未启用路径」（C3，吸取 #66 教训）；
   - 镜像 digest-pin（A6）；
   - 诊断日志默认零开销、按需开启（C5）。

6. **Watch items（MiaAI 未修，可作前瞻监控）**
   - #52 assistant 结尾无 generation header → 空回合自续循环（agent 会话韧性）。
   - #32 单冷 256K prefill 硬复位 GB10 head（我们长 ctx 关注点同源）。
   - #37 reasoning_effort=high + tools 退化（若生产走 tools+high 路径需验证）。
   - #66 thinking_token_budget 断崖修复进展（若未来需要该产品化能力）。

---

## 7. 数据来源与局限

- 所有 commit/issue 数据来自 GitHub REST API（2026-08-15 抓取）：`commits?per_page=30`、`issues?state=all&per_page=40&sort=created`、`branches`、`tags`、`releases`、`contents/docs/vllm-027-new-patches.md`、`contents/CHANGELOG.md`、`contents/docs/PATCHES.md`。
- 局限：① MiaAI 基线为 vLLM 0.25.2（Anemll 0.1.1）TP2，我们为 0.27 TP4，代码路径有差异，B 类项需在本环境验证后再定是否移植；② A 类中的 monkey-patch 需对照我们 serving 层源码锚点确认；③ issue #65/#66 的修复目前只在 issue 描述/分支中，尚未全部合入 main，落地时以最新 main 为准。
