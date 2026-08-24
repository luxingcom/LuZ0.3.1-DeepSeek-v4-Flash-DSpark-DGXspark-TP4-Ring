# SGLang NVFP4 测试环境验证策略与测试计划

**日期**：2026-08-13
**作者**：泰莎（Tessa）· 测试专家（工程保障团队）
**对象**：DGX Spark 四机集群（AICAD）TP4 环网上的 SGLang + DeepSeek-V4-Flash NVFP4 测试环境
**基线**：vLLM TP4 生产（anemll 0.2.1-v026.0，MXFP4 0731 原版，已有完整 tp4-* 基准报告）
**性质**：四层验证策略（权重→运行时→通信→性能）+ 用例清单 + 数据格式 + 风险盲区 + 行动清单

---

## 0. TL;DR

| 层 | 目标 | 核心结论/判据 | 用时估算 |
|---|---|---|---|
| L1 权重正确性 | 转换后、部署前的 NVFP4 产物可信 | bit-exact verify + **全量补扫**（verify 仅采样 64/35k 专家，不足）；scale 分布对照；布局不混用；精度冒烟 | 0.5~1 天 |
| L2 运行时正确性 | SGLang 起服务后输出可信 | 冒烟 + 精度对比（vs vLLM MXFP4）+ **TP4 输出 sanity check（防 garbage，vLLM CUTLASS 教训）** + MTP/DSPARK 接受率 | 0.5~1 天 |
| L3 环网通信 | 补丁原样适用 SGLang、通信正确 | NCCL 版本/PEER_HCA/ring 日志、all-reduce 0 误差、DeepEP NCCL fallback 行为实测、shim 绑核对 SGLang 进程名适用性 | 0.5 天 |
| L4 性能 A/B | SGLang NVFP4 vs vLLM MXFP4（+NVFP4 若有） | 复用 bench_prefill_decode_async.py 口径（per-request p50，**禁用 agg_* 跨组对比**）；prefill 期望 ≥1.1×，decode 期望 ≈1.0×（latency-bound） | 1~2 天 |

**三条红线**：
1. 跨组对比只用 per-request p50×并发，禁用 agg_*（asyncio prefill 串行）。
2. 任何 NVFP4 输出必须过内容 sanity check（vLLM CUTLASS grouped GEMM SM120 垃圾输出教训）。
3. SGLang TP4 与生产 vLLM TP4 **不可同机并存**（统一内存 ~79GiB/机已占满），必须错峰测试窗口。

**最大前置阻塞**：SGLang 对 DeepSeek-V4-Flash-NVFP4 的支持依赖 PR #25820 合入状态、SGLang 版本与容器选型（NGC 26.02 或主线自建）——C0/R0 预检必须最先做，未确认前不启动任何部署。

---

## 1. 测试目标与范围

### 1.1 被测对象
- **运行时**：SGLang（NVFP4 原生路径：flashinfer_fp4_gemm / modelopt_fp4 量化；SM120 动态检测；DeepSeek-V4-Flash 架构适配）
- **模型**：DeepSeek-V4-Flash-0731（33K routed experts / MLA / MTP / DSPARK），NVFP4 权重
- **拓扑**：4×DGX Spark（GB10 SM121）纯环网 01—02—04—03—01，NCCL ring-only 补丁（LD_PRELOAD /opt/nccl-ringonly + ALGO=RING + PEER_HCA per-peer + SUBNET_AWARE_ROUTING=1 + NET_PLUGIN=none + MERGE_NICS=0），NCCL 2.30.7（LD_LIBRARY_PATH 前插），QoS DSCP trust
- **对照基线**：vLLM TP4 生产（MXFP4 原版，`deliverables/engineering-assurance/tp4-r8-final-report-2026-08-12.md`、`tp4-r12-final-report-2026-08-13.md`、`tp4-opt-execution-final-2026-08-13.md`）

### 1.2 权重两条路线（验证均覆盖）
| 路线 | 来源 | input_scale | MTP | 佐证 |
|---|---|---|---|---|
| A 本地转换 | 官方 0731 MXFP4（48 shards 已在 01/02）→ `transcode_0731_to_nvfp4.py`（tsarihan 工具） | 固定 1.0（纯代数 2^-G） | **全转**（2026-08-09 起） | 工具自带 verify 子命令 + 需补全量扫 |
| B 预转换下载 | Rarri / MJPansa 0731-NVFP4（若已下载） | Rarri=ModelOpt 0.45 官方流程 / MJPansa=500K token 校准 | Rarri **全转**（mtp_nvfp4_build_report.json）/ MJPansa **不转** | conversion-receipt.json / mtp 报告 |

> ⚠️ MTP 处理方向相反是已知关键分歧：tsarihan 实证"不转 MTP → draft 接受率 0.31→0.121 崩溃"；MJPansa 在 vLLM 0.26.1"不转可运行"但无接受率数据。**SGLang 侧 DSPARK 的 MTP 策略必须独立 A/B**（见 R9）。

### 1.3 测试环境边界
- 测试仅限独立窗口（见 §6.4 资源争抢控制）；canary 03/04 仅可承载轻量验证（内存余量 ~28G），不可承载完整 rank。

---

## 2. 测试分层总览

```
L1 权重正确性（部署前，纯 CPU/磁盘）
   └─ 转换执行 → 三验证（bit-exact / scale 分布对照 / 精度冒烟）→ 布局不混用确认 → 分发校验
L2 运行时正确性（SGLang 起服务后）
   └─ 冒烟 → 精度对比 vs vLLM MXFP4 → TP4 一致性 + 输出 sanity → MTP/DSPARK 投机
L3 环网通信验证
   └─ NCCL 初始化日志 → 通信正确性 → 与 vLLM 经验对照（DeepEP NCCL fallback）→ 绑核/断链
L4 性能 A/B
   └─ 预检 → 核心矩阵 → TTFT/TPOT 分解 → decode 长度档 → vs vLLM 基线 → 判定
```

**执行顺序**：L1 全绿 → L2 → L3（可与 L2 并行观察）→ L4。任何一层 FAIL 即中止后续（权重错不部署、输出错不压测、通信错不 A/B）。

---

## 3. L1 权重正确性验证（转换后、部署前）

### 3.1 工具与命令（tsarihan transcode_0731_to_nvfp4.py）

工具已本地化：`delivery/nvfp4-investigation/transcode_0731_to_nvfp4.py`（516 行）。子命令：

```bash
python3 transcode_0731_to_nvfp4.py transcode SRC_DIR OUT_DIR [--shards lo:hi] [--resume]
python3 transcode_0731_to_nvfp4.py config    SRC_DIR REF_DIR OUT_DIR   # 合并 config.json（取 REF 的 quantization_config）
python3 transcode_0731_to_nvfp4.py verify    SRC_DIR OUT_DIR           # bit-exact 校验（默认采样 64 个专家）
python3 transcode_0731_to_nvfp4.py index     SRC_DIR OUT_DIR           # 重写 model.safetensors.index.json
```

转换语义（须在计划中固化，供核对）：
- 专家权重：I8 字节拷贝 → U8（**字节不变**）
- 专家 scale：E8M0 [R,C]（block-32）→ E4M3 [R,2C]（block-16，每字节复制成两半）；`G13=8-max(E8M0 exp over w1∪w3)`、`G2=8-max(exp over w2)`；`weight_scale_2 = 2^-G`（F32 标量）、`input_scale = 1.0`（F32 标量）；全零块 → 0x01
- 非专家张量（attn FP8 / shared FP8 / norms BF16 / embed / hc_* / MTP 若保留）→ **verbatim 拷贝**
- **MTP 全转**（本工具当前版本，`parse_expert` 将 mtp.B 映射到 L=1000+B，避免 draft 双 scale 约定崩溃）

### 3.2 用例清单（W 系列）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **W0** | 环境/工具预检 | 容器内（vllm-dsv4:src-sm121，torch/safetensors/numpy，纯 CPU）运行 `transcode_0731_to_nvfp4.py` 无参 → 打印 4 子命令用法；确认 SRC 48 shards/156G 在位；确认 OUT 磁盘 ≥250G | 4 子命令可见；SRC 文件数/大小符合；磁盘余量满足 |
| **W1** | 转换执行 | `transcode ... transcode SRC OUT --resume`；记录 pass1 G13/G2 range、pass2 每 shard 专家/verbatim 计数、最终 `DONE: 138,365 tensors / ~180GiB / 48 shards` | 退出 0；48 shards 全出；`_g_dict.json` 落盘；无 E8M0 0xFF abort、无 G 越界报错 |
| **W2** | config 合并 | `transcode ... config SRC REF OUT`（REF=参考 NVFP4 config.json，来自 nvidia/MJPansa 或 modelopt 约定）；`jq` 校验关键字段 | `quantization_config.moe_quant_algo == "NVFP4"`；`expert_dtype`/`scale_fmt` 正确；`dspark_block_size=5` 保留；架构字段与 0731 一致；index.json 可解析且 weight_map 键数=138,365 |
| **W3** | bit-exact verify（工具自带） | `transcode ... verify SRC OUT`；记录采样数/OK/BAD | 退出 0；`64 OK / 0 BAD`（默认 n_samples=64，种子 12345） |
| **W4** | **全量 bit-exact 补扫（必做）** | verify 仅采样 64/35,328 专家投影，**不足**。写补扫脚本：遍历**全部**专家投影（主层 43×3×256=33,024 + MTP 3×3×256=2,304 = 35,328），逐块校验 ①weight 字节一致 ②scale 关系 `(b4>>3)-7 == (b-127)+G` ③两半一致 ④全零→0x01；MTP 键（L≥1000）单列统计 | 0 失配；输出按 (L,E,wX) 计数表；MTP 单独 0 失配（见附录 A 脚本骨架） |
| **W5** | scale 分布对照 | 若 REF/报告可得：对照 MJPansa conversion-receipt.json 或 Rarri mtp_nvfp4_build_report.json 的 scale 分布；输出本地 `_g_dict.json` 的 G13/G2 range、weight_scale_2=2^-G 直方图 | 分布形态一致（G range 对齐）；与 REF 的差异**显式记录**（input_scale：本地=1.0，MJPansa=校准非 1.0，属预期差异非缺陷） |
| **W6** | **NVFP4/MXFP4 布局不混用确认** | ①config 校验（同 W2）；②扫描专家张量命名：应含 `.weight_scale`(F8_E4M3)/`.weight_scale_2`(F32)/`.input_scale`(F32)，**不得残留 `.scale`(F8_E8M0)**；③shape 校验：`weight_scale [R,2C]`（block-16）vs 源 `[R,C]`（block-32）；④随机抽 3 shard 用 safetensors 加载，核对 dtype 为 U8/F8_E4M3/F32 | 0 个 MXFP4 风格专家 scale 残留；block-16 确认；verbatim 张量与源字节一致；dtype 符合 |
| **W7** | 精度冒烟（CPU/transformers） | 用 HF Transformers `grouped_mm`/`batched_mm`（消费 float32 scales，无 SM120 FP4 硬件路径但可作正确性基准）加载 20 条固定 prompt，输出 logits 对比原版 MXFP4 | top-1 token 一致率 ≥0.99（前 32 token）；logit RMS diff < 阈值；无 NaN |
| **W8** | 产物完整性 + 分发校验 | ①75 文件清单（48 shard+config+index+tokenizer+报告）；②sha256 抽样（≥8 shard+config+index）；③01/02 NFS 导出 → 03/04 挂载后四机读校验 | 文件数/大小对齐；sha256 通过；四机可读（`head -c 1 <shard>` 成功）；NFS ro 权限正确 |

**W 层交付物**：`_archive_scratch/_tessa_sglang_bench/<date>/weights/`（verify 日志、_g_dict.json 副本、全量补扫报告、W5 对照表、sha256 清单）。

---

## 4. L2 运行时正确性验证（SGLang 起服务后）

### 4.0 预检（R0，阻塞项）
| 项 | 检查 | 判定 |
|---|---|---|
| PR #25820 合入状态 | GitHub 核验 DeepSeek-V4-Flash-NVFP4 支持是否在主线的 SGLang 版本内 | 合入→用主线/含该 PR 镜像；未合入→评估补丁或换权重路线 |
| SGLang 版本/容器 | NGC 26.02（SGLang 0.5.8 + flashinfer 0.6.1）或主线自建；`sglang.launch_server --help` 抓取真实 flag | 关键 flag 存在：`--moe-runner-backend`（含 flashinfer_mxfp4/fp4_gemm 取值）、`--speculative-algorithm DSPARK`、DeepEP/`--quantization modelopt_fp4`、多机 `--nnodes/--node-rank/--dist-url` |
| FlashInfer SM121 | `FLASHINFER_CUDA_ARCH_LIST=12.1a`；`SGLANG_DISABLE_DEEP_GEMM=1`（SM120 无 DeepGEMM） | NVFP4 GEMM 走 flashinfer_fp4_gemm；DeepGEMM 不触发 |
| 启动参数基线 | 对照 vLLM 生产（见 §1.2 服务部署指南）：kv-cache-dtype、max-model-len、cudagraph、prefix-cache | 记录 SGLang 等价参数；差异显式入表 |

### 4.1 冒烟与基础行为（R1-R4）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **R1** | 服务拉起/就绪 | head-first 四机启动；`/health` + `/v1/models` | 4/4 rank ready；日志 TP=4；model 名/served 正确；max_model_len 符合配置 |
| **R2** | 单请求冒烟 | 3 prompt（coding/json/prose）× max_tokens=64，temperature=0.6 | HTTP 200；content 非空；finish_reason=stop；无 NaN/UNK/garbage |
| **R3** | 多请求并发冒烟 | 8 并发混合任务，max_tokens=64 | 全部成功；无 hang；无 crash |
| **R4** | 生成长度/stop 行为 | max_tokens=128 → 核对 usage.completion_tokens ≤128+ε；stop=["\n\n"]；tool-call 场景 finish_reason | 长度被尊重；stop 生效；tool-call finish_reason 正确 |

### 4.2 精度对比与输出 sanity（R5-R7，核心）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **R5** | **输出内容 sanity check（防 garbage，硬门槛）** | 对 R2-R4 全部输出跑启发式检测：①token 级 NaN 不可见但检测 repetition loop（≥4 个相同 4-gram）②UNK 占比 ③语言中途切换 ④输出 token 数异常（0 或 1，或远超 max_tokens）⑤首 token 与 prompt 语言不符 | 0/20 触发任何启发式；**任何触发即判 FAIL 并停 L4**（参照 vLLM CUTLASS grouped GEMM SM120 垃圾输出教训——silent garbage 不报错） |
| **R6** | 精度对比 SGLang NVFP4 vs vLLM MXFP4 | 固定 20-prompt 集（coding/json/prose/GSM8K 型各 5），temperature=0（greedy）主测 + 0.6 辅测；指标：greedy top-1 token 一致率、ROUGE-L、1-4 gram overlap、确定性任务 exact match；若 API 暴露 logprobs 再比 top-k 分布 | greedy 一致率 ≥0.90（<0.85 判 FAIL）；无系统性 garbage；确定性答案 exact ≥0.8；量化格式不同（E4M3×2^-G vs E8M0）允许非逐位一致但须语义等价 |
| **R7** | TP4 分布式一致性 | 方式 A：同权重 SGLang TP1（单机）5 短 prompt vs TP4 输出；方式 B：TP4 temperature=0 同一请求重复 3 次 → 应确定性一致；方式 C（可行时）：per-rank 日志核对采样 token | TP4 vs TP1 greedy 输出一致（5 prompt）；TP4 重复 3 次全同；无 NaN/garbage；**任何 rank 相关不一致即 FAIL** |

### 4.3 MTP/DSPARK 投机（R8-R10）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **R8** | 长上下文无 hang/无超时 | 16k/32k ctx × max_tokens=256 × 4 并发 | 全部在预算内完成；TTFT/TPOT 无数量级 stall |
| **R9** | **MTP/DSPARK 接受率** | 确认日志加载 DSPARK/MTP 与 num_speculative_tokens；测 draft 接受率（SGLang 指标导出/日志解析/受控探针三选一，需 R0 确认机制） | **接受率 ≥0.40**（Rarri vLLM NVFP4 实测 49.4%）；**<0.20 即 FAIL**（对应 tsarihan 0.121 崩溃阈值，提示 MTP/scale 混用）；对比 SPEC_DECODE=off 基线 |
| **R10** | 精度冒烟 vs 已知答案 | GSM8K 抽样 50 题 greedy | 准确率 ≥ vLLM MXFP4 基线 −5pp；无系统性退化 |

### 4.4 API/缓存兼容（R11-R12）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **R11** | prefix-cache 行为 | 相同前缀重复请求 → 输出一致；bench 随机前缀（uuid4）应天然避开缓存 | 无跨请求内容污染；随机前缀确实绕过缓存 |
| **R12** | bench 脚本 API 兼容 | `/v1/chat/completions` 流式 + `stream_options.include_usage` 返回 usage.prompt_tokens/completion_tokens | 校准成功；usage 齐全（脚本依赖，否则 bench 无法跑） |

---

## 5. L3 环网通信验证

### 5.1 NCCL 初始化日志（C0-C1）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **C0** | NCCL 版本/补丁生效 | `docker logs <sglang-rank0> | grep -i "NCCL version"`；核对 LD_PRELOAD 生效 | banner=`2.30.7+cuda13.0`（补丁版）；**出现 13.3 = LD_PRELOAD 未生效，立即排查** |
| **C1** | per-rank IB peer 对/ring 路径 | 解析每 rank NCCL_DEBUG_FILE（`NCCL_DEBUG=INFO`）：PEER_HCA override 是否应用（sendSetup/recvSetup 到环邻对）、ring 路径 01-02-04-03-01、无 `ibv_modify_qp 110` | 每 rank 只连环邻对；0×110；4/4 rank；ring 顺序与 PEER_HCA 表一致（rank0→1,3；rank1→0,2；rank2→1,3；rank3→0,2） |

### 5.2 通信正确性（C2-C3）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **C2** | all-reduce/bcast/all-gather 已知张量 | 复用 nccl_probe（world=4，参照 TP4_prefill_bench_plan C4），张量 128M/512M/1G | 误差=0；记录带宽 GB/s 作基线 |
| **C3** | 无超时/重传异常 | 在 L4 全矩阵期间：NCCL 日志 grep timeout/retry；ethtool 16 口计数零增量；nvidia-smi link 错误 | 0 timeout/retry；物理层计数 0 增量 |

### 5.3 与 vLLM 经验对照（C4-C6，SGLang 特有）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **C4** | 环网补丁 env 原样适用性 | SGLang 启动沿用 vLLM 的 env 全集（LD_PRELOAD ring-only+shim、NCCL_ALGO=RING、PEER_HCA per-rank、SUBNET_AWARE_ROUTING=1、NET_PLUGIN=none、MERGE_NICS=0、TOS=46、GID_INDEX=2、SOCKET_IFNAME）；确认 SGLang node-rank 0..3 映射 = 01/02/04/03 | SGLang TP4 在 ring-only 补丁下正常初始化；0×110；PEER_HCA 按 rank 生效 |
| **C5** | **DeepEP NCCL fallback 行为** | ①关闭 DeepEP（TP4 纯 torch NCCL all-reduce）为主路径 → 全绿；②开启 DeepEP（`--enable-deepep-moe`）观察：EP communicator 是否在环网初始化、结果是否正确、有无 NVLink 假设报错 | 主路径（无 DeepEP）正确即可放行；DeepEP 若报 NVLink 相关错误 → 记录为"不支持/回退 TP-only"，**不得因此卡 L4**；DeepEP 可用则加跑正确性探针 |
| **C6** | shim 绑核对 SGLang 进程名适用性 | PSR 采样：NCCL 线程应落 8-9；SGLang 主线程（scheduler/tokenizer/detokenizer/server）PSR 分布 | NCCL 线程 8-9（应随 LD_PRELOAD 生效）；SGLang 主线程若未落 15-19（shim v8 按 vLLM "EngineCore" 进程名 pin）→ 用 taskset/cpuset 显式补绑，记录偏差（风险 R-04） |
| **C7** | 断链/恢复（可选） | 压测中断 1 rank 网络 → 观察客户端超时（非挂死）→ 恢复 → 复测 1 组合抽查 | 客户端超时而非 hang；恢复后吞吐 ±5% |

### 5.4 数据面隔离（C8）

| ID | 用例 | 步骤 | 通过标准 |
|---|---|---|---|
| **C8** | 与 vLLM 数据面隔离 | MASTER_PORT 与 vLLM 25999 区分（如 25998）；确认测试窗口内 vLLM 已停（见 §6.4）；无端口/接口冲突 | 无冲突；`ss -ltnp` 核对监听 |

---

## 6. L4 性能 A/B 测试

### 6.1 基准口径（铁律，复用 bench_prefill_decode_async.py）

- 脚本：`bench_prefill_decode_async.py`（asyncio wave 真并发，每组合 = ROUNDS 波 × conc 并发在飞）
- 指标：**per-request p50**（`p50_prefill_tps / p50_decode_tps / p50_ttft_s / p50_total_s / p50_tpot_s`）
- **禁用 agg_prefill_tps/agg_decode_tps 做跨组对比**（asyncio prefill 串行缺陷）
- 随机前缀强制（uuid4）防 prefix-cache；temperature=0.6；3 轮取中位
- 每次运行前 `--sanity-log`（若 head 日志可指定）做 NCCL init 判定，FAIL 即中止

```bash
# 组 S（SGLang NVFP4，head <MGMT_OCTET>:8001 或测试端口）
python bench_prefill_decode_async.py --group S \
    --endpoint http://<NODE_IP>:<sglang_port>/v1 \
    --key <key> --model deepseek-v4-flash-0731-nvfp4 \
    --concurrency 1,4,8,16,32 --ctx 16384,32768,65536 \
    --tasks coding,json,prose --rounds 3 --engine asyncio --out ./_archive_scratch/_tessa_sglang_bench/<date>/S
```

> 注：SGLang 服务端需 `max running seqs ≥ 32` 才能测满 conc=32（vLLM 生产 max-num-seqs=6 是其并发上限，SGLang 测试实例按需调高——这是**测试实例参数**，非生产参数）。

### 6.2 测试矩阵（P 系列）

| ID | 用例 | 输入 | 输出 |
|---|---|---|---|
| **P0** | 预检 | aiohttp 依赖；endpoint/models 可达；SGLang max-running-seqs 支持 conc=32；vLLM 已停/错峰确认 | 全部通过才开跑 |
| **P1** | 校准 | 脚本内置 calibrate（非流式短请求读 usage） | tokens/unit≈记录；失败即中止 |
| **P2** | SGLang NVFP4 核心矩阵 | ctx 16384/32768/65536 × conc 1/4/8/16/32 × coding/json/prose × 3 轮（45 组合） | `rows_S.csv` + `summary_S.json`（含 p50_ttft/tpot 分解） |
| **P3** | TTFT/TPOT 分解 | 从 summary 提取 p50_ttft_s / p50_tpot_s 逐组合 | 记录；与 vLLM 基线同口径对比 |
| **P4** | decode 长度档（128/1024 输出） | 脚本 `TASK_MAX_TOKENS` 固定（coding/json=512、prose=256），需小改：新增 `--max-tokens-override` 或加 long_decode task（max_tokens=1024）；固定 ctx 16384，conc 1/4/8 | decode 128 与 1024 两档吞吐/TPOT 记录 |
| **P5** | vLLM TP4 基线对照 | 复用既有 tp4-r8/r12/opt-execution 报告（MXFP4）；若有 vLLM NVFP4 数据一并入表；如需重跑：同脚本 `--group V --concurrency 1,3,5 --ctx ...`（受 vLLM max-num-seqs=6 限制，conc 仅 1/3/5 严格可比） | 对比表（每格 = 同 ctx×task×conc 的 p50 比值） |
| **P6** | 对比维度与判定 | prefill 吞吐、decode 吞吐、TTFT、端到端、TPOT；并发收益曲线（16K-32K 分界观察） | 判定见 6.3 |
| **P7** | 资源争抢控制 | 测试窗口 = 停 vLLM（或降 TP2 至 01/02）；SGLang 容器 `--cpuset-cpus 1-19`；NCCL 8-9；主线程 pin；显存预算核算（~79GiB/机）；前后各跑 vLLM 恢复验证（/health + 1 冒烟） | 无 OOM；无争抢；测试后 vLLM 恢复 100% |
| **P8** | 报告 | 生成 `bench-sglang-<cfg>-<date>.md`（表格+比值+判定）；raw 归档 | 落盘 `deliverables/engineering-assurance/` + `_archive_scratch/` |

### 6.3 预期收益定位与判定阈值

| 维度 | SGLang NVFP4 vs vLLM MXFP4（TP4 同口径） | 判定 |
|---|---|---|
| prefill 吞吐（大 ctx/带宽饱和档） | 期望 ≥1.1×（tsarihan cu130 实测 1.14-1.32×）；**目标 1.1×** | ≥1.05× 记 PASS（prefill 主战场） |
| decode 吞吐（latency-bound） | 期望 ≈1.0×，提升有限（与既有结论一致） | ≥0.95× 硬门槛（不得劣化 >5%） |
| TTFT | 不劣化 | ≤ 基线 ×1.1 |
| 并发收益 | 分界 16K-32K ctx；131072 decode 崩塌 = TTFT 阻塞非带宽（沿用既有结论） | 记录曲线，不判胜 |
| 硬门槛 | L2 全过 + 无超时 + 无 OOM | 任一 FAIL 则 A/B 结论作废 |

> 判定法沿用既有：3 轮中位数 + 90% 置信区间，比率下界 >1.0 才判胜；**只对比同 ctx×task×conc 的 per-request p50**。

---

## 7. 数据记录格式与输出目录

| 类型 | 目录 | 命名约定 |
|---|---|---|
| 本测试计划 | `deliverables/engineering-assurance/` | `sglang-nvfp4-test-plan-2026-08-13.md` |
| 性能报告 | `deliverables/engineering-assurance/` | `bench-sglang-<cfg>-<date>.md`（沿用 `bench-*`/`tp4-*` 系列） |
| 权重验证交付物 | `_archive_scratch/_tessa_sglang_bench/<date>/weights/` | `verify-<...>.log`、`fullscan-report-<...>.json`、`g_dict-<...>.json`、`scale-dist-<...>.csv`、`sha256-<...>.txt` |
| 运行时验证 | `_archive_scratch/_tessa_sglang_bench/<date>/runtime/` | `smoke-<...>.jsonl`、`sanity-<...>.json`、`precision-<...>.json`、`accept-<...>.csv` |
| 性能 raw（脚本产物） | `_archive_scratch/_tessa_sglang_bench/<date>/S|V|VN/` | `rows_<group>.csv` + `summary_<group>.json`（脚本原生格式） |
| 服务器端 scratch | `<INSTALL_DIR>/logs/sglang-test/`（NCCL_DEBUG_FILE、启动日志） | `nccl-<host>.log`、`sglang-rank<0-3>.log` |

**CSV/JSON 字段**（脚本原生，不自行发明）：`group, ctx, task, concurrency, wave, ok, prompt_tokens, completion_tokens, ttft_s, total_s, prefill_tps, decode_tps, err`；summary 含 `p50_prefill_tps / p50_decode_tps / p50_ttft_s / p50_total_s / p50_tpot_s / agg_*（仅记录，禁止跨组对比用）`。

---

## 8. 风险与盲区

### 8.1 风险表
| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R-01 | PR #25820 未合入 → SGLang 无法加载 DSV4-Flash-NVFP4 | 高 | R0 预检最先做；未合入则评估补丁或回退 MXFP4 路线 |
| R-02 | DeepEP 依赖 NVLink，环网上 NCCL fallback 失败/极慢 | 高 | C5 双路径：主路径禁用 DeepEP（TP-only）可放行；DeepEP 报错记录为不支持，不阻塞 L4 |
| R-03 | shim v8 按 vLLM "EngineCore" 进程名 pin，SGLang 主线程绑核不生效 | 中 | C6 实测 PSR；taskset/cpuset 显式补绑；偏差入报告 |
| R-04 | verify 子命令仅采样 64/35,328 专家 → 漏检 | 中 | W4 全量补扫（必做） |
| R-05 | input_scale=1.0（本地转换）vs 校准（MJPansa）精度差异 | 中 | W7 精度冒烟 + R6 输出对比门槛 |
| R-06 | MTP 策略（转/不转）× SGLang DSPARK 兼容性未验证 | 中 | R9 接受率门槛（≥0.40，<0.20 FAIL）；必要时切权重路线 |
| R-07 | 与生产 vLLM 资源争抢（统一内存不可并存） | 高 | P7 测试窗口 + 错峰 + 前后恢复验证 |
| R-08 | flashinfer JIT 在 SM121 重编译耗时（~35min release）与 arity-skew 坑 | 中 | 复用 `FLASHINFER_JIT_DEBUG=0` + fused_moe 修复（tsarihan serve-wrap-jitfix 经验）；预构建 .so 缓存 |
| R-09 | cuBLAS 13.4.0 NVFP4 tensor-wide scaling bug（仅 L2 cu132 场景） | 中 | SGLang 用 cu130 或 cu132 时规避 13.4.0；自建镜像 ≥13.2.2/13.4.1 |
| R-10 | NVFP4 silent garbage（vLLM CUTLASS SM120 教训重演） | 高 | R5 输出 sanity check 硬门槛 + R7 跨 rank 一致性 |
| R-11 | 协议/template 差异（SGLang vs vLLM tokenizer/chat-template）污染精度对比 | 中 | R6 同 prompt 集 + 同 template 参数；差异显式记录 |

### 8.2 盲区（已知无法覆盖或需上游输入）
1. **SGLang 容器本地未就绪**：CLI flag 均以"R0 预检确认"为准，本文不臆造最终参数。
2. **Draft 接受率在 SGLang 的度量机制未定**：vLLM 有现成指标；SGLang 可能需要日志解析或受控探针（R9 列为待确认项）。
3. **per-rank 可观测性有限**：TP4 跨 rank 一致性只能通过"TP1 vs TP4 + 重复确定性"间接验证，无法逐 rank 注入检查（除非加探针补丁）。
4. **DeepEP 环网数据缺失**：vLLM 经验不直接覆盖 SGLang 的 DeepEP 路径，只能实测（C5）。
5. **基准数据对齐**：vLLM 基线 conc 上限 6（max-num-seqs），与 SGLang conc=32 的矩阵只有 1/4 两档严格可比——大并发档只能做"绝对数值参考"而非严格 A/B。
6. **MTP 接受率参照系**：Rarri 49.4% 是 vLLM 数值，SGLang 接受率独立测量，仅作量级参照。

---

## 9. 行动清单

| 优先级 | 行动 | 负责 | 前置 |
|---|---|---|---|
| P0 | 核验 SGLang PR #25820 合入状态 + 选容器（NGC 26.02 / 主线） | architect/sre | — |
| P0 | 准备 SGLang 启动脚本（复用 start_tp4_*.sh 的 NCCL env 全集，MASTER_PORT 区分 25998） | sre | P0-1 |
| P0 | 本地转换执行 + W0-W8 权重验证（含全量补扫脚本落地，附录 A） | testing（本文档执行） | 权重源 48 shards |
| P1 | R0/C0 预检（flag 确认、FlashInfer SM121、NCCL banner） | sre/testing | SGLang 容器 |
| P1 | L2 运行时验证 R1-R12 | testing | L1 全绿 |
| P1 | L3 环网验证 C1-C8 | sre/testing | L2 可跑 |
| P2 | L4 性能 A/B（P0-P8，独立测试窗口） | testing/team-lead 排期 | L2/L3 全绿 |
| P2 | bench 脚本小改：`--max-tokens-override`（P4 需要） | testing | — |
| P3 | 报告归档 + Runbook 回填 + 回滚锚点（SGLang 镜像/权重 tag） | tech-writer/sre | L4 完成 |

---

## 附录 A：W4 全量 bit-exact 补扫脚本骨架（可直接使用）

基于 `transcode_0731_to_nvfp4.py` 的 verify 逻辑扩展到全量（去掉采样、覆盖 MTP 键 L≥1000）：

```python
# fullscan_nvfp4.py  —— 在容器内（torch/safetensors/numpy）运行
# 用法: python3 fullscan_nvfp4.py SRC_DIR OUT_DIR
import os, sys, json, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcode_0731_to_nvfp4 import read_header, read_tensor_bytes, parse_expert

def scan(src_dir, out_dir):
    with open(os.path.join(out_dir, 'model.safetensors.index.json')) as f:
        out_idx = json.load(f)['weight_map']
    with open(os.path.join(src_dir, 'model.safetensors.index.json')) as f:
        src_idx = json.load(f)['weight_map']
    exp = sorted({(p[0], p[1], p[2]) for n in src_idx if (p := parse_expert(n)) and p[3] == 'scale'})
    cache = {}
    def get(d, idx, name):
        sh = idx[name]
        if sh not in cache:
            cache[sh] = None  # (dir,path,header,base)
        return cache[sh]
    # —— 加载缓存（每个 shard 一次）——
    def load(dir_, idx):
        for sh in sorted(set(idx.values())):
            path = os.path.join(dir_, sh)
            h, hl = read_header(path)
            cache[sh] = (dir_, path, h, 8 + hl)
    load(src_dir, src_idx); src_cache = dict(cache)
    cache.clear(); load(out_dir, out_idx); out_cache = dict(cache)
    bad, ok, total = 0, 0, 0
    for (L, E, wX) in exp:
        is_mtp = L >= 1000
        wname = f'layers.{L}.ffn.experts.{E}.{wX}.weight'
        sname = f'layers.{L}.ffn.experts.{E}.{wX}.scale'
        osname = f'layers.{L}.ffn.experts.{E}.{wX}.weight_scale'
        ows2 = f'layers.{L}.ffn.experts.{E}.{wX}.weight_scale_2'
        def rd(cache_, name):
            d, path, h, base = cache_[src_idx[name] if cache_ is src_cache else out_idx[name]]
            # 注意：src 与 out 的 index 均需解析；此处简化为两字典分别查
            return h, read_tensor_bytes(path, base, *h[name]['data_offsets'])
        # —— 字节级 weight 比对 + scale 关系（E4M3 两半一致 + (b4>>3)-7 == (b-127)+G）——
        # 全量逐块；G 从 weight_scale_2=2^-G 恢复；统计 MTP 单独计数
        total += 1
        if w_check_and_scale_check(...):  # 逐块逻辑同工具 verify，但全量
            ok += 1
        else:
            bad += 1
            print(f"[FAIL] {'MTP' if is_mtp else 'MAIN'} L{L} E{E} {wX}")
    print(f"[FULLSCAN] {ok} OK / {bad} BAD / {total} total")
    return bad == 0

if __name__ == '__main__':
    sys.exit(0 if scan(sys.argv[1], sys.argv[2]) else 1)
```

> 建议输出：按 (MAIN/MTP, w1/w2/w3) 分组的计数表 + 全量失败清单；运行时间估计数十分钟（35,328 投影 × 逐块）。

## 附录 B：R5 输出内容 sanity check 启发式（可直接使用）

```python
# sanity_check.py —— 输入一串输出 token/文本，任一触发即 FAIL
def check(text: str, tokens_used: int, max_tokens: int) -> list[str]:
    flags = []
    import re
    if not text.strip():
        flags.append("empty-output")
    if tokens_used <= 1:
        flags.append("degenerate-length")          # 只出 1 个 token
    if tokens_used > max_tokens + 4:
        flags.append("over-length")
    words = re.findall(r"\S+", text)
    if len(words) >= 8:
        # ≥4 个相同 4-gram → repetition loop
        grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        for g in set(grams):
            if grams.count(g) >= 4:
                flags.append("repetition-loop"); break
    unk = text.count("\ufffd")
    if unk and unk / max(1, len(text)) > 0.01:
        flags.append("high-unk")
    return flags
```

---

## 附录 C：参考材料索引（本文档依据）

- 服务部署指南：`deliverables/engineering-assurance/tp4-service-deployment-guide-2026-08-13.md`（环网/NCCL env/补丁 MD5/启停纪律）
- Runbook 增量：`deliverables/engineering-assurance/runbook-tp4-append-2026-08-12.md`（A.3 env 全集 / A.4 PEER_HCA 表）
- 基准口径：`deliverables/engineering-assurance/TP4_prefill_bench_plan.md`、`bench_prefill_decode_async.py`（仓库根）
- vLLM 基线：`tp4-r8-final-report-2026-08-12.md`、`tp4-r12-final-report-2026-08-13.md`、`tp4-opt-execution-final-2026-08-13.md`
- NVFP4 运行时调查：`research-nvfp4-alternative-runtimes-2026-08-13.md`（SGLang 第一候选 ★★★）、`research-cublaslt-grouped-gemm-nvfp4-sm120-2026-08-13.md`（garbage 教训）、`research-deepseek-v4-flash-nvfp4-2026-08-13.md`（Rarri/MJPansa）
- 权重方案：`delivery/nvfp4-investigation/transcode_0731_to_nvfp4.py`、`serve-0731-nvfp4.sh`、`nvfp4-investigation-2026-08-13.md`
- 执行分析：`deliverables/engineering-assurance/nvfp4-upgrade-execution-analysis-2026-08-13.md`、`_fix_20260813/l1-env-prep-20260813.md`、`l2-build-plan-20260813.md`
- 验收范式：`_fix_20260813/tessa-acceptance.md`（fresh eyes 独立复测方法）

> 时效性：2026-08-13 时点。SGLang NVFP4×SM120 迭代极快，落地前复核 PR #25820 合入状态、SGLang 版本、flashinfer 版本。

---

# 10. V2 更新节（2026-08-14 · SGLang 0.5.14 定案 + bench 适配 + 数据目录初始化）

> 本文档增量节。v1（§1-§9、附录 A-C）保持有效；本节替换/补充以下 v1 内容：
> **R0（§4.0）状态更新、R9 拆分为 R9-A（§4.3，无投机基线）+ Phase-B R9（DSPARK 延后）、W 系列（§3.2）主路径切换、P7（§6.2）互斥序列细化、新增 TP1 冒烟检查点、bench 适配结论、数据目录、正式运行执行清单。**
> **Phase-A/B 两阶段**：Phase-A = 0.5.14 容器（**无 DSPARK**）→ prefill-only A/B + 运行时/权重/环网全量验证；Phase-B = 0.5.16+ 容器升级 → DSPARK 接受率 + decode 判定。

## 10.1 版本与环境定案（事实更新，替代 v1 推断）

| 项 | v1 推断 | v2 定案（2026-08-14 实测/核验） |
|---|---|---|
| 容器 | NGC 26.02（0.5.8，早于支持）或主线自建 | **NGC 26.07-py3 = SGLang 0.5.14+nv26.7.59534057**（PR #25820 合入线） |
| PR #25820 | "待核实"（R0 阻塞项） | **已确认合入**（v0.5.14 发布即含，2026-06-26）；R0 由"阻塞预检"改为"复核记录" |
| flashinfer | ≥0.6.15.post1（推荐） | **0.6.14**（NGC 校验组合，略低于推荐；如遇 NCCL 回退问题再钉 0.6.15.post1） |
| 权重 | 两条路线（本地转换 A / 预转换下载 B） | **主路径 = 四机已有 `<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`（164GB，48 shards，软链就绪）**；本地转换/MJPansa 下载降级为备用（仅 W 验证不通过时启用） |
| 端口 | 8010/8011/26000（待 preflight 核） | 定案：API **8010** / metrics **:8010/metrics**（**0.5.14 无 `--metrics-port`，用 `--enable-metrics`，metrics 挂在 API 端口**）/ TCPStore **26000**；原独立 8011 降级废弃 |
| 投机解码（DSPARK） | — | **0.5.14 实测无 DSPARK**（`--speculative-algorithm` 仅 EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH）→ Phase-A 走**无投机基线**（默认 SPEC_DECODE=off）；DSPARK 接受率验证**延后 Phase-B（0.5.16+ 容器）** |
| 运行模式 | 独立窗口（§6.4） | **生产 vLLM 运行中；本阶段只做适配与准备，正式压测等互斥窗口通知**（A/B 切换序列见 10.6）；**Phase-A/B 两阶段**：Phase-A=0.5.14 prefill-only A/B；Phase-B=0.5.16+ 容器升级后补 decode/DSPARK 判定 |

**基准铁律不变**：per-request p50×并发，禁用 agg_* 跨组对比；矩阵 16384/32768/65536 ctx × conc 1/4/8/16/32 × coding/json/prose；3 轮中位；随机前缀防缓存。**Phase-A 仅 prefill 维度判胜**（见 10.4.6）。

## 10.2 bench 脚本 SGLang 适配检查结论（bench_prefill_decode_async.py）

### 10.2.1 三分类结论

| 分类 | 内容 | 依据 |
|---|---|---|
| **零改动可用**（API 兼容，直接指向 SGLang） | ①端点 `/v1/chat/completions`（脚本 `--endpoint .../v1` + 拼接）——SGLang OpenAI 兼容端点 ✓；②流式 `stream:True` + `stream_options:{"include_usage":True}`——SGLang 支持 include_usage（最终 chunk 携带 usage，另有 continuous_usage_stats 可选）✓；③`usage.prompt_tokens/completion_tokens` 字段——SGLang usage 返回齐备 ✓；④就绪自检 `/v1/models`——SGLang 支持 ✓；⑤`--model` 不在 served 列表时脚本自动选第一个 ✓（SGLang served-name 默认取 model 路径 basename，可 `--served-model-name` 显式指定） | 官方 OpenAI 兼容 API 文档 + mintlify chat-completions 文档 |
| **需小改**（非阻塞，可选） | ①`--sanity-log` 的 NCCL 判定 ok_pat/fail_pat 按 vLLM 日志措辞编写，SGLang 日志可能只返回 WARN——**不作为硬失败**（WARN 人工复核放行即可）；如需自动 PASS 可把 `"NCCL"` 相关模式扩展为 SGLang 措辞（如 `initialized process group` / `rank 0.*rank`）；②P4 长 decode 档需新增 `--max-tokens-override`（v1 行动清单已有，保留） | 代码审查 §383-413 `check_nccl_init` |
| **需新增**（服务端/外部，非脚本改动） | ①服务端 `--max-running-requests 64`（conc=32 需 ≥32；参数名以 `sglang.launch_server --help` 实测为准，官方文档/社区均以 `--max-running-requests` 为主，若 0.5.14 同时接受 `--max-running-seqs` 别名则优先用别名并记录）；②服务端 `--enable-metrics`（0.5.14 无 `--metrics-port`，metrics 在 `:8010/metrics`，Phase-A 记录可访问、Phase-B 供接受率探测）；③bench 运行机装 aiohttp（脚本依赖，无则 `--engine threads` 回退） | SGLang 参数文档 + 社区实测 |

### 10.2.2 实测命令（TP1 冒烟时一并执行，替代"R0 待核实"）

```bash
# ①确认 --max-running-requests 存在（0.5.14 实参名）
docker run --rm nvcr.io/nvidia/sglang:26.07-py3 \
  python3 -m sglang.launch_server --help 2>&1 | grep -iE "max-running|max_running|max-num-seqs"

# ②确认 metrics 导出（服务起后；0.5.14 用 --enable-metrics，metrics 在 :8010/metrics，无独立 --metrics-port）
curl -s http://<head>:8010/metrics | grep -cE "sglang:"   # >0 即 metrics 可用
```

## 10.3 测试数据目录初始化

**本地镜像（已创建，2026-08-14）**：

```
_archive_scratch/_tessa_sglang_bench/2026-08-14/
├── weights/    # W0-W8 权重验证交付物
├── runtime/    # R1-R12 运行时验证交付物（smoke/sanity/precision/accept）
├── S/          # SGLang 性能 raw（rows_S.csv + summary_S.json）
└── V/          # vLLM 基线 raw（rows_V.csv + summary_V.json，重跑时用）
```

**服务器端（正式运行前执行，写入可执行；02 或 01）**：

```bash
# 在服务器（02 推荐，或 01）执行：
mkdir -p <INSTALL_DIR>/logs/sglang-test/_tessa_sglang_bench/2026-08-14/{weights,runtime,S,V}
chmod 777 <INSTALL_DIR>/logs/sglang-test/_tessa_sglang_bench/2026-08-14  # 或赋予运行用户写权限
ls -la <INSTALL_DIR>/logs/sglang-test/_tessa_sglang_bench/2026-08-14/
```

**W0-W8 权重验证交付物清单（输出文件命名，均落盘 `weights/`）**：

| ID | 交付物文件 | 说明 |
|---|---|---|
| W0 | `env-preq-2026-08-14.log` | 工具子命令可见、SRC/OUT 在位、磁盘余量 |
| W1 | `transcode-2026-08-14.log` + `transcode-summary-2026-08-14.json` | **注意：主路径为已有权重，W1 转为"产物核对"（shard 数/大小/hf_quant_config 在位）；本地重转仅备用路线** |
| W2 | `config-merged-2026-08-14.json` + `config-check-2026-08-14.json` | quantization_config / dspark_block_size / index 键数 |
| W3 | `verify-2026-08-14.log` | 工具自带 verify（64 OK / 0 BAD） |
| W4 | `fullscan-report-2026-08-14.json` + `fullscan-count-2026-08-14.csv` | **全量 bit-exact 补扫（必做）**；按 MAIN/MTP × w1/w2/w3 计数 |
| W5 | `scale-dist-2026-08-14.csv` + `g_dict-2026-08-14.json` | G13/G2 range、weight_scale_2 直方图 |
| W6 | `layout-scan-2026-08-14.json` | 无 MXFP4 残留、block-16、dtype 核对 |
| W7 | `precision-smoke-2026-08-14.json` | 20 prompt top-1 一致率 / logit RMS / NaN |
| W8 | `sha256-2026-08-14.txt` + `file-manifest-2026-08-14.csv` | 48 shard+config+index 等 75 文件清单 + 四机 sha256 一致 |

> 归档路径沿用 v1 §7：`_archive_scratch/_tessa_sglang_bench/2026-08-14/weights/`（服务器端镜像同构）。

## 10.4 0.5.14 更新要点（用例级）

### 10.4.1 R0 预检（§4.0 更新）

| 项 | v2 状态 | 判定 |
|---|---|---|
| PR #25820 | **已确认合入 v0.5.14**（镜像内实测 `pip show sglang` = 0.5.14+nv26.7） | ✅ 无需补丁 |
| SGLang 版本/容器 | NGC 26.07-py3；`--help` 抓取真实 flag（见 10.2.2） | 关键 flag：`--moe-runner-backend flashinfer_trtllm_routed`、`--quantization modelopt_fp4`、`--enable-metrics`（**无 `--metrics-port`**）、`--max-running-requests`、多机 `--nnodes/--node-rank/--dist-url`；**`--speculative-algorithm` 实测无 DSPARK（仅 EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH），DSPARK 延后 Phase-B** |
| flashinfer | **0.6.14**（记录在案；NCCL 回退问题出现时再钉 0.6.15.post1） | ✅ |
| SM121 | `FLASHINFER_CUDA_ARCH_LIST=12.1a`；`SGLANG_DISABLE_DEEP_GEMM=1`/`SGLANG_ENABLE_DEEP_GEMM=0` | NVFP4 走 flashinfer_fp4_gemm；TP1 冒烟实测 `is_sm120_supported()` 覆盖 |

### 10.4.2 R9-A 无投机基线（§4.3 更新，Phase-A；DSPARK 延后 Phase-B）

**架构师裁决（2026-08-14）**：0.5.14 实测无 DSPARK（`--speculative-algorithm` 仅 EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH）→ **Phase-A 不做投机接受率判定**。

- **R9-A（本窗口）**：无投机基线 = 默认形态（SPEC_DECODE=off），验证 **MTP 权重加载不报错** 即可（服务启动日志无 MTP/scale 相关 abort；R1 冒烟正常出 token）。
  - K8 判定由"接受率 ≥0.40"改为"**MTP 加载无异常**"：`docker logs <sglang-rank0> | grep -iE "mtp|draft|scale|error|abort"` → 无致命错误即 PASS。
  - metrics 记录项：`:8010/metrics` 可访问即可（`curl -s http://<head>:8010/metrics | grep -cE "sglang:"` >0）。
- **R9 正式用例（Phase-B，0.5.16+ 容器升级后）**：启用 DSPARK（`--speculative-algorithm DSPARK` + `SGLANG_RAGGED_VERIFY_MODE=compact`）后，从 `:8010/metrics` 抓 `sglang:spec_accept_rate` / `sglang:spec_accept_length`，沿用门槛 `≥0.40` PASS / `<0.20` FAIL；DSpark cap-accept 模式（`SGLANG_RAGGED_VERIFY_MODE=cap-accept`）暴露修剪接受上限作分析用。

```bash
# Phase-A R9-A 探测：metrics 可访问 + MTP 加载无异常
curl -s http://<head>:8010/metrics | grep -cE "sglang:"            # >0 即 metrics 可用
docker logs <sglang-rank0> 2>&1 | grep -iE "mtp|draft|scale|error|abort" | tail -20   # 无致命错误
```

### 10.4.3 W 系列权重验证更新（§3.2 主路径切换）

- **主路径 = 四机已有 NVFP4 权重**（`<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`，164GB，48 shards）验证，**不再以本地转换为前置**：
  - W1 从"转换执行"改为"**产物核对**"：shard 数/大小/hf_quant_config.json（producer=modelopt、quant_algo=MIXED_PRECISION、group_size=16、**"experts-mtp-fallback" 命名 → MTP 策略必须实测确认**）
  - W2/W3/W4/W5/W6/W7/W8 逻辑不变，输入改为已有权重目录
- **MTP 策略验证重点**：Phase-A 以 **R9-A "MTP 加载无异常"** 为门槛（W4 MTP 键单独 0 失配 + TP1/TP4 启动日志无 MTP 错误 + SGLang load 冒烟失败时走备用路线：本地全转 / MJPansa 版）；**接受率 ≥0.40 数值门槛延后 Phase-B（0.5.16+ DSPARK）**
- 若 SGLang load 冒烟（TP1）失败且指向权重问题 → 才启用备用路线（本地 transcode 或下载 MJPansa ~180GB）

### 10.4.4 P7 资源争抢控制（§6.2 更新，互斥切换窗口细化）

**执行序列（正式压测时，本阶段仅演练/记录）**：

```bash
# ① 停生产 vLLM（head+worker 四容器）——由运维/排期窗口触发
docker ps --format '{{.Names}}' | grep vllm          # 确认容器清单
docker stop <vllm-head> <vllm-worker-01..03>          # 或编排 stop

# ② 前恢复验证（vLLM 已停 + 资源门禁）
pgrep -f EngineCore && echo "STILL RUNNING" || echo "vLLM STOPPED"   # 期望 STOPPED
free -g | awk '/Mem:/{print $7}'                        # 期望空闲内存 ≥55G（门禁）
ss -ltnp | grep -E ':8010|:26000' || echo "ports free"              # 期望 free（metrics 随 :8010，无独立 8011）

# ③ 启动 SGLang TP4（head-first，见工程方案 §2.6）—— head 01 → 02 → 04 → 03
# ④ 测试窗口（R/C/P 全绿判定）
# ⑤ 停 SGLang
docker rm -f sglang-nvfp4-tp4-{0,1,2,3}

# ⑥ 后恢复验证（切回生产 vLLM）
docker start <vllm-head> <vllm-worker-01..03>
sleep 30 && pgrep -f EngineCore | wc -l                 # 期望 4
curl -s http://<head>:8001/health                        # vLLM 健康
# + 1 次冒烟请求（生产 prompt）确认输出正常
```

**判定**：切换全程无 OOM/无争抢（监控 GPU util/mem）；测试后 vLLM 恢复 100%（4 EngineCore + health + 冒烟 OK）；切换时间戳记录入报告。

### 10.4.5 新增 TP1 冒烟检查点（单机先行，对应 handoff §6.3）

**执行清单（本阶段即可执行，不等正式压测）**：

```bash
# 1) 容器内 GPU 可见
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/sglang:26.07-py3 nvidia-smi

# 2) 版本复核（0.5.14 + torch 2.13.0a0+nv26.07 + flashinfer 0.6.14）
docker run --rm nvcr.io/nvidia/sglang:26.07-py3 bash -c \
  "pip show sglang | head -2; python -c 'import torch, flashinfer; print(torch.__version__, flashinfer.__version__)'"

# 3) 单机 TP1 启动（低配并存冒烟：mem-fraction 0.2-0.3 + cpuset=0 + 内存门禁 ≥55G；门禁不足则等互斥窗口）
docker run -d --name sglang-tp1-smoke --restart no --network host --ipc=host --privileged \
  --gpus all --shm-size 64g --cpuset-cpus 0 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 -e SGLANG_ENABLE_DEEP_GEMM=0 \
  nvcr.io/nvidia/sglang:26.07-py3 python3 -m sglang.launch_server \
  --model <INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4 \
  --tp 1 --trust-remote-code --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm_routed \
  --host 0.0.0.0 --port 8010 --enable-metrics \
  --mem-fraction-static 0.3 --max-running-requests 64
# 注：0.5.14 无 --metrics-port，metrics 在 :8010/metrics（--enable-metrics）

# 4) 日志确认 SM121 kernel 加载（无 CUTLASS 崩溃）：grep 关键字
docker logs sglang-tp1-smoke 2>&1 | grep -iE "flashinfer|fp4|kernel|error|SM120|SM121" | tail -30

# 5) API 冒烟
curl -s http://<head>:8010/health
curl -s http://<head>:8010/v1/models
curl -s http://<head>:8010/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash-0731-nvfp4","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}'

# 6) metrics 有数据（:8010/metrics）
curl -s http://<head>:8010/metrics | grep -cE "sglang:"
```

**判定**：GPU 可见 ✓、版本=0.5.14/0.6.14 ✓、TP1 启动无 CUTLASS 崩溃 ✓、`/health`+`/v1/models`+chat 冒烟 200 且流式 usage 齐备 ✓、metrics :8010/metrics >0 ✓、MTP 加载无异常（R9-A）✓ → **全绿才进 TP4**。

### 10.4.6 L4 判定阈值与 P4 矩阵（Phase-A/B 拆分）

**架构师裁决（2026-08-14）**：0.5.14（无投机）下 decode 对比对 SGLang 不公平 → Phase-A 只做 **prefill-only A/B**（tsarihan 1.14-1.32× 是 prefill 带宽收益，不依赖 DSpark，0.5.14 可验证）；decode 判定随 Phase-B。

| 维度 | Phase-A（0.5.14，无投机） | Phase-B（0.5.16+，DSPARK） |
|---|---|---|
| 矩阵 | **prefill 聚焦**：16384/32768/65536 ctx × conc 1/4/8/16/32 × coding/json/prose × 3 轮 | 补 decode 长度档（128/1024）与投机对比 |
| prefill 吞吐 | 目标 ≥1.1×（vs vLLM MXFP4 同口径）；≥1.05× 记 PASS | 复测 |
| decode 吞吐 | **无硬门槛、不做正式对比**（仅记录绝对数值参考） | ≥0.95× 硬门槛 + TPOT 对比 |
| TTFT | ≤ 基线 ×1.1 | 复测 |
| 接受率 | —（R9-A 仅 MTP 加载无异常） | spec_accept_rate ≥0.40 / <0.20 FAIL |

> **P4 调整**：decode 长度档（128/1024，`--max-tokens-override` 小改）从 Phase-A 移除，延后 Phase-B；Phase-A 核心矩阵即 10.5 步骤 11。

## 10.5 正式运行前测试执行清单（handoff 给执行者）

> 从"接到运行通知"到"性能数据产出"的完整动作序列。每步含命令要点/判定/耗时预估。
> **Phase-A/B 说明**：以下步骤 1-15 为 **Phase-A（0.5.14，prefill-only A/B + 全量运行时/权重/环网验证）**；**Phase-B 前置 = 容器升级 0.5.16+**（步骤 16-18），用于 DSPARK 接受率 + decode 判定。

| # | 步骤 | 命令/动作要点 | 判定（通过才继续） | 耗时预估 |
|---|---|---|---|---|
| 1 | 收到运行通知（互斥窗口排期） | team-lead/运维确认窗口 | 窗口可用 | — |
| 2 | 服务器端数据目录初始化 | `mkdir -p <INSTALL_DIR>/logs/sglang-test/_tessa_sglang_bench/2026-08-14/{weights,runtime,S,V}` + chmod | 目录可见可写 | 2 min |
| 3 | preflight 四机核验 | `/proc/self/maps \| grep libnccl`=2.30.7；RoCE 对口 ping；MTU 9000；`ss -ltnp` 8010/26000 空闲（metrics 随 :8010，无独立 8011） | 全部通过 | 10 min |
| 4 | **互斥切换①：停 vLLM** | `docker stop <vllm 四容器>`；`pgrep -f EngineCore` 空 | vLLM STOPPED + 内存 ≥55G | 5 min |
| 5 | SGLang TP4 启动（head-first） | 01→02→04→03；`--max-running-requests 64` + `--enable-metrics`（无 `--metrics-port`）+ 环网 env 全集 | `/health` OK + 日志无 NCCL/kernel error + :8010/metrics 可访问 | 15-40 min（含权重加载） |
| 6 | TP1/TP4 冒烟（R1-R4） | `/v1/models`、单请求、8 并发、长度/stop | 4/4 rank ready + HTTP 200 + usage 齐备 | 10 min |
| 7 | L2 精度/一致性（R5-R7） | sanity check + 20-prompt 对比 + TP4 确定性 | R5 0 触发；R6 ≥0.90；R7 全同 | 20 min |
| 8 | MTP/无投机（R8-R10 + R9-A） | 长 ctx + **R9-A：`curl :8010/metrics \| grep -c sglang:` >0 + MTP 加载无异常（日志 grep mtp/error）** | R9-A PASS（DSPARK 接受率延后 Phase-B） | 15 min |
| 9 | L3 环网（C0-C8） | NCCL banner/PEER_HCA/DeepEP fallback/shim PSR | C0 无 13.3；C1 ring 对；C5 主路径可放行 | 30 min |
| 10 | bench 预检（P0-P1） | aiohttp 依赖；`--sanity-log`（WARN 放行）；calibrate | calibrate 成功（tokens/unit 记录） | 5 min |
| 11 | **性能核心矩阵（P2，Phase-A prefill 聚焦）** | `bench_prefill_decode_async.py --group S --endpoint http://<head>:8010/v1 --key <key> --model deepseek-v4-flash-0731-nvfp4 --concurrency 1,4,8,16,32 --ctx 16384,32768,65536 --tasks coding,json,prose --rounds 3 --engine asyncio --out .../S` | 45 组合全出；`rows_S.csv`+`summary_S.json` 落盘；**只比 prefill** | 1-2 h |
| 12 | ~~decode 长档（P4）~~ **延后 Phase-B** | Phase-A 跳过（0.5.14 无投机 decode 对比不公平）；Phase-B 补 `--max-tokens-override` 小改 + 128/1024 两档 | — | — |
| 13 | vLLM 基线对照（P5-P6，Phase-A prefill-only） | 复用 tp4-r8/r12/opt-execution；如需重跑 `--group V`（conc 1/3/5） | prefill 对比表生成（decode 仅记录不作判） | 视需要 |
| 14 | **互斥切换②：停 SGLang → 恢复 vLLM** | `docker rm -f sglang-*` → `docker start <vllm>` → pgrep=4 + health + 冒烟 | vLLM 恢复 100% | 10 min |
| 15 | 报告（P8） | `bench-sglang-<cfg>-<date>.md` 落盘 `deliverables/engineering-assurance/` + raw 归档 | 表格+比值+判定齐全（标注 Phase-A） | 30 min |

**Phase-B（0.5.16+ 容器升级后）**：

| # | 步骤 | 命令/动作要点 | 判定 | 耗时预估 |
|---|---|---|---|---|
| 16 | 容器升级 0.5.16+（**Phase-B 前置**） | 拉取/自建 0.5.16+ 镜像（含 DSPARK）→ 四机分发 → 版本复核 | 容器内 `--speculative-algorithm` 含 DSPARK | 0.5-1 天 |
| 17 | R9 DSPARK 接受率 | `--speculative-algorithm DSPARK` + `SGLANG_RAGGED_VERIFY_MODE=compact`；`curl :8010/metrics \| grep spec_accept_rate` | ≥0.40 PASS / <0.20 FAIL | 20 min |
| 18 | decode A/B 判定（P4/P6） | 128/1024 长档 + decode ≥0.95× 硬门槛 | decode 不劣化 >5% | 30 min |

**关键纪律**：任何一步 FAIL 即停，不强行续跑；互斥窗口外不碰 vLLM；raw 数据只增不改。

---

## 附录 D：V2 参考索引（新增）

- 容器交付：`handoff-sglang-2607-four-node-2026-08-14.md`（0.5.14 版本、镜像 md5、TP1 冒烟 §6.3）
- 工程方案：`sglang-nvfp4-tp4-setup-plan-2026-08-13.md`（PR #25820 已合入、端口/权重定案、env 全集）
- SGLang 官方文档（2026-08-13 检索）：OpenAI 兼容 API（`/v1/chat/completions`、stream_options.include_usage）、Prometheus Metrics（`sglang:spec_accept_rate`/`spec_accept_length`）、`--max-running-requests` 参数
- 架构师裁决（2026-08-14）：0.5.14 无 DSPARK（`--speculative-algorithm` 实测仅 EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH）→ R9-A 拆分 + Phase-A prefill-only A/B + decode 判定延后 Phase-B + metrics 改 `:8010/metrics`（`--enable-metrics`，无 `--metrics-port`）
- bench 脚本：`bench_prefill_decode_async.py`（仓库根，asyncio wave 真并发）
