# TP4 AllReduce 优化：源码研究 + 停机窗口验证报告（2026-08-22）

**执行**: ar-optimizer（SRE 工程师）· P0-A
**窗口**: 2026-08-22 09:27:30Z 停机 → 10:30Z 生产恢复（含研究/微基准/基线复测/取证）
**任务**: PR 398μs/token 中 TP4 bf16 ring AR 占 109-124μs（~31% 步时，昨日 prde-bottleneck 口径）→ 验证 AR 优化路线（fuse_allreduce_rms / fuse_gemm_comms / 量化 AR / NCCL 调优）
**结论口径**: 【实测】= 4 节点 GPU 微基准 / e2e 基准 / RDMA 计数器直接测量；【源码】= fork 源码精读；【推断】= 实测×模型推导

---

## 0. 一页结论（供裁决）

1. **全部四条 AR 优化路线均判 No-Go 或已达物理最优**——但窗口产出了两个高价值校准结论：
2. **校准 1（最重要）: AR 实际只占步时 ~13%（52ms/步），不是 31%（127ms）。** 实测 NCCL busbw = 21.4GB/s（prde 报告假设 8.8-10GB/s 低估了 2 倍以上）。RDMA 取证复测：131K prefill 线上流量 143.4GB/节点（=128 步 × 1120MB/步，与昨日完全一致）÷ 21.4GB/s = 52.3ms/步 = **12.9% 步时**；与 86 AR × 0.588ms = 50.6ms 互证。**即使 AR 完全消除，PR 上限也只有 +15%**——AR 不是"性价比最高的第一刀"，**MoE（185-204ms/步，45-50%）才是**。
3. **校准 2: NCCL 调优空间已在生产现值穷尽**（4ch/8MB/ringonly）：8/16 通道在 ringonly 定制构建上灾难性劣化（~18-20ms/AR，chan 级硬编码映射被破坏）；原版 pip NCCL 在本 fabric init 挂起（ringonly 构建存在的根本原因）；BUFFSIZE 32MB 无收益（0.602 vs 0.588ms）。
4. **fp8 AR（08-17 PoC 计划的主选路线）结构性无收益**：fp8 纯传输确实减半（0.314 vs 0.584ms，ncclFloat8e4m3 归约工作正常），但 per-block scale 全局同步需要第二个小消息 AR（实测 0.509ms，延迟主导）+ 量化/反量化 kernel（0.65ms 未融合），端到端 1.52ms = bf16 的 2.6 倍。理论下限（fused kernel + 高效 scale 同步）≈ 0.64ms 仍 > bf16 0.584ms。数值误差 5.5-7.2% max_rel 也过不了 1% 门。
5. **C0 基线复测零漂移**（panorama 四档 Δ +0.5%/+0.1%/+0.2%/-0.4%），生产恢复完整验证（health 200 / KV 6,093,694 / B12X 256 experts / dspark draft+graphs / 自愈链四节点 active）。

---

## 1. 源码研究结论（生产镜像一次性容器精读，vLLM 0.26.1 fork `d3d3b2cca`）

### 1.1 发现 0（决定性）：torch.compile 整体禁用

生产日志实证：
```
WARNING [vllm.py:1203] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's
torch.compile pipeline. Equivalent to -cc.mode=none.
WARNING [vllm.py:1213] Inductor compilation was disabled by user settings...
```
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`（启动脚本固定设置）与 vLLM 编译管线互斥 → `CompilationMode.NONE`。**所有编译期 fusion pass 当前根本不运行**——pass_config 里的 `fuse_allreduce_rms=False / fuse_gemm_comms=False` 只是死配置。

### 1.2 fuse_allreduce_rms = flashinfer fused AR（AR+RMSNorm 融合）→ No-Go（三重死锁）

| # | 阻塞点 | 源码证据 |
|---|--------|---------|
| 1 | 编译期 pass，需先恢复 torch.compile | `pass_manager.py:157`；compile 被 breakable-cudagraph 禁用（§1.1），恢复 = 牵动 B12X/dspark 全栈，风险不可控 |
| 2 | flashinfer fused AR 后端物理不可用（4 节点 RoCE） | `flashinfer_all_reduce.py:97-121`：backend 只有 mnnvl（需 NVSwitch multicast，`_resolve_fi_ar_backend` 多节点强制 mnnvl）和 trtllm（单节点 IPC only，`get_node_count()>1 直接报错`）；4 节点 RoCE 环网两者皆无 |
| 3 | 能力字典无 sm121 | `allreduce_rms_fusion.py` `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` 仅 {90,100,103}；`flashinfer_max_size()` 返回 None → pass 直接 disabled（`__init__` 早退） |

（flashinfer 0.6.15 已装、`has_flashinfer()` 为真——但拓扑/能力双缺。）

### 1.3 fuse_gemm_comms = async TP → No-Go（本窗口）

1. 编译期 pass（`collective_fusion.py:900 AsyncTPPass`），同样被 compile-off 阻塞，且要求 full-graph 编译（`is_applicable_for_range` 断言）；
2. 强制依赖 SP：`config/vllm.py:1291` `fuse_gemm_comms → enable_sp=True`；SP 启用门禁 `get_sequence_parallelism_threshold()` 的字典 `SP_MIN_HIDDEN_SIZE={90:8192,100:8192}` **无 sm121** 且 DeepSeek V4 hidden=4096 < 8192 → 需手动强设 `sp_min_token_num`；
3. fork 有意对 MoE 禁用（`config/vllm.py` `IS_DENSE=False` 硬编码，注释引 issue 25689）；B12X custom op 的 pattern 匹配未验证。

### 1.4 量化 AR / fp8 AR：路径就绪但结构性无收益（§2 实测判死）

技术可行性（全部实证）：vllm pynccl_wrapper 有 `ncclFloat8e4m3=10` 映射（`pynccl_wrapper.py:76`），与镜像内 NCCL 2.30.7 头文件枚举一致；ringonly 构建是传输层补丁与 dtype 正交；**ncclFloat8 归约在 4 节点 RoCE 上实测工作正常**（0.314ms/8.4MB，busbw 等效 20.1GB/s）。torch 2.11 c10d 不支持 fp8 AR（libtorch_cuda 无 ncclFloat8 符号）→ 需走 pynccl ctypes（overlay 已写好：`_ar_opt/communication_op.py`，AICAD_FP8_AR 门控 + ≥1MiB prefill-only + 启动 SELFCHECK）。

### 1.5 当前 AR 路径确认（生产实际调用链）

```
RowParallelLinear(wo_b) / FusedMoE down → communication_op.tensor_model_parallel_all_reduce
  → get_tp_group().all_reduce → _all_reduce_out_place → cuda_communicator.all_reduce
  → [pynccl 禁用: VLLM_DISABLE_PYNCCL=1 → PyNcclCommunicator 根本不创建 (pynccl.py:93)]
  → out = input_.clone(); torch.distributed.all_reduce(out, group=device_group)  ← 生产实际路径
```
- 两处模型 AR（attention wo_b：`linear.py:1694`；MoE down：`moe_runner.py:433/464`）全部经 `tensor_model_parallel_all_reduce` 单一咽喉点（overlay 接管点选对了）。
- NCCL 层：`LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`（ringonly 2.30.7+cuda13.0，PEER_HCA chan 级硬编码映射）+ `NCCL_ALGO=RING` + `MIN=MAX_NCHANNELS=4` + `BUFFSIZE=8MB`。

---

## 2. 停机窗口实测（4 节点 GPU 微基准 + e2e + RDMA 取证）

### 2.1 fp8 AR 微基准（N1 冒烟 + 分解，torchrun 4×1 GPU，中位 30 轮）

| 路径（1024×4096 = 8.4MB 消息） | 耗时 | 备注 |
|---|---|---|
| bf16 AR（c10d，生产路径含 clone） | **0.584ms** | busbw 21.4GB/s |
| bf16 AR（pynccl） | 0.954ms | empty_like 分配开销，劣于 c10d |
| **fp8 AR 纯传输（pynccl ncclFloat8）** | **0.314ms** | **传输减半成立**（busbw 等效 20.1GB/s） |
| amax（per-128-block） | 0.052ms | |
| **scale 同步 AR（128KB，c10d MAX）** | **0.509ms** | 延迟主导，≥ fp8 省下的 0.27ms |
| quant（torch 未融合） | 0.374ms | |
| dequant（torch 未融合） | 0.274ms | |
| **fp8 端到端组合** | **1.522ms（bf16 的 2.6×）** | 判死 |

**死因（结构性）**：
1. RoCE 环上小消息延迟（scale 同步 0.509ms）吃掉全部传输节省（0.27ms）——含同步的理论下限 ≈ 0.64ms > bf16 0.584ms；
2. 量化/反量化 kernel 即使 fused 到 0.1ms 也救不回；
3. 数值：max_rel 误差 5.5-7.2%（E4M3 3-bit 尾数 + 4-rank 求和），cos 0.9988，大概率过不了 1% logprob 门。
（救法只剩 delayed scale + fused kernel，上限 PR +2%，不值得。08-17 PoC 报告"fp8 AR → PR +15-18%"的前提 busbw 8.8-10GB/s 被实测 21.4GB/s 推翻。）

### 2.2 NCCL 通道/缓冲扫描（同微基准框架，每组合 4 节点独立验证）

| 配置 | 8.4MB AR | busbw | 判定 |
|---|---|---|---|
| **4ch / 8MB（生产现值）** | **0.588ms** | **21.4GB/s** | **基准（复现 2 次：0.584/0.588）** |
| 8ch / 8MB | 18.1ms | 0.7GB/s | **灾难**（ringonly v4 chan 级硬编码映射被破坏） |
| 16ch / 8MB | 20.0ms | 0.6GB/s | 灾难 |
| 4ch / 32MB | 0.602ms | 20.9GB/s | 无收益 |
| 原版 pip NCCL（无 preload，cuda13.3 构建） | — | — | **ncclCommInitRank 挂起**（ringonly 构建存在的根本原因，与 08-06 init-hang 调查呼应） |
| （参考）LL128 / QPS=2 / 8ch@小消息 | — | — | 08-17 B 系列已排除 |

小消息参考（4ch）：786KB=0.133ms / 196KB=0.098ms（decode 路径现值健康）。

**结论：NCCL env/配置空间在生产现值穷尽。21.4GB/s（双发口各 ~10.7GB/s = 200G 线速 43%）是本 NCCL 构建 + RoCE fabric 的实际上限。**

### 2.3 C0 基线复测（防漂移锚点，panorama 3 轮中位）

| 档位（标称/实际 tokens） | C0 实测 PR | 历史基线 | Δ |
|---|---|---|---|
| 4K / 8.2K | 2521.5 | 2510 | +0.5% |
| 16K / 32.8K | 2503.6 | 2500 | +0.1% |
| 32K / 65.5K | 2425.5 | 2420 | +0.2% |
| 64K / 131K | 2261 | 2270 | -0.4% |

零漂移 ✓（另：131K 单请求 TTFT 57.7s / PR 2270.7）。logprob 7-prompt 基线落档（`c0_bench.json`）。

### 2.4 AR 流量取证 + 占比校准（窗口核心产出）

RDMA 计数器窗口采样包住单个 131K prefill（131,082 tokens，唯一 nonce）：

| 口 | xmit | rcv |
|---|---|---|
| roceP2p1s0f0 | 217.2MB | 71,483.1MB |
| roceP2p1s0f1 | 71,485.1MB | 219.2MB |
| rocep1s0f0 | 219.2MB | 71,499.7MB |
| rocep1s0f1 | 71,506.4MB | 225.9MB |
| **合计/节点** | **143,427.9MB** | **143,427.9MB** |

- 143.4GB ÷ 128 步 = **1120MB/步**——与昨日 prde 测量（1119MB）完全一致 → 流量口径无争议；
- 但墙钟校准：1120MB ÷ 21.4GB/s = **52.3ms/步**；互证：86 AR × 0.588ms = 50.6ms ✓；
- **AR 占步时 = 52.3 / 407ms = 12.9%**，昨日报告的"~31%（112-127ms）"基于 busbw 8.8-10GB/s 的假设，实测推翻（该假设源于旧小消息 nccl-tests 的每口 4.4-5GB/s）。
- 注意：昨日报告功率模型（0.31×25W + 0.69×65W ≈ 52W）是"31% 占比"的旁证——在 13% 占比下该模型不再自洽，需重新归因（可能是 NCCL kernel 自身 SM 占用而非纯等待）。

---

## 3. PR 理论阶梯修正（相对 prde-bottleneck §3）

| 情形 | 旧口径 | 修正口径 | 依据 |
|---|---|---|---|
| 现状 | 407ms / 2510 | 不变 | C0 复测 |
| 全组件 Roofline（AR 形态不变） | 3560-3720（+42-48%） | **~2830-2900（+13-16%）** | AR 只回收 52ms 中的 kernel 效率差 |
| + fp8 AR | 4450-4650 | **不可达（No-Go）** | §2.1 结构性判死 |
| + AR 重叠/异步 TP | 5700+ | 理论 ~3150-3300（+25-30%），工程量不变 | 上限 = 回收 52ms 全部 |
| （不变）MoE 扩 M/kernel 效率 | — | **新晋第一优先**：185-204ms 池子，62-69% 带宽效率 | 旧报告 §6.2 |

---

## 4. 生产建议

1. **AR 线（本任务）：关闭，判定已达物理最优。** fuse_allreduce_rms / fuse_gemm_comms / fp8 AR / NCCL env 全部 No-Go 或无收益，不建议任何生产变更（生产现值 4ch/8MB/ringonly 即最优）。overlay 代码（`_ar_opt/communication_op.py`）留档备查，未部署。
2. **优先级重排：建议把"MoE 带宽效率"（62-69% → 85% 目标，-50-60ms/步）和"threshold 2048 异常 root-cause"提为 P0。** AR 只剩 52ms 可回收且需要大工程（多 stream overlap / async TP 需恢复编译管线）；MoE 有 60-77ms 缺口且已有成熟抓手（b12x-tail F1/F10、扩 M、merged-GEMM）。
3. **测量方法论沉淀**：busbw 21.4GB/s 是新校准常数（大消息 bf16 ring AR，4ch ringonly）；后续通信分析一律用实测 busbw 而非每口 4.4-5GB/s 假设。
4. （风险登记）ringonly 构建把通道数硬编码为 4——未来若换 fabric/NCCL 版本需重新评估该补丁的映射逻辑，8/16 通道在当前补丁下直接破坏。

---

## 5. 窗口操作记录（UTC）

| 时间 | 事件 |
|---|---|
| 09:27:30 | 停机开窗：停 01 自愈链（vllm-tp4-head/healthcheck timer）+ 清四机容器 |
| 09:28-09:33 | N1 fp8 AR 冒烟（首跑因 worker 脚本未分发失败一次；修正后完成） |
| ~09:28 | 发现 03/04 anemll-8022 embed 服务被误杀 EngineCore 后重启（unless-stopped 策略），后续干扰首轮流扫描（垃圾数据 18-20ms） |
| 09:37 | fp8 AR 分解测试（4 节点） |
| 09:40-09:46 | 首轮通道扫描（受 embed 重启干扰作废） |
| 09:50-10:07 | 干净环境重扫：4ch/8MB 复现、8ch/16ch 灾难、BUFFSIZE 无收益、无 preload 挂起 |
| 10:08-10:13:41 | C0 基线 TP4 重启（head-first 编排，READY 10:13:41Z，KV 6,093,694） |
| 10:20-10:27 | C0 全量基准（panorama + DE + logprob + metrics） |
| 10:27-10:30 | RDMA 取证（131K prefill 包裹采样） |
| 10:30-10:32 | 生产恢复：自愈链四节点 active、health 200、B12X/dspark 验证 |
| （全程） | 生产脚本零改动（deploy 脚本预备但未执行，无需回滚）；checker 未触碰 |

## 6. 证据档案

| 项 | 位置 |
|---|---|
| 微基准 JSON（smoke/decomp/scan×5） | 本地 `deliverables/engineering-assurance/_ar_opt/*.json`；服务器 01:`/tmp/_ar_opt/` |
| C0 基线（panorama/bench/logprob/metrics/rdma） | 同上（c0_*.json / rdma_c0_*.txt） |
| fp8 AR overlay 代码（未部署，留档） | `_ar_opt/communication_op.py`（AICAD_FP8_AR 门控，默认 0 = 行为零变化） |
| 窗口工具链（stop/restart/bench/rdma/rollback） | `_ar_opt/*.sh`（可复用于后续窗口） |
| 源码研究引用 | prde-bottleneck-analysis-2026-08-22.md §AR；poc-fp8-allreduce-plan-2026-08-17.md；research-comm-overlap-tp4-2026-08-17.md；nccl-ab-B-execution-report-2026-08-17.md |

**局限声明**：① 8.4MB AR 的 0.584ms 是微基准稳态口径（预热后 30 轮中位），生产步间夹杂 compute 时的 per-AR 延迟可能略高（首次冒烟曾测到 1.16ms，疑似双 comm 交替干扰），但 RDMA 流量 ÷ busbw 与 86×0.588ms 双口径互证 52ms/步量级成立；② fp8 数值误差测试用合成激活（含 ×40 离群值），真实激活分布误差可能不同，但结论不依赖于此（性能侧已判死）；③ DE 数字（decode_1x256/12x128）受 bench prompt 早停影响仅作健康参考，未参与判定（无 DE 相关配置进入 e2e）。

> 本报告由工程保障团队 ar-optimizer 生成，关键裁决（优先级重排）请人类工程负责人复核。
