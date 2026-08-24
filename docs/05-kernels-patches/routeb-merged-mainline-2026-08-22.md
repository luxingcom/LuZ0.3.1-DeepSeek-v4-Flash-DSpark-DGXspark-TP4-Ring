# merged-GEMM 插件主线收官：p29 NaN 定论 + 挂起复现 + 三臂 PR 矩阵（Task #4，2026-08-22）

**执行**: mainline-engineer-2（系统架构师，第三任接管，前两任状态全部自磁盘恢复）
**集群**: node01-04 · TP4 · DeepSeek V4 Flash（GB10/sm_121a）· batched 4096 · MTP dspark n=7
**窗口**: 2026-08-22 03:50 – 06:01 UTC
**结论一句话**: **插件 v2.1（merged + Triton 长尾架构）三臂实测 PR/DE 全面深负（-73%~-79%），判 No-Go，生产已回退基线并验证恢复；NaN 根因已定位并修复（torch.zeros 初始化）；45min 挂起不复现（v2.1 预编译有效）；插件资产封存待"纯 merged 混合路由"再评估。**

---

## 0. 执行摘要（判据对照）

| 判据（batch-analyst §3 Phase 0） | 阈值 | 实测（主臂 MIN_M=256） | 判定 |
|---|---|---|---|
| PR 四档 vs 基线 2510/2500/2420/2270 | ≥+10% 保留 / <-3% 回退 | **573.5/588.1/581.6/573.6（-77.2/-76.5/-76.0/-74.7%）** | **❌ 硬崩 → 回退** |
| DE C1/C12（接受率归一 ±5%，基线 92.8/408） | 带内 | **C1 ~19.5-32（-66%~-79%）、C12 ~100-109（-73%~-75%）**，接受率 3.2-5.0/轮（健康，基线带 3.65-4.4） | **❌ 硬崩（纯算力侧，非接受率）** |
| Needle 64K/128K | 输出无垃圾 | 64K 2/3、128K 0/2（基线自身 1/3~4/5 噪声带内；输出连贯无数值垃圾） | ⚠️ 噪声带内 |
| KV ≥ 5.15M | 带内 | **5,565,335 tokens** | ✅ |
| 45min 首爆挂起 | 不复现 | **不复现**（首 merged prefill 1634tok ≤30s 完成 + 加固探针全过） | ✅ |

**根因归因（∞ 臂隔离证实）**: ∞ 臂（纯 Triton，merged 永不触发）PR = 559/573/566/558 tok/s，与主臂（573/588/582/574）在轮间噪声内**统计不可区分**（差 ≤3%）。**-77% 回归 100% 由 Triton W4A16 grouped 路径贡献；合成语料下 merged 桶覆盖≈0（exact-set 桶 ≥256 几乎不存在），merged GEMM 的潜在收益完全被尾部拖累掩盖。**

---

## 1. 任务 1：Triton W4A16 间歇性 NaN——定论 ✅

### 1.1 证据链（前两任 + 本任补全）

| 步骤 | 工件 | 发现 |
|---|---|---|
| p29（前任 02:49 写并跑） | p29_triton_hunt.py | uniform 路由 PASS / 随机路由 NaN（真实权重 256 experts） |
| p30（前任 02:52 写并跑） | p30_triton_bisect.py | random M=16 **单独跑 PASS** → 非输入依赖，指向内存/状态 |
| p31（前任 02:53 写） | p31_state_bisect.py | uniform M=4096 大 case → random M=16 **复现 NaN** → 分配器复用位型 |
| 根因（前任 03:29 修复） | triton_moe.py v2.1+ | **`grouped_linear` 的 `out=torch.empty` 未初始化**：哨兵块早退 + pad 行掩码不写 → 未初始化行含垃圾位型（含 NaN/Inf）→ 加权 scatter `y * w_pad(0)` = `0×NaN=NaN` → `index_add_` 污染输出。间歇性 = 取决于分配器复用内存的位型 |
| 修复 | `out = torch.zeros(...)` | md5 `00c6ada9…` 已部署 TP4 生产并运行（本任核验三处一致：work 源/deployed/dist-packages） |

同批还有两项 CUDA graph 捕获期修复（均实证时间戳）：`bincount` 含 CPU↔CUDA 拷贝捕获期非法（03:26）→ 纯 GPU `index_add_`；空批次 M=0 捕获期越界（03:07）→ 零张量早退。

### 1.2 p32 后置验证（本任，停机窗口 GPU 独占运行）✅ 定论达成

- 设计：真实权重全 256 experts + p29 全场景（uniform/random×M×seed/hot/invalid-ids）+ p31 状态依赖序列（uniform M=4096 → random M=16）+ **NaN 毒化分配器**（预分配 NaN/Inf 大块后释放，复现最恶劣 uninit 位型）+ **阴性对照**（复刻 v2.0 旧 `torch.empty` 路径，证明 harness 能捕获该 bug）
- 结果（05:46-05:47，GPU 独占，free 104GiB）：
  - **Part A 修复活性**：cap=3936 中 3840 个未写行（哨兵+pad）全部为 0（期望 0）→ zeros 修复生效
  - **Part B 阴性对照**：旧 empty 路径 + NaN 毒化分配器 → **NaN 复现**（证明 harness 能捕获该 bug、根因诊断成立——同一 harness 换 zeros 即清洁）
  - **Part C 修复后全场景 18/18 PASS**：全部 rel=3.9e-3~4.9e-3（W4A16 本征水平）、零 NaN。含 C2b「大 case 后小 case」（p31 复现序列）与 C3「NaN 毒化后小 case」（最恶劣位型）——修复前正是这两类场景间歇 NaN
- **定论：`NAN_FIX_VERIFIED + NEGCTRL_REPRODUCED`。Triton W4A16 间歇 NaN 根因 = `torch.zeros` 修复前 `torch.empty` 未初始化行的 0×NaN 污染；修复后（已部署生产）全场景清洁。**
- 附注：生产运行期间（本任 6+ 个探针请求 decode 全量走 Triton 路径）输出全部连贯、无 NaN 垃圾——与修复一致的现场证据。

---

## 2. 任务 2：45min 首爆挂起——不复现 ✅

- 前任 watcher 失败根因：`docker top … | grep -i enginecore` 永不命中——comm 列 15 字符截断为 `VLLM::EngineCor`。本任修复 pid 发现（comm 前缀匹配 + `pgrep -f '^VLLM::EngineCore$'` 锚定兜底），并重写 watcher（pyspy_hang_watch2.sh）：**双进程监控**（EngineCore + Worker_TP0——挂起特征进程是 Worker）+ **瞬时 CPU**（/proc/stat 帧间差分；v1 用生命周期累计 %CPU，98% 挂起需 ~1h 才爬过 85% 阈值，属测量学 bug）
- 实测：首个 merged prefill（1634 tok ≥ MIN_M=256）**≤30s 完成**（首轮轮询即 DONE），64 tok decode 输出连贯
- 加固探针：1860-tok prefill 4.4s / 265-tok 2.2s / 4 并发×~800 tok 全部 6-7s 完成
- 机理侧证：v2.1 的 P0-A 启动期预编译确实执行（容器 cutlass cache 03:33/03:36 新编译条目）；DSL M_pad 档（256-16384）启动期吸收
- 结论：**v2.1（预编译 + Triton warmup + 捕获安全三修复）已消除 v2.0 时代的首 chunk 冷编译挂起条件。VERDICT=NO_HANG 落盘**（pyspy_hang2_040224/）
- 插件 INFO 日志被 vllm 日志配置吞掉（`init_logger(__name__)` 非 vllm 命名空间）——外观问题，建议后续插件日志用 vllm 前缀 logger 或 stderr

---

## 3. 任务 3：三臂 PR 矩阵

### 3.1 主臂（MIN_M=256，即生产插件配置）——04:07-04:59 UTC

**PR 四档（panorama prefill 口径：唯一 nonce、输出 1 token、pt/TTFT、3 轮中位；与基线 Task #25 同脚本同公式同 token 数）**

| 档（实际 tok） | 主臂 TTFT 中位 | PR | 基线 PR | Δ |
|---|---|---|---|---|
| 4K（8,203） | 14.30s | 573.5 | 2510 | **-77.2%** |
| 16K（32,773） | 55.73s | 588.1 | 2500 | **-76.5%** |
| 32K（65,544） | 112.70s | 581.6 | 2420 | **-76.0%** |
| 64K（131,082） | 228.51s | 573.6 | 2270 | **-74.7%** |

- 轮间方差 ±1%（564-588），信号极稳；基线侧 round1 无慢轮（3.4/3.3/3.2s）→ 排除方法论差异
- 全档水位 ~575-590 tok/s 平坦 → 与 prefill 长度无关的**单位 token MoE 计算成本 ×4.3**

**DE（de_acc.py，chat 口径 512 tok，接受率取 /metrics 差分；接受率解析两 bug 已修：401 无 auth header + 指标名后带 `{labels}` 导致 startswith 不命中——.bak 留档）**

| 项 | 主臂 | 基线 | Δ | 接受率（acc/draft_tokens×7/轮） |
|---|---|---|---|---|
| C1 | 19.5-32.0（中位 ~24） | 92.8 | **-66%~-79%** | 3.85-5.0（健康） |
| C12 | 99.6-111.1（中位 ~105） | 408 | **-73%~-75%** | 3.2-3.7（健康） |

**关键读数：接受率在基线带内 → DE 回归 100% 是算力侧（Triton decode 步时 ×4），不是数值质量。** 数值质量与 needle"输出无垃圾"互相印证。

**Needle**: 64K 2/3、128K 0/2（基线自身跨 run 1/3~4/5 噪声带内；失败模式为指令回声而非乱码 → 非数值破坏）。

### 3.2 ∞ 臂（MIN_M=99999999，纯 Triton 尾，merged 永不触发）——05:02-05:45 UTC

| 档 | ∞ 臂 TTFT 中位 | PR | 主臂 PR | 基线 PR | Δ vs 基线 |
|---|---|---|---|---|---|
| 4K | 14.67s | 559.2 | 573.5 | 2510 | **-77.7%** |
| 16K | 57.23s | 572.6 | 588.1 | 2500 | **-77.1%** |
| 32K | 115.79s | 566.0 | 581.6 | 2420 | **-76.6%** |
| 64K | 235.2s | 558.4 | 573.6 | 2270 | **-75.4%** |

- **∞ 臂 ≈ 主臂（差 ≤3%，轮间噪声内）** → 尾隔离定论：合成语料（panorama FOX 重复文本）下 merged 桶覆盖率 ≈ 0，主臂与纯 Triton 臂行为相同
- DE 不重跑：decode 路径两臂完全相同（M<256 恒 Triton），主臂数据即 ∞ 臂数据
- KV：5,518,544 tokens（主臂 5,565,335；差 47K ≈ 0.8%，插件 w2 combo 缓存驻留所致，均 ≥5.15M ✅）
- Needle：64K 1/3、128K 2/2（合计 3/5；主臂 2/5；均在基线噪声带 1/3~4/5 内；失败模式为指令回声而非乱码 → 非数值破坏，与 p32 定论互证）

### 3.3 回退臂（恢复 8/21 原版脚本 = 无插件基线）✅ 基线行为恢复

- 执行（05:46-05:54，pipeline_final.sh 自动链）：停集群 → p32（见 §1.2）→ 恢复 `start_tp4_{head,worker}.sh.bak-e2e-20260821`（插件部署前原版，含 checker 校验通过）→ 全链重启
- 行为验证：`Using B12xExperts`（原类，非 MergedB12xExperts）✅；KV **6,065,597 tokens**（基线水平；插件臂为 5.52-5.57M，插件驻留约占 0.8-0.9% KV）✅；READY 480s 正常 ✅
- PR 四档抽查（冷启动后 + 10 发短预热）：

| 档 | 回退臂 PR | 基线 | Δ | 备注 |
|---|---|---|---|---|
| 4K | 1895（冷态中位） | 2510 | -24.5% | **冷启动残差**：热态复测 5 发 = 2432/2444/2425/1906/2420，中位 ≈2425（-3.4%，正常方差内） |
| 16K | 2364 | 2500 | -5.4% | 正常方差 |
| 32K | 2355 | 2420 | -2.7% | 正常方差 |
| 64K | 2187 | 2270 | -3.6% | 正常方差 |

- **结论：env 关（脚本原版恢复）= 基线行为，回退验证通过。生产现以基线运行。**

### 3.4 KV

主臂 KV cache 5,565,335 tokens（≥5.15M ✅；插件派生驻留未挤占 KV 预算）。

---

## 4. 归因分析

**证据三角**：
1. 主臂（merged 可触发）PR ≈ 573-588；∞ 臂（merged 永不触发）PR ≈ 558-573——**两臂统计不可区分（差 ≤3%，轮间噪声内）**
2. 回退臂（无插件 B12X）PR 立即恢复 ~2355-2425（-3% 内）
3. DE 接受率健康（3.2-5.0/轮，基线带内）→ 回归与数值质量无关

**推论链**：
- **第一层（直接原因）**：Triton W4A16 grouped kernel（BM=16/BN=64/BK=128 小 tile + 逐元素 LUT 解包）在 prefill 与 decode 两个 M 域都比 B12X 慢 ~4.3×（估算 ~3-4% 峰值效率 vs B12X ~13%）。**prefill（-77%）与 decode（-75~-79%）同一根因。**
- **第二层（架构性）**：v2.1 设计"B12X 完全退出、Triton 接管全部长尾+decode"——在 Triton kernel 当前效率下，这个替换在所有 M 域都是净负。batch-analyst 预警的双向风险 #3（"若 Triton 尾部 prefill 慢于 B12X，大 M 场景可能净负"）**以最坏形式兑现，且不止大 M，decode 同样中招**。
- **第三层（merged 侧未定罪也未辩护）**：合成语料（panorama FOX 重复文本）下 exact-set 桶 ≥256 几乎不存在（主臂≈∞ 臂），merged GEMM 的收益/成本在本矩阵中**不可测**。按 batch-analyst §2.2，真实流量 ~27% 热桶下 merged 理论收益存在（单流 PR ×1.4 分析级），但当前架构下尾部净负会吞掉大部分收益。
- **数字自洽性核验**：8.2K tok prompt = 8×1024 chunks（threshold 钳制），每 chunk M=1024 → 27% 理论 merged 覆盖 + 73% Triton 尾（慢 4.3×）+ 27% merged（快 ~2×）→ 加权 ≈ 0.73×(1/4.3) + 0.27×2 ≈ 0.44×——与实测 0.23×（574/2510）同数量级（attention 等非 MoE 部分占 45% 不变会稀释回归，实际比此模型略深，说明 Triton 在 M=1024 档或比 4.3× 更慢，与 probe 的"65T@1024 对单流偏乐观"警示一致）。

**性能口径备注**：panorama "4K/16K/32K/64K" 档的实际 prompt tokens 为 8.2K/32.8K/65.5K/131.1K（FOX≈10 tok/rep，length//5 rep）——与基线同公式同 token 数，对比有效。

---

## 5. 生产处置（已执行完毕）

- **判据触发**：PR <-3% → 回退（实际 -77%，远超阈值）
- **执行结果**（pipeline_final.sh，05:46-06:01 全自动）：
  1. 停集群 → p32 NaN 修复定论验证（§1.2，18/18 PASS）
  2. 恢复 `start_tp4_{head,worker}.sh.bak-e2e-20260821` 四节点（插件部署前原版；checker 全过）
  3. 全链重启：`Using B12xExperts` + KV 6,065,597 + PR 抽查恢复基线（§3.3）
- **生产终态 = 基线运行**（无插件 env、无 pip 前缀、原版脚本）；插件资产封存于 `<INSTALL_DIR>/nvfp4/{plugin_merged, routeb_official_v2}`（无 env 不激活，零污染结构性保证）
- enabled 与否留用户裁定：如需重新启用，`start_tp4_*.sh.bak-fix-20260822` 含 v2.1 全套 env（MIN_M=256），但**基于本报告数据不建议在 Triton kernel 重写前启用**

## 6. 工件清单

| 位置 | 内容 |
|---|---|
| 01:/tmp/_routea_work/ | p29-p32 全套脚本+日志 / bench_{arm0a,arminf,rollback}.log + pr_/needle_/de_*.json / pyspy_hang2_040224/（NO_HANG 现场）/ pipeline_{inf,final}.log / bench_panorama_prefill.py / de_acc.py（两 bug 修复后）/ pyspy_hang_watch2.sh / arm0a_bench.sh / arm_inf_bench.sh / pipeline_{inf_arm,final}.sh / stop_cluster.sh / switch_arm.sh / final_rollback.sh |
| 本地 deliverables/ | 本报告 |

## 7. 建议与后续

1. **Triton W4A16 grouped kernel 是本架构的性能死穴**（prefill 与 decode 皆 ×4.3 慢）：BM=16/BN=64/BK=128 小 tile + 逐元素 LUT 解包，估算 ~3-4% 峰值效率。若重走"merged 主路 + 尾"混合架构，**尾部必须保留 B12X**（v2.1 的"B12X 完全退出"设计在实测数据面前不成立）
2. **merged GEMM 本身的 e2e 收益在本矩阵中不可测**（合成语料 merged 桶覆盖 ≈ 0，主臂=∞ 臂）——与 batch-analyst §2.2 "合成语料中位窗口加速 1.0×" 预警一致。真实流量路由重采集（P1-1 采集器现成）仍是任何后续 A/B 的前置门；若真实流量确有 ~27% 热桶，"merged（热桶）+ B12X（尾）"的混合路由仍是值得重估的架构——需要的只是把 Triton 从架构中拿掉、尾路径回退 B12X
3. **可复用资产**：NaN 根因+修复（torch.zeros）与三项 CUDA graph 捕获期修复；py-spy watcher v2（comm 截断 + 瞬时 CPU 两测量学修复）；de_acc.py 两处指标解析修复（auth header + Prometheus label 后缀）；arm 切换/停机/回退三脚本（switch_arm/stop_cluster/final_rollback，全自动流水线模式）
4. 插件 INFO 日志建议改用 vllm 命名空间 logger（当前被吞，靠 cutlass cache 时间戳侧证 warmup 执行）

## 8. 时间线（UTC）

| 时间 | 事件 |
|---|---|
| 03:50 | 本任接管（前两任状态磁盘恢复：p29/p30/p31 已跑、NaN 根因已修已部署、TP4 已带修复重启） |
| 03:56-04:02 | p32 首跑受阻（生产占 GPU 100.5GB/121GB，swap 耗尽，CUDA context 无法创建）→ 改停机窗口执行；watcher v2 上线（pid 截断/瞬时 CPU 修复） |
| 04:02 | 挂起测试：首 merged prefill（1634 tok）≤30s 完成 + 变尺寸/并发探针全过 → NO_HANG |
| 04:07-04:51 | 主臂基准：PR 四档 -77% / DE C1 -79% C12 -75%（接受率健康）/ needle 2/5（噪声带内）|
| 04:58-05:01 | 停机流水线 #1：stop → p32 首跑失败（routeb-v2 未挂载）→ ∞ 臂切换 |
| 05:03-05:45 | ∞ 臂（MIN_M=99999999）重启 + 基准：PR 四档 -75~-78%（=主臂）→ Triton 尾定罪 |
| 05:46-05:47 | 停机流水线 #2：p32 修复版 18/18 PASS → NAN_FIX_VERIFIED + NEGCTRL_REPRODUCED |
| 05:47-05:54 | 回退基线（.bak-e2e-20260821 恢复 + 全链重启）：Using B12xExperts / KV 6.07M |
| 05:54-06:01 | 回退臂 PR 抽查 + 4K 热态复测（2425，-3.4% 正常方差）→ 基线行为确认 |
| 06:01 | 生产终态 = 基线运行；报告定稿 |
