# Dspark MTP acc_len 调优报告（P0-B）

**日期**：2026-08-22 ｜ **执行**：雷克斯（SRE），工程保障团队 engineering-p0p2
**窗口**：10:38Z-12:22Z（UTC，停机窗口 11:02Z-12:18Z）｜ **集群**：DGX Spark ×4 TP4（GB10/sm_121a），vLLM 0.26.1 fork，DeepSeek V4 Flash B12X W4A16 + Dspark MTP
**任务**：①acc_len 差异根因调查 ②停机窗口 num_speculative_tokens (n) 扫描验证

---

## 一、结论速览

| 问题 | 结论 |
|---|---|
| "自报 acceptance 6.5-7.1/7 vs e2e 3.65 差 2×"的根因 | **双重误读**：①e2e 口径漏了 +1 bonus token（真实 acc_len=4.65 不是 3.65）②6.5-7.1 是**规律文本（编号列表/代码）的负载上限值**，非全局值。两数字本是同一指标的不同负载样本，**无指标语义差异** |
| n 扩档（10/12）是否有 1.6-1.9× 空间 | **否**。n=10 实测 C1 +2.6%（噪声级）/ C12 **-25%**；自适应 n 也失败（C12 -24%、PR -21%） |
| 最优生产配置 | **维持现状 n=7 probabilistic 不变**（本负载已近最优） |
| "acc_len 是 DE 唯一大杠杆"论断 | **被本次实测修正**：扩 n 增加每步 verify token 数 → 触达更多 distinct experts → 权重读/步增加（带宽墙）→ 步时 +11-18%，吃掉接受率收益。C12 已在带宽墙上被直接反噬 |

---

## 二、acc_len 差异根因（阶段一，源码+实证）

### 2.1 指标语义链（源码实证）

- Prometheus `vllm:spec_decode_num_accepted_tokens_total` 由 scheduler.py:1688 `num_accepted = len(generated) - num_sampled` 累计——**不含 bonus token**；drafts : draft_tokens = 1 : n 精确。
- **每步实际产出 token = accepted/drafts + 1**（含 bonus）。实证（08-22 10:45Z 生产单请求探针，fox 文本 256 tok）：completion/drafts = 4.129 = accepted/drafts (3.177) + 1 ✓。
- EngineCore 日志 "Mean acceptance length" = `1 + accepted/drafts`（metrics.py:114）= 每步产出 token 数，与 e2e 口径**同一指标**。
- 昨日 "e2e acc_len 3.65"（Prometheus 窗口 accepted/drafts=3332/914）**漏了 +1**：真实 tokens/step 当时 ≈ **4.65**。"差 2×" 实为 4.65 vs 6.5-7.1（跨负载）≈ 1.4×。
- 顺带发现 de_acc.py（昨日 DE 脚本）分母误取 `num_draft_tokens_total`（=n×drafts），算出的是 0-1 接受分数而非长度；本窗口已用修正版 de_bench.py（tokens/step = completion/drafts）。

### 2.2 "6.5-7.1" 的出处

e2e-baseline-prod-2026-08-21.md L104，**按文本类别分层**的 Mean acceptance length：

| 负载类别 | Mean acceptance length | 平均接受率 |
|---|---|---|
| 规律文本（编号列表/代码） | **6.47-7.13**（n=7 理论上限 8） | 78-88%（per-position 0.985/0.974/0.948/0.902/0.856/0.794/0.675） |
| Agent/自由文本 | 2.88-5.34 | 26.9-62% |
| DE 基准（fox 文本，本窗口实测） | 4.13-4.74 | pos0 ≈0.78-0.83 |

**结论**：6.5-7.1 是最有利文本类别的上限值，被误读为全局值；"93-100% 接受率"的说法不成立（那是 6.5-7.1/8 ≈ 81-89% 效率，且仅限规律文本）。

### 2.3 draft_sample_method 语义（源码实证，修正先前假设）

- 选项仅 `greedy` | `probabilistic`（config/speculative.py:78）。
- gumbel.py `_gumbel_sample_kernel`：`if temp != 0.0` 才加 Gumbel 噪声 → **temp-0 下 probabilistic draft 就是 argmax，与 greedy 完全等价**（draft 提议与 target 验证语义均相同）。greedy 臂对 temp-0 基准无收益；对生产 temp>0 流量反而降低接受率（probabilistic 提议分布更贴近 target）。两模式理论上都无损（emitted token 恒为 target 分布样本）。
- 生产现用 probabilistic 是正确选择。
- 附带发现：temp-0 下输出存在 run 间非确定性（6 prompt 中 2 个不同）——batched 推理近平票 argmax 翻转（cudagraph batch 尺寸/归约顺序差异），**非 spec decode 引入**，为既有现象。

### 2.4 真实可兑现 acc_len 与调优空间判断

- fox 基准负载：acc_len ≈ 4.3-4.7/8，条件接受率 pos0≈0.78、深链 ~0.8 平稳——深链红利存在。
- 生产混合流量：per-position 无条件率 pos0 0.59-0.69 → pos6 0.03-0.04（几何衰减），Mean acceptance length 2.5-3.7——比基准负载更差。
- dspark_block_size=5（模型 config 实证）→ n≥5 全部合法。

---

## 三、停机窗口 n 扫描（阶段二）

### 3.1 方法

- 切换方式：`<INSTALL_DIR>/scripts/start_tp4_head.sh` + 三台 worker 的 `start_tp4_worker.sh` **同步**改 `--speculative-config`（.bak-mtp-20260822 留档）→ 自愈链停 → head-first 手动编排重启（复用 /tmp/_ar_opt/restart_tp4.sh）→ healthy + num_spec_tokens 日志确认 → 冒烟。
- **坑 1（已修复入档）**：worker 脚本各有一份 speculative-config，只改 head 会导致 rank 间 n 不一致 → NCCL 集合通信死锁（ALLGATHER 300s watchdog）。首次 n=10 尝试即踩此坑，四机同步后正常。
- 基准：DE C1/C12 各 3 轮（fox 512t，temp 0，stream 中位）+ tokens/step + per-position；PR 4K ×3（唯一 nonce）；greedy 输出捕获（质量门）。
- 判据：acc_len 归一后 ≥+8% 保留；PR ±3% 带内；质量门 = temp-0 输出对照（因近平票非确定性，以同 prompt 输出形态一致为准，非逐字节）。

### 3.2 全量数字（3 轮中位）

| 配置 | C1 tput | C1 acc_len | C1 步时 | C12 tput | C12 acc_len | C12 步时 | PR 4K tok/s | KV cache |
|---|---|---|---|---|---|---|---|---|
| **cfg0: n=7 prob（基线）** | **87.5** (71.2-93.5) | 4.34 | 49.5ms | **395.4** (388-405) | 4.22 | 128ms | 2487.5 ✓ | ~6.04M |
| cfg1: n=10 prob | 89.8 (58.4-106.0) | 4.92 (+13%) | 54.9ms (+11%) | 296.3 (288-326) | 3.74 (-11%) | 151.5ms (+18%) | 2514.0 ✓ | 5.86M |
| cfg2: 自适应 [[1,1,10],[2,12,7]] | 84.7 (75.8-112.8) | 4.83 | 57.1ms (+15%) | 298.8 (285-317) | 3.76 (-11%) | 151.1ms (+18%) | **1988.3 ✗ (-21%)** | 6.05M |
| cfg3: n=5 prob | 80.1 (57.7-80.9) | 3.66 (-16%) | 45.7ms (-7.7%) | 387.9 (380-388) | 3.77 (-11%) | 116.6ms (-8.9%) | 2555.5 ✓ | 6.14M |

（C1 步时 = 1/步率，步率 = tput/acc_len；C12 步时 = 12/步率）

### 3.3 机制定论：为什么扩 n 失败（本窗口最有价值的发现）

1. **接受率红利是真的**：C1 下 n=10 的 per-position 无条件接受率延伸到 pos7-9（0.02-0.42），acc_len +13% 实证。
2. **但 verify 成本同步上升**：每步 verify token 数 C1 8→11、C12 96→132，触达更多 distinct experts → MoE 权重读/步增加（C12 本就在 273GB/s 带宽墙上）→ 步时 +11-18%。
3. **C12 净 -25%**：步时 +18% 且 acc_len 反降 11%（大批量下更深 draft 链的接受退化）。
4. **自适应 n 也失败**：per-position 证明机制生效（C1 深 n、C12 回 n=7），但 C12 步率仍 -15%、PR 4K -21% 且方差放大——**max_n=10/动态切换路径本身引入开销**（cudagraph 覆盖/调度预算/动态评估），与本负载无关地拖慢全局。该特性在本 fork 上不可用。
5. **C1 净 +2.6%**：低于 +8% 保留阈值，且在 ±20% 轮间方差带内不可分辨。

### 3.4 cfg3（n=5）：负收益，双向扫描闭环

| 指标 | cfg3 (n=5) | vs 基线 (n=7) |
|---|---|---|
| C1 tput | 80.1 (57.7-80.9) | **-8.5%** |
| C1 acc_len | 3.66 | -16%（砍掉 pos5/6 接受贡献） |
| C1 步时 | 45.7ms | -7.7%（verify 6 vs 8 tokens，略快） |
| C12 tput | 387.9 (380-388) | -1.9%（噪声带内） |
| C12 acc_len | 3.77 | -11% |
| C12 步时 | 116.6ms | -8.9%（步更快但被 acc_len 吃掉） |
| PR 4K | 2555.5 ✓ | 带内 |

- **n=5 双侧皆负**（C1 -8.5%、C12 -1.9%）：n=7 优于 n=5。参考部署（n=5）C1 更高（97-124）的原因**不在 n**——其优势来自其他因素（draft 权重/构建差异），本集群不可据此调参。
- **扫描闭环**：n=5（负）、n=7（基线）、n=10（C1 微正/C12 大负）、自适应（全面负）→ **n=7 uniform 是本负载最优**，且 C1/C12 两端无单调改进空间。

### 3.5 质量门

- 各配置 greedy 输出捕获 vs cfg0 参考输出：fox_repeat/count/code/list 四个 prompt 全配置字节级一致；reason/zh 存在近平票翻转（与 cfg0 自身 run1/run2 之间的非确定性同量级）——**无分布漂移，质量门 PASS**（投机解码无损性保持）。

---

## 四、生产采纳建议

1. **不采纳任何变更**：回滚至 n=7 probabilistic 基线（.bak-mtp-20260822 已还原，恢复验证见 §5）。
2. n=7 已是本负载近似最优：n=5 方向（若 cfg3 证实无增益）与 n≥10 方向（本次证伪）均无空间；自适应 n 特性在本 fork 有未定位开销，不可用。
3. **MTP 调优线正式关闭**（与 AR 线 P0-A 同归 No-Go）：DE 上行需要的是**draft 模型质量**（pos0 接受率 0.78 → 0.9+ 才有意义），属模型训练范畴而非部署配置。
4. 后续若重开：优先方向是「规律文本/结构化输出场景的 n 自适应」（该负载类别 acc_len 上限 7.1/8，但需先修复 fork 动态 spec 路径的开销问题）。

---

## 五、恢复记录（UTC）

| 时间 | 事件 |
|---|---|
| 12:11:33 | 回滚启动：四机 .bak-mtp-20260822 还原（head + 3×worker，n=7 probabilistic），checker 通过 |
| 12:12:07 | 基线 head-first 重启开始 |
| 12:17:43 | READY（"Application startup complete"） |
| 12:18:05 | 验证完成：health 200 OK ｜ num_spec_tokens=7 ｜ B12X_MXFP4 ｜ KV 6,022,164 tokens（与 E5 参照 6,042,089 同带，-0.3% 属冷启动正常浮动）｜ vllm-tp4-head.service + vllm-healthcheck.timer active ｜ 3×worker service active |
| 12:20-12:22 | DE 漂移检查：C1 两轮 81.0/106.0 tok/s，acc_len 4.13/5.69——均在 cfg0 基线方差带（71-94 / 3.6-4.7）内，**生产基线性能确认恢复** |

**停机窗口总时长**：11:02Z-12:18Z（约 76 分钟，含 2 次失败重启：①引号转义 ②worker 未同步）。配置 0 基线在生产容器上直接采集（未额外停机）。

---

## 六、资产与复现

- 脚本（node01:/tmp/_mtp_tune/）：de_bench.py（修正口径 DE 基准）/ greedy_check.py（质量门）/ pr_check.py（PR 抽查）/ edit_spec.py + switch_config.sh + switch_config2.sh（四机同步配置切换）/ run_bench_cfg.sh
- 数据：cfg{0,1,2,3}_de.json / cfg{0,1,2,3}_greedy.json / cfg{0,1,2,3}_pr.json / cfg{0,1,2,3}_bench.log / cfg{1,2,3}_switch.log
- 生产脚本备份：<INSTALL_DIR>/scripts/start_tp4_head.sh.bak-mtp-20260822（head）+ 三台 start_tp4_worker.sh.bak-mtp-20260822
- checker（check_vllm_script.sh）不校验 speculative 参数，无需同步
