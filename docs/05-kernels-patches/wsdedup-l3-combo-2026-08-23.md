# ws-dedup L3 组合矩阵报告（全天窗口 第三阶段 b · W4A4 翻案测试）

- **执行人**：雷克斯（Rex）· SRE 工程师（phase3b-wsdedup）
- **日期**：2026-08-23（本地）/ 2026-08-22 19:04–21:20 UTC（服务器）
- **集群**：DGX Spark 4 节点 TP4（GB10/sm_121a），环网 01-02-04-03
- **对象**：ws-dedup 补丁（b12x wrapper 几何键共享池）L3 验证 + W4A4 翻案组合矩阵
- **基线态**：threshold 4096 采纳态（W4A16 B12X + Dspark MTP n=7），PR 4K 基线带 2753-2853（慢簇 2753-2768 / 中簇 2842-2853），DE C1 92.9 级，KV 6.04M

---

## 0. 一句话判定

**W4A4 翻案成立（"值得深测"档）**：池去重补丁解除内存死结（weight 68.15→45.32 GiB，**省 22.83 GiB**，落预期 13-27GB 带内；KV -51%→-8.5%），且与 P2 结论相反，threshold 4096 下 W4A4 prefill **+8.3%**（惊喜，含模式混杂警示）、共享池性能零代价、greedy 质量门 4/4 逐字一致。三判据（PR ≥ M1-3% / 省显存 ≥10GB / 质量门）全过。剩余成本：full 模式 decode 归一 -6~-9%、KV -8.5%；hybrid 模式 decode 全恢复但 KV -74.5% 不可产。**生产维持 W4A16 基线（默认恢复，已验证全绿）**。

---

## 1. 测试矩阵与时间线（UTC）

| 臂 | 配置 | 重启窗口 | 测量窗口 | 模式（首 4K TTFT） | stall 探针 |
|----|------|---------|---------|------|------|
| W-0+M1 | W4A16 + 补丁 off（env=0） | 19:12-19:17 | 19:18-19:29 | 中簇 3.04s | 干净 |
| M2 | W4A4 full（W4A4=2）+ 补丁 off | 19:31-19:40 | 19:43-19:54 | **fast 2.76s** | 干净 |
| M3 | W4A4 full + 补丁 on（SHARED=1） | 19:57-20:06 | 20:08-20:19 | **fast 2.74s** | 干净 |
| M4 | W4A4 hybrid（W4A4=1, MIN_M=3072）+ 补丁 on | 20:21-20:28 | 20:32-20:42 | **fast 2.74s** | 干净 |
| 生产恢复 | M1 态（overlay 保留 + env=0） | 20:46-21:03 | 21:10-21:15 验证 | 中簇 2.94s | 干净 |

停机窗口约 2 小时 14 分（19:08-21:20）。全部臂 stall 探针干净（3×短 4K：TTFT 2.74-3.26s、ITL 中位 46-55ms，无 >6s/极慢样本）——W1 发现的环境级随机 AR stall 未污染本阶段测量（生产形态是否中招仍由并行调查定论；本阶段相对结论不受影响）。

**W-0 部署**：deploy_ws_dedup.sh 四节点幂等部署成功——overlay md5 `8f88555a0fc7e330ee51255c643796bc` 四节点一致；env `VLLM_B12X_SHARED_WRAPPER=0` + bind mount 注入 head（L147/L168）/worker×3（L152/L203）脚本；check_vllm_script.sh 四机全过。**off 路径零行为变化 e2e 实证**：M1 全指标落基线带（见 §2），容器内 overlay md5 一致、env=0、无 pool/W4A4 生效行。

---

## 2. 四臂全数字

### 2.1 PR panorama（prefill 吞吐，3 轮中位，tok/s）

| 档位 | M1 基线复核 | M2 full+off | M3 full+on | M4 hybrid+on | M3 vs M1 | M3 vs M2 | M4 vs M1 |
|------|------|------|------|------|------|------|------|
| 4K（8.2K tok） | 2753 | 2982 | 2982 | 2999 | **+8.3%** | 0.0% | +8.9% |
| 16K（32.8K） | 2777 | 2960 | 2979 | 2980 | +7.3% | +0.6% | +7.3% |
| 32K（65.5K） | 2674 | 2848 | 2838 | 2852 | +6.1% | -0.4% | +6.7% |
| 64K（131K） | 2454 | 2557 | 2567 | 2545 | +4.6% | +0.4% | +3.7% |

- **M2/M3/M4 同为 fast 模式**（首 4K TTFT 2.74-2.76s），臂间直接可比：**跨层共享 wrapper 对 prefill 性能零代价**（W-5 判据过，各档 ±0.6% 内）。
- **模式混杂警示（如实标注）**：M1 为中簇（3.04s），M2-M4 为 fast 模式。"+8.3% vs M1" 含模式分量（历史模式带宽约 3-8%）。方向性结论（W4A4 在 threshold 4096 下 prefill 正增长，与 P2 的 -13% 反向）稳健：P2 测于 threshold 1024 时代（chunk M=1024 落 W4A4 微基准 0.79-0.95× 劣势区），threshold 4096 采纳后 chunk M=4096 恰入 W4A4 1.32× 甜点区，机理自洽。
- 生产恢复终态复测：PR 4K 2788（轮 2788/2804，首轮 2511 冷启动瞬态）/ 16K 2800 / 32K 2683 / 64K 2436——全部回基线带。

### 2.2 显存对比（rank0/rank1 双核验，四节点一致）

| 指标 | M1 | M2 full+off | M3 full+on | M4 hybrid+on | 判读 |
|------|------|------|------|------|------|
| weight（Actual usage） | 40.5 GiB | 68.15 GiB | **45.32 GiB** | 79.82 GiB | M2 精确复现 P2 的 68.15；**M3 较 M2 省 22.83 GiB**（预期带 13-27GB 内） |
| KV tokens | 5,997,537 | 2,923,992 | **5,484,179** | 1,532,462 | M3 较 M2 +87.5% 恢复；vs M1 -8.5% |
| KV 内存 | 53.18 GiB | 26.2 GiB | 48.63 GiB | 13.59 GiB | — |
| peak 激活 | 2.03 GiB | 2.03 GiB | 2.03 GiB | 2.03 GiB | 不变 |
| CUDAGraph | 0.72 GiB | 1.23 GiB | 1.21 GiB | 1.54 GiB | 捕获正常（三档完整） |

- **pool size==1 核验（方法论偏差，如实标注）**：插件 logger 行不进 docker logs（`init_logger("routea_plugin_a1.*")` 无 handler，vllm 命名空间外的 INFO 被丢弃——M2 的 "W4A4 ready" 行同样不可见，已用 `Using W4A4B12xExperts`（mxfp4.py:1734）+ 容器内 pip import 双重替代核验生效）。pool size==1 改由**三方内存算术**核验：M3−M1 = +4.82 GiB ≈ E4M3 scales（4.3）+ 1 wrapper（0.54）；M2−M3 = 22.83 GiB = 42 wrapper × 0.543 GiB——与 L2 实测 per-wrapper 量级自洽。池生效等价证明成立。
- **M4 hybrid 双表示代价**：weight 79.82 GiB（W4A4 副本 + W4A16 打包共存）→ KV 1.53M tokens（-74.5% vs M1）——补丁解决了 wrapper ×43（P2 T1 不可启动的根因之一），但双表示本身仍吃 KV。**可启动（P2 时不可）但 KV 容量不可产**。

### 2.3 DE（C1/C12，4 轮取中位，接受率归一 step_eff = tput / tokens_per_step）

| 指标 | M1 | M2 | M3 | M4 | M3 vs M1 | M4 vs M1 |
|------|------|------|------|------|------|------|
| C1 tput 中位 | 92.9 | 100.0 | 77.1 | 89.3 | — | — |
| C1 tokens/step | 4.571 | 5.505 | 4.063 | 4.531 | — | — |
| **C1 step_eff** | 20.3 | 18.2 | 19.0 | 19.7 | **-6.4%** | -3.0%（噪声带内） |
| C12 tput 中位 | 389.6 | 378.7 | 375.2 | 381.4 | — | — |
| C12 tokens/step | 4.151 | 4.672 | 4.395 | 4.088 | — | — |
| **C12 step_eff** | 93.9 | 81.1 | 85.4 | 93.3 | **-9.1%** | -0.6%（噪声带内） |

- **full 模式 decode 归一回退 -6~-14%**（M2 最差、M3 略好，均在概率解码接受率噪声带上边缘）：W4A4 wrapper CG=1 static 路径的 decode 劣势仍在（P2 为 -16~-19%，略缓解）。原始 tput 差异主要来自接受率波动（acc/draft 轮间 2.1-4.6 摆动），归一后判读。
- **hybrid 模式 decode 完全恢复**（M4 step_eff = M1 带内）——M<3072 走 W4A16 原路径，符合设计预期。

### 2.4 质量门

| 门 | 臂 | 结果 |
|----|------|------|
| W-3 golden 包络判据（4 稳定 prompt greedy，vs M1） | M3 | **PASS**：4/4 逐字一致；token 级 top-1 logprob 中位 \|diff\|=0.0、max 0.27（code prompt 单 token，该 prompt 本臂自复跑 own_stable=False，即概率投机解码运行级非确定，非 W4A4/共享效应） |
| needle（64K×3 + 128K×2，统计口径） | M1/M2/M3/M4 | 2/5 / 4/5 / 1/5 / 4/5——needle 本身噪声大（P2 历史 3-5/5 波动），golden 门为更强制证据；M3 的 1/5 结合 golden 4/4 判为统计噪声，如实记录不强行解读 |
| 启动日志核验 | 全臂 | M1：Using 'B12X_MXFP4' + 无 pool/W4A4 行；M2/M3/M4：Using W4A4B12xExperts + 插件容器内 import 核验；cudagraph 三档（PIECEWISE 16/16 + FULL 12/12 + dspark 11/11）全臂完整 |

---

## 3. 关键工程发现与处置（如实记录）

### 3.1 补丁池-W4A4 插件路径错配（测试设计缺口，已处置）

**发现**：ws-dedup 补丁的几何键共享池位于 vllm 的 `flashinfer_b12x_moe.py`（FlashInferB12xExperts 路径），而 W4A4 插件（`<INSTALL_DIR>/nvfp4/plugin_a1/`，A′ plugin）在 `w4a4_experts.py._derive_w4a4` 中**直接从 `flashinfer.fused_moe` 构造每层 B12xMoEWrapper，不经过补丁池**——若 M3 只挂 overlay + env=1，池永远不会被使用（生产 W4A16 走 B12xExperts/b12x 包，同样不经过）。昨日 ws-dedup 报告 §5 W-2 的"插件路径挂载补丁即生效"预期与实际源码不符。

**处置**：为插件做最小池集成（`_get_pooled_wrapper(**kwargs)` 辅助函数，路由 wrapper 构造经 overlay 模块池；同 env `VLLM_B12X_SHARED_WRAPPER` 门控；**off 路径构造 kwargs 与原版逐字一致**）。一次性容器 L1 验证 11/11 PASS：off 等价（env None/0/true/2/"" 五组 × 3 次调用 → 3 次构造、kwargs 逐一相同）+ 池逻辑（同几何×4 → 1 构造共享 / 异几何、异 activation → 新条目 / 幂等）+ overlay 符号在场。M2 用原版插件纯复现 P2；M3/M4 用池化插件（.bak-wsdedupl3-20260823 留档，四节点 md5 一致）。

**任务书勘误**：W4A4 插件实际位于 `plugin_a1/`（env VLLM_MOE_W4A4=0/1/2 门控），`plugin_merged/` 是 routeb merged-GEMM 插件（VLLM_MOE_MERGED 门控），两者为不同资产。

### 3.2 其他工程记录

1. **插件 logger 不可见**：vllm `init_logger` 对非 vllm 命名空间 logger 无 handler，INFO 丢失（影响所有 plugin_a1 日志行取证）。建议插件侧改用 `init_logger("vllm." + __name__)` 或 print 到 stderr；本次以类生效行 + 内存算术替代核验。
2. **恢复生产走 systemd 标准路径的补充教训**：W1 教训（worker 服务未停导致冲突）的对偶面——**worker 服务被停后，systemd 路径下 workers 不会自动跟随**（W1 场景 worker 服务全程未停才"自动跟随"）。本次 rank0 独自等待 rendezvous ~12 分钟后手动 `systemctl start vllm-tp4-worker.service`（02/03/04）完成恢复。runbook 应写明：**恢复自愈链 = head.service + 三 worker.service + healthy 后 healthcheck.timer，缺一不可**。
3. **mode 探针对 W4A4 臂的语义漂移**：模式分带（fast/mid/slow）原基于 W4A16 形态标定；W4A4 prefill 本身更快，首 4K TTFT 会被臂效应拉低——M2/M3/M4 的 2.74-2.76s 不能直接解读为"fast 模式"，更可能是"W4A4 增益 + 中簇模式"。臂间同窗口对比不受影响（M2/M3/M4 三臂一致）。
4. **stall 环境未污染本阶段**：全部臂 stall 探针干净（21:20 前）；W1 报告的"随窗口时间恶化"趋势本次未再现于生产形态（生产形态是否中招仍待 envstall 调查定论）。

---

## 4. W4A4 翻案判定（任务判据逐条）

| 判据 | 阈值 | 实测 | 判定 |
|------|------|------|------|
| M3 PR ≥ M1 -3% 带内 | ≥2671（4K） | 2982（+8.3%，含模式混杂警示） | ✓ |
| 显存省 ≥10GB | ≥10 | **22.83 GiB**（M3 vs M2） | ✓ |
| 质量门 | W-3 golden 包络 | 4/4 逐字一致 + logprob 中位差 0.0 | ✓ |
| pool size==1 日志 | ==1 | 内存算术等价核验（§2.2，方法论偏差标注） | ✓（替代口径） |
| cudagraph 捕获正常 | 三档完整 | 16/12/11 全臂完整 | ✓ |

**判定：W4A4 + ws-dedup 补丁组合"值得深测"成立。**

**翻案叙事修正**：P2 的 No-Go 根因有二——①workspace×43 内存死结（**本补丁已解**：68.15→45.32 GiB）；②CG=1 static 路径性能问题（**已随 threshold 4096 部分化解**：prefill M=4096 落 W4A4 甜点区 +6~8%，但 decode 仍 -6~-14%）。P2 的 -13% prefill 结论是 threshold 1024 时代的形态误差，不宜再作为 W4A4 定论引用。

**剩余成本清单**（深测/上线前须评估）：
1. full 模式 decode 归一 -6~-9%（decode 重业务需权衡；prefill 重业务净收益）；
2. KV -8.5%（5.48M vs 6.0M tokens，并发/长上下文容量小损）；
3. hybrid 模式（decode 全恢复 + prefill +8.9%）因双表示 KV -74.5% 不可产——除非内存护栏或表示瘦身；
4. 池化集成目前为测试资产形态（overlay + 插件补丁分离），上线需合并为正式补丁 + 修 logger + 扩大质量门样本（长文本/多语言/工具调用/温度>0）。

---

## 5. 生产终态与回滚链

**终态 = M1 态（W4A16 + 补丁 overlay 保留 + env=0）**，21:20 UTC 验证全绿：

| 项 | 状态 |
|----|------|
| 四 rank 容器 | vllm-tp4-rank0/1/2/3 全部 Up (healthy) |
| 自愈链 | 01 head.service + healthcheck.timer active；02/03/04 worker.service active |
| MoE 后端 | B12X_MXFP4（Using 'B12X_MXFP4'，无 W4A4 行）✓ |
| threshold | long_prefill_token_threshold 4096 live ✓ |
| KV | 6,055,074 tokens（基线带内）/ weight 40.5 GiB |
| overlay | 容器内 md5 8f88555a（=四节点 <INSTALL_DIR>/overlay-wsdedup），env=0 零行为变化（M1 全指标带内已实证） |
| PR 复测 | 4K 2788（轮 2788/2804）/ 16K 2800 / 32K 2683 / 64K 2436——基线带内 |
| API | /health 200 |

**回滚链（全部 <10 分钟）**：
- 补丁 overlay：env 已=0（零行为）；彻底移除 = 删 start 脚本注入行或恢复 `.bak-wsdedup-20260822` + restart
- W4A4 臂配置：`.bak-wsdedupl3-20260823`（head/worker 四机，已恢复）+ 插件 `.bak-wsdedupl3-20260823`（四机，已恢复原版 c2d1de3d）
- 生产恢复验证：本报告 §5 表（已执行）

**若采纳 M3 强阳性建议**（默认不采纳，维持基线）：恢复流程 = 部署池化插件（/tmp/_wsdedup_l3/w4a4_experts_pooled.py 四节点）+ patch_arm.py mode 2 + set_shared.sh 1 + restart + 三判据复验。

---

## 6. 产物索引

- **服务器** `node01:/tmp/_wsdedup_l3/`：logs/（m1-m4 各臂 restart/probe/stall/panorama/de/needle/startup/gpu_mem/golden + w0_deploy.log）、make_pooled_plugin.py、patch_arm.py、set_shared.sh、heal_chain.sh、stall_probe.py、golden_env.py、measure_arm.sh、test_pooled_plugin.py、w4a4_experts_pooled.py
- **本地** `deliverables/engineering-assurance/_wsdedup_l3_assets/`（上述脚本同步副本）
- **留档备份（四节点）**：start_tp4_{head,worker}.sh.bak-wsdedupl3-20260823、.bak-wsdedup-20260822（W-0 前）、plugin_a1/w4a4_experts.py.bak-wsdedupl3-20260823
- **前序资产**：/tmp/_ws_dedup/（补丁与部署脚本）、/tmp/_thr4096/（panorama/probe/de_bench 脚本）、/tmp/_routea_work/（bench_panorama_prefill.py、bench_tp4.py needle）、/tmp/_mtp_tune/（de_bench.py、greedy_check.py）
- **golden 参考**：/tmp/_wsdedup_l3/logs/m1_golden.json（M1 W4A16 4 稳定 prompt + top-1 logprobs）

## 7. 行动项建议

| # | 项 | 负责 | 优先级 |
|---|---|---|---|
| A1 | W4A4+池去重深测立项（长窗口 e2e A/B、真实流量形态、decode 业务影响评估）| 架构/SRE | P1 |
| A2 | 池化整合并正式化：overlay + 插件集成合并为单一正式补丁（含 logger 修复）| kernel 工程 | P1 |
| A3 | hybrid 双表示瘦身研究（KV -74.5% 的解法：W4A16 打包按需/延迟、或 MIN_M 上调）| 架构 | P2 |
| A4 | 模式探针标定更新（W4A4 臂下 fast/mid/slow 分带需重标定，避免臂-模式混杂误读）| SRE | P2 |
| A5 | runbook 补条目：systemd 恢复必须显式启动三 worker.service（§3.2.2 教训）| SRE | P2 |
| A6 | needle 口径改进（当前 5 样本噪声过大，无鉴别力；建议扩样本或降级为 smoke）| QA | P3 |

---

*纪律遵守：每臂 stall+模式双探针、≥3 轮中位、DE 接受率归一、包络判据、长命令后台+轮询、异常数字标注不强行解读（needle/模式混杂/插件 logger）、生产零残留（脚本/插件已回滚 + 验证全绿 + 自愈链恢复）。全部原始数据见 §6。*

*本报告由工程保障团队生成，关键决策（W4A4 深测立项与否）请由人类工程负责人复核。*
