# TP4 深化报告：绑核实测 · 每 token 耗时分解 · v3 双口落地复测

**日期**：2026-08-12
**工作流**：工作流 4（性能深化）+ 优化实施
**参与成员**：Rex（绑定核实/v3 实施/耗时采集）、Archi（耗时分析）、Tessa（复测）、Zhen（汇编）
**状态**：✅ v3 双口已落地，prefill 单流达标；遗留 P2×4

---

## 📌 TL;DR（执行摘要）

- **绑核实测**：PyTorch 侧 NCCL 辅助线程（heartbt/watchdg/tcpstore）全部落在隔离核 1-4 ✅；vllm 主进程/EngineCore 5-9 ✅；⚠️ **发现 shim 竞态**——NCCL 自有数据面线程（Progress/IbAsync 等）实际落 5-19（thread_entry 无条件 pin 覆盖父线程命名），带宽影响可忽略（23.9GB/s 达上限）但延迟敏感路径有隐患，建议 shim v4。
- **每 token 耗时分解（实测）**：prefill 占 96.6%、decode 3.3%；**通信占比 <0.5%（decode）/~6.7%（prefill）**——decode 纯计算 bound（层间串行延迟 43×~230µs 主导），带宽优化价值在 prefill。
- **v3 双口补丁落地**：1GB busbw **13.87→23.86 GB/s（+72%）**，四机 MD5=b7784b49… 一致，停机仅 ~8min。
- **v3 复测（27 组合）**：prefill **单流档均值 1.296 ≥1.25 验收 ✅**；全档均值 1.206（c5 并发档 1.13 摊薄）；**两个严重异常点彻底修复**（16384/coding/c1 841→2543、65536/coding/c1 1662→2501）；decode/TTFT 无回归（TTFT 全面改善 ×0.87）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 v3 双口落地，prefill 单流达标 |
| 线程绑定 | 辅助线程 1-4 ✅；NCCL 数据面线程 ⚠️（shim 竞态，P2） |
| 耗时组成 | decode 计算 bound（通信 <0.5%）；prefill 通信 ~6.7% |
| v3 带宽 | 13.87→23.86 GB/s（+72%） |
| prefill 验收 | c1 单流 1.296 ✅ / 全档 1.206 ⚠️ / c5 并发 1.134 |
| 遗留 | P2×4（shim v4、并发档 prefill、131072 补测、v3 归档） |

---

## 🔬 A. NCCL 线程绑定核实（Rex 实测，四机一致）

| 线程/进程类 | 期望 | 实际 | 判定 |
|---|---|---|---|
| pt_nccl_heartbt/watchdg（8/机） | 1-4 | 1-4（PSR 稳定） | ✅ |
| pt_tcpstore_uv（head） | 1-4 | 1-4 | ✅ |
| vllm 主进程 / EngineCore / Worker_TP | 5-9 | 5-9 | ✅ |
| 其余（ZMQbg/gloo/cuda） | 5-19 | 5-19 | ✅ 设计 |
| **NCCL 自有数据面线程**（Progress/IbAsync/Service） | 1-4 | **5-19** | ⚠️ 见下 |

**shim 竞态发现**：NCCL 线程默认不命名 → 加了 `NCCL_SET_THREAD_NAME=1` 后已命名且 shim 日志显示 "=> CPU 0-4"，但实测 affinity 5-19、PSR 5/6/10/16/18——`thread_entry` 在子线程启动时无条件 pin 5-19，若父线程先调 setname(0-4) 被子线程默认 pin 覆盖。**建议 shim v4**：thread_entry 不再无条件默认 pin（或检测已 pin 则跳过）。带宽影响已证可忽略（23.86GB/s 达上限）。

## 📊 B. 每 token 耗时分解（Rex 实测 + Archi 理论）

实测（32768 ctx 单请求）：TTFT 12.47s、prefill 2384 tok/s（占 96.6%）、decode 每 token 21.4ms（占 3.3%）、投机接受率 69%（draft 178535/accepted 123115）。

| 组成 | 估算 | 占比 |
|---|---|---|
| 层间串行延迟（43 层×~230µs） | ~9ms | ~95%（decode） |
| 计算下限（激活~23GB） | 0.1-0.15ms | ~1.3% |
| **NCCL 通信** | decode ~0.16ms（<0.5%）；prefill ~0.84s（~6.7%） | 见左 |
| 调度/其他 | 0.2-0.5ms | 2-5% |

**结论**：decode 是"每层固定延迟×43 层"的延迟受限（非带宽受限）；**v3 双口的价值在 prefill**（大消息段通信占 5-10%）。

## 🔧 C. v3 双口补丁落地（Rex）

| 项 | v2 单口 | v3 双口 | 提升 |
|---|---|---|---|
| 1GB busbw | 13.87 | **23.86 GB/s** | +72% |
| 64MB | 13.53 | 22.64 | +67% |
| 16MB | 12.84 | 21.95 | +71% |

- 实现：`ncclIbPeerHcaOverride` 支持 "peer=devA,devB" 按 channelId%2 轮换（send/recvSetup 传 channelId）；双 dev 轮换实测确认（rank0 peer3 偶 chan→dev0/奇 chan→dev2）
- NCH=4 与 NCH=2 持平 → 已达双口物理上限（~12GB/s/口）
- 落地：四机脚本双 dev PEER_HCA 表 + `NCCL_SET_THREAD_NAME=1`；MD5=b7784b49885659c27765e648884e4edd 四机一致；回滚路径 .bak-v2 + .bak-tp4-v3
- 停机 ~8min；恢复 READY 220s、8001=200、四机 128 条 RING-ONLY v3、0 错误

## 📈 D. v3 复测（Tessa，27 组合 405 样本）

### prefill p50×conc（V3 vs TP2A 关键档）
| ctx | c1（单流） | c5（并发） |
|---|---|---|
| 512 | ×1.10-1.48 | ×1.11-1.52 |
| 16384 | ×1.27-1.30 | ×1.08 |
| 32768 | ×1.28-1.35 | ×1.10-1.13 |
| 65536 | ×1.28-1.29 | ×1.07-1.08 |
| 4096(c5) | — | ×1.03-1.06 |

**达标判定**：c1 单流（n=12）均值 **1.296 ≥1.25 ✅**；c5 并发（n=15）均值 1.134（带宽/算力共享摊薄）；全档均值 1.206（首轮 1.075 → 显著提升）。

### 异常点复测
| 点 | 首轮 | v3 | 判定 |
|---|---|---|---|
| 16384/coding/c1 | 841 | **2543**（×3.02） | ✅ 彻底修复（三样本 2510-2571 稳定） |
| 65536/coding/c1 | 1662 | **2501**（×1.50） | ✅ 修复 |
| 512/coding/c1 | 1120（预热） | 1239 | ⚠️ 预热消失，wave2 偶发单波抖动（p50 正常） |

### 回归
- decode V3/T4 mean=1.062（无退化，小幅提升）
- TTFT V3/T4 mean=0.869（全面改善，16384/c1 17.4s→5.7s）
- preemption 前=0 后=0

## 🎯 优化空间排序（Archi，按收益）

1. **MTP 投机解码**（已启用，接受率 69%；进一步调优预期 decode 1.3-1.8×）
2. **CUDA Graph / kernel 融合 / prefill-decode 流重叠**（-20-40% 层间延迟，直击 95% 大头）
3. Prefix KV 复用（长上下文多轮 prefill 省 60%+）
4. 调度参数（max-num-seqs/chunked-prefill，+10-30%）
5. v3 双口（已完成，prefill +3-8% 已兑现）
6. LL 协议：暂缓（decode 通信 <0.5%）
7. 绑核/IRQ：已做；shim v4 修复数据面线程落隔离核

## ✅ 行动清单（遗留 P2）

| # | 行动 | 负责 | 紧急度 | 预期 |
|---|------|------|--------|------|
| 1 | shim v4（修复 thread_entry 竞态，NCCL 数据面线程落 1-4） | Rex+Archi | P2 | 下维护窗口 |
| 2 | c5 并发档 prefill 提升（当前 1.03-1.13，冲全档 1.25：考虑 chunked-prefill/调度） | Tessa+Archi | P2 | 评估 |
| 3 | 131072 极限档补测 | Tessa | P2 | 下轮 |
| 4 | v3 源码/diff/产物归档（补 <INSTALL_DIR>/backup/tp4-20260812 + runbook） | Docu | P2 | 下轮 |

## ⚠️ 待完善 / 已知局限

- NCCL 数据面线程未落隔离核（shim 竞态）——带宽无影响、延迟敏感路径有隐患，v4 修复
- 全档 prefill 1.206 未达 1.25（c5 并发档差距）；单流已达标
- 1MB 小消息延迟 1539µs 与 v2 持平（bench 含 sync 开销，非回归）

---

## 📚 数据来源 & 成员产出索引

- Rex：绑定矩阵+shim 竞态（teammate-message 2026-08-12）、v3 实施（busbw 表/MD5/PEER_HCA 表）、耗时采集（32768 单请求指标 diff）
- Archi：耗时组成方法+理论估算+优化空间排序（teammate-message 2026-08-12）
- Tessa：v3 复测（27 组合 405 样本，_tessa_tp4_bench/summary_TP4V3*.json + rows_TP4V3*.csv）

> 本报告由工程保障团队 AI 协作生成（2026-08-12），关键决策请由人类工程负责人复核签字。
