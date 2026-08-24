# DeepGEMM 解耦移植 + MoE 三路径对比 + 社区最佳案例 综合报告

**日期**：2026-08-05
**工作流**：工作流 2（架构调研）+ 工作流 1（代码审查）+ 工作流 4（部署评估）组合
**参与成员**：Archi（架构师）/ Cody（代码审查师）/ Docu（技术文档师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **DeepGEMM Grouped MoE 解耦移植工作量小**：内核侧≈0（生产 deep_gemm 即 post-#324 nv_dev，wrapper 全库生效）、vLLM 侧已预埋映射——**值得低成本 A/B 冒烟**（`--moe-backend deep_gemm` vs b12x），预期 decode +5~20% 可匹配/微超 b12x 基线，不捆绑 Mega MoE。
- **MoE 三路径对比**：**b12x 已是 GB10 最优**（我们基线 48-54 tok/s ≥ 社区 Marlin 50）；CuteDSL 在 SM12x 是死路、CUTLASS 需重编译且有 SM120 垃圾输出 bug、Triton 可作低优先 fallback。
- **社区最佳案例**：DSpark k=5 投机是最大单一因素（+60-85%）；B12X+NVFP4 是「整个速度差异」（我们已具备）；社区甜点 = 1M ctx + seqs6 + regular cudagraph + 热机。
- **阻塞 / 非阻塞**：非阻塞。F 生产维持（b12x + greedy + 动态K）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 调研完成，A/B 冒烟方向明确 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条 |
| 建议下一步 | DeepGEMM Grouped MoE A/B 冒烟（低成本高信息量） |

---

## 🔬 一、DeepGEMM 解耦移植分析（Archi）

### Blackwell 架构差异（wiki 确认）
- SM100 独有：tcgen05 + TMEM(256KB) + cta_group::2 + cluster>1 + 228KB SMEM
- SM12x 仅：mma.sync（Ampere 系）+ 99KB SMEM + 单 CTA TMA，无 TMEM
- SM100 内核非可重编译（tcgen05→mma.sync 属重写）——**但 #324 已代劳**（nv_dev 分支内置完整 SM120 mma.sync 原生内核）

### 移植工作分解
| 内核 | F 生产状态 | 工作量 | 需改什么 |
|------|-----------|--------|---------|
| fp8_fp4_gemm dense | ✅ 已跑通（sm_121a cubin） | 0 | 无 |
| fp8/fp4 MQA + FP4 indexer | ✅ 已跑通 | 0 | 无 |
| **m_grouped fp8 GEMM（MoE）** | ⚠️ MoE 未走 DeepGEMM（现 flashinfer_b12x）；内核在 nv_dev wheel 懒编译即入 cache | **小** | vLLM 侧 `--moe-backend deep_gemm` 映射已存在 + 权重 repack 已有；仅需冒烟 sm120 门控 + 精度回归 + A/B |
| Einsum / HC Prenorm | 未用 | 小 | 按需接入 |
| Mega MoE | ❌ SM121 不可行 | 不可移植 | TMEM/cta_group::2 仅 SM100 |

### 解耦可行性结论
- **内核侧成本≈0**：生产 deep_gemm 即 post-#324 nv_dev，wrapper 对整库生效；Grouped MoE 按需进 cache，非新移植
- **vLLM 侧已预埋**：映射 + 格式转换已在 base（注意 `should_auto_disable_deep_gemm(model_type)` 可能自动禁用，需显式放行）
- **收益预期**：GB10 273GB/s 带宽瓶颈下，社区 TFLOPS（masked FP8 670 / FP4 1363 @ RTX PRO6000）在 GB10 不可达；真实收益 = 单核 masked 启动节省 + M=1..16 AB-swap 小 GEMM，**预期 decode +5~20%、可匹配/微超 b12x 基线（c5~90/c1~40）但不量级跃迁**
- **建议**：值得做低成本 A/B 冒烟（`--moe-backend deep_gemm` vs b12x），不值得正式移植；A/B 达标则切换，否则维持 flashinfer_b12x；不捆绑 Mega MoE
- **待确认**：① 当前 129 内核 cache 是否已含 grouped ② DeepGemmFP4Experts.is_supported_config 对 SM120 家族门控值

---

## ⚖️ 二、MoE 三路径对比（Cody）

### 社区性能（SM12x）
| 路径 | SM120/SM121 实测 | 结论 |
|------|-----------------|------|
| CUTLASS grouped GEMM | SM120 垃圾输出 bug；SM121 需 CUTLASS4.4+sm121f 重编译 | 成本高，SM120 有 bug |
| FlashInfer CuteDSL | SM12x 被 SM100 能力检查拦截**无基准**；SM12x 实际形态 = **b12x 树** | 死路（mainline） |
| Triton fused_moe | GB10 FP8：单流 48.9 / 并发 208 t/s；需自建 GB10 调优配置 | 零编译可 A/B，预期 5-15% 落后 |
| **b12x（我们）** | **48-54 tok/s（本地 vllm_bench）≥ 社区 Marlin 50** | **已是 GB10 最优** |

### 可移植性（我们 0.2.1-v026.0）
- Triton：默认后端零编译（`--moe-backend triton`），但需自建 NVIDIA_GB10.json 调优
- CUTLASS：需重编译（CUTLASS≥4.4 + sm121f + FP4_ARCHS/K=64 补丁），不支持 EP
- CuteDSL：SM12x 不可用（死路）；其 SM12x 变体即 b12x（我们已集成）

### 结论
- **b12x 已是 GB10 最优论据成立**（CuTe DSL + 组 GEMM + 原生 FP4 的 SM12x 专用精化）
- 值得测：① Triton 快速 A/B fallback（低优先）② 若投编译成本可试 CUTLASS+补丁（预期不超 b12x）
- 死路：mainline CuteDSL（仅 SM90/SM100）、SM120 原生 CUTLASS grouped

---

## 🏆 三、社区最佳案例（Docu，8 案例）

### 代表案例
| 方案 | 配置 | 性能 |
|------|------|------|
| MiaAI-Lab 0731 | 1M/seqs6/util0.8-0.835/DSpark k5 | x1 82.4 / x6 191.2 / C6 regular cudagraph 340.5 |
| tonyd2wild 1M NVFP4 | k5 probabilistic/util0.78/Patch4 | 单流峰值 84.3 / C6 197.3 / C16@200K 315.1 |
| llmrequirements | fp8 KV/util0.87/seqs16/k5 | 单流 61 / 16 并发 261（比自家 MTP +37%） |
| flowtivity Hermes | FP8 KV/util0.82/1M | 单流 41（真实 agent 负载） |
| r0b0tlab | MTP/fp8/65K/16 | c1 38.4 / c16 144.6 |
| NVIDIA 论坛 | FP8+MTP2/200K/2 | c1 44 / c8 96 |

### 共同特点
1. **DSpark k=5 投机是最大单一因素**：官方 +60-85% vs MTP-1；社区实测 +37% vs 调优 MTP；Patch4 修复后 32.7→55.4 tok/s
2. **B12X MoE + NVFP4 4bit KV**：VLLM_USE_B12X_MOE=1 是「整个速度差异」（50-60 vs ~29 t/s）——**我们已具备**
3. **1M ctx + seqs6 为社区甜点**（KV 共享池）
4. **regular cudagraph 而非 breakable**：MiaAI 实测 C1 +28.6%
5. **真实负载预热**：冷启动惩罚约 -30%
6. **性能强依赖内容类型**：结构化/代码接受率 60-78% vs 散文/推理 33-40%

### 最应借鉴 3 项
| # | 借鉴项 | 潜在收益 |
|---|--------|---------|
| 1 | **DSpark k=5 投机（probabilistic 方向）** | c1 40→60+（需评估 overlay 风险与 GSM8K 保持 99%） |
| 2 | **KV 池放大至 1M ctx**（共享池模式） | 批量头寸提升 |
| 3 | **基准口径对齐**（热机 + decode-only + 结构化负载） | 与社区数字可比（我们的 c5 87-92 有望对标 C6 150-190） |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **DeepGEMM Grouped MoE A/B 冒烟**（`--moe-backend deep_gemm` vs b12x 双机对比，低成本高信息量） | SRE+Testing | P0 | 本周 |
| 2 | 确认 129 内核 cache 是否含 grouped + DeepGemmFP4Experts 门控值（A/B 前置） | Archi+SRE | P0 | A/B 前 |
| 3 | 基准口径对齐（热机 + decode-only + 结构化负载）——与社区数字可比 | Testing | P1 | 本周 |
| 4 | DSpark k5 probabilistic 方向评估（结合 workload：greedy 保质量 vs probabilistic 提吞吐） | Archi+Testing | P1 | 2 周内 |

---

## ⚠️ 待完善 / 已知局限

- DeepGEMM Grouped MoE 无公开 GB10 端到端数据（截至 2026-08）——收益为推断
- 社区多数聚合 t/s 为 decode-only（排除 prefill），口径需对齐
- al-engr/hazyumps 未能抓取原文（数字与 MiaAI/tonyd2wild 同源，未独立核实）

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：blackwell wiki + DeepGEMM #324 移植分解 + 解耦可行性（待确认 2 项）
- **Cody（代码审查师）**：三路径社区数据 + 可移植性 + b12x 最优论据（5 来源链接）
- **Docu（技术文档师）**：8 案例表 + 共同特点 + 借鉴排序（MiaAI/tonyd2wild/llmrequirements/flowtivity/r0b0tlab/NVIDIA 论坛）
- **前置报告**：`dynk-megamoe-version-ab-2026-08-05.md`、`dynk-megamoe-research-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（DeepGEMM A/B 冒烟、b12x 维持、借鉴项优先级）请由人类工程负责人复核。
