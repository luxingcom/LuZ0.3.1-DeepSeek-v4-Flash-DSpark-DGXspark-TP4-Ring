# 动态 K 固化 + Mega MoE SM121 调研综合报告

**日期**：2026-08-05
**工作流**：工作流 4（部署验证）+ 工作流 2（架构调研）组合
**参与成员**：Rex（SRE 工程师）/ Archi（架构师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **动态 K 已固化进 F 生产**：生产真实流量复验通过（c5 +3.36%、接受率 +33%、c8 +37%、0 错误），部署基线已更新，回滚路径保留。
- **Mega MoE SM121 不可行**：微架构指令缺失（TMEM/UTC\*MMA 仅 SM100），非编译参数问题——官方无 sm120 mega_moe 实现，社区移植未成熟。
- **阻塞 / 非阻塞**：非阻塞。F 生产运行新基线（动态 K）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 动态 K 固化完成；Mega MoE 结论明确 |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条 |
| 建议下一步 | 启动 v0.26 vs 0.25 A/B 验证（任务③） |

---

## ⚙️ 一、动态 K 固化（Rex，生产复验通过）

### 生产流量采样（40% 短 out64-256 + 60% 中长 out512-2048；c3/c5/c8 = 20/50/30% 随机窗口；基线 5min vs 动态 K 20min）

| 指标 | 固定 K5 | 动态 K | Δ |
|------|--------|--------|-----|
| c5 agg t/s | 75.99 | 78.54 | **+3.36%** |
| 接受率 | 0.370 | 0.491 | **+32.65%** |
| c3 agg | 49.9 | 55.3 | +10.99% |
| c8 agg | 65.3 | 89.6 | **+37.28%** |
| TPOT | 46.2ms | 43.7ms | 更好 |
| 错误率 | 0/87 | 0/385 | 持平 |

> 注：c5 生产混合流量 75.99 vs 历史均匀 91.9 差异因含 40% 短输出（A/B 同流量可比，不影响结论）

### 固化执行
- ✅ 双机启动脚本 speculative-config → 动态 K（`num_speculative_tokens_per_batch_size:[[1,1,5],[2,4,4],[5,6,3]]`，保留 greedy），bash -n 通过，当前容器即新基线
- ✅ 旧固定 K5 备份：`start_{head,worker}_v026r.fixedK5.bak.20260805_*`
- ✅ 部署基线记录：`deploy-f-dynamic-k-baseline-2026-08-05.md`
- ✅ 回滚路径：备份脚本可恢复固定 K5

---

## 🔬 二、Mega MoE SM121 可行性（Archi，证据链完整）

### 结论：**不可行**（微架构指令缺失，非编译参数问题）

| 证据 | 内容 |
|------|------|
| DeepGEMM #305 | 官方不计划自维护 SM120，欢迎社区 PR |
| DeepGEMM #318（jasl） | SM12x 参考实现，**MegaMoE 明确 gated**（需独立 SM12x 实现） |
| DeepGEMM #324（已合入 nv_dev） | sm120 内核（Dense/Grouped MoE/Einsum/HC Prenorm/MQA）**无 mega_moe** |
| NVIDIA 官方 | SM120/121 用 OMMA/QMMA；**TMEM/UTC\*MMA 仅 SM100** |
| blackwell-wiki | SM120 移植 = 实质内核重写（tcgen05→mma.sync、TMEM→reg/SMEM、tile 缩至 99KiB） |
| hazyumps 实测 | GB10 跑 V4-Flash 只 fallback 常规路径，不走 Mega MoE |

### 关键澄清
- 我们已跑通的 `fp8_fp4_gemm sm_121a cubin` 是 #324/jasl 的 **sm120 常规 GEMM 路径**（不含 Mega MoE）——与结论不冲突
- nvcc_wrapper（sm_120f→sm_121a）仅改写 JIT arch 字符串，**无法补出缺失硬件指令集**

### 替代建议
- 用 #324 sm120 分组 MoE GEMM 分步实现（非融合，可量化融合收益差距）
- 或 CUTLASS grouped GEMM / FlashInfer CuteDSL MoE（显式支持 SM12x）/ Marlin FP8 / Triton fused_moe
- 跟踪社区 sm120 mega_moe 移植（截至 2026-08 无公开实现，官方时间线不乐观）

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **启动 v0.26.1 vs 0.25.1 A/B 验证**（DSpark/CUDA-graph 默认行为差异——用户指定排在此后） | SRE+Testing | P0 | 本周 |
| 2 | 动态 K 新基线观察期（生产运行 1-2 天，确认稳定 + 记录真实负载收益） | SRE | P1 | 持续 |
| 3 | 跟踪 DeepGEMM #318/#324 社区 sm120 进展（Mega MoE 移植可能性） | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- 动态 K 采样为 20min 单轮（非多日），观察期确认
- Mega MoE 替代路径（分组 MoE 分步）未实测，属建议方向

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE 工程师）**：生产采样数据（`~/fparam-exp/prod_sample.py` + prod_base_prod.json + prod_v3_prod.json）+ 固化执行 + 部署基线文档
- **Archi（架构师）**：DeepGEMM #305/#318/#324 + NVIDIA 官方 + blackwell-wiki 证据链
- **前置报告**：`f-param-verify-2026-08-05.md`、`weight-megamoe-forum-research-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（动态 K 固化、Mega MoE 不引入、A/B 验证启动）请由人类工程负责人复核。
