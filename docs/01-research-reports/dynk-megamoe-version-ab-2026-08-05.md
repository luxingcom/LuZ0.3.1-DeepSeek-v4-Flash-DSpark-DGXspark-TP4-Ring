# 动态 K 固化 + Mega MoE 调研 + v0.26 vs 0.25 A/B 验证 综合报告

**日期**：2026-08-05
**工作流**：工作流 4（部署验证）+ 工作流 2（架构调研）+ 工作流 1（测试对比）组合
**参与成员**：Rex（SRE 工程师）/ Archi（架构师）/ Tessa（测试专家）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **动态 K 已固化进 F 生产**：生产真实流量复验通过（c5 +3.36%、接受率 +33%、c8 +37%、0 错误），部署基线已更新，回滚路径保留。
- **Mega MoE SM121 不可行**：微架构指令缺失（TMEM/UTC\*MMA 仅 SM100），非编译参数问题——官方无 sm120 mega_moe 实现。
- **v0.26 vs 0.25 A/B 验证完成**：**0.26 相对 0.25 在 DSpark/CUDA-graph 默认行为无实质变化**（唯一新增为 block-size 校验）——论坛性能差距不能由版本默认行为解释，更可能源于配置差异。
- **阻塞 / 非阻塞**：非阻塞。F 生产运行新基线（动态 K）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 三任务全部完成，F 生产基线已优化 |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条 |
| 建议下一步 | 动态 K 新基线观察期；彻底隔离版本 vs 配置（同镜像双 spec 对比） |

---

## ⚙️ 一、动态 K 固化（Rex，生产复验通过）

### 生产流量采样（40% 短 out64-256 + 60% 中长 out512-2048；c3/c5/c8 = 20/50/30%；基线 5min vs 动态 K 20min）

| 指标 | 固定 K5 | 动态 K | Δ |
|------|--------|--------|-----|
| c5 agg t/s | 75.99 | 78.54 | **+3.36%** |
| 接受率 | 0.370 | 0.491 | **+32.65%** |
| c3 agg | 49.9 | 55.3 | +10.99% |
| c8 agg | 65.3 | 89.6 | **+37.28%** |
| TPOT | 46.2ms | 43.7ms | 更好 |
| 错误率 | 0/87 | 0/385 | 持平 |

### 固化执行
- ✅ 双机启动脚本 → 动态 K（`num_speculative_tokens_per_batch_size:[[1,1,5],[2,4,4],[5,6,3]]`，保留 greedy），bash -n 通过，当前容器即新基线
- ✅ 旧固定 K5 备份（`.bak.20260805_*`）；部署基线 `deploy-f-dynamic-k-baseline-2026-08-05.md`；回滚路径保留

---

## 🔬 二、Mega MoE SM121 可行性（Archi）：**不可行**

| 证据 | 内容 |
|------|------|
| DeepGEMM #305 | 官方不计划自维护 SM120 |
| #318（jasl） | SM12x 参考实现，**MegaMoE 明确 gated** |
| #324（已合入 nv_dev） | sm120 内核列表**无 mega_moe** |
| NVIDIA 官方 | **TMEM/UTC\*MMA 仅 SM100**；SM120/121 用 OMMA/QMMA |
| blackwell-wiki | SM120 移植 = 实质内核重写（tcgen05→mma.sync、TMEM→reg/SMEM、tile 缩至 99KiB） |

- **关键澄清**：我们已跑通的 `fp8_fp4_gemm sm_121a cubin` 是 #324/jasl 的 sm120 常规 GEMM 路径（不含 Mega MoE）；nvcc_wrapper 无法补出缺失硬件指令集
- **替代建议**：#324 sm120 分组 MoE 分步实现 / CUTLASS grouped GEMM / FlashInfer CuteDSL MoE / Triton fused_moe

---

## ⚖️ 三、v0.26 vs 0.25 A/B 验证（Tessa）

### 静态对比（envs.py / 源码）
| 项 | 0.25.2(E) | 0.26.1(F) | 差异 |
|----|-----------|-----------|------|
| BREAKABLE_CUDAGRAPH 默认 | False | False | 无 |
| FLASHINFER_SAMPLER 默认 | True | True | 无 |
| DraftSampleMethod 默认 | greedy | greedy | 无 |
| 动态 K 支持 | 支持 | 支持 | 无 |
| env_override.py | md5 相同 | md5 相同 | 完全相同 |
| **DSpark block-size 校验** | 无 | **有**（num_spec>=block_size 否则报错） | **0.26 新增** |

### 运行时
- CUDA-graph 捕获方式一致（Breakable enabled + FULL_AND_PIECEWISE + capture_sizes [1,2,4,8,16,24]）
- dspark FULL 图数量 E=3 vs F=11：由动态 K 三档驱动，非版本差异

### 性能 A/B（⚠️ 配置耦合：E=probabilistic 固定K5，F=greedy 动态K）
| 场景 | E(0.25.2) | F(0.26.1) | 备注 |
|------|-----------|-----------|------|
| c1 200→64 | 33.0 | 38.2 | F 快 ~16% |
| c5 200→64 | 30.3 | 69.4 | F 快 129%（动态K并发优势） |
| c1 2048→256 | 30.9 | 20.0 | E 快 55%（单流长 prompt） |
| c5 2048→256 | 42.0 | 42.2 | 持平 |

### 结论
- **0.26 默认行为无实质变化**（唯一新增为 block-size 校验，正确性保护非性能）
- 论坛性能差距**不能由版本默认行为解释**，更可能源于配置差异（probabilistic vs greedy、固定K vs 动态K）——A/B 显示两者在短 prompt 并发（F 赢）与单流长 prompt（E 赢）上表现相反
- **不建议因版本回退 0.25**；彻底隔离版本 vs 配置需同镜像双 spec 对比

### F 恢复确认 ✅
F（v0.26.1dev + 动态 K）已恢复：8001/8003/4000 正常，c1 200→64 = 36.8 t/s 回到基线

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 动态 K 新基线观察期（生产运行 1-2 天确认稳定） | SRE | P1 | 持续 |
| 2 | 彻底隔离版本 vs 配置：同镜像（0.26.1）分别跑 probabilistic/固定K5 vs greedy/动态K 对比 | Testing | P1 | 按需 |
| 3 | 跟踪 DeepGEMM #318/#324 社区 sm120 进展（Mega MoE 移植） | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- 动态 K 采样为 20min 单轮（观察期确认）
- A/B 版本对比与配置差异耦合（E=probabilistic/K5 vs F=greedy/动态K），非纯版本隔离
- Mega MoE 替代路径（分组 MoE 分步）未实测

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE 工程师）**：生产采样 + 动态 K 固化（`~/fparam-exp/prod_sample.py` 等）
- **Archi（架构师）**：DeepGEMM #305/#318/#324 + NVIDIA 官方 + blackwell-wiki 证据链
- **Tessa（测试专家）**：envs.py 静态对比 + 运行时行为 + 性能 A/B
- **前置报告**：`dynk-megamoe-research-2026-08-05.md`、`f-param-verify-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（动态 K 固化、Mega MoE 不引入、不回退 0.25）请由人类工程负责人复核。
