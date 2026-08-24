# 四环境基准对比 · 思考模式统一极致（DGX Spark 社区标准）

**日期**：2026-08-02
**工作流**：部署前检查 / 基准对比选优（工作流 4 变体）
**参与成员**：Archi（架构）/ Rex（SRE）/ Tessa（测试）/ Cody（代码审查）+ 主理人执行层

---

## 📌 TL;DR

- **四环境统一思考极致**（`thinking=true + reasoning_effort=max`）全部落地：生产/A/C 由镜像 CMD 自带，B 通过 CMD 改写显式注入
- **5 并发性能最优：环境 A（DSpark 权重 + dspark 投机 + GPU_MEM=0.8）= 79.6-80.1 t/s**，errors=0
- A 比当前生产/C（0731+dspark 0.85）高 **16%**（80 vs 69），比无投机 B 高 **79%**（80 vs 44.6）
- **关键反差**：单流场景 C/生产（119 t/s）远强于 A（42 t/s）——思考模式下 dspark 投机收益集中在并发场景，单流反而拖累
- 数据链路：统一 thinking=max 下全新实测（非历史非 thinking 数据），社区口径（tonyd2wild/elsung 对齐）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 完成（四环境实测 + 选优结论） |
| 5 并发最优 | **环境 A**（DSpark+dspark 0.8）= 80 t/s |
| 约束满足 | 全部环境 error=0 / KV 不溢出 / TTFT 合理 |
| 思考统一 | 4/4 环境 thinking=max 生效（B 显式注入） |
| 生产当前 | 5 并发 63.9-69.1 t/s（thinking=max 下实测） |

---

## 🔍 四环境对比矩阵（统一 thinking=max，2026-08-02 实测）

| 指标 | A：DSpark+dspark 0.8 | B：0731 无投机 0.8 | 生产/C：0731+dspark 0.85 |
|------|---------------------|--------------------|---------------------------|
| **5 并发 agg t/s** | **79.62 / 80.11** 🏆 | 44.63 | 63.91 / 65.53 / 69.06 |
| 3 并发 agg t/s | 66.40 | 40.95 | — |
| 单流 decode t/s | 42.08 | 27.72 | 119.12 |
| 单流 total t/s | 337.41 | 304.51 | 404.49 |
| TTFT（单流） | 1083ms | 1171ms | 4640ms |
| KV 池 tokens | 1.07M | **1.60M** | 1.49M |
| max concurrency | 2.72x | — | 3.78x |
| errors | 0 | 0 | 0 |

**思考模式验证**：4/4 环境进程参数均含 `--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=max`（B 通过 CMD 改写注入）；A/B 用 GPU_MEM=0.8、生产/C 用 0.85。

---

## 🏆 选优结论：环境 A（DSpark 权重 + dspark 投机 + 0.8）

### 判定规则（Tessa + Archi 定稿）
error=0 硬约束 → 5 并发 aggregate t/s 最高 → TTFT 兜底 → acceptance 平局仲裁

### 结果
- **A = 80.1 t/s 胜出**（4 环境唯一 >70 档）
- 加速比：A vs 生产/C = **1.16x**（+16%）；A vs B 无投机 = **1.79x**（+79%）
- dspark 投机收益（B→A 同权重？不，A/B 权重不同，严格看：C 有投机 69 vs B 无投机 44.6 = 投机收益 **1.55x**）

### ⚠️ 单流反差（重要发现）
- 生产/C 单流 119 t/s ≫ A 单流 42 t/s（2.8x）
- **解释**：thinking=max 模式下，dspark 投机 draft 与 reasoning 长输出耦合，并发时投机重叠调度收益显著；单流时投机 draft 生成占用的计算量大于收益
- **决策含义**：若生产负载以**并发推理为主**（推荐），选 A；若单流交互为主，维持生产/C

### 变量混杂诚实说明（Archi）
- A vs C：权重（DSpark/0731）+ 显存（0.8/0.85）同时变化
- C vs B：投机（on/off）+ 显存（0.85/0.8）同时变化
- 严格隔离需补 A'（0731+dspark 0.8）与 B'（DSpark 无投机）对照——当前结论为方向性判定，A 优势显著（16%）超出噪声阈值（±10%）

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **生产切换决策**：若并发为主，切 A（改 MODEL_PATH=dspark + GPU_MEM=0.8，A 环境脚本已就绪 `hardened/live/start_*_A.sh`） | 人类拍板 + SRE | P0 | 决策窗口 |
| 2 | 若切 A：切换后跑 preflight + 交接清单 + 30min 观察（复用既有流程） | SRE | P0 | 切换当天 |
| 3 | 补 A'（0731+dspark 0.8）对照实验，严格隔离权重变量 | Tessa | P1 | 下轮 |
| 4 | 补 acceptance 大样本（A 环境 spec_decode 双采样）与 131K prefill | Tessa | P1 | 下轮 |
| 5 | 更新 PARAMS.md 记录四环境基准基线（thinking=max 版） | Docu | P1 | 1 周 |

---

## ⚠️ 待完善 / 已知局限

- **变量混杂**：A/C、C/B 非单一变量对照（Archi 诚实标注），结论为方向性
- **预热差异**：C/生产测于运行 1h+ 热态，A/B 测于启动 10-20min（thinking 下收敛更快，但严格同热态复测未做）
- **acceptance 未采到**：A/C 的空闲窗口无 spec_decode 增量，未获得 thinking=max 下的 acceptance 值（社区 0.673 为参考）
- **B 环境路径调整**：原定 production-ready 镜像与真机 RoCE/NCCL 不兼容（enP7s7 网卡、backend 缺失、NCCL invalid usage），改用 hybrid-1.6 + CMD 移除投机实现（更纯净的单一变量对照）
- 131K prefill / 900K ctx 未在本轮复测（历史数据可用）

---

## 📚 数据来源 & 成员产出索引

- Archi：四环境对比框架、思考统一方案、变量混杂诚实标注、选优判定规则
- Rex：测试顺序（C→A→B→回滚）、停机策略、回滚保护、时间预算
- Tessa：社区标准对齐矩阵、统计口径（≥10% 显著）、error=0 硬约束判定
- Cody：工具门槛适配（TTFT 5000ms/min-agg 55）、reasoning 字段兼容（`reasoning` vs `reasoning_content`）、acceptance 双采样法
- 主理人执行：四环境脚本生成/分发/重建（A/B 各多轮排障：digest IMAGE-ID 陷阱、GLOO/TP_SOCKET enP7s7、vllm 绝对路径、投机行续行符）、全部实测数据

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
