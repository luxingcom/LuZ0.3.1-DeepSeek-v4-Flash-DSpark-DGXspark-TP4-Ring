# 四机扩展隐患复查 + 管理网延迟 + TileLang JIT + 负载自适应 综合报告

**日期**：2026-08-05
**工作流**：工作流 2（架构调研）+ 工作流 1（代码审查）+ 工作流 4（部署评估）组合
**参与成员**：Archi（架构师）/ Cody（代码审查师）/ Docu（技术文档师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **网络复查**：数据面安全（TP 100% 走 200G RoCE 直连），**管理面隐患**——worker 管理网实际走 WiFi（avg 11ms / max 79.8ms 抖动）；Top 3 待处理（WiFi 接有线 / QSFP56 线缆认证 / NCCL hang 保护）。
- **TileLang ~5s JIT 根因明确**：`tile_n/n_splits` 随 num_tokens 变化进 cache key → 每请求 miss 重编；**方案 B（缓存持久化）立即止血 + A+D（patch 解耦 + 预热）根治，c1 TTFT 18s→~1s**。
- **负载自适应**：难度=中等，推荐 **per-key 模板起步**（LiteLLM aliases 间接实现）+ 网关 hook 兜底。
- **probabilistic 切换已采纳**：请求端温度强制 >0.1。
- **阻塞 / 非阻塞**：非阻塞。F 生产维持。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 复查完成，行动项明确 |
| 阻塞项数量 | 0 |
| 关键行动项 | 6 条 |
| 建议下一步 | 网络 Top3 + TileLang 方案 B 立即执行 |

---

## 📡 一、网络复查（Archi）

### 实测数据
| 网 | 速率 | ping（head→worker） |
|----|------|---------------------|
| 管理网 | head 2.5G 有线 / **worker 走 WiFi**（有线 DOWN） | avg 11.0ms / max 79.8ms（jitter 15.8） |
| RoCE-136 | **200G** | avg 0.52ms |
| RoCE-137 | **200G** | avg 0.55ms |

### 结论
- ✅ **管理网不影响 TP 数据面**：NCCL_NET=IB + 双 HCA + MASTER_ADDR 全走 10.100.136.x → TP 100% 走 200G RoCE
- ⚠️ **管理面单点隐患**：worker WiFi（若误走管理网 TPOT 秒级恶化）——**锁死 10.100.136.x 是生命线**（P1/P2 变体已做）

### 知乎文章隐患映射（Top 3）
| # | 隐患 | 等级 | 建议 |
|---|------|------|------|
| 1 | worker 管理网走 WiFi | 🔴高 | 接有线 |
| 2 | QSFP56 线缆认证待核验 | 🔴高 | 换认证 DAC |
| 3 | RoCE 无 hang 保护 | 🟠中 | NCCL_IB_TIMEOUT=1000s + RETRY_CNT=7 |

---

## 🔧 二、TileLang JIT 优化（Cody）

### 根因
- `mhc_pre_big_fuse_*_tilelang`（MoE 前 MHC pre 融合算子）`@tilelang.jit` 动态 grid
- **cache key 含 `tile_n/n_splits` 等 int 参数**——随 num_tokens 变化 → 每请求 miss → ~5s 全量编译
- 冷启动 10-26s：容器 `~` 不持久，磁盘缓存丢失

### 方案比选
| 方案 | 内容 | 收益 | 风险 |
|------|------|------|------|
| **B（立即）** | TILELANG_CACHE_DIR 持久卷挂载 | 消冷启动 10-26s | 低 |
| **A+D（短期根治）** | patch tilelang.py 将 tile_n/n_splits 与 num_tokens 解耦（固定 prefill/decode 两档）+ 预热两档 | **每请求 5s→0，c1 TTFT 18s→~1s** | 中（需回归） |
| C（中期） | 镜像升级启用 b12x MHC（AOT 无 JIT） | 从根消除 | 中-高（版本兼容） |

**推荐**：B 立即止血 → A+D patch 根治 → C 中期纳入镜像升级。

---

## 🎛️ 三、负载自适应方案（Docu）

### 结论：难度=中等，推荐 per-key 模板起步 + 网关 hook 兜底
| 方案 | 改动量 | 风险 | 适用 |
|------|--------|------|------|
| **1. per-key 模板**（起步） | 小 | 低 | 应用边界清晰 |
| 2. 网关 pre_call_hook（兜底） | 中 | 中 | 混合流量 |
| 3. 客户端约定 | 最小 | 高 | 内部可信 |

### LiteLLM 能力（核实）
- ❌ per-key 直接覆盖 temperature 不支持
- ✅ **per-key aliases 间接实现**（逻辑模型名→不同 model_list 条目，各带独立参数）
- ✅ 网关 pre_call_hook 可行

### 实施要点
- 结构化 key→prob 模板（temp 0.7）、散文/思考链 key→greedy 模板（temp 0.1 + enable_thinking）
- **temp 用 0.1 而非 0**（LiteLLM 部分 provider 取整）
- 先灰度一个结构化应用验证（+20~47% 基线）再推广

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **TileLang 方案 B**：TILELANG_CACHE_DIR 持久卷（双机）——消冷启动 10-26s | SRE | P0 | 本周 |
| 2 | **网络 Top1**：worker 管理网接有线（WiFi 单点隐患） | SRE+用户 | P0 | 本周 |
| 3 | **网络 Top2**：核验 QSFP56 线缆认证 | SRE | P1 | 本周 |
| 4 | **网络 Top3**：补 NCCL_IB_TIMEOUT=1000s + RETRY_CNT=7 到双机脚本 | SRE | P1 | 本周 |
| 5 | **TileLang A+D**：patch tile_n/n_splits 解耦 + 预热两档（c1 TTFT 18s→~1s） | Cody+SRE | P1 | 2 周内 |
| 6 | **probabilistic 切换落地**：采样改 probabilistic + 请求端 temp>0.1 强制 + per-key 模板（结构化 prob / 散文 greedy） | SRE+LiteLLM | P0 | 本周 |

---

## ⚠️ 待完善 / 已知局限

- 知乎原文为线性扩展测试（非踩坑清单），隐患经社区文交叉验证
- LiteLLM 请求级 vs model 级参数优先级未实测（需环境验证）
- TileLang patch（A+D）需容器内回归（GSM8K/工具调用）
- b12x MHC（方案 C）需确认镜像版本兼容

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：网络实测（WiFi/RoCE 双 ping）+ 知乎/社区隐患映射（eugr/Tencent/双机实录）
- **Cody（代码审查师）**：tilelang.py 调度分支分析 + cache key 机制 + 方案比选（PR#43474 / route179 / Chthonic 配方）
- **Docu（技术文档师）**：负载自适应三方案 + LiteLLM aliases 核实 + 配置示例
- **前置报告**：`combo-ab-prob-eval-2026-08-05.md`、`deepgemm-moe-path-research-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（probabilistic 切换 + temp>0.1、TileLang B 方案、网络 Top3）请由人类工程负责人复核。
