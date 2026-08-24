# 权重差异 + A 冗长根因 + Mega MoE 适配 + 论坛方案差距 综合报告

**日期**：2026-08-05
**工作流**：工作流 2（架构调研）+ 工作流 1（代码审查）组合
**参与成员**：Cody（代码审查师）/ Archi（架构师）/ Docu（技术文档师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **权重核实**：config/编码结构一致属实，但**权重非字节级相同**（基础层近同 corr 0.9999、spec/draft 模块差异大）；0731 为 7-31 新快照。
- **A 冗长根因**：**serving 栈**——A 镜像 CMD 强制 `thinking=true + reasoning_effort=max`，注入极冗长 CoT 前缀；F 无 thinking kwargs → 简洁输出。**非权重、非 vLLM 版本本身**。
- **Mega MoE**：**GB10 硬件不可适配**（SM100 专属内核，微架构差异不可弥合）；nvfp4_ds_mla × Mega MoE **架构正交可共存**（但 F 硬件下不可用）。
- **论坛方案差距**：3 个立即借鉴项（probabilistic / BREAKABLE_CUDAGRAPH=0 / 动态 K）；**我们 vLLM 0.26.1dev 超出所有公开验证版本**。
- **阻塞 / 非阻塞**：非阻塞。F 生产维持。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 调研完成，4 项结论齐备 |
| 阻塞项数量 | 0 |
| 关键行动项 | 5 条 |
| 建议下一步 | 3 个立即借鉴参数验证 + A 方案修复可选项 |

---

## 🔍 一、权重版本差异核实（Cody，SSH 实测）

| 项 | 结论 |
|----|------|
| config/generation/tokenizer | ✅ 字节完全一致（diff IDENTICAL） |
| safetensors 48 分片 | **二进制全不同**（sha256 全异、大小相同） |
| embed.weight（BF16） | corr=0.999864、MAD≈0.0015 → **基础层仅 ~1% 漂移** |
| mtp.2.ffn.experts.99.w1 | ~99% 字节不同 → **spec/draft 模块差异大** |
| encoding_dsv4.py | dspark 仅 MAX 档；0731 新增 low/high/max 三档 + DEFAULT="low" |
| 0731 独有 | SHA256SUMS_weights_0731.txt（7-31 新快照） |

**结论**：权重**非字节级相同**（基础层近同、spec 层差异大），但此差异**不构成 GSM8K 3.5pp 差距主因**——「权重相同」表述需修正为「config 一致、权重基础层近同」。

---

## 🧠 二、A 方案思维冗长根因（Cody）

| 对比 | A（hybrid-1.6） | F（anemll 0.26） |
|------|----------------|-----------------|
| thinking kwargs | 镜像 CMD **强制** `thinking=true + reasoning_effort=max` | 无任何 kwargs（默认关闭） |
| CoT 表现 | 注入 "Reasoning Effort: Absolute maximum..." 前缀 → 平均 2422 字符，7/200 达 15-17k 字符在 4096 截断、content 空 | 简洁输出（GSM8K 99.0%） |
| tokenizer 语义 | 两镜像 bundled 默认均关 thinking | 同左 |

**根因结论**：**serving 栈**（A 强制 thinking=max），非权重、非 vLLM 版本本身。
**修复建议**：A 移除 CMD 强制 kwargs 或降 effort=low（0731 的 low 为空前缀最省 token）；或请求方显式传 `chat_template_kwargs={"thinking":false}`。

---

## 🔬 三、Mega MoE 适配性 + nvfp4_ds_mla 兼容性（Archi）

### 适配性：❌ 硬件阻塞（非编译问题）
- Mega MoE 内核 `sm100_fp8_fp4_mega_moe` 依赖 SM100 数据中心 Blackwell 的 UMMA/TMEM/集群屏障/3 角色 warp specialization
- DeepGEMM 仅含 sm90_*/sm100_*，**无 sm120_***；GB10 SMEM 99KB vs 228KB 资源预算超标
- nvcc_wrapper（sm_120f→sm_121a）**无法弥合微架构差异**
- b12x 正是为绕开 deep_gemm 不支持 SM120/121 而定制的路径（PR#40082）

### nvfp4_ds_mla × Mega MoE 兼容性：✅ 架构正交、格式无耦合
- 路径分离：nvfp4_ds_mla = KV cache（attention 层消费）；Mega MoE = FFN 专家路径（不触碰 KV）
- 配置独立：`--kv-cache-dtype` 与 `--moe-backend` 无耦合（源码无交叉引用）
- 实证：F 当前 b12x MoE + nvfp4_ds_mla 已共存
- **结论**：理论可共存；F 硬件下 Mega MoE 不可用

### 落地建议
- **不建议 sm121a 移植**（收益边际低、风险高）；GB10 decode 是 273GB/s 显存带宽瓶颈，**b12x+NVFP4+MTP5 已是 GB10 正确解**
- 未来上 SM100：`--moe-backend deep_gemm_mega_moe` 与 `nvfp4_ds_mla` 可直接叠加

---

## 🌐 四、论坛方案性能差距（Docu）

### 立即借鉴 ★（改动小收益大）
| # | 参数 | 他们 | 我们 | 预期收益 |
|---|------|------|------|---------|
| 1 | draft_sample_method | `probabilistic` | `greedy`（我们拍的） | greedy 疑为 C1 39.6 vs 他们 66-96 主因 |
| 2 | VLLM_USE_BREAKABLE_CUDAGRAPH | `0`（0731 核心修复） | 未知（0.26 默认可能开） | +28.6% C1 |
| 3 | 动态 K | `num_speculative_tokens_per_batch_size:[[1,1,5],[2,4,4],[5,6,3]]` | 固定 K=5 | +8-12% 并发 |

### 中等工程 ◎
④ `VLLM_USE_B12X_WO_PROJECTION=1`（省带宽/内存）⑤ util 0.80→0.85 扩 KV 池 ⑥ post-readiness warmup

### 关键警示 ⚠️
- **我们 vLLM 0.26.1dev 超出所有公开验证版本**（Anemll 公开最新 0.1.1/vLLM 0.25.1；他们 1M6 recipe 用 0.1.1）——若 0.26 改变 DSpark/CUDA-graph 默认行为，需 A/B 回退 0.25.1 定位
- 测量口径：output-only decode vs 含 prefill 需统一后再定基准

### ⚠️ greedy vs probabilistic 决策张力（重要）
- 早前 AB 对比：F greedy thinking-on 接受率 +67%（79.2% vs 74.4%）→ 拍板 greedy
- 论坛数据：常规负载（短输出/代码）probabilistic 吞吐更高（他们 C1 66-96 vs 我们 39.6）
- **解释**：greedy 收益在 thinking 重负载；probabilistic 在短输出常规负载更优——**需结合 workload 决策**，建议 A/B 实测两种采样在当前 600K 环境的吞吐差异

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **A/B 验证 draft_sample_method**（probabilistic vs greedy，当前 600K 环境实测吞吐差异）→ 结合 workload 定 F 生产采样 | SRE+Testing | P0 | 本周 |
| 2 | 验证并设置 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（+28.6% C1 潜在） | SRE | P0 | 本周 |
| 3 | 加动态 K 配置 `num_speculative_tokens_per_batch_size`（+8-12% 并发潜在） | SRE | P1 | 本周 |
| 4 | 核对 `VLLM_USE_B12X_WO_PROJECTION=1` + util 0.85 扩 KV 池评估 | SRE+Archi | P1 | 2 周内 |
| 5 | A 方案冗长修复（若保留 A 环境）：移除强制 thinking kwargs 或降 effort=low | SRE | P2 | 按需 |

---

## ⚠️ 待完善 / 已知局限

- 权重差异（spec 层）对性能/质量的具体影响未单独量化（与 serving 栈差异耦合）
- v0.26.1dev vs 0.25.1 的 DSpark/CUDA-graph 默认行为差异未 A/B 验证（需回退测试）
- 论坛数字与我们的测量口径可能不同（output-only vs 含 prefill）

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）**：权重 diff/sha256/corr 实测 + A 冗长根因（镜像 CMD kwargs 证据）
- **Archi（架构师）**：Mega MoE 适配性（DeepWiki/vLLM#42845/PR#40082）+ nvfp4 正交性
- **Docu（技术文档师）**：4 源调研（al-engr ×2 / Anemll / MiaAI-Lab）+ 差距表
- **前置报告**：`a-recheck-f128-optimize-2026-08-05.md`、`bench-compare-ABCDEF-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（draft 采样切换、BREAKABLE_CUDAGRAPH、Mega MoE 不引入）请由人类工程负责人复核。
