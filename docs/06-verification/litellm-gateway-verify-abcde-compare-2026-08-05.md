# LiteLLM 网关落地验证 + 多部署方案（ABCDD'E）横向比较综合报告

**日期**：2026-08-05
**工作流**：工作流 4（部署验证）+ 工作流 2（架构调研）组合
**参与成员**：Rex（SRE 工程师）/ Cody（代码审查师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **LiteLLM 网关验证 9/9 通过**（models/chat/embeddings/responses/工具/SSE/reasoning 字段全绿），并发稳定性优于自建网关 8003；**但发现 1 项关键回归**：LiteLLM 的 `/v1/responses` 端点丢失 thinking 思考链（reasoning 为 null），依赖 Responses API 思考链的客户端切换将回归。
- **多方案横向比较完成**：服务器存在 A/B/C/D/D'/E/E-v026r 共 7 套方案环境；**D vs E 核心差异 = SM121a 适配**（E 靠 nvcc wrapper 把 sm_120f→sm_121a 跑通 GB10，D' 无 wrapper 不可用），**E-v026r（:0.2.1）为当前生产**。
- **严重度分布**：🟠高 1 项（Responses+thinking 回归风险）/ 🟡中 1 项（虚拟 key 迁移）
- **阻塞 / 非阻塞**：非阻塞。LiteLLM 4000 已就绪并行验证，切换决策待定。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 LiteLLM 可行但 Responses+thinking 有回归，需权衡 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条 |
| 建议下一步 | 评估 Responses 思考链依赖度 → 决定 LiteLLM 切换 or 保留 8003 |

---

## 🚪 一、LiteLLM 网关落地验证（Rex，9/9 PASS）

### 部署
| 项 | 结果 |
|----|------|
| 镜像 | ghcr.nju.edu.cn/berriai/litellm:v1.83.7-stable（1.98GB，ghcr 直连失败走镜像源） |
| 容器 | litellm-proxy（--network host，端口 4000）+ litellm-pg（PostgreSQL，虚拟 key 持久化） |
| config | model_list：local-v4-flash + 别名 deepseek-v4-flash → hosted_vllm/deepseek-v4-flash-0731@8001；local-embedding → 8020 |
| 虚拟 key | <API_KEY>（LiteLLM 管理，与 <API_KEY>-* 体系不同） |

### 功能验证 9/9
| 端点 | 结果 |
|------|------|
| /v1/models | ✅ 200（含 3 模型） |
| /v1/chat/completions | ✅ 200（"2+2"→"4"；别名可用） |
| /v1/embeddings | ✅ 200（1024 维） |
| /v1/responses | ✅ 200（原生支持） |
| 工具调用 | ✅ 200（get_weather 上海） |
| SSE 流式 | ✅ data + [DONE] |
| **reasoning 字段** | ✅ chat 路径：reasoning_content（顶层）+ provider_specific_fields.reasoning（原字段保留）；流式逐片输出 |

### 性能对比（10 并发 × 2 轮）
| 网关 | r1 | r2 | 结论 |
|------|----|----|------|
| LiteLLM 4000 | 1.15s（p50 772ms） | 0.81s（435ms） | 并发稳定 |
| 自建 8003 | 5.98s（p50 5.6s 首轮排队） | 1.03s（649ms） | 首轮排队明显 |

### ⚠️ 关键风险：/v1/responses + thinking 回归
- LiteLLM responses 端点：`reasoning:null`、output 无 reasoning 项——**思考链丢失**
- 自建 8003 passthrough：输出 `type=reasoning` + `reasoning_text` 思考链
- **影响**：依赖 Responses API 思考链的客户端（非 chat 路径）切换即回归

---

## 🏗️ 二、多部署方案横向比较（Cody，ABCDD'E）

### 方案全景表

| 方案 | 镜像 | vLLM | ctx | KV | MEM | seqs | DSpark | SM121a | 定位 |
|------|------|------|-----|----|----|------|--------|--------|------|
| A | hybrid-1.6 | 0.11.2dev | 393K | fp8 | 0.80 | 128 | prob | ✓ AOT | dspark 模型基线 |
| B | hybrid-1.6 | 0.11.2dev | 393K | fp8 | 0.80 | 128 | 无（剥离 spec） | ✓ | A 去 spec 对照 |
| C | hybrid-1.6 | 0.11.2dev | 393K | fp8 | 0.85 | 128 | prob | ✓ | 修复基线 fix(0.85) |
| D | hybrid-1.6 | 0.11.2dev | **1M** | fp8 | 0.85 | **1** | prob | ✓ | 1M 单流（Anemll 降级） |
| D' | anemll 0.1.1 | 0.25.2dev | 1M | nvfp4_ds_mla | 0.85 | 6 | prob | ✗ 原生 sm_120f | Anemll 1M（无 wrapper） |
| **E** | anemll 0.1.1 | 0.25.2dev | 600K | nvfp4_ds_mla | 0.80 | 6 | prob | ✓ wrapper | E=D'+wrapper+认证 |
| E-v026 | anemll 0.2.0 | 0.26.1dev | 600K | nvfp4_ds_mla | 0.80 | 6 | greedy | ✓ | +11 overlay 补丁 |
| **E-v026r** | anemll 0.2.1 | 0.26.1dev | 600K | nvfp4_ds_mla | 0.80 | 6 | greedy | ✓ | **★当前生产** |

### D vs E 核心差异（Cody 结论）
1. **上下文与并发**：D=1M 但 seqs=1 单流 + KV 仍 fp8；E=600K × seqs6 并发 + FP4 MLA（nvfp4_ds_mla）省显存 → 吞吐/利用率远高
2. **SM 适配（决定性）**：anemll 0.1.1 镜像 .so 只编 sm_120f，GB10 硬件是 sm_121a；**D' 无 wrapper 无法出可用 kernel（半成品）**，E 加 nvcc wrapper（sm_120f→sm_121a）+ DG_JIT_NVCC_COMPILER=wrapper 才跑通
3. **E 额外**：api-key 认证、tool/reasoning parser、vllm-cache 持久化（免重编）
- **结论**：E 是唯一"sm121a 可用 + 并发 6 + 600K + 认证"完整方案；D=1M 单流降级尝试；D'=技术验证不可用

### v0.26 增量（E-v026r）
- 升级 vLLM 0.26.1dev + 采样改 greedy（AB 对比 +67% 接受率提升）
- 0.2.0 需 11 个 overlay 补丁；0.2.1（v026r）原生含修复免 overlay → 成为当前生产

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **决策**：Responses API 思考链依赖度评估 → LiteLLM 切换 or 保留 8003（chat 路径已无碍，responses 路径有回归） | 人类负责人 | P0 | 立即 |
| 2 | 若切换：客户端 key 迁移方案（<API_KEY>-* → LiteLLM 虚拟 key）+ 8003 下线窗口 | SRE | P1 | 决策后 |
| 3 | 双轨并行观察期：LiteLLM 4000 对外 + 8003 保留（responses 思考链兜底） | SRE | P1 | 1-2 周 |
| 4 | 废弃镜像/脚本清理确认（production-ready 底包、A/B/C/D 仅调试用） | SRE | P2 | 按需 |

---

## ⚠️ 待完善 / 已知局限

- LiteLLM /v1/responses 思考链丢失为 v1.83 实测行为，未确认新版是否修复（可跟踪 LiteLLM release）
- 虚拟 key 与现有鉴权体系不兼容，客户端需迁移
- 方案比较基于脚本 + 镜像版本实测（只读），未启动非生产容器验证

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE 工程师）**：LiteLLM 部署（9/9 验证 + 性能对比 + responses 回归风险）
- **Cody（代码审查师）**：7 方案横向比较表 + D vs E 核心差异（SM 适配决定性）+ v0.26 增量
- **前置报告**：`metrics-sm121-gateway-research-2026-08-05.md`（网关调研）、`ab-compare-v026-rebuild-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（LiteLLM 切换、Responses 思考链处理）请由人类工程负责人复核。
