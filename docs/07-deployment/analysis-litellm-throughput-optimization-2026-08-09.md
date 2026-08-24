# litellm 网关吞吐链路优化空间分析报告

**日期**：2026-08-09
**工作流**：性能诊断（网关吞吐瓶颈定位与优化评估）
**参与成员**：主理人（实测诊断）/ Tessa（口径判定）
**被测对象**：<MGMT_OCTET> litellm-proxy 1.83.7（单进程 uvicorn，:4000）→ embed 池（<MGMT_OCTET>:8022 + <MGMT_OCTET>:8022）

---

## 📌 TL;DR（执行摘要）

- **结论：优化空间大，且瓶颈定位清晰——不是 CPU、不是上游计算，而是 litellm 单进程的"每请求固定开销"（HTTP 解析 + 鉴权 + 1024 维 JSON 序列化 + 转发链）**
- 证据：压测期间 litellm CPU 仅 1%（/proc 精确采样）、上游 embed CPU 2.8%、直连单机 c16=553 req/s，但经网关封顶 ~380-400 req/s
- **关键发现 1（batch 杠杆）**：batch 16 时条/s 达 1100（vs 单条 239，**4.6× 提升**）→ 业务侧改批量调用即可免费吃到上游能力
- **关键发现 2（rpm 硬限流）**：PG 虚拟 key `embedding` / `chat-v4-flash` 配置 **rpm_limit=300**，历史日志 14452 条 429 来自业务侧 <MGMT_OCTET> → **业务侧吞吐被 300 req/min 卡死，这是当前最现实的瓶颈**
- 优化方案分级：P0 解除 rpm（立即可做）→ P1 batch 化（免费 4.6×）→ P2 多 worker（×2-4）→ P3 直连+LB（达上游上限）

---

## 1. 瓶颈定位证据链

### 1.1 压测期间资源占用（瓶颈排除法）

| 观测对象 | 采样方式 | 结果 | 结论 |
|---------|---------|------|------|
| litellm-proxy CPU | docker stats + /proc Δutime | **0.1-1.5%（近 0）** | ❌ 非 CPU 瓶颈 |
| litellm-pg CPU | docker stats | 0-4% | ❌ 非 DB 瓶颈 |
| 上游 <MGMT_OCTET> embed CPU | docker stats | 2.79% | ❌ 非上游计算瓶颈 |
| 上游 <MGMT_OCTET> GPU util | nvidia-smi | 0% | ❌ 非 GPU 瓶颈 |

### 1.2 吞吐封顶实测（经 litellm 双机，master_key）

| conc | 1 | 4 | 8 | 16 | 32 | 64 |
|------|---|---|---|---|----|----|
| tps | 44 | 122 | 260-310 | 362-383 | 377 | 402 |

→ c16 后进入平台（~380-400 req/s），**再加大并发吞吐不涨**，延迟线性上升（p50: c16=40ms → c64=151ms）→ 单进程事件循环串行瓶颈特征

### 1.3 batch 对照（区分"每请求" vs "每条文本"开销）

| batch | req/s | **条/s** | p50 |
|-------|-------|---------|-----|
| 1 | 239 | **239** | 29ms |
| 4 | 141 | **563** | 53ms |
| 16 | 69 | **1100** | 110ms |

→ **条/s 随 batch 近乎线性增长（4.6×）**，证明瓶颈在"每请求"固定开销（litellm 转发链），而 vLLM 上游 batch 效率极高。**上游能力远未用尽**（直连单机 batch=1 已 553 req/s，batch 16 理论更高）

### 1.4 直连 vs 经网关对照

| 路径 | c16 tps |
|------|---------|
| 直连 <MGMT_OCTET>:8022（batch=1） | 553 |
| 直连 <MGMT_OCTET>:8022（batch=1） | 551 |
| 经 litellm 双机（batch=1） | 362-383 |

→ 网关单请求链路吃掉上游 ~30% 能力

---

## 2. 关键发现：业务 key 存在 300 rpm 硬限流

### 2.1 PG 虚拟 key 限流配置（litellm-pg）

| key_alias | rpm_limit | tpm_limit |
|-----------|-----------|-----------|
| embedding | **300** | 100000 |
| chat-v4-flash | **300** | 50000 |
| dspark-prob / dspark-greedy / (空) | 无 | 无 |

### 2.2 证据

- 历史日志 **14452 条 429**：`Rate limit exceeded. Current limit: 300, Remaining: 0. Limit resets at ...`，来源 <NODE_IP>（<MGMT_OCTET> 业务侧）
- 本轮测速用 master_key（无 rpm）→ 未触发 429，但**业务侧用 embedding key 时 300 req/min 即被拒**
- 近 10 分钟无新 429（业务侧当前流量未触顶），但**一旦业务量上升立即触发**

> ⚠️ 这是**当前最现实的吞吐天花板**：无论网关/上游多快，业务侧 embedding key 只能 300 req/min（5 req/s）。业务高峰期若曾出现过 429 重试风暴，即源于此。

---

## 3. 优化方案（按性价比排序）

### P0 — 解除业务 key 限流（立即可做，成本≈0）

```sql
UPDATE "LiteLLM_VerificationToken" SET rpm_limit = NULL, tpm_limit = NULL
WHERE key_alias IN ('embedding', 'chat-v4-flash');
-- 或改为显式大值：rpm_limit = 100000
```

- 预期：业务侧不再被 300 rpm 卡死，吞吐上限立即变为网关瓶颈（~380-400 req/s）
- 风险：低（内部集群，无外部攻击面；如需防滥用可设 100000 量级）
- ⚠️ 需确认业务侧预期：chat-v4-flash 300 rpm 若为有意的风控，建议只解除 embedding

### P1 — 业务侧 batch 化调用（免费 4.6×）

- 现状：业务单条文本一次请求 → 网关每请求开销成瓶颈（239 条/s）
- 改法：一次请求带 N 条文本（如 batch 8-16）→ 实测 563-1100 条/s
- 预期：**不改任何基础设施，条/s 提升 4.6×**；vLLM embed 原生支持多输入（--max-num-seqs 32）
- 风险：低（上游 batch 语义已验证）；需业务侧适配

### P2 — litellm 多 worker（×2-4，官方支持）

- litellm 1.83.7 CLI 支持 `--num_workers N`（uvicorn/gunicorn workers），源码已确认（proxy_cli.py:383）
- 预期：单进程瓶颈解除，吞吐近线性扩展至 ~800-1600 req/s
- 注意：需确认 PG 并发安全（litellm-pg 已就绪）、Prometheus multiproc（当前无 prometheus callback 则免）
- 风险：中（需重启网关，短暂中断；建议维护窗口操作 + 先备份 config）

### P3 — 直连上游 + 前置 LB（达上游真实上限）

- 若业务量 >1600 req/s：应用层 LB（nginx/haproxy）直连 <MGMT_OCTET>/<MGMT_OCTET>:8022，绕过 litellm 单点
- 预期：达上游上限（batch=1 时 ~1100 req/s 双机；batch 16 时更高）
- 代价：丢失 litellm 的鉴权/监控/fallback 能力 → 需在 LB 层补鉴权
- 备选：litellm 多实例（同 config 起 2 容器 + nginx 前置），保留网关能力且横向扩展

---

## 4. 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期效果 |
|---|------|---------|--------|---------|
| 1 | 确认 chat-v4-flash 300 rpm 是否有意；解除 embedding key rpm（P0 SQL） | Tessa/主理人 | **P0** | 业务 5→380 req/s |
| 2 | 业务侧 embed 调用改 batch（8-16 条/请求） | BE | P1 | 条/s 4.6×（~1100） |
| 3 | 评估 litellm --num_workers 2-4（维护窗口灰度） | SRE | P2 | 网关 380→800-1600 |
| 4 | 若目标 >1600：规划 nginx 前置双网关/直连+LB 架构 | Archi | P3 | 达上游上限 |

---

## 5. 局限与说明

- 本次测速客户端与网关同机（<MGMT_OCTET>），网络 RTT 未计入；真实业务跨机延迟略高
- batch 16 时 p50=110ms（单请求聚合 16 条），业务需评估单请求延迟预算
- 429 历史日志 14452 条为累计（跨多日），非单次事件
- master_key 不受 rpm 限制，本报告吞吐上限数据（380-400）为"解除限流后"的网关能力基准
- 未实测 P2/P3（需重启/变更，建议维护窗口验证）

---

## 📚 数据来源

- 实测：<MGMT_OCTET> /tmp/embed_bench_litellm.py（并发/batch 压测）+ /proc CPU 采样 + docker stats
- 配置：<MGMT_OCTET> /home/<USER>/litellm/config.yaml（router_settings / litellm_settings）
- 数据库：litellm-pg `LiteLLM_VerificationToken`（rpm/tpm 字段，5 个 key）
- 日志：litellm-proxy 容器日志（429 计数 14452、来源 <MGMT_OCTET>）
- 本地副本：_archive_scratch/bench_B/embed_bench_data.py（8/8 测速基线）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
