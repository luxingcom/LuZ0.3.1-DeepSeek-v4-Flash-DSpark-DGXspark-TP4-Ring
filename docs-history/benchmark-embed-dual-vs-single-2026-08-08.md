# B 组 embed 恢复 + 双机 vs 单机路由入口测速报告

**日期**：2026-08-08
**工作流**：部署恢复 + 性能基准（embed 路由入口对比）
**参与成员**：主理人（实施与测速）/ Tessa（口径判定）
**涉及节点**：<MGMT_OCTET>(node01) / <MGMT_OCTET>(node01) / <MGMT_OCTET>(node01 网关) / <MGMT_OCTET>(node01)

---

## 📌 TL;DR（执行摘要）

- **恢复动作**：停掉 03/04 的 B 组 TP2 benchmark LLM（8001，benchmark 已完成、生产 LLM 走 <MGMT_OCTET>:8001 不受影响），在 03/04 各启动 **anemll 0.2.1 embed（8022 端口，单卡，VLLM_GPU_MEMORY_UTILIZATION=0.15）**，均 /health 200 + dim 1024
- **路由恢复**：litellm 网关 embed 池收敛为 **2 deployment（<MGMT_OCTET>:8022 + <MGMT_OCTET>:8022）**，修复了 2 个配置缺陷（<MGMT_OCTET> api_base 注释残留 active → 偶发 500；<MGMT_OCTET> 孤儿行）——修复后 0 错误
- **双机 vs 单机结论**：**双机更快（高并发下吞吐 +28~32%、p50 延迟 -32%）**；低并发（c1/c4）无差异（单请求延迟受单机 embed 决定）
- **上限对照**：直连上游单机 c16=553 req/s，经网关双机 c16=362 → **litellm 网关本身成为吞吐瓶颈**（simple-shuffle 转发 + 单点代理）
- 阻塞 / 非阻塞：非阻塞（恢复完成 + 测速完成）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（embed 双机恢复 + 双机优于单机验证） |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条（网关瓶颈评估/分摊偏差/超长输入策略/内存预算复核） |
| 建议下一步 | 若业务高并发 >400 req/s 需评估 litellm 多实例或直连 + LB |

---

## 1. 恢复操作记录

### 1.1 前置状态（08-08 20:37 检查）

| 节点 | 恢复前状态 | 问题 |
|------|-----------|------|
| <MGMT_OCTET>/<MGMT_OCTET> | 运行 B 组 TP2 benchmark LLM（vllm-groupb-head/worker，8001） | 统一内存占满（avail 仅 6-7G），无法共存 embed |
| <MGMT_OCTET> | litellm config embed 池指向 <MGMT_OCTET>:8020/<MGMT_OCTET>:8020/<MGMT_OCTET>:8020/<MGMT_OCTET>:8022 | 4 条 deployment 全部失效（上游无服务或端口错误） |
| <MGMT_OCTET> | 无 embed 容器 | — |

### 1.2 执行动作（时间线）

1. **停 TP2**：03 `docker stop/rm vllm-groupb-head`、04 `docker stop/rm vllm-groupb-worker`（先确认 8001 无活跃业务连接；litellm LLM 路由均指向 <MGMT_OCTET>:8001 = A 组，B 组 8001 仅 benchmark 用）→ 内存释放至 avail 117G
2. **起 embed**：03/04 各 `docker run -d --name anemll-embed-8022 --restart unless-stopped --gpus all -p 8022:8022 -e VLLM_GPU_MEMORY_UTILIZATION=0.15 --entrypoint vllm anemll/dspark-vllm-gx10:0.2.1-v026.0 serve /models/Qwen3-Embedding-0.6B --port 8022 --max-model-len 8192 --max-num-seqs 32 --enforce-eager --trust-remote-code`（**无 --task embed**，anemll 自动识别 embedding 架构）
3. **改网关**：litellm config.yaml embed 池 → `<MGMT_OCTET>:8022 + <MGMT_OCTET>:8022`（2 active），注释 <MGMT_OCTET>/<MGMT_OCTET>；备份 config.yaml.bak-20260808204220 / config.yaml.dual-bak
4. **修复 500**：重启后测速发现偶发 500 → 根因 `<MGMT_OCTET>` deployment 的 api_base 被注释但 deployment 仍 active（litellm 随机路由到无 api_base 条目）→ 整体注释该块，0 错误
5. **验证**：/v1/models 含 local-embedding；3 连发全 200；20 连发分布 03:04 = 13:7（simple-shuffle 随机分摊）

### 1.3 最终部署状态

| 节点 | 容器 | 端口 | 状态 | 预算 |
|------|------|------|------|------|
| <MGMT_OCTET>(node01) | anemll-embed-8022 | 8022 | ✅ Up / health 200 / dim 1024 | 0.15 ≈ 18G |
| <MGMT_OCTET>(node01) | anemll-embed-8022 | 8022 | ✅ Up / health 200 / dim 1024 | 0.15 ≈ 18G |
| <MGMT_OCTET>(node01) | litellm-proxy :4000 | — | ✅ 2 deployment 路由正常 | — |

---

## 2. 测速方法与口径

- 客户端：<MGMT_OCTET>（网关本机），httpx asyncio 并发；文本 ~300 字（模拟业务负载）；每并发档 25 req/worker × 3 轮；先 warmup 1 次
- 四种模式：
  - **单机**：经 litellm（临时注释 <MGMT_OCTET>，仅 <MGMT_OCTET> 路由）
  - **双机**：经 litellm（<MGMT_OCTET> + <MGMT_OCTET> simple-shuffle）
  - **直连 <MGMT_OCTET> / 直连 <MGMT_OCTET>**：绕过网关直连上游（理论上限对照）
- 指标：p50/p95 延迟（ms）、吞吐（req/s，含总耗时口径）、错误数

---

## 3. 测速结果

### 3.1 经 litellm：双机 vs 单机（核心对比）

| conc | 单机 tps | 双机 tps | **增益** | 单机 p50 | 双机 p50 | 单机 p95 | 双机 p95 |
|------|---------|---------|---------|---------|---------|---------|---------|
| 1 | 44.4 | 43.8 | -1% | 21.60ms | 21.76ms | 25.15 | 27.16 |
| 4 | 123.7 | 121.7 | -2% | 30.97ms | 31.53ms | 41.37 | 39.83 |
| 8 | 241.2 | **309.8** | **+28%** | 31.07ms | **23.83ms** | 44.40 | 35.98 |
| 16 | 274.0 | **362.2** | **+32%** | 59.41ms | **40.46ms** | 72.20 | 58.63 |
| 32 | — | 417.2 | — | — | 70.92ms | — | 108.62 |

### 3.2 直连上游（理论上限对照）

| 节点 | c1 tps | c1 p50 | c8 tps | c16 tps | c16 p50 |
|------|--------|--------|--------|---------|---------|
| 直连 <MGMT_OCTET>:8022 | 64.7 | 15.41ms | 368.8 | 553.1 | 27.32ms |
| 直连 <MGMT_OCTET>:8022 | 63.9 | 15.62ms | 358.8 | 551.4 | 26.33ms |

### 3.3 分布验证

20 连发：03=13、04=7（simple-shuffle 随机分摊，与 8/7 已知分摊偏差一致）

---

## 4. 结论（Tessa 判定）

### 4.1 双机比单机更快吗？→ **是，高并发下显著**
- **c8/c16 吞吐 +28%/+32%，p50 延迟 -23%/-32%**——并发 ≥8 时双机优势明显
- **c1/c4 无差异**（-1%/-2%）：单请求延迟由单台 embed 计算决定，路由开销相同 → 低并发下双机不亏不赚
- 结论：**业务并发 ≥8 时双机部署有明确收益；<4 并发场景双机仅提供 HA 冗余价值**

### 4.2 但 litellm 网关成为瓶颈（关键发现）
- 直连单机 c16=553 req/s vs 经网关双机 c16=362 req/s → **网关吞吐上限约 360-420 req/s，低于单台 embed 上游能力**
- 双机经网关仅达理论上限（~1100）的 1/3 → 瓶颈在 **litellm simple-shuffle 转发链路**（单点代理 + 序列化路由），不在 embed 计算
- 若业务需求 >400 req/s：需评估 litellm 多实例/多 worker 或业务直连 + 前置 LB

### 4.3 配置缺陷修复（本轮额外价值）
- 修复前：偶发 500（约 5%）——`<MGMT_OCTET>` deployment api_base 注释残留但条目 active
- 修复后：全档位 0 错误 → **生产 embed 路由可靠性恢复**

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 评估业务侧 embed 实际并发水位；若 >400 req/s 规划 litellm 多实例/worker 或直连+LB 方案 | Tessa/主理人 | P1 | 业务压测后 |
| 2 | 分摊偏差观察：simple-shuffle 随机分摊（13:7），若业务可接受则不动，否则评估 least-busy | SRE | P2 | 观察 24h |
| 3 | 超长输入（>8192 token）截断策略实测（Gate B 遗留） | SRE | P2 | 1 周内 |
| 4 | 确认 0.15 预算在业务峰值下与 <MGMT_OCTET> 后端栈无内存互斥（avail 观察） | SRE | P3 | 观察 24h |

---

## ⚠️ 待完善 / 已知局限

- 测速客户端与网关同机（<MGMT_OCTET>），网络 RTT 未计入；真实业务跨机调用延迟会略高（预估 +1-3ms）
- simple-shuffle 随机分摊导致瞬时分布不均（13:7），c16 双机增益 32% 略低于理想 2×
- 文本为 300 字中文（~150 token），未覆盖超长/批量多样本场景
- B 组 TP2 LLM（8001）已停——若需复跑 131072/c5 回归需重新编排（head-first）
- 未测 batch 多输入（单请求多文本）场景——vLLM embed 的 batch 语义（--max-num-seqs 32）与吞吐关系待业务实测

---

## 📚 数据来源 & 成员产出索引

- 实测数据：<MGMT_OCTET> /tmp/embed_bench_litellm.py（脚本）+ 三轮测速输出（本报告 §3 表）
- 本地副本：_archive_scratch/bench_B/embed_bench_litellm.py、embed_bench_data.py
- 配置备份：<MGMT_OCTET> /home/<USER>/litellm/config.yaml.bak-20260808204220（恢复前）、config.yaml.dual-bak（单机测试前双机态）
- 容器日志：<MGMT_OCTET>/<MGMT_OCTET> anemll-embed-8022（访问计数）、<MGMT_OCTET> litellm-proxy（500 根因定位）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
