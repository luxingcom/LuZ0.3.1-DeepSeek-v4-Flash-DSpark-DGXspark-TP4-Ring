# Grafana 双面板数据健康度核查报告（vllm-realtime + vllm-dspark-cluster）

**日期**：2026-08-02
**工作流**：部署前检查 / 监控面板健康核查（工作流 4 变体）
**参与成员**：Rex（面板核查与修复）/ Tessa（工程负载测试）/ 主理人（编排汇编）

---

## 📌 TL;DR（执行摘要）

- **两面板共 33 图全部有数据**：vllm-realtime 18/18 ✅、vllm-dspark-cluster 15/15 ✅（其中 1 处查询逻辑错误已修复）
- **工程负载验证通过**：3.7 分钟 24 请求 0 错误（S1/S3/S5 × 并发 1/3），负载期间 GPU 利用率 95~96%、RoCE 互联 205~225 MB/s、TTFT 416~3594ms、decode 28~43 t/s
- **修复项专项复验通过**：RoCE 206（IB 计数器）负载期 100+ MB/s/链路 ✅；GPU 功耗/占用率非零且持续 ✅
- 新修复 1 处：集群面板 CL-16 请求成功率旧公式恒 50%（`success/(success+success)`）→ 改为 finished_reason 过滤 + 5m 窗口，实测 100%
- 严重度分布：🔴严重 0 / 🟠高 0 / 🟡中 1（CL-16 逻辑错误，已修复）/ 🟢低 0
- 阻塞 / 非阻塞：**非阻塞**（核查通过，仅 1 条非阻塞建议）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（33/33 图有数据） |
| 阻塞项数量 | 0 |
| 关键行动项 | 2 条（datasource 显式化、修复项已闭环） |
| 建议下一步 | 面板健康基线纳入每周巡检；Anemll 8002 上线后补抓取目标 |

---

## 🔍 核查方法

- 对象：worker58 Grafana(:3000)，vllm-realtime（18 图，refresh 2s）与 vllm-dspark-cluster（15 图，refresh 1s）
- 方式：Grafana API 拉取两面板 JSON → 逐 panel 提取 target expr → Prometheus API（127.0.0.1:8191）验证语法/指标存在/query 与 query_range 非空 → datasource uid 校验 → 泰莎负载期间活性回查
- 抓取目标：node-58/60、dcgm-58/60、vllm 全部 **up**

## ✅ vllm-realtime 面板（18 图，全部正常）

| Panel | 查询（截断） | 负载期实测 | 状态 |
|---|---|---|---|
| TTFT P99/P50 | histogram_quantile(0.99/0.5, ...ttft_bucket[1m]) | 2.5s / 1.75s | ✅ |
| TPOT P99/P50 | rate(...output_token_bucket[1m]) | 0.05s / 0.03s | ✅ |
| 生成吞吐 | sum(rate(generation_tokens_total[30s])) | 14~38 tok/s | ✅ |
| Prompt 吞吐 | sum(rate(prompt_tokens_total[30s])) | 119~175 tok/s | ✅ |
| 请求成功率 | sum(rate(request_success_total[1m])) | 0.05~0.22 req/s | ✅ |
| 抢占率 | sum(rate(num_preemptions_total[1m])) | 0（无抢占，正常） | ✅ |
| KV Cache 水位 | vllm:kv_cache_usage_perc | 0~1.5%（大 KV 池低水位，正常） | ✅ |
| 请求队列 | num_requests_running/waiting | 有数据 | ✅ |
| 前缀缓存命中率 | hits/queries | 0（负载无前缀流量，正常） | ✅ |
| 投机解码 Acceptance | accepted/draft | 37~62% | ✅ |
| GPU 温度 | DCGM_FI_DEV_GPU_TEMP | 63/64℃ | ✅ |
| **GPU 功耗（202）** | DCGM_FI_DEV_POWER_USAGE | **37~51W**（修复项复验） | ✅ |
| **GPU 占用率（203）** | DCGM_FI_DEV_GPU_UTIL | **双卡 95~96%**（修复项复验） | ✅ |
| CPU 占用率 | 100-avg(rate(node_cpu...idle[1m]))*100 | 4.6~7.8% | ✅ |
| 外网速率 | rate(node_network...wlP9s9\|enP7s7) | 0.003~0.03 MB/s（管理网无负载，正常） | ✅ |
| **内网 TP 互联（206）** | rate(node_infiniband_port_data_*{rocep1s0f1\|roceP2p1s0f1})/1048576 | **总 205~225 MB/s、单设备 100~111 MB/s**（修复项复验） | ✅ |

## ✅ vllm-dspark-cluster 面板（15 图，14 正常 + 1 已修复）

| Panel | 查询（截断） | 负载期实测 | 状态 |
|---|---|---|---|
| GPU 利用率/温度 | DCGM_FI_DEV_GPU_{UTIL,TEMP}{job=~"dcgm-.*"} | 95~96% / 63~64℃ | ✅ |
| 请求吞吐 | sum(rate(gen/prompt_tokens[1m])) | 12/119 tok/s | ✅ |
| 运行中请求 | running/waiting | 有数据 | ✅ |
| KV Cache | vllm:kv_cache_usage_perc | 低水位 | ✅ |
| 节点内存可用 | node_memory_MemAvailable_bytes/GB | 3.6/4.2 GB | ✅ |
| TTFT/TPOT P95 | quantile(0.95,[5m]) | 4.25s / 0.048s | ✅ |
| Prefill/队列 P95 | quantile(0.95,[5m]) | 4.4s / 0.29s | ✅ |
| Token 吞吐明细 | gen/prompt/cached | cached=0（无前缀流量，正常） | ✅ |
| 前缀缓存命中率 | hits/queries | 0（正常） | ✅ |
| 请求队列状态 | running/waiting/by_reason | 有数据 | ✅ |
| 抢占数 | rate(num_preemptions[5m]) | 0（正常） | ✅ |
| 投机解码 | accepted/draft、drafts/s | 48.8%、1.12 drafts/s | ✅ |
| **请求成功率（CL-16）** | 旧：success/(success+success)*100 | **恒 50% ❌→已修复** | ✅ |

### 已修复：CL-16 请求成功率（Grafana API，PUT 持久化 status:success，未动 vllm-envc/8000/8001）
- 根因：旧公式 `success/(success+success)*100` 恒等 50%（分母错误）
- 修复：`sum(increase(vllm:request_success_total{finished_reason!~"abort|error"}[5m])) / clamp_min(sum(increase(vllm:request_success_total[5m])),1) * 100`
- 验证：实测返回 **100%**

## 🧪 工程负载测试（Tessa，为核查提供活性数据）

- 时段：**2026-08-02 13:07:15 → 13:10:54（UTC+8）**；3.7 分钟、24 请求、**0 错误**、无超时无 5xx
- 矩阵：S1/S3/S5 × 并发 1/3 × 2 轮/cell，全部成功（dev_tps 87~641、TTFT 416~3594ms、TPOT 27.7~36.5ms、turn 27.4~43.0 t/s）
- 服务端全程健康（v1/models 正常）；thinking=max 下 avg_ct < 设定 out 属模型提前收敛，正常

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 集群面板 7-16 panel 显式指定 datasource uid（PBFA97CFB590B2093），防多源漂移 | Rex | P1 | 下轮迭代 |
| 2 | 面板健康巡检纳入每周例行（PromQL 非空 + 负载活性抽查脚本化） | Rex+Tessa | P2 | 2 周内 |
| 3 | Anemll 环境 E 上线（8002）后：新增 scrape target + 面板核对 0.25 metrics 名 | Rex | P1 | Anemll 部署时 |

## ⚠️ 待完善 / 已知局限

- TTFT/TPOT 在无请求窗口偶现 NaN 属 histogram 正常现象；前缀缓存命中/抢占为 0 均因本次负载无对应流量（正常非异常）
- 集群面板 7-16 panelDS=null 靠唯一默认源解析（当前正常），已列显式化建议
- 核查覆盖 Prometheus 数据层与面板查询层；Grafana 前端渲染层（浏览器缓存）不在本次范围

## 📚 数据来源 & 成员产出索引

- Rex（SRE）：两面板 33 图逐项核查表、修复项专项验证（RoCE 206 / GPU 202-203）、CL-16 修复与验证（SendMessage 回传）
- Tessa（测试）：负载测试矩阵与时段（Temp\vllm_bench\short_load_summary.json）、24 请求 0 错误
- 主理人：任务编排、报告汇编

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
