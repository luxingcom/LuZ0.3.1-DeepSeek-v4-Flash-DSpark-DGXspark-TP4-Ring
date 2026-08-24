# Grafana 面板数据核查报告（2026-08-09）

> 核查对象：187（node01）Grafana「DGX Spark vLLM 实时分析」+「DGX Spark vLLM 集群」面板。逐项核实 8 个用户提出的问题，附证据。

## 结论摘要

| # | 问题 | 结论 | 性质 |
|---|------|------|------|
| 1 | 面板数据少了很多 | B 组 vLLM（.188:8001）当前未运行 → vLLM 指标只剩 A 组；TSDB 数据连续（72h up=1） | 现状合理，非故障 |
| 2 | CPU 占用率曲线不全 | **四台数据齐全**（01:4.5-29.9% 02:3.6-25.2% 03/04:0.2-2.7%）；03/04 曲线贴 0 轴看似缺失 | 数据真实 |
| 3 | KV cache 水位单位太小 | **确认 bug**：`vllm:kv_cache_usage_perc` 是 0-1，面板未乘 100（实测 0.37 应为 37%） | 需修复 |
| 4 | 只有一台数据 | vLLM 指标只在 head 暴露；.186 up、.188 down（B 组未跑）、worker 不暴露 → 仅 A 组 | 现状合理 |
| 5 | 推理延迟多条曲线无区别 | **确认 bug**：面板 target 重复粘贴（TTFT×2、TPOT×2、ITL×3 相同 expr） | 需修复 |
| 6 | 投机解码平均接受长度无数据 | **确认 bug**：用了不存在的 `vllm:spec_decode_num_emitted_tokens_total` | 需修复 |
| 7 | GPU 占用率 03/04 为 0 | **数据真实**：03 max=0、04 max=16/avg=0.3；原因=只跑 embed 且无流量 | 数据真实 |
| 8 | 内网网卡 RoCE 真实性 | **数据真实准确**：IB 计数器字节口径；01↔02 实测 653MB/s 吻合 | 数据真实（外网面板有标签误） |

---

## 逐项证据

### 1. 面板数据少了很多

- Prometheus TSDB 连续：`up{node-58}` 最近 72h 每小时采样全 = 1，**无采集中断**（数据卷持久化正常）
- **主因**：vllm job 配了 `.186:8001`（up）和 `.188:8001`（**down**）——B 组 TP2 已停（8/8 benchmark 后，当前仅 embed），故 vLLM 相关面板从"两台"变"一台"
- 监控栈容器 3 小时前重启过（Up 3 hours），重启分钟级 gap 在小时步长下不可见

### 2. CPU 占用率曲线不全

`dcgx:cpu_util_percent`（recording rule）最近 1h：

| machine | 样本数 | min | max |
|---------|-------|-----|-----|
| node01 | 61 | 4.5% | 29.9% |
| node01 | 61 | 3.6% | 25.2% |
| node01 | 61 | 0.2% | 2.7% |
| node01 | 61 | 0.2% | 2.7% |

**四台采集齐全**。03/04 曲线贴 0 轴（值 <3%）在 Grafana 默认 Y 轴下看似"没有曲线"，建议改 Y 轴 min=0 或开启 soft min，或调面板 min-height。

### 3. KV cache 水位单位（确认 bug）

- vLLM metrics 原文：`# HELP vllm:kv_cache_usage_perc KV-cache usage. **1 means 100 percent usage**.`
- 实测值 0.3726（A 组空闲时 0）——面板直接 `vllm:kv_cache_usage_perc` 未乘 100 → 显示 0.37 而非 37%
- **修复**：`vllm:kv_cache_usage_perc * 100`，或面板 Unit 设 percent

### 4. 只有一台数据

- vLLM metrics 仅由 **head 节点**暴露（.186:8001 = A 组 head ✅ up）
- B 组 head（.188:8001）down（TP2 未跑）；worker 节点（.187/.189）不暴露 vLLM metrics
- 故 KV/延迟/spec decode/吞吐等 vLLM 面板仅 node01 一条——**符合架构现状**，B 组恢复后自动两台

### 5. 推理延迟多条曲线没有区别（确认 bug）

面板 json 中重复 target：
- TTFT 面板：**2 个相同 expr**（`histogram_quantile(0.5, sum by (le, node, model_name)...` 重复）
- TPOT 面板：2 个相同 expr
- ITL 面板：**3 个相同 expr**

→ 同系列重复绘制，多 legend 数值完全相同。**修复：删重复 target，每个面板只留 1 个。**

### 6. 投机解码平均接受长度无数据（确认 bug）

- 面板 expr 用 `vllm:spec_decode_num_emitted_tokens_total` —— **该指标在 vLLM 中不存在**（metrics 列表无 emitted，只有 accepted/drafts/draft_tokens）→ 除法分母 clamp 后为 0.001 常量 → 结果 NoData
- 数据实际存在：accepted=22,499 / drafts=7,234（A 组有流量时）
- **修复**：平均接受长度 = accepted / drafts：
  ```
  sum by (node) (rate(vllm:spec_decode_num_accepted_tokens_total{job="vllm"}[5m]))
    / clamp_min(sum by (node) (rate(vllm:spec_decode_num_drafts_total{job="vllm"}[5m])), 0.001)
  ```

### 7. GPU 占用率 03/04 为 0（核实）

最近 1h `dcgx:gpu_util_percent`：

| machine | max | avg | 说明 |
|---------|-----|-----|------|
| node01 | 96% | 92.8% | A 组 TP2 LLM 在跑 |
| node01 | 96% | 91.3% | A 组 TP2 worker |
| **node01** | **0%** | **0.0%** | **只跑 embed，无请求 → GPU 空闲** |
| **node01** | **16%** | **0.3%** | **只跑 embed，近空闲** |

**03/04 GPU util 为 0 是真实数据**：它们不跑 LLM（B 组 TP2 已停），仅承担 embed（Qwen3-Embedding-0.6B），embed 当前几乎无流量 → GPU 利用率≈0。瞬时偶发（03 曾短暂 80%）来自 embed 处理请求瞬间。非监控故障；若 embed 有持续流量会看到波动。

### 8. 内网网卡四个图表（RoCE 真实性）

- 指标：`node_infiniband_port_data_received_bytes_total`（node_exporter 从 `/sys/class/infiniband` 采集，**字节**口径）
- 最近 1h `rocep1s0f1` 收速率：

| machine | 平均 | 峰值 |
|---------|------|------|
| node01 | 653.6 MB/s | 728.7 MB/s |
| node01 | 653.9 MB/s | 730.8 MB/s |
| node01 | 0 | 0 |
| node01 | 0 | 0 |

- **数据真实准确**：01↔02 链路实际 ~653MB/s ≈ 5.2Gbps（200G 的 2.6%），与 A 组 TP2 推理时 TP 通信量吻合
- 口0（rocep1s0f0/roceP2p1s0f0）无数据 = 该口未接线/未使用；03/04 为 0 = B 组未跑 LLM、无跨机通信（真实）
- **⚠️ 附带问题**：「外网速率」面板 device 正则 `enp1s0f0np0|enp1s0f1np1|enP2p1s0f0np0|enP2p1s0f1np1|...` **把 RoCE 数据面网卡也计入"外网"**——标签误导，RoCE 流量被归入外网。建议正则只保留 `enP7s7|wlP9s9`（管理口）

---

## 修复清单（按优先级）

| 优先级 | 面板 | 修复 |
|--------|------|------|
| P0 | KV Cache 水位 | expr 加 `* 100`（两处：集群+实时面板） |
| P0 | 投机解码平均接受长度 | 分母改用 `vllm:spec_decode_num_drafts_total`（emitted 不存在） |
| P1 | 推理延迟（TTFT/TPOT/ITL） | 删除重复 target（TTFT×2/TPOT×2/ITL×3 → 各留 1） |
| P1 | 外网速率 | device 正则剔除 RoCE 网卡（只留 enP7s7/wlP9s9） |
| P2 | CPU 面板 | Y 轴 min=0，避免 03/04 低值贴 0 轴看似缺失 |

## 数据真实性说明（无需修复）

- **GPU 03/04 = 0**：真实（只跑 embed 且无流量），B 组 LLM 恢复或 embed 有流量后会出现
- **只有一台数据**：真实（vLLM 指标仅 head 暴露，B 组 .188 未运行）
- **内网网卡 RoCE**：真实准确（IB 计数器字节口径，01↔02 实际 653MB/s 吻合）
- **CPU 四台齐全**：真实（03/04 值低仅因负载低）

## 附：环境事实

- Prometheus targets：node/dcgm 四台全 up；vllm .186 up / .188 down（B 组未跑）
- TSDB：数据连续 72h（`up` 全 1），监控栈 3h 前重启，无数据丢失
- 数据源：Prometheus（provisioning），Grafana admin 凭据在容器 env

---

## 修复记录（2026-08-09 20:35 已执行）

通过 Grafana API 更新 vllm-realtime 面板，8 项变更全部生效并验证：

| # | 面板 | 修复 | 验证 |
|---|------|------|------|
| 1 | CPU 占用率 | Y 轴 `min=0`（03/04 低值可见）；**根因：Prometheus 08-09 08:31 UTC 重启，recording rule 对 01/03/04 从重启后才产生 series（此前仅 02）→ 看历史时段仅 1 条线**。当前数据 4 条正常 | 即时查询 4 series；Grafana 模拟查询 4 frames |
| 2 | KV Cache 水位 | expr 加 `* 100`（0-1 → 百分比） | 模拟查询正常（当前空闲=0） |
| 3 | 投机解码·平均接受长度 | 分母 `emitted`（不存在）→ `drafts` | 完整 expr 确认 `accepted / drafts` |
| 4 | TTFT / TPOT / ITL | 删除重复 target（各留 1 个） | 面板 json 确认各 1 个 expr |
| 5 | 外网速率 | device 正则剔除 RoCE 网卡（仅留 `enP7s7|wlP9s9`） | 收发两条 expr 确认 |

**CPU 面板说明**：数据层始终 4 条正常；"只有 1 条线"是 recording rule 历史 gap（Prometheus 重启前 01/03/04 无 series）+ 浏览器缓存所致。已通过面板更新触发重载，请**硬刷新浏览器（Ctrl+Shift+R）**后确认；若需查看 16:31 前的历史，01/03/04 数据不可回填（属历史遗留，未来数据正常）。

### 预填充吞吐"单并发/总吞吐"核查（20:40 追加）

**结论：没有反，但 B 公式有真实 bug，已修复。**

- **A 总预填充** = `rate(prompt_tokens_total)`（负载视角：墙钟 token 速率）——正确
- **B 单并发** 旧公式 = `(prompt/req) / 平均TTFT`——**两个 bug**：
  1. TTFT 含排队时间 → 高并发时 B 偏低（非纯 prefill）
  2. 无请求时 TTFT count≈0 → `clamp_min(...,0.001)` 除零 → 历史出现 **inf** 和"单>总"异常（实测 10:24 A=412/B=920、10:32 A=426/B=769、10:36 B=inf）
- **低并发时 A≈B 是正常现象**（当前 0.128 req/s 基本串行，单请求吞吐=总吞吐）
- **修复**：B 改用 vLLM 专用 `request_prefill_time_seconds`（纯 prefill 时间 histogram）：
  ```
  sum by (node) (rate(vllm:prompt_tokens_total{job="vllm"}[5m]))
    / clamp_min(sum by (node) (rate(vllm:request_prefill_time_seconds_sum{job="vllm"}[5m])), 0.001)
  ```
  验证：历史 6h inf=0，公式为"每 prefill 秒处理的 token"（引擎视角）
- **剩余说明**：prefix cache 命中时 prefill 时间近零 → B 出现虚高尖峰（真实现象，非 bug）；如影响可读性，面板加 Y 轴上限
