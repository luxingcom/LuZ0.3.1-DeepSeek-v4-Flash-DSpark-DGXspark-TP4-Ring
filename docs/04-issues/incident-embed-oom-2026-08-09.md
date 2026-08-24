# 03/04 内存告急（快 OOM）事故复盘 — embed KV cache 吃满 110GB

**日期**：2026-08-09
**工作流**：工作流 3（事故响应）
**参与成员**：Rex（SRE 判定依据：vLLM 日志铁证）/ 主理人执行修复

---

## 📌 TL;DR（执行摘要）

- 整体结论：**已修复**。03/04 内存从 used=120G/available=1G（快 OOM）恢复到 used=12G/available=109G，**每台释放约 108GB**；embed 服务全程功能正常。
- 严重度分布：🔴严重 1 项（OOM 风险）/ 🟡中 2 项（KV cache 配置失效、容器重建参数坑）
- 阻塞 / 非阻塞：已解除（修复完成，embed API 200 + 向量实测 OK）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 已修复（根因明确、复发预防已固化） |
| 阻塞项数量 | 0（已解除） |
| 关键行动项 | 3 条（见行动清单） |
| 建议下一步 | 监控 available 水位；后续 TP4/环网部署时用 `--kv-cache-memory` 而非 util env |

---

## 事故时间线

| 时间（UTC+8） | 事件 |
|--------------|------|
| 01:07 | 用户报告 03/04 快 OOM（内存 used=120G / available=1G） |
| 01:09 | 侦察：进程树无 LLM 权重进程、docker stats 仅 embed 3GB；meminfo 全字段 + NUMA + dmesg 定位 ~119GB 为"GPU 侧占用" |
| 01:12 | 容器内 torch 视角铁证：GPU used=120.9GiB / free=0.8GiB；vLLM 日志确认 KV cache 预分配 **110.2 GiB** |
| 01:15 | 根因确认：`VLLM_GPU_MEMORY_UTILIZATION` 在 anemll 0.2.1 失效 → 回落默认 util 0.92 → KV cache 110GB |
| 01:18 | 重建容器（--kv-cache-memory=4GB）：第 1 次漏 --gpus all 报 device 推断失败；第 2 次 CMD 带 serve（镜像 ENTRYPOINT 已含）报 unrecognized；第 3 次修正后成功 |
| 01:25 | 终验：内存释放至 used=12G/available=109G；embed API 200、向量 dim=1024；litellm 配置指向正确 |

## 影响范围

- **03/04 系统内存**：available 仅 1G，随时可能触发 OOM killer（可能误杀 embed/监控进程）
- **embed 服务**：本次修复中短暂中断（约 5 分钟重建窗口），litellm 池有 2 个后端逐台重建，未全断
- **其他节点**：01/02 不受影响

## SEV 评级

**SEV-2**（生产服务受影响 + 资源耗尽风险）：内存告急但未实际 OOM 杀进程；修复后无遗留影响。若未及时发现，OOM killer 可能杀 embed → SEV-1。

## 根因（5 Why）

1. **Why 内存 used=120G？** → vLLM KV cache 预分配 110.2 GiB
2. **Why KV cache 110GB？** → vLLM 回落默认 gpu_memory_utilization=0.92（121.6×0.92≈112GB）
3. **Why 回落默认？** → 容器 ENV `VLLM_GPU_MEMORY_UTILIZATION=0.15` 未被识别（日志 WARNING "Unknown vLLM environment variable detected"）
4. **Why 未被识别？** → anemll 0.2.1（NVIDIA 定制 vLLM）移除了该环境变量，改为 `--kv-cache-memory` CLI 参数
5. **Why 当时用了 util env？** → 部署时沿用旧版 vLLM 习惯，未验证 anemll 0.2.1 的参数兼容性 → **根因：版本差异导致配置失效，且无内存水位告警**

## 行动项

| # | 行动 | 状态 |
|---|------|------|
| 1 | 重建 embed 容器：`--kv-cache-memory=4294967296`（4GB）+ `--gpus all` + CMD 不带 serve | ✅ 已完成 |
| 2 | 固化启动参数铁律到 MEMORY.md（util env 失效 / ENTRYPOINT 含 serve / 必须 --gpus all） | ✅ 已完成 |
| 3 | 建议：为 03/04 配置内存水位告警（available < 20G 时告警），避免再次静默 OOM | ⏳ 待办 |

## 预防措施

1. **启动参数模板**（已验证）：
```bash
docker run -d --name anemll-embed-8022 --restart unless-stopped --gpus all \
  -p 8022:8022 -v <MODELS_DIR>:/models \
  <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \
  /models/Qwen3-Embedding-0.6B --host 0.0.0.0 --port 8022 \
  --served-model-name Qwen3-Embedding-0.6B --max-model-len 8192 \
  --max-num-seqs 32 --enforce-eager --trust-remote-code \
  --kv-cache-memory=4294967296
```
2. **版本兼容性验证**：任何 vLLM/anemll 版本升级后，先验证内存控制参数有效性（看启动日志是否有 Unknown env 警告）
3. **内存监控**：Prometheus 已有 node_exporter（node_memory_*），建议 Grafana 加 available 水位告警

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | embed 容器已用 --kv-cache-memory=4G 重建（03/04） | 主理人执行 | P0 | ✅ 已完 |
| 2 | 内存水位告警接入（available<20G 告警） | Rex/SRE | P1 | 待排期 |
| 3 | Runbook 更新 embed 启动参数（含铁律） | Docu | P2 | 随文档更新 |

## ⚠️ 待完善 / 已知局限

- litellm 池带 key 验证未完成（需 LITELLM_UPSTREAM_KEY/master_key，非故障）；后端 03/04 直连均正常，池配置正确
- KV cache 设 4GB 后，若业务 embed 并发升高可能需上调（观察 max-num-seqs=32 实际水位）

## 📚 数据来源 & 成员产出索引

- vLLM 启动日志（03/04）：`Unknown vLLM environment variable detected: VLLM_GPU_MEMORY_UTILIZATION` / `Desired GPU memory utilization is (0.92, 111.9 GiB)` / `Current kv cache memory in use is 110.2 GiB` —— 根因铁证
- torch 容器内视角：`GPU used=120.9GiB / free=0.8GiB`
- 修复后日志：`reserved 4.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config`
- 工作日志：2026-08-09.md；长期记忆 MEMORY.md（embed 启动参数铁律）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
