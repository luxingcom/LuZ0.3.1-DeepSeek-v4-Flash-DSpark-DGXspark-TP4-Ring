# embed 4 机 HA 改造与 comfyui 内存限流 交付报告

**日期**：2026-08-07
**工作流**：工作流 4（部署前检查/变更实施）——embed 高可用 + 资源治理
**参与成员**：Archi（方案 ADR-embed-002）/ Rex（检查清单）/ 主理人（实施与验证）

---

## 📌 TL;DR（执行摘要）

- **任务一（embed 与 vLLM 共用镜像 + 16G 预算）**：验证通过并全量落地——四机 embed 统一改用 vLLM 镜像（免 20G 专用镜像部署），内存预算 4.87G → **12G（.58）/ 18G（新机）**，CUDA OOM 风暴根除；litellm 网关 4 deployment 负载均衡 + 故障自动切换
- **任务二（comfyui 内存上限）**：已设置 88G 上限（swap 禁用），热更新未中断
- 四机 embed 全部 Up + /health 200 + dim 1024；网关 12 连发全 200；**故障注入通过**（停 .55 → 5 连发全 200 → 自动回纳）
- 无阻塞项；遗留 2 项（分摊调优、超长输入策略）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（4 机 HA 已上线） |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条（见行动清单） |
| 建议下一步 | 观察 24h 稳定性；分摊调优（least-busy）；旧 anemll 镜像观察期后清理 |

---

## 1. 任务一：embed 与 vLLM 共用镜像（ADR-embed-002）

### 1.1 决策
embed 从专用镜像 `embed-gpu:anemll-0.1.1-st5.6.1`（20G，EMBED_MEM_FRACTION=0.04≈4.87G → CUDA OOM）迁移至 **`vllm-gb10:0.26.1-cu132-sha-fa87aea5`**（18.9G 四机/registry 已有），`--gpu-memory-utilization` 控制预算。

### 1.2 部署规格（四机一致，仅预算/挂载源差异）

```bash
docker run -d --name embed-qwen3-vllm --restart unless-stopped --gpus all -p 8020:8020 \
  -v <权重路径>:/models/Qwen3-Embedding-0.6B:ro \
  --entrypoint vllm <NODE_IP>:5000/vllm-gb10:0.26.1-cu132-sha-fa87aea5 \
  serve /models/Qwen3-Embedding-0.6B --port 8020 --host 0.0.0.0 \
  --gpu-memory-utilization {0.15 新机 | 0.10 .58} --max-model-len 8192 --max-num-seqs 32 \
  [--enforce-eager .58] --served-model-name Qwen3-Embedding-0.6B \
  --api-key <API_KEY>-...
```

| 节点 | 状态 | 预算 | 权重挂载源 |
|------|------|------|-----------|
| .55 | ✅ Up / health 200 / dim 1024 | 0.15 ≈ 18G | <MODELS_DIR> |
| .58 | ✅ Up / health 200 / dim 1024（已从 anemll 迁移） | 0.10 ≈ 12G + enforce-eager | /home/<USER>/models |
| .59 | ✅ Up / health 200 / dim 1024 | 0.15 ≈ 18G | <MODELS_DIR> |
| .60 | ✅ Up / health 200 / dim 1024 | 0.15 ≈ 18G | <MODELS_DIR>（rsync 自 .58） |

### 1.3 litellm 网关（4 deployment LB）

- config.yaml：local-embedding → 4 deployment（api_base .55/.58/.59/.60:8020/v1，model: `hosted_vllm/Qwen3-Embedding-0.6B`，api_key 沿用 LITELLM_UPSTREAM_KEY）
- router_settings：`routing_strategy: simple-shuffle`（该版本无 round-robin）、allowed_fails 2、cooldown 30s、retries 2
- **验证**：/v1/models 含 local-embedding；12 连发全 200；**故障注入**：停 .55 → 5 连发全 200 → 恢复自动回纳

### 1.4 OOM 根因修复闭环

原链路：AICAD 业务 → litellm → .58 embed（4.87G 预算）→ 336MB 激活超预算 → CUDA OOM → 500 → 23 次/分钟重试风暴 → CPU 峰值。现预算 12-18G + 4 机 LB，根除。

## 2. 任务二：comfyui 内存限流

`docker update --memory 88g --memory-swap 88g comfyui-h3`（热更新未中断，swap 禁用）——单机内存不超过 88G，为 .58 后端栈与 embed 留出 ~33G。

## 3. 实施中修复的关键问题

| # | 问题 | 修复 |
|---|------|------|
| 1 | fork 镜像 serve 不支持 `--task embed` | 依赖 config.json 自动推断（验证 dim 1024） |
| 2 | 镜像 ENTRYPOINT=[] | 显式 `--entrypoint vllm serve <模型位置参数>` |
| 3 | litellm `routing_strategy: round-robin` 启动失败 | 改 `simple-shuffle`（版本有效值列表） |
| 4 | litellm 上游 model `openai/` 前缀 → vLLM 404 | 改 `hosted_vllm/`（匹配 served-model-name） |
| 5 | .58 挂载源 <MODELS_DIR> 不存在 → 容器崩溃循环 | 改用 /home/<USER>/models（权重源路径） |
| 6 | CRLF 行尾致 config 替换失败 | 转 LF 后处理 |
| 7 | .60 无 /data 权限 | sudo mkdir + chown 后 rsync 权重 1.2G |

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 观察 24h：四机 embed 稳定性、comfyui 无 OOM、litellm 路由正常 | SRE | P1 | 24h 后 |
| 2 | 分摊调优：simple-shuffle 下 .58 请求偏多（123 vs 47/58）→ 评估 least-busy | SRE | P2 | 1 周内 |
| 3 | 超长输入（>8192）行为对照测试并定截断策略（Gate B 遗留） | SRE | P2 | 1 周内 |
| 4 | 旧 anemll 镜像保留 1 周观察后清理；四机 embed 纳入 Grafana 健康面板 | SRE | P3 | 1 周后 |

---

## ⚠️ 待完善 / 已知局限

- 分摊偏差：simple-shuffle 随机洗牌，.58 实际请求数偏高（可能受业务侧本地调用路径影响），least-busy 可改善但 .58 有 comfyui 负载会自然少派单
- litellm 仍为 .58 单点（网关 SPOF，后续可迁 .60 或双网关——团队已记录另开 ticket）
- vLLM embedding 的 batch 语义（--max-num-seqs 32 对齐 anemll batch_max=32）与超长输入截断行为需实测确认
- .58 预算 12G 低于用户"16G"预期——因 comfyui 88G 限流后余量约束（.58 剩余 ~33G，12G embed + 13G 后端 + 余量），若需 16G 需 comfyui 降 80G（评估后可选）

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）**：ADR-embed-002 方案更新（部署规格/迁移 SOP/litellm 适配/预算/门禁）——消息回传全文
- **Rex（SRE）**：embed 4 机 HA 检查清单（A/B/C 门禁与命令）——消息回传全文
- **实测**：四机 health/dim 验证、网关 12 连发、故障注入（停 .55 → 全 200 → 回纳）、comfyui 88G cgroup 确认

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
