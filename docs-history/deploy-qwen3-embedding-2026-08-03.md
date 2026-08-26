# Qwen3-Embedding-0.6B 本地部署交付报告

**日期**：2026-08-03
**工作流**：系统设计 + 部署前检查（工作流 2/4 组合）
**参与成员**：Archi（方案设计）/ Rex（模型核查与部署）/ Tessa（独立验收）/ 主理人（编排汇编）

---

## 📌 TL;DR（执行摘要）

- **Qwen3-Embedding-0.6B 已部署上线**（worker 58，CPU，8020 端口，ctx 8192，1024 维，OpenAI 兼容 `/v1/embeddings`），与生产 E-600k 共存零扰动
- **框架定案**：Python venv + sentence-transformers + FastAPI（否决 vLLM-Anemll 专用 fork 与 TEI——TEI 无 arm64 镜像，架构级排除）
- **认证双层**：8020 内部 key + 网关 8003 新增 `/v1/embeddings` 路由（客户 key 透传），与现有 API key 体系一致
- **验证双绿**：SRE 内部验证 7/7 + 泰莎独立客户视角验收 6/6（Go）——语义质量（cos 0.87 vs 0.28）、ctx 截断实证、性能 avg 124ms
- 严重度分布：🔴严重 0 / 🟠高 0 / 🟡中 0 / 🟢低 3（观察项：/v1/models 未列 embedding、usage 计数语义、模型名不校验）
- 阻塞 / 非阻塞：**非阻塞（Go）**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 Go（可交付） |
| 服务 | http://<NODE_IP>:8020/v1/embeddings（内部 key） |
| 客户路径 | http://<NODE_IP>:8003/v1/embeddings（客户 key） |
| 模型 | Qwen3-Embedding-0.6B，1024 维，ctx 8192（截断） |
| 验证 | SRE 7/7 + Tessa 6/6 全 PASS |
| 建议下一步 | 观察项处理（usage 语义/模型列表/名校验） |

---

## 🏗️ 架构与选型（Archi）

- **否决 vLLM-Anemll 同镜像**：vLLM 0.25.2.dev0 是 DeepSeek V4 专用 fork（sm_121a wrapper + nvfp4），容器内无 python、model registry 不注册 Qwen3Embedding、同容器双模型抢统一内存
- **否决 TEI**：API 非 OpenAI 兼容（需代理层）；**官方镜像无 arm64**（manifest 仅 amd64 + unknown 占位），GB10 aarch64 无法直接跑——架构级排除
- **定案**：Python venv + **sentence-transformers**（ModelScope repo 原生 ST 格式：last-token pooling + L2 normalize + query prompt 开箱即用）+ FastAPI 包装；**CPU 模式**（fp16 ~1.2GB + 峰值 <3GB，15Gi 余量充足）；不容器化
- **位置**：worker 58（8003 网关同机 → 网关→8020 loopback 零跨机 FW 改动）；端口 8020（实测空闲）

## 🚀 部署实施（Rex）

| 项 | 详情 |
|---|---|
| 模型 | ModelScope 下载（13 文件+1 目录，~1.21GiB，21MB/s）→ /home/<USER>/models/Qwen3-Embedding-0.6B/ |
| 环境 | embed-venv（python3.12 + torch 2.13.0+cpu aarch64 + transformers 5.14.1 + sentence-transformers 5.6.1） |
| 服务 | /home/<USER>/embed-svc/main.py：/v1/embeddings（OpenAI 兼容、1024 维、ctx 8192 截断、batch ≤32）+ /health；内部 key 认证 |
| 网关 | main.py v1.2.0：/v1/embeddings + /embeddings 双挂载（客户 key → 127.0.0.1:8020 注入内部 key）；**关键修复：start_gateway.sh 的 pkill 通配会误杀 embedding 进程（-venv 匹配），改绝对路径精确匹配** |
| systemd | 用户级 embed-qwen3.service（Restart=always）+ linger=yes（免 sudo 等效自启，重启自恢复已验证） |
| 无扰动 | embedding 仅 CPU + ~523MB；E-600k 容器未重启，8001 经网关 /v1/models 200 |

## ✅ 验证结果

### SRE 内部验证（7/7）
单条 1024 维 ✅ / cosine sanity（0.8725 vs 0.2797）✅ / 认证三态 ✅ / 超长截断 ✅ / batch 5 条 ✅ / 33 条 400（超限保护）✅ / OpenAI 兼容格式 ✅

### Tessa 独立客户视角验收（6/6，Go）
| 用例 | 结果 |
|---|---|
| TC1 内部 8020 | PASS（单条/batch 1024 维；无/错 key 401） |
| TC2 客户 8003 | PASS（透传 200；无 key 401） |
| TC3 语义质量 | PASS（cos 0.87 > 0.28；跨语言 cos(苹果,apple)=0.92） |
| TC4 ctx 边界 | PASS（9882/20902 token 均 200 截断，耗时证截断至 8192） |
| TC5 回归 | PASS（/v1/responses 200 + /v1/models 200，E-600k 未破坏） |
| TC6 性能 | PASS（5 次单条 avg 124ms / max 149ms） |

## 📇 交付卡片（客户调用）

```python
from openai import OpenAI
client = OpenAI(api_key="<API_KEY>-64b0374c6f2840fe", base_url="http://<NODE_IP>:8003")
resp = client.embeddings.create(model="Qwen3-Embedding-0.6B", input="你好")
print(len(resp.data[0].embedding))  # 1024
```
- 模型名：`Qwen3-Embedding-0.6B`；维度 1024；ctx 上限 8192（超长自动截断）；batch ≤32 条
- 内部直连（运维）：`http://<NODE_IP>:8020/v1/embeddings` + 内部 key

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **观察项 2：usage.prompt_tokens 语义**（超长输入报原始计数 20902 非 8192——若涉计费需改截断后计数；当前不计费可接受） | Rex | P2 | 计费前 |
| 2 | 观察项 1/3：/v1/models 列出 embedding 模型 + 模型名校验（可选，客户体验增强） | Rex | P2 | 下轮迭代 |
| 3 | 可选优化：GPU 模式（torch cu130 已可测，灰度验证 E-600k 稳定后启用） | Rex | P3 | 评估后 |
| 4 | 更新 PARAMS.md（embedding 服务拓扑） | Docu | P2 | 1 周内 |

## ⚠️ 待完善 / 已知局限

- /v1/models 未列出 embedding 模型（客户经文档获知模型名）；模型名不校验（传任意名仍 200，OpenAI 兼容宽松行为）
- usage.prompt_tokens 超长时报原始计数（实际已截断至 8192 处理）——计费场景需修正语义
- 单机 CPU 服务（worker 58）：高并发 embedding 需求需评估（当前单条 avg 124ms 良好）；GPU 模式为可选优化
- systemd 为用户级（免 sudo 等效）；如需系统级需 root 执行（命令已备）

## 📚 数据来源 & 成员产出索引

- Archi：方案 v1.0/v1.1（选型实证：Anemll fork 无 python、TEI 无 arm64、ST 格式模型）、接口/认证/运维设计
- Rex：模型源核查（ModelScope 首选/hf-mirror 备选/HF 不可达）、部署实施（时间线 02:50-03:01、验证 7/7、网关 v1.2.0 + pkill 修复、systemd+linger、无扰动确认）、归档 hardened/live/embedding/
- Tessa：独立验收 6/6（TC1-6 客户视角、观察项 3 条）
- 主理人：编排、汇编

---

> 本报告由工程保障团队 AI 协作生成，关键决策（usage 语义、GPU 模式）请由人类工程负责人复核。
