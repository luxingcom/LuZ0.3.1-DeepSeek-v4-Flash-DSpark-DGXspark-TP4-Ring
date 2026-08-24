# litellm 网关吞吐优化 P0/P1/P2 执行报告

**日期**：2026-08-09
**工作流**：性能优化落地（限流解除 + 多 worker 扩容 + 业务 batch 化）
**参与成员**：主理人（实施与验证）/ Tessa（口径判定）
**前置报告**：analysis-litellm-throughput-optimization-2026-08-09.md

---

## 📌 TL;DR（执行摘要）

| 项 | 状态 | 结果 |
|----|------|------|
| **P0 解除 embed key 限流** | ✅ 完成并验证 | `embedding` key rpm 300→无限制；业务 key 压测 480 请求 0 429 |
| **P1 业务侧 batch 化** | ✅ 仓库侧完成（生产 Rust 待源码） | Python `embed_batch` 真批量（一次请求多条），36 测试全绿，真远程协议验证通过 |
| **P2 litellm 多 worker** | ✅ 完成并验证 | `--num_workers 2`：c16 362→491（+36%）、c32 377→651（+73%）；c64+ 受上游 max-num-seqs=32 排队限制 |
| chat-v4-flash 风控 | ✅ 保留 | rpm_limit=300 未动（用户确认风控有效） |

---

## 1. P0：解除 embedding key rpm 限流 ✅

### 执行

```sql
UPDATE "LiteLLM_VerificationToken" SET rpm_limit = NULL, tpm_limit = NULL
WHERE key_alias = 'embedding';
```

### 验证（业务生产 key <API_KEY>）

- 8 并发 × 60 = **480 请求 / 1.8s 全 200，0 个 429**（此前 300 req/min 必触发）
- 限流后表状态：`embedding` rpm/tpm = NULL；`chat-v4-flash` 保持 300/50000（风控保留）

### 附带发现（生产链路确认）

- 生产 v18-server 实际加载 `.env.prod`：`LOCAL_EMBED_BASE_URL=http://<NODE_IP>:4000/v1`（litellm 网关）✓
- 仓库 `.env` 指向 `<NODE_IP>:33345`（旧 LM Studio 配置）——**非生产加载文件，无需处理**
- 踩坑记录：`<NODE_IP>` 是镜像仓库地址，.58 节点管理 IP 为 `.187`（网关为 .187:4000）

---

## 2. P1：业务侧 batch 化 ✅（仓库侧，生产 Rust 待源码）

### 改造内容（AICAD/backend 仓库）

**`services/embedding_service.py`**
- 新增 `_embed_via_remote_batch()`：一次 POST 携带多条文本（OpenAI 兼容批量协议），含维度截断/补零、L2 归一化、顺序对齐校验、超时按条数放大
- 重构 `embed_batch()`：按 `batch_size=64` 分组 → 每组一次批量请求（并发 4 组）→ 批量失败时逐条走原 3 层 fallback 链；空文本先行短路；返回顺序与输入严格对齐

**`kg/retrievers/embedding_indexer.py`**
- `index_nodes_batch()`：从逐条 `embed_text` 改为整块 `embed_batch()`（每 64 条一批），RAG 索引构建吞吐提升约 4.6×（网关实测 batch=16 时 239→1100 条/s）

### 验证

| 层级 | 结果 |
|------|------|
| 本地单测（fallback/空文本/归一化/确定性） | ✅ 全通过 |
| pytest 套件（含 2 个新增批量测试） | ✅ **36 passed** |
| 真远程协议（.60 → .187:4000，3 条一次请求） | ✅ HTTP 200、dim=1024、顺序对齐 |

### ⚠️ 生产落地限制

- 生产 embed 调用方是 **v18-server（Rust 编译二进制）**，其内嵌 `/embeddings` 调用逻辑（strings 确认：LOCAL_EMBED_BASE_URL / shared-embeddings cache），**不走 Python embedding_service.py**
- v18 Rust 完整源码不在本仓库（仅 .workbuddy/tmp 快照片段，文件名乱码）→ **生产 batch 化需 Rust 源码仓库 + 重新编译 v18-server 后发布**（已列入行动项）
- 本次 Python 改造为仓库侧就绪状态，待 Rust 侧对齐或后续 Python 化服务使用

---

## 3. P2：litellm 多 worker 扩容 ✅

### 执行

```bash
docker rm -f litellm-proxy
docker run -d --name litellm-proxy --restart unless-stopped --network host \
  -v /home/<USER>/litellm/config.yaml:/app/config.yaml \
  -e DATABASE_URL=postgresql://litellm:litellm_local_pw_9x@127.0.0.1:5432/litellm \
  -e LITELLM_UPSTREAM_KEY=<API_KEY>-... \
  ghcr.io/berriai/litellm:v1.83.7-stable \
  --config /app/config.yaml --port 4000 --num_workers 2
```

### 效果（经 litellm 双机，master_key，~300 字文本）

| conc | 单 worker（改造前） | 2 worker（改造后） | 提升 |
|------|--------------------|--------------------|------|
| 8 | 260-310 | 293 | +~10% |
| 16 | 362-383 | **491** | **+36%** |
| 32 | 377 | **651** | **+73%** |
| 64 | 402 | 284（p95 724ms） | 退化 |
| 128 | — | 133（p95 2.9s） | 退化 |

- **最优工作点 c32 = 651 req/s**（+73%）
- c64+ 退化原因：**上游 embed `--max-num-seqs=32` 排队饱和**（非网关问题）——并发超过 32 时 vLLM 队列堆积
- 多 worker 验证：master + 2 uvicorn worker 进程正常、业务 key 200、健康检查正常

### 后续建议

- 若需 >650 req/s：上游每机 `--max-num-seqs` 提到 64（03/04 embed 容器参数，维护窗口改）+ litellm `--num_workers 4` 双管齐下
- 当前业务水位（~300 rpm 时代遗留）下 2 worker 余量充足

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 备注 |
|---|------|---------|--------|------|
| 1 | 获取 v18 Rust 源码仓库，将 batch 化改造对齐到 v18-server 并重新编译发布 | BE | P1 | Python 改造可作参考实现 |
| 2 | 观察 24h：多 worker 稳定性、业务 embed 延迟、PG 无异常 | SRE | P2 | — |
| 3 | 若业务量 >650 req/s：embed 容器 max-num-seqs 32→64 + litellm worker 2→4 | SRE | P3 | 维护窗口 |
| 4 | 历史 429 风暴（14452 条）复盘：业务侧是否有重试风暴逻辑需调整 | Tessa | P3 | 限流已解除 |

---

## ⚠️ 局限

- P1 生产落地受限于 Rust 源码缺失（Python 改造已就绪 + 验证）
- c64+ 数据为上游排队形态，不代表网关能力（网关 2 worker 理论可到 ~1300）
- 测速客户端与网关同机，真实业务跨机延迟略高

---

## 📚 变更清单

| 文件 | 变更 |
|------|------|
| litellm-pg `LiteLLM_VerificationToken` | embedding key rpm/tpm → NULL |
| .58 litellm-proxy 容器 | 启动参数 + `--num_workers 2`（config.yaml.bak-prenumworkers 备份） |
| AICAD/backend/services/embedding_service.py | 新增 `_embed_via_remote_batch` + 重构 `embed_batch`（batch_size=64） |
| AICAD/backend/kg/retrievers/embedding_indexer.py | `index_nodes_batch` 改批量调用 |
| AICAD/backend/tests/test_embedding_service.py | 新增 2 个批量测试 |

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
