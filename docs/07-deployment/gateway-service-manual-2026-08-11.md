# DGX Spark 网关服务使用手册 + Key 清单（精简版）

**版本**：2026-08-11（含 502 修复与 embed 上游修正后实测状态）
**⚠️ 敏感**：含明文 Key，仅限授权人员，禁止外发

---

## 1. 服务总览

| 服务 | 地址 | 用途 |
|------|------|------|
| **LiteLLM 网关** | `http://<NODE_IP>:4000` | OpenAI 兼容 chat / embeddings / models，多 Key 管理 + 限流 |
| **自建网关** | `http://<NODE_IP>:8003` | chat / completions / **responses（保思考链）** / embeddings / models |
| vLLM 引擎（内部） | `http://<NODE_IP>:8001` | 推理引擎，**业务勿直连** |

> 所有端点均已实测连通（HTTP 200）。旧文档中的 `<NODE_IP>` / `<MGMT_OCTET>` 为节点改名前的地址，已废弃。

---

## 2. Key 清单（全部实测有效）

### LiteLLM 网关（4000）

| Key | 值 | 授权模型 | 限流 |
|-----|-----|----------|------|
| **Master（管理，勿外泄）** | `sk-litellm-master-b9158f0b67dec7d9e395d54cb462afe2` | Admin 全部 | 无 |
| **Chat** | `sk-U_cIbL63-5c27rayJO3S6w` | `local-v4-flash` / `deepseek-v4-flash` | rpm 300 / tpm 50k |
| **Embedding** | `<API_KEY>` | `local-embedding` | rpm 300 / tpm 100k |
| **Prob**（结构化高吞吐） | `<API_KEY>` | `dspark-prob` | 未设 |
| **Greedy**（散文/确定性） | `<API_KEY>_C8xuN9rqHhg` | `dspark-greedy` | 未设 |

### 自建网关（8003）

| Key | 值 | 权限 |
|-----|-----|------|
| **客户端** | `<API_KEY>-64b0374c6f2840fe` | 全模型 passthrough |

### 内部 Key（勿外泄，仅运维）

| Key | 值 | 用途 |
|-----|-----|------|
| **vLLM 内部** | `<API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa` | vLLM 8001 鉴权，仅网关转发用 |

---

## 3. 可用模型

| 网关 | 模型名 | 后端 |
|------|--------|------|
| 4000 / 8003 | `local-v4-flash`、`deepseek-v4-flash` | vLLM deepseek-v4-flash-0731（<MGMT_OCTET>:8001） |
| 4000 | `dspark-prob`（temp 0.7 默认）/ `dspark-greedy`（temp 0.1 默认） | 同上游，per-key 模板 |
| 4000 | `local-embedding` | embed 池 <MGMT_OCTET>:8022 + <MGMT_OCTET>:8022（双机） |
| 8003 | `Qwen3-Embedding-0.6B` | 直连 <MGMT_OCTET>:8022 |
| 8003 | `deepseek-v4-flash-0731` | vLLM 实际 served 名 |

---

## 4. 使用要点（违反即出错）

1. **model 名必须写全**：4000 上 `dspark-prob` / `dspark-greedy` 不能简写为 `dspark`（401）；8003 客户端需用 `deepseek-v4-flash-0731` 或 `local-v4-flash`。
2. **Prob Key 必须传 `temperature>0.1`**（建议 0.7）：否则静默回退 greedy，失去 +20~47% 吞吐增益。
3. **Key 权限隔离**：Chat Key 不能调 embeddings，Embedding Key 不能调 chat（401）。
4. **embed 模型名区分**：走 4000 用 `local-embedding`；走 8003 用 `Qwen3-Embedding-0.6B`（透明透传，不改名）。
5. **思考链**：需要 `reasoning` 内容的客户端走 8003 `/v1/responses`（4000 的 responses 思考链有回归）。

---

## 5. 快速示例

```bash
# 4000 chat
curl http://<NODE_IP>:4000/v1/chat/completions \
  -H "Authorization: Bearer <BEARER>" \
  -H "Content-Type: application/json" \
  -d '{"model": "local-v4-flash", "messages": [{"role": "user", "content": "2+2=?"}]}'

# 4000 embedding
curl http://<NODE_IP>:4000/v1/embeddings \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "local-embedding", "input": "你好"}'

# 4000 prob（结构化高吞吐）
curl http://<NODE_IP>:4000/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "dspark-prob", "temperature": 0.7, "messages": [{"role": "user", "content": "write python to parse json"}]}'

# 8003 responses（保思考链）
curl http://<NODE_IP>:8003/v1/responses \
  -H "Authorization: Bearer <API_KEY>-64b0374c6f2840fe" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash-0731", "input": "2+2=?"}'

# 8003 embedding（注意模型名）
curl http://<NODE_IP>:8003/v1/embeddings \
  -H "Authorization: Bearer <API_KEY>-64b0374c6f2840fe" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "你好"}'
```

---

## 6. 网关选择指引

| 场景 | 走哪个 |
|------|--------|
| 标准 OpenAI 兼容 chat / embeddings | **4000**（多 Key 管控 + 限流 + 用量统计） |
| 结构化负载高吞吐 | **4000 `dspark-prob`**（temp 0.7） |
| 散文 / 确定性输出 | **4000 `dspark-greedy`** |
| 需要思考链（responses） | **8003** |
| 故障排查 | 4000 用 `GET /v1/models` 拉模型名；401=Key 错/越权，404=模型名错，429=限流 |

---

> 参考：旧版完整手册 `litellm-api-key-manual-2026-08-05.md`（地址已过时）；v18-server 配置 `/opt/aicad/.env.prod`（已指向 <MGMT_OCTET>:4000/8003）。
