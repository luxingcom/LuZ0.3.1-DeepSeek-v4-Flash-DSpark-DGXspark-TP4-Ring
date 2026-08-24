# LiteLLM 分离 API Key 使用手册

**日期**：2026-08-05
**适用读者**：接入双轨网关的客户端开发者 / 集成工程师 / 运维
**适用环境**：LiteLLM 网关（4000） + 自建网关（8003）双轨运行
**⚠️ 敏感级别**：本文含明文 API Key，仅限内部授权人员访问，禁止外发

---

## 0. API 快速参考卡（复制即用）

> 按负载类型取用：选好场景 → 对照下表拿 `base_url` + `model` + `key` + `temperature`。

| 负载类型 | base_url | model | Key | temperature | 说明 |
|----------|----------|-------|-----|-------------|------|
| 结构化（code / json / tool）高吞吐 | `http://<NODE_IP>:4000/v1` | `dspark-prob` | Prob `<API_KEY>` | **0.7（必须 >0.1）** | probabilistic 采样 + 动态K，吞吐 +20~47% |
| 散文 / 长文 / 思考链 | `http://<NODE_IP>:4000/v1` | `dspark-greedy` | Greedy `<API_KEY>_C8xuN9rqHhg` | 0.1 | greedy 语义，质量优先 |
| 常规对话（原主模型） | `http://<NODE_IP>:4000/v1` | `local-v4-flash` / `deepseek-v4-flash` | Chat `sk-U_cIbL63-5c27rayJO3S6w` | 默认 | 聊天默认路径 |
| 向量嵌入 | `http://<NODE_IP>:4000/v1` | `local-embedding` | Embedding `<API_KEY>` | – | 1024 维 |
| Responses 思考链 | `http://<NODE_IP>:8003/v1` | 以 `/v1/models` 为准（如 `deepseek-v4-flash-0731`） | 8003 `<API_KEY>-64b0374c6f2840fe` | 默认 | 保留完整 reasoning（4000 的 responses 思考链有回归） |

**⚠️ 两条硬性约定（违反即 401 或失去加速）**：
1. **显式写 model 名**：LiteLLM 1.83.7 key 级 aliases 不生效——Prob / Greedy Key 必须写 `dspark-prob` / `dspark-greedy`，写 `dspark` 返回 401。
2. **Prob Key 必须显式传 `temperature>0.1`**（建议 0.7）：temp≈0 回退 greedy、失去加速；网关无 guardrail，由请求端强制（正式约定见 §2.5）。

**服务端口总览**：

| 服务 | 地址 | 用途 | 说明 |
|------|------|------|------|
| LiteLLM 网关（对外主网关） | `http://<NODE_IP>:4000` | OpenAI 兼容 chat / embeddings / models | 多 key 管理 + 限流 + 用量统计 |
| 自建网关 | `http://<NODE_IP>:8003` | chat / completions / **responses** / embeddings / models | passthrough，保留思考链 |
| vLLM 引擎（内部） | `http://<NODE_IP>:8001` | 推理引擎，仅网关转发 | 需内部 key `<API_KEY>-*`，**业务勿直连** |
| 嵌入服务（内部） | `:8020` | 嵌入，经网关转发 | 客户端一律走 4000 / 8003 |

---

## 1. Key 总览表

| Key 名称 | 值 | 用途 | 权限范围 | 限流 | 归属网关 |
|----------|-----|------|----------|------|----------|
| **Master Key（管理）** | `sk-litellm-master-b9158f0b67dec7d9e395d54cb462afe2` | LiteLLM 管理端（建 key / 轮换 / 查询），**勿外泄** | 全部（Admin API） | 无 | LiteLLM 4000 |
| **Chat Key** | `sk-U_cIbL63-5c27rayJO3S6w` | 对话生成（chat/completions） | 仅 `local-v4-flash` / `deepseek-v4-flash` | rpm 300 / tpm 50,000 | LiteLLM 4000 |
| **Embedding Key** | `<API_KEY>` | 向量生成（embeddings） | 仅 `local-embedding` | rpm 300 / tpm 100,000 | LiteLLM 4000 |
| **Prob Key（probabilistic）** | `<API_KEY>` | 结构化负载（code/json）采样，默认 temp 0.7，走 probabilistic 加速 | 仅 `dspark-prob` | 未设限流，可后续配置 | LiteLLM 4000 |
| **Greedy Key（greedy）** | `<API_KEY>_C8xuN9rqHhg` | 散文/思考链，默认 temp 0.1（≈greedy） | 仅 `dspark-greedy` | 未设限流，可后续配置 | LiteLLM 4000 |
| **客户端 Key（8003）** | `<API_KEY>-64b0374c6f2840fe` | 自建网关全量访问（chat/completions、responses、embeddings、models） | 全模型（passthrough） | 未配置（引擎默认） | 自建 8003 |
| **内部 Key（勿外泄）** | `<API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa` | vLLM 8001 引擎鉴权，仅网关转发用 | vLLM 引擎内部 | 无 | 引擎直连（不经网关） |

> **权限隔离已验证**：Chat Key 调用 embeddings 端点返回 401（LiteLLM 按 key 限模型隔离生效）。

---

## 2. LiteLLM 4000 使用说明

### 2.1 接入信息

| 项 | 值 |
|----|-----|
| base_url | `http://<NODE_IP>:4000` |
| SDK base_url | `http://<NODE_IP>:4000/v1` |
| 鉴权方式 | HTTP Header `Authorization: Bearer <key>` |
| 可用模型 | `local-v4-flash`（→ deepseek-v4-flash-0731 @ vLLM 8001）、`deepseek-v4-flash`（别名，同上）、`dspark-prob`（per-key 模板，temp 0.7）、`dspark-greedy`（per-key 模板，temp 0.1）、`local-embedding`（→ 8020） |
| 已支持端点 | `/v1/models`、`/v1/chat/completions`、`/v1/embeddings`、`/v1/responses`（注意：responses 路径思考链有回归，见 §4） |

> 模型名以 `GET /v1/models` 实际返回为准（使用任一非 Master 业务 key 即可）。

### 2.2 Chat 请求示例

**curl**
```bash
curl http://<NODE_IP>:4000/v1/chat/completions \
  -H "Authorization: Bearer <BEARER>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-v4-flash",
    "messages": [{"role": "user", "content": "2+2=?"}],
    "stream": false
  }'
```

**Python（openai SDK）**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<NODE_IP>:4000/v1",
    api_key="<API_KEY>",   # Chat Key
)

resp = client.chat.completions.create(
    model="local-v4-flash",                 # 或用别名 deepseek-v4-flash
    messages=[{"role": "user", "content": "2+2=?"}],
)
print(resp.choices[0].message.content)
```

### 2.3 Embedding 请求示例

**curl**
```bash
curl http://<NODE_IP>:4000/v1/embeddings \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "local-embedding", "input": "你好"}'
```

**Python（openai SDK）**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<NODE_IP>:4000/v1",
    api_key="<API_KEY>",    # Embedding Key
)

resp = client.embeddings.create(model="local-embedding", input="你好")
print(len(resp.data[0].embedding))          # 1024 维
```

> ⚠️ Chat Key 与 Embedding Key 权限互相隔离：**不要**用 Chat Key 调 embeddings 端点、也不要用 Embedding Key 调 chat 端点，否则返回 401。

### 2.4 per-key 采样模板（probabilistic / greedy）

**背景**：2026-08-05 probabilistic 切换落地后，F 生产 speculative decode 使用 `probabilistic` 采样（需请求端 `temperature>0.1` 才生效，temp≈0 会回退 greedy）。LiteLLM 侧新增两个模板模型，按业务负载类型发放不同 Key：

| 模型名 | Key | 默认 temperature | 适用负载 |
|--------|-----|-----------------|----------|
| `dspark-prob` | Prob Key | **0.7** | 结构化负载（code / json / tool call）——吞吐 +20~47%，质量无退化 |
| `dspark-greedy` | Greedy Key | **0.1**（LiteLLM 部分 provider 取整，故不用 0） | 散文 / 思考链 / 需确定性输出 |

> ⚠️ **使用前必读（两条硬性约定）**：
> 1. **必须显式指定 model 名**：本版 LiteLLM（1.83.7）**不支持 key 级 aliases 生效**（`/key/generate` 的 `aliases` 字段仅存储不应用；实测 `model=dspark` 返回 401 `key_model_access_denied`）。客户端**必须**显式使用 `model=dspark-prob` 或 `model=dspark-greedy`。
> 2. **Prob Key 必须传 `temperature>0.1`**（建议 0.7）：temp≈0 回退 greedy、失去 +20~47% 吞吐增益；LiteLLM 无 `min_temperature` guardrail，**强制由请求端承担**（正式约定见 §2.5）。

**curl 示例**
```bash
# Prob Key → 结构化应用
curl http://<NODE_IP>:4000/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "dspark-prob", "messages": [{"role": "user", "content": "write python to parse json"}], "temperature": 0.7}'

# Greedy Key → 散文 / 思考链
curl http://<NODE_IP>:4000/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>_C8xuN9rqHhg" \
  -H "Content-Type: application/json" \
  -d '{"model": "dspark-greedy", "messages": [{"role": "user", "content": "explain the plan"}], "temperature": 0.1}'
```

**Python（openai SDK）**
```python
from openai import OpenAI

prob_client = OpenAI(base_url="http://<NODE_IP>:4000/v1", api_key="<API_KEY>")
resp = prob_client.chat.completions.create(
    model="dspark-prob",                 # 必须显式 model 名（aliases 不生效）
    temperature=0.7,                     # 结构化应用请保持 >0.1，probabilistic 才生效
    messages=[{"role": "user", "content": "..."}],
)

# Greedy Key → 散文 / 思考链（确定性优先）
greedy_client = OpenAI(
    base_url="http://<NODE_IP>:4000/v1",
    api_key="<API_KEY>_C8xuN9rqHhg",   # Greedy Key
)
resp = greedy_client.chat.completions.create(
    model="dspark-greedy",                 # 必须显式 model 名（aliases 不生效）
    temperature=0.1,                       # greedy 语义；散文/思考链不要调高
    messages=[{"role": "user", "content": "..."}],
)
```

**temperature 语义说明**：
- config 中 `temperature` 是**默认值**（客户端未传时生效）；客户端可每次请求覆盖（实测 temp=0 覆盖后输出确定性回归，temp=0.7 输出有方差）。
- **probabilistic 生效前提**：temp>0.1（建议 0.7），temp≤0.1 回退 greedy——正式约定见 **§2.5 温度约定**。
- 散文 / 思考链应用若需深度思考，可用 Greedy Key + `enable_thinking`（vLLM 上游经请求体 `chat_template_kwargs` 控制；本版 LiteLLM 未确认 `chat_template_kwargs` 透传，建议客户端直接传 `extra_body` 或在 8003 直连路径处理）。

### 2.5 温度约定（Temperature Contract）

> **性质**：probabilistic 采样生效的**前提条件**，由请求端强制（LiteLLM 无 guardrail）。违反不报错，但会**静默失去吞吐增益**。

**约定正文**
- 所有走 LiteLLM 4000 且使用 `dspark-prob`（或期望 probabilistic 加速）的请求，`temperature` **必须 > 0.1**（建议 0.7）。
- `temperature ≤ 0.1` 视为退化为 **greedy 语义**：probabilistic 不生效，失去 +20~47% 吞吐增益（仅性能损失，无正确性风险）。
- 结构化应用（code/json/tool）建议显式 `temperature=0.7`（至少 ≥0.2）。

**客户端实现建议**（各 SDK 以请求体字段 `temperature` 透传，确认最终请求体含 `"temperature": 0.7` 即可）
- Python（openai SDK）：`create(model="dspark-prob", temperature=0.7, ...)`（完整示例见 §2.4）。
- Node.js（openai SDK）：
```javascript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://<NODE_IP>:4000/v1", apiKey: "<API_KEY>" });
await client.chat.completions.create({ model: "dspark-prob", temperature: 0.7, messages: [{ role: "user", content: "..." }] });  // 必须 >0.1
```
- 通用：若框架默认 temp=0 或未传（回退 config 默认 0.7），需显式覆盖。

**服务端现状 / 可选加固**：LiteLLM 1.83.7 无 `min_temperature` guardrail，网关不拦截 temp≤0.1，**强制由请求端承担**；可选加固（非阻塞）为 custom callback / 前置校验，对 `dspark-prob` 且 temp≤0.1 告警或注入 0.7（需 SRE 排期）。

**与其他参数交互**：与 `max_tokens` / `stream` 无关；思考链场景（走 8003 / responses API）**不受此约定约束**（8003 为 passthrough，按 greedy 语义处理）。

---

## 3. 自建网关 8003 使用说明

### 3.1 接入信息

| 项 | 值 |
|----|-----|
| base_url | `http://<NODE_IP>:8003` |
| SDK base_url | `http://<NODE_IP>:8003/v1` |
| 鉴权方式 | HTTP Header `Authorization: Bearer <API_KEY>-64b0374c6f2840fe` |
| 可用端点 | `/v1/models`、`/v1/chat/completions`、`/v1/responses`、`/v1/embeddings` |
| 模型名 | 以 `GET /v1/models` 返回为准（passthrough 至 vLLM 8001） |

### 3.2 Chat / Completions 请求示例

**curl**
```bash
curl http://<NODE_IP>:8003/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>-64b0374c6f2840fe" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "2+2=?"}]
  }'
```

**Python（openai SDK）**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<NODE_IP>:8003/v1",
    api_key="<API_KEY>-64b0374c6f2840fe",  # 8003 客户端 Key
)

resp = client.chat.completions.create(
    model="deepseek-v4-flash-0731",
    messages=[{"role": "user", "content": "2+2=?"}],
)
print(resp.choices[0].message.content)
```

### 3.3 Responses API（保思考链）

8003 为 passthrough 直连 vLLM，`/v1/responses` **保留完整思考链**：输出含 `type=reasoning` + `reasoning_text` 字段，`reasoning` 不为空。

**curl**
```bash
curl http://<NODE_IP>:8003/v1/responses \
  -H "Authorization: Bearer <API_KEY>-64b0374c6f2840fe" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "input": "2+2=?"
  }'
```

**Python（openai SDK）**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<NODE_IP>:8003/v1",
    api_key="<API_KEY>-64b0374c6f2840fe",
)

resp = client.responses.create(
    model="deepseek-v4-flash-0731",
    input="2+2=?",
)
# 思考链在 output 中：item.type == "reasoning" 时读取 item.reasoning_text
for item in resp.output:
    if item.type == "reasoning":
        print("reasoning:", item.reasoning_text)
```

---

## 4. 双轨选择指引

| 场景 | 推荐网关 | 原因 |
|------|----------|------|
| 标准 OpenAI 兼容（chat / embeddings） | **LiteLLM 4000** | 原生 OpenAI 语义，多 key 管理 + 限流 + 用量统计开箱即用，并发稳定（10 并发 p50 772ms） |
| 需要多 key 隔离 / 按 key 限流 / 权限管控 | **LiteLLM 4000** | 虚拟 key 体系：按模型授权、按 rpm/tpm 限流，越权返回 401 |
| 结构化负载（code/json/tool）且需高吞吐 | **LiteLLM 4000 `dspark-prob`** | probabilistic 采样 + temp 0.7 默认，吞吐 +20~47%（详见 §2.4） |
| 散文 / 思考链 / 确定性输出 | **LiteLLM 4000 `dspark-greedy`** | greedy 模板（temp 0.1），详见 §2.4 |
| 需要 `responses` API **思考链**（reasoning 内容） | **自建 8003** | 4000 的 `/v1/responses` 思考链有回归（reasoning 为 null）；8003 passthrough 保留 `type=reasoning` + `reasoning_text` |
| 纯 chat 路径且需要思考链 | **4000（chat 路径）** 或 8003 | 4000 的 `/v1/chat/completions` 思考链正常（reasoning_content 顶层 + provider_specific_fields.reasoning），两者均可 |
| 直接访问 vLLM 内部能力 / 调试引擎 | 8003 直连（不建议业务直连 8001） | 8001 仅内网，key 为内部 key |

> **一句话**：默认走 4000（OpenAI 兼容 + key 管控 + 限流）；**只有**强依赖 Responses API 思考链的客户端走 8003。

---

## 5. 安全须知

1. **Master Key（`sk-litellm-master-*`）与内部 Key（`<API_KEY>-*`）严禁外泄**：
   - Master Key 拥有全部 Admin 权限，可建 key、轮换、查用量，泄露即等于网关失守。
   - 内部 Key 仅网关转发 vLLM 用，客户端不应接触。
   - 文档中其余业务 key（chat / embedding / prob / greedy / 8003）也应仅按需发放。
2. **权限隔离**：LiteLLM 按 key 限模型、按 key 限流。发放给客户端时按最小权限原则：对话只发 Chat Key，向量只发 Embedding Key。
3. **Key 轮换（LiteLLM 4000）**：用 Master Key 调 `/key/regenerate` 生成新 key（旧 key 自动失效）：
   ```bash
   curl http://<NODE_IP>:4000/key/regenerate \
     -H "Authorization: Bearer <BEARER>" \
     -H "Content-Type: application/json" \
     -d '{
       "key": "sk-U_cIbL63-5c27rayJO3S6w",
       "models": ["local-v4-flash", "deepseek-v4-flash"],
       "rpm_limit": 300,
       "tpm_limit": 50000
     }'
   ```
   轮换后立即通知所有持 key 方更换，确认无旧调用后再销毁旧 key。
4. **8003 Key 轮换**：自建网关 key 为服务端配置项，轮换需在网关配置中替换并重启（具体流程待运维补充，见 §7）。
5. **Master Key 建议改环境变量注入**（当前在 config 中，属遗留待办），PG 仅监听 127.0.0.1。

---

## 6. 故障排查

| 状态码 | 可能原因 | 排查与处理 |
|--------|----------|-----------|
| **401** | ① Key 拼写/复制错误；② 权限越界（如 Chat Key 调 embeddings 端点、Embedding Key 调 chat）；③ Prob/Greedy Key 传了非授权 model（如 `dspark`——本版 aliases 不生效，必须用 `dspark-prob`/`dspark-greedy`） | 检查 `Authorization: Bearer <key>` 是否完整；核对 key 对应的授权模型（§1 权限范围）；业务 key 越权需申请对应 key，不要用 Master Key 顶替 |
| **404** | 模型名错误 / 模型未在网关注册 | 用 `GET /v1/models` 拉取实际模型名再比对；4000 用 `local-v4-flash` / `deepseek-v4-flash` / `dspark-prob` / `dspark-greedy` / `local-embedding` |
| **429** | rpm（每分钟请求数）或 tpm（每分钟 token 数）超限 | 查看响应中 `Retry-After` / 限流字段；等待窗口重置；或联系管理员调大限额（Chat 默认 rpm 300 / tpm 50,000；Embedding rpm 300 / tpm 100,000） |
| **连接超时 / 5xx** | 网关或后端 vLLM 未就绪、负载过高 | 检查 4000 / 8003 / 8001 端口连通性；4000 并发稳定，若首轮排队明显可对比 8003 表现 |

---

## 7. 待确认 / 补充项

**✅ 已确认（2026-08-05 per-key 模板落地后）**：
- [x] per-key 采样模板已上线：`dspark-prob`（temp 0.7）/ `dspark-greedy`（temp 0.1），Prob / Greedy Key 已发放并验证（见 §1、§2.4）
- [x] LiteLLM 1.83.7 **key 级 aliases 不生效**（`model=dspark` → 401 `key_model_access_denied`），客户端必须显式用 `dspark-prob` / `dspark-greedy`
- [x] Prob / Greedy Key **未设限流**（rpm/tpm 未配置）；如需限制可后续在 LiteLLM 侧配置 guardrail，当前靠客户端自觉（temp>0.1 约定已固化为 §2.5）
- [x] **temp>0.1 温度约定已固化**（见 §2.5）：Prob Key 必须显式传 temperature>0.1（建议 0.7），temp≤0.1 视为 greedy 语义；LiteLLM 无 min_temperature guardrail，强制由请求端承担

**仍待确认 / 补充**：
- [ ] 8003 网关的**模型名清单**（本手册示例使用 `deepseek-v4-flash-0731`，请以 `/v1/models` 实测返回为准并回填）
- [ ] 8003 客户端 Key 的**限流配置**（当前未配置，若需限流请运维补充）
- [ ] 8003 的 **key 轮换操作步骤**（自建网关服务端配置项，待 SRE 补充）
- [ ] Master Key 是否已改环境变量注入（遗留安全项）
- [ ] `chat_template_kwargs` 透传未确认（`enable_thinking` 相关；思考链客户端按 8003 直连处理）

---

> 本手册由工程保障团队 AI 协作生成，Key 为敏感信息，请按内部安全规范保管与分发。
