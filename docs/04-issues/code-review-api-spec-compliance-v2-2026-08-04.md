# 网关路由 DeepSeek 官方接口规范符合性审查报告（v2 严格深化）

**日期**：2026-08-04
**工作流**：工作流 1（代码审查 - 规范符合性专项深化）
**参与成员**：Cody（代码审查师）
**关联**：深化 `code-review-api-spec-compliance-2026-08-04.md`（v1，8 项发现）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：v1 报告的 8 项发现全部成立；v2 深化新增 6 项，合并 **14 项发现**。用户三个关注点已专项核查：① embed 模型已完全独立，#1 修复不影响 embed；② 思考模式当前兼容正常（WorkBuddy 走 Chat 路由透传 reasoning_effort），但 Responses 路由剥离 reasoning 对象阻碍未来 per-request 控制；③ 严格标准下两路由均缺少必填字段网关级校验。
- **严重度分布**：🔴严重 1 项 / 🟠高 4 项 / 🟡中 6 项 / 🟢低 3 项
- **阻塞 / 非阻塞**：非阻塞，但 #1（🔴未知模型放行）影响客户端错误体验，V2-1/V2-2（必填字段未校验）影响严格规范满足。
- **修复路径**：修复 #1、#3、V2-1、V2-2 后可达 🟢 严格满足。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（部分偏差，需修复后方可声明"严格满足"） |
| 阻塞项数量 | 0 |
| 关键行动项 | 7 条 |
| 建议下一步 | P0 修 #1 Responses 严格 404 + #3 reasoning 透传 + V2-1/V2-2 必填字段校验 -> P1 统一校验+错误格式+流式标记 |
| embed 影响 | ✅ 无影响（已确认独立路由） |
| 思考模式兼容 | ✅ 当前正常（Chat 透传）；⚠️ Responses 剥离 reasoning 阻碍未来 per-request 控制 |

---

## 🔍 三关注点专项结论

### 关注点1：未知模型放行是否为 embed 模型

**结论：无关。embed 已完全独立，#1 修复不影响 embed 功能。**

证据链：
- `/v1/embeddings` 路由（行 173-203）是**物理隔离的独立路由**：客户端鉴权 -> 解析 JSON -> 注入 UPSTREAM_API_KEY -> 直接透传到 `EMBED_URL`（`http://127.0.0.1:8020`，行 191）。全程**不经过 MODEL_ALIAS 映射**，不触碰 vLLM 8001。
- `/v1/responses` 路由（行 206-269）转发到 `VLLM_URL`（`http://<NODE_IP>:8001`，行 238）。两路由后端 IP 不同（8020 vs 8001），路径不同（`/v1/embeddings` ≠ `/v1/responses`）。
- embed 模型（Qwen3-Embedding-0.6B）请求**只命中 `/v1/embeddings`**，不可能走 `/v1/responses`。FastAPI 路由按路径匹配。
- 行 220 的 `MODEL_ALIAS.get(model, model)` 放行逻辑仅在 `/v1/responses` 内部执行，与 embed 路由无代码路径交集。

**因此**：将 #1 修复为严格 404 **不会影响 embed 功能**。

### 关注点2：思考模式与 WorkBuddy 兼容性

**结论：WorkBuddy 走 Chat 路由时兼容正常；Responses 路由剥离 reasoning 对象是潜在风险。**

| 检查项 | 结论 | 证据 |
|--------|------|------|
| Chat 路由是否透传 `reasoning_effort` | ✅ 是 | 行 272-336 无字段剥离，`reasoning_effort` 可到达 vLLM 8001 |
| Responses 路由是否剥离 `reasoning`/`reasoning_effort` | ✅ 是（剥离） | 行 225 `body.pop("reasoning")`；行 227-228 `body.pop("reasoning_effort")` |
| vLLM 引擎级默认能否被 per-request 覆盖 | ✅ 能 | vLLM `--default-chat-template-kwargs.thinking=true --reasoning_effort=max` 是引擎默认，Chat 请求中的顶层 `reasoning_effort` 按标准行为覆盖 |
| WorkBuddy 走 Chat 发 `reasoning_effort` 是否符合规范 | ✅ 符合 | DeepSeek Chat 规范中 `reasoning_effort` 是顶层字段，取值 `low`/`high`/`max`（默认 `high`） |
| WorkBuddy 走 Responses 路由的剥离风险 | ⚠️ 存在 | 若切到 Responses 路由，行 227-228 剥离 `reasoning_effort`，行 225 剥离 `reasoning` 对象，思考模式退回引擎默认，客户端无法 per-request 控制 |

**关键评估**：
- **当前状态**：WorkBuddy 走 `/v1/chat/completions`（v1.4.0 新增此路由正是为此），`reasoning_effort` 正常透传，**当前无功能缺陷**。
- **潜在风险**：Responses API 规范本就用 `reasoning` 对象（而非顶层 `reasoning_effort`），所以剥离顶层 `reasoning_effort` 在 Responses 路由**符合规范**。问题在于同时剥离了 `reasoning` 对象（行 225），这才是规范偏差（#3），且阻碍未来 Responses 路由支持 per-request 思考强度控制。

### 关注点3：严格规范补充发现

以"严格满足"标准重新审视，发现 v1 审查遗漏的 6 项规范偏差（详见下方发现表 V2-1 至 V2-6）。

---

## 🔍 合并审查发现（v1 8 项 + v2 新增 6 项，按严重度排序）

| # | 严重度 | 规范条款 | 文件:行 | 偏差描述 | 建议修复 | 来源 |
|---|--------|----------|---------|----------|----------|------|
| 1 | 🔴严重 | Responses model 必填且合法取值仅 [deepseek-v4-flash] | :220 | /v1/responses 用 `MODEL_ALIAS.get(model, model)`，未知模型原样放行转发上游。规范要求 model 必须是合法值。放行后客户端收到上游 400 而非网关标准错误体 | 改为严格校验：`if model not in <responses_allowlist>: return 404 _model_not_found(model)` | Cody v1 |
| 2 | 🟠高 | 两路由 model 校验策略应一致 | :220 vs :295-296 | Chat 路由严格 404，Responses 路由放行未知模型。同一网关对同语义字段采用相反策略 | 统一为严格校验；分别定义两路由合法模型集 | Cody v1 |
| 3 | 🟠高 | Responses API `reasoning` 是合法 nullable 字段 | :225 | /v1/responses 强制 `body.pop("reasoning", None)`，剥离客户端合法字段。规范将 `reasoning`(object,nullable) 列为正式字段。静默丢弃改变客户端意图，阻碍未来 per-request 思考控制 | 透传上游由 vLLM 处理；至少文档记录剥离行为 | Cody v1 |
| V2-1 | 🟠高 | Chat Completions 必填字段 `messages` 校验 | :288-291 | 网关仅校验 JSON 合法性，不校验 `messages` 是否存在/是否非空数组。规范要求 `messages` 必填。缺失时转发到 vLLM 返回上游错误，网关未拦截 | 网关层增加 `if not body.get("messages"): return 400` | Cody v2 |
| V2-2 | 🟠高 | Responses API 必填字段 `input` 校验 | :212-215 | 网关仅校验 JSON 合法性，不校验 `input` 是否存在。规范要求 `input` 必填。缺失时转发上游返回不透明错误 | 网关层增加 `if not body.get("input"): return 400` | Cody v2 |
| 4 | 🟡中 | 两路由 model 合法取值范围不同（Chat=flash+pro，Responses=仅flash） | :109-113 | MODEL_ALIAS 单一共享映射，未区分路由级白名单。未来为 Chat 添加 v4-pro 后 Responses 会错误放行 pro | 为两路由分别定义 RESPONSES_MODELS 与 CHAT_MODELS 白名单 | Cody v1 |
| 5 | 🟡中 | Chat model 合法取值含 deepseek-v4-pro | :295-296 | Chat 严格 404 未知模型，但 MODEL_ALIAS 不含 v4-pro。规范允许 v4-pro 作为 Chat 合法模型 | 若后端支持 v4-pro 添加映射；否则文档声明仅服务 flash | Cody v1 |
| 6 | 🟡中 | 错误响应 type 应使用标准值 | :247,:314 | 502 错误体 `type:"upstream_error"`，标准取值为 `api_error`/`server_error` 等，非标准值 | 改为 `type:"api_error"` 或 `"server_error"` | Cody v1 |
| V2-3 | 🟡中 | Chat 流式响应必须以 `data: [DONE]` 结尾 | :318-332 | 网关做 SSE 字节透传，不验证 `[DONE]` 标记存在。若上游未发送，客户端流无法正常终止。依赖 vLLM 行为正确 | 可选：透传后检测流末尾含 `[DONE]`，缺失则补发；或文档声明依赖上游 | Cody v2 |
| V2-4 | 🟡中 | Responses 流式响应不应有 `[DONE]` | :251-265 | 网关对 Responses 流做裸字节透传。Responses 用 `response.completed` 事件结束，不含 `[DONE]`。若 vLLM 错误注入了 `[DONE]`（复用 chat 模板），网关不拦截 | 可选：透传后过滤 Responses 流中的 `[DONE]` 行；或验证 vLLM 行为 | Cody v2 |
| V2-5 | 🟡中 | `max_tokens`(Chat) vs `max_output_tokens`(Responses) 混淆风险 | :272-336/:206-269 | 两路由均做裸透传不转换。若客户端在 Responses 请求中误传 `max_tokens`（Chat 字段），vLLM 可能忽略或报错 | 可选：Responses 路由将 `max_tokens` 映射为 `max_output_tokens`；或文档声明不可混用 | Cody v2 |
| 7 | 🟢低 | model 命名应使用官方名称 | :72 | PUBLIC_MODEL="local-v4-flash"，官方名为 deepseek-v4-flash | 如非必要对齐官方名；或在 /v1/models 同时列出官方名映射 | Cody v1 |
| 8 | 🟢低 | 路径前缀 /v1 与无前缀并存 | :206-207,:272-273 | 官方路径无 /v1 前缀；网关同时挂两形式。为兼容 openai SDK 常见做法，不构成违规 | 无需修复；文档说明两形式等价 | Cody v1 |
| V2-6 | 🟢低 | `/v1/models` 返回格式细节 | :155-170 | `created` 字段硬编码为 `0`（行 160）。规范示例中为 Unix 时间戳。值 `0` 不违反 schema 但不符惯例 | 设为实际时间戳 `int(time.time())` 或文档说明 | Cody v2 |

---

## 🏗️ 两路由一致性评估（更新）

### model 校验
- Chat（:295-296）：严格 404，符合规范精神。
- Responses（:220）：宽松放行，违反规范，且与 Chat 策略矛盾。
- **结论**：不一致。Responses 是薄弱环节。

### 字段处理
- Responses（:225-228）：剥离 `reasoning`（合法字段，规范偏差）与 `reasoning_effort`（非 Responses 字段，合理）。
- Chat（不剥离）：不剥离已废弃字段 `frequency_penalty`/`presence_penalty`（规范称"传入不产生效果"，合规）。
- **新增**：两路由均不校验必填字段（Chat 的 `messages`、Responses 的 `input`），缺失时转发上游返回不透明错误。
- **结论**：不一致且方向相反。Responses 过度处理（剥离合法字段），Chat 恰当忽略废弃字段。两路由对"非预期字段"处理哲学不统一；必填字段校验均缺失。

### 错误格式
- 两路由共享 `_unauthorized`/`_model_not_found` 工厂函数，格式一致且符合约定。
- 502 错误体 `type:"upstream_error"` 非标准（两路由重复出现）。
- 400 Invalid JSON 缺 code/param 但不影响符合性。
- **结论**：整体一致，502 type 偏差重复出现需同步修正。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 修复 /v1/responses model 严格校验：未知模型返回 404 标准错误体（不影响 embed） | Cody | P0 | 本周 |
| 2 | 停止剥离 /v1/responses 的 `reasoning` 字段：透传上游或按规范返 400 | Cody | P0 | 本周 |
| 3 | 网关层增加必填字段校验：Chat 校验 `messages`、Responses 校验 `input`，缺失返 400 | Cody | P0 | 本周 |
| 4 | 为两路由分别定义路由级 model 白名单（RESPONSES_MODELS / CHAT_MODELS） | Archi+Cody | P1 | 2 周内 |
| 5 | 502 错误体 type 改标准 `api_error`/`server_error`，两路由同步 | Cody | P1 | 2 周内 |
| 6 | 流式格式保证：Chat 检测 `[DONE]` 存在性；Responses 过滤可能的 `[DONE]` | Cody | P1 | 2 周内 |
| 7 | 若后端支持 v4-pro，Chat 白名单添加映射；否则文档声明仅服务 flash | Archi | P2 | 按需 |

---

## ⚠️ 待完善 / 已知局限

- 本审查基于线上 `hardened/live/responses_gateway_main.py` v1.4.0 静态代码与 DeepSeek 官方文档对照，未对 vLLM 后端实际行为做实机探测。
- vLLM Responses 端点是否会错误注入 `[DONE]`（V2-4）需实机验证。
- `reasoning` 字段透传后 vLLM 是否支持 per-request 覆盖需实机验证。
- embed 模型独立性已通过代码路径分析确认，但未做端到端请求验证。

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）v1 原始产出**：`_cody_spec_review_raw.md`（8 项发现）
- **Cody（代码审查师）v2 深化产出**：`_cody_spec_review_v2_raw.md`（三关注点专项 + 6 项补充发现 + 更新评级）
- **审查对象**：`hardened/live/responses_gateway_main.py` v1.4.0
- **对照基准**：DeepSeek 官方 Chat Completions / Responses API 文档

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
