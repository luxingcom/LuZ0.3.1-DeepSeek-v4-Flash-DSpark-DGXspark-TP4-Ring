# 网关路由 DeepSeek 官方接口规范符合性审查报告

**日期**：2026-08-04
**工作流**：工作流 1（代码审查 - 规范符合性专项）
**参与成员**：Cody（代码审查师）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：网关 `/v1/chat/completions` 路由基本符合 DeepSeek 官方规范；`/v1/responses` 路由存在**未知模型放行**（🔴严重）与**合法字段 `reasoning` 静默剥离**（🟠高）两项实质偏差，叠加共享 MODEL_ALIAS 未区分路由级白名单、502 错误 type 非标准值，整体为部分偏差。
- **严重度分布**：🔴严重 1 项 / 🟠高 2 项 / 🟡中 3 项 / 🟢低 2 项
- **阻塞 / 非阻塞**：非阻塞（无安全漏洞类阻塞），但 #1 影响客户端错误体验与网关错误体可控性，建议尽快修复。
- **与前次审计关联**：本次 #2（两路由校验矛盾）即前次审计报告 #10 的深化确认；本次进一步发现 model 合法取值范围两路由本应不同（Chat=flash+pro，Responses=仅 flash），当前共享 MODEL_ALIAS 未体现此差异。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（部分偏差） |
| 阻塞项数量 | 0 |
| 关键行动项 | 5 条 |
| 建议下一步 | 先修复 #1 Responses 严格 404 + #3 reasoning 透传，再区分路由级白名单与 502 type 标准化 |

---

## 🔍 规范符合性审查发现（按严重度排序）

| # | 严重度 | 规范条款 | 文件:行 | 偏差描述 | 建议修复 | 来源 |
|---|--------|----------|---------|----------|----------|------|
| 1 | 🔴严重 | Responses model 必填且合法取值仅 [deepseek-v4-flash] | responses_gateway_main.py:220 | /v1/responses 用 `MODEL_ALIAS.get(model, model)`，未知模型原样放行转发上游。规范要求 model 必须是合法值，不合法应返 400/404。放行后客户端收到上游 400 而非网关标准错误体，错误格式不可控 | 改为严格校验：`if model not in <responses_allowlist>: return 404 _model_not_found(model)`，与 Chat 路由一致 | Cody |
| 2 | 🟠高 | 两路由 model 校验策略应一致 | :220 vs :295-296 | Chat 路由严格 404（`if model not in MODEL_ALIAS`），Responses 路由放行未知模型（`get(model, model)`）。同一网关对同语义字段采用相反策略 | 统一为严格校验；若需差异化，分别定义两路由合法模型集后校验 | Cody |
| 3 | 🟠高 | Responses API `reasoning` 是合法 nullable 字段 | :225 | /v1/responses 强制 `body.pop("reasoning", None)`，剥离客户端合法字段。规范将 `reasoning`(object,nullable) 列为正式字段，客户端有权传入控制推理行为。静默丢弃改变客户端意图 | 将 reasoning 透传上游由 vLLM 处理（若不支持则按规范返 400）；至少文档记录此剥离行为 | Cody |
| 4 | 🟡中 | 两路由 model 合法取值范围不同（Chat=flash+pro，Responses=仅flash） | :109-113 | MODEL_ALIAS 为单一共享映射，未区分路由级白名单。当前仅含 flash 变体；未来为支持 Chat 的 v4-pro 而添加 pro 后，Responses 放行逻辑会同时接受 pro，违反 Responses 仅 flash 规范 | 为两路由分别定义 RESPONSES_MODELS 与 CHAT_MODELS 白名单 | Cody |
| 5 | 🟡中 | Chat model 合法取值含 deepseek-v4-pro | :295-296 | Chat 严格 404 未知模型，但 MODEL_ALIAS 不含 v4-pro。规范允许 v4-pro 作为 Chat 合法模型，网关不当拒绝合法模型 | 若后端支持 v4-pro，在 MODEL_ALIAS 添加映射；若不支持则文档声明仅服务 flash | Cody |
| 6 | 🟡中 | 错误响应 type 应使用标准值 | :247, :314 | 502 错误体 `type:"upstream_error"`，OpenAI/DeepSeek 标准取值为 `api_error`/`server_error`/`invalid_request_error`/`authentication_error`，`upstream_error` 非标准值，客户端按 type 分支时无法识别 | 改为 `type:"api_error"` 或 `"server_error"` | Cody |
| 7 | 🟢低 | model 命名应使用官方名称 | :72 | PUBLIC_MODEL="local-v4-flash"，官方名为 deepseek-v4-flash。用自定义名为主公开名，影响客户端对规范 model 名的直接可用性 | 如非必要对齐官方名；或在 /v1/models 中同时列出官方名映射 | Cody |
| 8 | 🟢低 | 路径前缀 /v1 与无前缀并存 | :206-207, :272-273 | 官方路径无 /v1 前缀；网关同时挂两形式。为兼容 openai SDK 两种 base_url 写法的常见做法，不构成违规，属超规范扩展 | 无需修复；保留兼容性，文档说明两形式等价 | Cody |

---

## 🏗️ 两路由一致性评估

### model 校验
- **Chat Completions**（:295-296）：严格校验，未知模型 -> 404。符合"model 必填且合法"的规范精神。
- **Responses**（:220）：宽松放行，未知模型 -> 原样转发上游。违反规范，且与 Chat 路由策略矛盾。
- **结论**：不一致。同一网关对同一语义字段采用相反校验策略，Responses 路由是薄弱环节。

### 字段处理
- **Responses**（:225-228）：剥离 `reasoning`（规范合法字段）与 `reasoning_effort`（非 Responses 字段）。剥离 reasoning 属规范偏差；剥离 reasoning_effort 合理（该字段不属于 Responses API）。
- **Chat**（不做剥离）：不剥离已废弃字段 `frequency_penalty`/`presence_penalty`。规范称"传入不产生效果"即接受但忽略，不剥离是合规的。
- **结论**：不一致且方向相反。Responses 过度处理（剥离合法字段），Chat 恰当忽略废弃字段。两路由对"非预期字段"的处理哲学不统一。

### 错误格式
- 两路由共享 `_unauthorized`（401）与 `_model_not_found`（404）工厂函数，格式一致且符合 OpenAI/DeepSeek error.message/type/code/param 约定。
- 502 错误体（:247, :314）两路由一致使用 `type:"upstream_error"`，但该 type 值非标准。
- 400 Invalid JSON 错误体（:215, :291）两路由一致，缺 code/param 字段但不影响符合性。
- **结论**：错误格式整体一致，但 502 的 type 值偏差在两路由中重复出现，建议同步修正。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 修复 /v1/responses model 严格校验：未知模型返回 404 标准错误体（`_model_not_found`），与 Chat 路由一致 | Cody | P0 | 本周 |
| 2 | 停止剥离 /v1/responses 的 `reasoning` 字段：透传上游由 vLLM 处理；若 vLLM 不支持则按规范返 400，不得静默丢弃 | Cody | P0 | 本周 |
| 3 | 为两路由分别定义路由级 model 白名单（RESPONSES_MODELS 仅含 flash 变体；CHAT_MODELS 可含 flash+pro），替代单一共享 MODEL_ALIAS 的校验职责 | Archi+Cody | P1 | 2 周内 |
| 4 | 502 错误体 type 值改为标准 `api_error` 或 `server_error`，两路由同步修正 | Cody | P1 | 2 周内 |
| 5 | 若后端支持 v4-pro，在 Chat 白名单添加映射；否则文档声明仅服务 flash 变体 | Archi | P2 | 按需 |

---

## ⚠️ 待完善 / 已知局限

- 本审查基于线上 `hardened/live/responses_gateway_main.py` v1.4.0 静态代码与 DeepSeek 官方文档对照，未对 vLLM 后端实际是否支持 `reasoning` 字段、是否支持 `deepseek-v4-pro` 做实机探测。
- 官方文档可能存在版本迭代，`reasoning.effort` 在 Chat 与 Responses 两接口的语义映射（Chat 用 `reasoning_effort` 顶层字段，Responses 用 `reasoning` 对象）需以后端 vLLM 实际行为为准。
- 路径前缀 `/v1` 与无前缀并存（#8）属常见兼容做法，不视为违规。

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）原始产出**：`deliverables/engineering-assurance/_cody_spec_review_raw.md`（8 项发现 + 两路由一致性评估 + 🟡 评级）；审查对象 `hardened/live/responses_gateway_main.py` v1.4.0；对照基准 DeepSeek 官方 Chat Completions / Responses API 文档。
- **前次审计报告关联**：`assurance-audit-dspark-cluster-2026-08-04.md` #10 两路由校验不一致项的深化确认。

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
