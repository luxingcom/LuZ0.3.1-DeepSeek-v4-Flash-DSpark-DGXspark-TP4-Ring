# 双机 vLLM 重启执行报告（reasoning-parser 生效验证）

**日期**：2026-08-04
**工作流**：工作流 4（部署执行 + 验证）
**参与成员**：Rex（SRE 工程师）/ Zhen（主理人，编排与复核）
**前置**：`fix-sync-remote-2026-08-04.md`（脚本已同步待重启）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：双机 vLLM 已重启，`--reasoning-parser deepseek_v4` **确认生效**（日志 `reasoning_parser='deepseek_v4'`），工具调用与非工具回归全绿。但发现**字段名差异**：该 vLLM 版本（0.25.2.dev0）返回 `reasoning` 而非 `reasoning_content`，且思考内容需 `enable_thinking:true` 触发才返回。
- **严重度分布**：🔴严重 0 项 / 🟠高 1 项（字段名适配，影响 WorkBuddy 折叠）/ 🟡中 1 项（enable_thinking 触发条件）
- **阻塞 / 非阻塞**：非阻塞。服务全部健康，但客户端折叠依赖的字段名需适配确认。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 重启成功；⚠️ 字段名适配待决策 |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（字段适配 / enable_thinking 决策 / 客户端验证） |
| 建议下一步 | 确认 WorkBuddy 读 `reasoning` 还是 `reasoning_content`，决定网关层适配方案 |

---

## 🔍 执行详情

### 阶段 1-2：双机重启（Rex）

| 节点 | 重启时刻 | 容器状态 | serve 参数确认 |
|------|---------|---------|---------------|
| worker (.58) | 13:14:44 | `vllm-envE-worker` Up healthy ✅ | 含 `--reasoning-parser deepseek_v4` |
| head (.60) | 13:15:12 | `vllm-envE-node` Up healthy ✅（READY 340s） | 含 `--reasoning-parser deepseek_v4` |

### 阶段 3：全链路验证（Rex）

| # | 验证项 | 结果 |
|---|--------|------|
| 1 | 直连 8001 /v1/models | HTTP 200 ✅ |
| 2 | reasoning_parser 生效 | 日志确认 `reasoning_parser='deepseek_v4'` ✅ |
| 3 | 思考内容返回 | 普通请求 `reasoning=null`（思考默认关闭）；带 `chat_template_kwargs:{"enable_thinking":true}` 后 `reasoning` 返回非空 ✅（网关 8003 + 直连 8001 均验证） |
| 4 | 工具调用回归 | 网关 8003 200，`tool_calls` 正常（get_weather 北京）✅ |
| 5 | 非工具回归 | 网关 8003 200，正常 content ✅ |

---

## ⚠️ 关键发现：字段名差异（需人类决策）

**现象**：此 vLLM 版本（vllm-0.25.2.dev0）响应字段名为 **`reasoning`**，而 DeepSeek 官方/WorkBuddy 折叠依赖的字段是 **`reasoning_content`**。

**影响链**：
- 根因诊断报告（diagnostic-thinking-collapse-failure）假设修复后返回 `reasoning_content`——实际返回 `reasoning`
- WorkBuddy 若读取 `reasoning_content` 折叠，将仍无法识别思考内容（字段名不匹配）
- 另：思考内容需 `enable_thinking:true` 显式触发（chat template 默认关闭思考），普通请求思考内容为空

**决策点（二选一或组合）**：
1. **网关层适配**：在 `responses_gateway_main.py` 的 chat/completions 响应透传处做字段改写 `reasoning` → `reasoning_content`（或同时保留两者），使客户端无需改动
2. **客户端适配**：WorkBuddy 侧读取 `reasoning` 字段（若客户端可配置）
3. **思考开关**：是否在网关层默认注入 `chat_template_kwargs.enable_thinking=true`（影响所有请求默认带思考，与之前 thinking 参数决策点同源）

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 确认 WorkBuddy 读取字段（reasoning vs reasoning_content），决定网关适配或客户端适配 | 人类负责人 | P0 | 立即 |
| 2 | 若网关适配：chat/completions 透传处做 reasoning→reasoning_content 字段映射（保留原字段或二选一） | Cody | P0 | 决策后 |
| 3 | 决策 enable_thinking 默认开关（网关注入 vs 客户端传参） | 人类负责人 | P1 | 决策后 |
| 4 | 在 WorkBuddy 实测思考折叠效果 | 用户 | P0 | 适配后 |
| 5 | 补测：工具调用全流程存证 + stress_test.py（沿用前审计 P0 项） | Testing | P1 | 本周 |

---

## ⚠️ 待完善 / 已知局限

- 字段名差异（`reasoning` vs `reasoning_content`）为 vLLM 0.25.2.dev0 实测行为，未对照该版本源码确认是否可配置输出字段名
- enable_thinking 触发条件为实测确认，未验证 Chat Completions 与 Responses 两路由行为是否一致（responses 路由此前剥离 reasoning 的修复已透传，但未做思考触发验证）
- 网关未做任何改动（保持 v1.4.0 修复后状态），字段适配待决策后实施

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE 工程师）原始产出**：双机重启时间线、容器状态、5 项验证结果、字段名差异发现（`reasoning` vs `reasoning_content`、enable_thinking 触发）
- **前置报告**：`fix-sync-remote-2026-08-04.md`、`diagnostic-thinking-collapse-failure-2026-08-04.md`、`code-review-api-spec-compliance-2026-08-04.md`
- **API key 资料**：源自远端 `responses-gateway.service` 实测（客户端 key `<API_KEY>-*` / 内部 key `<API_KEY>-*`）

---

> 本报告由工程保障团队 AI 协作生成，关键决策（字段名适配方案、enable_thinking 开关）请由人类工程负责人复核。
