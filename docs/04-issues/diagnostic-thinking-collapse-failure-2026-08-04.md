# WorkBuddy 思考内容无法折叠问题根因诊断报告

**日期**：2026-08-04
**工作流**：事故响应（根因诊断，非正式工作流）
**参与成员**：甄宇航（主理人，直接分析）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：WorkBuddy 折叠思考内容依赖 Chat Completions 响应中的 `reasoning_content` 字段，但 vLLM 8001 的 reasoning parser **完全未启用**（`reasoning_parser=''`，`enable_in_reasoning=False`），导致思考内容作为纯文本混入 `content` 字段，WorkBuddy 无法识别和折叠。
- **严重度**：🟠高（影响用户体验，但不影响功能正确性）
- **根因**：`start_head_E.sh` / `start_worker_E.sh` 启动脚本缺少 `--reasoning-parser deepseek_v4` 参数。
- **修复**：在 SERVE_CMD 中添加 `--reasoning-parser deepseek_v4`，重启 vLLM 后 `reasoning_content` 字段将正常返回。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（根因已定位，修复方案明确） |
| 阻塞项数量 | 0 |
| 关键行动项 | 1 条（添加 --reasoning-parser 参数） |
| 建议下一步 | 双机 SERVE_CMD 添加 `--reasoning-parser deepseek_v4`，重启验证 |

---

## 🔍 根因分析

### 问题现象
WorkBuddy 通过 local-v4-flash 模型调用时，思考内容（reasoning/thinking）无法折叠，直接显示在正文内容中。

### 根因链

| # | 环节 | 证据 | 状态 |
|---|------|------|------|
| 1 | WorkBuddy 折叠思考内容依赖 `reasoning_content` 字段 | DeepSeek 官方 Chat Completions 响应格式：`message.reasoning_content` 含思考过程，`message.content` 含最终回答 | 规范确认 |
| 2 | vLLM 8001 的 reasoning parser 未启用 | vLLM 启动日志：`StructuredOutputsConfig(reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False)` | 🔴 根因 |
| 3 | 思考内容混入 content 字段 | 无 reasoning parser 时，vLLM 将思考内容作为纯文本放入 `content`，不分离到 `reasoning_content` | 推导确认 |
| 4 | 启动脚本缺少 --reasoning-parser 参数 | `start_head_E.sh` SERVE_CMD（行 33-50）无 `--reasoning-parser`；`start_worker_E.sh` 同 | 代码确认 |
| 5 | 网关透传不影响此行为 | 网关 `/v1/chat/completions` 路由做裸透传，不修改响应体，vLLM 返回什么 WorkBuddy 就收到什么 | 代码确认 |

### 关键证据

**vLLM 启动日志**（`head_log_tail3.txt:27`）：
```
StructuredOutputsConfig(
    backend='auto',
    ...
    reasoning_parser='',
    reasoning_parser_plugin='',
    enable_in_reasoning=False
)
```

**启动脚本**（`start_head_E.sh:33-50`）SERVE_CMD 中：
- ✅ 有 `--tool-call-parser deepseek_v4`（工具调用已启用，但据科迪审查实际可能未落地到脚本）
- ❌ **无** `--reasoning-parser deepseek_v4`（思考内容解析未启用）
- ❌ **无** `--default-chat-template-kwargs.thinking=true`（引擎级思考开关未显式设置）
- ❌ **无** `--default-chat-template-kwargs.reasoning_effort=max`（引擎级推理强度未显式设置）

**网关代码**（`responses_gateway_main.py:272-336`）：
- `/v1/chat/completions` 路由做裸透传，不修改响应体
- 不影响 `reasoning_content` 字段的有无

### 代码中的佐证

工作区测试脚本 `bench_smoke.py:44` 同时检查 `reasoning_content` **和** `reasoning` 两个字段：
```python
if content.strip() or d.get('reasoning_content') or d.get('reasoning'):
```
这说明团队已知 vLLM 可能返回 `reasoning` 或 `reasoning_content` 两种字段名，但未启用 reasoning parser 时两者都不会正确填充。

### 为什么 thinking 参数也缺失？

benchmark 报告（`benchmark-four-env-dspark-2026-08-02.md`）称"4/4 环境进程参数均含 `--default-chat-template-kwargs.thinking=true`"，但 `start_head_E.sh` 的 SERVE_CMD 中实际**不含**此参数。这与科迪此前发现的"工具调用参数未落地到脚本"是**同一类问题**：参数可能在镜像 CMD 中，但被 heredoc SERVE_CMD 覆盖后丢失。

**需要确认**：vLLM 镜像的默认 CMD 是否包含 thinking 参数。如果 SERVE_CMD 覆盖了 CMD，则 thinking 参数也丢失了。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 双机 `start_{head,worker}_E.sh` SERVE_CMD 添加 `--reasoning-parser deepseek_v4`，重启 vLLM 后验证 `reasoning_content` 字段返回 | SRE | P0 | 立即 |
| 2 | 确认 vLLM 镜像 CMD 是否含 thinking 参数；若 SERVE_CMD 覆盖了 CMD，则需同时添加 `--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=max` | SRE+Cody | P0 | 立即 |
| 3 | 重启后在 WorkBuddy 中验证思考内容可折叠 | 用户 | P0 | 重启后 |

---

## ⚠️ 待完善 / 已知局限

- 本诊断基于启动脚本静态分析和 vLLM 启动日志，未实机验证添加 `--reasoning-parser` 后的行为。
- `--reasoning-parser deepseek_v4` 是否是 vLLM 0.25.2.dev0 支持的 parser 名称需确认（参考 `--tool-call-parser deepseek_v4` 的注册方式）。
- thinking 参数是否被镜像 CMD 隐式提供需确认（如果 CMD 有而 SERVE_CMD 覆盖了，则需显式添加）。
- 本报告由主理人直接分析，未经团队成员独立审查。

---

## 📚 数据来源 & 成员产出索引

- **证据文件**：`head_log_tail3.txt:27`（vLLM 启动日志）、`start_head_E.sh:33-50`（SERVE_CMD）、`responses_gateway_main.py:272-336`（网关透传）、`bench_smoke.py:44`（字段兼容性佐证）
- **关联报告**：`code-review-api-spec-compliance-v2-2026-08-04.md` #3（reasoning 字段剥离）、`assurance-audit-dspark-cluster-2026-08-04.md`（工具调用参数未落地）

---

> 本报告由工程保障团队主理人直接分析生成，关键决策请由人类工程负责人复核。
