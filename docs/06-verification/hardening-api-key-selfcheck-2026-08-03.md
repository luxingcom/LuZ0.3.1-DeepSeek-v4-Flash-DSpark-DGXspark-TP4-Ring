# 8001 安全加固与持久化自检报告

**日期**：2026-08-03
**工作流**：部署前检查 / 安全加固（工作流 4 变体）
**参与成员**：Rex（加固实施与自检）/ Tessa（独立认证验证）/ 主理人（编排汇编）

---

## 📌 TL;DR（执行摘要）

- **8001 API key 认证加固完成（方案 2 分层密钥）**：8001 内部 key + 网关 UPSTREAM_API_KEY 注入——客户 key 泄露也无法直连 8001（实测 401，纵深防御）
- **最大并发限制确认**：`--max-num-seqs 6`（scheduler 序列上限，超限排队不拒绝）
- **持久化完成**：deep_gemm JIT 缓存卷（冷启动 370s vs 426s）、GSM8K 数据、配置双端归档
- **独立验证 GO**：SRE 自检 12 项 + 泰莎认证/并发用例全部 PASS（无/错/客户 key 401、内部 key 200、网关端到端 200、7 并发排队、800k 边界 400）
- 严重度分布：🔴严重 0 / 🟠高 0 / 🟡中 0 / 🟢低 3（遗留项：autotune 缓存、systemd、密钥明文）
- 阻塞 / 非阻塞：**非阻塞（GO）**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 GO（加固 + 验证闭环） |
| 认证体系 | 8001 内部 key + 网关客户 key（双层） |
| 并发限制 | max-num-seqs 6（排队机制） |
| 持久化 | JIT 缓存卷 / GSM8K 数据 / 配置归档 |
| 建议下一步 | 按需处理遗留项（autotune 缓存卷、网关 systemd、密钥 vault） |

---

## 🔐 加固实施（Rex）

### API key 方案（方案 2 分层密钥）
- **理由**：.77 事件根因是 8001 匿名直连；方案 1（共用客户 key）下客户 key 泄露即可直打 8001，复现同类事故且绕过网关日志/别名层
- **实施**：
  - 8001 vLLM：双机 SERVE_CMD 加 `--api-key <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa`（内部密钥）
  - 网关 main.py v1.1.0：`UPSTREAM_API_KEY` 环境变量，透传注入 `Authorization: Bearer <UPSTREAM_API_KEY>`（原网关 `_DROP_HEADERS` 丢弃 authorization——两方案都需改，方案 2 增量成本≈0）；未设置时回退 API_KEY（可一键回退方案 1）
  - 网关启动脚本 `~/responses_gateway/start_gateway.sh` 固化全部 env
- **安全收益**：客户 key（<API_KEY>-64b0374c6f2840fe）只能走 8003 网关；8001 只认内部 key（实测客户 key 直连 8001=401）

### 最大并发限制
- `--max-num-seqs 6` 即 scheduler 同时处理序列数上限，超限按 capacity 排队（不拒绝）
- 证据：运行参数、启动日志 `Maximum concurrency for 800,000 tokens per request: 6.56x`、/metrics `num_requests_waiting{reason=capacity}`

### 持久化清单

| 项 | 位置 | 说明 |
|---|---|---|
| deep_gemm JIT 缓存卷 | 双机 `~/vllm-cache/deep_gemm` → 容器 `/root/.cache/vllm/deep_gemm:rw` | 预置 109 个 sm_121a kernel；移除启动 rm -rf；重启 0 次 nvcc 编译 |
| GSM8K 数据 | head `~/data/gsm8k_test.jsonl`（原 /tmp 会丢） | md5 6493e22f... 一致 |
| 配置归档 | 本机 `集群部署/hardened/live/` + 双机 `~/hardened/live/` | README+脚本+网关代码，双端 cmp 全 MATCH |
| 防火墙 | 无拦截规则（用户决策不加白名单） | 双机 INPUT 默认 ACCEPT，安全依托 api-key |

## ✅ 自检结果（Rex 12 项 + Tessa 独立验证，全部 PASS）

| # | 项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 8001 无 key | 401 | curl `{"error":"Unauthorized"}` |
| 2 | 8001 内部 key | 200 | /v1/models + chat "OK" |
| 3 | 8001 客户 key | 401（纵深防御） | 客户 key 直连 8001 被拒 |
| 4 | 网关 8003 端到端 | 200 | 客户 key 经网关透传输出 "OK" |
| 5 | 网关 8003 无 key/错 key | 401 | invalid_api_key |
| 6 | 网关流式 | 200 | SSE 9 事件含 response.completed |
| 7 | 并发超限（7>6） | PASS | 7 并发全 200 无 429，排队分布 |
| 8 | 800k 边界 | PASS | max_tokens=801000 → 400 清晰错误 |
| 9 | 容器 healthy | PASS | 双机 docker ps healthy |
| 10 | JIT 缓存复用 | PASS | 启动 0 次 nvcc；缓存 109→110 |
| 11 | 冷启动 | PASS | 370s vs 上版 ~426s |
| 12 | GSM8K 持久 | PASS | md5 一致 |
| 13 | 配置归档 | PASS | hardened/live/ 双端 MATCH |

## ⚠️ 遗留项（非阻塞）

| # | 项 | 建议 |
|---|---|---|
| 1 | flashinfer autotune cache（~1min 冷启动）未持久化 | 卷扩为整 `/root/.cache/vllm`（需一次重启） |
| 2 | 25000 master 口 head 监听 0.0.0.0 | 用户决策不加白名单，记录备查 |
| 3 | 密钥明文存脚本 | 如需更高级可接 KMS/vault |
| 4 | 网关为 nohup 进程（非 systemd） | 主机重启后需手动拉起 start_gateway.sh；可选做 systemd user service |
| 5 | 更换镜像/wrapper/架构后需清 JIT 缓存 | 运维注意事项 |

## 📚 数据来源 & 成员产出索引

- Rex：加固实施（方案 2 分层密钥、JIT 缓存卷、GSM8K/配置持久化）、12 项自检、回滚锚点（脚本 .bak.20260803_011837 等）
- Tessa：独立认证/并发验证（.tessa_test/REPORT.md）、4 类用例全 PASS、观察项（thinking 不返回独立 reasoning 字段，模型固有行为）
- 主理人：编排、方案 2 决策、汇编

---

> 本报告由工程保障团队 AI 协作生成，关键决策（遗留项处理、vault 引入）请由人类工程负责人复核。
