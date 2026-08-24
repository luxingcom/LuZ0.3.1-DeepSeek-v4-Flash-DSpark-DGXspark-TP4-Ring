# DeepSeek V4 Flash 双 DGX Spark 集群服务器保障状态质量核查报告

**日期**：2026-08-04
**工作流**：保障状态质量核查（运维自检 + 代码审查 + 测试验证 综合审计）
**参与成员**：Cody（代码审查师）/ Rex（SRE 工程师）/ Tessa（测试专家）

> 说明：本报告基于工作区既有日志/配置/测试证据与线上文件的真实审读，**未对双机现网做任何变更或真机高负载探测**。一切结论来源于可复现证据，读不到的状态已标注 ⚠️ 而非臆造。

---

## 📌 TL;DR（执行摘要）

- **整体结论**：双 DGX 集群**运行态健康**——三层鉴权、持久化、流式透传、防火墙、工具调用均可用；但存在**嵌入式 GPU→CPU 基线偏离、归档脚本与实况漂移、worker 拓扑 IP 记录错误**三处配置级风险，且工具调用/压测/600k 边界**「声称已验、缺可审证据」**。
- **严重度分布**：🔴严重 0 项 / 🟠高 4 项 / 🟡中 5 项 / 🟢低 2 项
- **性质**：**非阻塞** —— 无 SEV1/SEV2，无需要紧急停服的项；全部为运行期可控的加固/补证项。
- **整体评级**：🟡 黄灯 / 有条件通过。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 黄灯（有条件通过） |
| 阻塞项数量 | 0（无 SEV1/SEV2） |
| 高价值行动项 | 5 条 |
| SEV 评级 | SEV3：嵌入 GPU→CPU 偏离 + 脚本/拓扑漂移（团队级风险，不影响对外服务） |
| 建议下一步 | 先消除嵌入式 GPU↔CPU 基线漂移，再以实况为单一事实源重写归档启动脚本，随后补足测试证据 |

---

## 🔍 各组审计发现（按严重度合并去重后排序）

| # | 严重度 | 类别 | 归属 | 问题描述 | 建议修复 |
|---|--------|------|------|---------|---------|
| 1 | 🟠高 | 配置基线 | Rex | **嵌入后端实况运行在 CPU 模式**（torch==2.13.0+cpu、health `device:cpu`、存在 rollback_cpu.sh），而既定基线为 GPU Qwen3-Embedding。端点正常但能力/性能偏离基线 | 落地 embed-qwen3-gpu 真 GPU 并回归；或正式将 CPU 定为基线并更新文档，消除漂移 |
| 2 | 🟠高 | 配置/脚本 | Cody+Rex | **工具调用参数未落地归档启动脚本**：start_{head,worker}_E.sh 实测无 `--enable-auto-tool-choice`/`--tool-call-parser`；且 fix_toolcall.py 只在含 `--gpu-memory-utilization 0.80` 的行后插入，而归档 E 脚本实为 0.9 → **补丁脱靶**。归档≠实况，重建/回滚会丢失工具调用能力 | 以实况为准重写启动脚本（正确 gpu-mem + 工具参数），建立单一事实源 |
| 3 | 🟠高 | 安全 | Cody | **凭证散落 + 日志泄露**：SERVE_CMD 硬编码 `--api-key` 且 `echo "$SERVE_CMD"` 把 key 打进日志；内部/推送 key 散落 4+ 处（含 embed_main.py 源码硬编码 EMBED_API_KEY 默认值），轮换易漏 | key 全面 env/secret 化，代码零硬编码，echo 时脱敏 |
| 4 | 🟠高 | 拓扑/文档 | Rex | **worker 拓扑 IP 记录漂移**：基线记录 worker=<NODE_IP>，但全部证据为 **<NODE_IP>**（route_worker src .58、ping52 .58、ledger worker58） | 修正 worker 实际 IP 并统一拓扑记录，确认单点事实源 |
| 5 | 🟡中 | 测试证据 | Tessa | **工具调用全流程证据缺失**：gateway 200 有声称，但原始 `tool_calls`/`get_weather` JSON 响应未持久化，且「回传结果→二次完成」round-trip 未确认 | P0 补 1 次全流程：记录 8003 带 tools+auto 原始 JSON 存证 + 二轮回传 |
| 6 | 🟡中 | 测试证据 | Tessa | **压测未落地**：stress_test.py 存在但无任何运行输出，无高并发/p95 | P0 运行 stress_test.py（并发≥20，采 p95/错误率/超时）并存输出 |
| 7 | 🟡中 | 测试证据 | Tessa | **当前 600k 配置边界未复测**：max_len=600000，但 benchmark 明言未触及上限；`max_tokens=601000→400` 与 600k 长输入单请求均未实测（边界 400 是旧 800k 环境测得） | P1 在当前配置下补边界回归（601000→400；600k 长输入） |
| 8 | 🟡中 | 测试/容错 | Tessa | **上游故障路径未验证**：8001 down→网关 502/超时/重试行为从未测试（近期多次 503/500，网关容错是关键单点） | P1 上游故障注入：停 8001→验证网关 502+恢复自愈（低风险只读观察） |
| 9 | 🟡中 | 安全 | Cody | **鉴权细节待加固**：`_check_auth` 用 `==` 非恒定时间比较（时序侧信道）；`:70` UPSTREAM_API_KEY 缺省回退 API_KEY，若配置漏设则两层鉴权退化为单 key | hmac.compare_digest；缺省即拒绝启动，不静默回退 |
| 10 | 🟡中 | 正确性 | Cody | **两路由校验不一致**：/v1/responses 用 `MODEL_ALIAS.get(model,model)` 未知模型放行；/v1/chat/completions 严格 404 | 统一为严格 404 |
| 11 | 🟢低 | 安全 | Cody | `_DROP_HEADERS` 仅含 authorization，x-api-key/proxy-authorization 等含 key 头未脱，可透传上游 | 扩充脱敏头集合 |
| 12 | 🟢低 | 性能 | Cody | 网关 `timeout=None` 且无并发上限 / body 大小上限，上游挂起时连接与内存堆积 | 加读超时 + body 上限 + 并发信号量 |

---

## 🏗️ 运行态基线核查（Rex）

| 服务 | 端口 | 鉴权 | 持久化 | 健康 | 证据 |
|---|---|---|---|---|---|
| vLLM head | 8001 | 内部key <API_KEY>-*（客户key亦401=纵深防御） | docker --restart unless-stopped + docker.service enabled + deep_gemm JIT 卷 | ✅ Up 3h healthy，内部key=200 | rename-model、sre_v31 |
| vLLM worker | — | 同 | 同 | ✅ Up 3h healthy | sre_v30 |
| 响应网关 | 8003 | 客户key <API_KEY>-*（无key=401）+ 注入内部key至8001 | systemd user + linger + Restart=always | ✅ enabled+active，全链路200 | persistence-ledger、responses-gateway.service |
| 嵌入后端 | 8020 | 经网关鉴权，直连无独立鉴权 | systemd user + linger | ⚠️ **CPU 模式**（非 GPU 基线） | embed_requirements=cpu、sre_v29 device:cpu、rollback_cpu.sh |
| fw-25000 | 25000 | 仅内网数据面 | docker restart | ✅ Up healthy | sre_v31 |
| 监控栈 | 3000/8191/9400/9100 | — | compose unless-stopped | ✅ 200 | sre_v30、ledger |

**基线达成项**：✅ 三层鉴权 ✅ 25000 防火墙 ✅ systemd+linger ✅ docker restart ✅ deep_gemm 持久卷
**SEV 评级**：无 SEV1 / 无 SEV2；**SEV3**（团队级配置/文档风险，不影响对外服务）

---

## 🧪 测试覆盖评估（Tessa）

| 测试项 | 脚本/证据 | 状态 | 备注 |
|---|---|---|---|
| 三通道功能 | chat 08-04 新路由 10/10；responses .tessa_test REPORT；embeddings tc3/tc4* | ✅ | chat 通道新路由已回归 |
| 鉴权三层 | .tessa_test REPORT + test_results.log | ✅ | 8001 无/错/客户key 均401；8003 无/错401、客户key 200；UPSTREAM 注入透传验证 |
| 未知模型404 | rename-model doc#6、chat doc#4 | ✅ | 404 model_not_found |
| 工具调用 tools+auto | 仅 memory叙述 | 🟡 | 有声称，原始 tool_calls 未持久化 |
| 非工具回归 | chat doc#7/8/9、memory | ✅ | |
| 长序列600k | benchmark-E600k；dl10b仅加载 | 🟡 | 未触及上限，600k 单请求实测缺失 |
| 并发/压测 | .tessa_test 7路 | 🟡 | 仅7并发； stress_test.py 无输出 |
| 错误处理 | 无效JSON偶遇 | ⚠️ | 无正式用例；上游502/超时从未测 |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 消除嵌入式 GPU↔CPU 基线漂移：落地 embed-qwen3-gpu 真 GPU 并回归，或正式将 CPU 定为基线并更新文档 | SRE | P0 | 本周 |
| 2 | 以实况为单一事实源重写归档启动脚本（正确 gpu-mem + `--enable-auto-tool-choice` + `--tool-call-parser deepseek_v4`），防重建/回滚丢工具调用 | SRE+Cody | P0 | 本周 |
| 3 | 补工具调用全流程存证 + 二轮回传；运行 stress_test.py（并发≥20 采 p95） | Testing | P0 | 本周 |
| 4 | 密钥全面 env/secret 化，启动脚本 echo 脱敏；修正 worker 实际 IP 拓扑记录（.58 覆盖 .61） | Cody+SRE | P1 | 2 周内 |
| 5 | 统一 `/v1/responses` 与 `/v1/chat/completions` 严格 404 校验；`_DROP_HEADERS` 扩充；鉴权改恒定时间比较 | Cody | P1 | 2 周内 |

---

## ⚠️ 待完善 / 已知局限

- 本核查为**只读证据审计**：嵌入后端 GPU→CPU、vllm-envE 双机真机重启演练、上游故障注入等**未做实机验证**，仅据配置/日志判断。
- worker 拓扑 IP（.58 vs .61）存在记录争议，需现场确认真实地址。
- 工具调用压测、600k 边界、上游故障容错等「声称已验」项缺可审计的原始输出文件，需补证。
- 8001 早期(8/2) health 失败为重启窗口瞬时态，非当前故障。

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）原始产出**：痛点清单 10 项 + 架构影响评估（v1.4.0 三层鉴权健壮、风险集中在配置/运维脆土；工作区 hardened_main_gateway.py 为过期 v1.2.0 快照，线上实为 hardened/live/ v1.4.0）
- **Rex（SRE 工程师）原始产出**：服务清单 6 项自检表 + 保障基线差距 + SEV3 评级 + 5 条行动项
- **Tessa（测试专家）原始产出**：测试覆盖矩阵 8 行 + 5 项回归缺口 + 5 条补测建议

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
