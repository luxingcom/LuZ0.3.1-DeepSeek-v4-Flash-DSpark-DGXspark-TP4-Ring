# 生产环境综合收尾 + 网关端点 Benchmark 报告

**日期**：2026-08-06
**工作流**：综合收尾（状态复核 / 待办落实 / 清理修复加固 / 持久化韧性验证 / 网关压测）
**参与成员**：Rex（复核）/ Tessa（matrix 基线 + 网关压测）/ Cody（思考链修复）/ Docu（约定固化与清单）/ Zhen（编排执行清理与加固）

---

## 📌 TL;DR

- ✅ **服务器状态复核全绿**：双机全服务正常，持久化（卷/备份/restart）达标；思考链 8003 修复 + NCCL 加固已上线
- ✅ **清理修复加固完成**：docker 空间 -105GB、脚本归档、备份异地互备、NCCL_IB_TIMEOUT/RETRY_CNT
- ✅ **韧性实测结论明确**：TP=2 集群恢复 = 双机脚本重启（docker kill 不触发重启为预期行为）
- ✅ **网关 benchmark 0 错误**：8003 单流延迟更优且唯一保留思考链；4000 高并发吞吐更优
- ⏳ 遗留 3 项需用户拍板（防火墙 / 密码轮换 / 管理网有线）

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 收尾完成，生产稳定运行 |
| 阻塞项 | 0 |
| 关键行动项 | 5 条（3 条需用户拍板） |
| 建议下一步 | 按行动清单逐项关闭；双轨网关按负载分流使用 |

---

## 1️⃣ 服务器状态复核（Rex，只读）

| 维度 | 状态 | 要点 |
|------|------|------|
| 服务健康 | ✅ 全绿 | head 8001 / worker 8003 / LiteLLM 4000 / PG 5432 / 双机容器 healthy |
| 资源 | ⚠️ 观察 | 双机内存可用 10-13G（服务占高）；磁盘余量充足（head 20% / worker 30%） |
| 持久化 | ✅ 达标 | vllm-cache / tilelang-cache / models 等卷全挂；restart 策略全 unless-stopped；备份 cron 正常 |
| 安全 | ⚠️ 3 缺口 | NCCL 超时未设（本轮已修复✅）；worker 管理网走 WiFi；master_key/PG 密码明文 |

## 2️⃣ 待办落实（3 项全完成）

| 待办 | 状态 | 结果 |
|------|------|------|
| temp>0.1 温度约定固化 | ✅ | 手册新增 §2.5 Temperature Contract（Python/Node 示例 + 服务端现状 + 加固占位） |
| 新基线完整 matrix 固化 | ✅ | GSM8K 95.0%（temp=0.6 口径）、12 组合 0 错误、c5 聚合 80.8-96.5 t/s、负载 code 131.8/json 122.5 → bench-f-baseline-2026-08-05.md |
| 思考链走 8003 确认与修复 | ✅ | **根因**：F 配置仅在 enable_thinking=true 生成思考链、网关从不注入 → **修复**：网关 v1.5.0 新增 _inject_enable_thinking()（chat+responses 双路由，显式 opt-out 保留）→ 验证：responses 恢复 type=reasoning 事件、chat 恢复 reasoning_content |

## 3️⃣ 清理 / 修复 / 加固（Zhen 代执行 + Rex 协同）

| 类别 | 项 | 结果 |
|------|-----|------|
| 清理 | docker 空间 | head 镜像 283→236GB + buildcache 110→49.6GB；生产镜像（:0.2.1/:0.2.0/:0.1.1/hybrid）全保留 |
| 清理 | 脚本归档 | 双机 24 个实验脚本归档至 ~/archive_scripts/，仅保留 E 回滚 + v026r 生产 |
| 修复 | **NCCL 加固** | 双机 v026r 脚本追加 `NCCL_IB_TIMEOUT=1000` + `NCCL_IB_RETRY_CNT=7`（容器 env 实测生效，RoCE hang 防护） |
| 加固 | **备份异地互备** | head 每日 03:05 cron 拉取 worker 的 LiteLLM PG 备份（head→worker 免密 + 实测成功） |
| 加固 | 8020 确认 | embed-gpu 容器 docker-proxy 0.0.0.0:8020，建议绑 127.0.0.1（待排期） |

## 4️⃣ 持久化 / 自启 / 恢复能力验证（实测结论）

| 项 | 结论 |
|----|------|
| 持久化 | ✅ vLLM 双机 7 卷 + LiteLLM/PG 卷 + 备份 cron + 异地互备全确认 |
| 自启能力 | ✅ restart policy 全 unless-stopped（daemon 重启自动拉起） |
| **崩溃恢复** | ⚠️ **TP=2 集群恢复 = 双机脚本重启（worker 先 → head 后）**——docker kill（显式停止）不触发重启为预期行为（unless-stopped 语义）；容器内进程崩溃也不会自动恢复集群（NCCL 失联） |
| **关键运维经验** | vLLM TP 集群启动**偶发失败**（本轮 4 次重启 3 次卡 NCCL init，第 4 次成功；无 ERROR、RoCE 链路 0 错误、worker 侧 TCPStore IPv6 fdff:: 连接 head 无此地址）→ 属已知问题，双机重试即可恢复；**单边重建 head 无法恢复**（worker 失联 Broken pipe） |

## 5️⃣ 网关端点 Benchmark（Tessa，~170 请求 0 错误）

### 8003 vs 4000 对比（表 3 核心）

| 端点 | 并发 | 8003 | 4000 | 结论 |
|------|------|------|------|------|
| /v1/models | c1 | 28.9ms | 32.1ms | 持平 |
| chat 非流式 200→64 | c1 | **1303ms** | 1973ms | 8003 单流 -34% |
| chat 非流式 200→64 | c5 | 41.6 t/s | **83.3 t/s** | 4000 并发 +100% |
| chat 流式 | c1 | TTFT 367ms（content 延迟至思考链后） | TTFT 366ms（即时 content） | 4000 流式体验更佳 |
| /v1/responses | c1 | 1335ms **保留 reasoning** | 1964ms reasoning=0 | **思考链必须走 8003** |
| /v1/embeddings | c1/c10 | 45/88ms | 31/132ms | 4000 单流快，8003 并发稳 |

### 思考链专项
- 8003 responses 流式稳定输出 `reasoning_part.added → reasoning_text.delta ×13-23 → done`
- math 类思考开销 ~0（等长对比 2044 vs 2016ms）；trivial 请求 8003 强制思考 +1489ms（7.8×）——**思考链有成本，按需分流**
- 异常：4000 prob key 调 embedding 401（key 权限隔离，预期行为）；8003 `enable_thinking:false` 不生效（思考不可关）；8003 responses `usage.reasoning_tokens` 恒 0（以 output 事件为准）

### 双轨分流结论
**结构化/高吞吐/流式即时体验 → 4000（prob key）**；**思考链/Responses API/单流低延迟 → 8003**

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 防火墙 firewalld 白名单（内网段，需 sudo） | 用户+SRE | P1 | 用户授权后 |
| 2 | master_key / PG 密码轮换 + 去明文（改 env 注入） | SRE | P1 | 用户授权后 |
| 3 | worker 管理网回有线（当前 WiFi 79ms 抖动单点） | 用户+SRE | P1 | 物理接线后 |
| 4 | 8020 绑定 127.0.0.1（嵌入服务仅本机消费） | SRE | P2 | 下一维护窗 |
| 5 | 收尾清单回填（production-finalize-checklist 更新 ✅ 状态） | Docu | P2 | 随本报告 |

## ⚠️ 待完善 / 已知局限

- **vLLM TP 集群启动偶发失败率偏高**（4 次重启 3 次卡 NCCL）——建议后续专项调查（疑与多次快速重启后 RoCE 流表/TCPStore 端口状态相关），可考虑固化"双机重启 SOP + 重试策略"
- 8003 思考链强制开启（enable_thinking:false 不生效）——若需关闭思考的负载请走 4000 greedy key
- 4000 的 responses 思考链剥离（LiteLLM 层）——依赖思考链的客户端固定走 8003
- 新基线 GSM8K 95.0% 为 temp=0.6 口径（相对 temp=0 的 99.0% 有 -4pp 多样性代价，属预期）

## 📚 数据来源 & 成员产出索引

- Rex（SRE）：服务器复核报告（服务/持久化/安全清单）
- Tessa（测试）：`bench-f-baseline-2026-08-05.md`（matrix 基线）、`bench-gateway-endpoints-2026-08-06.md` + `_tessa_gateway_bench_raw_2026-08-06.txt`（网关压测）
- Cody（代码审查）：网关 v1.5.0 `_inject_enable_thinking`（`hardened/live/responses_gateway_main.py`）
- Docu（文档）：`litellm-api-key-manual-2026-08-05.md`（§2.5 温度约定）、`production-finalize-checklist-2026-08-05.md`
- Zhen（编排）：清理/加固执行记录、`_archive_scratch/`、NCCL 加固脚本（`start_*_v026r.sh` + `.bak.20260806_*`）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
