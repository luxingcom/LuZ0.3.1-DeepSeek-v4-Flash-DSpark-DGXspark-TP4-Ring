# 双 DGX Spark vLLM 生产集群运维手册（Runbook）

- **版本**：1.1（2026-08-06 23:58 修订）
- **v1.1 修订摘要**（2026-08-06 23:58）：① 新增 §0「当前状态」——LLM 双机停机（配合视频工作流释放 GPU）、embed 在线、恢复预案；② §1 补 4000 网关综合性能基线（27/27 0 错误、GSM8K 94.5%）+ RoCE/IB 结论（固件锁定不可切）；③ §3 新增方法学规范（随机前缀强制、温度统一、LiteLLM 真实开销 -6.8%~+6.9%）；④ §4 新增「大 ctx 吞吐下降」排查（prefill 主导 + prefix-cache 假象）
- **适用**：双 DGX Spark（GB10 sm_121a）TP=2 vLLM 生产集群，F 方案（probabilistic + 动态K + tilelang 两档 + b12x + nvfp4_ds_mla）
- **读者**：SRE / 运维 / 新成员
- **⚠️ 敏感**：本文含明文 API Key，仅限内部授权人员，禁止外发
- **维护**：工程保障团队；任何变更后 24h 内回填本手册

---

## 0. 当前状态（2026-08-06 23:58）⚠️

> **LLM 双机当前停机中**（配合视频工作流部署释放 GPU），**embed 服务在线**；chat 类调用不可用属预期。恢复时机由人类工程负责人协调。

| 服务 | 状态 | 说明 |
|------|------|------|
| vLLM head（8001） | ⛔ 停机 | `vllm-envE-node` Exited (0) 正常退出，8001 未监听 |
| vLLM worker | ⛔ 停机 | `vllm-envE-worker` 容器已移除 |
| 主模型权重 | ⛔ 已卸载 | 双机权重（~79GB/机）随容器释放给视频工作流 |
| **embed（8020）** | 🟢 **在线** | `embed-qwen3-gpu` Up；直连 8020 = 200 |
| 4000 / 8003 embeddings | 🟢 200 | 两网关 → 8020 链路正常 |
| 4000 / 8003 chat 路由 | ⚠️ 预期失败 | 后端 vLLM 停（500/502），属预期 |

**恢复预案（固化）**：
```bash
# ① 前置确认：视频工作流已释放 GPU（LLM 需 ~98GB/机，共享 121GB 统一内存）
# ② 一键拉起（head 机执行编排脚本）：
ssh aicad-server60 'nohup bash ~/start_v026r_cluster.sh > ~/v026r_cluster_runN.log 2>&1 &'
# ③ 恢复后全链路验证：8001/8003/4000 = 200 + e2e chat "2+2=?" → "4" + embeddings 200
```
- embed 服务全程未动，恢复 LLM **无需**操作 embed；完整启动/验证流程见 §2

---

## 1. 服务全景

### 1.1 拓扑（逻辑）

```
                业务客户端  base_url=http://<NODE_IP>:{4000|8003}
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 LiteLLM 4000 (worker <MGMT_OCTET>)        自建网关 8003 (worker <MGMT_OCTET>)
 master/prob/greedy/chat/embed    客户端 key <API_KEY>-*
        └───────────────┬───────────────┘
                        ▼
      vLLM head <MGMT_OCTET> (rank0, :8001) ── TP=2 NCCL/RoCE ──► vLLM worker <MGMT_OCTET> (rank1)
                        ▲        (<NODE_IP> ↔ <NODE_IP>, TCPStore :25000)
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Embed 8020 (127.0.0.1)         PG 5432 (127.0.0.1)
 (worker <MGMT_OCTET>, embed-qwen3-gpu)  (LiteLLM 依赖，本机消费)
```

- head = <NODE_IP>（spark-05cd，RoCE <NODE_IP>）；worker = <NODE_IP>（edgexpert-0c69，RoCE <NODE_IP>）
- 容器均为 **host 网络**：head `vllm-envE-node` / worker `vllm-envE-worker` / embed `embed-qwen3-gpu`

**网络说明（RoCE/IB，08-06 调研定论）**：
- 双机 ConnectX-7（MT4129）**固件锁定仅 RoCE/Ethernet，不可切 IB**（NVIDIA 官方明确无支持计划）→ **保持 RoCEv2**
- 2 节点直连场景 IB 收益 <5-10%（无拥塞、错误计数 0，无损优势被抹平）；带宽瓶颈在 **PCIe Gen5 x4 ~96Gbps** 而非链路模式
- NCCL 卡死与 fabric 无关（init 顺序竞态，IB 也避免不了）；监控方向：`NCCL_DEBUG=INFO` 确认 GDRDMA、MTU 9000、链路错误计数（当前 0）

### 1.2 服务清单

| 服务 | 主机 | 端口/地址 | 鉴权 key | 镜像/版本 | restart/守护 |
|------|------|-----------|----------|-----------|--------------|
| vLLM head（rank0） | <MGMT_OCTET> | 8001 | `<API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa` | `ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0`（vLLM 0.26.1dev） | unless-stopped |
| vLLM worker（rank1） | <MGMT_OCTET> | 无对外端口 | 同上（内部） | 同上 | unless-stopped |
| 自建网关 | <MGMT_OCTET> | 8003 | `<API_KEY>-64b0374c6f2840fe` | responses_gateway v1.5.0（hardened/live） | systemd active |
| LiteLLM | <MGMT_OCTET> | 4000 | 见 §6 双轨 key 表 | LiteLLM 1.83.7 | 容器 unless-stopped |
| PostgreSQL | <MGMT_OCTET> | 5432（仅 127.0.0.1） | PG 明文（待轮换） | LiteLLM 依赖 | 卷 + 备份 cron |
| Embed 服务 | <MGMT_OCTET> | 8020（已绑 127.0.0.1） | 内部（仅网关转发） | `embed-gpu:anemll-0.1.1-st5.6.1` | systemd + unless-stopped |

> 模型 `deepseek-v4-flash-0731`（SERVED）；seqs=6，600K ctx，spec decode：probabilistic + 动态K `[[1,1,5],[2,4,4],[5,6,3]]` + tilelang 两档 patch + `TILELANG_CACHE_DIR` 持久卷。

### 1.3 模型与性能基线速览（08-05 直连固化 + 08-06 网关补充）

- **GSM8K 95.0%**（190/200，temp=0.6 口径；旧 greedy temp=0 = 99.0%，-4.0pp 系多样性代价，非回归）
- **c5 聚合 80.8–96.5 t/s**；c1 单流 34.6–43.1 t/s；负载类型（c5）：**code 131.8 / json 122.5 / prose 81.7 t/s**
- **网关分流**：8003 单流更快（-34%）且**唯一保留思考链**；4000 高并发吞吐更优（c5 83.3 t/s，+100%）且流式无思考延迟

**4000 网关综合基线（bench-gw4000 08-06）**：
- coding/json/散文 × c1/c3/c5 × 512/8192/32768 = **27/27 组合 0 错误 0 超时**；GSM8K **94.5%**（189/200，-0.5pp/1 题，无回退）
- c5/512：**code 101.2 t/s**（接受率 82.5%）/ **json 93.5**（85.1%）/ **prose 67.7**（41.4%）；档级接受率 512/8K/32K = 60.7/58.9/57.4%
- **口径说明**：此表为「PARA 随机前缀填充 + temp 0.7」口径，与上方直连基线（短 prompt + temp 0.6）**不可直接相减**；经 4000 严格同条件 A/B 真实开销仅 **-6.8%~+6.9%（中位 ~0.5%）**，此前 ~23% 系口径差异（详见 §3.5）

---

## 2. 启动 / 重启 SOP（权威流程）

> **铁律**：① 必须用编排脚本按「head 先 → worker 后」启动；② **禁止**手动 worker→head 顺序（NCCL 卡死根因 H1）；③ **禁止单边重建**（kill 后必须双机重来）；④ 重启前确认双机容器已停止、25000 无残留。

### 2.1 全量重启（head 机执行编排脚本）

```bash
# 0) 前置：确认双机 vLLM 容器已停止（见 2.2 停机）
# 1) 启动（head 机，nohup 后台，日志留存）：
ssh aicad-server60 'nohup bash ~/start_v026r_cluster.sh > ~/v026r_cluster_runN.log 2>&1 &'
# 2) 轮询就绪（权重加载约 5-8 分钟）：
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" \
  http://127.0.0.1:8001/v1/models      # 期望 200
```

**编排脚本流程**（`~/start_v026r_cluster.sh`，head 机）：
1. 前置检查：双机容器须已停止
2. 启动 head(rank0) 容器（nohup，日志 `~/start_head_v026r_cluster.log`）
3. 轮询 head TCPStore :25000 就绪：`nc -z <NODE_IP> 25000`（或宿主 `ss -tln`），每 5s、最多 10min（实测 ~25s）
4. `ssh node0X` 执行 `~/start_worker_v026r.sh` 启动 rank1
5. 轮询 head :8001 `/v1/models` 就绪（internal key，每 5s、最多 10min）
6. 输出集群就绪

> 实测新顺序 **3/3 成功**（API 就绪 4m36s–5m18s，8001/8003/4000 全 200，e2e "2+2=?" → "4"）；旧顺序 4 次 3 挂。脚本含加固：`NCCL_IB_TIMEOUT=1000` + `NCCL_IB_RETRY_CNT=7`、`--distributed-timeout-seconds 300`、`VLLM_ENGINE_READY_TIMEOUT_S=600`、`NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE`（落 `~/vllm-logs/nccl-*.log`）。
> **停机后恢复**：若处于 §0 停机状态（配合视频工作流），先确认视频工作流已释放 GPU（LLM 需 ~98GB/机）再执行上方启动命令；恢复流程同 §0。

### 2.2 停机维护顺序（不可反）

```bash
ssh aicad-server 'docker rm -f vllm-envE-worker'   # 1) worker 先停
ssh aicad-server60 'docker rm -f vllm-envE-node'   # 2) head 后停
```

- 维护窗口内 8003 / 4000 网关与 LiteLLM 不受影响（仅转发后端不可用）
- 停机后重启一律走 2.1 编排脚本

### 2.3 就绪验证 + 全链路检查

```bash
# 容器 healthy（双机）
ssh aicad-server60 'docker ps --format "{{.Names}} {{.Status}}"' ; ssh aicad-server 'docker ps --format "{{.Names}} {{.Status}}"'
# 端口
for p in 8001 8003 4000; do curl -s -o /dev/null -w "port $p: %{http_code}\n" -H "Authorization: Bearer <对应key>" http://127.0.0.1:$p/v1/models; done
nc -z <NODE_IP> 25000 && echo "TCPStore OK"
# 推理 e2e：8001/8003/4000 均答 "4"（curl chat "2+2=?"）
# 思考链：8003 /v1/responses 输出应含 type=reasoning（4000 无 = 预期）
# 嵌入：4000/8003 /v1/embeddings 200，dim=1024
```

### 2.4 回滚方案

| 场景 | 步骤 |
|------|------|
| 启动脚本回滚（NCCL 修复） | head：`cp ~/start_head_v026r.sh.bak.20260806_ncclfix ~/start_head_v026r.sh`；worker：`cp ~/start_worker_v026r.sh.bak.20260806_ncclfix ~/start_worker_v026r.sh` |
| 编排脚本回滚 | 删除/弃用 `~/start_v026r_cluster.sh`，恢复手动顺序（**不推荐**，旧顺序已知 3/4 失败率） |
| 采样回滚 prob→greedy | 还原 `.bak.greedy-*` / `.bak-tilelang-2tier-*` 双机脚本 → 重启（LiteLLM per-key 模板独立，无需动） |
| 镜像回滚 | `:0.2.1-v026.0` / `:0.2.0-v026.0` / `:0.1.1` / hybrid 全保留；digest 不变 |
| 全部重置 | 双机停容器（worker 先 head 后）→ 恢复 `.bak.20260806_nccl` 脚本 → 手动顺序启动 |

---

## 3. 日常运维

### 3.1 健康检查命令集

| 层 | 命令（在对应主机执行） | 期望 |
|----|------------------------|------|
| 容器 | `docker ps` 双机 | `vllm-envE-node` / `vllm-envE-worker` healthy，RestartCount=0 |
| 端口 | `nc -z <NODE_IP> 25000`（宿主） | 通 |
| API | `curl -w "%{http_code}" -H "Authorization: Bearer <key>" http://127.0.0.1:8001|:8003|:4000/v1/models` | 8001=200 / 8003=200 / 4000=200 |
| 推理 | curl chat `"2+2=?"` 8001/8003/4000 | "4" |
| 思考链 | curl 8003 `/v1/responses` | 输出含 `type=reasoning` |
| 嵌入 | curl 4000 `/v1/embeddings`（Embedding key） | 200，`len==1024` |
| 系统 | 双机 `free -h` | 内存可用 10–13G 属正常低水位（服务占高） |

### 3.2 日志位置

| 日志 | 主机 | 用途 |
|------|------|------|
| `~/vllm-logs/nccl-*.log`（spark-05cd / edgexpert-0c69） | 双机 | NCCL INFO 级，卡死排查首选 |
| `~/v026r_cluster_runN.log` / `~/start_head_v026r_cluster.log` | head | 编排脚本与 head 启动 |
| 8003 网关日志（systemd `responses-gateway`） | worker | 网关 502/401/限流 |
| LiteLLM 日志（proxy 容器） | worker | key 鉴权、用量、429 |

### 3.3 备份与恢复

- **PG 备份**：worker 每日备份脚本；**head 每日 03:05 cron 异地拉取**（head→worker 免密，已实测成功）——双机互为备份
- **持久卷**：`vllm-cache` / `tilelang-cache` / `models` / LiteLLM / PG 卷全挂载
- **恢复**：PG 从备份 restore 后重启 LiteLLM；tilelang-cache 卷损坏时仅触发 JIT 重编（性能回落，可 `~/warmup_mhc.sh` 预热恢复）

### 3.4 性能基线对照表（判断回退）

矩阵口径：temp=0.6、直连 8001、max_tokens=128、流式+usage（`bench-f-baseline-2026-08-05.md`）：

| ctx | c1 agg_tps | c3 agg_tps | c5 agg_tps | c5 接受率 |
|-----|-----------|-----------|-----------|-----------|
| 512 | 38.95 | 64.68 | 96.52 | 56.2% |
| 2048 | 36.44 | 60.46 | 92.16 | 49.3% |
| 8192 | 36.61 | 58.84 | 83.34 | 48.0% |
| 32768 | 36.58 | 55.93 | 80.83 | 52.6% |

**4000 网关基线（08-06，口径：PARA 随机前缀填充 + temp 0.7）**：

| 负载 | c1/512 | c5/512 | c5/8192 | c5/32768 | 512 接受率 |
|------|--------|--------|---------|----------|-----------|
| code | 55.6 | **101.2** | 25.8 | 7.4 | 82.5% |
| json | 47.1 | **93.5** | 22.1 | 5.8 | 85.1% |
| prose | 39.2 | **67.7** | 22.4 | 6.8 | 41.4% |

> 两表口径不同（prompt 构建 + temperature），跨口径对比仅看趋势；**经 4000 真实开销 -6.8%~+6.9%（中位 ~0.5%）、TTFT ~0%**，GSM8K 经 4000 = 94.5%（无回退）。

- **负载类型**（c5 直连）：code 131.8 / json 122.5 / prose 81.7 t/s；GSM8K（temp=0.6）= 95.0%
- **判断回退规则**：与上表对照，c5 波动 ±10% 属轮间噪声（历史常见）；关键看①0 错误 ②c1 单流 ≥34 ③c5 聚合 ≥80 ④GSM8K ≥94%（temp=0.6 口径）。若大幅回退 → 检查 ①tilelang 预热/JIT 是否复现 ②probabilistic 是否退化（temp>0.1 是否被覆盖为 0）③版本是否被误改 ④**测试是否强制随机前缀**（固定文本命中 prefix-cache 会掩盖真实回退，见 §4.8）

### 3.5 方法学规范（基准测试强制项，08-06 固化）

1. **随机前缀强制**：矩阵/基线测试每请求必须带随机 `<rnd>` 前缀——固定文本会命中 prefix-cache 使 TTFT/吞吐数据失真（raw_final_matrix 32K TTFT≈370ms 即此假象）；报告须标注 prefill 口径（真实 TTFT vs cache 命中）
2. **温度统一**：同轮 A/B 或对比必须同 temperature（原 4000 vs 直连 23% 差异 = prompt 填充 + temp 0.6/0.7 不一致所致）
3. **LiteLLM 4000 开销结论**：严格同条件 A/B 实测 **-6.8%~+6.9%（中位 ~+0.5%）、TTFT 中位 ~0%**——LiteLLM 层无显著可测开销（鉴权/路由无固定延迟、spend 异步写 PG、日志无耗时）；此前 ~23% 判定为**口径差异非真实开销**，不再引用
4. **大 ctx 归因规范**：输出吞吐下降 ≠ 引擎退化；须先做时间拆解（TTFT/decode 占比）再归因（见 §4.8）

---

## 4. 故障排查手册

### 4.1 NCCL 卡死 / 启动挂起
- **症状**：重启后 head 日志停在 `parallel_state.py:1615 world_size=2 rank=0 backend=nccl`，8001 不监听、CPU 空闲、可挂 2h；worker 侧报 `TCPStore recvValue failed ... fdff::` / `Connection closed by peer`
- **诊断**：`ss -ltnp | grep 25000`（不监听=store 未建/H1；LISTEN=卡 barrier/H2）；`~/vllm-logs/nccl-*.log`（双机）；worker `nc -zv <NODE_IP> 25000`
- **根因**：H1 启动顺序竞态（rank0 才创建 TCPStore，worker 先启必失败）为主；H2 双 Spark GB10 系统性死锁（NVIDIA #366127）；H3 IPv6 污染（CVE-2025-47277 + 防火墙 DROP IPv6）
- **处置**：停双机容器（worker 先 head 后）→ `bash ~/start_v026r_cluster.sh` 重来，2-3 次重试可成功
- **预防**：只用编排脚本；重启前确认 25000 无残留；保持 `NCCL_DEBUG=INFO` 留证

### 4.2 容器 healthy 但 API 不可达
- **症状**：`docker ps` 双机 healthy，但 8001 无响应
- **诊断**：`docker logs vllm-envE-node --tail 200`；宿主 `nc -z <NODE_IP> 8001`；`ss -tln`（宿主非容器内）
- **根因**：权重加载未完成 / EngineCore 异常 / store 未就绪
- **处置**：等 5-8 分钟再探；仍不可达则双机脚本重启
- **预防**：就绪以编排脚本轮询 8001 为准，勿以容器状态判断

### 4.3 worker 失联（Broken pipe）
- **症状**：head 日志 Broken pipe；单边重建 head 无法恢复
- **根因**：TP 集群任一方退出，NCCL 连接即失效；docker kill 不触发 restart（unless-stopped 语义）
- **处置**：**必须双机重启**（worker 先 head 后 + 编排脚本）；禁止只重建 head
- **预防**：任何 kill/崩溃后一律全量重启；容器内进程崩溃不自动恢复集群

### 4.4 网关 502 / 5xx
- **症状**：8003 返回 502 `upstream_error`；4000 5xx/超时
- **诊断**：`systemctl status responses-gateway`；网关日志看上游 8001 是否可达；`curl 8001/v1/models` 直连验证
- **根因**：后端 vLLM 未就绪 / 负载过高 / 网关进程异常
- **处置**：确认 8001 200 → 重启网关服务；8001 不可达则走 4.2
- **预防**：网关 systemd 常驻，重启 vLLM 不影响网关进程

### 4.5 思考链丢失
- **症状**：responses/chat 无 reasoning 内容
- **诊断**：8003 `/v1/responses` 输出项是否含 `type=reasoning`；对比 4000（剥离 = 预期）
- **根因**：F 配置仅在 `enable_thinking=true` 生成思考链、网关不注入（旧版）；4000 LiteLLM 层剥离
- **处置**：思考链客户端**固定走 8003**（网关 v1.5.0 `_inject_enable_thinking` 已注入，chat+responses 双路由）
- **预防**：新客户端接入时按 §6 分流指引选网关；注意 8003 `enable_thinking:false` 不生效（思考不可关）

### 4.6 嵌入 401 / 异常
- **症状**：4000 embedding 401；8003 正常
- **诊断**：核对 key 权限——LiteLLM 按 key 限模型（Embedding key 才能调 embeddings；prob/greedy key 调 embedding = 401 属预期）
- **根因**：key 权限隔离（设计行为）；或 8020 绑定 127.0.0.1 后外部直连失败（预期）
- **处置**：用 `<API_KEY>`（Embedding key）调 4000；客户端一律走 4000/8003，勿直连 8020
- **预防**：发放 key 按最小权限；文档 §6 key 表为准

### 4.7 限流 429
- **症状**：4000 返回 429 + `Retry-After`
- **诊断**：查看响应限流字段；核对 key 限额（Chat rpm 300/tpm 50k；Embedding rpm 300/tpm 100k；Prob/Greedy 未设限流）
- **根因**：rpm/tpm 超限
- **处置**：等窗口重置；或联系管理员调大限额 / 换 Prob/Greedy key / 分流至 8003
- **预防**：客户端按 key 配额规划负载；高吞吐结构化走 prob key（未限流）

### 4.8 大 ctx 吞吐下降（非引擎缺陷）
- **症状**：8192/32768 ctx 输出吞吐骤降（c1 code 55.5→20.3→6.9 t/s），TTFT 线性放大（512→32K：0.48s→16.5s），易被误判为回退
- **根因**：**prefill 时间主导稀释**（~90-96%）：输出固定 max_tokens=128 时总耗时 = prefill（随 ctx 线性）+ decode（基本不变 1.83→1.96s），prefill 占比 21%→89%；次要 = chunked prefill 并发放大（c5 TTFT 16.5→50s、TPOT 15→244ms，~4-10%）；接受率下降 / KV 池竞争可忽略（<5%）
- **处置**：非故障，无需处置；大 ctx 场景改以 **prefill 吞吐（输入 t/s）** 与 TTFT 为考核指标（F 方案 prefill ~1700-2000 t/s 属健康）；优化方向 = prefill 加速或按 ctx 分实例/分路由，而非 decode 调优
- **⚠️ prefix-cache 假象**：基线若用固定文本（无随机前缀），warmup 后大 ctx 请求命中 prefix-cache → TTFT 全档平坦、吞吐"不随 ctx 降"（假象），掩盖真实 prefill 开销；**基准测试必须强制随机前缀**（§3.5），排查回退前先复核测试方法学

---

## 5. 坑位与经验教训（避免重复造轮子）

1. **NCCL 卡死 = 启动顺序反了**：vLLM TCPStore(25000) 由 rank0 在 `init_process_group` 时创建；worker 先启动是反的（旧顺序 4 次 3 挂，新顺序 3/3 成功）
2. **SSH 别名**：head 机访问 worker 用 `node0X`（<NODE_IP>），**不是** `aicad-server`（那是工作机侧别名；工作机侧 worker=`aicad-server`、head=`aicad-server60`）
3. **容器内 `ss` 看不到 25000**（无权限），端口探测一律用宿主机 `nc -z` / `ss -tln`
4. **docker kill = 显式停止，不触发 restart policy**（unless-stopped 语义）；容器内进程崩溃也不自动恢复 TP 集群 → 恢复必须双机脚本重启
5. **思考链走 8003**：F 配置仅 `enable_thinking=true` 生成思考链（网关 v1.5.0 已注入）；4000 LiteLLM responses 思考链被剥离——思考链必须走 8003
6. **LiteLLM key 级 aliases 不生效**（1.83.7 实测）：客户端须显式 `model=dspark-prob`/`dspark-greedy`（写 `dspark` → 401）；**prob key 必须 temp>0.1**（建议 0.7），temp≈0 回退 greedy 静默失去 +20~47% 吞吐
7. **8020 已绑 127.0.0.1**：嵌入调用统一走 4000/8003，勿直连 8020；PG 5432 仅 127.0.0.1
8. **tilelang JIT**：两档 patch + 预热 `~/warmup_mhc.sh` + `TILELANG_CACHE_DIR` 持久卷（`~/tilelang-cache`），重启后免重编；cache 丢失仅性能回落
9. **版本教训**：配置/结论**不可跨 vLLM 版本迁移**（0.25 vs 0.26 已验证，论坛结论需同版本复测）
10. **基准测试固定文本 = prefix-cache 假象**：矩阵/基线必须每请求随机 `<rnd>` 前缀；raw_final_matrix（08-05）32K TTFT≈370ms 即缓存命中假象，prefill 归因无效（方法学规范见 §3.5）
11. **RoCE 不可切 IB（硬件事实）**：ConnectX-7 固件锁定仅 RoCE；直连场景 IB 收益 <5-10%，无需再评估切换

**关键决策史**（勿重做）：E→F 演进（0.1.1 wrapper → 0.2.1-v026.0 固化版，12 挂载入镜像）；greedy→probabilistic（结构化负载 +20~47%，temp>0.1 强制）；动态K `[[1,1,5],[2,4,4],[5,6,3]]` 固化（c10 +36%）；Mega MoE 在 GB10 SM121 不可行（SM100 专属 UMMA/TMEM）；DeepGEMM/CUDAGRAPH=0/seqs=128 均不优于 b12x+动态K 组合；b12x+breakable+动态K 已是 GB10 最优；保持 RoCEv2（IB 固件锁定不可切）；4000 网关性能达标（27/27 0 错误、GSM8K 94.5%）；LiteLLM 无显著开销。

---

## 6. 安全与密钥

### 6.1 双轨 key 体系总表

| key | 值 | 用途 | 权限 | 网关 |
|-----|-----|------|------|------|
| Master | `sk-litellm-master-b9158f0b67dec7d9e395d54cb462afe2` | LiteLLM 管理（建 key/轮换） | 全部 Admin | 4000 |
| Prob | `<API_KEY>` | 结构化负载，model=`dspark-prob`，temp≥0.2（建议 0.7） | 仅 dspark-prob | 4000 |
| Greedy | `<API_KEY>_C8xuN9rqHhg` | 散文/确定性，model=`dspark-greedy`，temp 0.1 | 仅 dspark-greedy | 4000 |
| Chat | `sk-U_cIbL63-5c27rayJO3S6w` | 对话，model=`local-v4-flash`/`deepseek-v4-flash` | rpm300/tpm50k | 4000 |
| Embedding | `<API_KEY>` | 嵌入，model=`local-embedding` | rpm300/tpm100k | 4000 |
| 8003 客户端 | `<API_KEY>-64b0374c6f2840fe` | 自建网关全量（chat/completions/responses/embeddings/models） | passthrough | 8003 |
| 内部（勿外泄） | `<API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa` | vLLM 8001 / 嵌入 8020，仅网关转发注入 | 引擎内部 | 引擎直连 |

> **分流**：结构化/高吞吐/流式即时 → 4000（prob）；思考链/Responses/单流低延迟 → 8003。**两条硬约定**：显式 model 名（aliases 不生效）；prob key temp>0.1。

### 6.2 key 轮换流程
- **LiteLLM（4000）**：Master key 调 `/key/regenerate`（body 带原 key + models + 限额）→ 新 key 生效旧 key 自动失效 → 通知持 key 方更换 → 确认无旧调用后销毁
- **8003**：服务端配置项，改配置后重启网关（具体步骤由 SRE 补充）
- **内部 key / master_key**：改 env 注入后重启对应服务（master_key 当前仍明文在 config，属待办）

### 6.3 已加固项
✅ 双机 7 卷持久化 + restart=unless-stopped ✅ 备份异地互备（head 03:05 拉 worker PG）✅ NCCL_IB_TIMEOUT=1000/RETRY_CNT=7 ✅ NCCL_DEBUG=INFO 留证 ✅ 8003 思考链注入 v1.5.0 ✅ 8020 绑 127.0.0.1 ✅ PG 绑 127.0.0.1 ✅ 内部 key env 注入 ✅ docker 清理 -105GB、脚本归档

### 6.4 遗留待办（需用户拍板）
⏳ ① 防火墙 firewalld 白名单（需 sudo，P1）② master_key / PG 密码轮换 + 去明文（P1）③ worker 管理网回有线（当前 WiFi 79ms 抖动单点，P1）④ 8020 告警探针（P2）

---

## 7. 自反馈清单

**✅ 已闭环**
- NCCL 启动卡死根因定位（H1/H2/H3）并修复验证 3/3；编排脚本固化生产 SOP
- probabilistic + 动态K + tilelang 生产切换；新基线固化（GSM8K 95.0%，12 组合 0 错误）
- 8003 思考链修复（v1.5.0 注入）；双轨网关 benchmark 0 错误
- 清理/加固/持久化/韧性验证（docker -105GB、异地互备、restart 确认）
- 4000 网关综合性能达标（27/27 0 错误、GSM8K 94.5% 无回退）；LiteLLM 真实开销核实（中位 ~0.5%，23% 系口径差异）
- 大 ctx 吞吐下降归因（prefill 主导 90%+，非引擎缺陷）；基准方法学固化（随机前缀 + 温度统一）
- IB 调研定论（固件锁定不可切，保持 RoCEv2）；LLM 停机/恢复预案固化（§0）

**⏳ 待办**
- 防火墙 / 密码轮换 / worker 管理网有线（§6.4，需用户拍板）
- 收尾清单回填（production-finalize-checklist 更新 ✅ 状态）
- LLM 恢复时机协调（视频工作流释放 GPU 后一键拉起，见 §0；当前停机中）

**🔭 建议跟踪项（上游）**
- NVIDIA forum #366127 / vLLM #33041（双 Spark GB10 TP2 NCCL 死锁，需上游修复；复现可试 `NCCL_P2P_DISABLE=1 + --disable-custom-all-reduce`）
- CVE-2025-47277（store IPv6 双栈）修复版 vLLM 跟进
- Anemll 新镜像 / LiteLLM 新版（aliases 生效、`min_temperature` guardrail、`chat_template_kwargs` 透传）
- 8192 c5 轮间噪声复测；TCPStore 端口状态与快速重启关系专项
- 4000 网关大并发（>c5）与多 worker 评估（当前无需求）；final-baseline 脚本补随机前缀后 32K 基线复测

---

> 本手册由工程保障团队技术文档师汇编自 2026-08-01~08-06 全部生产报告与记忆日志；准确性优先，未验证结论不收录。关键决策请由人类工程负责人复核。
