# KV SSD 卸载 mem util 0.80→0.70 修正 · 验证测试方案

- **作者**：Tessa（测试专家，工程保障团队）
- **日期**：2026-08-19（设计方案，未执行）
- **执行人**：SRE Rex（SSH 执行）；本方案**不连接任何节点**
- **被测对象**：生产 TP4 集群（4×DGX Spark，vLLM 0.26，deepseek-v4-flash-0731，600K 上下文，KV SSD 卸载 io.py 补丁已上线）
- **变更**：`--gpu-memory-utilization` **0.80 → 0.70**（用户批准，已改四节点启动脚本 + check_vllm_script 校验）
- **验证目标**：0.7 修正后，**conc3×65536（65536/coding/conc3 事故格）不再触发 NCCL ALLREDUCE 300s 超时**；同时量化内存释放与 KV 卸载的真实内存代价

---

## 0. 事故与变更速览（采信主理人，本方案直接采用）

| 项 | 证据 | 对本方案的意义 |
|---|---|---|
| 事故 | benchmark 31/54（**65536/coding/conc3**）NCCL ALLREDUCE 300s 超时 → worker 死 → 集群宕 | Phase 3 主判定对象 |
| 可复现性 | **conc3×65536 可稳定复现** | Phase 3 单次受控复现即有效判定 |
| 复现工具 | `/tmp/verify_conc3_65536.py`（01 容器内，3 并发 × 65536 ctx 随机前缀） | Phase 3 主工具，直接复用 |
| 0.8 内存态 | used 110-115GB（121GB UMA）；03/04 avail 仅 4-6G；03 曾内存耗尽卡死；01 kvssd 曾 93% 满 | Phase 2 基线对照 + Phase 4 磁盘水位 |
| 0.8 正常项 | 单请求 65536 正常（46.2s）；conc1 短 ctx 全正常 | 不重复验证，采信 |
| rank 映射 | rank0=01(head)、rank1=02、rank2=04、rank3=03 | 日志检查目标（rank0/rank1） |
| 期望释放 | 0.7 vs 0.8 ≈ 0.1×121 ≈ **12GB/节点** | Phase 2 判定依据 |

**重要张力提示**：mem util 降低会**缩小 GPU KV 池**，可能使 SSD 卸载更激进（更多写盘、更多 CPU 侧压缩/IO），与"释放 DRAM 缓解内存压力"的修复意图存在方向张力。因此：
- **Phase 3（conc3×65536 复现）是唯一权威判定**——内存释放是否真能消除 NCCL 超时，以实测为准；
- **Phase 4（卸载代价）负责量化**这个张力是否把内存压力转移成了卸载/IO 压力。

---

## 1. 判定总纲（门禁一览）

| # | 门禁 | 阶段 | 通过条件 | 失败处理 |
|---|---|---|---|---|
| **G0** | 前置就绪 | Phase 0 | monitor 已 disable；四脚本 mem util=0.70；check 通过 | 不满足则 NO-GO，先补前置 |
| **G1** | 服务就绪 | Phase 1 | 4 rank Up(healthy) + /health 200 + 真实推理 200 + 卸载路径生效（KVZSTD01 魔数抽查） | 任一项失败 → 记录并报 SRE，不继续 |
| **G2** | 内存基线 | Phase 2 | 03/04 avail ≥8G；全节点 avail 相对 0.8 释放 ~12GB±3 | 03/04 avail <8G → 记录并报主理人，Phase 3 前必须人工决策（不可带病复测） |
| **G3** | **conc3×65536 复现（关键）** | Phase 3 | **3 请求全部 200 + 5min 观察窗口无 `Watchdog caught`/`died unexpectedly`/NCCL timeout + 4 容器仍 Up + kv_load_failure=0** | **FAIL：记录全部证据并报 SRE/主理人，不自动重试** |
| **G4** | 卸载代价 | Phase 4 | 卸载指标/du/swap 数据采集完整；bytes/token 维持 70.7KB 量级；kvssd 水位 <80%；swap 无异常上升 | 超标项记录定档，不阻断 G3 结论 |
| **G5** | benchmark 恢复 | Phase 5 | G3 PASS 后按分段恢复；每段后 NCCL 日志/健康检查全绿 | 任一段失败 → 停止，走 FAIL 备选表 |

> **执行顺序铁律**：G0 → G1 → G2 → **G3** → G4（随 G3 采集）→ G5。G3 未 PASS 前**禁止**恢复全量 benchmark。

---

## 2. 执行顺序与时间预算

| 段 | 内容 | 预估 | 说明 |
|---|---|---|---|
| P0 | 前置确认（G0） | 5 min | monitor disable、脚本核验、基线固化 |
| P1 | 就绪验证（G1） | 20-30 min | 含一次真实推理 + KVZSTD01 抽查 |
| P2 | 内存基线 0.7 vs 0.8（G2） | 10 min | 四节点 free/docker stats/cgroup |
| P3 | **conc3×65536 复现（G3）** | 30-40 min | 含 5min 观察窗口 + 证据采集 |
| P4 | 卸载代价核查（G4） | 随 P3 采集 | 指标 + du + swap + 磁盘水位 |
| P5 | benchmark 分段恢复（G5） | 3-4 h | 仅 G3=PASS 后执行；完成后恢复 monitor |

---

## 3. Phase 0 — 前置确认（G0，不占服务）

**目标**：复测窗口环境受控，杜绝"假通过/假失败"。

### 3.1 monitor 必须 disable（复测前确认，01 节点）
```bash
# 期望：service inactive/disabled + timer 无活动；若 active 必须先 stop/disable
systemctl is-active vllm-tp4-head.service        # 期望 inactive 或 disabled
systemctl list-timers vllm-healthcheck.timer     # 期望无该 timer 活动
```
> **原因**：monitor 会在 worker 死亡 30s 内自动拉起 rank0（`docker wait` 逻辑，见 200G 执行报告 §待完善-5）。若复测时 monitor 存活：
> - 失败被自动"修复"→ 误判 PASS；
> - 自动拉起与测试并发 → 污染证据、扩大爆炸半径。
> **测试全程保持 disable，直到 G5 全部结束后再恢复**（见 §8.3）。

### 3.2 脚本 mem util=0.70 核验（四节点）
```bash
# 四节点分别执行（01: start_tp4_head.sh；02/03/04: start_tp4_worker.sh）
grep -oE -- '--gpu-memory-utilization [0-9.]+' <INSTALL_DIR>/scripts/start_tp4_*.sh | sort -u
# 期望：四节点均为 0.70
bash <INSTALL_DIR>/scripts/check_vllm_script.sh <INSTALL_DIR>/scripts/start_tp4_head.sh   # 期望 ✅ 全部通过
```

### 3.3 0.8 基线固化（对照快照，四节点 + head）
```bash
# 四节点：free -m（MemAvailable/Swap used）
free -m > /tmp/util07_base_free_$(hostname).txt
# 四节点：容器 RSS + cgroup
docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}' vllm-tp4-rank* > /tmp/util07_base_stats_$(hostname).txt
# head：kv_offload 指标 + kvssd du + 磁盘水位
curl -s http://<NODE_IP>:8001/metrics | grep -E "^kv_offload|kv_cache_usage|prefix" > /tmp/util07_base_metrics.txt
du -sb /opt/aicad-kvssd | awk '{print "kvssd_bytes_base="$1}' > /tmp/util07_base_du.txt
df -h /opt/aicad-kvssd
```
> 0.8 已知对照值（采信）：used 110-115GB；03/04 avail 4-6G；01 kvssd 93% 满。Phase 2 用 0.7 实测与其对比。

### 3.4 证据目录
```bash
mkdir -p /tmp/util07_verify && cd /tmp/util07_verify
# 后续所有输出统一落此目录，文件命名带阶段前缀
```

---

## 4. Phase 1 — 就绪验证（G1）

**目标**：0.7 重启后集群真实可用，且 KV SSD 卸载路径真实生效（不是只响 health、不是静默回退到旧格式/未卸载）。

### 4.1 启动顺序（head-first 铁律）
`01(rank0) → 02(rank1) → 04(rank2) → 03(rank3)`；head 先启，轮询 TCPStore 25999 后依序启 worker；禁止 worker 先启/单边重建（Runbook §A.5）。

### 4.2 检查项与命令

| # | 检查项 | 命令 | 通过条件 |
|---|---|---|---|
| 1 | 4 rank 容器 | `docker ps --filter name=vllm-tp4 --format '{{.Names}}\t{{.Status}}'`（四节点） | 4 容器均 `Up (healthy)` |
| 2 | /health | `curl -s -o /dev/null -w '%{http_code}' http://<NODE_IP>:8001/health` | 200 |
| 3 | 模型规格 | `curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://<NODE_IP>:8001/v1/models` | served-model-name=deepseek-v4-flash-0731；max_model_len=600000 |
| 4 | **真实推理（非仅 health）** | 1×512 ctx，max_tokens 32，见 §4.3 | HTTP 200 + 合理输出 + usage.prompt_tokens>0 |
| 5 | 卸载配置 | `docker logs vllm-tp4-rank0 2>&1 \| grep -iE "TieringOffloadingSpec|OffloadingConnector" \| tail -5` | 有输出，root_dir=/opt/aicad-kvssd |
| 6 | io.py 补丁路径 | `docker exec vllm-tp4-rank0 python -c "import vllm.v1.kv_offload.tiering.fs.io as io; print(io.__file__)"` | 指向挂载路径（非镜像内原路径） |
| 7 | **KVZSTD01 魔数抽查** | `find /opt/aicad-kvssd -type f \| head -5 \| while read f; do echo "== $f"; head -c 8 "$f" \| xxd; done` | 至少一个文件头 8B == `4b56 5a53 5444 3031`（"KVZSTD01"） |
| 8 | 无 OOM | `dmesg 2>/dev/null \| grep -iE "oom|killed process" \| tail -5 \|\| echo no-oom`（四节点） | 无 OOM-kill 记录 |

### 4.3 真实推理示例（检查项 4）
```bash
source <INSTALL_DIR>/secrets/vllm.env
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://<NODE_IP>:8001/v1/chat/completions \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"hello, 1+1=?"}],"max_tokens":32,"temperature":0,"stream":false}' \
  | python3 -c "import sys,json; j=json.load(sys.stdin); print('status=200'); print(j.get('choices',[{}])[0].get('message',{}).get('content','')); print(j.get('usage'))"
```

### 4.4 G1 判定
- 表 4.2 全部 ✅ → **G1 PASS**，进入 Phase 2。
- 任一项失败 → 记录证据 + 报 SRE（可能需回滚/排查），**不进入 Phase 3**。
- 魔数抽查若因缓存为空（重启后无落盘）无文件：先发一个 ≥9K ctx 请求触发落盘（复用 G-4 触发方式，随机前缀），再抽查；仍无新格式文件 → 怀疑补丁/挂载未生效，报 SRE。

---

## 5. Phase 2 — 内存基线对比（0.7 vs 0.8，G2）

**目标**：量化 0.7 释放的内存量；**重点判定 03/04 avail 是否 ≥8G**（0.8 事故态仅 4-6G，是内存耗尽风险源）。

### 5.1 方法与口径
- **主机级 `free -m` 的 MemAvailable 为 UMA 压力的权威口径**（事故证据即来自 free）；容器 RSS/cgroup 用于归因（哪些进程吃了内存）。
- 每节点采集：`free -m`（Mem/ Swap）、`docker stats`（容器 MemUsage）、容器 cgroup `memory.current / memory.peak / memory.events(oom_kill)` + `memory.stat` 的 rss/anon。

```bash
# 四节点并行
echo "== $(hostname) =="
free -m | grep -E "Mem|Swap"
docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}' vllm-tp4-rank*
docker exec vllm-tp4-rank* sh -c 'echo current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); echo peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo NA); grep -E "^(rss|anon) " /sys/fs/cgroup/memory.stat 2>/dev/null; grep oom_kill /sys/fs/cgroup/memory.events 2>/dev/null'
```

### 5.2 对比表模板（SRE 回填）

| 节点 | 0.8 avail (G) | 0.7 avail (G) | Δ avail | 容器 RSS (G) | cgroup current (G) | cgroup peak (G) | oom_kill | 判定(≥8G) |
|---|---|---|---|---|---|---|---|---|
| 01 | 11 | | | | | | | |
| 02 | 11 | | | | | | | |
| 03 | 4-6 | | | | | | | ⚠️ 重点 |
| 04 | 4-6 | | | | | | | ⚠️ 重点 |

### 5.3 G2 判定
| 判据 | 条件 |
|---|---|
| **PASS** | 03/04 avail **≥8G**；全节点 Δavail 与预期 ~12GB（0.1×121）同量级（±3G 内视为符合，否则记录偏差） |
| **WARN** | 03/04 avail 6-8G：记录，Phase 3 前人工确认（可测但标注风险） |
| **FAIL** | 03/04 avail <6G 或 cgroup oom_kill>0 → **记录并报主理人，禁止进入 Phase 3**（带病复测会扩大宕机风险） |

> 附加观察：若 0.7 释放量显著 <12GB，可能 GPU KV 池未实际收缩（脚本未生效/未重启），需回查 Phase 0 脚本核验。

---

## 6. Phase 3 — conc3×65536 复现测试（G3，关键判定）

**目标**：复测事故格。0.7 修正有效 ⇔ 3 并发 × 65536 随机前缀全部 200 且无 NCCL Watchdog 超时。

### 6.1 前置（必须满足）
- [ ] G0/G1/G2 全 PASS（03/04 avail ≥8G）
- [ ] monitor 确认 disable（§3.1）
- [ ] 测试前快照：`free -m`、`du -sb /opt/aicad-kvssd`、`/metrics` 的 `kv_offload_*` 与 prefix 计数 → `/tmp/util07_verify/`（Phase 0 已建）

### 6.2 测试设计
- **工具**：01 容器内 `/tmp/verify_conc3_65536.py`（3 并发 × 65536 ctx 随机前缀）。
- **工具参数核验**（执行前确认脚本内，避免口径漂移）：
  - 3 条请求**同时发出**（asyncio.gather）；
  - 每条**随机前缀**（uuid4）→ prefill 全量计算、prefix 命中=0（命中会显著减小 KV 压力，导致假 PASS）；
  - `client timeout ≥ 600s`（**必须 > 300s**：NCCL watchdog 300s 超时是判定对象，客户端 300s 超时会掩盖/混淆；600s 足够区分"请求慢"与"NCCL 超时"）；
  - `max_tokens` 16-64 即可（判定点是 prefill 期 NCCL allreduce，非 decode 吞吐）；`temperature=0`；
  - 若工具默认非随机前缀或 timeout<600s，**先改工具再跑**，改动记录在案。
- **一次执行，不重试**：G3 只跑 1 轮。FAIL 后由主理人决策（见 §8.2），禁止自动重试。

### 6.3 执行序列（SRE）
```bash
cd /tmp/util07_verify

# 0) 测试前快照
free -m > conc3_before_free.txt
du -sb /opt/aicad-kvssd | awk '{print "kvssd_bytes_before="$1}' > conc3_before_du.txt
curl -s http://<NODE_IP>:8001/metrics | grep -E "^kv_offload|^vllm:num_prefix|kv_cache_usage" > conc3_before_metrics.txt

# 1) 复现（01 容器内）
docker exec vllm-tp4-rank0 python3 /tmp/verify_conc3_65536.py 2>&1 | tee conc3_run.txt
# 记录：3 请求各自 HTTP status + usage(PT/CT) + 首个异常（超时/连接错误/5xx）

# 2) 观察窗口（请求返回后 5 分钟，NCCL 300s 超时可能延迟暴露）
#    T+0 / T+1min / T+3min / T+5min 各打点一次：
date +%T; curl -s -o /dev/null -w 'health=%{http_code}\n' http://<NODE_IP>:8001/health
docker ps --filter name=vllm-tp4 --format '{{.Names}} {{.Status}}'
free -m | grep Mem
# 手动在 T+0、+1m、+3m、+5m 执行并追加到 conc3_observe_<t>.txt

# 3) 日志检查（关键：rank0=01, rank1=02）
for r in rank0 rank1; do
  echo "== $r docker logs =="
  docker logs vllm-tp4-$r 2>&1 | grep -iE "Watchdog caught|died unexpectedly|NCCL.*timed out|timed out after|allreduce" | tail -20
done
# NCCL_DEBUG 落盘文件（/var/log/vllm/nccl-<host>.log）
grep -ilE "Watchdog|timeout|error" /var/log/vllm/nccl-*.log 2>/dev/null | while read f; do
  echo "== $f"; grep -iE "Watchdog|timed out|error|allreduce" "$f" | tail -20
done

# 4) 卸载一致性（kv_load_failure 必须为 0）
curl -s http://<NODE_IP>:8001/metrics | grep -iE "kv_load_failure|kv_offload.*(fail|err)" | grep -v "^#"
```

### 6.4 G3 判定

| 判定 | 条件 |
|---|---|
| **PASS（修正有效）** | ① 3 请求全部 HTTP 200 且输出合理；② 5min 观察窗口 rank0/rank1 日志**无** `Watchdog caught` / `died unexpectedly` / NCCL timeout；③ 4 容器观察窗口全程 Up(healthy)；④ kv_load_failure=0 |
| **FAIL（修正无效）** | 任一请求非 200/超时/连接错误；或日志出现 `Watchdog caught`/`died unexpectedly`；或任一容器死亡/重启 |

**FAIL 处理（纪律）**：
1. **不自动重试**；2. 保存全部证据（conc3_run.txt、observe_*、docker logs 尾 200 行、NCCL_DEBUG 相关行、free/dmesg 快照）到 `/tmp/util07_verify/`；3. 报 SRE + 主理人，进入 §8.2 备选决策。

### 6.5 中止条件（执行中任何时刻）
- 任一节点 `free` avail < 2G → **立即中止**，记录内存快照，报 SRE（防 03 内存耗尽卡死复发）；
- 任一容器退出/重启 → 立即中止，报 SRE；
- dmesg 出现 OOM-kill → 立即中止。

---

## 7. Phase 4 — KV 卸载内存占用核查（G4）

**目标**：量化卸载的真实内存代价——0.7 释放的 DRAM 是否又被卸载路径（CPU 主层 + 压缩缓冲 + 页缓存 + swap）吃回去；同时复核落盘路径健康。

### 7.1 指标采集（head /metrics）
```bash
curl -s http://<NODE_IP>:8001/metrics | grep -E "^kv_offload" | grep -v "^#" > conc3_after_metrics.txt
diff conc3_before_metrics.txt conc3_after_metrics.txt || true
```
| 指标 | 用途 | 判读 |
|---|---|---|
| `kv_offload_store_bytes_total` | 落盘字节计数 | Δstore_bytes / Δtokens ≈ 70.7KB/token 量级（复核卸载路径真实运行且密度未恶化） |
| `kv_offload_store_tokens`（若暴露） | 写盘 token 计数 | 与 du 交叉验证（首选口径） |
| `kv_offload_lookup_*` / `load_*` | 读回路径 | 无异常放大（参考灰度 lookup 8-24ms） |
| `kv_offload_cpu_cache_usage`（若暴露） | CPU 主层占用 | 相对 2GiB 预分配的占用率；若 100%+持续 → 卸载热点 |
| `vllm:num_prefix_cached_tokens` / prefix 计数 | prefix 命中校验 | **必须无增长**（随机前缀铁律）；有增长 → 本轮作废 |

### 7.2 du 增量 + 磁盘水位 + swap（四节点）
```bash
du -sb /opt/aicad-kvssd | awk '{print "kvssd_bytes_after="$1}'
df -h /opt/aicad-kvssd                      # 01 事故 93% 满；期望 <80%
free -m | grep -i swap                      # 记录 Swap used；异常上升=内存压力信号
docker exec vllm-tp4-rank* sh -c 'grep oom_kill /sys/fs/cgroup/memory.events'   # 应为 0
```

### 7.3 G4 判定
| 判据 | 条件 |
|---|---|
| **PASS** | bytes/token 维持 70.7KB 量级（±20%）；kvssd 水位 <80%；swap used 无明显上升（<1G 增量）；oom_kill=0 |
| **WARN** | bytes/token 上升 >20%（卸载更激进，符合 0.7 KV 池变小的预期方向）→ 记录定档，不阻断 G3 结论 |
| **FAIL** | kvssd 水位 ≥80% 或 swap 显著上升或 oom_kill>0 → 记录并报主理人（卸载代价不可接受） |

---

## 8. Phase 5 — benchmark 恢复建议（G5）

### 8.1 G3=PASS 时的分段恢复（推荐顺序）

**原则**：不一次性重放 54 组合，按"内存/卸载压力递增"分段，段间设健康闸门。

| 段 | 组合 | 目的 | 段后闸门 |
|---|---|---|---|
| S1 | ctx **512,4096,16384,32768** × 3 task × 3 conc（36 格） | 低-中压力全量回归 | 4 容器 Up + /health 200 + rank0/rank1 日志无 Watchdog/died + 03/04 avail≥8G + kv_load_failure=0 |
| S2 | ctx **65536** × 3 task × 3 conc（9 格，**含事故格 65536/coding/conc3**） | 事故格直接回归 | 同上；**65536/coding/conc3 单独先跑**，200 后再放其余 8 格 |
| S3 | ctx **131072** × 3 task × 3 conc（9 格） | 最重负载 | 同上 + 全程盯 03/04 avail（<2G 即中止）+ dmesg OOM=0 |

**执行要点**：
- 每段前 warmup（3×512 ctx 请求）吸收 JIT 编译尖峰（TTFT 漂移报告 §四-P1）；
- 每段后 `grep -iE "Watchdog caught|died unexpectedly|timed out" /var/log/vllm/nccl-*.log` + docker logs；
- monitor 保持 disable 至 **S3 完成**；
- 任一段 FAIL → 停止，保存证据，报主理人，**不自动降级继续**。

### 8.2 G3=FAIL 时的备选方案（决策表，主理人裁决）

| # | 选项 | 操作 | 风险/代价 | 备注 |
|---|---|---|---|---|
| 1 | 回滚 mem util 0.8 | 恢复脚本 → 重启 | 回到事故态（03/04 avail 4-6G） | 单独使用不推荐；须与 2/3 组合 |
| 2 | 降并发 | 65536/131072 只跑 c1（或 c1+c3，跳过 c5） | 矩阵不完整 | 若根因是并发 prefill 聚合压力 |
| 3 | 降 ctx 上限 | 矩阵上限降为 32768 或 65536（放弃 131072） | 覆盖缩水 | 若根因与长 prefill 内存峰值相关 |
| 4 | 延长 NCCL 超时 | `--distributed-timeout-seconds 300→600` | **掩盖症状**、延迟故障暴露 | 仅主理人明确授权；不解决根因 |
| 5 | 关卸载对照（Tier-2） | 回滚 kv-transfer-config，A/B 判断卸载路径是否引入竞争 | **600K ctx 无卸载内存可能放不下（OOM 风险）**，慎用 | 判别"卸载 IO/压缩线程 vs NCCL progress 竞争"假设 |
| 6 | 深挖 NCCL 卡点 | 用 G3 的 NCCL_DEBUG 时间线定位 allreduce 卡在哪个 rank/段 | 无服务代价 | SRE task#4 并行推进；若指向卸载线程抢 CPU → 调低 offload 读写线程数（4→2）或降 zstd level 复测 |
| 7 | 再降 mem util / 缩 CPU 主层 | 0.70→0.65；或 cpu_bytes_to_use 2GiB→1GiB | 每步再释放 ~6GB；KV 池更小、卸载更激进 | 与 6 配合，逐级验证 |

### 8.3 收尾（G5 全绿后）
```bash
# 恢复生产自愈：重新启用 monitor（service + timer）
systemctl enable --now vllm-tp4-head.service
systemctl enable --now vllm-healthcheck.timer
systemctl is-active vllm-tp4-head.service; systemctl list-timers vllm-healthcheck.timer
# 确认自愈链路：docker ps 4 容器 Up(healthy) + /health 200
```

---

## 9. 交付物与数据留存

| 产物 | 路径 |
|---|---|
| 本方案 | `deliverables/engineering-assurance/kvssd-util07-verify-plan-2026-08-19.md` |
| Phase 0 基线快照 | `/tmp/util07_verify/*_base_*.txt`、`*_base_metrics.txt`、`*_base_du.txt` |
| Phase 3 复现运行 | `/tmp/util07_verify/conc3_run.txt`、`conc3_observe_<t>.txt` |
| Phase 3 日志证据 | rank0/rank1 docker logs 尾 200 行、NCCL_DEBUG 相关行 |
| Phase 4 指标 | `/tmp/util07_verify/conc3_after_metrics.txt`、`kvssd_bytes_after`、swap/oom 快照 |
| 对比表（Phase 2/5 模板回填） | `kvssd-util07-result-2026-08-19.md`（G3 后补） |

## 10. 覆盖重点与缺口说明

**已覆盖（高优先级）**：
- 关键路径：conc3×65536 复现（事故格）、真实推理、/health、KVZSTD01 落盘生效、读回一致性（kv_load_failure=0）
- 错误处理：NCCL Watchdog 超时检测、容器死亡检测、OOM 中止条件
- 边界情况：03/04 avail 地板（≥8G 门禁）、kvssd 磁盘水位（93%→<80%）、swap 用量
- 安全边界：monitor disable 防自愈掩盖、FAIL 不自动重试

**已知缺口（采信他人结论，不重复验证）**：
- 单请求 65536（46.2s）与 conc1 短 ctx 正常 → 采信，不重测
- 卸载 io.py 补丁正确性（Cody 已审查 + 容器内 6 项验证 + 200G 执行报告）→ 采信，Phase 1 仅抽查魔数
- 0.8 内存态数值 → 采信事故取证；若需更细归因由 SRE task#3 补充

**建议补充观测（超出本次必测，供 SRE/架构师）**：
- NCCL_DEBUG 时间线定位 allreduce 卡点 rank/段（判别"卸载 IO/压缩线程抢 NCCL CPU"假设，§8.2-6）
- offload 读写线程 CPU 占用 vs NCCL progress 线程（ps -eLo psr 抽样）——若为根因，比 mem util 修正更直接

> 本方案仅设计，不执行。G3（conc3×65536 复现）为唯一权威判定；G3 未 PASS 前禁止恢复全量 benchmark。关键决策（monitor 窗口、FAIL 备选、回滚）请由人类工程负责人复核。
