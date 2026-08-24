# KV SSD 卸载存储效率优化 · 验证测试方案（G-4 存储效率复测 + G-5 全量 benchmark 回归）

**作者**：Tessa（测试专家）
**日期**：2026-08-19（设计方案，未执行）
**执行人**：主理人（工程总监）按本方案 SSH 执行；本文件**不涉及任何节点连接**
**被测对象**：生产 TP4 集群（4×DGX Spark）KV 缓存 SSD 卸载 + `io.py` 补丁（去重 + trim + zstd-3）
**判定总纲**：G-4（落盘 ≤10KB/token）+ G-5（54 组合无性能回归，重点 131072 组 TTFT/吞吐）双门禁。

---

## 0. 采信基线速查（团队已实测，本方案直接采用，不重新验证）

| 项 | 数值 | 用途 |
|---|---|---|
| NVMe 写 | ~4.1 GB/s（256K/qd8） | 写吞吐上限 |
| NVMe 顺序读 | ~2.0 GB/s | 读上限参考 |
| NVMe 随机读 | 256K/qd16 ~1.8 GB/s、128K ~1.1 GB/s | lookup 带宽参考 |
| GPU KV 真实成本 | 9.6 KB/token | 物理下限 |
| MLA 压缩 KV | ~6.7 KB/token | 无冗余下限 |
| 磁盘现状 | 382 KB/token（5× 重复写 + 76% 零填充） | G-4 待改善基线 |
| 灰度实测 | kv_offload_store_bytes 6.08GB / store_time 0.143s ≈ 42 GB/s（GPU→CPU 主层）；lookup sync+async 8~24 ms | 触发方式与延迟基线 |
| 内存 | 01/02 available ~11G、03/04 ~6G；CPU 主层 2GiB；目标内存增量 ≤3GB | 风险监控水位 |
| 补丁后预期 | 写有效吞吐 2-3 GB/s（压缩瓶颈）；读仅有效 ~1GB/序列 | 吞吐合理性预期 |
| 基准脚本 | `<INSTALL_DIR>/bench_prefill_decode_async.py`，engine=asyncio，aiohttp ClientTimeout=1800s | G-5 主工具 |
| 基准口径 | 跨组对比禁用 `agg_*`；统一 per-request p50 × conc；rounds 3 | 判定口径 |

**采信前提**：以上数值来自灰度窗口实测；若 G-4 复测中 `du` 计量与 382KB/token 同量级，则先怀疑补丁未生效或实验口径偏差，按 §3.5 排查。

---

## 1. G-4 存储效率复测方案（阶段 4，≤30 min）

### 1.1 目标与判定

验证 `io.py` 补丁（去重 + trim + zstd-3）生效后，**落盘 bytes/token ≤ 10 KB/token**（对 382KB/token 基线 ≥38× 改善）。

| 判定 | 条件 |
|---|---|
| **PASS** | 多样文本组中位 bytes/token ≤ 10 KB/token，且 `kv_offload_store_bytes` 有增量（确认真触发落盘）、0 KV 读回失败 |
| **FAIL** | 中位 > 10 KB/token，或未触发落盘（无增量），或读回失败 > 0 |
| 超标 | 按 §3.5 排查分支定位失效层，30min 内给出结论（不自动判引擎故障） |

### 1.2 受控实验设计

**触发方式（复用灰度）**：CPU 主层 2GiB，策略为背景 flush/逐出——灰度 3×9K-token 请求已确认可触发 SSD 落盘（6.08GB store）。G-4 复用同构请求以直接可比。

**请求参数建议**：
- ctx = **9000 token**（随机前缀 + 中文文本；CHARS_PER_TOKEN=1.85 → ~16650 字符）
- `max_tokens=16`（最小输出，使新 KV ≈ prompt+completion 全量；completion_tokens 计入分母）
- `temperature=0`（确定性，读回一致性可比）
- `concurrency=1`（避免并发 flush 竞争污染计量）
- `stream=false`
- **随机前缀铁律**：每条请求 `content = uuid4().hex + "\n" + <文本>`，保证 prefill 全量计算、prefix 命中=0（`--enable-prefix-caching` 在生产开启，不随机前缀必然命中缓存 → 落盘减少 → 计量失真）

**两组文本（口径覆盖）**：
| 组 | 文本 | 测什么 | 判据适用 |
|---|---|---|---|
| A 多样文本 | 从生产语料池随机采样多字符/多 token 序列（避免全重复） | 接近真实负载的代表性 bytes/token | **主判据（≤10KB/token）** |
| B 重复文本 | 重复单一字符/短句（如 `数`×16650） | 验证**去重**特性是否生效 | 辅助：应与 A 显著更低（去重收益） |

> 若灰度 3×9K 用的是重复文本，则首轮与灰度同构（可比 382KB），再补 A 组作保守判据。若首轮未触发落盘（store_bytes 增量=0），逐步放大 ctx：9K→27K→81K，直至触发并记录实际触发规模。

**精确计量 bytes/token**：
```
bytes/token = ΔS / Σ新落盘 token
ΔS = du -sb /opt/aicad-kvssd 增量（S1 − S0，SSD 侧 flush 稳定后采样）
Σ新落盘 token = Σ (usage.prompt_tokens + usage.completion_tokens)（随机前缀→全部新算）
```
- **du 必须用 `-sb`（字节）**：默认块计数会让小文件/稀疏文件计量失真。
- **采样时机**：请求返回后 sleep ≥30s（或轮询 `kv_offload_store_bytes` 两次一致），等 fs 层 flush/GC 稳定再 du；否则增量偏小或偏大。
- **交叉验证（优先）**：若 /metrics 暴露 `kv_offload_store_tokens`（写盘 token 计数），直接用 `store_bytes_delta / store_tokens_delta`，不依赖请求侧 usage 记账，更精确。

**排除 prefix 缓存命中导致的落盘减少**：
1. 随机前缀 → 请求侧命中=0（主手段）。
2. 交叉验证：请求前后查 `/metrics` 的 `vllm:num_prefix_cached_tokens`（或 `prefix_cache_hits`）增量；若 >0 → 本轮作废重跑（不归一化，避免口径污染）。
3. 若命中发生（业务 prefix 污染），落盘 token 分母变小 → bytes/token 虚高 → 误判 FAIL；因此**命中>0 必须重跑**而非修正。

### 1.3 采样/统计

- **3 轮**（每轮 = 3×9K 请求），取 **bytes/token 中位数**（对单轮 du 抖动稳健）。
- 每轮间隔 ≥30s（flush 稳定窗口），并确认 `kv_offload_store_bytes` 增量 >0。
- 判定阈值：**中位 ≤ 10 KB/token → PASS**；同时记录 P25/P75 与最大值（供人工看分布）。

### 1.4 超标排查分支（bytes/token > 10KB）

| 观察值 | 疑似失效层 | 排查动作 |
|---|---|---|
| ~10–100 KB/token | trim 未完全生效（零填充削减不足） | 抽样落盘文件统计零字节占比（`od -An -tx1` / python `data.count(0)/len`）；对照补丁前 76% 零填充是否下降；对比补丁前后 du |
| ~380 KB/token | **去重失效**（5× 重复写仍在） | 确认 io.py 补丁已加载（启动日志 grep patch 标识 / python 模块 mtime）；检查去重 key 粒度与哈希实现；同文件重复序列是否仍重复落盘 |
| 10–30 KB 偏上限 | zstd-3 压缩率未达预期 | 校验文件头 zstd magic（0x28 B5 2F FD）；`zstd -lv <file>` 看 ratio；排查期可临时试 level 5/9 对比（**不改生产参数**） |
| 轮间波动大 | 并发 flush 竞争 / 生产负载干扰 | 重跑 1 轮；记录系统负载（loadavg、io util）；若业务突发则标注该轮污染 |
| 增量=0 | 未触发落盘 | 按 §1.2 放大 ctx 重试；确认 root_dir 挂载与权限 |

### 1.5 可执行命令序列（供主理人执行）

```bash
# 0) 前置：读 key、确认端点、清基线
source <INSTALL_DIR>/secrets/vllm.env
test -n "$VLLM_API_KEY" || { echo "key empty"; exit 1; }
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://<head>:8001/v1/models | head -c 300
df -h /opt/aicad-kvssd                       # 磁盘容量预检
S0=$(du -sb /opt/aicad-kvssd | awk '{print $1}'); echo "S0=$S0"
# /metrics 基线（prefix 命中与 store 计数，交叉验证用）
curl -s http://<head>:8001/metrics | grep -E "prefix_cache|kv_offload_store" | grep -v "^#" > /tmp/kv_metric_before.txt

# 1) 3×9K 请求（A 组多样文本；B 组把 TEXT 换成重复字符）
python3 - <<'EOF'
import asyncio, json, os, uuid, aiohttp
API=os.environ["VLLM_API_KEY"]; URL="http://<head>:8001/v1/chat/completions"
CHARS_PER_TOKEN=1.85; CTX=9000
# 多样文本：从常用汉字池随机采样（模拟真实 token 多样性）
import random; random.seed(42)
POOL=list("的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手")
def mk_text(n): return "".join(random.choice(POOL) for _ in range(int(n*CHARS_PER_TOKEN)))
async def req(i):
    payload={"model":"deepseek-v4-flash-0731",
      "messages":[{"role":"user","content":f"{uuid.uuid4().hex}\n{mk_text(CTX)}"}],
      "max_tokens":16,"temperature":0,"stream":False}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        async with s.post(URL, headers={"Authorization":f"Bearer {API}"}, json=payload) as r:
            j=await r.json(); u=j["usage"]
            print(json.dumps({"status":r.status,"pt":u["prompt_tokens"],"ct":u["completion_tokens"]}))
asyncio.run(asyncio.gather(*(req(i) for i in range(3))))
EOF

# 2) 等 SSD flush 稳定（轮询 store_bytes 两次一致 / 或 sleep 30）
sleep 30

# 3) 计量
curl -s http://<head>:8001/metrics | grep -E "prefix_cache|kv_offload_store" | grep -v "^#" > /tmp/kv_metric_after.txt
S1=$(du -sb /opt/aicad-kvssd | awk '{print $1}'); echo "S1=$S1"
echo "ΔS=$((S1-S0))"   # 除以 Σ(pt+ct) 得 bytes/token（脚本汇总自动算）
# 前缀命中校验：diff before/after 中 prefix 相关计数器，应无增长；有增长则本轮作废
diff /tmp/kv_metric_before.txt /tmp/kv_metric_after.txt || echo "PREFIX_CHECK: 有指标变化，人工核验命中"

# 4) 重复 3 轮（改 seed/前缀），取中位 → 判定 ≤10KB/token
```

**读回验证（G-4 后必做，暴露 KV 损坏）**：同前缀二次请求应走 SSD 读取路径（而非重新 prefill）：
```bash
python3 - <<'EOF'
# 固定 prefix P，发两次相同 9K 请求；第二次应命中 offload 读取路径（lookup 8~24ms）
# 校验：两次响应 usage 一致、输出一致、无 kv_load_failure
EOF
# 引擎日志：grep -iE "kv_load_failure|KV load failed|kv_offload.*(fail|err)" <head日志>
```

### 1.6 G-4 验收表模板

| 轮次 | 文本组 | ΔS (B) | Σ新落盘 token | bytes/token | prefix 命中 | store 增量>0 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | A 多样 | | | | 0 | ☐ | |
| 2 | A 多样 | | | | 0 | ☐ | |
| 3 | A 多样 | | | | 0 | ☐ | |
| 中位 | A | — | — | **≤10KB → PASS** | | | |
| 1 | B 重复 | | | | 0 | ☐ | 去重特性辅助 |

---

## 2. G-5 全量 benchmark 回归方案（阶段 5，约 3–4 h）

### 2.1 目标

补丁后全量 54 组合无性能回归；**重点：131072 组全 9 格跑通 + TTFT/吞吐对比**（卸载前 vs 382KB 时代 vs 补丁后）。

### 2.2 54 组合命令正确性确认（含 ⚠️ 校正）

- **54 组合 = 6 ctx（512/4096/16384/32768/65536/131072）× 3 task（coding/json/prose）× 3 conc（1/3/5）**。
- ⚠️ **校正项**：任务简报列出的 ctx 枚举（512/4096/16384/65536/131072）只含 5 档 → 5×3×3=**45**≠54，**缺 32768**（恰为并发收益分界点，必须保留）。执行命令须用 6 档 ctx；以脚本实际打印的组合数为准（脚本应输出 combos=54，若输出 45 判命令错误）。
- **--key 来源**：`source <INSTALL_DIR>/secrets/vllm.env` 后取 `$VLLM_API_KEY` 传给 `--key`。确认文件存在、key 非空、无 `\r` 污染（`tr -d '\r'`）；401 视为失败。
- **--endpoint**：`http://<head>:8001/v1`（直连，绕网关，与历史基线同口径）。
- **--model**：`deepseek-v4-flash-0731`；**--rounds 3 --engine asyncio**；`--sanity-log <head启动日志>`（事后核对 serve 参数与 offload 配置）；`--out ./results_kvssd_patch`。

```bash
source <INSTALL_DIR>/secrets/vllm.env
test -n "$VLLM_API_KEY" || { echo "key empty"; exit 1; }
cd <INSTALL_DIR>
python3 bench_prefill_decode_async.py \
  --group KVSSD_PATCH \
  --endpoint http://<head>:8001/v1 \
  --key "$VLLM_API_KEY" \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,3,5 \
  --ctx 512,4096,16384,32768,65536,131072 \
  --tasks coding,json,prose \
  --rounds 3 --engine asyncio \
  --sanity-log <head日志路径> \
  --out ./results_kvssd_patch
```

**Preflight（矩阵前 ~20min）**：
1. `/v1/models` 200，`max_model_len=600000`，served-model-name 正确；
2. TP=4 确认（head 日志 world_size / 4 rank）；
3. 启动日志确认 `kv-cache-dtype nvfp4_ds_mla` 与 TieringOffloadingSpec 配置（root_dir=/opt/aicad-kvssd、CPU 主层 2GiB、读写线程 4/4）与 io.py 补丁加载标识；
4. `df -h /opt/aicad-kvssd` 余量足够（131072 档单请求补丁后约 1.3GB，全矩阵累计数十 GB）；
5. warmup：3 个 512 ctx 请求触发 JIT/cudagraph 编译，避免首请求尖峰污染（历史：推理期 JIT 会产生一次性 TTFT 秒级尖峰）。

### 2.3 对比口径与方法（三基线）

| 基线 | 来源 | 说明 |
|---|---|---|
| B0 卸载前 | 历史 TP4 数据（如有）或同窗无 offload 对照（需重启切参数，成本高 → 优先历史） | 主对比基准 |
| B1 382KB 时代 | 灰度窗口同脚本数据 | 中间基线（标注窗口内可能含 JIT 冷启动噪声） |
| B2 补丁后 | 本次 G-5 实测 | 判定对象 |

- **指标字段**：`p50_prefill_tps` / `p50_decode_tps` / `p50_ttft_s`（per-request p50）；**跨组对比禁用 `agg_*`**。
- **并发口径**：per-request p50 × conc（c1 用单流值；c3/c5 用对应并发 p50 与基线同口径）。
- **历史参考（非 TP4 基线）**：TP2 全量矩阵 131072 格——c1 PR/DE = coding 1756.94/68.57、json 1780.57/73.70、prose 1707.58/34.90；c3 PR ~832-928、DE 3.5-6.7；c5 PR ~479、DE 5.8-6.4。**TP2≠TP4，仅作绝对合理性参考**；正式对比以 TP4 同口径基线为准。
- **已知离群复核**：16384/json/c1 历史曾现 795（瞬时干扰，TP2 复测 1953.52 平台值）；若本次该格异常，按「同 ctx 兄弟 task + 复测」判定，不单格判 FAIL。

### 2.4 对比表模板

**131072 全 9 格（重点区）**：

| ctx | task | conc | 指标 | 卸载前 B0 | 382KB 时代 B1 | 补丁后 B2 | Δ% vs B0 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 131072 | coding | c1 | PR p50 | | | | | |
| 131072 | coding | c1 | DE p50 | | | | | |
| 131072 | coding | c1 | TTFT s | | | | | |
| 131072 | coding | c3 | PR/DE/TTFT | | | | | |
| 131072 | coding | c5 | PR/DE/TTFT | | | | | |
| 131072 | json | c1/c3/c5 | 同上 | | | | | |
| 131072 | prose | c1/c3/c5 | 同上 | | | | | |

**全量 54 行骨架**（同列：ctx | task | conc | PR p50 | DE p50 | TTFT s | Δ vs B0 | 判定；由 `summary_KVSSD_PATCH.json` 展开，逐格填）：

| ctx | task | conc | PR p50 | DE p50 | TTFT s | 判定 |
|---|---|---|---|---|---|---|
| 512 | coding/json/prose | c1/c3/c5 | | | | |
| 4096 | … | | | | | |
| 16384 | …（含 16384/json/c1 离群复核格） | | | | | |
| 32768 | …（并发分界点） | | | | | |
| 65536 | … | | | | | |
| 131072 | …（重点区全 9 格） | | | | | |

### 2.5 131072 组超时风险与处理

- **1800s vs 300s 口径说明**：`aiohttp.ClientTimeout=1800s`（30min）是**压测脚本防挂死上限**，防止单请求异常把整轮拖死；300s 是**业务侧单请求 P99 SLA**。两者口径不同——压测不要求单请求 ≤300s，超 300s 记录实际值但不自动 FAIL。
- **正常预期**：131072/c1 prefill ~1756 t/s → 单请求 TTFT ~70-75s（TP2 参考）；c3/c5 并发下 per-request TTFT 随排队增长，c5@131K 单格 ~8min（v0.27 实测量级）——均在 1800s 内。
- **处理**：单格 1800s 超时 → 记为 timeout、重试 1 次；重试仍超时 → 检查 KV offload 是否阻塞 prefill（对比灰度 lookup 8-24ms 是否拉长到秒级 / store_time 是否异常），区分「SSD 卸载路径阻塞」与「历史已知的 131072 decode 崩塌（TTFT 阻塞，非带宽）」。
- **历史认知锚点**：131072 decode 曾崩塌是**长 prefill 占调度导致的 TTFT 阻塞，非带宽问题**（threshold/priority 已缓解，TP2 全量证实恢复）；补丁后若 c5@131K decode 与基线同构维持或改善 → 不判 FAIL。

### 2.6 通过/失败判据（Gate G-5）

| # | 判据 | 通过条件 |
|---|---|---|
| G5-1 | 131072 全 9 格跑通 | 无 1800s 超时、无 KV OOM、err=0 |
| G5-2 | 131072 TTFT | Δ ≤ +10% vs B0；**超标则记录实际值供人工定档，不自动 FAIL** |
| G5-3 | 131072 prefill/decode p50 | Δ ≥ −10% vs B0；decode c1 全平（±10%，参照历史 gate） |
| G5-4 | 全量 54 格完整性 | 54 格均有 summary、err=0、无缺失格（含 131072/prose/c5，历史缺失格本次补齐） |
| G5-5 | 全局健康 | kv_load_failure=0（`kv_load_failure_policy=fail` 下任何读失败都会硬暴露）、preemption 增量=0、无 OOM |

### 2.7 人工定档建议

| 场景 | 建议 |
|---|---|
| TTFT 劣化 ≤10% 且吞吐达标 | **PASS**，正常上线 |
| TTFT 劣化 10–25%、吞吐达标 | 记录并呈报：**可接受但需观测**（结合内存增量 ≤3GB 与 read 延迟综合判定） |
| TTFT 劣化 >25% 或 131072 超时/崩塌 | **FAIL**：回滚 io.py 补丁（或降级 offload），报架构师 |
| 仅 c5@131K decode 维持历史崩塌（与 B0/B1 同构） | 不判 FAIL，记录（历史已知特征，非补丁引入） |
| 内存增量 >3GB 或 03/04 OOM | **FAIL**（见 §3 风险） |

### 2.8 执行顺序与耗时（~3–4 h）

| 段 | 内容 | 预估 |
|---|---|---|
| P0 | Preflight（§2.2 五步 + warmup） | 20 min |
| P1 | 全量 54 格矩阵（短 ctx 快、131072 慢；c5@131K 每 task ~8min） | 2.5–3 h |
| P2 | 数据汇总 + 对比表 + 判定 | 20 min |
| P3 | 131072 异常专项复核（仅触发时） | +15 min |

---

## 3. 风险与坑

### 3.1 生产负载干扰
- **风险**：集群有真实流量（历史 kv_cache_usage ≤8%、prefix 命中 90%+），业务突发会污染压测与 G-4 计量。
- **对策**：
  1. G-4 计量窗口确认无其他请求（store_bytes 增量清零窗口）；G-5 与业务低峰对齐；
  2. 压测请求一律随机前缀（防业务 prefix 污染 + 自身命中=0）；
  3. 测试期间监控 `kv_cache_usage_perc` / `num_requests_waiting` / e2e latency；业务突发 → 标注该轮数据受污染并重跑。

### 3.2 内存水位（03/04 仅 6G available）
- **风险**：补丁（去重哈希 + zstd 缓冲）内存增量目标 ≤3GB；03/04 余量 6G → c5@131K 大并发是内存峰值场景，03/04 侧容器可能先 OOM。
- **对策**：
  1. 测试前 `free -g`/`cat /proc/meminfo` 记录 4 机 available；G-4/G-5 每阶段前后对比；
  2. 容器 RSS 增量监控（目标 ≤3GB）；任一机 available <2G 或出现 OOM-killer → 立即停止，报 SRE；
  3. 大并发长序列（c5@131K）放矩阵后段，单独盯内存。

### 3.3 补丁缺陷导致 KV 损坏的暴露（kv_load_failure_policy=fail）
- **暴露方式**：fail 策略下任何 KV 读失败都会硬失败 → 引擎日志 `kv_load_failure`/`KV load failed` 计数、客户端 500/空响应、decode 输出质量突变（乱码/无意义）。
- **对策**：
  1. **G-4 后立即读回验证**（§1.5）：同 prefix 二次请求走 SSD 读取路径，输出与首轮一致 → 在矩阵 3h 前尽早暴露损坏；
  2. G-5 每组合 err=0 硬约束；coding/json 可做结构合法性抽查（JSON 可解析、代码无乱码）；
  3. 全矩阵监控 `kv_load_failure` 指标增量=0；
  4. 回滚锚点：io.py 补丁为挂载/文件替换，可快速回滚；失败时回滚后重测 131072 关键格确认恢复。

### 3.4 其他坑
- **du 计量陷阱**：必须 `-sb`；fs 层若延迟删除（已删未回收），du 会含虚增 → 采样以两次一致为准；必要时 `fstrim` 后重测。
- **SSD 容量/寿命**：131072 档补丁后单请求 ~1.3GB、全矩阵累计数十 GB；`df -h` 预检 + 观察写放大（对比 du 与 kv_offload_store_bytes）。
- **口径陷阱**：TP2 历史数字 ≠ TP4 基线；382KB 时代数据含灰度窗口噪声；16384/json/c1 历史离群按平台值复核。任一跨配置对比须在报告标注口径边界。

---

## 4. 交付物与数据留存

| 产物 | 路径 |
|---|---|
| 本方案 | `deliverables/engineering-assurance/kvssd-perf-test-plan-2026-08-19.md` |
| G-4 计量原始输出 | `/tmp/kv_g4_round{1..3}.json`（含 pt/ct/ΔS/bytes-per-token） |
| G-5 矩阵原始行 | `results_kvssd_patch/rows_KVSSD_PATCH.csv` |
| G-5 矩阵汇总 | `results_kvssd_patch/summary_KVSSD_PATCH.json` |
| 对比表（模板 §2.4 填完） | `kvssd-vs-baseline-compare-2026-08-19.md` |
| 读回验证输出 | `/tmp/kv_g4_readback.json` |
| 引擎日志关键行 | 启动日志 grep（offload 配置 / kv_load_failure / JIT） |

> 本方案仅设计，不执行；待 io.py 补丁部署就绪后由主理人按 G-4 → G-5 顺序调度执行。关键决策请由人类工程负责人复核。
