# TP2 65536 上限全面测试方案（配置变更后首次全量）

**日期**：2026-08-10
**编制**：Tessa（测试专家）
**执行**：team-lead（本机驱动，测试目标 01 head .186:8001 + 02 worker）
**被测配置**：TP2（768K / seqs12 / batched4096 / threshold2048 / priority / regular CUDA graph / shim v3 / isolcpus 0-4 / IRQ 5-9 / NCCL 默认 tuner 去 LL）
**驱动**：`C:\Users\novAI\WorkBuddy\集群部署\bench_prefill_decode_async.py`（asyncio 引擎，rounds=3，uuid 随机前缀防 cache）
**口径铁律**：per-request p50 × conc；禁 agg_*；rounds=3；uuid 随机前缀

---

## 1️⃣ 测试范围：45 组合（5 ctx × 3 task × 3 conc）

| ctx | 512 | 4096 | 16384 | 32768 | 65536 |
|-----|-----|------|-------|-------|-------|
| task | coding / json / prose | 同左 | 同左 | 同左 | 同左 |
| conc | 1 / 3 / 5 | 同左 | 同左 | 同左 | 同左 |

- 排除 131072（用户指示收缩至 65536）。
- 行数合计 **405**：c1 15 格×3 波=45；c3 15 格×9=135；c5 15 格×15=225。

---

## 2️⃣ 分块执行计划（规避后台 60min 限制）

> 驱动组合顺序 = conc 主序（for conc → for ctx sorted → for task）。分块天然按并发切开，每组独立 CSV，最终合并。

**前置健康检查（B1 前，一次）**
```bash
# 1. 模型就绪
curl -s http://<NODE_IP>:8001/v1/models -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" | head -c 400
# 2. smoke（512/c1 单发，确认 200 + usage）
# 3. head 日志确认：NCCL init PASS + regular graph 生效（VLLM_USE_BREAKABLE_CUDAGRAPH=0 落日志）
# 4. shim 生效：docker inspect vllm-head | grep LD_PRELOAD → /opt/libncclpin.so
# 5. NCCL 线程落核 0-4：ps -eLo psr,comm | grep -i nccl（或 /proc/<pid>/task/*/status Cpus_allowed_list）
# 6. nvidia-smi：01/02 GPU 利用率基线 + 显存（确认无残留 embed 在 01/02）
```

**B1 — c1 15 格（预计 ~10-15min）**
```bash
cd "C:\Users\novAI\WorkBuddy\集群部署"
python bench_prefill_decode_async.py --group TP2C \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa \
  --model deepseek-v4-flash-0731 \
  --concurrency 1 --ctx 512,4096,16384,32768,65536 \
  --tasks coding,json,prose --rounds 3 --engine asyncio \
  --sanity-log <INSTALL_DIR>/logs/vllm/nccl-*.log \
  --out ./results_TP2C_c1
```

**B2 — c3 15 格（预计 ~15-20min）**：同上，仅 `--concurrency 3 --out ./results_TP2C_c3`

**B3 — c5 512-16384 9 格（预计 ~15-20min）**：同上，`--concurrency 5 --ctx 512,4096,16384 --out ./results_TP2C_c5a`

**B4 — c5 32768+65536 6 格（预计 ~20-25min）**：同上，`--concurrency 5 --ctx 32768,65536 --out ./results_TP2C_c5b`

> 若某块超 60min：B1 可再切 512-16384 c1(9格) + 32768-65536 c1(6格)；B4 可切单 ctx。

**每块验收（driver 输出）**：`rounds_ok=requests_total`（c1 3/3、c3 9/9、c5 15/15）且 errors=0；不满足即停并查因（见风险）。

---

## 3️⃣ 合并与校验

```bash
# merge_TP2C.py：读取 4 个 rows_*.csv → rows_TP2C.csv（断言 405 行 / ok=True / err 全空）
#                按 (ctx,task,conc) 重算 p50 → summary_TP2C.json（与 driver summary 语义一致）
# analyze_TP2C.py：Δ vs B 组 / Δ vs 8-10 全量 / 32768 ratio / decode c1 哨兵 / 🟢🟡🔴 标记
```

校验清单：
- [ ] 405 行，err=0，ok=True
- [ ] 无缺失格（5×3×3 = 45）
- [ ] c1 每格 3 样本、c3 9 样本、c5 15 样本
- [ ] 32768 ratio = decode_c5×5 / decode_c1×1 计算（三 task）
- [ ] decode c1 哨兵表（全 ctx coding/json/prose）

---

## 4️⃣ 对比维度与基线（预置数值）

### 4.1 vs B 组同格（65536 内）— 基线 rows_B.csv p50（t/s）

| ctx | task | B c1 prefill | B c1 decode | B c3 prefill | B c3 decode | B c5 prefill | B c5 decode |
|-----|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 512 | coding | 1118.3 | 71.1 | 615.7 | 47.39 | 519.7 | 36.08 |
| 512 | json | 1137.9 | 77.39 | 610.1 | 51.66 | 524.5 | 39.88 |
| 512 | prose | 1141.9 | 37.72 | 506.1 | 26.29 | 368.8 | 19.97 |
| 4096 | coding | 1993.1 | 77.63 | 933.9 | 42.44 | 604.6 | 29.71 |
| 4096 | json | 2001.9 | 76.1 | 892.4 | 45.13 | 616.0 | 30.49 |
| 4096 | prose | 2061.8 | 39.91 | 951.6 | 23.05 | 629.2 | 16.69 |
| 16384 | coding | 2037.2 | 77.84 | 948.9 | 30.07 | 633.3 | 18.19 |
| 16384 | json | 795.5† | 79.8 | 977.9 | 30.99 | 654.1 | 20.03 |
| 16384 | prose | 2037.6 | 39.82 | 994.5 | 15.23 | 662.0 | 9.72 |
| 32768 | coding | 1879.2 | 74.46 | 995.9 | 19.88 | 648.0 | 11.54 |
| 32768 | json | 1977.8 | 78.79 | 970.1 | 21.32 | 646.4 | 13.03 |
| 32768 | prose | 1983.1 | 39.46 | 964.3 | 11.28 | 630.8 | 6.31 |
| 65536 | coding | 1680.1 | 76.51 | 927.6 | 12.1 | 604.5 | 7.0 |
| 65536 | json | 1841.0 | 79.33 | 926.1 | 13.89 | 609.2 | 7.5 |
| 65536 | prose | 1859.7 | 39.09 | 940.2 | 6.82 | 603.6 | 3.55 |

† = B 组瞬时干扰离群，平台值 ~1950-2000（8/10 已复核）。

### 4.2 vs 8/9 36 格子集 / vs 8/10 全量 53 格（同格 65536 内）

- 8/10 全量报告矩阵值（变更前基线：seqs6/breakable/isolcpus 16-19）即本方案「vs 全量」对照源。
- 差异维度：seqs 6→12、regular graph（BREAKABLE 1→0）、isolcpus 16-19→0-4、max-len 131072→768000、shim v3 生效。
- **天然 A/B 提示**：8/10 全量 decode c1（68.6-77.9）为 breakable 口径；本次为 regular 口径 → 同集群跨配置可直接观测 regular 收益。

### 4.3 重点观察项

1. **decode c1 是否因 regular graph 提升**：8/10 全量 coding 68.6-77.9 / json 73.7-80.7 / prose 34.9-41.4；社区 regular +28.6% → 预期 coding ~85-96。判据见 §8。
2. **seqs12 下 c5 并发是否更好**：重点看 16384-65536 段 c5 decode 是否维持/超过 8/10 全量（+8%~+61% vs B）。
3. **65536/c5 长 ctx 并发表现**：c5 decode / 32768 ratio 连续性（65536 段 8/10 ratio 0.65-0.76，观察是否随 seqs12 改善）。
4. **32768 分界保持 ≥0.77**（8/10 为 1.065/0.994/1.207，本次须 ≥0.77 gate，期望维持 ≥0.99）。
5. **c3 短 ctx decode**：8/10 报告 512/4096 c3 偏低（-10.9%~-21.6% vs B），列为观察项。

---

## 5️⃣ NCCL 延迟测试方案

**目标**：验证隔离核 0-4（A725 能效核）下 16B all_reduce 延迟，对比历史 X925 16-19 基线。
**基线**：16-19（X925）+ LL avg 16.3µs / i_p99 19-26µs；+ Simple avg 24.6µs / i_p99 27.7-29.7µs（8/9 Phase B，01↔02，同节点对）。

**判据**：Simple 口径 **p99 ≤ 40µs**（验证 A725 降频轮询不拖垮 Simple 延迟）。

**命令序列**（在 01 执行；LD_LIBRARY_PATH 前插 /opt/nccl-2307；GID=2；per-node HCA；--bind-to none）
```bash
# 默认 tuner（生产口径，去 LL）——记录 tuner 实际选中协议（NCCL_DEBUG 日志）
mpirun --hostfile /tmp/nccl_hosts -np 2 --bind-to none \
  --mca oob_tcp_if_include enP7s7 --mca orte_tcp_if_include enP7s7 --mca btl_tcp_if_include enP7s7 \
  -x LD_LIBRARY_PATH=/opt/nccl-2307/lib:$LD_LIBRARY_PATH \
  -x NCCL_IB_GID_INDEX=2 -x NCCL_SOCKET_IFNAME=enP7s7 -x NCCL_DEBUG=INFO \
  taskset -c 0-4 /opt/nccl-tests/build/all_reduce_perf \
  -b 16 -e 16 -f 2 -g 1 -w 1000 -n 10000 -z 0 -I 1

# Simple 显式（对齐历史 Simple 24.6µs 基线）
... -x NCCL_PROTO=Simple ...（同上其余参数）

# LL 显式（诊断性，历史 16.3µs 基线参考）
... -x NCCL_PROTO=LL ...（同上其余参数）
```
- 每组 **5 run**（对齐 Phase B G1-G6 方法），记录 avg / i_p99 / i_max / 每 run 值。
- 判据矩阵：Simple p99≤40µs 🟢；40-50µs 🟡 复核；>50µs 🔴。
- **执行时机**：建议在 CUDA graph A/B 重启窗口（TP2 停机）或 LLM 45 格完成后 RoCE 空闲时跑，避免与在线 TP2 争用 HCA 造成互相污染。

---

## 6️⃣ Regular CUDA graph A/B 方案

**A = regular（当前，BREAKABLE=0）**：45 格全量 decode 数据已由 §2 采集（B1-B4）。

**B = breakable（BREAKABLE=1）**：需一次 TP2 重启（head-first），只跑关键 decode 格：

| 格 | 目的 |
|----|------|
| 512/c1 | 短 ctx decode 单流 |
| 16384/c1 | 中 ctx decode 单流 |
| 32768/c1 | 分界点 decode |
| 65536/c1 | 长 ctx decode 单流 |
| 65536/c5 coding | 长 ctx 并发 decode（regular 并发正确性 + 并发收益） |

**B 腿命令**（与 A 腿完全同参，仅 out 目录不同）：
```bash
python bench_prefill_decode_async.py --group TP2B \
  --endpoint http://<NODE_IP>:8001/v1 \
  --key <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,5 --ctx 512,16384,32768,65536 \
  --tasks coding,json,prose --rounds 3 --engine asyncio \
  --out ./results_TP2C_breakable
```
（c1 4 格 + c5 4 格 = 12 组合；也可按需砍到 team-lead 指定的 4 格：512/c1、16384/c1、65536/c1、65536/c5 coding）

**重启步骤（head-first）**：
1. 01 改启动脚本 env `VLLM_USE_BREAKABLE_CUDAGRAPH=0 → 1`（其余不动，保持隔离核/shim/seqs12）
2. 跑 `check_vllm_script.sh start_head_v026r.sh` 自检 + `bash -n`
3. head-first 重启：head 先起（EngineCore ≤10min）→ worker → :8001 就绪
4. 跑 B 腿关键格 → 对比 → 定稿

**判定**（decode c1 同格对比）：
- regular 优于 breakable（Δ>10%）：**维持 regular（当前态，无需再重启）**
- breakable 优于 regular（Δ>10%）：**切回 breakable（再重启一次设 =1）**
- 差异 ≤10%：**维持 regular**（社区 +28.6% 未复现，regular 无兼容性风险则取简单路径）
- **并发正确性检查**：B 腿 65536/c5 coding 必须 err=0、ok=15/15（regular 下动态 batch 重捕获风险观察项）。

---

## 7️⃣ Embed 测试方案

**拓扑**：03=<NODE_IP>:8022、04=<NODE_IP>:8022（anemll-embed-8022，Qwen3-Embedding-0.6B）；litellm 网关 02:4000（/v1/embeddings → 03/04 simple-shuffle）。
**历史基线**：直连单机 c16=553 req/s；网关双机 c16=362（上限 360-420）；双机经网关 c8+28%/c16+32%；p50 c1 ~21.8ms / c16 40.5ms。

**复用脚本**：`_archive_scratch/bench_B/embed_bench_litellm.py`（httpx asyncio；mode dual/single/direct3/direct4；--conc --requests）。
**命令序列**（客户端跑在 02 网关机，与历史同机口径；key 用当前 litellm master key）：
```bash
# 直连 03 / 直连 04（理论上限对照）
python3 embed_bench_litellm.py --conc 1,4,8,16 --requests 30 --mode direct3
python3 embed_bench_litellm.py --conc 1,4,8,16 --requests 30 --mode direct4
# 经 litellm 网关（双机池，生产口径）
python3 embed_bench_litellm.py --conc 1,4,8,16 --requests 30 --mode dual
# 批量输入变体（单请求多文本，payload input=[TEXT]*8，验证 --max-num-seqs 32 batch 语义）
python3 embed_bench_batch.py --conc 1,4,8,16 --requests 20 --batch 8
```
**指标**：p50/p95/p99（ms）+ 吞吐 tps + err 数。
**判据**：
- 直连单机 c16 ≥ 553（回归确认）；网关 c16 ≥ 360-420 区间 🟢；低于区间 20% 🔴
- 双机经网关 c8/c16 相对单机增益 ≥ +20% 🟢
- 全程 err=0 🟢
- **执行时机**：LLM 45 格与 A/B 完成后执行（网关在 02 与 TP2 worker 同机，并发压测会扰动 LLM 侧，避免同时）。

---

## 8️⃣ 判定阈值（预锁定）

| 维度 | 🟢 通过 | 🟡 复核 | 🔴 阻塞 |
|------|---------|---------|---------|
| vs B 组同格 Δ | \|Δ\| ≤10% | 10-20% | >20% |
| decode c1（regular 收益） | 较 8/10 全量提升 >10% | 0-10% | 回退 >10% |
| 32768 分界 ratio | ≥0.77（期望 ≥0.99） | 0.60-0.77 | <0.60 |
| 数据质量 | 405/405 ok，err=0 | 少量 err（<1%）可重跑 | err>1% 或缺失格 |
| NCCL Simple p99 | ≤40µs | 40-50µs | >50µs |
| embed | 回归达标 + 双机增益≥20% | 单点略低 | 吞吐腰斩 / err>0 |

**结论口径**：decode c1 提升 >10% → 判 regular graph 显著收益（社区 +28.6% 复现证据）；否则维持现状并记录。

---

## 9️⃣ 报告框架（产出文件）

`deliverables/engineering-assurance/benchmark-tp2-65536-config-audit-2026-08-10.md`

1. TL;DR + 整体判定（45 格全绿/有回退）
2. 45 格矩阵表（prefill/decode p50，c1/c3/c5 三源：本次 vs B vs 8/10 全量）
3. 对比维度分析（§4 四项重点观察逐一结论）
4. 32768 分界判定（ratio 表 + 机理）
5. decode c1 健康哨兵（regular graph 收益判定）
6. NCCL 延迟结果（0-4 A725 vs 16-19 X925 基线表 + 判据）
7. CUDA graph A/B 结论（regular vs breakable 表 + 定稿配置）
8. Embed 结果（直连/网关/批量 + 判据）
9. 配置变更影响分析（seqs12/regular/隔离核 0-4/shim/768K 各维度证据）
10. Gate 判定 + 风险提示 + 数据来源索引

---

## 🔟 执行顺序与风险

**建议顺序**：预检 → B1 → B2 → B3 → B4 → 合并校验 → NCCL（RoCE 空闲窗）→ CUDA graph B 腿（重启窗内可顺带跑 NCCL）→ Embed → 报告

**风险登记**：
1. **regular graph 首形状捕获尖峰**：c5 新 batch shape 首次可能捕获耗时 → 首波 TTFT 尖峰；p50 3 波吸收，若 err 则重跑该块。
2. **65536/c5 总时长**：5 个大 prefill 同时 chunked 切块，单格可能 2-4min；B4 预留 25min。
3. **60min 后台限制**：分块已控；超时预案见 §2。
4. **NCCL 与在线 TP2 争用**：绝不与 LLM 压测同时跑 nccl-tests（同 HCA）。
5. **embed 与 TP2 worker 同机（02）**：embed 压测放 LLM 块之后。
6. **B 腿重启风险**：head-first + check_vllm_script 前置自检（防 8/10 重启事故复发）。

---
> 本方案由工程保障团队测试专家编制，执行与判定由 team-lead 主导；阈值预锁定，报告统一由本方案框架产出。
