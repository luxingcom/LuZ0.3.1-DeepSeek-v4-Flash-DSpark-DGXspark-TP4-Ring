# E2E 基线测量 — 生产方案（Dspark MTP）vs 基准包参考值 — 2026-08-21

**执行**: Rex（SRE） · **窗口**: 2026-08-21 07:06 ~ 07:55 UTC · **被测**: 生产部署方案原样（start_tp4_head.sh 零修改，.bak-plugin-20260821 恢复版 diff 确认 IDENTICAL）
**基准包**: benchmark_package_20260819（口径严格遵循 docs/测试方案.md：decode-only 自首 token 计时 / 预热 25 请求 / 中位数聚合 / ≥3-10 轮）
**原始日志**: `e2e-baseline-results/results/`（00_warmup ~ 05_code_agent + master log）；服务器侧 /tmp/bench_pkg/results/

---

## 0. 执行摘要

生产恢复一次成功（绕过 B12X 门禁死锁的手动编排），基准全量跑完（07:43:40 ALL_BENCH_DONE）。**核心结论：高并发聚合与 prefill 吞吐全面超基准包参考值（C8 +13~20%、C12 +14~19%、PR +14~42%），单流接近参考中位，Agent Code/工具调用场景大幅超参考（+16~35%）**。停机窗口结束：生产保持运行，自愈链（head/worker systemd monitor + healthcheck timer）已评估并安全开启。

---

## 1. 生产恢复（任务前置）

### 1.1 启动方式
- `restart_run.sh 0`（P2 手动编排脚本，/tmp/_routea_work/）：MODE=0（A 基线）写入四节点 → 清理容器 → head 先起 → 轮询 TCPStore :25999 就绪 → 15s 错峰启 3 workers（rank1@02 / rank2@04 / rank3@03）→ 等 "Application startup complete"
- **绕过了 start_tp4_cluster.sh 的 B12X 门禁死锁**（该死锁不改脚本无法修复，集群脚本未使用）

### 1.2 启动验证（全部通过）
| 检查项 | 结果 |
|---|---|
| 4 rank 容器 | vllm-tp4-rank0/1/2/3 全部 Up (healthy)（01/02/04/03） |
| /health | 200 |
| MoE 后端 | `Using 'B12X_MXFP4' Mxfp4 MoE backend`（07:11:51） |
| **dspark speculative 生效** | speculative_config={'method':'dspark','num_speculative_tokens':7,'draft_sample_method':'probabilistic'}；`DSpark draft model loaded: 96 params`；`Capturing model for DSpark speculator` + dspark CUDA graphs 11/11 |
| KV cache | **6,024,962 tokens** |
| GPU 参与 | 四节点 nvidia-smi 均 **96%**（并发期快照） |

### 1.3 启动期关键数字（实际生效参数）
- max_num_seqs=12、max_num_batched_tokens=4096、**max_num_scheduled_tokens=4024**（vLLM 因 speculative n=7 自动下调，有 WARNING 建议增大 batched tokens——生产现状记录，未修改）
- gpu_memory_utilization=0.80、kv_cache_dtype=nvfp4_ds_mla、moe_backend=flashinfer_b12x
- cudagraph sizes 1..96、prefix caching + chunked prefill（threshold 1024）、flashinfer autotune
- vLLM v0.26.1.dev0+gd3d3b2cca.d20260805、quantization=deepseek_v4_fp8

---

## 2. 基准结果 vs 参考值对照表

> 我方 = 生产方案口径（dspark n=7）；参考值 = 基准包 8/19 复测（中位 | 最优），**仅对照不套用**

| 指标 | 我方（生产方案） | 参考（中位 \| 最优） | vs 中位 | vs 最优 |
|---|---|---|---|---|
| **C1 单流**（fox 512t，10 轮） | **92.8** 中位（范围 82.0-97.9，均值 90.3） | 97.1 \| 124.0 | -4% | -25% |
| C1 编号列表（3 轮） | 125.2 中位 | —（参考未单列） | — | — |
| **C4 聚合**（中位×C） | **237** | 218.0 \| 233.7 | **+9%** | +1% |
| **C8 聚合** | **343** | 286.3 \| 302.9 | **+20%** | **+13%** |
| **C12 聚合** | **408** | 342.8 \| 358.2 | **+19%** | **+14%** |
| Agent 代码生成（红黑树 g2048，3 轮） | **132.9** | 98.6 \| 102.4（Code 场景） | **+35%** | +30% |
| Agent 工具调用（g310，3 轮） | **126.9** | 105.8 \| 109.4 | **+20%** | +16% |
| Agent 5 场景平均 | 未测全（任务指定脚本不含 3rounds 全景） | 81.3 \| 84.6 | — | — |

**PR（Prefill Rate，panorama，prompt_tokens/TTFT 中位，3 轮）**:

| 上下文 | 我方 | 参考 | 差异 |
|---|---|---|---|
| 4K | **~2510**（2415-2527） | 2.2K | **+14%** |
| 16K | **~2500**（2494-2504） | 2.0K | **+25%** |
| 32K | **~2420**（2418-2429） | — | — |
| 64K | **~2270**（2261-2275） | 1.6K | **+42%** |

> 用户参考锚点 "TP4 PR 2500"：我方 4K-16K 段 2500-2510 完全持平，32K/64K 长上下文衰减更平缓（参考 64K 掉到 1.6K，我方仍 2270）。

**Panorama Decode 全景（聚合 tok/s，total_ct/wall 口径）**:

| ctx | C1 | C4 | C8 | C12 |
|---|---|---|---|---|
| 256t | 83.8 | 188.5 | 250.9 | 316.7 |
| 4096t | 70.1 | 121.2 | 170.8 | 178.0 |
| 16384t | 59.3 | 164.5 | 266.8 | 314.6 |
| 65536t | 77.1 | 95.3 | 121.9 | 119.6 |

（注意 panorama 与 conc_decode 的 C12 差异 316.7 vs 408 来自口径不同：panorama 固定上下文 + total/wall，conc 为每流中位×C；对参考值对照采用 conc_decode 口径，与参考数据同源）

### 与参考值差距一句话
**高并发与 prefill 全面领先参考部署（C8/C12 +13~20%、PR +14~42%），Agent Code/工具调用 +16~35%；唯单流 C1（92.8 vs 97.1|124.0）低于参考中位 4%**——单流对 MTP 接受率更敏感（dspark n=7 vs 参考 n=5 的接受长度差异），且我方 conc_decode 的 C1（73.0，短预热段）与 c1_10rounds 的 10 轮中位（92.8）本身有波动（82-98）。

---

## 3. 配置差异表（只列不套用）

| 参数 | 我方（生产方案） | 基准包参考部署 |
|---|---|---|
| 投机解码 | **dspark n=7, probabilistic**（生产脚本内） | vLLM V1 MTP n=5 |
| max_num_seqs | **12** | 16 |
| max_num_batched_tokens | **4096** | 8240 |
| gpu_memory_utilization | **0.80** | 0.82 |
| kv_cache_dtype | nvfp4_ds_mla（相同） | nvfp4_ds_mla |
| autotune | enable-flashinfer-autotune（相同） | True |
| MoE backend | flashinfer_b12x（B12X_MXFP4，生产特有） | —（未注明） |
| cudagraph | max 96 / 17 档 sizes | — |
| KV 池 | 6.02M tokens | 7.9M tokens |
| max_num_scheduled_tokens | 4024（speculative 自动下调） | — |
| 编排 | TP4 4 节点 head-first（restart_run.sh 手动编排） | TP4 4 节点 |

---

## 4. MTP（Dspark）生效证据

1. 启动日志：`DSpark draft model loaded: 96 params`；`Capturing model for DSpark speculator...`；`Capturing dspark CUDA graphs (FULL): 11/11`
2. metrics: `vllm:spec_decode_num_drafts_total 38398`、`num_draft_tokens_total 268786`
3. **Acceptance 采样（metrics.py 每 10s）**:
   - 规律文本（编号列表/代码类）: Mean acceptance length **6.47-7.13**（n=7 理论上限 8），Per-position 0.985/0.974/0.948/0.902/0.856/0.794/0.675，**Avg Draft acceptance 78-88%**
   - Agent/自由文本场景: Mean acceptance length 2.88-5.34，avg 26.9-62%
   - 混合均值（07:42-07:43 窗口）: 规律段 Accepted throughput 108-119 tok/s vs Drafted 129-136 tok/s

---

## 5. 停机窗口收尾：生产保持运行 + 自愈链开启

**基准完成后生产持续运行**（rank0 Up 45min healthy，health 200）。

**自愈链评估与开启**（任务授权评估）：
- 评估: vllm-tp4-head.service / vllm-tp4-worker.service(×3) 的 monitor 脚本均为**安全 attach 模式**（容器在跑 → `docker wait` 跟随，不扰动现有生产；容器退出 → systemd Restart 全链重建：head 清 workers + start_tp4_head.sh + D3 rank 门禁，worker 侧 TCPStore 门禁 + D1 模型门禁 + D2 指数退避）——重建路径用 head/worker 脚本，**不经过 start_tp4_cluster.sh 的 B12X 死锁门禁**，链路安全
- 一致性确认: rank 映射（02=rank1/<NODE_IP>、04=rank2/189、03=rank3/188）与 unit Environment 一致；master <NODE_IP>:25999 一致
- **已开启**: vllm-tp4-head.service（01）+ vllm-tp4-worker.service（02/03/04）+ vllm-healthcheck.timer（01，60s 探针/失败触发重建，cooldown 1800s）全部 active；开启后生产零扰动（容器 Up 未重启、health 200）
- ⚠️ 备注: 02/04 的 worker unit 有 "changed on disk, loaded version outdated" 警告（P1/P2 期间磁盘 unit 改过未 daemon-reload）；当前加载版本工作正常，未做 daemon-reload（避免未知磁盘版本生效引入风险），**建议下次维护窗口核查磁盘 unit 差异后统一 reload**

---

## 6. 遗留与建议

1. **[P1-建议] B12X 门禁死锁修复**: start_tp4_cluster.sh 的门禁等 "Using B12X_MXFP4"（模型加载后才打）但模型加载需 4-rank rendezvous 被 workers 被门禁挡 → 两次复现死锁。当前以 restart_run.sh 绕过，**自愈链已避开该路径**，但 vllm-cluster 体系恢复需修脚本
2. **[P2-建议] max_num_batched_tokens=4096 被 speculative 压到 4024**: vLLM 明确 WARNING 建议 4096→更大以容纳 draft slots（参考部署用 8240）。生产脚本未动；若后续调优可评估 8240（需重新压测）
3. **[P3-记录] 单流 C1 92.8 低于参考最优 124.0**: 建议后续单独排查（MTP n=7 vs n=5 的单流接受率差异 / cudagraph 96 档覆盖 / autotune 缓存状态）
4. [记录] 04/03 各有一个 anemll-embed-8022 容器在跑（embedding 服务，非 TP4 生产，未触碰）
5. [记录] bench 期间未复现昨日 nfsd 问题（03/04 NFS 全程响应正常）

## 7. 时间线（UTC）

| 时间 | 事件 |
|---|---|
| 07:06-07:09 | Preflight（四节点干净停机、脚本原版 diff、MODE=0、restart_run.sh 定位） |
| 07:09 | restart_run.sh 0 启动（head → TCPStore → 错峰 workers） |
| 07:11:51 | B12X_MXFP4 MoE backend 生效 |
| 07:14:39 | KV cache 6,024,962 tokens |
| 07:15:32 | DSpark speculator CUDA graphs 捕获完成 |
| ~07:16 | RUN READY → 基准包部署（scp + API 配置）+ warmup 25 请求 |
| 07:16-07:43 | 基准五阶段：decode_only → conc → c1_10 → panorama → code_agent |
| 07:43:40 | ALL_BENCH_DONE |
| 07:50-07:55 | 自愈链评估 + 开启（4 unit + timer，零扰动验证） |
