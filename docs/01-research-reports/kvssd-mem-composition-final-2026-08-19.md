# TP4 UMA 内存耗尽深度剖析 — 构成分解与参数优化（最终汇编）

**日期**：2026-08-19
**工作流**：技术分析（内存模型）
**参与成员**：Rex（实机构成数据）/ Cody（内存模型与参数矩阵）/ 主理人汇编
**数据来源**：`kvssd-mem-composition-data-2026-08-19.md`（Rex 实机）+ `kvssd-mem-model-and-tuning-2026-08-19.md`（Cody）

---

## 📌 TL;DR

- 121GB UMA 的 86G 由 vLLM 设备内存占走（权重 40.55 + **KV 池 42.43** + 激活 2.03 + cudagraph 0.9），另有 CUDA/框架开销 15-18G、03/04 的 embed 5.75G → **空闲 avail 仅 8.1/8.9/3.1/2.6G**，swap 已用 4-14G → **无安全边际**
- 内存耗尽机制：**KV 池 42.4G 对 60 万 ctx 满载（48G+）不足 → KV 卸载流量激增**（单条 64K 请求 store +28.9GB、cpu 主层 2GiB 缓冲冲至 60.8%、KV allocation_failure 单请求 +80）→ 突发并发（conc3×65536）时设备内存申请失败（NVRM NV_ERR_NO_MEMORY）→ 内核 OOM → NCCL 超时是受害者
- **优化杠杆排序**：①`max-num-seqs 12→8`（KV 最坏量 -33%）②03/04 embed 迁出（省 5.75G）③`cudagraph capture 96→64` ④cpu_bytes 2→1GiB（利用率 0-1%）⑤KV 池预留与卸载平衡再调
- 0.7 已实证有效（conc3×65536 4 连过），但 03/04 谷底 2.5G 仍薄，**需组合优化后复测**

---

## 一、为什么出现内存耗尽？（机制）

1. **UMA 结构约束**：DGX Spark 无独立显存，121GB 由 GPU 与 CPU 共享。vLLM 以设备内存方式持有大部分（**不计入容器 cgroup**，free 看不到细节）。
2. **固定盘（不可压缩）**：权重 40.55G + embed 5.75G（03/04）+ CUDA/框架 15-18G ≈ **62-64G 固定占用**。
3. **动态盘（可增长）**：KV 池 42.43G（0.7 util 预算）+ 激活峰值（长上下文 prefill 放大）+ 页缓存。60 万 ctx × 12 seq 满载 KV ≈ 48G+ **超出 KV 池预算** → 触发卸载。
4. **卸载流量放大**：KV 池不足时，LRU 驱逐 → SSD 卸载。单条 64K 请求即 store +28.9GB，cpu 主层 2GiB 缓冲冲至 60.8%，**KV allocation_failure 单请求 +80（876→956）**——卸载链路成为内存与 IO 压力放大器。
5. **触发链**：conc3×65536 突发 = 3 条长上下文并发 prefill（激活峰值 × KV 增长 × 卸载流量）→ 设备内存申请失败（03 于 03:18 NVRM NV_ERR_NO_MEMORY、03:20 avail=0）→ 内核 oom-killer（03:54 杀 Worker_TP 279G）→ 系统冻结 → NCCL allreduce 无进展 300s 超时。

## 二、哪些数据交换增加了内存占用？（按量排序）

| 项 | 实测/估算 | 性质 |
|---|---|---|
| **KV 池** | 42.43G/节点（0.7 util 预算） | 动态，60 万 ctx 满载 48G+ 不足 |
| **KV 卸载流量** | 单条 64K store +28.9GB；cpu 主层缓冲 60.8%；allocation_failure +80/请求；kvssd 落盘 40G | **最大数据交换**（GPU→CPU 主层→SSD） |
| **权重驻留** | 40.55G/节点（EP=TP4 分摊，43 层 MoE） | 固定 |
| **CUDA/框架开销** | 15-18G（自报 86G vs 实际 ~104G 差额） | 固定 |
| **embed（03/04）** | 5.75G（Qwen3-0.6B embed 服务同机） | 固定，可迁出 |
| **激活** | 2.03G 自报（prefill 峰值更高） | 动态，长 ctx 放大 |
| **cudagraph 池** | 0.9G（capture 96 档） | 固定 |
| **swap** | 已用 4.3-14.3G | 兜底（性能代价） |

## 三、真实占用与构成（121GB 分解）

```
121G UMA ≈ vLLM 设备内存 ~104G（nvidia-smi 实测）
  ├─ vLLM 自报 ~86G：权重 40.55 + KV 池 42.43 + 激活 2.03 + cudagraph 0.9 + non-torch 0.13
  ├─ CUDA/框架未记账 15-18G
  ├─ embed 服务（03/04）5.75G
  └─ 系统/页缓存/宿主机进程（容器 cgroup 外）
空闲 avail：01=8.1G 02=8.9G 03=3.1G 04=2.6G；swap 已用 4.3-14.3G → 无安全边际
```

## 四、参数优化矩阵（P0→P2）

| 优先级 | 参数 | 当前 | 建议 | 内存影响 | 风险/备注 |
|---|---|---|---|---|---|
| P0 | max-num-seqs | 12 | **8** | KV 最坏量 -33%（48G→32G 满载） | 并发吞吐降，R11 历史曾用 6 |
| P0 | embed（03/04） | 同机 5.75G | **迁出/独立机** | 03/04 释放 5.75G | 需另安排 embed 部署 |
| P1 | cudagraph capture | 96 | **64** | cudagraph 池缩减 | 覆盖 1-64 档，R11 曾用 64 |
| P1 | cpu_bytes_to_use | 2GiB | **1GiB**（利用率 0-1%） | 省 1G | 卸载缓冲降低，需复测卸载吞吐 |
| P1 | KV 池预留/--kv-cache-memory | util 0.70 | 评估显式 `--kv-cache-memory` 或下调预留 | 释放 KV 池 | 与卸载平衡：池小→卸载流量增，池大→余量减 |
| P2 | gpu-memory-utilization | 0.70 | 维持（勿再降） | — | 0.65 释放 6G 但 KV 池再缩 |
| P2 | max-num-batched-tokens | 4096 | 2048-8192 评估 | prefill 峰值 | 批大小权衡 |
| P2 | swap | 15G | 评估 zram/压缩 | 兜底 | 性能代价 |

**组合预期**：seqs 8 + capture 64 + cpu_bytes 1GiB + embed 迁出 → 03/04 谷底从 2.5G 抬至 **5-7G**，且不牺牲 KV 池容量、不放大卸载。落地后以 **conc3×65536 复测**为放行判据。

## 五、遗留风险

1. 0.7 空闲重启 warmup 仍触发 NVRM NV_ERR_NO_MEMORY（01@05:13、03@05:13/17）——**模型加载峰值已逼近极限**，无安全边际
2. KV 卸载 allocation_failure（+80/请求）需确认是否影响推理质量（kv_load_failure_policy=fail 下未报错，但值得监控）
3. 权重驻留 45-55G/节点估算需启动日志 `gpu_memory_used` 钉死

---

> 本报告由工程保障团队 AI 协作生成，数据以实机取证为准，参数调整须经用户批准后灰度。
