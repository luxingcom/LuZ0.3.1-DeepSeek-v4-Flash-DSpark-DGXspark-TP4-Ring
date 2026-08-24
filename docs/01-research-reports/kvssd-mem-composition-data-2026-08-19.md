# TP4 集群实机内存构成数据采集报告（4×DGX Spark UMA）

- **采集人**：雷克斯（Rex）· SRE 工程师
- **日期**：2026-08-19 05:35–06:20（节点时间）
- **范围**：node01~04（vllm-tp4-rank0~3，TP4，121GB UMA 共享内存，无独立显存）
- **方法**：SSH 只读采集（free / docker stats / cgroup v2 memory.* / /proc/meminfo / nvidia-smi / smaps_rollup / vLLM metrics+启动日志），未改动任何生产配置、未重启
- **状态**：采集时 4 rank 均 healthy（health 200），gpu-memory-utilization=0.70

> 说明：03/04 主机于 05:00 左右重启（up 39min），vllm 四节点均在 05:09–05:10 重启加载模型；01/02 主机已 up 1d21h。所有"空闲态"数据为 05:36 前后、无在途请求时采集；"轻载态"为 01 容器内跑 1 条 64K 单请求（conc1，安全）时/后采集。

---

## 1. 核心结论（TL;DR）

1. **空闲态即告急**：4 节点空闲时 `available` 仅 **8.1 / 8.9 / 3.1 / 2.6 GB**（03/04 更低是因为同机还跑着 embed 服务占 5.75GB 设备内存）。所谓"稳态 24/24/18/18G"在当前负载与模型组合下不复存在。
2. **121GB 去向的主体是 vLLM 的 GPU（UMA）设备内存分配**：nvidia-smi 显示每节点 vLLM Worker 持 **~104.1GB** 设备内存，其中 vLLM 自报：**权重 40.55GB + KV cache 池 42.43GB + 激活峰值 2.03GB + CUDA Graph 0.90GB + non-torch 0.13GB**，其余 ~15–18GB 为 CUDA/driver/框架开销（待分析）。
3. **换页已在兜底**：即使空闲，swap 已用 **11.8GB(01) / 14.3GB(02) / 4.3GB(03) / 4.3GB(04)**（16G swap.img）。01/02 的 swap 是 0.8 事故/长时间运行的遗留。
4. **KV 卸载是"数据交换占用内存"的主要来源**：单条 64K 请求即触发 **store_bytes +25.47GB**（累计写流量，非常驻；逻辑 KV 仅 ~1GB，~26× 放大），2GiB CPU tier 缓冲在预填充中冲到 **72.8%**，`allocation_failure_total` 单请求 +69（956→1025）；kvssd 净增 5.33GB（du）。详见 §8.5。
5. **NVRM 驱动 OOM 仍在发生**：即使 0.7、空闲启动，01 与 03 内核日志仍出现 `NVRM: NV_ERR_NO_MEMORY (0x00000051) from _memdescAllocInternal`（03 于 05:13/05:17 warmup 期，01 于 05:13 及 0.8 事故时段 03:18–04:47）。内存余量极薄。

---

## 2. 空闲态四节点 free 细览

`free -m`（采集于 05:36，无在途请求）：

| 节点 | total | used | free | shared | buff/cache | available | swap 已用/总 |
|---|---|---|---|---|---|---|---|
| node01 | 124546 | 116455 | 1129 | 2245 | 10325 | **8090** | 11811/16383 |
| node01 | 124546 | 115639 | 908 | 2242 | 11591 | **8906** | 14269/16383 |
| node01 | 124546 | 121479 | 1204 | 2319 | 5252 | **3067** | 4415/16383 |
| node01 | 124546 | 121971 | 1076 | 2320 | 5345 | **2574** | 4426/16383 |

连续 3 次采样确认可用内存稳定（01: 8.2G/8.2G/8.2G；03: 3.1G/3.1G/3.1G），非瞬时抖动。

### /proc/meminfo 关键字段（空闲态，01 示例）

| 字段 | 值 | 字段 | 值 |
|---|---|---|---|
| MemTotal | 127.5GB | MemFree | 1.1GB |
| MemAvailable | **8.3GB** | Buffers | 0.2GB |
| Cached | 9.8GB | AnonPages | 3.5GB |
| Active(anon) | 5.7GB | Inactive(file) | 6.4GB |
| Shmem | 2.3GB | Mapped | 2.9GB |
| SReclaimable | 0.5GB | SUnreclaim | 0.9GB |
| KernelStack | 25MB | PageTables | 60MB |
| VmallocUsed | 905MB | Committed_AS | 34.6GB |
| HugePages_Total | 0 | Hugetlb | 0 |

**页缓存归属要点**：常规可归因项（Anon+Cached+Buffers+SReclaimable+SUnreclaim+KernelStack+PageTables+Shmem+…）合计仅 ~17GB；`free` 的 used≈116GB 与这些项的差额 **~98GB 即 UMA 设备内存（CUDA 分配）**——它在驱动侧占用物理 DRAM、不可回收、也不计入任一进程 RSS，是"看不见的大头"。

---

## 3. 121GB 去向分解（每节点，空闲态）

### 3.1 GPU/UMA 设备内存（nvidia-smi 实测）

| 节点 | 进程 | used_gpu_memory |
|---|---|---|
| 01 | VLLM::Worker_TP0 | **104074 MiB (101.6GB)** |
| 02 | VLLM::Worker_TP1 | **104078 MiB (101.6GB)** |
| 03 | VLLM::Worker_TP3 | 104078 MiB + embed EngineCore **5750 MiB** |
| 04 | VLLM::Worker_TP2 | 104076 MiB + embed EngineCore **5750 MiB** |

> nvidia-smi 总/用/余均为 N/A（UMA 无独立 framebuffer 记账），仅进程级 used_memory 可读。

### 3.2 vLLM 自报分解（gpu_worker.py:857，启动日志）

节点 01：
```
Free memory on device (112.77/121.63 GiB) on startup. Desired GPU memory utilization is (0.7, 85.14 GiB).
Actual usage is 40.55 GiB for weight, 2.03 GiB for peak activation, 0.13 GiB for non-torch memory,
and 0.9 GiB for CUDAGraph memory. Current kv cache memory in use is 42.43 GiB.
```
节点 03（同机有 embed）：weight 40.55 / activation 2.03 / non-torch -0.51 / CUDAGraph 1.68 / KV cache 43.07 GiB。

| 去向 | 每节点（01 实测） | 说明 |
|---|---|---|
| 模型权重 (fp8, TP4) | **40.55 GiB** | 155.43GiB 模型 / TP4 分片 + 量化转换开销 |
| KV cache 池 | **42.43 GiB** | nvfp4_ds_mla，block_size=256 tokens（DEEPSEEK_SPARSE_SWA）；空闲时 kv_cache_usage_perc≈0（内存**预留**非使用） |
| 激活峰值 | 2.03 GiB | prefill/decode 工作集 |
| CUDA Graph | 0.90 GiB | breakable+f ull+dspark 图（16+12+11 次捕获） |
| non-torch | 0.13 GiB | 驱动/框架杂项 |
| **vLLM 可归因小计** | **≈86.0 GiB** | |
| CUDA 上下文/cuBLAS/FlashInfer/NCCL/spec-draft 等 | **≈15–18 GiB** | 104.1−86.0，**待 Cody 分析** |

### 3.3 其余物理内存（主机侧，01 空闲态实测）

| 去向 | 估算 | 证据 |
|---|---|---|
| 页缓存 (Cached+Buffers) | ~10.3 GiB | free buff/cache 10325MB |
| vLLM 容器 cgroup（host RSS+shm+文件页） | 12.5 GiB current / **76.6 GiB peak** | cgroup memory.current / memory.peak |
| 其他容器（aicad 全家桶，17 个容器） | ~0.9–1.1 GiB | docker stats 汇总（grafana 275M/neo4j 178M/prometheus 139M/minio 98M…） |
| OS/内核 (slab 不可回收/页表/栈/vmalloc) | ~1.5–1.9 GiB | meminfo |
| /dev/shm (tmpfs 61G) | 01 用 11G / 03 用 2.1G | df /dev/shm（含 NCCL psm_* 与 kv offload mmap） |
| MemFree | 1.1 GiB | |
| **可用余量** | **8.1 GiB (01/02) / 2.6–3.1 GiB (03/04)** | |

> 03/04 可用内存比 01/02 低 ~5GB，主因是 **anemll-embed-8022（Qwen3-Embedding-0.6B）同机占用 5750MiB UMA**（EngineCore 进程）。

### 3.4 vLLM 容器 cgroup v2 内存统计（空闲态）

路径：`/sys/fs/cgroup/system.slice/docker-<cid>.scope/`

| 节点 | memory.current | memory.peak | swap.current | 其中 anon | file | shmem |
|---|---|---|---|---|---|---|
| 01 | 12.5 GB | **76.6 GB** | 2.3 GB | 2.7 GB | 9.2 GB | 2.3 GB |
| 02 | 12.9 GB | **77.2 GB** | 3.1 GB | 1.5 GB | 11.0 GB | 2.3 GB |
| 03 | 7.6 GB | **72.7 GB** | 2.2 GB | 2.7 GB | 4.5 GB | 2.4 GB |
| 04 | 7.8 GB | **74.1 GB** | 2.2 GB | 2.7 GB | 4.7 GB | 2.4 GB |

**关键观察**：
- cgroup `memory.peak` 达 **71–77GB**（发生在模型加载 + cudagraph 捕获期），此后回落。
- cgroup `current`（7.6–12.9GB）远小于 nvidia-smi 的 104GB —— **设备内存不计入容器 cgroup**，由驱动在主机侧占用，这正是 free 里"消失的 ~98GB"。
- 容器内 `shmem` 约 2.3GB 对应 /dev/shm 中的 NCCL PSM / offload staging。

### 3.5 进程级内存（Worker 主进程，host PID）

| 项 | 01 Worker | 03 Worker |
|---|---|---|
| VmRSS | 4.0 GB | 5.5 GB |
| RssAnon | 1.45 GB | 2.62 GB |
| RssFile | 0.43 GB | 0.66 GB |
| **RssShmem** | **2.14 GB**（/dev/shm NCCL 缓冲） | **2.28 GB** |
| VmSwap | 1.67 GB | 0.44 GB |
| VmPin (mlock) | 108 MB | 108 MB |
| VmSize (虚拟) | 276 GB | 276 GB |

> 权重未以 file-backed 页常驻于进程 RSS（smaps 中 RssFile 仅 ~0.4–0.7GB，权重在设备内存），仅少量 mmap 文件页 + 共享内存。

---

## 4. KV 卸载 / 数据交换占用内存（重点）

rank0(01) 的 vLLM metrics 可读，其余节点 8001 拒绝（仅 rank0 暴露 API）。

### 4.1 空闲→轻载对比（conc1 单请求）

| 指标 | 空闲(05:36) | 64K 预填充中 | conc1 完成后 |
|---|---|---|---|
| `kv_cache_usage_perc` | 0.0 | **2.2%** | 0.0 |
| `kv_offload_cpu_cache_usage_perc` | 0.0 | **0.608（60.8%）** | 0.0 |
| `kv_offload_store_bytes_total` | 157.5 GB | 167.6 GB | **186.4 GB（+28.9GB）** |
| `kv_offload_allocation_failure_total` | 876 | 901 | **956（+80）** |
| `num_requests_running` | 0 | 1 | 0 |
| `prompt_tokens_total` | 1.218M | 1.243M | 1.345M |

- **单条 64K 请求（pt=101,468 tokens，42.8s）就卸载 ~29GB KV**，全部走 `GPU_to_CPU`；`CPU_to_GPU`=0（本次无提升/重载发生）。
- 2GiB `cpu_bytes_to_use` CPU tier 在预填充中被压到 **60.8%**，并产生 **+80 次分配失败** —— 说明 **2GiB CPU 缓冲对 64K 长上下文不够用**，conc3×4 事故时会饱和并叠加 GPU KV 池增长 → 内存峰值 → OOM。
- 卸载直方图：1376 次 store，单次 100–200MB 为主，累计 157GB→186GB，均摊 ~114MB/次。

### 4.2 配置（启动命令 kv-transfer-config）

```
OffloadingConnector / TieringOffloadingSpec
  kv_buffer_size        = 1,000,000,000 B (≈1GB, cuda 中转缓冲)
  cpu_bytes_to_use      = 2,147,483,648 B (2GiB, CPU 主层)
  eviction_policy       = lru
  secondary_tiers       = [fs: /opt/aicad-kvssd, n_read_threads=4, n_write_threads=4]
```
启动日志：`Created TieringOffloadingManager with primary tier (lru, 503 blocks)` —— CPU 主层仅 **503 blocks × 256 tokens/block**。

### 4.3 kvssd 落盘（SSD 层，与内存间接相关）

| 节点 | 设备 | 容量 | 已用 |
|---|---|---|---|
| 01 | /dev/loop19 | 196G | **40G**（conc1 后较 34G 净增 ~6G） |
| 02 | /dev/loop19 | 196G | ~0（28K） |
| 03/04 | /dev/loop0 | 196G | ~0 |

---

## 5. NCCL / 通信缓冲

- 4×RoCE IB HCA：`rocep1s0f0/1, roceP2p1s0f0/1` 全部 LinkUp。
- NCCL 环境：`NCCL_BUFFSIZE=8388608 (8MB)`、`NCCL_MAX/MIN_NCHANNELS=4`、`NCCL_ALGO=RING`、`NCCL_IB_TOS=46`、`NCCL_IB_SUBNET_AWARE_ROUTING=1`、`NCCL_IB_GID_INDEX=3`、`NCCL_SOCKET_IFNAME=enP7s7`。
- `/dev/shm`：容器 `--shm-size=64gb` → tmpfs 61G；**01 已用 11G**（NCCL PSM 共享内存文件 psm_* 为主），03 用 2.1G（另有 `vllm_offload_*.mmap`、`sem.mp-*` 多进程信号量）。
- 通信缓冲内存直接计入 `/dev/shm`（容器 shmem ~2.3GB 常驻 + 01 上 ~11GB 的 psm 残留）。

---

## 6. Swap / 页缓存 / OOM 证据

### 6.1 Swap
- 每节点 16GB `swap.img`（文件型，priority -2）。空闲已用 4.3–14.3GB。
- SwapCached：01=85MB / 02=67MB / 03=361MB / 04=350MB。

### 6.2 NVRM 驱动 OOM（`NV_ERR_NO_MEMORY`）——直接证据
- **01**（journalctl -k 持久）：`NVRM: ... Out of memory [NV_ERR_NO_MEMORY] (0x00000051) from _memdescAllocInternal @ mem_desc.c:1359`
  - 0.8 事故段：03:18 / 03:59 / 04:03 / 04:04 / 04:44 / 04:47（6 次）
  - **0.7 空闲重启段：05:13:32（1 次）**
- **03**（新内核 dmesg）：boot 后 1078s≈05:13:33、1325s≈05:17:40、1340s≈05:17:55（3 次，均在 0.7 warmup/加载期）
- 未在当前 boot 的 kernel ring 中检出 `oom-killer`（历史 oom-kill 由主理人/前次事故记录提供）；当前可用证据以 **NVRM NV_ERR_NO_MEMORY** 为准。

> 含义：即使 0.7 且空闲，驱动内存描述符分配仍会失败——内存预算已无安全边际；一旦 KV/激活/卸载缓冲叠加即触发驱动层 OOM（表现即 avial=0 + NV_ERR_NO_MEMORY + 进程被杀）。

---

## 7. 模型与加载

- 模型：`<INSTALL_DIR>/models/deepseek-v4-flash-0731`（symlink→01/02 为 /home/<USER>/models/...，03/04 为 <MODELS_DIR>/...），`du -shL`=**156G（155.43 GiB）**，48 个 safetensors 分片。
- vLLM：`quantization=deepseek_v4_fp8`，`dtype=bf16`，TP4 / nnodes=4，`distributed_executor_backend=mp`。
- 加载日志：checkpoint 155.43GiB，Available RAM 73.4GiB(01)/67.1GiB(03)；**auto-prefetch 因 checkpoint>90% RAM 被禁用**（01 EXT4 / 03 NFS4）。
- 权重加载耗时：01 首载 82.07s（含从磁盘读）+二次 16.52s；03 111.74s+20.09s。
- KV cache：`kv_cache_size_tokens=4,784,664`（log）／`2,566,890`（cache_config_info），`num_gpu_blocks=42743`，block_size=256（SWA）；**600K 上下文时最大并发 ≈7.97×**。

---

## 8. 可优化参数线索（实证驱动，具体调参待 Cody/主理人决策）

| 参数 | 现状 | 实证问题 | 方向 |
|---|---|---|---|
| `gpu-memory-utilization` | 0.70 | 空闲 avail 仅 2.6–8.9GB；0.7 仍触发 NVRM OOM | 下调 KV 池预留，给主机留余量；或改用 `--kv-cache-memory`（日志建议 `41.38GiB` 可贴合当前） |
| `cpu_bytes_to_use` | 2 GiB | 单条 64K 即冲到 60.8% + 分配失败 +80 | 增大 CPU tier 或限制并发；需评估主机 8GB 余量是否足够支撑更大的 CPU 缓冲 |
| `max_model_len` / 并发 | 600000 / 并发 7.97× | 长上下文放大 KV 池与卸载压力 | 视业务收紧或维持（影响功能） |
| `max_num_seqs` | 12 | 与 600K 长序列组合放大激活/卸载 | 低负载下可适当下调 |
| `kv_buffer_size` | 1GB | 中转缓冲占用设备内存 | 确认必要性，可减小 |
| 同机 embed（03/04） | 5750MiB/节点 | 直接吃掉 03/04 ~5GB 可用内存 | 评估迁移到专用节点或降低 embed KV cache（--kv-cache-memory=4GB） |
| swap 兜底 | 16G/节点已用 4–14G | 空闲即换页，延迟风险 | 长期应消除对 swap 的依赖 |

**待分析（Cody 接手）**：104GB 设备内存中 vLLM 自报项（86GB）之外的 ~15–18GB CUDA/driver/框架开销精确构成；KV 池预留 vs 物理常驻的量化；卸载失败后的重计算/提升路径对延迟的影响。

---

## 8.5 精确三方对比实测（追加：单条 64K conc1）

> 补充实测（08-19 05:51–05:55）：01 节点单条 64K 请求（随机 uuid 前缀，不命中 prefix cache），pt=**101,618**、ct=8、43.1s、status=200。
> **口径修正**：此前报告 `store +28.9GB` 是 conc1 完整脚本（16K+64K 两条顺序请求，pt=126,842）合计；**单条 64K 为 +25.47GB**，两者按 token 归一一致（~245KB/token）。

### 测量值（基线 → 事后 → 落定）

| 指标 | 基线 | 事后 | 增量 |
|---|---|---|---|
| `kv_offload_store_bytes_total` | 186.39 GB | 211.86 GB | **+25.47 GB（23.72 GiB）** |
| `kv_offload_store_size_count` | 1560 | 1684 | **+124 次**（均值 ~205.4 MB/次） |
| `kv_offload_allocation_failure_total` | 956 | 1025 | **+69** |
| `kv_offload_cpu_cache_usage_perc` | 0.0 | 0.728（落定 0.0） | 峰值 **72.8%** |
| `kvssd du -sb` | 41,765,503,171 B | 落定 47,095,877,827 B | **+5.33 GB（4.96 GiB）** |
| `kv_cache_usage_perc` | 0.0 | 0.0 | — |
| `CPU_to_GPU` | 0.0 | 0.0 | —（本次无提升/重载） |

### 三比值与理论对比

| 口径 | 计算 | 结果 |
|---|---|---|
| 引擎 store_bytes/pt | 25.47GB / 101,618 | **244.8 KB/token** |
| 落盘 du/pt | 5.33GB / 101,618 | **51.2 KB/token** |
| store / du（引擎 vs 落盘系数） | 25.47 / 5.33 | **4.78×** |
| 理论真实 KV（9.6KB/token） | 101,618 × 9.6KB | **0.976 GB** |
| store_bytes / 理论 | 25.47 / 0.976 | **26.1×** |
| du / 理论 | 5.33 / 0.976 | **5.5×** |

> 9.6KB/token 与引擎自证一致：GPU KV 池 42.43GiB / 4,784,664 tokens ≈ 9.5KB/token。

### 结论（供放大分析）
1. **`store_bytes` 是累计写流量，不是常驻 KV**：单条 64K 的**逻辑 KV 仅 ~1GB**（GPU 池内），但卸载路径产生 **25.5GB 累计写入**（~26×），即同一逻辑 KV 被反复写/多层写（GPU→CPU tier→FS tier 的 tiering 设计 + 预填充 chunk 循环 + LRU 逐出重写）。
2. **落盘净增 5.33GB（du）≈ 5.5× 理论**：FS tier 每 205MB 块粒度写入 + 块填充 + 两层 tier 累计，净保留仍明显高于逻辑 KV；精确拆分待 Cody。
3. **CPU tier 2GiB 是瓶颈**：单条请求即冲到 72.8%、+69 次分配失败；conc3 事故时必然饱和并叠加 GPU KV 增长 → 内存峰值 → OOM。

---

## 9. 采集脚本留存

- `collect_mem.sh` / `collect_mem2.sh` / `collect_mem3.sh` / `collect_mem4.sh`（本目录，全部只读）
- 复测命令：`ssh <USER>@node01 'bash -s' < collect_mem.sh`
