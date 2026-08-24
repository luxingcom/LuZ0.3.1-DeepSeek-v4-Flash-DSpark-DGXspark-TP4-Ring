# Code Review: KV 卸载 + io 补丁在内存压力下的风险评审（内存足迹专项）

- **审查人**：Cody（代码审查师，工程保障团队）
- **日期**：2026-08-19
- **审查对象**：
  1. `delivery/kvssd-offload-2026-08-18/kvpatch/io.py`（已上线的 KVZSTD01 格式补丁：trim + zstd-3 + per-path 锁/O_EXCL 去重，4 写 + 4 读线程内压缩/解压）
  2. vLLM 0.26 `OffloadingConnector` + `TieringOffloadingSpec`（`cpu_bytes_to_use=2147483648`=2GiB、`kv_buffer_size` 引擎默认 1e9、fs 层 root_dir=/opt/aicad-kvssd、读写线程各 4、`kv_load_failure_policy=fail`）
- **关联事故**：benchmark 31/54（65536/coding/conc3）NCCL ALLREDUCE 300s 超时 → worker 死 → 集群宕；conc3×65536 可稳定复现；0.8 util 下实测 used 110~115GB（理论 0.8×121=96.8GB，超 ~15GB）；03/04 avail 仅 4-6G，03 曾内存耗尽系统卡死
- **审查边界**：只做静态分析与代码审查，不 SSH 节点（现场由 SRE Rex 执行）

---

## 概要

**io 补丁的内存足迹是 MB 级（峰值 ~50-70MB、稳态 ~25-35MB），不是 110-115GB 与 96.8GB 之间 ~15GB 差异的来源**；vLLM 卸载框架的确定性增量 ~3-3.5GB（CPU 主层 2GiB 为主体），占 03/04 可用余量（4-6GB）的 50-75%，是**风险放大器**而非根因。~15GB 差异主要是「宿主机其它进程 + vLLM 运行时超额（cudagraph/NCCL/激活/非 expandable 分配器碎片）+ 卸载增量」的结构性构成，**不是 KV 卸载的内存溢出**。0.70 util 重启后预计释放 ~10-12GB 池容量，是把 03/04 拉回安全水位的正确动作；事故因果链为「0.8 util 余量过薄 + conc3×65536 突发（激活/KV/卸载 3GB 叠加）→ 内存压力触顶 → swap thrash/系统 stall → NCCL 进度停滞 300s → worker 死 → 集群宕」。补丁**非直接根因**（conc1 正常），但建议做 4 项小改进消除峰值分配与 5× 重复压缩 CPU。

---

## 一、io.py 补丁内存足迹分析

### 1.1 关键尺寸基线（实测定档）

| 量 | 值 | 依据 |
|---|---|---|
| block_size（单块槽位） | 4,263,936 B（4.26MB） | 实机取证 |
| 有效前缀（trim 后） | ~1.03MB | 实机取证（行首~最大非零末端） |
| zstd-3 payload | 614~807KB | 容器内实测（NVFP4 高熵，压缩率 ~81%） |
| 对齐落盘尺寸 | align_up(16+payload, 4096) ≈ 0.6~0.8MB | 补丁 `_align_up` |

### 1.2 写路径（`store_block`，运行于 4 个写线程）

| # | 步骤 | 分配 | 峰值/稳态 |
|---|---|---|---|
| W1 | `view = buffer.cast("B")[offset:offset+block_size]` | memoryview 切片，**零拷贝** | 0 |
| W2 | `_effective_len(view)`：`view.tobytes()` → **4.26MB 全量 bytes 拷贝**；`.rstrip(b"\x00")` → ~1.03MB 新 bytes（期间两对象并存） | 4.26 + 1.03 MB | **瞬态 ~5.3MB**（tobytes 释放后仅剩 ~1.03MB） |
| W3 | `compressor.compress(view[:valid_len])`（memoryview 切片，零拷贝） | zstd 输出 ~0.8MB | 稳态 ~0.8MB |
| W4 | `content = _MAGIC + _HEADER.pack + payload` | 新 bytes ~0.8MB | 稳态 ~0.8MB |
| W5 | `_write_aligned`：`mmap.mmap(-1, aligned_len)` 匿名页对齐缓冲（≤4KB+payload） | ~0.8MB | 稳态 ~0.8MB（写完即 close） |
| W6 | zstd 线程局部 `ZstdCompressor(level=3)` | 上下文 ~0.5MB | 常驻（8 线程共享计） |

**单写线程**：稳态 ~2.4MB；`_effective_len` 期间瞬态峰值 +4.26MB。
**4 写线程并发**：稳态 ~10MB；若 4 线程同时到达 trim 段，瞬态峰值 ~27MB。

### 1.3 读路径（`load_block`，运行于 4 个读线程）

| # | 步骤 | 分配 | 峰值/稳态 |
|---|---|---|---|
| R1 | `mmap.mmap(-1, aligned_size)` 读整文件 | ~0.8MB | 稳态 ~0.8MB |
| R2 | `os.readv` 读入 mmap（O_DIRECT 满足对齐） | 0（mmap 内） | 0 |
| R3 | `payload = bytes(buf[16:16+payload_len])` | ~0.8MB 拷贝 | 稳态 ~0.8MB |
| R4 | `decompressor.decompress(payload, max_output_size=block_size)` | 输出缓冲 ≤ **4.26MB**（实际 ~1.03MB） | 瞬态 ~4.26MB（有界，防炸弹） |
| R5 | `view_slice[len(data):] = b"\x00" * (block_size - len(data))` | **3.2MB 补零 bytes 一次性分配** | **瞬态 ~3.2MB** |
| R6 | 解压结果拷贝进主层 view + 补零 | 已计入 CPU 主层 | 0（view 属 2GiB 层） |
| R7 | zstd 线程局部 `ZstdDecompressor` | ~0.5MB | 常驻 |

**单读线程**：稳态 ~2.6MB；瞬态峰值 ~9MB（mmap + payload + 解压缓冲 + 补零并存）。
**4 读线程并发**：稳态 ~10MB；瞬态峰值 ~36MB。

### 1.4 峰值增量估算结论

| 项 | 稳态 | 峰值瞬态 |
|---|---|---|
| 4 写线程 | ~10MB | ~27MB（全到 trim 段） |
| 4 读线程 | ~10MB | ~36MB（全到解压+补零） |
| zstd 线程局部（8 线程） | ~4-8MB | 同左 |
| **合计** | **~25-35MB** | **~50-70MB** |

**结论：补丁内存足迹 = MB 级，与 ~15GB 差异相差两个数量级，可忽略。** 但 CPU 足迹（见 §三.3）在 conc3 突发时值得关注——zstd-3 压缩/解压 + 最多 5× 重复压缩占核，与 NCCL 进度线程争 CPU，是 NCCL 300s 超时的**次级嫌疑**。

---

## 二、vLLM KV 卸载框架确定性开销清单

> 依据：vLLM 源码（`kv_offload/tiering/fs/{manager,io,thread_pool}.py`、`config/kv_transfer.py`）+ 灰度注入实测报告 + G-3 门禁实测。fs 二级层**零拷贝**直接读写主层 memoryview，本身不额外分配大 staging buffer。

| # | 项 | 大小 | 说明/依据 |
|---|---|---|---|
| 1 | **CPU 主层（pinned LRU）** | **≤ 2GiB**（`cpu_bytes_to_use=2147483648`） | 灰度实测 usage 0.73 ≈ 1.5GiB（3×9K 请求）；LRU 驱逐上限即预算，**峰值可占满 2GiB** |
| 2 | **kv_buffer_size** | ≤ 1GB（默认 1e9） | vLLM `KVTransferConfig.kv_buffer_size=1e9`（引擎默认）；官方注释指向 TorchDistributedConnector 传输缓冲，**OffloadingConnector 路径是否实际占用需日志/内存归属确认**——保守计入上限 |
| 3 | fs 线程池（4 读 + 4 写） | ~5MB | Python 线程 + 栈（虚拟 8MB、RSS 小） |
| 4 | LRU/lookup 元数据（tracker） | ~1-5MB | `max_tracker_size` 64000 上限 × 数百 B |
| 5 | `_path_locks` 字典 | ~1-10MB | 每 dest 一条锁（数万条 × ~300B） |
| 6 | io 补丁工作缓冲（4+4） | ~25-70MB | §一 估算 |
| **合计确定性增量** | | **~3.0-3.5GB** | **与 G-3 门禁「内存增量 ≤3GB」实测吻合** |

**要点**：
- 这 ~3-3.5GB 是**常驻/可占满**的（CPU 主层会随卸载压力填到 2GiB），在 03/04 仅 4-6GB 余量的背景下吃掉 50-75% 余量——**这是卸载对本次事故的直接贡献上限**。
- 若需再压：`cpu_bytes_to_use` 2GiB→1GiB（灰度报告已建议），省 ~1GB。

---

## 三、关键问题回答

### 3.1 0.8 util 下 used 110-115GB（超 96.8GB 理论 ~15GB）的可能来源排序

**先纠正一个口径**：`0.8 × 121 = 96.8GB` 是 vLLM 为「权重 + KV 池」预留的 **GPU 预算**，**不是宿主 RSS 的硬上限**。宿主 `free -h` 的 used 天然包含 vLLM 预算 + vLLM 超额 + 所有其它进程。110-115GB vs 96.8GB 的 ~15GB 是**结构性构成**，不是异常溢出。按贡献排序：

| 排序 | 来源 | 估计 | 依据/机理 |
|---|---|---|---|
| ① | **宿主机其它进程**（AICAD 全套：redis/pg/neo4j/minio/grafana + litellm-proxy + monitor + 03/04 embed 服务 + OS） | **5-10GB** | 实测可用命令归因：`free -h` 减 vLLM 容器 RSS |
| ② | **vLLM 运行时超额**（cudagraph 池 `--max-cudagraph-capture-size 96`、NCCL 通信 buffer、flashinfer autotune workspace、投机 draft 7 tokens、激活、**非 expandable 分配器碎片**——`expandable_segments` 因连接器不兼容被置空） | **3-8GB** | 默认分配器预分配 + 不回收，易超预算 |
| ③ | **KV 池预分配本身** | 10-17GB（属「容量」非「溢出」） | 0.8 预算 = 96.8 − 权重(~79GB) − 激活预留 |
| ④ | **KV 卸载框架**（§二） | ~3-3.5GB | CPU 主层 2GiB + kv_buffer 1GB + 元数据 |
| ⑤ | **io 补丁** | ~30-70MB | §一，**可忽略** |

> ①+②+④ ≈ 11-21GB，覆盖 ~15GB 差异。**KV 卸载的 ~3GB 只占差异的 ~20%，补丁可忽略。**「UMA 语义下 util 不是硬上限」成立：util 只决定预算，进程 RSS 由分配器 + 碎片 + 其它进程共同决定。

### 3.2 0.7 util 应释放多少

- `0.7 × 121 = 84.7GB` 预算，KV 池容量缩小 ~**12.1GB**（0.1×121，权重 ~79GB 不变）。
- KV 池是启动时按预算**预分配的张量**（`set_util07.sh` 已做容器重启，util 在启动时生效），因此 **RSS 释放 ≈ 10-12GB**（接近池缩小量）。
- 预期：used 110-115GB → **~98-103GB**；available 4-6GB（03/04）→ **~16-22GB**，退出危险水位。
- **副作用（必须盯）**：KV 池缩小 → 更多块被 LRU 驱逐到 SSD → 卸载流量、zstd CPU、SSD 写放大上升；CPU 主层仍可填满 2GiB。**建议 0.7 后首个窗口复测 conc3×65536**（SRE 任务 #5），同时盯 `kv_offload_*`、CPU 占用、SSD 水位（200G 配额曾 186G/200G）。

### 3.3 是否存在内存溢出（03 卡死 + NCCL 超时因果）

**结论：不是「KV 卸载内存溢出」，是「余量过薄 + 突发超卖」型事故。**

因果链（与 03 卡死、NCCL 300s 超时一致）：
1. 0.8 util 下 03/04 稳态 available 仅 4-6GB（结构性：其它进程 + vLLM 超额 + 卸载 ~3GB）；
2. conc3×65536 突发：3 个 65K 长 prefill 并发 → **激活瞬时尖峰** + KV 池填充 + CPU 主层（2GiB）填满开始 LRU 驱逐 → 4 写线程 zstd 压缩 + 读回解压；
3. 内存压力触顶 → 内核回收/swap（15GiB swap）thrash → **系统级 stall（03 卡死）**；
4. NCCL ALLREDUCE 进度依赖 CPU/网络推进，被 stall + swap thrash 拖死 → **300s watchdog 超时 → worker 死 → 集群宕**。

证据自洽：conc1×65536 正常（46.2s，无并发突发、无卸载风暴）→ **补丁非直接根因**；conc3 复现 → **并发 + 内存压力为触发**；03 曾内存耗尽卡死 → **系统级内存压力**；卸载框架 ~3GB 占余量 50-75% → **放大器**。

**缓解**：0.7 util（已批准）为主；叠加建议：03/04 关 embed 服务腾余量、`cpu_bytes_to_use` 2→1GiB、`kv_buffer_size` 若确认占用可显式收紧、NCCL timeout 与 watchdog 复核（SRE 任务 #4 覆盖）。

---

## 四、补丁改进建议（按收益排序）

| # | 位置 | 问题 | 建议 | 类别 |
|---|---|---|---|---|
| 1 | `_effective_len`（L148-157） | `view.tobytes()` 全量 4.26MB 拷贝 + rstrip 再拷贝，峰值 ~5.3MB/线程（4 线程 ~27MB） | **分块反向扫描**：从末端按 4096 分块，逐块 `bytes(view[i-4096:i]).rstrip(b"\x00")`；命中含数据块即返回。峰值分配降为 4KB，典型（尾部 3.2MB 零）仅 ~800 次 4KB 扫描 | 内存 |
| 2 | 读路径补零（L405） | `b"\x00" * (block_size - len(data))` 一次性分配 3.2MB/线程 | 复用线程局部 4KB 零缓冲循环赋值（`for off in range(len(data), block_size, 4096): view_slice[off:off+n] = _zeros[:n]`），消除 3.2MB 瞬态分配 | 内存 |
| 3 | zstd level（L92） | `_COMPRESS_LEVEL=3` 硬编码；conc3 突发时 4 写线程 zstd + 5× 重复压缩与 NCCL 争 CPU | 暴露 `VLLM_KVSSD_ZSTD_LEVEL`（默认 3），突发可降 1 减 CPU；并评估在锁外压缩前加一次 `exists` 复查减少重复压缩（现有 L327 快速路径已覆盖「已写完」，未覆盖「同时到达」的 4 份浪费） | 性能/CPU |
| 4 | `_write_aligned`（L200） | 每写一次 mmap/munmap 系统调用 | 线程局部复用对齐缓冲（4 写线程 × ~0.8MB，内存无碍，省 syscall） | 性能 |
| 5 | `_path_locks`（L219-236） | 无界字典（前审已记 Low） | 可选分片锁表（N=256 取模）封顶内存 | 内存 |

> 格式头 `KVZSTD01` 已上线，**不建议**为存 valid_len 改头（会破坏兼容 + 需清缓存重算）；解压 `max_output_size=block_size` 已是有界防炸弹，保持。

---

## 五、做得好的地方

- **O_DIRECT 对齐完全隔离在 mmap staging**，KV 槽位本身零拷贝（W1/W3），设计干净。
- **解压有界**（`max_output_size=block_size`），补丁没有可被放大成 GB 级的内存路径。
- **fs 二级层零拷贝契约**（直接读写主层 memoryview）使 vLLM 框架侧增量被锁定在 ~3GB 内，与 G-3 门禁一致。
- 0.7 util 变更方向正确（`set_util07.sh`/`fix_check_util.sh` 同步改脚本与校验，运维闭环完整）。

---

## 六、结论

**Request Changes（低危，不阻断 0.7 复测）**

- **内存**：补丁 MB 级（~50-70MB 峰值），非 ~15GB 差异来源；卸载框架 ~3-3.5GB 确定性存在，占 03/04 余量 50-75%，是事故放大器非根因。
- **0.7 util**：重启后释放 ~10-12GB 池容量，应把 03/04 拉回 available ≥16GB；复测 conc3×65536 是放行判据。
- **事故定性**：余量过薄 + conc3 突发超卖 → 内存压力触顶 → swap/系统 stall → NCCL 300s 超时；补丁非直接根因（conc1 正常）。
- **建议动作**：① 0.7 复测后按 §四 打 4 项小改进（消除峰值分配 + zstd level 可调）；② `cpu_bytes_to_use` 2→1GiB 观察；③ 03/04 关 embed 腾余量；④ NCCL timeout 复核（SRE）。
