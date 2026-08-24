# KV 卸载 29× 流量放大根因代码级深挖（store_bytes / block→token / 5-group / 根治方案）

- **作者**：Cody（代码审查师，工程保障团队）
- **日期**：2026-08-19
- **触发**：用户质疑 64K 请求（101,559 prompt tokens）触发 `kv_offload_store_bytes` 28.9GB，而真实 KV 成本仅 ~1GB（101559×9.6KB=975MB）——**~29× 放大不合理**
- **方法**：本地资料（执行计划/灰度/200G 执行报告/前审）+ vLLM 开源 main 分支 `vllm/v1/kv_offload/` 源码级分析（FileMapper/CPUOffloadingSpec/OffloadingConfig/connector/fs tier）；SSH 与 SRE 数据未就绪，需 01 节点核对的项标 **「待实测」**
- **审查边界**：静态分析，不改生产

---

## 0. TL;DR（回答用户质疑）

**28.9GB 不是"数据量"，是"搬运计数"。** `kv_offload_store_bytes` 计的是 **GPU→CPU 主层的原始槽位传输字节**（未压缩、未裁剪、含 TP4 复制、5 个 KV group 各计一次），不是磁盘落盘字节、更不是有效 KV 数据。29× 放大由三个结构性因子叠加：

| 因子 | 倍数 | 来源 |
|---|---|---|
| 5 个 KV group 各 store 一份 | **5×** | MLA 的 latent/indexer/attn 等 5 组投影，内容不同（sha256 交集=0），不可合并 |
| TP4 world_size 复制 | **4×** | `kv_bytes_per_block = worker_kv_bytes_per_block × world_size`，CPU 主层为 4 个 worker 各留一份槽位 |
| 槽位对齐/dtype/块未满 | **~1.5-2×** | `round_up(..., 4096)` + 每块承载 token 数与槽位容量不匹配（57 token/4.26MB） |
| **合计** | **~30-40×** | 补丁前 fs 落盘 382KB/token = 9.6KB×40 ✓ 完美吻合 |

- **store_bytes（GPU→CPU 段）**：64K = 28.9GB/101559 = **284.6KB/token**（≈9.6×30）
- **fs 落盘（补丁后，du 口径）**：64K 预计 ~7.2GB = **70.7KB/token**（=9.6×7.4）
- **两者差 ~5×**：io 薄壳补丁只优化 fs 段（trim+zstd+去重），**GPU→CPU 主层段原样传输**——所以 store_bytes 指标即使补丁后也保持高值，**指标本身不能反映磁盘效率改善**。

**结论**：29× 是"5 group × TP4 复制 × 槽位冗余"的**架构性常数**，不是异常；真实放大已从 40×（补丁前）降到 7.4×（落盘口径）；要继续降需源码级（§4）或参数级（§5）。

---

## 1. `kv_offload_store_bytes` 计数口径（源码级）

### 1.1 指标定义

`vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`：

```
vllm:kv_offload_store_bytes  (counter)  "Total bytes stored from GPU to offload storage."
vllm:kv_offload_load_bytes   (counter)  "Total bytes loaded from offload storage to GPU."
```

### 1.2 计数链（谁在加、加多少）

```
worker 传输完成 → TransferResult.transfer_size (= num_bytes)
  → OffloadingWorkerMetadata.transfer_stats.store.record(num_bytes, time)
  → scheduler 聚合 → OffloadingConnectorStats.increase_counter(STORE_BYTES, bytes)
  → Prometheus vllm:kv_offload_store_bytes
```

`num_bytes` 在 `SingleDirectionOffloadingHandler.transfer_async()`（`vllm/v1/kv_offload/cpu/offload.py`）计算：

```python
num_transfer_bytes = 0
for g_idx, (group_size, block_idx) in enumerate(zip(group_sizes, block_indices)):
    ...
    num_bytes += group_size * data_ref.page_size_bytes   # 直接布局
    # canonical 布局：num_active_blocks * plan.total_bytes（writer 轮转过滤）
```

**精确语义**：
1. **按"完整槽位字节 × 提交块数"计**：`group_size`（该 group 提交的块数）× `page_size_bytes`（每块页大小）× ref 数。**与块内实际有效 token 数无关**，未满块也按整块计。
2. **按 KV group 独立累计**：5 个 group 各提交一批 keys，各自计入 `num_transfer_bytes` → 5×。
3. **含 TP 复制**：`page_size_bytes` 来自 `SharedOffloadRegion` 的槽位布局 = `worker_kv_bytes_per_block × world_size`（见 §2），TP4 → 4×。
4. **不含 trim/压缩/去重**：trim+zstd+去重发生在 fs 层 `io.py`（`batch_store_block` → KVZSTD01 格式），发生在 GPU→CPU 段**之后**，不影响 store_bytes 计数。

### 1.3 与 du 落盘的 5× 差异（用户问的"之前 3×9K 后 15.07G vs du 2.97G ≈ 5×"）

| 口径 | 3×9K (41,977 tokens) | 每 token | 位置 | 是否含补丁优化 |
|---|---|---|---|---|
| `kv_offload_store_bytes` | 15.07GB | 359KB | GPU→CPU 主层段 | **否**（补丁前原样传输） |
| fs 落盘（du，补丁后） | 2.97GB | 70.7KB | CPU 主层→SSD | 是（trim+zstd+去重） |
| **比例** | **5.07×** | — | — | 补丁只作用于 fs 段 |

→ **5× 差异是"GPU→CPU 段 vs fs 段"两个不同位置的口径差，不是计数错误**。这也解释为何补丁上线后 `store_bytes` 指标仍显示高值（灰度 6.08GB = GPU→CPU 段）——**要看磁盘效率必须用 du/`kv_offload_store_bytes` 之外的落盘指标，不能只看 store_bytes**。

---

## 2. block→token 映射：为何 4,263,936B 槽位只承载 ~57 token

### 2.1 槽位大小计算链（源码）

`build_offloading_config()`（`kv_connector/v1/offloading/config.py`）+ `CPUOffloadingSpec.__init__`（`kv_offload/cpu/spec.py`）：

```python
# config.py
worker_kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks
#   total_gpu_kv_bytes = packed ? kv_cache_tensors[0].size : sum(t.size for t in tensors)

# cpu/spec.py
num_copies = world_size                      # TP4 → 4（非 replicated）
kv_bytes_per_block = worker_kv_bytes_per_block * num_copies     # ← TP4 复制
kv_bytes_per_chunk  = kv_bytes_per_block * blocks_per_chunk      # 默认 blocks_per_chunk=1
aligned = round_up(kv_bytes_per_chunk, BLOCK_SIZE_ALIGNMENT)     # 4096 对齐
self.kv_bytes_per_chunk = aligned
self.num_blocks = cpu_bytes_to_use // aligned
```

- `worker_kv_bytes_per_block` = GPU KV 缓存总字节 ÷ 块数（单 worker 每块真实字节）。
- `kv_bytes_per_block` = 上述 × `world_size`（4）→ **每块在 CPU 主层占 4 份槽位**（TP4 各 rank 一份）。
- `BLOCK_SIZE_ALIGNMENT = mmap.PAGESIZE = 4096`，`round_up` 后槽位含 padding。
- `tokens_per_block = group.kv_cache_spec.block_size`（vLLM 默认 block_size，本项目 MLA 实测反推 ≈57 tokens/块，**待实测** `--block-size` 值）；`tokens_per_hash` 单组 = block_size×DCP。

### 2.2 8× 冗余的精确来源（64K 请求反推）

| 因子 | 值 | 说明 |
|---|---|---|
| 真实 GPU KV | 9.6KB/token | nvfp4 MLA 压缩后，实测口径 |
| × TP4 world_size 复制 | **4×** | `kv_bytes_per_block = worker_kv_bytes_per_block × world_size` |
| × 槽位对齐/dtype/块未满 | **~2×** | 4096 对齐 + `kv_bytes_per_chunk`（可能 blocks_per_chunk>1）+ 未满块按整块计 |
| = 每 token 槽位字节 | ~74.8KB/token | 4,263,936B ÷ 57 token |
| **总冗余** | **~7.8×** | 74.8KB / 9.6KB |

**"57 token/4.26MB"的直接含义**：一个卸载块（offload block）在 CPU 主层占用 `4,263,936B`（= worker 每块字节 ×4 + 对齐），但块内实际承载 ~57 个 token 的有效 KV（57×9.6KB=547KB）——**8× 中 4× 来自 TP 复制（架构必须，每个 rank 都要可读自己的槽位），其余来自对齐/块粒度不匹配**。

> ⚠️ 注意：**store_bytes 计数不因块未满而打折**——它按整块页大小计。所以"块内 token 越少、每 token 摊的 store_bytes 越高"：短请求（3×9K，359KB/token）比长请求（64K，284KB/token）放大更狠，正因短请求块更空。

---

## 3. 5 group 提交：调用链与合并可行性

### 3.1 调用链（源码确认）

- KV cache 分多组：`kv_cache_config.kv_cache_groups`，每组含 `tokens_per_block` + `layer_names`（MLA 的 latent/indexer/attn 等不同投影层）。
- 每个 group 生成独立 `OffloadKey = block_hash + group_idx(4B)`（`get_offload_group_idx` 取尾 4B）。
- `FileMapper.get_file_name()`：`<base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin` —— **group_idx 进文件名**，5 组 → 5 个文件。
- `prepare_store()` 对 `keys_to_store` 逐 key 处理，`CPUOffloadingManager` 的 block pool 按 key 分配槽位 → **5 组各占一个 CPU 槽位、各 store 一份**。
- fs tier `submit_store` → `batch_store_block(paths, view, offsets, block_size)` 按 path 写入。

### 3.2 能否文件层合并共享？

**实测结论（200G 执行报告发现 #2）**：g0~g4 内容 sha256 **交集=0**（150×150 抽样），同 group 内 500 文件全唯一 → **5 组是不同的数据（MLA 不同投影的 KV 视图），字节级不可去重**。所以：

- **不能按内容合并**（不是重复，是不同投影）。
- **可以按"行"合并**（同一 token 区间 5 组视图打包成一个文件、按偏移读）——**收益是 inode/目录 fan-out 减少（文件数 /5），不是字节减少**；字节量不变（内容不同）。io.py 补丁目前每 group 一个 KVZSTD01 文件，未来可扩展为"一行一文件、段内 5 偏移"格式，但落盘字节不变。
- **真正省字节的方向**：不是合并 5 组，而是 **消除 TP4 复制**（replicated_layout/canonical_layout，单份 MLA latent）与 **提高块内 token 密度**（§4）。

### 3.3 一个可查的杠杆：`replicated_layout`

`build_offloading_config()` 中 `replicated_layout` 判定：纯 MLA + 单 group + `worker_kv_bytes_per_block == page_size_bytes × len(layer_names)` + TP-only + mp executor + nnodes_within_dp=1 → **为 True 时 `num_copies=1`，TP4 复制从 4× 降到 1×**。本项目是否命中该分支**待实测**（`docker logs rank0` 查 `replicated_layout`/`canonical_layout` 日志）。若未命中，排查原因（如分布式 executor 非 mp、cp 维度 >1）可能直接砍掉 4× 里的 3×。

---

## 4. 根治方案评估（源码级，按收益排序）

> 现状：io 薄壳补丁（trim+zstd+去重）已达**不改源码**上限 = 70.7KB/token（=9.6×7.4）。要继续降必须动 vLLM 内部。

| # | 方案 | 目标 | 改动范围 | 预期收益 | 风险 | 交付形态 |
|---|---|---|---|---|---|---|
| **A** | **参数级：`blocks_per_chunk` 调大（如 4-8）+ 增大 `block_size`** | 块粒度不匹配 | `kv_connector_extra_config`，**零代码** | 每块 token 密度↑ → 每 token 摊槽位字节↓；文件数↓（chunk 合并）；查找粒度↑（官方文档明确支持） | 查找粒度变粗、命中率下降；需实测 | 卷挂载/环境变量即可 |
| **B** | **`store_threshold`（复用过滤）** | 只卸载被复用的块 | `kv_connector_extra_config.store_threshold>=2`，**零代码** | 一次性长 prompt 的"写完即弃"块不再卸载 → store 流量大降（长 ctx 尤其明显） | 需要复用的块会被延迟到第 2 次出现才卸载 | 环境变量 |
| **C** | **per-request `max_offload_tokens`** | 选择性卸载 | 请求参数（官方特性），**零代码** | 只缓存前缀/共享段，长 prompt 后段不卸载 | 命中率依赖前缀复用 | 请求侧 |
| **D** | **replicated_layout / canonical_layout 单份 MLA latent** | 消除 TP4 复制 3× | 配置 + 校验（源码已支持，需确认命中），卷挂载补丁级 | store_bytes 4×→1×（最大单项） | 布局校验失败会 fail-closed；需 01 日志确认当前未命中原因 | 配置 + 日志核对 |
| **E** | **fs 层"一行一文件"格式**（5 组段打包） | 文件数 /5 | io.py 补丁扩展（卷挂载，**无镜像重建**） | 文件数降 5×、目录 fan-out 降、inode 省；**字节不变** | 与现有 KVZSTD01 不兼容，需清缓存；读路径段偏移 | 卷挂载补丁 |
| **F** | **CPU 槽位 dtype 紧凑化**（nvfp4 打包直存 CPU 层） | 消除 dtype 冗余 ~2× | vLLM 源码/镜像重建（`SharedOffloadRegion` 布局） | 每 token 槽位字节↓ ~2× | 镜像重建、DMA 对齐、精度；成本最高 | **镜像重建** |
| **G** | **`--block-size` 调大**（如 16→64/128） | 每块 token 更多 | 启动参数，**零代码**（⚠️ 影响全局 KV 池粒度） | 块内 token 密度↑、每 token 摊槽位↓ | 前缀缓存粒度、调度对齐、大块内部碎片 | 启动参数 |

**推荐组合**：**短期（0 代码）**= A(blocks_per_chunk 4-8) + B(store_threshold≥2) + C(max_offload_tokens)；**中期** = D(replicated_layout 核对，若能命中省 3×) + E(io.py 行合并)；**长期（立项）** = F(CPU 层紧凑化)。G 谨慎评估。

---

## 5. 临时缓解（立即可做，纯配置）

| # | 动作 | 机制 | 收益 | 风险 |
|---|---|---|---|---|
| 1 | `store_threshold=2`（kv_connector_extra_config） | 只有复用≥2 次的块才卸载 | 长 prompt 一次性块不写 → store 流量显著降 | 首次复用会 miss（回算） |
| 2 | `max_offload_tokens`（请求级） | 只卸载前 N token | 长上下文后段不再触发卸载 | 前缀命中率决定收益 |
| 3 | `blocks_per_chunk=4-8` | 多块合并一 chunk，查找粒度粗化 | 文件数↓、元数据↓、每 token 摊↓ | 查找粒度变粗 |
| 4 | 提高 KV 池预算（util 0.70→0.75，**待内存余量确认**） | 减少驱逐 → 减少卸载触发 | 卸载流量↓ | 03/04 内存更紧（当前谷底 2.5G） |
| 5 | 限制长上下文并发（max-num-seqs 8 / 长 ctx 限 conc） | 减少同时驻留的长 KV | 卸载压力↓ | 吞吐↓ |
| 6 | **监控口径修正**：`kv_offload_store_bytes` 是 GPU→CPU 段，**评估磁盘效率必须用 du/落盘字节**，不能只看 store_bytes | 避免误判 | 运维可观测性 | — |

---

## 6. 对用户 29× 质疑的直接回答

1. **28.9GB 是什么**：GPU→CPU 主层的**原始槽位传输字节计数**（未压缩、未裁剪、5 group×TP4 复制），不是磁盘写入量，更不是有效 KV 数据。
2. **为什么 29×**：5 group（5×）× TP4 槽位复制（4×）× 槽位/块粒度冗余（~1.5×）≈ 30×，与 28.9GB/975MB=29.6× 吻合。**架构性常数，非 bug**。
3. **真实磁盘成本**：补丁后 ~70.7KB/token（du 口径，≈9.6×7.4），其中 7.4× 残余 = TP4 复制(4×) × 对齐/块粒度(1.8×)；trim+zstd+去重已把"40×"压到"7.4×"。
4. **还能降多少**：零代码最多再降 ~3-5×（store_threshold + blocks_per_chunk + max_offload_tokens）；源码级可再降 ~4-8×（replicated_layout 单份 latent + CPU 层紧凑化）。

---

## 7. 数据缺口（待 SRE / 01 节点核对）

1. `docker logs rank0` 中 `replicated_layout`/`canonical_layout` 是否命中（决定 4× 能否降到 1×）；
2. 实际 `--block-size`（tokens/block）与 `blocks_per_chunk` 生效值；
3. 64K 请求的 fs 落盘 du（验证 70.7KB/token 推算）；
4. CPU 槽位 dtype（nvfp4 打包直存 or FP8/FP16 展开）——决定 2× 因子；
5. `kv_offload_load_bytes` 对应读回量（load 侧同样有放大）。

---

## 8. 结论

- **29× 合理且可解释**：`kv_offload_store_bytes` = GPU→CPU 原始槽位字节（5 group × TP4 复制 × 槽位冗余），非磁盘量。
- **补丁已达薄壳上限**（70.7KB/token）；下一步**零代码优先**：`store_threshold≥2` + `blocks_per_chunk` + `max_offload_tokens`；**源码级**：优先核对 `replicated_layout`（省 3×），其次 fs 行合并/CPU 层紧凑化（立项）。
- **监控修正**：磁盘效率看 du，不看 store_bytes。
