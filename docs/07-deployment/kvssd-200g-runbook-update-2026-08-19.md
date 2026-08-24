# KV 缓存 SSD 卸载存储效率优化 — Runbook 更新素材与变更记录（2026-08-19）

- **版本**：1.0（2026-08-19）
- **维护**：工程保障团队 · 技术文档师 Docu
- **变更编号**：CHG-2026-08-19-001
- **适用**：TP4 生产集群（4×DGX Spark GB10 sm_121，vLLM 0.26 定制镜像，deepseek-v4-flash-0731，600k token 上下文）
- **前置阅读**：`runbook-dspark-vllm-2026-08-06.md`（v1.1）→ `runbook-tp4-append-2026-08-12.md`（§A–D）→ `runbook-tp4-v1.5-2026-08-12.md`（R11）→ `rollback-anchors-2026-08-12.md`
- **⚠️ 合并说明**：目标 Runbook `F:\AICADqwen\runbook-dspark-vllm-2026-08-08.md`（v1.4）当前不可达（F 盘可访问但无此文件，与 8/12 append 记录一致）。本文按团队既有增量章节格式撰写，**待正式 Runbook 恢复后合并第二部分并删除本文件**（处理方式同 `runbook-tp4-append-2026-08-12.md`）。
- **声明**：除「关键事实（采信）」外，本文涉及指标名/命令/路径中标注「⚠️ 待实测」的条目，均以部署后 /metrics 导出与脚本实测为准，**未验证结论不收录为定论**。

---

# 第一部分：变更记录（Change Record）

## 1. 变更元信息

| 项 | 值 |
|---|---|
| 变更编号 | **CHG-2026-08-19-001** |
| 变更标题 | KV 缓存 SSD 卸载存储效率优化（kvssd-200g） |
| 变更日期 | 2026-08-19 |
| 变更类型 | 性能优化（存储效率）+ 存储配额 + 挂载修正（生产灰度） |
| 影响面 | TP4 生产集群 4 节点（01–04）；KV 卸载 fs 层 `/opt/aicad-kvssd`；启动脚本 `start_tp4_head.sh` / `start_tp4_worker.sh`；磁盘配额与挂载形态 |
| 负责人 | ⚠️ 待主理人确认 |
| 状态 | 灰度中（待验收） |
| 关联资产 | 备份链 `.bak-kvssd200g-<ts>`、`.bak-kvoffload-20260819-075524`；zstd wheel 固化 `<INSTALL_DIR>/envs/zstd/` |
| 回滚预估 | 恢复 `.bak` 脚本 + 重启 TP4 ≈ **10–15 分钟** |

## 2. 变更明细

### 2.1 变更①：容器内 io.py 覆盖补丁（存储效率核心）

| 项 | 值 |
|---|---|
| 目标文件（容器内） | `/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/fs/io.py` |
| 实施方式 | **卷挂载覆盖补丁**（非改镜像；容器启动参数增加 `-v` 覆盖 io.py） |
| 去重（dedup） | 同 hash 只写 1 份（内容寻址；写前计算块内容 hash，已有条目则仅登记引用） |
| 裁剪（trim） | 只写有效 token 区间（满块有效段实测 ~1.04MB），不写整块容量 |
| 压缩 | 对有效段做 **zstd level 3** 压缩 |
| 效果目标 | **382KB/token → ≤10KB/token（约 38×）** |
| 前置依赖 | `zstandard` wheel 四节点固化 `<INSTALL_DIR>/envs/zstd/`（容器内须可 import） |
| 文件格式 | 新格式魔数 `KVZSTD01`（见第二部分 E.2）；**旧格式兼容读取** |

### 2.2 变更②：四节点磁盘缓存配额统一 200G

| 项 | 值 |
|---|---|
| 配额 | 每节点 `/opt/aicad-kvssd` 缓存配额统一 **200G** |
| 01/02 | 原 800G → **重建为 200G**（存量缓存重建） |
| 03/04 | 原已 200G（保持不变） |
| 目的 | 统一四节点缓存容量上限，防单节点 SSD 写满拖垮系统盘；配合去重/裁剪/压缩降低实际占用 |

### 2.3 变更③：配额生效修正（bind mount loop 镜像）

| 项 | 值 |
|---|---|
| 问题 | `/opt/aicad-kvssd` 当前是**根分区普通目录**，配额（项目配额）**未生效** |
| 修正 | 新建 **loop 镜像**挂载至 `/mnt/kvssd-quota`（200G 稀疏镜像 + xfs/ext4 + 项目配额），再 **bind mount** 到 `/opt/aicad-kvssd` |
| 效果 | 配额硬上限真实生效；`/opt/aicad-kvssd` 视角显示 200G 独立设备而非根分区 |
| 遗留 | ⚠️ 开机自动挂载（fstab / systemd unit）需固化并验证（见第四部分待办） |

## 3. 影响面评估

| 维度 | 影响 | 风险等级 |
|---|---|---|
| 存储 | 四节点 SSD 缓存统一 200G 硬上限；单块占用 382KB→≤10KB/token | 低（正向） |
| 性能 | 新增 zstd 解/压缩 CPU 开销；SSD I/O 减少（体积小 + 去重） | 中（需灰度观察） |
| 正确性 | 新文件格式引入；依赖旧格式兼容读取，否则重启后读旧缓存文件失败 | 中 |
| 依赖 | 容器内必须可 `import zstandard`；缺失则新格式写路径不可用 | 高（前置） |
| 运维 | 配额/挂载形态变化；清理流程需适配 dedup 引用语义 | 低 |
| 回滚 | 恢复 `.bak` 脚本重启 TP4（10–15 分钟）；loop 挂载可卸除回普通目录 | 低 |

## 4. 回滚指引

> 回滚动作由主理人/执行者在节点上执行，本文档仅固化步骤与依据。

| 场景 | 步骤 | 优先级 |
|---|---|---|
| **全量回滚（补丁+配额+挂载）** | ① 四节点恢复启动脚本：`cp start_tp4_head.sh.bak-kvssd200g-<ts> start_tp4_head.sh`（01）、`cp start_tp4_worker.sh.bak-kvssd200g-<ts> start_tp4_worker.sh`（02/03/04）；若需回退 io 补丁另恢复 `.bak-kvoffload-20260819-075524` 对应参数 ② 重启 TP4（head-first：`bash <INSTALL_DIR>/scripts/start_tp4_cluster.sh`）③ 验证 8001=200 + 四机 healthy | P0 |
| 仅回退 io 补丁 | 恢复 `.bak-kvoffload-20260819-075524` 对应的卷挂载参数（去掉 io.py 覆盖）→ 重启 TP4；**旧缓存文件可被未打补丁 io.py 的旧格式路径读取（向前兼容）** | P1 |
| 仅回退配额/挂载 | 卸 bind mount：`umount /opt/aicad-kvssd` → 恢复普通目录；如需要回 800G 则删除 200G loop 镜像重建 | P2 |
| zstd wheel 故障 | 见第二部分 E.4-3（恢复安装，非回滚） | P1 |

**回滚锚点登记**：⚠️ 四机 `.bak-kvssd200g-<ts>` 与 `.bak-kvoffload-20260819-075524` 的实际时间戳与 MD5 需部署后回填至 `rollback-anchors-2026-08-12.md`（见第四部分待办）。

## 5. 部署后验证清单（灰度验收）

- [ ] 四机启动脚本含 kvssd 200G / io 补丁参数，`.bak` 备份链在位
- [ ] 容器内 `python3 -c "import zstandard"` 四机成功
- [ ] `docker logs vllm-tp4-rank0` 出现 offload/zstd 相关 INFO（新格式写路径已走）
- [ ] `df -h /opt/aicad-kvssd` 四机显示 200G loop 设备（非根分区）
- [ ] 配额报表命令可查、用量随流量增长
- [ ] `/metrics` 中 `kv_offload_*` 指标族可见（导出全量清单核对命名）
- [ ] e2e chat 冒烟 + 大 ctx（≥32K）请求通过（验证 fs 层读写闭环）
- [ ] 灰度观察 ≥72h：SSD 占用、lookup 延迟、CPU 峰值、TTFT/吞吐 vs 变更前基线

## 6. 变更记录表（追加行，供正式 Runbook 收录）

| 日期 | 变更编号 | 变更内容 | 影响面 | 负责人 | 状态 |
|---|---|---|---|---|---|
| 2026-08-19 | CHG-2026-08-19-001 | KV SSD 卸载存储效率优化：io.py 补丁（dedup+trim+zstd-3）+ 配额统一 200G + bind mount 配额生效修正 | TP4 四节点 | ⚠️ 待填 | 灰度中 |

---

# 第二部分：Runbook 增量章节（可直接并入正式 Runbook 的 KV 卸载小节）

> 本部分按正式 Runbook「§E」编号编写（沿用 append 的 §A–D 之后顺序）。并入时替换/标注正式 Runbook 中已有的 KV 卸载小节。

## §E. KV 缓存 SSD 卸载（TieringOffloadingSpec）章节（新）

### E.1 KV 卸载架构说明

生产已启用 **KV 缓存多级卸载（TieringOffloadingSpec）**，缓解 600k 长上下文下 GPU HBM 不足：

```
GPU HBM（主缓存，最快，容量优先保证最近活跃块）
   │  evict（GPU KV 池满）
   ▼
CPU 主层（DRAM，2GiB LRU）          ← TieringOffloadingSpec 主层
   │  evict（LRU 逐出 / 长期不活跃）
   ▼
fs 层（SSD，/opt/aicad-kvssd，200G 配额，loop 镜像 + bind mount）
```

- **写入路径**：GPU 池满 → 逐出到 CPU 主层（2GiB LRU）→ LRU 逐出落盘到 fs 层。fs 层经 io.py 补丁后以 `KVZSTD01` 格式写入（去重 + 裁剪 + zstd-3）。
- **读取路径**：请求命中 → 优先 GPU → CPU LRU → fs 层 load（解压 + 校验 + 回填 CPU/GPU）。
- **fs 层形态**：`/opt/aicad-kvssd` 现为 **loop 镜像 + bind mount**（200G 独立设备），非根分区普通目录——配额才真正生效。
- **设计取舍**：以 SSD I/O + 解压 CPU 换取 600k 上下文可用性；命中率低时 SSD I/O 与 zstd 解压是主要成本，是灰度重点观察项（见第三部分监控）。
- **规模参考**：补丁目标 **382KB/token → ≤10KB/token**；四节点缓存配额统一 **200G**。

### E.2 io 补丁工作原理与文件格式

补丁覆盖容器内 `vllm/v1/kv_offload/tiering/fs/io.py`，写路径引入三种优化，读路径保持新旧格式兼容。

**① 去重（content-addressed）**
- 写前计算块有效内容 hash；store 中已有同 hash 文件/条目 → 跳过写，仅登记引用（⚠️ 引用计数/硬链接实现以补丁源码为准）。
- 效果：重复内容（如共享前缀、多请求相同片段）在 SSD 只占一份。

**② 裁剪（trim）**
- KV 块有整块容量，但有效 token 区间远小于容量；只序列化 `[0, valid_len)` 段。
- 满块有效段实测 **~1.04MB**；`valid_len=0`（空块）不落有效负载。

**③ zstd-3 压缩**
- 对裁剪后的有效段做 zstd level 3 压缩后再落盘；读时解压回原有效段。

**新格式文件布局（魔数 KVZSTD01）** ⚠️ 字段宽度/字节序以补丁源码实测为准：

| 偏移 | 长度 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | magic | ASCII `KVZSTD01`，标识新格式 |
| 8 | 8 | orig_len | 压缩前有效段长度（uint64 LE） |
| 16 | 8 | payload_len | zstd 压缩负载长度（uint64 LE） |
| 24 | payload_len | payload | zstd-3 压缩数据 |

**旧格式兼容**：读路径先读 8 字节 magic；`!= "KVZSTD01"` 走旧格式路径（整块原始读写/旧布局）——因此**未打补丁/回滚后仍可读存量缓存文件，打补丁后可读旧缓存**。文件头魔数校验同时也是损坏快速判据（见 E.4-1）。

### E.3 运维手册

#### E.3.1 验证补丁生效（容器内）

```bash
# ① 前置依赖：zstandard 可 import（补丁写路径依赖）
docker exec vllm-tp4-rank0 python3 -c "import zstandard; print(zstandard.ZSTD_VERSION)"
# 期望：zstd 版本号（如 1.5.x）；ImportError = 依赖缺失 → E.4-3

# ② 确认 io.py 为覆盖补丁版
docker exec vllm-tp4-rank0 grep -c "KVZSTD01" \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/fs/io.py
# 期望 ≥1（含魔数常量 = 新格式写路径编译进模块）

# ③ 日志关键词（启动 + 运行期）
docker logs vllm-tp4-rank0 --since 1h 2>&1 | grep -Ei 'kv_offload|kvssd|zstd|KVZSTD|dedup|offload'
# 期望：offload / fs write / load 相关 INFO；出现 KVZSTD / zstd 行 = 新格式写路径已走
```

#### E.3.2 监控 kv_offload_* 指标

- 采集：Grafana（01 承载）数据源 `http://<NODE_IP>:8191`（02 Prometheus）拉取 `kv_offload_*` 指标族。
- 面板与 PromQL 见**第三部分**；⚠️ 指标名以部署后 `/metrics` 导出实测为准。

#### E.3.3 配额检查命令

```bash
# ① 挂载形态（验证 bind mount + loop 镜像生效，四机各执一次）
df -h /opt/aicad-kvssd          # Size 应显示 200G 的 loop 设备，而非根分区
mount | grep -E 'kvssd|loop'     # 应见 /mnt/kvssd-quota loop 挂载 + /opt/aicad-kvssd bind
losetup -a | grep kvssd          # 应见 loop 设备指向 200G 稀疏镜像

# ② 配额用量报表（文件系统类型以实际为准）
xfs_quota -x -c 'report -h' /mnt/kvssd-quota    # xfs 项目配额
quota -s /opt/aicad-kvssd                        # ext4 项目配额

# ③ 容器内视角
docker exec vllm-tp4-rank0 df -h /opt/aicad-kvssd

# ④ 四节点一致性核对
for h in node01 node01 node01 node01; do
  ssh "$h" 'df -h /opt/aicad-kvssd | tail -1'
done
```

**判定**：① 显示 200G 独立设备 = 挂载修正生效；② 报表可用 = 配额生效；四机一致 = 变更②落地。

#### E.3.4 缓存清理流程

> fs 层是可重建缓存：清空仅触发重新计算/重写，不丢模型状态。**含 dedup 引用语义，禁止运行中逐文件删除**（会孤儿化引用 / 触发 load fail）。

```bash
# 前提：vLLM 停止（维护窗口）或低峰；四机依序执行
for h in node01 node01 node01 node01; do
  ssh "$h" 'rm -rf /opt/aicad-kvssd/*'
done
# 清空后复核：df -h /opt/aicad-kvssd 容量回收；/metrics kv_offload_store_bytes 归零
# 随后随流量自然重建（观察增长曲线符合预期）
```

**红线**：运行中勿删；`kv_load_failure_policy=fail` 下被删块会直接报错（见 E.4-1）。

#### E.3.5 zstd wheel 缺失恢复

```bash
# 固化位置：<INSTALL_DIR>/envs/zstd/（四机）
# 容器内安装（wheel 需已挂载进容器，或经 docker cp）
docker exec vllm-tp4-rank0 pip install <INSTALL_DIR>/envs/zstd/zstandard-*.whl
# 或：容器启动参数确保 -v <INSTALL_DIR>/envs/zstd:/opt/zstd-wheel:ro + 启动时安装
# 验证：import zstandard 成功 → 重启引擎（head-first）→ 新格式读写恢复
```

### E.4 故障排查 FAQ

#### E.4-1 KV 损坏 / 读 SSD 失败（`kv_load_failure_policy=fail`）

- **症状**：请求报错 / 引擎拒绝服务；日志含 load fail / magic 校验失败 / 截断读。
- **机制**：`kv_load_failure_policy=fail` 下，fs 层块读取失败（损坏/截断/被删）视为**致命**——依赖该块的请求失败（对比可配置为 ignore/重算的兜底策略，⚠️ 以引擎支持为准）。
- **处置**：① 确认损坏范围（日志中文件路径/哈希）② 维护窗口执行 E.3.4 清理该范围或全量 ③ 核对磁盘健康（loop 镜像所在卷 SMART/`df`）④ 重启 TP4 head-first。
- **预防**：文件头魔数 `KVZSTD01` 校验即损坏快速判据；监控 lookup 失败率（第三部分）。

#### E.4-2 压缩 CPU 占用过高（降 zstd level）

- **症状**：GB10 CPU 峰值升高、decode/TTFT 抖动；zstd 线程占满核。
- **处置**：① 将 zstd level 由 3 降为 1（改 io.py 补丁常量，重启生效）② 限制压缩线程数/绑定核（⚠️ 以补丁实现为准）③ 权衡：level 1 压缩比略降但 CPU 显著回落。
- **判定**：对照第三部分 CPU/延迟面板，确认瓶颈确在压缩路径（非 SSD I/O）。

#### E.4-3 配额满（镜像重建流程）

- **症状**：`df -h /opt/aicad-kvssd` Use% 100%；写入失败；`kv_offload_store_bytes` 触顶。
- **机制**：loop 镜像为 200G 硬上限；配额报表可提前预警。
- **处置（重建镜像）**：① 停止 vLLM ② `umount /opt/aicad-kvssd` 与 `/mnt/kvssd-quota` ③ 删除旧 200G 镜像文件 ④ 重建 200G 稀疏镜像 + 格式化 + 挂载 + bind mount ⑤ 恢复 E.3.3 校验 ⑥ 启动 TP4，缓存自然重建。
- **预防**：设置 180G（90%）告警；先确认去重/裁剪/压缩已生效（正常应远低于 200G）。

#### E.4-4 新旧格式混读 / 回滚后读缓存

- **场景**：灰度中途回滚、或重启前有旧格式存量文件。
- **行为**：读路径按 magic 分派，`KVZSTD01` 走新格式，否则走旧格式——**混存可读**。
- **注意**：若误将新格式文件交给未打补丁 io.py，旧路径可能按旧布局解析失败 → 此时执行缓存清理（E.3.4）让缓存重建，或直接恢复补丁。

---

# 第三部分：监控面板建议（Grafana）

> ⚠️ 指标名以下方「候选」为准，**部署后以 `/metrics` 导出 `kv_offload_*` 全量清单核对并回填**（见第四部分待办①）。数据源：Grafana 01 → `http://<NODE_IP>:8191`（02 Prometheus）。

| # | 面板 | 指标（候选名） | PromQL 示例 | 建议告警 |
|---|---|---|---|---|
| 1 | **SSD 存储总量（当前占用）** | `kv_offload_store_bytes` | `sum(kv_offload_store_bytes)` | 单机 >180G（90%）→ 配额预警 |
| 2 | **SSD 写入速率（增长曲线）** | `kv_offload_store_bytes_total` | `rate(kv_offload_store_bytes_total[5m])` | 突增 >阈值 15min → 查泄漏/未去重 |
| 3 | **CPU 主层用量** | `kv_offload_cpu_cache_usage`（或 `_percent`） | `kv_offload_cpu_cache_usage` | >90% 持续 → LRU 抖动 |
| 4 | **tiering lookup 延迟** | `kv_offload_tiering_lookup_duration_seconds`（histogram） | `histogram_quantile(0.95, sum(rate(kv_offload_tiering_lookup_duration_seconds_bucket[5m])) by (le))` | p95 劣化 → SSD I/O/解压瓶颈 |
| 5 | **fs 层读写延迟** | `kv_offload_store_load_duration_seconds`（⚠️ 候选） | `histogram_quantile(0.95, sum(rate(kv_offload_store_load_duration_seconds_bucket[5m])) by (le))` | 观察，异常时配合 E.4-1 |
| 6 | **DRAM 水位** | `kv_offload_cpu_usage_bytes`（⚠️ 候选）或节点内存 | `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100` | >85% → CPU 主层/整体内存压力 |
| 7 | **SSD 增长/文件数** | fs 层文件数（⚠️ 需暴露或 textfile） | 目录 `find /opt/aicad-kvssd -type f \| wc -l` 经 node_exporter textfile | 文件数异常激增 → dedup 失效检查 |

**面板组合建议**：面板 1+2+7 一组（存储效率），面板 3+6 一组（内存水位），面板 4+5 一组（卸载延迟），叠加 e2e TTFT/吞吐对照（与变更前基线对比，见 §E.1 取舍）。

---

# 第四部分：给主理人的待办清单（部署后补写/确认）

> 以下为文档侧与记录侧的收尾项，**建议部署验收后 24h 内完成**（对齐团队「任何变更后 24h 内回填」纪律）。

1. **[指标回填] 部署后导出 `/metrics` 中 `kv_offload_*` 全量指标名**，与第三部分候选名核对并修正 PromQL/面板（最关键，避免文档与生产漂移）。
2. **[回滚锚点] 登记四机 `.bak-kvssd200g-<ts>` 与 `.bak-kvoffload-20260819-075524` 的实际时间戳 + MD5**，回填 `rollback-anchors-2026-08-12.md`（新增 §kvssd-200g 小节）。
3. **[挂载固化] 确认 loop 镜像路径/大小/文件系统类型（xfs vs ext4）+ 开机自动挂载（fstab 或 systemd unit）已固化**并记录在维护手册。
4. **[脚本参数] 记录 start_tp4_head.sh / start_tp4_worker.sh 中 io.py 卷挂载参数（`-v 源:目标:ro`）实际值**，便于回滚复现与审计。
5. **[灰度数据] 收集灰度观察期（≥72h）数据**：SSD 占用增长曲线、tiering lookup p95、CPU 峰值、TTFT/吞吐 vs 变更前基线，形成验收结论。
6. **[规模复测] 若后续切换 400k→600k ctx**，复测全量 KV 落盘量是否仍在 200G 配额内（600k 全量落盘测算）。
7. **[文档合并] 正式 Runbook v1.4+（目标 `F:\AICADqwen\runbook-dspark-vllm-2026-08-08.md`）恢复后**，将本文件第二部分 §E 并入其 KV 卸载小节，并同步更新 `ops/server-maintenance-handbook.md` §2 参数表（新增 KV offload env/参数）。
8. **[负责人] 补全第一部分变更记录中「负责人」字段**。

---

> 本文由工程保障团队技术文档师 Docu 整理自 2026-08-19 变更任务说明与既有 Runbook/回滚锚点文档。准确性优先，未验证结论已标注 ⚠️；关键决策请由人类工程负责人复核。
