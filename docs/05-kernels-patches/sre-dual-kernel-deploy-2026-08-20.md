# SRE 双算子部署现场核查报告 — 2026-08-20

**执行人**: 雷克斯（Rex）· SRE 工程师
**窗口**: 生产停机窗口（TP4 全线停止，只读取证）
**取证方式**: SSH 只读命令 + 一次性容器（`docker run --rm`，用后即删，无 GPU 绑定，无生产写操作）
**涉及节点**: node01（head，<NODE_IP>）/ node01（.187）/ node01（.188）/ node01（.189）
**生产镜像**: `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（四节点同镜像，已确认本地存在）

---

## 1. 节点状态核查（4 节点）

### 1.1 状态总表

| 项目 | node01 (head) | node01 | node01 | node01 |
|---|---|---|---|---|
| uptime | 3d 1h47m | 3d 1h49m | 1d 5h03m | 1d 5h03m |
| 内存 used/total | 7.0G/121G | 6.9G/121G | 10G/121G | 10G/121G |
| load (1m) | 0.10 | 0.01 | 4.27 | 0.10 |
| swap used | 10G/15G | 12G/15G | 4.3G/15G | 4.3G/15G |
| Driver / CUDA | 580.173.02 / 13.0 | 580.173.02 / 13.0 | 580.173.02 / 13.0 | 580.173.02 / 13.0 |
| GPU util | 0% | 0% | 0% | 0% |
| GPU 进程 | 无 | 无 | VLLM::EngineCore 5750MiB* | VLLM::EngineCore 5750MiB* |
| TP4 容器 | 不存在（已删） | 不存在（已删） | 不存在（已删） | 不存在（已删） |
| vllm-* systemd | 全部 inactive | 全部 inactive | 全部 inactive | 全部 inactive |

\* 03/04 的 GPU VLLM::EngineCore 进程（PID 3827/3881，运行 1d5h）归属 `anemll-embed-8022` 容器（`vllm serve /models/Qwen3-Embedding-0.6B --port 8022 --kv-cache-memory=4294967296`），是独立的嵌入服务，**与 TP4 生产无关**。TP4 判定干净停止。

**OOM 恢复确认**: 四节点内存均正常（available ≥110G）；swap 残留占用为历史压力痕迹（GB10 统一内存常态），01 的 15 分钟 load=2.43 为历史均值衰减，1m/5m 负载已归零。恢复判定：✅。

### 1.2 systemd 服务状态与矛盾项记录

服务实际名称为 `vllm-cluster` / `vllm-healthcheck`（+timer）/ `vllm-tp4-head` / `vllm-tp4-worker`（注意：任务书中 "aicad-monitor/aicad-healthcheck" 单元名 not-found，实际单元名如上）。

| 单元 | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| vllm-cluster.service | disabled / inactive | 不存在 | 不存在 | 不存在 |
| vllm-healthcheck.service + .timer | disabled / inactive | 不存在 | 不存在 | 不存在 |
| vllm-tp4-head.service | disabled / inactive | 不存在 | 不存在 | 不存在 |
| vllm-tp4-worker.service | 不存在 | **enabled** / inactive | **enabled** / inactive | **enabled** / inactive |

**⚠ 矛盾项（风险，未修改，仅记录）**: 02/03/04 的 `vllm-tp4-worker.service` 处于 **enabled**（开机自启）而 head 侧全部 disabled。若 worker 节点在 head 未就绪时重启，worker 将自动拉起尝试连接不存在的 head，可能导致僵尸重试循环。建议下次变更窗口统一 `systemctl disable`（本窗口只读约束下不动）。

head 侧 journalctl 证实 monitor 最后一次尝试拉起为 08-20 08:08（此后服务 inactive，无自愈循环残留）。

---

## 2. nvfp4 目录与 md5 验证（4 节点）

### 2.1 md5 验证表

| 文件 | 基线 | 01 | 02 | 03 | 04 | 结论 |
|---|---|---|---|---|---|---|
| kernel1/nvfp4_4w4a_mmaf.py | 2d9cda46… | 2d9cda4686e2d3cb8fc406883c641873 | 同 | 同 | 同 | ✅ 4/4 一致，= routeA 基线 |
| kernel2/nvfp4_ds_mla_kv_linear_v17_triton.py | a795b2b4… | a795b2b4a486f8bd2b07366890e928af | 同 | 同 | 同 | ✅ 4/4 一致，= v17 交付基线 |

### 2.2 目录清单（4 节点结构一致）

- `kernel1/`: 仅 `nvfp4_4w4a_mmaf.py`（4316B，routeA）
- `kernel2/`: `benchmark_nvfp4_ds_mla_kv_linear_v17.py` / `nvfp4_ds_mla_kv_linear_torch.py` / `nvfp4_ds_mla_kv_linear_triton.py` / `nvfp4_ds_mla_kv_linear_v17_triton.py` / `test_nvfp4_ds_mla_kv_linear_v17.py`
- **关键补充**: 生产 `nvfp4_ds_mla_kv_linear_v17_triton.py` 中 **grep 不到任何 `BLOCK_SIZE` / `block_size` / `block_table`** → 生产运行的 v17 是**非 paged 变体**，交付包中的 paged 变体不在生产路径上。

### 2.3 plugin-src/ 清单（仅存在于 head=01，worker 无）

```
<INSTALL_DIR>/nvfp4/plugin-src/
├── ab_run.py
└── nvfp4_vllm_plugin/
    ├── setup.py
    ├── ab_routeA_vs_b12x.py
    ├── ab_v17_semantics.py
    └── nvfp4_vllm_plugin/
        ├── __init__.py
        ├── kv_writer.py
        ├── moe_method.py
        └── quant_config.py
```

---

## 3. ★关键验证：生产 nvfp4_ds_mla KV cache 块大小

### 3.1 结论（先行）

> **生产块大小 = 64（paged 形状 [64, 584]），kernel2 交付包 paged 变体的 BLOCK_SIZE=256 假设不成立，paged 变体按现状直接不可用。**
> 历史勘查记忆（paged [64,584]）正确；交付包注释中的 256 是错误的。

### 3.2 证据链（按验证途径排列）

**途径 a) 上次生产运行日志 — 容器已删，docker logs 无残留**；`~/vllm-logs/`（容器内 /var/log/vllm 挂载）仅有 nccl 检查文件，无引擎日志。但 journalctl 保留了完整 serve 命令：

```
8月 19 00:02 … [i] serve 命令: vllm serve --model /models --served-model-name deepseek-v4-flash-0731
  --kv-cache-dtype nvfp4_ds_mla --max-model-len 600000 --max-num-seqs 12
  --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024
  --scheduling-policy priority --gpu-memory-utilization 0.80 …
```
（该命令形态在 08-19 至 08-20 多次启动中一致，最近一次 08-20 08:08。）

**途径 c) 启动脚本参数** — head `start_tp4_head.sh` 与三台 worker `start_tp4_worker.sh` 均 grep 不到 `--block-size` → 块大小完全由镜像内 fork 逻辑决定。

**途径 b) 生产镜像内 fork 源码（一次性容器 grep，决定性证据）**:

1. `vllm/config/cache.py:48` — `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`（通用默认值，但 DSv4 MLA 后端覆盖之）
2. `vllm/v1/attention/backends/mla/sparse_swa.py:77` —
   ```python
   # The C4A KV block shape [256//4, head_dim] = [64, head_dim]
   # determines the SWA block size of 64 tokens per block.
   self.block_size = 64        # ← 硬编码
   ```
3. `vllm/v1/attention/backends/mla/flashmla_sparse.py:158` — `block_size: int = 64`；且 `get_supported_kernel_block_sizes() -> [64]`（**仅支持 64**）
4. `vllm/models/deepseek_v4/sparse_mla.py:105-107` —
   ```python
   # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
   return (num_blocks, block_size, 584)
   ```
5. `vllm/models/deepseek_v4/compressor.py:158` — `The KV block shape [256//4, head_dim] = [64, 584] determines: …`
6. `vllm/models/deepseek_v4/attention.py:620-626` — "use the proven 584-byte DSpark NVFP4 envelope" / `584`
7. nvfp4_ds_mla 布局 alignment: `sparse_swa.py get_kv_cache_spec()` 中 `alignment=584 if uses_nvfp4_ds_mla_layout`

### 3.3 推论

- 生产 paged KV cache：**每 token 584 字节（uint8: 448 NoPE + 128 RoPE + 8 scale），block = 64 token/page**，与记忆中 [64,584] 完全吻合。
- 交付包 paged 变体 `block_table[seq, pos//256]` 与生产 `block_table[seq, pos//64]` 语义不符：同一条目索引错位、页内偏移计算错误 → **写入/读取即数据损坏，直接不可用**（不是性能问题，是正确性问题）。
- 若要启用 paged 变体，须将 BLOCK_SIZE 改为 64 并重验；或维持现状（生产用非 paged v17）。

---

## 4. routeB P0 环境现状检查

| 检查项 | 结果 | 判定 |
|---|---|---|
| CUTLASS DSL 安装（生产镜像内） | **已安装**：`nvidia-cutlass-dsl 4.5.2`（含 libs-base 4.5.2、**libs-core 4.6.0、libs-cu12 4.6.0、libs-cu13 4.5.2**）；`import cutlass` 成功，版本 4.5.2 | ⚠ 与预期"未装"不符——P0 静态审查结论需修正；**但存在 4.5.2/4.6.0 混装**，routeB 编译目标对 DSL 版本敏感，须先验证混装组合可用 |
| Driver ≥ 580.142（P1 patch 前提） | 四节点全部 **580.173.02**，CUDA 13.0 | ✅ 满足 |
| Driver/CUDA 四节点一致性 | 580.173.02 / CUDA 13.0，完全一致 | ✅ |
| routeB 文件是否已部署 | 否——kernel1 md5 仍为 routeA 基线（2d9cda46…） | routeB 未进入生产路径 |

**routeB 环境小结**: 之前"precheck 静态审查认为 P0（安装 DSL）未执行"的判断与现场不符——镜像内实际已有 CUTLASS DSL。修正后的阻塞点是：(1) DSL 版本混装（core/cu12=4.6.0 vs cu13=4.5.2）与 routeB 交付包声明的目标版本是否匹配未验证；(2) routeB 内核文件本身未部署、未 A/B 验证。**P1 的 driver 前提已满足，P0 需重新定性为"已安装待版本核验"而非"未安装"。**

---

## 5. 双算子部署 Go / No-Go 矩阵

| 算子 | 前置条件 | 现状 | 决策 | 理由 |
|---|---|---|---|---|
| **kernel1 → routeB** | P0: CUTLASS DSL 就绪 | 镜像内已装 4.5.2，但与 libs-core/cu12 4.6.0 混装，未做 routeB 编译冒烟 | **NO-GO（本窗口）** | ① routeB 文件未部署（kernel1 仍为 routeA）；② DSL 版本混装未核验，routeB 编译产物正确性无保证；③ 无 A/B 基线数据支撑替换决策。**环境前提已具备，转条件性路径：先做版本核验+单算子离线编译冒烟，再排验证窗口** |
| **kernel1 → routeA（维持现状）** | 无变更 | md5 4/4 = 基线 | **Go（no-op）** | 不动即安全 |
| **kernel2 → v17** | 生产 = 交付包 | md5 4/4 一致，无差异 | **Go（no-op 确认）** | "部署"实为空操作：生产已是 v17。只需归档确认，无文件变更 |
| **kernel2 → paged 变体** | BLOCK_SIZE 与生产一致 | 交付包硬编码 256 ≠ 生产 64 | **NO-GO（硬阻塞）** | 块大小不匹配 = 正确性缺陷（见 §3），改 256→64 并重新验证前禁止上生产 |

---

## 6. 部署检查清单（停机窗口安全边界内）

**安全操作边界（本窗口已遵守）**: 只读命令 + `docker run --rm`（无 `--gpus`、无生产挂载写、用后即删）；不启动生产容器；不改 start 脚本；不动 <INSTALL_DIR> 下任何文件；不动 systemd 状态。

### 部署前（Go 决策成立后执行，本窗口只完成核验部分）
- [x] 四节点状态核验（本报告 §1）
- [x] kernel1/kernel2 md5 四节点一致性（§2）
- [x] 生产 KV cache 块大小取证（§3）
- [x] routeB 环境前置核验（§4）
- [ ] routeB 前置补验（NO-GO 解除条件）：
  - [ ] 核验镜像内 DSL 混装（core 4.6.0 / cu13 4.5.2）与 routeB 交付包要求版本一致；不一致则统一版本后重做 P0
  - [ ] 一次性容器内 routeB 单算子编译+数值冒烟（GPU 空闲可安全执行）
  - [ ] A/B 基线：routeA vs routeB 延迟/吞吐对比（plugin-src/ab_routeA_vs_b12x.py 已在 head 备好）
- [ ] 确认 02/03/04 `vllm-tp4-worker.service` disable 策略（矛盾项，防重启意外拉起）
- [ ] 快照回滚锚点：记录当前 md5（本报告 §2.1 即锚点）+ 备份目标文件时间戳

### 部署中（未来窗口）
- [ ] 备份 `<INSTALL_DIR>/nvfp4/kernel1/nvfp4_4w4a_mmaf.py` → `*.bak-routeA-<ts>`（仅当 routeB Go）
- [ ] 四节点分发改文件（同 md5 校验，一字不差）
- [ ] 不改 start 脚本、不改 systemd unit、不改镜像
- [ ] 观察 head 容器 `Application startup complete`（≤15min cold start）+ `/health` 通过
- [ ] 观察 KV cache 分配日志行确认块大小仍为 64（`grep 'GPU KV cache' docker logs`）

### 部署后
- [ ] 冒烟推理请求（短 prompt + 长 context 各一）
- [ ] 监控 30min：GPU util、显存、TTFT/TPOT 对比基线
- [ ] 回归确认无新 CUDA/JIT 错误（tilelang/b12x cache 首次编译属预期）

---

## 7. 回滚方案

| 算子 | 回滚触发条件 | 回滚动作 | 预期 RTO |
|---|---|---|---|
| kernel1 (routeB 上线后) | 启动失败 / 冒烟错 / TTFT 劣化超阈值 / 任意 SEV3+ 告警 | `cp kernel1/nvfp4_4w4a_mmaf.bak-routeA-<ts> kernel1/nvfp4_4w4a_mmaf.py`（四节点），md5 复核 = 2d9cda46…，重启 TP4 | 文件回退 <2min；含冷启动重启 ≤20min |
| kernel2 (v17) | 无需回滚 | 生产 = 交付基线，无变更 | 0 |
| kernel2 (若未来上 paged 修正版) | 同 kernel1 | 同上（v17 非 paged 文件即回退锚点，md5 = a795b2b4…） | 同 kernel1 |

兜底：`start_tp4_head.sh` 存在 20+ 个历史 .bak 脚本锚点（含 08-20 04:33/05:05 最新两版），脚本级回退链完整。

---

## 8. 风险与 SEV 评级（停机窗口内全 SEV4）

| # | 风险 | SEV | 缓解 |
|---|---|---|---|
| R1 | kernel2 paged 变体 BLOCK_SIZE=256 与生产 64 不匹配，误部署导致 KV 数据损坏 | SEV4（窗口内未部署，无实际影响；若误上线则升 SEV1） | §5 NO-GO 硬阻塞；部署清单含块大小复核项 |
| R2 | 02/03/04 worker 服务 enabled，节点重启时在无 head 下自启 → 僵尸重试 | SEV4 | 记录矛盾项，建议下窗口 disable |
| R3 | CUTLASS DSL 4.5.2/4.6.0 混装，routeB 编译行为不确定 | SEV4 | P0 重新核验版本；一次性容器冒烟后再谈部署 |
| R4 | 03/04 GPU 被 anemll-embed 占 5.75G，若误判"GPU 不空闲"可能延误窗口操作 | SEV4 | 本报告已定性归属（非 TP4）；生产 TP4 冷启动时需注意共存显存 |
| R5 | 01 swap 残留 10G，若直接重启可能延长冷启动 | SEV4 | 重启前 `swapoff -a && swapon -a`（需变更授权，本窗口未执行） |

---

## 附：取证命令记录（关键项）

```bash
# 节点状态
ssh <node> "uptime; free -h; docker ps -a; systemctl is-active vllm-tp4-head vllm-tp4-worker; nvidia-smi"
# md5
ssh <node> "md5sum <INSTALL_DIR>/nvfp4/kernel{1,2}/*.py"
# serve 命令历史（无 --block-size 证据）
ssh node01 "sudo journalctl -u vllm-tp4-head.service --since 2026-08-19 | grep 'serve 命令'"
# 镜像内源码取证（一次性容器，无 GPU）
docker run --rm --entrypoint bash dspark-vllm-gx10:0.2.1-v026.0 -lc \
  "grep -n 'block_size' .../vllm/v1/attention/backends/mla/sparse_swa.py; \
   sed -n '60,120p' .../sparse_swa.py; sed -n '145,175p' .../compressor.py; \
   pip show nvidia-cutlass-dsl-libs-cu13; python3 -c 'import cutlass'"
```

**报告完。所有取证均为只读；未对生产做任何变更。**
