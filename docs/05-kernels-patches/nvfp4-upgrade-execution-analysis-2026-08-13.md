# NVFP4/CUDA13.2 升级方案执行分析（对照服务器真实环境）

**日期**：2026-08-13 18:30
**依据**：nvfp4-cu132-upgrade-plan-2026-08-13.md（原文）+ 服务器四机实时核查 + 今日新增调查结论（hf.rimuru.work 通道 / Rarri 权重选择 / MTP 双证据 / L1-L2 分层）
**性质**：Phase 0 执行前逐项分析，含修订建议

---

## 一、方案 §1.1 环境事实核验（服务器实测 vs 方案声明）

| # | 项 | 方案声明 | 服务器实测（今日） | 判定 |
|---|----|---------|-------------------|------|
| ① | 驱动 | 580.173.02（四机） | **580.173.02**（01 实测） | ✅ |
| ② | 容器 torch | 2.11.0+cu130 | **2.11.0+cu130**（rank0 容器内） | ✅ |
| ③ | cuBLASLt | 13.1.1.3（CUDA 13.0 系） | libcublasLt.so.13 存在 | ✅ |
| ④ | 推理镜像 | anemll/dspark-vllm-gx10:0.2.1-v026.0 | **<NODE_IP>:5000/anemll/...:0.2.1-v026.0**（运行中） | ✅ |
| ⑤ | 权重 | deepseek-v4-flash-0731（MXFP4） | 48 shards / 156G / expert_dtype=fp4（01/02 各一，03/04 NFS） | ✅ |
| ⑥ | KV cache | nvfp4_ds_mla | **nvfp4_ds_mla**（运行参数实测） | ✅ |
| ⑦ | MoE 后端 | flashinfer_b12x | **flashinfer_b12x**（运行参数实测） | ✅ |
| ⑧ | 投机 | dspark num_spec=5 动态K | **dspark**（运行参数实测，[[1,1,5],[2,4,4],[5,6,3]]） | ✅ |
| ⑨ | ring-only NCCL | b7784b49（v3） | **b7784b49**（md5 实测） | ✅ |
| ⑩ | shim | ce43c688（v8） | **ce43c688**（md5 实测） | ✅ |

**结论：方案对生产环境的刻画 100% 准确，无漂移，可作为执行基线。**

---

## 二、今日新增关键事实（对方案的三处结构性修订）

### 修订 A：权重来源与目标变更（§2.2 缺口 2）
- 方案原案：用户代理 → HF 下载 MJPansa 0731-NVFP4（~180GB）
- **现状**：hf.rimuru.work（Cloudflare Workers 代理，GitHub AinzRimuru/HuggingfaceProxy）验证可用——API/resolve 全通、CDN Range 支持、无需 token。**已启动 Rarri/DeepSeek-V4-Flash-0731-NVFP4 下载**（02 服务器直连，aria2 8 连接 ≈ 16.8MB/s，164GB 预计 ~1.5-2 小时，已完成 30G/17 shard）
- **目标改为 Rarri 版**（非 MJPansa）：ModelOpt 0.45.0 官方流程、**MTP 全转**（2,304 张量 100% lossless）、DSpark 接受率 49.4% 实测（decode 1.78-1.96×）、MIT、48-shard 布局、conversion 报告齐全
- ⚠️ hf-mirror.com 已验证失效（Cloudflare 层全路径 308 到被墙官方；源站 160.16.86.14 不可达）

### 修订 B：MTP 处理必须落地（§2.4 投机解码兼容性）
- 方案原文："MJPansa 保留 DSpark/MTP，投机链路不丢"——隐含"不转 MTP"可行
- **双证据否定**：Rarri（2026-08-03 实测：MTP 保持 MXFP4 → 接受率崩至 ~15%；全转 → 49.4%）、tsarihan（0.31→0.121）独立证实；auroter 用运行时 routing patch（43 行）解决（接受率 41%）
- **决策**：采用 **Rarri 版权重（MTP 已全转）**，无需 patch、无需 A/B MTP 策略——此问题随目标切换自动消解
- 动态K `[5,6,3]` 的 3<block 5 校验问题仍存在（方案 §2.4 已指出）：Rarri 权重 dspark_block_size=5 不变，升级时**一并修为 `[5,6,5]`**（或删除 <5 档）

### 修订 C：收益分层 L1/L2（§2.1 核心思路）
- 方案原文：收益核心 = cuBLASLt 13.2 NVFP4 3×（prefill）
- **新增证据**：NVFP4 收益不依赖 cu132——tsarihan 在 **cu130 + FlashInfer CUTLASS JIT 修复**下实测 prefill 1.14-1.32×；Marlin 后端在 cu130 即 +16-20% decode
- **L1（先做）**：cu130 现有镜像 + NVFP4 权重 + `--kernel-config '{"moe_backend":"marlin"}'`（decode 保底）+ FlashInfer CUTLASS JIT（prefill 收益）——**零镜像构建**，下载完成后 1-2 天可 A/B
- **L2（增量）**：cu132 自建镜像 + cuBLASLt 13.2 原生 NVFP4（官方 release notes 点名 DGX Spark 大 M/N 3×）——自建 1-2 人日，A/B 定论
- **风险修正**：cuBLAS 13.4.0（13.2 U1）有 NVFP4 tensor-wide scaling bug → 自建镜像必须 13.2.2+ 或 cuBLAS 13.4.1 patch

---

## 三、Phase 0-5 逐项分析（环境现状 + 判定 + 前置 + 修订）

### Phase 0：前置准备（第 0 天，零风险）—— ✅ 已基本完成

| 项 | 环境现状 | 判定 |
|----|---------|------|
| 镜像 tag 锚点 | 0.2.1-v026.0（registry 前缀，四机一致） | ✅ |
| ring-only MD5 | b7784b49 实测一致 | ✅ |
| shim v8 MD5 | ce43c688 实测一致 | ✅ |
| 启动脚本 MD5 | head=72137b8a（01）；worker 02≠03/04（已知差异，非阻塞） | ✅ |
| 权重 .local-backup | 03 上 <MODELS_DIR>/deepseek-v4-flash-0731.local-backup = **156G** | ✅ |
| 02 下载通道 | **hf.rimuru.work 已验证可用，下载进行中（30G/164G）** | ✅（原方案"用户代理"已替换） |
| canary 环境（<MGMT_OCTET>/<MGMT_OCTET>） | 03/04 磁盘 622/631G 可用；03 内存充足（embed+TP4 已运行，资源有余量） | ✅（见 Phase 2 风险注） |

**修订**：Phase 0 补两项——①下载校验脚本（75 文件清单 + sha256 抽样 + mtp_nvfp4_build_report.json 核对）；②`[5,6,3]→[5,6,5]` 动态K 修正列入 Phase 4 参数固化清单。

### Phase 1：NVFP4 权重下载与校验 —— 🔄 执行中

| 项 | 状态 |
|----|------|
| 通道 | ✅ hf.rimuru.work（替代原"用户代理"案，无需 token） |
| 目标 | ✅ Rarri/DeepSeek-V4-Flash-0731-NVFP4（MTP 全转，替代 MJPansa） |
| 落点 | 02 `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4/`（磁盘 2.5T 可用） |
| 进度 | 30G / 164G（~17/48 shards），aria2 8 连接串行，预计 18:30+~1.5h 完成 |
| 校验计划 | ①75 文件数对照 ②config.json 关键字段（moe_quant_algo=NVFP4） ③shard 字节数对照 dl_list ④mtp_nvfp4_build_report.json（MTP 全转佐证） ⑤随机 sha256 抽样 |

**后续分发**（下载完成后）：02 → rsync 01（3.1T 可用）→ 01/02 NFS 导出 → 03/04 挂载（622/631G 可用，NFS 挂载不占本地空间）。

### Phase 2：CUDA 13.2 镜像自建（1-2 人日）—— ⚠️ 需重排（先 L1 后 L2）

| 项 | 分析 |
|----|------|
| 前置 | 03/04 canary 环境可用（磁盘/内存有余量）；但 03/04 正跑 TP4 worker + embed——**构建需用临时容器**（生产镜像 ENTRYPOINT=vllm serve，须 --entrypoint sleep 覆盖，Phase 1 已有先例） |
| torch 升级 | 2.11.0+cu130 → torch 2.12 系 cu132（需确认 0.26.1.dev0 兼容；上游无 cu132 支持记录 → 自建必经之路） |
| sm_121 内核 | TORCH_CUDA_ARCH_LIST=12.1a 重编（参考 PR #38126 arch guard） |
| ring-only NCCL | 方案 A（直接拷入 libnccl.so.2.30.7 验证 forward-compat）→ 5 分钟出结果；失败走 B（重编，源码在 <INSTALL_DIR>/backup/tp4-20260812/src，流程现成） |
| shim v8 | 重编（依赖 CUDA 版本）；⚠️ **v8 源码未归档**（仅有 .so 与 v3 源）——cu132 下若 shim 需重编，只能从 v3 源 + v4-v8 差异说明重建（P2 项，已列入归档待办） |
| cuBLAS 版本 | **必须 13.2.2 或 cuBLAS 13.4.1+**（13.2 U1 的 NVFP4 scaling bug） |
| 产物 | `dspark-vllm-gx10:0.2.1-cu132-nvfp4`（registry 分发四机） |

**修订（重要）**：Phase 2 降级为 **L2 第二阶段**——先做 L1（cu130 现有镜像 + NVFP4 权重 + marlin/CUTLASS 后端 env），1-2 天出 A/B 数据；L2 cu132 自建在 L1 数据基础上决策（若 L1 prefill 已达 1.3×，cu132 增量需权衡自建成本）。

### Phase 3：测试环境 A/B 验证（核心）—— ✅ 方案可行，测试档位补充

| 验证项 | 基线（现 FP8+MXFP4/b12x） | 目标 | 判定 |
|--------|--------------------------|------|------|
| prefill 131072/c1 | 2013-2016 tok/s | L1 ≥2400（1.2×） | GO 判据 |
| prefill 8192/32768/c1 | 基线 | 不劣化 | 硬门槛 |
| decode 131072/c1 | 110-115 | ≥100 | 硬门槛 |
| decode 32768/c1 | 95-103 | 不劣化 | 硬门槛 |
| **DSpark 接受率** | 0.73-0.93 | **≥0.4**（Rarri 实测 49.4%） | **新增必测项** |
| 精度抽样 | GSM8K/已知 prompt | 无漂移 | 硬门槛 |

**修订**：新增两项——①DSpark 接受率必测（Rarri 版权重接受率有 49.4% 背书，A/B 确认生产环境不退化）；②MoE 后端 A/B 矩阵（marlin vs flashinfer_cutlass vs b12x 基线），对齐 auroter 部署要点（cutlass 禁 expert-parallel，vLLM #42118 NaN 隐患；VLLM_USE_DEEP_GEMM 不可置 0）。

### Phase 3b：上游未合并 PR 移植与验证（2026-08-13 用户决策新增）

上游 Anemll/dspark-vllm-gx10 已停滞（最后 commit 2026-07-15 = 0.1.1/vLLM 0.25），4 个 open PR 中 **2 个与本环境相关**，纳入本次 L1 测试：

#### P-UPSTREAM-02（PR #2，L1 必做）：DSpark draft SWA cache 缺前缀修复
- **问题本质**（PR 注释原文）：prefix-cache 命中时 target 跳过计算 cached prefix → **draft 的 SWA 滑动窗口缓存缺失 → draft 退化**（接受率/质量）
- **适用性核验（已实测）**：①生产已启用 `--enable-prefix-caching`（F5 验证 78×）→ **bug 场景真实存在**；②0.26.1 容器内 `dspark_window_size` 逻辑**缺失**（scheduler.py/kv_cache_manager.py grep 为空）→ 修复未随 0.2.1 镜像提供
- **修复逻辑**（PR #2）：scheduler 从 draft HF config 读 `sliding_window` → `dspark_window_size` → 传入 KVCacheManager → `get_computed_blocks` 强制重算最后窗口 tokens
- **移植方式**：PR #2 为 0.25 overlay 整体替换（+2691 行），**不能直接 apply 0.26.1** → 提取逻辑增量按 0.26.1 结构适配（scheduler.py 已有 use_dspark/lookahead 分支可挂接；KVCacheManager 传参）
- **验证**：移植后 prefix 命中（≥32K 共享前缀）× draft 接受率/生成质量 vs 无前缀基线——若移植受阻，先做**行为探针**（不依赖 patch：构造共享前缀请求，观察 draft 接受率骤降/输出异常）暴露问题
- **状态**：PENDING（L1 测试环境移植）

#### P-UPSTREAM-01（PR #6，L2 可选）：kv_offload /dev/shm mmap 泄漏修复
- 修复对象：`overlay/vllm/v1/kv_offload/cpu/shared_offload_region.py`（mmap 文件 unlink）
- **适用性核验（已实测）**：生产**未启用 kv_offload**（docker args 无 offload 参数）→ 当前场景不触发；F6C 的 /dev/shm 残留是 IPC 信号量（psm_*/sem.mp-*），机制不同，**非同一问题**
- **处置**：降级 L2 可选——若未来启用 KV offload（NVFP4 大模型场景评估）再移植；L1 测试期**顺带监控 /dev/shm 使用量**（防类似泄漏）
- **状态**：DEFER（L2 评估）

（两个 PR 的 patch 已存 `delivery/nvfp4-investigation/pr2.full.diff` / `pr6.full.diff`；PR #2 核心逻辑已提取）

### Phase 4：生产灰度与固化 —— ✅ 可执行，参数固化清单更新

1. A/B 全过 → 生产窗口切换（head-first 纪律，既有流程）
2. 参数固化（修订后，含动态K 调研裁定 2026-08-13 18:56）：
   - `--kernel-config '{"moe_backend":"marlin"}'` 或 flashinfer_cutlass（按 A/B 结果）
   - **动态K `[5,6,3]` 保持不变**（Archi 裁定：上游 issue #50012 实证表内 3 无乱码、V1 runner 下 DSpark 恒按顶层 N=5 全块生成、block_size 校验只查顶层；生产 8/12 起稳定）——**不再修为 [5,6,5]**
   - **⚠️ `draft_sample_method` probabilistic → greedy（NVFP4 切换必须）**：Rarri 实测 NVFP4 路径 probabilistic 草稿乱码、greedy 干净——**优先级高于 K=3 问题**
   - KV cache 维持 nvfp4_ds_mla；--load-format 评估（Rarri 验证过 instanttensor，生产当前格式保留）

**动态K 4 条护栏（Archi，入 NVFP4 切换 checklist）**：
1. 禁止表内值下调 <3（尤其 2/0——7/13 事故与 PR #47737 K=0 ZeroDivisionError 同属小K脆弱性）
2. NVFP4 切换优先评估 draft_sample_method（probabilistic→greedy，见上）
3. 若启用 V2 runner（VLLM_USE_V2_MODEL_RUNNER=1），K 直传 propose（gpu_model_runner.py:5061/5079/5213），须对 K=3 档重跑 garble 探针
4. 监控 batch 5-6 档（K=3 生效档）接受率/吞吐，异常则回退 K=4/5 或固定 5

（动态K 调研全文：`_fix_20260813/dynk-research-20260813.md`；L1 环境准备：`_fix_20260813/l1-env-prep-20260813.md`）
3. 回填 Runbook + 回滚锚点（新增 NVFP4 权重回滚 = .local-backup 156G 兜底 + cu130 镜像 tag 秒级回退）
4. 补丁归档 backup/tp4-cu132-<date>/（含下载校验报告、A/B 数据、L1/L2 决策记录）

### Phase 5：监控与观察（14-30 天）—— ✅ 方案不变，补监控项

1. prefill 吞吐 / decode 劣化 / 投机接受率 / 显存水位（Prometheus+Grafana 已有，追加 KV cache 与 MoE 后端指标）
2. **新增**：decode TPOT 三日连续 >5% 下探 → 回滚评审（复用 Issue#22 时期的监控条件机制）
3. 回滚锚点：cu130 镜像 + MXFP4 权重（分钟级）

---

## 四、风险表更新（方案 §5 + 今日新证据）

| 风险 | 等级 | 缓解（修订后） |
|------|------|---------------|
| 自建 cu132 sm_121 内核成熟度不足 | 高 | **L1 先行**（cu130 镜像零构建出收益）；L2 在 03/04 canary 验证，失败退回 cu130 |
| ring-only 补丁与 CUDA 13.2 不兼容 | 高 | 选项 A→B 两级；源码已归档；流程现成 |
| decode 换 Marlin 劣化 | 中 | A/B 硬门槛 decode≥100；劣化>5% 回滚；**marlin 在 cu130 已有社区实测 +16-20%（avarok/论坛）** |
| DSpark 在 NVFP4 下异常 | 中 | **Rarri 权重 MTP 全转（接受率 49.4% 背书）**；A/B 必测接受率 ≥0.4 |
| cuBLASLt 3× 未兑现 | 中 | L1/L2 分层决策：<1.2× 则评估仅换 Marlin 或整体回退 |
| **cuBLAS 13.4.0 NVFP4 scaling bug** | 中 | **自建镜像用 13.2.2/13.4.1+**（新增） |
| **hf-mirror 失效/下载通道依赖** | 中 | hf.rimuru.work 已验证；权重下载后本地有 .local-backup 兜底（新增） |
| **shim v8 源码缺失** | 低 | L1 不需要重编 shim（镜像不变）；L2 若需重编从 v3 源重建（新增） |
| 下载中断/校验失败 | 低 | aria2 断点续传 + 3 轮补漏 + sha256 抽样；失败可重下（新增） |

---

## 五、执行序列（下载完成后）

```
T+0   （当前）权重修复下载中（00025/00048 进行中，4 个异常文件修复）
T+1   终验：75/75 大小一致 + index.json 可解析 + weight_map 138,365 键
T+2   分发：02 → rsync 01 → 01 NFS 导出 → 03/04 挂载验证（02→04 已通）
T+3   L1 测试：l1test 镜像（已就绪）+ NVFP4 权重 + marlin 后端 + draft_sample_method=greedy → 冒烟（/health、接受率、garble 探针）
T+3b  P-UPSTREAM-02 移植（PR #2 draft SWA prefix）：0.26.1 适配 dspark_window_size → prefix 命中 × draft 探针（暴露/验证 bug）；/dev/shm 使用量监控
T+4   L1 A/B：131072/32768 × c1/c5 矩阵 + DSpark 接受率 + 精度抽样（GSM8K）【独立测试窗口，canary 显存不足】
T+5   决策：L1 达标（≥1.2×）→ 生产灰度（维护窗口，head-first 纪律）+ draft greedy 固化 + P-UPSTREAM-02 随行
T+6   L2（可选）：cu132 镜像自建（13.2.2+）→ canary A/B → P-UPSTREAM-01（kv_offload）随 L2 评估
T+7   归档：Runbook 回填 + 回滚锚点 + backup/tp4-cu132-<date>/ + 上游 PR 跟踪（2 个 open PR 状态更新）
```

**待办移交**：团队三成员（Docu/Cody/Rex）此前派发任务无响应（环境限制），本分析由主理人直接编制；原行动清单（F5 文档落地、纪律入脚本、shim 归档核实）在下载等待期可继续以精简任务派发。

---
**证据来源**：四机实时 SSH 核查（驱动/torch/cuBLASLt/镜像/参数/MD5/磁盘/内存/.local-backup）+ 今日调查（hf.rimuru.work 通道验证、Rarri/auroter/MJPansa/tomsarihan 模型卡、CUDA 13.2 release notes、Marlin/avarok 社区实测）
