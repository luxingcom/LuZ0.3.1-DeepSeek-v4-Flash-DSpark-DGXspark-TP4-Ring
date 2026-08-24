# SGLang × DeepSeek-V4-Flash NVFP4 × TP4 环网 测试环境建立方案（综合报告）

**日期**：2026-08-13
**工作流**：工作流 2（系统设计）+ 工作流 4（部署前检查）综合
**参与成员**：Archi（系统架构师）/ Rex（SRE 工程师）/ Tessa（测试专家）
**主理人**：甄宇航（Zhen）· 工程督导
**任务**：基于现有 TP4 环网条件，创建 SGLang 测试环境，满足最新下载的 NVFP4 权重（DeepSeek-V4-Flash-0731）运行需求

---

## 📌 TL;DR（执行摘要）

- **SGLang 方案成立**：PR #25820（NVFP4 MoE for DSV4）已于 2026-06-22 合入主线，随 v0.5.14 发布——**不再需要打补丁**，主线原生支持 DSV4-Flash NVFP4 权重（自动识别 hf_quant_config.json）。
- **⚠️ 重要纠偏**：此前调研引用的 **NGC 26.02 容器内置 SGLang 0.5.8，早于 NVFP4 支持（0.5.14），不能直接用**；须改用 NGC 26.07-py3 或上游 `lmsysorg/sglang:v0.5.16`。
- **🔴 最大约束**：UMA 内存互斥——SGLang NVFP4 TP4（约 45~110GiB/rank）与生产 vLLM TP4（~79.5GiB/rank）**无法同机并存**，验证期必须 **A/B 互斥切换**（停 vLLM → 跑 SGLang → 切回），而非并存服务。
- **✅ 关键利好**：**NVFP4 权重已四机就绪**（164-165G/机，48 shards，modelopt 转换产物，软链已建）——无需再下载 180GB；验证重点转为权重产物可信度（MTP 策略/bit-exact）+ SGLang 加载冒烟。
- **推荐组合**：SGLang v0.5.16 + FlashInfer ≥0.6.15.post1 + CUDA 13.2（容器跟随 26.07 的 13.3.1）+ NCCL 2.30.7 ring-only（host 挂载沿用）；端口定案 API=8010 / metrics=8011 / TCPStore=26000。
- 严重度分布：🔴严重 2 项（内存互斥、NGC 26.02 误用风险）/ 🟠高 4 项 / 🟡中 5 项 / 🟢低 2 项；**无阻塞项**（权重/磁盘/NCCL 补丁/仓库均就绪）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（条件：①容器版本验证 ②内存互斥切换窗口 ③权重产物 MTP 策略确认） |
| 阻塞项数量 | 0（无硬阻塞；2 项高危需执行纪律约束） |
| 关键行动项 | 9 条（P0×4 / P1×3 / P2×2） |
| 建议下一步 | P0 四步：拉取并验证 NGC 26.07 容器内部版本 → 权重 W 系列验证（MTP 策略定案）→ 四机 preflight 补核验（8010/26000）→ TP1 冒烟确认 SM121 kernel 选择，全绿后进 TP4 |
| 预计落地 | 冒烟 0.5~1 天（容器+权重就绪后）→ 全量验证 2~3 天（含互斥窗口排期） |

---

## 1. 调研核实结论（Archi 交付）

### 1.1 支持状态（结论性证据）

| 项 | 结论 | 证据 |
|---|---|---|
| PR #25820（NVFP4 MoE for DSV4） | ✅ **已合入主线**（2026-06-22，13 commits） | github.com/sgl-project/sglang/pull/25820 |
| 合入版本 | v0.5.14（2026-06-26 发布） | SGLang releases |
| 用法 | `--moe-runner-backend flashinfer_trtllm_routed`，自动识别 `hf_quant_config.json`（moe_quant_algo: NVFP4） | PR 正文 + nvidia/DeepSeek-V4-Flash-NVFP4 模型卡 |
| 性能 | NVFP4 相对 MXFP4 约 **1.4× throughput**（Blackwell） | PR #25702 + mr.technology |
| SM120 支持 | PR #24692（2026-06-01 合入）：`mxfp4_moe_sm120_triton.py` + `flash_mla_sm120_triton.py`，`is_sm120_supported()` 守卫 | ⚠️ **SM121（Spark）是否被该函数覆盖需实测** |

### 1.2 版本组合（推荐）

| 组件 | 版本 | 说明 |
|---|---|---|
| SGLang | **v0.5.16**（2026-07-25） | 含 #25820 + #24692 + DSpark 完善；**NVFP4 现在强制走 FlashInfer**（QServe/FBGEMM 路径已移除） |
| FlashInfer | **≥0.6.15.post1**（sm12x wheel） | ⚠️ 0.6.16 安装可能把 NCCL 退回 2.29.7，**每节点必须重钉 2.30.7** |
| CUDA | 13.2（主机驱动栈一致；NGC 26.07 内置 13.3.1 则跟随） | — |
| NCCL | **2.30.7 ring-only**（/opt/nccl-ringonly，LD_PRELOAD，host 挂载不打进镜像） | MD5 b7784b49…（v3 双 dev） |
| shim | libncclpin v8（线程绑定 8-9/15-19） | MD5 ce43c688… |
| 容器 | **首选 `nvcr.io/nvidia/sglang:26.07-py3`**（需验证内部 SGLang ≥0.5.14 + flashinfer ≥0.6.15）；备选自建 `lmsysorg/sglang:v0.5.16` | ⚠️ **26.02 不可用**（0.5.8 早于支持） |
| 仓库 tag | `<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark`（四机保留 registry tag，运行 tag 另打） | — |

### 1.3 Spark（SM121）已知风险（Archi）

1. **无 GPUDirect RDMA**：跨机 NCCL 走 CPU staging——这正是 ring-only 补丁存在的根本原因，方向正确，沿用现有补丁栈
2. **NCCL 版本陷阱**：默认 2.28.9 报 "No available shared memory broadcast block"；验证用 `/proc/self/maps | grep libnccl`（**不能信 `torch.cuda.nccl.version()`**，读的是编译期宏）
3. **`is_sm120_supported()` 是否覆盖 SM121**：需启动日志实测；必要时显式 flag
4. **DeepGEMM SM100-only**：必须 `SGLANG_DISABLE_DEEP_GEMM=1` / `SGLANG_ENABLE_DEEP_GEMM=0`
5. **DeepEP 强依赖 NVLink/RDMA/NVSHMEM**：环网 EP 风险高 → **TP4 首选，EP 不建议首期**（有 4×Spark vLLM TP4 先例：prefill ~2500 / decode ~90 t/s）
6. `flashinfer_trtllm_routed` 的 TRTLLM kernel 在 SM121 兼容性待实测；备选降级链：`flashinfer`（CUTLASS fp4_gemm）→ `marlin`

---

## 2. 部署架构定案（Zhen 裁决合并 Archi+Rex）

### 2.1 拓扑与切分

```
管理网 <NODE_IP>~<NODE_IP>（2.5GbE：SSH/API/TCPStore 控制面）
[01 rank0] ══<RING_SUBNET>══ [02 rank1]     环序 01→02→04→03→01
   ║ <RING_SUBNET> (module0)      ║ <NODE_IP>/30+<NODE_IP>/30 (module0)
[03 rank3] ══<RING_SUBNET>══ [04 rank2]
RoCE：A=<RING_SUBNET>、B=<RING_SUBNET>；MTU9000；DSCP46→P5；NCCL RING（对角 2 跳禁用）
```

- **TP4 定案**（每节点 1 rank，NODE_RANK 对齐环序 01=0/02=1/04=2/03=3）；EP 仅作 P2 后续实测
- **DSpark 投机**：`--speculative-algorithm DSPARK` + `SGLANG_RAGGED_VERIFY_MODE=compact`
- 容器：`sglang-nvfp4-tp4-{0,1,2,3}`，`--restart no --network host --ipc=host --privileged --gpus all --shm-size 64g`，无 docker 内存硬限制

### 2.2 端口定案（Zhen 裁决）

| 项 | 定案 | 备选 | 说明 |
|---|---|---|---|
| API | **8010**（head 01） | 8004（Rex 实测空闲） | 8003 被生产 responses-gateway 占用；8010 与生产 8001 语义区分清晰 |
| metrics | **8011** | 8100 | 避免与 Prometheus 8191/dcgm 9400 冲突 |
| TCPStore | **26000** | 27000 | 25999 被 vLLM 占；26000 与 25999 相邻便于记忆 |

> ⚠️ **执行前补核验**：Rex 实测的是 8004/27000；8010/8011/26000 需在四机 preflight 补 `ss -tln` 核验（预计空闲，若有冲突按备选切换）。

### 2.3 权重路线定案（重大事实纠偏）

**SRE 实测：NVFP4 权重已四机就绪，无需再下载！**

| 节点 | 路径 | 大小 | 内容 |
|---|---|---|---|
| 01/02 | `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4` | 164-165G | 48 shards + config + `hf_quant_config.json` |
| 03/04 | `<MODELS_DIR>/deepseek-v4-flash-0731-nvfp4` | 164-165G | 同上 |
| 统一 | `<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`（软链已建） | — | 四机一致挂载路径 |

- `hf_quant_config.json`：producer=modelopt（`dsv4-nvfp4-experts-mtp-fallback`）、quant_algo=MIXED_PRECISION、专家层 NVFP4、group_size=16
- ⚠️ **"experts-mtp-fallback" 命名暗示 MTP 走 fallback（可能未转）**——与 tsarihan"全转"方案相反，**MTP 策略必须列为 W 系列验证重点**（对应 Tessa R9 接受率门槛）
- **裁决**：①四机已有权重为主路径，直接进入 W0-W8 验证（bit-exact 全量补扫 + hf_quant_config.json 完整性 + MTP 策略确认 + SGLang load 冒烟）；②MJPansa ~180GB 下载降级为可选黄金参照（仅当 W 验证不通过时再下载）

### 2.4 网络环境变量（SGLang 容器内，镜像生产值）

```bash
# NCCL 环网补丁栈（沿用 vLLM TP4 生产实测值）
export LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2"
export LD_LIBRARY_PATH="/opt/nccl-ringonly:${LD_LIBRARY_PATH}"     # 前插防 2.28.9 遮蔽
export NCCL_ALGO=RING
export NCCL_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none
export NCCL_MERGE_NICS=0
export NCCL_IB_HCA="rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1"
export NCCL_IB_PEER_HCA="<沿用生产 vLLM 按 rank 映射，v3 双 dev 轮换>"
export NCCL_IB_GID_INDEX=3          # ⚠️ 生产实测=3（非 2/4），以实测为准
export NCCL_IB_TOS=46               # DSCP46→P5 无损
export NCCL_SOCKET_IFNAME=enP7s7
export NCCL_IB_TIMEOUT=1000
export NCCL_IB_RETRY_CNT=7

# SGLang × SM12x 专用
export SGLANG_DISABLE_DEEP_GEMM=1
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_SM120_TRITON_FLASHMLA=1
export SGLANG_RAGGED_VERIFY_MODE=compact
```

### 2.5 内存互斥与执行模式（Zhen 裁决）

| 模式 | 场景 | 前提 | 资源参数 |
|---|---|---|---|
| **A/B 互斥切换（主）** | 完整 TP4 验证/性能 A/B | 停 vLLM TP4（head+worker）→ 门禁 → SGLang → 测完切回 | `--mem-fraction-static 0.90`、cpuset 1-19、NCCL 8-9/EngineCore 15-19 |
| 低配并存冒烟（可选） | 功能冒烟（R1-R4 级） | 内存门禁 ≥55G/节点（≤180s 超时中止）；接受换页延迟 | `--mem-fraction-static 0.2~0.3`、cpuset=0（仅 CPU0 完全空闲）、max-model-len 4K 起步 |

**纪律**：门禁不达标绝不强起；任何异常 `docker rm -f` 即撤（--restart no）；SGLang 测试期间 vLLM 生产日志/显存监控（H 系列检查）。

### 2.6 启动编排（head-first）

1. preflight：四机 `/proc/self/maps | grep libnccl` → 2.30.7；RoCE 对口 ping；MTU 9000；端口 8010/8011/26000 空闲
2. （互斥模式）停 vLLM → monitor 退出确认 → 清残留容器
3. 门禁：内存 ≥55G（并存模式）/ GPU 探测 ≤180s
4. head(01 rank0) → 02(rank1) → 04(rank2) → 03(rank3)；worker 等 head TCPStore :26000（120s）
5. 健康：`curl :8010/health`（期望 `{"status":"OK"}`）+ `/v1/models`；快速失败关键字：NCCL error / kernel image / CUDA error

---

## 3. SRE 核验发现（Rex 交付，只读）

### 3.1 就绪项（利好）

- ✅ **NVFP4 权重四机就绪**（164-165G/机，48 shards，软链已建）→ 无需下载
- ✅ 磁盘充足（01=2.9T/02=2.3T/03=622G/04=631G 可用）
- ✅ NCCL ring-only 2.30.7 四机存在，生产 env 全套可镜像
- ✅ 本地仓库 <NODE_IP>:5000 可达；NGC 可达（需登录）；02 已有 NGC 拉取先例
- ✅ 生产 vLLM 四容器 healthy，健康判定以 EngineCore 进程 pgrep 为准

### 3.2 发现的问题（严重度排序）

| # | 严重度 | 问题 | 处置 |
|---|---|---|---|
| 1 | 🔴 SEV2 | **内存余量不足**：可用仅 26-33GiB/节点，SGLang 与生产 vLLM 并存有 OOM/换页风险 | 互斥切换主路径 + 门禁 ≥55G（Zhen 已裁决，见 §2.5） |
| 2 | 🟠 SEV3 | **8003 被生产 responses-gateway 占用**（enabled） | 端口改 8010/8011/26000（已核验 8002/8004/8100/27000 空闲，8010 段待 preflight 补核） |
| 3 | 🟠 SEV3 | **四机无 SGLang 镜像**，本地仓库无 sglang repo | P0：02 上 `docker login nvcr.io`（需凭据）→ pull 26.07-py3 → 验证内部版本 → tag/push 分发 |
| 4 | 🟡 SEV4 | CPU 争用：仅 CPU0 完全空闲 | 并存模式 cpuset=0；互斥模式 1-19 |
| 5 | 🟡 SEV4 | 生产 /health 返回空体（非 HTTP 健康） | SGLang 健康检查用 HTTP + pgrep 双轨 |

---

## 4. 验证策略摘要（Tessa 交付，详见测试计划全文）

四层 39 用例：**W0-W8 权重 / R1-R12 运行时 / C0-C8 环网 / P0-P8 性能**。三条红线：

1. 跨组对比只用 per-request p50×并发，**禁用 agg_***（asyncio prefill 串行）
2. NVFP4 输出必须过 **sanity check 硬门槛**（vLLM CUTLASS SM120 垃圾输出教训）
3. SGLang TP4 与生产 vLLM TP4 **不可同机并存**，须互斥窗口（与 §2.5 一致）

**关键判据**：W4 全量 bit-exact 补扫（verify 仅采样 64/35,328 专家，不足）必做；MTP/DSPARK 接受率 **≥0.40、<0.20 即 FAIL**（tsarihan 0.121 崩溃阈值）；L4 判定 prefill ≥1.1×（目标）、decode ≥0.95× 硬门槛、TTFT ≤1.1×。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 02 拉取 `nvcr.io/nvidia/sglang:26.07-py3`（需 NGC 凭据）→ `docker exec` 验证 SGLang ≥0.5.14 / flashinfer ≥0.6.15 → tag/push 到 <NODE_IP>:5000 → 四机 pull；版本不合则改拉 `lmsysorg/sglang:v0.5.16` | Rex/Archi | P0 | 容器就绪后当日 |
| 2 | 权重 W 系列验证：hf_quant_config.json 完整性 + **MTP 策略定案**（experts-mtp-fallback 是否可被 SGLang 接受）+ W4 全量 bit-exact 补扫 + 四机 sha256 一致 | Tessa（+Rex 执行） | P0 | 1 天 |
| 3 | 四机 preflight 补核验：8010/8011/26000 空闲、`/proc/self/maps` NCCL 2.30.7、RoCE 对口、MTU、GID=3 | Rex | P0 | 半日 |
| 4 | **TP1 冒烟（先单机）**：确认 `is_sm120_supported()` 覆盖 SM121、NVFP4 MoE kernel 选择（flashinfer_trtllm_routed vs 降级链）、首 token 生成——全绿才进 TP4 | Tessa/Rex | P0 | 半日~1 天 |
| 5 | 编写 `start_sglang_tp4_cluster.sh`（head-first、门禁、互斥守卫、health、参数化镜像 tag；只新建不改生产脚本） | Archi/Rex | P1 | TP1 冒烟后 |
| 6 | TP4 环网启动 → R1-R4 冒烟 → C0-C8 环网验证（NCCL banner、PEER_HCA、DeepEP 降级确认、shim 绑核 PSR） | Rex/Tessa | P1 | 1 天 |
| 7 | 性能 A/B（互斥窗口排期）：bench_prefill_decode_async.py 同口径，矩阵 16384/32768/65536 × conc 1/4/8/16/32 vs vLLM 基线 | Tessa/Zhen 排期 | P1 | 1~2 天 |
| 8 | 备选 MoE backend 降级测试（flashinfer/marlin）；EP 可行性实测（仅在有收益信号时） | Archi | P2 | 后续 |
| 9 | 稳定后建 systemd 单元 + 文档同步 01/02 <INSTALL_DIR>/docs/ + Runbook 回填 + 回滚锚点登记 | Docu/Rex | P2 | L4 完成后 |

---

## ⚠️ 待完善 / 已知局限

1. **容器内部版本未实证**：26.07-py3 的 SGLang/flashinfer 版本为公开信息推断，须 `docker exec` 实测；若内部版本不合需走自建路径（1-2 轮构建）
2. **8010/8011/26000 端口未经四机实测**（Rex 实测的是 8004/27000），preflight 必须补核
3. **四机现有 NVFP4 权重的 MTP 策略存疑**（"experts-mtp-fallback"）——若 SGLang load 冒烟失败，备选：①确认/重转 MTP ②下载 MJPansa 版
4. **SGLang DSV4-Flash 未获官方 Spark 验证**（官方矩阵在 DGX Station 页），SM121 kernel 兼容性只能实测
5. **性能基准对齐限制**：vLLM 基线 conc 上限 6（max-num-seqs），与 SGLang conc=32 矩阵仅 1/4 两档严格可比
6. **NGC 凭据缺失**：拉取官方容器需要 NGC API Key，需用户提供或走自建
7. SGLang 的 draft 接受率度量机制待确认（日志解析/受控探针三选一）

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）原始产出**：`deliverables/engineering-assurance/sglang-nvfp4-arch-design-2026-08-13.md`（调研证据链接：PR #25820/#24692、SGLang releases、NGC catalog、NV 论坛 4×Spark 基准、DeepEP 依赖声明等 12 条）
- **Rex（SRE）原始产出**：`deliverables/engineering-assurance/sglang-nvfp4-sre-checklist-2026-08-13.md`（四机只读核验逐项 + 40+ 检查项 + 回滚方案 + 行动清单）
- **Tessa（测试专家）原始产出**：`deliverables/engineering-assurance/sglang-nvfp4-test-plan-2026-08-13.md`（39 用例 + W4 全量补扫脚本骨架 + R5 sanity check 启发式）
- 前置调研：`research-nvfp4-alternative-runtimes-2026-08-13.md`（SGLang 第一候选）、`delivery/nvfp4-investigation/hf-nvfp4-mirror-survey-2026-08-13.md`（权重转换路线）
- 生产基线：`tp4-service-deployment-guide-2026-08-13.md`（vLLM TP4 部署指南）、`rollback-anchors-2026-08-12.md`
- 基准口径：`TP4_prefill_bench_plan.md`、`bench_prefill_decode_async.py`、tp4-r8/r12 报告

> 本报告由工程保障团队 AI 协作生成，关键决策（端口定案、内存互斥策略、权重主路径）请由人类工程负责人复核后执行。
