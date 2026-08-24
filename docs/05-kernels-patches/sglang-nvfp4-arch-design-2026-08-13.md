# SGLang × DeepSeek-V4-Flash NVFP4 × TP4 环网 部署架构设计与调研核实报告

**日期**：2026-08-13
**作者**：阿奇（Archi）· 系统架构师
**状态**：调研核实完成 + 部署架构设计（Draft，待 SRE/测试审核后 Accepted）
**前置文档**：`research-nvfp4-alternative-runtimes-2026-08-13.md`（运行时选型）、`tp4-service-deployment-guide-2026-08-13.md`（vLLM TP4 生产基线）、`rollback-anchors-2026-08-12.md`

---

## TL;DR

- **PR #25820 已合入主线（2026-06-22），随 SGLang v0.5.14（2026-06-26）发布**——NVIDIA 官方 DeepSeek-V4-Flash-NVFP4 的 SGLang 支持前提已满足，不再需要打补丁。
- **推荐版本组合：SGLang v0.5.16 + FlashInfer ≥0.6.15.post1 + CUDA 13.2 + NCCL 2.30.7（ring-only）**，容器优先验证 `nvcr.io/nvidia/sglang:26.07-py3`（内含 DGX Spark 支持），内部 SGLang 需 ≥0.5.14；否则回退上游 `lmsysorg/sglang:v0.5.16` 自建。
- **⚠️ 重要纠偏**：NGC 26.02 容器（SGLang 0.5.8）**早于** DSV4 NVFP4 支持（0.5.14），"26.02 明示支持 DGX Spark NVFP4"仅覆盖 DeepSeek-R1/Llama 等早期模型，**不能直接用于 DSV4-Flash-NVFP4**。
- **TP4 首选，EP 不建议首期**：四机纯环网 + DGX Spark 无 GPUDirect RDMA（NCCL 跨机走 CPU staging）+ DeepEP 强依赖 NVLink/RDMA/NVSHMEM，EP 的 all-to-all 在环网上通信放大；TP4 有社区 4 机 Spark 先例（vLLM：prefill ~2500 t/s、decode ~90 t/s）。
- **关键约束：UMA 内存互斥**。SGLang NVFP4 TP4 单 rank 约需 ~110GB（mem-fraction 0.90），与生产 vLLM TP4（~79GB/rank）**无法同机并存**——SGLang 验证期作为与 vLLM TP4 的 **A/B 互斥切换轨道**，而非常驻并存服务。
- **端口规划**：SGLang API **8010**（8003 已被 aicad 应用栈占用）、metrics **8011**、TCPStore **26000**（25999 被 vLLM TP4 占用）。
- **NVFP4 权重**：优先下载 `MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（~180GB，source-preserving，HF 走 127.0.0.1:7890）→ 01 校验 → rsync 四机本地；本地 tsarihan 转换（input_scale=1.0）作备选，二者均需 conversion-receipt + SGLang load 冒烟验证（scale 字段/padding 对齐风险）。

---

## 1. 任务 A：调研核实结论

### 1.1 PR #25820 当前合入状态 ✅ 已合入

| 项 | 结论 | 证据 |
|---|---|---|
| 标题 | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 | https://github.com/sgl-project/sglang/pull/25820 |
| 状态 | **Merged into main**，2026-06-22，13 commits，+385/-17 | 同上 |
| 合入版本 | 随 **v0.5.14**（2026-06-26 发布，"DeepSeek-V4 NVFP4 MoE 量化" 列入 release notes） | LinkedIn/SGLang v0.5.14 发布 |
| 用法 | `--moe-runner-backend flashinfer_trtllm_routed`；从 `hf_quant_config.json` 自动检测 NVFP4（权重以 FP8 存储，`moe_quant_algo: NVFP4`） | PR 正文 + 模型卡 |
| 性能依赖 | 依赖 #25702（perf），NVFP4 相对 MXFP4 约 **1.4× throughput**（Blackwell） | PR + mr.technology |

**结论**：主线已有原生支持，**无需再打 #25820 补丁**。任何 SGLang ≥0.5.14 的发布版（含 NGC 容器）即可原生加载 `nvidia/DeepSeek-V4-Flash-NVFP4` 布局。

### 1.2 SGLang 最新稳定版 + DSV4 支持声明

- **最新稳定版：v0.5.16（2026-07-25 发布）**；前序 v0.5.14 / v0.5.15(.post1)。
- **v0.5.16 关键 breaking changes（必须知晓）**：
  - **NVFP4 现在要求 FlashInfer**（移除实验性 QServe / FBGEMM FP8 路径，`--fp4-gemm-backend cutlass` 移除）。
  - flag 改名：`--enable-deepep-waterfill` → `--enable-waterfill`；`--optimistic-prefill-retries` → `--optimistic-prefill-attempts`。
  - UnifiedRadixTree 成为 SWA/Mamba/DSA 默认；DSpark 完善（`--speculative-algorithm DSPARK` + `SGLANG_RAGGED_VERIFY_MODE=compact`）。
- **DSV4 支持声明**：
  - 官方模型卡 `nvidia/DeepSeek-V4-Flash-NVFP4`（2026-05-28，ModelOpt 0.44.0）：SGLang 需 PR #25820，命令 `python3 -m sglang.launch_server --model ... --tensor-parallel-size 8`（TP8 参考）。
  - SGLang DSV4 cookbook：DeepSeek-V4-Flash-0731 验证矩阵 = 8×B200 / 4×GB300 / 4×H200（**未验证 GB10/DGX Spark**）。
  - NVIDIA Spark SGLang playbook（build.nvidia.com/spark/sglang，2026-07-31 更新）：Spark 已验证矩阵含多个 NVFP4 模型（`--quantization modelopt_fp4`），**但 DeepSeek-V4-Flash 列在 DGX Station 页而非 Spark 页** → 官方未在 Spark 验证 DSV4-Flash，需自行实测。
- **证据**：SGLang GitHub releases、模型卡、docs.sglang.io/cookbook/DeepSeek/DeepSeek-V4、build.nvidia.com/spark/sglang。

### 1.3 NGC SGLang 容器最新 tag

| 容器 tag | 内置（据公开信息） | 备注 |
|---|---|---|
| **26.07-py3（Latest）** | CUDA 13.3.1；SGLang 0.5.x 上游构建；支持 B300/GB300/RTX PRO 6000/**DGX Spark**/Jetson Thor | 首选验证对象；需 `docker exec` 确认内部 SGLang ≥0.5.14 且含 sm12x kernel |
| 26.06 / 26.05 / 26.04 / 26.03(.post1) | 月度滚动 | 26.04 release notes 存在（2026-04-29 更新） |
| 26.02 | CUDA 13.1.1 / SGLang 0.5.8 / flashinfer 0.6.1 / sgl-kernel 0.3.21 | **仅支持 NVFP4 × DeepSeek-R1/Llama，不含 DSV4 NVFP4** |

- NGC SGLang 容器按月 tag（26.02~26.07 均有），release notes 文档于 2026-04-29 更新到 26.04，catalog 上 26.06/26.07 已存在。
- **结论**：26.02 不满足 DSV4 NVFP4；必须用 **26.06/26.07 或上游 0.5.16**。

### 1.4 Spark（SM121）运行 DSV4 已知问题

1. **GPUDirect RDMA 在 DGX Spark 不受支持**（NVIDIA CUDA 移植指南）：ConnectX-7 + RoCE 不能直接访问普通 CUDA allocation，跨机 NCCL 需经 CPU-visible memory staging → 这是"stock NCCL 纯环无解、ring-only 补丁 + NET_PLUGIN=none"的根本原因，方向正确。
2. **NCCL 版本陷阱**：Spark 默认容器带 NCCL 2.28.9，会报 "No available shared memory broadcast block"；需 **2.30.4+（社区验证 2.30.7）**。验证用 `/proc/self/maps | grep libnccl` 或 `ncclGetVersion()`，**不能信 torch.cuda.nccl.version()**（读的是编译期宏）。
3. **SM120 vs SM121 检测**：SGLang PR #24692 提供 SM120（compute 12.0）Triton fallback（`mxfp4_moe_sm120_triton.py`、`flash_mla_sm120_triton.py`、`sm120_mqa_triton.py`），由 `is_sm120_supported()` 守卫；DGX Spark 是 sm_121（12.1a）。社区说明 CUDA kernel 按 sm12x 家族编译、多数库覆盖，**但 SGLang 的 is_sm120_supported() 是否匹配 12.1 需实测确认**。
4. **DeepGEMM 不可用**：SM100/SM103 专用（tcgen05/TMEM），SM12x 必须 `SGLANG_DISABLE_DEEP_GEMM=1` / `SGLANG_ENABLE_DEEP_GEMM=0`。
5. **DeepEP 强依赖 NVLink + RDMA + NVSHMEM**：无 NVLink + 无 GPUDirect RDMA → internode/low-latency 能力被禁用，回退 NCCL；SGLang DSV4 cookbook 默认 DeepEP A2A 在四机环网上**风险高**。
6. **社区已验证软件栈（两 Spark 0731）**：Torch 2.13.0 / Triton 3.7.1 / FlashInfer 0.6.15.post1 / NCCL 2.30.7；FlashInfer 0.6.16 维护者警告安装可能把 NCCL 退回 2.29.7，**必须每节点重钉 2.30.7**。
7. **4 机 TP4 先例（vLLM）**：NV 论坛 4×Spark DSV4-Flash-0731 DSpark：prefill ~2500 t/s、decode ~90 t/s；使用 QRS812 fabric、RoCE GID 11、NVFP4 DS-MLA KV——证明 TP4 四机可行，但为 vLLM 栈，SGLang 需自己复测。

### 1.5 官方容器不合适时，自建成本与风险

- **需要的额外 patch**：主线 0.5.16 已含 #24692（SM120）+ #25820（NVFP4），**理论上不需要代码 patch**；自建主要是"重装验证"而非"打补丁"。
- **构建链**：lmsysorg/sglang:v0.5.16（aarch64）为基础 → 确认/替换 flashinfer ≥0.6.15.post1（sm12x wheel）→ sgl-kernel sm12x → NCCL 2.30.7 ring-only（host 挂载即可，不需要打进镜像）→ libncclpin shim（host 挂载）。
- **成本**：镜像 ~20-35GB 传输、构建 1-2 轮、JIT 首次加载需 flashinfer 本地编译（Triton 3.6.0+ 修复 sm12x）。
- **风险**：CUDA 13.x 与 PyTorch/Triton 组合漂移、JIT 缓存缺失导致启动变慢、维护负担。
- **建议**：先验 NGC 26.07；若内部 SGLang/flashinfer 版本不合或 kernel 缺，再自建 0.5.16。

### 1.6 NVFP4 权重兼容性（MJPansa vs 本地 tsarihan）

- **NVFP4 布局**：主权重 E2M1 4-bit + FP8 E4M3 块缩放（16 元素块）；HF 仓库以 FP8 E4M3 safetensors 存储 + `hf_quant_config.json`（`moe_quant_algo: NVFP4`）描述；SGLang 靠它自动识别（PR #25820）。
- **MJPansa/DeepSeek-V4-Flash-0731-NVFP4**：source-preserving 0731 转换，tensor 类型含 F8_E4M3，304B params，~180GB；两节点 vLLM 已验证；是 **首选下载源**。
- **本地 tsarihan transcode（input_scale=1.0）**：input_scale 为激活缩放；=1.0 表示不缩放激活 → 布局可兼容但精度有损；**加载兼容的关键在 hf_quant_config.json 字段完整**（moe_quant_algo / scale_fmt / weight_scale 等）。vLLM 侧 compressed-tensors 曾有 `scale_fmt` 缺失 → KeyError 的先例，SGLang 侧同样需 load 冒烟验证。
- **padding 对齐**：SGLang NVFP4 权重需 padding（CUTLASS 32 对齐 / TRTLLM 128 对齐，`pad_nvfp4_weight`）；本地转换产物若未按目标 kernel 对齐，可能 load 失败或走慢路径 → 转换后必须跑 SGLang load + 首 token 冒烟。
- **结论**：两者对 SGLang 均为"FP8 存储 + NVFP4 元数据"的合法布局，**兼容性以 hf_quant_config.json 完整性与 load 冒烟为准**，不能仅凭生成器假设。

---

## 2. 推荐版本组合

| 组件 | 版本 | 说明 |
|---|---|---|
| SGLang | **v0.5.16**（2026-07-25） | 含 #25820（NVFP4 MoE）+ #24692（SM120 Triton fallback）+ DSpark 完善 |
| FlashInfer | **≥0.6.15.post1**（sm12x wheel） | 0.5.16 要求 NVFP4 用 FlashInfer；警惕 0.6.16 重钉 NCCL 问题 |
| CUDA | **13.2**（与主机驱动/现有 vLLM 栈一致） | NGC 26.07 内置 13.3.1，若用该容器则跟随 |
| NCCL | **2.30.7 ring-only**（/opt/nccl-ringonly，LD_PRELOAD） | 沿用现有补丁 v3（双 dev PEER_HCA），MD5 `b7784b49…` |
| shim | **libncclpin v8**（<INSTALL_DIR>/lib） | 线程绑定 8-9 / 15-19，MD5 `ce43c688…` |
| 容器 | **首选 `nvcr.io/nvidia/sglang:26.07-py3`**；备选自建 `lmsysorg/sglang:v0.5.16` | 26.07 需验证内部 SGLang ≥0.5.14 + flashinfer ≥0.6.15 |
| 仓库 tag | `<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark`（或 `26.07-py3-nvfp4`） | 四机拉取后保留 registry tag，运行 tag 另打 |
| 权重 | `MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（首选）/ 本地 tsarihan（备选） | ~180GB |

---

## 3. 部署架构（文字版拓扑）

```
┌──────────── 管理网 <NODE_IP>~189（2.5GbE：SSH / API / TCPStore 控制面）────────────┐
│                                                                                        │
│   [01 A-head rank0] ════ 10.100.136/137 ════ [02 A-worker rank1]   环网边1 module1      │
│         ║                                              ║                                 │
│    10.100.140/141 (module0)                     <NODE_IP>/30 + <NODE_IP>/30 (module0)  │
│         ║                    (TP2 遗留段)              ║                                 │
│   [03 B-head rank3] ════ 10.100.138/139 ════ [04 B-worker rank2]   环网边2 module1      │
│                                                                                        │
└──────── RoCE：A=136/137(GID2)、B=138/139(GID4)；MTU9000；DSCP46→P5；NCCL RING ──────────┘

逻辑环：01(rank0) → 02(rank1) → 04(rank2) → 03(rank3) → 01
  - TP4 数据面：NCCL all-reduce 沿环 1 跳邻居中继（对角 2 跳不可用，禁用）
  - 控制面：TCPStore 01（管理网 <NODE_IP>:26000）
  - API：head 01 :8010；metrics :8011

每节点运行时（4× 相同拓扑）：
┌──────────────────────────────────────────────────┐
│ DGX Spark GB10（sm_121，UMA 121.6GiB）            │
│  docker: sglang-nvfp4-tp4-<rank>（--restart no）  │
│   ├ SGLang 0.5.16 + FlashInfer ≥0.6.15            │
│   ├ LD_PRELOAD: /opt/libncclpin.so + /opt/nccl-ringonly/libnccl.so.2 │
│   ├ LD_LIBRARY_PATH: /opt/nccl-ringonly（前插）    │
│   ├ DeepGEMM 禁用；NVFP4 MoE → flashinfer(TRTLLM/CUTLASS)；SM12x MLA/MoE → Triton fallback │
│   └ 权重 <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4（本地，ro；NFS 兜底）│
└──────────────────────────────────────────────────┘
```

### 3.1 TP/EP 切分建议（结论：TP4，EP 不建议首期）

**推荐 TP4**（每节点 1 rank，NODE_RANK 对齐环序 01=0/02=1/04=2/03=3）：

| 维度 | TP4 | EP（备选，不建议首期） |
|---|---|---|
| 内存 | 权重 167GB÷4 ≈ 42GB/rank + KV + 激活，128GB/机宽裕（mem-fraction 0.90） | 同样可分，但 33K 专家需全量驻留/调度 |
| 通信 | all-reduce 沿环 1 跳中继，RING + 双 dev PEER_HCA 可收敛（现有 vLLM 已实证 23.86GB/s） | all-to-all（dispatch/combine）跨环大量中继，DeepEP 无 NVLink/RDMA/GPUDirect 时退化，通信放大 |
| 先例 | 4×Spark TP4（vLLM）prefill ~2500 / decode ~90 t/s 实证 | 无 Spark 环网 EP 实证 |
| 风险 | 低（复用现有 NCCL 补丁栈） | 高（DeepEP 需 NVSHMEM + NVLink 域优化失效，可能不如 TP） |

- **内存互斥硬约束**：SGLang NVFP4 TP4 单 rank ≈ 42GB 权重 + KV/激活（mem-fraction 0.90 ≈ 109GiB）；生产 vLLM TP4 单 rank ≈ 79GiB（util 0.65）。**同一 UMA 池无法双 TP4 并存** → SGLang 验证期必须 stop vLLM TP4 后启动（A/B 互斥切换），这是与"端口隔离"同级的**编排级隔离**。
- **DSpark**：NVFP4 0731 自带 DSpark draft head，SGLang v0.5.16 用 `--speculative-algorithm DSPARK` + `SGLANG_RAGGED_VERIFY_MODE=compact`；可选 `--speculative-dspark-block-size`。
- **EP 后续**：如需，实测 `--moe-a2a-backend deepep`（NCCL fallback）与 NIXL/Mooncake，在环网上建立基线再定。

### 3.2 容器与镜像

1. **拉取/验证顺序**：
   - 02 上 `docker pull nvcr.io/nvidia/sglang:26.07-py3` → `docker exec` 验证 `python3 -c "import sglang, flashinfer; print(sglang.__version__, flashinfer.__version__)"`，SGLang 须 **≥0.5.14**、flashinfer **≥0.6.15**。
   - 若版本不合或 sm12x kernel 缺失 → 自建：`lmsysorg/sglang:v0.5.16`（aarch64）为基础，重装 flashinfer sm12x wheel。
2. **推送/分发**：`docker tag` → `<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark` → push → 四机 pull → 本地保留 registry tag + 运行 tag 另打（`sglang-nvfp4:0.5.16`）。
3. **NCCL 补丁不打包进镜像**：沿用 host 挂载 `/opt/nccl-ringonly`、`<INSTALL_DIR>/lib/libncclpin.so` → 容器内 `/opt/nccl-ringonly`、`/opt/libncclpin.so`（与 vLLM 同一套，保证四机一致性与回滚锚点）。

### 3.3 权重准备

1. **首选**：01 经 HF（代理 127.0.0.1:7890）下载 `MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（~180GB）到 `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4`。
2. **备选**：本地 tsarihan transcode（input_scale=1.0）产物落 `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4-local`。
3. **分发**：01 → 02 rsync（RoCE/管理网）；01 → 03（RoCE <NODE_IP>）、02 → 04（RoCE <NODE_IP>）——沿用现有 NFS 对口，或 rsync 本地化（03/04 磁盘 916G 足够）。
4. **验证（conversion-receipt）**：
   - `sha256sum -c manifest.sha256`（四机一致）；
   - `hf_quant_config.json` 含 `"moe_quant_algo":"NVFP4"`、scale/scale_fmt 字段齐全；
   - SGLang load 冒烟：`/v1/models` 返回 + 首 token 生成 + 与 MXFP4 0731 同一 prompt 抽样对比；
   - 记录 receipt：来源/日期/sha256/转换脚本版本/抽查结果（存 `<INSTALL_DIR>/docs/`）。

### 3.4 网络环境变量清单（SGLang 容器内）

```bash
# ── NCCL 环网补丁栈（沿用 vLLM TP4 生产值）──
export LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2"   # shim v8 + ring-only 2.30.7
export LD_LIBRARY_PATH="/opt/nccl-ringonly:${LD_LIBRARY_PATH}"           # 前插防系统 2.28.9 遮蔽
export NCCL_ALGO=RING
export NCCL_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none
export NCCL_MERGE_NICS=0
export NCCL_IB_PEER_HCA=<沿用 vLLM TP4 四机对口映射，v3 双 dev 轮换>      # 生成后四机一致，逐对核对
export NCCL_IB_GID_INDEX=2                                               # RoCE GID（A=2；B 侧按实测 4 或统一 2，以 preflight 为准）
export NCCL_IB_TOS=46                                                    # DSCP46→P5 无损
export NCCL_IB_TIMEOUT=22
# 注：NCCL_DEBUG_FILE 落容器内 ~/.sglang-logs/nccl-*.log（不污染 stdout）

# ── SGLang × SM12x 专用 ──
export SGLANG_DISABLE_DEEP_GEMM=1        # DeepGEMM SM100-only，SM121 必须关
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_SM120_TRITON_FLASHMLA=1    # SM120/121 MLA Triton fallback（PR #24692 默认开）
export SGLANG_SM120_MQA_FALLBACK=0       # 默认关（走 Triton 快路径）
export SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16   # 可选：压缩态池 BF16 降内存
export SGLANG_RAGGED_VERIFY_MODE=compact # DSpark（v0.5.16）
# 仅当启用 EP 时才需：SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK（> max_running×MTP_draft）
# 控制面网卡（若默认选错）：GLOO_SOCKET_IFNAME=<管理网 ifname，与现有 vLLM 脚本一致>
```

### 3.5 端口与并存隔离

| 项 | 值 | 说明 |
|---|---|---|
| SGLang API | **8010**（head 01） | **8003 已被 aicad 应用栈占用**，避开 |
| SGLang metrics | **8011**（head 01，Prometheus 可选接入） | 避免与 Prometheus 8191/dcgm 9400 冲突 |
| TCPStore | **26000** | **25999 已被 vLLM TP4 占用** |
| 容器名 | `sglang-nvfp4-tp4-{0,1,2,3}` | 与 vLLM `vllm-tp4-*` 区分 |
| 资源 | `--cpuset-cpus 1-19 --shm-size 64g --restart no` | 与生产一致；**不加** docker 内存硬限制 |
| 互斥 | stop vLLM TP4（head+worker）→ GPU 门禁 → 启动 SGLang | 同 UMA 池无法并存 |
| systemd | 验证期脚本先行；稳定后再建 `sglang-nvfp4-tp4-{head,worker}` 单元 | 复用 monitor/门禁/退避模式 |

### 3.6 启动编排（head-first，复用 start_tp4_cluster.sh 模式）

1. 预检：四机 `/proc/self/maps | grep libnccl` 应指向 2.30.7；RoCE 对口可达（ping 10.100.x）；MTU 9000；GID 正确。
2. `stop vllm-tp4-{head,worker}`（互斥）→ 确认 monitor 退出 → `docker rm -f` 残留容器。
3. GPU 门禁：`nvidia-smi` 探测 ≤180s 且确认无 vLLM TP4 存活（防抢占）。
4. head-first：01（rank0）→ 02（rank1）→ 04（rank2）→ 03（rank3）；worker 等 head TCPStore :26000（120s），head 就绪后 60s 缺秩 exit(1) 全链重建。
5. 健康检查：`curl -s http://<NODE_IP>:8010/health` → 200；`/v1/models` 确认模型加载（免鉴权与否按配置）。
6. 快速失败：日志关键字（NCCL error / No available / kernel image / CUDA error）→ 立即终止并回滚。

### 3.7 风险与回滚

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | UMA 内存互斥 | 无法与 vLLM TP4 并存 | A/B 互斥切换；回滚=切回 vLLM 镜像 tag |
| R2 | 环网 2 跳中继 + 无 GPUDirect RDMA | 跨机带宽/延迟损失，prefill 不达预期 | RING + 双 dev PEER_HCA；实测基线；接受比 B200 低 |
| R3 | `is_sm120_supported()` 是否覆盖 SM121 | kernel 选择错 → 慢路径/崩溃 | 启动日志核对 kernel 选择；必要时显式 flag/升级 |
| R4 | `flashinfer_trtllm_routed` 的 TRTLLM kernel 在 SM121 兼容性 | load/首个 token 失败 | 备选 `--moe-runner-backend flashinfer`（CUTLASS）或 `marlin` |
| R5 | NGC 26.02 不含 DSV4 NVFP4 | 误用旧容器 → 不支持 | 用 26.07/上游 0.5.16；验证容器内版本 |
| R6 | NVFP4 权重布局/字段/padding 不兼容（本地转换） | load 失败或静默精度损失 | conversion-receipt + load 冒烟 + 抽样对比 |
| R7 | NCCL 2.28.9 遮蔽 2.30.7 | "No available shared memory" 崩溃 | LD_LIBRARY_PATH 前插 + `/proc/self/maps` 验证 |
| R8 | 端口/控制面冲突 | 8003、25999 已被占 | 8010/26000；preflight 检查 |
| R9 | 官方未在 Spark 验证 DSV4-Flash | 未知性能/稳定性 | 预研定位；小步验证（TP1→TP2→TP4） |

**回滚锚点**：镜像 tag（`sglang-nvfp4:0.5.16` ↔ vLLM `0.2.1-v026.0`）+ 权重目录版本化（`-nvfp4` / `-local`）+ 启动脚本 `start_sglang_tp4_cluster.sh`（镜像 tag 参数化）。任何异常 → 停 SGLang → 起 vLLM TP4（现有 systemd 自愈兜底）。

---

## 4. 启动命令草案（head 01 示例，worker 依 rank 类推）

```bash
docker run -d --name sglang-nvfp4-tp4-0 \
  --gpus all --restart no \
  --cpuset-cpus 1-19 --shm-size 64g --network host \
  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro \
  -v <INSTALL_DIR>/lib/libncclpin.so:/opt/libncclpin.so:ro \
  -v <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4:/models:ro \
  -e LD_LIBRARY_PATH=/opt/nccl-ringonly:$LD_LIBRARY_PATH \
  -e LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2" \
  -e NCCL_ALGO=RING -e NCCL_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none -e NCCL_MERGE_NICS=0 \
  -e NCCL_IB_PEER_HCA="<对口映射>" -e NCCL_IB_GID_INDEX=2 -e NCCL_IB_TOS=46 -e NCCL_IB_TIMEOUT=22 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 -e SGLANG_ENABLE_DEEP_GEMM=0 \
  -e SGLANG_SM120_TRITON_FLASHMLA=1 -e SGLANG_SM120_MQA_FALLBACK=0 \
  -e SGLANG_RAGGED_VERIFY_MODE=compact \
  <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark \
  python3 -m sglang.launch_server \
    --model-path /models --trust-remote-code \
    --tp 4 --nnodes 4 --node-rank 0 \
    --dist-init-addr <NODE_IP>:26000 \
    --host 0.0.0.0 --port 8010 \
    --moe-runner-backend flashinfer_trtllm_routed \
    --speculative-algorithm DSPARK \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 4096
# worker（02/04/03）：同命令，--node-rank 1/2/3，去掉 --port（或 --port 8010 仅 head）
# 验证：
#   curl -s http://<NODE_IP>:8010/health
#   curl -s http://<NODE_IP>:8010/v1/models
```

> 说明：`--moe-runner-backend flashinfer_trtllm_routed` 为 #25820 官方推荐（NVFP4）。若 SM121 上报 TRTLLM kernel 不支持，依次降级测试 `flashinfer`（CUTLASS fp4_gemm）→ `marlin`。

---

## 5. 行动清单

**P0（阻塞）**
- [ ] 拉取并验证 NGC `sglang:26.07-py3` 内部 SGLang/flashinfer 版本；不合则拉 `lmsysorg/sglang:v0.5.16`（aarch64）验证
- [ ] 下载 MJPansa NVFP4 权重（~180GB，127.0.0.1:7890）或本地转换 → conversion-receipt
- [ ] 四机 rsync 权重 + sha256 校验
- [ ] 验证 SM121 检测与 NVFP4 MoE kernel 选择（TP1 冒烟）

**P1（本周）**
- [ ] 编写 `start_sglang_tp4_cluster.sh`（head-first、GPU 门禁、health 检查、互斥守卫）+ 参数化镜像 tag
- [ ] 预检：NCCL `/proc/self/maps`、RoCE GID/PEER_HCA 映射、MTU 9000、端口 8010/26000 空闲
- [ ] TP4 环网启动 → /health → /v1/models → 首 token → 长 prompt 生成
- [ ] 性能 A/B（同 vLLM 口径）：prefill/decode/TTFT/DSpark acc
- [ ] 与 sre-engineer 对齐部署检查清单、与 testing-expert 对齐验收基准

**P2（后续）**
- [ ] 备选 MoE backend 降级测试（flashinfer/marlin）
- [ ] EP 可行性实测（DeepEP NCCL fallback）——仅在有明确收益信号时
- [ ] 稳定后建 systemd 单元（sglang-nvfp4-tp4-*）+ 文档同步 01/02 `<INSTALL_DIR>/docs/`

---

## 6. 证据链接

- PR #25820（NVFP4 MoE for DSV4，Merged 2026-06-22）：https://github.com/sgl-project/sglang/pull/25820
- PR #24692（SM120 DSV4 Triton fallback，Merged 2026-06-01）：https://github.com/sgl-project/sglang/pull/24692
- SGLang v0.5.14 / v0.5.16 releases：https://github.com/sgl-project/sglang/releases
- SGLang DSV4 cookbook：https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- nvidia/DeepSeek-V4-Flash-NVFP4 模型卡：https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4
- MJPansa/DeepSeek-V4-Flash-0731-NVFP4：https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4
- NVIDIA Spark SGLang playbook（2026-07-31）：https://build.nvidia.com/spark/sglang/overview
- NVIDIA SGLang NGC 容器（26.02~26.07）：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/sglang
- NVIDIA SGLang release notes：https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes
- 4-node DGX Spark DSV4-Flash-0731 DSpark benchmark（prefill 2500/decode 90 t/s）：https://forums.developer.nvidia.com/t/378878
- DGX Spark 不支持 GPUDirect RDMA / SM121 软件栈讨论：https://forums.developer.nvidia.com/t/357663
- DeepEP（NVLink/RDMA/NVSHMEM 依赖）：https://github.com/Marks101/DeepEP
- NVFP4 × MXFP4 吞吐对比（1.4×，NVFP4 pack）：https://mr.technology/payloads/deepseek-v4-flash-0731-inference-stack-caught-up-august-2026
- SGLang 26.07 容器（CUDA 13.3.1，DGX Spark 支持）：https://dreaming.press/posts/sglang-26-07-blackwell-ultra-container-what-founders-rent.html

---

> 时效性说明：2026-08-13 时点。SGLang/flashinfer/NGC 容器均在快速迭代；落地前复核：26.07 容器内实际 SGLang/flashinfer 版本、`is_sm120_supported()` 对 SM121 的匹配、`flashinfer_trtllm_routed` 在 SM121 的 kernel 支持。

---

# V2 节：SGLang 0.5.14 启动参数终稿核实 + 容器内调整方案定稿

**日期**：2026-08-14（v2 追加，容器定稿后更新）
**作者**：阿奇（Archi）· 系统架构师
**触发**：容器实测定稿 **NGC 26.07-py3 = SGLang 0.5.14+nv26.7 / torch 2.13.0a0 / flashinfer 0.6.14**（v1 草案基于 0.5.16，v2 全部以 0.5.14 为准重新核实）

---

## V2-0. 摘要：0.5.14 与 0.5.16 的差异修正（必须先读）

| # | 项目 | v1 草案（按 0.5.16） | **v2 终稿（按 0.5.14）** | 影响 |
|---|---|---|---|---|
| C1 | `--moe-runner-backend` | `flashinfer_trtllm_routed`（#25820 官方推荐） | 🔴 **SM121 上改为 `flashinfer_cutlass`**。`flashinfer_trtllm_routed` 走 TRTLLM kernel，**上游仅在 B200(SM100) 验证，SM121 不可运行（issue #26324）**；SM121 正确路径是 CUTLASS FP4 GEMM（flashinfer `fp4_gemm_cutlass_template_sm120.h`，已在本机 vLLM/flashinfer 栈实证可跑） | 最大差异修正，直接影响 TP1 冒烟预期 |
| C2 | `SGLANG_RAGGED_VERIFY_MODE=compact` | 0.5.16 DSpark 新增 | ❌ **0.5.14 不设置**。compact 为 PR #30261/#31434（0.5.16）"confidence-driven DSpark"特性；0.5.14 的 DSPARK 为早期形态，无该 env | 0.5.14 DSPARK 用默认验证模式；若 0.5.14 支持 static 再显式设置 |
| C3 | `--enable-waterfill` | 0.5.16 改名 | ✅ 0.5.14 用旧名 `--enable-deepep-waterfill`（仅 EP 才需要，本方案 TP 首期不启用） | 无实际影响（首期不启用 EP） |
| C4 | NVFP4 强制 FlashInfer | 0.5.16 移除 QServe/FBGEMM | ⚠️ 0.5.14 仍含 QServe/FBGEMM 残留路径，但 NVFP4 首选用 FlashInfer（#25820 默认） | 显式指定 backend，不依赖默认 |
| C5 | flashinfer 版本 | ≥0.6.15.post1 | ⚠️ **0.6.14（NGC 校验组合）**。0.6.14 是 2026-07-02 发布、首个系统覆盖 DGX Spark/SM12x 的版本；**NCCL 回退警告仅见于 0.6.16**，0.6.14 未见公开报告 → 标记待 sre 实测 | 通常无需处理；如遇 NCCL 问题再钉 |
| C6 | `--quantization modelopt_fp4` | 0.5.16 支持 | ⚠️ 0.5.14 是否接受 `modelopt_fp4` 取值待 `--help` 确认；**主路径靠 hf_quant_config.json 自动检测（#25820）**，不传 `--quantization` 亦可 | 待确认 |

**核心结论**：0.5.14 容器**可用**（#25820 合入线、NVFP4 原生支持、SM121 Triton MLA path #24692 含内），但必须把 **`flashinfer_trtllm_routed` → `flashinfer_cutlass`** 和 **`SGLANG_RAGGED_VERIFY_MODE=compact` 移除**这两处从 v1 草案修正过来。

---

## V2-1. 启动参数终稿表（0.5.14）

**四机一致项（head 01 + worker 02/04/03 仅 `--node-rank` 不同）**：

| # | 参数 | **0.5.14 终稿值** | 来源 | 待确认 |
|---|---|---|---|---|
| 1 | `--model-path` | `/models`（挂载 NVFP4 权重，ro） | 容器定稿 | 无 |
| 2 | `--tp` / `--nnodes` / `--node-rank` | `4 / 4 / {0,1,2,3}`（环序 01=0, 02=1, 04=2, 03=3） | 公开文档 multi-node | 无 |
| 3 | `--dist-init-addr` | `<NODE_IP>:26000`（head 01 TCPStore） | 公开文档；端口定案 | 无 |
| 4 | `--host` / `--port` | `0.0.0.0` / `8010` | 设计定案 | 无 |
| 5 | `--moe-runner-backend` | **`flashinfer_cutlass`**（首选） | 社区实证（#26324 说明 trtllm_routed SM121 不可用；CUTLASS fp4_gemm 在 flashinfer 0.6.14 有 SM120 模板 + 本机 vLLM CUTLASS 冒烟实证） | `flashinfer_cutlass` 是否为 0.5.14 合法取值（`--help` 确认）；备选 `flashinfer`（CUTLASS 通用）/ `marlin` |
| 6 | `--quantization` | **不传**（靠 hf_quant_config 自动检测）；若 `--help` 有 `modelopt_fp4` 则显式传 | PR #25820 自动检测；NVIDIA Spark playbook | `modelopt_fp4` 是否 0.5.14 合法取值 |
| 7 | `--speculative-algorithm` | `DSPARK` | 公开（DSpark 自 0.5.10+ 支持） | 0.5.14 的 DSPARK 是否需额外 `--speculative-dspark-*` 参数（`--help`） |
| 8 | `SGLANG_RAGGED_VERIFY_MODE` | **不设置**（compact 是 0.5.16） | 公开 release notes | 若 0.5.14 支持 static 可显式设 `static` |
| 9 | `--mem-fraction-static` | `0.90`（互斥完整验证）/ `0.2~0.3`（低配并存冒烟） | 设计定案 | 无 |
| 10 | `--chunked-prefill-size` | `4096` | 设计草案 | 0.5.14 默认值可能不同（`--help`） |
| 11 | `--max-model-len` | `65536`（测试矩阵上限；若内存预算不足降至 32768） | 设计 | 与 mem-fraction 匹配性 |
| 12 | `--kv-cache-dtype` | **默认**（DSV4 NVFP4 自动走 NVFP4 DS-MLA KV；vLLM 侧 `nvfp4_ds_mla` 实证） | vLLM 侧实证（同栈） | SGLang 0.5.14 对 NVFP4 KV 的自动 dtype（`--help`/日志） |
| 13 | `--enable-metrics` / `--metrics-port` | `--enable-metrics --metrics-port 8011` | 公开文档 | `--metrics-port` 是否 0.5.14 支持（`--help`）；若否用默认 metrics 端口（10200）并做端口映射 |
| 14 | `--trust-remote-code` | 是 | 公开 | 无 |

**容器内环境变量终稿（0.5.14 对齐）**：

```bash
# NCCL 环网补丁栈（沿用 vLLM TP4 生产实测值；task 已确认 GID_INDEX=3）
export LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2"   # shim v8 + ring-only 2.30.7
export LD_LIBRARY_PATH="/opt/nccl-ringonly:${LD_LIBRARY_PATH}"           # 前插防 2.28.9 遮蔽
export NCCL_ALGO=RING
export NCCL_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none
export NCCL_MERGE_NICS=0
export NCCL_IB_PEER_HCA="<沿用 vLLM TP4 四机对口映射，v3 双 dev 轮换>"   # 生成后四机一致
export NCCL_IB_GID_INDEX=3            # 生产实测=3
export NCCL_IB_TOS=46
export NCCL_IB_TIMEOUT=22             # 或沿用生产 1000，以生产脚本为准
export NCCL_SOCKET_IFNAME=enP7s7      # 管理网控制面
export NCCL_DEBUG_FILE=/root/.sglang-logs/nccl-<rank>.log   # 不污染 stdout

# SGLang × SM12x 专用（0.5.14 全部支持）
export SGLANG_DISABLE_DEEP_GEMM=1     # DeepGEMM SM100-only，SM121 必须关
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_SM120_TRITON_FLASHMLA=1 # PR #24692：SM120/121 MLA Triton fallback
export SGLANG_SM120_MQA_FALLBACK=0    # 默认关（走 Triton 快路径）
export SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16   # 可选：压缩态池降内存
# ❌ 0.5.14 无 SGLANG_RAGGED_VERIFY_MODE=compact（0.5.16 特性），不要设置
```

> 注：v1 草案 §3.4 的 `SGLANG_RAGGED_VERIFY_MODE=compact` 和 §4 启动命令里的 `flashinfer_trtllm_routed` **均按 v2 修正**：backend 改 `flashinfer_cutlass`、移除 compact env。Docker run 其余参数（`--restart no --network host --ipc=host --privileged --gpus all --cpuset-cpus 1-19 --shm-size 64g`）不变。

---

## V2-2. SM121 kernel 验证方案定稿

### V2-2.1 `is_sm120_supported()` 在 0.5.14 的行为

| 项 | 结论 | 来源 | 待确认 |
|---|---|---|---|
| PR #24692 是否含在 0.5.14 | ✅ **大概率含**：#24692 合入 2026-06-01，v0.5.14 发布 2026-06-26，时间线满足 | 公开 PR 日期 + release | 容器内 `grep -r is_sm120_supported` 源码核实 |
| SM121（12.1a）是否被覆盖 | ✅ **是**：#24692 以 `major==12` 判断（社区明确 "major==12, covers GB10"），SM120 与 SM121 同属 sm12x 家族 | 社区（xomoxcc/dgx-spark-sglang + NV 论坛） | TP1 启动日志最终确认 |
| 触发路径 | MLA → `flash_mla_sm120_triton.py`；MoE NVFP4 → CUTLASS fp4_gemm（非 Triton fallback 主路径） | PR #24692 + flashinfer 模板实证 | 日志确认实际 kernel 名 |

### V2-2.2 NVFP4 MoE 在 SM121 的 kernel 路径

- **首选 `flashinfer_cutlass`**：SGLang MoE runner 走 flashinfer CUTLASS FP4 GEMM（`fp4_gemm_cutlass_template_sm120.h` 已在本机 flashinfer 0.6.15 栈/生产 vLLM 实证；0.6.14 同理有该模板）。
- **`flashinfer_trtllm_routed` 不可用于 SM121**（#26324，TRTLLM kernel 仅 B200/SM100 验证）——v1 草案的官方推荐在 SM121 不成立，这是本次最关键的修正。
- **降级链（0.5.14 触发方式）**：
  1. `--moe-runner-backend flashinfer_cutlass`（首选，CUTLASS fp4_gemm）
  2. 若报 kernel/API 错误 → `--moe-runner-backend flashinfer`（flashinfer 通用 CUTLASS 路径，SGLang 内部 fp4_gemm fallback）
  3. 若仍失败 → `--moe-runner-backend marlin`（Marlin W4A16 兜底，精度/性能较低，仅验证正确性用）
  - 切换依据：**TP1 冒烟 load + 首 token**，不进 L4 性能。

### V2-2.3 TP1 冒烟脚本级验证清单（启动日志关键字）

| # | 检查点 | 启动日志应看到 | 失败判定 |
|---|---|---|---|
| K1 | NCCL 版本 | `NCCL version 2.30.7+cuda13.0`（`/proc/self/maps` 实测，非 torch 宏） | 出现 2.28.9 → 停 |
| K2 | SM12x 检测 | 日志出现 SM120/SM121 相关 fallback 选择（`is_sm120_supported` 命中，Triton MLA 路径启用） | `kernel image` / `no kernel image` 报错 → 停 |
| K3 | MoE backend | 日志显示选用 **flashinfer_cutlass**（或确认 CUTLASS fp4_gemm） | 隐式回退到 marlin/慢路径 → 记录并分析 |
| K4 | NVFP4 检测 | `Detected NVFP4` / 从 hf_quant_config 读取 `moe_quant_algo=NVFP4` | 未识别 → 权重兼容性问题（见 V2-4） |
| K5 | DeepGEMM | 无 DeepGEMM 初始化（SGLANG_DISABLE_DEEP_GEMM=1 生效） | 出现 DeepGEMM JIT → 停 |
| K6 | CUDA error | 无 `CUDA error` / `out of memory` / `no kernel image` | 出现 → 立即终止回滚 |
| K7 | 首 forward | `/v1/chat/completions` 返回非乱码、非 NaN、非 token0 垃圾（复用 R5 sanity check 启发式） | garbage 输出 → FAIL（CUTLASS SM120 教训） |
| K8 | DSPARK | DSPARK/MTP draft 加载成功、无报错 | draft 缺失/加载失败 → 走 MTP 兼容性排查 |

> 冒烟执行顺序：单机 TP1（head 01，mem-fraction 0.2~0.3 + 门禁）→ 全绿 → TP4 环网。TP1 冒烟**不测性能**，只验证 K1-K8 全绿。

---

## V2-3. JIT 预构建方案（flashinfer 0.6.14 @ SM121）

### V2-3.1 预构建命令

flashinfer 0.6.14 为模块化安装（`flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache`）。NGC 26.07 容器已内置，只需在首次启动前预热 JIT：

```bash
# 容器内（head 01 预构建 + 分发到 02/03/04；或四机各自预热）
export FLASHINFER_CUDA_ARCH_LIST=12.1a          # SM121 显式指定，防止默认 arch 不匹配
export FLASHINFER_JIT_DEBUG=0
# 预热命令（首 token 触发 JIT 编译；用最小模型路径即可）
python3 -c "import flashinfer; print(flashinfer.__version__)"   # 0.6.14
# 或直接跑一次 TP1 冒烟，让 fused_moe_120 / mla 等 kernel 完成 JIT
```

**缓存目录规划**：
- flashinfer JIT cache 默认落 `~/.cache/flashinfer`（或 `FLASHINFER_CACHE_DIR` 指定）；SGLang 另用 `~/.cache/sglang`。
- **建议 host 挂载**（与 vLLM 栈一致，`/home/<USER>/sglang-cache:/root/.cache:rw`）→ 四机共享式预热、容器重建后缓存保留、回滚不丢。
- 若采用容器内可写卷：`-v sglang-jit-vol:/root/.cache`，但容器删除后卷保留需显式 `docker volume rm` 清理，且四机各自独立预热（时间×4）。
- **推荐：host 挂载**。复用 `~/vllm-cache` 同机制（`/root/.cache/vllm` 绑定已实证可行）。

### V2-3.2 0.6.14 已知问题

| 问题 | 0.6.14 状态 | 结论 |
|---|---|---|
| NCCL 回退（0.6.16 把 NCCL 退回 2.29.7） | **未见公开报告**（0.6.16 才有该警告） | ⚠️ 仍按纪律用 LD_LIBRARY_PATH 前插 + `/proc/self/maps` 实测，不依赖"无问题"假设；**待 sre 实测确认** |
| SM12x 覆盖 | ✅ 0.6.14 首个系统覆盖 DGX Spark/SM12x（2026-07-02 release） | 利好，SM121 预期可跑 |
| CuteDSL JIT 下标调用（cute.compile 变为可下标） | vLLM 侧有 JIT monitor 兼容问题（PR #47669）；SGLang 侧未单独报告 | 若 JIT 编译报 `RuntimeError` 记录并上报，非阻塞评估 |
| 依赖 cutlass-dsl | 0.6.14 与 nvidia-cutlass-dsl 4.6.0 配对（社区镜像实证） | 容器已校验组合，一般无需处理 |

---

## V2-4. 权重兼容性终稿（hf_quant_config.json）

### V2-4.1 识别可行性

| 项 | 结论 | 依据 | 待确认 |
|---|---|---|---|
| `hf_quant_config.json` 能否被 0.5.14 识别 | ✅ **大概率可识别**：PR #25820 靠读 `moe_quant_algo == NVFP4` 自动检测；本地权重 quant_algo=MIXED_PRECISION + 专家层 NVFP4 + group_size=16 属 ModelOpt 混合精度标准产物 | 公开 PR + sre 核验的 hf_quant_config 字段 | 字段实际名是否 `moe_quant_algo`（`--help` 无此信息，靠 TP1 冒烟 K4 判定） |
| MIXED_PRECISION 是否被接受 | ✅ 预期接受：dense 层 FP8 + MoE 专家 NVFP4 正是 ModelOpt 标准混合布局，vLLM 侧已加载 | 同栈实证 | SGLang load 冒烟 |
| group_size=16 | ✅ NVFP4 标准块（E2M1 + FP8 E4M3 块缩放，16 元素块） | 公开格式 | 无 |

### V2-4.2 "experts-mtp-fallback" 命名风险（MTP 策略）

- **含义**：命名暗示 **MTP draft head 走 fallback（未转 NVFP4）**，仅主模型 experts 为 NVFP4。与 tsarihan"全转"方案相反。
- **风险**：SGLang 0.5.14 的 DSPARK 需要读取 MTP 权重。若 MTP 层保持 FP8/BF16，需确认 SGLang 能加载混合精度 MTP；若 MTP 权重命名/布局不被 DSPARK 解析，draft 无法加载 → 启动失败或接受率崩塌。
- **验证点（对应 Tessa R9 接受率门槛）**：
  1. TP1 冒烟 K8：DSPARK/MTP draft 加载成功；
  2. R9：**draft 接受率 ≥0.40**（Rarri vLLM NVFP4 实测 49.4% 参照）；**<0.20 即 FAIL**（tsarihan 0.121 崩溃阈值）；
  3. 若 FAIL → 两条路：①确认/重转 MTP（全转 NVFP4 或转 BF16 显式化）②下载 MJPansa 版（其 MTP 策略独立确认过 vLLM 可运行，但无 SGLang 接受率数据）。
- **MJPansa 下载**：**仍列为备选**（仅当 W 验证不通过 / R9 FAIL / hf_quant_config 不可识别时启用）；主路径用四机现有 NVFP4 权重。

---

## V2-5. 与生产并存的最后核对（0.5.14 特有问题）

### V2-5.1 A/B 互斥切换执行序列（不变）

```
停 vLLM TP4（head+worker，确认 monitor 退出）→ 清残留容器
→ GPU/内存门禁（≤180s，无 vLLM 进程）→ 启动 SGLang TP4（head-first）
→ 验证（/health 200 + /v1/models + 首 token）→ 性能 A/B
→ 切回：停 SGLang → 起 vLLM TP4（systemd 自愈兜底）
```

**回滚锚点（不变）**：镜像 tag（`sglang-nvfp4:0.5.14` ↔ vLLM `0.2.1-v026.0`）+ 权重目录（`-nvfp4`）+ `start_sglang_tp4_cluster.sh` 参数化 + 四机 NCCL/脚本 MD5 登记（沿用 cutlass-run 锚点机制）。

### V2-5.2 0.5.14 特有补充核对

| # | 项目 | 核对结论 |
|---|---|---|
| A1 | **backend 变更风险**：flashinfer_trtllm_routed → flashinfer_cutlass | 属参数级变更，不涉及镜像/权重/网络；回滚 = 改回参数重启，分钟级 |
| A2 | **compact env 移除**：DSpark 回到 0.5.14 早期形态 | 接受率预期可能低于 0.5.16 compact；R9 门槛仍然有效，若接受率低但 ≥0.20 记录为"0.5.14 特性差异" |
| A3 | **flashinfer 0.6.14 vs 0.6.15.post1**：NCCL 回退未见报告 | 预检 `/proc/self/maps` 强制验证 2.30.7；若 0.6.14 隐式依赖 NCCL 2.28.9 冲突 → 重钉 2.30.7（沿用 vLLM 方法） |
| A4 | **互斥守卫优先级**：生产 vLLM 运行中，正式 TP4 等通知 | 本阶段只做参数定稿与方案；实际切换由主理人排窗口 |
| A5 | **低配并存冒烟**（可选）：mem-fraction 0.2~0.3 + 门禁 ≥55G + cpuset=0 | 仅限功能冒烟（K1-K8），不做性能对比 |

---

## V2-6. 风险增量（相对 v1 表新增/修正）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| V2-R1 | 🔴 **flashinfer_trtllm_routed 在 SM121 不可运行**（#26324）——v1 推荐 backend 不适用 | 高 | v2 改 `flashinfer_cutlass`；降级链 flashinfer→marlin |
| V2-R2 | 🟠 **SGLANG_RAGGED_VERIFY_MODE=compact 误用于 0.5.14**（env 无效/被忽略） | 中 | v2 移除；0.5.14 DSpark 用默认验证模式 |
| V2-R3 | 🟠 **"experts-mtp-fallback" MTP 未转 NVFP4** → SGLang DSPARK 加载/接受率风险 | 中 | TP1 冒烟 K8 + R9 接受率门槛（≥0.40，<0.20 FAIL）；备选 MJPansa |
| V2-R4 | 🟡 flashinfer 0.6.14 NCCL 回退**未见报告但未实证** | 中 | `/proc/self/maps` 强制 2.30.7；LD_LIBRARY_PATH 前插 |
| V2-R5 | 🟡 `flashinfer_cutlass` / `modelopt_fp4` / `--metrics-port` 是否为 0.5.14 合法取值未确认 | 中 | sre `launch_server --help` 实测；非法则按备选值/默认值回退 |
| V2-R6 | 🟡 0.5.14 DSPARK 接受率可能低于 0.5.16 compact | 低 | 记录差异；不阻塞功能验证 |

---

## V2-7. 待 sre / testing 实测闭环清单

1. 【sre】容器内 `python3 -m sglang.launch_server --help` 抓取 0.5.14 真实 flag：确认 `--moe-runner-backend` 候选值（含 `flashinfer_cutlass`）、`--quantization modelopt_fp4`、`--metrics-port`、`--kv-cache-dtype`、DSPARK 相关参数。
2. 【sre】容器内 `grep -r is_sm120_supported` 核实 #24692 是否在 0.5.14；`/proc/self/maps | grep libnccl` 确认 2.30.7。
3. 【sre】flashinfer 0.6.14 NCCL 回退实测（启动后 ncclGetVersion）。
4. 【testing】TP1 冒烟按 V2-2.3 K1-K8 逐项核对；R9 接受率。
5. 【team-lead】正式 TP4 启动窗口排期（生产 vLLM 停机）。

---

## V2-8. 定稿裁决（sre 0.5.14 `launch_server --help` 实测后）——本节替代 v2-1/v2-2 冲突行

**日期**：2026-08-14（sre 实测回传后定稿）
**实测依据**：sre 在 26.07 容器内 `launch_server --help` 结果
**生效范围**：凡与本节冲突的 v2-1/v2-2 内容，以本节为准。

### V2-8.1 🔴 DSPARK 缺失裁决（0.5.14 无 DSpark）

**实测事实**：`--speculative-algorithm` 在 0.5.14 的 choices 仅为 `EAGLE / EAGLE3 / NEXTN / NGRAM / STANDALONE / DFLASH`，**无 DSPARK、无 MTP**。
**公开证据交叉确认**：DSpark（confidence-driven speculative decoding）是 **SGLang v0.5.16（2026-07-25）才合入**的特性（PR #30261/#31434），0.5.14 的发布线不含。

**裁决**：推荐**两阶段策略**（不用无谓地推翻已就绪的 26.07，也不在 0.5.14 上死等 DSpark）。

| 阶段 | 容器 | 目标 | 说明 |
|---|---|---|---|
| **Phase-A（立即）** | 0.5.14（现有 26.07） | **功能验证 + prefill-only 性能 A/B** | 无投机运行；prefill 收益（NVFP4 vs MXFP4）不依赖 DSpark，仍可测；R9 改为"无投机基线"记录 |
| **Phase-B（后续窗口）** | **0.5.16+**（上游 lmsysorg 或 NGC 更新 tag） | **DSpark 性能裁决（decode + 接受率）** | DSpark decode +85%（官方 383.7 tok/s @TP8 B300）；R9 接受率门槛只在 0.5.16 才有意义；最终性能判定在 Phase-B |

**三选项对答**：
1. **接受无投机运行？** ✅ **接受（Phase-A 主路径）**。功能/正确性/环网验证不受影响；prefill A/B 仍有效（tsarihan 1.14-1.32× 是 **prefill 吞吐收益**，与 DSpark 无关，仍可在 0.5.14 验证）。
2. **有无其他 MTP 利用途径（EAGLE3/DFLASH）？** ❌ **无**。EAGLE/EAGLE3 需要独立 EAGLE draft 权重（DSV4-Flash 未携带）；DFLASH 需独立 draft 小模型；0.5.14 的 MTP 支持仅限 Nemotron（非 DSV4）。DSV4 的 MTP 模块只能被 **0.5.16 的 DSPARK** 消费。
3. **升级容器路线？** ✅ **Phase-B 必要投资**。0.5.16 的 breaking changes 对本方案无影响（我们已用旧 flag 名）；DSpark decode +85%、SM121 自动选 flashinfer_cutlass（#26496）、compact verify 全在 0.5.16。**建议 Phase-A 与 Phase-B 并行推进镜像准备**，不阻塞。

**对测试计划（R9）的影响**：
- R9（MTP/DSPARK 接受率）→ **0.5.14 阶段改名为 R9-A：无投机基线**（记录 decode tps、TTFT、TPOT，供 Phase-B 对比）；原接受率门槛（≥0.40 / <0.20 FAIL）**延后到 Phase-B（0.5.16）**执行。
- K8（DSPARK 加载）→ 0.5.14 改检：**MTP 权重被忽略/加载不报错**（0.5.14 不消费 MTP，但须确认加载 MTP 权重不崩）。
- 新增 **K9：无投机模式**（`--speculative-algorithm STANDALONE` 显式确认）。

**对性能预期（tsarihan 1.14-1.32×）的影响**：
- 1.14-1.32× 是 **prefill 带宽收益**（NVFP4 vs MXFP4），**不含 DSpark** → 0.5.14 仍可验证 prefill。
- **decode 硬门槛（≥0.95×）在 0.5.14 无投机下不公平**（vLLM 基线带 dspark）→ **decode A/B 延后 Phase-B**；0.5.14 只做 prefill A/B + decode 无投机基线记录。

### V2-8.2 flag 名修正（sre 实测为准）

| 参数 | v2-1 草案 | **0.5.14 实测/终稿** | 影响 |
|---|---|---|---|
| `--tp` | `--tp 4` | **`--tp-size 4`** | 0.5.14 用 `--tp-size` |
| `--metrics-port` | `--metrics-port 8011` | **无此 flag**；用 `--enable-metrics`，metrics 暴露在主端口 `/metrics` | **8011 端口规划降级**：metrics 走 `:8010/metrics`；Grafana/Prometheus scrape 改为 `http://<NODE_IP>:8010/metrics`（或前置反代把 /metrics 映射到 8011）。端口表 8011 改为"由 8010/metrics 承担" |
| `--quantization` | 不传（自动检测） | **`modelopt_fp4` 合法，显式传** `--quantization modelopt_fp4` | 显式声明更稳（#26496 在 SM120 也依赖 quantization=modelopt_fp4 触发自动后端选择） |
| `--moe-runner-backend` | `flashinfer_cutlass`（推断） | **`trtllm_routed` flag 合法但 SM121 不可运行；`flashinfer_cutlass` 是否在 0.5.14 choices 待 sre 补测** | 见 V2-8.3 |
| `--speculative-algorithm` | `DSPARK` | **`STANDALONE`（0.5.14 Phase-A）**；Phase-B 用 `DSPARK` | 无投机运行 |

### V2-8.3 `--moe-runner-backend` 0.5.14 实际可行值（SM121）

**裁决依据**：PR #26496（"Changes for SM120 perf and usability for NVFP4"，晚于 0.5.14）证实——在 `quantization=modelopt_fp4` + SM120 上，SGLang 自动选 `flashinfer_cutlass`，因为默认 `flashinfer_trtllm` **仅支持 SM100**。SM121 与 SM120 同属 sm12x，同样适用此结论。

**优先级**（sre 按序补测确认）：
1. **首选 `flashinfer_cutlass`**——若在 0.5.14 choices 中则直接用（同 #26496 的 SM120 决策）。
2. 若 0.5.14 无 `flashinfer_cutlass` → 测 `flashinfer`（通用 CUTLASS fp4_gemm，#25820 的 fallback 路径）。
3. 若均不可用 → `marlin`（W4A16 兜底，仅正确性验证，**性能不作数**）→ 这显著强化 Phase-B 升级必要性。
4. **禁止** `flashinfer_trtllm_routed` 上 SM121（#26324：TRTLLM kernel 仅 SM100）。

**需 sre 补测的 flag（直接给出命令）**：
```bash
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "moe-runner-backend"   # 确认 choices 是否含 flashinfer_cutlass / flashinfer
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "speculative-algorithm" # 确认无 DSPARK/MTP
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "quantization"          # 确认 modelopt_fp4
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "kv-cache-dtype"        # 确认 NVFP4/DS-MLA 取值
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "metrics"               # 确认 --enable-metrics 与默认端口
python3 -m sglang.launch_server --help 2>&1 | grep -A3 "tp-size\|tp "          # 确认 --tp-size 语义
grep -rn "dspark\|DSPARK\|DSPARK" /usr/local/lib/python3.12/dist-packages/sglang/srt/ 2>/dev/null | head   # 确认源码无 DSpark
```

### V2-8.4 修正后参数终稿表（0.5.14 Phase-A，替代 v2-1）

```bash
python3 -m sglang.launch_server \
  --model-path /models --trust-remote-code \
  --tp-size 4 --nnodes 4 --node-rank {N} \
  --dist-init-addr <NODE_IP>:26000 \
  --host 0.0.0.0 --port 8010 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_cutlass \   # 若 0.5.14 无此值 → flashinfer → marlin（见 V2-8.3）
  --speculative-algorithm STANDALONE \        # 0.5.14 无 DSPARK
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --max-model-len 65536 \
  --enable-metrics                            # 无 --metrics-port；metrics 在 :8010/metrics
```

环境变量（同 v2-1，删除 SGLANG_RAGGED_VERIFY_MODE）：
```bash
export SGLANG_DISABLE_DEEP_GEMM=1
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_SM120_TRITON_FLASHMLA=1
export SGLANG_SM120_MQA_FALLBACK=0
# 不设 SGLANG_RAGGED_VERIFY_MODE（0.5.14 无此 env）
# NCCL 环网 env 沿用生产实测（GID_INDEX=3 / ring-only / LD_PRELOAD），见 v2-1
```

### V2-8.5 风险增量（本轮裁决新增）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| V2-R7 | 🔴 **0.5.14 无 DSpark**：decode 无法与 vLLM(DSpark) 基线公平对比 | 高 | Phase-A 只做 prefill A/B + decode 无投机基线记录；decode 裁决延后 Phase-B(0.5.16) |
| V2-R8 | 🟠 **0.5.14 可能无 `flashinfer_cutlass`**（#26496 晚于 0.5.14）→ SM121 NVFP4 无可用加速路径 | 高 | sre 补测；若无 → marlin 仅正确性，性能阶段直接升 0.5.16 |
| V2-R9 | 🟡 **8011 端口规划失效**（metrics 在 :8010/metrics） | 中 | Grafana/Prometheus scrape 指向 :8010/metrics；或反代映射 8011 |
| V2-R10 | 🟡 MTP 权重在 0.5.14 被加载但未消费，可能引发加载告警/失败 | 低 | K8 改检"加载不报错"；若报错记录并评估 0.5.16 |

---

*v2 由 architect 基于公开资料 + 社区实证 + 容器实测定稿（2026-08-14）；凡标"待确认"项以 sre `launch_server --help` 与 TP1 冒烟为准。v2-8 定稿裁决基于 sre 0.5.14 `--help` 实测，替代 v2-1/v2-2 冲突行。*

---

# V3 节：SGLang 升级版本选型（0.5.14 → 支持 DSPARK 的最新版本）

**日期**：2026-08-14（v3 追加）
**作者**：阿奇（Archi）· 系统架构师
**触发**：用户指令"升级到最新，DSPARK 必须启用"；v2-8 已裁决 Phase-B（0.5.16+）为 DSPARK 必要投资，本节约定升级目标版本、路径与执行序列。
**基线**：NGC 26.07-py3 = SGLang **0.5.14+nv26.7** / torch **2.13.0a0+9186a08b2c** / flashinfer **0.6.14** / transformers 5.8.1 / CUDA 13.3.1 / NCCL 2.30.7（四机就绪，层 md5 96e467d4… 一致）

---

## V3-0. 摘要与裁决（TL;DR）

- **目标版本：SGLang v0.5.16**（2026-07-25，DSPARK 引入版）。**不是 0.5.17**。
- **推荐路径：C —— 现有 26.07 容器内 `pip install --no-deps sglang==0.5.16` 升级 → 重打包 → 推内网 registry → 四机分发**。改动面最小，保留 NGC 已验证基础栈。
- **flashinfer 兼容性结论（关键）**：**0.5.16 锁 flashinfer 0.6.14（`flashinfer_python[cu13]==0.6.14`）＝ NGC 26.07 现装版本 → 无需升级 flashinfer**。`≥0.6.15.post1` 是 **0.5.17** 的要求（0.6.15 曾在 0.5.16 周期内被回滚 #31502/#31625，稳定性存疑）。
- **路径 A（NGC 26.08+）当前不可用**：截至 2026-08-14，NGC SGLang 容器 release notes 最新仅到 **26.07**，26.08 未发布。
- **路径 B（lmsysorg:v0.5.16 aarch64）不推荐**：上游镜像基于 torch **2.11.0**（非 NVIDIA 校验的 torch 2.13.0a0+nv26.07）+ CUDA 13.0.1，SM121 NVFP4 验证弱，下载/验证成本高。
- **0.5.16 完整参数终稿**见 V3-4；**升级执行序列 + 回滚锚点**见 V3-6。

---

## V3-1. 版本盘点表

| # | 候选 | NGC tag / 上游版本 | 发布日期 | 内置组件（关键） | DSPARK | 本方案可用性 |
|---|---|---|---|---|---|---|
| 0 | **现状** | `nvcr.io/nvidia/sglang:26.07-py3` | 2026-07 月 | CUDA 13.3.1 / **SGLang 0.5.14+nv26.7** / torch **2.13.0a0** / flashinfer **0.6.14** / transformers 5.8.1 / flash-attn 2.7.4.post1 / xgrammar 0.2.1 | ❌（实测仅 EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH） | 已就绪（四机 md5 一致） |
| A | NGC 26.08 / 26.09 | **尚未发布**（release notes 最新 26.07，2026-08-04 最后更新） | 预计 8 月下旬（按月滚动） | 未知（内置 SGLang 版本待发布后确认） | 未知（若内置 ≥0.5.16 则 ✅） | 🔴 当前不可用；后续波次候选 |
| B | `lmsysorg/sglang:v0.5.16`（aarch64） | 2026-07-25 | CUDA 13.0.1 基础 / torch **2.11.0** / flashinfer 0.6.14 / sgl-kernel 0.4.5 / TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a | ✅ | 🟠 可行但需自验（arm64 标准 tag 可用性、SM121 kernel 完整性、JIT 自配）；与 NGC torch 栈不同 |
| **C** | **26.07 容器内升级 `sglang==0.5.16`** | 2026-07-25 | 保留 torch **2.13.0a0+nv26.07** / flashinfer **0.6.14** / CUDA 13.3.1 / NCCL 2.30.7；升级 sglang→0.5.16（+sgl-kernel 0.4.5、transformers 5.12.1 按需核对） | ✅ | 🟢 **推荐** |
| D（延伸） | 上游 **v0.5.17**（Latest） | 2026-08-08 | flashinfer **0.6.15.post1** / sgl-deep-gemm 0.1.5.post1 / torch 2.11.0 / CUDA 13.0.1 基础 | ✅（含 Kimi K3 DSpark 等） | 🟠 需连带升 flashinfer（0.6.15 曾回滚）；增量收益（Kimi K3/Rust 前端/DWDP/更快加载）与本 DSV4+DSPARK 用例无关 → **下波候选** |
| E（延伸） | 0.5.18 / 0.6.x | **不存在**（截至 2026-08-14，GitHub releases 0.5.16 之后仅 0.5.17） | — | — | — |

**关键交叉验证**：
- 0.5.16 PyPI `requires_dist`（实抓）：`torch==2.11.0`、`flashinfer_python[cu13]==0.6.14`、`transformers==5.12.1`、`sglang-kernel==0.4.5`、`sgl-deep-gemm==0.1.4.post1`、`nvidia-cutlass-dsl[cu13]==4.6.0`；**cp312-manylinux_2_34_aarch64 wheel 存在（14.4MB）**。
- PR #26496（SM120 NVFP4：`modelopt_fp4` + SM120 → `moe_runner_backend=flashinfer_cutlass`、fp4_gemm 移 `flashinfer_cudnn`→`flashinfer_cutlass`）**合并于 2026-06-05，早于 v0.5.14（06-26）** → 0.5.14 大概率已含，**0.5.16 必含**（修正 v2-8 中"晚于 0.5.14"的保守假设）。
- 0.5.15.post1 修复 #31001（flashinfer trtllm FP4 MoE 长输入 NaN）→ 进一步佐证 SM121 上避开 `trtllm_routed`、走 `flashinfer_cutlass` 的方向。

---

## V3-2. 升级路径三选一评估

### 路径 A：NGC 26.08+ 官方容器
| 维度 | 评估 |
|---|---|
| 可用性 | 🔴 **当前不可用**（26.08 未发布，release notes 最新 26.07，2026-08-04 更新；按月滚动预计 8 月下旬） |
| 优点 | 官方校验组合、干净、内置版本组合（SGLang/flashinfer/torch/CUDA）由 NVIDIA 保证 |
| 缺点 | 需重新拉取分发 20.8GB×4；内置 SGLang 是否 ≥0.5.16 未验证（若 26.08 仍带 0.5.15/0.5.16 早期则需再等）；时间不可控 |
| 结论 | ❌ **不选**（等不起且不确定）；作为"0.5.17 波次"并行观察项 |

### 路径 B：上游 `lmsysorg/sglang:v0.5.16`（aarch64）
| 维度 | 评估 |
|---|---|
| 优点 | 含 DSPARK + PR #26496；社区镜像 ~12GB 框架版 |
| 缺点 | torch **2.11.0**（非 NGC 2.13.0a0）——SM121 NVFP4 组合未获 NVIDIA 校验；CUDA 13.0.1 基础与主机/现有 vLLM 栈的兼容需重验；**arm64 标准 v0.5.16 tag 可用性未实测**（仅见 kimi-k3-…-arm64 构建，SGL_VERSION=0.5.16）；flashinfer/JIT 需自配；丢失 NGC 已验证的 torch/flashinfer 组合 |
| 结论 | 🟠 **备选不推荐**；仅当路径 C 遇到无法逾越的 torch ABI 问题时启用 |

### 路径 C：现有 26.07 容器内 pip 升级 sglang==0.5.16 → 重打包推 registry（**推荐**）
| 维度 | 评估 |
|---|---|
| 优点 | 保留 NGC 基础栈（torch 2.13.0a0+nv26.07 / flashinfer 0.6.14 / CUDA 13.3.1 / NCCL 2.30.7 LD_PRELOAD 全部不动）；**flashinfer 0.6.14 与 0.5.16 锁版完全一致，零 flashinfer 变更**；改动面=1 个 Python 包（+按需 2-3 个小包）；wheel 仅 14.4MB；DSPARK 完整可用；回滚=改镜像 tag |
| 缺点 | 0.5.16 wheel 基于 torch 2.11.0 构建，需在 torch 2.13.0a0 上验证向前兼容（NGC 0.5.14 已在 2.13 上跑，预期低风险，但须 TP1 冒烟）；**必须 `--no-deps`**（wheel 锁 `torch==2.11.0`，普通 pip 会降级 torch 破坏 NGC 栈）；sgl-kernel/transformers 版本需核对 |
| 结论 | 🟢 **推荐** |

**推荐理由**：约束先行（DSPARK 必须启用 + 保留已验证 NGC 基础栈 + 生产 vLLM 互斥不变）→ 0.5.16 是唯一满足"含 DSPARK **且** flashinfer 0.6.14 完全匹配"的版本；路径 C 是唯一不动 torch/flashinfer/NCCL 的升级方式，风险面最小、分发最快（增量镜像小）、回滚最简。

**明确回答 team-lead 假设**：**"0.5.16 要求 flashinfer ≥0.6.15.post1？" → 否**。0.5.16 锁 **0.6.14**（= 现装），无需连带升级 flashinfer；≥0.6.15.post1 属于 0.5.17。因此本升级 **不改 flashinfer、不改 NCCL 2.30.7 LD_PRELOAD**。

---

## V3-3. 0.5.16 breaking changes 对现有启动脚本的影响（脚本更新必须项）

| # | 项 | 0.5.14（现） | **0.5.16** | 动作 |
|---|---|---|---|---|
| B1 | `--speculative-algorithm` | `STANDALONE`（无 DSPARK） | `DSPARK`（新增，必须） | 改值 + 加 `--speculative-dspark-block-size 5` + `SGLANG_RAGGED_VERIFY_MODE=compact` |
| B2 | NVFP4 GEMM 后端 | `--fp4-gemm-backend cutlass`（0.5.14 若用） | **`cutlass` 已移除**；合法值 `auto / flashinfer_cutlass / flashinfer_cutedsl / flashinfer_cudnn`；`auto`=SM100→cutedsl、SM120→cutlass（SM121 未文档覆盖） | 显式 `--fp4-gemm-backend flashinfer_cutlass` |
| B3 | `--moe-runner-backend` | `flashinfer_cutlass`（v2-8.3 裁决） | 合法值含 `flashinfer_trtllm / flashinfer_trtllm_routed / flashinfer_mxfp4 / flashinfer_cutedsl / flashinfer_cutlass / triton / marlin`；PR #26496 在 SM120+modelopt_fp4 自动选 `flashinfer_cutlass` | 保留显式 `flashinfer_cutlass`（SM121 不依赖 auto） |
| B4 | NVFP4 依赖 | FlashInfer 首选（仍有 QServe/FBGEMM 残留） | **NVFP4 强制 FlashInfer**（QServe/FBGEMM 移除） | 一致，无额外动作 |
| B5 | flag 改名（若脚本用到才需处理） | `--enable-deepep-waterfill` / `--optimistic-prefill-retries` / `num_tokens_per_bs` | `--enable-waterfill` / `--optimistic-prefill-attempts` / `num_tokens_per_req`（**无 deprecated 别名，旧名直接报错**） | 本方案 TP 首期不用 EP → 预期无影响；grep 脚本确认 |
| B6 | `--tp` vs `--tp-size` | `--tp-size`（sre 实测） | 0.5.16 文档示例混用 `--tp`；建议保持 `--tp-size`，`--help` 确认（若报未知则回退 `--tp`） | `--help` 实测定稿 |
| B7 | metrics | `--enable-metrics`（无 `--metrics-port`，走 `:8010/metrics`） | 0.5.16 若支持 `--metrics-port` 则恢复 8011 规划 | `--help` 实测；否则维持 `:8010/metrics` |

---

## V3-4. DSPARK 启用参数终稿（0.5.16，四机 TP4 + NVFP4 + DSPARK）

**四机一致项（head 01 + worker 02/04/03 仅 `--node-rank` 不同；环序 01=0/02=1/04=2/03=3）**：

```bash
python3 -m sglang.launch_server \
  --model-path /models --trust-remote-code \
  --tp-size 4 --nnodes 4 --node-rank {0,1,2,3} \
  --dist-init-addr <NODE_IP>:26000 \
  --host 0.0.0.0 --port 8010 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \      # 0.5.16 显式指定（SM121 不依赖 auto）
  --moe-runner-backend flashinfer_cutlass \    # PR #26496 逻辑，SM121 显式更稳
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \          # DSV4 0731 config dspark_block_size=5（task 确认）；官方默认 8，按模型 config 取 5
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --max-model-len 65536 \
  --enable-metrics                            # 若 0.5.16 支持 --metrics-port 8011 则恢复端口规划（B7）
# 可选：--speculative-dspark-sps-table-path sps_table.json（SPS 成本表，非必需）
```

**容器内环境变量终稿（新增/修改相对于 0.5.14）**：

```bash
# ── 新增（0.5.16 DSpark + SM121 JIT）──
export SGLANG_RAGGED_VERIFY_MODE=compact          # DSpark 紧凑验证（PR #30261/#31434，0.5.16 必须）
export FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"    # SM121 JIT 显式双架构编译（防默认 arch 不匹配；社区实测必需）

# ── 保留（0.5.14 已定稿，不变）──
export SGLANG_DISABLE_DEEP_GEMM=1                 # DeepGEMM SM100-only，SM121 必须关
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_SM120_TRITON_FLASHMLA=1             # PR #24692：SM120/121 MLA Triton fallback
export SGLANG_SM120_MQA_FALLBACK=0
# NCCL 环网 env 全部保留（LD_PRELOAD shim v8 + ring-only 2.30.7 / NCCL_ALGO=RING / GID_INDEX=3 / TOS=46 / TIMEOUT=22 / PEER_HCA 对口映射）
```

**DSpark 生效判定（TP1 冒烟 K8 升级版）**：
- `--help` 确认 `speculative-algorithm` choices 含 `DSPARK`；
- 启动日志出现 DSPARK draft 加载、`SGLANG_RAGGED_VERIFY_MODE=compact` 生效（verify 走 ragged compact 路径）；
- R9 接受率门槛（≥0.40，<0.20 FAIL）在 0.5.16 正式生效（v2-8.1 已延后至此）；
- 官方参考：DSV4-Pro TP8 B300 bs=1 = 383.7 tok/s、accept length ~5；**本机为 4×DGX Spark(SM121) 环网，绝对值会低，只做相对基线对比**。

---

## V3-5. 升级执行序列（给执行代理）+ 回滚锚点

```
Phase 0 预检（02，下载源+registry）
  ├─ 记录基线版本：python3 -c "import sglang,flashinfer,torch,transformers;print(sglang.__version__,flashinfer.__version__,torch.__version__,transformers.__version__)"
  │   并查 sglang_kernel 版本（import sglang_kernel 或 pip list | grep sglang-kernel）——决定是否需要连带升级
  ├─ 回滚锚点①：原镜像 tag 不动（nvcr.io/nvidia/sglang:26.07-py3），并另打 docker tag … <NODE_IP>:5000/sglang/sglang:0.5.14-nv26.07-rollback
  └─ 下载 wheel 到 02（PyPI/代理；vendor /tmp/wheels）：sglang==0.5.16(cp312-aarch64)、sglang-kernel==0.4.5、
     sgl-deep-gemm==0.1.4.post1、nvidia-cutlass-dsl[cu13]==4.6.0、transformers==5.12.1（按需）

Phase 1 构建/升级镜像（02）
  ├─ 方案 C-w（首选，wheel）：Dockerfile: FROM nvcr.io/nvidia/sglang:26.07-py3 → COPY wheels → 
  │   RUN pip install --no-deps <sglang==0.5.16> [sglang-kernel==0.4.5 若现有版本旧] [transformers==5.12.1 若 load 报 config 错]
  │   ⚠️ 必须 --no-deps（wheel 锁 torch==2.11.0，普通 pip 会降级 torch 破坏 NGC 栈）
  │   → docker build → tag <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark → push
  ├─ 回滚锚点②：新 tag 只增不改，绝不覆盖 0.5.14 旧 tag
  ├─ 容器内验证：版本核对 + launch_server --help 抓取（B1/B6/B7 定稿：DSPARK choices / --tp-size / --metrics-port / fp4-gemm choices）
  └─ 方案 C-s（回退，源码重建）：若 wheel 在 torch 2.13.0a0 下 ABI 报错 → 容器内 /opt/sglang checkout v0.5.16，
     pip install -e python[all] --no-deps --no-build-isolation（NGC 自带构建链），sgl-kernel 0.4.5 同样源码对 torch 2.13 重建

Phase 2 四机分发
  ├─ 02 push → 01/03/04 pull（内网 registry 分钟级）；⚠️ 沿用教训：勿三机并行 pull 与 push 并发，错峰
  └─ 本地 tag：docker tag …:0.5.16-nvfp4-spark sglang-nvfp4:0.5.16；记录镜像 digest，四机核对一致

Phase 3 四机版本验证（本阶段只验证，不启动 TP4 服务）
  ├─ 每机 docker run --rm 检查：sglang.__version__==0.5.16、flashinfer.__version__==0.6.14、
  │   torch.__version__==2.13.0a0+nv26.07、/proc/self/maps|grep libnccl → 2.30.7（LD_PRELOAD 生效）
  └─ DSPARK flag 实测：launch_server --help | grep -A3 speculative-algorithm（choices 含 DSPARK）
     + grep -A3 "fp4-gemm-backend\|moe-runner-backend"（含 flashinfer_cutlass）

Phase 4 启动脚本参数更新
  ├─ start_sglang_tp4_head.sh / worker 脚本：STANDALONE→DSPARK、加 --speculative-dspark-block-size 5、
  │   加 SGLANG_RAGGED_VERIFY_MODE=compact、加 FLASHINFER_CUDA_ARCH_LIST、加 --fp4-gemm-backend flashinfer_cutlass、
  │   镜像 tag 参数化 0.5.16-nvfp4-spark；grep 确认无旧改名 flag（B5）
  └─ 回滚锚点③：脚本版本化/备份；回滚=还原脚本 + 镜像 tag 指回 0.5.14

Phase 5 TP1 冒烟 DSPARK 加载确认（等停机窗口，生产 vLLM 互斥约束不变）
  ├─ 维护窗口：停 vLLM TP4 → GPU 门禁 → TP1（01，mem-fraction 0.2~0.3）按 v2-2.3 K1-K8 全项 + DSPARK 生效判定（V3-4）
  ├─ 通过后再 TP4 环网冒烟（K1-K8 + 首 token + 接受率 R9 ≥0.40）
  └─ 正式服务启动仍等主理人通知（本阶段不启动 TP4 正式服务）
```

**回滚锚点汇总**（全部只增不改、分钟级回滚）：
1. 镜像：`nvcr.io/nvidia/sglang:26.07-py3`（原始）+ `…:0.5.14-nv26.07-rollback` 双保险；新 `…:0.5.16-nvfp4-spark` 只增不改。
2. 脚本：`start_sglang_tp4_*.sh` 参数化（镜像 tag + speculative 参数）；版本化备份。
3. 权重/NCCL/网络栈：**零变更**（路径 C 不动 torch/flashinfer/NCCL/CUDA 层）。
4. 与生产 vLLM 互斥：A/B 切换序列沿用 v2-5.1，正式切换窗口由主理人排期。

---

## V3-6. 风险增量（相对 v2 表新增/修正）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| V3-R1 | 🔴 **pip 未用 `--no-deps` → torch 被降级 2.11.0，破坏 NGC 栈** | 高 | 强制 `--no-deps`；装后四机版本核对（Phase 3） |
| V3-R2 | 🟠 **0.5.16 wheel（torch 2.11 构建）跑在 torch 2.13.0a0+nv26.07** 向前兼容未验证 | 高 | TP1 冒烟 K1-K8；失败回退 C-s 源码重建（对 torch 2.13 编译）；回滚锚点① |
| V3-R3 | 🟠 **sglang-kernel 0.4.5 / transformers 5.12.1 在 `--no-deps` 下不自动装**，旧版本可能不兼容 | 中 | Phase 0 先核对容器内现有版本；按需定向 `--no-deps` 安装这两个小包 |
| V3-R4 | 🟠 **NGC fork 特有补丁（0.5.14+nv26.7）在上游 0.5.16 wheel 中缺失** | 中 | 冒烟验证兜底；0.5.16 上游已含更多修复；回滚锚点① |
| V3-R5 | 🟡 **SM121 下 fp4_gemm/moe autoselect 仅文档覆盖 SM100/SM120**，SM121 需显式 | 中 | 显式 `--fp4-gemm-backend flashinfer_cutlass` + `--moe-runner-backend flashinfer_cutlass` + `FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"` |
| V3-R6 | 🟡 0.5.16 锁 flashinfer 0.6.14 的 NCCL 回退未见报告但未实证 | 中 | `/proc/self/maps` 强制 2.30.7 + LD_LIBRARY_PATH 前插（沿用 v2 纪律） |
| V3-R7 | 🟡 0.5.17 若被误选需 flashinfer 0.6.15.post1（0.6.15 曾回滚） | 低 | 本波固定在 0.5.16；0.5.17 列入下波（等社区稳定或 NGC 26.08 内置） |
| V3-R8 | 🟡 `--tp-size` 在 0.5.16 是否仍合法未实测（文档混用 `--tp`） | 低 | Phase 1 容器内 `--help` 定稿；不支持则回退 `--tp` |

---

## V3-7. 待 sre / testing 实测闭环清单

1. 【sre·Phase 0】02 容器内记录 sglang/flashinfer/torch/transformers/**sglang-kernel** 基线版本。
2. 【sre·Phase 1】升级后容器内 `launch_server --help`：确认 DSPARK choices、`--fp4-gemm-backend`/`--moe-runner-backend` 含 flashinfer_cutlass、`--tp-size`/`--tp`、`--metrics-port`。
3. 【sre·Phase 3】四机版本 + NCCL 2.30.7 + DSPARK flag 验证。
4. 【testing·Phase 5】TP1 冒烟 K1-K8 + DSPARK 生效 + R9 接受率（≥0.40，<0.20 FAIL）。
5. 【team-lead】正式 TP4 启动窗口排期（生产 vLLM 停机 + 正式服务启动通知）。

---

## V3-8. 证据链接（v3 新增）

- SGLang releases（0.5.16 / 0.5.17）：https://github.com/sgl-project/sglang/releases
- sglang 0.5.16 PyPI（requires_dist 实抓：torch==2.11.0 / flashinfer_python[cu13]==0.6.14 / sglang-kernel==0.4.5）：https://pypi.org/project/sglang/0.5.16/
- PR #26496（SM120 NVFP4 → flashinfer_cutlass，合并 2026-06-05）：https://github.com/sgl-project/sglang/pull/26496
- PR #30261 / #31434（DSpark + compact ragged verify）：https://github.com/sgl-project/sglang/pull/30261
- SGLang NGC release notes（最新 26.07，无 26.08）：https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/index.html
- DSpark 运行指南（block-size 默认 8 / compact env）：https://dreaming.press/posts/how-to-run-dspark-speculative-decoding-sglang-0-5-16.html
- NVIDIA 论坛 DGX Spark SM121 NVFP4 CUTLASS + FLASHINFER_CUDA_ARCH_LIST 实证：https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163/46
- lmsysorg/sglang kimi-k3-…-arm64（SGL_VERSION=0.5.16 / CUDA 13.0.1）：https://hub.docker.com/layers/lmsysorg/sglang/kimi-k3-c6ad1f26-20260729-arm64

---

*v3 由 architect 基于公开资料 + PyPI/镜像元数据实抓 + NGC release notes 盘点定稿（2026-08-14）。裁决：目标版本 0.5.16、路径 C（容器内 pip --no-deps 升级）、flashinfer 0.6.14 不动。凡标"实测/待确认"项以 sre 执行与 TP1 冒烟为准。*

---

# V4 节：SGLang 0.5.16 升级后 sgl_kernel ABI 不兼容 —— 三方案评估与重打包修复

**日期**：2026-08-14（V4 追加）
**作者**：阿奇（Archi）· 系统架构师
**触发**：Phase 5a 实测失败（08-14 15:15）：0.5.16 镜像启动即死 `undefined symbol: torch::TensorBase::const_data_ptr<int,0>`（`sgl_kernel sm100/common_ops.abi3.so`，sgl_kernel **0.4.5** PyPI wheel）。生产 vLLM 已回滚 healthy；0.5.14 rollback 锚点保留。
**本节约束**：① DSPARK 必须启用（0.5.16）② 保留 NGC 已验证基础栈（torch 2.13.0a0+nv26.07 / flashinfer 0.6.14 / CUDA 13.3.1 / NCCL 2.30.7）③ 生产 vLLM 互斥切换纪律不变。

---

## V4-0. 摘要与裁决（TL;DR）

- **根因（一句话）**：PyPI `sglang-kernel==0.4.5` 按 **torch==2.11** ABI 编译；容器实装 torch **2.13.0a0+nv26.07** 未导出其引用的 `TensorBase::const_data_ptr<int,0>` 符号 → 任何 GPU 加载 `common_ops`（sm100 路径）即 dlopen 失败。
- **裁决**：
  - **主路径 = 方案 A**：保留 NGC 自带 **sgl_kernel 0.4.4+nv26.7**（对 torch 2.13 编译、SM121 已实证可用），只装 sglang 0.5.16 主包，**绝不安装 sglang-kernel==0.4.5**。
  - **兜底 = 方案 C**：sgl-kernel 0.4.5 源码对 torch 2.13 / SM121 重建（仅当 A 冒烟发现缺 API 时启用，成本 1.5-3h）。
  - **排除 = 方案 B**：lmsysorg 0.5.16 镜像基于 torch 2.11 + `TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a`（**无 sm12x**）→ 其 sgl_kernel 既 ABI 不匹配（torch 2.11）又无 SM121 kernel，双重不满足。
- **0.4.4 vs 0.4.5 API 差异结论（关键）**：**sglang 0.5.16 的 CUDA 路径不硬依赖 sgl_kernel 0.4.5 的任何新 API**。0.4.5 新增算子全部为「CPU 专用」（PR #27862：`*_cpu` 投机/拷贝算子，Intel AMX 路径）或「InfLLM-v2 稀疏注意力」（PR #29383：`infllmv2_attn_stage1`/`max_pooling_1d_varlen`，仅 InfLLM 模型用）。0.5.16 CUDA 路径实际调用点（activation/moe_sum_reduce/fast_topk/build_tree_kernel_efficient/concat_mla/rmsnorm/fused_q_norm_rope 系列/transfer_kv_*）**全部在 0.4.4 已存在**，且 sglang 主包无 sgl_kernel 版本运行时校验。
- **⛔ V3 纠偏**：V3 Phase 1 括号里的 `[sglang-kernel==0.4.5 若现有版本旧]` 是 v6 事故的直接指令来源。**本 V4 正式撤销该建议**：NGC 26.07 栈下 `sglang-kernel==0.4.5` 不得安装。
- **发布门槛新增（必须）**：任何 SGLang 镜像发布前必须过「**带 GPU common_ops 冒烟**」——`nvidia-smi` 可见后 `import sgl_kernel`（触发 arch-specific dlopen）+ 1 个真实 kernel 调用。无 GPU 的 `import sgl_kernel` 因无法选 arch 而不加载 common_ops，**无法暴露 ABI 错误**，不算数。

---

## V4-1. 事故复盘与根因

| # | 事实 | 证据/说明 |
|---|---|---|
| F1 | 0.5.16 启动即死：`undefined symbol: torch::TensorBase::const_data_ptr<int,0>` | `sgl_kernel sm100/common_ops.abi3.so` 来自 PyPI `sglang-kernel==0.4.5`（v6 按 V3 指示定向安装） |
| F2 | PyPI sglang-kernel 0.4.5 按 torch==2.11 编译 | PyPI README 明示 `Requires torch == 2.11.0`；0.5.16 requires_dist 锁 `torch==2.11.0`/`sglang-kernel==0.4.5`（pyoven 实抓） |
| F3 | 容器实装 torch **2.13.0a0+9186a08b2c.nv26.07** 未导出该符号 | NGC 26.07 栈（四机 md5 一致） |
| F4 | NGC 原镜像自带 **sgl_kernel 0.4.4+nv26.7**（NVIDIA 定制，对 torch 2.13 编译），**同 GPU SM121 下 common_ops OK** | 0.5.14 容器已在四机跑通；即 0.4.4 的 kernel 面在 SM121 可用 |
| F5 | 加载机制：`import sgl_kernel` → `_load_architecture_specific_ops()` 按 GPU capability 选 dir dlopen | GB10 SM121 无 sm120 dir → 落 sm100 路径 → 才触发 ABI dlopen 失败；**无 GPU 时选不了 arch，不加载 → 无错**（这就是"无 GPU import 验证不够"的机理） |
| F6 | v6 升级序列：`pip install --no-deps sglang==0.5.16` 后**又**装 `sglang-kernel==0.4.5` → 覆盖 NGC 0.4.4 → ABI 崩溃 | 已回滚：生产 vLLM healthy；0.5.16（坏）镜像仍在 registry；0.5.14 rollback 锚点保留 |

**根因归类**：不是"版本号冲突"，而是 **ABI 层面二进制不兼容**——同一版本号 0.4.5 的 wheel 是 torch 2.11 ABI，与 NGC torch 2.13 运行时不匹配。同类问题（`const_data_ptr<int>`）在 PyTorch 大版本升级后 torch 扩展重编译时常见。

---

## V4-2. 关键事实核查：sgl_kernel 0.4.4 vs 0.4.5 API 差异（0.5.16 是否硬依赖 0.4.5）

### V4-2.1 源码面差异（sgl-kernel `__init__.py`，v0.5.14 标签≈0.4.4 vs v0.5.16 标签≈0.4.5）

| 类别 | 0.4.5 相对 0.4.4 的变化 | 是否影响 DSV4/CUDA 路径 |
|---|---|---|
| **新增（CPU 专用）** | `assign_draft_cache_locs_contiguous_cpu`、`assign_extend_cache_locs_cpu`、`assign_req_to_token_pool_cpu`、`build_draft_decode_metadata_cpu`、`build_tree_kernel_efficient_cpu`、`fill_accept_out_cache_loc_cpu`、`fill_bonus_tokens_cpu`、`rotate_input_ids_cpu`、`verify_tree_greedy_cpu`、`copy_all_layer_kv_cache_cpu` | ❌ 仅 CPU（PR #27862：Intel AMX CPU 投机解码）。CUDA 下对应分支不导入 |
| **新增（InfLLM-v2）** | `infllmv2_attn_stage1`、`max_pooling_1d_varlen` | ❌ 仅 InfLLM 稀疏注意力模型（PR #29383）；DSV4 不用；AOT/JIT 内核树 |
| **移除/改名** | `qserve_w4a8_per_chn_gemm`、`qserve_w4a8_per_group_gemm`、`dsv3_router_gemm`、`kimi_k2_moe_fused_gate`、`moe_fused_gate`、`fp8_blockwise_scaled_mm` | 与 0.5.16 移除 QServe/FBGEMM 一致；本方案用 flashinfer_cutlass，不受影响 |
| **两者均有（0.5.16 CUDA 路径实际调用点）** | `gelu_and_mul`/`gelu_tanh_and_mul`/`silu_and_mul`、`moe_sum_reduce`、`fast_topk`/`fast_topk_v2`/`fast_topk_transform_fused`、`build_tree_kernel_efficient`、`concat_mla_absorb_q`/`concat_mla_k`、`rmsnorm`/`fused_add_rmsnorm`、`fused_q_norm_rope`/`fused_k_norm_rope_flashmla`/`fused_q_indexer_rope_hadamard_quant`、`transfer_kv_all_layer(_mla)`/`transfer_kv_per_layer(_mla)`、`topk_softmax`/`moe_align_block_size`/`prepare_moe_input`、`segment_packbits`/`verify_tree_greedy`/`tree_speculative_sampling_target_only` | ✅ **全部 0.4.4 已存在** |

### V4-2.2 sglang 0.5.16 主包对 sgl_kernel 的调用点核查（v0.5.16 标签源码 + grep.app 交叉）

| 文件 | 导入（CUDA 分支） | 0.4.4 有？ |
|---|---|---|
| `srt/layers/activation.py` | `gelu_and_mul`/`gelu_tanh_and_mul`/`silu_and_mul` | ✅ |
| `srt/layers/moe/moe_runner/triton_utils/fused_moe.py` | `moe_sum_reduce`/`silu_and_mul` | ✅ |
| `srt/speculative/spec_utils.py` | `fast_topk`（CUDA）；`assign_extend_cache_locs_cpu`（仅 CPU） | ✅ |
| `srt/speculative/eagle_utils.py` | `build_tree_kernel_efficient`（CUDA）；`*_cpu`（仅 CPU） | ✅ |
| `srt/layers/attention/dsa/dsa_topk_backend.py` | `fast_topk_v2`/`fast_topk_transform_fused` | ✅ |
| `srt/layers/attention/dsv4/...`（elementwise） | `fused_q_norm_rope`/`fused_k_norm_rope_flashmla`/`fused_q_indexer_rope_hadamard_quant` | ✅ |
| `multimodal_gen/runtime/layers/layernorm.py` | `fused_add_rmsnorm`/`rmsnorm` | ✅ |
| `srt/speculative/dspark_components/*`（dspark_verify/accept/worker_v2） | **不直接引用 sgl_kernel**；DSPARK 验收/窗口内核为 Triton（`accept_greedy_triton` 等） | — |

**附加核查**：sglang 主包**无** `sgl_kernel.__version__` / `sglang-kernel` 版本运行时校验（grep 全仓无命中）→ 不会因版本号 <0.4.5 拒绝启动。

### V4-2.3 结论

- **sglang 0.5.16 不硬依赖 sgl_kernel 0.4.5 的新 API**；0.4.4 覆盖 0.5.16 CUDA 路径全部调用点。
- 残余风险仅剩：NGC fork `0.4.4+nv26.7` 与社区 0.4.4 的 API 面差异（NVIDIA 定制可能缺个别社区算子）→ **用带 GPU 冒烟 + TP1 兜底**（见 V4-5/V4-6）。

---

## V4-3. 三方案对比表

| 维度 | **A：保留 NGC sgl_kernel 0.4.4+nv26.7，只装 sglang 0.5.16 主包** | **B：从 lmsysorg/sglang:v0.5.16 提取 wheel** | **C：源码重建 sgl-kernel 0.4.5 for torch 2.13/SM121** |
|---|---|---|---|
| ABI 兼容（torch 2.13.0a0+nv26.07） | ✅ 0.4.4+nv26.7 就是 torch 2.13 编译 | ❌ 上游 0.5.16 锁 torch 2.11 → 其 sgl_kernel 仍 torch 2.11 ABI → 同款 `const_data_ptr` 崩溃 | ✅ 对 torch 2.13 现场编译 |
| SM121 kernel 支持 | ✅ 同 GPU 已实证（0.5.14 容器跑通，common_ops 落 sm100 路径） | ❌ 镜像 `TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a`（dockerhub 元数据实抓）**无 sm12x** → 即使绕过 ABI 也 `no kernel image` | ⚠️ 需 `TORCH_CUDA_ARCH_LIST="12.1a"`（或 12.0a 12.1a）显式编译，NVIDIA 未校验该组合 |
| 0.5.16 所需 API 覆盖 | ✅ CUDA 路径调用点全在 0.4.4（V4-2）；残差= NGC fork 差异 | ⚠️ 不解决 ABI，无从谈覆盖 | ✅ 0.4.5 源码全覆盖（含 0.4.4 全部） |
| 改动面 | **1 个 Python 主包**（`--no-deps`）；torch/flashinfer/NCCL/transformers 零/按需 | 拉 20.8GB×1 镜像 + 提取 wheel + 冒烟；torch 栈仍不匹配 | 整个 sgl-kernel CUDA 源码构建；可能连带 0.5.16 主包也需源码编译 |
| 成本 | 🟢 **分钟级**（wheel 14.4MB，build 1-2 轮） | 🟠 小时级（镜像下载/提取）但**无效** | 🔴 1.5-3h（20 核）+ 编译期/链接期排错风险；内存峰值风险 |
| 风险 | 低：ABI 已知匹配；残差= NGC fork API 面 | **高且无效**：双重不满足 | 中高：CUDA 13.3.1 × torch 2.13 × aarch64 组合未经 NVIDIA 验证；pTXAS/SM121 编译坑 |
| 回滚 | 改 tag，分钟级 | 无意义 | 复杂（需保留编译环境） |
| **结论** | 🟢 **推荐主路径** | 🔴 **排除** | 🟡 **兜底（仅当 A 冒烟缺 API）** |

**方案 B 排除的硬证据**：
- 上游 0.5.16 锁 `torch==2.11.0`（pyproject 实抓）→ 镜像内 sgl_kernel 为 torch 2.11 ABI → 与 NGC 2.13 栈**同样不匹配**（不解决事故）。
- dockerhub kimi-k3-c6ad1f26-20260729-arm64 元数据：`CUDA_VERSION=13.0.1`、`TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a` → 构建目标为 GH200/B200/B300，**无 SM120/SM121 kernel** → GB10 SM121 上连 kernel 都没有。
- 结论：B 同时踩 ABI 与 SM121 两个坑，无继续价值。

---

## V4-4. 推荐方案与理由

**推荐：方案 A（主）+ 方案 C（兜底），排除 B。**

理由（约束先行）：
1. **保留 NGC 已验证基础栈**：torch 2.13.0a0+nv26.07 / flashinfer 0.6.14 / CUDA 13.3.1 / NCCL 2.30.7 LD_PRELOAD 全部不动（同 V3 路径 C 的目标）。
2. **ABI 是硬约束**：只有 NGC 0.4.4+nv26.7（torch 2.13 编译）满足；PyPI 0.4.5 与 lmsysorg 产物均 torch 2.11，直接出局。
3. **API 面已核查**：0.5.16 CUDA 路径调用点全在 0.4.4 已有（V4-2），0.4.5 新增算子与本用例（DSV4-Flash-NVFP4+DSPARK）无关。
4. **务实胜过完美**：A 改动最小（1 个 Python 包）、分发最快、回滚最简；C 的源码构建在 aarch64×CUDA 13.3.1×SM121 上风险不可控，仅作最终兜底。
5. **发布门槛升级**：本次事故证明"无 GPU import 验证"不可靠；方案 A 自带新的「带 GPU common_ops 冒烟」验证（V4-6），正好把门槛补上。

---

## V4-5. 重打包执行步骤（方案 A）

### V4-5.1 Dockerfile（02 构建机）

```dockerfile
# Dockerfile.sglang-0.5.16-abi2
# 基线：NGC 26.07（内部 sglang 0.5.14+nv26.7 / sgl_kernel 0.4.4+nv26.7 / torch 2.13.0a0+nv26.07）
# ⛔ 关键纪律：绝不安装 sglang-kernel==0.4.5（PyPI，torch 2.11 ABI）——保留 NGC sgl_kernel 0.4.4+nv26.7
FROM nvcr.io/nvidia/sglang:26.07-py3

# 预先 vendor wheel 到 02 /tmp/wheels（PyPI/代理）：sglang-0.5.16-cp312-cp312-manylinux_2_34_aarch64.whl（14.4MB）
ARG SGLANG_WHEEL=sglang-0.5.16-cp312-cp312-manylinux_2_34_aarch64.whl
COPY wheels/${SGLANG_WHEEL} /tmp/wheels/

RUN pip install --no-deps /tmp/wheels/${SGLANG_WHEEL} \
    # 按需（仅当模型加载报 transformers/其他 版本错误时取消注释）：
    # && pip install --no-deps transformers==5.12.1 \
    && rm -rf /tmp/wheels \
    && python3 -c "import sglang, sgl_kernel, torch, flashinfer; print('sglang', sglang.__version__); print('sgl_kernel', sgl_kernel.__version__); print('torch', torch.__version__); print('flashinfer', flashinfer.__version__)"
# 预期输出：
#   sglang 0.5.16
#   sgl_kernel 0.4.4+nv26.7        ← 必须是 NGC 定制版，不是 0.4.5
#   torch 2.13.0a0+9186a08b2c.nv26.07
#   flashinfer 0.6.14
```

### V4-5.2 构建 / 推送 / 分发

```bash
# 02：
cd <INSTALL_DIR>/backup/sglang-0516-repack
docker build -f Dockerfile.sglang-0.5.16-abi2 -t <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2 .
docker push <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2
# 四机 pull（错峰，勿与 push 并发）：
#   for h in dgxspark0{1..4}; do ssh $h docker pull <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2; done
# 本地运行 tag：
#   docker tag <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2 sglang-nvfp4:0.5.16
# 记录镜像 digest，四机核对一致（docker inspect --format '{{index .RepoDigests 0}}'）
```

### V4-5.3 四机版本验证（不带 GPU 的快速层，不够但先做）

```bash
docker run --rm --entrypoint bash sglang-nvfp4:0.5.16 -c '
python3 -c "import sglang,sgl_kernel,torch,flashinfer;print(sglang.__version__,sgl_kernel.__version__,torch.__version__,flashinfer.__version__)"
# 期望：0.5.16  0.4.4+nv26.7  2.13.0a0+9186a08b2c.nv26.07  0.6.14
grep -c "sglang-kernel==0.4.5\|sglang_kernel-0.4.5" /dev/null 2>/dev/null; pip list 2>/dev/null | grep -i sglang
'
```

### V4-5.4 启动脚本参数（沿用 V3-4 终稿，仅镜像 tag 改为 `0.5.16-nvfp4-spark-abi2`）

- `--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --fp4-gemm-backend flashinfer_cutlass --moe-runner-backend flashinfer_cutlass --quantization modelopt_fp4 --tp-size 4 --nnodes 4 --mem-fraction-static 0.90 --chunked-prefill-size 4096 --max-model-len 65536 --enable-metrics`
- env：`SGLANG_RAGGED_VERIFY_MODE=compact`、`FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"`、`SGLANG_DISABLE_DEEP_GEMM=1`、`SGLANG_ENABLE_DEEP_GEMM=0`、`SGLANG_SM120_TRITON_FLASHMLA=1`（NCCL 环网 env 全部保留）。
- 切换走 **A/B 互斥**（停 vLLM TP4 → GPU 门禁 → SGLang；正式窗口由主理人排期）。

---

## V4-6. 发布门槛（新增，必须）：带 GPU common_ops 冒烟

**为什么必须带 GPU**：`import sgl_kernel` 时 `_load_architecture_specific_ops()` 需 GPU capability 才能选 dir（SM121 无 sm120 → 落 sm100）并 dlopen `common_ops`；无 GPU 时选不了 arch → 不加载 → ABI 错误被掩盖。**本次事故正是被"无 GPU import 验证"放行**。

**门槛命令（容器内，GPU 可见）**：

```bash
docker run --rm --gpus all --entrypoint bash <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2 -c '
set -e
echo "== 1) GPU 可见 ==" && nvidia-smi -L | head -1          # GB10
echo "== 2) torch ==" && python3 -c "import torch;print(torch.__version__)"   # 必须 2.13.0a0+nv26.07
echo "== 3) import sgl_kernel（⚠️ 关键：触发 common_ops dlopen，ABI 错误在此暴露）==" \
  && python3 -c "import sgl_kernel;print(sgl_kernel.__version__)"              # 必须 0.4.4+nv26.7
echo "== 4) import sglang ==" && python3 -c "import sglang;print(sglang.__version__)"  # 必须 0.5.16
echo "== 5) 真实 kernel 执行（SM121）==" && python3 - <<"PY"
import torch, sgl_kernel
assert torch.cuda.is_available(), "no GPU"
x = torch.randn(8, 8, device="cuda"); w = torch.randn(8, device="cuda")
# 简单 op 冒烟（签名以容器内 0.5.16 源码为准；若 rmsnorm 签名不符改调 fast_topk 等）：
out = sgl_kernel.rmsnorm(x, w, 1e-5)[0]
print("rmsnorm OK", tuple(out.shape))
print("COMMON_OPS_SM121_OK")
PY
'
```

**判定**：①-⑤ 全过 = 本镜像 ABI 兼容 + SM121 kernel 可执行 → 才允许进入 TP1 冒烟（K1-K8 + DSPARK 生效 + R9 接受率）→ TP4 环网冒烟 → 正式发布。任一失败 → 停止并按 V4-7 回滚/转 C。

---

## V4-7. 回滚锚点

| # | 锚点 | 说明 |
|---|---|---|
| R1 | `nvcr.io/nvidia/sglang:26.07-py3`（原始） | 不动，始终可回 0.5.14 栈 |
| R2 | `<NODE_IP>:5000/sglang/sglang:0.5.14-nv26.07-rollback` | 0.5.14 回滚锚点，保留 |
| R3 | 坏镜像 `…:0.5.16-nvfp4-spark`（含 sgl_kernel 0.4.5） | 标记 deprecated / 保留取证；确认后可删，不影响回滚 |
| R4 | 新镜像 `…:0.5.16-nvfp4-spark-abi2` | **只增不改**；digest 四机核对 |
| R5 | 脚本/权重/NCCL/网络栈 | **零变更**（方案 A 不动 torch/flashinfer/NCCL/CUDA） |
| R6 | 与生产 vLLM 互斥 | A/B 切换序列沿用 V2-5.1；正式切换窗口主理人排期 |

---

## V4-8. 风险增量（相对 V3 表）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| V4-R1 | 🟠 **NGC fork 0.4.4+nv26.7 与社区 0.4.4 API 面差异**（可能缺个别社区算子） | 中 | 带 GPU 冒烟（V4-6）+ TP1 全项；缺 API 报 `ImportError` 点名符号 → 转 C |
| V4-R2 | 🟠 **transformers 5.8.1（NGC）vs 5.12.1（0.5.16 锁）** 版本差导致模型加载 config 错误 | 中 | 0.5.16 主包 `--no-deps` 安装后 TP1 试加载；报错则 `--no-deps` 补装 `transformers==5.12.1` |
| V4-R3 | 🟡 **0.5.16 需要而 0.4.4 未提供的其他新 op（残余）** | 低 | 已核查 CUDA 路径全部调用点（V4-2）；DSPARK 用 Triton 不依赖 sgl_kernel 新 API |
| V4-R4 | 🔴 **再次误装 sglang-kernel==0.4.5**（V3 指令残留/他人操作） | 高 | V4 撤销 V3 括号建议；Dockerfile 注释警示；装后强制版本核对（0.4.4+nv26.7） |
| V4-R5 | 🟡 **发布门槛被跳过**（无 GPU import 验证放行） | 高 | V4-6 列为发布必过门槛；SRE checklist 同步更新 |

---

## V4-9. 待 sre / testing 实测闭环清单

1. 【sre】02 构建 `0.5.16-nvfp4-spark-abi2` → push → 四机 pull + digest 核对。
2. 【sre】四机执行 V4-5.3（无 GPU 版本核对）+ **V4-6 带 GPU common_ops 冒烟**（新门槛）。
3. 【sre】容器内 `launch_server --help` 复核 DSPARK / fp4-gemm / moe-runner / tp-size / metrics（沿用 V3-7）。
4. 【testing】维护窗口内 TP1 冒烟（K1-K8 + DSPARK 生效 + R9 接受率 ≥0.40，<0.20 FAIL）。
5. 【testing】TP4 环网冒烟 + prefill A/B（对照 vLLM 基线）。
6. 【team-lead】正式 TP4 启动窗口排期（生产 vLLM 互斥）。

---

## V4-10. 证据链接（V4 新增）

- sglang-kernel PyPI（0.4.4/0.4.5：`Requires torch == 2.11.0`）：https://pypi.org/project/sglang-kernel/0.4.4/ 、https://pypi.org/project/sglang-kernel/0.4.5/
- sglang 0.5.16 PyPI requires_dist（torch==2.11.0 / sglang-kernel==0.4.5 / flashinfer 0.6.14）：https://pyoven.org/package/sglang 、https://pypi.org/project/sglang/0.5.16/
- sgl-kernel 源码面差异（__init__.py @ v0.5.14 vs v0.5.16）：https://raw.githubusercontent.com/sgl-project/sglang/v0.5.16/sgl-kernel/python/sgl_kernel/__init__.py
- PR #27862（CPU 投机解码 → 新增 *_cpu 算子）：https://github.com/sgl-project/sglang/pull/27862
- PR #29383（InfLLM v2 注意力 kernel → 新增 infllmv2_attn_stage1）：https://github.com/sgl-project/sglang/pull/29383
- lmsysorg/sglang kimi-k3-arm64（CUDA 13.0.1 / TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a）：https://hub.docker.com/layers/lmsysorg/sglang/kimi-k3-c6ad1f26-20260729-arm64
- sgl-kernel 架构预编译说明（SM80/89/90/100/120 分目录，非 SM121）：https://aiwiki.ai/wiki/sglang

---

*v4 由 architect 基于事故复盘 + PyPI/镜像元数据实抓 + sgl-kernel 源码面 diff + sglang 0.5.16 调用点核查定稿（2026-08-14）。裁决：方案 A（保留 NGC sgl_kernel 0.4.4+nv26.7，只装 sglang 0.5.16 主包）为主、方案 C（源码重建）兜底、方案 B 排除；新增「带 GPU common_ops 冒烟」发布门槛。凡标"实测/待确认"项以 sre 执行与 TP1 冒烟为准。*

---

# V5 节：DSpark SM120 路由修复 —— 三方案评估与定案（topk=192 / num_tokens>64 阻断项）

**日期**：2026-08-15（V5 追加）
**作者**：阿奇（Archi）· 系统架构师
**触发**：Phase 5a 重跑失败（sre 实测 08-15 23:20，3 轮启动 FAIL）：SGLang 0.5.16 + flashinfer 0.6.14 在 SM121 的 DSpark draft verify 兼容性缺口（topk=192 不在实例化桶内 + `num_tokens>64` 断言），**非启动参数可解**，需代码层修复。
**本节约束**：① DSPARK 必须启用（用户硬性要求，STANDALONE 仅评估对照）② 保留 NGC 已验证基础栈（torch 2.13.0a0+nv26.07 / flashinfer 0.6.14 / CUDA 13.3.1 / NCCL 2.30.7 / sgl_kernel 0.4.4+nv26.7）③ 80G 上限运行期验收必须达标（容器 ≤80G、free ≥30G）④ abi2 镜像基础不变，增量最小化。

---

## V5-0. 摘要与裁决（TL;DR）

- **根因（一句话）**：DSPARK draft indexer 发射 `topk=192`，不在 flashinfer 0.6.14 DSV4 sparse-MLA 实例化表内（decode `{128,512,1024}`、prefill `{128,512,1024,2048}`）；`_flash_mla_flashinfer`（`flash_mla_sm120.py:459` 调用点）对 ≤64 token 的 draft verify 落入 flashinfer `_paged_attention` 内部 dispatch → `_decode_dsv4_dispatchable()` 失败 → **fall-through 到 prefill orchestrator** → prefill 断言 `num_tokens>64` → 崩溃。CUDA graph capture（B>64）则被 prefill 配置检查拒绝（`(64,192)` 无实例化）。
- **上游核查结论（关键）**：**没有任何已发布版本修复此问题**。上游 PR **#33407**（Fix DSPARK SM120 decode dispatch for non-instantiated topk widths）精确对口，但**仍未合并**（target main，open 状态）；**v0.5.17 与 main 均已核实不含该修复**（`_next_topk_bucket` 缺失）；flashinfer #4309 已关闭被 #4380 取代（需 flashinfer 版本升级 + 重编，且 0.6.16 在 SM120 graph capture 会 segfault）。**0.5.17 升级路径同时失效**：既不含修复，又需连带 flashinfer 0.6.15.post1 + torch ABI 风险（V3 结论维持）。
- **裁决**：**方案 A = SGLang 本地 patch（移植 PR #33407 逻辑到 0.5.16）为唯一可行路径**。纯 Python 层，~95 行，不改 flashinfer、不重编 kernel、不动 torch ABI；已实例化宽度完全透传，主模型热路径零改动。
- **落地**：abi2 基础 Dockerfile 重打包 → tag **`0.5.16-nvfp4-spark-abi3`**；补丁脚本已产出 `scripts/patch_dspark_sm120_topk192.py`（锚点式、幂等、带 dry-run）。
- **关联风险（必须同步处理）**：上游 issue **#33800** 报告 SM120 上 **DSPARK depth=5（即本方案 `--speculative-dspark-block-size 5`）损坏输出**（根因 NCCL symmetric pool einsum 瞬时张量碰撞，修复 #29927/#34021）。**sm_121 无数据、未确认**，但本机恰用 block-size 5 → 必须加入损坏冒烟验证 + 回退预案（block-size 4 或并打 #34021）。

---

## V5-1. 失败链与根因（精确到 dispatch 表）

### V5-1.1 sre 实测三连败

| 轮次 | 时间 | 动作 | 结果 |
|---|---|---|---|
| 1 | 22:34 | abi2 镜像原样启动 | `sglang-kernel 0.4.4+nv26.7 < 要求 0.4.5` 断言失败 → 官方开关 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 已解（保留，非缺陷） |
| 2 | 22:36 | +skip 开关重启 | DSpark verify CUDA graph capture：`Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=64 topk=192 page_block_size=64` → 加 `--disable-cuda-graph` |
| 3 | 23:05 | +`--disable-cuda-graph` | 运行时 draft verify：`Check failed: num_tokens > 64 (5 vs. 64)`（`flash_mla_sm120.py:459`） |

网络/NCCL/TCPStore/权重加载/DSPARK draft runner/FlashInfer autotune 均正常 → **唯一缺陷 = DSpark draft verify 的 SM120 sparse-MLA 路由**。

### V5-1.2 dispatch 表核查（flashinfer 0.6.14 `_sparse_mla_sm120.py` 实抓）

```python
_DECODE_MAX_TOKENS = 64                     # num_tokens > 64 → prefill orchestrator
_DECODE_DSV4_DISPATCH = frozenset({         # DSV4 decode 实例化表
    (8,128),(8,512),(8,1024), (16,128),(16,512),(16,1024),
    (32,128),(32,512),(32,1024), (64,128),(64,512),(64,1024),
    (128,128),(128,512),(128,1024),
})
# prefill orchestrator: {128,512,1024,2048} × {16,32,64,128} @ page_block_size=64

def _decode_dsv4_dispatchable(num_tokens, num_heads, topk, d_qk, page_block_size, extra_topk=0):
    return (num_tokens <= 64 and d_qk == 512 and page_block_size == 64
            and (num_heads, topk) in _DECODE_DSV4_DISPATCH)
```

**失败机理**：SGLang `_flash_mla_flashinfer`（v0.5.16）对任何 B 直接调 flashinfer `_paged_attention`；`(64,192)` 不在 decode 表 → 跳过 decode → **无条件 fall-through 到 `module.sparse_mla_sm120_paged_attention`（prefill 编排器）** → prefill 对 ≤64 token 断言崩溃；对 graph capture（B>64）则 prefill 配置检查拒绝 `(64,192)`。**双失败同源**：192 不在任何实例化桶。

### V5-1.3 上游同款问题确认

- **#33134**（DGX Spark sm_121 TP=2，DeepSeek-V4-Flash-0731 + DSPARK）：一模一样的 topk=192 失败，作者本地 workaround = 把 idx 192→512 零填充后**可正常跑通 sm_121**（零填充因 topk_length 封顶扫描而不会被读取，故未破坏正确性）。
- **#33407**（PR，作者 hassellof，4× RTX PRO 6000 sm_120）：精确修复——192→512 右填充 **-1 skip 哨兵** + topk_length 封顶 + Triton 兜底；含封闭式单测（`CUDA_VISIBLE_DEVICES=99` CPU 跑）。**仍未合并**（8× RTX 5090 TP=8、sm_121、4× RTX PRO 6000 三个环境独立复现后 open）。
- **flashinfer #4309**（feat topk=192）已 closed，被 **#4380** 取代；且 flashinfer 0.6.16 在 SM120 graph capture segfault → **flashinfer 侧短期内不可依赖**。

---

## V5-2. 上游修复核查（方案 B）

| 核查项 | 结论 | 证据 |
|---|---|---|
| 0.5.17 是否含修复 | ❌ **不含** | v0.5.17 `flash_mla_sm120.py` 实抓：无 `_next_topk_bucket`/`_SUPPORTED_TOPK_WIDTHS`，仍仅 split-K 逻辑 |
| main 是否已合并 | ❌ **未合并** | main 同文件实抓：无 bucket 填充逻辑；PR #33407 仍 open |
| 0.5.18 / 更高版本 | ❌ **不存在** | GitHub releases：v0.5.17 为 Latest（2026-08-08），无 0.5.18 |
| NGC 26.08 | ❌ 未发布 | V3 结论维持（截至 08-15 无 26.08） |
| 升级 0.5.17 的代价 | 🔴 高且无效 | 需连带 flashinfer 0.6.15.post1 + torch 2.13 ABI 风险（V3-2 路径 D 结论维持）；**且升级后缺陷仍存在**（不含 #33407） |
| cherry-pick #33407 到 0.5.16 | 🟢 可行 | PR 为 95 行纯 Python 新增、无删除，`flash_mla_sm120.py` 在两版本间结构高度一致（v0.5.16 与 PR base 差异仅 dispatch 检查位置，见 V5-3） |

**结论：方案 B（等上游/升级）不可行，时机不可控且升级不改问题。方案 A（本地 patch = 移植 #33407）为必然选择。**

---

## V5-3. 三方案对比表

| 维度 | **A：SGLang 本地 patch（移植 #33407）** | **B：上游升级（0.5.17+ 等修复）** | **C：flashinfer 侧加 topk=192** |
|---|---|---|---|
| 可行性 | ✅ **高**：纯 Python 层，锚点式脚本注入，已验证可插入 | ❌ 0.5.17/main 均无修复；PR open 无时间表；0.5.18 不存在 | ❌ #4309 已关闭被 #4380 取代；需 flashinfer 版本升级/重编；0.6.16 SM120 graph capture segfault |
| 风险 | 🟢 **低**：已实例化宽度（128/512/1024/2048）透传零改动；热路径（topk=2048 decode）不受影响；sm_120 作者实测 + sm_121 workaround 实证 | 🔴 高：flashinfer 0.6.15.post1 连带 ABI 风险；升级后缺陷仍在 | 🔴 高：重编 flashinfer（aarch64×CUDA13.3×SM121 未验证）；Docker/镜像面大改；版本锁定被打乱 |
| 成本 | 🟢 **分钟级**：1 个 .py 文件 + 重打包 abi3（增量镜像 <1MB）+ 四机分发 | 🔴 小时级~天级且**无效** | 🔴 小时级~天级，编译/链接/验证链条长 |
| 对热路径影响 | 无（透传） | 无（但没用） | 无（但不可达） |
| 回滚 | 改 tag（abi2 完好保留） | 无意义 | 复杂 |
| 单测 | ✅ 封闭式单测可移植（PR 自带 278 行） | — | PR 自带 sm_120 单测（需卡） |
| **结论** | 🟢 **推荐** | 🔴 **排除** | 🔴 **排除** |

**附：`SGLANG_SM120_FLASHMLA_BACKEND=triton` 立即解阻（非最终）**
- 0.5.16 的有效环境变量是 **`SGLANG_SM120_FLASHMLA_BACKEND`**（默认 `flashinfer`，`environ.py` 实抓）；**`SGLANG_SM120_TRITON_FLASHMLA=1` 在 0.5.16 已不存在/无效**（V3-4 env 里的旧变量应清理）。
- 设 `SGLANG_SM120_FLASHMLA_BACKEND=triton` 可立即绕过崩溃（8×RTX5090 TP8 实证），但**把包括热路径 topk=2048 在内的所有 sparse-MLA 都路由到 Triton**，吞吐损失——只作诊断/临时解阻，不作最终方案。

---

## V5-4. 推荐方案与理由

**推荐：方案 A——移植 PR #33407 到 0.5.16（主路径）；方案 B/C 排除；STANDALONE 仅作回退对照。**

理由（约束先行）：
1. **DSPARK 必须启用** → 唯一同时满足"保留 0.5.16 + flashinfer 0.6.14 组合"且修复路由的是本地 patch。
2. **改动面最小**：1 个 Python 文件 `flash_mla_sm120.py`，纯 Python 逻辑（桶对齐 + topk_length 封顶 + Triton 兜底），不动 kernel/flashinfer/torch/NCCL，符合 V4 已固化的"不动已验证基础栈"纪律。
3. **风险最低**：已实例化宽度完全透传（PR 有中性保证单测）；sm_120 作者 12,334 请求零损坏验证；sm_121 有等效 workaround（#33134，零填充版）实证可跑。
4. **可回滚可退休**：abi2 保留为回滚锚点；上游 #33407 合并后，本补丁与上游同构，可直接退役/被 flashinfer #4380 原生 192 取代。
5. **对热路径正确性透明**：192→512 填充的 -1 哨兵 + topk_length=192 封顶 = 内核永不读取填充位，注意力数学不变（PR 精度 atol=5e-2 单测）。

---

## V5-5. Patch 草案与落地方式

### V5-5.1 改动文件与锚点（v0.5.16 wheel）

**文件**：`site-packages/sglang/kernels/ops/attention/flash_mla_sm120.py`

**三处插入（锚点式，见 `scripts/patch_dspark_sm120_topk192.py`）**：

**① 模块级**（锚点 `_sm120_default_backend = envs.SGLANG_SM120_FLASHMLA_BACKEND.get()` 之后）：
```python
_SUPPORTED_TOPK_WIDTHS = (128, 512, 1024, 2048)
_noted_bucket_pad = False
_warned_triton_fb = False

def _next_topk_bucket(topk: int):
    """Smallest instantiated topk width >= topk, or None if wider than every kernel."""
    return next((w for w in _SUPPORTED_TOPK_WIDTHS if w >= topk), None)
```

**② `_flash_mla_flashinfer` 内、`extra_idx` squeeze 之后**（桶对齐；d_qk==512 且 topk 非实例化才触发）：
```python
    _topk = idx.shape[-1]
    _d_qk = q.shape[-1]
    if _d_qk == 512 and _topk not in _SUPPORTED_TOPK_WIDTHS:
        _next_w = _next_topk_bucket(_topk)
        if _next_w is not None:
            if topk_length is None:
                topk_length = torch.full((B,), _topk, dtype=torch.int32, device=dev)
            idx = torch.nn.functional.pad(idx, (0, _next_w - _topk), value=-1)
            logger.info("SM120 sparse-MLA: padding topk %d -> %d (-1 skip sentinel; scan capped via topk_length).", _topk, _next_w)
```

**③ 同函数 `if B <= _FI_DECODE_MAX_TOKENS:` 之后**（decode 规模仍不可分派 → Triton 兜底，防 prefill 断言）：
```python
        from flashinfer.mla._sparse_mla_sm120 import _decode_dsv4_dispatchable
        _extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
        if not _decode_dsv4_dispatchable(B, H, idx.shape[-1], _d_qk, _PBS_DST, _extra_topk):
            from sglang.kernels.ops.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
            out, lse = flash_mla_sparse_decode_triton(q, k_cache, indices, topk_length, attn_sink,
                                                      head_dim_v, softmax_scale, extra_k_cache,
                                                      extra_indices, extra_topk_length)
            return (out, lse)
```

**机理**：填充后 `idx` 宽度=512 → flashinfer `_paged_attention` 内 `_decode_dsv4_dispatchable(5,64,512,512,64)` = True → 走 **CUTLASS decode 内核**（不再 fall-through 到 prefill）；`topk_length=192` 使内核只扫描 192 个真实候选，-1 填充永不读取。graph capture（B>64）场景：填充后 `(64,512)` 命中 prefill 实例化表 → capture 通过。**故 `--disable-cuda-graph` 应移除恢复默认**（若仍失败再设回）。

### V5-5.2 落地方式对比

| 方式 | 说明 | 适用 |
|---|---|---|
| **Dockerfile 重打包 → abi3（推荐）** | FROM abi2 + COPY 补丁脚本 + RUN python3 打补丁 + 断言验证 + 重打 tag | **生产主路径**：可复现、可审计、digest 四机可核对、回滚=改 tag |
| 运行时 overlay 挂载 | `-v <主机>flash_mla_sm120.py:<site-packages>/...:ro` 覆盖 | **仅诊断/快速 A/B**：四机文件须一致，镜像 digest 不体现差异，不作生产 |
| 容器内改 + docker commit | ad-hoc | ❌ 不推荐（不可复现） |

**Dockerfile.sglang-0.5.16-abi3（草案，02 构建机）**：
```dockerfile
FROM <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2
COPY scripts/patch_dspark_sm120_topk192.py /tmp/
RUN SP=$(python3 -c 'import site;print(site.getsitepackages()[0])') \
 && python3 /tmp/patch_dspark_sm120_topk192.py --site-packages "$SP" \
 && python3 -c "import sglang.kernels.ops.attention.flash_mla_sm120 as m; assert m._next_topk_bucket(192)==512; assert hasattr(m,'_flash_mla_flashinfer'); print('PATCH_OK topk192->', m._next_topk_bucket(192))" \
 && rm /tmp/patch_dspark_sm120_topk192.py
```
```bash
docker build -f Dockerfile.sglang-0.5.16-abi3 -t <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi3 .
docker push <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi3
# 四机 pull + digest 核对（沿用 V4-5.2 纪律）
```

### V5-5.3 启动参数/env 变更（相对 V3-4/V4-5.4）

```bash
# 移除（恢复默认，让 graph capture 走通）：
#   --disable-cuda-graph            ← 删除；若 capture 仍失败再设回
# 移除（0.5.16 已无效的旧变量，避免误导）：
#   SGLANG_SM120_TRITON_FLASHMLA=1  ← 删除（0.5.16 真实变量是 SGLANG_SM120_FLASHMLA_BACKEND）
# 保留：
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1   ← 仍必须（sgl_kernel 0.4.4 断言豁免）
#   SGLANG_SM120_FLASHMLA_BACKEND 不设置      ← 默认 flashinfer = CUTLASS 热路径
#   --speculative-algorithm DSPARK --speculative-dspark-block-size 5  ← 用户硬性要求（风险见 V5-7 R4）
```

---

## V5-6. 重跑验证清单（含 80G 上限运行期验收）

| # | 步骤 | 判定/门槛 | 责任人 |
|---|---|---|---|
| 1 | 补丁验证（无 GPU 层）：容器内 `python3 -c "import sglang.kernels.ops.attention.flash_mla_sm120 as m; print(m._next_topk_bucket(192))"` | 输出 `512`；`hasattr(m,'_flash_mla_flashinfer')` True | sre |
| 2 | 带 GPU common_ops 冒烟（沿用 V4-6 门槛，含 rmsnorm 调用） | ①-⑤ 全过 | sre |
| 3 | TP1 启动（单机或 TP4 head-first）：DSPARK + block-size 5，**无 `--disable-cuda-graph`** | 服务达 /health 200；**Draft verify CUDA graph capture 成功（target + draft）**；日志无 `num_tokens > 64` / `topk=192` 报错 | sre |
| 4 | 运行时 draft verify 冒烟：发 1 个 chat + 1 个 tool-call 请求 | 正常返回；日志无 `Check failed: num_tokens` | sre/testing |
| 5 | **损坏冒烟（新增，针对 block-size 5）**：连发 ≥30 请求（含结构化/JSON/工具调用），检测重复循环（`for for for`）、token 乱码、`</think>` 泄漏、`finish_reason` 异常；辅助检测器：Decode 批量 accept length ≤2.0 区间出现即告警 | **0 损坏事件**；有 1 例即停并转预案（block-size 4 或并打 #34021，见 V5-7 R4） | testing |
| 6 | DSPARK 生效判定：accept length、decode 相对 STANDALONE 加速比 | accept length ≥3（参考）；相对 STANDALONE 有正加速 | testing |
| 7 | **80G 运行期验收（必须达）**：`docker stats`/`ps` 观察容器 RSS ≤80G；`free -g` 余量 ≥30G；UMA×cgroup 兼容性观察（CUDA 分配未报 cudaErrorMemoryAllocation） | 容器 ≤80G 且 free ≥30G | sre |
| 8 | TP4 环网启动（01 head + 02/04/03 worker）+ K1-K8 全链路冒烟 + L3 | 沿用既有 checklist | testing |
| 9 | 四机镜像 digest 核对 + abi2 回滚锚点保留确认 | digest 一致；abi2 不动 | sre |
| 10 | vLLM 恢复（由主理人排期） | 生产 vLLM healthy | team-lead |

---

## V5-7. 风险增量（相对 V4 表）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| V5-R1 | 🟠 **补丁与 wheel 实际内容锚点不匹配**（v0.5.16 wheel 与上游 raw 可能存在微小差异） | 中 | 补丁脚本锚点式 + 找不到锚点即抛错（fail-safe，不写坏文件）；dry-run 先跑；失败回传锚点原文 |
| V5-R2 | 🟡 **topk_length 语义**：合成 `torch.full((B,),192)` 与调用方自带的 topk_length 冲突 | 低 | 仅当 `topk_length is None` 才合成（PR 语义）；已有值则透传，填充位由 -1 哨兵兜底 |
| V5-R3 | 🟡 **Triton 兜底路径未被本配置触发**（padding 已覆盖 192），若未来触发可能不熟 | 低 | 本配置实际不经过；作为安全网保留；封闭单测覆盖分派逻辑 |
| V5-R4 | 🟠 **DSPARK depth=5 损坏输出（#33800，SM120 实证）在 sm_121 无数据** | 中高 | 用户硬性 block-size 5 → 先按 5 跑，V5-6 #5 损坏冒烟兜底；一旦命中 → A) `--speculative-dspark-block-size 4`（SM120 实证干净）；B) 并打 #34021（mHC combine einsum 移出 symmetric pool，单 hunk，SM120 323/323 干净；B300 无效但与 sm_121 同属未知区，作为决策项由主理人拍板） |
| V5-R5 | 🟡 **graph capture 重开风险**：B>64 的 draft verify 经填充后走 prefill 实例化表，若仍有其他 quirk | 中 | V5-6 #3 显式验证；失败则设回 `--disable-cuda-graph`（功能可用、吞吐下降，非阻断） |
| V5-R6 | 🟡 **上游 #33407 合入后补丁需对齐/退役** | 低 | 补丁与上游同构，且 flashinfer #4380 原生 192 落地后 bucket-pad 可退休；登记技术债跟踪 |
| V5-R7 | 🟠 **80G 上限 × UMA × cgroup 兼容性未实测**（前轮启动未达运行期） | 中 | V5-6 #7 专项验收；若 CUDA 分配命中 cgroup → 回传主理人，勿自行去掉 `--memory 80g` |
| V5-R8 | 🟡 **`SGLANG_SM120_FLASHMLA_BACKEND=triton` 被误用为生产方案** | 低 | 明确仅诊断/临时解阻；吞吐损失（热路径 topk=2048 走 Triton） |

---

## V5-8. 对照方案：DSPARK → STANDALONE（仅记录，不推荐）

| 项 | 说明 |
|---|---|
| 触发 | 仅当方案 A 补丁后仍被未知问题阻断、且用户同意放弃 DSPARK |
| 启动参数 | `--speculative-algorithm STANDALONE`（其余不变） |
| 功能损失 | **无投机解码**：SM120 参考数据 DSpark 相对无投机约 **+93%**（conc16）；4× RTX PRO 6000 单流 78-87 tok/s vs 62（无投机）；本机 TP4 环网绝对值更低，只做相对基线 |
| 意义 | 仅用于 **A/B 性能对照** 与「DSPARK 关闭后是否同样有 depth-5 损坏」的判别测试（#33800 的判别手法） |
| 结论 | 不作为最终方案；用户 DSPARK 硬性要求下仅评估对照 |

---

## V5-9. 证据链接（V5 新增）

- PR #33407（Fix DSPARK SM120 decode dispatch，**open**）：https://github.com/sgl-project/sglang/pull/33407
- Issue #33134（DGX Spark sm_121 topk=192 复现 + 零填充 workaround）：https://github.com/sgl-project/sglang/issues/33134
- Issue #33800（DSPARK depth 5 损坏输出，SM120 根因 #29927/#34021，sm_121 无数据）：https://github.com/sgl-project/sglang/issues/33800
- PR #34021（mHC combine einsum 移出 symmetric pool，最小修复）：https://github.com/sgl-project/sglang/pull/34021
- flashinfer #4309（feat topk=192，closed→#4380）：https://github.com/flashinfer-ai/flashinfer/pull/4309
- v0.5.16 `flash_mla_sm120.py`：https://raw.githubusercontent.com/sgl-project/sglang/v0.5.16/python/sglang/kernels/ops/attention/flash_mla_sm120.py
- v0.5.16 `environ.py`（`SGLANG_SM120_FLASHMLA_BACKEND` 默认 flashinfer）：https://raw.githubusercontent.com/sgl-project/sglang/v0.5.16/python/sglang/srt/environ.py
- flashinfer 0.6.14 `_sparse_mla_sm120.py`（`_DECODE_DSV4_DISPATCH`/`_decode_dsv4_dispatchable`）：https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.6.14/flashinfer/mla/_sparse_mla_sm120.py
- SGLang releases（v0.5.17 Latest，无 0.5.18）：https://github.com/sgl-project/sglang/releases

---

*v5 由 architect 基于上游 PR/issue 实抓 + v0.5.16/main/v0.5.17 源码面核查 + flashinfer 0.6.14 dispatch 表核查定稿（2026-08-15）。裁决：方案 A（移植 PR #33407 到 0.5.16）为主、B/C 排除、STANDALONE 仅对照；产出补丁脚本 `scripts/patch_dspark_sm120_topk192.py` 与 abi3 重打包方案；新增 depth-5 损坏冒烟（V5-6 #5）与 80G 运行期验收（V5-6 #7）。凡标"实测/待确认"项以 sre 执行与 TP1 冒烟为准。*
