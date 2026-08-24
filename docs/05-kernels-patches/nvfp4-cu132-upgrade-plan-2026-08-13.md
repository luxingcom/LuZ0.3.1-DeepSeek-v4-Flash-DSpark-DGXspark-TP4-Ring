# DeepSeek-V4-Flash 集群 NVFP4 权重 + CUDA 13.2 升级方案（PR 提速专项）

**日期**：2026-08-13
**目标**：改善 PR（prefill）速率，DE（decode）保持不劣化、不做额外优化
**前提**：不改硬件、不改四机环网拓扑、不升驱动（580.173.02 经 forward-compat 支撑 CUDA 13.2）

---

## 一、调查分析结论（已逐项实测核实）

### 1.1 当前服务器环境实测

| 项 | 实测值 | 来源 |
|----|--------|------|
| 宿主机驱动 | **580.173.02**（四机一致） | nvidia-smi |
| 容器内 torch / CUDA | **2.11.0+cu130 / CUDA 13.0** | torch.version |
| cuBLASLt | **13.1.1.3**（CUDA 13.0 系，无 NVFP4 3× 路径） | /usr/local/cuda/lib64 |
| 推理镜像 | anemll/dspark-vllm-gx10:0.2.1-v026.0（vLLM 0.26.1.dev0，自建） | docker images |
| 模型权重 | deepseek-v4-flash-0731，`expert_dtype=fp4`(MXFP4) + `quant_method=fp8` | config.json |
| KV cache | `nvfp4_ds_mla`（走 flashmla FP8 快内核，Issue#22 已修） | 启动参数 |
| MoE 后端 | `flashinfer_b12x` | 启动参数 |
| 投机解码 | dspark num_spec=5 动态K | 启动参数 |
| 通信补丁 | ring-only NCCL v3（libnccl.so.2.30.7，MD5 b7784b49，CUDA 13.0 下容器内构建） | /opt/nccl-ringonly |

### 1.2 三个关键澄清（纠正常见误解）

**澄清①：当前生产是 CUDA 13.0，不是 13.2。**
"驱动 580 经 forward-compat 支撑 CUDA 13.2 容器"是 8/7 审计的**评估结论**，从未落地——镜像清单中无任何 cu132 镜像。

**澄清②：NVFP4 KV 缓存 ≠ NVFP4 权重解算。**
`--kv-cache-dtype nvfp4_ds_mla` 走 `flashmla_sparse.py` 的 FP8 快内核，与 cuBLASLt 的 NVFP4 GEMM 无关。用了 NVFP4 KV 缓存不代表 NVFP4 权重解算就绪。

**澄清③："595 驱动"非严格约束。**
CUDA 13.2 官方配对 595，但 580 驱动 + NVIDIA forward-compat（cuda-compat-13-2）实测可跑 CUDA 13.2 容器（8/7 ADR-2 + 社区 580.142 跑 13.2 佐证）。**驱动不必升 595**（595 仍是 beta、未进 DGX OS 验证通道）。

### 1.3 上游镜像状态（决定方案形态）

| 上游 | 状态 |
|------|------|
| anemll/dspark-vllm-gx10 官方 | 最新 **0.1.1 = vLLM 0.25.2 + torch cu130（CUDA 13.0）**，**无 cu132 版本** |
| NGC cu132 vLLM | 仅 0.19/0.20，远旧于 0.26.x，不可平替 |
| 社区 SM121 cu132 实践 | eugr/avarok 已跑通（580 驱动 + CUDA 13.2 + NVFP4 + Marlin），非官方镜像 |

**结论：CUDA 13.2 镜像必须自建**（基于现有 0.2.1-v026.0 的 Dockerfile + 补丁资产升级 CUDA 层）。

### 1.4 cuBLASLt NVFP4 3× 加速的精确边界（决定收益预期）

CUDA 13.2 官方 headline：cuBLASLt 对 NVFP4/MXFP8 提供 **up to 3× 加速，仅对 large M/N 大 GEMM**——这正是 prefill 的 MoE 专家 GEMM 场景（M=token 数、N=expert 维度、K=hidden）。

但 SM121 **缺 tcgen05**，导致：
- CUTLASS FP4 内核在 SM121 输出结构性垃圾（不可用）
- **prefill 大 GEMM** → cuBLASLt 13.2 NVFP4 路径（3× 潜力，本次升级目标）
- **decode 小 GEMM** → Marlin W4A16（dequantize FP4→BF16，memory-bound，收益小）

> 收益预期必须诚实：3× 只作用于 MoE 专家 GEMM 这一环，prefill 总耗时还含 attention、路由、通信(~6.7%)，故**整体 prefill 预期 1.3~2×（需 A/B 实测定论），不是整 3×**。

---

## 二、技术方案设计

### 2.1 核心思路

```
当前：FP4(MXFP4) 权重 + flashinfer_b12x → MXFP4 kernel（prefill 无 cuBLASLt NVFP4 加速）
目标：NVFP4 权重 + CUDA 13.2 cuBLASLt → prefill 大 GEMM 走 NVFP4 3× 路径
      decode 走 Marlin W4A16（不劣化即满足目标，不做额外优化）
```

两个变化同时发生：① 权重格式 FP4→NVFP4；② 后端 flashinfer_b12x → cuBLASLt NVFP4（prefill）/ Marlin（decode）。

### 2.2 依赖缺口与补齐（三缺二 + 补丁重编）

| # | 缺口 | 补齐方案 |
|---|------|---------|
| 1 | **CUDA 13.2 镜像**（现 13.0） | 自建：现有 0.2.1-v026.0 Dockerfile 升级 torch cu132 + CUDA 13.2 runtime + 重编 ring-only NCCL |
| 2 | **NVFP4 权重** | MJPansa/DeepSeek-V4-Flash-0731-NVFP4（33,024 routed-expert 投影全量 NVFP4，保留 DSpark/MTP） |
| 3 | **后端 env** | `VLLM_NVFP4_GEMM_BACKEND=marlin`（decode）+ cuBLASLt NVFP4（prefill，CUDA 13.2 自动生效）+ `VLLM_USE_FLASHINFER_MOE_FP4=0` |
| — | 驱动 | 无需动（580.173.02 + forward-compat） |
| — | KV cache | 无需动（nvfp4_ds_mla 已是 NVFP4 压缩，走 FP8 快内核） |

### 2.3 关键工程点：ring-only NCCL 补丁重编

当前 ring-only v3（NCCL 2.30.7）是在 **CUDA 13.0 下容器内构建**的。换 CUDA 13.2 镜像后需决策：

| 选项 | 做法 | 风险 |
|------|------|------|
| A（优先验证） | 直接把现有 libnccl.so.2.30.7 拷入 cu132 镜像，验证 forward-compat 是否可加载 | 低，NCCL ABI 对 CUDA runtime 依赖稳定，大概率可跑 |
| B（兜底） | 在 CUDA 13.2 下重新构建 NCCL 2.30.7 + 重打 ring-only v1/v2/v3 补丁（transport.cc/net.cc） | 中，补丁源码已归档（backup/），构建流程有现成（8/11 部署记录） |

**策略**：先 A 后 B——A 验证 5 分钟出结果，失败再走 B（1-2 人日）。

### 2.4 投机解码兼容性

MJPansa 0731-NVFP4 **保留 DSpark/MTP 模块**（报告已确认），投机解码链路不丢。但需注意：
- dspark_block_size=5 硬约束不变（num_spec≥5）
- 动态K `[[1,1,5],[2,4,4],[5,6,3]]` 里的 `[5,6,3]` 仍违反校验（3<5），需一并修为 `[5,6,5]` 或删除低于 5 的档位

---

## 三、分阶段执行计划

### Phase 0：前置准备（第 0 天，零风险，不碰生产）

1. 备份锚点固化：当前镜像 tag + ring-only MD5(b7784b49) + shim v8 MD5(ce43c688) + 启动脚本 MD5 + 权重 .local-backup
2. 确认 02 下载通道（16:55 已定案：用户代理 → 本地下载 MJPansa 0731-NVFP4 → scp 02 → rsync 01 → NFS 03/04）
3. 在 <MGMT_OCTET>/<MGMT_OCTET>（canary 机）预留 cu132 镜像构建环境

### Phase 1：NVFP4 权重下载与校验（并行于镜像构建）

```
用户代理 → 本地 HF 下载 MJPansa/DeepSeek-V4-Flash-0731-NVFP4
→ scp 到 02 → rsync 到 01 → NFS 双源同步 03/04
→ 校验：conversion-receipt.json + safetensors shard 数(48) + 字节级抽样
```

### Phase 2：CUDA 13.2 镜像自建（1-2 人日）

1. 基于 0.2.1-v026.0 Dockerfile，升级：torch 2.12 系 cu132 + CUDA 13.2 runtime + cuBLASLt 13.2
2. 重编 sm_121 原生 kernel（TORCH_CUDA_ARCH_LIST=12.1a，参考已合并 PR #38126 的 arch guard）
3. ring-only NCCL：先验证现有 libnccl.so.2.30.7 直接加载（选项 A），失败则重编+重打补丁（选项 B）
4. shim v8 重新编译（依赖 CUDA 版本）
5. 产物 tag：`dspark-vllm-gx10:0.2.1-cu132-nvfp4`

### Phase 3：测试环境 A/B 验证（核心，prefill 专项）

在测试环境（生产镜像副本同一硬件）拉起 cu132 镜像 + NVFP4 权重，跑 **prefill 专项对比**：

| 验证项 | 基线（现 FP8+FP4/b12x） | 目标（NVFP4+cuBLASLt） | 判定 |
|--------|----------------------|----------------------|------|
| prefill 131072/c1 | 2013-2016 tok/s | 预期 1.3~2× | 提速 ≥1.2× 即 GO |
| prefill 8192/32768/c1 | 基线值 | 对比 | 不劣化 |
| decode 131072/c1 | 110-115 tok/s | ≥100（不劣化） | 硬门槛 |
| decode 32768/c1 coding | 95-103 | 不劣化 | 硬门槛 |
| 投机接受率 | 0.73-0.93 | 不劣化 | 不劣化 |
| 精度抽样 | GSM8K/已知 prompt | 无漂移 | 硬门槛 |

### Phase 4：生产灰度与固化

1. A/B 全过 → 生产窗口切换 cu132 镜像 + NVFP4 权重
2. 启动参数固化：`--moe-backend` 决策（prefill 走 cuBLASLt，需验证是否需保留 b12x）+ Marlin env + 修动态K `[5,6,3]`→`[5,6,5]`
3. 回填 Runbook + 回滚锚点 + 补丁归档 backup/tp4-cu132-<date>/

### Phase 5：监控与观察（14-30 天）

1. 观察 prefill 吞吐、decode 是否劣化、投机接受率、显存水位
2. 若 decode 劣化 >5% 或精度异常 → 回滚 FP8 权重 + cu130 镜像（锚点秒级回退）

---

## 四、量化目标（诚实口径）

| 指标 | 基线 | 目标 | 说明 |
|------|------|------|------|
| **prefill 131072/c1** | 2013-2016 tok/s | **≥2400（1.2×，保守）/ 理想 3000+（1.5×）** | 3× 仅作用于 MoE GEMM，非整体 |
| prefill 8192-32768/c1 | 基线 | ≥1.1× 或不劣化 | 中小档大 GEMM 也受益 |
| **decode 131072/c1** | 110-115 tok/s | **≥100（不劣化）** | DE 不动，硬门槛 |
| 精度 | — | 无漂移 | GSM8K/抽样 |
| 硬件/拓扑/驱动 | — | 0 变更 | 硬约束 |

---

## 五、风险与回滚

| 风险 | 等级 | 缓解 |
|------|------|------|
| 自建 cu132 镜像 sm_121 原生 kernel 成熟度不足 | 高 | Phase 2 在 canary <MGMT_OCTET>/<MGMT_OCTET> 先验证；失败退回 cu130 |
| ring-only 补丁与 CUDA 13.2 不兼容 | 高 | 选项 A→B 两级；补丁源码已归档；重编流程现成 |
| decode 换 Marlin 后劣化 | 中 | A/B 硬门槛 decode≥100；劣化>5% 回滚 |
| 投机解码 dspark 在 NVFP4 下异常 | 中 | MJPansa 保留 DSpark；A/B 验证接受率 |
| cuBLASLt NVFP4 3× 未兑现（prefill 收益不足 1.2×） | 中 | 若 <1.2×，评估是否保留 NVFP4 仅换 Marlin，或整体回退 |

**回滚锚点**：cu130 镜像 tag + FP8 权重 + 原启动脚本 MD5，切回即可（分钟级，权重 .local-backup 已保留）。

---

## 六、依赖与资源清单

| 资源 | 内容 | 状态 |
|------|------|------|
| 权重 | MJPansa/DeepSeek-V4-Flash-0731-NVFP4（~160GB） | 待下载（走代理） |
| 镜像 | 自建 dspark-vllm-gx10:0.2.1-cu132-nvfp4 | 待构建（1-2 人日） |
| 补丁源码 | ring-only v1/v2/v3 + shim v8 | 已归档 backup/ |
| 驱动 | 580.173.02 | ✅ 无需动 |
| 环境 | <MGMT_OCTET>/<MGMT_OCTET> canary + 测试环境（生产镜像副本） | 就绪 |

---

## 七、一句话总结

PR 提速的完整路径 = **自建 CUDA 13.2 镜像（cuBLASLt NVFP4 3× 路径）+ 换 MJPansa 0731-NVFP4 权重 + decode 走 Marlin 兜底不劣化**。驱动 580 不动、KV cache 不动、拓扑不动；核心风险在自建镜像的 sm_121 原生 kernel 与 ring-only 补丁重编，均已备两级方案与回滚锚点。预期 prefill 1.3~2×，decode 保持 ≥100 tok/s 不劣化。
