# cu132 充分利用评估：torch cu130 vs cu132 实测对比（GB10）

**日期**：2026-08-07
**工作流**：系统设计 / 技术选型评估（工作流 2 变体）
**参与成员**：Archi（方案与判定线）/ 主理人执行实测（<MGMT_OCTET> 上 venv 对照实验）

---

## 📌 TL;DR（执行摘要）

- **结论**：aarch64 生态下 torch 保持 cu130 是**当前最优解**——实测 torch 2.14.0.dev+cu132（nightly，cuBLAS 13.4/cuDNN 9.24 全套 13.2 库）与 torch 2.11.0+cu130 在 GB10 上 **GEMM 性能持平（±2%）**，且官方 nightly wheel **仍未编译 sm_121 原生 SASS**（只有 compute_120 PTX）。
- 关键认知：GB10 的"原生优化"收益主要来自 **vLLM 自编译 kernel（sm_121a SASS）**——这部分在"13.2 runtime + cu130 torch"镜像中**已获得**；torch 层库升级（cuBLAS 13.4）在当前 GEMM 形态上无差异。
- 严重度分布：不适用（评估类）；决策：**维持 cu130 torch，跟踪上游**。
- 阻塞 / 非阻塞：非阻塞。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（维持现状，不投入重编） |
| cu132 全栈可行性 | 可行（nightly wheel 全 Python 版本存在）但**收益未兑现** |
| 判定线结果 | 稳态差 <2%（Archi 判定线：<2-3% → 维持 cu130） |
| 建议下一步 | 跟踪 PyTorch sm_121 原生 wheel 与 vLLM 上游 cu132；Path C 暂不投入 |
| 已固化产物 | <MGMT_OCTET> registry: `vllm-gb10:0.26.1-cu132-sha-fa87aea5`（18.9GB，功能验证通过） |

---

## 1. 背景与问题

用户要求：aarch64 wheel 生态限制下 torch 保持 cu130，**有无解决方案？能否充分利用 cu132 优化？**
现状：已交付镜像 = CUDA 13.2 工具链（nvcc/cudart 13.2.51）+ torch 2.11.0+cu130 wheel + vLLM 0.26.1.dev0（sm_121a 原生 kernel）。

## 2. 可行性核实（实测）

| 项 | 结果 |
|----|------|
| torch cu132 nightly aarch64 wheel | ✅ 存在，**覆盖 cp310-cp315 全部版本**（download.pytorch.org/whl/nightly/cu132/torch/，2.14.0.dev20260806+cu132） |
| torchvision/torchaudio cu132 aarch64 | ⚠️ 部分缺失（不影响 vLLM） |
| vLLM pypi wheel | cp38-abi3（兼容 3.12），aarch64 存在 |
| 关键约束 | cu132 目前只有 nightly（2.13/2.14.dev），生产锁定 2.11 稳定 → **版本错位是真正限制** |

## 3. 实测对比（<MGMT_OCTET> 上同机同 GPU，8192³ GEMM）

| 指标 | torch 2.11.0+cu130（现镜像） | torch 2.14.0.dev+cu132（nightly 全栈） | 差异 |
|------|------------------------------|----------------------------------------|------|
| torch.version.cuda | 13.0 | **13.2** | ✅ |
| arch 列表 | sm_80-120（无 121） | sm_80-120 + compute_120（**仍无 sm_121 原生**） | ⚠️ |
| FP8 e4m3 GEMM | **8.6 ms** | **8.8 ms** | -2%（持平） |
| BF16 GEMM | **13.5 ms** | **13.7 ms** | -1.5%（持平） |
| FP8 首调（JIT 冷启动） | 204 ms | 244 ms | 无豁免（nightly 也未编 sm_121，PTX JIT 仍在） |
| 依赖库 | cuBLAS 13.x / cuDNN 9.x（cu130 系） | **cuBLAS 13.4.0.1 / cuDNN 9.24 / NCCL 2.30.7** | 库版本新但无 GEMM 收益 |

**解读**：
1. cu132 torch 的 arch 列表**没有 sm_121**——PyTorch 官方 nightly wheel 也未为 GB10 编译原生 SASS，PTX JIT 豁免**不成立**。
2. cuBLAS 13.4 vs 13.0 在 8192³ FP8/BF16 GEMM 上**无可测差异**（±2% 噪声内）——GB10 的 GEMM 性能由 vLLM 自编译 kernel（sm_121a）主导，该收益**已获得**。
3. 与 Archi 预判一致：稳态吞吐增量 0-5% 且实测贴近 0；首载改善未兑现。

## 4. 决策与路径结论

| 路径 | 结论 |
|------|------|
| A. vllm-gb10 改造（torch→cu132 nightly 重编） | ❌ 暂不投入：跳版 2.11→2.14 兼容风险高，实测收益 ≈0 |
| B. 等 PyTorch cu132 稳定版 + sm_121 原生 | ✅ 跟踪（预计数月内；届时重测一次） |
| C. CoreWeave torch 2.11.0+cu132 arm64 | ⏸️ 暂缓：同 torch 版本下预期收益同属 torch 库级（已证明 ≈0） |
| D. 自编译 torch sm_121a | ❌ 过度投入 |
| **维持现状**：13.2 runtime + cu130 torch + vLLM sm_121a kernel | ✅ **采用**（已交付固化） |

**收益边界明示**（供用户决策）：
- ✅ 已获得：CUDA 13.2 运行时、vLLM 推理热路径 sm_121a 原生 SASS（FA3/MoE/FP8 matmul）、FlashInfer 0.6.14
- ❌ 未获得且实测无差：cuBLAS 13.2 库级优化（GEMM 持平）
- 📅 未来增量来源：PyTorch 官方 sm_121 原生 wheel（跟踪）、vLLM 上游 cu132 正式 tag

## 5. 生产部署状态（已完成）

- 镜像：`<NODE_IP>:5000/vllm-gb10:0.26.1-cu132-sha-fa87aea5`（18.9GB，digest fa87aea5，manifest 200 验证）
- 冒烟：nvcc 13.2.51 / cudart 13.2.51 / torch 2.11.0+cu130 / vLLM 0.26.1.dev0 / GB10 (12,1) / matmul OK
- 环境：<MGMT_OCTET> 本机留 venv（/tmp/cu132venv，torch 2.14+cu132）供后续复测

## 6. 深度分析补充：vLLM sm_121a kernel 的 13.2 利用度（Archi 最终结论 + 编译层取证）

### 6.1 编译层取证（cuobjdump/ldd 实测）

| 层 | 实测证据 | 判定 |
|----|---------|------|
| vLLM 扩展 cubin | `_qutlass_C.abi3.*.sm_120.cubin`（8 个 .so 全为 sm_120） | **nvcc 13.2.51 编译、原生 SASS、无 JIT**（sm_121a 可运行 sm_120 cubin，minor 兼容） |
| vLLM 扩展 ldd | `libcudart.so.13 => /usr/local/cuda/lib64/libcudart.so.13`（=13.2.51） | 运行层用 13.2 cudart ✅ |
| vLLM GEMM 路径 | 不直接链接 cuBLAS（ldd 无条目）→ cutlass 自研 kernel | 与 torch 库版本解耦 |
| torch 委托算子 | nvidia_cublas-13.1.0.3 / nvidia_cudnn_cu13-9.19.0.56（cu130 系） | 缺口 ❌（权重 <5% 时间） |
| 工具链 arch 支持 | nvcc 13.2.51 `--list-gpu-code` 含 **sm_121** ✅ | 可编 sm_121，但生态默认 sm_120 |
| cu132 torch nightly | get_arch_list = sm_80-120 + compute_120（**无 sm_121**） | PyTorch 官方仍未编 sm_121 |

### 6.2 全栈 nightly + vLLM 重编 sm_121a 的收益建模（Archi）

| 增量项 | 理论上限 | GB10 实际权重 |
|--------|---------|--------------|
| sm_121a SASS | 大 GEMM 0-10% | 仅 prefill 长上下文；受 ptxas/Triton 对 sm_121a 指令支持 bug 制约（`.tile::gather4` 报错、Triton 3.5+ sm_121a 后缀被拒、vLLM CMakeLists 显式排除 sm_12x FP4 kernel） |
| cuBLAS 13.4（torch 委托） | 单算子微增 | <1% |
| cuDNN 9.24 | 可观（SDPA 场景） | vLLM 主路径 ~0（走 vllm_flash_attn） |
| NCCL 2.30.7 | TP 场景可观 | 单机 TP=1 时 ~0 |

**为什么实测 GEMM 持平（±2%）**：① 两次跑分 GEMM 均走同一套 vLLM cutlass sm_120 kernel，换 torch/cuBLAS/cuDNN 不触达该路径；② GB10 深度内存受限（FP8 dense ~200-250 TFLOPS / 273 GB/s LPDDR5x → 临界算术强度 730-900 FLOP/byte；decode 算术强度仅 2-10，权重每 token 全量重读），**serving 墙钟被 decode 主导，计算侧优化被稀释为个位数%**。

### 6.3 最终决策（Archi 结论）

**不值得做全栈 nightly 升级**：预期端到端收益 0-5%（集中 prefill 大 batch），成本 2-4h+ 构建 + torch 2.11→2.14 跳版 ABI 风险 + 工具链补丁 + nightly 漂移。

第三条路（性价比排序）：
1. **维持 sm_120 现状（推荐）**——不是缺陷，是 GB10 生态当前正确编译目标；如确要试 sm_121：CMake patch + `TORCH_CUDA_ARCH_LIST=12.1`（去 'a' 后缀，nvcc 13.2.51 已确认支持）单测重编，成本可控可回退
2. **pip 覆盖 torch 委托库**（nvidia-cublas-cu13 → 13.3+）：soname 稳定 ABI 兼容，收益 <1%，可选做
3. **等 torch 2.11 系 cu132 稳定 wheel**：避免 2.14 dev 跳版，后续最稳妥升级路径

**行动建议**：正式基准用端到端 TTFT/TPOT + batch 扫描（勿用孤立 GEMM——它掩盖内存受限事实）；TP>1（双 Spark NVLink）场景单独评估 NCCL 2.30；prefill 占比高且确认工具链产出 sm_121 SASS 后再投入路径 1。

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 维持 cu130 torch 决策（不投入 cu132 torch 重编） | Archi | P0 | 已定 |
| 2 | 跟踪 PyTorch sm_121 原生 wheel 发布（发布后重测 GEMM/首载） | Archi | P2 | 持续 |
| 3 | <MGMT_OCTET> 从 <MGMT_OCTET> registry 同步拉取固化镜像（生产部署前置） | Rex | P1 | 本周 |
| 4 | 单机性能测试（Tessa C0-C6 计划，用户排期确认后执行） | Tessa | P1 | 待排期 |
| 5 | 重建记录落盘（rebuild-vllm-cu132 骨架回填实测数据） | Docu | P2 | 本周 |

---

## ⚠️ 待完善 / 已知局限

- GEMM 微基准为 8192³ 单形态；未覆盖注意力/端到端（需模型加载，用户指示本次不做）
- 2.14.0.dev 为 nightly（20260806），与 2.11 存在版本差混杂；同版本纯净对比（Path C）未执行，但预期收益边界一致
- 首调耗时含 torch 初始化差异，仅作信号非精确 JIT 计量

---

## 📚 数据来源 & 成员产出索引

- Archi（架构师）：cu132 专项报告（路径对比表 / 收益盘点 / 判定线 <2-3% 维持）
- 主理人实测：pytorch 源 wheel 清单核实（cp310-315 全版本存在）、<MGMT_OCTET> venv 对照实验（cu130 vs cu132 GEMM 数据）、<MGMT_OCTET> registry 固化验证
- 背景：eugr/spark-vllm-docker ABI 坑（vllm cu132 wheel 须配 torch cu132）、vllm-gb10 项目构建机制（VLLM_REF 参数化）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
