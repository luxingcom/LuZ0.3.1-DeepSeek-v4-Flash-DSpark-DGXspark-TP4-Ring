# 驱动 / CUDA 升级必要性评估（DGX Spark 双机，2026-08-07）

> 背景：项目组认为"当前服务器内驱动与 CUDA 不是最新"，要求调研最新驱动带来的优化及升级必要性。
> 真机核实（2026-08-07，head=60/worker=58）：驱动 **580.173.02**、内核 **6.17.0-1029-nvidia**、固件全最新（fwupdmgr 无可用更新，含 ConnectX-7 MT2910）、vLLM 0.26.1（Anemll v026，TORCH_CUDA_ARCH_LIST=12.1a，基于 nvidia/cuda 官方基础镜像）。

---

## 0. 结论摘要

| 组件 | 当前 | 最新（官方线） | 有无升级必要 | 理由 |
|---|---|---|---|---|
| **驱动** | 580.173.02 | **580.173.02（已是最新）** | ❌ **无需升** | 2026-06-29 发布 = Release 580 当前最新；DGX 软件栈 07-17 同版本；595.x 官方明确不支持 DGX Spark |
| **内核** | 6.17.0-1029-nvidia | 与驱动配对正确 | ❌ 无需动 | 论坛实证这是 OTA2607 后的正确内核组合 |
| **固件** | 全最新 | 全最新 | ❌ 无需 | fwupdmgr 零可用更新 |
| **CUDA（容器内）** | 13.x（镜像自带） | **13.2 值得跟进** | ⚠️ **等镜像升级，勿手动** | cuBLASLt NVFP4/MXFP8 最高 3×、Grouped GEMM MXFP8、Tensor Memory 并发 bug 修复——全与 MoE GEMM 相关 |
| **DGX Spark OTA** | 已含（580.173.02 = OTA2607） | July 2026 Release | 🟡 确认 OOM 增强是否生效 | 官方公告：GB10 unified memory 的 OOM 处理增强（跑大模型有价值） |

**一句话**：**驱动/固件已是最新，无需升级；真正的机会在"容器镜像的 CUDA 13.2"（等 Anemll/社区新镜像），以及确认 July 2026 OTA 的 OOM 增强已生效。**

---

## 1. 驱动层面：已是官方最新（"不是最新"的判断不成立）

### 1.1 版本对照

| 来源 | 版本 | 说明 |
|---|---|---|
| 本机（真机核实） | **580.173.02** | 与官方最新一致 |
| NVIDIA Tesla Release Notes | 580.173.02（2026-06-29） | Release 580 家族最新 |
| DGX 软件栈（2026-07-17） | GPU Driver **580.173.02** | EL8/9/10 统一版本 |
| 论坛 OTA2607（378200） | 580.173.02 + 内核 6.17.0-1029-nvidia | 与本机完全一致 = DGX Spark 最新 OTA 状态 |

### 1.2 580.173.02 带来的优化（与本项目相关性）

| 更新项 | 内容 | 相关性 |
|---|---|---|
| GB10y GDMA 死锁 WAR | 高负载下 GDMA 硬件死锁防护 | 🟡 中（GB10 平台稳定性，与负载相关） |
| P2P 建议 | 关闭 IOMMU/ACS 以获得 PCIe P2P 性能 | 🟡 中（本项目无 intra-node GPU P2P，跨机走 RoCE，不适用） |
| UVM/NVML/ISR 等修复 | 底层稳定性 bug 修复 | 🟢 低（无直接推理收益） |
| 已知问题 | 与 EUD ≤580.126.12 不兼容（需 ≥580.159.x） | ✅ 本机 580.173 已满足 |

**结论：580.173.02 的更新主要是稳定性修复，无针对 vLLM 推理的显式性能提升。驱动不升。**

### 1.3 为什么不能升 595.x（除非走实验路线）

- NVIDIA 员工（aniculescu）在论坛明确："**Driver 595 is not yet supported on DGX Spark**"；
- 有社区用户在 Ubuntu 26.04 自装 595.71.05 + CUDA 13.2 跑通（vLLM 容器内 CUDA 13.2），但属**非官方实验**，官方明确可能 broken system；
- 且 595 需要 Ubuntu 26.04 系 + 新内核（7.0.0-1006），本项目 DGX OS 24.04 线不匹配。

---

## 2. CUDA 层面：机会在容器镜像，不在宿主机

### 2.1 架构澄清

- 宿主机 CUDA toolkit **对容器化 vLLM 无影响**（vLLM 在容器内用镜像自带的 CUDA 运行时）；
- 本机容器（Anemll v026）基于 nvidia/cuda 官方基础镜像 + `TORCH_CUDA_ARCH_LIST=12.1a`（SM121 目标）；
- **"升级 CUDA"的实际动作 = 等 Anemll/社区新镜像（CUDA 13.2 基础）或自行构建，不是动宿主机。**

### 2.2 CUDA 13.2 对 GB10/SM121 的大件（论坛 363182，官方/社区总结）

| 特性 | 内容 | 对本项目（DS-V4 MoE + MLA）的价值 |
|---|---|---|
| **cuBLASLt NVFP4/MXFP8 性能 ↑ 3×** | 大 M/N 问题尺寸下 NVFP4/MXFP8 最高 3× | 🔴 **高**——MoE GEMM 若走 cuBLASLt，收益直接 |
| **cuBLASLt Grouped GEMM（实验）支持 MXFP8** | SM121 上 grouped GEMM（MoE 场景） | 🔴 **高**——与 DeepGEMM grouped GEMM 路线互补/竞争 |
| **关键 bug 修复**：cublasLtMatmul 与 Tensor Memory 并发错误结果 | 影响 CC 10.x/11.x，自 cuBLAS 12.8 起 | 🟡 中——可能解释社区遇到的输出质量退化（lmxxf Marlin 乱码类） |
| CUDA Tile 支持 SM120/SM121 | 新 tile 编程模型 | 🟡 中——未来 SM121 NVFP4 kernel 更干净路径 |
| Unified Tegra+Desktop toolkit | GB10 是 Tegra 派生 SoC，统一减少 SM121 专属 bug | 🟢 低-中（长期收益） |
| PTX ISA 9.2 / aarch64 编译改进 | 新指令/编译器修复 | 🟢 低 |

**结论：CUDA 13.2 对本项目是"值得关注的下一个镜像基线"**——尤其 cuBLASLt NVFP4/MXFP8 3× 与 Grouped GEMM 两项与 MoE decode 直接相关（与本项目 DeepGEMM 调研同方向）。但**不建议手动升级容器内 CUDA**（风险大、收益需实测），正确路径 = 跟进 Anemll/社区镜像发布。

### 2.3 注意（诚实提醒）

- 本项目 0731 权重为 FP8，NVFP4 的 3× 收益需要权重是 NVFP4 才完整兑现（Anemll 环境 E 的 nvfp4_ds_mla 是 KV cache 维度，不是权重维度）——FP8 权重的 cuBLASLt 收益低于 NVFP4 宣传值；
- 3× 是针对"大 M/N"（prefill/大 batch），decode 小 GEMM 的收益需要实测；
- 最终以新镜像 + 现有 bench 方法论（bench_matrix.py）实测为准。

---

## 3. DGX Spark July 2026 Release（OTA2607）——OOM 增强

- 官方公告（论坛 376736）：该 release 驱动包含 **GB10 unified memory 的 OOM 处理增强**（内存压力下系统更健壮、有用户反馈）；Display Reserved Memory 2/4GB 可调（appliance 模式默认 2GB 无需改）；Cloud-Init 增强；
- 本机 580.173.02 + 6.17.0-1029 内核 = 论坛实证的 OTA2607 状态 → **大概率已含该 release 的驱动部分**；
- **待确认**：OOM 增强是否还需配套固件/内核组件（本机固件已最新，无可用更新，大概率齐了）；
- 验证方法：跑一次内存压力场景（本项目已有 mem_test.py 系列），观察 OOM 处理表现。

---

## 4. 升级风险与操作建议

1. **升级风险（论坛 378200 实证）**：部分 Spark 在 apt upgrade 到 580.173.02 后出现 "No devices found"（GPU 掉）——修复 = `apt dist-upgrade` 对齐内核 6.17.0-1029-nvidia 或重装 `nvidia-driver-580-open` 系列；**升级必须避开生产窗口**（vllm-envE 在跑）并留 SSH 回退通道；
2. **推荐动作**：
   - 驱动/固件：**不动**（已最新）；
   - 容器镜像：**跟进 Anemll/社区发布**，CUDA 13.2 基础镜像出来后，用 bench_matrix.py 对 0731 权重做 A/B（重点看 MoE GEMM 路径：cuBLASLt vs 现有后端）；若 MoE 后端是 Triton/CUTLASS/DeepGEMM，则 CUDA 13.2 收益需重新评估；
   - OTA：维持官方更新节奏（DGX Dashboard 或 apt dist-upgrade），升级后核对内核版本对齐；
   - 长期：CUDA Tile（SM121）成熟后，SM121 NVFP4 kernel 可摆脱 CUTLASS patch-and-pray（lmxxf/社区补丁链），降低维护成本。

---

## 5. 自行升级 CUDA 13.2 的路径调研（2026-08-07 补充）

### 5.1 决定性事实：580.173.02 驱动与 CUDA 13.2 兼容 ✅

- CUDA 13.2 官方定位："establishes a new baseline for driver compatibility, **requiring NVIDIA drivers version 580 or higher** for minor version compatibility across the 13.x toolkit family"（Linux）；
- CUDA 13.2.1 release notes 打包配套驱动为 595.58.03，但那是"随附新版驱动"，**非必需**——580+ 驱动即可运行 CUDA 13.2 库（CUDA minor version compatibility 原则：同 major 驱动向后兼容所有 minor toolkit）；
- **结论：本机 580.173.02 可直接承载 CUDA 13.2 的库/工具链（容器内或宿主机 runfile），无需换驱动**。595 驱动是另一码事（解锁 595 专属修复，见 5.3 路径 C）。

### 5.2 四条路径矩阵

| 路径 | 操作 | 获得 | 风险 | 适用 |
|---|---|---|---|---|
| **A. 容器内 CUDA 13.2**（推荐首选） | 用 `nvidia/cuda:13.2.x-devel-ubuntu24.04`（aarch64）基础镜像重建 vLLM（Anemll 0.2.1 的 Dockerfile 改基础镜像版本），宿主驱动不动 | cuBLASLt 3×（NVFP4/MXFP8）、Grouped GEMM MXFP8、Tensor Memory 并发修复、CUDA Tile SM121、统一 Tegra toolkit——**全部软件侧收益** | 🟢 低（容器级，旧镜像可随时回退） | 推理性能优化（本项目主路径） |
| **B. 宿主装 CUDA 13.2 toolkit（保留 580.173.02）** | CUDA 官方 repo（`developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/`）加 keyring → `apt install cuda-toolkit-13-2`；或 runfile toolkit-only（`cuda_13.2.x_580.173.02_linux_sbsa.run`，UNCHECK Driver） | 宿主编译/调试工具链 13.2 | 🟢 低（不动驱动；论坛 349881 有 13.0.2 runfile 保留 580.95.05 的先例） | 宿主编译需要时 |
| **C. 宿主升级 595 驱动 + CUDA 13.2**（社区 beta） | purge 580 → `nvidia-open-dkms-595` + `cuda-toolkit-13-2` → reboot（Dre Dyson 完整流程） | **CUDA graph capture 修复（Qwen3.5-397B 实测 +35-40%）**、NVFP4 计算解锁（FP4 训练/推理）、590 内存泄漏修复 | 🔴 **高**（官方明确不支持 595 于 DGX Spark；unbootable/chroot 恢复风险；Grace 强制 openrm） | 追求极限且能承担风险的非生产节点 |
| **D. 等官方/Anemll 认证** | 无操作，跟进镜像发布与未来 DGX OTA | 稳定、经认证 | 🟢 无 | 生产优先 |

### 5.3 路径 C（595 驱动）的详细实证与警告

**Dre Dyson 实测结论（对比分析文）**：
- "The CUDA 13.2 libraries in the container give you **most of the software-side improvements without the driver risk**"——**容器内 13.2 库已含大部分收益**；
- 595 驱动专属（容器无法替代）：CUDA graph capture 修复（Qwen3.5-397B **+35-40%**）、NVFP4 计算解锁、590 内存泄漏修复；
- 必须用 `nvidia-open`（Grace 强制 openrm，"If you see nvidia in lsmod instead of nvidia_open, something went wrong"）；装错包可导致系统无法启动（作者经历过 chroot 恢复）；
- **go/no-go 测试**：换 595 后先跑 vLLM 内存释放测试（Ctrl-C 后显存回落）——不通过则不可用于推理，立即回滚（`apt remove nvidia-open-dkms-595 + cuda-toolkit-13-2` + 重装 580）；
- 官方无 595 稳定时间线承诺。

**对本项目的针对性评估**：0731 权重为 FP8 → NVFP4 解锁收益小；本项目已用 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（breakable CUDA graph）→ 595 的标准 graph capture 修复是否叠加收益需实测（大概率边际）；**路径 C 收益-风险比不划算，不推荐**。

### 5.4 推荐行动（结合本项目）

1. **首选路径 A**：检查 Anemll 上游是否已发布 CUDA 13.2 基础镜像；若无，用其 Dockerfile 改 `FROM nvidia/cuda:13.2.x-devel-ubuntu24.04` 自建（aarch64），保持 0731 权重与现有启动脚本；
2. 新镜像出来后用 **bench_matrix.py** 对 0731 权重做 A/B（双机基线 53.8-78.8 t/s 对比），重点验证：MoE GEMM 后端是否走 cuBLASLt、NVFP4/MXFP8 3× 是否兑现于 FP8 权重、Tensor Memory 并发 bug 修复后输出质量是否更稳；
3. 宿主 CUDA 13.2 toolkit（路径 B）仅在需要宿主编译时安装（runfile toolkit-only，保留 580.173.02）；
4. 路径 C 标记为高风险实验，仅在非生产节点 + 有回退预案时考虑；
5. 同步关注 Anemll/社区（eugr spark-vllm-docker 已在用 13.2.0-devel 基础镜像——可直接复用其构建产物）。

### 5.5 关键参考资料

- CUDA 13.2.1 Release Notes（Linux driver 595.58.03 配套；arm64-sbsa 支持）
- CUDA Installation Guide for Linux（Ubuntu 24.04 aarch64 支持；deb/runfile 两种方式）
- CUDA 13.2 Downloads（arm64-sbsa → Ubuntu → deb/runfile）
- Dre Dyson《Fix Nvidia drivers 595.45.04 and CUDA 13.2...》（5 分钟流程：purge 580 → nvidia-open-dkms-595 + cuda-toolkit-13-2；graph capture +35-40%）
- Dre Dyson《I Tested Every Solution...》（容器内 13.2 库=大部分收益；595=graph capture/NVFP4/内存泄漏；go/no-go 内存测试；chroot 恢复教训）
- 论坛 364688（Ubuntu 26.04 + 595.71.05 + CUDA 13.2 + vLLM 容器实测 GB10）
- 论坛 349881（13.0.2 runfile toolkit-only 保留 580.95.05 驱动的先例）
- TechnoTim《Ubuntu Server on DGX Spark》（官方 dgx-repo + nvidia-driver-580-open + cuda-toolkit-13-0 标准流程参考）

---

## 6. 参考

- NVIDIA Tesla Release Notes 580.173.02（06/29/2026：GB10y GDMA WAR、P2P IOMMU 建议、EUD 兼容）
- DGX EL9 软件栈 Release Notes（07-17：Driver 580.173.02 / CUDA 13.0 Update 3 / NCCL 2.30.7 / DCGM 4.6）
- 论坛 376736（DGX Spark July 2026 Release：OOM 增强、Display Reserved Memory、Cloud-Init）
- 论坛 378200（apt upgrade 580.173.02 翻车案例与修复；4×Spark 正常实例）
- 论坛 364688（595.x 官方不支持 DGX Spark；社区 Ubuntu 26.04 自装案例）
- 论坛 363182（CUDA 13.2 DGX Spark 影响：cuBLASLt NVFP4/MXFP8 3×、Grouped GEMM、Tensor Memory bug 修复、CUDA Tile SM121、Tegra 统一 toolkit）
- CUDA Toolkit Archive（13.3.1 最新 / 13.4.0 preview / 13.2.2 2026-07）
