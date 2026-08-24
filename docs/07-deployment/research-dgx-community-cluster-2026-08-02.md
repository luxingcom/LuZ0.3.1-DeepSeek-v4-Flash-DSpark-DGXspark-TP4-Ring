# DGX Spark 多机集群社区调研报告（双机 / 四机 / 八机 + 4 机环网专项）

> 调研日期：2026-08-02
> 范围：GitHub、NVIDIA Developer Forums、Reddit (r/LocalLLaMA)、CSDN/知乎/腾讯云社区、ModelScope（魔塔）、个人博客
> 关联项目：本集群（2× DGX Spark GB10，DeepSeek-V4-Flash-0731，TP=2 vLLM）

---

## 0. 摘要（TL;DR）

- **双机是绝对主流**：几乎所有社区项目都从双机起步（QSFP 直连 200G RoCE，TP=2），性能共识 **decode 30-80 tok/s**（取决于模型/量化/投机解码），扩展性接近线性。
- **四机两种流派**：①官方/主流 = **RoCE 200GbE 交换机**（MikroTik CRS812/CRS804），TP=4；②**无交换机环网确实有搞法**——NVIDIA 官方只支持到**三机环**（NCCL 补丁分支），社区（nccl-mesh-plugin / H-AOC 交叉线缆）已跑通**四机环**，但官方工具链明确建议 4+ 用交换机。
- **八机**：单交换机（400DD→4×100G breakout 或 400G→2×200G）+ TP=8，可跑 550B 级模型；实测证明 **decode 与网络带宽基本无关**（UMA 内存带宽是瓶颈），网络只影响 TTFT/冷 prefill。
- **魔塔（ModelScope）**：主要作为模型分发源（Qwen 官方仓库），没有独立的 DGX Spark 集群项目；国内经验集中在 CSDN/知乎/腾讯云开发者社区。
- **关键教训**：vLLM 必须用社区 fork（eugr/jasl 系，b12x MoE 内核、SM121 补丁）；NCCL 需 2.29.7+ 且 GID 索引、HCA 列表、memlock/shm 必须配对；GB10 瓶颈是 273 GB/s LPDDR5X 带宽，不是网络。

---

## 1. 社区生态地图

| 社区 | 主要形态 | DGX Spark 集群相关内容 | 活跃度 |
|---|---|---|---|
| **GitHub** | 开源项目 + fork 生态 | `NVIDIA/dgx-spark-playbooks`（官方，321 commits）、`eugr/spark-vllm-docker`（事实标准容器栈）、`jasl/vllm`（SM120+ fork）、`vllm-dgx-spark` 系（社区双机脚本，62+ commits）、`autoscriptlabs/nccl-mesh-plugin`（4 机环网 NCCL 插件）、`lmxxf/deepseek-v4-deployment-on-dgx-spark`（**与本项目同模型**） | 极高，2026 年上半年爆发 |
| **NVIDIA Developer Forums** | 实战帖 + 官方人员（eugr_nv 等）回帖 | 三机环官方支持公告、8 节点集群 Build Report、DeepSeek V4 Flash 1M ctx 配方、SGLang 4 节点 MTP 实测 | 高，日均多帖 |
| **Reddit (r/LocalLLaMA)** | 购机/踩坑讨论 | "DGX Spark 是否值得买"、$10k 硬件方案、单机/双机性能引用来源 | 中高 |
| **CSDN / 知乎 / 腾讯云社区** | 中文教程（含 AI 生成内容混杂） | 4 节点搭建踩坑、Qwen 系部署教程、vLLM/SGLang 参数详解 | 高（质量参差） |
| **ModelScope（魔塔）** | 模型仓库 + 教程 | 仅作模型下载源（Qwen 官方仓库 + git clone），无集群部署项目 | 低（集群层面） |
| **个人博客** | 深度长文 | keithtyser（双机家庭集群）、Dre Dyson（4 机无交换机 / 8 机商业化）、rthpc（Qwen3.5-397B-INT4 双机） | 中，质量最高 |

**GitHub 关键仓库**：

| 仓库 | 定位 | 要点 |
|---|---|---|
| `NVIDIA/dgx-spark-playbooks` | 官方 playbook 集合 | `connect-two-sparks`（双机直连）、`connect-three-sparks`（**三机环**）、`multi-sparks-through-switch`（多机过交换机）、`nccl`（多机 NCCL）、`vllm`、`sglang`、`trt-llm` |
| `eugr/spark-vllm-docker` | 社区事实标准容器栈 | 提供 `sparkrun` CLI，**支持 3 节点 ring 拓扑自动检测**（topology: ring 注入 NCCL_IB_SUBNET_AWARE_ROUTING 等变量）；最新镜像含 NCCL 2.29.7+ 与 b12x MoE 内核 |
| `jasl/vllm` | SM120+ Triton 内核 fork | DeepSeek-V4 系列 MLA/attention 内核重写的基础（本项目环境 D 同源路线） |
| `vllm-dgx-spark`（gary109/volfco 等 5+ fork） | 双机一键部署脚本 | 13 模型预设、IB 自动探测、benchmark 套件；注意已不再维护旧版 |
| `autoscriptlabs/nccl-mesh-plugin` | **4 机环网 NCCL 网络插件** | 支持 ring/line/mesh 拓扑、子网感知网卡选择、中继路由、双路径负载均衡；142 commits，2026-03 仍活跃 |
| `lmxxf/deepseek-v4-deployment-on-dgx-spark` | DeepSeek V4 Flash 双机配方 | TP=2 约 12 tok/s、PP=2 约 10-11 tok/s；Marlin+DeepGEMM 混合后端；基于 `NVIDIA/dgx-spark-playbooks` 的 netplan 配置 |

---

## 2. 双机部署（社区绝对主流）

### 2.1 实现方法（共识流程）

1. **物理连接**：两台 Spark 的 ConnectX-7 QSFP56 口**背靠背直连**（一根 DAC 线，无需交换机），200 Gb/s。
2. **网络配置**：官方 netplan（`NVIDIA/dgx-spark-playbooks/nvidia/connect-two-sparks/assets/cx7-netplan.yaml`）→ 两台各配静态 IP（如 <NODE_IP>/2 或 169.254.x.x）；`ib_write_bw` 验证接近线速。
3. **NCCL 环境变量**（社区反复强调的黄金组合）：
   - `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`（列 HCA）
   - `NCCL_SOCKET_IFNAME`（对应管理/直连网口）
   - `NCCL_IB_GID_INDEX=3`（GID 必须存在，社区默认 3，出错试 0-3）
   - 容器必须 `shm_size=64gb` + `ulimits memlock=-1`（否则 `ibv_reg_mr_iova2 failed: Cannot allocate memory`——本项目已踩过同样坑）
4. **推理框架**：vLLM TP=2（Ray 或 mp executor）+ `--kv-cache-dtype fp8` + `--gpu-memory-utilization 0.8-0.92`；部分场景 `--enforce-eager`（SM121 CUDA graph 兼容性）。
5. **启动顺序**：worker 先起、head 后起（等 10s），本项目 8-01 已实证该顺序的必要性。

### 2.2 双机性能实测（社区数据汇总）

| 模型 | 量化 | 双机 decode | 来源 |
|---|---|---|---|
| DeepSeek-V4-Flash | FP8 + MTP | **30-45 tok/s**（1M ctx，980K 时 30.4）；并发 2 达 54.4 | NVIDIA 论坛 Aiden recipe（2026-05） |
| DeepSeek-V4-Flash | FP4 Marlin | ~12 tok/s（TP=2）/ 10-11（PP=2） | lmxxf GitHub |
| **本项目 DeepSeek-V4-Flash-0731** | dspark 投机 | **53.8-78.8 tok/s**（并发 1-3 基准，2026-08-02 实测） | 本集群基准 |
| Qwen3.5-397B-A17B | INT4 | ~40 tok/s（单流 25） | rthpc / NVIDIA 论坛 TP=4 对比帖 |
| Qwen3.5-122B-A10B | INT4 | ~40 tok/s | Reddit/bswen 汇总 |
| GPT-OSS-120B | MXFP4 | ~75 tok/s | keithtyser |
| Llama 3.3 70B | NVFP4 | ~80 tok/s；TPOT 133ms（TP2 vs TP1 269ms，近 2×） | NVIDIA 官方 blog / bswen |
| Qwen2.5-32B | Q4 | ~100 tok/s | bswen 汇总 |
| Qwen VL32B | BF16 | 6.14 tok/s（单机 3.58，**线性扩展**） | 腾讯云社区 |

**结论**：双机 TP=2 扩展效率约 1.5-2×（TPOT 近线性），模型越大、网络占用越小（MLA KV 极省，decode 与 fabric 无关）；**本项目 53-78 t/s 显著优于社区同模型配方（12-54 t/s）**，dspark 投机 + 0731 权重组合是当前社区已知的最优解之一。

---

## 3. 四机部署

### 3.1 官方/主流方案：RoCE 200GbE 交换机（推荐）

- **拓扑**：4× Spark → 1× MikroTik CRS812（8×QSFP56 200G，双 400G 口）→ TP=4。
- **官方 playbook**：`multi-sparks-through-switch` + `nccl`；NVIDIA 官方定位"四节点 = 本地推理服务器，支持 700B 级模型"。
- **性能（官方 blog，Llama 3.3 70B NVFP4 / TRT-LLM / 32K ctx / batch=1）**：

| 指标 | 1 机 TP1 | 2 机 TP2 | 4 机 TP4 |
|---|---|---|---|
| TTFT | 33,415 ms | 21,384 ms | 15,552 ms |
| TPOT | 269 ms | 133 ms | **72 ms**（≈4× 近线性） |

- **社区实测（腾讯云社区，Qwen VL32B BF16，64GB 权重）**：单机 3.58 → 双机 6.14 → **四机 11.36 tok/s**（近线性）；交换机中转延迟 ~3μs vs 直连 ~2μs。
- **SGLang 路线（NVIDIA 论坛，Gemma-4-31B + MTP，TP=4 EP=1）**：n=1 → 26.7 tok/s，n=8 → **153.2 tok/s**（MTP +80%）；注意 FlashInfer 对 head_dim=256 崩溃 → 必须 `attention_backend: triton`。
- **NCCL 细节**：4 节点 ring+tree NCCL init 约 5s；每节点双虚拟接口各 100G = 单节点 200Gbit/s 总带宽。

### 3.2 无交换机方案（环网/直连 mesh）——见第 5 节专项

---

## 4. 八机部署

### 4.1 实现方法（两个已知案例）

| 方案 | 硬件 | 网络 | 实测内容 |
|---|---|---|---|
| **方案 A（NVIDIA 论坛 Toshi.A）** | 4× ASUS Ascent GX10 + 4× Lenovo ThinkStation PGX | **单台 CRS812** + 2× 400DD→4×100G breakout（交换机侧 400G QSFP-DD → 节点侧 4×100G QSFP28），**8 节点全部 100G** | TP=8 Nemotron 3 Ultra 550B-A55B NVFP4 |
| **方案 B（Dre Dyson 商业化）** | 8× DGX Spark | MikroTik **CRS804-4DDQ**（98DX7335）+ 4× 400G→2×200G 线缆，200G RoCEv2 | Qwen3.5-397B FP8 / Kimi K2.6，vLLM eugr fork v0.21.1 TF5 |

**方案 A 关键实测（100G vs 200G 对比，TP=4 Qwen3.5-397B INT4）**：
- 单流 decode：100G 与 200G **几乎无差**（±3%）——LPDDR5X UMA 带宽是瓶颈，NCCL all-reduce 不是；
- 聚合吞吐（n=4）：短上下文（8K-16K）100G 掉 ~20%，64K+ 差距消失；
- 温 TTFT：100G 翻倍（8K: 2.02s→4.15s），绝对值仍在秒级；**冷 prefill 受影响最大**；
- **结论：8 机 100G 是生产可接受的折衷**。

**方案 A 八机 TP=8 性能（Nemotron 3 Ultra 550B NVFP4，SGLang 0.5.12）**：
- 启动：NCCL init ~5s，113 分片权重加载 ~9min，就绪 ~10min；
- 单流 decode **13.5 tok/s**（p50），TPOT ~74ms；KV 池 17.6M tokens（50.4 GB 集群级）；
- 冷 prefill ~1,380 tok/s（8K→64K TTFT 5.7s→46.7s）；前缀缓存命中时 TTFT 秒级。

### 4.2 八机踩坑（社区共识）
- CRS812 只有 2 个 400G 口 → 8 机必须 breakout；CRS804 有 4×400G → 每节点 200G；
- **ConnectX-7 PCIe Power Throttle 卡死**（热插拔线缆后链路卡在 13 Gbit/s，需重启主机）；
- 散热：8 节点 + 交换机发热极大（~50℃），需通风；CRS804 不可发往中国/香港/俄罗斯/委内瑞拉（出口限制，**国内采购注意**）；
- Jumbo Frame（MTU 9000）全集群统一。

---

## 5. 专项：4 机环网（无交换机）——**有搞法，三条路线**

### 5.1 结论先行

**4 机环网可行，但分三个层次**：

| 路线 | 拓扑 | 官方支持 | 社区验证 | 代价 |
|---|---|---|---|---|
| **A. 官方三机环**（NCCL 补丁） | 3 机环（每机 2 口，首尾相连） | ✅ NVIDIA playbook + sparkrun | ✅ 多帖实测（训练/推理） | 只到 3 机；4 机官方明确建议交换机 |
| **B. 社区四机环**（nccl-mesh-plugin） | 4 机环（每机 2 口 A-B-C-D-A，各链路独立子网） | ❌ | ✅ Qwen2.5-14B DeepSpeed ZeRO-3 训练 + vLLM 推理（生产环境） | 非相邻节点走**中继转发**（~1 RTT/跳），NCCL 需换插件 |
| **C. 四机准全互联**（H-AOC 交叉线缆） | 4 机，每机 3 卡 6 接口 → 与其它 3 机各 2 条直连 | ❌ | ✅ Dre Dyson SaaS 生产（2026-05 长文） | 需第 3 块 ConnectX-7（扩展槽）+ $900 线缆 |

### 5.2 路线 A：官方三机环（无交换机，NVIDIA 支持）

- **物理**：每台 Spark 有 2 个 CX7 QSFP 口（Port0 靠网口侧 / Port1 远离）；接线：Node1(P0)→Node2(P1)，Node2(P0)→Node3(P1)，Node3(P0)→Node1(P1)；**每个物理口暴露 2 个逻辑接口**（enp1s0f0np0/rocep1s0f0 与 enP2p1s0f0np0/roceP2p1s0f0）。
- **NCCL**：标准 NCCL 不支持 → 必须用补丁分支 `git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git`（NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"）。
- **环境变量（官方/社区黄金组合）**：
  ```bash
  NCCL_NET_PLUGIN=none
  NCCL_IB_SUBNET_AWARE_ROUTING=1   # 多子网感知路由（mesh 必需）
  NCCL_IB_MERGE_NICS=0
  NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1  # 全部 4 个 HCA
  NCCL_SOCKET_IFNAME=<管理口>
  NCCL_IB_GID_INDEX=3
  ```
- **工具链**：`sparkrun`（eugr/spark-vllm-docker）集群定义写 `topology: ring` 即自动注入上述变量 + 自动检测；3 节点 PP=3 配方已发布（Qwen3.5-397B-INT4-AutoRound 3x-vLLM）。
- **实测带宽**：nccl-tests ~24 GB/s avg bus bandwidth；训练可跑通（NVIDIA 论坛有成功案例）。
- **常见报错**："ring topology requires exactly 3 hosts"（4+ 会被拒，提示用交换机）；"requires 2 physical ports"（只插了 1 根线）。

### 5.3 路线 B：社区四机环（nccl-mesh-plugin）

- **项目**：`autoscriptlabs/nccl-mesh-plugin`（142 commits，2026-03 活跃，NCCL 2.29 v9 API 支持）。
- **原理**：标准 NCCL 假设"同子网交换式 IB"或 TCP，无法处理**异子网直连 RDMA mesh**；插件实现：Multi-Address Handle Exchange（通告所有子网 IP）+ Subnet-Aware NIC Selection（按对端 IP 子网选本地网卡）+ 后台握手线程（消除 connect 死锁）+ **Store-and-forward 中继**（非相邻节点经邻居转发）+ 环上双路径负载均衡。
- **4 节点环 IP 规划**（每链路独立子网是硬性要求）：
  ```
  A↔B: <NODE_IP>/24   B↔C: <NODE_IP>/24
  C↔D: <NODE_IP>/24   D↔A: <NODE_IP>/24
  管理网: <NODE_IP>/24（NCCL_SOCKET_IFNAME 引导）
  ```
- **环境变量**：`NCCL_NET_PLUGIN=<libnccl-net.so>`、`NCCL_MESH_ENABLE_RELAY=1`、`NCCL_MESH_MAX_HOPS=4`、`NCCL_MESH_RING_LOAD_BALANCE=1`、`NCCL_MESH_GID_INDEX=3`（出错试 0-3）。
- **通信代价**：A↔C 需经 B 或 D 中继（2 跳）；插件自动 CW/CCW 双路径负载均衡（阈值 1MB 切换）；计划中 cut-through 转发可降延迟。
- **验证场景**：4 节点环 + Qwen2.5-14B + DeepSpeed ZeRO-3 分布式训练 + vLLM 推理，双通道 200G 跑满。
- **注意**：该项目 vLLM 补丁"已提交上游未合并"，采用面较窄（个人/小团队），无 switch 方案成熟。

### 5.4 路线 C：四机准全互联（H-AOC，Dre Dyson 生产方案）

- **核心洞察**：ConnectX-7 每物理口 = 2 逻辑接口；每节点插 **3 块 CX7**（2 内置 + 1 扩展），配合 **SR4 光模块一分二** 或 **MFS1S90-HxxxE H-AOC 交叉线缆**（2×200G→2×200G，内部 lane 交叉，$450/根），每节点得到 6 个活跃接口 → **与其它 3 节点各 2 条直连路径**（环 + 对角线），无需交换机。
- **实测**：H-AOC 即插即用，与 breakout 方案等价但更干净；需 udev 规则固定接口名（`/etc/udev/rules.d/70-spark-net.rules`，按 PCI 地址 pin 名字，重启不漂移）。
- **代价**：约 $900 线缆 + 扩展卡；属于"工程上绕开官方限制"的进阶方案，无现成一键脚本。

### 5.5 环网 vs 交换机的工程判断（对 4 机扩容的建议）

| 维度 | 交换机（官方） | 环网（社区） |
|---|---|---|
| 非相邻节点通信 | 直连，全带宽 | 中继 2 跳，延迟/带宽折损（decode 场景影响小） |
| NCCL | 标准版即可 | 需补丁（3 机）或插件（4 机） |
| 稳定性 | 官方验证 | 社区验证，工具链兼容性需自测 |
| 成本 | 交换机 ~$1-3k + 线缆 | 0 线缆成本（3 机环）；$900+（4 机 H-AOC） |
| 扩展 | 4→8 加口/换 CRS804 即可 | 环只能到 4（每机 2 口上限）；8 机必须交换机 |

**参考依据**：8 机 100G vs 200G 实测证明 decode 与网络带宽解耦——**环网 2 跳中继对 decode 吞吐的伤害远小于直觉**（瓶颈是 273 GB/s UMA）；但 TTFT/冷 prefill 会受影响。若本项目扩 4 机且以并发 serving 为主：**交换机仍是首选；若纯内部实验/省预算，4 机环 + nccl-mesh-plugin 可行**，三机环则有官方全程支持。

---

## 6. 全社区共识与关键参数基线（对部署有普适价值）

1. **vLLM 版本**：必须 ≥0.21 且带 SM121 补丁（b12x MoE 内核、PR #40082 cherry-pick、FlashInfer/cutlass sm_121a 放宽）；社区事实标准 = eugr/spark-vllm-docker 镜像（`vllm-node-tf5/tf6`）或 `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`（DS-V4 专用，b12x + CUDA 12.1 + vLLM 0.21.1）。
2. **NCCL**：2.29.7+（多机）/ 补丁分支（3 机环）；`NCCL_IB_GID_INDEX=3` 是社区默认值；`NCCL_IB_MERGE_NICS`、`NCCL_CROSS_NIC` 按拓扑配置。
3. **容器**：`--shm-size 64gb` + `ulimits memlock=-1` + `stack 64MB`（NCCL 注册内存必需）——本项目已全部对齐。
4. **性能真相**：GB10 decode 是 **273 GB/s LPDDR5X 带宽瓶颈**，与网络/互连几乎无关；网络决定 TTFT 与 prefill。量化（NVFP4/INT4/FP8）收益巨大：NVFP4 单流可到 74.75 tok/s（Nemotron 3 Nano，带宽天花板 ~80）。
5. **投机解码**：MTP/dspark 是唯一能突破带宽天花板的途径（SGLang MTP +80% @n8；本项目 dspark 投机使 5 并发 80.1 t/s 居社区前列）。
6. **中国环境注意**：模型下载走 `hf-mirror.com` 或 ModelScope（本项目已用）；CRS804 交换机有出口限制，国内采购需确认渠道。
7. **DeepSeek-V4-Flash 双机社区基线**（与本项目直接对标）：
   - Aiden recipe：1M ctx + TP=2 + MTP + b12x = 30-45 t/s，`--block-size 256`、`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`、`--distributed-executor-backend mp`、worker 先启动；
   - lmxxf 配方：FP4 + Marlin/DeepGEMM 混合 = ~12 t/s（Marlin 42 层 + 末层 DeepGEMM 保精度）；
   - **本项目（0731 权重 + dspark 投机 + hybrid-1.6）：53.8-78.8 t/s，全面领先社区公开配方。**

---

## 7. 参考链接

- NVIDIA 官方 blog：Scaling Autonomous AI Agents and Workloads with DGX Spark（4 机支持声明 + TP1/2/4 数据）
- NVIDIA Playbooks：`github.com/NVIDIA/dgx-spark-playbooks`（connect-two-sparks / connect-three-sparks / multi-sparks-through-switch / nccl）
- build.nvidia.com/spark/connect-three-sparks/three-sparks-ring（三机环官方指南）
- NVIDIA 论坛：Three node Spark clusters without a switch（sparkrun ring 支持）；Is training on 3 nodes without a switch supported（NCCL 补丁 + 环境变量）；8x DGX Spark Cluster Build Report（CRS812 + TP=8）；DeepSeek V4 Flash at 1M Context on Dual DGX Spark（Aiden recipe）；Multi-node SGLang Gemma-4-31B MTP
- GitHub：`eugr/spark-vllm-docker`、`jasl/vllm`、`autoscriptlabs/nccl-mesh-plugin`、`lmxxf/deepseek-v4-deployment-on-dgx-spark`、`gary109/vllm-dgx-spark` 等
- 博客：keithtyser.com（双机家庭集群）、dredyson.com（4 机无交换机 / 8 机商业化）、rthpc.com（Qwen3.5-397B-INT4 双机）、ai-muninn.com（Nemotron 3 Nano NVFP4 74.75 t/s 调优）
- 中文社区：腾讯云开发者社区《DGX Spark 多节点集群搭建这些坑千万别踩》、CSDN 多篇 Qwen 系部署

---

*报告完。结论一句话：双机 TP=2 是社区最成熟路径且本项目已是同模型最优；4 机环网存在 3 条可行路线（官方三机环 / nccl-mesh-plugin 四机环 / H-AOC 准全互联），官方立场是 4+ 用交换机；8 机单交换机 + TP=8 已验证生产可行。*
