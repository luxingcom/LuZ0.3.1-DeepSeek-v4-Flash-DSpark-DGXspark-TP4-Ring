# GB10 CPU 缓存系统设计与 NCCL 专用最低延迟调配方案

> 日期：2026-08-10
> 方法：官方资料（Tom's Hardware 评测 / Android Authority / Arm 官方）+ 真机核实（lscpu -e、/sys cache 拓扑、PCIe NUMA 归属、/proc/cmdline，head=60 与 worker=58 一致）
> 目的：GB10 20 核如何分组、共享什么缓存/硬件、如何调配获得最低延迟用于 NCCL 专用。

---

## 0. 结论摘要（TL;DR）

| 项 | 结论 |
|---|---|
| **核分组** | 20 核 = 10×Cortex-X925（性能核，3900MHz）+ 10×Cortex-A725（能效核，2808MHz），分 **2 个集群**，每集群 5+5 |
| **缓存共享** | 每核私有 L2（X925=2MB / A725=512KB）；**集群内 10 核共享 L3**：集群 0（CPU 0-9）=**8MB**、集群 1（CPU 10-19）=**16MB（不对称）**；另有 GPU 侧 16MB system cache（L4，CPU 不可见） |
| **NUMA** | **单 NUMA 节点（UMA 128GB）**——网卡/内存对两集群距离无差异，NUMA 距离优化不适用 |
| **NCCL 最低延迟组合** | **集群 1 的 X925 隔离核（CPU 16-19，当前已隔离）+ IRQ 绑集群 1 + nohz_full/rcu_nocbs**——现状配置已是较优解，微调空间见第 4 节 |
| **关键洞察** | 单 NUMA 下优化维度 = ①核性能（X925>频率+L2）②L3 容量（集群 1>集群 0）③隔离免干扰 ④IRQ 局部性——**全部指向"集群 1 的 X925"** |

---

## 1. GB10 CPU 缓存系统设计（官方 + 真机）

### 1.1 核心组成与分组

| 属性 | 值 | 证据 |
|---|---|---|
| CPU 设计方 | **联发科**（MediaTek 设计，类似 Dimensity 移动 SoC 架构） | Tom's / Android Authority |
| 核组成 | 10× Cortex-X925（性能核）+ 10× Cortex-A725（能效核） | 官方/真机 |
| 集群 | **2 个集群 × 10 核**，每集群再分 5×A725 + 5×X925 | Tom's + 真机 lscpu |
| 频率 | X925 = 3900MHz（真机 MAXMHZ）、A725 = 2808MHz | 真机 lscpu -e |
| 微架构 | Armv9（非 Neoverse，是 off-the-shelf Arm 消费级核心） | Tom's |

**真机拓扑（lscpu -e，head=60）**：

```
集群 0（L3=8MB）：CPU 0-4  = A725（CORE 0-4，2808MHz）
                  CPU 5-9  = X925（CORE 0-4，3900MHz）
集群 1（L3=16MB）：CPU 10-14 = A725（CORE 5-9，2808MHz）
                  CPU 15-19 = X925（CORE 5-9，3900MHz）
```

**A725/X925 结对**：CPU 0(A725) 与 CPU 5(X925) 共享 CORE ID 0；CPU 10(A725) 与 CPU 15(X925) 共享 CORE ID 5——**集群内每个 core 位置是一个 X925 + 一个 A725 搭档**（big.LITTLE 结对设计，共享集群接口/功耗域）。

### 1.2 缓存层次（官方 + 真机交叉验证）

| 缓存 | 每核/集群容量 | 共享范围 | 证据 |
|---|---|---|---|
| L1d / L1i | 64K / 64K（每核） | 核私有 | 真机 lscpu -C |
| **L2** | **X925 = 2MB / A725 = 512KB**（每核私有） | 核私有（真机 index2 shared=自身） | Tom's / Android Authority + 真机 |
| **L3（集群 0）** | **8MB**（CPU 0-9 共享） | 集群 0 内 10 核 | 真机 /sys index3 size=8192K |
| **L3（集群 1）** | **16MB**（CPU 10-19 共享）——**不对称** | 集群 1 内 10 核 | 真机 index3 size=16384K + Tom's |
| System Cache（L4） | 16MB（GPU 侧，CPU 不可见） | CPU+GPU 跨芯片（经 NVLink-C2C） | Android Authority / Tom's（GPU L4） |

> 注：lscpu -C 显示 L2 ONE-SIZE=512K 是**错误的统一值**——真机 /sys 只能看到"私有"，X925 实际 2MB（官方数据）。L3 不对称（8MB vs 16MB）真机确凿。

### 1.3 内存与互连

- **UMA 单 NUMA 节点**：128GB LPDDR5X-4266，256-bit，273GB/s（真机 numactl：1 node，124GB）
- **NVLink-C2C**：CPU↔GPU 600GB/s 双向（统一地址空间）
- 2.5D 封装：S-die（CPU，联发科，含内存控制器）+ G-die（GPU，Blackwell）

### 1.4 ConnectX-7 与 CPU 的关系（真机核实）

- 4 PF / 2 PCIe domain（0000、0002），全部 `numa_node=-1`、`local_cpulist=0-19`
- **单 NUMA 节点下，网卡对两个集群距离完全对称**——不存在"哪边核离网卡近"的 NUMA 差异（PCIe RC 物理上可能有近端，但 Linux 层不可见、不可利用）
- **推论：NCCL 核选择优化必须转向"核性能 + L3 容量 + 隔离 + IRQ 局部性"四个维度，NUMA 距离维度失效**

---

## 2. NCCL 专用最低延迟调配原则

单 NUMA 环境下，四个可用的优化杠杆（按收益排序）：

### 2.1 杠杆 1：X925 优先（核性能）

- X925：3900MHz（vs A725 2808MHz，**+39%**）、2MB L2（vs 512KB，**4×**）
- NCCL 通信线程 = **轮询/进展线程**（polling progress thread），纯延迟敏感——**必须跑在 X925 上**
- L2 4× 对 NCCL 小消息（握手、元数据、控制面）的缓存驻留收益显著

### 2.2 杠杆 2：大 L3 集群优先（集群 1 = CPU 10-19，16MB）

- 集群 1 的 L3 是集群 0 的 **2×**——NCCL 的通信缓冲（控制消息、小包聚合、状态）在同一集群内访问缓存命中率更高
- 集群间通信（CPU 0-9 ↔ CPU 10-19）走片内互连，有额外延迟——**NCCL 线程与其使用的内存/IRQ 应同集群**

### 2.3 杠杆 3：隔离核 + 免中断（现状已做）

- 当前 `/proc/cmdline`：`isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19`——**隔离的恰好是集群 1 的 X925（CORE 6-9）**，最优组合（16MB L3 集群 + X925 + 无调度器/时钟/RCU 干扰）
- 建议：如需更多 NCCL 核，扩展为 **15-19**（多一个 X925，CORE 5）或 **14-19**（+A725 做辅助 memcpy 核）

### 2.4 杠杆 4：IRQ 亲和性（网卡中断绑集群 1）

- mlx5 的 RoCE 中断（/proc/interrupts 的 mlx5 条目）应绑到**集群 1**（CPU 10-14 或 15 的 A725/X925），避免中断打断隔离核 16-19 的 NCCL 轮询，也避免中断落在集群 0 造成跨集群访问
- ARM64 下 `/proc/irq/<n>/smp_affinity` 可用（GIC 支持）；此前"IRQ 绑核 ARM64 不可行"的结论应复核——至少可将 IRQ 绑到集群 1 的非隔离核（10-15）

---

## 3. 推荐调配方案（结合本项目现状）

### 3.1 核分工矩阵

| 用途 | 推荐核 | 理由 |
|---|---|---|
| **NCCL 通信线程（主）** | **CPU 16-19**（集群 1 X925，已隔离） | X925 2MB L2 + 16MB L3 集群 + 零干扰 |
| NCCL 通信线程（扩展） | CPU 15（集群 1 X925） | 若需 5 个 NCCL 核 |
| NCCL 辅助（memcpy/搬移） | CPU 10-14（集群 1 A725） | A725 足够做数据搬移，省 X925 |
| **网卡 IRQ（RoCE）** | CPU 10-15（集群 1） | 与 NCCL 线程同集群，避免跨集群中断 |
| vLLM EngineCore 主线程 | CPU 5-9（集群 0 X925）或 10-14 | 与 NCCL 分离，避免争抢 |
| 管理面/OS | CPU 0-4（集群 0 A725） | 低负载任务 |

### 3.2 具体配置

```bash
# 1. 核隔离（已生效，如需扩展改 GRUB 后重启）
# GRUB_CMDLINE_LINUX: isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19
# 建议扩展: isolcpus=15-19 nohz_full=15-19 rcu_nocbs=15-19（多一个 X925）

# 2. IRQ 亲和（RoCE 中断绑集群 1 的非隔离核 10-15）
for irq in $(grep -l mlx5 /proc/irq/*/actions 2>/dev/null | grep -oE '[0-9]+' ); do
  echo 0f00 > /proc/irq/$irq/smp_affinity   # 集群1: 核10-15 → 位掩码 0x0FC0
done
# 注: 0x0FC0 = 核 10-15（bit10-15）

# 3. NCCL 线程绑核（vLLM 启动时）
taskset -c 16-19 vllm serve ...   # 或对 EngineCore 线程 taskset
# 保持 NCCL_IGNORE_CPU_AFFINITY=1 则让 vLLM 自行调度；实测对比后取优

# 4. 内存: UMA 单节点无需 interleave；如需大页可开 hugepages（对 NCCL 缓冲收益小，可选）
```

### 3.3 与现状对比（微调点）

| 现状 | 评估 | 建议 |
|---|---|---|
| isolcpus=16-19（集群 1 X925） | ✅ 已是最优组合 | 保持；如 NCCL 单核不够再扩 15 |
| NCCL_IGNORE_CPU_AFFINITY=1 | ⚠️ NCCL 不自行绑核（交给 vLLM） | 实测 vLLM 是否把 EngineCore 绑到好核；必要时 taskset 显式绑 16-19 |
| IRQ 亲和未配置 | ❌ 中断可能落在任意核 | **新增**：绑集群 1（10-15） |
| 网卡双 domain 对称 | ✅ 无 NUMA 距离差异，无需处理 | 无需动作 |

---

## 4. 验证方法（用数据选优）

```bash
# 绑定不同核组合跑 nccl-tests，对比 busbw 与延迟
# 方案 A：绑定 16-19（集群1 X925，推荐）
taskset -c 16-19 mpirun ... nccl-tests/build/all_reduce_perf -b 4M -e 128M -f 2 -g 1
# 方案 B：绑定 5-9（集群0 X925，对照）
taskset -c 5-9   mpirun ... nccl-tests/build/all_reduce_perf -b 4M -e 128M -f 2 -g 1
# 方案 C：绑定 0-4（集群0 A725，对照下限）
taskset -c 0-4   mpirun ... nccl-tests/build/all_reduce_perf -b 4M -e 128M -f 2 -g 1

# 小消息延迟（最敏感）：all_reduce 128B-4K
taskset -c 16-19 mpirun ... nccl-tests/build/all_reduce_perf -b 128 -e 4K -f 2 -g 1
# 预期：X925(16-19) < X925(5-9) < A725(0-4)；16-19 因 16MB L3 优于 5-9
```

预期结论：**16-19（集群 1 X925）在延迟与带宽上均最优**（频率 + L2 + L3 + 隔离四重优势）；5-9（集群 0 X925）次之（差在 8MB L3）；A725 最差。四机 K4 方案落地后按此验证。

---

## 5. 参考

- Tom's Hardware《Nvidia DGX Spark review》（联发科 CPU、2 集群×10 核、5+5 分组、X925 2MB L2/A725 512KB、L3 16MB+8MB 不对称、GPU L2 24MB+16MB L4、273GB/s）
- Android Authority《RTX Spark...built like a smartphone》（X925 2MB/A725 512KB、16MB L3+16MB system cache、NVLink-C2C 600GB/s、128GB LPDDR5X-4266）
- Arm 官方《Cloud-Class AI, Now on Your Desk》（X925 驱动计算密集/防 GPU 饥饿；A725 预处理/分词/推理）
- 腾讯科技《DGX Spark 评测》（S-die/G-die 2.5D 封装、联发科 S-die、NVLink-C2C 600GB/s、128GB 256-bit 273GB/s）
- 真机核实（2026-08-10 head=60）：lscpu -e / lscpu -C / /sys cache index3 size（8M vs 16M）/ numactl（1 node）/ PCIe numa_node=-1 + local_cpulist=0-19 / /proc/cmdline（isolcpus=16-19 nohz_full rcu_nocbs）

---

## 5. 集群 0（CPU 0-9）满足性核查（2026-08-10，NCCL 实际负载评估）

### 5.1 NCCL 实际负载（本项目实测数据，TP2 双机）

| 指标 | 实测值 | 来源 |
|---|---|---|
| 每 token 全模型通信 | **~368KB**（352KB all-reduce + 11KB KV all-gather） | analysis-tp2-tp4-communication-2026-08-09 |
| decode 单步 | 0.38MB/token，**88 次 all-reduce/step**（44 层 × 2），单次消息 ~4KB（小消息） | 同上 |
| 全模型纯通信 | TP2 ~2.6ms（每层 ~58µs），vLLM 流水线部分掩蔽 | 同上 |
| 带宽占用 | 200G 链路 prefill 峰值仅 **25-45%**（带宽远充裕） | 同上 |
| NCCL 单跳延迟 | 16B 消息 29µs（GPU launch 占大头，wire 仅 3.27µs） | 8-09 实测 |
| Worker 线程 CPU 时间 | 累计 **8 分钟** vs EngineCore 230 分钟（当前生产） | 2026-08-10 top |

**负载画像：decode 场景 = 小消息（KB 级）高频 + 带宽需求极低 + CPU 开销占比小；prefill 大消息（MB-GB 级）走流式。**

### 5.2 集群 0 vs 集群 1 的差异对 NCCL 的影响

| 差异 | 对 NCCL 的影响 | 判定 |
|---|---|---|
| X925 核本身（5-9 vs 15-19） | 同频率（3900MHz）、同 L2（2MB）——**核能力零差距** | 无影响 |
| L3 8MB vs 16MB | ①decode 小消息（4KB）工作集 KB-MB 级 ≪ 8MB → 缓存驻留无压力；②prefill 大消息走 NIC→GPU RDMA 流式，不依赖 L3 驻留 | **影响极小** |
| 集群间互连（跨集群访问） | 若 NCCL 线程与 IRQ/缓冲同集群则无跨集群开销——绑好即可 | 可控 |

### 5.3 实证：当前生产 NCCL 已在用集群 0 且正常

- 2026-08-10 实测：vLLM Worker 线程（含 NCCL 进展线程）**散落在核 3/7/9（集群 0）与 10/11/13/14/15（集群 1）**，亲和性 0-19 未绑核；
- 生产 vLLM 双机 TP2 运行正常（基准 53.8-78.8 t/s），**集群 0 的核参与 NCCL 工作无瓶颈迹象**。

### 5.4 结论：集群 0 完全满足，边界条件明确

**✅ 集群 0 可以满足 NCCL 要求**：
- 用集群 0 的 **X925（5-9）**：与集群 1 X925 性能等价（L3 差异对 NCCL 无感）→ **推荐**
- 用集群 0 的 A725（0-4）：频率 -39%、L2 -75%，小消息延迟略增，但 NCCL CPU 负载轻（每 step 88×4KB），**可兜底**（延迟略增可接受）

**边界条件（满足的前提）**：
1. **不与计算竞争**：NCCL 线程须与 vLLM EngineCore 计算线程分离（隔离或分区绑核）；
2. **IRQ 同集群**：RoCE 中断绑 NCCL 所在集群（集群 0 则绑 5-9，掩码 0x03E0），避免跨集群中断；
3. 大消息 prefill（流式）对集群选择不敏感——两集群无差异。

**工程建议**（按负载优先级二选一）：
- 若 **vLLM 计算是瓶颈**（当前 EngineCore 230min CPU 时间远高于 Worker 8min）→ **集群 1 的 X925（15-19）给 EngineCore 计算、NCCL 用集群 0 的 X925（5-9）**——把强核让给计算，NCCL 用集群 0 完全够；
- 若 **NCCL 延迟是瓶颈**（四机 K4 高并发 all-reduce）→ 维持现状：NCCL 用集群 1 隔离核 16-19。
- 验证：维护窗口用双机 nccl-tests（或容器内 torch.distributed）绑核对照（5-9 vs 15-19 vs 0-4），128B-4K 小消息延迟差异预期 <10%。

### 5.5 NCCL 全绑 A725 评估（2026-08-10）——是否影响 NCCL？

**结论：decode 场景下 NCCL 全绑 A725（能效核）不影响性能，且是释放 X925 的合理工程选择。**

#### 5.5.1 A725 vs X925 差异（官方/真机）

| 维度 | A725 | X925 | 差距 |
|---|---|---|---|
| 频率 | 2808 MHz | 3900 MHz | -39% |
| L2 | 512KB | 2MB | -75% |
| 单核性能（综合） | 能效核（较低 IPC） | 旗舰性能核 | **约 1.5-1.7×** |
| 每集群核数 | 5 | 5 | 相同 |

#### 5.5.2 为什么 A725 全绑不影响 NCCL（掩蔽关系量化）

NCCL CPU 开销 vs GPU 计算的相对关系是决定性因素：

| 量 | 数值 | 说明 |
|---|---|---|
| decode 单 step NCCL CPU 开销（X925） | ~130µs（88 次 AR × ~1.5µs） | 协议固定开销 |
| decode 单 step NCCL CPU 开销（A725） | ~200-260µs（×1.5-1.7） | 绝对值差 ~100µs |
| **单 step GPU 计算时间**（70 t/s 单流） | **~14ms** | 实测基线 |
| 掩蔽余量 | **50-100×**（CPU 开销 ≪ GPU 计算） | vLLM 流水线重叠 |

**只要 CPU 侧 NCCL 开销 << GPU 计算时间，绑什么核都不形成瓶颈**——当前余量 50-100×，A725 的 1.5-1.7× 慢完全在余量内。带宽路径同理：decode 368KB/step 对 200G 链路是零头（传输 ~2µs），A725 不参与数据搬运（GPUDirect RDMA，CPU 只做协议）。

#### 5.5.3 三种绑核方案对比

| 方案 | NCCL 延迟 | 释放的核 | 适用 |
|---|---|---|---|
| X925 隔离核 16-19 | 最优（基准） | 无 | 极致延迟/四机高并发 |
| 集群 0 X925（5-9） | 与 16-19 等效（L3 差异无感） | 15-19 | 中核分配 |
| **A725 全绑（0-4 或 10-14）** | **绝对差 ~50-100µs/step（被完全掩蔽）** | **全部 X925 给计算** | **计算瓶颈优先（当前正是）** |

#### 5.5.4 边界条件（A725 方案的适用前提）

1. **GPU 计算时间 ≥ ~5ms/step**（当前 14ms，余量大）：若未来并发翻倍、GPU 计算快于 ~3ms，NCCL CPU 开销占比上升，A725 需复核；
2. **IRQ 分散**：RoCE 中断不要与 NCCL 进展线程挤在同一 A725 核（分散到 0-4 或 10-14 的不同核）；
3. **prefill 大消息**：走 GPUDirect RDMA（NIC→GPU），CPU 仅协议，A725 不受影响；
4. **128B 级极限延迟压测**（非生产形态）：X925 29µs → A725 ~45-50µs，仅影响纯 NCCL benchmark，不影响 vLLM 端到端（掩蔽）。

#### 5.5.5 推荐（当前项目）

**NCCL 全绑 A725（建议 10-14，集群 1 的 A725：16MB L3 集群 + 与计算集群同侧），X925 全部释放给 vLLM EngineCore/高负载任务**——当前 EngineCore CPU 时间（230min）远高于 Worker（8min），计算才是瓶颈，强核让给计算是正确方向。验证：维护窗口绑核对照（A725 vs X925 跑 nccl-tests 小消息 + vLLM bench），预期端到端差异 <3%。
