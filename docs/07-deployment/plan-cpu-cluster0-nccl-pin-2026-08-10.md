# 四台 CPU 处理器调配方案：集群 0 A725(0-4) 隔离给 NCCL + IRQ 绑 X925(5-9)

> 状态：**方案已出，待用户批准重启窗口后执行**
> 日期：2026-08-10
> 依据：四台实测拓扑 + IRQ 绑核复核成功（推翻旧结论）

## ① 实测结论（四台一致）

### 1.1 CPU 拓扑（lscpu -e + cpufreq + cluster_id 实测）

| 集群 | cluster_id | CPU 范围 | 核类型 | 最大频率 | L3 |
|------|-----------|---------|--------|---------|-----|
| **集群 0** | 56 | **0-4** | **A725 能效核** | 2808 MHz | instance 0 (12MB) |
| 集群 0 | 56 | **5-9** | **X925 性能核** | 3900 MHz | instance 0 (12MB) |
| 集群 1 | 1144 | 10-14 | A725 能效核 | 2808 MHz | instance 1 (12MB) |
| 集群 1 | 1144 | 15-19 | X925 性能核 | 3900 MHz | instance 1 (12MB) |

- 单 NUMA 域（0-19）；L3 2×12MB（每集群 12MB，非 8MB）
- **关键发现：现有隔离核 16-19 = 集群 1 的 X925 性能核** —— 旧方案把 4 个高频核隔离给 NCCL 轮询（浪费），且 NCCL（集群 1）与 vllm 计算线程（默认 0-15，主落集群 0）跨集群访问

### 1.2 mlx5 IRQ 现状（01 实测）

- 每卡 20 个 comp IRQ（+1 async），当前亲和 = compN → CPU N（0-19 全覆盖）
- **部分 mlx5 IRQ 正落在隔离核 16-19 上** → 隔离核上的 NCCL 轮询被中断打断（待修复）
- irqbalance 四台 **inactive** → 手动亲和不会被覆盖，可持久化

### 1.3 IRQ 绑核复核（03 实测）✅ 推翻旧结论

```
echo 5-9 > /proc/irq/395/smp_affinity_list  →  写后=5-9  ✅ 成功
还原 echo 0-19 → 0-19 ✅
```

- **结论：ARM64 GIC IRQ 绑核完全可行**。旧结论"IRQ 绑核 ARM64 不可行"错误——当时失败是因为绑**隔离核**（isolcpus=16-19 写 smp_affinity 返回 EINVAL）；绑**非隔离核**（5-9）正常
- 用户判断正确：至少可将 IRQ 绑到集群 0 非隔离核（5-9）

## ② 用户方案可行性分析

### 2.1 方案内容
| 项 | 目标 |
|----|------|
| NCCL 隔离核 | **0-4**（A725 能效核，集群 0，与计算同 L3） |
| mlx5 RoCE IRQ | 绑 **5-9**（X925 集群 0）→ 不打隔离核、不跨集群 |
| EngineCore 主线程 | 绑 5-9（X925，集群 0） |
| 其余 vllm 线程 | 仅线程绑定（5-19，isolcpus 自动排除 0-4） |

### 2.2 可行性判定：**✅ 可行且显著优于现状**

| 维度 | 现状（16-19 隔离） | 新方案（0-4 隔离） | 收益 |
|------|------|------|------|
| 计算可用 X925 | 6 个（5-9 + 15） | **10 个**（5-9 + 15-19） | **+67% 性能核**（计算 29× 更忙，直接受益） |
| 计算可用核 | 0-15（16 核） | 5-19（15 核） | -1 核（全为 A725 换出，可接受） |
| NCCL 核频率 | 3900 MHz | 2808 MHz | -28%（轮询够用，需实测确认） |
| NCCL-中断同集群 | 否（16-19 集群1，中断散落 0-19） | **是**（0-4 与 5-9 同集群同 L3 12MB） | 消除跨集群访问 |
| IRQ 打断隔离核 | **存在**（comp16-19 落 16-19） | 无（全绑 5-9） | 消除轮询被中断打断 |
| 能效比 | 高频核空转轮询 | 能效核轮询、高频核计算 | 节能 + 计算提速 |

**核心逻辑（用户判断正确）**：EngineCore CPU 时间 230min vs Worker/NCCL 8min（29×）→ 计算才是资源饥渴方，X925 应全力给计算；NCCL 是延迟敏感（轮询等待为主）非计算密集，A725 能效核足够，且与中断处理核同集群同 L3 是延迟最优布局。

### 2.3 需要注意的点
1. **核 0 隔离**：isolcpus 含 CPU0 是 boot CPU，建议 isolcpus=0-4 时**不配 nohz_full=0-4**（CPU0 tick 无法全停），用 `isolcpus=0-4 rcu_nocbs=0-4`；housekeeping 核 = 5-19（15 核）充裕（vllm+litellm+监控）
2. **A725 轮询延迟**：2808MHz + 小核 IPC，NCCL 16B 延迟可能比 X925 略增（预估 +1-2µs，CPU 轮询仅占 ~1-2µs/步）。**必须实测**（nccl-tests 绑 0-4 vs 绑 16-19）
3. **vllm 绑核落地**：isolcpus=0-4 后普通线程自动排除 0-4 → vllm 进程 `taskset -c 5-19`（计算核全集）即可让计算线程天然落 X925；NCCL 线程需**显式**绑 0-4（隔离核显式 sched_setaffinity 允许，已由 nccl-tests taskset -c 16-19 验证）
4. **NCCL 线程绑 0-4 的手段**：LD_PRELOAD shim（拦截 pthread_create，NCCL* 线程绑 0-4，其余继承 5-19）——Tessa 已设计过 per-thread shim 方案，工程量 ~100 行 C；或用 cgroup cpuset 二段式
5. **B 组（03/04）**：同步落地（对称），03 未接线但设备可见（hotplug DISABLED），IRQ 绑定可先配好

## ③ 四台落地清单（执行序列，待批准）

### Phase 1：隔离核切换（需重启，P0）
```
01/02/03/04: /etc/default/grub.d/90-isolcpus.cfg
  从 isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19
  改 isolcpus=0-4 rcu_nocbs=0-4        # 不配 nohz_full=0（CPU0 boot 核）
  备份 .bak-20260810-cluster0 → update-grub → 逐台重启（01→03→04→02，02 业务最后）
  重启后验证：/proc/cmdline isolcpus=0-4、nproc=15、0-4 无用户线程
```

### Phase 2：mlx5 IRQ 绑 5-9（无需重启，P0，重启前即可配）
```
每台 /usr/local/sbin/mlx-irq-pin.sh：
  遍历 /sys/class/net/{enp1s0f0np0,enp1s0f1np1,enP2p1s0f0np0,enP2p1s0f1np1}/device/msi_irqs/
  逐个 echo 5-9 > /proc/irq/<N>/smp_affinity_list
systemd oneshot（mlx-irq-pin.service，After=network-online.target，仿 mlnx-qos.service）
  或 udev 规则（设备加载即生效，更稳）
验证：/proc/interrupts mlx5 计数全落 5-9 列；隔离核 0-4 上 0 mlx5 中断
```

### Phase 3：vllm 绑核落地（需 TP2 重启，P1）
```
start_head_v026r.sh / start_worker_v026r.sh（01/02）+ groupB（03/04）：
  ① 整进程 taskset -c 5-19（计算线程天然排除 0-4）
  ② NCCL 线程：LD_PRELOAD shim（libncclpin.so，NCCL* 线程 → 0-4）
     或首期用 taskset -c 0-4 单独验证 shim 效果后再并入
  注释理由 + 备份 + bash -n
验证：ps -eLo psr,comm | grep NCCL → 0-4；EngineCore → 5-9 附近
```

### Phase 4：验证与回测（P1）
```
① nccl-tests 16B allreduce：绑 0-4(A725) vs 绑 16-19(X925) 延迟对比（Simple 生产协议）
   - 判据：avg ≤ 27µs（现状 24.6µs + 预估降频余量）、i_p99 ≤ 40µs（SLO 保持）
   - 若 A725 轮询延迟超标 → 回退评估（NCCL 改绑 10-14 集群1 A725？或接受）
② IRQ 绑核前后 i_p99 稳定性对比（interrupts 分布 + i_max 尖峰是否收敛）
③ TP2 32768 以内子集快测（12 格，~30min）确认端到端无回退
④ 生产 4-rank 环网前：03/04 同步验证
```

### 回滚
```
隔离核：恢复 90-isolcpus.cfg 备份 + update-grub + 重启
IRQ：还原 smp_affinity 0-19（或重跑脚本参数化）
vllm：还原脚本备份
```

## ④ 风险清单

| # | 风险 | 缓解 |
|---|------|------|
| 1 | A725 轮询延迟超 SLO（p99>40µs） | Phase 4 实测前置；超标则评估 10-14 或混合方案 |
| 2 | 核 0 隔离的 RCU/housekeeping 异常 | 不配 nohz_full=0；rcu_nocbs 兜底；重启后观察 ksoftirqd/rcu 负载 |
| 3 | 重启中断 TP2（01/02 head-first 重编排 ~10min） | 全量测试采完后再执行；02 litellm 自动恢复 |
| 4 | 02 litellm/监控 挤占计算核 | 5-19 共 15 核，vllm+litellm+监控充裕；Phase 0 已实测 idle 99%+ |
| 5 | IRQ 绑核后单核中断过载（5-9 共 80 个 mlx5 IRQ/卡×4） | 5 核分摊；实测中断吞吐；必要时扩到 5-9 全集合即可 |
| 6 | shim 工程风险（vllm 线程误绑） | 只匹配 "NCCL" 前缀 comm；staging 先验 |

## ⑤ 相关参考
- benchmark-nccl-cpu-pin-4core-matrix-2026-08-09.md（16-19 绑核基线）
- benchmark-tp2-noll-subset-2026-08-09.md（TP2 去 LL 全量 gate 数据）
- Tessa per-thread shim 设计（LD_PRELOAD pthread 亲和 shim 方案）
