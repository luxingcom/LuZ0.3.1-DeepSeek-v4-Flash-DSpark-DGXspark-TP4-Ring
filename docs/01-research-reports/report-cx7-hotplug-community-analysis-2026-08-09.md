# 社区调查报告：ConnectX-7 网卡在 DGX Spark 上"消失"——热插拔省电特性（cx7-pcie-hotplug）

**日期**：2026-08-09
**调查人**：Archi（架构师）· 工程保障团队
**主题**：03 的 ConnectX-7 网卡从 PCIe 总线消失的社区根因分析、解决方案矩阵与处置建议
**性质**：纯分析任务，未执行任何服务器操作

---

## 📌 TL;DR（执行摘要）

- **根因已确认（非个例，社区广泛报告）**：DGX OS GA2 OTA2 / OTA 7.5.0 引入的 **ConnectX-7 Hot-Plug Power Saving** 特性。`cx7-pcie-hotplug` 驱动（ACPI 设备 `MTKP0001`）在**端口未插线**时，自动将 ConnectX-7 从 PCIe 总线**热移除**（省电 ~18W：19W→1W），插线后自动恢复。**这不是 bug，是 feature**。
- **03 现状完全吻合**：两个 200G 口均未插线 → lspci 无 mlx5、IB class 为空、dmesg 见 "Cable removal" + AER RxErr correctable（官方确认为良性）→ 均与热插拔特性一致，**不是硬件故障、也不是 MODULE_SPLIT 后遗症**。
- **归因修正**：此前误判的 "MODULE_SPLIT 写入后遗症" 应撤回。Secure FW 下读不到 NVConfig 明细、且未执行任何写入——与本次消失**无因果关系**。
- **处置推荐（详见 §3）**：**稳态保持热插拔启用**（省电 + 成环插线自动恢复）；仅在"单机带载联调/排障需要设备恒可见"的临时窗口内，对个别节点**临时禁用**热插拔，联调完成后再恢复。
- **社区要点**：禁用开关 = `rm /etc/nvidia/cx7-hotplug-enabled` + 重启；恢复 = `touch` 同路径。OEM/白牌 GB10（如 Gigabyte）有"热插拔重连后链路降到 25Gbps"的个例，需在成环后用 ethtool 复核协商速率；**务必使用 NVIDIA 认证线缆**（Amphenol / Luxshare），第三方线缆（如 FS）可能被 EEPROM 校验误判为 Cable removal。

---

## 1. 根因分析

### 1.1 特性来源与版本依赖

| 项 | 内容 |
|----|------|
| 特性名称 | ConnectX-7 Hot-Plug Power Saving（热插拔省电） |
| 引入版本 | DGX OS GA2 OTA2（MSI/官方确认）；中文社区观测对应 **OTA 7.5.0** |
| 实现驱动 | `cx7-pcie-hotplug`（ACPI 设备 `MTKP0001`） |
| 管理文件 | `/etc/nvidia/cx7-hotplug-enabled`（存在=启用；删除=禁用） |
| 处理脚本 | `/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh`（子命令 `plug-in` / `removal`） |
| 省电量 | ~18W（idle 19W → 1W）；整机 idle 实测 ~37W → ~25W（带显）/ ~22W（headless），降幅 32%+ |
| 预期行为 | 未插线 → 热移除 → lspci 不再显示 Mellanox 端点；插线 → 自动恢复 |

> 官方论坛版主 aniculescu 原话（Thread #366076）：
> "There is a hotplug feature which disables the CX7 cards if there is no cable attached for power saving purposes. If you attach a cable you should see it in lspci. Alternatively, you can completely disable hotplug feature by removing the file `/etc/nvidia/cx7-hotplug-enabled`. If you want to enable it again, just create that file: `touch /etc/nvidia/cx7-hotplug-enabled`"

### 1.2 完整因果链（社区共识）

```
OTA 升级到 GA2 OTA2 / 7.5.0
  → ACPI / 固件配置变更，启用 hotplug_enabled = 1
    → cx7-pcie-hotplug 驱动开始监控 ConnectX-7 外部线缆状态（QSFP 口）
      → 检测到端口无线缆（NO-CARRIER）
        → 触发 "cx7-pcie-hotplug MTKP0001:00: Cable removal" 事件
          → 关闭 PCIe 链路（LnkSta Width x0）+ 触发 PCIe 设备热移除
            → lspci / ip link / IB class 均不可见（设备被动态移除，非硬件消失）
```

配套内核日志（社区多贴一致）：
- `cx7-pcie-hotplug MTKP0001:00: Cable removal`
- `mlx5_core ...: Detected insufficient power on the PCIe slot (27W)` —— 未插线时 PCIe slot 功率预算 27W 低于 CX7 有链路阈值，**良性告警**
- root port `AER: ... RxErr` **correctable** —— 同一链路状态转换的已知良性行为（MSI Q4 官方答复），**不是硬件故障**

### 1.3 为什么"自行恢复可见"后又消失

- 每次**冷启动**时，CX7 会先被 PCIe 固件正常枚举（约前 20-22 秒内 lspci 可见、驱动正常加载、接口改名完成）；随后 `cx7-pcie-hotplug` 驱动完成线缆判定 → 触发 "Cable removal" → **在启动后约 20-22 秒**将设备热移除。
- 因此 03 上午 12:30"自行恢复可见"完全可解释为：**启动/枚举窗口内的瞬时可见**（或某次 cable 状态重扫后的短暂恢复）；**重启后流程重演 → 设备再次消失**。行为与热插拔特性完全吻合，与硬件故障无关。
- 社区已有同样观测：Thread #374275（Lausanne）——boot 后 20-22 秒设备消失，必须整机断电才回来。

### 1.4 关键认知

1. **设备不可见 ≠ 设备不存在**：PCIe 热插拔可在运行时动态添加/移除设备；lspci 无输出不代表硬件损坏。
2. **内核日志是最可靠证据**：`journalctl -b -k` 保留完整启动日志，可看到"检测到 → 移除"完整生命周期。
3. **03 此前 dmesg 的 E-Switch cleanup + AER correctable 均与链路 down/热移除转换一致**，不是故障信号。
4. **field diagnostics（a_id/5767）不能当作 CX7 存在性检查**：社区报告（Thread #361099）显示 fieldiag 在 CX7 完全缺失时仍返回 PASS（跳过缺失组件）——它只用于**硬件健康**判断，不用于判断"设备是否被热移除"。

---

## 2. 解决方案矩阵

| # | 方案 | 操作 | 生效方式 | 功耗 | 适用场景 | 风险/备注 |
|---|------|------|---------|------|---------|-----------|
| **A** | **插线（成环时自然恢复）** | 将 QSFP 线缆插入 03 的 200G 口（module0/module1） | 自动热插拔恢复，无需重启 | 插线后 ~19W（有链路） | **成环/组网、有物理线缆可用** | 推荐首选。需用 NVIDIA 认证线缆；插线后 1 分钟内复核 lspci + 协商速率 |
| **B** | **禁用热插拔（设备恒可见）** | `sudo rm -f /etc/nvidia/cx7-hotplug-enabled` + 重启 | 重启后生效 | 每台 +18W（idle 回到 ~19W） | 排障窗口、单机联调、设备需要恒可见 | 需重启；OTA/重装后可能重新启用（需复核）；见 §5 |
| **C** | **官方 field diagnostics** | 运行 a_id/5767 工具（`dgx-spark-fieldiag`），把日志 DM/交给 NVIDIA 支持 | 诊断输出 | — | 怀疑硬件故障、走 RMA 前置 | **不能**仅凭它判定 CX7 缺失；PASS ≠ 设备在 lspci 中可见 |
| **D** | **RMA / 现场** | 联系 NVIDIA DGX Spark 支持（Founders Edition），引用论坛 thread | 硬件更换 | — | 仅在**确认硬件故障**时（如插线后仍不出现、fieldiag 报硬件 FAIL） | 社区确认热插拔场景多为误报，RMA 前先走 A/B/C |

### 推荐排序

1. **首选 A（插线）**——根因是"没插线"，成环插线即恢复，零配置、零风险、符合官方设计。
2. **排障/联调期临时用 B（禁用）**——当 03 需要单独带载联调、验证 RoCE/netplan、或 A 未自动恢复时，临时禁用以获得确定性可见性；验证完成后恢复。
3. **C（field diagnostics）**——仅当"插线后仍不出现"或怀疑硬件时执行，作为 RMA 前置证据。
4. **D（RMA/现场）**——仅 C 确认硬件故障后；不要把热插拔误判为硬件故障走 RMA（社区已有大量此类误判先例）。

---

## 3. 对 03 的处置建议

### 3.1 结论：**稳态保持热插拔启用 + 成环插线时自动恢复**（主路径）

**推荐动作**：不对 03 做任何"修复"操作；在成环接线时把 200G 线缆插入 03 的 module0/module1 口，设备应自动恢复枚举与 RoCE 功能。

### 3.2 理由

| 维度 | 评估 |
|------|------|
| 根因确定性 | 社区多来源交叉验证，03 现状（双口未插线 + Cable removal + AER correctable）与特性行为完全吻合；无硬件故障证据 |
| 成环时间线 | 03 补线是既定部署动作（接线调查已明确"03 补线后即可扩展"）。**插线即恢复，无需额外变更** |
| 操作成本 | 保持启用 = 零操作；禁用 = 每台需重启 + 配置漂移风险 + 需在 OTA 后复核 |
| 功耗 | 集群 24×7 常开：4 节点保持启用可省 ~72W idle；对常开实验集群是真实收益（~32% idle 降幅） |
| 可观测性代价 | 未插线时设备不可见是**预期行为**，应通过告警基线修正（把"无 cable 且无 CX7"降级为 INFO）而非禁用特性 |
| 一致性 | 四机统一策略，避免个别节点行为漂移；03 禁用而 01/02/04 启用会造成排查口径混乱 |

### 3.3 例外窗口（何时改用 B）

满足以下任一，**建议在 03 上临时禁用热插拔**（做完再恢复）：

1. **成环前需要 03 单独联调**：如验证 netplan / RoCE GID / 单机 RDMA 连通性，需设备恒可见；
2. **插线后 1-2 分钟内未自动恢复**：此时先用 `mtk-hotplug-handler.sh plug-in` 触发重扫；仍无效再临时禁用排查；
3. **监控持续误报、影响值班判断**：先改告警基线，不改设备行为（避免为了监控破坏省电特性）。

> 决策判据一句话：**有物理线缆就插线（走 A）；没线缆又要动它（走 B 临时窗口）；两个都不沾就什么都不做。**

---

## 4. 验证步骤（禁用热插拔后的确认流程）

> 以下为操作预案，实际执行需主理人/相关成员在服务器上运行。

### 4.1 确认热插拔状态与禁用

```bash
# 查看当前是否启用（文件存在=启用）
ls -la /etc/nvidia/cx7-hotplug-enabled

# 若需临时禁用（B 方案）
sudo rm -f /etc/nvidia/cx7-hotplug-enabled
sudo reboot

# 恢复启用
sudo touch /etc/nvidia/cx7-hotplug-enabled
sudo reboot
```

### 4.2 禁用后确认设备重新枚举

```bash
# 1) PCI 层：应重新看到 Mellanox 端点（0000:01:00.0/1 + 0002:01:00.0/1，共 4 个 function 项）
lspci | grep -i mellanox
# 期望输出形如：
#   0000:01:00.0 Ethernet controller: Mellanox Technologies MT2892 Family [ConnectX-7]
#   0000:01:00.1 ...
#   0002:01:00.0 ...
#   0002:01:00.1 ...

# 2) sysfs：确认 link 状态（应有 x4 lanes / width 或正常）
for d in 0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1; do
  echo "== $d =="; cat /sys/bus/pci/devices/$d/enable 2>/dev/null; \
  lspci -s $d -vv | grep -E "LnkSta:|LnkCap:" 2>/dev/null
done

# 3) 网络接口：enp1s0f0np0 / enp1s0f1np1 / enP2p1s0f0np0 / enP2p1s0f1np1 应出现
ip link | grep -E "enp1s0f0np0|enp1s0f1np1|enP2p1s0f0np0|enP2p1s0f1np1"

# 4) 驱动信息
ethtool -i enp1s0f0np0   # driver=mlx5_core, firmware-version=28.x

# 5) hotplug 处理脚本状态（脚本存在性 + 子命令用法）
ls -la /opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh
sudo /opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh plug-in   # 仅插线场景；先确认子命令
```

### 4.3 恢复后验证 RoCE 功能（成环后）

```bash
# RDMA 层
rdma link show
ibstat 2>/dev/null || ibv_devinfo -l

# 环内对端互通（以 03↔某对端为例，替换对端 IP/GID 模式）
ib_write_bw -d mlx5_0 --report_gbits -x 3 -F <peer_ip>   # 直连测试
ib_read_lat -d mlx5_0 --report_gbits -R <peer_ip>        # rdma_cm 模式（支持路由，2 跳场景）

# 协商速率复核（防 OEM 个例降速）
ethtool enp1s0f0np0 | grep -E "Speed|Link detected"
# 期望 200000Mb/s；若为 25000Mb/s → 属热插拔重连降速个例，重启恢复
```

### 4.4 保持启用态下的快速判据

```bash
# 未插线：设备不可见 = 正常，不必当事故
# 插线后：等待 ≤1 分钟自动恢复；未恢复 → 手动 plug-in 重扫；仍不行 → 查线缆认证/临时禁用排查
```

---

## 5. 风险与注意事项

| # | 风险/注意 | 说明与缓解 |
|---|-----------|-----------|
| 1 | **禁用热插拔的功耗代价** | 每台 +18W（idle 19W→1W 的收益消失）；4 台全禁 ≈ +72W。对常开集群是真实成本，故推荐只在临时窗口用 B |
| 2 | **OTA 升级 / 系统重装可能重新启用** | 热插拔启用状态由驱动/ACPI 管理，`cx7-hotplug-enabled` 可能在后续 OTA、OS 重装、OOBE 更新后被重建。**每次升级后复核该文件存在性**；若发现设备再次"消失"，先查此文件，勿误判硬件 |
| 3 | **第三方/非认证线缆** | 社区报告 FS 等第三方 QSFP 线缆可能被 EEPROM/厂商校验误判 → 反复 Cable removal / NO-CARRIER。成环务必用 NVIDIA 认证线缆：**Amphenol** NJAAKK-N911（0.4m）/ NJAAKK0006（0.5m）、**Luxshare** LMTQF022-SD-R |
| 4 | **OEM/白牌 GB10 个例：热插拔重连后降速** | Gigabyte GB10 用户报告重连后链路降至 25Gbps，需重启恢复；DGX Spark FE + 认证线缆一般不会出现，但成环后仍应 ethtool 复核协商速率（200G） |
| 5 | **AER RxErr correctable 属良性** | 不要因 correctable 报错走 RMA；它伴随链路状态转换出现，安装 GA2 OTA2 后显著减少 |
| 6 | **fieldiag 不能判断设备存在性** | `dgx-spark-fieldiag` 在 CX7 缺失时可能仍 PASS（跳过缺失组件）。**判断"热移除 vs 硬件故障"要以"禁用热插拔后 lspci 是否重现"为准** |
| 7 | **勿在业务流量/训练任务运行中手动 removal** | `mtk-hotplug-handler.sh removal` 会热移除设备；仅在无活跃 NCCL/RDMA 流量时使用，避免会话中断 |
| 8 | **多机一致性** | 四机启用/禁用状态保持一致（或明确记录各节点状态），避免排查口径混乱；建议在文件注册表/交接文档中登记每节点 hotplug 状态 |
| 9 | **监控告警基线** | Grafana/值班告警应将"无 cable 且 CX7 不在 lspci"从 CRITICAL 降为 INFO；否则 03 这类节点会持续误报 |

---

## 6. 社区/证据来源清单

| # | 来源 | 关键内容 |
|---|------|---------|
| 1 | NVIDIA 论坛 Thread #366076 "Mellanox Cards Not Detected on DGX Spark After Updates" | 版主 aniculescu：热插拔省电特性；`rm/touch /etc/nvidia/cx7-hotplug-enabled` 开关 |
| 2 | NVIDIA 论坛 Thread #362667 "My QSFP ports on my DGX Spark are not working" | NVIDIA 员工 relc：field diagnostics a_id/5767、RMA 路径；ericmwilliams/aniculescu 确认 power-saving 行为 |
| 3 | NVIDIA 论坛 Thread #374275 "ConnectX-7 network cards disappear after DGX Spark system update" | Lausanne：boot 后 20-22s 消失、27W 告警、FS 第三方线缆 EEPROM 校验疑点 |
| 4 | NVIDIA 论坛 Thread #361099 "CX7 not discoverable even with secureboot disabled" | jasonaduclos：fieldiag PASS 但 CX7 缺失（跳过缺失组件）；插线+冷启动后恢复 |
| 5 | NVIDIA 论坛 Thread #371031 "CX7 throughput drops to 25 Gbps after hot-plug/cable reconnection (Gigabyte GB10)" | OEM 个例：热插拔重连后降速 25G，重启恢复；禁用 hotplug 需 rm + reboot |
| 6 | MSI 论坛 MS-C931 帖（Q3/Q4） | 官方答复：热插拔预期行为 + mtk-hotplug-handler.sh plug-in/removal 验证命令；AER RxErr correctable 已知良性 |
| 7 | NVIDIA DGX Spark 官方文档（spark-clustering） | QSFP 200G、twin 双逻辑口、每口 2 个 PCIe 地址、认证线缆清单 |
| 8 | 中文博客 nnsay.cn（OTA 7.5.0 因果链详解） | OTA 7.5.0 → ACPI → hotplug_enabled=1 → 无线缆触发 Cable removal → PCIe 热移除；journalctl -b -k 证据链 |
| 9 | 媒体评测（Tom's Hardware/Yahoo/Tectack） | idle 功耗 ~37W→25W/22W（-32%），与 ~18W NIC 省电口径一致 |

---

## 7. 结论与行动建议

1. **撤回** "MODULE_SPLIT 写入后遗症" 的归因；**确认根因 = 热插拔省电特性 + 未插线**。
2. **03 不做修复操作**；成环接线时插线，验证自动恢复 + RoCE。
3. **本次调查的落盘产出**：本文件 `report-cx7-hotplug-community-analysis-2026-08-09.md`。
4. **后续动作**：①更新监控告警基线（无 cable+CX7 不可见=INFO）；②注册登记各节点 hotplug 状态；③OTA 升级后复核 `/etc/nvidia/cx7-hotplug-enabled`；④成环使用 NVIDIA 认证线缆并复核 200G 协商速率。

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
