# RoCEv2 与 InfiniBand 切换可行性调研报告

**日期**：2026-08-06
**工作流**：技术调研（网络架构）
**参与成员**：Archi（架构师）

---

## 📌 TL;DR

- **❌ 不能切换 IB**：DGX Spark 内建 ConnectX-7（MT4129）硅片支持 IB（VPI 双模），但被 NVIDIA **固件锁定为仅 Ethernet/RoCE 模式**——官方明确"不支持 IB，无未来计划"
- **✅ 即便能切收益也小**：2 节点物理直连（无交换机、无拥塞、错误计数 0），RoCE 的 PFC/ECN 短板基本不生效，IB 无损优势被直连"抹平"到 <5-10%
- **✅ 建议保持 RoCEv2**；近期 NCCL 卡死是软件竞态（init 顺序），与网络 fabric 无关，IB 也避免不了

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 保持 RoCEv2（不切换） |
| 阻塞项 | 0（硬件固件锁定，无切换路径） |
| 关键行动项 | 2 条（监控链路 / 确认 GDRDMA） |

---

## 1️⃣ 硬件结论（官方资料 + 本机核实）

| 项 | 结论 |
|----|------|
| 网卡型号 | **ConnectX-7（MT4129）**，vendor_part_id=4129，fw 28.45.4028，board NVD0000000087（NVIDIA 定制） |
| 硅片能力 | 标准 CX7 是 VPI 双模卡（IB NDR200/HDR + 200GbE，官方 datasheet） |
| **实际模式** | **固件锁定仅 Ethernet/RoCE**——NVIDIA 论坛版主明确："DGX Spark does not support InfiniBand mode, no plans to add in the future" |
| 本机核实 | 4 个 RDMA 设备全命名 `roce*`，sysfs link_layer=Ethernet，活动口 200 Gb/sec (4X HDR)；mlx5_ib/ib_uverbs/ib_core 已加载（RoCE verbs 完整）；**无 mlxconfig/mst**（需另装 MFT，且受固件限制） |
| 带宽瓶颈 | STH 实测：PCIe Gen5 x4，双口约 96Gbps——**瓶颈在 PCIe 而非链路模式** |

## 2️⃣ IB vs RoCE 对比（对我们的直连场景）

| 维度 | IB | RoCEv2 | 对 2 节点直连差异 |
|------|-----|--------|------------------|
| 无损 | 链路级 credit 流控，原生 lossless | 依赖 PFC/ECN | 直连无拥塞、错误计数 0 → **差异 ≈ 0** |
| 延迟 | ~1µs | ~1.5-2µs | 可忽略（TP=2 通信量小） |
| NCCL all-reduce | 40-55 GB/s | 35-50 GB/s（无损配置） | 单链路直连 <10% |
| 运维 | 需 Subnet Manager | 以太网栈 | 直连均不需 SM |

## 3️⃣ 结论与建议

**不切换**，理由：
1. **固件锁定不可切**——官方无开放计划、无成功解锁案例；强刷第三方固件风险高（保固 + 变砖），不可取
2. **收益小**——2 节点点对点直连，IB 无损优势被"抹平"到 <5-10%；TP=2 下 ~1µs 延迟差无感
3. **NCCL 卡死与 fabric 无关**——已确认为 init 顺序竞态（软件），IB 也避免不了

**保持 RoCEv2 的优化方向**：
- `NCCL_DEBUG=INFO` 确认 "NET/IB"（GDRDMA）生效
- 保持 MTU 9000
- 持续监控链路错误计数（当前 0）

## 📚 数据来源

- NVIDIA 论坛（官方确认不支持 IB）：forums.developer.nvidia.com/t/connecting-dgx-spark-to-mellanox-infiniband-sb7800/355444
- NVIDIA 论坛（mlxconfig missing）：forums.developer.nvidia.com/t/nvidia-dgx-spark-mlxconfig-missing/363477
- ConnectX-7 datasheet（VPI 双模能力）：nvidia.com（infiniband-connectx7-data-sheet.pdf）
- vCluster IB vs RoCE 实测：vcluster.com/blog/gpu-cluster-networking-infiniband-roce
- NVIDIA NIM on DGX Spark（官方仅 RoCE 部署）：docs.nvidia.com/nim/.../deploy-on-dgx-spark.html

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
