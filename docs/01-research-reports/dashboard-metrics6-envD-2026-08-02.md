# Grafana 实时分析页补充 · 6 大常规指标 + 环境 D 核查

**日期**：2026-08-02
**工作流**：监控完善（工作流 4 变体）
**参与成员**：主理人执行层（指标设计/推送/验证）

---

## 📌 TL;DR

- **实时分析页 12 → 18 图**：新增 6 大常规指标（GPU 温度 / 设备功耗 / GPU 占用率 / CPU 占用率 / 外网速率 / 内网交换速率），全部验证出数
- **环境 D 核查结论：未部署**——当前真机仅生产环境（vllm-envc 8h healthy）运行，A/B/C 为测试时临时部署已回收；无第 5 个环境 D 配置
- **参考资料特点**：已梳理社区 3 大参考项目（tonyd2wild / elsung / dredyson）的能力边界与适用场景

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 面板补充 | vllm-realtime 12 → **18 图**（version 3） |
| 新增指标 | 6/6 全部出数验证通过 |
| 环境 D | ❌ 未部署（如需可安排，见行动清单） |
| 数据链路 | DCGM（温度/功耗/占用）+ node-exporter（CPU/网络）双节点 |

---

## 📊 6 大常规指标面板（id 201-206，y=24/32 行）

| # | 面板 | PromQL | 实测值 | 布局 |
|---|------|--------|--------|------|
| 201 | GPU 温度 (℃) | `DCGM_FI_DEV_GPU_TEMP` | 53°C | y=24 x=0 w=6 |
| 202 | GPU 设备功耗 (W) | `DCGM_FI_DEV_POWER_USAGE` | 12.3W | y=24 x=6 w=6 |
| 203 | GPU 占用率 (%) | `DCGM_FI_DEV_GPU_UTIL` | 0%（空闲） | y=24 x=12 w=6 |
| 204 | CPU 占用率 (%) | `100 - avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[1m]))*100` | 1.15% | y=24 x=18 w=6 |
| 205 | 外网速率 (管理网 MB/s) | `rate(node_network_*_bytes_total{device=~"wlP9s9\|enP7s7"}[1m])` rx/tx | 3.2KB/s | y=32 x=0 w=12 |
| 206 | 内网交换速率 (RoCE MB/s) | `rate(node_network_*_bytes_total{device=~"enp1s0f1np1\|enP2p1s0f1np1"}[1m])` rx/tx | 453B/s | y=32 x=12 w=12 |

**接口归属说明**（已真机核实）：
- 外网/管理网：head `wlP9s9`（<NODE_IP>）、worker `enP7s7`（<NODE_IP>）
- 内网 RoCE 数据面：`enp1s0f1np1`（<NODE_IP>）+ `enP2p1s0f1np1`（<NODE_IP>）
- 全部按节点分组显示（{{instance}}），rx/tx 双曲线

---

## 🔍 环境 D 核查结论（未部署）

**当前真机状态**：
- ✅ 生产环境 vllm-envc-node（hybrid-1.6 + dspark5 + 0.85 + thinking=max）healthy 8h
- ⏸ vllm-node（production-ready 镜像）Exited——历史 nomtp 容器
- ❌ **无环境 D 独立部署**（A/B/C 均为昨日基准测试临时部署，测试后已回收）

**如需部署环境 D，候选配置**（基于社区参考资料）：
1. **900K ctx 长上下文**（tonyd2wild 主打：max_model_len=900000，单流 62 t/s，KV 池 962K）——与当前 393K 形成长上下文对照
2. **MTP 投机**（dredyson：num_speculative_tokens=2，acceptance ~68%）——注意本集群 ADR-4 已否决 method=mtp（必崩），dspark 为替代
3. **DSpark 权重 + 无投机**（B' 对照：严格隔离权重变量，补 Archi 标注的实验缺口）

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 刷新 Grafana 实时分析页查看 6 大指标（已就绪） | 用户 | P0 | 已就绪 |
| 2 | 环境 D 部署决策：选 900K ctx / MTP / B' 对照哪种？ | 人类拍板 | P0 | 决策窗口 |
| 3 | 若部署环境 D：脚本生成（复用 start_*_A.sh 改造）+ preflight + 基准 | SRE/Tessa | P0 | 拍板后 |
| 4 | 压测时观察 6 大指标联动（GPU 占用/功耗/内网速率随负载上升） | SRE | P2 | 下轮压测 |

---

## ⚠️ 待完善 / 已知局限

- 外网接口按节点固定匹配（wlP9s9/enP7s7），若网络配置变更需同步更新表达式
- RoCE 字符设备（rocep*）无 node-exporter 流量计数，内网速率取的是 IP 网卡（enp1s0f1np1/enP2p1s0f1np1）口径
- 环境 D 未部署（本次仅核查）

---

## 📚 数据来源 & 成员产出索引

- 主理人执行：指标名采集（DCGM/node 双源）、接口归属真机核实（ip -o addr）、6 面板生成/推送（Grafana API，overwrite=true）、出数验证
- 社区参考资料：tonyd2wild（900K ctx 62 t/s）、elsung（TP2 FP8 41/350）、dredyson（MTP 68%）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
