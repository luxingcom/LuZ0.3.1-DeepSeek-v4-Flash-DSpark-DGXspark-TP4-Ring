# isolcpus 16-19 四核扩展方案 + 四核绑核实测准备（2026-08-09）

**作者**：Rex（SRE）
**状态**：待用户批准（不执行重启，不改 grub）
**目标**：①隔离核 18-19 → 16-19 四核；②四核绑核 + 最佳协议（LL）实测 16B allreduce，验证 SLO P99 ≤ 40µs；③评估 0-15 核剩余负载是否够用。

---

## 一、现状核查（只读，2026-08-09 08:08-08:12 UTC 实采）

### 1.1 isolcpus 配置状态

| 节点 | IP | 90-isolcpus.cfg | /proc/cmdline 生效 | /sys isolated | 可调度核 |
|---|---|---|---|---|---|
| node01 | .186 | `isolcpus=18-19 nohz_full=18-19 rcu_nocbs=18-19` | ✅ 已生效 | 18-19 | 0-17（18 核） |
| node01 | .187 | 同上 | ✅ 已生效 | 18-19 | 0-17（18 核） |
| node01 | .188 | **无** | ❌ 无 isolcpus | 空 | 0-19（20 核） |
| node01 | .189 | **无** | ❌ 无 isolcpus | 空 | 0-19（20 核） |

- grub.d 目录含 NVIDIA 出厂 cfg（iommu/earlycon/pci 等），90-isolcpus.cfg 为独立新增，追加到 GRUB_CMDLINE_LINUX_DEFAULT 尾部。
- 引导方式：**UEFI**；update-grub 可用（/usr/sbin/update-grub）。

### 1.2 核负载现状（mpstat 1s×2 / 10s 采样）

- **01/02/04 全核基本空闲**：平均 %idle 99.7-99.9%（当前无 TP2 生产，仅监控/embed/litellm 轻负载）。
- **18-19 隔离核**：三台均 0% 占用（隔离核当前闲置）。
- **02 的 0-15 核**：平均 %idle 99.6%（litellm ~1-2%、neo4j ~0.6%、prometheus ~0.6%，余全闲）。
- **结论**：扩展到 16-19 隔离后，剩余 0-15 共 16 核，当前总负载不足 1 核当量，**空间充裕**（litellm+监控+PG 等全部放得下）。TP2 训练启动时（8 卡并跑）才可能触及压力，但按现有规划 TP2 进程亲和 0-19 不受 isolcpus 影响。

### 1.3 02 生产/服务清单（重启影响评估核心）

| 服务 | 端口 | 形态 | restart policy | 重启后自动恢复 |
|---|---|---|---|---|
| **litellm-proxy（生产在用）** | 4000 | 容器 host 网络 | **unless-stopped** | ✅ docker 自动拉起 |
| responses_gateway（8003→.60:8001） | 8003 | systemd **user** service | Linger=**yes** + enabled + Restart=always | ✅ 自动恢复 |
| registry（镜像仓库） | 5000 | 容器 | always | ✅ |
| aicad compose 栈（prometheus/grafana/postgres/redis/neo4j/chromadb/alertmanager/minio） | 8191/3000/8082/6379/7474/8180/9093/19000… | compose（docker compose v5） | unless-stopped/always | ✅ |
| dcgm-exporter / node-exporter | 9400/9100 | 容器 | unless-stopped | ✅ |
| docker.service | — | systemd | enabled | ✅ |

- **02 上当前无 TP2/vllm worker 进程**（无 vllm/torchrun/25000 监听）→ 02 重启不涉及正在运行的 LLM 训练。
- **⚠️ 关键风险：02 的 L2 地址 `<NODE_IP>/30`（enp1s0f0np0）与 `<NODE_IP>/30`（enP2p1s0f0np0）未在 netplan 持久化**（97-roce-mtu.yaml 仅写 MTU，99-nvidia-sync 仅含 module1 的 10.100.x）。重启后 **L2（02↔04）地址丢失**，需补回，否则 NCCL L2 链路断。

### 1.4 01 / 03 / 04 服务状态

| 节点 | 服务 | 备注 |
|---|---|---|
| 01 | 仅 aicad 监控栈 + aicad-fw-25000（防火墙容器, unless-stopped）+ 桌面会话 | **无 TP2 生产**（无 8001/25000 监听）；10.100.x 已 netplan 持久化 |
| 03 | anemll-embed-8022（生产 embed, 8022） | RoCE 四口全 NO-CARRIER（未接线），无 isolcpus |
| 04 | anemll-embed-8022（生产 embed, 8022, unless-stopped） | **⚠️ L2 地址 <NODE_IP>/14 未持久化**；无 isolcpus |

- litellm 上游 embed 指向 **.188:8022 和 .189:8022**（03/04 均在生产 embed，互为兜底）。

### 1.5 重启后需复核的持久化项（现状均已就绪）

| 项 | 状态 | 重启后行为 |
|---|---|---|
| netfilter-persistent | enabled + /etc/iptables/rules.v4 存在（01/02/04） | ✅ 自动恢复（内网限 TCP、FORWARD DROP） |
| mlnx-qos.service | enabled + active（01/02/04；03 未装） | ✅ 自动恢复（PFC P3/P5） |
| MTU 9000（16 RoCE 口） | netplan 97-roce-mtu（module0）+ 99-nvidia-sync-cluster（module1）已持久化 | ✅ 自动恢复 |
| sysctl 99-sec.conf（rp_filter=1 等） | 存在（01/02/04） | ✅ 自动恢复 |
| **10.20.0.x L2 地址（02/04 module0）** | **❌ 未持久化** | **⚠️ 丢失，需补回** |

### 1.6 时区

- 01 = HKT(+8)，02/03/04 = UTC(+0)。**执行窗口与告警时间线请以此换算**（02 的"业务低谷"以 UTC 计）。

---

## 二、变更方案（待批准，未执行）

### 2.1 目标配置

```
/etc/default/grub.d/90-isolcpus.cfg 改为：
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19"
```

每节点完整步骤（01/02/04）：
1. 备份：`sudo cp -a /etc/default/grub.d/90-isolcpus.cfg /etc/default/grub.d/90-isolcpus.cfg.bak-$(date +%Y%m%d)`
2. 改值 18-19 → 16-19（04 需新建文件）
3. `sudo update-grub`
4. 验证生成：`grep -o "isolcpus=[^ ]*" /boot/grub/grub.cfg | head -1` 应为 `isolcpus=16-19`
5. **重启（单独批准，不在本方案内执行）**
6. 重启后复核（见 §2.3）

### 2.2 变更清单（每台：改什么 / 影响 / 回滚）

| # | 节点 | 改什么 | 影响 | 回滚 |
|---|---|---|---|---|
| C1 | **02**（.187） | isolcpus 18-19→16-19 + update-grub + 重启 | ① **litellm(4000) 生产中断约 5-15 分钟**（重启+容器自动拉起；unless-stopped 保证恢复）② 监控/Grafana 断档同窗口 ③ registry/DB 断档同窗口 ④ **L2 <NODE_IP>/13 地址丢失需补回** ⑤ responses_gateway(8003) 自动恢复 | 恢复 cfg 备份 + update-grub + 重启（同样窗口） |
| C2 | **01**（.186） | 同上 | ① 无业务生产，仅监控短断档 ② aicad-fw-25000 自动恢复 ③ 10.100.x 自动恢复 | 同上 |
| C3 | **04**（.189） | **新建** isolcpus=16-19 + update-grub + 重启 | ① embed(8022) 中断约 5-15 分钟（**有 03 embed 兜底**，litellm 上游双活）② **L2 <NODE_IP>/14 地址丢失需补回** | 删除文件 + update-grub + 重启 |
| C3' | **04 不加（备选）** | 不动作 | 04 绑核落到普通核（无 nohz_full/rcu_nocbs 保护），T2_24/对角数据降权（与现状一致）；零中断 | — |
| C4 | **03**（.188） | **不加 isolcpus** | 03 未接线，无 NCCL 参与，加了无收益且有 embed 中断风险；**待接线成环后再补** | — |
| C5 | **02/04** | L2 地址持久化（写 netplan 或重启后手动 `ip addr add`） | 防止 10.20.0.x 丢失导致 L2 断 | 删除地址配置 |

**建议**：C5 作为 C1/C3 的前置/伴随项，**推荐把 10.20.0.x 补进 97-roce-mtu.yaml**（该文件已含 MTU+dhcp4:false，加 addresses 即可，重启自动恢复），一次性解决。

### 2.3 重启后复核清单（逐台）

```
1. /proc/cmdline 含 isolcpus=16-19 nohz_full=16-19 rcu_nocbs=16-19
2. cat /sys/devices/system/cpu/isolated  = 16-19
3. nproc = 16（可调度 0-15）
4. iptables -S：内网限 TCP + FORWARD DROP 在（netfilter-persistent）
5. systemctl is-active mlnx-qos = active；mlnx_qos 输出 P3/P5
6. ip -o link show ... mtu=9000（16 口）
7. 02/04：ip addr show enp1s0f0np0/enP2p1s0f0np0 有 10.20.0.x（补回）
8. 02：curl -s localhost:4000/health（litellm 恢复）+ ss :8003（responses_gateway）
9. 04/03：curl -s localhost:8022/health（embed 恢复）
10. docker ps 全部容器 Up（对照 §1.3 清单）
11. 01：docker logs aicad-fw-25000（规则已加载）
```

### 2.4 建议执行窗口

- **顺序建议**：先 **01**（无业务，验证 grub+重启流程）→ 再 **04**（embed 有 03 兜底）→ **最后 02**（litellm 生产，选业务低谷）。
- **02 窗口**：UTC 低峰 = **UTC 18:00-22:00（HKT 02:00-06:00）** 最稳妥；若业务确认可短停，也可选任意低请求时段。litellm 有 --num_workers 2 与 PG 层限流，突发请求在重启窗口会 5xx/超时，需业务侧接受。
- 01/04 可与 02 同批执行（一次维护窗口重启 3 台，每台间隔 ~10-15 分钟，总窗口约 1 小时）；或 04 单独成批。

---

## 三、四核绑核实测准备（Task B，等协议结论 + 用户批准后执行）

### 3.1 taskset 预演结论（已实测，**不阻塞**）

- `taskset -c 16-19` 在当前 18-19 隔离下**可用**：16-17 普通核 + 18-19 隔离核均允许绑定，返回 affinity `16-19`（01/02 均验证）。隔离核不拒绝 taskset 绑定（ARM64 限制仅在 IRQ 亲和，进程亲和不受影响）。
- `taskset -c 18-19` 同样可用（对照）。
- 04 未配 isolcpus 也能 taskset 绑 16-19（普通核），但无 nohz/rcu 隔离保护。

### 3.2 四核绑核 wrapper：pin_ar4.sh

基于既有 t2wrap.sh/t2_24wrap.sh 结构，per-node HCA + LINK 参数化（L1=01↔02 module1，L2=02↔04 module0），NCPIN=1 绑 16-19：

```bash
#!/bin/bash
# pin_ar4.sh: 四核绑核(16-19) all_reduce_perf wrapper
# 用法: LINK=L1|L2 NCPIN=1 pin_ar4.sh <args>   (经 mpirun -x 传入)
export LD_LIBRARY_PATH=/opt/nccl-2307/nvidia/nccl/lib:$LD_LIBRARY_PATH
H=$(hostname)
LINK=${LINK:-L1}
case "$H:$LINK" in
  node01:L1|node01:L1) export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 ;;
  node01:L2|node01:L2) export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 ;;
  node01:ALL) export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 ;;
  *) echo "[pin4] no HCA map for $H:$LINK"; exit 1 ;;
esac
export NCCL_SOCKET_IFNAME=enP7s7
echo "[pin4-$$] $(hostname) LINK=$LINK pre-aff: $(taskset -pc $$ 2>/dev/null)"
if [ "$NCPIN" = "1" ]; then
  taskset -c 16-19 /opt/nccl-tests/build/all_reduce_perf "$@" &
  PID=$!
  for i in $(seq 1 20); do
    [ -r /proc/$PID/status ] && { echo "[pin4-$$] $(hostname) PID=$PID aff: $(grep Cpus_allowed_list /proc/$PID/status)"; break; }
    sleep 0.2
  done
  wait $PID
  echo "[pin4-$$] $(hostname) exit=$?"
else
  /opt/nccl-tests/build/all_reduce_perf "$@"
  echo "[pin4-$$] $(hostname) exit=$?"
fi
```

### 3.3 实测命令序列（等批准后执行）

环境（沿用既有）：NCCL 2.30.7、nccl-tests 2.19.7、OpenMPI 4.1.6、管理口 MPI、GID_INDEX=2（per-node HCA 规避 169.254 污染）、--bind-to none。

```
# 前置：部署 pin_ar4.sh 到 01/02(/04)
scp pin_ar4.sh <USER>@<NODE_IP>:/tmp/nccl_pin/
ssh ... 'sed -i ...'  (可选：04 需先部署到 .189)

# hostfile
# hostfile_12:  <NODE_IP> slots=1 / <NODE_IP> slots=1
# hostfile_24:  <NODE_IP> slots=1 / <NODE_IP> slots=1

# T2_12 四核绑核 + LL（主测）
mpirun --allow-run-as-root -np 2 --hostfile /tmp/nccl_pin/hostfile_12 \
  --map-by node --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_PROTO=LL -x NCPIN=1 -x LINK=L1 \
  /tmp/nccl_pin/pin_ar4.sh -b 16 -e 16 -f 2 -g 1 -w 100 -n 5000 -z 0 -I 1

# T2_24 四核绑核 + LL（04 需加 isolcpus 才有隔离保护；不加则降权）
mpirun --allow-run-as-root -np 2 --hostfile /tmp/nccl_pin/hostfile_24 \
  --map-by node --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_PROTO=LL -x NCPIN=1 -x LINK=L2 \
  /tmp/nccl_pin/pin_ar4.sh -b 16 -e 16 -f 2 -g 1 -w 100 -n 5000 -z 0 -I 1

# 对照：T2_12 四核绑核 Simple 协议（隔离绑核收益 vs 协议收益解耦）
... -x NCCL_PROTO=Simple ...

# 对照：T2_12 双核绑核（现状 18-19，NCPIN 改 18-19）→ 四核 vs 双核差异
# 对照：单核 18（NCPIN=18，t2wrap 既有）→ 复核"2-rank 只用 1 核"
# 短窗抖动：-w10 -n100 快速采样若干组，验证 i_p99 分布
# 对角（如成环后）：3-rank 04-02-01，LINK=ALL on 02，t3wrap 改 16-19

# 采样：每组 3-5 runs，记录 OOP avg / i_p99 / IP avg / IP i_p99，比对 SLO P99≤40µs
```

### 3.4 验收判据

- **主验收**：绑核 16-19 + LL，16B allreduce 2-rank，`i_p99 ≤ 40µs`（新 SLO），avg 目标 ≤ 17.5µs（对齐 18-19 双核最佳 17.3µs 或更优）。
- 四核 vs 双核对比预期：2-rank 单通信实际只占 1-2 核，**四核主要收益在 4-rank/TP4 或并发场景**；若 2-rank 无改善属预期，需如实记录。
- 抖动指标：i_max 尖峰记录（IRQ/softirq 无法绑隔离核为已知限制，偶发 1.5-3ms 尖峰预计仍存在）。

---

## 四、风险与红线

- **红线遵守**：本方案只出方案，**未执行任何重启/grub 改动/容器操作**；仅做只读核查 + taskset 进程级验证（sleep 进程，未触碰生产）。
- **最大风险**：02 重启中断 litellm 生产；L2 10.20.0.x 地址丢失。缓解：选业务低谷 + C5 持久化 + 启动后按 §2.3 复核。
- **已知限制**：IRQ/softirq 无法绑隔离核（ARM64 GIC）；04 不加 isolcpus 时 T2_24 数据降权；对角 3-rank 在链式拓扑不可 init（需成环）。
- **时区**：执行窗口以 UTC 为准（02/03/04），01 为 HKT(+8)。
