# TP4 环网部署总计划（维护窗口执行清单）

**日期**：2026-08-11
**工作流**：工作流 2（系统设计）+ 工作流 4（部署前检查）+ 加固实施
**参与成员**：Docu（操作须知）、Rex（加固+QoS 持久化）、Archi（TP4 playbook）、Zhen（汇编）
**状态**：🟡 准备完成，待维护窗口执行

---

## 📌 TL;DR（执行摘要）

- **基础条件全部就绪**：环网 L3 闭合（8 口全配 IP）、iptables 白名单四台复核 ✅、QoS DSCP trust 四台统一 + systemd 持久化 ✅（含 NCCL 优先级映射验证）、rules.v4 拆分工具已部署（净化动作留维护窗口）。
- **关键内存纠正**：TP4 单机权重 **38.9GiB**（155.43/4，比 TP2 的 77.7GiB 减半）→ KV 余量 ~58GiB（3×），**max-num-seqs=12 轻松维持**（用户指示 4）。
- **架构决策**：rank=环序（01=0→02=1→04=2→03=3，全 1 跳）；**控制面走管理网**（GLOO/NCCL_SOCKET=enP7s7，MASTER_ADDR=192.168.5.x），数据面走 RoCE；进程绑定延续 01/02 经验（EngineCore 0-17 + NCCL 线程隔离核 18-19 + IRQ 5-9）。
- **操作红线 12 条已梳理**（restart no / head-first / GPU-gate / iptables 同步放行 / rules.v4 勿固化 docker 链 / 自检用对外 IP 等），全部有来源文件。
- ⚠️ 待确认：维护窗口执行时机（卸载 TP2 为生产变更，会中断推理）；NCCL_IB_TOS=46 应用侧确认。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 可进入维护窗口执行 |
| 前置加固 | QoS 持久化 ✅ / iptables 复核 ✅ / rules.v4 工具 ✅ |
| 内存结论 | TP4 权重 38.9GiB/机，KV 余量 3×，max-num-seqs=12 ✓ |
| 操作红线 | 12 条（Docu，含来源） |
| 回滚 | 双保险（TP2 脚本/镜像/权重未动 + 备份） |
| 待确认 | 维护窗口时间；NCCL_IB_TOS=46 |

---

## 🛡️ 操作红线 12 条（Docu，来源：runbook/SRE审查/ring-close/Grafana 事故）

1. head/worker 容器**必须 --restart no**（embed 用 unless-stopped 是有意设计勿动）
2. **head-first**：head(rank0) 先启、:25000 就绪后 worker 再 join；停机反向（worker 先 rm → head 后停）
3. GPU-gate ≤180s + 健康门禁（NV_ERR 扫描、显存≥32GiB，各≤300s；受限子项跳过不阻塞）
4. 编排含双脚本前置自检（check_vllm_script.sh）、8001 释放验证、对端门禁 120s
5. **管理网 192.168.5.x 禁改**；新 RoCE 网段配 IP 必须同步放行 iptables（白名单模式）
6. `iptables-save` 前先 `iptables-restore --test`；**rules.v4 不得固化 docker 动态 DOCKER 链**；改前备份
7. 自检/健康检查**必须用对外 IP**（127.0.0.1 是 docker-proxy 假象）
8. 保持：max-model-len 768000 / max-num-seqs 12 / VLLM_USE_BREAKABLE_CUDAGRAPH=1 / LD_PRELOAD=/opt/libncclpin.so
9. 脚本路径 `<INSTALL_DIR>/scripts/`（连字符）；hosts 勿覆盖；GID 改 IP 后须重启防空洞
10. 驱动保持 580.173.02 勿升（580.178.04 需 DKMS 不可装）；CUDA13.2 暂缓
11. 改启动脚本后必须 `check_vllm_script.sh + bash -n` 通过
12. 直连无交换机勿配 PFC（QoS 用 DSCP trust）；每 QSFP=2 逻辑接口全配 IP；MTU 全 9000

## 🔧 前置加固结果（Rex，已完成）

| 项 | 结果 |
|---|---|
| QoS 复测 | 发现 01(<RING_SUBNET>)/03(3口) 漏配 pcp → **已全部修复 4 口 dscp**；NCCL 映射：DSCP46→prio5→tc5 高优先+PFC P5 无损队列（**应用侧需 NCCL_IB_TOS=46**） |
| QoS 持久化 | systemd mlnx-qos.service + mlnx-qos-setup.sh（03 新建 enable+start；01 脚本补 4 口）；模拟验证：reset pcp → restart service → 恢复 dscp ✓（shebang 已修） |
| iptables 复核 | 四台全 ✅：peer 白名单全覆盖（01 含 <RING_SUBNET>/<RING_SUBNET>、03 含 <RING_SUBNET>/<RING_SUBNET> 等）；管理网默认 ACCEPT；白名单=RoCE 口 TCP 定向 DROP（UDP 数据面放行，不影响 NCCL）；rules.v4 与生效规则 diff 空 |
| rules.v4 拆分 | 根因确认（netfilter-persistent 先恢复旧快照+docker 后启 IP 变）；**iptables-save-custom.sh 已部署四台，dry-run 通过**；净化 rules.v4 属维护窗口动作（备份后执行，失败 cp 回滚） |

## 🚀 TP4 部署 playbook（Archi，维护窗口执行）

### 阶段 0：准备
1. 03/04 补 isolcpus=18-19（grub.d/90-isolcpus.cfg + update-grub）
2. 01/02 docker restart（runtime 生效）——⚠️ **重启后必须复查对外端口**（Grafana DNAT 教训：docker 重启 IP 漂移）
3. **rules.v4 净化**（iptables-save-custom.sh，先备份 rules.v4，失败回滚）
4. 四机顺序重启（01→02→03→04，embed 双活兜底）；重启后复核 isolcpus/nproc=18/MTU9000/GID/对外端口
5. 03/04 同步环境：rsync <INSTALL_DIR>/{envs,lib,scripts} + vllm-cache/tilelang-cache + 权重软链检查

### 阶段 1：卸载 TP2（01/02）
```
systemctl stop vllm-cluster.service            # 01
docker inspect vllm-envE-node > <INSTALL_DIR>/backup/tp2-node.json
docker rm -f vllm-envE-node                    # 01
ssh node01 "docker inspect vllm-envE-worker > backup/tp2-worker.json; docker rm -f vllm-envE-worker"
ss -tln | grep -E ':8001|:25000' → 空；pgrep -f VLLM::EngineCore → 空
```
镜像/日志/权重不动 = 回滚锚点；litellm/8003 窗口内 suspend LLM deployment（上游 8001 502）

### 阶段 2：部署 TP4
- 新脚本 <INSTALL_DIR>/scripts/start_tp4_{head,worker,cluster}.sh；容器 vllm-tp4-rank0~3
- rank：01=0、02=1、04=2、03=3（环序）；`--tensor-parallel-size 4 --nnodes 4`；`--restart no`
- head-first：01 起 → 轮询 :25000 → 02 → 04 → 03；worker 存活校验+对端门禁(120s)+快速失败；GPU-gate/健康增强沿用

### 阶段 3：验证
1. NCCL world4 all_reduce（测试容器 `--device /dev/infiniband --ulimit memlock=-1`）
2. 8001 /health=200、/v1/models；chat/tool-call 冒烟
3. bench 对比 TP2：prefill_tps≥1.25×、TTFT p95≤1.1×、preemption=0
4. **Grafana 面板核对**（用户任务 7）：4 台 GPU/RoCE/服务数据正常（数据面调整后连接修复确认）

### 阶段 4：收尾持久化
- 新网段/端口变更 → iptables 同步放行（wrapper 重存）+ QoS setup.sh 覆盖 → mlnx-qos.service restart
- 确认修复可靠后**及时持久化**（rules.v4 / netplan / 脚本归档 <INSTALL_DIR>/backup/tp4-<date>/）
- 回填 Runbook

## 📐 进程绑定矩阵（v2，2026-08-11 用户拍板 + Rex 核实）

**隔离条件（四台已生效）**：`isolcpus=0-4 rcu_nocbs=0-4`（grub.d/90-isolcpus.cfg，2026-08-10 用户批准；项目记录"18-19"已过时）。核拓扑：**0-4=Cortex-A725 低功耗（2808MHz）**、**5-9=Cortex-X925 高性能（3900MHz）**、10-14=A725、15-19=X925（双簇 L3）。

| 线程类 | CPU | 手段 |
|---|---|---|
| NCCL 线程/延迟敏感 | 隔离核 **1-4**（A725；避开 CPU0 boot 核，nohz_full 未配） | LD_PRELOAD=/opt/libncclpin.so + taskset |
| EngineCore/主进程 | **5-9**（X925） | taskset -c 5-9 |
| 控制面/数据面（GLOO/TCP） | 5-9（轻量，与引擎同区） | taskset |
| mlx5 RoCE IRQ | **5-9**（用户拍板定案；与 EngineCore 同区，硬中断抢占风险已接受，落地时 EngineCore taskset 与 IRQ smp_affinity 明确分区顺序） | /proc/irq smp_affinity |
| litellm/embed/监控 | 10-19（默认调度） | 默认 |

⚠️ nproc=16 已查明：= **CPU0 + CPU5-19**（isolcpus=0-4 但 CPU0 为 boot 核无法隔离，实际隔离 `cpuset.cpus.isolated=1-4`），**非故障**；无 cgroup 限制，taskset 绑 1-4/5-9/15-19 全部实测可行。**容器需显式 `--cpuset-cpus=1-19`**（避免 docker 默认亲和 0,5-19 丢失隔离语义），进程层再 taskset 分绑（1-4 NCCL / 5-9 EngineCore / 15-19 备用），CPU0 留给系统。

## 🔗 NCCL 配置要点

- `NCCL_ALGO=RING`、`NCCL_MIN_NCHANNELS=2`（双链路）
- `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`（全 twin 4 口）
- **控制面**：`NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=enP7s7`、MASTER_ADDR/VLLM_HOST_IP=<NODE_IP>~<MGMT_OCTET>（非邻机 QSFP 无路由）
- GID：四机 show_gids 逐机核对；一致则 NCCL_IB_GID_INDEX=2，不一致留空让 2.30 自动选
- `NCCL_IB_TOS=46`（QoS 映射已就绪，应用侧确认）、`NCCL_IB_TIMEOUT=1000/RETRY_CNT=7`、MTU 9000
- LD_LIBRARY_PATH 前插 NCCL 2.30.7

## ⚠️ 风险与回滚

- **风险**：GID 双口 index 不一致 → comm init 失败（前置核对）；控制面误走 QSFP → 非邻机握手挂（管理网规避）；03/04 embed 共存内存紧 → util 0.75；4 机直连非官方拓扑 → 手动验证
- **回滚**：TP4 容器 stop/rm → 跑原 start_v026r_cluster.sh 恢复 TP2（脚本/镜像/权重未动）→ 03/04 恢复 embed 单卡；全部配置先入 <INSTALL_DIR>/backup/tp4-<date>/

## ✅ 行动清单（维护窗口）

| # | 行动 | 负责 | 紧急度 | 验收 |
|---|------|------|--------|------|
| 1 | 03/04 isolcpus 补齐 + update-grub | SRE | P0 | 重启后 nproc=18 |
| 2 | rules.v4 净化（备份后 iptables-save-custom.sh，失败回滚） | SRE | P0 | dry-run 通过、docker 链残留 0 |
| 3 | 01/02 docker restart + **对外端口复查**（3000/8191/8001） | SRE | P0 | 外部可达，无 DNAT 错位 |
| 4 | 四机顺序重启 + 复核（isolcpus/MTU/GID/端口） | SRE | P0 | 全通过 |
| 5 | 卸载 TP2（worker→head，backup json） | SRE | P0 | 8001/25000 释放 |
| 6 | TP4 部署（start_tp4_*，head-first + 绑定矩阵） | SRE | P0 | 4 rank 全 ready |
| 7 | NCCL world4 all_reduce + 冒烟 + bench 对比 | SRE | P0 | 达标 |
| 8 | Grafana 4 台面板核对 + 持久化收尾 + 回填 Runbook | SRE + Docu | P1 | 面板数据完整 |

## ⚠️ 待确认

1. **维护窗口执行时间**（卸载 TP2 中断推理，需用户指定窗口）
2. NCCL_IB_TOS=46 应用侧确认（或部署时统一注入）

---

## 📚 数据来源 & 成员产出索引

- Docu：teammate-message tech-writer @ 2026-08-11（12 红线/坑/回滚/TOP10/执行顺序）
- Rex：teammate-message sre-engineer @ 2026-08-11（QoS 修复表、iptables 复核表、rules.v4 拆分方案+工具）
- Archi：teammate-message architect @ 2026-08-11（TP4 playbook、绑定矩阵、内存账、NCCL 要点）

> 本报告由工程保障团队 AI 协作生成（2026-08-11），关键决策请由人类工程负责人复核签字。
