# SRE 可运维性与可靠性评估 — TP4 集群 + NVFP4 落地

> **SRE 工程师**：雷克斯（Rex） | **日期**：2026-08-20
> **范围**：DGX Spark GB10 TP4 四机集群（01-04 / sm_121a / vLLM 0.26）NVFP4 落地，支撑「交接文档落实」
> **审查方式**：只读研究 + 运行态风险评估。输入依据 = 阿奇架构审查（architecture-nvfp4-2026-08-20.md）、TP4 runbook（v1.5 / append / 分布 / newnode / grafana）、集群自检基线（audit-08-07 / audit-doc-vs-server-08-13）、08-19 SEV1 事故日志、08-20 NVFP4 落地交接上下文、长期记忆。
> **关键口径（勿推翻）**：4 节点环网 10.20.0.x + RoCE 136-139；TP4 08-11 上线；生产容器 `vllm-tp4-rank0~3`，镜像 `dspark-vllm-gx10:0.2.1-v026.0`；`/vllm-workspace` 容器内部目录（重建丢失）；宿主机 `<INSTALL_DIR>`（scripts/lib/models/envs 挂载进容器）；NVFP4 路线 A 为主路径。

---

## 一、可运维性风险清单（按严重度）

**总体评级**：🟡 有条件通过——方案已齐（持久化落位 <INSTALL_DIR> + 回退三防线 + DRSOP），但**生产恢复决策未定、自愈机制在 SSD 事故后被 disable、监控告警覆盖有盲区**，这三项是接手团队需立即关切的。

### 🔴 严重（P0，接手后优先处理）

| # | 风险 | 现状与证据 | 影响 | 缓解 / 处置 |
|---|---|---|---|---|
| R1 | **生产 4 rank 未恢复（GPU 0%）** | 08-19 起按用户要求保持未恢复；4 rank healthy 但 GPU 0%。恢复决策归 P2 | 集群当前**不承载推理**，任何"已上线"表述都只代表容器拉起，不代表可用 | P2 收尾统一决策恢复；恢复须走 `start_tp4_cluster.sh` head-first + GPU-gate + 回归三件套，勿直接 `docker run` 单机拉起 |
| R2 | **自愈机制处于 disable 状态** | 08-19 SSD 事故期间 monitor 被 disable（`vllm-tp4-head.service` + `vllm-healthcheck.timer`）；且 `vllm-healthcheck.timer is-enabled` 与 disable 记录不符，需显式 mask | 集群失去故障自愈、失去健康探活 → 宕机无人感知（08-07 已现 Exited 13h 无人发现的先例） | 恢复生产前**必须**核对并恢复 head monitor + healthcheck timer（`systemctl is-enabled/masked` 逐一确认）；建议补内存<2G 告警 |
| R3 | **`/vllm-workspace/` 非持久 = 重建丢失** | 生产产物若仅存容器内 `nvfp4-landing/` 同步副本，重建即丢 | NVFP4 落地产物丢失、回退失效、恢复时间拉长（见 RTO） | **P0**：routeA 适配层 + v17 落宿主机 `<INSTALL_DIR>/`（scripts/nvfp4 + kernel2/v17），软链进 site-packages，`import` 透明；容器重建后自动可见 |

### 🟠 高（P1）

| # | 风险 | 现状与证据 | 影响 | 缓解 |
|---|---|---|---|---|
| R4 | **UMA 内存耗尽复发（08-19 根因仍在环境）** | 根因 = UMA 内存耗尽（03 NVRM NV_ERR_NO_MEMORY→avail 0→NCCL 300s 超时→oom-killer→03 冻结 50min）。0.7 验证有效，但 SSD 回滚后 mem util **回到 0.80** | conc3×长上下文并发下可能再次触发 NCCL 超时 + 节点冻死 | 恢复 0.70 或保留 0.80 + 硬约束：03/04 持续负载头寸仅 ~2.5G → **内存 avail<2G 告警 + 后手 0.65/降 max-num-seqs**；benchmark 分段恢复，事故格（65536/coding/conc3）单独验证 |
| R5 | **监控告警覆盖盲区** | Grafana 数据源 01→`http://<NODE_IP>:8191`（02 Prom，非 9090）、scrape 5s、面板按 node 分组。但：job 名/标签仍用旧节点末段；vllm 抓取目标含 worker（无 API 端口）；节点卡死时 Prometheus 本身中断（03 冻结时采集中断） | 节点冻死、内存告警、进程守护无可靠信号 → 事件发现延迟 | 清理 Prometheus 旧命名与失效抓取目标；**为 avail 内存、KUBE/UMA、NCCL 超时日志、GPU 0% 添加显式告警规则**；node_exporter 卡死视为节点级 SEV 信号 |
| R6 | **配置漂移历史复发**（rank 映射颠倒、MTU/shm/capture-sizes/LD_PRELOAD 多处文档失准） | 08-13 双向审核确认运行态与文档系统性漂移 10+ 处，rank 曾颠倒（照文档部署会 TP4 无法组网） | 交接团队照文档误部署 → 组网失败、参数错位 | 交接文档必须以**运行态实测**为准（§部署检查清单的 Go 即为运行态核验）；关键锚点：NCCL MD5 `b7784b49`（v3 双口，旧文档仍写 `4cc43e3b`）、shim v8 md5 `ce43c688` |
| R7 | **监控/自愈曾因停机窗口误伤生产** | monitor 在 SSD 停机窗口自动拉起 rank0，需先 stop timer+service 才能安全停机 | 维护窗口若未先停 monitor 会意外拉起服务 | 停机 SOP 固化：**先 systemctl stop timer+service → 再容器停机 → 完成后恢复**；此教训写入事故预案 |

### 🟡 中（P2）

| # | 风险 | 缓解 |
|---|---|---|
| R8 | 双 Grafana（01/02 各一个 :3000）未收敛 | 按 r12 建议保 02 权威，删去重；更新监控文档口径 |
| R9 | `.local-backup` 删除决策（03/04 各 156G 兜底） | **暂缓删除**至 NFS 连续稳定≥7 天或补 HA；当前 03/04 走本地 serving，NFS 集中化恢复中 |
| R10 | 安全暴露面：sudo 密码明文于 file-registry.md、API key 硬编码脚本（755）、litellm master_key 明文+5 .bak、ssh 全默认密码认证 | P0 已列安全轮换；交接团队须在接手窗口内完成轮换 + 脚本收敛 700/750 + sshd 显式收敛 |
| R11 | 01 时区漂移（+0800 vs 他机 UTC） | 统一 UTC 后再做日志/监控时间线对齐 |
| R12 | SSD KV 卸载模块（kvssd/zstd/io 补丁）对**已知不可行**的结论保留 | 回滚已做（util 0.80、无 kv-transfer）、fstab 行注释留档；**不得重新启用**除非新立项；校验 check_vllm_script.sh 为 0.80 版（md5 472c58bb） |

### 🟢 良好与信任锚点（已在位，保持）
- 补丁 MD5 一致（libnccl.so.2.30.7=`b7784b49`、libncclpin v8=`ce43c688`）四机一致；/opt/nccl-ringonly 归档、rollback-anchors 在
- 隔离核 isolcpus=8-9（专 NCCL 数据面）、EngineCore 15-19、shim v8 mark-then-pin 竞态已修复（runbook v1.5 D6）
- 网络：环网 4 段 IP 全配、MTU 9000、iptables 白名单与 rules.v4 一致、QoS mlnx-qos 持久化（DSCP46→P5）、FEC P1 解除
- 回退三防线（kernel①→v15、kernel②→v11、全量重挂载）已设计齐备（见阿奇 §4.2）
- 服务健康基线：TP4 容器 Up/RestartCount=0/systemctl --failed=0、环邻 ping 0 丢包

---

## 二、部署检查清单（Go / No-Go）

> 用途：新团队接手 TP4 集群 + NVFP4 落地的**投产前门禁**。任一 **No-Go** 项未满足即不投产，先解决再进下一步。顺序 = 环境 → 镜像 → 持久化 → GPU 门禁 → 网络 → 回归。

### 阶段 A：基础环境核验（所有 No-Go 均阻断）

| # | 检查项 | 命令 / 判据 | Go（通过） | 备注 |
|---|---|---|---|---|
| A1 | 4 机在线、开机顺序 | `ping <NODE_IP>~<NODE_IP>` | 4 机全通 | head-first 铁律 |
| A2 | 时区一致 | `timedatectl` 四机 | UTC | 01 需先修漂移 |
| A3 | 隔离核生效 | `cat /proc/cmdline` 含 `isolcpus=8-9` | 四机一致 | 数据面铁律 |
| A4 | 内存头寸 | `free -g`，01/02/03/04 | 生产前 03/04 avail≥4G 余量 | 03/04 仅 ~2.5G 头寸时降配 |
| A5 | 磁盘水位 | `df -h /` | <70%，无满盘 | kvssd 已清 191G→20K |
| A6 | 镜像二进制锚点 | `md5sum` | NCCL=`b7784b49`、shim v8=`ce43c688` | 防漂移/防错版 |

### 阶段 B：镜像与容器

| # | 检查项 | 判据 | Go |
|---|---|---|---|
| B1 | 生产镜像在位 | `docker images \| grep dspark-vllm-gx10` 含 `0.2.1-v026.0` | 四机一致 |
| B2 | 容器不存在残留 | `docker ps -a \| grep vllm-tp4` | 无 Exited 残留（或已知 cleanup 计划） |
| B3 | 补丁 banner | `docker logs vllm-tp4-rank0 --tail 50 \| grep "NCCL version"` | `2.30.7+cuda13.0`（**出现 13.3 = LD_PRELOAD 失效，No-Go**） |

### 阶段 C：持久化（NVFP4 P0 门禁）

| # | 检查项 | 判据 | Go |
|---|---|---|---|
| C1 | routeA 适配层落宿主机 | `<INSTALL_DIR>/scripts/nvfp4/nvfp4_4w4a_mmaf.py` 存在 | 是 |
| C2 | v17 内核落宿主机 | `<INSTALL_DIR>/kernel2/v17/` 存在 + md5 校验 | 四机一致 |
| C3 | 容器内 import 透明 | 容器重建后 `python3 -c "import nvfp4_4w4a_mmaf"` | OK，且跑通 preprocess_weights+__call__ |
| C4 | **唯一权威源纪律** | `grep -r "vllm-workspace" <INSTALL_DIR>/scripts/` 无生产依赖 `/vllm-workspace` | 无（产物不落容器内目录） |

### 阶段 D：GPU 门禁

| # | 检查项 | 判据 | Go |
|---|---|---|---|
| D1 | GPU 可用 | 容器内 `nvidia-smi` | 4 rank 各识别 1 GPU |
| D2 | NVIDIA runtime | `docker info \| grep nvidia` | runtime 在位 |
| D3 | NCCL ring 组网 | 4 rank 全连 TCPStore:25999、rank 汇总正确 | rank0~3 全 Up |
| D4 | SASS 门禁（NVFP4） | `grep -iE "mma.*e2m1\|mmaf"` 产物 | 出现（**勿用 tcgen05**，SM10x 指令） |

### 阶段 E：网络

| # | 检查项 | 判据 | Go |
|---|---|---|---|
| E1 | 环网 4 段 IP | `ip -4 addr` <RING_SUBNET> + 10.20.0.x + <RING_SUBNET> + <RING_SUBNET> | 全配 |
| E2 | MTU | 环邻口 | 9000 |
| E3 | iptables 白名单 | 新 RoCE 网段已放行 + rules.v4 持久化 | 放行对应 peer 侧 |
| E4 | QoS | `mlnx-qos` active | 是，DSCP trust |
| E5 | 环邻连通 | 环邻 `ping` | 0 丢包；iperf3 抽查段带宽 |

### 阶段 F：回归验证（最终 Go）

| # | 检查项 | 判据 | Go |
|---|---|---|---|
| F1 | 服务 health | `curl -H "Authorization: Bearer <key>" http&#58;//127.0.0.1:8001/v1/models` | 200（**需带 key，已启用鉴权**） |
| F2 | chat 冒烟 | 容器内 `2+2=?` | 返回"4" |
| F3 | benchmark 正确性 | 8/8 rel<0.02 vs 官方 `dequantize_to_dtype` | PASS |
| F4 | benchmark 性能 | kernel① ≥200 TFLOPS（当前峰值 187，P1 冲刺）；kernel② v17 大 T ≥120 GB/s | 达门槛或记录偏差 |
| F5 | 对照 v15 | kernel① ≥1.5× v15 | PASS |
| F6 | 恢复自愈核验 | `systemctl is-active` head monitor + `is-enabled` healthcheck timer | 自愈在位（非 disable） |

> **综合判定**：阶段 A-E 全 Go 且 F1-F2 Go → 可投产；F3-F6 为 NVFP4 专项深化，若 F4 未达 200 门槛需**单独立项决策**（阿奇 P1），不构成基础投产阻断，但须明确记录偏差。

---

## 三、事故响应预案（SEV 分级 + 响应）

### 3.1 严重度分级

| 等级 | 标准 | 响应时限 | 升级 |
|---|---|---|---|
| **SEV1** | 全集群推理不可用 / 多节点宕机 / 数据权威源损坏 | 立即，全员 | 主理人 + SRE + 全团队 |
| **SEV2** | 单节点卡死 / 单 rank 掉线且自愈未生效 / 主要功能降级 | 15 分钟 | 主理人 + SRE |
| **SEV3** | 单点异常但服务可用 / 监控盲区 / 配置漂移 | 1 小时 | SRE |
| **SEV4** | 低影响（面板读值误判、日志缺失等） | 下个工作日 | 记录 + P2 |

### 3.2 SEV 处置 SOP（按序）

**S0 分诊（即时）**
- 判定等级；识别受影响 rank/sx 与影响面（是否全链）。
- 角色分配：事故指挥官（主理人）、SRE 修复通道、测试做回归、文档师留档。

**S1 止血（优先于根因）**
- 通用止血三板斧：
  1. **先确认监控是否在**（<NODE_IP>:8191/Prom）——若 Prometheus 本身中断 = 节点级冻死信号（08-19 03 即此），高度疑似 UMA 耗尽。
  2. 节点冻死：`free -g` 看 avail，若 <2G → 按 SEV1 处理（清下载任务 / 降 mem util 至 0.65 / 降 max-num-seqs）→ 击杀高内存进程 → 等待 oom-killer 或手动 kill 冻结进程，**删容器后自愈**（08-19 经验）。
  3. NCCL 超时：先查 `docker logs vllm-tp4-rank0 --tail | grep -i "NCCL version"` 是否 `13.3`（LD_PRELOAD 失效 → 立即修挂载）→ 再查环网/PEER_HCA/隔离核 → NCCL 超时多为**被动受害者**，先查内存再查网。

**S2 状态沟通（要点模板）**
```
【SEV-X 状态】受影响：<rank/机>；影响面：<全链可用/降级>
- 我们知道的：<事实>
- 我们已做的：<止血动作+时间戳>
- 下一步：<SOP + 责任人>
- 用户影响：<有无/预计恢复>
```
- 更新节奏：SEV1 每 15min、SEV2 每 30min 内部同步；事实驱动，不猜根因。

**S3 缓解**
- 记录时间线（发现/止血/恢复三时刻）；跟踪每一步。
- 恢复走 `start_tp4_cluster.sh` head-first（rank0→:25999→02→04→03），**禁止单机 docker run 拉起**。
- 停机窗口铁律：**先 stop monitor timer+service → 再停容器 → 完成 → 恢复 monitor**（防自愈误拉）。

**S4 复盘（无责）**
- 模板：时间线（含事件触发点） → 5-Why 根因 → 行动项（owner+deadline）→ 是否沉淀进 runbook。
- 铁律：聚焦系统与流程，不追责个人；复盘产出必回填 runbook（本次 08-19 SEV1 的教训即应固化）。

### 3.3 已知高发场景 × 处置速查

| 症状 | 可能根因（按序查） | 处置 |
|---|---|---|
| NCCL 300s 超时 | ①UMA 内存耗尽（最可能）②LD_PRELOAD 失效(banner 13.3)③环网/PEER_HCA 漂移 | 查 avail→修挂载→查 banner→查网络→回归 |
| 节点冻死/sshd 无响应 | UMA 耗尽 → oom-killer | kill 冻结进程或删容器自愈；降 util |
| 8001 不监听/卡 init | 权重加载中（117-135s）或 TCPStore 未就绪 | `ss -ltnp\|grep 25999` + NCCL_DEBUG；head-first 重来 |
| healthy 但 API 401/403 | 已启用 `--api-key`，未带 Authorization 头 | 带 `Authorization: Bearer <internal-key>` |
| GPU 0% 全 rank | 生产未恢复（当前常态，非故障）| 按用户决策恢复 |
| SSD kvssd 相关 | 已知不可行已回滚 | 不得重新启用；校验 util=0.80 check 版 |

---

## 四、历史事故教训回顾

### 4.1 2026-08-19 SEV1（NCCL 超时 + 03 卡死）

**根因（已 RCA 闭环）**：主因 = **UMA 内存耗尽**，NCCL 300s 超时是被动受害者。证据链：03 于 03:18 NVRM NV_ERR_NO_MEMORY → 03:20 avail=0 → 03:23 NCCL 超时 → 03:54 oom-killer 杀 Worker_TP(279G)+EngineCore → 03 冻结 ~50min。SSD 满盘 + 卸载 CPU 竞争为次要放大器。

**行动项是否已沉淀？**：⚠️ **部分沉淀，尚有缺口**
- ✅ 已沉淀：0.7 验证有效（conc3×65536 连续 4 次过，内存谷底未归零）；SSD 卸载方案判定不可行并回滚到 util 0.80；回滚检查脚本修复（md5 472c58bb 留档）。
- ⚠️ **未彻底闭环（复发风险）**：
  1. 生产当前 util 回到 **0.80**（0.7 虽验证有效但被 SSD 回滚覆盖）——**复发场景仍存在**，需恢复 0.70 或硬性内存告警 + 降配后手。
  2. **03/04 持续负载头寸仅 ~2.5G**，avail<2G 告警未见落地。
  3. **monitor/healthcheck 自愈处于 disable**，需恢复以保"宕机有人发现"。
  4. `vllm-healthcheck.timer is-enabled` 与 disable 记录矛盾，需显式 mask 定格。
  5. benchmark 事故格（65536/coding/conc3）尚未按 Tessa 计划单列复验。

**复发风险评估**：**中-高**。根因（UMA 耗尽）× 复现组合（conc3 长上下文并发）均未根除，仅靠"不跑 crash 组合"回避。缓解 = 恢复 0.70 / 内存告警 / 降 max-num-seqs 后手 / benchmark 分段恢复并在事故格单独冒烟。

### 4.2 其他历史事故要览（已闭环，交接参考）
- **08-11 Grafana 外部不可达**：容器 IP 漂移 + iptables DOCKER 链错位 + 自检用 127.0.0.1 绕过 docker-proxy 造假象 → 教训：自检须用对外 IP、rules.v4 不得固化 docker 动态 DOCKER 链。✅ 已闭环。
- **08-11 iptables 白名单阻断新链路**：白名单只放行旧邻段，新 RoCE 段全 DROP → 教训固化红线 5"新网段配 IP 必同步放行"。✅ 已闭环。
- **08-11 环网 FEC / 08-06 NCCL init hang / 08-02 GPU 指标**：均已闭环并回填 runbook。
- **共同模式**：本集群历史事故大部分为**配置漂移 + 监控盲区 + 内存头寸**三者叠加；交接团队应优先补齐这三个维度的护栏。

---

## 五、RTO / RPO 评估

### 5.1 关键事实
- 容器 `--restart no`（无自动重启策略，靠 systemd monitor 自愈）。
- `/vllm-workspace` 容器内部（重建丢）；`<INSTALL_DIR>/{scripts,lib,models,envs}` 持久挂载。
- 权重加载 5-8 分钟（NFS 117-135s 实测 + 模型初始化）。
- 生产 4 rank 当前 GPU 0%（未恢复，按用户要求）。

### 5.2 场景 × RTO / RPO

| 场景 | RPO | RTO | 说明 |
|---|---|---|---|
| **单容器崩溃（无自愈）** | 0（无状态业务丢失） | **25-40 min**（手动拉起 head-first + 权重加载 5-8min + 4 rank 全连） | 若 monitor 在位可降为自动，RTO≈权重加载 5-8min |
| **单节点冻死（UMA 耗尽）** | 0 | **30-60 min**（杀冻结 → 恢复自愈 → head-first 重建） | 03/04 低头寸节点风险最高 |
| **容器重建（NVFP4 产物丢失）** | **高（RPO 失效）**：若产物仅存 /vllm-workspace → 全部丢失 | **+15-30 min**（重建后重分发 routeA/v17 + import 核验 + 回归） | **P0 落 <INSTALL_DIR> 后 RPO=0、RTO 回落至基线** |
| **全量重挂载恢复（DRSOP）** | 0（源在宿主机） | **40-60 min**（重建镜像容器 + 重挂 4 子目录 + DRSOP 三步回归） | 阿奇 §4.3 路径 |
| **回退 kernel①→v15** | 0 | **0（删除引用即回退，不改 vLLM 本体）** | 最低代价回退 |
| **回退 kernel②→v11** | 0 | 0（调用点换回） | 文件留存 |
| **回退到旧镜像/TP2** | 0 | **≤15 min（容器级）** | runbook append §A.6 回滚 TP2；镜像 0.2.0/0.2.1 双 tag 在位 |

### 5.3 结论与建议
- **当前 RPO 关键风险**：只要生产产物仍依赖 `/vllm-workspace`，RPO=∞（重建即丢）。**P0 落位 <INSTALL_DIR> 后 RPO=0**，这是本评估最重要的单一动作。
- **当前 RTO 基线**：无自愈主动恢复 ≈25-40min；有 monitor 自愈 ≈5-8min（仅权重加载）。
- **改善项**：
  1. 恢复 monitor/healthcheck 自愈 → RTO 从 25-40min 降到 ~5-8min（SEV 期间最有效的单一投资）。
  2. 把 `<INSTALL_DIR>` 的 routeA/v17 也纳入 `docker run` 命令（或 DRSOP）的挂载清单，确保重建即完整。
  3. 为 03/04 低内存节点加 avail 告警 ≤2G，RTO 前置 → 避免进入冻死的大 RTO 场景。

---

## 附一：交接文档应纳入的本评估关键项（供 tech-writer）
1. R2 自愈恢复（monitor+healthcheck 从 disable 恢复）+ 停机窗口先停 monitor 铁律。
2. R4 UMA 内存告警 + 0.70 恢复决策 + 03/04 低头寸后手。
3. 部署检查清单 A-F 段作为投产门禁（含 NCCL MD5 `b7784b49`、带 key 的 health 校验）。
4. DRSOP + 回退三防线 + RTO/RPO 表。
5. SEV 分级 + SOP + 状态沟通模板 + 复盘模板。

---

*本报告由工程保障团队 SRE 雷克斯基于只读研究 + 历史文档生成，未修改任何生产配置/脚本。恢复决策、0.70/0.80 取舍、事故格复验请由人类工程负责人复核签字。*