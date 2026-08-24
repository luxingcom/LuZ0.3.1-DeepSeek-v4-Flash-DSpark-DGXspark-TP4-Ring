# Runbook 待回填内容：TP4 攻坚成果持久化（2026-08-12）

> **回填说明**：目标 runbook `F:\AICADqwen\runbook-dspark-vllm-2026-08-08.md` 不存在（F 盘可访问但无此文件），回退版本 `runbook-dspark-vllm-2026-08-06.md` 亦不在 F 盘。基准内容取自 `deliverables/engineering-assurance/runbook-dspark-vllm-2026-08-06.md`。以下为待回填至正式 Runbook 的 TP4 增量章节，由工程保障团队技术文档师 Docu 于 2026-08-12 整理，依据 8/11 TP4 部署报告、start_tp4_* 脚本实测与 8/11 事故报告。**请在正式 Runbook 恢复访问后合并本节并删除本文件。**

---

## §A. TP4 四机环网部署章节（新）

### A.1 拓扑与 rank/IP 映射（环序）
纯环网直连（每机 2 OSFP 口、无交换机），4×DGX Spark（GB10 sm_121）TP=4。

| rank | 主机 | 管理网 | 环邻（RoCE） | 角色 |
|---|---|---|---|---|
| 0 | node01 | <NODE_IP> | 02(f1)、03(f0) | head / TCPStore |
| 1 | node01 | <NODE_IP> | 01(f1)、04(f0) | worker |
| 2 | node01 | <NODE_IP> | 02(f0)、03(f1) | worker |
| 3 | node01 | <NODE_IP> | 04(f1)、01(f0) | worker |

- 控制面走管理网：MASTER_ADDR=<NODE_IP>、**MASTER_PORT=25999**（TP4 专用，非 TP2 的 25000）
- RoCE 环网 IP：01↔02 = 10.100.136/137；02↔04 = 10.20.0.x；04↔03 = 10.100.138/139；03↔01 = **10.100.140/141**（8/11 补闭环）；MTU 全 9000
- 容器：`vllm-tp4-rank0~3`，`--restart no`，`--tensor-parallel-size 4 --nnodes 4`，host 网络；镜像 `anemll/dspark-vllm-gx10:0.2.1-v026.0`

### A.2 补丁方案（ring-only NCCL，社区 GLM-5.2 路线）
- **背景**：stock NCCL 对全 rank 对建 IB QP + 按 HCA index 统一配对，纯环网非直连必 `ibv_modify_qp 110`；无交换机 NCCL 无法感知 fabric 拓扑。
- **v1（src/transport.cc, ncclTransportP2pConnect）**：跨机非环邻对跳过（`peer != ringPrev && peer != ringNext`）→ 根治 PAT 为 distance-2 对建连死结。
- **v2（src/transport/net.cc）**：`ncclIbPeerHcaOverride()` 读 `NCCL_IB_PEER_HCA="peerRank=devName;..."`，sendSetup/recvSetup 强制环邻对用物理对口。
- **构建**：容器内 `--user root --entrypoint bash` + `make -j src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`；产物 `/opt/nccl-ringonly/libnccl.so.2.30.7`（四机 MD5=`4cc43e3b25ddf275701c11b3d566b686`，符号链接 libnccl.so→.2→.2.30.7 完整）。
- **版本区分**：banner `2.30.7+cuda13.0`（补丁版）vs `2.30.7+cuda13.3`（pip 原版）。
- **加载**：`-v /opt/nccl-ringonly:/opt/nccl-ringonly:ro -e LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`
- **归档**：`<INSTALL_DIR>/backup/tp4-20260812/`（src 源码+diff+artifacts+scripts+logs+README）

### A.3 env 全集（start_tp4_head.sh 实测）
`NCCL_ALGO=RING`、`NCCL_CROSS_NIC=1`、`NCCL_IB_SUBNET_AWARE_ROUTING=1`、`NCCL_NET_PLUGIN=none`、`NCCL_IB_MERGE_NICS=0`、`NCCL_MIN_NCHANNELS=2`、`NCCL_NET=IB`、`NCCL_IB_GID_INDEX=2`、`NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`、`NCCL_IB_TIMEOUT=1000`、`NCCL_IB_RETRY_CNT=7`、`NCCL_IB_TOS=46`（QoS DSCP trust 已配）、`NCCL_SOCKET_IFNAME=enP7s7`、`NCCL_IGNORE_CPU_AFFINITY=1`、`NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log`

### A.4 PEER_HCA 表（NCCL_IB_PEER_HCA，per-rank 已写入脚本）
| rank(机) | NCCL_IB_PEER_HCA | 语义 |
|---|---|---|
| 0(01) | `1=rocep1s0f1;3=rocep1s0f0` | →02 f1、→03 f0 |
| 1(02) | `0=rocep1s0f1;2=rocep1s0f0` | →01 f1、→04 f0 |
| 2(04) | `1=rocep1s0f0;3=rocep1s0f1` | →02 f0、→03 f1 |
| 3(03) | `0=rocep1s0f0;2=rocep1s0f1` | →01 f0、→04 f1 |

### A.5 启动 / 恢复流程（head 01 执行编排）
```bash
# 前置：四机已开机（01→02→03→04）、/opt/nccl-ringonly 四机 lib 在位（md5 校验）、
#       01/02 内存余量 ≥10G（必要时清 buff/cache 或降 max-num-seqs）、
#       重启后复核 isolcpus(nproc=18)/MTU9000/GID/对外端口 3000/8191/8001/iptables
cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh
```
- **流程**：head(rank0=01) 先启 → 轮询 TCPStore :25999（每 5s）→ 依序启动 rank1(02)/rank2(04)/rank3(03) → 轮询 :8001 /v1/models 就绪。head-first 铁律；禁止 worker 先启/单边重建；worker 脚本自检走 `check_vllm_script.sh`。
- **验证**：`curl -H "Authorization: Bearer <internal-key>" http://127.0.0.1:8001/v1/models`=200；四机 `docker ps | grep vllm-tp4` 全 Up (healthy)；chat 冒烟 `"2+2=?"`→`"4"`；补丁 banner 校验见 §A.7。
- **停机反向**：worker 先 rm（03→04→02）→ head(01) 后停。

### A.6 回滚到 TP2
1. 四机 `docker rm -f vllm-tp4-rank0~3`
2. 01 上运行既有 `start_v026r_cluster.sh` 恢复 TP2（start_tp4_* 脚本已备份 `.bak-tp4-patch2`；TP2 脚本/镜像/权重全程未动 = 锚点）
3. 验证 8001/8003/4000 全 200 + e2e chat
4. TP2 无需 LD_PRELOAD 补丁库，/opt/nccl-ringonly 保留不影响

### A.7 重启恢复流程要点（操作手册，归档 README 附录同步收录）
- 开机顺序 01→02→03→04；启动走 `start_tp4_cluster.sh`；8001=200 就绪（权重加载 5-8 分钟）
- **补丁 banner 校验**：`docker logs vllm-tp4-rank0 --tail 50 | grep -i "NCCL version"` 期望 `2.30.7+cuda13.0`；出现 `13.3` = LD_PRELOAD 未生效立即排查
- 常见故障：①NCCL 110 → 查 PEER_HCA 表/环拓扑/banner；②8001 不监听卡 init → 宿主机 `ss -ltnp|grep 25999` + NCCL_DEBUG 日志，head-first 重来；③healthy 但 API 不可达 → 权重加载中等待；④OOM → 清缓存/降 max-num-seqs；⑤链路错误 → ethtool 16 口计数 + iperf3 复核

---

## §B. 隔离核记录修正（1-4，替代旧"18-19"）
- **实际生效**：`isolcpus=0-4 rcu_nocbs=0-4`（grub.d/90-isolcpus.cfg），**四机已生效**。核拓扑：0-4=A725（低功耗 2808MHz）、5-9=X925（高性能 3900MHz）、10-14=A725、15-19=X925。
- **NCCL 线程/延迟敏感 = 隔离核 1-4**（A725；避开 CPU0 boot 核，CPU0 无法隔离）。**旧记录"18-19"已过时，勿再引用。**
- nproc=16 已查明非故障：= CPU0 + CPU5-19（isolcpus 后 cpuset.isolated=1-4）；容器显式 `--cpuset-cpus=1-19` + 进程 taskset 分绑（1-4 NCCL / 5-9 EngineCore / 15-19 备用）。
- 实施：`LD_PRELOAD=/opt/libncclpin.so` + taskset；mlx5 RoCE IRQ smp_affinity 绑 5-9。

---

## §C. 8/11 全链事故与修复记录（纳入 Runbook 故障章节）

### C.1 Grafana 外部不可达（P0，已修复）
- 根因：docker 重启容器 IP 漂移 → iptables DOCKER 链 DNAT 指向历史 IP（:3000→prometheus、:8191→redis 等全错位）；ring-close 的 iptables-save 固化错位快照；127.0.0.1 自检被 docker-proxy 绕过造成假象。
- 修复：`cd /opt/aicad && docker compose restart`（7 容器按当前 IP 重建 DOCKER 链）→ 立即 `iptables-save > /etc/iptables/rules.v4` 覆盖错位快照；外部 3000/8191 恢复 200。
- 教训：自检必须用**对外 IP**（红线 7）；rules.v4 不得固化 docker 动态 DOCKER 链（红线 6）；compose 容器建议固定 ipv4_address 防漂移。

### C.2 iptables 白名单阻断新链路（已根治）
- 四台 iptables 为白名单模式，只放行旧相邻段 IP；新链路 10.100.140/141 TCP 全 DROP → iperf3 初测失败暴露。
- 修复：放行 10.100.140/141 并持久化 rules.v4；各机放行各自 peer 侧（01 放 140.2/141.2 等）。
- **流程固化**：任何新 RoCE 网段配 IP 必须同步放行 iptables（红线 5）。

### C.3 环网 FEC 错误率（P1 解除）
- 03 module0 FEC corr 2280 → 0（重插复位，down/up ~24s）；16 口 PHY/IP 错误全 0。
- iperf3 四段双向 99-110Gbps（CPU 受限非线速）、重传 0-6；压测后计数仍 0 → 物理层 P1 **正式解除**。
- 观察：01/02 共享计数（2281/2503/341/329）为设备级累计，无活动错误证据，需 mft/mlxlink 精确基线（P2）。

---

## §D. 待正式 Runbook 合并时需同步更新
1. §0 当前状态 → TP4 上线（8001=200、fingerprint `-tp4-`）
2. §1.1 拓扑图 → 四机环网 rank/IP/端口（25999）
3. §2 启动 SOP → `start_tp4_cluster.sh`（替代 start_v026r_cluster.sh 为 TP4 权威流程）；停机顺序 worker 先 head 后
4. §4 故障章节 → 增补 C.1/C.2/C.3 + A.7 故障表
5. §5 坑位 → 隔离核 1-4 修正；NCCL 110 根因（全对建连+index 配对）；iptables 白名单教训
6. §7 自反馈 → TP4 攻坚闭环、Grafana/iptables/FEC 事故闭环
7. 遗留跟踪：双口带宽优化（NCCL_IB_MERGE_NICS）、TP4 完整 bench 对比 TP2、QoS 开机持久化复验、mft/mlxlink 基线
