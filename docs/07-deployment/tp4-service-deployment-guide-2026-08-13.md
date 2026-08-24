# DGX Spark 四机 TP4 服务部署指导文档 v1.1

**适用**：运维团队 / 交付团队
**环境**：4× NVIDIA DGX Spark（GB10，UMA 121.6GiB，20 核异构）纯环网
**基线日期**：2026-08-13（基于 8/9-8/13 完整生产落地，含 R11 修复与 TP4FINAL 测试）
**修订日期**：2026-08-13（双向审核后修订）
**文档位置**：服务器 01/02 `<INSTALL_DIR>/docs/`（镜像）+ 本地 `deliverables/engineering-assurance/`
**密钥声明**：本文档不含任何明文密钥/密码。sudo 密码、vLLM API key、litellm master_key 一律经环境变量注入，不入文档、不入仓库。

---

## 1. 总览

### 1.1 架构摘要
- **推理**：4 机环网 Tensor Parallel=4（vLLM），head=01，单实例 8001 API
- **网关**：litellm（02 :4000，并发 12）+ litellm-pg（02，postgres:16）
- **Embed**：03/04 :8022（anemll，Qwen3-0.6B）
- **监控**：Grafana 01 与 02 双实例均 :3000（去重待执行，目标保 02 权威）+ Prometheus（02 :8191 对外）+ alertmanager（02 :9093）+ dcgm/node-exporter 四机
- **镜像仓库**：02 内建 registry `<NODE_IP>:5000`（四机 insecure-registries）
- **权重存储**：01/02 数据节点权威源（NFS 导出，源路径 `/home/<USER>/models/deepseek-v4-flash-0731`）→ 03/04 运行节点 RoCE 挂载至 `<MODELS_DIR>/...`（ro）
- **编排**：systemd 自愈（head/worker 单元 + monitor wrapper + 门禁/退避/互杀守卫）

### 1.2 关键配置基线（R12 定稿）
| 参数 | 值 |
|---|---|
| max-num-seqs | 6（曾 12→6，缓解并发堵塞） |
| gpu-memory-utilization | 0.65（0.60 为 400k×12 地板） |
| max-model-len | 400000（原生支持，YaRN 1048576） |
| 隔离核 | isolcpus=8-9（NCCL 绑 8-9） |
| nproc | 18（系会话 affinity 排除隔离核 8-9 所致，`nproc --all`=20） |
| EngineCore 绑定 | 15-19（X925） |
| CUDA Graph | max-cudagraph-capture-size 64 + capture-sizes `1 2 4 8 16 24 32 36 40 48 56 64`（含 36，seqs=6 稳态；无 72/80） |
| Prefix KV | --enable-prefix-caching + retention 4096 + long-prefill 1024 |
| 投机解码 | dspark（num_speculative_tokens=5） |
| 分布式 | MASTER_PORT 25999、--distributed-timeout-seconds 300 |
| NCCL | RING + SUBNET_AWARE_ROUTING=1 + NET_PLUGIN=none + MERGE_NICS=0 + PEER_HCA 双 dev |
| NCCL 延迟优化 R13（2026-08-16 生产生效） | T1aM4+MAX_CH16：`MIN_NCHANNELS=4` + `PROTO=Simple` + `BUFFSIZE=8388608` + `MAX_NCHANNELS=16`。依据 deliverables/engineering-assurance/nccl-p0-scan-results-2026-08-16.md 与 nccl-maxch16-e2e-verification-2026-08-16.md。368KB allreduce 923→173µs（-81%）；端到端 32K PR+5.5%/DE+6.5%，131K PR+21.7%/TTFT-17.2%。回滚：.bak-ncclMAXCH16-20260816 四机。 |
| 内存 | 容器 --shm-size 64g、无 docker 内存硬限制（Memory=0） |
| API 鉴权 | --api-key（已启用，`/health` 免鉴权、`/v1/*` 需 `Authorization: Bearer`） |

---

## 2. 镜像方案

### 2.1 镜像清单
| 服务 | 镜像 | 版本基线 | 来源 | 大小 | 部署位置 |
|---|---|---|---|---|---|
| TP4 推理 | `ghcr.io/anemll/dspark-vllm-gx10` | 0.2.1-v026.0（vLLM 0.26.1.dev0 / B12x / FlashInfer） | ghcr.io/anemll → 02 registry | 01=34.2GB、02/03/04=21.6GB | 四机 |
| Embed | 同镜像 | 0.2.1-v026.0（serve --task embed） | 同上 | 21.6GB | 03/04 |
| 网关 | `ghcr.io/berriai/litellm` | v1.83.7-stable | ghcr（非 Docker Hub）→ 02 registry | 1.98GB | 02 |
| 监控 | grafana/prometheus/alertmanager/dcgm-exporter/node-exporter | 13.1.1 / 2.x | 官方 → 02 registry | 各 <1G | 01/02+四机 exporter |
| 镜像仓库 | registry:2 | 2.x | 官方 | 40G（/data/registry） | 02 |
| aicad 应用栈 | neo4j/redis/postgres/minio/dashboard/aicad-fw 等 | 运行中（审核新发现，待逐一建档） | 02 registry catalog | — | 01/02 |

### 2.2 仓库管理
- **权威源**：02 `<NODE_IP>:5000`（旧 `.58` 已废弃；四机 daemon.json insecure-registries 含新旧地址）
- 拉取后**保留 registry tag**，运行 tag 另打（避免直接引用远端导致重拉）
- 测试镜像（archi-test 等）用后即删（rmi），避免积压（曾回收 98.9G）
- **未文档镜像仓**：registry catalog 实测含 20+ 未文档镜像仓（minio/neo4j/chroma/pgvector/comfyui/vllm-gb10/embed-gpu 等），目录待逐一建档（P2）

### 2.3 拉取策略
- 四机本地 `docker images` 常备（启动不依赖 registry 在线）
- 新版本：推 02 registry → 四机 `docker pull` → 脚本 tag 切换 → 滚动验证

---

## 3. 调度约束方案

### 3.1 说明
本环境为**裸机 systemd + docker 编排**（非 K8s）。调度语义通过 systemd 单元依赖 + docker 运行参数 + 隔离核/线程绑定实现。3.6 给出等价 K8s YAML 供迁移参考。

### 3.2 节点选择与角色
| 节点 | IP（管理/RoCE） | 角色 | 存储 |
|---|---|---|---|
| 01 | .186 / 10.100.136/137 + 10.100.140/141 | head（rank0）+ 数据源 | 3.6T |
| 02 | .187 / 10.100.136/137 + <NODE_IP>/30 + <NODE_IP>/30 | worker(rank1) + 监控 + registry + 数据源 | 3.6T |
| 03 | .188 / 10.100.138/139 + 10.100.140/141 | worker(rank3) + embed | 916G |
| 04 | .189 / 10.100.138/139 + <NODE_IP>/30 + <NODE_IP>/30 | worker(rank2) + embed | 916G |

> 环序 **01(0)→02(1)→04(2)→03(3)→01**（NODE_RANK 以 systemd 实测为准，见 §3.5）。

### 3.3 资源限制（等价 requests/limits）
```bash
# docker run 关键参数（start_tp4_head.sh 实证）
--gpus all
--cpuset-cpus 1-19              # 容器核范围（含隔离核 8-9 供 NCCL + 15-19 供 EngineCore）
--shm-size 64g                  # 无 -m/--memory-swap 内存硬限制（docker inspect Memory=0）
-e VLLM_ENGINE_READY_TIMEOUT_S=600
```
- **GPU 占用核算**（util 0.65）：权重 40.5 + 峰值激活 ~1.8 + CUDAGraph ~1.2-1.4 + non-torch ~0.1-2.4 + KV ~34.4-36.7 ≈ **79GiB**（util 0.65×121.63GiB 预算）；KV 池口径统一为 **~1.2-1.4M tokens**（随 util 波动，r9 1.2M / r12 1.43M 两说并存，以日志 block 核算为准）
- **显存地板**：400k×12 配置 util 0.60 是地板（0.55 即 `No available memory for cache blocks`）

### 3.4 亲和性/隔离（等价 nodeAffinity/taints）
- **隔离核**：grub `isolcpus=8-9 rcu_nocbs=8-9`（四机，update-grub 后重启生效）
- **线程绑定**（shim v8，LD_PRELOAD 实际路径（容器内）`/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`）：
  - NCCL 全部线程（Progress/IbAsync/Service/RAS/UDS/tcpstore/pt_nccl_*）→ **8-9**
  - EngineCore（vLLM 主进程，prctl PR_SET_NAME 捕获）→ **15-19**
  - 其余默认 → 5-19
- **GPU-gate**：启动前置 `nvidia-smi` 探测 ≤180s 才允许拉起（等价 Ready 门控）；注：该门禁仅存在于 `start_tp4_cluster.sh` 编排路径，systemd 自愈路径（monitor→start_tp4_*.sh）暂未经过（待下沉，P2）
- **对端门禁**：worker 等待 head TCPStore :25999（120s）；head 就绪后 60s 缺秩 exit(1) 全链重建
- **污点/容忍等价**：worker 失败退避（**线性 60s×n**，60..600s，≤10 次）+ StartLimitBurst=20/1800s 防永久 failed

### 3.5 调度依赖（systemd 单元要点）
```ini
# /etc/systemd/system/vllm-tp4-worker.service（03/04 实证）
[Unit]
Description=vllm-tp4-worker
After=network-online.target docker.service remote-fs.target
RequiresMountsFor=<MODELS_DIR>/deepseek-v4-flash-0731   # NFS 权重就绪（R12 修复）
StartLimitIntervalSec=1800
StartLimitBurst=20
[Service]
Type=simple
Restart=always
RestartSec=15
TimeoutStartSec=1500
# 注：NODE_RANK 与 LD_PRELOAD 由 start_tp4_worker.sh 内 docker -e 传入（非 unit Environment）
# 04 对应 NODE_RANK=2、03 对应 NODE_RANK=3（01=0、02=1）
ExecStart=<INSTALL_DIR>/scripts/start_tp4_worker.sh
```

### 3.6 K8s 迁移等价 YAML（参考）
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: {name: tp4}
spec:
  replicas: 4
  podManagementPolicy: OrderedReady      # head-first（rank0 先起）
  # rank 映射对齐环序：01=0、02=1、04=2、03=3（03/04 为 worker(rank3)/(rank2)）
  template:
    spec:
      nodeSelector: {role: spark}        # 四机
      tolerations: [{key: isolated-cpu, operator: Exists}]
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: {matchLabels: {app: tp4}}
            topologyKey: kubernetes.io/hostname   # 每机 1 rank
      containers:
      - name: vllm
        image: <NODE_IP>:5000/dspark-vllm-gx10:0.2.1-v026.0
        resources:
          limits: {nvidia.com/gpu: "1", cpu: "19"}   # 生产无内存硬限制（Memory=0），此处可加 limits 仅作示例
        env:
        - {name: LD_PRELOAD, value: "/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2"}
```

---

## 4. 组网方案

### 4.1 网络拓扑（文字拓扑）
```
            01 ════════ 02
           ║             ║
           ║             ║
          03 ════════ 04

  环网 01-02-04-03-01（四边全连，每边双链路；每机 2 OSFP = 4 逻辑口全配 IP）
  ├ 01↔02：10.100.136/137
  ├ 02↔04：<NODE_IP>/30 + <NODE_IP>/30（TP2 遗留段）
  ├ 04↔03：10.100.138/139
  └ 03↔01：10.100.140/141
  管理网 <NODE_IP>~189（2.5GbE，四机实测 speed=2500）——控制面/SSH/API/监控
```
- **环网 L3 闭合**（每边双链路）：01↔02（10.100.136/137）、02↔04（<NODE_IP>/30 + <NODE_IP>/30，TP2 遗留段）、04↔03（10.100.138/139）、03↔01（10.100.140/141）
- **RoCE**：GID_INDEX=3；**MTU=9000（jumbo）**
- 数据面 10.20.0.x 为 TP2 遗留段（环网 4 边中 02↔04 边沿用该段）；对角无直连（RDMA 2 跳已放弃）

### 4.2 服务间通信
| 流量 | 路径 | 端口 |
|---|---|---|
| 控制面（TCPStore） | head 25999 ← workers（管理网/RoCE） | 25999 |
| NCCL 数据面 | 环邻对（RoCE 直连，PEER_HCA 对口） | IB/RoCE |
| 推理 API | 客户端 → 186:8001 | 8001 |
| 网关 | 客户端 → 02:4000 → 8001 | 4000 |
| Embed | 03/04:8022（litellm 池 .188/.189） | 8022 |
| 监控 | Prometheus 8191 ← dcgm 9400/node 9100 四机；alertmanager 02:9093；02 Grafana 3000 | 8191/9400/9100/9093/3000 |
| 权重 | NFS 01→03（<NODE_IP>，源 `/home/<USER>/models/deepseek-v4-flash-0731`）、02→04（<NODE_IP>，同源） | 2049 |
| aicad 应用栈 | Neo4j（01/02）、Redis（01/02）、Postgres（02）、MinIO（01/02）、aicad-fw、dashboard、02 应用 | 7474/7687、6379、8082(→5432)、19000/50081、25000、11000、8003 |

### 4.3 对外暴露与防火墙
- **iptables 白名单**（四机持久化 rules.v4）：只放行已知 peer IP 与端口；曾因白名单只放旧相邻 IP 导致新链路全 DROP（见避坑）
- 内网仅 TCP + `rp_filter=1`；管理网 Wi-Fi 禁用
- **QoS**：mlnx-qos.service（DSCP trust，DSCP46→P5 无损），应用 `NCCL_IB_TOS=46`
- **风险注记**：Neo4j/MinIO 监听 0.0.0.0 且不在 iptables 白名单体系（白名单仅覆盖 RoCE 环邻），管理网侧收敛为 P2 整改项

### 4.4 负载均衡与网关（等价 Ingress/Service）
```yaml
# K8s 等价：litellm = Service(ClusterIP:4000) + Deployment；vLLM = Service(NodePort:8001)
apiVersion: v1
kind: Service
metadata: {name: llm-gateway}
spec:
  type: NodePort
  selector: {app: litellm}
  ports: [{port: 4000, nodePort: 30000}]   # 对外统一入口
---
apiVersion: v1
kind: Service
metadata: {name: vllm-tp4}
spec:
  type: ClusterIP
  selector: {app: tp4, rank: "0"}          # 仅 head 暴露
  ports: [{port: 8001}]
```
- 生产（裸机）：客户端统一走 litellm :4000（并发 default_max_parallel_requests=12）；直连 8001 供调试

---

## 5. 补丁功能

### 5.1 补丁清单
| # | 补丁 | 版本链 | 位置 | 作用 |
|---|---|---|---|---|
| P1 | NCCL ring-only v1 | v1→v2→v3 | /opt/nccl-ringonly/libnccl.so.2.30.7 | 纯环网 4 rank TP4 可行性 |
| P2 | NCCL PEER_HCA | v2（单 dev）→v3（双 dev 轮换） | 同上 | 单口→双口带宽 13.9→23.9GB/s |
| P3 | libncclpin shim | v1→v4→v8 | <INSTALL_DIR>/lib/libncclpin.so | 线程绑定隔离核（8-9/15-19） |
| P4 | vLLM capture-sizes | 默认→显式 capture（历史命名 fix72） | start_tp4 脚本参数 | 修复稳态 batch 截断（TPOT 17×）；最终运行配置 capture 1..64 含 36（无 72） |
| P5 | 编排修复 | R11 批 | systemd/monitor 脚本 | NFS 依赖/退避/互杀守卫/门禁 |

### 5.2 关键补丁前后对比
| 补丁 | 前 | 后 | 量化收益 |
|---|---|---|---|
| P1 v1 | stock NCCL 纯环 4 rank 110 崩溃 | RING 过滤非环邻对，连通 | 4 rank sum=6 零 110 |
| P2 v3 | 单口 13.87 GB/s | 双 dev 轮换 23.86 GB/s | **+72%** |
| P3 shim v8 | NCCL 线程落 5-19（未隔离） | NCCL 8-9 / EngineCore 15-19 | PSR 实测一致 |
| P4 fix72（历史命名） | decode 稳态 batch 72 落慢路径（TPOT 7.4s） | capture 显式化 → 快路径（0.43s） | **TPOT 17×**（修复成果保留为历史事实；最终配置 capture 1..64 含 36，无 72） |
| P5 互杀守卫 | 冷启动 head/worker 互杀循环 | 集群成形才触发重建 | 重启 HEAD_KILL=0 |

### 5.3 升级路径与回滚
- 构建：容器内 `make src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE sm_121`（源码 <INSTALL_DIR>/backup/tp4-20260812/）
- 分发：`shim-deploy.sh {check|deploy|rollback|diff}`（MD5 校验，部署前自动备份锚点）
- 回滚锚点（详见 rollback-anchors-2026-08-12.md）：`.bak-v2`（库）、`.bak-tp4-*`（脚本）、`.bak-v7/v6`（shim）、ring-fix（网络）、tp4-20260812（源码）
- **注**：01 无 start_tp4_head.sh 回滚锚点（与 rollback-anchors §1.1 声明不符），补齐为 P2 整改
- **MD5 注（2026-08-14 已核对）**：rollback-anchors / runbook 均已更新为 b7784b49（ringonly）与 ce43c688（shim v8），本注作废

### 5.4 验证方法
```bash
# 补丁生效验证（四机）
md5sum /opt/nccl-ringonly/libnccl.so.2.30.7      # 期望 b7784b49885659c27765e648884e4edd（v3 双口，四机一致）
md5sum <INSTALL_DIR>/lib/libncclpin.so         # 期望 ce43c688c5164ac7efd5105c94fdab77（v8，四机一致）
# RING-ONLY banner 在 NCCL_DEBUG_FILE（容器内 ~/.vllm-logs/nccl-*.log），不在 docker logs stdout
# 线程绑定 PSR（负载下采样）
ps -eLo pid,tid,psr,comm | grep -E "NCCL|EngineC"
# 期望：NCCL* → 8-9、EngineCore → 15-19
curl -s <NODE_IP>:8001/health                 # 200（/health 免鉴权，可裸测）
curl -s -H "Authorization: Bearer <VLLM_API_KEY>" <NODE_IP>:8001/v1/models   # 需鉴权；期望 max_model_len=400000
```

---

## 6. 模块选择方案

### 6.1 选型与启停
| 模块 | 选型依据 | 启停 | 配置要点 | 启停影响 |
|---|---|---|---|---|
| vLLM TP4 | 4×121.6GiB UMA 池（约 522GB，营销口径 512GB）、长上下文 | systemd vllm-tp4-head/worker | seqs=6/util 0.65/400k/capture 64 | 停=推理中断（全链） |
| Embed | 轻量 Qwen3-0.6B | docker anemll-embed-8022（03/04） | --kv-cache-memory=4294967296 | 停=embed 不可用，TP4 不受影响 |
| litellm 网关 | 统一入口/并发控制 | docker（02 :4000） | default_max_parallel_requests=12 | 停=API 入口断，直连 8001 可用 |
| NCCL 补丁库 | 纯环网必需 | LD_PRELOAD 常驻 | PEER_HCA 双 dev | 缺失=TP4 无法组网 |
| shim 绑定 | 隔离核/延迟保障 | LD_PRELOAD 常驻 | 目标 8-9/15-19 | 缺失=线程落 5-19 |
| Grafana/Prom | 监控 | 01/02 双 Grafana（去重待执行，保 02 权威）/ 02 Prom | 数据源 .187:8191 | 停=无监控，不影响推理 |
| NFS | 权重集中化 | 01/02 export + 03/04 mount | ro,hard,timeo=600,nconnect=4 | 断=03/04 权重不可读（软链兜底，删除决策待裁决 P1） |

- **NFS 兜底**：目录实际为 `deepseek-v4-flash-0731.local-backup`（03=156G + 04=156G，共 312G）；**删除决策待裁决（P1）**——删除后 01/02 任一 NFS 故障 → 对应 worker 断权 → TP4 全链断、无本地兜底
- **断链影响**：01 挂 → 03 断权；02 挂 → 04 断权；任一 rank 权重不可读即全链断

### 6.2 启停顺序（纪律）
- **停机**：worker（03→04→02）→ head（01）
- **启动**：head-first（01 → workers，TCPStore 门禁）
- **防自愈干扰**：`systemctl stop vllm-tp4-{head,worker}.service` → 确认 monitor 退出 → `docker rm -f` 容器 → 无残留后再操作

---

## 7. 分步部署指导

### 7.1 前置条件
- 4× DGX Spark 物理连线：环网（每机 2 OSFP 4 逻辑口全配 IP）+ 管理网
- 基础系统：UTC 时区（注：01 现为 Asia/Hong_Kong，非 UTC，其余三机 UTC，整改 P2）、管理网静态 IP 186-189、SSH 免密 01→02/03/04
- 4 机 nproc=18（会话 affinity 排除隔离核 8-9 所致，`nproc --all`=20）、UMA 121.6GiB

### 7.2 阶段 A：网络与系统（可复现步骤）
```bash
# 1. 环网 IP（示例 01）—— 全 4 口配 IP 后才可达
ip addr add <NODE_IP>/24 dev enp1s0f0np0   # 01↔02
ip addr add <NODE_IP>/24 dev enP2p1s0f0np0 # 01↔03
# 2. RoCE MTU 9000（jumbo，四机）
ip link set dev <roce_if> mtu 9000
# 3. 隔离核（四机）
echo 'GRUB_CMDLINE_LINUX_DEFAULT="... isolcpus=8-9 rcu_nocbs=8-9"' > /etc/default/grub.d/90-isolcpus.cfg
update-grub && reboot
# 4. QoS（DSCP trust，四机）
systemctl enable --now mlnx-qos.service
# 5. iptables 白名单 + 持久化
iptables-save > /etc/iptables/rules.v4   # 只放行 peer IP/端口
# 6. hosts / SSH 别名 node01~04
```

### 7.3 阶段 B：仓库与镜像
```bash
# 02 建 registry；四机 daemon.json 加 insecure-registries
docker run -d -p 5000:5000 -v /data/registry:/var/lib/registry registry:2
# 拉取权威镜像 → push 02 → 四机 pull 并本地保留
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0
docker tag ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0 <NODE_IP>:5000/dspark-vllm-gx10:0.2.1-v026.0
```

### 7.4 阶段 C：NCCL 补丁构建与部署
```bash
# 01：容器内构建（源码 backup/tp4-20260812/src）
docker run --rm --entrypoint bash --user root -v <INSTALL_DIR>/backup/tp4-20260812/src:/src \
  ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0 -c "cd /src && make src.build \
  CUDA_HOME=/usr/local/cuda NVCC_GENCODE=sm_121"
# 分发（shim-deploy 同型）：scp 四机 /opt/nccl-ringonly/ + md5 校验
# shim 构建：gcc 编译 libncclpin_v8.c → <INSTALL_DIR>/lib/libncclpin.so
./shim-deploy.sh deploy    # MD5 校验 + 备份锚点 + 四机分发
```

### 7.5 阶段 D：TP4 启动编排
```bash
# 1. systemd 单元（head/worker + monitor wrapper），单元含 NFS 依赖与门禁
# 2. 启动（head-first）
systemctl start vllm-tp4-head.service    # 01
systemctl start vllm-tp4-worker.service  # 02/04/03
# 3. 验证
curl -s <NODE_IP>:8001/health        # 200（/health 免鉴权，可裸测）
curl -s -H "Authorization: Bearer <VLLM_API_KEY>" <NODE_IP>:8001/v1/models   # max_model_len=400000（/v1/* 需鉴权）
```

### 7.6 阶段 E：监控与网关
```bash
# litellm（02）：config.yaml 加 default_max_parallel_requests:12
# Grafana（01/02 双实例，去重待执行保 02 权威）数据源 http://<NODE_IP>:8191；Prometheus scrape 四机 dcgm/node
# 面板：vllm-dspark-cluster（15）、vllm-realtime（30）
```

### 7.7 阶段 F：验收（对标 TP4FINAL，已实测全部满足 ✅）
| 指标 | 验收值（c1 档） |
|---|---|
| prefill（131072） | ≥2000 tok/s |
| decode | ≥100 tok/s（coding/json） |
| 投机接受率 | coding/json ≥0.75 |
| 满长并发 | 400k 下 ≈3.58/机 |

---

## 8. 避坑指南

| # | 报错/现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `ibv_modify_qp 110`（4 rank） | stock NCCL 纯环建全 rank 对 QP + index 配对 | ring-only 补丁 v1+v2 |
| 2 | 单口 4.4GB/s 上不去 | PEER_HCA 单 dev | v3 双 dev channelId%2 轮换 |
| 3 | `No available memory for cache blocks` | util < 地板（400k×12 需 0.60） | util 提到 0.65；核算内存账 |
| 4 | 重启后 worker failed（StartLimit） | NFS 挂载失败 + 秒退烧限流 | fstab 正确 RoCE 源 + RequiresMountsFor + 退避 + StartLimit 1800/20 |
| 5 | 冷启动 head/worker 互杀循环 | health=pgrep 误判 + 双删 | 互杀守卫（集群成形才触发）+ 门禁 |
| 6 | 新链路 TCP 全 DROP | iptables 白名单只放旧相邻 IP | 白名单补新段 + rules.v4 持久化 |
| 7 | NFS "Network is unreachable for <NODE_IP>" | fstab 用 02 改名前的旧 IP（stale） | fstab 改正确 RoCE 源（<NODE_IP>/<NODE_IP>） |
| 8 | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 后 prefill -25% | 单设 0 逼 torch.compile（0.25.x 不支持） | 保持 =1（fix72 需 1+capture 组合） |
| 9 | capture-sizes 参数不生效 | nargs='+' 需空格分隔 | `--cudagraph-capture-sizes 1 2 ... 64` |
| 10 | 副本测试 WorkerProc 失败 | TCPStore 死锁（head-first 顺序）或 util 差异 | 并发启动 + util 对齐生产 |
| 11 | 管理网 NFS 10min 超 300s | 2.5GbE 带宽不足 | 走 RoCE 对口（45-90s） |
| 12 | 容器反复 Exited(1) | systemd 自愈 monitor 未停 + 副本抢 GPU | 维护窗口先 systemctl stop + 确认 monitor 退出 |
| 13 | `curl /v1/models` 返回 401 | API 鉴权已启用（--api-key） | 需带 `Authorization: Bearer <VLLM_API_KEY>`（/health 免鉴权） |
| 14 | 误记 MTU 1500 | 实测 9000（jumbo） | 排障勿按 1500 推演，QoS/RoCE 均按 jumbo 口径 |

---

## 9. 测试佐证与问题报告

### 9.1 性能基线（TP4FINAL，45 组合 0 错误；315 格数据核验 0 错误）
| 档位/并发 | prefill | decode | TTFT | 投机 acc |
|---|---|---|---|---|
| 131072/c1 | 2013-2016 | 110-115 | 57.4s | coding 0.81 |
| 32768/c1 | 2208-2228 | 60-119 | 13.0s | 0.82-0.86 |
| 8192/c1 | 2210-2222 | 106-122 | 3.3s | 0.73-0.88 |
| 131072/c5 | 633-642 | **7.8-8.2（断崖）** | 182.7s | 0.86-0.92 |
| 峰值 decode | 41-87（c5 单请求窗口） | — | — | — |
- 并发结论：小档 c5 TPS 1.7-1.9×（亚线性）；**长档（≥32768）建议并发 ≤c3**（c5 反噬）
- kv_cache 峰值 14.7%、零 OOM、零超时
- **基线对比注记**：TP4S 数据不完整（运行中断 14/45），以同配置 TP4R 为参考基线，无回退、decode 略升（131072 coding c1：prefill 1801.6→2014.7、decode 92.1→110.0）

### 9.2 可靠性测试（R11 复核 7 项）
- 故障注入：kill worker → 15s 重建；藏 config → 门禁等待非秒退；重加入 → 自愈收敛（P1 遗留已修）
- 长稳 30min 零错误；重启验证 7/7（NFS 自动挂载/单次拉起/PSR/互杀 0）

### 9.3 缺陷记录与处理结论
| # | 缺陷 | 处理 | 结论 |
|---|---|---|---|
| D1 | 03/04 worker 永久 failed（NFS） | RequiresMountsFor + 门禁 + 退避 | 已修复（R12 重启验证） |
| D2 | head/worker 互杀循环 | 互杀守卫 | 已修复（HEAD_KILL=0） |
| D3 | decode 稳态 72 截断 | capture-sizes 显式化（历史命名 fix72，最终配置无 72） | 已修复（TPOT 17×） |
| D4 | 131072×c5 decode 断崖 | 记录为 P2（seqs=6 约束） | 建议长档 ≤c3 |
| D5 | NFS fstab stale IP | fstab 修正 + 文档 | 已修复 |
| D6 | shim 竞态（线程 pin 覆盖） | shim v8 mark-then-pin | 已修复（PSR 实测） |

---

## 10. 安全整改与待办

> 本文档不含明文密钥/密码。sudo 密码、vLLM API key、litellm master_key 一律经环境变量注入，不入文档、不入仓库。

### 10.1 安全整改（P0）
- sudo 密码轮换 + 移除 `docs/file-registry.md:102` 明文条目
- API key / litellm master_key 环境变量化 + 清理 5 份 .bak
- 核心脚本权限收敛（700/750）

### 10.2 文档与配置整改（P1）
- 本文档同步 01/02 `<INSTALL_DIR>/docs/` 镜像
- `.local-backup` 删除裁决（03=156G + 04=156G，共 312G）
- 文档批次 B：rollback-anchors §2.1 / runbook §A.3 的 NCCL MD5 更新为 `b7784b49`（v3 双口）

### 10.3 待办（P2）
- 01 时区改 UTC（现为 Asia/Hong_Kong）
- 补齐 01 head 脚本（start_tp4_head.sh）回滚锚点
- sshd 显式收敛（禁密码认证 / 禁 root 登录）
- 双 Grafana 去重（保 02 权威）
- Neo4j/MinIO 端口绑管理网白名单收敛
- Prometheus job 名/标签清理（旧 .55/.58/.59/.60 命名）与失效抓取目标

---

## 附录

- 回滚锚点手册：`docs/rollback-anchors-2026-08-12.md`
- 运维文档体系：`docs/ops/`（启停纪律/维护手册/工具索引/容错/自恢复）
- 脚本引用：`docs/scripts/REFERENCE.md`（10 脚本 → 文档映射）
- 测试数据：`_tessa_tp4_bench/TP4FINAL/`（rows/summary/report）

> 本文档基于生产实测数据编制（2026-08-13），配置变更须同步更新 01/02 `<INSTALL_DIR>/docs/` 镜像。

---

## 修订记录

| 版本 | 日期 | 修订内容 | 依据 |
|---|---|---|---|
| v1.1 | 2026-08-13 | 全文版本号 v1.0→v1.1，标题与文首标注"双向审核后修订" | 双向审核总报告 |
| v1.1 | 2026-08-13 | §3.2 节点表 rank 修正：03=rank3、04=rank2，环序 01-02-04-03-01 | F1 四路独立佐证 |
| v1.1 | 2026-08-13 | §1.2/§5 CUDA Graph：capture 1..64 含 36，无 72/80；fix72 降级为历史命名 | F3/Rex B10/Archi D.14 |
| v1.1 | 2026-08-13 | §3.3 --shm-size 32g→64g；删除 -m 100g --memory-swap 100g 声明 | F3/Rex B8/Cody B6 |
| v1.1 | 2026-08-13 | §3.3 显存账改为实测值并修正算术（40.5+1.8+1.2~1.4+0.1~2.4+34~37）；KV 池口径统一 | Archi B.8/ADR-3 |
| v1.1 | 2026-08-13 | §3.4/§3.5/§3.6 LD_PRELOAD 实际路径 /opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2 | F3/Rex B11/Cody B7 |
| v1.1 | 2026-08-13 | §3.4 退避改"线性 60s×n"；GPU-gate 补覆盖缺口注记 | Cody issue 6/15 |
| v1.1 | 2026-08-13 | §4.1 MTU 1500→9000（jumbo）；ASCII 图补全 4 边；网段表述修正（删除"环网 TP4 用 10.100.x"） | Archi A.2/ADR-1/2 |
| v1.1 | 2026-08-13 | §2.1 镜像 tag 0.2.1→0.2.1-v026.0；大小 01=34.2G、02/03/04=21.6G；litellm 来源 ghcr | F3/Cody D18/Archi D.15 |
| v1.1 | 2026-08-13 | §1.1 补 alertmanager/litellm-pg/双 Grafana；§2.1/§4.2 补 aicad 应用栈；§2.2 补 20+ 未文档镜像仓 | F4/Rex F27 |
| v1.1 | 2026-08-13 | §1.2 补 API 鉴权与内存基线行；§5.4/§7.5 验证命令补 API key 说明 | Cody A3/Tessa D |
| v1.1 | 2026-08-13 | §5 MD5 双值确认 + 注记 rollback/runbook 陈旧值 4cc43e3b；§5.3 补 01 无 head 锚点 | Cody A1/A2/B8/Archi D.14 |
| v1.1 | 2026-08-13 | §6 NFS 兜底目录名/删除裁决 P1；补断链影响行 | Archi C.11/ADR-5 |
| v1.1 | 2026-08-13 | §7.1 时区注记（01=Asia/Hong_Kong）；§7.2 补 MTU 9000 | Rex B13/Archi A.2 |
| v1.1 | 2026-08-13 | §8 追加 #13（401 鉴权）、#14（MTU 9000） | Cody/Tessa |
| v1.1 | 2026-08-13 | §9 "vs 基线"改注 TP4R 参考基线；D3 缺陷注记 fix72 历史命名 | Tessa C.7 |
| v1.1 | 2026-08-13 | 新增 §10 安全整改与待办（P0/P1/P2） | 审核报告行动清单 |


---

## 附录：NCCL per-size tuner（Stage B）部署记录（2026-08-16 追加）

> 状态：**未上线**。容器验证通过，生产部署被 GLIBC 兼容性阻断，已回滚至 v3 原状。本节记录部署路径与回滚，供重编后执行。

### 背景
- Stage B 目标：per-size 协议 tuner（≤40KB allreduce 走 LL，>40KB 走 Simple），decode 小消息提速 21-31%（受控 A/B 实测，见 nccl-stageb-verification-2026-08-16.md）
- Stage B 库：/tmp/nccl-official-2307/build/lib/libnccl.so.2.30.7，md5 `9cdb26dc`（官方源 + v1 + v4 + enqueue.cc 双带 tuner）
- git commit `02449f6` / tag `nccl-2307-stageB-v1` / 归档 <INSTALL_DIR>/backup/nccl-official-2307-stageB-20260816

### 部署步骤（方案 A，2026-08-16 执行后回滚）
1. 四机备份现网 v3 → `libnccl.so.2.30.7.bak-stageB`（md5 b7784b49）
2. 四机替换为 Stage B 库（md5 9cdb26dc），符号链接完整
3. 四机脚本（start_tp4_head.sh / start_tp4_worker.sh）移除 `NCCL_PROTO=Simple`（.bak-stageB-20260816 留档 + bash -n 通过）
4. 恢复顺序：worker（02/04/03）→ head（01）

### 阻断：GLIBC 不兼容
- 报错：`GLIBC_2.38 not found (required by /opt/nccl-ringonly/libnccl.so.2)`
- 原因：Stage B 库在 host（新工具链）编译，readelf 要求 GLIBC_2.38；生产镜像 `anemll/dspark-vllm-gx10:0.2.1-v026.0` glibc < 2.38；现网 v3 库最大 GLIBC_2.34
- LD_PRELOAD 机制在容器 bash 启动即失败

### 回滚（已完成，生产恢复原状）
- 四机库回滚至 v3（b7784b49）
- 四机脚本还原 NCCL_PROTO=Simple
- 重启 worker→head，健康确认：4 容器 Up(healthy) + /health 200 + /v1/models 200 + NCCL 日志（NET_PLUGIN=none + PROTO=Simple + 无 110）

### 重编后部署注意（后续执行）
- **必须**用兼容生产镜像 glibc（≤GLIBC_2.34）的工具链或直接在 dspark-vllm-gx10 容器内重编 Stage B 库
- 重编后容器复验（受控 A/B）→ 生产窗口按上述步骤部署
- env 要求：移除 `NCCL_PROTO=Simple`；保留 `NCCL_ALGO=RING`、`NCCL_NET_PLUGIN=none`（tuner 生效前提）；可选 `NCCL_TUNER_THRESHOLD=40960`；无需新增 `NCCL_TUNER_PLUGIN=none`（NET_PLUGIN=none 已使 else 分支可达）
- 可选加固：PerSizeTuner 补丁从 else 分支移到 if/else 双分支（防未来改 NET_PLUGIN 静默失效）


### 更新（2026-08-16 GLIBC 修复重编后）
- **重编成功**：Stage B 库已用生产镜像 `anemll/dspark-vllm-gx10:0.2.1-v026.0` 重编（命令见 README/ADR-015 S1.10），新库 md5 `3d9cf539d2ed269ccaa49a46a98d7eb0`，GLIBC_MAX=2.34，生产镜像 LD_PRELOAD 加载验证通过。
- **当前状态**：生产仍为 v3（b7784b49 + NCCL_PROTO=Simple），healthy。Stage B 库待生产窗口容器 A/B 复验（小消息 LL 21-31%、368KB 持平）。
- **上线前必做**：容器受控 A/B 复验通过 → 按上文"重编后部署注意"执行（四机备份 v3→部署 3d9cf539→移除 NCCL_PROTO=Simple→worker 先 02/04/03→head 01→健康确认 /v1/models + NCCL 日志）。
- **env**：移除 `NCCL_PROTO=Simple`；保留 `NCCL_ALGO=RING`、`NCCL_NET_PLUGIN=none`；可选 `NCCL_TUNER_THRESHOLD=40960`；无需 `NCCL_TUNER_PLUGIN=none`。

## Stage B 部署记录（2026-08-16 16:30 终态）

**当前生产 NCCL 配置基线（R14）**：
- 库：/opt/nccl-ringonly/libnccl.so.2.30.7（md5 3d9cf539，官方 2.30.7-1 + v1 环邻过滤 + v4 硬编码 per-peer 映射 + enqueue 双带 tuner）
- env：**无 NCCL_PROTO**（已移除，env 优先级高于 tuner）；NCCL_ALGO=RING / NCCL_NET_PLUGIN=none / MIN_CHANNELS=4 / MAX_CHANNELS=16 / BUFFSIZE=8388608
- tuner：PerSizeTuner（≤40KB→LL / >40KB→Simple，仅 allreduce）；可选 NCCL_TUNER_THRESHOLD=40960
- 说明：NCCL_IB_PEER_HCA 已弃用（v4 硬编码映射内置，库忽略该 env）

**验证结果**：全量 4 档端到端 0 错误；32K PR 2425（+1.2% vs MAX_CH16）/ TTFT 11.93s；131K PR 2200 / TTFT 52.6s；vs T1aM4 全面改善；🟢 放行保留。

**回滚**：库 .bak-stageB-prod-20260816（=v3 b7784b49）+ 脚本 .bak-stageB-prod-20260816（含 NCCL_PROTO=Simple），还原后重启即回滚。

## Stage B 加固更新（2026-08-16 17:20）

- 库更新：/opt/nccl-ringonly/libnccl.so.2.30.7 md5 **2be94172**（双分支加固版，含 SPCX 防御；替换 3d9cf539）
- 回滚：.bak-hardened-20260816（=3d9cf539）→ .bak-stageB-prod-20260816（=v3 b7784b49）
- env 不变：无 NCCL_PROTO / NET_PLUGIN=none / ALGO=Ring / MIN_CH4 / MAX_CH16 / BUFFSIZE 8M

---

## B1 通道数更新（2026-08-17）

> **生产基线变更**：`NCCL_MAX_NCHANNELS` 16→**4**（B1 窗口 A/B 胜出，2026-08-17 18:5x 固化）。**此前本文档及历史附录中所有 MAX_CH16 描述均为历史记录，当前生产以本节为准（已由 B1 取代）。** 库 2be94172 与其余 env 均未变，唯一变更 = MAX_NCHANNELS。

### 变更内容

| 项 | 变更前（R13/StageB 基线） | 变更后（B1） |
|---|---|---|
| `NCCL_MIN_NCHANNELS` | 4 | **4（不变）** |
| `NCCL_MAX_NCHANNELS` | 16 | **4** |
| 启动脚本 | head L116 / worker L121 `NCCL_MAX_NCHANNELS=16` | `NCCL_MAX_NCHANNELS=4` |
| 备份 | `.bak-ncclMAXCH16-20260816`（R13） | **`.bak-ncclB1`**（head @01 + worker @02/03/04） |
| 库 | 2be94172 | **2be94172（未变）** |

### 收益（B1 报告，nccl-ab-B-execution-report-2026-08-17.md）

- **nccl-tests**（4-rank 环网 avg µs）：112KB allreduce **126→83µs（-34%）**、224KB **160→86µs（-46%）**；14KB +2µs（噪声带）
- **端到端**：c1@131K PR 2180.75 / DE 104.07 / TTFT 52.4s（DE +4%）✅；c1@32K 2387.91 / 96.83 / 11.93s（持平）
- **机制**：368KB/16ch=23KB 分片 Simple 延迟不友好 → 4ch 分片更大（92KB）延迟更优；14KB LL 由 per-size tuner 保证（≤40KB→LL），不受通道数影响
- **关闭项**：B3（LL128）无净收益、B4（QPS）大消息劣化；B2（8ch）不稳定淘汰

### env 表更新（生产启动脚本 NCCL 通道行）

```bash
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4        # B1：16→4（2026-08-17）
```

### 回滚

还原 `.bak-ncclB1`（`start_tp4_head.sh.bak-ncclB1` @01 + `start_tp4_worker.sh.bak-ncclB1` @02/03/04）+ `start_tp4_cluster.sh`（~8min）即回 MAX_CH16。B1 配置启动收敛 ~6min、health 200、四机容器 healthy。

### 文档标注

- §1.2 R13 行、§5 补丁表中 MAX_CH16 相关描述、§附录 Stage B 各节均为**历史记录**，不做修改。
- 当前生产 NCCL env 基线：`ALGO=RING / NET_PLUGIN=none / MIN_CH4 / MAX_CH4（B1）/ BUFFSIZE 8M / TUNER_THRESHOLD 40960 / 无 NCCL_PROTO / 无 NCCL_IB_PEER_HCA`。
- 依据：`deliverables/engineering-assurance/nccl-ab-B-execution-report-2026-08-17.md` / ADR-015 S1.14 / `b1-compat-adjudication-criteria-architect-2026-08-17.md`。
