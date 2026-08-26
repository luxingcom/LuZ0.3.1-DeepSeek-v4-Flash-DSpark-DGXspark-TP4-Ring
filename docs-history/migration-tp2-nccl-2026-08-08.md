# DGX Spark 四机集群 TP2 恢复 + NCCL 2.30.7 升级综合报告

**日期**：2026-08-08
**工作流**：部署迁移（工作流 2 变体：TP2 部署 + 环境升级 + embed 迁移编排）
**参与成员**：Zhen（主理人·编排与执行）、Rex（SRE 分诊）、Archi（架构 ADR）、Tessa（测试验收）、Cody（安全审查）、Docu（文档）
**关联文档**：`followup-actions-cluster-2026-08-07.md`（P0 清单源）、`memory-cleanup-plan-2026-08-07.md`（记忆清理）、`runbook-dspark-vllm-2026-08-06.md`（生产 Runbook v1.1，Docu 更新 v1.2 中）

---

## 📌 TL;DR（执行摘要）

- **P0 完成**：58+60 TP2 生产恢复成功（head-first 双机重启，/health 200，推理正常），根因=8/6 H1 启动顺序竞态复发 + 单边重建残留 + 双镜像 digest 不一致
- **P2 完成**：NCCL 2.30.7 升级成功（LD_LIBRARY_PATH 前插 pip nccl 目录，双端运行时验证 0x59df），利用新版 NCCL（PXN 死锁修复/RoCE LAG 负载均衡）改善多机互联
- **P1 通过**：双镜像（34.2G head + 21.6G worker）vLLM 版本串一致，gloo "128 vs 8" 归因启动竞态而非版本差异
- **P3 进行中**：embed 迁移 anemll（<MGMT_OCTET>/<MGMT_OCTET>），<MGMT_OCTET>/<MGMT_OCTET> 保留 vllm-gb10；镜像拉取中，对拍验证待执行
- **P4 待门禁**：清理 vllm-gb10 需 embed 迁移完成 + 无引用扫描 + 24h 观察窗口（Cody Request Changes）
- 严重度分布：🔴 0（已解决）/ 🟠 1（<MGMT_OCTET>/<MGMT_OCTET> dockerd 脏缓存）/ 🟡 2（口令轮换、embed 内存互斥）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（核心目标已达成，P3/P4 待完成） |
| 阻塞项数量 | 1（P4 清理门禁：embed 迁移未完成） |
| 关键行动项 | 5 条（见行动清单） |
| 建议下一步 | P3 embed 迁移对拍 → 24h 稳定观察 → P4 条件清理 |

---

## 🚨 事故时间线（58+60 TP2 停机恢复）

| 时间 | 事件 |
|------|------|
| 08-07 14:31Z | <MGMT_OCTET> worker 先启（违反 SOP） |
| 08-07 15:39-15:45Z | <MGMT_OCTET> head 三次重建（单边重建，等 rank1 join 300s 超时循环） |
| 08-07 15:45Z | head Restarts=9 循环重启，worker Broken pipe 失联，LLM 生产停机（SEV1） |
| 08-07 23:41Z | 主理人接管，四机现状采集 |
| 08-07 23:50Z | Rex 分诊确认：H1 竞态 + 镜像不一致 + 脏 socket |
| 08-08 00:00Z | 34.2G push registry + 双机脚本 IMG 改本地源 |
| 08-08 00:05Z | head-first 双机重启，TP2 恢复（/health 200，推理 "1+1"→"2"） |
| 08-08 00:15Z | P2 NCCL 2.30.7 升级（LD_LIBRARY_PATH 前插），双端验证 0x59df |
| 08-08 00:20Z | 并发 3 推理稳定，Restarts=0 |

---

## 🔍 根因分析（Rex 分诊 + 主理人实测）

### 主因：8/6 H1 启动顺序竞态复发
- worker(<MGMT_OCTET>) 先启、head(<MGMT_OCTET>) 后启，违反 8/6 固化 SOP（head 先→轮询 25000→worker 后）
- vLLM TCPStore(25000) 由 rank0 在 init_process_group 时才创建；worker 先启必失败
- head 单边重建 3 次，每次等 rank1 join 300s 超时→退出→restart 循环（Restarts=9）

### 次因：双镜像 digest 不一致
- head=34.2G（e100ddad568a）/ worker=21.6G（9ea563a724d4），同 tag 不同构建
- 曾现 gloo "op.preamble.length<=op.nbytes 128 vs 8" collective mismatch
- **实测推翻**：双镜像 vLLM 版本串完全一致（0.26.1.dev0+gd3d3b2cca.d20260805 / torch 2.11.0+cu130）→ mismatch 实为启动竞态时序问题，非版本差异

### 网络证据
- head TCPStore *:25000 LISTEN 但仅 rank0 自连，无 worker 连接
- worker 表内 SYN_SENT 打到 <NODE_IP>:25000(WiFi) 被防火墙 DROP（脏 socket 残留）
- overlay <NODE_IP>:25000 实测可达（主链路通）

---

## ✅ P2 NCCL 2.30.7 升级详情

### 背景
- 用户需求：升级生产环境利用新版 NCCL 改善多机互联效率
- 初始基线：双端运行时 NCCL 2.28.9（系统 lib 优先加载）

### 关键发现（实测）
- pip 包 `nvidia-nccl-cu13==2.30.7` 已安装，但被 LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu 遮蔽（系统 2.28.9 优先）
- `torch.cuda.nccl.version()` 返回 (2,28,9) 是**编译期常量**不可信；须用 `ncclGetVersion` + dladdr 实测运行时版本

### 升级方案（Archi 裁决方案 A：零构建）
```bash
# start_head_v026r.sh / start_worker_v026r.sh 修改（备份 .bak-20260807-nccl）：
# 原: -e 'LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu'
# 改: -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu'
```

### 验证结果
- <MGMT_OCTET> worker：ncclGetVersion=0x59df=**2.30.7**，dladdr=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 ✅
- <MGMT_OCTET> head：同 ✅
- TP2 并发 3 推理稳定（7+1=8/7+2=9/7+3=10），Restarts=0 ✅

### 安全复核（Cody）
- LD_LIBRARY_PATH 前插遮蔽风险：pip nccl 目录需核实仅含 libnccl 系（待执行）
- 作用域限容器/目标进程，勿写全局 profile
- 覆盖 dockerd/容器重启后复验 0x59df（防环境变量丢失）

---

## 🏗️ 架构决策（Archi ADR 摘要）

| ADR | 决策 | 状态 |
|-----|------|------|
| ADR-001 | 运行基线=anemll 0.2.1（head 34.2G + worker 21.6G 混搭，版本一致） | ✅ Accepted（G1 通过） |
| ADR-002 | cu132 路径=LD_LIBRARY_PATH 前插 pip nccl（零构建），非整栈换镜像 | ✅ Accepted（P2 完成） |
| ADR-003 | 55+59 Wi-Fi TP2 不可行（RTT~103ms），保持 embed-only | ✅ Accepted |
| ADR-004 | embed 迁移 anemll：<MGMT_OCTET>/<MGMT_OCTET> 迁移、<MGMT_OCTET>/<MGMT_OCTET> 保留 vllm-gb10（防异构漂移） | ✅ Accepted（P3 执行中） |
| ADR-005 | vllm-gb10 条件清理（embed 迁移完成 + 无引用 + 24h 观察） | ⏳ Proposed（门禁未过） |

---

## 🧪 测试验收计划（Tessa）

### P3 embed 迁移（门禁 A）
- dim=1024：同句集 100 条 cos ≥0.99（先归一化）
- 性能：batch16 ≥425 条/s（基线 472×90%）
- 内存互斥：LLM 107G + embed ≤14G（--gpu-memory-utilization 0.10），OOM 自动停 embed 保 LLM
- litellm：4 deployment 12 连发全 200，切换后保留 8020 容器观察 24h

### P4 清理门禁（Cody Request Changes）
1. TP2 稳定：health 200 ✅ + ≥24h 观察窗口无回退
2. embed 迁移 anemll：**未完成 ❌（硬门禁）**
3. 无引用扫描：docker ps + 脚本/config digest 引用全节点扫描 **未执行 ❌**

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | P3 embed 迁移：<MGMT_OCTET> 起 anemll embed 8021 测试容器 + golden 对拍（cos≥0.99） | Zhen+Tessa | P0 | 镜像拉取后 30min |
| 2 | <MGMT_OCTET>/<MGMT_OCTET> 内存互斥验证：LLM 107G + embed ≤14G 共存无 OOM | Zhen | P0 | 对拍通过后 |
| 3 | litellm 切换 8020→8021 + 12 连发验证 + 24h 观察窗 | Zhen+Tessa | P1 | 对拍通过后 1h |
| 4 | NCCL 遮蔽安全复核：pip nccl 目录内容 + 重启后复验 0x59df | Cody | P1 | 24h 内 |
| 5 | <MGMT_OCTET>/<MGMT_OCTET> dockerd 脏缓存修复（维护窗口重启 + by-digest 复验） | Rex | P1 | 下次维护窗口 |
| 6 | P4 清理：门禁全绿后定向 rmi vllm-gb10（保留 anemll 双 tag） | Zhen+Cody | P2 | embed 稳定 48h 后 |
| 7 | 口令轮换 + sshd 禁密码（全集群同密码 <PASSWORD> 风险） | Zhen+Cody | P0 | 1-2 天 |

---

## ⚠️ 待完善 / 已知局限

- **<MGMT_OCTET>/<MGMT_OCTET> dockerd 存储脏缓存**（pull 报 up-to-date 但无该 digest 镜像，疑似 containerd 快照元数据损坏）——需维护窗口重启 dockerd，勿在 TP2 运行期动 <MGMT_OCTET>
- **口令轮换 P0**：全集群共用密码 <PASSWORD> + 记忆已脱敏但用户对话暴露过；密码已在 8/7 报告列为 P0
- **embed 内存互斥临界**：<MGMT_OCTET> 余 14GiB，embed 预算 ≤14G 为临界值，OOM 自动停 embed 需实现
- **NCCL 遮蔽风险**：LD_LIBRARY_PATH 前插 pip 目录的长期安全性待 Cody 复核（目录内容核查）
- **registry 34.2G 覆盖 21.6G tag**：8/7 报告"registry 指向 21.6G"认知已过时，现指向 34.2G
- 55+59 仅 Wi-Fi：TP2 需有线接线（ConnectX-7 已存在），硬件前置

---

## 📚 数据来源 & 成员产出索引

- **Rex（SRE）**：SEV1 分诊报告（根因 3 因叠加 + 处置步骤 + 检查清单）；P2 方案 A 决策；<MGMT_OCTET> 镜像同步决策（B 混搭 + A 并行 + C 兜底）
- **Archi（架构师）**：ADR-001~005 + P0-P4 分阶段计划 + 门禁 G0-G4；P3 embed 拓扑补充（<MGMT_OCTET>/<MGMT_OCTET> 迁移）；P4 拍板（vllm-gb10 保留 + 34.2G 有条件 push）
- **Tessa（测试专家）**：5 项验收门禁（A/B/C 分级）；P3 5 阶段方案（docker run 命令 + golden 对拍脚本 + litellm 切换验证）
- **Cody（代码审查师）**：安全审查 4 项🔴（镜像边界/口令轮换/明文密码/registry 认证）；P4 清理方案（P4清理方案_vllm镜像_ADR005_20260807.md）；NCCL 遮蔽风险复核
- **Docu（技术文档师）**：memory-cleanup-plan-2026-08-07.md（记忆清理）；runbook v1.2 更新中
- **Zhen（主理人）**：四机现状采集、P0 执行（head-first 重启）、P2 执行（LD_LIBRARY_PATH 修改 + 双端验证）、镜像统一（34.2G push registry）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
> ⚠️ 重点提醒：P4 清理 vllm-gb10 前必须 embed 迁移完成 + 无引用扫描 + 24h 观察窗口；anemll 0.2.1 双 tag（34.2G/21.6G）+ registry 三处保留作不可再生资产。

---

## 🆕 追加：55/59 TP2 互联配置完成（2026-08-08 01:00）

### 用户关键信息补充
- 55/59 已物理接通 200G RoCE 直连（此前 ADR-003"Wi-Fi 不可 TP2"判断已过时）
- 用户要求：对照 58/60 修复 55/59 内网配置 → 测试 embed 在 TP2 下表现（防 OOM）→ 四机互联后 512G 内存可 TP4

### 执行成果
1. **网络配置**（参照 <MGMT_OCTET> 样板 /etc/netplan/99-nvidia-sync-cluster.yaml）：
   - <MGMT_OCTET> → <NODE_IP>/24 + <NODE_IP>/24（mtu 9000）
   - <MGMT_OCTET> → <NODE_IP>/24 + <NODE_IP>/24（mtu 9000）
2. **修复坑位**：<MGMT_OCTET>/<MGMT_OCTET> 各有损坏的 90-NM-2edf06d6.yaml（全 NUL 字节 715B）导致 netplan 解析失败 → 移走 .corrupt.bak
3. **验证通过**：jumbo ping（-s 8972）0% 丢包，RTT 0.6-0.8ms（vs Wi-Fi 103ms，提升 130 倍）；邻居表 REACHABLE
4. **镜像同步**：<MGMT_OCTET>/<MGMT_OCTET> 均有 anemll 0.2.1（21.6G 9ea563a724d4，版本串一致）

### Archi 架构裁决（补充）
- **IP 规划**：<RING_SUBNET> 给 55↔59，<RING_SUBNET> 预留四机全互连（TP4 铺路）
- **embed TP2 澄清**：Qwen3-Embedding-0.6B（1.2G）单卡绰绰有余，TP2 无内存收益且损 HA → **生产 embed 维持 4×单卡**；TP2 仅作互联通路演练（为未来 LLM TP2/TP4 预热）；防 OOM 靠单卡显存预算 + OOM 停 embed 脚本
- **ADR-003 修正**：55+59 可通过 RoCE 直连跑 TP2（原"Wi-Fi 不可行"已不成立）

### 执行中发现的环境障碍（待 Archi 定夺）
1. **anemll 0.2.1 镜像无 ray**（import ray → ModuleNotFoundError）→ ray-head/ray-worker 方案不可行
2. **anemll 0.2.1 不支持 --task embed**（实测 unrecognized arguments）→ embed 模型启动参数需调整
3. 待 Archi 确认：改 mp executor（与 58/60 一致）+ embed 参数调整，或降级用 vllm-gb10 做 TP2 互联演练

---

## 🆕 追加：55/59 TP2 embed 验证成功（2026-08-08 01:30）

### 执行成果
1. **TP2 embed 部署成功**：anemll 0.2.1 + mp executor（<MGMT_OCTET> rank0 + <MGMT_OCTET> rank1，master <NODE_IP>:25055），`/health 200` + `/v1/embeddings` 正常
2. **golden 对拍 PASS**：TP2(8021) vs 单卡(8020) 6 组向量，**cos_min=0.9998 / cos_mean=0.9999**，dim 均 1024——TP2 分布式推理与单卡完全一致
3. **内存分摊**：TP2 每机 ~2.8GiB（vs 单卡 3.1GiB）——TP 分摊显存验证有效

### 关键环境事实（踩坑记录）
1. **vllm-gb10 的 mp executor 有 follower bug**：`collective_rpc should not be called on follower node`（multiproc_executor.py:355）——正是 8/7 记录"新镜像 mp 多机 KV broadcast 不支持"；**anemll fork 已修复**（58/60 与 55/59 均验证）
2. **vllm-gb10 不支持 --task embed**（实测 unrecognized arguments）；anemll 无 --task 自动识别 embedding 架构（pooler_config 正确初始化）
3. **TP2 需 GLOO_SOCKET_IFNAME + VLLM_HOST_IP** 防止 gloo 回环 127.0.0.1 连接失败
4. vllm-gb10 镜像默认 entrypoint=null（cmd=["bash"]），必须显式 `--entrypoint vllm`

### 架构确认（Archi）
- ADR-003 修正：55+59 可通过 200G RoCE 跑 TP2（RTT 0.8ms，NCCL all_reduce PASS）
- embed TP2 演练通过 → 生产 embed 维持 4×单卡（TP2 无内存收益且损 HA）
- TP2 能力留作未来 LLM TP2/TP4 复用（四机互联 512G 内存，IP 段 <RING_SUBNET> 预留）

---

## 🆕 第三阶段追加：embed 验证 + 性能基线 + P1/P4 规划（2026-08-08 08:30）

### ✅ 完成项
1. **<MGMT_OCTET> anemll embed 单卡验证通过（ADR-004 待验项）**：8022 端口（util 0.10），health 200 + golden 对拍 vs <MGMT_OCTET>:8020 cos=0.9999（3 组 dim 1024 一致）；LLM+embed 共存 117G/121G 无 OOM，TP2 推理无回退
2. **litellm 3 端点 embed 池落地**：<MGMT_OCTET>(vllm-gb10 8020) + <MGMT_OCTET>(vllm-gb10 8020) + <MGMT_OCTET>(anemll 8022)，经 4000 转发验证通过（config 备份 .bak-20260808-embed58）
3. **TP2 性能基线补测**：c1/512=32.5 t/s、**c5/512=81.7 t/s**（8/5 基线 80.8-96.5 ✅达标，NCCL 2.30.7 无回退）、c1/8192=19.5-22.5 t/s（prefill 稀释 + embed 并存内存影响，Tessa 待专业解读）

### ❌ <MGMT_OCTET> embed 失败（内存临界，待补位）
- 根因：LLM head 占 107G，GPU free 14G < embed 需 12.16G（util 0.10），util 0.09 仍 CUDA OOM
- 主杠杆（Tessa）：收敛 <MGMT_OCTET> LLM util 到 0.80（~97G）释放 ~10G，P4 清理 + head 重启窗口补位
- 补位顺序：TP2 基线 → P1(<MGMT_OCTET> 重启+recreate util 0.80) → sanity → P4 → embed 补位 → TP2 全量回归

### 📋 P1/P4 规划（Rex + Cody 产出）
- **P1 dockerd**：诊断先行（内存态 vs 磁盘态），凌晨窗口 <MGMT_OCTET>→<MGMT_OCTET>，不做 prune -af；修复后 <MGMT_OCTET> 拉 34.2G 建第二份回滚锚点
- **P4 清理**：门禁收窄 <MGMT_OCTET>/<MGMT_OCTET> 定向清理（vllm-gb10 5a2a5e99a5a6 18.9G + ac38a938a8d5 19.2G ≈ 38G）；registry vllm-gb10 暂保留；6 组无引用扫描命令已备；24h 观察窗维持

### 🔑 关键环境事实
- <MGMT_OCTET> SSH 管理网口（<NODE_IP>）超时，但 RoCE 内网（<NODE_IP>）正常——embed 服务在线（8020=200）
- <MGMT_OCTET> embed 模型曾空目录（从未同步），已从 <MGMT_OCTET> rsync 1.2G 补齐
- GB10 统一内存：LLM head 107G + embed 12G 临界（<MGMT_OCTET> 余 4G 可、<MGMT_OCTET> 余 2G 不可）

---

## 🆕 追加：突发事故处置（<MGMT_OCTET> 系统重启 + TP2 恢复 + Grafana 4 台修复）（2026-08-08 09:00）

### 🔴 事故概述
<MGMT_OCTET> 系统意外重启 → 管理网/RoCE 全断 + TP2 中断（8001 health 000）。容器自动恢复但 worker/embed 挂起。

### 恢复链（完整记录）
1. **worker Exited(127)**：/tmp/env-e-build/nvcc_wrapper.py 被误建为**目录**（重启后 /tmp 清理 + 进程重建错误类型）→ 从 <MGMT_OCTET> base64 传输正确文件（1710B）
2. **head 重建三连坑**：
   - 缺模型挂载 → HFValidationError('/models') → 补 `-v /home/<USER>/models/deepseek-v4-flash-0731:/models:ro`
   - 缺 nvcc_wrapper 挂载 → deepgemm JIT Assertion → 补挂载
   - **NCCL ibv_modify_qp failed 61（GID index 3 空）** → head 加 `NCCL_IB_GID_INDEX=2`（<MGMT_OCTET> 重启后 GID3 全零，GID2=<NODE_IP> 有效）
3. **Gloo Connection closed by peer**（时序竞争）→ head-first 严格双停双启 → TP2 恢复（8001=200，推理 3+4=7）

### ✅ Grafana 4 台数据修复（用户任务 2 部分完成）
- <MGMT_OCTET> Wi-Fi 重连成功（nmcli connection up 珉珉家 UUID）→ <NODE_IP> 管理网恢复
- Prometheus target 改回**管理网直采**（撤销 <MGMT_OCTET> 代理方案）→ node/dcgm 4 台 8/8 up

### 📌 用户约束确认（已执行）
- 内网(RoCE)仅模型分发/多机互联/环境同步；**数据采集走管理网**
- 断联优先 Wi-Fi 重连（<MGMT_OCTET> 已执行成功）
- <MGMT_OCTET> 管理网走有线 enP7s7（<NODE_IP>），Wi-Fi 不影响

### 🔑 关键坑位（待 Docu 写 Runbook）
- <MGMT_OCTET> 系统重启后 RoCE GID index 3 为空 → head 需 `NCCL_IB_GID_INDEX=2`（已持久化）
- head/worker 完整挂载清单：模型 /models:ro + nvcc_wrapper + vllm-cache + tilelang-cache + vllm-logs
- TP2 恢复必须 head-first 严格时序（head 先起等 worker）

---

## 🆕 第四阶段：容错加固 + Grafana 修订 + 服务清理（2026-08-08 12:30）

### ✅ 完成项
1. **/tmp 挂载迁出（核心加固）**：nvcc_wrapper.py + vllm-envc-cache → <INSTALL_DIR>/envs/（容器内路径不变，宿主源持久化）——重启不再丢失挂载源
2. **目录规范**：四机 <INSTALL_DIR>/{models,envs,scripts,configs,cache,logs,backup,docs}；models 软链统一入口
3. **文件调用索引**：file-registry-4node-2026-08-08.md（30 条 + 重启 SOP + 故障速查）→ 四机 docs/file-registry.md
4. **脚本体系**：start_head_v026r.sh 加固（GID_INDEX 3→2 修复今晨故障根因、挂载持久化）；**start_worker_v026r.sh 新建**（此前缺失）；start_v026r_cluster.sh **v2.0**（幂等清理+端口/挂载预检+trap 诊断+双阶段健康）；systemd vllm-cluster.service（oneshot, enable）
5. **桌面服务关闭**：<MGMT_OCTET>/<MGMT_OCTET> gdm+gnome-remote-desktop、<MGMT_OCTET> gnome-remote-desktop → disable + multi-user
6. **NVIDIA 服务分析**：dgx-dashboard 不占 GPU 保留；nvidia-sync 手动 CLI 无常驻服务，干扰已隔离
7. **Grafana 指标修订（provisioning 方式，零口令依赖）**：
   - vllm-realtime 104/105：解码速度 Decode t/s / 预填充速度 Prefill t/s（标题+表达式 `sum by (node, model_name) (rate(vllm:generation_tokens_total/prompt_tokens_total{job="vllm"}[$__rate_interval]))`）
   - vllm-dspark-cluster：5 个吞吐 panel（decode/prefill/cached 三式）
   - Tessa 四面板复核通过（vllm-realtime v19, provisioning 生效）
8. **Prometheus**：预加 <MGMT_OCTET>:8001 target（组 B，node=head-55, machine=node0X）

### ✅ SRE 验收（Rex 清单）
A 挂载持久化 0 个 /tmp ✅ | C 桌面关闭 ✅ | D systemd ✅ | E GID_INDEX=2 双端 ✅ | F 三脚本语法 ✅ | G head-60 up ✅（head-55 待组 B）
B nvidia-sync 守卫 + 自愈冒烟 → 后续窗口

### 🔄 进行中
- A 组（58+60）benchmark：TP2 已就绪（8001=200 推理 8*7=56），等 Tessa bench 脚本
- 组 B（55+59）TP2 部署：待 A 组 benchmark 完成
