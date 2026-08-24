# vLLM TP=2 双机启动「卡 NCCL init」根因调查报告

**日期**：2026-08-06
**工作流**：事故调查（架构 + 代码双线）
**参与成员**：Archi（架构/网络/时序）/ Cody（vLLM 源码层）

---

## 📌 TL;DR

- **主因（H1）：启动顺序竞态**——当前 SOP 为 worker(rank1) 先启动、head(rank0) 晚 12s，但 vLLM 的 TCPStore(master, 25000) 由 **rank0 在 init_process_group 时才创建**；实测 worker 比 head 早 31s 到达 NCCL init → worker 连接时 store 未就绪 → join 失败进程退出 → head 永远等不到 rank1 而静默空转（挂 2h）
- **次因（H2）**：DGX Spark GB10 双机 TP=2 NCCL 死锁为**社区已知系统性类别**（NVIDIA forum #366127 / vLLM #33041），RoCE 与 TCP 均复现
- **次因（H3）**：IPv6 路径污染——store 绑 IPv6 双栈（CVE-2025-47277）+ head 防火墙静默 DROP IPv6 + `get_ip()` 的 IPv6 fallback（fdff:: ULA）
- **修复方向**：颠倒启动顺序（head 先、轮询 25000 就绪后 worker 后）+ 缩短超时快速失败 + 提升 NCCL 日志留证

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🔴 根因已定位（H1 主因 + H2/H3 次因），需修复实验确认 |
| 阻塞项 | 1（生产重启偶发失败，当前已恢复运行） |
| 关键行动项 | 5 条（颠倒顺序 / 缩短超时 / NCCL 日志 / SOP 固化 / 防 IPv6 污染） |

---

## 1️⃣ 现象与诊断素材

- 环境：2×DGX Spark（GB10 sm_121a），head <MGMT_OCTET>（RoCE <NODE_IP>）/ worker <MGMT_OCTET>，host 网络容器，vLLM 0.26.1.dev0（anemll 0.2.1-v026.0），TP=2 mp 后端
- **4 次双机重启 3 次卡死**（第 4 次成功）；卡死时 head 日志停在 `parallel_state.py:1615 world_size=2 rank=0 backend=nccl` 后无输出（无 ERROR），EngineCore 进程 ALIVE、8001 不监听、CPU 空闲（可挂 2h）
- worker 侧失败错误：`TCPStore recvValue failed remote=[fdff:ffff::a457:9e7c:a7eb]:17300`（IPv6 ULA，head 无此地址）/ `Connection closed by peer [<NODE_IP>]:18349`
- 已排除：TIME_WAIT 端口残留（store 用 SO_REUSEADDR）、RoCE 链路（双 HCA 计数 0 错误）、OS（Ubuntu 24.04 合规）

## 2️⃣ 根因分析

### H1（主因）启动顺序竞态 —— 时序证据链

| 时间 | 事件 |
|------|------|
| 00:04:14 | worker(rank1) 到达 NCCL init（`parallel_state.py:1615`） |
| 00:04:22 | worker TCPStore recvValue failed → 初始化失败退出 |
| 00:04:45 | head(rank0) 才到达 NCCL init → **永远等不到已死的 rank1** → 静默空转 |

- vLLM 事实（Cody 源码证据）：**rank0 在 `init_process_group` 时才创建 TCPStore server**（parallel_state.py:1677）；worker 作 client 连它
- 当前 SOP 为 **worker 先启动、head 晚 12s**——**顺序反了**；head 是 store/zmq 广播的 host，必须先就绪
- 成功的那次（第 4 次）为时序巧合（worker 初始化恰好慢于 head）
- 佐证：vLLM 维护者 youkaichao（issue #18634）："TCPStore server has shut down too early 通常意味着 rank0 进程死了"——即 rank0 未建好 store

### H2（次因）双 Spark GB10 NCCL 系统性死锁（社区已知）

- NVIDIA forum #366127：双 DGX Spark GB10 TP=2 在 channel 建立后首个 all-reduce 死锁，NCCL 2.27.7/2.28.8/2.29.7 全复现，vLLM + TRT-LLM 均中招
- vLLM #33041（Blackwell TP2 hang）：缓解 = BIOS 关 IOMMU/ACS 或 `NCCL_P2P_DISABLE=1 + --disable-custom-all-reduce`
- 我们 4 次 3 挂的特征与竞态性死锁吻合

### H3（次因）IPv6 路径污染

- store 绑定 `*:25000`（IPv6 双栈 `[::]`）——**CVE-2025-47277** 默认行为（fork 未应用修复）
- **head 防火墙（aicad-fw-25000 容器 iptables）静默 DROP IPv6**（仅放行 established+::1）→ 任何 IPv6 尝试静默挂死
- `get_ip()`（network_utils.py:33-72）IPv4 fallback 失败时走 **IPv6 UDP connect → 返回本机 IPv6 ULA**（fdff:ffff::a457:9e7c:a7eb）——若某子进程未继承 VLLM_HOST_IP 即触发
- 17300/18349 = fork coord store 动态随机端口（parallel.py:550 / core_client.py:1607），报错指向"不存在的 store 实例"= 陈旧/错配 store 纪元（H1 的次生症状）

## 3️⃣ 社区已知问题对照

| 来源 | 内容 | 对本环境 |
|------|------|---------|
| vLLM #18634/#26769/#30579 | `TCPStore server has shut down too early` = rank0 进程死，HeartbeatMonitor 症状 | H1 佐证 |
| NVIDIA forum #366127 | 双 DGX Spark GB10 TP2 NCCL 全版本 hang（vLLM+TRT-LLM） | H2 佐证 |
| vLLM #33041 | Blackwell TP2：关 IOMMU/ACS 或 NCCL_P2P_DISABLE=1 修复 | H2 缓解参考 |
| route179.dev | 双 Spark 推荐 mp 后端 + leader 先就绪再起 worker | H1 修复参考 |

## 4️⃣ 修复建议（按优先级）

### 短期（立即实施）
| # | 项 | 内容 |
|---|-----|------|
| 1 | **颠倒启动顺序** | **先 head(rank0) → 轮询 `nc -z <NODE_IP> 25000` 就绪 → 再 worker(rank1)**（head 脚本完成 store 创建后 worker 才启）——H1 直接消解 |
| 2 | **缩短超时** | 容器加 `--distributed-timeout-seconds 300` + `VLLM_ENGINE_READY_TIMEOUT_S=600`——卡死 5 分钟快速失败（不静默挂 2h） |
| 3 | **NCCL 日志留证** | `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=~/vllm-logs/nccl-%h.log`（当前 WARN 无证据） |
| 4 | **SOP 固化** | 绝不单边重建（kill 后必须双机 worker→head 重来）；重启前确认 25000 无残留 |
| 5 | **防 IPv6 污染** | 容器确认所有子进程继承 `VLLM_HOST_IP=10.100.136.x`；评估 store 绑定指定接口（修 CVE-2025-47277 暴露面）或放宽 head IPv6 DROP 使异常快速失败 |

### 治本（跟踪/实验）
- 跟踪 NVIDIA forum #366127 / vLLM #33041 / Anemll 更新（GB10 双机 NCCL 死锁需上游修复）
- 若 H2 复现：实验 `NCCL_P2P_DISABLE=1 + --disable-custom-all-reduce`（#33041 已验证有效）
- 上游 patch 方向：mp executor rank1 连 master store 失败应显式报错退避重试；MessageQueue connect_ip 强制用 master_addr（不用 get_ip IPv6 fallback）

## 5️⃣ 验证实验（确认 H1/H2/H3 占比）

1. 颠倒顺序重启 2-3 次，统计成功率（预期显著提升 → H1 坐实）
2. 卡死时 `ss -ltnp | grep 25000`：LISTEN → 卡 nccl barrier 等 rank1（H2 方向）；不监听 → store 未建（H1 方向）
3. `NCCL_DEBUG=INFO` 重跑定位 worker 死前最后一步
4. 双机 `ip -6 addr` 全量核对 fdff:: 归属 + worker `nc -zv <NODE_IP> 25000`

## 📚 数据来源

- Archi（架构）：端口/TIME_WAIT/store 双栈/防火墙 DROP/RoCE/avahi 解析实测 + NVIDIA forum #366127 + CVE-2025-47277
- Cody（代码）：vllm main≈v0.26 源码逐行（parallel_state.py:1604-1683、multiproc_executor.py:133-135、shm_broadcast.py:528、network_utils.py:33-72、parallel.py:317-325/586-611、engine/utils.py:1206）+ vLLM #18634/#33041 社区对照
- 参考：github.com/vllm-project/vllm/issues/18634,33041；forums.developer.nvidia.com/t/366127；CVE-2025-47277

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。修复实验需重启生产容器（短中断），建议排维护窗口执行。
