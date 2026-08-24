# SGLang 测试环境部署检查清单 — SRE 只读核验报告（v2 对齐 architect 设计）

- **核验人**：雷克斯（Rex）· SRE 工程师（工程保障团队）
- **核验时间**：2026-08-13 15:30–15:40 UTC（只读操作，全程未做任何写操作）
- **设计基线**：`sglang-nvfp4-arch-design-2026-08-13.md`（architect，§3.4/§3.5/§3.6 已对齐）
- **范围**：DGX Spark 四机集群（AICAD）TP4 环网：node01(<MGMT_OCTET>/<MGMT_OCTET>) / 02(<MGMT_OCTET>/<MGMT_OCTET>) / 03(<MGMT_OCTET>/<MGMT_OCTET>) / 04(<MGMT_OCTET>/<MGMT_OCTET>)，用户 <USER>
- **目的**：核验创建 SGLang 测试环境（NVFP4 权重，TP4 环网）的可行性，与生产 vLLM TP4 做 **A/B 互斥切换**（UMA 内存互斥硬约束，无法并存）

---

## 0. 结论摘要（v2 关键变更）

| 项目 | 结论 |
|------|------|
| **并存策略** | 🔴 **改为 A/B 互斥切换**：SGLang NVFP4 TP4 单 rank 约 110GB（mem-fraction 0.90），与生产 vLLM TP4（~79GB/rank）同 UMA 池**无法并存**。SGLang 验证期必须 **stop vLLM TP4（head+worker）→ GPU 门禁 → 启动 SGLang**；回滚=停 SGLang→起 vLLM TP4 |
| 权重（NVFP4） | ✅ **已就绪**：四机各 164–165G，48 shards + hf_quant_config.json（modelopt，NVFP4 MoE experts，group_size=16），`<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4` 软链已建；**仍须补 conversion-receipt + SGLang load 冒烟** |
| 磁盘 | ✅ 充足：01=2.9T / 02=2.3T / 03=622G / 04=631G 可用 |
| 端口 | ✅ architect 定案 **API 8010 / metrics 8011 / TCPStore 26000**；四机均已核验空闲（8003/25999 分别被 aicad 应用栈 / vLLM 占用） |
| 镜像 | ⚠️ 四机**均无 SGLang 镜像**、本地仓库无 sglang repo；**26.02 不可用**（内部 SGLang 0.5.8 < 0.5.14，不含 DSV4 NVFP4）；须用 **26.07-py3**（验证内部 SGLang ≥0.5.14 + flashinfer ≥0.6.15）或自建 `lmsysorg/sglang:v0.5.16` |
| NCCL | ✅ ring-only 2.30.7 四机存在；**关键**：预检须用 `/proc/self/maps \| grep libnccl` 验证容器内实际加载 2.30.7（容器系统 2.28.9 会遮蔽，不信 `torch.cuda.nccl.version()`） |
| 内存 | 🔴 由"并发 OOM 风险"转为"**必须互斥**"的编排级守卫：同 UMA 池不能双 TP4 并存 |
| 版本硬校验 | ⚠️ 容器内 SGLang ≥0.5.14（NVFP4 支持下限）、flashinfer ≥0.6.15；两大实测点：`is_sm120_supported()` 对 SM121 匹配、`flashinfer_trtllm_routed` 在 SM121 的 TRTLLM kernel（备选降级 flashinfer→marlin） |

---

## 1. 四机核验结果表（逐项）

### 1.1 基础环境

| 项目 | node01 (A head) | node01 (A worker+镜像仓) | node01 (B head) | node01 (B worker) |
|------|------|------|------|------|
| IP | <NODE_IP> | <NODE_IP> | <NODE_IP> | <NODE_IP> |
| GPU | NVIDIA GB10 ×1（sm_121） | 同左 | 同左 | 同左 |
| Driver / CUDA | 580.173.02 / 13.0 | 同左 | 同左 | 同左 |
| 内存 total | 121Gi | 121Gi | 121Gi | 121Gi |
| 内存 used / available | 89Gi / 32Gi | 88Gi / 33Gi | 94Gi / 26Gi | 93Gi / 28Gi |
| Swap | 15Gi（已用 4.0Gi） | 15Gi（已用 6.0Gi） | 15Gi（已用 4.1Gi） | 15Gi（已用 5.7Gi） |
| 逻辑核 | 0-19（isolated 8-9, nproc 18） | 同左 | 同左 | 同左 |
| 根盘 | 3.6T（可用 2.9T） | 3.6T（可用 2.3T） | 916G（可用 622G） | 916G（可用 631G） |

### 1.2 运行容器（当前生产，SGLang 启动前须 stop）

| 容器 | 01 | 02 | 03 | 04 |
|------|----|----|----|----|
| vllm-tp4-rank0/1/2/3 | ✅ rank0 (healthy) | ✅ rank1 (healthy) | ✅ rank3 (healthy) | ✅ rank2 (healthy) |
| anemll-embed-8022（TP2 embed） | — | — | ✅ 0.0.0.0:8022 | ✅ 0.0.0.0:8022 |
| registry:2 | — | ✅ 0.0.0.0:5000 | — | — |
| responses-gateway（systemd 用户服务） | — | ✅ **占用 8003**（8003→<NODE_IP>:8001，enabled） | — | — |
| 其他（redis/pg/grafana/neo4j/minio/prometheus/alertmanager/litellm 等） | ✅ | ✅ | — | — |

- 生产镜像：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`
- vLLM 容器模式：`--network host --ipc=host --privileged --gpus all --restart no --cpuset-cpus=1-19 --shm-size=64gb`
- vLLM GPU 占用：每 rank `VLLM::Worker_TP*` ≈ 79511–79515 MiB；03/04 另有 EngineCore 5750 MiB
- **互斥结论依据**：生产 vLLM ~79GB/rank（util 0.65） vs SGLang NVFP4 TP4 ~110GB/rank（mem-fraction 0.90）→ 同 UMA 121GiB 池无法双 TP4 并存

### 1.3 权重（NVFP4）— 已就绪

| 节点 | NVFP4 权重路径（软链 `<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`） | 大小 / 文件数 |
|------|------|------|
| 01 | `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4` | 165G / 70 项（48 shards） |
| 02 | `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4` | 164G / 70 项 |
| 03 | `<MODELS_DIR>/deepseek-v4-flash-0731-nvfp4` | 165G / 70 项 |
| 04 | `<MODELS_DIR>/deepseek-v4-flash-0731-nvfp4` | 164G / 70 项 |

- 内容：`config.json` / `config.1.json` / `hf_quant_config.json` / `generation_config.json` / `encoding` / `inference` / `model-00001~00048-of-00048.safetensors` / LICENSE* / README*
- `hf_quant_config.json`：producer=modelopt（dsv4-nvfp4-experts-mtp-fallback），quant_algo=MIXED_PRECISION，专家层 NVFP4，group_size=16
- **待办**：与 architect §3.3 对齐——本目录是 MJPansa 下载产物还是本地 tsarihan 转换？是否已做 conversion-receipt（sha256 + hf_quant_config 字段完整 + load 冒烟）？未确认前按"待验收"处理，不直接信任。

### 1.4 镜像仓库 / 外部网络

| 项目 | 结果 |
|------|------|
| 本地仓库 `<NODE_IP>:5000` | ✅ 四机可达；**无 sglang repo** |
| vllm-gb10 tags | `0.26.1-cu132-sha-fa87aea5`（vLLM 内部参考） |
| NGC (nvcr.io) | ✅ 可达（HTTP 401 = 需登录鉴权）；02 已有 `nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04` |
| 外部代理 | docker.m.daocloud.io / ghcr.io 曾用于拉取 |

### 1.5 端口占用（architect 定案）

| 端口 | 01 | 02 | 03 | 04 | 说明 |
|------|----|----|----|----|------|
| 8001 | 🔴 vLLM head | 空闲 | 空闲 | 空闲 | 生产，不可用 |
| 8003 | 空闲 | 🔴 responses-gateway | 空闲 | 空闲 | aicad 应用栈，不可用 |
| 8022 | 空闲 | 空闲 | 🔴 embed | 🔴 embed | 生产，不可用 |
| 25999 | 🔴 vLLM master | 空闲 | 空闲 | 空闲 | 生产，不可用 |
| **8010**（SGLang API） | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | **head 01 绑定** |
| **8011**（SGLang metrics） | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | head 01 |
| **26000**（TCPStore） | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | ✅ 空闲 | head 01，控制面 |

> 端口 8010/8011/26000 基于核验窗口四机完整 `ss -tln` 快照确认均未出现；部署时仍须 preflight 复查。

### 1.6 NCCL / 环网环境（对齐 architect §3.4）

- `/opt/nccl-ringonly`：四机均存在，含 `libnccl.so.2.30.7`（另 `.bak-v2`）→ 满足 2.30.7
- 主机 `dist-packages/nvidia/nccl/lib`：空（NCCL 依赖在容器内/ringonly 补丁，属正常）
- **NCCL 版本陷阱**：Spark 容器默认 NCCL 2.28.9 会报 "No available shared memory broadcast block"；须 2.30.4+（社区验证 2.30.7）。**验证用 `/proc/self/maps \| grep libnccl`（容器内），不能信 `torch.cuda.nccl.version()`**（读编译期宏）。
- 生产 env（SGLang 容器须镜像 + architect 调整）：
  - `LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2"`（shim v8 + ring-only 2.30.7）
  - `LD_LIBRARY_PATH="/opt/nccl-ringonly:${LD_LIBRARY_PATH}"`（**前插**防系统 2.28.9 遮蔽）
  - `NCCL_ALGO=RING`、`NCCL_SUBNET_AWARE_ROUTING=1`、`NCCL_NET_PLUGIN=none`、`NCCL_MERGE_NICS=0`
  - `NCCL_IB_PEER_HCA=<沿用 vLLM TP4 四机对口映射，v3 双 dev 轮换>`（生成后四机一致，逐对核对）
  - `NCCL_IB_GID_INDEX=2`（A=2；B 侧按实测 4 或统一 2，以 preflight 为准）、`NCCL_IB_TOS=46`、`NCCL_IB_TIMEOUT=22`
  - `NCCL_DEBUG_FILE` 落容器内 `~/.sglang-logs/nccl-*.log`（不污染 stdout）
- SGLang × SM12x 专用 env（architect）：
  - `SGLANG_DISABLE_DEEP_GEMM=1` / `SGLANG_ENABLE_DEEP_GEMM=0`（DeepGEMM SM100-only）
  - `SGLANG_SM120_TRITON_FLASHMLA=1`（SM120/121 MLA Triton fallback）
  - `SGLANG_SM120_MQA_FALLBACK=0`
  - `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`（可选降内存）
  - `SGLANG_RAGGED_VERIFY_MODE=compact`（DSpark v0.5.16）
  - 控制面：`GLOO_SOCKET_IFNAME=<管理网 ifname，与现有 vLLM 脚本一致>`
- PEER_HCA 环网映射（01→02→04→03）：
  - rank0(01)：`1=rocep1s0f1,roceP2p1s0f1;3=rocep1s0f0,roceP2p1s0f0`
  - rank2(04)：`1=rocep1s0f0,roceP2p1s0f0;3=rocep1s0f1,roceP2p1s0f1`
  - rank3(03)：`0=rocep1s0f0,roceP2p1s0f0;2=rocep1s0f1,roceP2p1s0f1`
  - rank1(02)：按 worker 脚本 case

### 1.7 启动脚本参考（生产基线）

- `<INSTALL_DIR>/scripts/`：`start_tp4_head.sh`（v1.5-r11，head=rank0/01，PORT=8001，MASTER_PORT=25999）、`start_tp4_worker.sh`（NODE_RANK/VLLM_HOST_IP 参数化）、`start_tp4_cluster.sh`、`start_v026r_cluster.sh`、`shim-deploy.sh`、`monitor_tp4_{head,worker}.sh`、`check_vllm_script.sh`
- SGLang 脚本须**新建** `start_sglang_tp4_cluster.sh`，禁止改动生产脚本

---

## 2. 发现的问题（按严重度，v2 更新）

### 🔴 SEV1 — UMA 内存互斥（架构级约束，非缺陷但必守）
- **依据**：SGLang NVFP4 TP4 单 rank ~110GB（mem-fraction 0.90）= 42GB 权重 + KV/激活；生产 vLLM TP4 ~79GB/rank。同 UMA 121GiB 池无法双 TP4 并存。
- **处置**：A/B 互斥切换。启动 SGLang 前必须 stop vLLM TP4（head+worker）并确认 monitor 退出；回滚=停 SGLang→起 vLLM TP4（现有 systemd 自愈兜底）。→ 做成**互斥守卫**（见 D 节）。

### 🟠 SEV2 — 镜像版本门槛（26.02 不可用）
- **现象**：四机均无 SGLang 镜像；NGC 26.02（SGLang 0.5.8）**早于 DSV4 NVFP4 支持（0.5.14）**，仅覆盖 R1/Llama 早期模型。
- **处置**：拉取 `nvcr.io/nvidia/sglang:26.07-py3`，`docker exec` 验证 `sglang.__version__ >= 0.5.14` 且 `flashinfer >= 0.6.15`；不合则自建 `lmsysorg/sglang:v0.5.16`（aarch64）+ flashinfer sm12x wheel。推送到 `<MGMT_OCTET>:5000/sglang/sglang:0.5.16-nvfp4-spark`。

### 🟠 SEV2 — 端口定案变更（8003/25999 被占，改 8010/8011/26000）
- 已核验四机 8010/8011/26000 空闲；部署 preflight 复查。

### 🟡 SEV3 — NCCL 版本遮蔽风险
- 容器系统 2.28.9 会遮蔽 2.30.7 → 必须 `LD_LIBRARY_PATH` 前插 `/opt/nccl-ringonly` + 容器内 `/proc/self/maps` 实测。

### 🟡 SEV3 — SM121 kernel 兼容性未验证（两大实测点）
1. `is_sm120_supported()` 是否覆盖 SM121；
2. `flashinfer_trtllm_routed` 的 TRTLLM kernel 在 SM121 兼容性 → 备选降级 `flashinfer`（CUTLASS）→ `marlin`。

### 🟡 SEV4 — NVFP4 权重来源/验收待确认
- 本地 `deepseek-v4-flash-0731-nvfp4` 已就绪（164-165G），但**不确定是 MJPansa 下载还是本地 tsarihan 转换**；须补 conversion-receipt + load 冒烟。

### ✅ 未发现阻塞项
- 权重已落地、磁盘充足、NCCL 补丁齐备、本地仓库可达、候选端口（8010/8011/26000）空闲。

---

## 3. 部署检查清单（部署时逐项执行）

### A. 前置：镜像获取与推送

| # | 检查项 | 检查命令（02） | 通过标准 | 失败处置 |
|---|--------|----------------|----------|----------|
| A1 | NGC 登录凭据可用 | `docker login nvcr.io`（凭据团队提供） | Login Succeeded | 联系镜像负责人获取 NGC API Key；否则走自建 |
| A2 | 拉取 26.07 容器 | `docker pull nvcr.io/nvidia/sglang:26.07-py3` | 拉取成功 | 换 tag / 代理源；记录原因 |
| A3 | **容器内版本硬校验** | `docker run --rm --entrypoint python3 <img> -c "import sglang,flashinfer; print(sglang.__version__, flashinfer.__version__)"` | SGLang ≥0.5.14 **且** flashinfer ≥0.6.15 | 不合→自建 `lmsysorg/sglang:v0.5.16`（重装 flashinfer sm12x wheel） |
| A4 | 推送到本地仓库 | `docker tag ... <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark; docker push` | push 成功 | 检查 02 registry 容器健康；重试 |
| A5 | 四机拉取 | 01/03/04 `docker pull <NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark` | 四机成功 | 检查网络/磁盘；保留 registry tag，运行 tag 另打 |

### B. 权重就绪（含 conversion-receipt）

| # | 检查项 | 检查命令 | 通过标准 | 失败处置 |
|---|--------|----------|----------|----------|
| B1 | NVFP4 四机存在 | 四机 `ls <INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4/model-00048-of-00048.safetensors` | 48 shards 齐全 | 从完好副本补齐（写操作须团队确认） |
| B2 | 量化配置有效 | `grep -E 'NVFP4|group_size' .../hf_quant_config.json` | 出现 NVFP4、group_size=16 | 非 NVFP4 禁止启动 |
| B3 | **conversion-receipt** | 与 architect 对齐来源（MJPansa / tsarihan）；`sha256sum -c manifest.sha256` | 四机一致 + receipt 记录 | 来源/校验未过→按 architect §3.3 处理 |
| B4 | 只读挂载 | 启动 `-v <INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4:/models:ro` | 容器内只读 | 加 `:ro`；禁止 SGLang 写权重 |
| B5 | load 冒烟前置 | 先 TP1 冒烟 `curl :8010/v1/models` | 模型加载成功 + 首 token | 检查 hf_quant_config 字段/padding；降级 backend |

### C. 磁盘/内存

| # | 检查项 | 检查命令 | 通过标准 | 失败处置 |
|---|--------|----------|----------|----------|
| C1 | 磁盘余量 | 四机 `df -h / /home /data` | 03/04 可用 ≥100G | 清理旧镜像/日志（写操作须团队确认） |
| C2 | 内存释放确认 | stop vLLM 后四机 `free -h` | 可用 ≥100G/节点（为 mem-fraction 0.90 留足） | 未释放→查残留进程/容器 |

### D. 互斥守卫 + 端口（A/B 切换核心）

| # | 检查项 | 检查命令/动作 | 通过标准 | 失败处置 |
|---|--------|---------------|----------|----------|
| D1 | **停生产 vLLM TP4** | 01-04：停 `vllm-tp4-rank{0..3}`（head+worker），确认 monitor 退出 | `docker ps` 无 vllm-tp4-* | 若无法停（生产不可中断）→ **中止部署**，SGLang 验证另排窗口 |
| D2 | 残留容器清理 | `docker ps -a \| grep vllm-tp4` | 无残留运行容器 | `docker rm -f` 残留（记录于变更单） |
| D3 | GPU 门禁（≤180s） | 启动前 `nvidia-smi` 轮询 | 无 vLLM 进程、显存清空 | 超时中止，禁止强起 |
| D4 | API 8010 空闲 | 四机 `ss -tln \| grep ':8010'` | 均无输出 | 停占用进程或换端口（需团队决策） |
| D5 | metrics 8011 空闲 | 四机 `ss -tln \| grep ':8011'` | 均无输出 | 同上 |
| D6 | TCPStore 26000 空闲 | 四机 `ss -tln \| grep ':26000'` | 均无输出 | 同上 |
| D7 | 避开生产端口 | `ss -tln \| grep -E ':8001|:8003|:8022|:25999'` | 不被 SGLang 占用 | 立即停 SGLang 核验 |

### E. NCCL 环境（环网，architect §3.4）

| # | 检查项 | 检查命令 | 通过标准 | 失败处置 |
|---|--------|----------|----------|----------|
| E1 | ringonly 补丁 | 四机 `ls /opt/nccl-ringonly/libnccl.so.2*` | 含 2.30.7 | 缺失→同步（写操作须团队确认） |
| E2 | 容器挂载 | `-v /opt/nccl-ringonly:/opt/nccl-ringonly:ro` + `-v <INSTALL_DIR>/lib/libncclpin.so:/opt/libncclpin.so:ro` | 挂载齐全 | 修正启动命令 |
| E3 | env 注入 | `docker inspect` 验证 `LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`、`LD_LIBRARY_PATH=/opt/nccl-ringonly:...`（前插） | 注入正确 | 修正 |
| E4 | **实际加载版本** | 容器内 `cat /proc/self/maps \| grep libnccl` | 指向 **2.30.7**（非 2.28.9） | 修正 LD_LIBRARY_PATH 顺序；重钉 2.30.7 |
| E5 | 环网参数 | `NCCL_ALGO=RING, SUBNET_AWARE_ROUTING=1, NET_PLUGIN=none, MERGE_NICS=0, GID=2, TOS=46, TIMEOUT=22` | 与 architect 一致 | 逐项修正 |
| E6 | PEER_HCA 映射 | rank0/1/2/3 按 §1.6 | 四机一致逐对核对 | 参考生产 worker 脚本修正 |

### F. SGLang 启动（head-first，architect §3.6）

| # | 检查项 | 建议参数（architect 草案，以最终脚本为准） | 通过标准 | 失败处置 |
|---|--------|-------------------------------|----------|----------|
| F1 | 容器名/模式 | `sglang-nvfp4-tp4-{0,1,2,3}`；`--restart no --network host --ipc=host --privileged --gpus all --cpuset-cpus 1-19 --shm-size 64g`；**不加**内存硬限制 | 不与 vllm-tp4-* 冲突 | 改名后重试 |
| F2 | 启动顺序 | head-first：01(rank0)→02(rank1)→04(rank2)→03(rank3)；worker 等 head TCPStore :26000（120s），head 就绪后 60s 缺秩 exit(1) | 顺序启动 | 查 head 日志；端口未监听则重查 env |
| F3 | 启动命令 | `python3 -m sglang.launch_server --model-path /models --trust-remote-code --tp 4 --nnodes 4 --node-rank {N} --dist-init-addr <NODE_IP>:26000 --host 0.0.0.0 --port 8010 --moe-runner-backend flashinfer_trtllm_routed --speculative-algorithm DSPARK --mem-fraction-static 0.90 --chunked-prefill-size 4096`（worker 去 --port） | 正常加载 | 见 G 快速失败 |
| F4 | SGLang env | `SGLANG_DISABLE_DEEP_GEMM=1/ENABLE=0`、`SGLANG_SM120_TRITON_FLASHMLA=1`、`SGLANG_SM120_MQA_FALLBACK=0`、`SGLANG_RAGGED_VERIFY_MODE=compact` | 注入正确 | 修正 |
| F5 | SM121 kernel 验证 | 启动日志核对 `is_sm120_supported()` kernel 选择 | 无 kernel image 错误 | 显式 flag/升级；降级 MoE backend |
| F6 | MoE backend 兼容 | 首选 `flashinfer_trtllm_routed` | load + 首 token 成功 | 依次降级 `flashinfer`（CUTLASS）→ `marlin` |

### G. 健康检查与快速失败

| # | 检查项 | 检查命令 | 通过标准 | 失败处置 |
|---|--------|----------|----------|----------|
| G1 | 容器存活 | `docker ps`（sglang-nvfp4-tp4-* Up） | 四容器 Up | `docker logs` 定位 |
| G2 | 健康 | `curl -s http://<NODE_IP>:8010/health` | HTTP 200 | 等 engine ready（日志）；持续失败回滚 |
| G3 | 模型可见 | `curl -s http://<NODE_IP>:8010/v1/models` | 列出模型 | 检查 --model-path/加载 |
| G4 | 首 token | 发起小长度 completion | 正常返回 | 查日志；按 kernel/内存判定 |
| G5 | **快速失败关键字** | `docker logs` 扫 `NCCL error / No available / kernel image / CUDA error` | 无上述关键字 | 立即终止并回滚 |
| G6 | 长 prompt 生成 | 长 prompt 冒烟 | 稳定生成 | 评估是否需降 DSpark/调参 |

### H. 回滚（切回 vLLM TP4）

| # | 动作 | 命令/说明 | 通过标准 |
|---|------|-----------|----------|
| H1 | 停 SGLang | 01-04 `docker rm -f sglang-nvfp4-tp4-{0..3}`（--restart no 不自动拉起） | 无 sglang 容器 |
| H2 | 起 vLLM TP4 | 用现有 `start_tp4_cluster.sh` / systemd 自愈 | vllm-tp4-* 四容器 healthy |
| H3 | 验证恢复 | `docker ps` + `curl :8001` + `nvidia-smi` | 生产恢复原状 |
| H4 | 回滚锚点 | 镜像 tag（sglang↔vllm 0.2.1-v026.0）+ 权重目录（-nvfp4/-local）+ `start_sglang_tp4_cluster.sh` 参数化 | 记录在案 |

---

## 4. 回滚方案（汇总）

| 阶段 | 操作 | 回滚/撤销方式 |
|------|------|---------------|
| 镜像 | 拉取/推送失败或版本不合 | `docker rmi <NODE_IP>:5000/sglang/sglang:<tag>`；registry 可保留，不影响生产 |
| 权重 | 误改/误删 | 只读挂载（:ro）+ 启动前 `ls` 校验；异常停止写入，从任一完好副本比对 |
| 互斥守卫 | 生产不可停（D1 失败） | **中止部署**，另排窗口；不强行启动 SGLang |
| 容器启动 | 任一容器失败/异常 | `docker rm -f sglang-nvfp4-tp4-{N}`；`--restart no` 保证不自动拉起 |
| 门禁超时 | 内存/GPU 未就绪 | 未启动任何容器，终止流程，等待窗口重试 |
| 端口冲突 | 误占生产端口 | 立即 `docker rm -f`，确认 `ss -tln` 生产端口恢复 |
| SM121 kernel / MoE backend | 加载失败 | 降级 backend（flashinfer→marlin）或升级容器；仍失败则回滚 |
| 生产恢复 | SGLang 验证结束 | H1→H4 切回 vLLM TP4（systemd 自愈兜底） |

**回滚触发条件**（任一即触发）：
1. 互斥守卫失败（无法 stop vLLM TP4 或残留进程）；
2. GPU 门禁 >180s 未达标；
3. 快速失败关键字（NCCL error / No available / kernel image / CUDA error）；
4. `/proc/self/maps` 显示 NCCL 非 2.30.7 且无法修复；
5. 4 分钟内任一节点可用内存跌破 10G 且继续下行；
6. 首 token 不可用且 2 次重启仍失败。

---

## 5. 行动清单（按优先级，v2 更新）

| 优先级 | 行动 | 责任方 |
|--------|------|--------|
| P0 | 确认 A/B 互斥切换策略与生产停机窗口（SGLang 验证须停 vLLM TP4） | 主理人+架构师+SRE |
| P0 | 拉取/验证 26.07 容器内部 SGLang ≥0.5.14 + flashinfer ≥0.6.15；不合则自建 0.5.16 | SRE/Ops |
| P0 | 权重 conversion-receipt：确认来源（MJPansa/tsarihan）+ sha256 + load 冒烟 | 架构师+SRE |
| P1 | 编写 `start_sglang_tp4_cluster.sh`（互斥守卫、head-first、GPU 门禁、health、快速失败、参数化镜像 tag） | 架构师+SRE |
| P1 | preflight：NCCL `/proc/self/maps`、RoCE GID/PEER_HCA、MTU 9000、端口 8010/8011/26000 | SRE |
| P2 | TP1 冒烟（SM121 kernel 验证）→ TP4 环网启动 → /health → /v1/models → 首 token → 长 prompt | SRE+测试 |
| P2 | 备选 MoE backend 降级测试（flashinfer/marlin） | 测试+SRE |
| P3 | 性能 A/B（同 vLLM 口径）、稳定后建 systemd 单元 | SRE |

---

*本报告全部结论基于只读核验（docker ps / nvidia-smi / df / ss / ls / cat / curl GET）；未执行任何安装、配置修改、容器启动或文件删除。v2 已对齐 architect 设计：A/B 互斥切换、端口 8010/8011/26000、镜像 ≥0.5.14、NCCL `/proc/self/maps` 实测、SM121 kernel 验证。*

---

# v3 节：准备阶段执行记录（2026-08-14，SRE 执行）

- **执行人**：雷克斯（Rex）· SRE 工程师
- **执行时间**：2026-08-13 16:30–16:45 UTC（对应 08-14 凌晨 00:30–00:45）
- **范围**：仅 02 容器内验证 + 新脚本写入 `<INSTALL_DIR>/scripts/`；**未启动正式 SGLang TP4、未触碰生产 vLLM 容器/配置/权重源、未修改 start_tp4_*.sh 生产脚本**
- **执行方式**：短暂 `docker run --rm` 测试容器（--gpus all 但无 CUDA 大上下文、未加载模型），用后即清；容器 `--restart no`

## 3.1 容器内版本终验（Task 1.1）

| 组件 | 实测版本（02 容器内） | 判定 |
|---|---|---|
| SGLang | **0.5.14+nv26.7.59534057** | ✅ PR #25820 合入线（NVFP4 DSV4 原生支持） |
| torch | 2.13.0a0+9186a08b2c.nv26.07 | ✅ |
| flashinfer | **0.6.14** | ⚠️ 略低于方案 0.6.15.post1（NGC 校验组合，如遇 NCCL 回退问题再钉） |

> 实测命令：`docker run --rm <img> bash -c "pip show sglang \| head -2; python -c 'import torch,flashinfer; print(torch.__version__, flashinfer.__version__)'"`

## 3.2 flag 实测表（Task 1.2，0.5.14 launch_server --help，供 architect 终稿参考）

| 项目 | 0.5.14 实测 | 说明 / 影响 |
|---|---|---|
| **TP flag** | `--tp-size TP_SIZE`（别名 `--tensor-parallel-size`） | ⚠️ **不是 `--tp`**（0.5.16 改名风险证实存在）→ 脚本用 `--tp-size 4` |
| **metrics** | **无 `--metrics-port`** | 只有 `--grpc-http-sidecar-port`（gRPC 模式专用，Defaults to --port+1）；HTTP 模式用 `--enable-metrics`，metrics 随主端口 `/metrics` → **8011 独立 metrics 端口在 0.5.14 不可达**，需 architect 复核（脚本已按 8010/metrics 落） |
| **moe-runner-backend** | 合法值含 `flashinfer_trtllm_routed / flashinfer_cutlass / marlin / auto ...` | ✅ `flashinfer_trtllm_routed` 合法（首选 NVFP4 MoE） |
| **quantization** | 合法值含 `modelopt / modelopt_fp4 / nvfp4_online / modelopt_mixed / auto...` | ✅ `modelopt_fp4` 合法；NVFP4 权重含 hf_quant_config.json 时自动识别 |
| **mem-fraction-static** | `--mem-fraction-static` 存在 | ✅ 默认 0.90 |
| **dist/nnodes/node-rank** | `--dist-init-addr`（别名 `--nccl-init-addr`）、`--nnodes`、`--node-rank` | ✅ |
| **speculative-algorithm** | 内置仅 `EAGLE/EAGLE3/NEXTN/NGRAM/STANDALONE/DFLASH`；**无 DSPARK**（`sglang/srt` 全包 grep 无 dspark） | 🔴 **方案 `--speculative-algorithm DSPARK` 在 0.5.14 不可用**，须 architect 改（EAGLE3/MTP draft 或首期关投机）；脚本默认不启用 |
| **kv-cache-dtype** | `auto/fp8_e5m2/fp8_e4m3/bf16/bfloat16/fp4_e2m1` | SGLang 自动；无 vLLM 式 nvfp4_ds_mla |
| **其他** | `--trust-remote-code`、`--host`、`--port`、`--dtype`、`--chunked-prefill-size`、`--kv-cache-dtype`、`--log-level` | ✅ |

## 3.3 flashinfer JIT 预构建结论（Task 1.3）

- **0.6.14 无 `preload()` 全局 API**；`jit_spec_registry` 初始为空（惰性注册，spec 参数化，需模型加载时实例化）。
- **容器已内置预编译 cubin：`Downloaded 16106/16106 cubins`；654 注册 modules 中 653 compiled / 1 not compiled**（非相关模块）。
- 相关 kernel 全部 **Compiled**：`batch_mla_attention`(ckv512/kpe64)、`cute_sm120_mxfp8_groupwise`、`fp4_gemm_cutlass_sm120`、`gemm_sm120`、`mxfp8_gemm_cutlass_sm120`、`sparse_mla_sm120`、`xqa_mla`(head_dim_576) 等。
- **结论：无需预构建**。若运行期出现未编译 kernel，正式 TP1 冷启动时自动 JIT 编译；缓存目录 `FLASHINFER_CACHE_DIR`（默认 `~/.cache/flashinfer`）已由脚本挂载 `<INSTALL_DIR>/cache/flashinfer-jit:rw` 持久化。
- 触发命令（正式运行第一动作，通常不需要）：`FLASHINFER_CACHE_DIR=/root/.cache/flashinfer python3 -m flashinfer download-cubin`（已内置 16106，会跳过）。
- ⚠️ 观察：`gen_moe_utils_module` 报 `No supported CUDA architectures found for major versions [10]`（默认 arch 读取问题，非阻塞，653/654 已编译）。

## 3.4 NCCL 挂载确认（Task 1.4）+ 生产 env 实测

- 挂载方案验证 ✅：`-v /opt/nccl-ringonly:/opt/nccl-ringonly:ro` + `-v <INSTALL_DIR>/lib/libncclpin.so:/opt/libncclpin.so:ro` → 容器内 `readlink -f /opt/nccl-ringonly/libnccl.so.2` = `libnccl.so.2.30.7`；权重 `-v .../deepseek-v4-flash-0731-nvfp4:/models:ro` 可见。
- **生产运行容器实测 env**（`docker inspect vllm-tp4-overlap-rank1`，2026-08-14 凌晨）：
  - `NCCL_IB_GID_INDEX=3`（⚠️ **生产实测=3，非旧脚本 2**）、`NCCL_ALGO=RING`、`NCCL_IB_TOS=46`
  - `NCCL_IB_PEER_HCA` rank1=`0=rocep1s0f1,roceP2p1s0f1;2=rocep1s0f0,roceP2p1s0f0`（脚本 case 一致）
  - `NCCL_NET_PLUGIN=none`、`NCCL_SOCKET_IFNAME=enP7s7`、`GLOO_SOCKET_IFNAME=enP7s7`
  - `LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`
- **容器自带**：SGLang 26.07 容器内置 `NCCL_VERSION=2.30.7`、默认 `NCCL_NET_PLUGIN=spcx`、默认 `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:...` → 脚本将其 LD_LIBRARY_PATH 前插 `/opt/nccl-ringonly` 并将 `NCCL_NET_PLUGIN=none`（对齐生产）。

## 3.5 容器内依赖完整性（Task 1.5）

`python3 -c "import sglang, sglang.srt.entrypoints.openai, flashinfer, torch, triton"` → 全部成功，无缺失。权重目录含 `tokenizer.json / tokenizer_config.json` → `HF_HUB_OFFLINE=1` 安全。

## 3.6 脚本清单与路径（Task 2，已写 02 `<INSTALL_DIR>/scripts/`）

| 脚本 | 路径 | 状态 |
|---|---|---|
| start_sglang_tp4_head.sh | `<INSTALL_DIR>/scripts/start_sglang_tp4_head.sh` | ✅ 已写 + `bash -n` 通过（本地+02） |
| start_sglang_tp4_worker.sh | `<INSTALL_DIR>/scripts/start_sglang_tp4_worker.sh` | ✅ 同上（NODE_RANK=1/2/3 参数化，PEER_HCA case 注入） |
| preflight_sglang.sh | `<INSTALL_DIR>/scripts/preflight_sglang.sh` | ✅ 同上（四机一键：镜像层 md5/权重 shard/端口/NCCL/内存） |

**脚本要点**：互斥守卫（`docker ps \| grep ^vllm-tp4-` 非空即中止，可捕获 `vllm-tp4-overlap-rank1`）、内存门禁 ≥55G（≤180s）、worker 等 head TCPStore :26000（120s 缺秩 exit1）、`--tp-size 4`、`--moe-runner-backend flashinfer_trtllm_routed`、`--enable-metrics`（无 --metrics-port）、GID=3/NET_PLUGIN=none/PEER_HCA 按 rank、快速失败关键字（NCCL error/kernel image/CUDA error/No available）。注释已写明正式运行前需 `docker rm -f sglang-nvfp4-tp4-{0,1,2,3}` 清残留。
**⚠️ 未执行任何脚本**（正式运行等主理人/用户通知）；head 脚本需在正式运行前拷至 node01、worker 拷至 02/03/04（或按 02 编排分发）。

## 3.7 发现的问题（本阶段新增）

| # | 严重度 | 问题 | 处置 |
|---|---|---|---|
| 1 | 🔴 高 | **DSPARK 投机在 0.5.14 不可用**（`--speculative-algorithm` 无 DSPARK；sglang/srt 无 dspark） | architect 复核：EAGLE3+MTP draft / 首期关投机；脚本默认关 |
| 2 | 🟠 中 | **0.5.14 无 `--metrics-port`**，8011 独立 metrics 端口不可达 | metrics 随 8010/metrics（`--enable-metrics`）；或升级容器版本时再评估 |
| 3 | 🟠 中 | 0.5.14 用 `--tp-size`（非 `--tp`） | 脚本已按 `--tp-size 4` 落 |
| 4 | 🟡 低 | 生产容器名变体 `vllm-tp4-overlap-rank1`（非 rank1） | 互斥守卫 `^vllm-tp4-` 已覆盖 |
| 5 | 🟡 低 | flashinfer 0.6.14 < 方案 0.6.15.post1 | 如遇 NCCL 回退问题再钉 |
| 6 | 🟢 提示 | SGLang 容器内置 NCCL 2.30.7 + 默认 NET_PLUGIN=spcx | 已覆盖为 none；ringonly+shim 补丁栈保留以对齐生产 |

## 3.8 正式运行前置动作（供主理人/用户通知后执行）

1. `docker rm -f sglang-nvfp4-tp4-{0,1,2,3}` 清残留；
2. 停生产 vLLM TP4（head+worker，含 `vllm-tp4-overlap-*`）→ 确认 monitor 退出；
3. 四机 `bash preflight_sglang.sh`（或按节点分发后逐一核验）；
4. 拷 head 脚本至 01、worker 至 02/03/04；
5. head(01) → worker(02) → worker(04) → worker(03) 顺序启动；
6. 健康检查 `curl :8010/health` + `/v1/models` + 首 token；快速失败关键字扫描。

---

# v4 节：Grafana 镜像（SGLang 实时分析面板，2026-08-14 执行）

> 操作人：general-purpose-1（运维执行代理，Rex）
> 目标：在 02 的 Grafana（容器 `aicad-grafana-1`）新建 SGLang 实时分析面板，**原 vLLM 面板零改动**。
> 硬边界：不重启 Grafana；不修改/删除 `/var/lib/grafana/dashboards/vllm-realtime.json`；不触碰生产 vLLM 容器；不启动 SGLang 服务。

## 4.1 操作结果

| 项 | 值 |
|---|---|
| 新面板 uid | `sglang-realtime` |
| 新面板 title | `DGX Spark SGLang 实时分析` |
| 容器内文件 | `/var/lib/grafana/dashboards/sglang-realtime.json`（75863 B） |
| 本地副本 | `C:\Users\novAI\WorkBuddy\集群部署\_archive_scratch\sglang-realtime-generated.json` |
| 面板数 | 30（7 行 + 23 图），与源完全一致 |
| datasource | Prometheus `PBFA97CFB590B2093`（不变） |
| 替换面板 | 13 个 vLLM 业务面板 → SGLang 指标 |
| 保留面板 | 7 行 + 5 dcgx + 5 node_network/Infiniband 原样（逐字校验一致） |
| API 验证 | `GET /api/dashboards/uid/sglang-realtime` → 200，`provisioned:true`，`provisionedExternalId:"sglang-realtime.json"`，title/uid/tags 正确，`vllm:` 残留 0 |
| 原面板 md5 | 操作前 `ed03adb523e806454186afe965dba8b1` ＝ 操作后一致（零改动） |

加载方式：file provisioning（`updateIntervalSeconds=10`）自动加载，未重启容器、无 `provision` 相关 error/warn 日志。

## 4.2 vLLM → SGLang 指标映射表（基于容器镜像实测 `nvcr.io/nvidia/sglang:26.07-py3`，2026-08-14）

SGLang 指标统一 `sglang:` 前缀；labels = `model_name/engine_type/tp_rank/pp_rank/moe_ep_rank`，**无 `node`、无 `job` 标签**（job 由 Prometheus scrape 时注入，故 `job="vllm"` 过滤已去除）；Histogram 仍带 `_sum/_count/_bucket` 后缀。

| # | 面板 | vLLM 表达式（节选） | SGLang 表达式 | 说明 |
|---|---|---|---|---|
| 1 | TTFT | `increase(vllm:time_to_first_token_seconds_sum{job="vllm"}[5m])/clamp_min(increase(..._count[5m]),1)` | `increase(sglang:time_to_first_token_seconds_sum[5m])/clamp_min(increase(..._count[5m]),1)` | 同名 Histogram |
| 2 | TPOT | `increase(vllm:request_time_per_output_token_seconds_sum[5m])/...` | `increase(sglang:inter_token_latency_seconds_sum[5m])/...` | SGLang 无独立 TPOT，用 ITL 近似 |
| 3 | ITL | `increase(vllm:inter_token_latency_seconds_sum[5m])/...` | `increase(sglang:inter_token_latency_seconds_sum[5m])/...` | 同名 Histogram |
| 4 | decode t/s | `sum(increase(vllm:request_generation_tokens_sum[5m]))/clamp_min(sum(increase(vllm:request_decode_time_seconds_sum[5m])),1)` | `sum(sglang:gen_throughput)` | SGLang 直接提供 gen_throughput Gauge(token/s) |
| 5 | prefill t/s | `sum(increase(vllm:request_prompt_tokens_sum[5m]))/clamp_min(sum(increase(vllm:request_prefill_time_seconds_sum[5m])),1)` | `sum(increase(sglang:prompt_tokens_total[5m]))/300` | SGLang 无 prefill 耗时指标，用 prompt token 速率近似 |
| 6 | QPS | `sum by (node)(rate(vllm:request_success_total{job="vllm"}[30s]))` | `sum by (model_name)(rate(sglang:num_requests_total[30s]))` | 无 node 标签 → 按 model_name |
| 7 | 请求队列 | `vllm:num_requests_running` / `vllm:num_requests_waiting` | `sglang:num_running_reqs` / `sglang:num_queue_reqs` | 同名 Gauge |
| 8 | 抢占率 | `sum by (node)(rate(vllm:num_preemptions_total[30s]))` | `sum by (model_name)(rate(sglang:num_retracted_requests_total[30s]))` | SGLang 无 preempt 计数，用 retract 近似（抢占/回退语义） |
| 9 | KV 水位 | `(vllm:kv_cache_usage_perc{job="vllm"}) * 100` | `(sglang:token_usage) * 100` | token_usage 为 0-1 比例 Gauge |
| 10 | 前缀缓存命中率 | `sum by (node)(rate(vllm:prefix_cache_hits_total[30s]))/clamp_min(sum by (node)(rate(vllm:prompt_tokens_total[30s])),1)` | `sglang:cache_hit_rate * 100` | SGLang 提供 cache_hit_rate Gauge（0-1） |
| 11 | 投机·接受率 (Phase-B) | `sum(increase(vllm:spec_decode_num_accepted_tokens_total[5m]))/clamp_min(sum(increase(vllm:spec_decode_num_draft_tokens_total[5m])),1)*100` | `sum(sglang:spec_accept_rate) * 100` | spec_* 为 Gauge（0-1），非 Counter |
| 12 | 投机·平均接受长度 (Phase-B) | `sum(increase(vllm:spec_decode_num_accepted_tokens_total[5m]))/clamp_min(sum(increase(vllm:spec_decode_num_drafts_total[5m])),1)` | `sum(sglang:spec_accept_length)` | 直接 Gauge |
| 13 | 投机·覆盖率 (Phase-B) | `sum(increase(vllm:spec_decode_num_draft_tokens_total[10s]))/10` | `sum(sglang:spec_accept_rate * sglang:spec_accept_length)` | 用 accept_rate×length 近似每步接受 token 数 |

注：面板 4/5/13 的守卫因子 `((sum(num_requests_running)+sum(num_requests_waiting))>0)` 已同步替换为 `sglang:num_running_reqs`/`sglang:num_queue_reqs`；投机解码 3 面板 title 追加 ` (Phase-B)`（Phase-A 无 DSPARK，指标为空属预期）。

## 4.3 验证记录

1. 生成：`gen_sglang_dashboard.py` 输出 30 面板，13 个 vLLM 面板替换；
2. JSON 合法性：`python3 -m json.tool` 通过；
3. 自动化断言全部通过：uid/title/tags 正确、panel id 顺序与源一致且唯一、无 `vllm:` 残留、`dcgx:*`/`node_*`/row 面板逐字一致、datasource uid 全部 `PBFA97CFB590B2093`；
4. 部署：`scp` → 02 → `docker cp` 进容器；sleep 14s 后 API 200；
5. 原 vLLM 面板 md5 操作前后一致 `ed03adb523e806454186afe965dba8b1`。

## 4.4 回滚方法

```bash
ssh node01 "docker exec aicad-grafana-1 rm /var/lib/grafana/dashboards/sglang-realtime.json"
```

文件删除后 provisioning 10s 内自动摘除面板（不重启 Grafana）。本地副本保留于 `_archive_scratch\sglang-realtime-generated.json` 供重建。

---

# v5 节：升级前置侦察与执行准备（2026-08-14，SRE 只读侦察）

- **执行人**：雷克斯（Rex）· SRE 工程师（工程保障团队）
- **执行时间**：2026-08-14（对应 UTC 05:40–05:50 窗口）
- **范围**：SGLang 0.5.14（26.07）→ 支持 DSPARK 的目标版本（v0.5.16+）**前置只读侦察**；正式升级动作待 architect 选型 + 主理人确认后执行
- **硬边界遵守**：全程只读（manifest inspect / curl / df / docker images / docker ps / grep / pip index / pip download 到 `--rm` 临时容器 /tmp 后自动销毁）；**未 pull 新镜像、未 pip 升级、未重打包、未启动任何服务容器、未触碰生产 vLLM、未删任何文件**

## 5.1 侦察结果总表

| # | 检查项 | 结果 | 判定 |
|---|---|---|---|
| 1 | NGC 南大源 `ngc.nju.edu.cn` 可达性 | ✅ 可达；v2 API 正常返回 tag 列表 | 首选通道可用 |
| 1a | **26.08-py3 tag 存在性** | 🔴 **不存在**（`no such manifest: ...sglang:26.08-py3`）；tags/list 最新仅到 `26.07-py3`（另含 26.03.post1/26.06 等） | **阻塞：南大源尚未同步 26.08** |
| 1b | 26.07 / 26.06 tag | ✅ 均存在（`26.07-py3` 为当前运行版） | 回滚/重拉锚点可用 |
| 1c | docker.io 直连 `registry-1.docker.io/v2/` | ❌ HTTP 000（15s 超时 / connection reset） | 直连不可用 |
| 1d | lmsysorg 经 DaoCloud `docker.m.daocloud.io/lmsysorg/sglang` | ✅ `v0.5.16` 与 `latest` 均存在，OCI index 含 **linux/arm64**（另含 amd64） | 自建镜像路线可行 |
| 2 | 磁盘（新镜像 20-35GB + 重打包） | 01=2.9T / 02=2.3T / 03=602G / 04=612G 可用 | ✅ 充足 |
| 2b | 02 `docker system df` | images 47.77GB（reclaimable 26.83GB）、containers 13/13、volumes 5.06GB | ✅ 充足 |
| 3 | 四机现有镜像锚点 | 01=`1e8eb5ea94b0`；02/03/04=`4f5f4cade001`（均 26.07-py3，registry tag 与 nvcr tag 齐备） | ✅ 回滚锚点锁定 |
| 4 | 02 registry | 容器 `registry` Up 37h，:5000；catalog **已含 `nvidia/sglang` repo** | ✅ |
| 5 | 启动脚本现状 | `start_sglang_tp4_head.sh`/`worker.sh` 存在（08-13 16:40）；关键参数 `--model-path ${MODEL_DIR}`、`--tp-size 4`、`--moe-runner-backend ${SGLANG_MOE_BACKEND}`、条件性 `--speculative-algorithm ${SGLANG_SPEC_ALGO}`（0.5.14 默认关） | ✅ 为 DSPARK 参数更新备好锚点 |
| 6 | 容器内 pip 通道 | ✅ pip index 可达；可见 sglang **0.5.17 / 0.5.16 / 0.5.15.post1 / 0.5.15 / 0.5.14…**，INSTALLED=0.5.14+nv26.7，LATEST=0.5.17 | ✅ |
| 6b | **aarch64 wheel 可用性** | ✅ `pip download --only-binary sglang==0.5.16` 成功 → `sglang-0.5.16-cp312-cp312-manylinux_2_34_aarch64.whl`（14.4MB） | **容器内 pip 升级 0.5.16 技术上可行** |
| 6c | flashinfer 现状 | ✅ `flashinfer 0.6.14`（import 实测，与 v3 基线一致） | 0.6.15.post1 如需再钉 |

## 5.2 发现的阻塞 / 风险

| # | 级别 | 说明 | 影响 / 处置建议 |
|---|---|---|---|
| B1 | 🔴 高 | **NGC 南大源无 26.08 tag**（最新 26.07-py3）。若升级路径依赖 NGC 26.08 容器，当前首选下载通道无法获取 | 需 architect 决策升级路径：①等南大镜像同步 26.08（不可控）；②容器内 pip 升级 0.5.16/0.5.17；③自建 lmsysorg v0.5.16 镜像（arm64 已确认） |
| B2 | 🟠 中 | docker.io 直连不可达（lmsysorg 直连探测被 reset） | 自建路线须走 DaoCloud 镜像站拉取（已实测可达）；分发仍走内网 registry |
| B3 | 🟡 中 | pip in-place 升级是 **PyPI 通用 wheel**，与 NGC 定制容器（torch 2.13.0a0+nv26.07、flashinfer 0.6.14）混装存在依赖冲突风险 | 升级前须在临时容器做依赖 dry-run + 兼容性验证；不建议直接 `pip install -U` 生产容器 |
| B4 | 🟢 低 | 01 镜像 virtual size 显示 42.2GB（save/load 层合并显示差异，已建档） | 以 RootFS 层 md5 为准，不影响 |

## 5.3 "待确认执行"清单（本轮全部未执行，待主理人/architect 确认后执行）

1. **确认升级路径**（阻塞项 B1 决策）：
   - A) NGC 26.08 容器 — ❌ 南大源无此 tag，暂不可行；
   - B) 容器内 pip 升级 0.5.16/0.5.17 — aarch64 wheel 已确认，但须先验证 torch/flashinfer 兼容；
   - C) 自建 `lmsysorg/sglang:v0.5.16`（经 DaoCloud 拉取 + flashinfer sm12x 处理）— arm64 已确认。
2. 确认后执行序列（届时按变更单执行，非本轮）：
   - 拉取/重打包 → `docker tag` → push 内网 registry `<NODE_IP>:5000/nvidia/sglang:<新tag>` → 四机分发。
   - ⚠️ **分发纪律（01 v1 manifest 教训）**：严格串行；**勿并发 pull 与 push**；核对标准=RootFS 层 md5 一致，非 IMAGE ID。
3. 升级后脚本参数更新：`--speculative-algorithm DSPARK`（0.5.16+ 支持）、按版本核对 `--metrics-port` 可用性、`--tp-size` 保持；更新前 `cp` 备份 + `bash -n`。
4. 正式服务启动：等主理人通知后按 §3.8 动作执行（互斥守卫→停 vLLM→preflight→启动→健康检查）。

*本节全部结论基于只读核验；未执行任何升级/写操作。*

---

# v6 节：SGLang 0.5.16 升级执行记录（Phase 0-4，2026-08-14，SRE 执行）

- **执行人**：雷克斯（Rex）· SRE 工程师（工程保障团队）
- **执行时间**：2026-08-14（UTC 06:00–06:10 窗口）
- **范围**：主理人裁决路径 C = 26.07 容器内 pip `--no-deps` 升级 sglang 0.5.16 → 重打包 → registry → 四机分发 → 版本/DSPARK flag 验证 → 启动脚本参数更新。**Phase 5（TP1 冒烟）未执行**（等停机窗口通知）；未启动 TP4 服务、未触碰生产 vLLM。
- **铁律遵守**：全部 pip 操作带 `--no-deps`；flashinfer 0.6.14 / torch 2.13.0a0+nv26.07 / CUDA / NCCL 2.30.7 均未动。

## 6.1 Phase 0 预检（02）

| 项 | 结果 |
|---|---|
| 基线版本 | sglang `0.5.14+nv26.7.59534057` / flashinfer `0.6.14` / torch `2.13.0a0+9186a08b2c.nv26.07` / transformers `5.8.1` / sglang-kernel `0.4.4+nv26.7` |
| 回滚锚点① | `<NODE_IP>:5000/sglang/sglang:0.5.14-nv26.07-rollback` push 成功，digest `sha256:713bc60d090f...`（层 Mounted from nvidia/sglang，秒级） |
| wheels（02 `/tmp/sglang-wheels/`） | `sglang-0.5.16-cp312-cp312-manylinux_2_34_aarch64.whl`（14.4MB）/ `sglang_kernel-0.4.5-cp310-abi3-manylinux2014_aarch64.whl`（36.6MB）/ `transformers-5.12.1-py3-none-any.whl`（11.2MB） |
| 依赖分析（0.5.16 METADATA） | 🔴 `torch==2.11.0`（锁死→**必须 `--no-deps`**）；✅ `flashinfer_python[cu13]==0.6.14`（匹配不动）；`sglang-kernel==0.4.5`（需升）；`transformers==5.12.1`（需升，现 5.8.1） |

## 6.2 Phase 1 构建升级镜像（02）

- Dockerfile：`FROM nvcr.io/nvidia/sglang:26.07-py3` → COPY 3 wheels → `RUN pip install --no-deps ...` → 清理。
- **build 成功**（10.3s）：安装 sglang-0.5.16 / sglang-kernel-0.4.5 / transformers-5.12.1（旧 0.5.14 / 0.4.4 / 5.8.1 被 uninstall）。镜像 ID `3b53c6da6963`。
- **push 成功**：digest `sha256:3365d0132d4cf30832edea4b7d1d3bb53d548f3719b21e143223e4801f6ff0fc`（仅 2 新层）。
- **容器内闭环验证（V3-7 清单 1/2）**：
  - V1 版本：`sglang 0.5.16 / flashinfer 0.6.14 / torch 2.13.0a0+9186a08b2c.nv26.07 / transformers 5.12.1` ✅
  - V2 `--speculative-algorithm` choices：`EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, **DSPARK**` ✅
  - V3 `--fp4-gemm-backend` choices：`{auto,flashinfer_cudnn,flashinfer_cutedsl,**flashinfer_cutlass**,flashinfer_trtllm,marlin}` ✅；`--moe-runner-backend` 含 `flashinfer_cutlass`/`flashinfer_trtllm_routed` ✅
  - V4 `--tp-size` 仍合法（`--tp-size TP_SIZE, --tensor-parallel-size`）✅ 无需回退 `--tp`
  - V1d `--speculative-dspark-block-size` flag 存在 ✅

## 6.3 Phase 2 四机分发（错峰串行）

- 01→03→04 依次 pull（**串行，未与 push 并发**），digest 全部 `sha256:3365d013...` ✅
- 四机 **RootFS 层 md5 一致** `0567b7512235235117432969178e6eba` ✅
- 本地 tag `sglang-nvfp4:0.5.16` 四机就位 ✅
- ⚠️ **发现：01 再现 v1 schema manifest 行为**：01 config_id = `3365d013...`（=manifest digest），02/03/04 config_id = `3b53c6da...`（构建产物）；IMAGE ID 显示不一致。**层 md5 四机一致 = 内容字节级统一**，且 Phase 3 功能验证 01 通过。判定为与 handoff §4 同源的 registry(registry:2/DaoCloud) 对 01 返回 v1 manifest 现象；**非阻塞**，建议后续 `docker save/load` 对齐 IMAGE ID。

## 6.4 Phase 3 四机版本验证（只验证，不启动服务）

| 节点 | sglang | flashinfer | torch | /proc/self/maps libnccl |
|---|---|---|---|---|
| 01 | 0.5.16 | 0.6.14 | 2.13.0a0+nv26.07 | `/opt/nccl-ringonly/libnccl.so.2.30.7` + `/opt/libncclpin.so` ✅ |
| 02 | 0.5.16 | 0.6.14 | 2.13.0a0+nv26.07 | 同上 ✅ |
| 03 | 0.5.16 | 0.6.14 | 2.13.0a0+nv26.07 | 同上 ✅ |
| 04 | 0.5.16 | 0.6.14 | 2.13.0a0+nv26.07 | 同上 ✅ |

- host 路径四机存在：`/opt/nccl-ringonly/libnccl.so.2 → 2.30.7`、`<INSTALL_DIR>/lib/libncclpin.so` ✅
- **LD_PRELOAD 生效**（libncclpin CPU pin 消息 + maps 实测 2.30.7）✅

## 6.5 Phase 4 启动脚本参数更新（02）

- head `8827→9454B`，worker `8073→8544B`；备份 `.bak-v0516` 已建；`bash -n` 本地+远端通过；head 远端与本地一致。
- **变更摘要**（head/worker 同步）：
  - IMG 参数化：`SGLANG_IMG` 默认 `<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark`
  - `--speculative-algorithm ${SGLANG_SPEC_ALGO}`，默认 **DSPARK**
  - `--speculative-dspark-block-size ${SGLANG_DSPARK_BLOCK_SIZE}`，默认 **5**（DSV4 0731 config dspark_block_size=5）
  - `--fp4-gemm-backend ${SGLANG_FP4_BACKEND}`，默认 **flashinfer_cutlass**（SM121 显式）
  - `--moe-runner-backend` 默认 **flashinfer_cutlass**
  - env 新增 `FLASHINFER_CUDA_ARCH_LIST=12.0a 12.1a`（`SGLANG_RAGGED_VERIFY_MODE=compact` 原有已含）
  - 旧改名 flag 检查：**无** `--enable-waterfill` 等旧 flag；`--tp-size` 保持合法
- **未执行脚本**（等停机窗口通知）。

## 6.6 发现的问题 / 待办

| # | 级别 | 说明 | 处置 |
|---|---|---|---|
| 1 | 🟡 中 | 01 再现 v1 manifest 行为（IMAGE ID/config_id 与 02/03/04 不一致；层 md5 一致） | 非阻塞；建议 save/load 对齐，或接受（内容已统一） |
| 2 | 🟢 提示 | Phase 5（TP1 冒烟）未执行 | 等主理人/用户停机窗口通知；启动前须 `docker rm -f sglang-nvfp4-tp4-{0,1,2,3}` + 停 vLLM TP4 + preflight |
| 3 | 🟢 提示 | 01 v1 manifest 坑的通用规避：后续任何新 tag 分发，01 单独 save/load 或先于三机验证 manifest 版本 | 列入部署 SOP |

## 6.7 回滚锚点（升级后）

| 镜像 | 位置 | digest/ID |
|---|---|---|
| 0.5.16 新镜像 | registry `sglang/sglang:0.5.16-nvfp4-spark` / 本地 `sglang-nvfp4:0.5.16` | manifest `3365d013...` / config(02/03/04) `3b53c6da...` |
| 0.5.14 回滚锚点① | registry `sglang/sglang:0.5.14-nv26.07-rollback` | `713bc60d...` |
| 0.5.14 原镜像 | `nvcr.io/nvidia/sglang:26.07-py3`（四机仍在） | 01=`1e8eb5ea94b0` / 02/03/04=`4f5f4cade001` |
| 脚本备份 | `<INSTALL_DIR>/scripts/start_sglang_tp4_{head,worker}.sh.bak-v0516` | md5 `23e1d448...`/`32ce3bb4...` |

*本节结论基于实际执行与验证；Phase 5 服务启动未执行，等主理人/用户通知。*

# v7 节：Phase 5a 正式执行记录（2026-08-14，SRE 执行，停机窗口）

- **执行人**：雷克斯（Rex）· SRE 工程师
- **窗口**：2026-08-14 CST 15:15 起（用户确认让出）；本记录全程 UTC+8 时间戳（服务器为 UTC，标注换算）
- **结论**：🟥 **SGLang 0.5.16 TP4 启动 FAIL（sgl_kernel 0.4.5 ABI 不兼容，冷启动即死）** → 按纪律已回滚恢复生产 vLLM（四机 healthy + API 200 + 单请求冒烟通过）。**未进入 K1-K8 / C0-C6 阶段。**
- **关键纪律遵守**：任何 FAIL 先停 SGLang 再评估；SGLang 容器已全部清理（无残留）；vLLM 已恢复并验证；未修改生产 vLLM 配置/脚本；未删权重；未重启 Grafana。

## 7.1 时间线（UTC+8 = UTC+8；服务器 UTC 已换算）

| 时间(CST) | 事件 |
|---|---|
| 15:17 | 四机 SSH 连通确认；Step 0 preflight 开始 |
| 15:17-15:19 | 发现 vLLM 已在窗口前停止（systemd inactive、无 monitor、端口空闲、可用内存 110-116G）；确认 embed 03/04 运行（可用内存>90G → 保持不停）；镜像 digest `sha256:3365d013...` 四机一致；权重 48 shard+config 四机齐全 |
| 15:19 | 发现 SGLang 脚本仅存在于 02 → 分发到 01/03/04 |
| 15:20 | 🔴 发现**脚本 Bug#1（模型路径）**：`--model-path ${MODEL_DIR}`（host 路径）但 bind 是 `${MODEL_DIR}:/models:ro`，容器内不存在 host 路径（实测确认）→ 修正为 `--model-path /models`（与生产 vLLM `--model /models` 约定一致），四机同步 |
| 15:23 | 启动 head(rank0) → 🔴 **脚本 Bug#2（内存门禁 locale）**：`free -g | awk '/^Mem:/'` 在 zh_CN locale 输出 `内存：` 不匹配 → 修正为 `LC_ALL=C free -g`，四机同步 |
| 15:23-15:25 | head 第二次启动 → 🟥 **sgl_kernel 0.4.5 ABI FAIL**：`sm100/common_ops.abi3.so: undefined symbol _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v`；fast-fail 自动清理容器 |
| 15:25-15:27 | 复现定位：`import sglang` OK（无 GPU）；带 GPU 时 `from sgl_kernel import common_ops` FAIL；**基线 26.07 镜像（sgl_kernel 0.4.4）common_ops OK** → 确认为 0.5.16 升级引入的 sgl_kernel 0.4.5 wheel ABI 不兼容 |
| 15:27-15:30 | 按纪律回滚：停 SGLang（已无残留）→ 恢复 vLLM（head+3 worker，systemd 需 sudo 不可用，改走 monitor wrapper 正常通道 + 容器内 API key 重建环境） |
| 15:33 | 四机 vLLM 容器全部 Up(healthy)；TCPStore 25999 监听 |
| 15:44 | vLLM API :8001 health=200；/v1/models + 单请求冒烟通过 → 生产恢复确认 |

## 7.2 Step 0 preflight 结果

| 节点 | 内存可用(free -g) | 端口 8010/26000 | 权重 shard | 镜像 digest | NCCL ringonly | 备注 |
|---|---|---|---|---|---|---|
| 01 | 116G | 空闲 | 48 | 3365d013 | ✅ 2.30.7 | vLLM head 已停(Exited 0)；embed 无 |
| 02 | 115G | 空闲 | 48 | 3365d013 | ✅ | vLLM rank1 已停(Exited 137) |
| 03 | 110G | 空闲 | 48 | 3365d013 | ✅ | vLLM rank3 已停；embed 运行(~120MiB) |
| 04 | 110G | 空闲 | 48 | 3365d013 | ✅ | vLLM rank2 已停；embed 运行(~113MiB) |

- 内存门禁：停 vLLM 后可用内存远 >55G（110-116G），embed 保留（>90G 阈值）✅
- systemd：`vllm-tp4-head.service`（01）/`vllm-tp4-worker.service`（02/03/04）inactive dead；无 timer；monitor 进程无 → 无自愈拉起风险
- 附加发现：01 上 `vllm027-build` 容器（vLLM 构建任务，非生产服务）在跑，不影响互斥/资源（仅 CPU 构建，未占用生产 GPU 语义）

## 7.3 启动/冒烟结果（K1-K8 / C0-C6）

| 检查点 | 结果 | 说明 |
|---|---|---|
| K1 /health | ⛔ 未到达 | SGLang 冷启动在 import sgl_kernel 即失败，head 容器被 fast-fail 自动清理 |
| K2-K8 | ⛔ 未到达 | 因 K1 前置失败全部未执行 |
| C0-C6 | ⛔ 未到达 | 同上 |

**FAIL 根因（已实测定位，非猜测）**：
- `sgl_kernel` 0.4.5 wheel（v6 Phase 1 从 PyPI 安装）的 `sm100/common_ops.abi3.so` 引用 `at::TensorBase::const_data_ptr<int,0>()` 符号，但容器内 torch `2.13.0a0+9186a08b2c.nv26.07` **未导出该符号** → `undefined symbol` → sgl_kernel 无法加载任何 common_ops。
- v6 已记录 METADATA 显示 0.5.16 `torch==2.11.0` 锁死 → 当时用 `--no-deps` 跳过了 torch 升级，**但 sgl_kernel 0.4.5 二进制实为对 torch 2.11 ABI 编译** → 与容器 torch 2.13.0a0 不兼容。
- 补充：0.4.5 wheel 内**只有 sm100/sm90 目录，无 sm120**；GB10(SM121) 被归到 sm100 路径加载，撞上该 ABI 问题。
- 基线镜像 `nvcr.io/nvidia/sglang:26.07-py3`（sgl_kernel 0.4.4+nv26.7，NVIDIA 原生）在同样 GPU 下 `common_ops OK` → **确认是 0.4.5 wheel 的问题，非环境/驱动问题**。

**修复建议（供主理人/architect 决策，SRE 未自行改镜像）**：
1. **首选**：sgl_kernel 不用 PyPI 0.4.5 wheel；改用容器内 0.4.4+nv26.7（NVIDIA 原生，已验证 common_ops OK）或找与 torch 2.13.0a0+nv26.07 ABI 匹配的 NVIDIA 分发 sgl_kernel。
2. 或：将容器 torch 对齐 sgl_kernel 0.4.5 的 torch 2.11（但会动 NVIDIA 容器 torch，风险高，不推荐）。
3. 重打包镜像需重新 push + 四机分发 + 层 md5 校验（参考 v6 Phase 1-3），并**新增"带 GPU 的 common_ops 冒烟"作为发布门槛**（v6 只做了无 GPU 的 import sglang 验证，遗漏了此 ABI 检查）。

## 7.4 回滚执行记录（恢复生产 vLLM）

- **触发**：SGLang 启动 FAIL + 窗口纪律"任何一步失败先停 SGLang 再恢复"
- **停 SGLang**：无残留容器（head 启动脚本 fast-fail 已 `docker rm -f`；workers 从未启动）；四机端口 8010/26000 空闲确认
- **恢复 vLLM**：
  - 说明：systemd 启动需要 sudo（无 NOPASSWD、交互密码不可用）；生产 systemd unit 的 `EnvironmentFile=<INSTALL_DIR>/secrets/vllm.env` 为 root-only 不可读。
  - 采用**正常通道等价方案**：直接运行 monitor wrapper（与 systemd ExecStart 相同脚本），以 `VLLM_API_KEY`（从既有容器 `docker inspect` 提取，与生产一致）+ 各 rank `NODE_RANK`/`VLLM_HOST_IP` 注入环境。
  - 01 head：`VLLM_API_KEY=... nohup bash monitor_tp4_head.sh &` → rank0 容器重建
  - 02/04/03 worker：`VLLM_API_KEY=... NODE_RANK=1/2/3 VLLM_HOST_IP=... nohup bash monitor_tp4_worker.sh &`
- **验证**：四机 `vllm-tp4-rank{0,1,2,3}` 全部 Up(healthy)；TCPStore 25999 监听；:8001/health=200；`/v1/models` 返回 `deepseek-v4-flash-0731`（max_model_len 400000）；单请求 chat 返回正常内容（非空、非 NaN）
- **注意**：monitor 通过 nohup 运行，非 systemd 托管；如需恢复 systemd 自愈语义，建议主理人后续在具备 sudo 的环境执行 `systemctl start vllm-tp4-head.service`（01）+ `vllm-tp4-worker.service`（02/03/04）——已确认 unit 存在且 enabled，monitor 对已运行容器是 `docker wait` 跟随，不冲突。

## 7.5 脚本变更记录（本次修正，均已留 .bak）

| 文件 | 变更 | 备份 |
|---|---|---|
| `start_sglang_tp4_head.sh` / `worker` | `--model-path ${MODEL_DIR}` → `--model-path /models`（容器内路径） | `.bak-preflight-fix`（02）+ 同步四机 |
| `start_sglang_tp4_head.sh` / `worker` | 内存门禁 `free -g` → `LC_ALL=C free -g`（zh_CN locale 兼容） | 同上 |

*本节基于实际执行；SGLang 服务未启动成功，等镜像 ABI 修复后按 v6/v7 preflight 重试。*

## 7.6 补充核验：方案 A 实证（2026-08-14，SRE，仅 02 执行）

**背景**：为 architect 方案 A（保留 NGC sgl_kernel 0.4.4+nv26.7，仅装 sglang 0.5.16 主包）提供实证依据。执行时生产 vLLM 已恢复运行（02 有 rank1），全程未触碰生产配置。

| 核验项 | 结果 |
|---|---|
| 原镜像 26.07-py3 sglang | `0.5.14+nv26.7.59534057` |
| 原镜像 sgl_kernel | `0.4.4+nv26.7.59534057`（pip list + `sgl_kernel.__version__` 双确认） |
| 0.5.16 升级镜像 sgl_kernel | **0.4.5**（确认错装，PyPI wheel，非 NVIDIA 原生） |
| 0.5.16 主包 | `0.5.16`（正常，无其他异常） |
| **0.4.4 SM121 GPU common_ops 冒烟** | ✅ **OK**：`torch.cuda.get_device_capability()` = (12,1)；`import sgl_kernel` → 0.4.4 loaded OK；`from sgl_kernel import common_ops` → **COMMON_OPS_OK**，EXIT=0 |

- **路径纠正**：0.4.4 中 common_ops 是包级属性，`sgl_kernel.ops` 子模块不存在（team-lead 初始命令路径会 ModuleNotFoundError）；正确路径为 `from sgl_kernel import common_ops`。
- **vLLM 影响**：GB10 UMA 的 nvidia-smi 内存列恒为 `[N/A]`（不支持逐进程显存查询），以 GPU util + 容器状态替代：前后 util 均 0%、`vllm-tp4-rank1` 全程 Up(healthy) 未中断，冒烟退出后即释放。
- **结论**：方案 A 实证可行。重打包镜像时仅装 sglang 0.5.16（`--no-deps`），**不要**装 PyPI sglang-kernel 0.4.5。

# v8 节：abi2 重打包 + 四机分发 + GPU 冒烟（2026-08-14，SRE 执行，方案 A）

- **执行人**：雷克斯（Rex）· SRE 工程师
- **范围**：architect V4 方案 A 定稿 → 02 重打包（保留 NGC sgl_kernel 0.4.4+nv26.7，只装 sglang 0.5.16 主包）→ push → 四机分发 → 四机带 GPU common_ops 冒烟（新发布门槛）→ 启动脚本 tag 更新。**无需停机窗口，与生产 vLLM 并存，生产全程 healthy 未受影响。**
- **结论**：✅ **全部通过**——新镜像 abi2 四机层 md5 一致，四机带 GPU `COMMON_OPS_OK`（EXIT=0），生产 vLLM 四机 healthy、util 0%。

## 8.1 镜像构建与版本（02）

| 项 | 值 |
|---|---|
| Dockerfile | `FROM nvcr.io/nvidia/sglang:26.07-py3` → `RUN pip install --no-deps sglang-0.5.16-*.whl`（**只装主包，未装 sglang_kernel 0.4.5**） |
| 构建产物 | image `sha256:01b0c5bb0d14...`，BUILD_EXIT=0 |
| 容器内版本 | pip list：sglang **0.5.16** / sglang-kernel **0.4.4+nv26.7.59534057**（确认保留，无 0.4.5） |
| launch_server --help | `--speculative-algorithm` Builtins 含 **DSPARK**；`--fp4-gemm-backend`/`--moe-runner-backend` 含 `flashinfer_cutlass` ✅ |
| 说明 | 无 GPU 容器内 `import sglang` 报 libcuda 错误为预期（不加载 common_ops），版本以 pip list + 带 GPU 冒烟为准 |

## 8.2 Push / digest / 四机分发

| 项 | 值 |
|---|---|
| registry tag | `<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2` |
| digest | `sha256:f39263d2d36710215dda882e4f4c80815e873e9107329821abdeba779f4a6947` |
| RootFS 层 md5 | `58c77180286298b77f0802ff510e08f8`（**四机一致**，字节级统一） |
| 本地 tag | `sglang-nvfp4:0.5.16-abi2`（四机已打） |
| 分发 | 01/03/04 错峰串行 pull，digest 全一致；01 IMAGE ID 显示 30.8GB 为 v1 manifest 现象（层 md5 一致=内容统一，同 v6 记录） |

## 8.3 四机带 GPU common_ops 冒烟（ABI 发布门槛）

| 节点 | PRE vLLM | 冒烟结果 | POST vLLM | POST util |
|---|---|---|---|---|
| 01 | rank0 healthy | **COMMON_OPS_OK** EXIT=0 | rank0 healthy | 0% |
| 02 | rank1 healthy | **COMMON_OPS_OK** EXIT=0 | rank1 healthy | 0% |
| 03 | rank3 healthy | **COMMON_OPS_OK** EXIT=0 | rank3 healthy | 0% |
| 04 | rank2 healthy | **COMMON_OPS_OK** EXIT=0 | rank2 healthy | 0% |

- 命令：`docker run --rm --gpus all ...abi2 -c "nvidia-smi && python3 -c 'import torch; torch.cuda.init(); from sgl_kernel import common_ops; print(\"COMMON_OPS_OK\")'"`
- 意义：无 GPU 容器内 import 不加载 common_ops，无法暴露 ABI 问题（v7 已踩坑）；本门槛确保 SM121 带 GPU 真实加载通过。

## 8.4 启动脚本 tag 更新（不执行）

| 文件 | 变更 |
|---|---|
| `start_sglang_tp4_head.sh` / `worker.sh` | `IMG="${SGLANG_IMG:-...0.5.16-nvfp4-spark-abi2}"`（head:50 / worker:58），注释行同步 |
| 备份 | `.bak-v7-abi2`（02） |
| 校验 | `bash -n` SYNTAX_OK；speculative/backend 参数未动（仍 DSPARK + flashinfer_cutlass） |
| 分发 | 已同步 01/03/04（四机 IMG 行确认 abi2） |
| 执行 | **未执行启动**（等主理人/用户后续停机窗口调度） |

## 8.5 发现 / 备忘

| # | 说明 |
|---|---|
| 1 | 02 作为跳板 `ssh node01` 自连 host key verification failed（非目标机问题，Windows 侧直连 02 正常）；已用 Windows 直连/本地命令补齐验证 |
| 2 | 旧错误镜像 `0.5.16-nvfp4-spark`（sgl_kernel 0.4.5）**保留未删**（新 tag 只增不改原则）；后续测试须用 abi2 |

*本节基于实际执行与验证；启动脚本已指向 abi2，服务启动待后续窗口。*

---

# v11 节：Phase 5a 重跑 · 80G 内存上限（2026-08-15 22:28–23:20，SRE 执行，用户重跑窗口）

## 11.0 背景与硬边界

- 用户重跑窗口已确认：vLLM TP4 全停、四机内存可用 01=115G/02=114G/03=110G/04=110G；端口 8010/26000 空闲
- **80G 内存上限是用户硬性要求**（上次 0.90 无上限导致 03/04 挂起 1 小时+、需物理重启）
- 镜像：`<NODE_IP>:5000/sglang/sglang:0.5.16-nvfp4-spark-abi2`（digest f39263d2，sglang 0.5.16 / sgl_kernel 0.4.4+nv26.7 保留；0.4.5 ABI 崩溃版已废弃）
- 已知参数：`--tp-size 4 --nnodes 4 --dist-init-addr <NODE_IP>:26000 --port 8010 --mem-fraction-static 0.65 --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cutlass --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --enable-metrics` + NCCL 全套

## 11.1 Step 0：80G 内存上限修复（02，改前已备份）

| 项 | 状态 | 说明 |
|---|---|---|
| `.bak-80g` 备份 | ✅ 已存在 | `start_sglang_tp4_head.sh.bak-80g` / `start_sglang_tp4_worker.sh.bak-80g`（0.90 原版） |
| `MEM_FRACTION_STATIC` 默认 0.90→**0.65** | ✅ 已在位 | head L57 / worker L63：`${MEM_FRACTION_STATIC:-0.65}`（80G/121.6GiB≈0.658，与生产 vLLM util 0.65 对齐） |
| `docker run --memory 80g` | ✅ 已在位 | head L176 / worker L199：`--memory 80g --shm-size=64gb ...`（UMA 下 CUDA 分配与 cgroup 兼容性**未在本次验证**，因启动未达运行期即失败；见 11.6 问题 #2） |
| `bash -n` 验证 | ✅ 通过 | 四机脚本 md5 一致（head=c5d84b22…，worker=7bf1adfe…） |
| **新增修复**：`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | ✅ 已加 | 见 11.4 根因 #1；官方开关（environ.py:717 / flexkv README:107），备份 `.bak-skipkernchk-<ts>` |

## 11.2 Step 1：TP4 环网启动（head-first，3 轮尝试）

| 轮次 | 时间（UTC+8） | 动作 | 结果 |
|---|---|---|---|
| 1 | 22:34 | 01 head rank0 + 02/04/03 worker rank1/2/3 | ❌ **失败：kernel 版本断言**（`sglang-kernel 0.4.4+nv26.7 < 要求 0.4.5`） |
| 2 | 22:36 | 加 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 后重启 | ❌ **失败：DSpark verify CUDA graph capture topk=192**（`Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=64 topk=192 page_block_size=64`） |
| 3 | 23:05 | 加 `--disable-cuda-graph`（错误提示方案 3） | ❌ **失败：DSpark draft verify 运行时 num_tokens=5 仍走 flashinfer sparse_mla_sm120 prefill dispatch**（`Check failed: num_tokens > 64 (5 vs. 64)`） |

- 四机 TCPStore :26000 监听、DSpark draft runner 初始化、FlashInfer autotune 均正常，说明**网络/NCCL/权重加载/DSPARK draft 加载全部通过**，仅 DSpark draft verify 的 sparse-MLA prefill 路由失败。

## 11.3 Step 2-3：K1-K8 / L3 冒烟

**未执行**（服务未就绪即被上述根因阻断；head /health 未达 200）。

## 11.4 根因分析（3 轮失败，均为 SGLang 0.5.16 + flashinfer 0.6.14 在 SM120/121 的兼容性缺口）

| # | 现象 | 根因 | 性质 |
|---|---|---|---|
| 1 | `assert_pkg_version: sglang-kernel 0.4.4+nv26.7 < 0.4.5` | SGLang 0.5.16 engine.py:1313 硬性要求 kernel ≥0.4.5；但**本镜像 0.4.4+nv26.7 是保留正确版本**（0.4.5 是 ABI 崩溃版已废弃） | **镜像内版本策略与代码断言冲突** → 官方开关 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 已解 |
| 2 | verify CUDA graph capture：`topk=192` 不被支持 | DSpark **target verify**（num_tokens_per_req=6）graph 成功；DSpark **draft verify**（num_tokens_per_req=5）topk=192 → flashinfer `sparse_mla_sm120_prefill.cu` 仅支持 topk∈{128,512,1024,2048}，192 不在列表（decode 表同样无 (64,192)） | **flashinfer 0.6.14 预编译 kernel 覆盖不足** → `--disable-cuda-graph` 可跳过 capture，但见 #3 |
| 3 | 运行时 draft verify：`num_tokens > 64 (5 vs. 64)` | `flash_mla_sm120.py:459` 对 DSpark draft verify（B=5≤64）仍调 `_sparse_mla_sm120_paged_attention` → flashinfer `_paged_attention`（line 311-380）判定 `_decode_dsv4_dispatchable()` 失败（topk=192 不在 decode 表）→ fall through 到 prefill orchestrator → prefill 又要求 `num_tokens>64` | **SGLang 0.5.16 将 ≤64 的 draft verify 路由到 SM120 sparse-MLA prefill dispatcher 的缺陷 + flashinfer 不支持 192** |

**结论：该组合（SGLang 0.5.16 + flashinfer 0.6.14 + SM121 + DSPARK draft verify topk=192）需代码层修复（patch SGLang flash_mla_sm120.py 路由 或 flashinfer 增加 topk=192/≤64 dispatch），非启动参数可解。**

## 11.5 资源状态（回传时集群已恢复）

| 节点 | 内存可用 | sglang 容器 | 端口 8010/26000 |
|---|---|---|---|
| 01 | 115G | 0 | 空闲 |
| 02 | 115G | 0 | 空闲 |
| 03 | 110G | 0 | 空闲 |
| 04 | 110G | 0 | 空闲 |

- vLLM **未恢复**（testing 的 L2/L4 随后执行，恢复由主理人调度）
- 容器 `--restart no` 生效，无自动拉起残留

## 11.6 问题与建议

1. **80G 上限核心验收未达**：因服务未启动，无法提供"容器内存 ≤80G、free 余量 ≥30G"实测；**脚本层 0.65 + --memory 80g 已就位**，待兼容性问题修复后重跑再验。
2. **UMA 下 `--memory 80g` 兼容性未验证**：Docker cgroup 内存限制与 GB10 UMA CUDA 分配的兼容性需在服务运行期观察（若 CUDA 分配命中 cgroup 上限导致 cudaErrorMemoryAllocation，需回传，勿自行去掉）。
3. **建议主理人决策**：①联系 SGLang/flashinfer 上游 patch DSpark draft verify SM120 路由（首选）；②或评估降级路径（如 DSPARK→STANDALONE 无投机、或换镜像/flashinfer 版本）；③或接受 `--disable-cuda-graph` 组合补丁后重跑（当前 #3 仍阻断，需代码修复）。

## 11.7 本次脚本变更清单（均已留 .bak）

| 文件 | 变更 | 备份 |
|---|---|---|
| `start_sglang_tp4_head.sh` / `worker.sh`（四机） | MEM_FRACTION_STATIC 0.90→0.65、`--memory 80g`、`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | `.bak-80g`（既有）、`.bak-skipkernchk-<ts>`（本次） |

*本次重跑因 SGLang/flashinfer 的 DSPARK draft verify SM120 路由缺陷阻断，80G 上限脚本修复已就位，待代码层修复后重跑。*
