# LuZ0.3.1 部署指导教程（DEPLOYMENT GUIDE）

- **适用版本**：LuZ0.3.1（生产采纳形态：W4A4 full + 池补丁 + FI 0.6.16 + threshold 4096 + util 0.82）
- **目标读者**：负责在 4× DGX Spark 上部署/维护本套件的 SRE、DevOps 与平台工程师
- **前置阅读**：`README.md`（仓库导航）→ `docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`（指标基线）→ `docs/07-deployment/LuZ0.3.1-release-notes.md`（构成与回滚链）
- **维护**：工程保障团队（Docu/SRE）｜**口径**：本文所有地址、用户名、密钥均为占位符（`REGISTRY_HOST` / `<NODE_IP>` / `<USER>` / `<API_KEY>`），按 `REDACTION-MAP.md` 语义替换为现场值

> 本文给出**可直接复现的生产镜像容器方案**：既支持「运行时挂载（零 bake，随脚本默认）」，也支持「bake 自包含镜像 LuZ0.3.1（一键回滚/分发）」。两种路径共用同一套组装材料，见 §3。

---

## 目录

1. [概览与 5 分钟快速开始](#1-概览与-5-分钟快速开始)
2. [环境要求](#2-环境要求)
3. [LuZ0.3.1 镜像构成与构建（生产镜像容器方案）](#3-luz031-镜像构成与构建生产镜像容器方案)
4. [启动部署（head-first 编排 + systemd 自愈）](#4-启动部署head-first-编排--systemd-自愈)
5. [生产参数表（脱敏通用）](#5-生产参数表脱敏通用)
6. [验证与质量门](#6-验证与质量门)
7. [回滚](#7-回滚)
8. [安全说明](#8-安全说明)
9. [参考链接](#9-参考链接)

---

## 1. 概览与 5 分钟快速开始

LuZ0.3.1 是 DeepSeek V4 Flash 在 4× DGX Spark（TP4 环网）上的生产调优形态。其核心价值：

- **W4A4 full 量化 + 跨层几何键共享池（池补丁）**：MoE 权重显存 45.32 GiB（池化前 68.15 GiB，省 22.83 GiB），PR 4K 单流 2950.5 tok/s（+6.6% vs W4A16 基线），并发 C6/C12 +11.4%/+11.6%。
- **FlashInfer 0.6.16 定制树 + ring-only NCCL 2.30.7**：prefill 路径全部落带内，RING 物理环序消除 stall。
- **自包含恢复镜像 + 一键回滚**：`restore_luz031.sh` 支持 `--dry-run`，全部回滚 <10 分钟。

### 1.1 快速开始（已验证路径）

```bash
# 0) 前置：四机 SSH 免密（head→worker）、registry 可达、权重已放置
# 1) 生成 API key（部署者自行生成，勿使用示例值）
export VLLM_API_KEY="<API_KEY>"        # 由部署者生成，见 §8

# 2) head 上执行四机编排（head-first、幂等、R12 版 TCPStore 门禁 + B12X 错峰）
cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh

# 3) 验证就绪（§6）
curl -sf http://<NODE_IP>:8001/health          # 期望 200
systemctl is-active vllm-tp4-head.service       # active
for h in node02 node03 node04; do ssh $h "systemctl is-active vllm-tp4-worker.service"; done  # active
```

> 冷启动约 16 分钟（权重加载 + 4 rank NCCL rendezvous + B12X JIT）。完整验证清单见 §6。

### 1.2 两种镜像容器方案（先看结论）

| 方案 | 形态 | 适合场景 | 回滚 |
|---|---|---|---|
| **A. 运行时挂载**（默认） | 基座镜像 + 只读 bind-mount 注入 overlay/插件/库 | 参数调优期、灰度、开发 | 改回 `.bak` 快照 + head-first 重建 |
| **B. bake 自包含镜像**（推荐生产/发布） | 基座 + 全部组装层 bake 进 `LuZ0.3.1` tag | 生产终态、多机分发、一键恢复 | `docker pull` 恢复镜像 + 启动 |

---

## 2. 环境要求

### 2.1 硬件

| 项 | 要求 |
|---|---|
| 节点 | **4× NVIDIA DGX Spark**（GB10，`sm_121a`，单机约 121.6 GiB UMA 可用） |
| 拓扑 | **TP4 环网（Ring）**：4 机按环序直连（`node01↔node02↔node04↔node03↔node01`），无交换机 |
| 互联 | RoCE/QSFP 双平面直连（每机 ≥2× 200G RoCE 口，与两邻居各 1 条链路） |
| 管理面 | 独立管理网承载 SSH/监控/TCPStore 控制面（`<NODE_IP>` 占位） |
| 网络要求 | MTU 9000（RoCE 数据面）；RoCEv2 GID 固定（现场核对 `NCCL_IB_GID_INDEX`） |
| 存储 | 模型权重可本地 serving 或 NFS 双源挂载（`fault-tolerance.md` §2） |

> 不要求交换机：环网直连。单链路/单节点故障会降级或中断 TP4（架构性限制，见 `fault-tolerance.md` §6）。

### 2.2 软件栈（推荐）

| 组件 | 版本/说明 |
|---|---|
| OS | Ubuntu Server（aarch64） / DGX 官方基座 |
| 驱动 + CUDA | **CUDA 13.0 运行时**（容器内 `/usr/local/cuda`）；nvcc wrapper 将 `sm_120f` → `sm_121a`（`<INSTALL_DIR>/envs/nvcc_wrapper.py`） |
| NCCL | **2.30.7 ring-only 定制**（环序强制补丁；`NCCL_ALGO=RING` + 4 通道 MIN/MAX=4）。库以 md5 + 构建来源描述发布，**二进制不随仓库**（见 §3.4） |
| 容器运行时 | Docker + 本地 registry `REGISTRY_HOST:5000` |
| 编排 | systemd（head/worker service + healthcheck timer）+ `start_tp4_cluster.sh`（R12） |

### 2.3 前置资源清单

| 资源 | 来源 | 说明 |
|---|---|---|
| 模型权重 `deepseek-v4-flash-0731` | 独立获取（不随仓库/镜像） | MXFP4/nvfp4 格式；挂载到容器 `/models` |
| 基座镜像 | `REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0` | Anemll 0.2.1 / vLLM 0.26.1 fork |
| API key | 部署者生成 | `VLLM_API_KEY` 环境变量，见 §8 |
| 四机 SSH 免密 | 现场配置 | `start_tp4_cluster.sh` 通过 ssh 下发 worker |

---

## 3. LuZ0.3.1 镜像构成与构建（生产镜像容器方案）

### 3.1 镜像构成总览

LuZ0.3.1 = **基座镜像 + 6 类组装层**。组装材料在本仓库 `kernels/`、`patches/`、`scripts/` 内（含 md5 记录）；其中 `libncclpin.so` 二进制**不随仓库**（按源码 + md5 构建，§3.4），ringonly 提供补丁 + 构建记录 + 构建产物参考副本（`patches/ringonly-v5-2026-08-23/`，md5 2b8669ec）。

| # | 层 | 内容/来源 | 完整性（md5） | 载体 |
|---|---|---|---|---|
| 1 | **基座镜像** | `REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（Anemll 0.2.1 / vLLM 0.26.1 fork） | digest `sha256:<BASE_IMAGE_DIGEST>`（现场 resolve） | registry |
| 2 | **plugin_a1**（W4A4 池化插件） | `kernels/server-nvfp4/plugin_a1/`（routea-plugin-a1 0.1.0，vLLM general_plugins 注册） | `w4a4_experts.py` md5 `e5ed0c853c4964846d782686e9decb9c` | 容器内 `pip install --no-deps` |
| 3 | **overlay-wsdedup**（池补丁） | `patches/server-overlay/flashinfer_b12x_moe.py`（`_get_pooled_wrapper` 几何键共享池） | md5 `8f88555a0fc7e330ee51255c643796bc` | bind-mount / bake 到 vllm experts 路径 |
| 4 | **flashinfer-0.6.16 overlay** | rebased-experimental 定制树（0.6.16 官方 wheel + 5 fork 补丁 + 58 fork 文件） | tarball md5 `7aac3857220eb5865a70a9ee50e7b8a8` | 目录级 bind-mount / bake 到 `dist-packages/flashinfer` |
| 5 | **overlay-mask**（api_utils 脱敏） | `patches/server-overlay/api_utils.py`（对齐上游 PR#89，掩码 api-key 防落日志） | md5 `d9c7aeb62458848c5547b02c43e4133a` | bind-mount / bake 到 serve utils 路径 |
| 6 | **libncclpin**（shim v8） | CPU 绑核 shim（NCCL→8-9 / EngineCore→15-19） | md5 `ce43c688c5164ac7efd5105c94fdab77` | `LD_PRELOAD`，**二进制不随仓库**（§3.4 源码构建） |
| 7 | **ringonly 库**（NCCL 2.30.7 环序强制） | `patches/ringonly-v5-2026-08-23/`（v5-incremental.patch + 构建记录 + 构建产物参考副本） | v5 md5 `2b8669eceebd633120cd8053a5be3089`（生产 ref `2be94172…`） | `LD_PRELOAD` |

> 基座镜像与 bake 镜像的 digest 均为占位符（`<BASE_IMAGE_DIGEST>` / `<BAKE_IMAGE_DIGEST>`）：发布副本不发布具体内容哈希值，现场以 `docker inspect --format '{{.RepoDigests}}'` 解析。

### 3.2 基座镜像（方案 A/B 共用起点）

```bash
# 拉取并锚定基座（现场替换 REGISTRY_HOST）
docker pull REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0
docker tag REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \
       REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1-base
# 校验：digest 应匹配部署基线 sha256:<BASE_IMAGE_DIGEST>
docker inspect --format '{{index .RepoDigests 0}}' REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0
```

### 3.3 组装材料在仓库内的路径（方案 A/B 共用）

| 组装层 | 仓库内路径 | 容器内目标路径 |
|---|---|---|
| plugin_a1 | `kernels/server-nvfp4/plugin_a1/` | `pip install /tmp/plugin_a1_install`（vLLM plugin 注册） |
| overlay-wsdedup | `patches/server-overlay/flashinfer_b12x_moe.py` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py` |
| flashinfer-0.6.16 | 见 `docs/01-research-reports/fi016-replacement-2026-08-23.md`（tarball 自建） | `/usr/local/lib/python3.12/dist-packages/flashinfer`（目录级） |
| overlay-mask | `patches/server-overlay/api_utils.py` | `/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py` |
| libncclpin | 源码见 `docs/04-issues/code-review-cluster-2026-08-20.md`（v8 布局 8-9/15-19） | `/opt/libncclpin.so` |
| ringonly | `patches/ringonly-v5-2026-08-23/`（patch + 构建记录） | `/opt/nccl-ringonly/libnccl.so.2` |

### 3.4 两个原生库的构建说明（md5 校验，二进制不随仓库）

**libncclpin（shim v8）**：源码级绑核 shim，v8 布局 = NCCL/pt_nccl/pt_tcpstore 线程 → CPU 8-9（isolcpus），EngineCore/VLLM::EngineC → 15-19。构建后 `md5sum libncclpin.so` 应等于 `ce43c688c5164ac7efd5105c94fdab77`；不一致即版本漂移（历史事故，见 `code-review-cluster-2026-08-20.md`）。

**ringonly（NCCL 2.30.7 环序强制）**：从上游 NCCL 2.30.7 源码 + `patches/ringonly-v5-2026-08-23/v5-incremental.patch`（connect.cc 环序强制 32 行 + transport.cc 物理邻接过滤 11 行）构建：

```bash
# 在基座镜像的 CPU 容器内构建（不占 GPU；GLIBC_MAX 约束 ≤2.34）
docker run --rm -v "$PWD/patches/ringonly-v5-2026-08-23":/src \
  REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \
  bash -lc "cd /src && rm -rf build && make -j src.build \
    CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'"
# 产物校验
md5sum /src/build/lib/libnccl.so.2.30.7   # 期望 2b8669eceebd633120cd8053a5be3089
```

> 构建记录/分发清单见 `patches/ringonly-v5-2026-08-23/staging/MD5-RECORD-v5.txt`。

### 3.5 方案 A：运行时挂载（零 bake，随脚本默认）

`scripts/server-production/start_tp4_head.sh` 即运行时挂载参考实现。核心是「**组装层只读注入** + **可写持久缓存**」两组挂载：

**① 组装层注入（只读）**

| 宿主 | 容器内 |
|---|---|
| `<INSTALL_DIR>/models/deepseek-v4-flash-0731` | `/models`（权重，ro） |
| `<INSTALL_DIR>/envs/nvcc_wrapper.py` | `/tmp/env-e-build/nvcc_wrapper.py`（ro） |
| `<INSTALL_DIR>/overlay-wsdedup/flashinfer_b12x_moe.py` | `…/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`（ro） |
| `<INSTALL_DIR>/nvfp4/flashinfer-0.6.16/flashinfer` | `…/dist-packages/flashinfer`（ro，目录级） |
| `<INSTALL_DIR>/overlay-mask/api_utils.py` | `…/vllm/entrypoints/serve/utils/api_utils.py`（ro） |
| `<INSTALL_DIR>/lib/libncclpin.so` | `/opt/libncclpin.so`（ro，LD_PRELOAD） |
| `/opt/nccl-ringonly` | `/opt/nccl-ringonly`（ro，LD_PRELOAD） |

**② 可写持久缓存**

| 宿主 | 容器内 |
|---|---|
| `$HOME/vllm-cache` | `/root/.cache/vllm`（rw） |
| `$HOME/tilelang-cache` | `/root/.tilelang/cache`（rw） |
| `$HOME/b12x-cache` | `/root/.cache/b12x`（rw） |
| `$HOME/flashinfer-cache` | `/root/.cache/flashinfer`（rw） |
| `~/vllm-logs` | `/var/log/vllm`（rw，NCCL 日志） |

**③ 容器运行时骨架（节选，完整 ENV_ARGS 见 §5.2 与 `start_tp4_head.sh`）**

```bash
docker run -d --name vllm-tp4-rank0 \
  --restart no --network host --ipc=host --privileged --gpus all \
  --shm-size=64gb --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --cpuset-cpus=1-19 \
  --health-cmd "curl -sf -o /dev/null -m 5 http://127.0.0.1:8001/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 5 --health-start-period 900s \
  -v <INSTALL_DIR>/models/deepseek-v4-flash-0731:/models:ro \
  -v <INSTALL_DIR>/overlay-wsdedup/flashinfer_b12x_moe.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro \
  -v <INSTALL_DIR>/nvfp4/flashinfer-0.6.16/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer:ro \
  -v <INSTALL_DIR>/overlay-mask/api_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py:ro \
  -v <INSTALL_DIR>/lib/libncclpin.so:/opt/libncclpin.so:ro \
  -v /opt/nccl-ringonly:/opt/nccl-ringonly:ro \
  -v "$HOME/vllm-cache:/root/.cache/vllm:rw" \
  -v "$HOME/flashinfer-cache:/root/.cache/flashinfer:rw" \
  -v ~/vllm-logs:/var/log/vllm \
  --entrypoint /bin/bash \
  REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \
  -lc "rm -rf /tmp/plugin_a1_install; cp -r <INSTALL_DIR>/nvfp4/plugin_a1 /tmp/plugin_a1_install; pip install --no-deps -q /tmp/plugin_a1_install; vllm serve ... "
```

> worker（rank1-3）与 head 共用同一套组装层注入；rank 差异仅 `NODE_RANK`/`VLLM_HOST_IP`/`NCCL_IB_HCA`（per-rank 邻接口）。生产脚本为权威实现，本节为节选；完整 `ENV_ARGS` + serve 命令见 `start_tp4_head.sh`（§5 参数表）。

### 3.6 方案 B：bake 自包含镜像 LuZ0.3.1（推荐生产/发布）

将全部组装层 bake 进镜像，产物 `REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`（自包含：FI 0.6.16 树 + ws-dedup overlay + 池化插件全 bake，约 34.4 GB，digest `sha256:<BAKE_IMAGE_DIGEST>`）。

```dockerfile
# Dockerfile.luz031 — 自包含恢复镜像（生产 bake 方案）
FROM REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 AS base

# --- 组装层 1：overlay-wsdedup（池补丁）---
COPY patches/server-overlay/flashinfer_b12x_moe.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py

# --- 组装层 2：flashinfer-0.6.16 定制树（目录级替换）---
# 前置：现场从 md5=7aac3857 的 tarball 解包（见 fi016-replacement 报告）
COPY nvfp4/flashinfer-0.6.16/flashinfer/ \
     /usr/local/lib/python3.12/dist-packages/flashinfer/

# --- 组装层 3：overlay-mask（api_utils 脱敏）---
COPY patches/server-overlay/api_utils.py \
     /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py

# --- 组装层 4：plugin_a1（W4A4 池化插件，vLLM general_plugins 注册）---
COPY kernels/server-nvfp4/plugin_a1/ /opt/plugin_a1/
RUN pip install --no-deps -q /opt/plugin_a1

# --- 组装层 5/6：原生库（可选 bake，或运行时挂载）---
# 二进制不随仓库：现场构建后 COPY /opt/libncclpin.so 与 /opt/nccl-ringonly/ 进镜像，
# 或保持运行时 -v 挂载（推荐，便于回退锚点管理）
COPY lib/libncclpin.so /opt/libncclpin.so
COPY nccl-ringonly/ /opt/nccl-ringonly/

# --- 运行参数（与 §5 参数表一致）---
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    DG_JIT_NVCC_COMPILER=/tmp/env-e-build/nvcc_wrapper.py \
    DG_JIT_USE_NVRTC=0 \
    DSPARK_SLOT_CLAMP=1 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    LD_PRELOAD="/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2" \
    NCCL_ALGO=RING \
    NCCL_CROSS_NIC=1 \
    NCCL_IB_GID_INDEX=3 \
    NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 \
    NCCL_IB_TIMEOUT=1000 \
    NCCL_IB_RETRY_CNT=7 \
    NCCL_IB_TOS=46 \
    NCCL_CUMEM_HOST_ENABLE=0 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    NCCL_MIN_NCHANNELS=4 \
    NCCL_TUNER_THRESHOLD=40960 \
    NCCL_BUFFSIZE=8388608 \
    NCCL_MAX_NCHANNELS=4 \
    NCCL_NET=IB \
    NCCL_IB_SUBNET_AWARE_ROUTING=1 \
    NCCL_NET_PLUGIN=none \
    NCCL_IB_MERGE_NICS=0 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_DISABLE_PYNCCL=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=600 \
    VLLM_USE_B12X_MOE=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_DSPARK_LOCAL_ARGMAX=1 \
    VLLM_TRITON_MLA_SPARSE=1 \
    VLLM_MOE_W4A4=2 \
    VLLM_MOE_W4A4_MIN_M=3072 \
    VLLM_MOE_W4A4_CG=1 \
    VLLM_B12X_SHARED_WRAPPER=1 \
    PYTHONPATH=/opt/kernel1:/opt/kernel2
```

构建 + 分发：

```bash
# 构建（增量层，约 34.4GB 产物）
docker build -f Dockerfile.luz031 -t REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1 .
# 锚定基座 tag（回滚锚点）
docker tag REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 \
       REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1-base
# push + 四机 pull
docker push REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1
for h in node02 node03 node04; do ssh $h "docker pull REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1"; done
# 校验 digest
docker inspect --format '{{index .RepoDigests 0}}' REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1
```

> bake 后启动脚本的 `IMG` 指向 `…:LuZ0.3.1` 即可；运行时不再需要 overlay 挂载（已固化进镜像），仅保留 `/models` 权重挂载与可写缓存卷。

### 3.7 权重 / checkpoint（独立获取，不随仓库/镜像）

- 模型：**DeepSeek-V4-Flash-0731**（MXFP4/nvfp4 权重）。
- 获取方式：独立渠道获取 checkpoint，校验格式（FP8 block linear + MXFP4 experts）后放置 `<INSTALL_DIR>/models/deepseek-v4-flash-0731/`（含 `config.json`）。
- 不随仓库、不 bake 进镜像；容器内挂载为 `/models`（只读）。
- 换权重版本（如 0731 之后 ckpt）需重新走质量门 + needle 抽验（§6）。

---

## 4. 启动部署（head-first 编排 + systemd 自愈）

### 4.1 head-first 四机编排：`start_tp4_cluster.sh`（R12）

**唯一推荐冷启动/重建入口**（在 `node01`/head 上执行，幂等）：

```bash
cd <INSTALL_DIR>/scripts && export VLLM_API_KEY="<API_KEY>" && bash start_tp4_cluster.sh
```

R12 版执行序列：

1. GPU-gate（`nvidia-smi` ≤180s）→ 四机 `check_vllm_script.sh` 前置自检（A1-A3/C，B1 对编排器豁免）。
2. 清理四机残留容器（幂等）→ head `:8001` 释放校验。
3. **step 1/4**：启动 head（rank0）容器，后台日志 `$HOME/start_tp4_cluster.log`。
4. **step 2/4**：轮询 head **TCPStore `:25999`** 就绪（连续 2 次探测，最长 600s）。
5. **step 2.5/4（R12 门禁）**：复核 TCPStore 就绪（`TCPSTORE_GATE_WAIT` 默认 300s）——**门禁信号为 TCPStore 而非 B12X 日志**（修复 8/20 冷启动死锁：head 引擎核心初始化需 4 rank 入 NCCL 域，原"等 B12X_MXFP4 日志"永不出现 → 互等 300s）。
6. **step 3/4**：按环序错峰启动 worker（rank1=`node02` → rank2=`node04` → rank3=`node03`），每 worker 间隔 `B12X_JIT_STAGGER`（默认 20s），防多 worker 并行撞 B12X JIT 竞态；worker 通过 ssh 携带 `VLLM_API_KEY`/`NODE_RANK`/`VLLM_HOST_IP`/`NCCL_IB_HCA`（per-rank 邻接口）下发。
7. **step 4/4**：worker 容器出现门禁（每 rank 120s）→ 轮询 head `Application startup complete`（≤15min）→ READY。

调参环境变量：`TCPSTORE_GATE_WAIT`、`B12X_JIT_STAGGER`（若仍偶发 JIT 瞬时失败可上调 20→30s）。

### 4.2 systemd 自愈链（三件套）

| 单元 | 位置 | 作用 |
|---|---|---|
| `vllm-tp4-head.service` | `node01` | head monitor（`docker wait` 跟随），容器缺失 → 清 worker → 重建 head |
| `vllm-tp4-worker.service` | `node02/03/04` | worker monitor，带互杀守卫（head API 健康 + 集群成形才允许动 head） |
| `vllm-healthcheck.timer` | `node01` | 只读健康探针（`healthcheck.sh`），异常触发重建 |

- 容器 `--restart no`：docker daemon 不直拉，生命周期完全由 systemd 掌控（防重启风暴）。
- monitor 恒 exit 非零 → `Restart=always` + `RestartSec=15`；`StartLimitIntervalSec=1800` / `StartLimitBurst=20`（防崩溃循环误入 failed）。
- **正确重启姿势（唯一推荐）**：服务保持 active，`docker rm -f vllm-tp4-rank0` → head-first 全链重建（monitor 清 worker 容器 → 各机 systemd 自愈重建，worker 带 TCPStore 门禁防冷启动互杀，全程零人工干预）。
- **注意**：`systemctl stop vllm-tp4-head.service` ≠ 容器停（服务停不停容器），且停服务后 workers 不会自动跟随——配置变更请用 `docker rm -f rank0` 姿势，不要停服务。

### 4.3 停机 / 恢复顺序

**停机维护（防自愈误触发）**：

```bash
# worker 先 → head 后；每台 systemctl stop 后等 15-30s
for h in node03 node04 node02; do ssh $h "sudo systemctl stop vllm-tp4-worker.service"; done
ssh node01 "sudo systemctl stop vllm-tp4-head.service"
# 确认无残留容器 / 无 active 服务
```

**恢复（head-first）**：

```bash
# 整机重启后开机顺序：node01 → node02 → node03 → node04（systemd 已 enable，勿手动启）
ssh node01 "cd <INSTALL_DIR>/scripts && export VLLM_API_KEY='<API_KEY>' && bash start_tp4_cluster.sh"
# 复核：8001=200 + 四机容器 Up(healthy) + NCCL banner（2.30.7 + cuda13.0）
```

> 基准作业纪律：测量前 `systemctl stop vllm-healthcheck.timer`（防探针误判触发重建打断测量），测量后恢复（§6.4）。

---

## 5. 生产参数表（脱敏通用）

> 来源：`start_tp4_head.sh` serve 命令 + ENV_ARGS（权威）。以下为 LuZ0.3.1 生产采纳值；**调参一次只改一档**，改后 `.bak` 留档 + `check_vllm_script.sh` 通过 + head 先行验证。

### 5.1 serve 参数

| 参数 | 生产值 | 含义 | 调参指引 |
|---|---|---|---|
| `--max-model-len` | `600000` | 最大上下文（600K） | 业务长上下文需求；上调需核对 KV 预算（KV tokens ≥5.7M 门） |
| `--max-num-seqs` | `12` | 最大并行序列数 | 并发扩容优先调此项；上调增大显存/调度压力 |
| `--gpu-memory-utilization` | `0.82` | GPU 显存利用率 | 0.80→0.82 曾 +0.23M KV；上调需观察 KV 回补与碎片 |
| `--long-prefill-token-threshold` | `4096` | 长 prefill 阈值 | 4096 为采纳值（旧 2048）；越大越倾向 chunked prefill |
| `--max-num-batched-tokens` | `4096` | 单步 batched token 上限 | 与 threshold 同步；调低控延迟、调高提吞吐 |
| `--speculative-config` | dspark n=7, probabilistic | 投机解码（DSpark MTP） | n=7 采纳；调高提升接受率但增显存/延迟 |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | KV 缓存精度 | 保持默认；换 dtype 需全量回归 |
| `--moe-backend` | `flashinfer_b12x` | MoE 后端 | 保持默认 |
| `--max-cudagraph-capture-size` | `96` | CUDAGraph 捕获上限 | capture-sizes 1..96（16 档）；业务序列长度分布变化时核对 |
| `--distributed-timeout-seconds` | `300` | 分布式超时 | 冷启动互等防护；保持默认 |
| `--scheduling-policy` | `priority` | 调度策略 | 保持默认 |
| `--enable-prefix-caching` | on | 前缀缓存 | 长上下文命中率高时收益大 |

### 5.2 关键 env

| env | 生产值 | 含义 | 调参指引 |
|---|---|---|---|
| `VLLM_MOE_W4A4` | `2`（full） | MoE W4A4 量化模式（0=off 1=hybrid 2=full） | **LuZ0.3.1 核心**；`0` = 降级 W4A16（单开关回滚） |
| `VLLM_B12X_SHARED_WRAPPER` | `1` | 池补丁开关（几何键共享池） | 与 overlay-wsdedup 配套；0=off 零行为 |
| `NCCL_CUMEM_HOST_ENABLE` | `0` | 关闭 host-side CUMEM | 08-24 落地；保持关闭 |
| `NCCL_ALGO` / `MIN/MAX_NCHANNELS` | `RING` / `4/4` | 环序强制 + 4 通道 | ring-only 定制库配套；勿随意改通道数 |
| `NCCL_BUFFSIZE` | `8388608`（8MB） | NCCL 通信缓冲 | 已调优（368KB allreduce 923→173us）；勿改 |
| `LD_PRELOAD` | `/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2` | 绑核 shim + ring-only | 缺任一即启动失败（checker B1 强制项） |
| `DG_JIT_NVCC_COMPILER` | `/tmp/env-e-build/nvcc_wrapper.py` | sm_120f→sm_121a JIT wrapper | 缺则 JIT 编译失败（checker 校验） |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | 屏蔽 FI 版本检查 | dist-info 滞后 0.6.15 属已知（`__version__`=0.6.16 为准） |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | 前缀缓存保留间隔 | 保持默认 |
| `VLLM_DP_MASTER_IP` / `VLLM_HOST_IP` | `<NODE_IP>`（head） | 多节点 coord_store / 数据面地址 | worker 由编排脚本注入 per-rank IP |

---

## 6. 验证与质量门

### 6.1 就绪验证（部署后必跑）

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://<NODE_IP>:8001/health   # 200
systemctl is-active vllm-tp4-head.service                                 # active
for h in node02 node03 node04; do ssh $h "systemctl is-active vllm-tp4-worker.service"; done  # 全 active
for r in 0 1 2 3; do docker inspect -f '{{.State.Health.Status}}' vllm-tp4-rank${r}; done        # 全 healthy
# 启动核验清单（含 flashinfer 版本项，防误回滚复发）
docker exec vllm-tp4-rank0 python -c "import flashinfer; print(flashinfer.__version__)"   # 0.6.16
```

### 6.2 质量门（quality_gate.py，4/4）

```bash
# 首次/换基座：抓参考快照
VLLM_API_KEY="<API_KEY>" python3 <INSTALL_DIR>/scripts/quality_gate.py capture
# 回归：与 reference-latest 比对
VLLM_API_KEY="<API_KEY>" python3 <INSTALL_DIR>/scripts/quality_gate.py compare
# 期望：exact_match 4/4（fox_repeat/count/code/list，temp=0 逐字一致）；
#       不一致时包络判据：token 级 top-1 logprob 总和漂移 ≤1% 兜底
```

- 退出码 `0=PASS 1=FAIL 2=用法/环境错误`。LuZ0.3.1 采纳验收：**4/4 exact match，own_stable 4/4**。
- 参考快照管理：`<INSTALL_DIR>/backup/quality-gate/`（latest 机制；换基座时 capture 并保留旧快照）。

### 6.3 模式探针（stall / TTFT）

- **stall 探针**：重启后 3× 短 4K 请求，TTFT <6s 且 `SUSPECT=False` 为干净（LuZ0.3.1 实测 2.77-3.12s）。
- **模式探针**：首 4K TTFT ≈2.79s（W4A4-fast 类），用于区分正常/异常启动路径。
- 工具：`scripts/server-production/quality_gate.py` 同类请求即可（或现场探针脚本）。

### 6.4 DE 抽验（decode 接受率归一）

- 口径：`step_eff = tput / tokens_per_step`（接受率归一），4 轮中位。
- LuZ0.3.1 参考：C1 step_eff **18.2**（中性）、C12 step_eff **80.2**（落已知 W4A4 full decode 代价带）。
- **基准纪律**：测量前 `systemctl stop vllm-healthcheck.timer`，测量后恢复（勿忘，属三件套）。

### 6.5 最终指标引用

性能/资源权威口径见 **`docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`**（含 PR 四档、并发、DE、KV、质量门、400K 长上下文）。注意：旧口径（08-05 `raw_final_matrix.json` 等）与本表**不可混用**。

---

## 7. 回滚

### 7.1 一键恢复：`restore_luz031.sh`

位置：`<INSTALL_DIR>/backup/luz031-checkpoint-20260823/restore_luz031.sh`（状态快照 20MB/18 文件 + md5-manifest 17 项）。

```bash
cd <INSTALL_DIR>/backup/luz031-checkpoint-20260823/
bash restore_luz031.sh --dry-run      # 演练（只打印，不执行）
bash restore_luz031.sh                # 实际恢复：md5 核验 → 分发 → FI 树核验 → overlay → checker → Prometheus → head-first 重建 → 启动核验 → 三件套恢复
```

9 步：md5 核验 → 分发 → FI 0.6.16 树核验 → overlay → checker → Prometheus 恢复 → head-first 重建 → 启动核验 → 自愈链三件套恢复。

### 7.2 镜像回滚链（bake 方案）

| 目标 | 动作 |
|---|---|
| 回到 **LuZ0.3.1 自包含镜像**（回滚到本版本） | `docker pull REGISTRY_HOST:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`（digest `sha256:<BAKE_IMAGE_DIGEST>`）→ 改脚本 IMG → head-first 重建 |
| 回到 **W4A16 基线**（B1） | 四机 `cp .bak-luz031-20260823`（head/worker/checker）+ 插件恢复 `.bak-wsdedupl3-20260823`（原版 c2d1de3d）+ checker 过 + head-first 重建 |
| **单开关降级**（不回滚插件） | `sed -i "s/VLLM_MOE_W4A4=2/VLLM_MOE_W4A4=0/"` 四机脚本 → W4A4 关闭（池 overlay env=0 零行为） |
| **util 单独回退** | `sed -i "s/gpu-memory-utilization 0.82/gpu-memory-utilization 0.80/"`（脚本 + checker 同步） |

> 恢复后自愈链三件套核验：`vllm-tp4-head.service` active（01）+ 三 `vllm-tp4-worker.service` active（02/03/04）+ `vllm-healthcheck.timer` active（01）+ `/health` 200。
> **跨窗口恢复教训**：核对 `.bak` 快照的时序覆盖范围（FI 0.6.16 误回滚事件，见 `LuZ0.3.1-release-notes.md` §6.1）。

---

## 8. 安全说明

- **API key**：由部署者自行生成（`VLLM_API_KEY` / `--api-key`），本文不含真实 key，仅占位符 `<API_KEY>`。参考生成：`openssl rand -hex 24` 前缀 `sk-`。
- **日志脱敏**：overlay-mask（api_utils）在 serve 层掩码 api-key（对齐上游 PR#89）；head 脚本对 `--api-key` 值做 `********` 掩码后再打印，防密钥落操作日志。
- **开放端口**：
  - `8001`：vLLM OpenAI API（对外，需鉴权）；建议仅管理网可达 + 白名单放行。
  - `25999`：TCPStore 控制面（四机间）；不对外暴露。
  - `8191`：Prometheus（监控）；`3000`：Grafana（若部署）；建议管理网 + 鉴权。
  - 数据面 RoCE（`NCCL_IB_HCA` 对口）：不路由到公网。
- **鉴权建议**：API 端口强制 `--api-key`；网关层（litellm 等）额外 Bearer；外部访问走白名单/防火墙（`iptables-save-custom.sh` 固化）。
- **容器安全**：`--privileged --gpus all`（DGX 运行所需）；保持镜像只读挂载组装层 + 可写卷最小化；不 bake 密钥进镜像。
- **敏感项纪律**：本仓库为开源发布副本，内网 IP/主机名/用户名/路径/密钥均占位符化（`REDACTION-MAP.md`）；现场替换占位符时勿将真实值回写仓库。

---

## 9. 参考链接

- 最终指标：`docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`
- 版本说明/回滚链：`docs/07-deployment/LuZ0.3.1-release-notes.md`
- 落地报告：`docs/07-deployment/luz031-deployment-2026-08-23.md`
- Runbook（重启姿势/基准纪律/needle 口径）：`docs/07-deployment/runbook-tp4-v1.5-2026-08-12.md`
- 自愈：`docs/07-deployment/ops/self-recovery.md`｜容错：`docs/07-deployment/ops/fault-tolerance.md`｜维护：`docs/07-deployment/ops/maintenance-plans.md`｜工具索引：`docs/07-deployment/ops/tools-index.md`
- R12 门禁修复：`docs/01-research-reports/b12x-gate-fix-2026-08-24.md`
- FI 0.6.16 替换：`docs/01-research-reports/fi016-replacement-2026-08-23.md`
- ring-only v5：`patches/ringonly-v5-2026-08-23/ringonly-v5-brief-2026-08-23.md`
- 脚本：`scripts/server-production/start_tp4_cluster.sh`、`start_tp4_head.sh`、`check_vllm_script.sh`、`healthcheck.sh`、`quality_gate.py`
