# SGLang NGC 26.07 容器下载与集群部署指南

**日期**：2026-08-13 ｜ **目标镜像**：`nvcr.io/nvidia/sglang:26.07-py3` ｜ **适用**：DGX Spark 四机集群（aarch64）

---

## 1. 镜像信息（已核实）

| 项 | 值 |
|---|---|
| NGC 路径 | `nvcr.io/nvidia/sglang:26.07-py3`（NGC 目录页确认存在，`latest` 即此 tag） |
| 基座 | **CUDA 13.3.1** |
| 覆盖硬件 | Blackwell 全家族：B300/GB300、RTX PRO 6000 Blackwell Server Edition、**DGX Spark**、Jetson Thor |
| 架构 | multi-arch：**amd64 + aarch64**（DGX Spark 需 arm64 层） |
| 版本语义 | 26.07 = NGC 容器版本（2026 年 7 月构建），**不是** SGLang 上游版本号（上游为 0.5.x） |

## 2. 前置检查（无需升级）

- **驱动**：26.07 基座 CUDA 13.3.1 属于 13.x 系列 → **最低驱动 580 分支**（官方 minor-version 兼容表：13.x >= 580）。集群现驱动 **580.173.02 = R580 分支，满足 ✓**，无需升级驱动。
- **登录凭据**：NGC 免费账号（https://org.ngc.nvidia.com/setup/api-key 生成 API Key）。

## 3. 下载（两步：登录 + pull）

```bash
# ① 登录 NGC（用户名固定 $oauthtoken，密码 = NGC API Key）
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key>

# ② 拉取（在 aarch64 机器上执行会自动取 arm64 层）
docker pull nvcr.io/nvidia/sglang:26.07-py3
```

⚠️ **架构陷阱**：若在本机（x86_64 Windows）执行 pull 会拉到 **amd64** 层，拷到 Spark（aarch64）无法运行。两个正确姿势：
- 直接在 **02（aarch64，有外网）** 上 pull；或
- x86 机器上必须显式 `docker pull --platform linux/arm64 nvcr.io/nvidia/sglang:26.07-py3`

## 4. 国内网络

NGC（nvcr.io）直连在国内通常慢或不通。本项目 **02 有稳定外网**（8/13 已实测从 hf.rimuru.work 下载 164GB 权重），**推荐路径：02 直连 pull**。若 02 也慢，配置 docker daemon 代理后重试。

## 5. 四机分发（走内网 registry，复用现有基建）

```bash
# 02（下载机）：
docker tag nvcr.io/nvidia/sglang:26.07-py3 <NODE_IP>:5000/nvidia/sglang:26.07-py3
docker push <NODE_IP>:5000/nvidia/sglang:26.07-py3

# 01/03/04：
docker pull <NODE_IP>:5000/nvidia/sglang:26.07-py3
docker tag <NODE_IP>:5000/nvidia/sglang:26.07-py3 nvcr.io/nvidia/sglang:26.07-py3
```

（四机均已配置 <NODE_IP>:5000 为 insecure-registry，免额外配置。）

## 6. 运行冒烟（单机先行）

```bash
docker run --rm --gpus all --ipc=host --network=host \
  nvcr.io/nvidia/sglang:26.07-py3 \
  nvidia-smi
# 若报 "CUDA driver version is insufficient"（不应发生，580 已满足）：
#   装 cuda-compat-13-3 或用 NGC 26.02（CUDA 13.1.1）降级验证
```

## 7. 版本核对（拉取后必做）

```bash
docker run --rm nvcr.io/nvidia/sglang:26.07-py3 \
  bash -c "pip show sglang | head -2; python -c 'import torch; print(torch.__version__)'"
```

## 8. 备注

- 26.07 是 **stable 非 nightly**；nightly 镜像另在 `nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest`（不推荐生产）。
- 若后续要跟上游主线（非 NGC 校验栈），可改用 Docker Hub `lmsysorg/sglang:latest`（社区镜像，非"官方 26.07"）。
- 26.07 属 CUDA 13.3 系，与此前 NVFP4 调查结论兼容（cuBLASLt 13.6.0.2 已含 grouped GEMM NVFP4 支持迭代；SM120 侧仍以 SGLang flashinfer 路径为准）。
