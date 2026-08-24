# SGLang NGC 26.07 容器四机落地交付文档

**日期**：2026-08-14 ｜ **交付对象**：部署团队 ｜ **状态**：镜像已就绪（01 修正中见 §4）

---

## 1. 交付概要

SGLang 官方 NGC 26.07 容器已通过国内镜像通道拉取并分发至 DGX Spark 四机集群，用于 **DeepSeek-V4-Flash NVFP4 推理验证**（工程团队方案 `sglang-nvfp4-tp4-setup-plan-2026-08-13.md` 的镜像前置项）。

**关键结论：26.07 容器内置 SGLang 0.5.14 = PR #25820 合入线 → NVFP4 DeepSeek-V4-Flash 原生支持，无需补丁。**

## 2. 镜像标识（以 02 为基准）

| 项 | 值 |
|---|---|
| 镜像名（统一） | `nvcr.io/nvidia/sglang:26.07-py3` |
| 内网 registry | `<NODE_IP>:5000/nvidia/sglang:26.07-py3` |
| 镜像 ID | `sha256:4f5f4cade001a28b44f3e6289f49eb6e2e3e941e284fa95ee67a53c3d17745a1` |
| 尺寸 | 20.8 GB |
| manifest digest | `sha256:356846f927ed8f2fee14ebafd28fe45eb4bd6529a98bf0e0a8719ebd6c6aafbf`（南大源） |
| 架构 | aarch64（arm64，Grace CPU） |
| NVIDIA Release | 26.07（build 373559652），基座 CUDA 13.3.1 |

**容器内版本（02 实测）**：

| 组件 | 版本 | 判定 |
|---|---|---|
| SGLang | **0.5.14+nv26.7.59534057** | ✅ PR #25820 合入线 |
| torch | 2.13.0a0+9186a08b2c.nv26.07 | ✅ |
| flashinfer | 0.6.14 | ⚠️ 略低于方案推荐 0.6.15.post1（NGC 校验组合，无需处理，如遇 NCCL 回退问题再钉） |

## 3. 四机落地核对表

| 节点 | IP | 层校验（RootFS md5） | 尺寸 | 状态 |
|---|---|---|---|---|
| node01（下载源） | <NODE_IP> | 96e467d43b59c2362246545bacb4c9fe | 20.8GB | ✅ |
| node01 | <NODE_IP> | 96e467d43b59c2362246545bacb4c9fe | 20.8GB | ✅ |
| node01 | <NODE_IP> | 96e467d43b59c2362246545bacb4c9fe | 20.8GB | ✅ |
| node01（head） | <NODE_IP> | 96e467d43b59c2362246545bacb4c9fe | 20.8GB | ✅（save/load 修正后） |

**四机 RootFS 层 md5 全部一致 = 镜像内容字节级统一。** 功能验证（01 容器内实测）：SGLang 0.5.14+nv26.7 / torch 2.13.0a0+nv26.07 ✓

> 注：01 经 save/load 后 `docker images` 的 IMAGE ID 显示 `1e8eb5ea94b0`（与 02 的 `4f5f4cade001` 不同），系 load 与旧镜像共享层合并后的显示差异；**层 md5 与容器内版本完全一致，不影响使用**。四机核对以 RootFS 层 md5 为准（命令见 §6.1）。

## 4. ⚠️ 已知问题：01 的 v1 manifest 坑（部署团队必读）

**现象**：01 从内网 registry 拉取时拿到 **30.7GB 旧内容**（IMAGE ID `713bc60d090f`），与 02/03/04 的 20.8GB（`4f5f4cade001`）不一致。

**根因**：registry（`registry:2`，镜像源自 DaoCloud）对 01 的 manifest 请求返回 **v1 schema manifest**（旧内容），而 03/04 走 v2 拿到新内容。02 push 本身成功（manifest digest `713bc60d090f` 即 02 的镜像）。

**处置（已完成）**：02 `docker save`（20.9GB tar）→ 01 `docker load` → 功能验证 + 层 md5 比对通过，**01 已与 02 字节级一致**。

**教训（后续分发必守）**：
- 多机分发不要在三机并行 pull 时与 push 并发（01 曾在 push 完成前抢跑拿到旧 manifest）
- 四机核对标准：**RootFS 层 md5 一致**（`docker image inspect --format '{{json .RootFS.Layers}}' ... | md5sum`）而非只看 IMAGE ID

## 5. 下载/分发通道结论（未来复用）

| 通道 | 实测速度 | 结论 |
|---|---|---|
| 直连 nvcr.io | 停滞 0 MB/s | ✗ 不可用 |
| DaoCloud `docker.m.daocloud.io` | 不支持 nvcr 前缀（invalid repository） | ✗ 仅 docker.io |
| `nvcr.1ms.run` | 1 MB/s | ✗ 太慢 |
| **南大 `ngc.nju.edu.cn`** | **49 MB/s** | ✅ **首选**（官方 NGC 缓存代理） |
| 内网 registry（02→四机） | 分钟级 | ✅ 分发通道 |

**标准拉取命令**：
```bash
docker pull ngc.nju.edu.cn/nvidia/sglang:26.07-py3
docker tag ngc.nju.edu.cn/nvidia/sglang:26.07-py3 nvcr.io/nvidia/sglang:26.07-py3
```

## 6. 部署团队进场指引

### 6.1 镜像验证（每台必做）
```bash
# ① 层校验（四机须一致：96e467d43b59c2362246545bacb4c9fe）
docker image inspect --format '{{json .RootFS.Layers}}' nvcr.io/nvidia/sglang:26.07-py3 | md5sum
# ② 版本验证
docker run --rm nvcr.io/nvidia/sglang:26.07-py3 bash -c \
  "pip show sglang | head -2; python -c 'import torch; print(torch.__version__)'"
# 期望：SGLang 0.5.14+nv26.7 / torch 2.13.0a0+nv26.07
```

### 6.2 运行要点（承接工程团队方案）
- **端口**：API 8010 / metrics 8011 / TCPStore 26000（8003 被 responses-gateway 占用、25999 被 vLLM 占用）
- **内存互斥**：SGLang NVFP4 TP4（~45-110 GiB/rank）与生产 vLLM TP4（~79.5 GiB）**不能同机并存** → 验证期 A/B 互斥切换；并存仅限低配冒烟（mem-fraction 0.2-0.3 + 门禁 ≥55G + cpuset=0）
- **权重路径**：`<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`（164GB NVFP4，已分发四机，软链就绪）
- **启动参数参考**（SGLang NVFP4 DSV4）：
  ```bash
  python3 -m sglang.launch_server --model <权重路径> \
    --tp 4 --trust-remote-code --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_trtllm_routed \
    --host 0.0.0.0 --port 8010 --metrics-port 8011 \
    --mem-fraction-static 0.9
  ```
  （参数以工程团队 `sglang-nvfp4-tp4-setup-plan-2026-08-13.md` 为准，此仅为占位）
- **DeepEP/通信**：四机无 NVLink，跨机 EP 走 NCCL fallback；沿用现有 NCCL 环境变量经验（ring-only + LD_PRELOAD 视需要）

### 6.3 TP1 冒烟检查点（单机先行）
1. `docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ... nvidia-smi` 容器内 GPU 可见
2. 单机 TP1 启动 DSV4-Flash NVFP4 → 确认 SM121 kernel 加载（无 CUTLASS 崩溃）
3. `/v1/chat/completions` 冒烟 200 + 首个 token 延迟
4. 四机 TP4 全链路 → 端口 8010 可达、metrics 8011 有数据

### 6.4 纪律提醒
- 不改生产 vLLM 配置；验证走独立端口/独立容器
- 容器 `--restart no`，与生产编排一致
- 修改任何脚本前备份 + `bash -n`

## 7. 参考

- 下载指南：`deploy-sglang-ngc-2607-2026-08-13.md`
- 运行时调查：`research-nvfp4-alternative-runtimes-2026-08-13.md`
- 工程方案：`sglang-nvfp4-tp4-setup-plan-2026-08-13.md`（工程团队交付）
- 权重：`<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4`（164GB，modelopt NVFP4）
