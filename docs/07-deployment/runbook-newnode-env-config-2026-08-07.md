# 新节点环境配置手册（gx10-3f4d / <MGMT_OCTET>）

> 面向对象：SRE / 后续新节点接入人员
> 适用范围：DGX Spark GB10 新节点从裸机到可用基线的环境配置
> 配套文档：《分发机制操作手册》 runbook-distribution-ops-2026-08-07.md
> 最近更新：2026-08-07

---

## 1. 概述

本文档描述新节点 **gx10-3f4d（<NODE_IP>）** 的完整环境配置过程与结果，作为集群新增节点接入的**权威基线**。所有配置均已实施并验证，命令可直接复制执行（面向 <MGMT_OCTET> 及后续同类新节点复用）。

新节点是 DGX Spark GB10 单机（aarch64），当前仅接入 Wi-Fi 管理网（<NODE_IP>/24），尚未有线/RoCE 组网。配置目标：使其与既有 worker（<MGMT_OCTET>）基线对齐，具备 Docker + NVIDIA 容器栈 + 镜像/权重分发接入能力。

**核心结论**：<MGMT_OCTET> 已通过全部验收，可正常拉取内网 registry 镜像、挂载 <MGMT_OCTET> NFS 权重、运行 GPU 容器。唯一已知限制为当前仅 Wi-Fi 链路，大文件传输受带宽约束（详见第 7 节）。

---

## 2. 节点清单与现状表

| 节点 | 角色 | IP | 硬件/配置 | 系统 | GPU 驱动 | 存储 | 网络 | 备注 |
|---|---|---|---|---|---|---|---|---|
| **gx10-3f4d** | **新节点** | <NODE_IP>/24 | DGX Spark GB10，aarch64，121Gi 统一内存 | Ubuntu 24.04.4 | 580.173.02 | NVMe 931.5G（可用 822G） | 仅 Wi-Fi（无有线/RoCE） | 用户 <USER>（sudo+docker gid 988） |
| **<MGMT_OCTET> worker** | 分发源/worker | <NODE_IP> | 3.6T 大盘 | — | — | 权重 312G @ /home/<USER>/models | 有线 + RoCE | registry:2 :5000、NFS 导出、14 容器（embed 8020 / litellm 4000 / 8003 网关等） |
| **<MGMT_OCTET> head** | 备份接收端 | <NODE_IP> | 3.6T | — | — | 备份 ~/backup-images/ | 有线 + RoCE | 接收 <MGMT_OCTET> save 的 vLLM+embed 镜像备份 |
| 本机（管理机） | 运维入口 | — | — | — | — | — | — | SSH config 别名 gx10-55，三机互信 |

> RoCE 平面：<MGMT_OCTET>/<MGMT_OCTET> 双机 10.100.136.x，TP=2 vLLM（当前停机，配合视频工作流）。<MGMT_OCTET> 未接入 RoCE。

---

## 3. 配置逐项说明

### 3.1 Docker 修复（buildkit 损坏库清理）

**背景**：<MGMT_OCTET> 初始 Docker 服务异常（buildkit 依赖库损坏，服务无法启动）。

**处理方式**：清理损坏的 buildkit 库文件后 Docker 恢复正常。

**当前状态**：

```bash
systemctl is-active docker    # active
systemctl is-enabled docker   # enabled
docker version --format '{{.Server.Version}}'   # 29.2.1
```

### 3.2 daemon.json（/etc/docker/daemon.json）

配置要点（已生效）：

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "insecure-registries": ["<NODE_IP>:5000"],
  "registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerproxy.com"],
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

说明：
- **日志轮转**：单个容器日志 100m × 3 份，防 /var/lib/docker 被日志撑爆。
- **insecure-registries**：内网 registry `<NODE_IP>:5000` 无 TLS/认证（内网信任，见 ADR-4），必须显式声明，否则 pull 被拒。
- **registry-mirrors**：外网拉取兜底（daocloud / dockerproxy），用于无内网缓存的新镜像。
- **nvidia runtime**：GPU 容器透传必需。

> 修改后需 `systemctl restart docker` 生效。修改前建议 `docker info` 先确认当前状态。

### 3.3 NVIDIA 容器栈

**验证命令**（GPU 透传闭环）：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**期望输出**：识别到 GB10 GPU，driver 580.173.02，CUDA 12.4 容器内可调用。

- 使用统一内存 121Gi，无需独立显存分配逻辑。
- aarch64 平台，镜像需匹配 arm64 架构（registry 已按平台分发）。

### 3.4 SSH 互信

- **本机 → <MGMT_OCTET> 免密**：本机公钥已写入 <MGMT_OCTET> 的 authorized_keys。
- **<MGMT_OCTET> 密钥对**：已生成（默认 ~/.ssh/id_ed25519 等）。
- **三机互信**：<MGMT_OCTET> / <MGMT_OCTET> / <MGMT_OCTET> 之间已互信，便于后续脚本跨机 rsync / scp。
- **known_hosts 预置**：避免首次连接指纹确认，供脚本非交互执行。
- **本机 config 别名**：

```sshconfig
# ~/.ssh/config
Host gx10-55
    HostName <NODE_IP>
    User <USER>
    IdentityFile ~/.ssh/<本机私钥>
```

使用：`ssh gx10-55` 直达。

### 3.5 基础工具

已安装：`git curl jq tmux rsync nfs-common htop iotop sysstat net-tools dnsutils vim unzip`

- `nfs-common`：NFS 客户端挂载必需。
- `rsync`：权重增量同步核心。
- `sysstat`/`iotop`/`htop`：巡检与性能诊断。

### 3.6 时区 / 内核

```bash
timedatectl set-timezone Etc/UTC      # 对齐 <MGMT_OCTET>
timedatectl set-ntp true              # NTP 自动同步
sysctl vm.max_map_count=1048576        # 持久化到 /etc/sysctl.d/（默认已足，显式对齐）
```

- 时区统一 **Etc/UTC**，避免多节点日志时间错位。
- `vm.max_map_count=1048576`：vLLM/大模型场景常需提高 mmap 上限，<MGMT_OCTET> 默认值已满足，显式固定。

### 3.7 权限

- 用户 **<USER>**：sudo 权限 + docker 组（gid 988）。
- docker 组授权后无需 sudo 即可执行 docker 命令（`docker ps` 等），便于脚本与日常操作。

---

## 4. 验收命令表

> 全部通过视为节点基线达标。逐条执行并核对期望输出。

| # | 命令 | 期望输出 |
|---|---|---|
| 1 | `systemctl is-active docker` | `active` |
| 2 | `systemctl is-enabled docker` | `enabled` |
| 3 | `docker version --format '{{.Server.Version}}'` | `29.2.1` |
| 4 | `docker info \| grep -A3 'Registry Mirrors'` | 列出 daocloud / dockerproxy |
| 5 | `docker info \| grep -A2 'Insecure Registries'` | `<NODE_IP>:5000` |
| 6 | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` | 显示 GB10，driver 580.173.02 |
| 7 | `ssh gx10-55 'hostname'` | `gx10-3f4d`（免密直达） |
| 8 | `ssh gx10-55 'ssh <MGMT_OCTET> 主机名或IP' 'hostname'` | <MGMT_OCTET> hostname（三机互信） |
| 9 | `timedatectl \| grep -E 'Time zone\|NTP'` | `Etc/UTC`，`yes` |
| 10 | `sysctl vm.max_map_count` | `1048576` |
| 11 | `id <USER>` | 含 `sudo` 与 `docker`(gid 988) 组 |
| 12 | `df -h /` | 931.5G 总量 / 822G 可用 |
| 13 | `uname -m` | `aarch64` |
| 14 | `for t in git curl jq tmux rsync nfs-common htop iotop sysstat net-tools dnsutils vim unzip; do command -v $t; done` | 全部返回路径 |
| 15 | `mount \| grep models-nfs` | `<NODE_IP>:/home/<USER>/models on /mnt/models-nfs type nfs4 (ro,...)` |

---

## 5. 常见故障

### 5.1 Docker 起不来

**现象**：`systemctl start docker` 失败，或启动即退出。

**排查**：

```bash
journalctl -u docker -n 50 --no-pager
systemctl status docker
```

**处置**：
- 检查 daemon.json 语法（`dockerd --validate` 或 `python3 -m json.tool /etc/docker/daemon.json`），JSON 错误会导致 dockerd 拒绝启动。
- 历史问题为 buildkit 损坏库：清理损坏库后重启。
- 恢复后验收第 1-3 条。

### 5.2 拉镜像超时 / 拉取失败

**现象**：`docker pull` 卡住或 `timeout`。

**处置**（按序）：
1. 内网优先：`docker pull <NODE_IP>:5000/<repo>:<tag>`（走 <MGMT_OCTET> registry，Wi-Fi 内网延迟低）。
2. 外网兜底：registry-mirrors（daocloud / dockerproxy）已配置，Docker Hub 直连失败时自动走 mirror。
3. 重试：Wi-Fi 链路偶发抖动，`--pull=always` 或重跑 pull。
4. 确认 insecure-registries 已含 `<NODE_IP>:5000`，否则返回 `http: server gave HTTP response to HTTPS client`。

### 5.3 GPU 透传失败

**现象**：`docker run --gpus all` 报 `could not select device driver "" with capabilities: [[gpu]]` 或容器内看不到 GPU。

**排查**：

```bash
nvidia-smi                          # 宿主机驱动是否正常
docker info | grep -A2 Runtimes     # 是否含 nvidia runtime
ls /usr/bin/nvidia-container*       # toolkit 是否安装
```

**处置**：
- 驱动异常：重装/恢复 580.173.02。
- runtime 缺失：确保 daemon.json 含 nvidia runtime 并 `systemctl restart docker`。
- 确认使用 `--gpus all` 或 `--gpus '"device=0"'`。

### 5.4 NFS 挂载失效

**现象**：`/mnt/models-nfs` 目录空或挂载丢失（系统重启后）。

**说明**：fstab 已配置 `_netdev,nofail`，挂载失败**不会阻塞系统启动**；挂载参数 `ro,soft,timeo=50,retrans=2` 保证客户端不 hang。

**处置**：

```bash
showmount -e <NODE_IP>          # <MGMT_OCTET> 是否仍导出
mount -a                          # 重挂 fstab 条目
mount | grep models-nfs
```

### 5.5 Wi-Fi 链路带宽受限

**说明**：当前仅 Wi-Fi，大镜像/大模型首拉慢。156G 级模型预计首拉 25-35 分钟（未实测）。
**缓解**：使用 `sync-model.sh --bwlimit=30000`（30MB/s 限速防抖），后台执行 + tmux；未来有线/RoCE 组网后同步源切换 10.100.136.x 自动提速。

---

## 6. 安全基线（当前）

- **registry 无 TLS/认证**：内网信任模型，仅限 <NODE_IP>/24 使用（ADR-4 决策）。
- **NFS 仅 <NODE_IP>/24**：`ro,sync,no_subtree_check,root_squash` 导出。
- **防火墙本次未启用**：集群层待办（见分发手册「安全注意」）。

---

## 7. 已知限制与后续

| 项 | 说明 | 计划 |
|---|---|---|
| 仅 Wi-Fi 链路 | <MGMT_OCTET> 未接有线/RoCE，大流量受限 | 有线/RoCE 组网后同步源切 10.100.136.x |
| 156G 大模型首拉未实测 | 预计 25-35 分钟 | 接入后实测并登记 |
| 防火墙未启用 | 集群层待办 | 统一防火墙策略 |
| registry 单点（<MGMT_OCTET>） | 无副本 | 备份至 <MGMT_OCTET>，规划高可用 |
