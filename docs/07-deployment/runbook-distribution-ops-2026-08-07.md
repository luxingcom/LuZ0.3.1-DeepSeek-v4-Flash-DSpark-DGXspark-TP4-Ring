# 分发机制操作手册（镜像 + 权重 + 清理）

> 面向对象：SRE / 新节点接入人员
> 适用范围：集群内镜像分发（.58 registry）、权重分发（NFS + rsync）、磁盘清理与巡检
> 配套文档：《新节点环境配置手册》 runbook-newnode-env-config-2026-08-07.md
> 最近更新：2026-08-07

---

## 1. 架构拓扑

```
                    ┌─────────────────────────────┐
                    │  管理机 / 本机（运维入口）     │
                    │  SSH alias: gx10-55          │
                    └──────────────┬──────────────┘
                                   │ 三机互信 SSH
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼─────┐              ┌─────▼─────┐              ┌─────▼─────┐
   │ .58 worker│             │ .55 新节点 │             │ .60 head   │
   │           │             │           │             │            │
   │ registry:2│             │ NFS 挂载   │             │ 备份接收端  │
   │  :5000    │◄── pull ────│ /mnt/models│             │ ~/backup-  │
   │           │  (192.168.5 │ -nfs (ro) │             │ images/    │
   │ NFS server│    .58:5000)│           │             │            │
   │ /home/liu │── NFS ro ──►│ <MODELS_DIR>│            │            │
   │ xiaoya/   │  (192.168.5 │ (LRU 200G) │             │            │
   │ models    │   .0/24)    │           │             │            │
   │ (312G)    │             │ /opt/     │             │            │
   └───────────┘             │ distribution│           └────────────┘
                             │ scripts   │
                             └───────────┘

   RoCE 平面：.58/.60 10.100.136.x（TP=2 vLLM，当前停机）
   Wi-Fi 平面：<NODE_IP>/24（.55 仅在此平面）
```

**角色划分**：
- **.58**：镜像分发源（registry:2）、权重权威源（NFS 导出 + 本地 312G 模型）、worker。
- **.55**：消费端。镜像按需 pull；权重 NFS 只读挂载 + 可选 rsync 落本地 <MODELS_DIR>。
- **.60**：备份接收端（接收 .58 导出镜像备份）。

---

## 2. 数据流

### 2.1 镜像流（registry 拉取）

```
.58 registry（/data/registry, :5000） ──docker pull──►  .55 本地镜像存储
       33 镜像 / 27 repos                             按需拉取（已验证：redis 7.4.10 拉取→运行→清理闭环）
```

- 镜像来源：.58 现有镜像全量推送至 registry（33 镜像 → 27 repos）。
- 消费方式：`docker pull <NODE_IP>:5000/<repo>:<tag>`。
- 无认证内网信任（ADR-4），insecure-registries 已配。

### 2.2 权重流（NFS 只读 + rsync 按需）

```
路径 A（直接读）：.58 NFS 导出 /home/<USER>/models ──ro 挂载──► .55 /mnt/models-nfs
路径 B（落本地）：.58 NFS ──sync-model.sh rsync 增量──► .55 <MODELS_DIR>（LRU 200G 管理）
```

- 路径 A：零拷贝、省 .55 盘；只读 + soft 挂载，客户端不 hang。
- 路径 B：适合需本地高性能读取/断网可用场景，由 sync-model.sh 管理。

### 2.3 清理流（LRU + 磁盘水位）

```
.55 <MODELS_DIR>（LRU，超 200G 触发清理）
        │ cleanup-weights.sh：按 .last-used 淘汰最久未用，keep/allowlist 豁免
        ▼
.55 磁盘水位 disk-watch.sh：70% 告警 / 85% 严重
        │ distribution-watch.timer 每日 00:00 UTC 触发
        ▼
日志 /var/log/distribution.log
```

---

## 3. 日常操作

### 3.1 拉取镜像

```bash
# 内网 registry（推荐）
docker pull <NODE_IP>:5000/redis:7.4.10

# 重打 tag 便于本地使用
docker tag <NODE_IP>:5000/redis:7.4.10 redis:7.4.10

# 验证容器可运行后按需清理
docker run --rm <NODE_IP>:5000/redis:7.4.10 redis-server --version
docker rmi <NODE_IP>:5000/redis:7.4.10   # 无需保留时清理
```

### 3.2 同步模型到本地

```bash
sudo /opt/distribution/sync-model.sh <模型名或路径>
# 例：同步 embed 模型（已验证 1.2G 闭环）
sudo /opt/distribution/sync-model.sh embed
```

### 3.3 清理本地权重

```bash
sudo /opt/distribution/cleanup-weights.sh          # 默认：LRU 淘汰至 200G 阈值内
sudo /opt/distribution/cleanup-weights.sh --dry-run # 预演，不实际删除
```

### 3.4 巡检命令

```bash
# 分发状态日志
tail -50 /var/log/distribution.log

# NFS 挂载健康
mount | grep models-nfs && df -h /mnt/models-nfs

# 磁盘水位
df -h / && du -sh <MODELS_DIR>

# registry 健康（.58 上）
curl -s http://<NODE_IP>:5000/v2/_catalog | jq '.repositories | length'

# 最近使用记录
cat <MODELS_DIR>/.last-used 2>/dev/null
```

---

## 4. 脚本用法

### 4.1 sync-model.sh — 增量同步模型

**位置**：`/opt/distribution/sync-model.sh`
**核心**：rsync 增量 + flock 防并发 + 限速 + 日志 + 使用记录

```bash
sudo /opt/distribution/sync-model.sh <源> [--bwlimit=<速率KB/s>]
```

| 参数 | 说明 |
|---|---|
| `<源>` | 模型名或 .58 上路径（默认同步源 <NODE_IP>:/home/<USER>/models 下对应目录） |
| `--bwlimit=` | 默认 30000（≈30MB/s），Wi-Fi 链路防抖 |

**内置行为**：
- `rsync -aP --partial --inplace --bwlimit=30000`：断点续传、原地写、限速。
- `flock`：防止并发同步冲突（同一模型同时跑两次）。
- 日志写入 `/var/log/distribution.log`。
- 成功后更新 `<MODELS_DIR>/.last-used`（供 cleanup LRU 判定）。

**示例**：

```bash
# 默认限速同步 embed 模型
sudo /opt/distribution/sync-model.sh embed

# 自定义限速 10MB/s（弱网时段）
sudo /opt/distribution/sync-model.sh embed --bwlimit=10000

# 后台执行（大模型，建议 tmux 内跑）
tmux new -s sync
sudo /opt/distribution/sync-model.sh llama-156g
```

### 4.2 cleanup-weights.sh — LRU 清理

**位置**：`/opt/distribution/cleanup-weights.sh`
**目标**：`<MODELS_DIR>`，阈值 **200G**

| 参数 | 说明 |
|---|---|
| `--dry-run` | 只打印将删除项，不实际删除 |
| （无参） | 按 `.last-used` 淘汰最久未用，直至低于阈值 |
| keep / allowlist | 豁免清单内的模型永不清理（关键权重保护） |

**示例**：

```bash
sudo /opt/distribution/cleanup-weights.sh --dry-run   # 先看将删什么
sudo /opt/distribution/cleanup-weights.sh             # 执行清理
```

### 4.3 disk-watch.sh — 磁盘水位告警

**位置**：`/opt/distribution/disk-watch.sh`
**阈值**：70% 告警 / 85% 严重

**示例**：

```bash
sudo /opt/distribution/disk-watch.sh        # 手动触发一次巡检
```

### 4.4 distribution-watch.timer — 定时巡检

- 触发：`distribution-watch.timer`，每日 **00:00 UTC**。
- 动作：执行 disk-watch.sh（水位检查），超阈值输出告警。
- 查看状态：

```bash
systemctl list-timers distribution-watch.timer
systemctl status distribution-watch.timer
journalctl -u distribution-watch.service -n 30 --no-pager
```

### 4.5 allowlist-images.txt — 镜像白名单

- 位置：`/opt/distribution/allowlist-images.txt`
- 用途：登记允许在 .55 等节点拉取/保留的镜像清单（配合清理与合规审计）。
- 新镜像加入 registry 时同步登记。

---

## 5. 新增节点接入 SOP（S1–S8）

> 供第四台及以后节点复用。每步执行后核对对应验收项（见环境配置手册第 4 节）。

### S1 网络与基础
1. 节点加电，接入 <NODE_IP>/24（Wi-Fi 或有线）。
2. 配置静态 IP（如 .56+）、主机名；`ping <NODE_IP>` 通。
3. 创建运维用户（如 <USER>），加入 sudo 与 docker 组（gid 988）。

### S2 系统基线
4. `timedatectl set-timezone Etc/UTC`；`timedatectl set-ntp true`。
5. `apt update && apt upgrade`；安装基础工具（见环境手册 3.5）。
6. `sysctl vm.max_map_count=1048576` 并持久化。

### S3 Docker 栈
7. 安装 Docker 29.2.1；检查 buildkit 完整性（避免 .55 同款损坏问题）。
8. 写入 daemon.json（log 轮转 / insecure-registries / mirrors / nvidia runtime），`systemctl restart docker`。
9. `systemctl enable --now docker`。

### S4 NVIDIA 栈
10. 确认宿主机 `nvidia-smi` 正常（driver 580.173.02 基线）。
11. `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` 透传验证。

### S5 SSH 互信
12. 生成节点密钥对；三机互信（含 .55/.58/.60 及本机）。
13. 本机 config 增加新别名；known_hosts 预置。

### S6 镜像分发接入
14. 验证 `docker pull <NODE_IP>:5000/redis:7.4.10` 闭环（拉取→运行→清理）。
15. 将新增镜像登记入 allowlist-images.txt。

### S7 权重分发接入
16. fstab 增加 NFS 挂载 `/mnt/models-nfs`（ro,soft,timeo=50,retrans=2,nfsvers=4,_netdev,nofail），`mount -a` 验证。
17. 按需 `sync-model.sh` 同步本地权重（可选）。

### S8 验收与收尾
18. 执行环境手册「验收命令表」全项。
19. 登记节点清单（更新本文档第 1 节拓扑与节点表）。
20. 加入监控（disk-watch.timer）、纳入备份/巡检轮值。

---

## 6. 故障处理

### 6.1 镜像拉取失败

```bash
# 1. 检查 .58 registry 是否存活
curl -s http://<NODE_IP>:5000/v2/_catalog | jq '.repositories | length'

# 2. 检查本地 registry 容器
ssh <NODE_IP> 'docker ps | grep registry'

# 3. 确认 tag 存在
curl -s http://<NODE_IP>:5000/v2/<repo>/tags/list

# 4. 重试 pull；网络抖动重试，或临时切 mirror
```

- 若报 HTTPS 相关错误 → 检查 insecure-registries 是否含 `<NODE_IP>:5000`。
- 若镜像平台不符（x86 镜像拉 arm64）→ 确认 repo 有 aarch64 标签。

### 6.2 NFS 断链

**设计缓解**：`ro,soft,timeo=50,retrans=2,nofail` — 客户端不 hang、不阻塞启动。

```bash
# .58 端确认导出
ssh <NODE_IP> 'exportfs -v | grep models'

# 客户端查看
showmount -e <NODE_IP>
mount -a                          # 重挂
mount | grep models-nfs
df -h /mnt/models-nfs
```

- 持续断链 → 检查 Wi-Fi 链路、.58 NFS 服务（`systemctl status nfs-server`）。

### 6.3 磁盘满

1. `df -h /`、`du -sh <MODELS_DIR> /var/lib/docker` 定位占用。
2. `cleanup-weights.sh --dry-run` 预演，再执行清理。
3. 清理旧容器/悬空镜像：`docker system prune -af`（谨慎，先确认 allowlist）。
4. 检查 docker 日志轮转是否生效（100m×3）。
5. 若仍满：评估扩容 / 迁移到 .58 大盘 NFS 只读方案。

### 6.4 .58 单点

**风险**：registry 与 NFS 权威源均在 .58，单机故障即分发中断。

**对策**：
- 镜像备份：`docker save` vLLM+embed 等关键镜像 → .60 `~/backup-images/`（已实施，后台执行中）。
- 权重备份：.58 模型 312G 需定期备份/校验；.60 为备份接收端。
- 规划：registry 高可用（多副本/负载均衡）、NFS 主备切换（后续迭代）。

---

## 7. 安全注意

### 7.1 内网信任边界（当前基线）

- registry 无 TLS/认证：仅限 **<NODE_IP>/24** 内网使用（ADR-4 决策）。
- NFS 导出仅限 **<NODE_IP>/24**：`ro,sync,no_subtree_check,root_squash`。
- **防火墙未启用**：集群层待办，禁止将 5000/NFS 端口暴露到公网或不可信网段。

### 7.2 加固选项（后续可选）

**registry 加认证（htpasswd）+ TLS**：

```bash
# 生成密码文件（在 .58）
docker run --rm --entrypoint htpasswd registry:2 -Bbn <user> <password> > /data/registry/htpasswd

# registry 容器增加参数（需调整）
#  -v /data/registry/htpasswd:/auth/htpasswd:ro
#  -e "REGISTRY_AUTH=htpasswd" -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
#  -e "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd" \
#  -e "REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt" \
#  -e "REGISTRY_HTTP_TLS_KEY=/certs/domain.key"
```

> 启用后所有节点 daemon.json 需同步调整（insecure-registries → 证书信任），并更新所有 pull 脚本。

**NFS 加固**：保持只读 + root_squash；若需读写用子目录最小授权。

**防火墙**：启用 ufw/nftables，仅放行 22、5000（<NODE_IP>/24）、NFS（2049,111,20048 等，限内网）。

---

## 8. 监控与告警

| 项 | 配置 | 说明 |
|---|---|---|
| 定时巡检 | `distribution-watch.timer` 每日 00:00 UTC | 触发 disk-watch.sh 水位检查 |
| 磁盘告警阈值 | 70% 告警 / 85% 严重 | disk-watch.sh 输出 |
| 同步日志 | `/var/log/distribution.log` | sync-model.sh / cleanup 动作全记录 |
| 使用记录 | `<MODELS_DIR>/.last-used` | LRU 判定依据 |
| 镜像清单 | `/opt/distribution/allowlist-images.txt` | 白名单登记 |

**巡检示例**：

```bash
# 查看 timer 下次触发
systemctl list-timers distribution-watch.timer

# 查看最近一次巡检
journalctl -u distribution-watch.service -n 30 --no-pager

# 查看同步/清理日志
tail -50 /var/log/distribution.log
```

---

## 附录：关键路径速查

| 资源 | 位置 |
|---|---|
| registry 数据 | .58 `/data/registry` |
| registry 服务 | .58 `docker run registry:2` :5000（restart=always） |
| 权重权威源 | .58 `/home/<USER>/models`（312G） |
| NFS 挂载点 | .55 `/mnt/models-nfs`（ro） |
| 本地权重缓存 | .55 `<MODELS_DIR>`（LRU 200G） |
| 分发脚本 | `/opt/distribution/` |
| 分发日志 | `/var/log/distribution.log` |
| 备份接收 | .60 `~/backup-images/` |
