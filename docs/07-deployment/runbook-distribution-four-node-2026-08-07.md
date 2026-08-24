# DGX Spark 四机集群模型分发机制操作手册

- 版本：v1.0（四机版）
- 读者对象：SRE / 集群运维
- 配套文档：《环境配置手册（四机版）》

> **主理人汇编注（2026-08-07）**：本文由 Docu 产出，落地时按实测修正：① 权重热源实际为 `<MGMT_OCTET>:/home/<USER>/models`（312G）；② 脚本实际位于 `/opt/distribution/`；③ 实测参数：sync-model 默认 bwlimit 20000 KB/s、重试 5 次（可传第 3 参调大）；④ 备份实际存 `<MGMT_OCTET>:~/backup-images/`（vllm-0.2.1-v026.0.tar 21G + embed-0.1.1.tar 19G）。

## 1. 拓扑与数据流

### 1.1 拓扑
```
                         <NODE_IP>/24（Wi-Fi 内网）
   ┌───────────────────────────────┐
   │ <MGMT_OCTET> aicad-server（大容量中心） │  RoCE <NODE_IP> + Wi-Fi
   │  · registry :5000（27 repos）  │
   │  · NFS 导出 /home/<USER>/models │  312G 只读
   │  · embed 8020 等 14 容器        │
   └──────┬──────────────┬─────────┘
          │ 镜像 pull    │ rsync 增量 / NFS ro
   ┌──────▼─────┐  ┌─────▼──────┐        ┌────────────────┐
   │ <MGMT_OCTET> gx10-55│  │ <MGMT_OCTET> gx10-59│        │ <MGMT_OCTET> aicad-      │
   │ 931G 仅WiFi│  │ 931G 仅WiFi│        │ server60 (head) │
   │ 156G 同步中│  │ embed 闭环 │        │ RoCE .1 + WiFi  │
   └────────────┘  └────────────┘        │ 镜像备份端       │
                                         │ vllm21G+embed19G│
                                         └────────────────┘
```

### 1.2 四类数据流

| 数据流 | 源 | 目标 | 通道 | 特点 |
|--------|----|------|------|------|
| 镜像流 | <MGMT_OCTET> registry :5000 | <MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET> | docker pull（按需） | 无 TLS/认证（ADR-4 内网信任）；27 repos |
| 权重流-热 | <MGMT_OCTET> /home/<USER>/models | <MGMT_OCTET>/<MGMT_OCTET> <MODELS_DIR> | rsync 增量 | bwlimit 默认 20MB/s、断点续传、sha256 硬校验 |
| 权重流-冷 | <MGMT_OCTET> NFS 导出 | <MGMT_OCTET>/<MGMT_OCTET> /mnt/models-nfs | NFS4 ro | 只读冷访问/校验清单读取 |
| 清理流 | <MGMT_OCTET>/<MGMT_OCTET> <MODELS_DIR> | —— | cleanup-weights.sh（LRU 200G + prune until=24h） | 与 sync 共用 flock 防误删 |
| 备份流 | <MGMT_OCTET> 关键镜像 | <MGMT_OCTET> ~/backup-images | docker save + scp | vllm 21G + embed 19G；sha256 对账 |

## 2. 日常操作

### 2.1 拉镜像
```bash
curl -s http://<NODE_IP>:5000/v2/_catalog | jq -r '.repositories[]'      # 期望 27
curl -s http://<NODE_IP>:5000/v2/<repo>/tags/list | jq -r '.tags[]'
docker pull <NODE_IP>:5000/<repo>:<tag>
```

### 2.2 同步模型
```bash
bash /opt/distribution/sync-model.sh Qwen3-Embedding-0.6B          # 默认 20MB/s、重试 5
bash /opt/distribution/sync-model.sh deepseek-v4-flash-0731 10000 15   # 限速 10MB/s、重试 15
```
进度：`tail -f /var/log/distribution.log`；大权重首拉（156G）约 1.5-3h（20MB/s），支持断点续传（已验证 SHA256 一致）。

### 2.3 清理
```bash
bash /opt/distribution/cleanup-weights.sh          # 阈值默认 200G；低于阈值 skip
```
说明：与 sync 共用 /var/lock/sync-model.lock；路径兜底仅限 <MODELS_DIR>/[A-Za-z0-9._-]；镜像 prune until=24h。

### 2.4 巡检
```bash
df -h / /data /var/lib/docker 2>/dev/null
curl -s http://<NODE_IP>:5000/v2/_catalog | jq '.repositories | length'   # 27
showmount -e <NODE_IP>
mount | grep models-nfs
tail -50 /var/log/distribution.log
systemctl list-timers distribution-watch
ssh aicad-server 'docker ps | wc -l'      # <MGMT_OCTET> 容器 ≥14
```

## 3. 脚本用法

### 3.1 sync-model.sh —— 权重热同步
```
用法: bash /opt/distribution/sync-model.sh <MODEL> [bwlimit_kbps] [retry_max]
示例: bash /opt/distribution/sync-model.sh deepseek-v4-flash-0731 20000 10
```
特性：MODEL 白名单（防路径穿越）/ rsync -aP --partial --inplace --timeout=600 / TCPKeepAlive / 自动重试×5-10 / flock 防重入 / sha256sums.txt 硬校验（清单缺失时 WARN 跳过）/ .last-used 标记 / 日志 /var/log/distribution.log。

### 3.2 cleanup-weights.sh —— 权重 LRU 清理
```
用法: bash /opt/distribution/cleanup-weights.sh [threshold_gb]
```
特性：与 sync 共用锁（同步中跳过）/ 路径兜底 / 按 mtime LRU 删除 / 镜像 prune until=24h / 日志。

### 3.3 disk-watch.sh —— 磁盘水位（systemd timer 每日 00:00 UTC）
```
手动执行: sudo systemctl start distribution-watch.service
```
特性：/ 与 /data 使用率 ≥70% 写 WARN、≥85% 写 CRITICAL 至 /var/log/distribution.log。

## 4. 故障处理

### 4.1 镜像拉取失败
```bash
curl -s http://<NODE_IP>:5000/v2/_catalog | jq .     # registry 可达？
ssh aicad-server 'docker ps | grep registry'             # registry 容器在跑？
df -h /var/lib/docker                                    # 目标磁盘？
```
处理：registry 容器异常 → <MGMT_OCTET> `docker restart registry`；Wi-Fi 抖动 → 重试 pull（按层续传）。

### 4.2 NFS 断链
症状：ls /mnt/models-nfs 卡住、df 挂起、读权重超时。
```bash
showmount -e <NODE_IP>
ssh aicad-server 'exportfs -v'
sudo umount -l /mnt/models-nfs && sudo mount -a
```
预防：fstab 已配 soft,timeo=50,retrans=2,_netdev,nofail。

### 4.3 Wi-Fi 断流 / 同步中断
```bash
tail -100 /var/log/distribution.log
bash /opt/distribution/sync-model.sh deepseek-v4-flash-0731 10000 15   # 降速重试，自动续传
```
机制：自动重试×5-10 + TCPKeepAlive + --partial 断点续传（已验证中断 296MB→恢复→SHA256 一致）。连续失败多为 Wi-Fi 链路问题；组网（RoCE）后同步源切 10.100.136.x 提速。

### 4.4 磁盘满
```bash
df -h / /data /var/lib/docker
bash /opt/distribution/cleanup-weights.sh   # 小盘节点清理
```
小盘节点规划：embed 1.2G + deepseek 156G ≈ 157G，配合 LRU 200G 阈值控制总量。

### 4.5 <MGMT_OCTET> 单点故障
现状：registry 与 NFS 均在 <MGMT_OCTET>（SPOF 已记录）。缓解：关键镜像已备份 <MGMT_OCTET>（40G）；权重可 NFS 冷访问 + rsync 重拉。
恢复顺序：<MGMT_OCTET> 服务恢复 → registry 自启检查 → NFS export 检查 → 各节点 mount -a。
长期：registry 高可用/备用 NFS（集群待办）。

### 4.6 备份恢复
- 镜像恢复：<MGMT_OCTET> `docker load < backup-images/*.tar` 或从 registry 重 pull。
- 权重恢复：NFS 冷目录回拷或重新 sync；恢复后 `sha256sum -c` 校验。
- 已知限制：备份 sha256 对账运行中（2026-08-07）；deepseek 156G 校验清单待生成。

## 5. 监控与告警

| 项 | 方法 | 期望 |
|----|------|------|
| disk-watch timer | systemctl list-timers distribution-watch | next 00:00 UTC |
| 同步/清理日志 | tail -f /var/log/distribution.log | START/DONE/FAIL |
| 各节点磁盘 | df -h / /data | 可用 > 阈值（70% WARN/85% CRITICAL） |
| registry 健康 | curl _catalog | 27 repos |
| NFS 状态 | showmount -e <NODE_IP> | 导出正常 |
| <MGMT_OCTET> 容器 | ssh aicad-server 'docker ps \| wc -l' | ≥14 |

告警：当前仅日志无通知通道（待办：cron 健康检查 / 接 Prometheus+Grafana）。

## 6. 安全注意

### 6.1 内网信任边界
- registry 无 TLS/认证（ADR-4）：仅限 <NODE_IP>/24，严禁暴露公网；
- NFS 导出 <NODE_IP>/24 只读（ro,root_squash）；
- 集群防火墙未启用（集群待办）——当前信任依赖内网隔离。

### 6.2 加固选项（推荐按序）
1. 启用防火墙（ufw）：仅放行 <NODE_IP>/24 与必要端口；
2. registry 增加 TLS + htpasswd 认证；
3. SSH 仅密钥、禁密码登录、各机差异化口令；
4. NFS 收窄 IP + root_squash 复核；
5. 备份加密 + sha256 对账（已启动）；
6. 脚本/白名单 git 集中管理。

### 6.3 已知限制与演进
- <MGMT_OCTET>/<MGMT_OCTET> 仅 Wi-Fi：组网后同步源切 10.100.136.x；
- deepseek 156G 校验清单待生成；
- <MGMT_OCTET> 为 registry+NFS 单点 → 规划高可用；
- 四机并发同步带宽竞争 → 大任务错峰。

---
*本文由工程保障团队 Docu 产出、主理人汇编落盘（2026-08-07）。*
