# DGX Spark 四机集群环境配置手册（含新增节点接入 SOP）

- 版本：v1.0（四机版）
- 适用平台：NVIDIA DGX Spark GB10 / Ubuntu 24.04.4 LTS / aarch64 / 121Gi / driver 580.173.02
- 读者对象：SRE / 集群运维
- 配套文档：《分发机制操作手册（四机版）》

> **主理人汇编注（2026-08-07）**：本文由 Docu 产出，落地时按实测修正以下差异：① 分发脚本实际部署路径为 `/opt/distribution/`（非 /usr/local/bin）；② NFS 导出源实际为 `<MGMT_OCTET>:/home/<USER>/models`（非 /mnt/models-nfs）；③ daemon.json 实测含 registry-mirrors 与 nvidia runtime（见 §2 S5）；④ 四机互信已含 <MGMT_OCTET>。第 5+ 台节点按本 SOP 复用，仅替换 IP/主机名/SSH 别名。

## 1. 四机节点清单

| 节点 | IP | 角色 | 磁盘 | 网络 | 主机标识 | 状态 | 说明 |
|------|----|------|------|------|---------|------|------|
| <MGMT_OCTET> | <NODE_IP> | worker（大容量中心） | 3.6T（2.4T 可用） | RoCE <NODE_IP> + Wi-Fi | aicad-server | 运行中 | registry :5000（27 repos）/ NFS 导出 312G / embed 8020 等 14 容器 |
| <MGMT_OCTET> | <NODE_IP> | head | 3.6T（2.8T 可用） | RoCE <NODE_IP> + Wi-Fi | aicad-server60 | 运行中 | 关键镜像备份端（vllm 21G + embed 19G） |
| <MGMT_OCTET> | <NODE_IP> | 新增小盘节点 | 931G（821G 可用） | 仅 Wi-Fi | gx10-3f4d | 已配置 | 156G 大权重同步中（首拉） |
| <MGMT_OCTET> | <NODE_IP> | 新增小盘节点 | 931G（822G 可用） | 仅 Wi-Fi | gx10-31c4 | 已配置完成 | embed 1.2G 同步闭环 |

SSH 别名（本机 Windows `~/.ssh/config`）：`gx10-55` / `gx10-59` / `aicad-server`(<MGMT_OCTET>) / `aicad-server60`(<MGMT_OCTET>)。
四机互信：<MGMT_OCTET>/<MGMT_OCTET> 公钥已入 <MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET> 的 authorized_keys。

## 2. 新增节点接入 SOP（适用于第 5+ 台复用）

> 操作序列 S1–S11（对应 <MGMT_OCTET> 实际实施序列）；第 5+ 台仅替换示例 IP/主机名。

### S1 前置检查
```bash
ssh gx10-59
sudo -i
hostname                            # 期望: gx10-31c4
grep PRETTY_NAME /etc/os-release    # 期望: Ubuntu 24.04.4 LTS
uname -m                            # 期望: aarch64
nvidia-smi                          # 期望: Driver 580.173.02 / 121Gi
lsblk -d -o NAME,SIZE,MODEL         # 期望: nvme0n1 931G
ip -4 -o addr show                  # 期望: 192.168.5.x
systemctl is-system-running         # 期望: running
df -h /                             # 记录可用空间基线
```
常见坑：
- 提示 "dpkg was interrupted" → 先 `sudo dpkg --configure -a`（见 S4）。
- `nvidia-smi` 无输出 → 驱动未就绪，先报障不继续。
- is-system-running 返回 degraded → `systemctl --failed` 处理后再继续。

### S2 Docker 修复（buildkit）
出厂镜像偶发 buildkit 缓存库损坏，报错形如 `error creating buildkit instance: invalid database`。
```bash
sudo systemctl status docker --no-pager -l
# 确认无容器/镜像后（新机）：
sudo systemctl stop docker
sudo mv /var/lib/docker/buildkit /var/lib/docker/buildkit.bak.$(date +%Y%m%d%H%M%S)   # 先备份再删
sudo systemctl start docker
docker version --format '{{.Server.Version}}'   # 期望: 29.2.1
```
常见坑：
- **buildkit invalid database**：先 `mv` 备份，确认正常后再删除（<MGMT_OCTET>/<MGMT_OCTET> 实测均为此根因）。
- docker 起不来：`journalctl -u docker -n 50`；`df -h /var/lib/docker` 确认空间。

### S3 docker 组（免 sudo）
```bash
sudo usermod -aG docker $USER
# 重新登录或 newgrp docker 后：
docker ps        # 期望免 sudo 可执行
getent group docker   # 期望 docker:x:988:<USER>（gid 988 对齐集群）
```
常见坑：不重新登录/不 newgrp，当前 shell 仍报 permission denied。

### S4 基础工具 + dpkg 修复
```bash
sudo dpkg --configure -a            # 若 apt 报中断
sudo apt update
sudo apt install -y git curl jq tmux rsync nfs-common htop iotop sysstat net-tools dnsutils vim unzip
which git curl jq tmux rsync mount.nfs   # 期望全路径
```

### S5 daemon.json（registry 信任 + mirror + runtime）
```json
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "100m", "max-file": "3"},
  "insecure-registries": ["<NODE_IP>:5000"],
  "registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerproxy.net"],
  "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}
}
```
```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'   # 内容如上
EOF
sudo systemctl restart docker
docker info | grep -A2 'Insecure Registries'   # 期望 <NODE_IP>:5000
docker info | grep -A3 Runtimes                # 期望含 nvidia
```
常见坑：daemon.json 语法错误导致 docker 起不来——改完先 `docker info` 验证；已有 daemon.json 先备份再覆盖。

### S6 NVIDIA 容器栈
```bash
nvidia-ctk --version                 # 期望 v1.19.1
sudo nvidia-ctk runtime configure --runtime=docker   # 若未配置
sudo systemctl restart docker
```
冒烟（S7 合一）：
```bash
docker pull nvidia/cuda:12.4.0-base-ubuntu22.04   # 直连失败时走 mirror/内网
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# 期望: NVIDIA GB10 / Driver 580.173.02
```

### S7 SSH 互信
```bash
# 新节点上
ssh-keygen -t ed25519 -a 100 -N '' -f ~/.ssh/id_ed25519 -C '<USER>@gx10-<host>'
# 预置 known_hosts（防批量脚本卡 host key 确认）
ssh-keyscan -H <NODE_IP> <NODE_IP> <NODE_IP> <NODE_IP> >> ~/.ssh/known_hosts
# 公钥分发：管理员（本机）将新节点公钥追加到各机 ~/.ssh/authorized_keys
# 验证
ssh -o BatchMode=yes <USER>@<NODE_IP> echo SSH_OK
```
常见坑：
- **pkill -f 自匹配**：脚本内 `pkill -f 'rsync.*deepseek'` 可能杀掉自身命令行——用 `pgrep` 取 PID 再 kill。
- authorized_keys 权限 600、.ssh 700，否则 sshd 忽略。

### S8 NFS 挂载（<MGMT_OCTET> 只读冷访问）
```bash
sudo mkdir -p /mnt/models-nfs <MODELS_DIR> /opt/distribution
sudo mount -t nfs -o ro,soft,timeo=50,retrans=2,nfsvers=4 <NODE_IP>:/home/<USER>/models /mnt/models-nfs
mount | grep models-nfs     # 期望 ro, vers=4.2, soft, timeo=50
# fstab 持久化（_netdev,nofail 防 Wi-Fi 未就绪阻塞启动）：
echo '<NODE_IP>:/home/<USER>/models /mnt/models-nfs nfs ro,soft,timeo=50,retrans=2,nfsvers=4,_netdev,nofail 0 0' | sudo tee -a /etc/fstab
sudo chown -R <USER>:<USER> <MODELS_DIR>
sudo touch /var/log/distribution.log && sudo chown <USER>:<USER> /var/log/distribution.log
```
常见坑：`mount.nfs: access denied` → 检查 <MGMT_OCTET> exports（<NODE_IP>/24 已覆盖）；`ls` 卡住 → showmount -e 判断断链。

### S9 脚本部署（/opt/distribution）
```bash
# 从 <MGMT_OCTET> 或 <MGMT_OCTET> 拷贝（或 git 统一管理）：
# /opt/distribution/sync-model.sh      （加固版：MODEL 白名单/--timeout=600/TCPKeepAlive/重试×5-10/bwlimit 默认 20MB/s/flock）
# /opt/distribution/cleanup-weights.sh （与 sync 共用锁/路径兜底/prune until=24h）
# /opt/distribution/disk-watch.sh      （70% 告警/85% 严重）
# /opt/distribution/allowlist-images.txt
sudo chmod +x /opt/distribution/*.sh
# systemd timer（每日 00:00 UTC）：
sudo tee /etc/systemd/system/distribution-watch.timer >/dev/null <<'EOF'
[Unit]
Description=Distribution disk watch
[Timer]
OnCalendar=daily
Persistent=true
[Install]
WantedBy=timers.target
EOF
sudo tee /etc/systemd/system/distribution-watch.service >/dev/null <<'EOF'
[Unit]
Description=Distribution disk watch service
[Service]
Type=oneshot
ExecStart=/opt/distribution/disk-watch.sh
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now distribution-watch.timer
systemctl list-timers distribution-watch   # 期望 next 00:00 UTC
```

### S10 验证闭环
```bash
# 1) 镜像拉取（registry 冒烟）
docker pull <NODE_IP>:5000/library/redis:7-alpine && docker run --rm <NODE_IP>:5000/library/redis:7-alpine redis-server --version
# 2) 权重同步（小模型闭环）
bash /opt/distribution/sync-model.sh Qwen3-Embedding-0.6B 20000
# 3) 完整性校验（硬校验）
cd <MODELS_DIR>/Qwen3-Embedding-0.6B && sha256sum -c sha256sums.txt --quiet; echo RC=$?   # 期望 0
# 4) 大权重首拉（156G，Wi-Fi 1.5-3h，断点续传）
bash /opt/distribution/sync-model.sh deepseek-v4-flash-0731 20000 10
# 5) 日志
tail -f /var/log/distribution.log
```
闭环判定：镜像 run 成功；同步 DONE + .last-used；sha256sum -c 全 OK；清理/水位 timer 正常。

### S11 登记
- 本机 ~/.ssh/config 加别名；节点信息登记集群 inventory；Runbook 回填。

## 3. <MGMT_OCTET> 与 <MGMT_OCTET> 配置差异记录

结论：**无配置差异（同构）**——仅主机标识（gx10-3f4d / gx10-31c4）、IP、SSH 别名不同；同步进度不同（<MGMT_OCTET> 大权重 9.5G+ / <MGMT_OCTET> embed 闭环 + 首拉中）。

## 附录 A：本机 Windows SSH 别名
```
Host gx10-55     # HostName <NODE_IP>
Host gx10-59     # HostName <NODE_IP>
Host aicad-server    # HostName <NODE_IP>
Host aicad-server60  # HostName <NODE_IP>
# 均 User <USER>, IdentityFile ~/.ssh/id_ed25519
```

## 附录 B：新节点接入后检查清单
- [ ] `nvidia-smi` 121Gi / driver 580.173.02
- [ ] `docker info` Insecure Registries 含 <MGMT_OCTET>:5000、Runtimes 含 nvidia
- [ ] `ssh aicad-server60 hostname`、`ssh gx10-55 hostname` 免密
- [ ] `mount | grep models-nfs` ro 挂载
- [ ] `sync-model.sh Qwen3-Embedding-0.6B` 闭环 + `sha256sum -c` OK
- [ ] `systemctl list-timers distribution-watch` next 00:00 UTC
- [ ] 已追加进本机 SSH config 别名

---
*本文由工程保障团队 Docu 产出、主理人汇编落盘（2026-08-07）。*
