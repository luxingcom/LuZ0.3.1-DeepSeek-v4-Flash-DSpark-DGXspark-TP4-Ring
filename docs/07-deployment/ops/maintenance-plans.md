# 维护方案（Maintenance Plans）v1.5-R11

**日期**：2026-08-12（R11 修订）｜**维护**：Docu｜**前置**：必须先读 ops/ops-discipline-quickref.md（纪律权威）
**R11 变更**：灰度示例改 util 0.65/seqs 6 已落地；shim 回退锚点 v7；重启复核项改 isolcpus=8-9。

---

## 1. 维护窗口 SOP（停机维护）

### 1.1 窗口申请与前置
- 维护动作按 **P0（立即）/P1（15min 内）/P2（窗口期）** 分级（速查 §5）。
- 窗口时间：避开业务高峰（建议凌晨）；**通知 litellm 下游**（02:4000 并发队列会堆积）。
- 变更三件套：`cp <script> <script>.bak-<tag>` → `check_vllm_script.sh` 通过 → Rex 审批。

### 1.2 停机顺序（防自愈）
```bash
# worker 先 → head 后；每台 systemctl stop 后等 15-30s
for h in node01 node01 node01; do ssh $h "sudo systemctl stop vllm-tp4-worker.service"; done
ssh node01 "sudo systemctl stop vllm-tp4-head.service"
# 确认无残留容器/无 active 服务（速查 §1 复查命令）
```

### 1.3 启动顺序（head-first）
```bash
ssh node01 "cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh"
# 验证 8001=200 + 四机容器 Up(healthy) + NCCL banner 2.30.7+cuda13.0（速查 §2）
```

### 1.4 窗口结束清单
- 8001/4000 双端 200；四机 systemd active；`docker ps` 全 healthy。
- 若动了 rules.v4/netplan → 重新 `iptables-save-custom.sh` 落盘 + 外部 IP 自检（红线 7）。

## 2. 灰度发布流程（参数/补丁变更）

1. 目标参数一次只改一档（R11 已落地：util 0.65 / seqs 6 / capture 64；后续如 0.65→0.70 单步 +0.05）。
2. 改脚本 → `.bak-<tag>` → `check_vllm_script.sh` → **先 head 验证**（容器起 + API 200 + 性能冒烟）。
3. head 稳定后 worker 依次上线；全程保留 30 分钟观察窗口（prefix 命中、latency、OOM 日志）。
4. 异常 → 立即还原 `.bak-<tag>`（R11 前版本见 rollback §2.8）重走启动流程。

## 3. 回滚流程（速查）

| 场景 | 动作 | 锚点文档 |
|---|---|---|
| TP4→TP2 降级 | 四机 rm vllm-tp4-rank0~3 → 01 `start_v026r_cluster.sh` | rollback §3 |
| 脚本还原 | 用 `.bak-r11-20260812-2222xx` 覆盖（R11 前各档见 rollback §1.1/§2.8） | rollback §1.1 |
| NCCL 库回退 | `.bak-v2` 覆盖 → head-first 重启 | rollback §2.1 |
| shim 库回退 | `libncclpin.so.bak-v7` 覆盖（v8→v7） | rollback §2.2 |
| 网络回退 | `backup/ring-fix-20260811/<host>/` 还原 | rollback §2.3 |

## 4. 硬件/环境类维护

- **重启整机**：开机顺序 01→02→03→04；重启后复核 isolcpus=**8-9**(nproc=18)/MTU9000/GID/iptables/模型软链指向。
- **网络变更**：任何新 RoCE 网段配 IP → 同步放行 iptables（红线 5）；改 netplan/hosts 须窗口执行。
- **QoS**：`mlnx-qos-setup.sh` 目前无持久化 → 重启后手动执行并复验（P2 遗留，建议固化为 systemd 单元或 rc.local）。
