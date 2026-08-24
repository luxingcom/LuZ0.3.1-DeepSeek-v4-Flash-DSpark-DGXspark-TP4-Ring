# 自恢复方案（Self-Recovery）v1.5-R11

**日期**：2026-08-12（R11 修订）｜**维护**：Docu｜**目标**：容器异常退出后无需人工介入自动恢复，且恢复链路保证 vLLM rank 全量重新注册
**R11 变更**：StartLimit 600/8→**1800/20**（退避放大）；新增**互杀守卫**（见 §3.3）；新增 GPU-gate/就绪门禁说明。

---

## 1. 自愈机制架构

```
systemd(vllm-tp4-head.service) ── ExecStart ── monitor_tp4_head.sh (01)
systemd(vllm-tp4-worker.service)── ExecStart ── monitor_tp4_worker.sh (02/03/04)
        │
        ├─ 容器在(docker ps 命中) → docker wait（跟随退出，退出码非零）→ systemd Restart
        ├─ 容器不在(head)  → ssh 清 worker(02/04/03) → start_tp4_head.sh 重建 head
        └─ 容器不在(worker) → head API 200 且集群成形? → rm rank0 触发全链重建；否则等 TCPStore:25999
```

- **容器 `--restart no`**：docker daemon 不直拉，避免重启风暴；生命周期完全由 systemd 掌控。
- **monitor 恒 exit 非零**：让 systemd Restart=always 接管，RestartSec=15 节流。
- **head 重建前置**：先清 worker 容器（各机 systemd 感知自己容器消失→依次自愈重建），保证所有 rank 重新注册 TCPStore。

## 2. systemd 单元参数（R11 实测）

| 参数 | 值 | 含义 |
|---|---|---|
| Type | simple | monitor 前台运行 |
| Restart | always | 恒重启 |
| RestartSec | 15 | 失败冷却 15s |
| StartLimitIntervalSec | **1800** | 1800s 窗口（R11，原 600） |
| StartLimitBurst | **20** | 窗口内最多 20 次（R11，原 8；防崩溃循环空转） |
| TimeoutStartSec | 1500 | 启动最长 25 分钟（权重加载） |
| User | <USER> | 非 root 运行 |
| Environment | HOME、NODE_RANK、VLLM_HOST_IP、NCCL_IB_HCA | worker 参数注入 |

- 单元文件：01 `/etc/systemd/system/vllm-tp4-head.service`；02/03/04 `vllm-tp4-worker.service`。
- 单元内 `Documentation=file://<INSTALL_DIR>/scripts/start_tp4_*.sh` → 帮助头引用速查文档。

## 3. 恢复链路（逐级）

1. **容器崩溃** → monitor docker wait 感知 → systemd RestartSec=15 后重跑 monitor → 重建该 rank。
2. **head 容器消失** → head monitor：先 `ssh` 清 02/04/03 的 worker 容器 → 重建 head（NO_WAIT=1）→ 各 worker 自愈感知容器消失 → 等 head TCPStore 重建。**注**：此处清理顺序（02/04/03）是"沿环邻递归清理以便全量重注册"，与停机维护顺序（worker 先 03→04→02 后 head）**用途不同**——重建清理不涉及停机节奏，仅保证 head 先起、无 worker 残留；勿混淆为停机顺序。
3. **worker 容器消失且 head 健康且集群成形** → worker monitor 判断 head API=200 **且 rank 全连 TCPStore** → `rm vllm-tp4-rank0`（主动触发 head 全链重建）→ sleep 25 → 等 TCPStore 再重建本 rank。**设计意图**：单 rank 缺失不静默，强制全链一致性。
3.1 **互杀守卫（R11）**：worker monitor **先确认集群已成形（rank 全连 TCPStore）才允许动 head**；冷启动阶段（TCPStore 未成形）head API 即使可达也**不动 head**，仅等待重建本 rank。防"冷启动互杀"（worker 先起把 head 误杀）。判别依据 `journalctl -b -u vllm-tp4-worker | grep "head API 健康"`。
4. **worker 容器消失且 head 不可用** → 等 head TCPStore（≤300s 每 5s 探测）→ 重建。
5. **超限**：1800s 内 20 次失败 → systemd failed（停止空转）→ 人工介入（速查 §4 错误码表）。

## 4. 验证方法（自愈演练）

```bash
# 演练 A：worker 容器崩溃自愈（以 02 为例）
ssh node01 "docker kill vllm-tp4-rank1"
sleep 60   # RestartSec=15 + 权重加载窗口
ssh node01 "systemctl is-active vllm-tp4-worker.service; docker ps | grep vllm-tp4-rank1"   # active + Up
ssh node01 "curl -o /dev/null -w '%{http_code}\n' http://<NODE_IP>:8001/health"          # 200

# 演练 B：head 容器崩溃全链自愈
ssh node01 "docker kill vllm-tp4-rank0"
sleep 180   # head 重建 + 各 worker 依次自愈 + 权重加载
ssh node01 "systemctl is-active vllm-tp4-head.service; docker ps | grep vllm-tp4-rank0"
for h in 02 03 04; do ssh dgxspark$h "docker ps | grep vllm-tp4-rank; systemctl is-active vllm-tp4-worker.service"; done
ssh node01 "curl -o /dev/null -w '8001=%{http_code}\n' http://<NODE_IP>:8001/health"
# 全链路恢复判定：四机容器 Up + systemd active + 8001=200 + NCCL banner 正确
```
> 演练在窗口期执行；演练前通知 litellm 下游；完成后检查 `/var/log/vllm/nccl-*.log` 无异常。

## 5. 观测点

- 状态：`systemctl status vllm-tp4-*`、`journalctl -u vllm-tp4-head -f`。
- 失败计数：`systemctl show vllm-tp4-head -p NRestarts`（20 次/1800s 上限参考，R11）。
- 自愈痕迹：monitor 日志输出 `[i] head API 健康但 ${NAME} 缺失 => 触发 head 全链路重建`。
- **互杀守卫痕迹**：`journalctl -b -u vllm-tp4-worker | grep -c "head API 健康"`（期望当前 boot 为 0-1）。

## 6. 已知边界

- 自愈解决**容器级**故障；**进程级**（容器在但 vLLM 死锁/API 挂）不会被 docker wait 感知——需外部健康探针（建议增加 curl :8001/health 判活后主动 kill 容器，P2 增强项）。
- systemd failed（20 次超限）后不会自愈，须人工按速查 §2/§4 处理。
- 整机重启后自愈由 `multi-user.target` 依赖拉起（unit 已 enable），开机顺序仍须 01→02→03→04。
