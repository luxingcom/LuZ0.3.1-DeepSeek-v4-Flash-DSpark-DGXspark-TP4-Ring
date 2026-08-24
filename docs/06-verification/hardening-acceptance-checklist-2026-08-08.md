# P0/P1 加固验收验证清单（TP2 重建完成后执行）

**负责**：Tessa（测试专家）· 工程保障团队
**前置**：A 组（58+60）TP2 加固版编排重建完成（head .60:8001 /health 200）
**方法**：.58 上执行（ssh aicad-server），全部 PASS 才判验收通过；FAIL 记录定位

## 1. TP2 层（LLM 推理）
| # | 检查项 | 命令 | 通过标准 |
|---|--------|------|---------|
| T1 | head 健康 | `curl -s http://<NODE_IP>:8001/health` | 200/OK |
| T2 | 推理 sanity | 并发 3 次短问答（1+1→2 等） | 输出正确、无超时 |
| T3 | 容器状态 | `docker ps`（.58/.60） | head+worker 双端 Running、Restarts=0 |
| T4 | NCCL 2.30.7 运行时 | `python3 -c "import torch;print(hex(torch.cuda.nccl.version()))"` + dladdr 查 libnccl 路径 | 0x59df 且路径=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 |
| T5 | 挂载持久化 | `docker inspect <head/worker> | jq '.[].Mounts[].Source'` | 模型 /models:ro、nvcc_wrapper、vllm-cache、tilelang-cache、vllm-logs 全在 |
| T6 | TP 配对 | head 启动日志 NCCL init 判定（脚本 check_nccl_init） | PASS（rank0/1 真正互连，非仅 HTTP 200） |

## 2. P0/P1 加固项（SRE 落地，主理人执行）
| # | 检查项 | 命令 | 通过标准 |
|---|--------|------|---------|
| H1 | 环境迁移 nvidia-sync | `ls <INSTALL_DIR>/envs/`、`docker inspect <容器> | grep -i env` | envs 目录存在且容器引用正确 |
| H2 | 桌面服务 disable | `systemctl list-units --type=service --state=running \| grep -iE 'gdm|gnome|lightdm|kde'` | 无桌面服务运行 |
| H3 | vllm-cluster.service enable | `systemctl is-enabled vllm-cluster.service`（.58/.60） | enabled（自启动/自愈） |
| H4 | 编排 v2.0 生效 | `systemctl status vllm-cluster.service` | active；编排含 head-first + NCCL_IB_GID_INDEX=2 + 双停双启顺序 |
| H5 | 端口监听 | `ss -ltnp \| grep -E ':(8001|25000|25055)'` | head 8001 LISTEN、TCPStore 25000 LISTEN |
| H6 | 自愈判定（vllm-cluster.service） | 见下方 H6 详述 | 场景 A/B 均达判 |
| H7 | 幂等性 | 已就绪重跑 `start_v026r_cluster.sh` | exit 0 且 head/worker `State.StartedAt` 不变（未重复启停） |
| H8 | 守卫触发 | `rm nvcc_wrapper.py && mkdir 同名` → 跑 `/usr/local/bin/guard_mount.sh` | 自动修复为普通文件 + 容器可正常启动 |
| H9 | 日志轮转 | `ls /var/log/vllm-cluster/` + `logrotate -d /etc/logrotate.d/vllm-cluster` | 目录存在 + dry-run 无错 |

### H6 自愈判定详述（SRE 补充）
- **场景 A 崩溃自愈**：`docker kill $(docker ps -qf name=head)`（或 kill 主进程）→ 15s 内 systemd 自动拉起整套（Restart=on-failure, RestartSec=15）。**成功判据**：`systemctl status vllm-cluster` restart 计数 ≥1 且最终 `curl -sf 127.0.0.1:8001/v1/models` 可用；head 容器 `State.StartedAt` 晚于 kill 时刻。
- **场景 B 熔断保护**：连续失败触发 StartLimitBurst=3/600s 后，`systemctl status vllm-cluster` 进入 **failed**（不允许无限循环重启），`systemctl reset-failed` 可复位。**判定 = 有熔断不是 bug**。
- **顺序保证**：`docker inspect -f '{{.Name}} {{.State.StartedAt}}'` 比较 head 与 worker，**必须 head < worker**（head-first）；ExecStop 反向（先 worker 后 head）。
- **容器 restart 策略**：`docker inspect --format '{{.HostConfig.RestartPolicy.Name}}'` 应为 **no / on-failure:5**，而不是 unless-stopped（并行重启会复现 Gloo 竞态，这是加固点）。

### T4 细化（SRE 补充）：NCCL_IB_GID_INDEX=2 持久化
- 同时查 `docker inspect .Config.Env`（写死在容器定义，节点重启后仍在）与容器内 `printenv NCCL_IB_GID_INDEX`，**两者一致才算持久化生效**（不仅看运行时 0x59df）。

## 3. 监控层（Grafana/Prometheus）
| # | 检查项 | 命令 | 通过标准 |
|---|--------|------|---------|
| M1 | vllm target 在线 | Prometheus `up{job="vllm"}` | head-60 up |
| M2 | 指标有数据 | `vllm:generation_tokens_total` / `vllm:prompt_tokens_total` rate>0（压测时） | 面板出数 |
| M3 | 面板解耦 | vllm-realtime 104/105 + cluster 3/10 已复核（v19 通过） | 已通过 ✅ |

## 验收判定
- T1~T6 + H1~H9 + M1 全部 PASS → **加固验收通过**，回报 team-lead 出验收报告（cc SRE）
- 任一 FAIL → 记录证据（命令输出/日志片段），回报 team-lead + SRE 定位（SRE/Rex 分诊）
- T4 若重启后回退 2.28.9 → 环境变量丢失，需复插 LD_LIBRARY_PATH（Cody 关注项）
- H6 场景 B：进入 failed 是**预期熔断**非 bug；复位用 `systemctl reset-failed`

## 执行入口（.58）
```bash
# 关键检查一把梭（.58 上）
ssh aicad-server '
  curl -s http://<NODE_IP>:8001/health
  docker ps --format "{{.Names}} {{.Status}}"
  docker inspect $(docker ps -q --filter name=vllm-head) | jq -r ".[0].Mounts[].Source" 2>/dev/null
  systemctl is-enabled vllm-cluster.service 2>/dev/null
  curl -sG "http://localhost:8191/api/v1/query" --data-urlencode "query=up{job=\"vllm\"}" | jq -c .data.result
'
```
