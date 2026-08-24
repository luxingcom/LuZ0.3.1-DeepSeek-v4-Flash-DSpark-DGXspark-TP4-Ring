# 生产启停纪律与运行反馈响应速查（Ops Discipline QuickRef）v1.5-R11

**日期**：2026-08-12（R11 修订）｜**维护**：Docu｜**依据**：生产脚本实测 + 8/11-8/12 事故复盘
**引用**：本文件为脚本帮助头/维护手册/tools-index 的**唯一纪律权威**。变更前先读本文件。
**R11 变更**：隔离核 **8-9**（NCCL 绑 8-9）、EngineCore **15-19**、seqs=**6**、util=**0.65**、StartLimit **1800/20**、互杀守卫（冷启动未成形不动 head）、本地 serving（NFS 恢复中）。

---

## §0 铁律速记

1. **启序**：head-first（01 rank0 先）→ worker（02→04→03）；**停序反向**：worker 先停（03→04→02）→ head(01) 最后。
2. **禁止**：worker 单边重建 / head 未就绪先启 worker；TP4 编排唯一入口 `start_tp4_cluster.sh`（01）。
3. **防自愈**：任何"停机维护"必须先 **systemctl stop**（停 systemd 防自动重建），否则 docker rm 会被 monitor 秒级拉起。
4. **改脚本**：先 `cp` 留 `.bak-<tag>` + `check_vllm_script.sh` 通过 + Rex 验证，禁止裸改。
5. **回滚**：见 `rollback-anchors-2026-08-12.md`；TP2 降级唯一入口 `start_v026r_cluster.sh`（01，全程未动）。
6. **rules.v4**：只允许 `iptables-save-custom.sh` 落盘（剔除 docker 动态 DOCKER 链）；新 RoCE 网段配 IP 必须同步放行 iptables。
7. **自检用对外 IP**：`curl http://127.0.0.1:8001/...` 会被 docker-proxy 绕过造成假象，必须用 192.168.x 或环内地址。

## §1 停机（维护窗口）标准流程

```bash
# 四机：先停 systemd（防 monitor 自动拉起）
ssh node01 "sudo systemctl stop vllm-tp4-worker.service"
ssh node01 "sudo systemctl stop vllm-tp4-worker.service"
ssh node01 "sudo systemctl stop vllm-tp4-worker.service"   # worker 先停
ssh node01 "sudo systemctl stop vllm-tp4-head.service"     # head 最后

# 确认 monitor 已退出、无残留容器（避免自愈残留进程占用 GPU）
for h in node01 node01 node01 node01; do
  ssh $h "docker ps -a --filter name=vllm-tp4-rank --format '{{.Names}} {{.Status}}'; systemctl is-active vllm-tp4-head.service vllm-tp4-worker.service" 
done
# 期望：无容器输出；systemd 全 inactive/failed（未 stop 则为 active 需补停）
# 残留处置：docker rm -f vllm-tp4-rank0~3（确认 monitor 已停才可删）
```

**防自愈停法要点**：①stop 顺序先 worker 后 head；②stop 后**等 15-30s** 再操作容器（monitor 的 RestartSec=15）；③`systemctl is-active` 全部非 active 才算停净；④docker rm 后立即 `docker ps` 复查 5 分钟内无"幽灵重建"。

## §2 启动标准流程

```bash
# 前置（重启后必查）：
#   isolcpus=8-9(nproc=18) / MTU 9000 / GID index=2 / /opt/nccl-ringonly md5
#   01/02 内存余量 ≥10G（必要时清 buff/cache）
#   四机 libncclpin.so 在位（shim v8，8-9 绑定）
cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh   # head-first 编排

# 验证：
curl -s -o /dev/null -w '%{http_code}' http://<NODE_IP>:8001/health   # =200
for h in 01 02 03 04; do ssh dgxspark$h "docker ps | grep vllm-tp4"; done  # 全 Up(healthy)
docker logs vllm-tp4-rank0 --tail 50 | grep -i "NCCL version"  # 期望 2.30.7+cuda13.0
```

**启动即停**：systemd 已在 `multi-user.target` 开机自启，正常重启**勿手动启**（head monitor 会自动拉起 head+清 worker 链式自愈）。

## §3 灰度 / 参数调整（R11：util 0.65 / seqs=6 已落地）

1. 改 `start_tp4_head.sh`（+ worker 若参数一致）→ 先 `cp` `.bak-<tag>`。
2. `bash check_vllm_script.sh start_tp4_head.sh` 必须全过（check 关键参数已同步 `max-model-len 400000` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`）。
3. 停→改→启走 §1/§2。**执行口径**：全局参数（util/seqs/capture）采用四机并发重建（head+worker 同窗口一次到位）；单机局部参数先 head 后 worker。
4. 回退：还原 `.bak-<tag>`（R11 前版本见 rollback §2.8）重走 §2。

## §4 运行反馈错误码速查（帮助头 EXITCODES 引用此表）

| 现象/错误码 | 判定 | 响应动作 |
|---|---|---|
| 8001 不通 / curl 000 | head 未就绪或崩溃 | `ss -ltnp\|grep 25999` 查 TCPStore；head-first 重来；权重加载需 5-8 分钟 |
| 容器崩溃循环（Up(restarting)） | systemd 自愈反复失败 | `journalctl -u vllm-tp4-head -n 100` 定位；NCCL_DEBUG 日志 `/var/log/vllm/nccl-*.log`；banner 查 `NCCL version` 是否 13.3（LD_PRELOAD 失效） |
| NCCL 110（ibv_modify_qp） | 补丁库/PEER_HCA 失配 | 查 PEER_HCA 表（runbook §A.4）+ banner `2.30.7+cuda13.0`；`/opt/nccl-ringonly` md5 `b7784b49885659c27765e648884e4edd` |
| KV 不足 / OOM | util 或 max-num-seqs 过高 | 清 buff/cache；降 `--max-num-seqs`（6→4）或 util 0.65→0.60；勿动 400k 长度 |
| prefix 命中为 0 | 缓存未生效 | 确认 `--enable-prefix-caching` 在 rank0；四机 KV 状态 `docker logs ... \| grep prefix`；请求需含相同前缀 |
| 单 rank 掉线但 head API 200 | worker 故障 | monitor 互杀守卫：仅集群成形时触发 head 重建（rm rank0）；人工勿插手，观察 5 分钟 |
| healthy 但推理错/超时 | 权重加载中或队列积压 | 等待；`curl /v1/models` 确认 served-model-name；litellm 并发 12 满则排队 |
| PSR 线程不在 8-9/15-19 | shim v8 绑定失效 | 查 `/proc/<pid>/status` Cpus_allowed_list；重跑容器或还原 `.bak-v7`（rollback §2.2） |

## §5 运行反馈响应时限

| 等级 | 定义 | 响应 |
|---|---|---|
| P0 | 全链路不可用（8001 全挂） | 立即 → 联系 Rex/oncall；按 §2 重建 |
| P1 | 单节点/性能劣化 | 15 分钟内定位；不擅自重启 |
| P2 | 观察项（FEC 计数/QoS 持久化复验） | 记录到日报，窗口处理 |
