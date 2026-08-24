# vLLM TP=2 双机 NCCL init 卡死修复部署报告

**日期**：2026-08-06
**工作流**：SRE 生产修复实施（事故响应 阶段 3 缓解 → 阶段 4 验证）
**执行**：雷克斯（SRE 工程师）
**关联文档**：`investigate-nccl-init-hang-2026-08-06.md`（根因调查）

---

## 📌 TL;DR

- **修复落地并 3/3 验证通过**：H1 启动顺序竞态通过「编排脚本颠倒启动顺序（head 先 → 轮询 25000 → worker 后）」直接消解
- 3 次全新重启全部成功（head+worker 双机 healthy、8001/8003/4000 全 200、e2e "2+2=?" 答 "4"）
- 新增 `NCCL_DEBUG=INFO + NCCL_DEBUG_FILE` 双机留证（`~/vllm-logs/nccl-*.log`）
- **生产标准启动流程已切换**：今后重启用 `bash ~/start_v026r_cluster.sh`（head 机执行），**禁止**再手动 worker→head 顺序

---

## 1. 修复内容

### 1.1 启动脚本修改（双机）

| 文件 | 变更 |
|------|------|
| `~/start_head_v026r.sh`（head <MGMT_OCTET>） | ① serve 加 `--distributed-timeout-seconds 300` ② `VLLM_ENGINE_READY_TIMEOUT_S` 7200→600 ③ `NCCL_DEBUG` WARN→INFO ④ 新增 `NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log`（容器内 `/var/log/vllm` = 宿主机 `~/vllm-logs` 挂载点） |
| `~/start_worker_v026r.sh`（worker <MGMT_OCTET>） | 同上 4 项（worker serve 无 `--port`，其余一致） |

- 修改前均备份：`~/start_head_v026r.sh.bak.20260806_ncclfix` / `~/start_worker_v026r.sh.bak.20260806_ncclfix`
- 双机 `bash -n` 校验通过；`grep` 确认 5 处变更点全部命中
- **SRE 注**：`/vllm-logs` 在容器内不存在，实际使用容器内已挂载的 `/var/log/vllm` 路径（对应宿主机 `~/vllm-logs`），保证 NCCL 日志落到预期位置

### 1.2 编排脚本（核心修复）

**路径**：`~/start_v026r_cluster.sh`（head 机，已 chmod +x，版本 v1）

```
流程: ①前置检查(双机容器须已停止) 
      ②启动 head(rank0) 容器 (nohup, 日志 ~/start_head_v026r_cluster.log)
      ③轮询 head TCPStore :25000 就绪 (nc -z <NODE_IP> 25000 或 host ss -tln, 每 5s, 最多 10min)
      ④ssh worker 执行 ~/start_worker_v026r.sh 启动 rank1
      ⑤轮询 head :8001/v1/models (internal key, 每 5s, 最多 10min)
      ⑥输出集群就绪
```

- **机制**：head(rank0) 先启动并创建 TCPStore(25000) 后 worker(rank1) 才启动 join——消除 H1 竞态
- `bash -n` 校验通过
- **坑位记录**：head 机访问 worker 的 SSH 别名是 `node0X`（<NODE_IP>），不是本地机的 `aicad-server`；`docker exec ss` 在容器内看不到 25000（容器无 ss 权限），端口探测统一用宿主机 `nc -z`/`ss -tln`

---

## 2. 三次重启验证结果

**验证方法**：每次停容器（worker 先停 → head 后停）→ 编排脚本重启 → 等 head 8001 就绪 → 全链路验证

| 第 N 次 | 启动顺序 | TCPStore 就绪耗时 | API 就绪耗时(总) | 成功 | 8001 | 8003 | 4000 | e2e(2+2=?) |
|--------|---------|------------------|-----------------|------|------|------|------|-----------|
| 1 | head→worker（新） | ~25s | ~5m18s | ✅ | 200 | 200 | 200 | "4" |
| 2 | head→worker（新） | ~25s | ~4m36s | ✅ | 200 | 200 | 200 | "4" |
| 3 | head→worker（新） | ~25s | ~5m13s | ✅ | 200 | 200 | 200 | "4" |

- 3/3 成功（旧顺序 4 次 3 挂）；每次 head+worker 容器均 `healthy`
- 时间线样本（Run 1）：`11:54:17` 启 head → `11:54:42` TCPStore 25000 就绪 → `11:54:55` worker 启动成功 → `11:59:35` API 200
- 完整日志：`~/v026r_cluster_run{1,2,3}.log`（head 机）

---

## 3. NCCL_DEBUG 日志产出确认

| 主机 | 文件 | 大小 | 说明 |
|------|------|------|------|
| head (<MGMT_OCTET>) | `~/vllm-logs/nccl-spark-05cd.log` | ~21KB | NCCL INFO 级，含 Bootstrap/RoCE 握手等 254 行 |
| worker (<MGMT_OCTET>) | `~/vllm-logs/nccl-edgexpert-0c69.log` | ~17KB | 同上 186 行 |

- 已确认每次重启都会覆盖写入（时间戳随运行更新），后续卡死排查可直接取用

---

## 4. 生产最终状态

- **保持运行**：Run 3 之后的健康状态即为生产状态（head `vllm-envE-node` + worker `vllm-envE-worker` 双机 `healthy`）
- 8001（internal key）/ 8003（客户端 key）/ 4000（LiteLLM master key）全部 200
- 8003/4000 LiteLLM 网关全程未触碰，不受 vLLM 重启影响
- vLLM 版本不变：0.26.1.dev0（anemll 0.2.1-v026.0），probabilistic + 动态K + tilelang + NCCL 加固（TIMEOUT=1000/RETRY_CNT=7）全部保留

---

## 5. 回滚方法

| 场景 | 步骤 |
|------|------|
| 启动脚本回滚 | head：`cp ~/start_head_v026r.sh.bak.20260806_ncclfix ~/start_head_v026r.sh`；worker：`cp ~/start_worker_v026r.sh.bak.20260806_ncclfix ~/start_worker_v026r.sh` |
| 编排脚本回滚 | 删除/弃用 `~/start_v026r_cluster.sh`，恢复手动顺序（但**不推荐**——旧顺序已知 3/4 失败率） |
| 容器回滚 | 镜像 digest 不变（`ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0`），旧脚本 `.bak.20260806_nccl` 亦可用 |
| 全部重置 | 双机停容器（worker 先 head 后）→ 用 `.bak.20260806_nccl` 恢复脚本 → 手动顺序启动 |

---

## 6. SOP 固化

**今后双机 vLLM 重启标准流程（生产）**：

```
1. 停容器（顺序不可反）：
   ssh aicad-server        'docker rm -f vllm-envE-worker'
   ssh aicad-server60      'docker rm -f vllm-envE-node'
2. 启动（head 机执行编排脚本，后台）：
   ssh aicad-server60      'nohup bash ~/start_v026r_cluster.sh > ~/v026r_cluster_runN.log 2>&1 &'
3. 轮询就绪（5-8 分钟权重加载）：
   curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer <API_KEY>-*" http://127.0.0.1:8001/v1/models  # 期望 200
4. 验证：8001/8003/4000 全 200 + e2e chat
```

- **禁止**单边重建、禁止手动 worker→head 顺序
- 卡死排查证据：`~/vllm-logs/nccl-*.log`（双机）+ `~/v026r_cluster_runN.log` + `~/start_head_v026r_cluster.log`

---

## 📊 验证数据留档

- 编排日志：`~/v026r_cluster_run1.log` / `run2` / `run3`（head 机）
- NCCL 证据：`~/vllm-logs/nccl-spark-05cd.log`（head）/ `~/vllm-logs/nccl-edgexpert-0c69.log`（worker）
- 脚本：`~/start_v026r_cluster.sh`、`~/start_head_v026r.sh`、`~/start_worker_v026r.sh`（含 `.bak.20260806_ncclfix` 备份）

---

> 本报告由工程保障团队 SRE 生成。修复验证已完成（3/3），生产保持最终状态运行；如需调整请由工程负责人复核。
