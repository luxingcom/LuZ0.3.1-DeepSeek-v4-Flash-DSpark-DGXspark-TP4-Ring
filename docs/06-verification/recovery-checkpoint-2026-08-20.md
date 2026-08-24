# NVFP4 B12X 事故 —— 恢复检查点记录（镜像备份锚点）

**日期**：2026-08-20

## 一、事故恢复状态（锚点建立时）

- ✅ 4 rank 全部 healthy（vllm-tp4-rank0~3），GPU 0%
- ✅ 端到端推理通过（`cluster recovered OK`, TP4 fingerprint）
- ✅ 各 rank 均 `Using 'B12X_MXFP4'` + `Prewarmed route-pack`
- 事故根因：启动期 b12x cute/JIT 多 worker 并行加载竞态（非代码回归，未回滚任何文件）

## 二、镜像备份锚点

**生产运行镜像**：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`
- **镜像 ID**：`e100ddad568a9d6a0f64ec899b8af00871e6208e3a895dcbb0ffcceeb81b1c62`
- **创建时间**：2026-08-05T08:25:02+08:00

**恢复检查点 tag 锚点（本日新增）**：
```
<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0-b12x-recovered-20260820
```
- 指向同一镜像 ID e100ddad568a
- 用途：独立 tag 保护，防止原 tag 被覆盖/回滚失效后仍可引用该健康基线

**各节点本地镜像**：02/03/04 均含 `0.2.1-v026.0`（grep=1），与 01 一致。

## 三、恢复命令（Go-Recover 参考）

若以后再次遇到同类 B12X 启动失败/镜像问题，恢复步骤：

```bash
# 1. systemd 受控拉起（head-first，错峰）
sudo systemctl start vllm-tp4-head.service      # 01 rank0
# 确认 head 日志出现 "Using 'B12X_MXFP4'" + "Prewarmed route-pack" 后再：
sudo systemctl start vllm-tp4-worker.service    # 02=rank1, 04=rank2, 03=rank3
# 2. 验证
docker ps --filter name=vllm-tp4 --format '{{.Names}} {{.Status}}'   # 全 healthy
# 3. 端到端冒烟
#    content="cluster recovered OK" 且 finish_reason=stop, tp4 fingerprint
# 4. 若需回退到锚点镜像：
docker tag <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0-b12x-recovered-20260820 \
           <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0
```

## 四、镜像导出备份（如已执行）

- 目标：`<INSTALL_DIR>/backup/`
- 文件名：`vllm-tp4-image-b12x-recovered-20260820.tar`（~34GB）
- 校验：导出后 `sha256sum` 记录锚点镜像 ID
- 承载镜像 ID 一致性校验：docker images 的 IMAGE ID = 导出的 manifest 校验

## 五、隐患提示

- 该镜像 ~34GB，导出耗时较长，请后台执行
- 镜像仓库地址 <NODE_IP>:5000（02 节点，Prom 对外 8191 亦在 02）
- 锚点 tag 使用 `-b12x-recovered-20260820` 命名以标识"事故后健康基线"，勿与后续发布 tag 混淆

## 六、预防措施落地记录（2026-08-20，随检查点一并固化）

以下改动已落地并校验，均在 `start_tp4_*.sh` 启动脚本层（不涉及镜像层，对运行中容器无影响）：

### 6.1 head healthcheck 升级（防"进程活但引擎未就绪"盲区）
- 文件：`<INSTALL_DIR>/scripts/start_tp4_head.sh`（.bak-healthchk-20260820 留档）
- 改动：`--health-cmd "pgrep -f VLLM::EngineCore ..."` → **`curl -sf -o /dev/null -m 5 http://127.0.0.1:8001/health || exit 1`**
- 语义：head 暴露 8001，`/health` 返回 HTTP 200 才判 healthy（引擎 fully ready 才绑定）
- 校验：check_vllm_script ✅ 全部通过；worker 无 HTTP API，维持进程探针（合理，未动）

### 6.2 B12X 启动期隔离门禁（防多 worker 并行撞 b12x JIT 竞态——事故根因）
- 文件：`<INSTALL_DIR>/scripts/start_tp4_cluster.sh`（.bak-b12xgate-20260820 留档）
- 改动：新增 **step 2.5**——head TCPStore 就绪后、启 workers 前，轮询 head 日志出现 `Using 'B12X_MXFP4' Mxfp4 MoE backend` 才放行（限时 `${B12X_GATE_WAIT:-300}s`，超时中止+diagnose）
- 语义：确保核心 B12X 加载成功后再错峰启 3 worker，从编排层阻断并发加载竞态
- 校验：SYNTAX OK

### 6.3 恢复路径与门禁的配合
- 完整恢复走 `start_tp4_cluster.sh`（head-first + TCPStore + B12X gate）或 systemd 错峰：先 `systemctl start vllm-tp4-head.service` 确认 B12X 后，再启 3 worker

### 6.4 待办（需镜像层改代码/评审，未在本轮落地）
- b12x import 重试 + DEBUG 捕获瞬时 ImportError（镜像层 `b12x_mxfp4_moe.py`，风险高待评审）
- Exited/8001/引擎未就绪告警链路
- SEV1 受控拉起预案演练