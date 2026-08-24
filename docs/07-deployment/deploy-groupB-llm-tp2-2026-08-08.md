# 组B（03/04）LLM TP2 部署 SRE 交付：GID 检查 + 启动脚本 + 部署步骤 + 风险

**日期**：2026-08-08
**角色**：Rex（SRE 工程师）| 团队 engineering-groupb-deploy
**关联**：A组 start_head_v026r.sh(v3.3, GID_INDEX=2) / start_worker_v026r.sh / migration-tp2-nccl-2026-08-08.md / file-registry-4node-2026-08-08.md

---

## 1. 03/04 GID 表检查命令与决策规则

**背景**：A组 <MGMT_OCTET> 系统重启后 RoCE GID index 3 为空 → `ibv_modify_qp failed 61` → 需 `NCCL_IB_GID_INDEX=2`（GID2=<NODE_IP> 有效）。03/04 是新建 RoCE 段（<RING_SUBNET>），部署前必须逐机查 GID 表。

```bash
# ① 列出 RoCE/IB 设备（对照 NCCL_IB_HCA 取值）
ssh node01 "ls /sys/class/infiniband/"
ssh node01 "ls /sys/class/infiniband/"

# ② 逐设备打印 GID 表（index 0-7）
for h in node01 node01; do
  echo "========== $h =========="
  ssh $h 'for d in /sys/class/infiniband/*/; do
            echo "[$d]";
            for i in 0 1 2 3 4 5 6 7; do
              printf "  idx%s: " $i; cat ${d}ports/1/gids/$i 2>/dev/null; echo;
            done
          done'
done
```

**决策规则**（逐 HCA、逐 port）：
- GID 值 **全零**（`00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00`）= 无效/空 → **不可用**
- 可用 GID 应为 RoCEv2 形态：`fe80::...`（link-local）或 `0000:...:ffff:<iphex>`（IPv4 GID，如 <NODE_IP> → `0000:0000:0000:0000:0000:ffff:0a64:8a02`）
- **取第一个非零且为 RoCEv2 形态的 idx 作为 NCCL_IB_GID_INDEX**；03/04 双 HCA（rocep1s0f1 / roceP2p1s0f1）通常一致，若不一致以 rocep1s0f1 为准
- **判定输出**：若有效 GID 在 idx2 → `NCCL_IB_GID_INDEX=2`（与 A组 <MGMT_OCTET> 相同）；若在 idx3 → `=3`；全空 → 检查 `ibstat`/驱动（部署阻塞）
- 脚本默认 `NCCL_IB_GID_INDEX=2`，检查结果不同时用 `NCCL_IB_GID_INDEX=<N> bash start_*.sh` 覆盖

**辅助验证（RoCE 链路/MTU）**：
```bash
# jumbo ping 互达（MTU 9000），期望 0% 丢包、RTT<1ms
ssh node01 "ping -c 3 -s 8972 <NODE_IP>; ping -c 3 -s 8972 <NODE_IP>"
ssh node01 "ping -c 3 -s 8972 <NODE_IP>"
# 邻接表：03 只应见 <RING_SUBNET>/<RING_SUBNET>（独立段确认）
ssh node01 "ip -4 addr show | grep 10.100.13; ip neigh | grep 10.100.13"
```

---

## 2. 启动脚本设计（已落盘）

| 文件 | 节点 | 关键参数 |
|------|------|---------|
| `start_head_groupB.sh` | 03/node01 | master <NODE_IP>:25055, rank0, port 8001 |
| `start_worker_groupB.sh` | 04/node01 | master <NODE_IP>:25055, rank1, 无 API |
| `start_groupB_cluster.sh` | 03 编排 | head-first → TCPStore → worker → API |

**相对 A组脚本的 8 处改造**：
1. **镜像**：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（21.6G, 9ea563a724d4）双机同 digest（03/04 已有；部署前 `docker images --digests` 复核一致）
2. **挂载路径**：模型 `<MODELS_DIR>/deepseek-v4-flash-0731:/models:ro`（03/04 是 <MODELS_DIR>，非 /home/<USER>）；nvcc_wrapper、vllm-envc-cache、vllm-cache、tilelang-cache、vllm-logs 全部持久化到 `<INSTALL_DIR>/{envs,cache,logs}`
3. **GID_INDEX**：`NCCL_IB_GID_INDEX=${GID_INDEX:-2}`（按 §1 检查覆盖）
4. **数据面 IP**：`VLLM_HOST_IP`/`VLLM_DP_MASTER_IP` = head <NODE_IP>（03）/ worker <NODE_IP>（04）；`MASTER_ADDR=<NODE_IP>`、`MASTER_PORT=25055`（避开 A组 25000）
5. **网卡**：`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`=`enp1s0f1np1,enP2p1s0f1np1`（138 数据面 + 139 备份面）；`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`
6. **显存/容量**：`--gpu-memory-utilization 0.88`（参照 A组）；`--max-model-len 131072`（组B 仅 95G 可用，见 §4 风险 2；A组 800000 需 ≥107G/机）
7. **NCCL 2.30.7**：`LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/aarch64-linux-gnu`（前插）；`NCCL_IB_TIMEOUT=1000`/`RETRY_CNT=7`/`DEBUG=INFO`+`NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h-%p.log`
8. **hostname 守卫**：head=`node01`，worker=`node01`；容器名 `vllm-groupb-head/worker`（与 A组 `vllm-envE-node/worker` 隔离）

**serve 参数基线（与 A组一致）**：`--kv-cache-dtype nvfp4_ds_mla`、`--max-num-seqs 6`、`--speculative-config dspark/5-token/probabilistic`、`--moe-backend flashinfer_b12x`、`--distributed-executor-backend mp`、`--enable-flashinfer-autotune`、`--max-cudagraph-capture-size 24`、`--api-key <API_KEY>-11282...`、`--distributed-timeout-seconds 300`、`--served-model-name deepseek-v4-flash-0731`。

**可选未启用**：`/opt/patch-v026`（A组 tilelang 两档 patch）——21.6G anemll fork 已含 mp KV fix；若组B JIT/性能异常再从 <MGMT_OCTET> 拷贝 patch 对齐。

---

## 3. 部署执行步骤（embed 卸载 → head → TCPStore → worker → 验证）

```bash
# ════════ 阶段 0：预检（Go/No-Go） ════════
# 03/04 双机：
for h in node01 node01; do
  echo "== $h =="
  ssh $h "docker images | grep 0.2.1-v026.0; ls -ld <MODELS_DIR>/deepseek-v4-flash-0731;
          du -sh <MODELS_DIR>/deepseek-v4-flash-0731; free -g | head -2;
          test -f <INSTALL_DIR>/envs/nvcc_wrapper.py && echo nvcc_wrapper:OK || echo nvcc_wrapper:MISSING"
done
# 若 nvcc_wrapper MISSING（关键坑位：<MGMT_OCTET>/<MGMT_OCTET> 曾误建为目录）→ 从 <MGMT_OCTET> 拷贝并校验 1710B：
ssh node01 "mkdir -p <INSTALL_DIR>/envs && scp node01:<INSTALL_DIR>/envs/nvcc_wrapper.py <INSTALL_DIR>/envs/ && wc -c <INSTALL_DIR>/envs/nvcc_wrapper.py"
ssh node01 "mkdir -p <INSTALL_DIR>/envs && scp node01:<INSTALL_DIR>/envs/nvcc_wrapper.py <INSTALL_DIR>/envs/ && wc -c <INSTALL_DIR>/envs/nvcc_wrapper.py"
# 权重完整性：48 分片/156G + 校验清单（若 <MODELS_DIR> 有 sha256sums.txt）
ssh node01 "cd <MODELS_DIR>/deepseek-v4-flash-0731 && ls | wc -l && sha256sum -c sha256sums.txt 2>/dev/null | tail -3"

# ════════ 阶段 1：embed 卸载（03/04 腾内存） ════════
ssh node01 "docker rm -f embed-qwen3-vllm 2>/dev/null || true"
ssh node01 "docker rm -f embed-qwen3-vllm 2>/dev/null || true"
ssh node01 "ss -tlnp | grep 8020 || echo '8020 free'"; ssh node01 "ss -tlnp | grep 8020 || echo '8020 free'"
# ↑ 只停容器不动镜像（测试完恢复）。litellm 池影响见 §4 风险 4。

# ════════ 阶段 2：脚本就位 + head 先起 ════════
# 将三脚本 scp 到 <INSTALL_DIR>/scripts/（chmod +x），在 node01 执行编排：
ssh node01 "bash <INSTALL_DIR>/scripts/start_groupB_cluster.sh"   # 内含 head→TCPStore→worker→API 全时序

# 手动分步（等价，便于定位）：
ssh node01 "bash <INSTALL_DIR>/scripts/start_head_groupB.sh"      # 1) head
ssh node01 "nc -z <NODE_IP> 25055 && echo TCPStore_OK"           # 2) 轮询 TCPStore（≤10min，实测 ~25s）
ssh node01 "bash <INSTALL_DIR>/scripts/start_worker_groupB.sh"    # 3) worker 后起（严禁先 worker）
# 4) 轮询 8001 就绪（权重加载 5-8min）

# ════════ 阶段 4：验证（见 §5） ════════
```

---

## 4. 风险清单

| # | 风险 | 评估 | 缓解 |
|---|------|------|------|
| 1 | **A组（01+02）不受影响** | 🟢 低 | RoCE 段独立（03/04=<RING_SUBNET> vs 01/02=<RING_SUBNET>）、宿主不同、端口不重叠（A组 TCPStore 25000@<RING_SUBNET>、API 8001@<MGMT_OCTET>；组B 25055@<RING_SUBNET>、8001@<MGMT_OCTET>）。确认：§1 `ip neigh` 只含对端；部署前后各打一次 A组 `http://<NODE_IP>:8001/health` 应持续 200 |
| 2 | **内存临界（主要风险）** | 🟠 中高 | 组B 03/04 仅 95G 可用/机，A组基线 ~98-107G/机（LLM 权重 split ~78G/机 + KV）。脚本用 util 0.88 + max-model-len 131072 压入；若 head 加载 OOM/`CUDA OOM` → 降 `--gpu-memory-utilization 0.80` 或 `--max-num-seqs 4`；全程 `free -g` 监控 |
| 3 | **nvcc_wrapper.py 缺失/被建目录** | 🟠 中 | <MGMT_OCTET>/<MGMT_OCTET> 曾现 `/tmp/env-e-build/nvcc_wrapper.py` 为目录 → 挂载失败/deepgemm JIT Assertion。部署前强制预检 + 从 <MGMT_OCTET> 拷贝 1710B（阶段 0） |
| 4 | **embed 卸载对 litellm 池** | 🟠 中（用户已批准） | 当前池成员需实查：`ssh node01 "grep -A20 local-embedding /home/<USER>/litellm/config.yaml | grep -E 'api_base|model'"`。若池={03:8020,04:8020} → 卸载后 **0 节点 → embeddings 500/超时（预期）**；若含 <MGMT_OCTET>:8022(anemll) → 剩 1 节点（无 HA、单点负载）。测试期如必须保留生产 embed，可暂缓卸载 04 或用 <MGMT_OCTET>:8022 兜底。恢复：测试完 `docker run` 重启 embed-qwen3-vllm → /health 200 → litellm cooldown(30s)/allowed_fails 后自动回纳 |
| 5 | **GID index 空** | 🟠 中 | 03/04 新建 RoCE 段未经验证 → 先 §1 检查；`ibv_modify_qp failed 61` 即 GID 空 → 按表覆写 `NCCL_IB_GID_INDEX` |
| 6 | **镜像 digest 不一致** | 🟡 低 | 03/04 均需 9ea563a724d4（21.6G）；用同 IMG tag，部署前 `docker images --digests` 复核 |
| 7 | **21.6G vs 34.2G head 差异** | 🟡 低 | A组 head 用 34.2G 完整版；组B 双 21.6G。版本串已实测一致（0.26.1.dev0，mismatch 归因时序竞态），但 tilelang 两档 patch 未验证 → 若 JIT/性能异常从 <MGMT_OCTET> 拷 patch 或评估拉 34.2G |

---

## 5. 验证命令

```bash
# ① health（期望 200）
curl -s -o /dev/null -w 'B组 8001: %{http_code}\n' -m 10 \
  -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" \
  http://<NODE_IP>:8001/health

# ② models（期望含 deepseek-v4-flash-0731）
curl -s -m 10 -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" \
  http://<NODE_IP>:8001/v1/models | head -c 400

# ③ 推理 smoke（期望 content 含 "4"）
curl -s -m 90 -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" \
  http://<NODE_IP>:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":64}' \
  | grep -o '"content":"[^"]*"'

# ④ NCCL 无 GID 错 + TCPStore（双机）
ssh node01 "docker logs vllm-groupb-head 2>&1 | grep -iE 'ibv_modify_qp|error' | tail -5; nc -z <NODE_IP> 25055 && echo TCPStore_OK"
ssh node01 "docker logs vllm-groupb-worker 2>&1 | grep -iE 'ibv_modify_qp|error' | tail -5"

# ⑤ 容器健康 + 内存
ssh node01 "docker ps --filter name=vllm-groupb --format '{{.Names}}|{{.Status}}'; free -g | head -2"
ssh node01 "docker ps --filter name=vllm-groupb --format '{{.Names}}|{{.Status}}'; free -g | head -2"

# ⑥ A组不受影响复核（期望持续 200）
curl -s -o /dev/null -w 'A组 8001: %{http_code}\n' -m 10 \
  -H "Authorization: Bearer <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa" \
  http://<NODE_IP>:8001/health

# ⑦ NCCL 运行时版本（期望 0x59df=2.30.7，若与 A组对齐）
ssh node01 "docker exec vllm-groupb-head python -c \"import torch;print(hex(torch.cuda.nccl.version()))\""
```

**恢复（测试完成后）**：worker 先停 → head 停（`docker rm -f vllm-groupb-worker` 04 → `vllm-groupb-head` 03）→ 重启 embed-qwen3-vllm（03/04）→ /health 200 → litellm 池回纳确认。
