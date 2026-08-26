# 生产加固方案：部署能力强化 + 运维健壮性强化

**文档类别**：生产加固规范（由克隆环境分层预防修复报告衍生）
**编制**：Archi(架构师) — 供 Rex(SRE) 落地
**基线**：start_tp4_cluster.sh v1.5-r12 / start_tp4_head.sh v1.5-r11 / start_tp4_worker.sh v1.5-r12（2026-08-25 稳态）
**原则**：判断式语言 + 明确 exit code 语义 + 决策表。改动须 `check_vllm_script.sh` 通过 + `.bak-<tag>` 留档 + 更新 REFERENCE.md。

---

## 一、目标与范围

**目标**：把「分层预防性修复思想」（MEMORY.md 崩溃分层裁决：运行中 QP 断 vs 重启 GID 空洞 vs 宿主内存三线分离 + connect.cc:317-321 写死 GID_INDEX 盲信跳动态选择）固化为**生产启动与运维规范**，使布局不一致在**连接建立前**被识别/降级/阻断，崩溃**证据必留**，卡死被**守卫活性探针**闭合。

**范围（四支柱 + 运维）**：
1. 连接建立前的有效识别与管理
2. 崩溃问题有效记录
3. 布局不一致时的启动检查/降级/报错
4. 环网补丁与连接建立的加载一致性
5. 运维健壮性（守卫活性探针/看门狗窗口化/timeout 空窗/日志留存）

**非目标**：不改模型/推理路径；不改变既有 NCCL T1aM4+MAX_CH16 延迟优化参数结论；不动 memdesc NV_ERR 使能条件裁决（已排除为触发项）。

**边界**：本方案针对 **生产四机**；克隆环境另按克隆预防报告走 Go/No-Go 清单。

---

## 二、部署能力强化（四支柱）

### A. 连接建立前的有效识别与管理

**判据总纲（GID / RoCE 布局，插入 cluster.sh step 2 之前 = 连接建立前）**：
新增独立函数 `preflight_gid()` 于 head/worker 的 `check_vllm_script` 之后、`docker rm` 之前（head line157-161 / worker line185-189 之间插入）。

**A1. GID/RoCE 布局预检（脚本级）**：
- **index3 空洞判定**：枚举 `/sys/class/infiniband/*/ports/*/gids/[0-9]*`；任一 `gids/N` 为全 0 → HOLE。判定 index3（现网写死值）位 = 空洞 → **No-Go**。
- **子网前缀一致性**：校验 index3 GID 的 subnet prefix == 该机 RoCE 实际网段 L2 前缀；且**四台 index3 子网前缀必须逐位一致**（不一致 → 写死 index 跨网段 → No-Go）。
- **口名映射失效判定**：预期口名 `rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1`；任一非此名 → HCA/PEER_HCA 全要重算 → **No-Go**（禁止带旧 HCA 启动）。
- **连通判定**：`ibping` 验证 indexGID 可达对端（可降级为 P3 软门，见二级响应）。

**A2. NCCL 关键 env 一致性识别**：
- `NCCL_IB_GID_INDEX`（head line102 / worker line107 写死 3）、`NCCL_IB_HCA`（line103/108 4 twin 口）、`NCCL_IB_PEER_HCA`（worker line124 per-peer）、`NCCL_IB_RETRY_CNT=7`（line105/110）、`NCCL_IB_TIMEOUT=1000`（line104/109）须与实际口映射匹配。预检校验脚本内 HCA 列表 ⊆ 实测可用 RoCE 口集。

**A3. 布局不一致时的三级响应（关键新增）**：
| 级 | 触发条件 | 响应动作 | exit code |
|----|---------|---------|----------|
| **fail-fast 硬停** | index3 空洞 / 同台 I/O 空洞 / 四台子网前缀不一致 / 口名非预期集 | `preflight_gid` 立即打印判据不满足项，退出，**绝不在创建容器后诊断** | **3**（GID 布局 No-Go，新增保留值） |
| **降级（自动退动态选择）** | GID[3] 有效但连通判据 ibping 未达；或写死 index 跨台不稳 | 自动替换 `NCCL_IB_GID_INDEX=3` → **设 `-1`**（交 connect.cc 动态选择），打印降级告警 | 0（带 WARN） |
| **报错（明确 code+日志）** | env 一致性校验发现 HCA 列表/PEER_HCA 与实测口失配 | printf 明确错误码 `E_HCA_MISMATCH/<rank>`，写日志，**阻断启动** | **4**（env 一致性失配，新增） |

> **语义说明**：`exit 3`、`exit 4` 为新增保留位，务必写入三个脚本头部 EXITCODES 注释（cluster line11 / head line11 / worker line11），并纳入 check_vllm_script 校验白名单。与既有 0/1/2/130 语义不冲突：0=成功、1=业务失败、2=用法错误(可重试)、3=GID 布局 No-Go(不可重试需人工)、4=env 一致性失配、130=signal。

---

### B. 崩溃问题有效记录（闭合 RCA 最大缺口）

**B1. NCCL_DEBUG_FILE 落持久卷（现网已具备，需固化核验）**：
- head line101 `NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log` 已配，且 line192 `-v ~/vllm-logs:/var/log/vllm` 已挂宿主持久目录 ✅。worker line106/220 对称 ✅。
- **强化**：在 head/worker 启动成功分支（head line202 / worker line230 `[ok] READY`）后核验四机 `ls -la ~/vllm-logs/` 是否已生成 `nccl-*.log`；未生成 → WARN（首启核验兜底，防"声称已落地实测未落地"重复模式）。

**B2. 崩溃 ExecStopPost dump docker logs（新增，当前缺失）**：
- 在 systemd unit（monitor_tp4_head/worker 生成的 unit）的 ExecStopPost 追加：
```bash
ExecStopPost=/bin/sh -c 'docker logs vllm-tp4-rank${R} > /opt/aicad-prod/backup/crash-dump-$(date +%Y%m%d-%H%M%S)-rank${R}.log 2>&1 || true'
```
- 触发条件：仅在 `ExecStart` 返回非 0 时 dump（用 ExecStopPost 的 `$SERVICE_EXIT_CODE` 判断，=SUCCESS 不 dump）。dump 目标 `/opt/aicad-prod/backup/` 须确认在持久盘。

**B3. 崩溃指纹留存（新增）**：
崩溃时采集三件套到同一 crash-dump 文件：
1. **dmesg** RoCE 事件：`dmesg | grep -iE 'NM_ERR|carrier|link (down|up)|roce' | tail`
2. **容器日志**：docker logs（见 B2）
3. **时间戳**：`date -u +%Y-%m-%dT%H:%M:%SZ` + 对应 alarm 时间戳
> 目的：任何崩溃先分层（QP 断 vs GID 空洞 vs 宿主内存）再归因——指纹三件套支持三线分类。

---

### C. 布局不一致启动检查/降级/报错完善

**C1. 三级启动决策表（汇总）**：
| 输入判据 | 判定 | 启动动作 | exit |
|----------|------|---------|------|
| 任一 GID 空洞（含 index3） | No-Go | 阻断，打印判据 | 3 |
| 四台 index3 子网前缀不一致 | No-Go | 阻断，打印判据 | 3 |
| 口名非预期集 / HCA 失配 | No-Go | 阻断，打印 E_HCA_MISMATCH | 4 |
| GID[3] 有效，ibping 未达但对端 soft | 降级 | GID_INDEX=3 → -1 动态 | 0 WARN |
| GID[3] 有效 + 子网一致 + 口名对齐 + ibping 通 | 通过 | 维持写死 3（或按 probe 注入） | 0 |

**C2. exit code 语义扩展（落脚本头 + check_vllm_script 白名单）**：
- 新增：`3=GID/RoCE 布局 No-Go（需人工处理）`、`4=NCCL env 一致性失配（需重算 HCA/PEER_HCA）`
- 保留：`0=成功 1=业务失败 2=用法错误(可重试) 130=被signal`
- **铁律**：exit 3/4 必须在 docker run 之前返回（连接建立前）。

**C3. probe_gid_index.sh 建议值逻辑（可落为独立脚本）**：
- 枚举各口 index，输出建议值 = 首个「type==RoCEv2/IPv4 且非 HOLE」的 index；
- 若四台该 index 子网前缀不一致 → 输出 `REMOVE / -1`（动态）；
- 否则输出该 index；cluster.sh 据此决定注入或移除 `-e NCCL_IB_GID_INDEX`。

---

### D. 环网补丁与连接建立

**D1. nccl-ringonly 补丁加载一致性校验（新增）**：
- head line95 / worker line100 `LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2` 顺序固定（libncclpin 先、ringonly 后）。
- **强化**：启动前核验 `/opt/nccl-ringonly/libnccl.so.2` 的 **MD5 == 4cc43e3b**（MEMORY 记录源码 v2.30.7-1 容器构建）且 LD_PRELOAD 顺序未被调换 → 不一致 exit 4（归入 env 一致性码）。
- 四机 MD5 一致性校验（cluster.sh 编排在 step 2 前 ssh 比对）。

**D2. NCCL_IB_HCA per-rank 对端口映射校验**：
- cluster.sh line189-193 `RANK_HCA` per-rank 对端、worker line82-84 `PEER_HCA` per-peer 须与实际环阵链路一致（01↔02、02↔04、04↔03、03↔01）。
- **强化**：预检校验每 rank 的 `NCCL_IB_HCA` 子集 == 该节点环邻实际口；`NCCL_IB_PEER_HCA` 的 peer index 与 rank 映射一致。失配 → preflight exit 4。

---

## 三、运维能力健壮性强化

**E1. 守卫活性探针（闭合卡死漏判）**：
- 现守卫只查 `/health`（head line189 / worker line217），卡死 100s+（NCCL shm hang）仍返回 healthy。
- **新增**：推理活性探针 = 周期 POST `/v1/chat/completions` 最小探针（token 极少）或监控 `docker stats` CPU/请求进度；卡死窗口（探针超时且无请求进度）判 unhealthy。保留 `--health-start-period 900s` 冷启动宽限（line190/218 已配 ✅）。

**E2. 看门狗改速率窗口（弃纯累计阈值）**：
- 现看门狗纯累计 NV_ERR 阈值 → flap 卡死期 NV_ERR=0 必失效。
- **修改**：改为**「加载期/运行期分段 + 单位时间新增速率」窗口**，对 `ERR_PATTERNS`（cluster line54，已含 ibv_modify_qp/NV_ERR_NO_MEMORY/ncclSystemError/DistStoreError ✅）按 rate-window 判定。

**E3. distributed-timeout 空窗评估**：
- `--distributed-timeout-seconds 300`（head line66 / worker line65）。生产 05:34 clean exit 0 与该 300s 吻合。
- **结论**：300s→**120-180s** 减空窗、防误杀长 prefill。**本方案定为 P1**。需回归验证长 prefill 不误杀后再落。

**E4. 日志留存规范 + 检查点/回滚锚点**：
- 日志：~/vllm-logs（nccl debug）+ /opt/aicad-prod/backup（crash-dump）双路径，均须持盘。
- 检查点/回滚锚点已具：head line48 / worker line47 `docker inspect > rollback_*.json` ✅；每轮改动 `.bak-<tag>` 留档。

---

## 四、落地清单（脚本级动作映射）

| # | 动作 | 脚本 | 插入/修改点 | 优先级 |
|---|------|------|------------|--------|
| 1 | 新增 `preflight_gid()`：空洞/子网/口名/连通三级响应 | head+worker | check_vllm_script 后、docker rm 前（head≈L157-161 / worker≈L185-189） | P0 |
| 2 | RANK_HCA / PEER_HCA / LD_PRELOAD-MD5 env 一致性校验 | cluster(ssh 比对)+head+worker | 并入 preflight_gid | P0 |
| 3 | 新增 exit 3/4 语义 + 头部 EXITCODES 注释 + check 白名单 | cluster/head/worker | 头部注释 L11 | P0 |
| 4 | GID_INDEX=3 → probe 注入或 -1 降级逻辑 | head L102 / worker L107 | REPLACE 写死行为 | P0 |
| 5 | ExecStopPost dump docker logs + crash 指纹三件套 | monitor unit | systemd unit | P0 |
| 6 | READY 后核验 nccl-*.log 已生成 | head/worker | L202/L230 READY 分支 | P0 |
| 7 | 看门狗改速率窗口 | monitor 看门狗 | rate-window 逻辑 | P0 |
| 8 | 守卫活性探针 | 守卫脚本 | 探针注入 | P0 |
| 9 | distributed-timeout 300→120-180 | head L66 / worker L65 | 需回归验证 | P1 |
| 10 | 四机 nccl md5 首启核验 | cluster | step2 前 | P0 |

---

## 五、风险与回滚

**风险**：
1. **GID_INDEX 改 -1 动态化**：入 10.20.0.x(/30) + 10.100.x 双子网下可能选错 index → busbw 倒退。**缓解**：preflight 保留 ibping 连通判据，仅 index3 有效且连线通才维持写死；回归验证 4 环网双子网下动态选择 busbw 不倒退。
2. **fail-fast exit 3 过度阻断**：若 ibping 判据过严误判，可能误停健康启动。**缓解**：ibping 设 P3 软门（降级而非硬停），硬停仅限空洞/子网/口名三类明确判据。
3. **活性探针开销/误杀长 prefill**：探针最小 token + cold-start 900s 宽限已覆盖；长 prefill 由 distributed-timeout P1 评估协同。
4. **LD_PRELOAD MD5 误报**：ringonly 若被升级则 MD5 变 → 误拦。**缓解**：MD5 基线更新走明确变更流程（更新 MEMORY + 白名单）。

**回滚**：
- 每改动保留 `.bak-<tag>`；回滚 = `cp .bak-<tag> 原路径` + check_vllm_script 通过。
- GID_INDEX 若动态化后不稳定 → 一键回退写死 3（前提：preflight 判据全过）。
- distributed-timeout 若误杀长 prefill → 立即回退 300。

---

> 本方案由工程保障团队 AI 协作生成，供人类工程负责人复核后由 SRE 落地。