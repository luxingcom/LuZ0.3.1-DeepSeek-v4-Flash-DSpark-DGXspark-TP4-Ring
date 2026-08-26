# 克隆环境「多次会话后崩溃」事故：bug 构成 + 分层预防性修复

**日期**：2026-08-25
**工作流**：工作流 3（事故响应 / 预防性根因治理）
**参与成员**：Archi(架构师·bug构成+预防设计) / Rex(SRE·落地命令+自检清单)
**环境边界**：本次事故发生于【克隆环境】（旁路/克隆镜像部署，复用生产镜像 LuZ0.3.1 与部署脚本），**非生产环境**

---

## 📌 TL;DR（执行摘要）

- **整体结论**：克隆环境「多次会话后崩溃」= **上游 `carrier flap`（物理层，克隆放大）→ L1 运行时 QP 断（worker 静默死）→ L2 重启卡死（写死 `NCCL_IB_GID_INDEX=3` 在克隆 GID 表差异下踩空 → shm_broadcast hang，**connect.cc:317-321 源码实锤**）→ 系统观测缺口（日志不落卷/守卫盲区/看门狗阈值失效）** 多层叠加的 NCCL 故障。
- 严重度分布：🔴 确定 3 项 / 🟡 推断 2 项 / 🟢 待证(低优先) 1-2 项；阻塞 0 项当前活动，但克隆**启动前的 GID 预检为硬性 No-Go 门**。
- **最危险认知**：不是「缺配置」而是「**生产写死 index=3 恰好可用，克隆盲目照抄 index=3 在克隆上洞了**」。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过——按 Go/No-Go 清单预检后放行 |
| 阻塞项数量 | 0 项当前活动（观测缺口闭合前禁止无预检启动） |
| 关键行动项 | 6 项（P0×3 硬门 / P0×3 观测与基线） |
| 建议下一步 | 克隆每台跑 GID fail-fast 门 → probe_gid_index 判定 → 日志落持久卷 → 再启动 |

---

## 🔍 事故裁决正文

### 〇、直答结论（一句话）

> 克隆环境「多次会话后崩溃」= 上游 `carrier flap`（物理层克隆放大）+ L1 运行时 `QP 断`（worker 静默死）+ **L2 重启卡死（写死 `NCCL_IB_GID_INDEX=3` 在克隆 GID 表差异下踩空 → shm_broadcast hang，connect.cc:317-321 源码证实）** + 观测缺口（日志不落卷 / 守卫盲区 / 看门狗阈值失效）多层叠加；**预防 = 治物理 flap 根因 + 移写死 GID_INDEX 改动态选择/实测注入 + 持久日志与活性探针闭环观测**。

---

### 一、多层 bug 构成清单（克隆环境独有放大 + 确信度 + 证据）

| # | 层 | Bug | 克隆放大 | 确信度 | 证据依据 |
|---|----|-----|---------|--------|---------|
| T0 | 🔴 触发源 | `carrier flap`（物理层） | 高 | 🟡 | 克隆复用生产镜像/脚本但走非生产口/SFP/线缆 → flap 事件密度高于生产稳态；生产 RoCE A=136/137(GID2)、B=138/139(GID4) |
| L1 | 🟡 运行时 | QP 断 → `IBV_WC_RETRY_EXC_ERR` → worker 静默死 | 中 | 🟢 | flap 活跃时 QP 重建，`NCCL_IB_TIMEOUT=1000/RETRY_CNT=7`(head:104-105) 试图吸收但克隆 flap 更密仍可击穿；worker 静默死 / head /health 200 / clean exit——与 05:34 观测链同构 |
| L2 | 🟠 重启卡死 | GID 空洞 → 写死 index → `shm_broadcast` hang | **高** | 🔴（源码级） | **connect.cc:317-321 命中 `NCCL_IB_GID_INDEX` 后直接 return，跳过 328-332 动态选择**；head.sh:102 / worker.sh:107 均 `-e NCCL_IB_GID_INDEX=3` |
| — | ⚪ 使能背景 | memdesc `NV_ERR` | 无/低 | 🔴（排除） | 已裁决「两次皆使能条件非触发」（burst 仅权重加载期，卡死窗口 NV_ERR=0）；克隆复用同镜像/加载路径，非克隆特有放大项 |
| — | 🔴 观测缺口 | 日志缺失 / 守卫盲区 / 看门狗阈值失效 | **高** | 🔴 | (a) `NCCL_DEBUG_FILE` 指容器本地 `/var/log/vllm` 未挂持久卷，四机实测无文件；(b) 守卫只查 /health，卡死 100s+ 仍判 healthy；(c) 看门狗纯累计 NV_ERR 阈值，而 flap 卡死期 NV_ERR=0 → 必失效 |

**⚠️ 形态判读**：克隆环境是「L1/L2 两线 NCCL 故障被放大 + 观测缺口被放大」，memdesc 线不放大。**严禁与生产 05:34 的宿主内存主因合并**——那是生产特定（util0.82/swap/MemLimit=0），克隆主打 NCCL 线。

---

### 二、分层预防性修复设计

#### P0-治本｜carrier flap 物理根因定位
1. **四机同频 vs 单台判别**：
   - 同频（4 台同时 flap）→ 查公共电源 / 时钟同步 / fw 一致性（生产 fw 28.45.4028，克隆须先对齐 fw 版本）
   - 单台 flap → 定位该机光模块/线缆/SFP（`ethtool -S` 看 link_down_events_phy / CRC / tx-rx errors，按 08-11 基线口径）
2. `ethtool -S` 时序取证 + mlnx 已知修复核查（fw 已知 bug 是否在 28.45.4028 修正列表）
3. flap 未根治前提下的 L1 缓解：
   - 适度抬 `NCCL_IB_RETRY_CNT`（现 7→12-15 区间 A/B，吸收 flap 瞬时重试，但注意同增 hang 时间，需权衡 RETRY 语义）
   - RoCE carrier 中断绑定隔离核（irq affinity → isolcpus 8-9，避免中断打到 vLLM/NCCL 关键线程；配合 shim mark-then-pin）
   - 若克隆口布局与生产 HCA 列表不一致，重设 `NCCL_IB_HCA`（head:103 写死 4 twin 口）到克隆实际口集合

#### P0-配置｜GID 写死（克隆最高优先）
1. **首选**：移除 `-e NCCL_IB_GID_INDEX=3` 或显式设 `-1`，让 connect.cc:328-332 动态选择（官方 2.21+ 本就不该设）。**回归验证点**：4 环网 2 子网（10.20.0.x /30 + 10.100.x）下是否稳定选中正确 index、busbw 不倒退。
2. **次选**：若必须写死，则按**克隆实测**注入实际 index（非复用生产 3），并加**启动前 GID 空洞预检 fail-fast**。
3. **⚠️ 铁律**：**克隆严禁盲目复用生产 `NCCL_IB_GID_INDEX=3`**——这是克隆「重启卡死」最大诱因，必须「实测当前 GID 表 → 选型」而非照抄。

#### P0-观测｜闭合 RCA 缺口
1. `NCCL_DEBUG_FILE` 改挂**持久卷**（mount 独立 volume，非容器本地 /var/log/vllm；容器销毁日志仍在）
2. 容器崩溃 **ExecStopPost dump `docker logs` 到持久目录**（生产四机历史日志已删=RCA 最大缺口，克隆必须自带崩溃指纹留存）
3. 守卫加**推理活性探针**（NCCL comm 心跳/请求进度，闭合「卡死 100s+ 仍判 healthy」盲区；配合 health-start-period 宽限 900s）。注：生产曾「明确暂不设」——若克隆仍暂缓，至少并行部署 1/2 项
4. 看门狗阈值**改「加载期/运行期分段 + 单位时间新增速率」窗口**，弃纯累计 NV_ERR 阈值（flap 卡死期 NV_ERR=0，纯累计必失效）

#### P1｜`--distributed-timeout-seconds 300` 空窗优化评估
300s→120-180s 减空窗、防误杀长 prefill（生产 05:34 clean exit 0 与该 300s 吻合）。P1 非 P0，因克隆主打 NCCL 线，内存线仅冗余兜底。

---

### 三、克隆环境安全部署检查清单（Go/No-Go，可执行）

#### 检查点 1 · GID / RoCE 布局核查（硬门）
**命令 1a — dump 四台 GID 表**：
```bash
for p in /sys/class/infiniband/*/ports/*/gids/[0-9]*; do
  idx=${p##*/}; dev=${p%/*}
  gid=$(cat "$p" 2>/dev/null)
  [ "$gid" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && st=HOLE || st=ok
  printf "%-24s idx=%2s %-45s %s\n" "$dev" "$idx" "$gid" "$st"
done | sort
```
（`dev` 应展开为 `rocep1s0f0/rocep1s0f1/roceP2p1s0f0/roceP2p1s0f1`，**若非此名 → 直接 NO-GO**，口名不同则 HCA/PEER_HCA 全要重算）

**命令 1b — index3 有效性 + IPv4 RoCEv2 子网判定**：枚举 `gid_attrs/types/` 判读 3 号位类型；任一 `gids/N` 全 0 即 HOLE。

**命令 1c — 可执行判据**（Jack）：
1) 空洞判据：index3 为纯 0 → HOLE → No-Go
2) 子网判据：index3 GID 的 subnet prefix 须 == 克隆实际 RoCE 网段 L2 前缀
3) 一致性判据：四台 index3 子网前缀必须一致（不一致 → 写死 index 会跨网段 → No-Go）
4) 连通判据：`ibping` 验证 index3 GID 确实可达对端
→ 任一不满足 → **No-Go**，必须改动态选择，禁止带 index3 启动。

#### 检查点 2 · NCCL_IB_GID_INDEX 适配
- **默认移除/设 -1**（走官方动态选择）。仅当 1b/1c 判据证明克隆四台 index 布局与生产逐位一致，才允许按节点注入。
- 检测脚本 `probe_gid_index.sh`：枚举各口 index，输出建议值 = 首个「type 为 RoCEv2/IPv4、非 HOLE」的 index；若四台该 index 子网前缀不一致 → 输出 `REMOVE / -1`（动态），否则输出该 index。

#### 检查点 3 · 启动前预检 fail-fast（硬门，防卡 shm_broadcast）
`preflight_gid.sh`（插入 start_tp4_head/worker.sh 的 check 之后、docker run 之前）：任一 GID 空洞 → `exit 3` 终止，绝不在创建容器后才诊断。**克隆必须在此阻断，而不是等容器起来再卡死诊断（RCA 最大教训）。**

#### 检查点 4 · 日志与观测
```bash
-e 'NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log'
-e 'NCCL_DEBUG=INFO'
-v ~/vllm-logs:/var/log/vllm        # 确认宿主 ~/vllm-logs 在持久盘
--log-opt max-size=100m --log-opt max-file=3
# systemd 崩溃 dump：
ExecStopPost=/bin/sh -c 'docker logs vllm-tp4-rank${rank} > /opt/aicad-prod/backup/crash-dump-$(date +%s).log 2>&1 || true'
```

#### 检查点 5 · 物理层 carrier flap
**采集**：`ethtool -S <口>` 抓 `link_down_events/link_down_events_phy/rx_errors/tx_errors/*crc*err*`；多次采样（30s×N）对比计数增量定 flip 频率。
**判据**：
- 4 台同源同频增长（时间戳对齐）→ 公共电源/时钟/固件（排查 PDN/时钟同步/固件）
- 仅单台/单口增长（CRC 同时非 0）→ 光模块/线缆/SFP，隔离直连替换
- CRC 全 0 且计数稳定 → carrier 健康，物理层 P1 解除（对齐生产 08-11 iperf3 四段重传≈0 判据）

---

### 四、机制关系图（文字版）

```
carrier flap (T0, 克隆放大——非生产口/SFP/线缆)  [🟡 物理根因, 治本点]
   │
   ├─[运行期]─→ NCCL QP 底层瞬断 → 重试超限 → IBV_WC_RETRY_EXC_ERR → worker 静默死 (L1) [🟢]
   │             (NCCL_IB_RETRY_CNT=7 可调吸收, irq affinity→隔离核缓解)
   │
   └─[重启期]─→ 内核重建 GID 表 → index 漂移/空洞(克隆布局 ≠ 生产)
                 → 生产写死 NCCL_IB_GID_INDEX=3 在克隆踩空(connect.cc:317-321 盲信)
                 → ibv_modify_qp 22/61 → 卡 shm_broadcast hang (L2) [🔴 源码实锤, 克隆放最大]
   │
memdesc NV_ERR ────────────────────────────→ 毫无放大/非触发, 排除 [🔴]
观测缺口(日志未落卷/守卫盲区/看门狗累计阈值失效) ─→ RCA 缺口放大 [🔴]
```

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 克隆每台跑 GID fail-fast 门（检查点3 preflight_gid.sh，空洞/diff → No-Go 停启） | Rex (SRE) | **P0 硬门** | 启动前强制 |
| 2 | probe_gid_index.sh 判定 NCCL_IB_GID_INDEX：默认移除/-1 动态；仅四台 index3 一致才按节点注入 | Rex (SRE) | **P0 硬门** | 启动前强制 |
| 3 | 口名/HCA 全量复核（检查点1a dump：非 rocep1s0f0/roceP2p1s0f0/1 即 NO-GO，HCA/PEER_HCA 重算） | Rex (SRE) | **P0 硬门** | 首次部署 |
| 4 | NCCL_DEBUG_FILE→持久卷 + ExecStopPost dump docker logs（杜绝日志缺口） | Rex (SRE) | P0 | 部署时 |
| 5 | 守卫活性探针 + 冷启动宽限 900s（防卡死漏判/误杀；如生产暂缓则作观察项） | Rex (SRE) | P0 | 评估后 |
| 6 | carrier flap 物理采集与同频/单台判据基线（ethtool 时序；治本路径） | Rex + Archi | P0 | 持续 |

---

## ⚠️ 已知局限

- **未连服务器**：本清单为脚本级可执行设计，克隆环境实跑逐条回贴命令输出即可；`probe_gid_index.sh` / `preflight_gid.sh` 可再细化成可直接 scp 的 .sh。
- GID 空洞机理（L2）为**源码级实锤**，但克隆现场「是否已踩到洞」需实跑检查点 1 confirm。
- carrier flap 物理根因（T0）为🟡推断（克隆放大），未获得克隆现场 ethtool 数据。
- 守卫活性探针生产侧曾「暂不设」——若克隆沿此决定，活性盲区保持开放，需如实记录。

---

## 📚 数据来源 & 成员产出索引

- **Archi（架构师）原始产出**：克隆 bug 分层（T0/L1/L2/memdesc/观测）+ 预防设计 + 源码实锤（connect.cc:317-321 写死跳动态；head.sh:102/worker.sh:107 写死；NCCL_DEBUG_FILE 未落卷四机无文件）+ 铁律。
- **Rex（SRE 工程师）原始产出**：Go/No-Go 检查单（GID dump / index3 判据 / probe_gid_index.sh / preflight_gid.sh / 日志持久卷 / ExecStopPost dump / irq affinity / 看门狗改造 / ethtool flap 采集与同频判据）+ 落地优先级 P0-1~6。
- 历史引用：`.workbuddy/memory/2026-08-25.md`、`MEMORY.md`（崩溃分层裁决、克隆环境边界）、生产脚本 `start_tp4_head/worker.sh`。

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。