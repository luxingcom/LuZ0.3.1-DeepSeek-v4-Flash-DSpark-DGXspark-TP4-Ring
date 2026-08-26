# 四机落地记录 — SRE (sre-1) 2026-08-26

> 本记录为 sre-1 对「生产加固脚本+参数修订 8192 四机落地」的实际落地证据与执行状态。
> **只落地不改动运行容器、不触碰在跑服务、不重启**（重启验证由 testing 环节负责）。

---

## 0. 结论摘要（重要）

| 项目 | 状态 |
|------|------|
| 6 个加固脚本 | ✅ **已完成四机落地**，md5 四机 + 本地完全一致 |
| 3 个内嵌启动脚本 (head/worker/cluster) | ✅ **已完成四机落地（主理人裁决修正后）** — 按 6 处修正（head 镜像.187+注释R12、worker 去 PEER_HCA + health-cmd 回 pgrep），md5 四机一致 |
| crash_dump systemd ExecStopPost | ⏸ 需人工（触碰 active systemd 单元，属重启窗口；已批准 testing 窗口代做） |
| watchdog timer | ⏸ 需人工（交付物无 systemd/timer 模板；已批准 testing 窗口若补模板则装，否则待人工） |
| healthcheck 替换 | ⏸ 需人工（需重建容器；已批准 testing 窗口代做 — head curl / worker pgrep） |
| 自愈链 | ✅ 未触碰，保持原样 |

---

## 1. 已落地：6 个加固脚本（四机一致）

### 1a. 目标（每台 server）
`/opt/aicad-prod/scripts/{preflight_roce_gid.sh, probe_gid_index.sh, gid_index_env.sh, crash_dump.sh, healthcheck_hardened.sh, watchdog_hardened.sh}`，`chmod +x`

### 1b. md5 一致性（四机 + 本地源头）

| 文件 | 本地源头 | 01 | 02 | 03 | 04 | 是否一致 |
|------|---------|----|----|----|----|---------|
| preflight_roce_gid.sh | `70fd4bf...69ace` | 同 | 同 | 同 | 同 | ✅ |
| probe_gid_index.sh | `4530e9a...6641d` | 同 | 同 | 同 | 同 | ✅ |
| gid_index_env.sh | `863b3f4...7c886` | 同 | 同 | 同 | 同 | ✅ |
| crash_dump.sh | `13ff68c...97e40` | 同 | 同 | 同 | 同 | ✅ |
| healthcheck_hardened.sh | `a0bedd7...b4bc` | 同 | 同 | 同 | 同 | ✅ |
| watchdog_hardened.sh | `fe8f07f...95686` | 同 | 同 | 同 | 同 | ✅ |

（上表"同" = 与本地源头 md5 逐位一致；bash -n 全部通过）

### 1c. 落地前冲突核查
- 6 个加固脚本在四机均 **不存在**（additive 新增，零覆盖风险）。
- 生产 scripts 目录现有大量 `.bak-*`（check_vllm_script / start_tp4_* / sglang 等），均未触碰。

---

## 2. 启动脚本 STOP → 主理人裁决 → ✅ 修正后已落地

> 初落地时对 head/worker 集成版发现超出「仅预检+8192」的生产语义变更触发 STOP，team-lead 实查确诊并裁决：
> **裁决 1** head 镜像修正为 .187；**裁决 2** worker 回退 PEER_HCA + health-cmd 回 pgrep；**裁决 4** head 头注释 R11→R12。
> 已按裁决修正 6 处后完成四机落地。详见 §2E「修正后落地」。

### 2A. 原 STOP 发现（head 镜像回归，已按裁决修正）
- 生产现网(live)：`IMG="<registry_ip>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0"`，运行中容器就是该镜像，registry 实测 HTTP 200。
- 集成版误写 `.186:5000`（HTTP 000 不可达）→ 触发 STOP。
- **→ 已按裁决修正为 .187，与现网一致。**

### 2B. worker 额外语义变更（已按裁决回退 2 处）
集成 worker 原含：**①新增 `NCCL_IB_PEER_HCA` 注入（生产无）②health-cmd 改 curl**。
- worker 无对外 HTTP 8001，curl 会误判 unhealthy → 自愈误重启。
- **→ 已按裁决：①移除 PEER_HCA 行；②health-cmd 回 `pgrep -f VLLM::EngineCore`（head 保留 curl，因 head 有 8001）。**
- 另有「移除 NCCL MAXCH16 注释块」— 仅注释，无行为影响。

### 2C. cluster — 依赖 2A/2B 定稿，现一并落地
- 集成版新增 STEP0 四机一致性核验（GID/RINGONLY MD5），逻辑 OK，随 head/worker 定稿后落地。

### 2D. 备份 .bak-20260826（现网原版 = 回滚基线）
四机已对启动脚本留 `cp -p` `.bak-20260826`，复核 bak md5 == 现网原版：

| 服务器 | 已备份文件 |
|--------|-----------|
| dgxspark01 | start_tp4_head.sh.bak-20260826, start_tp4_cluster.sh.bak-20260826 |
| dgxspark02 | start_tp4_head.sh.bak-20260826, start_tp4_worker.sh.bak-20260826, start_tp4_cluster.sh.bak-20260826 |
| dgxspark03 | start_tp4_worker.sh.bak-20260826 |
| dgxspark04 | start_tp4_worker.sh.bak-20260826 |

### 2E. ✅ 修正后四机落地 + md5 一致性
按裁决修正后的 3 个启动脚本已 scp 到各机并 `chmod +x`，逐字节 md5 复核：

| 文件 | 修正后源 md5 | 01 | 02 | 03 | 04 |
|------|-------------|----|----|----|----|
| start_tp4_head.sh | `b0733370...99510` | ✅ | ✅ | — | — |
| start_tp4_worker.sh | `ad13cecd...d5ead` | — | ✅ | ✅ | ✅ |
| start_tp4_cluster.sh | `7ee20fba...14c73` | ✅ | ✅ | — | — |

落地后关键参数实核：
- **01 head**: `IMG=<registry_ip>:5000` ✅ · `seqs=12` ✅ · `max-num-batched-tokens 8192` ✅ · health-cmd=`curl`(head 有 8001, 正确) ✅ · R12 头注释 ✅
- **02 worker**: `max-num-batched-tokens 8192` ✅ · health-cmd=`pgrep VLLM::EngineCore`(正确) ✅ · **PEER_HCA 计数=0**(已移除) ✅
- **01 cluster**: STEP0 四机一致性核验在 (4 处命中) ✅

---

## 3. 系统观测接线 — 三项均「需人工/重启窗口」

自愈链实况：
- head: `vllm-tp4-head.service`（active running，ExecStart=monitor_tp4_head.sh）
- worker: `vllm-tp4-worker.service`（active running，ExecStart=monitor_tp4_worker.sh）
- 观测: `vllm-healthcheck.service`+.timer（每 60s 调 healthcheck-rebuild.sh）
- monitor_tp4_head.sh 内 `NO_WAIT=1 bash start_tp4_head.sh` → 说明启动脚本路径已被自愈链引用，覆盖=影响下次重启。

| 观测项 | 决定 | 理由 |
|--------|------|------|
| crash_dump (ExecStopPost) | ⏸ **需人工** | 需改 active 的 vllm-tp4-head/worker.service + daemon-reload；且 ExecStopPost 仅在 systemd 触发重启时生效。改动/生效均属重启窗口，且当前会触碰在跑服务 → 不在本落地窗口做 |
| watchdog timer | ⏸ **需人工** | 交付物仅 watchdog_hardened.sh，**无 systemd/timer 模板**。需先补模板，再 daemon-reload。脚本本身已就位四机，可直接被人工定时调用 |
| healthcheck 替换 | ⏸ **需人工** | 需重建容器（改 docker --health-cmd）或改 active 自愈服务引用 healthcheck_hardened。均在重启窗口 |

---

## 4. 保留自愈链
- 自愈链相关（monitor_*.sh / *.service / *.timer）**全程未触碰**，保持原样。
- 已落地的 6 加固脚本为纯新增，不影响自愈链解析。

---

## 5. 主理人裁决结果（已执行）
team-lead 已实查确诊并裁决，均已在本轮落地中执行：
1. **head 镜像** → 以 **.187** 为准（修正集成 head，与现网一致）✅
2. **worker** → 移除 `NCCL_IB_PEER_HCA` + health-cmd 回 `pgrep VLLM::EngineCore`（head 保留 curl）✅
3. **观测接线** → 批准在 **testing 重启窗口内**由 sre-1 代做 crash_dump ExecStopPost + healthcheck（重建容器时用 head-curl/worker-pgrep）+ watchdog timer（若补出模板则装，否则待人工）
4. **head 头注释** → R11→R12（`seqs=12/util=0.78`）同步 ✅

## 6. 待 testing 重启窗口执行（系统观测接线）
- **crash_dump ExecStopPost**：改 vllm-tp4-head/worker.service 加 `ExecStopPost=.../crash_dump.sh <container>` + daemon-reload（已在重启窗口授权）
- **healthcheck 替换**：重建容器时 health-cmd 用 `head=curl /8001/health`、`worker=pgrep VLLM::EngineCore`（已内嵌于落地脚本，重建即生效）
- **watchdog timer**：补 systemd 模板则装；否则记录待人工
> 当前落地窗口仅落文件未重启，上述改动随 testing 重启窗口由 sre-1 执行。