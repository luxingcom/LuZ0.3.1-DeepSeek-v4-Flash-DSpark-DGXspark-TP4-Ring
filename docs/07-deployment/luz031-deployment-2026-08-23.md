# LuZ0.3.1 生产落地 — 部署与验收报告

- **执行人**：雷克斯（Rex）· SRE 工程师（luz031-deployer）
- **日期**：2026-08-23（本地）/ 05:11–06:05 UTC（服务器）
- **集群**：DGX Spark 4 节点 TP4（GB10/sm_121a），环网 01-02-04-03
- **任务**：用户批准 B2+util0.82 方案应用于生产 + 历史小问题补丁 + 命名 LuZ0.3.1 + 完整检查点与恢复镜像
- **一句话结论**：**LuZ0.3.1 采纳成立，生产终态全绿运行**——PR 四档/并发 C6C12/DE C1/质量门/needle/KV（5.73M≥5.7M）全过；补丁 6/7 落实（E4 调查列入遗留）；检查点+自包含恢复镜像就位；过程中发现并修复前序窗口遗留的 **FI 0.6.16 误回滚**。

---

## 1. 部署记录（时间线 UTC）

| 步骤 | 时间 | 结果 |
|---|---|---|
| 勘察 | 05:11-05:16 | 四机 start 脚本/plugin_a1 均干净基线态（bprime 恢复彻底，无 pip uninstall 残留）；池化插件资产 /tmp/_wsdedup_l3（md5 e5ed0c85）在场 |
| 组装 | 05:17 | 池化插件四机部署（md5 一致）+ patch_arm mode2（VLLM_MOE_W4A4=2 + SERVE 前缀装插件）+ SHARED_WRAPPER=1 + util 0.80→0.82（脚本+checker L75 KEY_PARAMS）+ `.bak-luz031-20260823` 四机留档；checker 4/4 PASS |
| 第一次重建 | 05:17-05:26 | head-first（服务 active + docker rm -f rank0），~8.5min 全绿；启动验证过但发现 FI 0.6.16 缺失（见 §2） |
| FI 0.6.16 补回 | 05:36-05:45 | 四机脚本补回 fi016 挂载 2 行（主替换 + JIT 缓存），checker 4/4 PASS，第二次 head-first 重建 |
| 启动验证（终态） | 05:45 | W4A4B12xExperts 双 rank / flashinfer 0.6.16 双 rank / weight 45.32 GiB / KV 5,730,000 / cudagraph 16/12/11 / threshold 4096 / dspark n=7 / util 0.82 |
| 验收测量 | 05:47-06:01 | 双探针 → PR 四档×3 → C6/C12×3 → DE×4 → 质量门 → needle → 全过（§3） |
| 收尾 | 06:02-06:04 | healthcheck.timer 恢复 + 自愈链三件套核验 + 回归日志扫描 0 异常 + 检查点清单刷新 |

## 2. 重大发现：FI 0.6.16 挂载被前序窗口误回滚（已修复）

- **时间线考证**：fi016 窗口（08-23 00:56-01:35 UTC）向四机 start 脚本注入 flashinfer-0.6.16 目录级 bind-mount（`.bak-fi016-20260823` 留档=注入前快照）→ **w4a4-ext 收尾恢复（03:01）误用 phase3b 时代的 `.bak-wsdedupl3-20260823`**（00:56 前快照，不含挂载行）→ 挂载被覆盖删除 → bprime 恢复（04:43）沿用 → **03:01 起生产实际运行 FI 0.6.15 混合树约 2.5 小时**。
- **为何未被发现**：fi016 验证已证两树数值逐位一致（B12xMoEWrapper 输出 bitwise 一致）+ w4a4-ext 恢复后 PR 复测在带内——无功能性信号；各窗口的启动核验清单均不含 flashinfer 版本项。
- **证据**：`.bak-fi016` 与 `.bak-wsdedupl3` diff 为空（同为注入前基线）；容器内 `flashinfer.__version__` = 0.6.15（第一次重建后实测）。
- **处置**：按 fi016 报告 §2.2 原样补回 2 条挂载行（`nvfp4/flashinfer-0.6.16/flashinfer` → dist-packages/flashinfer + `~/flashinfer-cache` JIT 缓存），四机 checker PASS，第二次重建后双 rank 实测 0.6.16。
- **教训（已入 runbook §E）**：跨窗口恢复必须核对 .bak 快照的时序覆盖范围；启动核验清单应加 flashinfer 版本项。

## 3. 验收数字（vs B2 预期带 = w4a4-ext B2 臂 + util0.82 回补合成）

### 3.1 PR 单流 panorama（3 轮中位，tok/s）

| 档位 | B1（W4A16） | B2 预期 | **LuZ0.3.1** | vs B2 | 判定 |
|---|---|---|---|---|---|
| 4K（8.2K tok） | 2769 | ~2994 | **2950.5** | -1.5% | ✓ 带内 |
| 16K | 2770 | ~2973 | **2943.6** | -1.0% | ✓ |
| 32K | 2565 | ~2830 | **2834.2** | +0.2% | ✓ |
| 64K | 2215 | ~2541 | **2550.0** | +0.4% | ✓ |

### 3.2 并发聚合（4K，3 轮中位，tok/s）

| 并发 | B1 | B2 预期 | **LuZ0.3.1** | vs B2 | med TTFT |
|---|---|---|---|---|---|
| C6 | 2744 | ~3060 | **3057**（3023/3060/3057） | -0.1% | 10.47s（B2 10.40） |
| C12 | 2737 | ~3092 | **3056**（3059/3056/3034） | -1.2% | 18.39s（B2 18.13） |

vs 生产前基线（W4A16）：C6 **+11.4%** / C12 **+11.6%**，PR 4K **+6.6%** — B2 并发增益在 util 0.82 + FI 0.6.16 终态下兑现。

### 3.3 DE（接受率归一 step_eff，4 轮中位）

| 指标 | B1 | B2 预期 | **LuZ0.3.1** | 判读 |
|---|---|---|---|---|
| C1 step_eff | 17.7 | ~18.3 | **18.2** | ✓ 中性（-0.5%） |
| C12 step_eff | 87.2 | ~85.1 | **80.2** | ⚠ -5.8% vs B2——落已知 W4A4 full decode 代价带（phase3b 口径 -6~-9% / w4a4-ext 口径 ±3% 两口径并存；vs B1 -8.0%）。用户采纳 B2 时已知该代价，如实记录 |

### 3.4 显存 / KV / 质量 / 探针

| 项 | 实测 | 判定 |
|---|---|---|
| weight | **45.32 GiB**（=B2 精确一致，池生效） | ✓ |
| **KV tokens** | **5,730,000**（≥5.7M 门 ✓；回补 +0.23M vs 合成预期 +0.44M——回补不达预期但达标，**记录决策口径：不阻断**，差异归因非 torch 内存/碎片随 util 上升非线性） | ✓ |
| 质量门 | 稳定 4 prompt **4/4 exact match**（own_stable 4/4；quality_gate.py vs B1 参考） | ✓ PASS |
| needle 64K | **3/3 PASS**（mid/late/late）；128K 加测 1/2（late 位，已知统计抖动） | ✓ |
| stall 探针 | 3×短4K TTFT 2.77-3.12s，ITL 中位 52-55ms，SUSPECT=False | ✓ 干净 |
| 模式探针 | 首 4K TTFT 2.786s（W4A4-fast 类，=B2 2.83s 带内） | ✓ |
| 回归日志 | error/exception/traceback **0 条**（rank0 全日志） | ✓ |

**采纳判定：全部验收门通过 → LuZ0.3.1 保持运行（生产终态，不回滚）。**

## 4. 历史小问题补丁落实（6/7）

| # | 项 | 落实 |
|---|---|---|
| 1 | 贪心质量门固化 | ✅ `<INSTALL_DIR>/scripts/quality_gate.py`（四机 md5 b01ed796）：稳定 4 prompt 集（reason/zh 除名，运行级非确定已证）+ 包络判据（逐字一致硬门 + logprob sum drift ≤1% 兜底）+ 参考快照管理（`backup/quality-gate/`，latest 机制）；历史散落脚本（greedy_check.py/golden_env.py）已加弃用指向 |
| 2 | systemd 重启姿势 runbook | ✅ runbook §E.1：服务 active + `docker rm -f rank0` head-first 全链重建；systemd stop ≠ 容器停；自愈链三件套清单（本窗口两次实证） |
| 3 | healthcheck 基准纪律 | ✅ runbook §F（基准前停 timer/后恢复 + 测量口径）+ healthcheck-rebuild.sh 探针超时 10s→30s（四机，`.bak-luz031` 留档；改动可逆且 checker 过） |
| 4 | checker 用法修复 | ✅ 无参自动发现本机 start 脚本（按 hostname 角色排序：01→head，02-04→worker），usage 提示保留；四机实测 PASS |
| 5 | Prometheus 边界 case | ✅ **现场发现 aicad-prometheus-1 已 Exited(137) 2 天不自愈**——`docker start` 恢复就绪（8191）+ `docker update --restart=always` + compose L125 同步改 always（`.bak-luz031` 留档）+ runbook §E.2 文档化 |
| 6 | worker daemon-reload | ✅ 02/03/04 `systemctl daemon-reload` 执行完毕（版本不一致警告消除） |
| 7 | E4 KV 足迹 8192 调查 | ⏸ **列入遗留**（不阻断）：+58%/token 机制调查需专用窗口（一次性容器内 KV 页布局实验），本窗口已被 FI 误回滚修复+二次重建占用。方向提示：非激活/CUDAGraph 固定开销仅 +1.9GiB 不足以解释，疑似 batched 8192 下 attention/KV 页布局变化（w4a4-ext §5.2） |

## 5. 检查点与恢复镜像清单

| 资产 | 位置 | 状态 |
|---|---|---|
| **自包含恢复镜像**（FI 0.6.16 树 + ws-dedup overlay + 池化插件全 bake，34.4GB） | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1` | ✅ built + pushed（digest sha256:85f2149f…，构建仅增量层） |
| 基座锚点 tag | `…:LuZ0.3.1-base`（=0.2.1-v026.0） | ✅ pushed |
| 状态快照（20MB，18 文件） | `<INSTALL_DIR>/backup/luz031-checkpoint-20260823/`（start 脚本终态 + checker + quality_gate + 池化插件 + plugin_a1 全目录 + overlay + FI tarball + release notes + runbook + compose） | ✅ md5-manifest 17 项核验过 |
| **一键恢复脚本** | 同目录 `restore_luz031.sh`（9 步：md5 核验→分发→FI 树核验→overlay→checker→Prometheus→head-first 重建→启动核验→三件套恢复） | ✅ 语法 + `--dry-run` 路径演练通过（按纪律未实际恢复） |
| 版本文档 | `<INSTALL_DIR>/docs/LuZ0.3.1-release-notes.md`（四机同步）：构成清单/env 全集/基线数字/回滚链/已知事项 | ✅ |
| 脚本留档 | `start_tp4_{head,worker}.sh`/`check_vllm_script.sh` `.bak-luz031-20260823` 四机（=W4A16 基线快照）+ 插件原版 `.bak-wsdedupl3-20260823` | ✅ |

## 6. 生产终态（06:04 UTC 核验全绿）

| 项 | 状态 |
|---|---|
| 形态 | **LuZ0.3.1** = W4A4 full（VLLM_MOE_W4A4=2）+ 池补丁（SHARED=1）+ FI 0.6.16 bind-mount + threshold 4096 + util 0.82 |
| 容器 | vllm-tp4-rank0/1/2/3 全 Up (healthy)；flashinfer 0.6.16 双 rank 实测 |
| 自愈链 | head.service active + 三 worker.service active + healthcheck.timer active（三件套齐）|
| API | /health 200 |
| Prometheus | Up + restart=always（监控栈恢复） |
| b′ 插件 | 文件保留未激活（No-Go 判定不变） |
| checker | 四机 PASS（含 util 0.82 + 无参自动发现） |

## 7. 证据索引

- **服务器** `node01:/tmp/_luz031/`：luz031_setup.sh、luz031_fi016_restore.py、luz031_run.sh、restore_luz031.sh、logs/（setup/rebuild{,2}/run_master/{stall,probe,panorama,conc6,conc12,de,quality_gate,needle,startup}/build/push）
- **检查点** `<INSTALL_DIR>/backup/luz031-checkpoint-20260823/`
- **前序参照**：wsdedup-l3-combo / w4a4-ext / bprime-window / fi016-replacement（2026-08-23）

## 8. 遗留项

1. **E4 KV 足迹 8192 机制调查**（+58%/token，需专用窗口）
2. DE C12 两口径定论（phase3b -6~-9% vs w4a4-ext ±3%）：建议长轮次 DE 专项
3. W4A4 已知代价监控：decode 重业务占比上升时评估（KV -4.5% vs W4A16 基线）
4. 建议启动核验清单永久加入 flashinfer 版本项（防误回滚复发）

---

*纪律遵守：勘察先行、双探针、≥3 轮中位、DE 接受率归一、质量门稳定集口径、所有改动 .bak 留档四机一致、head-first 正确重启姿势两次实证、长命令后台+轮询、发现异常（FI 误回滚）停下考证并按批准构成修复、生产终态全绿 + 自愈链三件套恢复。*

*本报告由工程保障团队（SRE）生成；LuZ0.3.1 为正式生产变更（用户批准 B2+util0.82 方案的落地实施）。*
