# 生产加固实现脚本 · 集成说明 (INTEGRATION.md)

**工作流**: 工程保障 · 生产加固 (部署能力 + 运维健壮性)
**日期**: 2026-08-26
**依据**: `incident-clone-roce-prevention-2026-08-25.md` §分层预防 (P0-配置 / P0-观测)
**交付目录**: `deliverables/engineering-assurance/prod-hardening-2026-08-26/`

> 本目录脚本均为**可直接 scp 到生产/克隆机**的独立 bash 脚本，不连服务器执行。变更规范对齐现有：
> 改脚本须 `bash -n` + `check_vllm_script.sh` 通过 + `.bak-<tag>` 留档 + 更新 REFERENCE.md。

---

## 0. 交付物清单 (7 文件)

| 文件 | 一句话用途 |
|------|-----------|
| `preflight_roce_gid.sh` | 连接建立前 RoCE GID 布局预检 (空洞/IPv4 RoCEv2/index3 子网一致性), 布局异常→`exit 3` No-Go 阻断启动。 |
| `probe_gid_index.sh` | 实测当前 RoCEv2/IPv4 GID 实际 index, 输出建议 `NCCL_IB_GID_INDEX` (数值 / -1 动态 / REMOVE)。 |
| `crash_dump.sh` | 崩溃指纹留存 (systemd ExecStopPost 调用): dump docker logs + dmesg 采集到 `crash-<ts>/` 持久目录。 |
| `healthcheck_hardened.sh` | 加固守卫: 加推理活性探针 (小请求带超时判定卡死) 闭合「卡死 100s+ 仍判 healthy」盲区 + 保留冷启动宽限。 |
| `watchdog_hardened.sh` | 加固看门狗: 加载期/运行期分段 + 单位时间 NV_ERR 新增速率窗口, 弃纯累计阈值, 补 carrier flap 计数探针。 |
| `gid_index_env.sh` | start_tp4_*.sh 可 source 的 env 决策片段: 实测后注入 `NCCL_IB_GID_INDEX` (动态/-1 或实际值), 含降级栅栏。 |
| `INTEGRATION.md` | 本文件: 每个脚本插入现有脚本的位置、.bak 留档 tag、check_vllm_script.sh 验证。 |

---

## 1. 各脚本插入位置 (对齐现有脚本行号)

### 1.1 `start_tp4_head.sh` (v1.5-r12)

| 位置 | 插入内容 | 说明 |
|------|---------|------|
| `check_vllm_script.sh "$0"` 通过**之后**、`docker rm -f` 之前 (当前 line 157-161) | `preflight_roce_gid.sh --degrade` + 若返回 3 → `exit 1` | **P0 硬门**: 空洞/布局不一致 → 启动前阻断, 绝不创建容器后才诊断 (RCA 最大教训)。 |
| 同一处 (可与 preflight 并列) | `probe_gid_index.sh` 采集当前 index 供人工核对 (可选打印) | 记录"实测 index"到操作日志。 |
| **ENV_ARGS 组装之前** (当前 line 81 前) | `source /opt/aicad-prod/scripts/gid_index_env.sh` | 注入 `NCCL_IB_GID_INDEX` 决策 (动态/-1 或实际值)。 |
| ENV_ARGS 中替换 `-e 'NCCL_IB_GID_INDEX=3'` (当前 line 102) | 改为 `-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"` | **严禁再写死 3** — 用 source 片段注入实测值。 |
| systemd unit 的 `ExecStopPost` | `ExecStopPost=/opt/aicad-prod/scripts/crash_dump.sh vllm-tp4-rank0` | 崩溃指纹留存 (闭合日志缺口)。 |

### 1.2 `start_tp4_worker.sh` (v1.5-r12)

| 位置 | 插入内容 | 说明 |
|------|---------|------|
| D1 模型就绪门禁 (line 157-183) **之后**、`check_vllm_script.sh` 之前 (line 185) | `preflight_roce_gid.sh --degrade` + 若非 0 → `exit 1` | worker 同样跑 GID 硬门 (四台须交叉核对, 见 `--peers`) |
| ENV_ARGS 组装之前 (当前 line 86 前) | `source .../gid_index_env.sh` | 注入 GID_INDEX 决策。 |
| ENV_ARGS 中替换 `-e 'NCCL_IB_GID_INDEX=3'` (当前 line 107) | 改为 `-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"` | 同上, 禁止写死。 |
| systemd unit `ExecStopPost` | `ExecStopPost=.../crash_dump.sh vllm-tp4-rank${NODE_RANK}` | 崩溃指纹留存。 |

> **四台交叉核对 preflight**: 各机独立跑 preflight 后, 汇总四台 index 用
> `preflight_roce_gid.sh --peers 'rank0:3' 'rank1:3' 'rank2:3' 'rank3:3'`
> 任一 index 不一致 → 判不一致, 输出 -1 动态。此步骤在 head 侧协调机统一执行一次。

### 1.3 `check_vllm_script.sh` 前置自检

`gid_index_env.sh` 引入了对 `probe_gid_index.sh` / `preflight_roce_gid.sh` 的依赖:
- 两者路径可通过 env 覆写 (`TP4_PROBE_BIN` / `TP4_PREFLIGHT_BIN`), 默认 `/opt/aicad-prod/scripts/probe_gid_index.sh`。
- 若部署时因依赖不存在而降级为 `NCCL_IB_GID_INDEX=-1` (动态), `check_vllm_script.sh` 的 B 组关键参数
  若检查 `NCCL_IB_GID_INDEX=3` 字面量需同步放宽 (改检 `NCCL_IB_GID_INDEX=(-1|[0-9]+)` 或移除对该键的硬性检查)。

### 1.4 守卫/看门狗 (独立, 不插启动)

| 用途 | 替换/新增 | 说明 |
|------|----------|------|
| systemd/手动健康探针 | `healthcheck.sh` → `healthcheck_hardened.sh` | 保留冷启动宽限；head 加推理活性探针。 |
| 常驻看门狗 | 新增 `watchdog_hardened.sh` (timer/cron 60s 或 while 循环) | 分段 + 速率窗口, 补 carrier flap 探针。 |

---

## 2. `.bak-<tag>` 留档建议

按现有规范 (`.bak-<tag>` 留档), 建议 tag:

| Tag | 时点 |
|-----|------|
| `.bak-gid-dyn-20260826` | 生产 head/worker 替换 `NCCL_IB_GID_INDEX=3` → source 片段注入 (**生产首改建议先留此档, 可秒回滚**) |
| `.bak-preflight-20260826` | 插入 preflight/probe 硬门后 |
| `.bak-crashdump-20260826` | 加入 systemd `ExecStopPost` 后 |
| `.bak-health-20260826` | 换用 healthcheck_hardened 后 |

回滚顺序 (若异常): 还原 ENV_ARGS 为 `-e 'NCCL_IB_GID_INDEX=3'` → 移除 preflight 门 → 还原 healthcheck。
(注: 回滚到写死 3 仅限**生产四机 index3 已实测一致且无洞**情形; 克隆环境必须走动态, 见铁律。)

---

## 3. 用 check_vllm_script.sh 验证

```bash
# 1) 每个新脚本过 shellcheck + bash -n
shellcheck preflight_roce_gid.sh probe_gid_index.sh crash_dump.sh \
          healthcheck_hardened.sh watchdog_hardened.sh gid_index_env.sh
bash -n <script>                       # 全部已通过 (见交付时测试)

# 2) 对改动后的 start_tp4_head.sh / worker.sh 跑既有自检
/opt/aicad-prod/scripts/check_vllm_script.sh start_tp4_head.sh
/opt/aicad-prod/scripts/check_vllm_script.sh start_tp4_worker.sh
# 预期在满足 §1.3 的放宽前提下返回 0

# 3) 功能 smoke (无真实 IB 主机也能测的路径)
bash preflight_roce_gid.sh --help      # 用法 OK
bash probe_gid_index.sh --help
bash -c 'source gid_index_env.sh'      # 无 probe 时降级 -1, GID_INDEX_DECIDED=degraded
```

---

## 4. 分层预防对应关系 (脚本 ↔ 方案)

| §分层预防 行动 | 落到脚本 |
|--------------|---------|
| P0-观测 #1 #2 日志落持久卷 + ExecStopPost dump | `crash_dump.sh` (需配合 `-v ~/vllm-logs:/var/log/vllm` 已存在) |
| P0-观测 #3 守卫活性探针 + 冷启动宽限 | `healthcheck_hardened.sh` |
| P0-观测 #4 看门狗分段+速率窗口+carrier 探针 | `watchdog_hardened.sh` |
| 检查点 1 GID dump / index3 判据 / 一致性 | `preflight_roce_gid.sh` |
| 检查点 2 probe_gid_index 判定注入 | `probe_gid_index.sh` + `gid_index_env.sh` |
| 检查点 3 启动前 preflight fail-fast | `preflight_roce_gid.sh` (插入 start 脚本, exit 3 阻断) |

---

> 本交付为脚本级实现, 未连生产/克隆服务器实跑。上生产前由人类工程负责人按 Go/No-Go 清单逐条在目标机复核输出。