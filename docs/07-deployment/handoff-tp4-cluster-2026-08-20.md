# DGX Spark TP4 集群 工程审查交接文档

> **交接人/单**：工程保障团队（engineering-audit-handoff）主理人
> **交接日期**：2026-08-20
> **整体评级**：🟡 **有条件通过**（CONDITIONAL PASS）
> **适用对象**：接手 DGX Spark TP4 集群 + NVFP4 4W4A 落地的新团队（拿到即可接管执行）

---

## 0. 交接摘要（一句话）

> **DGX Spark 4 节点（01-04 / sm_121a / vLLM 0.26）TP4 集群生产功能已实测稳定，NVFP4 路线 A 主路径已打通（8/8 正确性 rel=0.00141、60~187 TFLOPS、零构建）；四位成员（架构/代码/运维/测试）联合审查给 🟡 有条件通过，剩余 P0/P1/P2 待办为「工程化收尾」而非「方案不确定」——新团队按本文十章 + 待办清单即可接管执行。**

- **🟡 有条件项的集中点**：可复现性与密钥卫生（科迪 S1/S2/S3）、生产恢复与自愈未就绪（雷克斯 R1/R2/R3）、验收测试框架未自动化（泰莎 G1/G2/G3）。三者均不阻断已上线运行，但**阻挡「声明自洽、可长期维护」的验收标准**。
- **接手后第一动作**：🔴 修密钥泄漏（S1）→ 🔴 消除 shim 源码/二进制漂移（S2）→ 统一网络寻址权威表（S3）→ P0 持久化落 `<INSTALL_DIR>` → 恢复自愈（R2）。详见 §5。

---

## 1. 环境事实（接手必读，勿再推翻）

> ⚠️ **铁律：全部以运行态实测为准**。历史文档存在多处配置漂移（见科迪 S3、雷克斯 R6），照文档部署可能组网失败。动手前先用 §7 测试矩阵（尤其 Go 段）做运行态核验。

| 项 | 运行态事实 |
|---|---|
| **机型/架构** | DGX Spark 4 节点 / GB10 / sm_121a / Triton 3.6.0 / torch 2.11+cu130 |
| **生产容器镜像** | `dspark-vllm-gx10:0.2.1-v026.0`，vLLM 0.26，flaṡhinfer 0.6.15 |
| **生产容器** | `vllm-tp4-rank0`（01 head）+ rank1~3（02/03/04 worker） |
| **节点/角色/IP** | 01=head `<NODE_IP>`；02 worker `<NODE_IP>`；03 worker `<NODE_IP>`；04 worker `<NODE_IP>` |
| **网络** | 环网 4 段：`10.20.0.x`（控制面） + RoCE 数据面（`10.100.x.x`/`10.20.0.x`/`<RING_SUBNET>`）；MTU 9000；iperf3 99~110G、重传≈0 |
| **NCCL** | RING-only 补丁 `nccl-ringonly-v2.30.7-patch.diff`；`libnccl.so.2.30.7` MD5=`b7784b49`（v3 双口，**旧文档 `4cc43e3b` 已作废**） |
| **shim** | `libncclpin.so` v8 MD5=`ce43c688`（布局 `NCCL→8-9 / EngineCore→15-19`；隔离核 `isolcpus=8-9`） |
| **SSH** | `ssh -o BatchMode=yes node01 "docker exec ..."`；root 免密 sudo（NOPASSWD） |
| **⛔ 持久化边界** | `/vllm-workspace/` 是**容器内部目录，未挂载宿主机 → 容器重建即丢**；宿主机持久目录 `<INSTALL_DIR>/`（`lib/`、`models/`、`envs/`、`scripts/` 已挂载进容器） |
| **生产状态** | 4 rank 全程 healthy、GPU 0%、**未恢复**（按用户要求）；当前不承载推理 |
| **监控** | Grafana 数据源 01→`http://<NODE_IP>:8191`（02 Prom）；scrape 5s；面板按 node 分组 |

### 1.1 科迪 S3 网段口径问题（接手必清）
- 控制面（TCPStore 25999 + vLLM 分布式）走**管理网镜像** `<NODE_IP>`（`start_tp4_cluster.sh:31` `MASTER_ADDR`）；RoCE 数据面走环网 `10.100.x / 10.20.0.x`。
- 工作区现存**三套网段描述**（`192.168.5.x` / `10.100.x.x` / `10.20.0.x`）并存未统一 → 运维极易拿错 IP。
- **处置**：建单一权威 `hosts`/netplan 映射表，固化「管理网=控制面 / 环网=数据面」，更新 `97-roce-mtu.yaml` 覆盖 10.20.0.x（当前仅 10.100.x，缺少 MTU 的 10.20.0.x 需核对）。

### 1.2 NCCL 锚点（勿再改）
- `libnccl.so.2.30.7` MD5=`b7784b49` 四机一致 = 生产部署值。补丁可复现、可回滚，是本轮最大亮点。
- `libncclpin.so` v8 MD5=`ce43c688` 四机一致，与 `shim-deploy.sh` 期望值闭环（备 .bak-pre-deploy/.bak-v7 锚点）。

---

## 2. 决策基线（5 条 ADR — 维护中勿改关键参数）

| ADR | 决策 | 状态 | 关键参数（勿改） |
|---|---|---|---|
| **ADR-1** | **kernel① 主路径 = 路线 A**（vLLM 内置 `cutlass_scaled_fp4_mm`，原生 FP4，零构建） | ✅ Accepted | 主=路线A；备=路线B（FlashInfer 待升级）；回退=v15（bf16）；性能≥200 TFLOPS 目标 |
| **ADR-2** | **kernel② KV-Linear v17 替换 v11** | ✅ Accepted | 生产=v17；回退=v11；**paged 维持 v11（R3 待做）** |
| **ADR-3** | **生产持久化单一权威源 = `<INSTALL_DIR>/`** | ✅ Accepted（P0 待落实） | routeA → `<INSTALL_DIR>/scripts/nvfp4/`；v17 → `<INSTALL_DIR>/kernel2/v17/`；复用 libncclpin 挂载蓝本 |
| **ADR-4** | **SASS 门禁判据 = sm_120a `mma.*e2m1`/`mmaf`** | ✅ Accepted | ⚠️ **勿用 `tcgen05`**（那是 SM10x）；sass 历史脚本含 tcgen05 为陈旧，须更新 |
| **ADR-5** | **对照基准用 vLLM 官方语义** | ✅ Accepted | kernel① 对照 = `dequantize_to_dtype`（rel<0.02）；kernel② 对照 = 逐字节 `torch.equal`。**不要用旧 torch 32-group ref**（rel 0.19/1.35 属误报） |

**详细背景与选项对比**：见 `deliverables/engineering-assurance/architecture-nvfp4-2026-08-20.md` 第三章 ADR-1~5。

---

## 3. 交付成果与证据（已完成）

| 交付物 | 证据 | 位置 |
|---|---|---|
| **NVFP4 路线 A**（kernel① 原生 FP4） | 正确性 8/8 rel=0.00141（vs 官方 `dequantize_to_dtype`）；性能 60~187 TFLOPS（大 shape 12288 峰值 187）；零构建；SASS 门禁 sm_120 cubin + 1349 FP4 符号 | `nvfp4-landing/kernel1/nvfp4_4w4a_mmaf.py` |
| **kernel② v17** 替换 v11 | 8/8 逐字节；大 T 194~262 GB/s（3.5~4.6× v11，T=1024 达 96% 理论 273）；安全审计 4 项通过 | `nvfp4-landing/docs/runbook-kernel2-v17.md` |
| **统一资料库** | **32 个文件**：README 入口 + kernel1 适配层 + docs×6 + tests 全量 | `deliverables/engineering-assurance/nvfp4-landing/` |
| **四份审查报告** | 架构 / 代码 / SRE / 测试 | 见下方 §3.1 |
| **路线 B 结论**（备选阻塞） | flashinfer 0.6.15 四 backend 全阻塞，记录备用 | `nvfp4-landing/docs/ROUTE-B-FEASIBILITY.md` |
| **旧料清理** | 6 个历史交付目录归档 `_archive_old_deliveries/old-deliveries-2026-08-20.tgz`（1.4M/336 文件） | 容器内归档 |

### 3.1 四份审查报告落盘位置
| 报告 | 成员 | 评级 | 路径 |
|---|---|---|---|
| 架构审查 | 阿奇 | —（方案已定） | `deliverables/engineering-assurance/architecture-nvfp4-2026-08-20.md` |
| 代码/配置审查 | 科迪 | 🟡 有条件 | `deliverables/engineering-assurance/code-review-cluster-2026-08-20.md` |
| SRE 可运维/可靠性 | 雷克斯 | 🟡 有条件 | `deliverables/engineering-assurance/sre-ops-reliability-2026-08-20.md` |
| 测试策略/覆盖 | 泰莎 | 🟡 有条件 | `deliverables/engineering-assurance/testing-strategy-2026-08-20.md` |

---

## 4. 审查发现汇总（四位成员按严重度合并去重）

> 严重度：🔴=严重 / 🟠=高 / 🟡=中。已按问题合并去重，来源注明成员。

### 4.1 🔴 严重项（接手第一优先）

| # | 来源 | 问题 | 建议处置 | 负责人 |
|---|---|---|---|---|
| **S1** | 科迪 | **API key 明文泄漏到启动日志**：`start_tp4_head.sh:77` / `start_tp4_worker.sh:76` 把含 `--api-key <明文>` 的 `$SERVE_CMD` echo 到 `$HOME/start_tp4_*.log`（若未 chmod 600 会暴露生产密钥） | ① echo 前脱敏 `--api-key ****`；② 改从 `VLLM_API_KEY` 环境变量读取；③ `start_tp4_*.log` chmod 600 | SRE/编排 |
| **S2** | 科迪 | **shim 源码↔二进制↔文档三处漂移**：kit 内 `ncclpin.c` 标注 v3（布局 0-4/5-9），但 `libncclpin.so` 是 v8（布局 8-9/15-19）；按 .c 重编译会得到错误绑核，与验收的 isolcpus=8-9 相悖 | ① 把 kit 内 `ncclpin.c` 升级到 v8 源码并核对头注释；② BUILD 文档补「.c 必须能复现 .so MD5」检查项；③ 统一 default 区间 | Docu/Cody |
| **S3** | 科迪 | **三套网段口径未统一**（192.168.5.x / 10.100.x / 10.20.0.x），运维易拿错 IP | 建单一权威 `hosts`/netplan 映射；README 固话四机 IP/环口/角色；补 10.20.0.x MTU | SRE/Docu |
| **R1** | 雷克斯 | **生产 4 rank 未恢复（GPU 0%）**：08-19 起按用户要求未恢复；恢复决策归 P2，但「已上线」≠「可用」 | P2 统一决策恢复；恢复须走 `start_tp4_cluster.sh` head-first + GPU-gate + 回归三件套，**禁止单机 `docker run` 拉起** | 主理人/用户裁决 |
| **R2** | 雷克斯 | **自愈机制 disable**：08-19 SSD 事故后 monitor（`vllm-tp4-head.service` + `vllm-healthcheck.timer`）被 disable；timer `is-enabled` 与 disable 记录矛盾需显式 mask；宕机无人感知（08-07 现 Exited 13h 无人发现先例） | 恢复前核对并恢复 head monitor + healthcheck timer（`systemctl is-enabled/masked` 逐一确认）；建议补内存<2G 告警 | SRE |
| **R3** | 雷克斯 | **`/vllm-workspace/` 非持久 = 重建丢失**：NVFP4 产物若仅存容器内即重建丢 → RPO=∞、回退失效 | **P0**：routeA 适配层 + v17 落宿主机 `<INSTALL_DIR>/`，软链进 site-packages，`import` 透明 | 交付/QA |
| **G1** | 泰莎 | **P0 持久化 import 验收无自动化测试**：只有手工命令（阿奇 §4.1 / 雷克斯 C3），无脚本/退出码 → 唯一单点不可自动化验证 | 写 `tests/prod-gate/check_c3_persist_import.sh`（exit 0/1 门禁，含 RouteA import + preprocess + `__call__` shape 断言）纳入 C 段 | 测试 |
| **G2** | 泰莎 | **「≥200 TFLOPS」无法通过/无门禁**：`bench_big.py` 只打印 TFLOPS 无 PASS/FAIL 断言，峰值 187 < 200（差 7%），大 shape 不跑正确性对照 | 给 `bench_big.py` 加 200 门槛断言 + 每 shape 附 rel<0.02；`exit 1` if 峰值<200；记录 187 为已知偏差 | 测试/交付 |
| **G3** | 泰莎 | **A–F 投产门禁无可执行测试套件**：手工 checklist 无脚本串联、无 Go/No-Go 汇总、无退出码 | 建 `tests/prod-gate/`，每项抽 `check_*.sh`，统一入口 `run_prod_gate.sh A`~`F`；重点自动化 B3（NCCL banner）、F1（鉴权 health） | 测试/SRE |

### 4.2 🟠 高项

| # | 来源 | 问题 | 建议处置 | 负责人 |
|---|---|---|---|---|
| H1 | 科迪 | shim 二进制为剥离产物，无法静态审计绑核；仅靠 SHA 判断可能有 MD5 相同语义不同的替换 | 提供带调试信息 v8 源码 + 可复现 MD5 构建产物；签名的 sha256 + 部署表注各机备份锚点 | 交付 |
| H2 | 科迪 | 防火墙默认 `.env` `:INPUT ACCEPT`（默认放行新进非 25000 端口），监控栈 3000/8191/Prom、registry 5000 暴露 | 评估 `:INPUT DROP` 白名单化；至少先给 3000/8191/9093 加 IP 白名单 | SRE |
| H3 | 科迪 | kit 内 `prometheus.yml` 仍是旧 TP2 拓扑（旧节点末段），未含 03/04 新节点，与生产不一致 | 同步为四机 <NODE_IP>（186/187/188/189 末段）+ vllm rank 聚合；明确 <NODE_IP>:8191 权威配置源 | SRE/Monitor |
| H4 | 科迪 | SSH 编排无 `StrictHostKeyChecking`/超时；worker 无引号复合命令解析风险 | `ssh -o ConnectTimeout -o StrictHostKeyChecking=accept-new`；here-doc/base64 传参 | SRE |
| H5 | 科迪 | `start_tp4_cluster.sh` 无 `set -e`，关键步骤无显式失败语义，可能「假成功」 | 对 TCPStore 就绪/worker 容器/startup complete 用显式返回值 + 统一 exit | SRE |
| H6 | 科迪 | `ncclIbPeerHcaOverride` 用 2048 固定缓冲 + `atoi` 无校验 + 精确匹配 | 动态长度解析；`atoi` 校验；dev 匹配 prefix 兜底 + WARN；标注长度上限 | 交付 |
| **R4** | 雷克斯 | **UMA 内存耗尽复发（08-19 根因未根除）**：生产 util 回到 0.80，03/04 仅 ~2.5G 头寸；conc3×长上下文并发可能再触发 NCCL 超时+节点冻死 | 恢复 0.70 或保留 0.80+硬约束；内存 avail<2G 告警 + 后手 0.65/降 max-num-seqs；事故格单独复验 | SRE/**用户裁决 0.70/0.80** |
| **R5** | 雷克斯 | 监控告警覆盖盲区：job 名旧节点末段；vllm 抓取含 worker（无 API 口）；节点卡死时 Prom 中断 | 清理旧命名与失效目标；为 avail 内存/KUBE/UMA/NCCL 超时/GPU 0% 加显式告警；node 卡死视为 SEV 信号 | SRE/Monitor |
| **R6** | 雷克斯 | 配置漂移历史复发（rank 映射颠倒、MTU/shm/LD_PRELOAD 多处失准） | 交接以运行态实测为准；关键锚点 NCCL=`b7784b49`、shim v8=`ce43c688` | 全团队 |
| **R7** | 雷克斯 | monitor 曾在停机窗口自动拉起 rank0，误伤生产 | 停机 SOP 固化：**先 stop timer+service → 再停容器 → 完成后恢复**（写入事故预案） | SRE |

### 4.3 🟡 中项（P2）

| # | 来源 | 问题 | 建议处置 |
|---|---|---|---|
| M1 | 科迪 | NVFP4 RouteA 尚未落 `<INSTALL_DIR>/scripts`（=R3/G1 同一问题） | 同 R3 P0 |
| M2 | 科迪 | `_dequant_w_our` 每次全量反量化+重量化（21ms/层）、`W_scale` 假定 E8M0 | 常频用 `RouteA` 类缓存；抽查 W_scale 语义 |
| M3 | 科迪 | `RouteA` 多入口状态耦合、`A.float().half()` 显式截断、非线程安全 | 明确单线程逐层调用约定；补并发 guard 注释 |
| M4 | 科迪 | 文档/内核参数/源码三处 CPU 布局歧义（isolcpus=8-9 vs v3 源码注释 0-4/5-9） | 统一到 isolcpus=8-9 已验收事实，清理 v3 残留 |
| M5 | 科迪 | 5s 高频 scrape 磁盘/网络开销；告警规则待加载未闭合 | 复核 retention/disk；确认 alert_rules 四机加载对齐 |
| M6 | 科迪 | 根目录 103 个 `tmp_*` 临时脚本爆炸，无清理契约 | 归一化到 tools/scripts；`_archive_scratch/`；定期归档 |
| R8 | 雷克斯 | 双 Grafana（01/02 各 :3000）未收敛 | 保 02 权威，删去重 |
| R9 | 雷克斯 | `.local-backup` 删除决策（03/04 各 156G 兜底） | 暂缓删除至 NFS 持续稳定≥7 天或补 HA |
| R10 | 雷克斯 | 安全暴露面：sudo 密码明文、API key 硬编码 755、litellm master_key 明文、ssh 默认密码认证 | P0 安全轮换 + 脚本收敛 700/750 + sshd 收敛 |
| R11 | 雷克斯 | 01 时区漂移（+0800 vs UTC） | 统一 UTC 再对齐日志时间线 |
| R12 | 雷克斯 | SSD KV 卸载模块结论保留「不可行」，已回滚 | 不得重新启用；校验 util=0.80 版（md5 472c58bb） |

---

## 5. P0 / P1 / P2 待办清单

> 合并 HANDOFF-TO-TEAM.md 原有待办 + 四位成员新发现。每条含**完成判据**（可验证）。

### 5.1 P0（接手第一优先，安全 + 持久化 + 自愈）

| # | 任务 | 完成判据 | 参考脚本/文件 |
|---|---|---|---|
| **P0-1** | **修复 API key 明文日志泄漏**（S1）：echo 脱敏 + 改环境变量读取 + log chmod 600 | grep 生产启动日志无明文 64-hex key 泄漏；`ss`/`docker logs` 均脱敏 | `start_tp4_head.sh:77`、`start_tp4_worker.sh:76` |
| **P0-2** | **消除 shim 源码↔二进制漂移**（S2）：kit 内 `ncclpin.c` 升 v8，三处文档统一 isolcpus=8-9 | `ncclpin.c` 头注 v8 + 布局 8-9/15-19 + BUILD 记录 MD5=ce43c688；无 0-4/5-9 残留 | `lib/ncclpin.c`、`runbook-tp4-v1.5`、`shim-deploy.sh` |
| **P0-3** | **统一四机网络寻址权威表**并同步 netplan（S3） | kit README/`hosts` 一张表覆盖 4 机 IP+环口+角色；configs 与 RANK_HOST 一致；补 10.20.0.x MTU | `start_tp4_cluster.sh:31`、`.env.example`、`97-roce-mtu.yaml` |
| **P0-P 持久化落位** | **routeA + v17 落宿主机 `<INSTALL_DIR>/`**（R3/M1/G1） | 四机重建容器后 `import nvfp4_4w4a_mmaf` + routeA `preprocess+__call__` 跑通（见 C3 判据）；产物不落 `/vllm-workspace` | `nvfp4_4w4a_mmaf.py`、`<INSTALL_DIR>/scripts/nvfp4/` |
| **P0-S 恢复自愈** | **恢复 monitor + healthcheck 自愈**（R2）：从 disable 恢复并显式 mask | `systemctl is-active vllm-tp4-head.service` = active；`is-enabled vllm-healthcheck.timer` = enabled 且与 mask 一致 | `vllm-tp4-head.service`、`vllm-healthcheck.timer` |

### 5.2 P1（投产深化）

| # | 任务 | 完成判据 | 参考脚本/文件 |
|---|---|---|---|
| **P1-1** | 防火墙默认 ACCEPT→DROP + 监控栈加 IP 白名单（H2） | 白名单化后外部仅白名单可达，25000 链不受影响 | `configs/iptables/rules.v4` |
| **P1-2** | 同步 kit `prometheus.yml` 到 TP4 四机拓扑；确认 8191 retention/ disk/ 告警闭环（H3/M5） | kit 与生产 8191 比对一致；4 条告警规则在 <NODE_IP>:8191 均加载 | `configs/monitoring/prometheus.yml` |
| **P1-3** | 按 HANDOFF §3 P0 落 routeA 到 `<INSTALL_DIR>/scripts/nvfp4/`（交付/QA，=P0-P） | 四机重建后 import 跑通 smoke | `nvfp4_4w4a_mmaf.py` |
| **P1-4** | 生产性能简测 + **200 TFLOPS 冲刺**（G2/F4a）：A 量化融合/CUDA Graph/cutlass backend；加 200 断言 | `bench_big.py` 峰值≥200 TFLOPS 且 exit 0；否则单独立项决策 200 是否硬门槛（**用户裁决**） | `tests/bench_big.py` |
| **P1-5** | kernel② v17 四节点分发 + 切换调用点 + md5 校验（R3/paged） | 四个节点 v17 文件 md5 一致；调用点切 `_v17_triton`；8/8 逐字节 + 带宽达标 | `docs/runbook-kernel2-v17.md` |
| **P1-6** | 对照 v15 用真 v15 Triton 内核作基线（G5/F5） | `compare_v15.py` 改真 v15 基线后 ≥1.5× 达标 | `tests/compare_v15.py`、`nvfp4_4w4a_prefill_gemm_v15_triton.py` |
| **P1-7** | SASS 路线A 门禁替换陈旧脚本（G6/D4）：只认 `mma.*e2m1`/`mmaf`，删 tcgen05 | 新建 `sass_gate_routeA.sh` 固化 _C_stable .so cubin 判据；历史脚本归档标注旧路径 | `tests/sass/sass_gate.sh`、`sass_check_prefill_gemm.sh` |
| **P1-8** | 自愈/内存告警/4rank 恢复回归（G7）：补 UMA avail<2G 告警 + 自愈断言脚本 | Prometheus 存在 `avail_mem_bytes` 告警且阈值生效；自愈 is-active/is-enabled 断言脚本通过 | R4/R5/R7 |

### 5.3 P2（收尾）

| # | 任务 | 完成判据 | 参考脚本/文件 |
|---|---|---|---|
| **P2-1** | **4 rank 生产恢复决策与执行**（R1）：当前按用户要求未恢复 | 用户决定恢复后：`start_tp4_cluster.sh` → 4 rank healthy + 4 GPU + `/v1/models` 200（F1-F3）；决定不恢复则记录归档 | `start_tp4_cluster.sh`、prod-gate F 段 |
| **P2-2** | kernel① edge-case / 负向覆盖（G4） | 仿 kernel② 写 `test_nvfp4_4w4a_edge.py`：全零/±6/1e30/1e-30/-0.0 rel<0.02 | `tests/kernel2/test_..._v17.py` 为范本 |
| **P2-3** | kernel① 确定性/显存回归（G9） | 为 RouteA 补 determinism + 无显存增长 + Warmup（R2 加固）测试 | 新增 `test_routeA_determinism.py` |
| **P2-4** | 修正 safety 套件 3 个脚本缺陷（G8） | `test_saturation`=144、`test_sign_zero`=24、`test_boundary_T` 加 seed → pytest 全绿 | `tests/kernel2/test_..._v17_safety.py` |
| **P2-5** | 事故格复验（RC1–RC6）：65536/coding/conc3 + UMA 告警 + NCCL 被动受害者识别 + 停机窗口防误拉演练 | conc3×65536 连续≥4 次不触发 UMA 耗尽；RC2-RC6 逐项断言通过 | 见 §7 测试矩阵 |
| **P2-6** | `evidence/` 归档基准原始输出（G10） | bench_mmaf/bench_big/compare_v15/benchmark_v17 一次干净输出落 `evidence/` | `nvfp4-landing/evidence/` |
| **P2-7** | 清理 tmp 脚本爆炸 + 收敛安全暴露面（M6/R10） | `tmp_*` 归一化；sudo 密码/API key/litellm master_key 轮换 + 脚本 700/750 + sshd 收敛 | R10 / tech-debt 方案 |
| **P2-8** | 时区统一 UTC（R11）、双 Grafana 收敛（R8）、.local-backup 决策（R9）、SSD 模块不复启（R12） | 各项状态固化 | — |

---

## 6. 可运维性

### 6.1 自愈铁律 + 停机先停 monitor
- **自愈 = monitor + healthcheck timer**（当前 disable，P0-S 必须先恢复）。
- ⛔ **停机窗口铁律**：**先 `systemctl stop vllm-tp4-head.service` + `vllm-healthcheck.timer` → 再停容器 → 完成后恢复**。否则 monitor 会在停机窗口自动拉起 rank0 误伤生产（R7/RC6 教训）。

### 6.2 UMA 告警 + 0.70 决策
- **根因**：UMA 内存耗尽（08-19 SEV1）→ avail 0 → NCCL 超时 → oom-killer → 节点冻死。
- **当前**：生产 util 回 **0.80**（0.7 验证有效但被 SSD 回滚覆盖）；03/04 仅 ~2.5G 头寸。
- **决策点（需用户裁决）**：恢复 0.70 或保留 0.80 + 硬约束（avail<2G 告警 + 后手 0.65/降 max-num-seqs）。
- **内存头寸基线**：生产前 03/04 avail≥4G（A4）；<2G 触发告警并按 SEV1 处理。

### 6.3 A–F 投产门禁
> 完整矩阵见 §7。A 环境 / B 镜像 / C 持久化 / D GPU / E 网络 / F 回归。任一 🔴 行 FAIL 即不投产。
> **阶段 A–C 全 Go 且 F1–F2 Go → 可投产**；F3–F6 为 NVFP4 专项深化，F4a 未达 200 需**单独立项决策**（不构成基础阻断）。

### 6.4 DRSOP 容器重建恢复流程
1. 重建容器 `dspark-vllm-gx10:0.2.1-v026.0`，重挂 `<INSTALL_DIR>/{scripts,lib,models,envs}`。
2. 校验 routeA + v17 在 `/opt/nvfp4/` 自动可见、`import` 成功。
3. 跑三类回归：SASS 门禁 → 正确性(8/8) → 性能(≥200 TFLOPS / v17 GB/s)。
4. 通过即恢复生产；失败即按 §6.5 回退。

### 6.5 回退三防线

| 层级 | 回退动作 | 代价 |
|---|---|---|
| kernel① | 删除 `nvfp4_4w4a_mmaf.py` 引用 → 落回 v15 | 0（不改 vLLM 本体） |
| kernel② | 调用点换回 `_triton`(v11) | 0（文件留存） |
| 全量 | 恢复 `<INSTALL_DIR>` 挂载 + 重建容器 | 低（源在宿主机） |

### 6.6 RTO / RPO 表

| 场景 | RPO | RTO | 说明 |
|---|---|---|---|
| 单容器崩溃（无自愈） | 0 | **25-40 min** | monitor 在位可降为 5-8min |
| 单节点冻死（UMA） | 0 | **30-60 min** | 03/04 低头寸风险最高 |
| 容器重建（NVFP4 产物丢失） | **高(∞)** 若只存容器内 | **+15-30 min** | **P0 落 <INSTALL_DIR> 后 RPO=0、RTO 回落** |
| 全量重挂载（DRSOP） | 0 | 40-60 min | 阿奇 §4.3 |
| 回退 kernel①→v15 | 0 | 0（删引用） | 最低代价 |
| 回退 kernel②→v11 | 0 | 0（换调用点） | 文件留存 |
| 回退旧镜像/TP2 | 0 | ≤15 min | runbook append §A.6 |

### 6.7 SEV 事故响应预案
- **严重度分级**：SEV1=全集群不可用/多节点宕机/权威源损坏（立即全员）；SEV2=单节点卡死/单 rank 掉线且自愈未生效（15min）；SEV3=单点异常但服务可用（1h）；SEV4=低影响（下工作日）。
- **SOP 按序**：S0 分诊（判等级/识别影响面）→ S1 止血（先确认监控是否在，节点冻死看 avail<2G，NCCL 超时先查内存再查网）→ S2 状态沟通（SEV1 每 15min/SEV2 每 30min）→ S3 缓解（head-first 恢复，禁止单机 docker run）→ S4 无责复盘（5-Why → 行动项 → 回填 runbook）。
- **已知高发速查**：NCCL 300s 超时①UMA②LD_PRELOAD(13.3)③网络；节点冻死→降 util；8001 卡 init→查 25999；healthy 但 401→带 `Authorization: Bearer <key>`；GPU 0% 全 rank=生产未恢复（非故障）。

---

## 7. 测试验收矩阵

> 采纳泰莎矩阵（testing-strategy-2026-08-20.md 第三章）。用法：每行 = 可执行检查项（判据 + 脚本路径）。状态列填 ✅PASS / ❌FAIL / ⬜待跑。任一 🔴 行 FAIL 即不投产。

### 阶段 A–B：环境 / 镜像 / 补丁
| 项 | 检查 | 判据 | 脚本/命令 | 优先级 |
|---|---|---|---|---|
| A1 | 4 机在线 | 4 ping 通 | `ping <NODE_IP>~<NODE_IP>` | 🔴 |
| A2 | 时区一致 | UTC（01 需先修漂移） | `timedatectl` | 🔴 |
| A3 | 隔离核 | cmdline 含 `isolcpus=8-9` | `cat /proc/cmdline` | 🔴 |
| A4 | 内存头寸 | 03/04 avail≥4G | `free -g` | 🔴 |
| A5 | 磁盘水位 | `<70%` 无满盘 | `df -h /` | 🔴 |
| A6 | NCCL/shim MD5 | `b7784b49`/`ce43c688` 四机一致 | `md5sum` | 🔴 |
| B1 | 生产镜像在位 | 含 `0.2.1-v026.0` 四机一致 | `docker images` | 🔴 |
| B2 | 无容器残留 | 无 Exited 残留 | `docker ps -a` | 🔴 |
| B3 | NCCL banner | `2.30.7+cuda13.0`，**无 13.3**（13.3=LD_PRELOAD 失效 No-Go） | `docker logs vllm-tp4-rank0 \| grep "NCCL version"` | 🔴 |

### 阶段 C：持久化（P0 门禁，新增自动化）
| 项 | 检查 | 判据 | 脚本 | 优先级 |
|---|---|---|---|---|
| C1 | routeA 落宿主机 | `<INSTALL_DIR>/scripts/nvfp4/nvfp4_4w4a_mmaf.py` 存在 | `test -f ...` | 🔴 |
| C2 | v17 落宿主机 | 存在 + md5 四机一致 | `test -d <INSTALL_DIR>/kernel2/v17/` | 🔴 |
| C3 | **容器重建 import** | import + preprocess + out shape OK，exit 0 | **`tests/prod-gate/check_c3_persist_import.sh`（G1 待写）** | 🔴 |
| C4 | 唯一权威源 | 无生产依赖 `/vllm-workspace` | `grep -r "vllm-workspace" <INSTALL_DIR>/scripts/` | 🔴 |

### 阶段 D / F：GPU / 回归（NVFP4 专项）
| 项 | 检查 | 判据 | 脚本/命令 | 优先级 |
|---|---|---|---|---|
| D1 | GPU 可用 | 4 rank 各识别 1 GPU | 容器内 `nvidia-smi` | 🔴 |
| D2 | NVIDIA runtime | runtime 在位 | `docker info \| grep nvidia` | 🔴 |
| D3 | NCCL ring 组网 | rank0~3 全连 TCPStore:25999 | rank 汇总 | 🔴 |
| D4 | SASS 门禁 | `mma.*e2m1\|mmaf` 出现（**勿用 tcgen05**） | 固化 `tests/sass/sass_gate_routeA.sh`（G6 待写） | 🔴 |
| F1 | health（鉴权） | HTTP 200（带 `Authorization: Bearer $KEY`） | `curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/v1/models` | 🔴 |
| F2 | chat 冒烟 | `2+2=?` → "4" | 容器内 curl chat | 🔴 |
| F3 | kernel① 正确性 | 8/8 rel<0.02 vs 官方 dequant | `python tests/bench_mmaf_final.py` | 🔴 |
| F4a | kernel① 性能 | **峰值≥200 TFLOPS**（当前 187，已知偏差；**加 200 断言，G2 待写**） | `python tests/bench_big.py` | 🔴 |
| F4b | kernel② 带宽 | 大 T ≥120 GB/s（实测 194~262） | `python tests/kernel2/benchmark_..._v17.py` | 🔴 |
| F5 | 对照 v15 | **≥1.5×（用真 v15 Triton 内核，勿用 bf16 代理，G5 待改）** | `python tests/compare_v15.py` | 🔴 |
| F6 | 自愈在位 | head monitor active + healthcheck timer enabled | `systemctl is-active/is-enabled` | 🔴 |

### 附加专项（🟠 P1/P2）
| 项 | 检查 | 判据 | 脚本 | 优先级 |
|---|---|---|---|---|
| kernel② 正确性 | 8/8 逐字节 | `pytest tests/kernel2/test_..._v17.py` | 🟠 |
| kernel② 安全 | 修脚本缺陷（G8）后全绿 | `pytest tests/kernel2/test_..._v17_safety.py` | 🟠 |
| kernel① edge | zeros/±0/±6/1e30/1e-30 rel<0.02 | 新增 `tests/kernel1/test_nvfp4_4w4a_edge.py`（G4 待写） | 🟠 |
| kernel① 确定性/泄漏 | byte 一致 + 无泄漏 | 新增 RouteA determinism/mem 测试（G9 待写） | 🟠 |

### 判定规则
- 阶段 A–C 全 Go（🔴 全 PASS）→ 可进入 D/F；F1–F2 Go → 基础可投产。
- F3–F6 为 NVFP4 专项深化；**F4a 未达 200 需单独立项决策并记录偏差**（不构成基础阻断，但 P1 冲刺必须启动）。
- **F4a/F5 判据补齐前不得宣称「性能验收通过」。**

---

## 8. 风险与权衡

| 风险 | 等级 | 缓解 | 是否需用户裁决 |
|---|---|---|---|
| **性能距 200 门槛 7%**（峰值 187 TFLOPS） | Med | P1 简测定位；A 量化融合/CUDA Graph/cutlass backend；**若不可达需确认 200 是否硬门槛** | ✅ **需用户裁决 200 门槛** |
| **v15 对照 1.5× 真实性**：当前用 bf16 matmul 代理非真 v15 Triton 内核 | Med | 改用真 `nvfp4_4w4a_prefill_gemm_v15_triton.py`（G5/F5）后再确认 1.5× | ✅ **需用户裁决对照口径** |
| **路线 A 依赖 vLLM 0.26 内置符号** | Med | 不改 vLLM 本体；适配层独立 .py 可回退；记录 API 契约供升级核对 | 否 |
| **路线 B wheel 阻塞** | Low（备选） | 保持记录；FlashInfer TOT/重编后再评估 | 否 |
| **持久化中断**（工件仅存 `/vllm-workspace`） | Med | **P0 立即落 `<INSTALL_DIR>`**；全部产物双写宿主机 | 否 |
| **生产 4 rank 恢复决策未定**（当前 GPU 0%） | Low-Med | P2 收尾统一决策；当前按用户要求保持未恢复 | ✅ **需用户裁决生产恢复** |
| **0.70/0.80 内存 util 取舍**（UMA 复发） | Med | 恢复 0.70 或保留 0.80+硬约束；avail<2G 告警 + 降 max-num-seqs 后手 | ✅ **需用户裁决 0.70/0.80** |
| **密钥卫生**（S1 泄漏 / R10 暴露面） | High | P0 立即修日志脱敏；接手窗口内完成密钥轮换 + 脚本收敛 700/750 + sshd 收敛 | ✅ **需用户裁决密钥轮换时机** |

---

## 9. 文件索引（关键文件/脚本/报告/资料库位置一览）

| 文件 | 路径 |
|---|---|
| **交接文档（本文件）** | `deliverables/engineering-assurance/handoff-tp4-cluster-2026-08-20.md` |
| 架构审查（阿奇） | `deliverables/engineering-assurance/architecture-nvfp4-2026-08-20.md` |
| 代码审查（科迪） | `deliverables/engineering-assurance/code-review-cluster-2026-08-20.md` |
| SRE 可靠性（雷克斯） | `deliverables/engineering-assurance/sre-ops-reliability-2026-08-20.md` |
| 测试策略（泰莎） | `deliverables/engineering-assurance/testing-strategy-2026-08-20.md` |
| 统一资料库 README | `nvfp4-landing/README.md` |
| routeA 适配层 | `nvfp4-landing/kernel1/nvfp4_4w4a_mmaf.py` |
| v17 替换手册 | `nvfp4-landing/docs/runbook-kernel2-v17.md` |
| 落地手册（路线 A/B） | `nvfp4-landing/docs/landing-runbook.md` |
| 测试对照矩阵 | `nvfp4-landing/docs/testing-matrix.md` |
| 路线 B 可行性 | `nvfp4-landing/docs/ROUTE-B-FEASIBILITY.md` |
| 清理清单 | `nvfp4-landing/docs/cleanup-inventory.md` |
| 原有交接草稿 | `nvfp4-landing/HANDOFF-TO-TEAM.md`（含 P0/P1/P2 + 环境事实 + 文件索引） |
| 正确性+性能 bench | `nvfp4-landing/tests/bench_mmaf_final.py`、`bench_big.py` |
| 对照 v15 | `nvfp4-landing/tests/compare_v15.py` |
| SASS 门禁 | `nvfp4-landing/tests/sass_fp4_check.py` |
| kernel② 测试 | `nvfp4-landing/tests/kernel2/test_..._v17.py` |
| kvssd 200G 执行/runbook | `kvssd-200g-execution-report-2026-08-19.md`、`kvssd-200g-runbook-update-2026-08-19.md` |
| 回退锚点 | `rollback-anchors-2026-08-12.md` |
| 落地审核 | `audit-doc-vs-server-tp4-2026-08-13.md` |
| NCCL/TP4 部署 | `tp4-service-deployment-guide-2026-08-13.md`、`start_tp4_cluster.sh` |

---

## 10. 重要教训

1. **08-19 SEV1 根因 = UMA 内存耗尽**，NCCL 300s 超时是**被动受害者**。教训：遇 NCCL 超时**先查内存 avail 再查网**；根因（UMA）× 复现组合（conc3 长上下文并发）**未根除**，仅靠"不跑 crash 组合"回避 → 复发风险中-高，须补 avail<2G 告警 + 恢复 0.70/降配后手。
2. **停机先停 monitor**（R7）：否则自愈在维护窗口自动拉起 rank0 误伤生产。固化到事故预案。
3. **对照基准必须用官方语义**（ADR-5）：旧 torch 32-group ref 会误报（rel 0.19/1.35 不代表错）；v15 对照须用真 v15 Triton 内核，勿用 bf16 matmul 代理（G5）。
4. **SASS 门禁勿用 tcgen05**：SM12x 用 `mma.*e2m1`/`mmaf`；历史脚本含 tcgen05 已陈旧须更新（G6）。
5. **持久化纪律**：`/vllm-workspace` 重建即丢，一切生产产物双写宿主机 `<INSTALL_DIR>`（唯一权威源）。
6. **bench 缓存陷阱**：`RouteA` 缓存按 W data_ptr；批量 bench 复用同 data_ptr 会踩缓存。
7. **框架 bug**：`nvfp4_emulation_utils.break_fp4_bytes` 用 CPU `kE2M1ToFloat_handle` 索引 GPU → 需先 `.cuda()`。
8. **配置漂移是本集群最大事故源**：历史多为「配置漂移 + 监控盲区 + 内存头寸」三者叠加；交接以运行态实测为准（NCCL MD5 `b7784b49`、shim `ce43c688`、util=0.80 版）。
9. **历史事故要览（已闭环）**：08-11 Grafana 外部不可达（容器 IP 漂移 + iptables 错位 + 自检用 127.0.0.1 造假象）；08-11 白名单阻断新链路（新网段配 IP 必同步放行）；08-06 NCCL init hang；08-02 GPU 指标。均回填 runbook。
10. **SSD KV 卸载判定不可行**：已回滚（util 0.80、无 kv-transfer），**不得重新启用**除非新立项。

---

## 附录：需人类负责人 / 用户最终裁决项清单

| 裁决项 | 当前状态 | 决策选项 |
|---|---|---|
| **② 200 TFLOPS 是否为硬门槛** | 峰值 187（差 7%） | 冲刺或明确降门槛（如 180） |
| **③ 生产 4 rank 是否恢复** | 未恢复（GPU 0%） | 恢复（head-first）或维持归档 |
| **④ 0.70 / 0.80 内存 util 取舍** | util=0.80，复发风险中-高 | 恢复 0.70 或 0.80+硬告警 |
| **① 密钥轮换时机**（S1/R10） | 明文泄漏 + 暴露面 | 接手窗口内完成轮换 |
| **⑤ v15 对照 1.5× 真实性** | bf16 代理（不准） | 改真 v15 后复核 |

*本文档由工程保障团队技术文档师多库基于四位成员（阿奇/科迪/雷克斯/泰莎）报告 + 现有交接草稿整合而成。所有待办均含可验证完成判据。关键裁决项标注需人类负责人/用户最终拍板。*