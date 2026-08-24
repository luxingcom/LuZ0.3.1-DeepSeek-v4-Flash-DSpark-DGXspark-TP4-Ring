# 代码/配置审查 —— DGX Spark TP4 四机集群工作区工程资产

**审查人**：科迪（Code Reviewer）｜**日期**：2026-08-20
**审查方式**：只读（证据取自有权限访问的工作区工程资产，不改动任何生产脚本/配置）
**审查范围**：NCCL ring-only 补丁与 libncclpin/shim、TP4 启动编排、<INSTALL_DIR> 持久化结构、监控、网络安全、NVFP4 落地适配层、tmp 临时脚本残留

> 依据团队给定的背景事实（勿推翻）：4 节点 01/02/03/04（管理网末段占位），环网，TP4 8-11 上线、ring-close 已闭环、iperf3 99-110G 重传≈0；生产容器 vllm-tp4-rank0，镜像 dspark-vllm-gx10:0.2.1-v026.0、vLLM 0.26；NVFP4 RouteA rel=0.00141、8/8 PASS、60-187 TFLOPS。

---

## 审查目标与证据来源

| # | 目标 | 审查对象 | 证据 |
|---|------|---------|------|
| 1 | NCCL ring-only 补丁 + shim | `nccl-ringonly-v2.30.7-patch.diff`、`lib/ncclpin.c`、`lib/libncclpin.so`、`scripts/shim-deploy.sh`、`BUILD.md` | 补丁 99 行、shim 源码、MD5 |
| 2 | TP4 启动编排 | `scripts/start_tp4_cluster.sh`、`start_tp4_head.sh`、`start_tp4_worker.sh`、`monitor_tp4_head.sh`、`check_vllm_script.sh` | 代码直读 |
| 3 | <INSTALL_DIR> 持久化 | `start_*.sh` BINDS、`nvfp4-landing/HANDOFF-TO-TEAM.md` | 挂载清单 |
| 4 | 监控 | `configs/monitoring/prometheus.yml`、告警/数据源若干 audit 报告 | 配置直读 |
| 5 | 网络安全 | `configs/iptables/rules.v4`、`99-sec.conf`、`97-roce-mtu.yaml`、`90-isolcpus.cfg`、QoS 报告 | 配置直读 |
| 6 | NVFP4 落地适配层 | `nvfp4-landing/kernel1/nvfp4_4w4a_mmaf.py` | 源码直读 |
| 7 | tmp 临时脚本残留 | 工作区根目录 `tmp_*` | glob 统计 |

---

## 严重问题（按严重度排序）

### 🔴 严重

| # | 文件:行/对象 | 问题 | 建议修复 |
|---|--------------|------|---------|
| S1 | `start_tp4_head.sh:77`、`start_tp4_worker.sh:76`（`echo "[i] serve 命令: $SERVE_CMD"`）+ 行 68/67（`--api-key "${VLLM_API_KEY}"`） | **API key 泄漏**：SERVE_CMD 内嵌明文 `--api-key` 后被 `echo` 打到启动日志（`$HOME/start_tp4_*.log`），同时是 `monitor`/systemd 输出的一部分；日志若未限制 0600 会暴露生产密钥。安全审查位 P0。 | ① `echo` 脱敏：把 `SERVE_CMD` 中 `--api-key <val>` 替换为 `--api-key ****` 再输出；② 或改 `vllm serve` 从 `VLLM_API_KEY` 环境变量读取（vLLM 支持 `--api-key` 也读环境/配置文件），避免命令串内嵌明文；③ `$HOME/start_tp4_*.log` chmod 600。 |
| S2 | `lib/ncclpin.c`（v3） vs `lib/libncclpin.so`（v8）+ `runbook-tp4-v1.5:18`、`shim-deploy.sh:19` | **shim 源码↔二进制↔文档三处不一致（可复现性风险）**：工作区 kit 内 `ncclpin.c` 标注 v3、布局 `NCCL→0-4 / EngineCore→5-9`；但 kit 内 `libncclpin.so` MD5=`ce43c688…` = `shim-deploy.sh` 声明的 **v8 期望值**（含正确 v8 布局 `NCCL→8-9 / EngineCore→15-19`）；runbook §A.1-6 亦记录 v8 为 `8-9/15-19`。**源码不是产物的源** —— 直接以该 .c 重编译会得到错误的 CPU 绑定布局（0-4/5-9），与已验收的 isolcpus=8-9 语义相悖。 | ① 将 kit 内 `ncclpin.c` 升级到 v8 源码（补 pt_nccl/pt_tcpstore→8-9、EngineCore/VLLM::EngineC→15-19、default→? 的实际主线），并核对头注释版本号；② BUILD 文档补充「释放出的 .c 必须能复现 .so MD5」检查项，避免再次发生源码/二进制版本漂移；③ 明确 default 区间（源码 v3 是 5-19，运行接受的是 15-19？需统一）。 |
| S3 | `start_tp4_cluster.sh:31` `MASTER_ADDR="<NODE_IP>"` 与 `RANK_HOST` 均用**管理网 192.168.5.x**；`.env.example:24-26`；但 `nccl-ringonly/BUILD.md` 与 `HANDOFF` 标注环网 `10.20.0.x`、`.env.example:29-36` 环网段含 `<RING_SUBNET>`、`10.20.0.x` | **网络拓扑寻址口径多处不一致（正确性/可维护性风险）**：控制面 TCPStore(25999)、vLLM 分布式均走**管理网镜像**（<NODE_IP>），而 RoCE 数据面走管理网（10.100.x / 10.20.0.x，见 head `NCCL_IB_HCA` 对口即有 `172`? 实为 `2026-08-13` 报告）。工作区 3 套网段描述（192.168.5.x / 10.100.x.x / 10.20.0.x）并存且未统一一处权威映射，运维极易拿错 IP。 | ① 建立单一权威 `hosts/netplan` 与 `RANK_HOST` 映射表，明确「管理网=控制面」「环网=数据面」；② 在 kit 顶部 README 用一张表固化四机 IP/环口/角色；③ 更新 `97-roce-mtu.yaml`/`99-nvidia-sync-cluster.yaml`（当前仅 10.100.x 无 MTU 的 10.20.0.x，需核对是否覆盖 TP4 实际使用的 10.20.0.x）。 |

### 🟠 高

| # | 文件:行/对象 | 问题 | 建议修复 |
|---|--------------|------|---------|
| H1 | `lib/libncclpin.so`（strip，无法 strings 出 CPU 布局）；`shim-deploy.sh:41-48` 仅 MD5 校验 | **shim 二进制为剥离产物，无法静态审计真实绑核范围**；仅靠 SHA 判断"等于某版本"，若攻击者/误操作替换为一个 MD5 相同但语义不同的库无法察觉（当前无签名）。 | 提供**带调试信息**的 v8 源码 + 可复现 MD5 的构建产物；对 libncclpin.so 生成并校验更强的完整性（如签名的 sha256 + 部署表注明各机备份锚点）。 |
| H2 | `configs/iptables/rules.v4:11-12` `:INPUT ACCEPT` / `:OUTPUT ACCEPT`（默认接受） | **防火墙默认策略 ACCEPT**（IN/FORWARD 非 DROP，只有 FORWARD=DROP）。虽然 roce 口有显式 DROP、conntrack 归还已有流，但**新进的非 25000 端口流量全部放行**；监控栈 3000/8191/Prom 对外、registry 5000 均暴露。`incident-grafana-unreachable` 也证明端口映射历史错位反复出问题。 | 评估收紧 `:INPUT DROP`（白名单化）——历史已做过 25000 收敛，建议把需要开放的端口（25000/8001/9100/9400/8191/3000/5000 等）显式 ACCEPT 后默认 DROP；至少先给 3000/8191/9093 加 IP 白名单。 |
| H3 | `prometheus.yml`（kit）仍是 vLLM `<NODE_IP>`/旧节点末段/旧 TP2 拓扑 | **监控 scrape 配置与 TP4 拓扑不一致（可维护性）**：kit 内 `prometheus.yml` 的 8 个 job 面向旧 TP2 双机（vllm head 末段 :8001），未含 03/04 新节点，也未含四机 DCGM/node-exporter 的 TP4 全集；实际生产 8191 由另一份 catalina/镜像维护。`scrape_interval` 全局 15s、vllm job 5s。 | kit 的 prometheus.yml 落后于生产 TP4 拓扑，应同步为四机（管理网末段）+ vllm rank 聚合 + 面板 node 分组口径；明确 8191 对外/Prom 数据源 <NODE_IP>:8191 的权威配置源，避免 kit 与 /opt 漂移。 |
| H4 | `start_tp4_cluster.sh:97、156-157` `ssh -o BatchMode=yes` 无 `StrictHostKeyChecking=no/accept-new`、无超时聚合 | **SSH 编排容错**：`BatchMode=yes` 但未显式处理 host-key 首次/变更；无 `ServerAliveInterval`；worker 启动命令为**无引号包裹的远程复合命令**（`NODE_RANK... nohup bash ... &`），ssh 串拼接多变量，rank 2/3 的 `RANK_HCA`/`PEER_HCA` 若含空格/特殊字符有解析风险（当前为逗号安全）。 | 使用 `ssh -o ConnectTimeout -o StrictHostKeyChecking=accept-new`；把远程命令改用 here-doc 或 base64 传参；对 RANK_HCA 做显式引号封装。 |
| H5 | `start_tp4_cluster.sh:122` `nohup bash ... > log 2>&1 &` 但 `set -uo pipefail` 下未 `set -e` | `start_tp4_cluster.sh` 用 `set -uo pipefail`（无 `-e`），多处 `cmd || true`、`if ! cmd` 已做，但 **grep 链/诊断不失败即中止**；若中间某 ssh 意外返回但未匹配 `$?` 检查，编排可能"假成功"。 | 明确编排主路径失败语义：对关键步骤（TCPStore 就绪、worker 容器出现、Application startup complete）用显式返回值判断并统一 `exit`，避免依赖 `trap` 副作用。 |
| H6 | `nccl-ringonly-v2.30.7-patch.diff:50-79` `ncclIbPeerHcaOverride()` 用 `strtok_r`+`sscanf→atoi` 解析 env，2048 固定缓冲 | **NCCL_IB_PEER_HCA 解析健壮性（低风险但值得留意）**：`buf[2048]` 截断、`atoi` 无错误校验、dev name 匹配用 `strstr` 语义的 `strcmp(props.name,name)` 精确匹配（若 props.name 带后缀/前缀会失配退回 fallback）；环境变量长度超过 2KB 会静默截断。当前 4 rank 表很小无碍，但属边界隐患。 | 用动态长度解析；对 `atoi` 校验；dev 匹配提供 prefix 匹配兜底并在失配时产 WARN（现已有 WARN）；文档标注 NCCL_IB_PEER_HCA 长度上限。 |

### 🟡 中

| # | 文件:行/对象 | 问题 | 建议修复 |
|---|--------------|------|---------|
| M1 | `<INSTALL_DIR>` 持久化（`HANDOFF-TO-TEAM.md:53-58` P0 未做） | **NVFP4 RouteA 适配层尚未落到 `<INSTALL_DIR>/scripts`**，容器 `/vllm-workspace/` 非持久，容器重建即丢——当前 `nvfp4_4w4a_mmaf.py` 仅存在于工作区 + 容器临时目录。这使"零构建落地"仍处脆弱态。 | 按 HANDOFF §3 P0 执行：把 `nvfp4_4w4a_mmaf.py` 落到 `<INSTALL_DIR>/scripts/nvfp4/`（已挂载进容器），容器内建 softlink/site-packages 引用，重启后 `import` 验证。 |
| M2 | `nvfp4_4w4a_mmaf.py:34` `_dequant_w_our` 用 `E2M1F` 查表 + `W_scale.to(float32)-127` | 适配层把"既有 W 格式"经查表+scale 反量化后再 `scaled_fp4_quant` 转官方格式——**每次 preprocess_weights 全量反量化+重量化（21ms/层）**；且 `_dequant_w_our` 假定 scale 为 E8M0 `2^(x-127)`，若调用方 W_scale 语义不同会被误解析。`nvfp4_4w4a_prefill_gemm` 便捷入口每次都重做 W 预处理（无缓存），高频调用性能差。 | ① 常用路径用 `RouteA` 类缓存（已有 `use_cached_w`），文档明确"生产高频须用类 + 预量化"；② 抽查 W_scale 语义与 E8M0 假设，若生产已是官方格式则跳过反量化→直接复用官方 `scaled_fp4_quant` 权重。 |
| M3 | `nvfp4_4w4a_mmaf.py:60-76` `__call__` 内 `alpha=1.0`、`out[..., :N]` 裁剪、`A.float().half()` | 正确性铁证 rel=0.00141 成立，但 `__call__` 存在**多入口状态耦合**：`use_cached_w` 与 `preprocess_weights` 之间 `self._N/_wq/_wsf` 共享状态，非线程安全；`A.float().half()` 对大 M 有显式半精度截断。 | 明确 RouteA 非线程安全/单线程逐层调用约定；`nvfp4_4w4a_prefill_gemm` 便捷入口每次 new `RouteA` 已规避全局态，标注清楚；补一个并发/重入的 guard 注释。 |
| M4 | `90-isolcpus.cfg` `isolcpus=8-9` + `rcu_nocbs=8-9`；`ncclpin.c` 头注释"CPU 0-4 (A725) 隔离" | **文档/内核参数语义与 shim 布局的历史版本歧义**：grub 是 `8-9`，但 shim v3 源码注释是 `0-4/5-9`，runbook v8 是 `8-9/15-19`。虽当前 v8 二进制正确，但三处文本互相矛盾，后续维护者易改错。 | 统一所有文档到"isolcpus=8-9 供 NCCL；EngineCore 15-19"这一已验收事实，清理 shim 源码注释里的 v3 残留。 |
| M5 | 监控 `scrape_timeout:5s` / `scrape_interval:5s`（vllm job）+ node/random TSDB 增长 | 5s 高频 scrape 对 4 节点 DCGM+node+vllm 的磁盘/网络开销需评估；prometheus 默认保留期与规则文件加载（`checklist-audit` 报告 4 条告警规则"待加载"）未闭合。 | 复核 8191 实例的 retention 与 disk 容量；确认 alert_rules 四机已加载并对齐 <NODE_IP>:8191 数据源；面板按 node 分组已成（grafana-optimize 报告），建议把 kit/vllm 的 5s job 与全局统一管理避免疯狂写盘。 |
| M6 | 工作区根目录 **103 个 `tmp_*` 残留**（30 个 `tmp_fi*.sh`、51 个 `tmp_probe*.sh` + 各自 `_fc_*`/`_b*_*` 临时落盘） | **临时脚本爆炸（可维护性）**：`tmp_probe51.sh`、`tmp_fi30.sh` 等一次性只读调查脚本 + `.txt` 输出直接堆在工作区根目录，命名无业务语义、无清理契约，203 个文件混淆生产/临时。 | 按 `tech-debt-workspace-cleanup` 方案把可复用 probe 归一化到 `tools/` 或 `scripts/`，一次性脚本进 `_archive_scratch/`；建立 `tmp_*` 定期归档规则；把仍需要的（如 probe_a1_final.py）挪进 nvfp4-landing。 |

### 🟢 低

| # | 文件:行/对象 | 问题 | 建议修复 |
|---|--------------|------|---------|
| L1 | `.env.example` 无 `NCCL_IB_PEER_HCA`/`NCCL_ALGO=RING` 等运行面关键 env 注释 | .env 只覆盖 API key 与网段，未对齐 head/worker 里实际注入的 NCCL env 全集。 | 在 .env.example 补运行面 env 模板（NCCL_ALGO/IB_HCA/PEER_HCA/TOS 等），作权威记录。 |
| L2 | `start_tp4_head.sh:99` `NCCL_DEBUG_FILE=/var/log/vllm/nccl-%h.log` + `NCCL_DEBUG=INFO` | NCCL_DEBUG=INFO 日志量大，长期运行占盘；属于可接受但建议降级/轮转。 | `NCCL_DEBUG_FILE` 加 logrotate；无外部排障时 `NCCL_DEBUG=WARN`。 |
| L3 | `shim-deploy.sh` `runsudo "sudo -n"` 依赖 NOPASSWD | 编排脚本运行需远端 root 免密 sudo，属既定运维模式，但应在 README 强调最小权限与密钥保护。 | README 补"NOPASSWD 仅限受控运维机"说明。 |

---

## 做得好的地方

- **ring-only 补丁质量高**：`nccl-ringonly-v2.30.7-patch.diff` 逻辑清晰（v1 环邻过滤 + v2 per-peer HCA 映射），带 BUILD/产物校验/历史 MD5 轨迹，且 kit 内 `libnccl.so.2.30.7` MD5=`b7784b…` 与 runbook 生产部署完全一致 —— 补丁可复现、可回滚，是本轮最大亮点。
- **启动编排健壮**：head-first 幂等、GPU-gate≤180s、TCPStore 连续 2 次探测、worker 120s 对端门禁、错误模式 `ERR_PATTERNS` 诊断、`trap` 诊断钩子、`check_vllm_script.sh` 前置自检、monitor 互斥自愈与 D3 快速失败 —— 面向"2026-08-10 重启事故"的防御网很到位。
- **shim v8 二进制 MD5 与部署工具锚点闭环**：kit 内 `libncclpin.so`=v8（`ce43c688…`）与 `shim-deploy.sh` 期望值一致，配备份/回滚锚点（.bak-pre-deploy/.bak-v7），运维可复验。
- **NVFP4 落地证据链完整**：RouteA rel=0.00141、8/8 PASS、60-187 TFLOPS、SASS 门禁脚本、测试矩阵/runbook/handoff 文档齐备；HANDOFF 明确 P0 持久化待办与"vLLM 官方语义比对"教训，工程严谨度高。
- **网络安全纵深**：rp_filter=1、RoCE 口 TCP DROP、25000 白名单链、isolcpus 隔离核、MTU 9000（管理网环口）均有持久化配置；iptables rules.v4 结构清晰可分表。
- **文档纪律**：runbook v1.5-R11 的 A/B/C/D 增量结构、`CHANGE` 变更门槛注释、verify 7 项验收 —— 是可维护性典范。

---

## 整体评级

### 🟡 有条件通过（CONDITIONAL PASS）

生产功能（TP4 ring-close、55+ 基准、NVFP4 正确性/性能）已被实测证明稳定，且本审查**未发现影响当前正确性/性能的运行时缺陷**；核心基础设施（ring-only 补丁可复现、编排防御、shim 锚点闭环、隔离核、rp_filter）质量过硬。

但有条件项集中于**可复现性与密钥卫生**：
1. 🔴 S1 明钥泄漏到启动日志（必须立即修）；
2. 🔴 S2 shim 源码↔二进制版本漂移（一旦按源码重编会得到错误绑核）；
3. 🔴 S3 三套网段口径未统一（运维出错源）；
4. 🟠 默认 ACCEPT 防火墙 / 监控配置与 TP4 拓扑漂移。

以上不影响已上线运行，但**阻挡"声明自洽、可长期维护"的验收标准**。

---

## 行动项（P0/P1）

| 优先级 | 行动项 | 责任建议 | 验收标准 |
|--------|--------|---------|---------|
| **P0-1** | **修复 API key 明文日志泄漏**：SERVE_CMD 打日志前脱敏 `--api-key`；或改从环境读取；`start_tp4_*.log` chmod 600 | SRE/编排 | grep 生产启动日志无明文 64-hex key 泄露；`ss`/`docker logs` 均脱敏 |
| **P0-2** | **消除 shim 源码↔二进制版本漂移**：kit 内 `ncclpin.c` 升到 v8（NCCL→8-9/EngineCore→15-19），并验证可复现 `ce43c688…`；三处文档统一 isolcpus=8-9 语义 | Docu/Cody | `ncclpin.c` 头注 v8 + 布局 8-9/15-19 + BUILD 记录 MD5 一致；无 0-4/5-9 残留 |
| **P0-3** | **统一四机网络寻址权威表**并同步 netplan（补齐 10.20.0.x 环网 MTU/寻址），固化"管理网=控制面 / 环网=数据面"映射 | SRE/Docu | kit README/`hosts` 一张表覆盖 4 机 IP+环口+角色；configs 与 RANK_HOST 一致 |
| P1-1 | 评估防火墙默认 ACCEPT→DROP + 监控栈(3000/8191/9093)加 IP 白名单（维护窗口） | SRE | 白名单化后外部仅白名单可达，25000 链不受影响 |
| P1-2 | 同步 kit `prometheus.yml` 到 TP4 四机拓扑；确认 8191 retention/disk 容量与告警规则加载闭环 | SRE/Monitoring | kit 与生产 8191 比对一致；4 条告警规则在 <NODE_IP>:8191 均加载 |
| P1-3 | 按 HANDOFF §3 P0 把 NVFP4 RouteA 落到 `<INSTALL_DIR>/scripts/nvfp4/`（容器重建不丢），重启 import 验证 | 交付/QA | 四机重建容器后 `import nvfp4_4w4a_mmaf` 跑通 smoke |

---

*本审查为工作区只读评审，产出不影响运行中生产。建议合并至本次工程审计交接主报告。*