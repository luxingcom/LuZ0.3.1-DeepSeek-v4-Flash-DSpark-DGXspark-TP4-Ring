# DGX Spark 四机部署文档 ↔ 服务器双向审核报告

**日期**：2026-08-13
**工作流**：双向审核（文档声明→服务器事实 / 服务器现实→文档覆盖，自定义工作流）
**参与成员**：Rex（SRE 运行态核验）、Cody（代码审查·文件配置核验）、Archi（架构评审·文档质量审计）、Tessa（测试专家·数据核验）、Zhen（汇编）
**审核对象**：
- `tp4-r12-final-report-2026-08-13.md`（r12 全矩阵测试报告）
- `tp4-service-deployment-guide-2026-08-13.md`（TP4 部署指南 v1.0）
- 交叉参考：`rollback-anchors-2026-08-12.md`、`runbook-tp4-v1.5-2026-08-12.md`、`tp4-r9-optimization-report-2026-08-12.md` 等
**审核方式**：四路并行、全程只读（SSH 密钥 + sudo 只读，未修改服务器任何配置、未重启任何服务）

---

## 📌 TL;DR（执行摘要）

- **数据层面高度可信**：r12 性能表 45 组合 × 7 指标 = 315 格逐格比对原始 CSV/summary，**0 数据错误**（13 格仅展示取整）；"45 组合 0 错误 / hit=0 / 验收三值"全部有据，验收真实通过。
- **配置类声明存在系统性漂移**：rank 映射颠倒（四路独立佐证）、MTU 1500（实为 9000）、shm-size 32g（实为 64g）、内存限制（实为无限制）、capture-sizes 含 72（实为无 72 且上限 64）、LD_PRELOAD 路径、显存账数字（算术自相矛盾）等 10+ 处与运行态不符。
- **安全暴露面确认**：sudo 密码明文于 docs/file-registry.md:102（自标 P0 待轮换）、API key 硬编码于 start 脚本（755 全局可读）、litellm master_key 明文 + 5 份 .bak；01 时区非 UTC。
- **最大服务器→文档漂移**：aicad 应用栈（Neo4j/Redis/Postgres/MinIO/dashboard/02:8003 等）完全未文档化。
- **两处运行风险**：.local-backup 按 r12 计划 8/14 删除后将使 NFS 变为无兜底硬依赖（4 rank 全链断风险）；head 脚本无回滚锚点。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（数据可信；配置文档需修订后可作为部署依据） |
| 判定统计 | Rex 29 项 ✅14/❌6/⚠️9 · Cody 19 项 ✅12/❌1/⚠️6 · Archi 16 项 ✅9/❌2/⚠️5 · Tessa 315 格 ✅302/❌0/⚠️13 |
| 严重度分布 | 🔴严重 4 项 / 🟠高 6 项 / 🟡中 10 项 / 🟢低 6 项（去重合并后） |
| 阻塞项 | 无（当前运行态健康：四机容器 Up 7h、RestartCount=0、全服务 200、环邻 ping 0 丢包） |
| 关键行动项 | 9 条（P0×2 / P1×4 / P2×3） |
| 建议下一步 | 安全轮换（P0）→ 文档修订批次（P1）→ .local-backup 删除决策暂缓 |

---

## 一、双向审核方法与成员分工

| 成员 | 方向与范围 | 产出 |
|------|-----------|------|
| Rex（SRE） | 文档→服务器：运行态声明 29 项（服务/参数/网络/NFS/资源/安全）；服务器→文档：漂移发现 11 条 | sre-runtime-verify.md |
| Cody（代码审查） | 文档→服务器：文件/配置/MD5/unit 19 项；脚本代码级 issue 16 条 | code-config-verify.md |
| Archi（架构） | 文档→服务器：拓扑/容量/SPOF 16 项；文档质量审计（无证据声明/内部矛盾） | arch-verify.md |
| Tessa（测试） | 文档→数据：315 格逐格重算；跨文档一致性；P1/P2 复测计划 | test-data-verify.md |

---

## 二、🔴 严重发现（4 项，去重合并、多路佐证）

### F1. 指南 NODE_RANK 标注颠倒（四路独立佐证，最高置信度）
- 指南 §3.2 表：「03=worker(rank2)、04=worker(rank3)」；§3.5 示例 NODE_RANK=2 标在 03 上。
- 服务器实测：**03(188)=rank3、04(189)=rank2**，环序实为 01(0)→02(1)→04(2)→03(3)→01。
- 证据：systemd `NODE_RANK` 环境变量、容器名（vllm-tp4-rank2/rank3）、进程名（Worker_TP2/TP3）、01 容器 `NCCL_IB_PEER_HCA` rank 映射四重佐证（Rex/Cody/Archi/Tessa 四路独立确认）。
- 影响：**照文档部署会导致 rank 错位、TP4 无法组网**。§4.1 环网文字叙述本身正确，矛盾仅在 §3.2 表与 §3.5 示例。

### F2. 明文凭据暴露面（安全 P0）
| 位置 | 内容 | 证据来源 |
|------|------|---------|
| `<INSTALL_DIR>/docs/file-registry.md:102` | 统一 sudo 密码 `<PASSWORD>` 明文（条目自标"⚠️ P0 待轮换"） | Rex + Cody 双路 |
| `start_tp4_head.sh:70` / `start_tp4_worker.sh:63` | vLLM API key `<API_KEY>-11282...` 硬编码，脚本 755 全局可读；归档/archi-test 副本同样泄露 | Cody |
| `litellm/config.yaml`（02） | `master_key` 明文，664 可读，另有 **5 份 .bak** 同样泄露 | Cody |
| `shim-deploy.sh` | sudo 密码经 `echo | sudo -S` 明文传递，远端 ps 可窥 | Cody |

### F3. 核心配置参数与运行态系统性漂移（文档失准，非服务器故障）
| 参数 | 文档声明 | 实测 | 发现人 |
|------|---------|------|--------|
| --shm-size | 32g | **64g** | Rex + Cody |
| 内存限制 | -m 100g --memory-swap 100g | **Memory=0（无限制）** | Rex + Cody |
| cudagraph-capture-sizes | "1..80 含 36/72" | **无 72、上限 64**（1 2 4 8 16 24 32 36 40 48 56 64） | Rex + Archi |
| LD_PRELOAD | <INSTALL_DIR>/lib/libncclpin.so | **/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2** | Rex + Cody |
| 镜像大小 | TP4 34.2G 四机 | 01=34.2G，**02/03/04=21.6G** | Cody |
| 运行镜像 tag | 0.2.1 | 0.2.1-**v026.0** | Rex + Archi |
| MTU | 1500 | **9000（jumbo，runbook 正确）** | Rex + Archi |

注：capture-sizes 无 72 与指南 §5.2「P4 fix72：capture 含 72 → TPOT 17×」的叙事直接矛盾——fix72 的修复成果真实存在（Tessa 数据佐证 TPOT 正常），但当前运行配置已演进为「max-cudagraph-capture-size=64 + sizes 1..64 含 36」，文档仍停留在历史叙述。

### F4. aicad 应用栈完全未文档化（服务器→文档最大缺口）
01/02 实际运行大量未文档服务：Neo4j(7474/7687)、Redis(6379)、Postgres(8082→5432)、MinIO(19000/50081)、aicad-fw-25000、dashboard(11000)、02:8003、02 上第二个 Grafana(:3000)、alertmanager(9093)、litellm-pg(postgres:16)。指南 §2.1 镜像清单与 §4.2 端口表均未覆盖；registry catalog 另有 20+ 未文档镜像仓（minio/neo4j/chroma/pgvector/comfyui 等）。未文档 cron：backup_pg_daily.sh、comfy_unload.sh（每 5 分钟）。

---

## 三、🟠 高（6 项）

1. **.local-backup 删除决策风险**：r12 计划 8/14 删除 312G 兜底。实测该目录名为 `deepseek-v4-flash-0731.local-backup`，03=156G + 04=156G（合计 312G，非单目录）。删除后 01/02 NFS 任一故障 → 对应 worker 断权 → **TP4 全链断且无本地兜底**（4 rank 强耦合）。当前兜底为手动软链切换，非自动 failover。（Archi）
2. **01 时区漂移**：01=Asia/Hong_Kong(+0800)，02/03/04=UTC——与指南 §7.1「UTC 时区」不符，墙钟差 8h 影响跨机日志对齐与排障。（Rex）
3. **head 回滚能力缺口**：01 无任何 start_tp4_head.sh 回滚锚点（.bak-tp4-*/.bak-r11-* 均缺），与 rollback-anchors §1.1 声明不符；另 01 残留一份 7424B 陈旧 start_tp4_worker.sh（与 02=8414B、03/04=8649B 不一致）。（Cody）
4. **回滚文档 NCCL MD5 未更新**：rollback-anchors §2.1 / runbook §A.3 仍写 v2 的 `4cc43e3b`，生产已为 v3 双口的 `b7784b49`（四机实测一致✅）——回滚锚点会误指向旧版本。（Archi）
5. **双 Grafana 未收敛**：01/02 各跑一个 Grafana(:3000)；r12 已列"去重建议（保 02 权威）"但未执行，指南 §1.1 仍写单点，两文档口径矛盾。（Rex + Archi）
6. **重启窗口与 r12 声明不符**：r12 称"四机顺序重启测试"，实测 02/03/04 于 8/12 17:10-17:12 重启、**01 迟至 8/13 01:02**（且 01 另有 22:34、23:33 两次额外重启）——非单一顺序窗口，报告叙述有美化之嫌。（Rex）

---

## 四、🟡 中（10 项）

1. **指南 §4.1 网段自相矛盾**：L152「02↔04=<NODE_IP>/30」vs L154「环网 TP4 用 10.100.x」——实测 02↔04 边在 10.20.0.x（8/30+12/30 双链路），环网 4 边中 1 边非 10.100.x；ASCII 图亦未画 02-04、03-01 两条边。（Archi）
2. **§3.3 显存账过时且算术不自洽**：40.5+6.38+32.91=79.79≠79.06；实测权重 40.5✅、激活/graph ≈3.1~5.6（非 6.38）、KV 34.38~36.67（非 32.91）。疑似 r9/util0.60 时代残留。（Archi）
3. **§5.4/§7.5 验证命令失效**：`docker logs | grep RING-ONLY` 无效（banner 在 NCCL_DEBUG_FILE）；`curl /v1/models` 因已启用 `--api-key` 鉴权返回 401，文档未提示需带 Authorization 头。（Cody + Tessa 实测）
4. **"vs 基线 TP4S 持平"证据不足**：TP4S 原始 CSV 为 0 字节（运行中断于 14/45）；用同配置最近基线 TP4R 对比可支持"无回退、decode 略升"（131072 coding c1: prefill 1801.6→2014.7、decode 92.1→110.0），建议报告改为引用 TP4R。（Tessa）
5. **GPU-gate 覆盖缺口**：nvidia-smi≤180s 门禁仅存在于 start_tp4_cluster.sh 编排路径，systemd 自愈路径（monitor→start_tp4_*.sh）未经过。（Cody）
6. **退避算法表述不符**：monitor 实为线性 60×n（60..600s），指南 §3.4 称"指数退避"。（Cody）
7. **NFS 加载秒数历史不一致**：r9 曾记 03=181.9s/04=212.4s，r12 记 117.5s/135.7s（本次实测日志精确吻合 r12：117.47/135.66 ✅），但两轮差异大且无原始计时文件，门禁值"<270s"与 runbook"120s 等 head"、指南"distributed-timeout 300"三者口径不一。（Archi + Rex）
8. **sshd 全默认配置**：PermitRootLogin/PasswordAuthentication 均为注释（默认密码认证开启），未显式收敛为仅公钥；核心脚本 -rwxrwxr-x group 可写。（Rex）
9. **Neo4j/MinIO 端口未收敛**：监听 0.0.0.0 且不在 iptables 白名单体系内（白名单仅覆盖 RoCE 环邻），管理网侧可达性未收敛。（Rex）
10. **nproc=18 机制表述不准**：实测 `nproc --all=20`（20 核全 online），nproc=18 系 SSH 会话 affinity 掩码（排除隔离核 8-9）所致，非"20-2 减核"。（Archi）

---

## 五、🟢 低（6 项）

1. Prometheus job 名/标签仍用旧 .55/.58/.59/.60 命名；vllm 抓取目标含 188:8001（worker 无 API 端口）。
2. monitor_tp4_worker.sh:59-60 存在不可达死代码（`exit 1` 之后的 `docker wait ...; exit 1`）。
3. daemon.json registry-mirrors 指向公网 daocloud/dockerproxy 加速器（镜像投毒面）。
4. worker unit `Environment=NCCL_IB_HCA=rocep1s0f1,rocep1s0f0`（2 口）被脚本 4 口覆盖，成死配置。
5. r12 TL;DR "c1 prefill 1591-2228 / decode 106-122"未标注为 coding/json 口径（prose 未含）。
6. 峰值 prefill 叙述"647-1006"仅覆盖大档；512/2048 段峰值达 1748-2252。

---

## 六、✅ 强确认清单（审核中完全成立的声明，供信任锚点）

- **r12 测试数据 100% 自洽**：315 格 0 数据错误；"0 错误/45 组合/hit=0"全有据；指南 §7.7 验收三值（131072 prefill≥2000、decode≥100、acc≥0.75）实测全部满足。
- **补丁 MD5**：libnccl.so.2.30.7=`b7784b49...`、libncclpin.so=`ce43c688`（v8）四机一致。
- **NFS 加载耗时**：03=117.47s / 04=135.66s，与 r12 声明精确吻合；挂载参数 nfs4.2/ro/hard/timeo=600/nconnect=4、源 <NODE_IP> / <NODE_IP> 全部属实。
- **内核与网络**：isolcpus=8-9 rcu_nocbs=8-9 四机一致、环网四段 IP 全配、iptables 白名单与 rules.v4 一致、rp_filter=1 持久化、mlnx-qos active、GID_INDEX=2、环邻 ping 0 丢包、Wi-Fi 禁用。
- **NCCL 运行参数**：RING/SUBNET_AWARE_ROUTING/NET_PLUGIN=none/MERGE_NICS=0/TOS=46/PEER_HCA 双 dev 全部命中。
- **服务健康**：8001/4000/8022/3000/8191/9400/9100/5000 全部存活，四机 TP4 容器 Up 7h(healthy)、RestartCount=0、systemctl --failed=0。
- **文档镜像**：01/02 `<INSTALL_DIR>/docs/` 与本地一致。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 轮换 sudo 密码；删除 file-registry.md:102 明文条目；API key 与 litellm master_key 改环境变量注入并清理 5 份 .bak；核心脚本收敛为 700/750 | 运维 | **P0** | 本周 |
| 2 | 修正指南 §3.2/§3.5 rank 映射（03=rank3、04=rank2），同步 01/02 docs 镜像——防照文档部署 rank 错位 | 文档/运维 | **P0** | 本周 |
| 3 | 文档修订批次 A（一次性订正）：MTU=9000、shm-size 64g、删除 -m 100g 声明、capture-sizes 改"1..64 含 36"、fix72 降级为历史命名、LD_PRELOAD 实际路径、镜像大小与 tag、§4.1 网段表述与 ASCII 图补全 | 文档 | **P1** | 8/15 前 |
| 4 | 补写 aicad 应用栈文档（Neo4j/Redis/Postgres/MinIO/dashboard/02:8003 端口、备份 cron、镜像清单、registry 仓目录） | 文档/运维 | **P1** | 8/15 前 |
| 5 | **暂缓 .local-backup 删除**：保留至 NFS 连续稳定 ≥7 天，或补 NFS 导出方 HA；正式决策留痕（ADR-5） | 架构/运维 | **P1** | 8/14 前裁决 |
| 6 | 文档修订批次 B：§3.3 显存账改为实测值并修正算术、§5.4/§7.5 验证命令补 API key 说明、r12 "TP4S 持平"改引 TP4R、rollback-anchors/runbook 的 NCCL MD5 更新为 b7784b49 | 文档 | **P1** | 8/15 前 |
| 7 | 01 时区改 UTC；补齐 head 脚本回滚锚点；sshd 显式收敛（禁密码认证/禁 root 登录）；Neo4j/MinIO 端口绑管理网白名单 | 运维 | **P2** | 8/20 前 |
| 8 | 执行双 Grafana 去重（保 02 权威）并更新指南 §1.1；清理 Prometheus 旧命名与失效抓取目标 | 运维 | **P2** | 8/20 前 |
| 9 | 维护窗口执行 P1 复测（重启 7/7、HEAD_KILL=0、PSR 负载采样）；P2 复测（iptables 行为、178G 清理核对、RoCE 双口带宽）按窗口排期 | 测试/运维 | **P2** | 下个维护窗口 |

---

## ⚠️ 待完善 / 已知局限

- 本次为**只读审核**：HEAD_KILL=0、PSR 负载下采样、iptables 白名单实际阻断行为、RoCE 23.86GB/s 等需负载/变更窗口的声明，仅给出复测计划未执行（见 Tessa 报告 E 节）。
- "~178G 清理"声明无 journal/history 直接证据（archi-test 镜像已不存在为间接佐证），判定 ⚠️。
- registry catalog 仅抽查，未逐一核对 20+ 未文档镜像仓的版本与用途。
- aicad 应用栈的功能正确性（Neo4j/Redis/Postgres 等）未纳入本次核验范围，仅确认"存在且未文档"。
- 首轮派员曾因环境未注册专家子代理类型而失败，改用通用代理承接同角色任务，团队协作流程保持一致；不影响产出质量。

---

## 📚 数据来源 & 成员产出索引

- Rex（SRE）原始产出：`_raw_audit_20260813/sre-runtime-verify.md`（29 项判定表 + 11 条漂移 + 5 条安全发现）
- Cody（代码审查）原始产出：`_raw_audit_20260813/code-config-verify.md`（19 项判定 + 16 条 issue + 8 条矛盾注记）
- Archi（架构）原始产出：`_raw_audit_20260813/arch-verify.md`（16 项判定 + 无证据声明清单 + ADR 1-5）
- Tessa（测试）原始产出：`_raw_audit_20260813/test-data-verify.md`（315 格核验 + 跨文档一致性 + P1/P2 复测计划）
- 服务器证据：四机 SSH 只读采集（docker inspect/logs、systemctl、iptables -S、/proc、sysctl、ss -tlnp、timedatectl、last reboot 等）

---

> 本报告由工程保障团队 AI 协作生成（2026-08-13），关键决策（尤其 P0 安全轮换与 .local-backup 删除裁决）请由人类工程负责人复核签字。
