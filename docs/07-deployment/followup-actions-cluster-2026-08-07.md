# DGX Spark 四机集群 — 后续任务与改进清单

**日期**：2026-08-07
**工作流**：技术债评估 / 运维待办汇编
**参与成员**：Zhen（主理人汇编，基于全天实施与排查结论）、Archi（迁移评估）、Rex（清理 SOP）、Cody（安全审查）、Tessa（验收评审）

---

## 📌 TL;DR

- 全天完成：镜像清理（~225G）、<MGMT_OCTET> 对齐、embed 4 机 HA、Grafana 面板修复与优化、环境迁移评估
- **当前 2 项 P0 阻塞**：① <MGMT_OCTET> 生产 LLM head 镜像被误删需重新获取（34.2G 完整版）② 口令轮换（全集群同密码）
- **关键决策**：TP=2 生产保留 anemll 0.2.1（新镜像 mp/ray 双路径均受限），新镜像承担 embed 单机任务
- 遗留安全加固 6 项（防火墙/ssh/registry 认证等）+ 架构改进 5 项 + 验证任务 4 项

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件运行（核心服务正常，2 项 P0 待处理） |
| P0 项 | 2（LLM head 镜像恢复、口令轮换） |
| P1 项 | 6（安全加固 4 + embed 回迁评估 + 迁移决策落实） |
| P2/P3 项 | 12+ |
| 建议下一步 | 优先恢复 LLM 生产（head 镜像），随后完成安全基线加固 |

---

## 🔴 P0 — 立即处理（阻塞生产/高风险）

### 1. 恢复 <MGMT_OCTET> 生产 LLM head（vllm-envE-node）
- **背景**：今日 `docker system prune -af`（<MGMT_OCTET> 对齐任务）误删 Exited 状态的 vllm-envE-node 容器及其镜像（ghcr 34.2G 完整版 0d9d37607520）
- **现状**：已从 registry 拉 21.6G 版重建 head，但**初始化反复失败**（NCCL 后卡死/循环重启，日志停在 autotune 配置，内存未加载权重）；worker（<MGMT_OCTET>）healthy 等待中；**LLM 生产服务当前停机**
- **行动**：
  1. <MGMT_OCTET> 重新获取原始完整版镜像：`docker pull ghcr.io/anemll/dspark-vllm-gx10:0.2.1-v026.0`（34.2G，验证 ID 是否为 0d9d37607520；ghcr 直连不通则经 dockerproxy.net 镜像源）
  2. 用完整版重建 head（启动参数与今日重建命令一致，含全部 fork ENV 与 patch 挂载）
  3. 验证：/health 200 + 1 次 chat/completions 推理 + litellm 上游恢复
- **回滚锚点**：worker 与全部配置未动；registry 版 21.6G 保留可作应急
- **负责人**：Zhen / 预期：镜像获取后 30 分钟

### 2. 口令轮换（全集群）
- **背景**：<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET> 共用密码 <PASSWORD> + 相同 sudo（Cody 审查 #2 🔴）；管理脚本曾明文落盘（已清理）
- **行动**：
  1. 各机差异化强口令（或统一轮换新口令），更新 sudo 密码
  2. sshd 关闭 PasswordAuthentication，全面密钥登录（改前确认密钥互通）
  3. 管理脚本凭据改走环境变量/vault
- **负责人**：Zhen / 预期：1-2 天

## 🟠 P1 — 短期（1 周内）

### 3. 安全基线加固（Cody 审查遗留）
| 项 | 说明 | 行动 |
|----|------|------|
| 防火墙 | 四机均未启用（集群待办） | ufw 默认 deny + 放行：ssh 22（管理网段）、registry 5000（<NODE_IP>/24 + <NODE_IP>/16）、docker/NFS 按需；**启用前评估对 NFS/registry 的影响** |
| registry 认证 | 内网无认证（ADR-4 记录） | htpasswd 基础认证（REGISTRY_AUTH=htpasswd）+ 可选 TLS；配置后各机 docker login |
| sshd 加固 | 未确认 PasswordAuthentication 状态 | 对齐基线：PasswordAuthentication no / PubkeyAuthentication yes / PermitRootLogin prohibit-password |
| logrotate | /var/log/distribution.log 无限增长 | 配 logrotate daily + 保留 14 份 |
| NFS 收窄 | /etc/exports <NODE_IP>/24 过宽 | 收窄为具体 IP（<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET>）；评估 sec=krb5p |

### 4. embed 服务恢复策略
- **背景**：为给 LLM 腾内存，<MGMT_OCTET>/<MGMT_OCTET> 的 embed-qwen3-vllm 已停（restart=no）；litellm 已自动 failover 到 <MGMT_OCTET>/<MGMT_OCTET>（验证通过 dim=1024）
- **行动**：LLM 生产恢复后评估双机内存余量，决定 embed 是否回迁 <MGMT_OCTET>/<MGMT_OCTET>（<MGMT_OCTET>/<MGMT_OCTET> 承担 embed 亦可，作为长期拓扑）

### 5. 环境迁移决策落实（任务③结论）
- **结论**：新镜像 vllm-gb10:0.26.1-cu132 无法承载 TP=2 生产（mp executor 多机 KV broadcast 不支持 + ray 连接失败，fork 差异）；**生产保留 anemll 0.2.1**
- **行动**：
  1. 更新集群架构文档：LLM=anemll 0.2.1（<MGMT_OCTET> head/<MGMT_OCTET> worker）+ embed=vllm-gb10（<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET>/<MGMT_OCTET> 按需）
  2. anemll 0.2.1 镜像**三处保留**（<MGMT_OCTET>/<MGMT_OCTET>/registry）作为不可再生资产，禁止再 prune 误删
  3. 跟踪 vLLM 上游 mp 多机修复或 anemll 新版本，再评估迁移

### 6. 镜像与磁盘清理收尾
- <MGMT_OCTET> 对齐完成（registry pull 19.2G 版 + 旧版删除 + prune 85G）；registry GC 完成（71G→40G）
- **剩余**：<MGMT_OCTET> 可再 prune（RECLAIMABLE 尚有空间）；sglang 两镜像（<MGMT_OCTET> 27.3G/<MGMT_OCTET> 38.7G）为视频工作流资产，**确认保留或清理**；<MGMT_OCTET>/<MGMT_OCTET> 旧 LLM 相关文件（patch-v026、tilelang-cache、env-e-build）保留至 LLM 稳定

## 🟡 P2 — 中期（2-4 周）

### 7. 集群监控完善
- Grafana：vLLM 恢复后验证新面板（ITL P50/P99、投机 Mean Length≈1.6、KV 水位 ×100 修复）；为 KV 水位与统一内存加阈值线（80%/95%）
- Prometheus：recording rules 已生效；compose 已补挂载；告警通道（alertmanager 接 webhook）待配置
- 各节点磁盘水位告警（/data≥70% WARN/85% CRIT 已有 disk-watch，接入告警通道）

### 8. 分发机制改进
- 分摊调优：litellm simple-shuffle 偏差（<MGMT_OCTET> 偏多）→ 评估 least-busy 或 weight 配置
- embed 超长输入（>8192）行为测试：定截断策略（vLLM vs 调用侧）
- litellm 网关单点：迁移 <MGMT_OCTET> 或双网关（<MGMT_OCTET>/<MGMT_OCTET> 各一）
- 备份 cron 化：<MGMT_OCTET> 关键镜像备份（vllm+embed tar 40G）目前一次性，需周期化 + sha256 对账自动化
- deepseek 156G 校验清单：<MGMT_OCTET> 生成 sha256sums.txt（约 30-60 分钟），<MGMT_OCTET>/<MGMT_OCTET> 补跑硬校验

### 9. 稳定性改进
- <MGMT_OCTET> dockerd 存储异常记录：load 同 tar 持续产出旧 ID（疑似 containerd 快照元数据损坏），已用 registry pull 绕过；若复发需重建 dockerd 存储
- vLLM 容器启动内存规划：LLM(97G 预算) + embed(12-18G) 共存超 121G 物理限制 → 明确"同机 embed 与 LLM 互斥"策略（HA 已支持）
- systemd timer 首跑验证（distribution-watch 已自动触发一次 ✅）

## 🟢 P3 — 长期/低优先

- anemll 0.2.1 观察 1 周后：确认无回滚需求再清理 registry 旧副本（保留 ghcr 原始源）
- sglang 用途确认：视频工作流如果长期使用，纳入正式运维（版本/监控/更新）；否则清理 66G
- 密钥登记表 + 季度轮换流程（SSH 密钥、registry 口令）
- 文档更新：四机版环境配置手册/分发手册已交付（2026-08-07），需补充：镜像清单变更、LLM 恢复 SOP（head 重建命令）、embed/LLM 互斥策略
- 组网规划：<MGMT_OCTET>/<MGMT_OCTET> 有线/RoCE 后同步源切 10.100.136.x（已记录，待硬件）

---

## ✅ 行动清单（按优先级）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 恢复 <MGMT_OCTET> LLM head（获取 ghcr 34.2G 完整版镜像重建） | Zhen | P0 | 镜像获取后 30min |
| 2 | 全集群口令轮换 + sshd 禁密码 | Zhen | P0 | 1-2 天 |
| 3 | 防火墙基线（ufw）+ registry 认证 | Zhen+Rex | P1 | 1 周 |
| 4 | logrotate + NFS 收窄 + sshd 加固 | Zhen | P1 | 1 周 |
| 5 | embed 回迁评估（LLM 恢复后） | Zhen | P1 | LLM 恢复后 |
| 6 | 更新架构文档（LLM/embed 双镜像体系） | Docu | P1 | 1 周 |
| 7 | Grafana 面板数值验证（vLLM 恢复后）+ 阈值线 | Zhen+Rex | P2 | 2 周 |
| 8 | litellm 分摊调优 + 单点治理 | Zhen | P2 | 2-4 周 |
| 9 | 备份 cron 化 + deepseek 校验清单 | Zhen | P2 | 2 周 |
| 10 | sglang 处置确认 + anemll 副本清理决策 | Zhen | P3 | 1 月 |

## ⚠️ 待完善 / 已知局限

- LLM 生产 head 停机中（P0-1 未完成前 AICAD LLM 业务不可用，embed 不受影响）
- <MGMT_OCTET> dockerd 存储异常根因未明（已绕过，未根治）
- 新镜像 ray executor 支持不完整（client server 失败），ray 迁移路径暂不可行
- prune -af 误删教训：**Exited 容器与无引用镜像会被 system prune -a 删除**，执行前必须确认（已记录坑位）

## 📚 数据来源 & 成员产出索引

- Cody（代码审查师）：20 条安全审查发现（8/7 上午）
- Archi（架构师）：生产迁移方案 + ADR-001（mp/ray 评估）、ADR-embed-002
- Rex（SRE）：四机清理 SOP、部署检查清单
- Tessa（测试专家）：<MGMT_OCTET>/<MGMT_OCTET> 验收、条件通过（A/B/C 门禁）
- Zhen（主理人）：全天实施记录、生产容器配置采集、1b 冒烟实测结论

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
> ⚠️ 重点提醒：LLM 生产 head 镜像（ghcr 34.2G 完整版）不可再生，恢复后务必三处备份；后续任何 prune 操作前先 `docker ps -a` 检查 Exited 容器。
