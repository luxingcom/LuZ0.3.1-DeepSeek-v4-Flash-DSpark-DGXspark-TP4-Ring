# AICAD 集群四机全面工程审查 + CUDA 13.2 升级评估（综合报告）

**日期**：2026-08-07
**工作流**：全面工程审查（任务一）+ CUDA 13.2 升级 Go/No-Go 评估（任务二）
**参与成员**：Cody（代码审查师）/ Archi（架构师）/ Rex（SRE）/ Tessa（测试专家）/ Docu（技术文档师）

---

## 📌 TL;DR（执行摘要）

- **整体结论**：四机为统一 DGX Spark GB10 集群（Ubuntu 24.04.4 / 驱动 580.173.02 / CUDA 13.0.3），健康度 🟡 有条件通过——.58 内存耗尽风险与 LLM TP2 停机（Exited 137 疑似 OOM）需立即处理；全机无防火墙/registry 无认证属高危安全暴露。
- **CUDA 13.2 结论**：**能升，但分层处理**——① 驱动层不动（580.173.02 已实测支撑 CUDA 13.2 容器，官方配对 13.2 需 595 beta 驱动不进 DGX 验证通道）；② 本机 toolkit 13.0→13.2 低风险可选（与容器完全解耦）；③ **vLLM 容器内 13.0→13.2 暂缓**（NGC cu132 镜像版本过旧、sm_121 原生支持未成熟、现无收益，且 LLM 停机根因是 OOM 与 CUDA 无关）。
- 严重度分布：🔴 严重 3 项 / 🟠 高 8 项 / 🟡 中 9 项 / 🟢 良好基线 6 项。
- 阻塞 / 非阻塞：**非阻塞**——无任何 CUDA 13.2 硬件级阻塞（13.2 容器已实测可运行），但生产升级须走 canary 灰度。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（审查整改项 20 项 + 升级路线已定但生产暂缓） |
| 阻塞项数量 | 3 项（🔴：.58 内存耗尽风险 / LLM TP2 停机 / 全机无防火墙+弱认证） |
| 关键行动项 | 10 条（P0×3 / P1×4 / P2×3） |
| CUDA 13.2 Go-No-Go | 🟡 有条件——能力已验证（容器级 13.2 可跑），生产执行暂缓，等 .55/.59 canary 验证 + 上游 cu132 tag |
| 建议下一步 | ① 限 comfyui 内存 + 恢复 vLLM TP2；② 防火墙/registry/口令安全三连；③ .55/.59 启动 cu132 canary 验证 |

---

# 模块 A：四机全面工程审查（任务一）

## A1. 四机健康度总览（实测 2026-08-07）

| 主机 | 角色 | 磁盘 | 内存 | 负载 | GPU/驱动 | 关键状态 |
|---|---|---|---|---|---|---|
| .58 edgexpert-0c69 | head/网关/存储/registry | 3.6T 36% | 🔴 110/121G（可用 10G，swap 3.1G） | 0.31 | 54°C/0%/13W | LLM worker Exited(137) 13h；comfyui RSS~63.7G；16 容器运行 |
| .60 spark-05cd | worker | 3.6T 25% | 108G 可用 | 2.2-2.8 | 56°C/0%/12W | sglang JIT 编译 sm_121 中（CPU 100%+）；vllm-node Exited(0) |
| .55 gx10-3f4d | 空闲 canary | 916G 24% | 118G 可用 | 0.13 | 42°C/0%/5W | 无容器；NFS 挂载模型；GNOME 桌面 |
| .59 gx10-31c4 | 空闲 canary | 916G 21% | 118G 可用 | 0.28 | 42°C/0%/4W | 同 .55 |

**共同基线**：Ubuntu 24.04.4 / 内核 6.17.0-1029-nvidia / aarch64 / 20 核 121G / 驱动 580.173.02 / 本机 CUDA 13.0.3 / Docker 29.2.1 / **四台 ufw 均未启用**。GB10 为统一内存架构（GPU 显存显示 N/A，GPU 工作集计入主机 RAM——comfyui 高 RSS 即此）。

## A2. 审查发现清单（合并去重，按严重度排序）

| # | 严重度 | 类别 | 位置 | 问题描述 | 建议修复 | 来源 |
|---|--------|------|------|---------|---------|------|
| 1 | 🔴严重 | 资源 | .58 | 内存耗尽风险：可用仅 10G、swap 已用 3.1G；comfyui 无内存上限占 ~63.7GB，再 OOM 将连锁杀 embed/litellm/网关 | 先给 comfyui 限内存（<64G）再拉起 vLLM；加内存告警阈值 | Rex |
| 2 | 🔴严重 | 服务 | .58+.60 | LLM TP2 停机：worker Exited(137)=SIGKILL 疑似 OOM（需 dmesg 确认）、node Exited(0)；推理不可用 13h 无人感知拉起 | dmesg 确认根因 → 双机同版本 0.2.1-v026.0 成对重启（head 先+轮询 25000）；补进程守护/探活 | Rex/Cody |
| 3 | 🔴严重 | 安全 | 四机 | 全机无防火墙 + 弱认证：ufw 均未启用；.58 暴露 registry:5000/NFS/minio:9001/neo4j:7474 至 Wi-Fi 段；registry 无认证无 TLS；.55/.58 同口令 | ufw deny+白名单（22/5000/2049 限内网）；registry htpasswd+TLS 或迁 Harbor；口令差异化+禁密码登录 | Rex/Cody |
| 4 | 🟠高 | 兼容 | start_head_E.sh:22,57,88 / start_worker_E.sh:22 | 启动脚本仍钉 vLLM 0.1.1（v0.25），与生产 0.2.1-v026.0（v0.26）脱节；TORCH_CUDA_ARCH_LIST=12.1a 硬编码 | 以生产 0.2.1-v026.0 重写脚本并入库；arch 参数化 | Cody |
| 5 | 🟠高 | 兼容 | start_head_E.sh:109 / dryrun_head.sh:31 | 运行时 bind 挂载 /opt/patch/core.py 与 /usr/local/cuda/bin/nvcc；换 13.2 镜像后 patch 不匹配、JIT cache 失效 | 改镜像内 overlay（rebuild 流程）；nvcc 用容器默认 | Cody |
| 6 | 🟠高 | 安全 | runbook/ADR-4 | NFS <NODE_IP>/24 白名单过宽、root_squash 待复核；registry+NFS 单点于 .58 | 收窄 IP、复核 root_squash、规划 HA | Rex/Cody |
| 7 | 🟠高 | 运维 | .60 | sglang 用 dev-cu13 非稳定镜像，JIT 编译期 CPU 100%+ 争抢资源 | 换稳定 tag + 预编译 kernel + JIT cache 持久化 | Rex |
| 8 | 🟠高 | 运维 | .58/.60 | 时区不一致（.58 UTC / .60 HKT）→ 日志与监控时间线错乱 | 统一时区（Asia/Shanghai）+ NTP 复核 | Rex |
| 9 | 🟠高 | 安全 | 多文件 | API key 硬编码于脚本/unit/代码默认值（start_head_E.sh:44 / baseline_head.sh:5,8 / responses-gateway.service:21 / embed_main.py:42） | env/secrets 注入 + 轮换；unit 改 EnvironmentFile | Cody |
| 10 | 🟠高 | 安全 | 互信配置 | SSH 私钥无口令、互信无 from= 限制 | from= 收窄 + 密钥轮换登记 | Cody |
| 11 | 🟠高 | 运维 | .58 | vLLM 无自动回滚脚本（现靠 ADR-0008 手工 + 0260-bak 容器，RTO≤15min 手工） | 补 rollback 脚本并演练；保留 bak 镜像 | Cody |
| 12 | 🟡中 | 运维 | .58/.60 | 废弃容器与镜像堆积：.58 三个 Exited、.60 残留 47GB+46GB 旧 vLLM 镜像 | 镜像 GC（保留 0.2.1/0.2.0-bak/embed 现行 tag） | Rex |
| 13 | 🟡中 | 兼容 | vLLM 镜像 | torch arch 列表无 sm_121，GB10 靠 PTX 兼容首载慢 | 确认上游是否支持编 sm_121；canary 验证 PTX 路径 | Rex/Archi |
| 14 | 🟡中 | 运维 | 四机 | daemon.json 配置漂移（insecure/registry-mirror/nvidia runtime 不一）；本机 nvcc 未入 PATH | 统一配置管理（ansible/脚本）；nvcc 补 PATH | Rex |
| 15 | 🟡中 | 运维 | .55/.59 | 生产节点跑 GNOME 桌面 | headless 化 | Rex |
| 16 | 🟡中 | 安全 | .60 | head 主 IP .52 副 .60 双地址，易 ARP 冲突 | 收敛单地址 | Cody |
| 17 | 🟡中 | 兼容 | entrypoint_gpu.sh:13 | 容器内运行时 pip install sentence-transformers 不可复现，13.2 下或拉错 torch | 烘焙依赖进镜像 | Cody |
| 18 | 🟡中 | 网络 | .55/.59 | 仅 Wi-Fi 接入，大镜像同步慢 | 组网后切 10.100.136.x | Cody |
| 19 | 🟡中 | 维护 | /opt/distribution | 分发脚本未入库（仅文档在仓库） | git 管理 + logrotate | Cody |
| 20 | 🟡中 | 供应链 | 镜像 | 0.2.1-v026.0 本地构建 build.commit=unknown、无 SBOM；镜像经第三方 mirror 拉取未全部 digest pin | 补 SBOM/digest pin 策略 | Cody |

**🟢 良好基线**：驱动版本四机统一（580.173.02）、GPU 温度正常（42-56°C）、磁盘余量充足（≥21%）、监控栈完整（prometheus/grafana/dcgm/alertmanager）、NTP 运行、digest pin（0.1.1）、sha256 硬校验、flock 防误删、prune until=24h、--timeout 重试、embed OOM 兜底、gateway 双 key 分离（UPSTREAM_API_KEY）等良好实践已在生产生效。

---

# 模块 B：CUDA 13.2 升级评估（任务二）

## B1. 核心事实（实测证据，非推测）

| 层级 | 现状 | CUDA 13.2 兼容性证据 |
|------|------|---------------------|
| 驱动 | 580.173.02（四机一致） | ✅ **comfyui ubuntu24_cuda13.2-dgx 容器已在 .58/.60 实际运行**（容器内 nvidia-smi：CUDA 13.2, Build cuda_13.2.r13.2/compiler.37953736_0）；.55/.59 已验证 13.2.0-base 容器可跑。580 官方配对为 CUDA 13.0，13.2 官方配对为 595.x（beta，未进 DGX OS 验证通道）→ 容器级 13.2 经 NVIDIA forward-compat（cuda-compat-13-2 兼容 580+）支持 |
| 本机 toolkit | 13.0.3（/usr/local/cuda-13.0） | ⚠️ 可升但**非必需**：容器内 CUDA 与本机 toolkit 完全解耦（nvidia-container-toolkit 只挂载宿主驱动库，与 /usr/local/cuda 无关）；仅本机 nvcc 编 13.2 代码才需要 |
| vLLM 容器 | CUDA 13.0（torch 2.11.0+cu130 / vllm 0.26.1.dev0 / libcudart 13.0.96） | ⚠️ **可行但暂不建议**：NGC cu132 vLLM 容器仅 0.19.0/0.20.1（远旧于 0.26.x，特性不可平替）；保留 0.26.x 需自建 cu132 镜像，风险集中在 sm_121 原生 kernel |
| embed 容器 | torch 2.11.0+cu130 / vllm 0.25.2.dev0 | ⚠️ 同 vLLM；torch cu132 可能破坏现有依赖，升级前 D2 前置验证 |
| GB10 sm_121 | 现经 PTX 兼容运行（torch arch 无 sm_121） | ⚠️ 上游 vLLM #38484：发布构建 TORCH_CUDA_ARCH_LIST 仍未含 12.1；自建需 TORCH_CUDA_ARCH_LIST="12.1"+FlashInfer AOT 12.1+Triton PTXAS 指向；sglang dev-cu13 已为 sm_121 原生 JIT（__CUDA_ARCH__=1210）→ 原生支持技术上可行 |

## B2. 兼容矩阵（Archi 核实 + 实测）

| 组件 | 版本 | 官方配对/最低驱动 | sm_121(GB10) | 结论 |
|------|------|------------------|--------------|------|
| 宿主驱动 | 580.173.02 | 配对 CUDA 13.0；13.2 需 ≥595 | 支持 GB10 | ✅ 13.2 运行时经 forward-compat 可用（实测） |
| CUDA 13.2 runtime（容器内） | 13.2.0/1 | 官方最小 595.x | CUPTI 支持 | ✅ 实测可用 |
| CUDA 13.2 toolkit（本机 nvcc） | 13.2.x | 需驱动 ≥595 | 原生 | ⚠️ 需先升驱动 |
| torch cu130 | 2.11.0+cu130 | — | sm_120+PTX→JIT | ✅ 现用 |
| torch cu132 | 2.12 系 | — | 发行版多为 120+PTX；原生 121 需自编译 | ⚠️ |
| vLLM 0.26.x | 0.26.1.dev0 | — | 现经 PTX 兼容 | ✅ 现用 cu130 |
| vLLM NGC cu132 | 0.19.0/0.20.1 | CUDA 13.2.1 | 未标注 121 原生 | ⚠️ 版本过旧 |
| CUDA 13.3 | 待发布 | 驱动 610 | — | 待核实，勿追 |

## B3. 升级决策（ADR 摘要）

| ADR | 决策 | 结论 | 理由 |
|-----|------|------|------|
| ADR-1 | 升级路径选择 | ✅ **容器内升级**（B 本机升级暂缓） | 改动面仅镜像、可灰度回滚；本机升级需 595 beta 驱动 + 全节点重启，曾现 nvidia-smi 失效案例 |
| ADR-2 | 驱动升级 580→595 | ❌ 暂缓 | 595 为 beta、未进 DGX OS 验证通道；当前驱动已实测满足 13.2 容器运行，无收益 |
| ADR-3 | 本机 toolkit 13.0→13.2 | ⏸️ 可选、低风险 | 与容器解耦、alternatives 秒级回滚；仅在需要本机 nvcc 编 13.2 时执行，建议 .55/.59 先行 |
| ADR-4 | 生产 vLLM 容器 13.0→13.2 | ⏸️ **暂缓（推荐）** | NGC cu132 版本过旧不能平替；自建 cu132 镜像 sm_121 原生成熟度不足；当前停机根因是 OOM 非 CUDA；跟踪 anemll 上游 cu132 tag，成熟后灰度 |
| ADR-5 | canary 环境 | ✅ .55/.59 作为 canary | 空闲、已验 13.2.0-base；在此重建 cu132 vLLM 镜像、评估 595 beta（如需原生 sm_121 + CUDA Graph）、跑 NVFP4 回归 |

## B4. 逐机升级决策

| 主机 | 决策 | 说明 |
|------|------|------|
| .58（head） | 不动 | 保持 cu130 + 580 驱动；待 cu132 重建镜像在 canary 验证通过后再灰度 |
| .60（worker） | 不动 | 与 .58 同镜像同版本 lockstep；sglang dev-cu13 保持（首次 JIT 属预期，建议持久化 cache） |
| .55/.59 | canary | 重建/拉取 cu132 vLLM 镜像、验证 sm_121 原生、跑回归矩阵 |

## B5. 升级验证测试计划（Tessa 5 阶段门禁，详见 upgrade-cuda-13-2-2026-08-07.md）

**A 容器冒烟（.55/.59）→ B vLLM 单机 → C TP2 集群 → D 全链路回归 → E 稳定性（4h 长稳）**。核心门禁：C3 GSM8K ≥94.5%（对标 95.0% 基线）、C4 吞吐落 80.8-96.5 t/s 区间、D1 网关 27 组合 0 错误。**方法学铁律**：随机前缀防 prefix-cache 假象、温度统一（GSM8K=0.6）、新旧同硬件同口径。升级前必须先恢复旧栈（bash ~/start_v026r_cluster.sh）采基线核对 08-05/06 历史基准。

## B6. 回滚方案（层级化）

1. **容器级（首选，分钟级）**：vLLM/embed 回切 cu130 旧 tag（.58 已有 0.2.0/0.2.1 双 tag）；comfyui 回旧镜像
2. **toolkit 级（秒级）**：本机 toolkit 走 update-alternatives 切回 13.0
3. **驱动级（兜底，需停机）**：DGX Spark 封闭平台须走 NVIDIA 官方源/DGX OS 通道恢复，禁手动 .run；需带外控制台
4. **回滚触发阈值**：GSM8K <94.5% / 吞吐 <旧基线 90% / e2e 非偶发错误 / 拉起连续 2 次失败 / 长稳卡死 OOM

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | .58 限制 comfyui 内存（<64G）→ dmesg 确认 Exited(137) 根因 → 双机成对恢复 vLLM TP2（0.2.1-v026.0，head 先+轮询 25000） | Rex | P0 | 今日 |
| 2 | 四机 ufw 防火墙基线（deny+白名单 22/5000/2049 限内网）+ 禁用 SSH 口令登录 | Rex | P0 | 本周 |
| 3 | registry:5000 加认证（htpasswd/TLS）或迁 Harbor；.55/.58 口令轮换差异化 | Rex/Cody | P0 | 本周 |
| 4 | 以生产 0.2.1-v026.0 重写启动脚本入库 + 生成 vLLM 自动回滚脚本并演练 | Cody | P1 | 本周 |
| 5 | 统一时区（Asia/Shanghai）、NFS 白名单收窄+root_squash 复核、SSH 互信 from= 收窄、API key 迁移 secrets 并轮换 | Rex/Cody | P1 | 本周 |
| 6 | .55/.59 canary：重建 cu132 vLLM 镜像（TORCH_CUDA_ARCH_LIST=12.1/sm_121a）+ Tessa A/B 阶段验证 | Archi/Tessa | P1 | 1-2 周 |
| 7 | 镜像 GC、sglang 换稳定 tag、daemon.json 统一配置管理、JIT cache 按 CUDA 版本分键 | Rex | P2 | 本月 |
| 8 | 跟踪 anemll/dspark-vllm-gx10 cu132 官方 tag 与 vLLM sm_121 原生支持；成熟后生产灰度（.58/.60） | Archi | P2 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- **待核实外部事实**：CUDA 13.2 官方最低驱动对 580 的正式声明（实测可用但官方矩阵为 595）；anemll cu132 tag 是否已发布；download.pytorch.org cu132 稳定 wheel 是否开放；vLLM 官方 aarch64 cu132 tag。
- 数据采集为单时间点快照（2026-08-07 03:45/11:45），.60 sglang 首次 JIT 的 CPU 100% 为瞬时状态。
- 未采集：dmesg OOM 证据（需 sudo）、prometheus 历史指标、comfyui 容器详细资源限制配置。
- 团队成员无法直接 SSH，全部结论基于主理人采集的原始数据 + 公开资料核实。

---

## 📚 数据来源 & 成员产出索引

- **原始数据（主理人采集）**：`_audit_20260807/raw/{58,60,55,59}.txt` + vLLM 镜像/容器深度探测（gather2/gather3）
- **Rex（SRE）**：四机健康度总览、20 项问题清单、CUDA 13.2 运维风险评估、P0-P2 行动建议
- **Archi（架构师）**：兼容矩阵（驱动×CUDA×torch×vLLM×sm_121）、双路径对比、ADR 决策、逐机决策、vLLM #38484 等上游核实
- **Cody（代码审查师）**：13 项发现表（含 start_head_E.sh:22,57,88 / dryrun_head.sh:31 等文件:行）、安全专项、升级兼容要点、Request Changes 结论
- **Tessa（测试专家）**：5 阶段门禁测试计划（A-E 22 用例）、回滚决策矩阵、基线口径铁律
- **Docu（技术文档师）**：报告双主线结构、升级实施文档章节草案、回滚 Runbook 要点、文档资产清单（SSOT 原则）
- **配套文档**：`upgrade-cuda-13-2-2026-08-07.md`（升级实施+回滚手册，本报告 B5/B6 完整展开）；既有 `runbook-dspark-vllm-2026-08-06.md` 建议升 v1.2 增补 CUDA 环境章节（只链接不复制）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。涉及生产变更（恢复 TP2、防火墙、registry 认证）请按变更管理流程执行并保留回滚通道。
