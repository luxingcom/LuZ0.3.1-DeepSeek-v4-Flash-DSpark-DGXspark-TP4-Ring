# API key 环境变量化 + 开源发布包交付报告

**日期**：2026-08-13
**工作流**：密钥改造与开源发布（自定义工作流：代码改造 → 维护窗口重启 → 打包消毒 → 独立验收）
**参与成员**：Cody（代码改造）、Rex（重启验证+打包）、Docu（README）、Tessa（独立验收）、Zhen（编排+终审）

---

## 📌 TL;DR（执行摘要）

- **API key 已完成环境变量化与轮换**：旧 key（<API_KEY>，已暴露于脚本/备份）从四机脚本与 unit 全部清除，新 key 入 `<INSTALL_DIR>/secrets/vllm.env`（600/root:root，四机 md5 一致），脚本 `${VLLM_API_KEY:?}` fail-fast 注入，litellm 上游 8 处对齐 `os.environ/VLLM_API_KEY`。
- **维护窗口重启验证 7/7 全过**：新 key 200 / **旧 key 401（轮换确认）** / 冒烟 / litellm 端到端 / 四机 healthy / PSR 绑核 / 日志无 error；TP4 服务零中断期外无异常。
- **开源发布包已产出并双轮审计**：`dgxspark-tp4-deploy-kit-2026-08-13.tar.gz`（~43MB / 1372 文件 / md5 e5627a95...），**零密钥、零密码、零免密配置、零模型文件**；README 617+ 行经独立验收→修订→终审三轮打磨。
- **遗留观察**：EngineCore 偶发线程落 PSR=13（shim v8 绑核口径待复核，P2）；包内 shim 源码为 v3 基线（v8 源码未归档，已在 lib/README 注明）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（服务/包/README 三段验收全过） |
| 密钥改造 | 环境变量化 + 轮换 + fail-fast 三件套完成 |
| 重启验证 | 7/7 通过，旧 key 401 确认轮换生效 |
| 发布包 | 43MB/1372 文件，消毒扫描 0 命中 |
| README | 9 处路径错位已全部修正并自查通过 |
| 遗留 | P2：shim v8 绑核口径复核、v8 源码归档 |

---

## 一、密钥改造（Cody）

| 项 | 内容 |
|----|------|
| 旧 key 定位 | start_tp4_head.sh:68、start_tp4_worker.sh:67（四机）、start_tp4_cluster.sh:34、start_v026r_cluster.sh:19（额外发现）、litellm config.yaml 上游 |
| 新 key 管理 | `<INSTALL_DIR>/secrets/vllm.env`（600/root:root，64hex，四机 md5 一致） |
| 脚本注入 | `--api-key "${VLLM_API_KEY:?VLLM_API_KEY is not set}"` fail-fast（未设置即启动失败，防无鉴权裸奔） |
| unit 注入 | vllm-tp4-head/worker.service + vllm-cluster.service 加 `EnvironmentFile=<INSTALL_DIR>/secrets/vllm.env` |
| litellm | config.yaml 8 处上游 key 对齐 `os.environ/VLLM_API_KEY`；master_key 未动（客户端在用的网关密钥） |
| 关键偏差发现 | litellm-proxy **非 compose 管理**（手动 docker run）→ 需 rm -f 重建注入变量，已转交 Rex 优先处理 |
| 清除核验 | 四机 `grep <API_KEY>` scripts/ + systemd/ = 0 命中 |

## 二、维护窗口重启与验证（Rex）

- **litellm 重建**（优先，消除窗口期竞态）：按原参数重建 + `--env-file` 注入新 key，Up/healthy，env 前缀确认
- **TP4 全链重启**：停机 worker 03→04→02 → head 01（monitor 确认退出 + 残留容器清理）→ head-first 拉起 → 300s 内四容器 Up(healthy)
- **验证矩阵 7/7**：/health 200 ｜ 新 key /v1/models 200 且 max_model_len=400000 ｜ **旧 key 401** ｜ 200-token 冒烟 200 ｜ litellm :4000 端到端 200 ｜ 四机 healthy + --failed=0 + RestartCount=0 + PSR(NCCL→8-9/EngineCore→15-19) ｜ 日志无 error
- 无回滚、无异常

## 三、开源发布包（Rex 制作 + Tessa 审计）

- **路径**：`deliverables/dgxspark-tp4-open-kit-2026-08-13/dgxspark-tp4-deploy-kit-2026-08-13.tar.gz`（44,788,088 字节，md5 `e5627a9521aa581f1ded08e8e3dfd592`，1372 文件）
- **结构**：docs/ scripts/ lib/ nccl-ringonly/ systemd/ configs/ secrets/ verify/ + README.md + LICENSE
- **消毒铁律落实**（双轮扫描 0 命中）：无 <PASSWORD> / sk-* / master_key 值 / SSH 私钥 / authorized_keys / 新 key 前缀 <KEY_PREFIX_OLD>；shim-deploy.sh 消毒版无 SUDO_PW/sudo -S（改 ssh sudo -n，README 说明 NOPASSWD 配置）；**无模型文件**（无 .safetensors/.gguf/.bin 模型类文件，仅 libnccl.so 58MB 属正常）
- **双库 MD5 与 verify/expected-md5s.txt 一致**：b7784b49885659c27765e648884e4edd（NCCL v3 双口）、ce43c688c5164ac7efd5105c94fdab77（shim v8）
- **已知偏差（已注明）**：① v8 shim 源码未归档，包内为 v3 基线源码 ncclpin.c（lib/README.md 注明，建议用预构建 v8 .so）；② registry 建仓脚本提取自部署指南

## 四、README（Docu 编写 → Tessa 验收 ❌9 处 → Docu 修订 → 终审 ✅）

- **终版**：617+ 行，15 章节（0 包清单 / 1 概览+拓扑 / 2 硬件接线 / 3 系统基线 / 4 免密自建 / 5 镜像仓库 / 6 补丁安装双路径 / 7 systemd+secrets 自建 / 8 NFS / 9 启动编排 / 10 验证验收 / 11 排障 14 条 / 12 安全清单 / 13 LICENSE）
- **验收修复**：9 处路径错位全部以包内实际为准修正；fstab 示例改"部署者自行 APPEND 模板"；PSR 期望值改"EngineCore 主要落 15-19（允许个别线程偏离）"
- **终审复核**（主理人执行）：旧错误路径残留=0、8 处抽检路径全存在、密钥终扫=0、重新打包后 1372 文件

## 五、独立验收结论（Tessa，fresh eyes）

| 段 | 判定 | 要点 |
|----|------|------|
| 服务状态 | ✅ | 8 项=7✅+1⚠️（PSR 个别线程 13），新/旧 key 行为全部实测符合预期 |
| 发布包 | ✅ | 9 项审计全过（密钥/模型/免密扫描、MD5、20 个 .sh 语法、结构完整性） |
| README | ❌→✅ | 9 处路径错位已修复，修订后自查+主理人终审通过 |

---

## ✅ 行动清单

| # | 行动 | 紧急度 | 说明 |
|---|------|--------|------|
| 1 | sudo 密码轮换（docs 明文已清，但口令本身未换，历史暴露面仍在） | **P0** | 与本次 API key 轮换同源，建议尽快窗口执行 |
| 2 | shim v8 绑核口径复核（EngineCore 偶发 PSR=13） | P2 | 观察性结论，不影响功能；后续 shim v9 时一并处理 |
| 3 | v8 shim 源码（libncclpin_v8.c）归档补录 | P2 | 当前仅 v3 基线在包内，生产 v8 源码应找回并归档 |
| 4 | 开源包发布前人工复核 LICENSE 声明与上游许可条款 | P1 | NCCL(BSD-style)/vLLM(Apache-2.0) 注记已附，正式开源前建议法务/合规复核 |
| 5 | 新服务器首次部署演练（按 README 从零走一遍） | P1 | 建议在一台备用机或全新环境做端到端验证后再对外发布 |

---

## ⚠️ 待完善 / 已知局限

- README 的部署流程基于生产环境实测（环网拓扑/版本/参数），但**尚未在全新硬件上端到端演练过**——发布前建议按行动清单 #5 演练一次。
- 包内 IP/网段为默认示例（README 已注明可调整）；现场自定义网段时需同步改 iptables 白名单与 hosts。
- 模型权重不在包内（按用户要求），部署者需自备权重并经 NFS 导出。
- 旧 key 仍存在于服务器历史备份目录（`backups/key-env-20260813/*.bak`），属运行环境内部文件、未进发布包；如后续清理需注意这些备份仍含旧凭据。

---

## 📚 数据来源 & 成员产出索引

- Cody：`_fix_20260813/key-env-refactor.md`（改造清单/diff/secrets 校验/待重启清单）
- Rex：`_fix_20260813/restart-and-openkit.md`（验证矩阵证据/打包清单/消毒扫描/偏差记录）
- Tessa：`_fix_20260813/tessa-acceptance.md`（A 8 项/B 9 项判定 + README 9 处修复建议）
- 发布物：`deliverables/dgxspark-tp4-open-kit-2026-08-13/`（tar.gz + 解包目录 + README）

---

> 本报告由工程保障团队 AI 协作生成（2026-08-13）。P0 项（sudo 密码轮换）与开源发布前的合规/演练请由人类工程负责人安排。
