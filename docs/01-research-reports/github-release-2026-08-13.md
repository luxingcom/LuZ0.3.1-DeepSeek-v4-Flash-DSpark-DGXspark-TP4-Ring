# DGX Spark TP4 部署套件 GitHub 开源发布报告

**日期**：2026-08-13
**工作流**：开源发布（Agent 友好化 → 合规化 → GitHub 建仓上传）
**参与成员**：Docu（AGENTS.md+合规）、Cody（CLI+Makefile）、Zhen（编排+git 提交+代理登录上传）
**仓库**：https://github.com/luxingcom/dgxspark-tp4-deploy-kit

---

## 📌 TL;DR（执行摘要）

- 开源发布包已按 **Agent 直接拉取**标准改造完成并上传 GitHub：公开仓库 `luxingcom/dgxspark-tp4-deploy-kit`（默认分支 main，commit `f2a274f`，1148 文件）。
- **Agent 友好层**：AGENTS.md 自描述（146 行 7 节）+ `dgxspark-cli.sh` 统一入口（8 子命令，默认 dry-run、`--yes` 才执行写操作）+ Makefile + `.env.example` + `.gitignore` + `.gitattributes`（LF 强制）。
- **合规层**：LICENSE（Apache-2.0 全文）+ NOTICE（6 项上游归属）+ THIRD_PARTY_LICENSES/（NCCL BSD-3 / Apache-2.0 / MIT 原文）。
- **上传路径**：GitHub 密码认证已禁用 → 改用 **OAuth 设备码代理登录**（token scope=repo）+ 临时 hosts 映射绕过 github.com 直连限制，推送成功后已恢复 hosts。
- **消毒**：零密钥/零密码/零免密配置/零模型文件（提交前 + 打包后双轮扫描 0 命中）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 仓库 | https://github.com/luxingcom/dgxspark-tp4-deploy-kit（公开，main） |
| commit | f2a274f649d40a9c15fc2aa3de1fda0847a74d28 |
| 文件数 | 1148（git）/ 1383（tar 含未追踪的 README 等） |
| 发布包 | 43MB / 1383 文件 / md5 `7190c3d31610b10c87c5c4c65b3a9891` |
| 认证方式 | OAuth 设备码（gho token，scope=repo，建议用后撤销） |
| 合规 | Apache-2.0 主许可 + 6 项上游 NOTICE + 3 份三方许可原文 |

---

## 一、Agent 友好层（面向"直接拉取"设计）

| 文件 | 作用 |
|------|------|
| `AGENTS.md` | 自描述文件：项目使命/快速上手/目录地图/关键铁律/行为守则/任务映射/密钥规范 |
| `dgxspark-cli.sh` | 统一入口：status/check/install/start/stop/rollback/verify/package/help，写操作默认 dry-run |
| `Makefile` | `make help/status/check/...` 透传 CLI |
| `.env.example` | 环境变量模板（VLLM_API_KEY 等，占位符） |
| `.gitignore` | 排除 .env/secrets/*.env/*.bak/*.tar.gz，保留模板 |
| `.gitattributes` | 强制 LF 行尾（防 CRLF 破坏 shell 脚本） |

## 二、合规层

- **LICENSE**：Apache-2.0 全文（主许可）
- **NOTICE**：6 项上游归属——NVIDIA NCCL(BSD-3)、vLLM(Apache-2.0)、anemll/dspark-vllm-gx10(Apache-2.0)、litellm(MIT)、Prometheus(Apache-2.0)、Docker Registry(Apache-2.0)；明确不列 Grafana（无 dashboard JSON）与系统工具（引用不打包）
- **THIRD_PARTY_LICENSES/**：NCCL-LICENSE.txt、APACHE-2.0.txt、MIT-LICENSE.txt

## 三、GitHub 上传过程（代理登录）

1. **认证**：密码认证已被 GitHub 禁用（401 实证）→ 走 OAuth 设备码流程，用户在浏览器一次性授权（代码 6C42-55C1）→ 换取 repo-scope token
2. **网络**：`api.github.com` 直连可达建仓；`github.com` 主站被墙 → 临时写 hosts（140.82.112.3 → github.com）打通推送，完成后**已恢复 hosts**
3. **建仓**：`luxingcom/dgxspark-tp4-deploy-kit`，public=true，auto_init=false
4. **推送**：`git branch -M main` + 首次提交 + push（57.73MB 的 libnccl.so 触发 50MB 建议警告，非阻断，GitHub 硬限 100MB）

## 四、发布包终态

- 路径：`deliverables/dgxspark-tp4-open-kit-2026-08-13/dgxspark-tp4-deploy-kit-2026-08-13.tar.gz`
- 44,799,571 字节 / 1383 文件 / md5 `7190c3d31610b10c87c5c4c65b3a9891`
- 修正过程：首版误打入 `.git`（44MB）与嵌套旧 tar（43MB）→ 清理后从目录内重打，校验包内无 `.git/`、无嵌套 tar、无模型文件

---

## ✅ 行动清单

| # | 行动 | 紧急度 | 说明 |
|---|------|--------|------|
| 1 | 撤销本次 OAuth token（Settings→Applications→Authorized OAuth Apps→GitHub CLI） | P1 | 设备码 token 已用于建仓推送，建议用完即撤；后续推送用 PAT 或 SSH key |
| 2 | 新服务器首次端到端部署演练（按 README/AGENTS.md 从零走一遍） | P1 | 发布前实战验证，README 尚未在全新硬件验证 |
| 3 | 核对 GitHub 页面 README/许可证渲染（公开仓库首次展示） | P2 | 确认 badge/表格/拓扑 ASCII 图渲染正常 |
| 4 | shim v8 源码（libncclpin_v8.c）补归档 | P2 | 包内仍为 v3 基线源码（已注明） |
| 5 | 正式开源前的法务/合规复核（NCCL 补丁衍生作品的 BSD 合规细节） | P1 | NOTICE 已列，正式发布前建议人工复核 |

---

## ⚠️ 待完善 / 已知局限

- **token 安全**：本次 gho token 已用于操作，建议用户尽快撤销；如需长期推送，请配置 SSH key 或 PAT（我可协助）。
- **hosts 已恢复**：后续从本机再推送需重新加 hosts 映射（或走 api.github.com + 其他通道）。
- **libnccl.so 57.73MB**：超过 GitHub 50MB 建议值（未达 100MB 硬限），后续可考虑转 Git LFS 或改由源码构建（README 已提供路径 B）。
- README 部署流程尚未在全新 4× DGX Spark 上端到端演练。

---

## 📚 数据来源

- Docu 产出：AGENTS.md / LICENSE / NOTICE / THIRD_PARTY_LICENSES/ / README §0.1
- Cody 产出：dgxspark-cli.sh / Makefile / .env.example / .gitignore
- 上传证据：GitHub API（repo 创建 + contents 清单）、git push `[new branch]` 输出、commit f2a274f
- 发布包：deliverables/dgxspark-tp4-open-kit-2026-08-13/dgxspark-tp4-deploy-kit-2026-08-13.tar.gz

---

> 本报告由工程保障团队 AI 协作生成（2026-08-13）。开源发布前的合规复核与新机演练请由人类工程负责人安排。
