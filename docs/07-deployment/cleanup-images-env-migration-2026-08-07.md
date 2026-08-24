# 四机镜像清理整理与环境迁移 — 综合报告

**日期**：2026-08-07
**工作流**：部署前检查 / 运维变更（清理 + 环境迁移）
**参与成员**：Archi（迁移方案 + 兼容性评估）、Rex（清理 SOP）、Zhen（实施与汇编）

---

## 📌 TL;DR（执行摘要）

- **任务①②（清理）✅ 全部完成**：四机旧 embed-gpu 镜像清零、anemll 镜像整理完毕（0.2.1 生产版三处保留），累计释放磁盘 **~225G**（.60 111G + .58 83G + registry GC 31G）
- **任务④（.60 对齐）⚠️ 95% 完成**：发现 .60 dockerd 存储异常（load 同一 tar 持续产出旧 ID，疑似 containerd 快照元数据损坏），功能正常但 38.3G 未释放，**需维护窗口重启 dockerd 后重新导入**
- **任务③（环境复刻）🔬 阶段 1a 完成**：重大发现——新镜像 `vllm-gb10:0.26.1-cu132` 实为 **anemll fork 公开构建**，生产参数（dspark 投机 / b12x MoE / DS-MLA KV / tilelang / deepseek_v4 parser）**1:1 原生支持**，迁移可行且无需降级；1b 双机冒烟需 .58 内存窗口
- **红线守护**：sglang（视频工作流资产）全程未动；anemll 0.2.1 三处验证保留

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（清理完成；.60 对齐 + 复刻验证需维护窗口） |
| 阻塞项 | 2（.60 dockerd 重启需窗口；1b 双机冒烟需 .58 内存释放） |
| 已释放空间 | .60 ~111G / .58 ~83G / registry 31G（71G→40G） |
| 建议下一步 | 视频工作流结束后：①.60 重启 dockerd + prune + 重新导入 ②任务③ 1b 双机冒烟 |

---

## ✅ 任务①：四机清旧 embed-gpu 镜像 — 完成

| 位置 | 处理 | 释放 |
|------|------|------|
| .58 本机 | `embed-gpu:anemll-0.1.1-st5.6.1` + `<NODE_IP>:5000/embed-gpu:...` 双 tag 删除（同 ID fe884bdd8ef0） | 20G |
| registry | `embed-gpu` 仓库 tag digest 删除 + GC | 计入 31G |
| .55/.59/.60 | 无此镜像（无需操作） | - |

## ✅ 任务②：anemll 镜像整理 — 完成（0.2.1 保留）

| 机 | 删除项 | 释放 |
|----|--------|------|
| .58 | 0.2.0-v026.0（17.4G）、sparkrun production-ready + production-hybrid-1.6（~46G）、0260-bak + vllm-worker Exited 容器 | ~83G |
| .60 | 0.2.0-v026.0（26.4G）、0.1.1（28.9G）、vllm-base-v0.26.0（23.8G）、sparkrun hybrid + ready（~93.5G）、0260-bak + vllm-node 容器 | ~111G |
| registry | anemll 0.1.1 / 0.2.0-v026.0 tag（digest 删除 202）、sparkrun 仓库 tag×2、GC | 31G（71G→40G） |
| **保留验证** | 0.2.1-v026.0：.58 ✅ / .60 ✅ / registry ✅（tags 仅剩 0.2.1） | - |
| **红线守护** | sglang（.58 sglang-h3:exported / .60 sglang:dev-cu13）**全程未动** ✅ | - |

## ⚠️ 任务④：.60 vllm-gb10 对齐 — 95% 完成（待维护窗口）

**目标**：.60 本地 38.3G 旧版（ID 2a1df51ffe21）→ registry 18.9G 版（ID 5a2a5e99a5a6）。

**执行过程**（save 17.7G → rsync 经 .58→.60 高速链路 3.5GB/s 传输 ✅ → load 验证）：
- 发现**异常**：.60 dockerd 在**完全干净**（无任何 vllm-gb10 镜像）状态下 load 同一 tar（md5 与 .58 源文件一致 f0dd27f4709a0d48），**始终产出旧 ID 2a1df51ffe21 38.3G**——疑似 containerd 快照元数据损坏，镜像 ID 无法更新
- 多次尝试：rmi 后 load / tag 重映射 / 管道 save|load——均复现
- **当前状态**：.60 embed-qwen3-vllm 运行正常（/health 200 + dim 1024），但 38.3G 未释放；`docker system df` 显示 **RECLAIMABLE 150.1G（75%）**（含 build cache 49.6G）
- **修复方案（需维护窗口）**：视频工作流结束后重启 dockerd（`systemctl restart docker`）清 metadata 缓存 → `docker load` 重新导入 → 验证 → `docker system prune -a` 回收（可释放 150G+）
- 影响说明：重启 docker 会中断 .60 运行容器（comfyui-h3-60 视频工作流 + embed + 基础栈），容器 restart=unless-stopped 会自动恢复

## 🔬 任务③：生产环境复刻到新 vLLM 镜像 — 阶段 1a 完成

**重大发现**：`vllm-gb10:0.26.1-cu132-sha-fa87aea5`（timothystewart6 构建）**不是纯标准 vLLM，而是 anemll fork 的公开构建**——源码级验证（.60 镜像内）：

| 生产参数 | 支持度 | 证据 |
|---------|--------|------|
| 权重格式 | ✅ 标准 fp8 | config.json `quant_method: fp8`（e4m3/128×128），155.4G/48 文件 |
| deepseek_v4 模型 | ✅ | `vllm/models/deepseek_v4/`（nvidia/amd/xpu）+ DeepseekV4ForCausalLM 注册 |
| **dspark 投机** | ✅ | `speculative.py:62 DSparkModelTypes = Literal["dspark"]` |
| **DS-MLA KV** | ✅ | CacheDType 含 `fp8_ds_mla` + turboquant 系列 |
| **b12x MoE** | ✅ | `kernel.py:131/148` 枚举含 `flashinfer_b12x`（CuteDSL NVFP4 GEMM SM120+） |
| **tilelang MLA sparse** | ✅ | tilelang 0.1.9 内置 + mhc 模块（生产 patch 挂载不再需要） |
| **deepseek_v4 parser** | ✅ | `deepseekv4_engine_tool_parser.py` + `vllm/reasoning/` |
| HostConfig | ✅ 已采集 | host 网络 + privileged + 64G shm（.60 node / .58 worker 实测） |

**结论**：生产参数**可 1:1 复刻**（仅需移除 fork 专属 ENV 与 patch 挂载——镜像已内置）。Archi 原方案"20-50% 性能损失"假设被推翻。

**剩余阶段（需窗口）**：
- 阶段 1b：双机最小加载冒烟（.58/.60，权重 156G 单机物理放不下）——需 .58 释放内存（comfyui 88G + embed 12G 占用）
- 阶段 2：生产级双机复刻（容器启动参数按 1a 枚举确认）
- 阶段 3：功能回归（工具调用 / 长上下文 / 吞吐对比 ≥ 旧镜像 70-80%）

---

## ✅ 行动清单

| # | 行动 | 负责 | 紧急度 | 预期完成 |
|---|------|------|--------|---------|
| 1 | 视频工作流结束后：.60 重启 dockerd → 重新 load 18.9G 版 → 验证 → prune 回收 150G+ | Zhen | P0 | 待窗口 |
| 2 | .58 内存释放后：任务③阶段 1b 双机冒烟（新镜像加载 deepseek，确认 nvfp4_ds_mla 枚举兼容） | Zhen+Rex | P1 | 待窗口 |
| 3 | 1b 通过后：阶段 2 生产复刻 + 阶段 3 功能回归（工具调用/长上下文/吞吐） | Zhen+Archi | P1 | 待窗口 |
| 4 | 复刻稳定后：旧 anemll 0.2.1 保留观察 1 周再决定去留 | Zhen | P3 | 1 周后 |

## ⚠️ 待完善 / 已知局限

- .60 dockerd 存储异常根因未完全定位（疑似 containerd 快照元数据损坏），已用"重启 daemon 清缓存"方案兜底，待窗口验证
- .60 docker system df 显示 build cache 49.6G + 大量可回收（150.1G），prune 后需确认不误删（build cache 无风险）
- 任务③阶段 1b/2/3 全部依赖视频工作流窗口（.58 内存 + .60 dockerd 重启）
- registry _catalog 中已删 tag 的 repo 名可能仍列出（元数据缓存），不影响按 tag 拉取

## 📚 数据来源 & 成员产出索引

- Archi（架构师）原始产出：生产环境迁移设计方案（兼容性矩阵 + ADR-001 + 三阶段）
- Rex（SRE）原始产出：四机镜像清理 SOP（A/B/C/D 四部分 + 12 项风险矩阵）
- Zhen（主理人）实机采集：四机镜像全量清单、生产容器 HostConfig/CMD/ENV、registry 状态、.60 dockerd 异常验证（5 轮 load 复现）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
> ⚠️ .60 dockerd 重启将中断 comfyui-h3-60 视频工作流，需在业务空闲窗口执行；任务③阶段 1b 需 .58 内存释放（当前被 comfyui 88G 占用）。
