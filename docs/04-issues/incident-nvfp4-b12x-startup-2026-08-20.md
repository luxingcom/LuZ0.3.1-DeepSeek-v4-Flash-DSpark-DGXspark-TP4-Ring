# 事故复盘报告：DGX Spark NVFP4 B12X MoE 启动事故

**日期**：2026-08-20
**工作流**：事故响应工作流 3（复盘）
**参与成员**：Rex（SRE 分诊与根因）、team-lead（systemd 受控恢复执行）
**SEV 评级**：**SEV1**（生产 4 rank 全挂，推理服务完全不可用）

---

## 📌 TL;DR（执行摘要）

- **现象**：8 月 20 日 03:27/03:31/03:32 三次启动，生产 `vllm-tp4-rank0~3` 4 rank 全部 `Exited(1)`，head 与 workers 同因崩溃，推理服务完全不可用，无法降级。
- **直接报错**：`ValueError: Mxfp4 MoE backend 'B12X_MXFP4' does not support the deployment configuration since kernel does not support current device cuda.`（Worker_TP0 于 `multiproc_executor.py:901`）。
- **根因（5 Why 收敛）**：**启动期 b12x cute/JIT 多 worker 并行加载竞态**——瞬时资源/时序问题，**非代码回归**。同一镜像与配置 08-19 06:33 曾稳定加载 `B12X_MXFP4` 并成功 prewarm，08-20 三次同批并行拉起时 `B12xExperts._supports_current_device()` 在 worker 进程返回 False。
- **处置**：**未 touch 任何生产文件、未回滚**（官方 v0.27.1 回滚锚点经评估会破坏 `flashinfer_b12x`→B12X 映射，禁用）。通过 `systemctl` 受控拉起 head+workers，**4 rank 全部 healthy、端到端推理通过、GPU 0%**，服务恢复。
- **关键教训**：健康检查/告警对"进程活着但引擎未就绪"盲区；B12X 这类 JIT 编译后端缺乏启动期隔离验证与竞态缓解。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 事故编号 | incident-nvfp4-b12x-startup-2026-08-20 |
| SEV 评级 | 🔴 **SEV1**（服务宕机、全 rank 受影响、无降级路径） |
| 影响范围 | 4 rank 全 down，推理完全不可用；GPU 空转 0% |
| 根因定性 | 启动期 b12x cute/JIT 多 worker 并行加载竞态（瞬时/时序），**非代码回归、非文件损坏** |
| 是否回滚 | ❌ 否（无文件可回滚；官方锚点禁用） |
| 恢复方式 | systemctl 受控单批拉起 head+workers |
| 当前状态 | ✅ 4 rank healthy、端到端推理通过、GPU 0% |
| 待办 | 5 条预防措施（见行动项） |

---

## 🚨 事故时间线

| 时间 (08-19/08-20) | 事件 |
|------|------|
| 08-19 06:32:49 | 生产 rank0(head) 启动请求（moe_backend=flashinfer_b12x, tp=4） |
| **08-19 06:33:37** | ✅ worker **成功**：`mxfp4.py:426 Using 'B12X_MXFP4' Mxfp4 MoE backend` |
| **08-19 06:35:02-06:35:11** | ✅ 成功：`Using B12xExperts` → `Prewarmed B12X route-pack ... on cuda:0 (experts=256, topk=6)`——**同一镜像/配置基线曾稳定运行** |
| 08-19 10:57 ~ 08-20 03:15 | kernel①/② 交付与 landing-export 活动（**均未部署到生产调用点**，与本次事故无因果） |
| **08-20 03:27:37** | ❌ 启动 #1：APIServer 起，Worker_TP0 03:29:01 报 B12X `current device cuda` 不支持 → EngineCore 失败 |
| **08-20 03:31:56** | ❌ 启动 #2：同因失败 |
| **08-20 03:32:05** | ❌ 启动 #3：同因失败 |
| 03:29 起 | 4 rank 全部 `Exited(1)`，head(rank0 / <NODE_IP>:25999) 与 workers(rank1/2/3) 同因崩溃，服务全 down |
| 03:27-03:29 间 | 关键旁证：worker 曾成功初始化 `cuda_communicator`（GPU/CUDA 环境正常），随后 `load_model` 阶段才在 B12X experts 选择处失败 |
| **08-20 ~12:xx** | ✅ 恢复：`systemctl start vllm-tp4-head.service`(01) + `vllm-tp4-worker.service`(02/04/03)，**4 rank 全部 healthy**，各 rank `Using 'B12X_MXFP4'`，NCCL world_size=4 就绪，DeepGEMM warmup 991/991，head 8001 绑定，**端到端冒烟通过**（`cluster recovered OK`，finish_reason=stop，tp4 fingerprint，prompt 10/completion 4 tokens），GPU 各 rank 0% |

---

## 📊 影响范围

- **直接受害面**：生产 4 rank 推理服务完全不可用（无降级、无单 rank 兜底），服务中断自 03:29 至 ~12:xx 恢复。
- **资源**：4 节点 GPU 在故障期间空转（进程退出后 0%，无泄漏占用）。
- **数据/配置**：**未** touch 任何生产文件、未回滚；权重/镜像未变更。kernel①/kernel② 交付包仍躺在交付目录，**未部署到生产调用点**。

---

## 🔍 根因分析（5 Why）

**触发面（报错层）**
- W1：`modular_kernel.py:547-548` `if not cls._supports_current_device(): return False, "kernel does not support current device cuda"` → `B12xExperts._supports_current_device()` 返回 **False**。
- W2：`b12x_mxfp4_moe.py:577-578` `_supports_current_device = is_cuda() and is_device_capability_family(120) and _has_b12x()`，三条件中至少一个在 worker 进程为 False。
- W3：已实测干净容器带 GPU 时三条件**全 True**（`is_cuda=True`、`fam120=True`、`_has_b12x()=True`），b12x 0.15.3 可 import，`cute_compile/*.o` 缓存完整存在；且 08-19 06:33 曾成功——**强指运行态差异，非代码/文件差异**。
- W4：故障批次为 4 rank **并行同批拉起**，worker 在 `load_model` 阶段于 experts 选择处触发 b12x cute/JIT 解析；多 worker 并发 + torch.compile/线程环境下 `b12x.integration.tp_moe` 的 import/JIT 解析出现**瞬时失败**（最可能 `_has_b12x()` 中 import 抛 ImportError）。
- **W5（系统性缺陷）**：**B12X（cute/JIT 编译型）后端在启动期缺乏隔离验证与足够的 warmup 竞态缓解**——引擎就绪判定未覆盖"编译型后端解析成功"这一前置，告警/健康检查也未覆盖。

**⊕ 反证（排除项，支撑"非回归"结论）**
- 运行时加载的 `mxfp4.py` 是**镜像内层**（md5 de16a8ca…），并非主理人初判的 patch-v026 版（md5 1f0b097a，且该文件**未被 bind 挂载**，仅 `tilelang.py` 被挂载）。
- 官方回滚锚点 `vllm-0.27.1/mxfp4.py` **不含 B12X_MXFP4**，回滚会破坏 `flashinfer_b12x` 映射→后端降级，**经评估禁用并弃用**。
- kernel② v17 / prefill kernel① 尚未部署到生产调用点，与本事故无因果。

---

## 🎯 行动项（事故闭环）

| # | 行动 | 优先级 | 负责角色 | 预期完成 |
|---|------|--------|---------|---------|
| 1 | **B12X 启动期隔离验证**：单 rank 先启动 head 并确认 `Using 'B12X_MXFP4'` + `Prewarmed route-pack` 后再启 worker（错峰），将「编译型后端解析成功」纳入启动 gate | P0 | SRE | 下周 |
| 2 | **warmup 竞态缓解**：对 `_has_b12x()` / b12x import 增加**重试**（如 N 次带退避）与失败日志捕获（`VLLM_LOGGING_LEVEL=DEBUG` 打印 ImportError 真因），避免瞬时 ImportError 直接判 False | P0 | SRE / BE | 下周 |
| 3 | **healthcheck 增强为 API 可达**：容器/进程探针从 `pgrep`/进程存活升级为**端口 + `/health` + 一次最小推理冒烟**可达才算 healthy | P0 | SRE | 2 周 |
| 4 | **告警规则补盲**：对「4 rank 任一 `Exited`」「head 8001 端口不可达」「引擎未就绪(无 `Using 'B12X_MXFP4'` 日志)」配置告警并验证链路 | P1 | SRE | 2 周 |
| 5 | **SEV1 预案演练**：补充本类"编译型后端并行加载竞态"的预案（错峰拉起 + 受控 systemctl 恢复），并演练一次全量恢复与冒烟 | P1 | SRE + 主理人 | 1 月 |

---

## 🛡️ 预防措施（防复发机制）

1. **启动顺序守护（B12X 隔离验证）**：head 加载 B12X 成功并 prewarm 完成后，才允许 workers 并行拉起；将「`Using 'B12X_MXFP4'` + route-pack prewarm」作为启动 gate，避免多 worker 同时撞编译型后端解析。
2. **JIT import 韧性**：b12x import 增加重试与超时；失败时输出 DEBUG 级 ImportError 根因（而非静默判 False 后抛泛化 `current device` 错误），使偶发竞态可重试自愈。
3. **健康检查语义升级**：由「进程存活」升级为「端口 + /health + 最小推理冒烟」，防止"进程在但引擎未就绪"盲区。
4. **告警闭环**：覆盖容器 Exited、8001 不可达、引擎未就绪三类信号，并做实链路验证（非仅规则配置）。
5. **SEV1 预案**：沉淀 systemd 受控拉起 + 错峰 + 冒烟的标准化恢复 SOP，并定期演练（含本次恢复路径回放）。

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 |
|---|------|---------|--------|
| 1 | B12X 启动期隔离验证（head 先行 + 错峰 workers） | SRE | P0 |
| 2 | b12x import 重试 + DEBUG 失败根因捕获 | SRE / BE | P0 |
| 3 | healthcheck 增强为 API 可达（端口+/health+冒烟） | SRE | P0 |
| 4 | 容器 Exited / 8001 / 引擎未就绪告警并验证链路 | SRE | P1 |
| 5 | SEV1 预案沉淀 + 全量恢复演练 | SRE + 主理人 | P1 |

---

## ⚠️ 待完善 / 已知局限

- 根因 W4（多 worker 并行下 b12x import 瞬时失败）为**高置信推断**，基于：干净容器全绿实测 + 08-19 成功/08-20 三次失败的同配置对照 + worker 已成功初始化 cuda communicator 却仅在校验 B12X 处失败的日志。**未捕获到当次失败的 ImportError 原始堆栈**（当时未开 `VLLM_LOGGING_LEVEL=DEBUG`），如需 100% 闭环需在开启 DEBUG 后复现。
- gateway/litellm 上游切换行为、故障期间是否出现用户请求超时/报错的具体数量，需在数据面侧取证（本次未采集）。
- 恢复时间 ~12:xx 为执行端报告口径，精确 RTO 尚未以时间戳定量记录。

## 📚 数据来源 & 成员产出索引

- **取证主机**：node01（head，<NODE_IP>），辅助 node01/03/04（worker）。
- **容器**：`vllm-tp4-rank0~3`，镜像 `dspark-vllm-gx10:0.2.1-v026.0`，vLLM 0.26.1.dev0+gd3d3b2cca，`kv-cache-dtype=nvfp4_ds_mla`，`moe-backend=flashinfer_b12x`，tp=4，`enforce_eager=False`（常量，非本次变更）。
- **Rex（SRE）分诊产出**：`container logs`（三次失败栈 + 08-19 成功 prewarm）、`mxfp4.py` / `modular_kernel.py` / `b12x_mxfp4_moe.py` 源码定位、干净容器带 GPU 全绿实测（`_has_b12x()/_supports_current_device()`）、文件时间戳与 md5 比对、backup 锚点评估。
- **关键证据行**：`mxfp4.py:290`（`"flashinfer_b12x": [B12X_MXFP4]`）、`mxfp4.py:599`（`_return_or_raise(B12X_MXFP4)` 无 fallback）、`modular_kernel.py:547-548`（current device 判 False）、`b12x_mxfp4_moe.py:577-578`（三条件）、日志 `08-19 06:33 Using 'B12X_MXFP4'` / `08-20 03:29 ValueError ... current device cuda`。

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。