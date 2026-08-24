# TP4-Opt 五阶段执行计划 · 最终交付报告

**日期**：2026-08-13
**工作流**：性能优化执行（调研→核实→补丁→归因→调优→回滚闭环）
**参与成员**：Rex（SRE·切换/回滚/插桩）、Cody-2（补丁件/兼容验证/插桩设计）、Tessa（A/B 与归因执行）、Zhen（编排/裁决/终审）
**最终状态**：🟢 生产已回滚原镜像 `0.2.1-v026.0`，验证矩阵 7/7 全过，集群健康

---

## 📌 TL;DR

- **五阶段执行计划全部闭环**：Issue #22 补丁（Phase 1）→ P-ISSUE22B 兼容验证 → F2 断崖四归因 → F3~F7 配置级 A/B → 生产回滚。
- **关键认知修正**：活跃注意力后端实为 `FLASHINFER_MLA_SPARSE_DSV4`（模型专用路径），**nvfp4_ds_mla 一直在走 FlashInfer SM120 原生快内核**——Issue #22 补丁打在了非活跃文件（flashmla_sparse.py，通用后端），**无运行效果**（A/B 无收益即此因）；P-ISSUE22B"解锁"对本集群同样无效果。
- **F2 归因定论**：131072×c5 断崖 = **访存延迟主导**（每步串行遍历 4.4GB MLA KV，步长 0.36-0.48s 为带宽下界 50-65×）+ **深位草稿接受率崩落次级共因**（acc_len 2.83，pos3/4 接受率 <16%）；preemption 与调度饥饿双排除。
- **F3~F7 结论：配置零变更**——batch 8192（引擎崩溃）、long-prefill 2048（decode -81%）、num_spec 降深（block_size=5 硬约束+启动挂起）全部回滚；**F5 prefix caching PASS（64K 共享前缀 TTFT 78×）**、F7 CUDA Graph 覆盖确认。
- **两起测试隐患已闭环**：①`num_spec=2 < dspark_block_size=5` 非法配置导致全链崩溃（代码注释实锤"产出乱码"）；②head/worker speculative 配置不一致引发 shm_broadcast 初始化死锁——均已在测试中修复并沉淀为纪律教训。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 完成（生产已恢复原状，无任何测试配置转生产） |
| 生产状态 | 原镜像 0.2.1-v026.0（head e100ddad568a / worker 9ea563a724d4），补丁行已还原 |
| 性能结论 | 配置零变更；F5 业务建议保留（共享前缀 TTFT 78×） |
| 归因结论 | 断崖 = 访存延迟主导 + 深位草稿次级共因 |
| 测试资产 | issue22fix/f2inst 镜像 + F2 探针 + 各备份留档（不转生产） |
| 遗留 | cudagraph_metrics 观测开启（P2）、shim v8 源码归档（P2） |

---

## 一、执行阶段总览

### Phase 1：Issue #22 补丁（✅ 完成，裁定 TEST_ONLY_NOT_PROMOTED）

| 步骤 | 结果 |
|------|------|
| 生产快照 | 集群空闲确认；发现镜像双变体（head 34.2G / worker 21.6G 同名不同内容，worker 不在 registry→docker save 备份） |
| 补丁镜像构建 | `-issue22fix-head`（01）+ `-issue22fix-worker`（02/03/04），补丁行验证+SYNTAX OK |
| 生产切换 | 纪律停机/head-first 拉起，验证矩阵 7/7（含旧 key 401 确认轮换生效） |
| A/B 验证 | **无显著改善**（prefill -1~2%、decode -2~6% 统计不显著、400K 稳定）；裁定 🟡有条件固化 → 用户裁决 TEST_ONLY |

### P-ISSUE22B：NVFP4 更快路径排查（✅ 完成，结论：无需解锁）

- **重大认知修正**：引擎日志 + 代码级证据确认活跃后端 = `FLASHINFER_MLA_SPARSE_DSV4`（DeepseekV4FlashInferSM120Attention，模型专用），**nvfp4_ds_mla 已在走 FlashInfer SM120 原生稀疏 MLA 快内核**（584 字节 padded 布局，实际内容 FP8 e4m3，与内核期望逐字节一致）
- Issue #22 补丁（flashmla_sparse.py:861）**不在 DSV4 运行路径** → 零运行效果 → A/B 无收益的完整解释
- flashinfer_mla_sparse_sm120.py 的 dtype 硬限制只作用于通用非 DSV4 模型 → 补丁件保留为卫生修复，A/B 取消
- 结论："针对 NVFP4 的更快的路径" = **当前已在用的 FlashInfer SM120 DSV4 内核**；真 416 字节原生布局社区未解决，不可达

### F2：131072×c5 断崖四因素归因（✅ 完成）

| 因素 | 结论 | 证据 |
|------|------|------|
| ① preemption | ❌ 排除 | num_preemptions_total=0（排队 reason=capacity） |
| ② 访存延迟 | ✅ **主导** | 断崖段 ~530-600 GB/s（未饱和），步长 0.36-0.48s = 带宽下界 50-65×，每步串行遍历 5×878MB=4.4GB KV |
| ③ 深位草稿崩落 | ✅ **次级共因** | acc_rate 0.85 高，但 acc_len 2.83（vs c1 3.96）；pos3/4 接受率 76%→16%/69%→8% |
| ④ 调度饥饿 | ❌ 排除 | 断崖段 running_decode 恒 ≥3，纯 decode 步频 ~20/s 无节流 |

- 生产指引维持：>64K ctx ≤c1、>32K ≤c3
- 新发现指导：131K×高并发可评估降 num_spec——但被 dspark block_size=5 硬约束否决（见隐患 1），实际不可行

### F3~F7：配置级逐项 A/B（✅ 完成，配置零变更）

| 项 | 测试 | 裁定 |
|----|------|------|
| F6 num_spec | 顶层 3 被校验拒绝（≥block 5）；per_batch→2 / 移除 spec 均启动挂起 | KEEP 基线（5/4/3 动态） |
| F3 batch 4096→8192 | c5 全 10 请求 HTTP 500 + rank0 崩溃；c1 prefill 回退 | ROLLBACK 4096 |
| F4 long-prefill 1024→2048 | c3 decode 崩落 8.57（-81%）；TTFT 假性提升 | ROLLBACK 1024 |
| F5 prefix caching | **PASS**：64K 共享前缀冷 26.54s→热 0.34s（**78×**） | ✅ 业务建议写入运营文档 |
| F7 CUDA Graph | dspark 专属 capture 12/12 覆盖 spec 批次（c1:6/c5:20 tok） | ✅ 确认；cudagraph_metrics 待开启 |

### 生产回滚（✅ 完成，验证 7/7）

- 四机容器回原镜像 `0.2.1-v026.0`；**补丁行已还原**（`== "fp8_ds_mla"`，回滚彻底）
- 验证矩阵：/health 200、/v1/models 200（max_model_len=400000）、镜像确认、补丁行还原、--failed 空、litellm e2e 200、冒烟 200-token
- 回滚资产留档：四机原镜像 + 02 tar 21.7GB + 脚本备份 .bak-rollback

---

## 二、两起测试隐患复盘（闭环）

### 隐患 1：num_spec < dspark_block_size 非法配置
- **现场**：动态K 表 `[[1,1,5],[2,4,4],[5,6,2]]`（表内 batch≥5 档=2 < block=5）→ 07:43 全链崩溃（head+worker 全 exit 1）
- **根因**（vllm/config/speculative.py:1005-1025 代码注释实锤）：DSpark 是 block 式草稿，num_spec < block_size 产出**乱码而非降速**；顶层校验 raise，但动态K 表内值不经过该校验 → 运行期错误
- **处置**：撤回为 `[[5,6,3]]`；F6 修订为仅测 ≥5 档位；教训：动态K 表内所有值必须 ≥ block_size

### 隐患 2：shm_broadcast 初始化死锁（"严重缓存问题"）
- **现场**：8001=000、`Application startup complete` 从未出现、EngineCore "No available shared memory broadcast block found in 60 seconds"×N
- **根因**：Tessa 改动移除了 head 脚本 speculative-config 行 → head（无投机）与 worker（有投机）**配置不一致** → 引擎初始化广播阶段死锁（非资源型问题，/dev/shm 仅 1% 占用）
- **处置**：恢复 spec 配置 + 清理 /dev/shm IPC 残留（psm_*/sem.mp-*）+ 重启 → 集群健康、capture 10s
- **教训**：改脚本必须"备份 + 只改目标参数 + bash -n + 四机配置一致性检查（head/worker spec 配置必须同步）"

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | F5 业务产出落地：结构化共享前缀（system prompt/RAG ≥32K）写入运营文档与 Runbook | 运维/文档 | P1 | 本周 |
| 2 | 下次重启窗口开启 cudagraph_metrics 观测（F7 后续） | 运维 | P2 | 下次窗口 |
| 3 | shim v8 源码（libncclpin_v8.c）补归档；ring-only v3 diff 回填 Runbook | 运维 | P2 | 本周 |
| 4 | 台账更新：P-ISSUE22-001=TEST_ONLY_NOT_PROMOTED（已）、P-ISSUE22B-002=NO_RUNTIME_EFFECT_ON_DSV4、P-ISSUE22B-003 建议保留 | 文档 | P2 | 本周 |
| 5 | 测试纪律沉淀：动态K 表值 ≥ block_size、head/worker spec 配置一致性检查纳入 check_vllm_script.sh | 代码审查 | P1 | 本周 |
| 6 | 生产长期遗留：sudo 密码轮换（P0 仍未动）、registry 认证、.local-backup 裁决 | 运维 | P0/P1 | 待窗口 |

---

## ⚠️ 待完善 / 已知局限

- **F2 归因的②③为共因叠加**：未做单因素隔离（如禁 draft 下测带宽），但①②④排除证据完整、③有逐位接受率数据支撑，结论置信度足够
- F5 的 78× 为共享前缀+热命中理想场景；实际业务收益取决于调用方 prompt 结构化程度
- 测试期间发现的 F3（batch 8192 崩溃）未深挖根因（已回滚保护），若未来需大 batch 需单独立项
- cudagraph_metrics 未开启，F7 的覆盖确认基于日志证据（capture 计数），运行期命中率待观测

---

## 📚 数据来源 & 成员产出索引

- Rex：`phase1-issue22-snapshot-image.md`、`phase2-issue22-switch.md`、`f2-instrument-deploy.md`、`emergency-revert-dynk-capture.md`、`emergency-recovery-shmbroadcast.md`、`prod-rollback-20260813.md`
- Cody-2：`issue22-patch.md`、`issue22b-patch.md`、`f2-attribution-prep.md`、`f2-instrument.patch`
- Tessa：`tessa-issue22-ab.md`、`f2-attribution-run.md`、`f3-f7-ab-series.md`
- 回滚资产：`rollback-anchor-issue22.md`（RA-ISSUE22-001）、02 `prod-worker-image-0.2.1-v026.0.tar`（21.7GB）
- 决策依据：`delivery/aicoding-arch-2026-08-13/交叉验证与执行计划-2026-08-13.md`（五阶段计划）

---

> 本报告由工程保障团队 AI 协作生成（2026-08-13）。生产已恢复原状，测试产物留档；关键长期项（sudo 轮换等）请人类工程负责人安排。
