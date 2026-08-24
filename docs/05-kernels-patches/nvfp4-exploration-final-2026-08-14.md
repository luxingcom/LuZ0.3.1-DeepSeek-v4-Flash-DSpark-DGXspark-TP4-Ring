# DGX Spark 四机 TP4 NVFP4 探索最终汇总报告

**日期**：2026-08-14 ｜ **状态**：正式收官归档 ｜ **生产决策**：维持 b12x + MXFP4 不变

---

## 1. 探索背景与目标

在 DGX Spark 四机 TP4 集群（vLLM 0.26.1.dev0 / SM121 / 环网 NCCL ring-only）上，系统性验证 **NVFP4 权重升级**（Rarri/DeepSeek-V4-Flash-0731-NVFP4，164G，MTP 全转）的性能价值，对比生产基线 **b12x + MXFP4**（156G）。共完成 **9 组实验、3 次测试窗口、2 次方法论修正**。

## 2. 实验矩阵与结论总览

| # | 路径/实验 | 结果 | 判定 |
|---|-----------|------|------|
| 1 | L1：cu130 + NVFP4 + marlin | prefill +1% / decode -13~15% | ❌ 证伪（后更正：env 为 no-op，未真测） |
| 2 | 真测 marlin（--moe-backend marlin） | decode 92.1/91.8，无 +16-20% | ❌ 无收益 |
| 3 | CUTLASS W4A4（TP4） | prefill 0.95-1.09× | 🟡 未达门槛（TP4 all-gather 摊薄） |
| 4 | cuBLASLt 3×（L2 cu132） | SM121 不支持 Grouped GEMM NVFP4 + vLLM 无后端 | ❌ 判死刑 |
| 5 | b12x + NVFP4 | oracle 拒绝（swiglu_limit 无 clamp） | 🔴 死路 |
| 6 | moe_ep（CuteDSL） | 三重拦截（枚举/swiglu/设备闸门） | 🔴 死路 |
| 7 | 通信-计算重叠补丁 | 长档 +5.7% / 短档 -5.5% | 🟡 功能验证通过，弃用 |
| 8 | 权重直接对比（生产 b12x 换 NVFP4） | oracle 拒绝；CUTLASS 备选成功 | 见 #9 |
| 9 | **三轨裁决 + 并发全矩阵** | **CUTLASS 最优但 decode -10-14%；b12x 并发全面略优** | ✅ **生产维持 b12x** |

## 3. 关键性能数据（最终裁决表）

### 3.1 三轨 c1（per-request p50，coding/greedy）

| 配置 | 131072 prefill | 131072 decode | 32768 prefill | 32768 decode | accLen |
|------|---------------|---------------|---------------|---------------|--------|
| **b12x+MXFP4（生产）** | 1896.4 | **104.1** | 2222.2 | **109.7** | 4.04 |
| CUTLASS+NVFP4 | **1955.9** | 96.3 | **2349.5** | 92.5 | 4.56 |
| Marlin+NVFP4 | 1826.8 | 92.1 | 2009.5 | 91.8 | 4.63 |

### 3.2 并发全景（b12x 生产，c1-c5 × 双 ctx）

| ctx | c1 | c3 | c4 | c5 |
|-----|-----|-----|-----|-----|
| 131072 prefill | 1896.4 | 732.63 | 635.96 | 595.53 |
| 131072 decode | 104.1 | 34.48 | 37.45 | **7.01** ⚠️ |
| 32768 prefill | 2222.2 | 847.2 | 692.13 | 678.68 |
| 32768 decode | 109.7 | 38.93 | 48.7 | **16.95** ⚠️ |

**并发崩塌点 = c5**（131072/c5 decode 7.01）——**平台共性**（TP4+长上下文+K 降窗+并发竞争，与后端无关；CUTLASS c5 同样 6.59）。

### 3.3 能耗/ALU（信息）

- prefill 段：P_avg 193.8W（四机合计）/ decode 段：163.3W / GPU util 87-90%（ALU 忙，计算在跑）

## 4. 核心发现与归因

1. **NVFP4 权重本身无损**：接受率 accLen 4.56-4.63（反超 MXFP4 的 4.04）、GSM8K 0.733 无退化、MTP 全转（cast_lossless_pct=100）——**权重质量不是瓶颈**
2. **decode -10-14% 为 CUTLASS 内核结构性差异**：大 M 设计瓦片跑 M=1~6 空转 + dequant 开销；b12x CuTe DSL 有 decode 专用微内核（小 M≤16 全融合）——**非配置可解**
3. **c5 并发 decode 崩塌是平台共性**（非 CUTLASS 特有）——F2 断崖同源（访存延迟主导）
4. **overlap 补丁破坏 dspark FULL graph 捕获**（decode 54 的根因，已坐实弃用）
5. **Marlin 无额外价值**（GB10 NVFP4 下与 CUTLASS 相当）
6. **cuBLASLt 3× 无未来**（SM121 不支持 + vLLM 0.27.1/main 均无后端）

## 5. 生产变更（本次执行）

| 变更 | 内容 | 状态 |
|------|------|------|
| **litellm 路由并发限制** | `default_max_parallel_requests: 12 → 5`（02 config.yaml，备份 .bak-maxconc5-20260814） | ✅ 已生效（重启 litellm-proxy，e2e 验证 OK） |
| 生产 TP4 | b12x + MXFP4 原状（未动） | ✅ 健康（/health 200） |

**路由限制依据**：并发测试显示 c5 为崩塌点（131072 decode 7.01），限制最大并发 5 可避免更高并发恶化（c4 仍有 37.45 良好表现）。

## 6. 资产清单（归档）

| 资产 | 位置 |
|------|------|
| NVFP4 权重（Rarri，164G） | 四机 /home/<USER>/models/...-nvfp4/（03/04 NFS） |
| CUTLASS 测试脚本 | 四机 <INSTALL_DIR>/scripts/start_tp4_*_cutlass.sh（未启用） |
| 重叠补丁 v3 | deliverables/engineering-assurance/_fix_20260813/overlap-patch/ |
| 测试报告 | combo-test-run-20260814.md / combo-acd-run-20260814.md / l2-build-plan-20260813.md（§D-§N） |
| 基准数据 | _tessa_tp4_bench/CONC_COMPARE/（c3/c4/c5 全并发） |
| 归因文档 | §F 通信量化 / §J 接受率归因 / §K decode 开销 / §L graph 修复 / §M CUTLASS vs B12x / §N 决策点 |

## 7. 未来方向（长期候选）

1. **vLLM 0.27 升级窗口**：上游 #4495（B12x Direct low-token MoE，M=1 1.668×）+ 量化 Markov heads（#50424）——NVFP4 decode 差距的可能收敛点
2. **flashinfer 升级**：0.6.15→0.6.17 无 decode 收益（#4253 性能中性），持续跟踪
3. **cuBLASLt 路线**：需 vLLM 上游支持（当前无在途 PR）
4. **路由并发策略**：c4 为 decode 质量边界（37-49 tok/s），可评估按 ctx 长度动态限流（>64K 限 c3）

## 8. 结论

**生产最优解确认：b12x + MXFP4（decode 104-110 tok/s 无可替代）。NVFP4 升级在 vLLM 0.26.1 / SM121 / TP4 生态下无生产价值**（唯一亮点 = c1 prefill +3%）。全部 9 组实验闭环，资产归档完毕，等待 vLLM 0.27 生态成熟后再评估。
