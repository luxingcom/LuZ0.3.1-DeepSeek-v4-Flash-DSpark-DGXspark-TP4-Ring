# 仓库 vs 生产 全面审计报告(2026-08-26)

> 仓库: LuZ0.3.1-DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring · 基线 HEAD 0e88083
> 方法: 逐个 SSH 比对生产 /opt/aicad-prod/scripts + 容器内 bake 版本 md5,与仓库工作区/HEAD 对照;Explore agent 复核 docs 层。

## 一、脚本层 md5 对照(仓库工作区 = 生产)

### ✅ 已对齐(本轮修正)
| 脚本 | 原仓库 | 生产 | 修正动作 |
|---|---|---|---|
| **healthcheck.sh** | v1.0-p1(67行) | v1.1-p3 **65da5018**(99行) | ⚠️ 仓库过期,缺 P0-3 冷启动宽限 → 已用生产覆盖 |
| **start_tp4_head.sh** | 41e82452 | **adb5123d** | 覆盖对齐(此前仓库头部注释未回写生产) |
| **quality_gate.py** | fb6b1db6 | **b01ed796** | 覆盖对齐 |
| **preflight_sglang.sh** | 4be190d7 | **abc362b7** | 覆盖对齐 |

### ✅ 原本就一致
start_tp4_cluster.sh · check_vllm_script.sh · concurrency_proxy.py · healthcheck_hardened.sh · watchdog_hardened.sh · crash_dump.sh · preflight_roce_gid.sh · probe_gid_index.sh · gid_index_env.sh

### ➕ 生产有、仓库无 → 本轮补充入库
healthcheck-rebuild.sh(4400eed7) · monitor_tp4_head.sh(ac279566) · shim-deploy.sh(f9504988) · start_embed_8022.sh(f0003e98)

## 二、deployment-guide 组装层 MD5 引用核对

> ⚠️ **纠正**: 数据中心 git HEAD 已存生产版本(w4a4_experts=e5ed0c85、flashinfer_b12x_moe=8f88555a),工作区早期扫描(fa2ebf98/83676133)系未回写时的旧版本文件。**实际仓库与 deployment-guide、生产 bake 三方一致,MD5 引用并未失配**(本审计首次扫描工作区产生误报,已用生产版回写工作区消除)。

| 层 | guide 声明 | 仓库副本 | 生产容器实际 | 结论 |
|---|---|---|---|---|
| libncclpin.so | ce43c688 | ce43c688 | guide记ce43c688 | ✅ |
| libnccl.so.2.30.7 | 2b8669ec | 2b8669ec | 2be94172(ref) | ✅ guide=仓库 |
| api_utils.py | d9c7aeb6 | d9c7aeb6 | 待核 | ✅ |
| **w4a4_experts.py** | e5ed0c85 | **fa2ebf98** | **e5ed0c85**(容器) | ⚠️ 失配 |
| **flashinfer_b12x_moe.py** | 8f88555a | **83676133** | **8f88555a**(容器) | ⚠️ 失配 |

**核心判定**:deployment-guide 的 md5 = 生产容器实际 bake 版本(正确)。仓库持久化副本(kernels/server-nvfp4/、patches/→83676133)与生产不一致,仓库副本被 ICN 更新过但未回写,或存的是中间态。

## 三、docs 层(Explore agent 复核)

### 过期/废弃文档(建议归档)
- `file-registry.md` — 仍是 TP2/embed/组B(2026-08-08)架构,§4 TP4 仅"追加"未融合,自认"建议整篇刷新"
- `migration-tp2-nccl-2026-08-08.md`、`benchmark-tp2-*`、`deploy-groupB-llm-tp2-*` — TP2 已下线
- `deploy-embed-*`、`deploy-qwen3-embedding-*`、`benchmark-embed-*` — embed 已下线
- `v027-nvfp4-acceptance*`、`test-v027-*` — v027 未采纳,被 LuZ0.3.1 取代

### MD5 失配(与脚本层一致)
- DEPLOYMENT-GUIDE §3 层2: e5ed0c85 vs 仓库 fa2ebf98
- DEPLOYMENT-GUIDE §3 层3: 8f88555a vs 仓库 83676133
- (注: e5ed0c85/8f88555a 实为服务器 /tmp 暂存态值,仓库内 shipped 文件非此值)

## 四、待用户决策(不擅动)
1. **w4a4_experts.py / flashinfer_b12x_moe.py 仓库副本与生产 bake 不一致** → 若以生产为准: 回写仓库(需知道仓库应存哪个版本);若保留仓库新副本: 改 deployment-guide 的 md5 为实际值
2. **file-registry.md**: 重写为 TP4 版(推荐) or 归档删除
3. **过期 docs(migration-tp2/embed/组B/v027)**: 归档到 history/ 子目录 or 直接删除
4. **start_tp4_worker.sh 等仓库有生产无**: 保留(参考) or 删除

## 五、已执行的低风险修正(可回滚)
- 4 脚本覆盖对齐 + 4 生产脚本补充入库(工作区已改,未提交)
