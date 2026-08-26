# 仓库 vs 生产 审计(2026-08-26) — 脚本层第一轮

## 一、脚本 md5 对照(仓库工作区 = 生产)

| 脚本 | 仓库 md5 | 生产 md5 | 状态 |
|---|---|---|---|
| start_tp4_head.sh | adb5123d | adb5123d | ✅ 已对齐(用生产覆盖) |
| start_tp4_cluster.sh | 00b231d9 | 00b231d9 | ✅ 一致 |
| check_vllm_script.sh | 875ac21b | 875ac21b | ✅ 一致 |
| concurrency_proxy.py | c9e3b4db | c9e3b4db | ✅ 一致 |
| healthcheck.sh | 65da5018 | 65da5018 | ✅ 已对齐(**原仓库 v1.0-p1 过期→生产 v1.1-p3** 含 P0-3 冷启动宽限) |
| healthcheck_hardened.sh | a0bedd79 | a0bedd79 | ✅ 一致 |
| watchdog_hardened.sh | ff50fed0 | ff50fed0 | ✅ 一致 |
| crash_dump.sh | 13ff68c7 | 13ff68c7 | ✅ 一致 |
| preflight_roce_gid.sh | df747a3c | df747a3c | ✅ 一致 |
| probe_gid_index.sh | 4530e9a0 | 4530e9a0 | ✅ 一致 |
| gid_index_env.sh | 863b3f4c | 863b3f4c | ✅ 一致 |
| quality_gate.py | b01ed796 | b01ed796 | ✅ 已对齐(原 fb6b1db6) |
| preflight_sglang.sh | abc362b7 | abc362b7 | ✅ 已对齐(原 4be190d7) |

## 二、deployment-guide 组装层 MD5 引用核对

| 层 | guide 声明 | 仓库副本实际 | 生产容器实际 | 结论 |
|---|---|---|---|---|
| libncclpin.so | ce43c688 | ce43c688 | (生产 ref 未知, guide 记) | ✅ 一致 |
| libnccl.so.2.30.7 | 2b8669ec | 2b8669ec | 2be94172(ref) | ✅ guide=仓库一致 |
| w4a4_experts.py | e5ed0c85 | fa2ebf98 | **e5ed0c85**(容器内) | ⚠️ **仓库副本 ≠ guide/生产** |
| flashinfer_b12x_moe.py | 8f88555a | 83676133 | **8f88555a**(容器内) | ⚠️ **仓库副本 ≠ guide/生产** |
| api_utils.py | d9c7aeb6 | d9c7aeb6 | (待核) | ✅ 一致 |

> 关键发现:deployment-guide 记录的 MD5 与**生产容器实际 bake 版本一致**。仓库里存放的持久化副本(kernels/server-nvfp4/...、patches/...)的 md5 与生产不一致——仓库副本可能是更新(实验)版,或同步滞后。**需决策以哪方为准**。

## 三、待决策项
1. **w4a4_experts.py / flashinfer_b12x_moe.py 仓库副本与生产 bake 版本不一致** → 以生产为准回写仓库,还是保留仓库的更名实验版?
2. **目录结构**:生产平铺 /opt/aicad-prod/scripts/;仓库分 server-production/ + hardening/。是否要按生产平铺重整,还是保持归档分类?
3. **仓库有、生产无的文件**:start_tp4_worker.sh(生产仅partial在backup)、bench_v12_real.py、collect_mem*.sh、start_groupB 系列等旧架构脚本 → 过期待删?
4. **生产有、仓库无的脚本**:healthcheck-rebuild.sh、monitor_tp4_head.sh、shim-deploy.sh、start_embed_8022.sh 需补充入库?
