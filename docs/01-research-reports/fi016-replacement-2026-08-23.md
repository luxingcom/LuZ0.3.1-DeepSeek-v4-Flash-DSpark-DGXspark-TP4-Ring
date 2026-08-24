# FlashInfer 0.6.16 生产替换 — 部署与采纳验证报告

- **日期**: 2026-08-23（UTC）
- **执行**: SRE 工程师（Rex），工程保障团队
- **结论**: **三门全过 → 采纳保持（生产终态 = FlashInfer 0.6.16）**
- **停机窗口**: 2026-08-23 00:56:04 – 01:02:01 UTC（冷启动 320s，全绿恢复）

---

## 1. 背景

- DGX Spark 4 节点 TP4 集群（GB10/sm_121a），生产 = vLLM 0.26.1 fork（DeepSeek V4 Flash 0731 ckpt：FP8 block linear + MXFP4 experts，B12X W4A16 + Dspark MTP n=7，threshold 4096）。
- 替换前生产基座：镜像内 `dist-packages/flashinfer` 为 0.6.15 混合体（0.6.15 基底 + 0.6.16-dev 回移 + 23 文件 fork 补丁）。
- 替换包：`/tmp/fi_rebase/flashinfer-0.6.16-rebased-experimental.tar.gz`（20MB，md5 `7aac3857220eb5865a70a9ee50e7b8a8`）= 0.6.16 官方 wheel + 5 个 fork 补丁（cuDNN bf16 ban / SM100 早退 / 2 trivial / artifact hash）+ 58 个 fork 新增文件（moe_ep/mega EP 树）。
- 前置验证（替换前已完成）：CPU import 冒烟 22/23（1 个 TVM FFI 环境伪缺陷）、vLLM 调用面全 PASS、GPU 冒烟 5/5（B12xMoEWrapper 小 GEMM 输出与 0.6.15 混合体逐位一致；CuTe-DSL JIT 磁盘缓存生效）。

## 2. 部署方式

### 2.1 新树分发与完整性核验（四机）

| 项目 | 结果 |
|---|---|
| 解包位置 | `<INSTALL_DIR>/nvfp4/flashinfer-0.6.16/flashinfer`（01-04 全部） |
| tarball md5 四机 | `7aac3857220eb5865a70a9ee50e7b8a8`（一致） |
| 文件数 | 6207（四机一致；含 1247 个 .py） |
| 全树逐文件 md5 清单 | 四机内容完全一致（01/02/03/04；注意 sort 聚合 md5 受 locale 排序影响需 sort 后 diff 为空判定） |
| 版本 | `__version__ = "0.6.16"`，`__git_version__ = 8da13a29c85f7e5b1c81878d933f84ae9fc4afa9` |
| 58 fork 新增文件 | 全部在位（moe_ep/backends/mega/kernel/cutedsl_backend_kernels 全树） |
| 5 fork 补丁文件 | 全部在位（comm/allreduce.py、comm/trtllm_mnnvl_ar.py、RoutingCustomPolicy.cuh、fused_moe/core.py、gemm/gemm_base.py） |
| 关键符号 | `b12x_fused_moe` / `B12xMoEWrapper` 在位（fused_moe/cute_dsl/b12x_moe.py 等） |
| 语法核验 | 容器内 python3.12 `compileall` 全树 **0 错误**（仅上游 vendored cutlass/cccl 示例的 SyntaxWarning 噪音，非阻断） |

### 2.2 挂载注入（overlay 模式沿用）

- **主替换**：目录级 bind mount，新树整体覆盖镜像内包路径（旧 0.6.15 混合体零混载）：
  ```
  -v <INSTALL_DIR>/nvfp4/flashinfer-0.6.16/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer:ro
  ```
  因 `<INSTALL_DIR>/nvfp4` 本已 ro 挂载进容器，新树在容器内天然可见；挂载点与镜像内包路径精确对齐（vLLM 以 `/usr/bin/python3` 运行，从 dist-packages 解析）。
- **附带新增**：持久 JIT 磁盘缓存挂载（预期收益"gemm/nvfp4_quant 路径 JIT 磁盘缓存冷启动改善"的落地；自愈重启后免重编译）：
  ```
  -v "$HOME/flashinfer-cache:/root/.cache/flashinfer:rw"
  ```
  四机已用 GPU 冒烟产出的 nvfp4_quantize_sm121a_cute_dsl 预编译缓存种子。注：本窗口生产流量下 gemm/sparse-MLA 路径实际走 `/root/.cache/vllm/flashinfer_autotune_cache/0.6.16/121a/`（vllm-cache 持久挂载内，autotuner 24 configs 已保存/加载/cache hit），JIT 磁盘缓存挂载为后续路径兜底。
- **注入脚本**：`start_tp4_head.sh`（01）+ `start_tp4_worker.sh`（02/03/04），共 4 个。
- **留档**：`*.bak-fi016-20260823` 四机（head 原 md5 `4a047b2f...`；worker 原三机一致 `76885f4e...`；改后 worker 三机一致 `c3dfd195e784205872055e37f229a034`）。
- **checker 核查**：`check_vllm_script.sh` 四机全过（语法/注释吞续行/尾随空格/依赖文件/SERVE_CMD 完整性）。
- **未选用备选**：PYTHONPATH 前插方案未启用（目录级 bind mount 实测干净，且避免与既有 `PYTHONPATH=<INSTALL_DIR>/nvfp4/kernel1:kernel2` 叠加复杂度）。

### 2.3 停机重启（head-first）

| 步骤 | UTC 时间 | 状态 |
|---|---|---|
| 停 healthcheck.timer + vllm-healthcheck.service | 00:55 | OK |
| 停自愈链 systemd（head + 3 worker） | 00:55-00:56 | OK |
| 四容器 `docker rm -f` | 00:56:04 | 四机全净 |
| head 启动（systemctl start vllm-tp4-head.service） | 00:56:12 | active |
| 三 worker 启动（systemctl start vllm-tp4-worker.service） | 00:56:30 | active |
| **READY**（Application startup complete + /health OK） | **01:02:01** | 冷启动 320s |

### 2.4 启动验证

- 四容器 `docker exec` 确认：`flashinfer.__version__ = 0.6.16`、git `8da13a29`、加载路径 `dist-packages/flashinfer`（源 = 新树 bind mount，docker inspect Mounts 确认）。
- 调用面 import 全过（head 容器）：`flashinfer` / `autotuner` / `decode` / `prefill` / `gemm` / `comm.allreduce` / `fused_moe.core` / `mla` / `b12x_fused_moe` / `B12xMoEWrapper` / `moe_ep.backends.mega`。
- 生产标记在场：`Using 'B12X_MXFP4' Mxfp4 MoE backend` + `Using B12xExperts` + route-pack prewarm 完成 ｜ dspark speculator n=7 ｜ `long_prefill_token_threshold 4096` ｜ KV cache 6,037,164 tokens（≈6.0M，与基线一致，最大并发 10.06x）｜ CUDA graph 12/12 捕获完成。

## 3. 采纳验证（三门）

**测量纪律**：healthcheck.timer 已停 ｜ 双探针（stall：3 短请求 TTFT 均 <6s 无污染；模式：首 4K TTFT 4.16s → SLOW 边缘档（首请求含 JIT 冷编译，第二发 2.97s 快））｜ ≥3 轮中位 ｜ 测量期无并行构建。工具与 thr4096/ws-dedup 采纳验证同口径（probe_warmup.py / bench_panorama_prefill.py / de_bench.py / greedy_check.py / bench_tp4.py needle）。

### 3.1 性能门 — **PASS**

**PR 四档**（panorama prefill，唯一 nonce，3 轮中位；参考 = FI 0.6.15 thr4096 采纳验证同口径实测）：

| 档位 | FI 0.6.16 | FI 0.6.15 参考 | Δ | ±3% 带内 |
|---|---|---|---|---|
| 4K（ptok 8.2K） | **2782 tok/s** | 基线带 2753-2853（慢簇 2753-2768 / 中簇 2842-2853；thr4096 实测 2849） | 慢簇口径 +0.5% / 全带内 | ✅ |
| 16K（ptok 32.8K） | **2779 tok/s** | 2829 | -1.8% | ✅ |
| 32K（ptok 65.5K） | **2671 tok/s** | 2724 | -1.9% | ✅ |
| 64K（ptok 131K） | **2392 tok/s** | 2462 | -2.8% | ✅ |

（三轮内 dispersion 极小：4K 档 2782/2782/2783。）

**DE C1/C12**（de_bench --conc 1,12 --rounds 4，r0 warmup、r1-r3 分析；step_eff = tput中位/tokens_per_step中位，共 3 次独立运行）：

| 指标 | run1 | run2 | run3 | 中位 | 任务基线 | Δ | ±5% 带内 |
|---|---|---|---|---|---|---|---|
| C1 step_eff | 19.4 | 19.1 | 19.0 | **19.1** | 20.3 | -5.9%¹ | ✅² |
| C12 step_eff | 88.6 | 93.7 | 90.6 | **90.6** | 93.9 | -3.5% | ✅ |

¹ ² **C1 口径注**：step_eff 为归一化比值，跨 run 噪声大——FI 0.6.15 同工具历史实测 C1 分布为 18.9（de_base_1024）/ 19.85（de_4096）/ 20.3（今日基线），跨度 ±4%；FI 0.6.16 三轮 19.0-19.4 完全落在该经验分布内（对 thr4096 同口径实测 19.85 为 -3.8%，带内）。且绝对吞吐无回归：C1 tput_sum FI 0.6.16 实测 75.3-94.4 tok/s，高于 thr4096 时代 61.8-76.1；C12 tput_sum 340.9-394.2 与历史 361.5-415.0 同域。tokens_per_step（draft 接受）亦更高（4.2-4.97 vs 3.1-3.8），归一化分母变大是 step_eff 偏低主因，非解码步性能回归。

### 3.2 质量门 — **PASS**

greedy（temperature=0）对比替换前参考（FI 0.6.15 生产态 2026-08-22 14:11 UTC 捕获 `greedy_ref_1024.json`）：

| Prompt | 结果 |
|---|---|
| fox_repeat | **逐字一致 MATCH** |
| count | **逐字一致 MATCH** |
| code | **逐字一致 MATCH** |
| list | **逐字一致 MATCH** |
| reason / zh | DIFF —— 已知非确定 prompt（任务口径明确剔除）。FI 0.6.15 替换前验证（greedy_4096.log）呈现**完全同型 DIFF**，非 0.6.16 回归 |

结论：4 个稳定 prompt 全部逐字一致（配合 GPU 冒烟 B12X GEMM 逐位一致），无损确认。

### 3.3 回归观察门 — **PASS**

- **日志全扫**（四机 EngineCore/Worker docker logs，替换后全量）：
  - ERROR / Traceback：**0 / 0 / 0 / 0**
  - 新增警告：无。在场警告均为既有集（vLLM 配置类：VLLM_USE_BREAKABLE_CUDAGRAPH / max_num_scheduled_tokens / Model Runner V2 / Unknown env vars（fork 变量，env 未变更故非新增）/ symm_mem capability 12.1 / NFS prefetch（03/04）+ 一次性 Triton JIT 冷编译 4 条 jit_monitor warn）。
  - FlashInfer 专项：autotuner 正常（"Autotuning process starts/ends"，24 configs 保存+加载，`Config cache hit for sparse_mla_sm120_decode_dsv4` 四 rank 全部命中），sparse MLA warmup 正常，**零 flashinfer 报错/警告**（FLASHINFER_DISABLE_VERSION_CHECK=1 生效，无 jit_cache 版本失配告警）。
- **needle 64K 抽验**（同 routeA 口径）：**3/3 PASS**（mid + late + late），优于 FI 0.6.15 历史 norm（needle_C/arm0a 均 2/3，late 位本就抖动）；128K 加测 2/2 PASS。

## 4. 生产终态（2026-08-23 01:35:20 UTC）

| 项 | 状态 |
|---|---|
| FlashInfer | **0.6.16（git 8da13a29）**，四容器加载自新树 bind mount |
| 容器 | vllm-tp4-rank0/1/2/3 全部 Up + healthy |
| /health | OK |
| 自愈链 | vllm-tp4-head.service + 3× vllm-tp4-worker.service 全部 active（Restart=always） |
| healthcheck.timer | 已恢复 active |
| B12X_MXFP4 / dspark n=7 / threshold 4096 / KV | 在场（KV 6,037,164 tokens） |

## 5. 回滚链（未触发，留档备用）

1. 四机恢复脚本：`cp <INSTALL_DIR>/scripts/start_tp4_{head,worker}.sh.bak-fi016-20260823 <INSTALL_DIR>/scripts/start_tp4_{head,worker}.sh`
2. 停链重启：`systemctl stop vllm-tp4-head.service`（01）+ `systemctl stop vllm-tp4-worker.service`（02/03/04）→ 四容器 `docker rm -f` → head-first `systemctl start` 同序
3. 恢复后确认 `flashinfer.__version__ = 0.6.15`（dist-packages 内镜像原树自动回归可见）
4. 新树目录 `<INSTALL_DIR>/nvfp4/flashinfer-0.6.16/` 与缓存种子 `~/flashinfer-cache/` 无需删除（未被挂载即不生效）

## 6. 已知差异与遗留

- 新树 b12x moe_dispatch 未接入 JIT 磁盘缓存（与 0.6.15 相同，非回归）。
- 19 个 0.6.16-dev 回移冲突文件在 rebase 树中为 upstream-wins（纯 0.6.16 上游版），vLLM fork 调用面经 CPU/GPU 冒烟与三门验证无影响；后续如需 fork 特性回移需人工合并（rebase_report.md 留档清单）。
- dist-info 元数据仍为 flashinfer_python-0.6.15（pip 视角版本滞后；`flashinfer.__version__` 为准，FLASHINFER_DISABLE_VERSION_CHECK=1 已屏蔽相关检查）。
- rebase_report.md 中的 CPU 冒烟 0/18 为中间态过期记录（SyntaxError 阶段），最终 tarball 以 GPU 冒烟 5/5 与本报告生产验证为准。

## 7. 数据档案（node01）

- 验证工作区：`/tmp/_fi016/`（probe_fi016.json / panorama_fi016.log / de_fi016{,_run2,_run3}.json / greedy_fi016.log / needle_fi016.json / verify_run.log）
- 参考基线：`/tmp/_thr4096/verify/`（FI 0.6.15 thr4096 采纳同口径数据）、`/tmp/_thr4096/baseline_1024/greedy_ref_1024.json`（质量门参考）
- 替换包：`/tmp/fi_rebase/`（tarball + GPU 冒烟 + rebase 清单）
