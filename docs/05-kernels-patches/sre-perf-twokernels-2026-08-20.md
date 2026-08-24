# SRE 性能测试报告：两算子对总体推理性能影响

> 执行人：雷克斯（SRE 工程师）| 日期：2026-08-20 | 生产集群 DGX Spark 4 节点
> 文档：`deliverables/engineering-assurance/sre-perf-twokernels-2026-08-20.md`
> 镜像：生产 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（非 v0.27）

---

## 0. 执行摘要（TL;DR）

| 算子 | 方案 | 实测性能 | 对照基线 | 提速 | 判定 |
|------|------|---------|---------|------|------|
| **kernel①** prefill GEMM（4W4A） | **路线A**（vLLM 内置 `cutlass_scaled_fp4_mm` 原生 FP4） | **84~131 TFLOPS** | v15（Triton bf16）9.6~21.3 TFLOPS | **≈5.5~10×（均值 6.7×）** | ✅ 显著提升 |
| **kernel②** KV-Linear（DS-MLA） | **v17** | **189~278 GB/s** | v11（53~61 GB/s） | **3.4~4.8×** | ✅ 与既有定论一致 |

**关键结论**：两算子（kernel① 路线A + kernel② v17）均在生产镜像 `0.2.1-v026.0` 环境/依赖下**可用且达到预期性能**，相对各自回退实现（v15 / v11）有 3.4~10× 算子级提速。

**端到端 serve 10 负载 bench 未能执行**（见 §6）：单节点（1×GB10，~121GB 内存）无法承载 DeepseekV4 DS-MLA 模型（48 shards，生产 TP4 跨 4 节点载入）。已交付设计好的 10 负载清单（§7），待 TP4 恢复后按清单执行。

---

## 1. 环境确认（step 1，已完成）

- 4 rank 容器全部**生产 serve**、healthy 20h+、镜像 `dspark-vllm-gx10:0.2.1-v026.0`（非 v0.27），与授权一致。
- 连接采用 SSH 别名 `node01-04`（<NODE_IP>-189，serve master-addr 所在网络），10.20.x 旧路由不可达。
- rank0 内 serve 进程确认：port 8001、host 网络、`--moe-backend flashinfer_b12x --speculative dspark/7 tokens --gpu-memory-utilization 0.80 --max-model-len 600000 --tensor-parallel-size 4`。
- 自愈（monitor/timer）确认关闭状态，本轮未改动。

**容器/镜像快照（恢复依据）**：

| 节点 | IP | 容器名 | 角色 | 容器 ID 前缀 |
|------|-----|--------|------|-------------|
| 01 | <NODE_IP> | vllm-tp4-rank0 | rank0(head) | 893afb6999fb |
| 02 | <NODE_IP> | vllm-tp4-rank1 | rank1 | 5cdbcb4ba4db |
| 03 | <NODE_IP> | vllm-tp4-rank3 | rank3 | 5d2038c78a55 |
| 04 | <NODE_IP> | vllm-tp4-rank2 | rank2 | d4a626b0bac3 |

---

## 2. 卸载生产腾内存（step 2，已执行）

按用户②⑨授权「测试需卸载生产环境腾出内存」，`docker stop` 4 个 rank（只停不删，保留可 start 恢复）：

| 节点 | 容器 | 停止结果 |
|------|------|---------|
| 01 | vllm-tp4-rank0 | rc=0 → exited |
| 02 | vllm-tp4-rank1 | rc=0 → exited |
| 04 | vllm-tp4-rank2 | rc=0 → exited |
| 03 | vllm-tp4-rank3 | rc=0 → exited |

**内存释放（available）**：所有节点由 `6~12G` → `110~114G`，约释放 **100G/节点**。

> ⚠️ 过程中曾误 `docker start vllm-tp4-rank0` 一次，立即纠正回 `exited`（见 §3 修正记录）。恢复阶段统一按本节状态处理。

---

## 3. test 容器创建（step 3，已执行）

**命名**：`vllm-tp4-test`，位于 node01，镜像=生产 `0.2.1-v026.0`，`--entrypoint /bin/bash` + `tail -f /dev/null` 保持存活（镜像 ENTRYPOINT 是 vllm CLI，需覆盖）。

**挂载（复用生产关键路径）**：
- `<INSTALL_DIR>/models/deepseek-v4-flash-0731 → /models`
- `<INSTALL_DIR>/lib/libncclpin.so → /opt/libncclpin.so`
- `/opt/nccl-ringonly → /opt/nccl-ringonly`
- `<INSTALL_DIR>/envs/vllm-envc-cache → /cache/huggingface`
- `/home/<USER>/vllm-cache → /root/.cache/vllm`
- `/home/<USER>/tilelang-cache → /root/.tilelang/cache`
- `/home/<USER>/b12x-cache → /root/.cache/b12x`
- tilelang patch → `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py`
- `<INSTALL_DIR>/nvfp4-landing-export → /vllm-workspace/nvfp4-landing`（**内核脚本，由 stop 的 rank0 经 `docker cp` 导出，见下**）
- host 网络、`--gpus all`、`--ipc=host`

**内核脚本来源**：交接文档明确 `/vllm-workspace/nvfp4-landing` 为容器内路径、**未挂载宿主机、容器重建丢失**。因此先从 stop 的 rank0 用 `docker cp` 导出到 `<INSTALL_DIR>/nvfp4-landing-export`，再挂载进 test 容器。

**可用性验证（全部通过）**：
- GPU 可见（GB10）；CUDA 可用
- `/models`、nccl-ringonly、b12x、vllm-cache、nvfp4-landing、tilelang patch 全部就位
- vLLM `_custom_ops` 的 `cutlass_scaled_fp4_mm` / `scaled_fp4_quant` 均可导入

---

## 4. kernel① 路线A 微基准（核心交付）

**脚本**：`mini_bench_k1.py`（自写，内联 make_weights，绕开 µpytest/matplotlib 依赖；routeA 与 v15 各自按布局喂权重，同一 fp32 权源）。

**原始数据（routeA vs v15，TFLOPS）**：

| M | K | N | routeA ms | routeA TF | v15 ms | v15 TF | 提速 |
|---|---|---|---|---|---|---|---|
| 256 | 4096 | 4096 | 0.088 | **97.2** | 0.893 | 9.6 | **10.11x** |
| 512 | 4096 | 4096 | 0.204 | **84.1** | 1.143 | 15.0 | 5.60x |
| 1024 | 4096 | 4096 | 0.367 | **93.5** | 1.771 | 19.4 | 4.82x |
| 256 | 8192 | 8192 | 0.371 | **92.6** | 2.942 | 11.7 | 7.93x |
| 512 | 8192 | 8192 | 0.524 | **131.1** | 3.692 | 18.6 | 7.04x |
| 1024 | 8192 | 4096 | 0.611 | **112.4** | 3.540 | 19.4 | 5.79x |
| 256 | 4096 | 16384 | 0.389 | **88.3** | 2.781 | 12.4 | 7.14x |
| 512 | 4096 | 16384 | 0.587 | **117.1** | 3.226 | 21.3 | 5.50x |

**结论**：
- routeA（原生 FP4 CUTLASS MMA）= **84~131 TFLOPS**
- v15（Triton bf16 回退）= **9.6~21.3 TFLOPS**
- **平均提速 ≈6.7×，范围 4.8×~10.1×**
- 与交接文档定论一致（v15 26-81/routeA 60-187 TFLOPS 量级）。本表采用每 shape 独立量化+计时更严格口径，v15 略低，但路线A 相对增益方向与量级完全吻合。
- routeA 适配层 smoke（`nvfp4_4w4a_mmaf.py`）通过：`(256,4096,4096) GEMM ok`。
- 对照探针（纯 cutlass GEMM、未含 preprocess/W 反量化）实测 routeA 达 **255~321 TFLOPS**。

---

## 5. kernel② v17 微基准（核心交付）

**脚本**：`mini_bench_k2.py`（自写，绕开 perf_report/matplotlib），口径 = v17 官方 `BYTES_PER_TOKEN=4680`（读4KB fp32 + 写584B/每 token）。

**原始数据（v17 vs v11，GB/s）**：

| T | v11 ms | v11 GB/s | v17 ms | v17 GB/s | 提速 |
|---|---|---|---|---|---|
| 256 | 0.023 | 52.1 | 0.015 | 80.3 | 1.54x |
| 1024 | 0.082 | 58.2 | 0.017 | **277.7** | 4.77x |
| 4096 | 0.314 | 61.1 | 0.075 | **254.8** | 4.17x |
| 16384 | 1.285 | 59.7 | 0.350 | **219.2** | 3.67x |
| 65536 | 5.544 | 55.3 | 1.620 | **189.3** | 3.42x |

**结论**：
- **v17 = 189~278 GB/s**（大 T 区间 219~278；T=1024 达 277.7 ≈ 理论 273 上限）
- **v11 = 52~61 GB/s**
- **提速 3.42~4.77×**，与交接文档 v17 194-262 GB/s、3.5-4.6× 完全一致。
- 确认 kernel② v17 生产替换 v11 有效，DS-MLA KV-Linear 带宽瓶颈大幅缓解。

---

## 6. 端到端 serve 基准（未能执行，原因说明）

任务要求「test 容器内起 serve（生产参数）跑 10 负载端到端」。**无法执行**，原因客观存在：

- 模型 `deepseek-v4-flash-0731` 为 **DeepseekV4ForCausalLM（DS-MLA）**：43 层、hidden 4096、**64 头 × head_dim 512**、q_lora_rank 1024、1 KV 头，权重 **48 个 safetensors shards**，生产以 **TP4 跨 4 节点**载入。
- 单节点仅 1×GB10（host ~121GB 内存，CPU+GPU 共享），**无法容纳完整模型权重**。
- 生产 serve 当前为卸载状态；即使恢复 4 rank，其运行的也是 **未接入两算子** 的 vLLM 默认路径（两算子未集成任何 serve 调用点，见 §8），测得的只是「基线」非「接入后」。

**结论**：端到端「两算子接入 vs 不接入」的 serve 级对比，需要先完成 **P0（生产持久化+把 routeA/v17 集成进可 serve 调用点）**，属于方案 B 集成改造，超出本轮授权边界（team-lead 已裁定不走方案 B）。本轮交付为**算子级可用性+微基准**，属任务 step2 原文目标。

---

## 7. 10 负载典型清单（已设计，供 TP4 恢复后执行）

`bench_prefill_decode_async.py`（宿主机 `<INSTALL_DIR>/bench_prefill_decode_async.py`，生产脚本）参数：
`--endpoint http://<head>:8001/v1 --model deepseek-v4-flash-0731 --rounds 3 --engine asyncio`

**10 组 conc × ctx**（覆盖 prefill 浅/深 × decode 并发）：

| # | ctx | conc | 覆盖 |
|---|-----|------|------|
| 1 | 512 | 1 | prefill 浅 / 低并发 |
| 2 | 512 | 3 | prefill 浅 / 中并发 |
| 3 | 4096 | 1 | prefill 中 / 低并发 |
| 4 | 4096 | 3 | prefill 中 / 中并发 |
| 5 | 16384 | 3 | prefill 中深 / 中并发 |
| 6 | 16384 | 5 | prefill 中深 / 高并发 |
| 7 | 65536 | 3 | prefill 深 / 中并发 |
| 8 | 65536 | 5 | prefill 深 / 高压 |
| 9 | 131072 | 3 | prefill 极深 / 中并发 |
| 10 | 131072 | 5 | prefill 极深 / 高压 |

- tasks: coding/json/prose（各含代表性生成长度），rounds=3。
- 衡量：总吞吐 / prefill p50 / decode p50×conc（脚本原生输出）。
- 两套对照：kernel① 路线A 接入 vs 不接入（**待 P0 集成改造后**），或当前基线先跑。

---

## 8. 重要发现（本次实测）

**两算子尚未接入任何生产 serve 路径**：
- `grep -rlE 'nvfp4_4w4a_mmaf|nvfp4_ds_mla_kv_linear_v17|RouteA' <INSTALL_DIR>/{scripts,lib} /home/<USER>/patch-v026` → **零命中**。
- 内核脚本仅存在于容器内 `/vllm-workspace/nvfp4-landing/`（本轮已导出到 `<INSTALL_DIR>/nvfp4-landing-export`）。
- 运行中的 serve 用 vLLM 0.26 默认 + flashinfer_b12x + tilelang hc patch。
- 交接文档 P0（生产持久化）、P1（生产性能简测）均标注未完成。
- 这解释了为何「serve 级端到端接入对比」无落点，本轮以算子级微基准替代（team-lead 裁定）。

---

## 9. 待办与建议

1. **P0 生产持久化**：把 `nvfp4_4w4a_mmaf.py`（routeA）与 kernel② v17 落到 `<INSTALL_DIR>/scripts/`（或新建 `nvfp4/`，libncclpin 模式），并建立容器内 import 路径，保证重建后可用。
2. **serve 集成（方案 B）**：若需端到端「接入 vs 不接入」总性能对比，须把 routeA/v17 接入可 serve 调用点（修订 start_tp4_head 新增 env/调用），再跑 §7 的 10 负载两套对照。
3. **TP4 恢复后端到端基线**：按 §7 清单跑一次基准（当前 vLLM 默认路径），作为后续接入后的对照基线。

---

## 10. 恢复到生产（step 6）—— ⚠️ 恢复受阻，全组未起

按用户裁决②，测试后 `docker start` 4 个 rank 恢复生产。**结果：整个 TP4 未能起起来**（详见本报告末尾《附录：生产恢复诊断》）。

- **rank0（head/01） exited**：`Worker_TP0` 在 `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py select_deepseek_v4_mxfp4_moe_backend` 选 `B12X_MXFP4` 后端时报 **`ValueError: Mxfp4 MoE backend 'B12X_MXFP4' does not support the deployment configuration since kernel does not support current device cuda`**（`is_supported_config` 需 `is_cuda() && is_device_capability_family(120) && _has_b12x()` 全真，head 处未通过）。
- **worker rank1/2/3 实际也失败**：rank1 日志 `Worker_TP1` 成功选到 `B12X_MXFP4`（`Using 'B12X_MXFP4' Mxfp4 MoE backend`），但随后 `Connection closed by peer [<NODE_IP>]`（head 先挂）→ `EngineCore failed to start`。worker 容器 `Status=healthy` 是**进程存活的假阳性**，EngineCore 未就绪。
- 已排除静态差异：4 节点同为 GB10/同驱动/同镜像 `0.2.1-v026.0`/同 env(`VLLM_USE_B12X_MOE=1`)/b12x-cache 一致(1.1M, 子目录相同)/同挂载。head nvidia-smi 正常、无残留 CUDA 进程。
- 干净重启（全停→head 先起→worker）**复现同样失败**。head 用 `docker run` 同一镜像读源码/查库均正常。

**判断**：重启后才暴露的、可复现的 pre-existing head（rank0）设备相关的 `B12X_MXFP4` MoE 初始化问题，深在 vLLM MoE kernel 层，超出本轮授权边界，需 MoE/kernel 专项介入（建议 `VLLM_LOGGING_LEVEL=DEBUG` 复现取详细 unsupported 原因；排查 head 上 device-capability 探针在 rank0 分布式启动上下文的行为）。

**当前生产状态**：未恢复（4 节点 TP 组 down）。自愈(monitor/timer)按用户要求保持关闭，未动。

---

## 附录：生产恢复诊断（原始证据）

### 容器现状（恢复后）
- rank0(head/01)=exited/unhealthy，出 `Exited(1)`，OOM=false
- rank1(02)/rank2(04)/rank3(03)=running/healthy（但 EngineCore 未就绪，见 rank1 日志）

### 关键错误（head, `mxfp4.py:428/599/609`，`b12x_mxfp4_moe.py:578`）
```
ValueError: Mxfp4 MoE backend 'B12X_MXFP4' does not support the deployment configuration
since kernel does not support current device cuda.
is_supported_config = p.is_cuda() and p.is_device_capability_family(120) and _has_b12x()
_has_b12x() = import b12x.integration.tp_moe.b12x_moe_fp4  (image 内存在 b12x-0.15.3)
```

### head vs worker 静态核对（均一致）
- GPU：NVIDIA GB10，driver 580.173.02，4 节点相同
- env：均含 `VLLM_USE_B12X_MOE=1`、`--moe-backend flashinfer_b12x`、`VLLM_HOST_IP` 各异（正常）
- b12x-cache：`/home/<USER>/b12x-cache` 均 1.1M，子目录 `cute_compile/{8f,a6,40,4c,c6,30}` 相同
- 挂载：均将 b12x-cache → `/root/.cache/b12x`

### 复现与影响
- `docker start` 一次 + 干净重启一次，两次均在 head 复现 B12X device 不支持 → 全组 EngineCore failed。
- 原生产 serve 在测试前 healthy 20h；一次 stop/start 后触发该问题。

### 后续建议
1. 由 MoE/kernel 负责人独立专项排查 head 为何在 rank0 分布式启动时 `is_device_capability_family(120)` / `_has_b12x()` 判定失败。
2. 复现可加 `VLLM_LOGGING_LEVEL=DEBUG` 获取 `_make_log_unsupported` 详细 reason。
3. 恢复期间自愈保持关闭（用户裁决），勿自动拉起失败组。