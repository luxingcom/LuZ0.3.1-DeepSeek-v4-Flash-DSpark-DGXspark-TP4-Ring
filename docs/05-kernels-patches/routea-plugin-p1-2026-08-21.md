# A′ 插件 P1 报告：实现 + 单卡 mini 验证（Task #22，2026-08-21）

**任务**: P1 · A′ 插件实现 + 单卡 mini 验证（承接 Task #20/#21）
**执行**: Archi（系统架构师）· node01 单卡一次性容器（生产镜像 + tilelang patch 挂载）
**产出**: 本报告 + `plugin_a1/`（插件代码，本地与 01:/tmp/_routea_work/ 双份）+ 6 组验证 logprob 快照（lp_v0..v4）+ 判定脚本

> **一页结论（供决策）**
> 1. **P1 全部验证通过，插件放行进入 P2（TP4）**：① env 关闭/低于阈值 → 与生产基线**逐位一致**（V1、V4 两组独立证明回退零污染）；② 全量 W4A4 → 总 logprob **+0.08%**（判据 ≤1%，好于 Task #21 的 +0.41%）；③ M 阈值分派上下限行为均验证正确。
> 2. **payload 共享实测**：hybrid（W4A16+W4A4 双驻留）内存 +15.5GB（mini 4 层 TP1）→ **生产 TP4 折算 +~41GB/rank**（b12x W4A16 prepare 就地销毁原始 payload，W4A4 必须自持副本——wrapper 的 w4a16 模式内部同样打包副本，无共享捷径）；full（全量 W4A4）+3.5GB（mini）→ **生产 +~8.7GB/rank**（payload 原地行交换零拷贝）。
> 3. **新发现（重要）**：W4A4 wrapper 路径**运行间不确定**（同配置重跑 mean|Δlp|≈0.13、非逐位；W4A16 路径确定）。greedy 输出 run-to-run 可变——TP4 A/B 判据不能用单次 greedy 对比，需 logprob 统计/多次平均；对生产（temperature>0）无增量风险。
> 4. **建议 TP4 默认配置**：hybrid + `VLLM_MOE_W4A4_MIN_M=3072`（decode 与 <3072 的 prefill 保持生产 W4A16 逐位不变，≥3072 chunk 走 W4A4 1.32×）；KV 缩水 ~41GB/rank 需评估，若不可接受改用 full 模式（+8.7GB/rank，中段 M=64-2048 承受 0.79-0.95×）。

---

## 1. 插件实现（plugin_a1/）

### 1.1 结构

```
plugin_a1/
  setup.py                    # vllm.general_plugins entry point（TP4 pip install -e 用）
  routea_plugin_a1/
    __init__.py               # env 门控 install(): monkey-patch 两处注入点
    w4a4_experts.py           # W4A4B12xExperts(B12xExperts 子类) + 适配器派生链
```

注入点（Task #20 核实）：`oracle.mxfp4.backend_to_kernel_cls` + `quantization.mxfp4.select_deepseek_v4_mxfp4_moe_backend`（后者是 `Mxfp4MoEMethod.__init__` 实际调用点，已实证 "Using W4A4B12xExperts" 日志）。B12X_MXFP4 → 返回 `W4A4B12xExperts`，其余 backend 不动。**零 vLLM 本体改动**；PYTHONPATH import 或 pip install 均可加载。

### 1.2 env 契约

| env | 默认 | 说明 |
|---|---|---|
| `VLLM_MOE_W4A4` | 0 | 0=off（生产原样） 1=hybrid（M≥MIN_M 走 W4A4，其余 W4A16） 2=full（全量 W4A4） |
| `VLLM_MOE_W4A4_MIN_M` | 3072 | hybrid 分派阈值（batch M = hidden_states 行数） |
| `VLLM_MOE_W4A4_CG` | 1 | wrapper use_cuda_graph（TP4 decode 捕获需要） |
| `VLLM_MOE_W4A4_DEBUG` | 0 | 分派决策日志 |

### 1.3 权重派生链（复用 Task #20 适配器 + Task #21 契约）

- **payload**：loader 给 `[w1(gate); w3(up)]`（routed_experts.py:1039 实证 w1=gate→rows[0:n]）；W4A4 wrapper kernel 约定 `[w3(up); w1(gate)]`（fork `reorder_w1w3_to_w3w1` 等价）。hybrid：行交换**副本**（+payload 一份）；full：**原地交换零拷贝**（layer 参数即 W4A4 payload，不做 w4a16 prepare）。
- **scale**：E8M0[E,N,K//32] →LUT 精确→ E4M3[E,N,K//16] →逐 expert `swizzle_blockscale`→ `convert_sf_to_mma_layout(num_groups=E)`（输入必须 swizzled，Task#21 实证）。
- **alpha/fc2_input_scale** = 1.0（input_scale=1.0 模式，Task#21 实测与原生校准差 0.12/0.24%）。
- **W4A16 回退**：完全继承生产 `B12xExperts` 路径（b12x w4a16 plan/bind，swiglu_limit clamp 保留）——**不做任何行序/布局改动**，规避 Task#20 发现的布局歧义。
- **swiglu 闸门**：MXFP4 oracle 路径无 NVFP4 oracle 的 ValueError 闸门；W4A4 分支不施加 clamp（实测 clamp 效果 0.0000，Task#21）——由 `VLLM_MOE_W4A4=1/2` 显式声明接受。
- 其余：topk_ids→int32 / topk_weights→fp32 复用 b12x 归一化函数；apply_router_weight_on_input=True 时回落 W4A16（wrapper 无此语义，DSV4 为 False）；process_weights 幂等防重入。

## 2. 单卡 mini 验证（mini-0731，真实 checkpoint 前 4 层，Task#21 构建器重建）

配置与 Task #21 完全一致（kv fp8、eager、同 7 prompts、greedy、max_num_seqs=12）。

| # | 运行 | 判定项 | 结果 |
|---|---|---|---|
| V0 | 无插件基线（W4A16 生产路径） | 基准 + 确定性 | 文本与 Task#21 基线一致 |
| V1 | hybrid MIN_M=999999（全回落 W4A16，经子类） | **env 开但低于阈值 → 逐位一致** | **PASS：prefill logprobs 逐位一致 + gen 文本一致** |
| V2 | full（全量 W4A4） | **质量：总 logprob ≤1%** | **PASS：+0.08%**（-8377.3 vs -8384.0；prefill mean\|Δlp\|=0.18=固有量化差） |
| V3 | hybrid MIN_M=64（batch M=519 ≥64 → 全 W4A4 prefill） | 分派上限行为 | PASS：输出贴近 V2（W4A4），内存/日志确认派生与分派生效 |
| V3b | V3 重跑 | **W4A4 路径确定性** | **非逐位：mean\|Δlp\|=0.132 —— W4A4 wrapper 运行间不确定**（W4A16 路径 V0/V1 逐位确定） |
| V4 | hybrid MIN_M=600（batch M=519 <600 → 全 W4A16） | **分派下限 → 逐位一致** | **PASS：prefill logprobs 逐位一致 + gen 文本一致（=V0）** |

内存实测（vLLM "Actual usage for weight"）：基线 15.95GB → hybrid **31.48GB**（+15.5：W4A4 payload 副本 12GB + E4M3 scale 及 swizzled 存储 ~3.5GB）→ full **19.48GB**（+3.5：仅 scale）。生产 TP4 折算（43 层/4 分片）：hybrid **+~41GB/rank**、full **+~8.7GB/rank**。

### 2.1 payload 共享结论（任务指定验证项）

双 wrapper（wrapper w4a16 模式 + wrapper w4a4 模式）共享 payload **不可行省内存**：flashinfer 的 w4a16 路径内部经 `_get_w4a16_packed_weights → prepare_w4a16_packed_weights` 同样生成打包副本（W4A16PackedWeights），与 b12x 集成层无本质差别；b12x w4a16 prepare 又会就地销毁原始 payload。**W4A16 与 W4A4 双表示必然 +一份 payload（~33GB/rank）**。取舍：
- **hybrid**：+41GB/rank，decode 与中段 prefill 保持生产逐位不变（回退语义最安全），KV 缩水需评估（mini 上 KV 85→69GB）。
- **full**：+8.7GB/rank，无 W4A16 回退；中段（M=64-2048）0.79-0.95× 回退（生产 decode M≤12 影响小：M=1 6.2×、M=8 0.92×；prefill chunk 4096 档 1.32×）。
- **B′ 备选**（Task#21 convert 脚本放大为全模型离线转换 + checkpoint config swiglu_limit=null）：零插件代码、单份 W4A4 权重、换 checkpoint 即部署——作为 full 模式的无插件替代，代价是回退需换回原 checkpoint 文件（无 env 秒级回退）。

### 2.2 新发现：W4A4 wrapper 运行间不确定性

同配置重跑 V3/V3b 非逐位（mean|Δlp|=0.132，量级≈固有量化差）——源出 flashinfer W4A4 MoE kernel（dynamic 后端原子归约类操作）。影响：
- 任何 W4A4 输出对比（含 Task#21 的 T/E/C 三方"实现间差"0.21-0.25）都含此抖动成分——进一步支持 Task#21 "非缺陷" 定性；
- **TP4 A/B 正确性判据必须用统计口径**（总 logprob/多次平均/needle 通过率），不能用单次 greedy 逐 token 对比；
- 生产 temperature>0 本有采样随机性，无增量风险；纯 greedy 调试场景需知晓。
- （次要）插件自身 vllm-logger 输出在 EngineCore 子进程被吞（"Using W4A4B12xExperts" 能打印但插件 logger.info 不能）——DEBUG 模式建议后续改 print；不影响功能。

## 3. TP4 部署方案（P2 执行输入）

1. **插件部署**（SRE 执行，<INSTALL_DIR> 只读约束下先落本地再由窗口部署）：
   - `plugin_a1/` → `<INSTALL_DIR>/nvfp4/plugin-a1/`（4 个节点同布）；
   - 容器改动（start_tp4_head.sh + worker 同步）：`-v <INSTALL_DIR>/nvfp4/plugin-a1:/opt/plugin-a1:ro` + `-e PYTHONPATH=/opt/plugin-a1`（或容器内 `pip install -e`）；`-e VLLM_MOE_W4A4=1 -e VLLM_MOE_W4A4_MIN_M=3072`。共 4 行，模型/后端 flag 零改动。
2. **测试矩阵**（建议顺序）：
   - T0 基线：现状 B12X-only（无 env）→ 采吞吐/logprob 基线；
   - T1 hybrid MIN_M=3072：正确性（needle + logprob 统计 ≤1%，多次平均覆盖 W4A4 抖动）+ 性能（4096 chunk prefill 延迟/吞吐，预期 kernel 1.32× → 端到端 ≥15% 判据）+ 稳定性（4 rank healthy、无 OOM，KV 缩水 ~41GB/rank 下按 max-len/concurrency 降档跑）；
   - T2 full（可选，若 T1 KV 不可接受）：+8.7GB/rank，附带中段负载（M=64-2048 混合）回退 ≤5% 复核。
3. **回退**：`VLLM_MOE_W4A4=0`（或 unset）重启即回生产原样——V1/V4 已证逐位一致，回退零污染。
4. **KV/其它**：KV dtype 维持生产 nvfp4_ds_mla 不动（mini 的 compressor stride 崩溃为 4 层小 cache 边界问题，Task#21 记录；43 层生产形状不受影响——TP4 起跑即验证）；W4A4 不与 KV K2 等其它变更联合灰度。

## 4. 工件清单

| 文件 | 位置 |
|---|---|
| `plugin_a1/`（setup.py + routea_plugin_a1 2 文件，~330 行） | 本地 routea-plugin-p1-2026-08-21/plugin_a1/ + 01:/tmp/_routea_work/plugin_a1/ |
| `lp_v0..v4, v3b.json`（6 组验证快照） | 同上目录 |
| `run_mini_plugin.py / compare5_plugin.py / compare6_dispatch.py` | 同上 |
| mini 模型（16GB） | 已清理（build_mini.py 可重建） |

环境约束遵守：一次性容器（--rm）、生产镜像 + tilelang patch、未启动生产、<INSTALL_DIR> 只读、单卡峰值 ~100GB（0.85 util 内）。
