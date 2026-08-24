# PoC：FP8 量化 AllReduce 核心算子方案（含 FP8 vs NVFP4 通信选型论证）

- 日期：2026-08-17
- 专家：KernelGen / 通信优化线
- 状态：**选型已定（FP8 胜出）**，patch 待落库
- 关联：`nccl-operator-latency-analysis-2026-08-17.md`（算子延迟分析）、`nccl-ab-B-execution-report-2026-08-17.md`（B1 固化）、`research-comm-overlap-tp4-2026-08-17.md`（overlap 调研）

---

## 0. 结论先行（TL;DR）

**fp8 量化 allreduce 是本集群通信减半的正选；NVFP4 量化 allreduce 在通信场景不可行、无收益。**

决定性差异不在"比特数"，而在**集合通信库的值域归约能力**：

| 维度 | FP8 量化 AR | NVFP4 量化 AR |
|---|---|---|
| 通信量 | bf16 16bit → **8bit（-50%）** | 16bit → **4bit（-75%）** |
| **NCCL 原生归约** | ✅ `ncclFloat8`（NCCL 2.19.3+，我们 2.30.7 原生） | ❌ **NCCL 无 4-bit 类型**，无法值域归约 |
| 精度（求和场景） | E4M3 3-bit 尾数 + 共享缩放（FP8-LM 验证等效 bf16） | E2M1 1-bit 尾数、max=6.0，**求和误差累积不可控** |
| 硬件路径 | SM120 原生 fp8 转换指令 + NCCL FP32/FP16 accumulator | FP4 tensor core 优势**只在 GEMM 计算路径**，归约用不上 |
| 实现成本 | torch 原生 `float8_e4m3fn` 转换，改 2 个接入点 | 需自定义打包 + allgather 本地归约，工程量大 |
| decode 小消息收益 | 14KB→7KB，LL 协议延迟主导（~41µs），**收益小** | 3.5KB，同样延迟主导，**收益接近 0** |
| prefill 大消息收益 | 368KB→184KB，Simple 带宽主导，**收益明确** | 理论再减半，但 NCCL 无原生支持而不可行 |

**一句话**：NVFP4 是**权重存储与 GEMM 计算格式**（4W4A 铁律的用武之地），FP8 是**激活/通信传输格式**——各司其职，NVFP4 强行用于通信属于"拿着计算格式干传输的活"。

---

## 1. 背景：为什么做量化 allreduce

（承接 `nccl-operator-latency-analysis-2026-08-17.md`）

- decode 每 token：61 层 × 2 个 allreduce（attention `wo_b` + MoE down）≈ **2.6ms 通信**，占 decode 24%
- 14KB 小消息实测 43µs（B1 后），其中 **~15µs 是 6 跳 RDMA 物理下界**（不可消），~25µs 是协议+流水（可压）
- 368KB prefill 实测 86µs（B1 后 224KB），Simple 协议带宽主导，**压缩通信量直接减传输时间**

量化 allreduce 是"砍掉字节数"的算子级手段：**通信量减半 = prefill 大消息时间近似减半**；decode 小消息因延迟主导收益有限但仍有几 µs/层。

## 2. 生产 0.26.1 DSV4 通信结构（实锤，2026-08-17 代码核对）

生产模型是重度定制实现，**每层恰好 2 个 allreduce 点**，均藏在组件内部：

```
DecoderLayer.forward:
  x ──mhc_pre_tilelang（融合 norm，无 AR）──> attn
  attn: fused_wqa_wkv(Column) → attention → wo_a einsum → wo_b(RowParallelLinear, 内部 AR #1)
  x ──mhc_fused_post_pre_tilelang（融合 norm，无 AR）──> ffn
  ffn: gate(Column) → FusedMoE(gate_up Column + experts + down RowParallel reduce_results=True, AR #2)
```

- AR #1：attention `wo_b`（RowParallelLinear，`reduce_results=True` 默认）→ 14KB bf16
- AR #2：MoE `down`（FusedMoE 构造 `reduce_results=True`）→ 14KB bf16
- `mhc_pre/mhc_fused_post_pre_tilelang` 均**无 allreduce**（grep 实锤），纯计算融合
- 已有多 stream 基建：attention 内 `attn_gemm_parallel_execute` + 3 路 aux stream + `ln_events`（GEMM×GEMM 并行），但**无通信×计算并行**，且 c1 单序列下依赖链（wo_b AR → mhc norm → gate_up）严格串行，**stream 切分无接入点**（修正上一轮 overlap 结论）

**结论**：overlap 路线在本结构下收益≈0 且工程复杂；**量化通信量是唯一干净、可测、低风险的算子级优化**。

## 3. 为什么 FP8 而不是 NVFP4（选型论证）

### 3.1 决定性：集合通信库的值域归约

allreduce 不是字节搬运，是**数学归约**（SUM 等）。NCCL 对每种数据类型有对应的归约 kernel：

- **FP8（E4M3/E5M2）**：NCCL 2.19.3+ 原生支持（`ncclFloat8`），NCCL 2.27 更进一步在 NVLS 上用 FP16 accumulator 归约 FP8。PyTorch 侧 `torch/distributed/ops/fp8_ops.py` 提供了 FP8 感知的 AllReduce（先 cast 再通信再 cast 回），**带宽减 4x（相对 FP32）**。我们 NCCL 2.30.7 完全覆盖。
- **NVFP4（E2M1）**：NCCL **不存在** `ncclFloat4` 类型。4-bit 打包数据无法在 GPU 归约 kernel / 网卡上直接做值域求和。要通信只能：
  - 解包成 fp8/bf16 再归约 → **通信量仍是 8/16bit，没有 -75% 收益**（解包在通信前，传的还是高精度）
  - allgather 原始 4-bit + 本地解包求和 → 通信量 (N-1)/N×4bit，但需自研打包/解包/求和 kernel，且 allgather 延迟特性差于 ring allreduce

**结论：NVFP4 的 -75% 在"值域归约"语义下根本吃不到**——这是硬件/软件栈层面的硬约束，不是工程努力能绕过的。

### 3.2 精度：求和场景是 FP8 的舒适区、NVFP4 的雷区

- FP8 E4M3：3-bit 尾数 + 8bit 动态范围。FP8-LM 论文（社区大规模训练验证）证明：**自动缩放 + 共享标量（先收集全局最小缩放因子再统一量化）→ FP8 梯度/激活 AllReduce 与 BF16 等效精度**，SNR 对比 pre/post scaling 全面最优。
- NVFP4 E2M1：1-bit 尾数，max=6.0，有效精度 ~3-4 bit。它靠**双级缩放**（16 元素 E4M3 block scale + 全局 FP32 scale）补偿——但那是**静态权重/GEMM 输入**场景的设计。**求和是累加运算，量化噪声随 rank 数与层数累积放大**：4 rank 求和 + 61 层传播，E2M1 的 ~1/8 级相对误差会滚雪球。NVIDIA 论坛实测也证实 NVFP4 在 draft 场景精度劣于其他量化法。
- 本集群 4W4A 铁律的适用边界：NVFP4 是**权重格式**（MoE expert I8 148GB、deep_gemm 原生 E8M0→NVFP4 路径）；**激活/通信保持 FP8 是行业共识**（DeepSeek MLA 的 fp8_ds_mla KV、vLLM fp8 allreduce 均为通信侧 fp8）。

### 3.3 硬件：SM120 的 FP8 归约路径现成，FP4 优势在别处

- GB10/SM120：fp8 转换有原生指令，NCCL 归约用 FP32 accumulator（不依赖 tensor core fp8），**路径现成、零开发**
- NVFP4 的 2× FP8 tensor core 吞吐优势**只在 GEMM 计算**（权重×激活），通信归约是 NCCL kernel 内部的事，**用不到 FP4 tensor core**
- 我们的 ringonly 补丁（PEER_HCA 映射）+ stageB tuner 都是**传输层**优化，与 dtype 正交——fp8 AR 完全兼容

### 3.4 收益量化（对本集群实测基线）

| 场景 | 现状（B1 固化） | FP8 AR 预期 | NVFP4 AR 预期 |
|---|---|---|---|
| decode 14KB×122 次/token | 43µs/次 ≈ 5.2ms/token 通信 | 7KB：~36-40µs/次（省 ~0.4-0.8ms/token） | 3.5KB：~34-38µs/次（**增量极小**，且不可行） |
| prefill 224KB | 86µs | 112KB：~45-55µs（**-35~-45%**） | 56KB：理论 ~25-30µs，但不可行 |
| 端到端 c1@131K | DE 104 | DE +1~3% | — |

### 3.5 结论

1. **主选：FP8 量化 allreduce**（E4M3 + 共享 block scale），实现路径 = torch 原生 fp8 转换 + NCCL fp8 归约 + 反量化
2. **NVFP4 通信关闭**：NCCL 无 4-bit 归约 = 硬不可行；即便 allgather 方案，收益被工程复杂度与精度风险吞没
3. NVFP4 的职责边界明确：**权重存储 + GEMM 计算**（与本集群 4W4A 路线一致），不扩展到通信

## 4. PoC patch 设计（落库待执行）

### 4.1 接入点（2 处，均为组件内部 reduce 前插量化）

- AR #1：`vllm/models/deepseek_v4/nvidia/ops/o_proj.py` → `deep_gemm_fp8_o_proj` 返回值后，对 wo_b 输出做 fp8 AR（wo_b 构造改 `reduce_results=False`，由 env 控制）
- AR #2：`vllm/model_executor/layers/fused_moe/layer.py` → down reduce 前插 fp8（`reduce_results=False` 已支持，`skip_final_all_reduce` 逻辑现成）

### 4.2 核心算子（新增 `vllm/.../ops/fp8_allreduce.py`）

```python
def fp8_quant_all_reduce(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    # 1. per-block amax → 全局共享 scale（先收集各 rank amax，allreduce 取 max）
    # 2. x / scale → float8_e4m3fn（torch 原生，SM120 指令）
    # 3. NCCL all_reduce（ncclFloat8 原生）
    # 4. 反量化 scale * y
```

- 缩放策略：**共享 block scale**（FP8-LM 模式：先 allreduce scale 再统一量化），避免 per-rank 独立 scale 导致归约语义错位
- 精度保护：block=128（14KB→112 个 scale，开销可忽略）；可选 per-token scale 对照档

### 4.3 env 开关与回滚

- `AICAD_FP8_AR=1` 启用，默认 0（生产行为零变化）
- 挂载式 patch（仿 v027-test overlay 机制），不动镜像；回滚 = 去挂载重启

### 4.4 测试矩阵（工程测试窗）

| 档 | 内容 | 判定门槛 |
|---|---|---|
| U1 | fp8 算子精度单元测试（随机张量，4 进程模拟归约） | 相对误差 < 1e-2（bf16 基线） |
| N1 | nccl-tests fp8 vs bf16（14K/28K/56K/112K/224K） | 224KB ≤ 50µs，14KB 不劣化 >3µs |
| E1 | c1@131K 端到端（fp8 AR on） | PR/DE/TTFT 不劣化 >3%，DE 期望 +1~3% |
| E2 | c1@131K 生成质量抽查（同 prompt 对比 bf16 基线） | 语义/风格无退化 |
| E3 | c4@32K 并发档回归 | 无劣化 |

### 4.5 风险与预案

- **精度风险**：若 E2 检出质量退化 → 回退 block 到 per-token scale 或关闭 AR#2（MoE down 对精度更敏感）
- **NCCL fp8 与 ringonly 兼容**：预期正交，N1 档同时验证
- **cudagraph**：fp8 转换 kernel 为普通 elementwise，可 capture；PIECEWISE 各档需验证

## 5. 交付物清单

- [ ] 本方案文档（选型论证 + 设计）
- [ ] `fp8_allreduce.py` 核心算子（落库 `<INSTALL_DIR>/scripts/`）
- [ ] 2 个接入点 patch（o_proj.py / fused_moe layer.py）
- [ ] env 开关 + 挂载/回滚脚本
- [ ] U1 精度自检脚本（含 4 进程模拟）
- [ ] 测试矩阵执行（用户安排窗口后）

## 参考

- FP8-LM（自动缩放 + 共享标量 FP8 AllReduce，SNR 等效 bf16）
- NCCL 2.19.3+ `ncclFloat8`；NCCL 2.27（FP8 用 FP16 accumulator）
- PyTorch `torch/distributed/ops/fp8_ops.py`
- NVFP4 规范（Transformer Engine：E2M1 + 16 元素 E4M3 block scale + 全局 FP32 scale）
- NVIDIA 论坛：FP4 on DGX Spark（NVFP4 精度劣化讨论）
