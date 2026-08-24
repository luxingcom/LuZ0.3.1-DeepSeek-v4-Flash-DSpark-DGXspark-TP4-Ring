# 通信隐藏（Overlap）落地方案调研与代码交叉验证

**日期**：2026-08-17 ｜ **分析**：KernelGen ｜ **目标**：decode 每 token ~2.6ms 通信开销（61 层 × 14KB allreduce）的隐藏
**验证方式**：生产 0.26.1 容器内源码（vllm-tp4-rank0 挂载）+ 上游 GitHub v0.27.1 源码逐行对照

---

## 0. TL;DR

| 方案 | 面向架构 | 本集群（纯 TP4 环网）适配 | 结论 |
|---|---|---|---|
| **vLLM DBO（--enable-dbo）** | **仅 DP+EP**（文档明文） | ❌ 无 DP/EP，无 MoE all-to-all 可重叠 | 不适用 |
| **SGLang TBO（Two-Batch Overlap）** | DP+EP（DeepEP dispatch/combine） | ❌ 同上 | 不适用 |
| **vLLM SP（#46789，0.27.1）** | EP+DP/mega_moe（`_use_sequence_parallel` 硬条件） | ❌ 开关不满足；c1 单序列无切分收益 | 不适用 |
| **自研 TP4 allreduce 双 stream overlap** | 纯 TP（本集群） | ✅ 唯一正解 | **落地候选** |

**核心结论**：业界主流通信隐藏（DBO/TBO/SP）全部围绕 **EP 的稀疏 all-to-all（dispatch/combine）** 设计——而本集群 TP4 下 MoE 是列并行+allreduce，无 EP 通信可 ping-pong。纯 TP4 的 overlap 只能走**自研：把 RowParallel allreduce 与相邻 GEMM 用双 stream 重叠**。

---

## 1. 代码交叉验证（生产 0.26.1 vs 上游 0.27.1）

### 1.1 生产 0.26.1（容器内实测）

| 检查点 | 结果 | 含义 |
|---|---|---|
| `vllm/distributed/communication_op.py` | 仅 `tensor_model_parallel_all_reduce` 同步调用，**无 stream 参数/异步变体** | 无现成 overlap 原语 |
| DSV4 `nvidia/model.py` L93-99 | `is_sequence_parallel` 参数存在（FFN 侧 `disable_tp` 预留） | **半成品 SP 接口** |
| DSV4 attention.py | **零 SP/AG/RS 支持** | SP 不完整 |
| 全树 `is_sequence_parallel=True` 调用点 | **0 处** | SP 从未启用 |
| `compilation_config.py` fuse_* 开关 | **不存在** | 无编译级融合 |

### 1.2 上游 0.27.1（GitHub raw 逐行对照）

| 检查点 | 结果 | 含义 |
|---|---|---|
| `vllm/models/common/ops/sequence_parallel.py` | **新增模块**：`sp_all_gather`/`sp_reduce_scatter`，支持 `custom_all_gather`/`custom_reduce_scatter` 钩子 | SP 原语完整 |
| DSV4 model.py L960-968 | `if use_sequence_parallel: x=sp_all_gather(x)` → attn → `x=sp_reduce_scatter(x)` | **标准 SP 结构已接入** |
| `_use_sequence_parallel()` L805-812 | `return (PP==1 and enable_expert_parallel and TP>1 and (mega_moe or DP>1))` | **硬条件：必须 EP** |

**判定**：0.27.1 SP 是为 **EP 部署配套**（EP 时序列切分 + 与专家路由通信重叠）；纯 TP4（本集群）不满足启用条件，且 c1 单序列 decode 下序列切分无实义。`custom_*` 钩子为未来自定义通信算子预留（可插拔点）。

---

## 2. 为什么主流方案都不适用（架构本质）

- **DBO/TBO** 重叠的对象 = MoE **dispatch/combine（all-to-all）**：EP 下 token 跨 rank 送专家，通信稀疏且可与另一 microbatch 计算乒乓。TP4 下无此通信类型（专家在本地列分片，聚合用 allreduce）。
- **SP** 重叠对象 = attention 前后 AG/RS 与 norm/FFN 计算：需 EP 承载序列切分语义。
- 本集群 4 机环网 2 邻居拓扑：即便改 EP，all-to-all 在环上更差（非全连接），无意义。

---

## 3. 落地候选：TP4 allreduce × GEMM 双 stream overlap（自研）

### 3.1 原理
decode 每层关键路径：
```
attn 输出 ──(allreduce)──> x ──(ffn_norm)──> gate_up GEMM ──> ... ──> down_proj ──(allreduce)──> ...
```
当前 allreduce 与后续 GEMM **串行**（同 stream）。目标：allreduce 放 comm stream，gate_up GEMM 提前在 compute stream 启动，两者并行。

### 3.2 改动点（对照生产 0.26.1 源码）
| 位置 | 改动 |
|---|---|
| `vllm/distributed/communication_op.py` | 新增 `tensor_model_parallel_all_reduce_async(input_, stream)`：NCCL 调用在 comm stream 上发（torch 原生支持 `nccl` 通信与任意 stream 混用，record event 定序） |
| DSV4 `nvidia/model.py` DecoderLayer.forward | attention 输出处：norm 计算保持 compute stream → allreduce 切 comm stream → 同时 gate_up GEMM 在 compute stream 排起 → `wait_event` 后继续 |
| CUDA graph 兼容性 | **最大风险**：graph capture 下多 stream 需 capture 两条 stream + event 定序；vLLM PIECEWISE 模式每 size 档 capture，需验证 capture 期间 stream 切换合法（NCCL graph 支持 `ncclConfig.cudaGraph`） |

### 3.3 收益预期
- decode 61 层：若隐藏 ~50% allreduce 等待 → **~1.3ms/token 回收 → decode +5~8%**
- 保守估计（CUDA graph 限制下部分层生效）：**+3~5%**
- 无带宽代价，纯延迟隐藏（与 B1 正交，可叠加）

### 3.4 风险与前置
- CUDA graph capture 兼容性需 PoC 验证（PIECEWISE 各 size 档）
- shim 线程 pin（NCCL→8-9）与 comm stream 的交互需复测
- 改 vLLM 源码 → 需**重建镜像**（anemll 0.2.1-v026.0 系），或 overlay 挂载 patch（仿 v027-test patch 机制，改动 2 个文件可 overlay）

---

## 4. 建议行动

1. **PoC（下一步，测试窗口）**：按 §3.2 改 2 个文件（communication_op + model.py），overlay 挂载到 v027 测试环境（复用 0.27 测试链，非生产镜像）→ 验证：①CUDA graph capture 通过 ②nccl-tests 层面 allreduce 与 GEMM 并行性（nsys 看 gap）③c1@131K decode 变化
2. **备选对照**：0.27.1 SP 强开（hack `_use_sequence_parallel` 返回 True）评估 c4/c6 大 batch 场景收益——但 c1 无收益，优先级低
3. **长期**：跟踪 vLLM 上游 `custom_all_gather/reduce_scatter` 钩子生态（未来官方 TP overlap 或复用自定义算子接口）
4. **明确不做**：改 EP 架构、SGLang 迁移（环网拓扑不支持高效 all-to-all）

## 5. 证据档案
- 生产 0.26.1：`vllm-tp4-rank0:/usr/local/lib/python3.12/dist-packages/vllm/`（container 实测 grep）
- 上游 0.27.1：raw.githubusercontent.com/vllm-project/vllm/v0.27.1（model.py L805/829/960-968、sequence_parallel.py、attention.py）
- 社区：vLLM DBO 文档（vllm.ooos.top/design/dbo，明文"仅 DP+EP"）、vLLM Blog（DBO for decode on Wide-EP）、SGLang TBO（DeepEP 前置）
