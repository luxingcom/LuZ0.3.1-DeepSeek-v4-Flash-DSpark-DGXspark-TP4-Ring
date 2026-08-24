# 环境 D→E：Anemll 真方案升级部署计划与验收策略

**日期**：2026-08-02
**工作流**：系统设计 + 部署前检查（工作流 2/4 组合）
**参与成员**：Archi（升级方案与 ADR）/ Tessa（验收测试策略）/ Rex（NCCL 与前置坐实）/ 主理人（NJU 预检、编排汇编）

---

## 📌 TL;DR（执行摘要）

- **目标**：将环境 D（hybrid-1.6 降级版：seqs=1、KV fp8）升级为 **Anemll 真方案**（dspark-vllm-gx10:0.1.1：SM120 kernels、nvfp4_ds_mla KV、**max-num-seqs 6**、1M ctx），参考知乎实战（单流 74 t/s、4 并发 agg 150 t/s、25K prefill 1.1K t/s）
- **可行性已确认**：NJU 镜像代理（ghcr.nju.edu.cn）可达且 tags 含 0.1.1（主理人预检 HTTP 200）
- **风险可控**：最高风险为 0731 权重 vs nvfp4 KV 兼容性 → head 单机 dry-run 先行，失败回退 fp8 KV（不换权重）；Anemll 用专属端口 **8002**，生产 C(8001)/8000 不动
- **验收门槛**：conc3 全场景 err=0 且 TTFT<5000ms（seqs=6 关键提升）、decode≥24 t/s、文章基线 ±20% 容忍
- 阻塞 / 非阻塞：**非阻塞**（待人类拍板停机窗口后执行，总预算 2.5-3h）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 方案就绪，待拍板执行 |
| 阻塞项数量 | 1（停机窗口确认 + 0731/nvfp4 dry-run 结果） |
| 关键行动项 | 6 条（见行动清单） |
| 建议下一步 | 人类拍板停机窗口 → P1 拉镜像 → dry-run → 全量部署验收 |

---

## 需求与目标

- **现状**：环境 D 降级版（hybrid-1.6 + seqs=1 + KV fp8 + 1M ctx）无法承载并发（并发 3 TTFT 11s，S2/S3/S4 排队未测），且无 SM120 专属内核（hc_pre 撞墙风险路径）
- **目标**：Anemll 真方案——1M ctx 全功能 + seqs=6 可并发 + nvfp4_ds_mla KV（4-bit 压缩、KV 池更大）+ SM120 baked 内核免自编译；对标文章实测（单流 74 t/s、4 并发 150 t/s）

## 高层设计（Archi 方案，落盘 `hardened/live/PLAN-envE-anemll-upgrade.md`）

| 维度 | 方案 |
|---|---|
| 镜像 | `ghcr.nju.edu.cn/anemll/dspark-vllm-gx10:0.1.1`（28.9GB，vLLM 0.25，SM120 kernels），双机各自 pull（~4min），digest pin `name@sha256:` 双机核对 |
| docker run | `--network host --ipc=host --privileged --gpus all --shm-size=64gb --ulimit memlock=-1 --ulimit stack=67108864` + healthcheck start-period 900s（cold start 8-15min）；BINDS 保留 `/models:ro`+缓存+libibverbs/mlx5 宿主库 |
| serve 参数 | `vllm serve /models/deepseek-v4-flash-0731 --served-model-name deepseek-v4-flash-0731 --kv-cache-dtype nvfp4_ds_mla --max-model-len 1048576 --max-num-seqs 6 --gpu-memory-utilization 0.85 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' --moe-backend flashinfer_b12x --distributed-executor-backend mp --enable-flashinfer-autotune --tensor-parallel-size 2 --nnodes 2 --master-addr <NODE_IP> --master-port 25000`（0.25 CLI：--model 非 positional；spec 勿加 dspark_block_size） |
| env 关键项 | 文章必配：DSPARK_SLOT_CLAMP=1 / VLLM_USE_B12X_MOE=1 / VLLM_TRITON_MLA_SPARSE=1 / VLLM_DSPARK_LOCAL_ARGMAX=1 / VLLM_USE_FLASHINFER_SAMPLER=1 / DG_JIT_USE_NVRTC=0 / DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc / **VLLM_USE_BREAKABLE_CUDAGRAPH=0** / TORCH_CUDA_ARCH_LIST=12.1a / FLASHINFER_DISABLE_VERSION_CHECK=1 / TILELANG_CLEANUP_TEMP_FILES=1 / PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；本项目：MASTER_ADDR=<NODE_IP> / MASTER_PORT=25000 / NODE_RANK / HEADLESS / PORT=8002 / VLLM_ENGINE_READY_TIMEOUT_S=7200 / HF_HUB_OFFLINE=1；NCCL 双链路（见下） |
| NCCL | 无 bond0 → 本项目用双 socket 链路：NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1；保留 NCCL_NET=IB / HCA=rocep1s0f1,roceP2p1s0f1 / CROSS_NIC=1 / PROTO=LL,LL128,Simple；**NCCL_IB_GID_INDEX 改 3 或移除**（Rex 坐实 index 5 GID 为空） |
| 不注入 | hybrid-1.6 特有：VLLM_DSPARK_IMPL=upstream / VLLM_DSPARK_KV_QAT / LD_PRELOAD=libnccl-local-inference / VLLM_NCCL_SO_PATH（镜像内建 DSpark path 自带，避免冲突） |
| 启动脚本 | start_head_E.sh / start_worker_E.sh：沿用回滚锚点模式；CMD 改写逻辑重写（0.25 语法）；就绪判定改 `docker logs` 轮询 "Application startup complete"（head 多进程 /health 早响应不可靠）；worker 先→12s→head 后 |

## 关键决策记录（ADR）

- **ADR-0013**：升级 Anemll 0.1.1，接受 vLLM 0.25 断代 + metrics 核对成本，换 SM120 真路径
- **ADR-0014**：保留 0731 权重（48 片双机已就位），nvfp4 dry-run 失败回退 `--kv-cache-dtype fp8`（保 kernel/seqs），不切换权重（避免 156GB 重拉）
- **ADR-0015**：NCCL 双 socket 链路（10.100.136/137）替代文章 bond0，与 Rex 坐实结果收敛

## 可运维性（Rex 联动）

- RoCE 通道已坐实：verbs RC QP 双链路满用（200Gb×2），面板已改 IB 计数器监控（见 incident-roce-monitor 报告）
- **NCCL_IB_GID_INDEX 隐患**：index 5 GID 为空（有效 IPv4 GID 在 2/3），当前靠自定义 NCCL 库兜底；升级 Anemll（新库/新版本）前必须修正，防 verbs 初始化失败回退 socket
- 部署顺序：worker 先 → head 后 12s（铁律）；cold start 8-15min；回滚锚点模式 RTO≤15min
- 停机窗口：cold start ×2 + 全矩阵 ~2.5-3h，需人类确认

## 测试策略（Tessa 验收矩阵）

**部署成功判据**：双机 images 含 0.1.1（28.9GB）；双容器 healthy（worker→head+12s，轮询 "Application startup complete" ≤15min）；/v1/models 返 max_model_len=1048576；**8002/metrics 可抓（Prometheus 需新增 8002 scrape target）**

**功能测试**：① 1M ctx 长输入单请求（500K→512 输出，可选 1M→256；prefill≈1.1K t/s → 500K≈8min、1M≈15min，超时 1800s）；② thinking 冒烟（reasoning 字段非空）；③ tool-call json-schema（get_weather）；④ **质量抽查（GSM8K 式数学，防 nvfp4 4-bit KV 精度损失）**

**性能矩阵**：复用 S1-S5 × 并发 1/3（预热 1+3 轮中位数）+ **新增 L128K/L256K（输入→512 输出）单流**（L512K/L1M 扩展项）；collect_prom 采 kv_cache_usage/num_preemptions/spec 接受率

**对比维度（vs 现 D 基线 results_D_*.jsonl）**：seqs=1→6 并发承载（conc3 全场景可跑、TTFT 11s→<5s）；nvfp4 vs fp8（KV 池水位/质量/TPOT）；decode（conc1 turn 33→?、conc3 聚合）；spec5 接受率 vs D 35.9%

## Go/No-Go 门槛（Tessa 定稿）

- err=0；短场景 TTFT(thinking=max) P50<5000ms；decode P50≥24 t/s
- **conc3 S1-S5 全场景 err=0 且 TTFT<5000ms**（seqs=6 关键提升验证）
- 文章单流 74 t/s ±20%（≥59 t/s）容忍，超差**调查不自动拒**（权重/拓扑/RoCE 差异）
- 1M 长输入 prefill-bound 不作 TTFT 门槛；conc3 聚合 S1≥90 t/s 参考

## 风险与回滚

| 风险 | 等级 | 缓解 |
|---|---|---|
| 0731 权重 vs nvfp4_ds_mla 不兼容 | 🔴 高 | head 单机 dry-run 先行；失败回退 fp8 KV（保 kernel/seqs），不换权重 |
| vLLM 0.25 metrics 断代 | 🟠 中 | 启动后核对 8002/metrics 指标名（现无 request_error_total），必要时仅改 prometheus-alerts.yml expr；Prometheus 加 8002 target |
| NCCL GID/库变更致 verbs 失败 | 🟠 中 | 部署前修正 GID index；启动后 rdma QP 复核 |
| nvfp4 下 spec5 接受率<20% | 🟡 低 | decode 增益失效评估，回退 fp8 |
| 长上下文 KV 池满 → preemption | 🟡 低 | 2×128K 并发实测 seqs=6 边界；collect_prom 监控 |
| 回滚 | — | docker inspect 锚点 + 回滚 start_*_D.sh（已知良好基线），权重不动，RTO≤15min |

## 执行计划（P0-P9，总预算 2.5-3h）

P0 预检(5min：端口/基线/prom target) → P1 拉镜像(4-15min 双机，NJU) → P2 部署(20min) → P3 部署验证(5min) → P4 功能冒烟(10min) → P5 短矩阵 conc1/3(40min) → P6 长上下文 L128K/L256K/L1M(30-60min) → P7 长上下文并发+prom 佐证(10min) → P8 对比报告(10min) → P9 恢复生产(15min：停 Anemll→恢复 C@8001→smoke)

**约束**：8000/生产 C(8001) 全程不动；Anemll 专属端口 8002；每阶段超时预算执行

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **人类拍板停机窗口**（2.5-3h，Anemll 部署期间 8002 独立运行，可并行） | 人类 + 主理人 | P0 | 决策窗口 |
| 2 | 修正 NCCL_IB_GID_INDEX（5→3 或移除）后执行 P1 拉镜像 + P2 部署（start_head/worker_E.sh） | Rex | P0 | 拍板后当天 |
| 3 | head 单机 dry-run 验证 0731×nvfp4 兼容性（失败即回退 fp8 KV） | Tessa+Rex | P0 | 部署当天 |
| 4 | Prometheus 新增 8002 scrape target + 核对 0.25 metrics 名 | Rex | P1 | 部署当天 |
| 5 | 执行验收矩阵（功能/短矩阵/长上下文/对比）+ Go/No-Go 判定 | Tessa | P1 | 部署后 |
| 6 | 验收通过后：更新 PARAMS.md（环境 E 基线）+ 面板 RoCE 告警补强 | Docu+Rex | P2 | 1 周内 |

## ⚠️ 待完善 / 已知局限

- 0731 权重（48 片双机）与 nvfp4_ds_mla 的兼容性未实测——dry-run 是唯一判据；文章权重为 fp4 DSpark 156GB（逻辑同构字节不同）
- Anemll 镜像 28.9GB × 2 机拉取（NJU ~4min/机），期间占用双机带宽；若 NJU 抖动需重试策略
- vLLM 0.25 与现有监控/告警规则（vllm-dspark.rules）的指标名差异待启动后核对
- 1M 长输入单请求耗时 8-15min，属 prefill-bound，验收时需单独时间窗
- 环境 D 现基线以 results_D_*.jsonl（seqs=1/KV fp8）为准（8001 现为恢复后的 C）

## 📚 数据来源 & 成员产出索引

- Archi：PLAN-envE-anemll-upgrade.md（完整方案，落盘 hardened/live/）+ ADR-0013/0014/0015
- Tessa：验收矩阵、对比维度、Go/No-Go 门槛、执行计划 P0-P9、只读探测（8002 端口、metrics 现状）
- Rex：NCCL 通道坐实（verbs/双链路）、GID 隐患、digest 双机核对流程
- 主理人：知乎文章抓取与要点提取、NJU 代理可达性预检（HTTP 200 + tags 0.1.1）、编排汇编
- 参考资料：知乎《Deepseek V4 Flash 74 t/s 实测 DGX SPARK 双机部署》（zhuanlan.zhihu.com/p/2067050069652772697）、al-engr ds4-f-anemll-1m6-recipe、Anemll/dspark-vllm-gx10

---

> 本报告由工程保障团队 AI 协作生成，关键决策（停机窗口、镜像升级、权重兼容性回退策略）请由人类工程负责人复核拍板。
