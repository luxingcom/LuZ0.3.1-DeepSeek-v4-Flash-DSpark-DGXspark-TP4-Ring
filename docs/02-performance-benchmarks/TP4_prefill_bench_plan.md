# TP4 组网 prefill 主导基准测试计划（prefill:decode=100:1）

## 一、基准口径

**场景定义**：长 prompt 批量处理（RAG/长文档）。测例参数：
- prompt 档位：8k / 32k / 128k / 256k（现有 calib 仅到 32k，需新生成 128k/256k；preflight 校验服务端 max_model_len≥262144）
- 输出：max_tokens=8~32（8k@32=250:1，比 100:1 更极端、更保守；如需精确 100:1 用 8k@80）
- 温度统一 0.6；每档 warmup 2~3 请求；每档 3 轮取中位
- 随机前缀铁律：每请求 prompt=`<rnd:{随机}>`+校准文本，防 prefix-cache（现有 final_matrix 复用静态 calib 文本，同档请求共享前缀、存在 cache 污染，必须修）

**指标**：
- prefill_tps（聚合）= Σprompt_tokens / wall_s（usage 聚合；若 fork 暴露 vllm:prompt_tokens_total，用增量交叉验证）
- TTFT p50/p95（含调度+prefill+网络）
- decode_tps（辅助）；吞吐-延迟曲线（TTFT vs prefill_tps，逐 batch）
- 完整性：preemption 增量=0、无 OOM、记录 KV cache 占用

**batch 定义**：burst 模式=固定 batch（同刻发 B 请求测 batch prefill）；sustained 模式=B 并发在飞（稳态≈batch）。batch 扫描 1/4/8/16/32；256k 档 KV 预算受限，批上限按 preflight 收窄（如 1/2/4），避免 preemption 污染数据。

**CPU/IO 侧**：客户端 prompt 构建+发送耗时单独记录（256k≈1.7MB/请求，防 client 成瓶颈）；服务端 tokenizer CPU 采样；TP4 另采各节点 nvidia-smi 与网卡计数。若 client 无法灌满负载，判无效并加线程。

## 二、对比矩阵

| 组 | 拓扑 | 镜像/版本 | 隔离变量 |
|---|---|---|---|
| A：TP2 基线 | 2 机（.58+.60） | 现有 cu130 / 0.26.1.dev0 | —（历史口径） |
| B：TP4 | 4 机（.55/.59 有线组网） | 同 A（固化镜像） | 仅组网+TP 并行度 |
| C：TP4+cu132（可选） | 4 机 | .55 cu132 镜像 | 仅软件栈 |

A→B 隔离"组网收益"，B→C 隔离"全栈收益"，A→C 总收益。三组同脚本、同温度、同随机前缀口径。

## 三、用例表

| # | 用例 | 输入 | 指标 | 验收 | 耗时 |
|---|---|---|---|---|---|
| C0 | 预检/校准 | 生成 calib_prefill.json；preflight max_model_len/KV/批上限 | token 计数偏差<2% | 全过 | 30min |
| C1 | A 基线矩阵 | 8k~256k × batch 1~32，mt=32，temp=0.6，3轮中位 | prefill_tps、TTFT p50/p95、preemption | 与历史口径对齐 | 2-4h |
| C2 | B：TP4 矩阵 | 同 C1 | 同上 | 见验收标准 | 2-4h |
| C3 | 长 ctx 衰减 | 128k/256k × batch 1/2/4 | 各档 prefill_tps、缩放比 | 记录 256k vs 8k 衰减率 | 1h |
| C4 | NCCL 正确性 | world=4，all-reduce/bcast/all-gather 已知张量，128M/512M/1G | 误差=0、带宽 GB/s | 全过 | 30min |
| C5 | TP4 拉起 SOP | head 先起+轮询 4 机 /health 与 rank | 4/4 ready、TP=4 日志、worker 分布 | 30s 内 ready | 20min |
| C6 | 断线/恢复 | 断 1 rank→观察→恢复 | 客户端超时非挂死；恢复后复测 C2 抽查 | 吞吐±5% | 30min |
| C7 | cu132 全栈 | C 组重跑 C2 | 同上 | ≥0.95×B | 2-4h |

## 四、脚本改造

**final_matrix.py**：加 `--prefill-only`（max_tokens 默认 32、ctx 默认 8192,32768,131072,262144、--conc 语义=batch 扫描）；修随机前缀；拆分 prefill_tps/decode_tps；rounds 默认 3 取中位。

**新 prefill_bench.py 结构**：
```
gen-calib → preflight → burst(--batch --ctx) → sustained(--ctx --batches --window 60s --rounds 3) → report(--baseline)
```
示例：`python3 prefill_bench.py sustained --base http://<head>:8001 --ctx 32768 --batches 1,4,8,16,32 --max-tokens 32 --temp 0.6 --rounds 3 --out raw_tp4.json`

## 五、TP4 特有验证

- C4：扩展 nccl_probe.py 至 world=4；校验 rank/设备映射、all-reduce 数据一致性、大消息带宽计时（记录实测 GB/s 作为后续基线）
- C5：head 先起→轮询 4 机 /health 与 rank 日志→全部 ready 再压测；日志确认 TP=4、权重分片分布
- C6：断 1 rank 网络观察客户端超时行为（必须超时而非挂死），恢复后复测 C2 抽查

## 六、预期值设定方法（不臆造数字）

- prefill 计算受限 → TP4 聚合算力=2×TP2 → 理论上限≈2×TP2 的 prefill_tps；comm（activation all-reduce 随 seq×batch 增长）与内存带宽使其衰减，256k/大 batch 更明显
- 区间：上限 2.0×、验收下限 1.25×；<1.1× 判组网/并行效率异常
- decode 内存受限，TP4 收益小于 prefill，不作为主判据
- cu132：单 kernel ±2% → 端到端区间 [0.95,1.15]×B；下限 0.95（升级主因是正确性/可维护性而非性能）
- 判定法：3 轮中位数 + 90% 置信区间，比率下界>1.0 才判胜

## 七、验收标准

1. 方法学齐备：随机前缀、temp=0.6、3 轮中位、新旧同口径
2. C1 与历史 total_tps/prefill 口径可对齐
3. C2 prefill_tps ≥1.25×C1（全档均值）；256k 档 ≥1.1×
4. 同 batch 下 TP4 TTFT p95 不劣于 TP2×1.1（并行不得显著增延迟）
5. NCCL 0 误差；preemption=0；无 OOM
6. C7 ≥0.95×C2，无正确性回退
7. C6 无挂死；恢复后复测在 ±5% 内
