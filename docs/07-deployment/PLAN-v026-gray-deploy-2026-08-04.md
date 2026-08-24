# v0.26 定制镜像灰度部署预案（dspark-vllm-gx10:0.2.0-v026.0）

**日期**: 2026-08-04 ｜ **SRE**: Rex ｜ **状态**: 构建中，待镜像就绪后实施
**镜像**: `ghcr.io/anemll/dspark-vllm-gx10:0.2.0-v026.0`（vLLM 0.26.0 @ 568afb3a，overlay-v026 13 文件定制）
**基础镜像**: `nvidia/cuda:13.0.2-devel-ubuntu22.04@sha256:1c517d4f...`（本地已复用，无 docker.io 拉取）

---

## 1. 构建状态跟踪
- [x] 基础镜像本地就绪（digest 1c517d4f，buildx `#6 CACHED` 复用）
- [x] overlay-v026 13 文件齐全、upstream.lock v0.26.0@568afb3a
- [x] 构建后台重启（wrapper PID=2788923，setsid+nohup，日志 ~/v026-build.log）
- [ ] 镜像构建完成（预计 2-4h 编译期）
- [ ] 镜像 digest resolve 并 pin（沿用 ADR-0011，`docker inspect --format '{{index .RepoDigests 0}}'`）

## 2. 灰度 SERVE_CMD 配置（用户决策：dspark 投机解码改 greedy）

**核心变更**：`draft_sample_method=probabilistic`（生产/0.1.1 同口径）→ **`draft_sample_method=greedy`**（官方推荐配方，A/B 对比验证）。

```bash
vllm serve /models/deepseek-v4-flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --kv-cache-dtype nvfp4_ds_mla \
  --max-model-len 1048576 --max-num-seqs 6 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
  --moe-backend flashinfer_b12x \
  --distributed-executor-backend mp \
  --enable-flashinfer-autotune \
  --tensor-parallel-size 2 --nnodes 2 \
  --master-addr <MASTER_ADDR> --master-port 25000
```

**决策记录**：
- `num_speculative_tokens=5` **保持不变**（=2×DGX 社区标准，与生产同口径，保证 A/B 单一变量）
- 仅 `draft_sample_method` 改 greedy；`dspark_block_size` 不加（pydantic 白名单，沿用 0.1.1 经验）
- 备选：官方配方 num_spec=7 可作 stretch 变体，**但须在 greedy@5 vs probabilistic@5 首轮对比结束后再测**（避免一次改两变量污染 A/B）

## 3. SRE 对 greedy 与 target 采样一致性的判断

**结论：无正确性风险，建议采纳；接受率以灰度实测为准。**

1. **理论正确性**：投机解码 rejection sampling 对 draft 分布无偏性要求不因 greedy 破坏——greedy 使 draft 分布成为 argmax 点质量 q(x*)=1，接受规则 min(1, p(x*)/q(x*))=p(x*)，输出仍服从 target 分布 p。**不引入分布偏移。**
2. **接受率预期**：dspark 为高精度 draft，top-1 与 target 对齐时接受率≈p(x*)；当 target 用 temperature 采样时，在低置信（分布平坦）位置 greedy 与采样可能分歧，接受率略降但速度收益仍为正。整体通常持平或优于 probabilistic（官方推荐依据）。
3. **监控判据**（灰度期核心指标，与生产 probabilistic 基线对比）：
   - spec decode acceptance rate（vllm spec decode 指标，须在 preflight 核对指标名）
   - `time_per_output_token_seconds`（ITL）、TTFT、decode 吞吐
   - 回滚触发：acceptance/ITL 显著劣于生产基线（阈值按首轮 15min 基线数据定）
4. **快速回滚优势**：draft_sample_method 是容器启动 SERVE_CMD 参数，**改回 probabilistic 无需重建镜像**，仅重启容器即可——灰度风险低。

## 4. Preflight 待核对（构建完成后）
- [ ] greedy 与 `VLLM_DSPARK_LOCAL_ARGMAX=1` 兼容性（argmax 路径 vs greedy sample 是否互斥/冗余）
- [ ] vLLM 0.26 metrics 名核对（沿用 0.1.1 经验，prometheus-alerts.yml expr 必要时微调）
- [ ] 镜像 digest 双机核对（ADR-0011）
- [ ] 双机同镜像同版本（ZMQ bind 对齐），worker 先 → sleep 12 → head

## 5. 回滚锚点
- 回滚目标：生产 probabilistic 配置（0.1.1 / 现网），或 `start_*_D.sh` 已知良好基线
- 回滚 = 改 SERVE_CMD 重启容器（config-only，RTO ≤15min，ADR-0008 流程）
