# ② c5 并发档恢复执行方案（decode 崩塌 7.01 → ≥35）

- **编制**：testing-expert-1（测试专家 Tessa）
- **日期**：2026-08-14（v2 修订：C1–C4 已拍板，C3 接受并发限 ≤c4）
- **前置**：`test-0.2.1-v027` TP4 冒烟通过（或按窗口降级到 0.26 同参执行，判定口径一致）
- **GPU**：TP4 全量占用
- **预估总耗时**：**2–2.5 h**（基线复测 30min + B1–B4 每变体 ~25–30min；H1 证实后 c5 档不再追加复测）

---

## 0. 问题定义

**症状**：c5（conc=5）@131K decode p50 崩塌——
- 生产基线：c5@131K PR 595.53 / **DE 7.01**；c5@32K PR 678.68 / DE 16.95
- 对照（r12 全矩阵 0.26）：c1@131K DE 104–115；c3@131K DE 45；c4@131K DE 37.45；c5@131K DE 7.8/8.2/4.2（coding/json/prose，TPOT 0.121–0.236s）
- 注：c5 峰值 decode 仍可达 80–87（单请求窗口正常），说明**不是硬件/内核级崩塌**，而是调度/排队层面的服务质量劣化。

**目标**：c5@131K DE p50 **≥ 35**（c4 水平），且 c1/c4 不回退、PR 不劣化 >5%。

---

## 1. 根因假设（决定 A/B 方向）

| # | 假设 | 证据 | 对应手段 |
|---|---|---|---|
| H1 | **prefill 排队稀释**：asyncio bench 串行 prefill，131K×5 并发时 decode 请求被长 prefill 阻塞，DE p50 被 TTFT 稀释（c1 单流 131K decode 恒定 73.8 即证明非带宽瓶颈） | review-mla-compression-decode-collapse（决定性证据） | 记录 TTFT 逐 wave；B2（batched-tokens）让 prefill 可切块、decode 可插队 |
| H2 | **seqs=6 槽位压力**：5 个长 ctx 请求占满 prefill，decode 无槽位/被抢占 | r12「seqs=6 约束」 | B1（seqs 6→8） |
| H3 | **投机开销放大**：高并发下 dspark draft+verify 额外计算/通信放大 decode 步延迟 | draft 在长 ctx 下不「便宜」 | B3（收窄 spec / 关投机对照） |
| H4 | **CUDA graph 重捕获**：稳态 batch 72 > capture 64 → 重捕获停顿 | r12「72 截断 bug」 | B4（capture 加 72 档） |

> **判读要点**：每档同时记录 TTFT p50（逐 wave）。若 TTFT 占主导而「纯 decode 段」本身不低，则 c5 崩塌主要是**测量口径 + 调度**问题，恢复手段应侧重 B2/B1；若纯 decode 段也低，则侧重 B3/B4。

---

## 2. A/B 矩阵（每变体独立重启验证）

> 执行方式：复制 0.27 测试启动脚本 → 改目标参数 → 备份 `.bak-c5B<序号>` → 顺序停 → 起 → health → warmup → 矩阵 → 记录 → 回滚/下一档。

| # | 参数变更 | 生产基线 | 试验值 | 假设 | 风险 |
|---|---|---|---|---|---|
| B0 | 基线复测 | seqs=6 / batched=4096 / spec 动态K / capture≤64 | 同生产 | 同窗对照 | 低 |
| B1 | `--max-num-seqs` | 6 | **8** | H2 槽位 | 中（KV/显存：fp8 KV 减半容量） |
| B2 | `--max-num-batched-tokens` | 4096 | **8192** | H1 切块/插队 | 中（chunk 放大，prefill 干扰 decode） |
| B3a | spec 动态K 窗口 | `[[1,1,5],[2,4,4],[5,6,3]]` | `[[1,1,5],[2,3,3],[4,5,3]]`（收窄） | H3 | 中（投机质量↓风险） |
| B3b | spec 档位 | 动态K | `num_speculative_tokens=4`（固定） | H3 | 中 |
| B3c | spec 开关 | 开 | **关闭（对照）** | H3 上限 | 中（DE 可能↓但可判 H3） |
| B4a | cudagraph capture | `--max-cudagraph-capture-size 64`，sizes 1..64（含 36） | `--max-cudagraph-capture-size 72` + sizes 追加 72 | H4 | 低 |
| B4b | cudagraph capture | 同上 | `--max-cudagraph-capture-size 96`（若显存允许） | H4 | 低-中 |

**每变体标准流程（~25–30 min）**：

```bash
# 1) 改参数 → 备份 → 重启（SRE 执行容器切换；改 start 脚本中 SERVE_CMD 相应项）
cp start_tp4_head_b12x_ab.sh .bak-c5B1    # 以 B1 为例
#    修改 --max-num-seqs 6 → 8（worker 变体同步）
#    顺序停 → 起 → curl health 200 → warmup 2 个短请求

# 2) 主矩阵（c5 判据单元格 + c1/c4 回退检查）
python3 bench_prefill_decode_async.py --group C5B1 \
  --endpoint http://<NODE_IP>:8001/v1 --key <KEY> \
  --model deepseek-v4-flash-0731 \
  --concurrency 5 --ctx 32768,131072 --tasks coding,json,prose \
  --rounds 3 --engine asyncio --out ./results_c5_b1
# 回退检查（c1/c4 不回退）
python3 bench_prefill_decode_async.py --group C5B1CTL \
  --endpoint http://<NODE_IP>:8001/v1 --key <KEY> \
  --model deepseek-v4-flash-0731 \
  --concurrency 1,4 --ctx 131072 --tasks coding \
  --rounds 3 --engine asyncio --out ./results_c5_b1_ctl

# 3) 记录（含 TTFT 逐 wave）
#    summary_C5B1.json 中取 c5@131K coding p50_decode_tps / p50_prefill_tps / p50_ttft_s
#    附：若需要 wave 级分解，读 rows_C5B1.csv 按 wave 聚合 TTFT 与 decode 段

# 4) 判定
#    PASS：c5@131K DE ≥35 且 c1@131K DE ≥104.1×0.95（≥98.9）且 c4@131K DE ≥37.45×0.95（≥35.6）
#          且 c5@131K PR ≥595.53×0.95（≥565.8）
#    FAIL：任一不满足 → 回滚 .bak，记录，进入下一档
```

**判定汇总模板**：

| # | c5@32K DE | c5@131K DE | c5@131K PR | c1@131K DE | c4@131K DE | TTFT p50 特征 | 判定 |
|---|---|---|---|---|---|---|---|
| B0 基线 | 16.95 | 7.01 | 595.53 | 104.1 | 37.45 | 长 prefill 排队 | 基准 |
| B1 (seqs8) | | | | | | | |
| B2 (bt8k) | | | | | | | |
| B3a (窄K) | | | | | | | |
| B3b (fixK4) | | | | | | | |
| B3c (spec off) | | | | | | | |
| B4a (cap72) | | | | | | | |
| B4b (cap96) | | | | | | | |

**组合策略（若单档不足）**：当单档无法达标时，允许 **1 次组合复测**（如 B1+B2 或 B2+B4a），组合档记 `C5B1B2`，+30min；组合档判定同样按 PASS 条件。

---

## 3. 结论规则与后续

1. 选「c5@131K DE 最高且 c1/c4 无回退」的档位（或组合）为**候选恢复参数集**。
2. 候选参数集需在 0.26 生产同参复测确认（若 0.27 与 0.26 参数面一致，则直接作为生产切换候选参数；若 0.27 特有，需在切换时同步）。
3. **C3 已拍板**：若所有档位均无法使 c5@131K DE ≥35，但 TTFT 证据支持 H1（纯 decode 段正常）→ 接受「**生产长 ctx 并发限制 ≤c4**」作为处置建议写入报告；**c5 档保留诊断不复测**（不再为恢复 c5 追加窗口）。
4. 恢复参数集与方案①/③结论合并 → 统一生产切换建议。

---

## 4. 回滚与恢复

- 每档回滚：`.bak-c5B<序号>` 还原 + 重启（~5min）。
- 全程不修改生产脚本；窗口结束恢复生产：
```bash
ssh node01 "cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh"   # 约 8 min
# 验证：8001=200 + 四机 healthy + PSR（NCCL→8-9、Engine→15-19）
```

---

## 5. 决策点（督导/用户已拍板）

- **C1（已拍板）**：B1（seqs 6→8）若触发 KV OOM/preemption，**默认不降** `--gpu-memory-utilization`、不缩短 `--max-model-len`（保持生产同参），记录 OOM 为失败原因。
- **C2（已拍板）**：B3c（关投机对照）仅用于归因 H3，**不参与候选恢复参数评选**。
- **C3（已拍板）**：H1 证实后接受「生产长 ctx 并发限制 ≤c4」为处置建议；c5 档保留诊断不复测（已写入 §3 规则 3）。
- **C4（已拍板）**：组合档多变量耦合局限**默认接受并文档化**。
- **执行顺序（team-lead 传达）**：③ v0.27 主矩阵 → 同窗 0.26 对照 → ① NCCL A/B → **② c5 诊断**。

> 本方案由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
