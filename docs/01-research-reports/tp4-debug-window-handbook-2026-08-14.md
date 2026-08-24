# 调试窗口执行手册（① NCCL 参数 A/B + ② c5 复测 + ③ v0.27 验证）

- 日期：2026-08-14 15:20 编制
- 前置：vLLM 0.27.1 编译（vllm027-build 容器，后台）+ 测试镜像构建完成
- GPU 状态：TP4 已停（systemctl SOP），GPU 已释放交其他工作组；**本手册在调试窗口回切 GPU 时执行**

---

## 0. 环境基线（不得修改项）

- 镜像：<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0（生产同款）
- 启动：01 `bash start_tp4_cluster.sh`（head-first，需 VLLM_API_KEY 环境变量）
- shim v8：<INSTALL_DIR>/lib/libncclpin.so（ce43c688，**保持不动**；v9 可选细分默认不做）
- NCCL 基线 env：ALGO=RING / MIN_NCHANNELS=2 / NET_PLUGIN=none / MERGE_NICS=0 / IB_PEER_HCA per-peer / GID_INDEX=3 / RETRY_CNT=7 / TOS=46 / SUBNET_AWARE_ROUTING=1
- 基准脚本：bench_prefill_decode_async.py（per-request p50 口径）
- 参考基线（c1@131K）：PR/DE=1896.4/104.1；c4@131K：635.96/37.45；c5@131K：595.53/7.01（崩塌待恢复）

## 1. ① NCCL 通信参数 A/B（直击 368KB all-reduce）

执行方式：改 `start_tp4_head_b12x.sh` 中容器 env（NCCL_* 行）→ 重启容器 → nccl-tests + bench 对比 → 记录 → 下一档。**每档先备份 env（.bak-ncclA<序号>）**。

| # | 参数变更 | 依据 | 预期 |
|---|---|---|---|
| A1 | NCCL_MIN_NCHANNELS 2→4 | 环网双 HCA，多通道并行 | busbw 4.4→6-8 GB/s |
| A2 | NCCL_MIN_NCHANNELS 2→8 | 通道数上限测试 | busbw 上限探测 |
| A3 | NCCL_IB_QPS_PER_CONNECTION=4 | 每连接多 QP 提升小消息并发 | all-reduce 延迟↓ |
| A4 | NCCL_BUFFSIZE=8M/16M | 368KB×N 大消息聚合 | 大消息吞吐↑ |
| A5 | NCCL_IB_SPLIT_DATA_ON_QPS=1 | QP 数据拆分 | 与 A3 组合评估 |
| A6 | （谨慎）MERGE_NICS=1 | 双 NIC 合并 | **风险：破坏 per-peer 对口，默认不做** |

验证矩阵（每档）：
1. `nccl-tests all_reduce`（4 rank，16MB-512MB 消息，busbw）
2. `bench_prefill_decode_async.py --group X`（c1@131K PR/DE）
3. 判定：busbw ↑ 且 PR/DE 不劣化 → 记录为候选；劣化 → 回滚 env（.bak 还原 + 重启）

## 2. ② c5 并发档恢复

- 目标：c5@131K DE 从 7.01 → ≥35（c4 水平）；PR 不降
- 手段（依序 A/B，同窗口）：
  - B1：max-num-seqs 6→8（更多并发槽）
  - B2：max-num-batched-tokens 4096→8192（chunk 放大）
  - B3：dspark 动态K 窗口 [[1,1,5],[2,4,4],[5,6,3]] → 尝试收窄 spec 档位
  - B4：cudagraph-capture-sizes 检查（含 32/36 附近档位）
- 每项记录 c5@32K/131K PR/DE，与基线对比

## 3. ③ vLLM 0.27 验证（编译完成后）

1. **镜像构建**（01）：基于 NGC 26.07 + 编译产物 → `test-0.2.1-v027`（或基于 0.2.1-v026.0 overlay）；push registry → 四机 pull + md5 校验
2. **冒烟**：TP1 起服（DeepSeek-V4-Flash 权重，不加载 NVFP4 转换）→ /health 200 + 1 个推理请求
3. **性能 A/B**：c1/c3 @32K/131K PR/DE vs 生产基线（1896.4/104.1 等）
4. **优化项验证**：B12x Direct M=1（#4495 预期 1.668×）、空 c128 skip（#48957）、topk/router skip（#49486）、q-head padding 移除（#48047）
5. **质量抽查**：GSM8K 抽样 / 自洽性（与生产输出对比）
6. 判定：PR 提升 ≥20% 且质量无劣化 → 生产切换候选；否则维持 0.26

## 4. 回滚与恢复（每轮测试后）

```bash
# 恢复生产（任意时间点可执行，约 8min）
ssh node01 "cd <INSTALL_DIR>/scripts && bash start_tp4_cluster.sh"   # head-first
# 验证：8001=200 + 四机 healthy + PSR（NCCL→8-9、Engine→15-19）
```

## 5. 状态记录
- [x] 14:5x TP4 停机（GPU 释放）
- [x] 15:15 vLLM 0.27.1 源码（gh-proxy 39MB）
- [x] 15:17 编译容器 vllm027-build 启动（后台）
- [ ] vLLM 0.27.1 编译完成（预期 2-4h，日志 <INSTALL_DIR>/backup/vllm027-src/vllm-0.27.1/build.log）
- [ ] test-0.2.1-v027 镜像构建 + 四机分发
- [ ] NCCL A/B 首档（A1）执行
- [ ] c5 B1-B4 执行
