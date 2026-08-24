# v17 轮执行日志（2026-08-20 00:51–00:56 GMT+8）

## 输入
- `nvfp4-kernels-delivery(5).zip` → 容器 `/vllm-workspace/nvfp4-delivery-v17/`
- kernel② v17（手写优化：S1 多 token/多组 + M1 连续向量化 + S3 zeros 内联 pad）
- kernel① 文件与 #4 完全一致（diff 确认 v16 triton / .cu 均 SAME）→ 双路径 = Triton v16 fp8（A）+ CUTLASS（B）

## kernel② v17
1. shipped pytest：**8/8 PASS**（逐字节）
2. 带宽对照（4680B/token）：v17 **75.6~262.3 GB/s** vs v11 50.8~61.0 → 大 T **3.5~4.6×**；T=1024 达 262.3（96% 理论）
3. 配置归因 T=65536：BLOCK_G=32/TPP=1/warps=1 = **213.5 GB/s**（最优）
4. 验收 ≥120 GB/s：达标（194~262）

## kernel① 双路径
1. 路径 A v16-fixed（F1+F2，#4 轮完成）：8/8 误差 0；0.1~0.2 TFLOPS（bf16+模拟）
2. **新探针（决定性）**：普通 tl.dot 配 fp8 e4m3 → PTX `mma.sync...f32.f16.f16`（FP16 降级）、SASS `HMMA.16816.F32` 无 E4M3 类型 → **Triton 3.6.0 sm_121 无任何原生 FP8 MMA codegen**（scaled/plain 双降级）→ v16 思路 3.6 上不可救
3. 路径 B CUTLASS：.cu 无更新；B1/B2/B3 阻塞不变；round12 判改道 PR #42209
4. A/B 结论：A 组唯一可用 = v15（26.7~81.4）；v16 弃用；400 目标仅 B 组（PR #42209/Triton 3.7+）

## 交付
- deliverables/engineering-assurance/nvfp4-v17-round-2026-08-20/（报告+日志+脚本+evidence）+ zip
- 生产 4 rank healthy GPU 0% 未恢复
