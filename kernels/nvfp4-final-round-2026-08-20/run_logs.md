# 最终交付包执行日志（2026-08-20 07:55–08:05）

## 输入
- `nvfp4-kernels-delivery-final.zip` → 容器 `/vllm-workspace/nvfp4-delivery-final/`
- 新增：kernel1_planB_pr42209_integration.md（方案 B）、kernel2_v17_safety_reliability.md + test_v17_safety.py

## kernel② v17
1. shipped `test_v17_safety.py`：**4 通过 / 3 失败**（saturation/sign_zero/boundary_T）
2. 逐项定性（探针 `probe_v17_edge.py`）：v17 与 torch ref 在 zeros/+0/-0/1e6/1e30/1e-30/6.0 **全部 byte_equal=True**（scale=24/24/24/144/224/24/127）
   → 3 失败均为测试脚本缺陷：saturation 期望 255 实为 144；sign_zero 期望 1 实为 24；boundary_T 未 seed 比了两次不同 randn
3. 消费端探针（`probe_kv_layout.py`）：FlashInfer nvfp4_kv_quantize 输出 [T,256] packed + [T,32] e4m3 scale，与 v17 584B/E8M0 **语义不同不字节兼容**（预期：服务不同 reader）
4. 结论：**v17 验证合格可替换**（正确性 8/8 + 带宽 194~262 GB/s + 边界全等；部署落实 R1/R2，R3 paged 待做；回退 v11）

## kernel① 方案 B
1. FlashInfer 0.6.15 API 面完整（mm_fp4/nvfp4_quantize/nvfp4_attention_sm120/nvfp4_kv_quantize 均在）
2. mm_fp4 契约逐步厘清：a 需 fp4 [M,K//2]、b 需 [K//2,N]、descale 需 uint8（e4m3 位模式）2D、nvfp4_quantize 输入需 bf16
3. backend 实测全阻塞：b12x/cute-dsl=CuTe DSL 未编入 wheel；trtllm=cap121 不支持；cudnn=描述符错误；cutlass=TVM 参数错误
4. 结论：**该 wheel 无 sm_121 可用原生 FP4 GEMM backend** → 方案 B 需完整 vLLM 源码构建 / 重编 FlashInfer wheel / FlashInfer TOT；SASS 门禁修正 mma.*e2m1|mmaf

## 交付
- deliverables/engineering-assurance/nvfp4-final-round-2026-08-20/（报告+日志+7 探针脚本+evidence）+ zip
- 生产 4 rank healthy GPU 0% 未恢复
