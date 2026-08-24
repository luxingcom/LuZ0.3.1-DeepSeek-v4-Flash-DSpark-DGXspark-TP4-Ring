# v12.1 最终汇总报告：kernel② linear grid 修复 + kernel① 真实性能

> 日期：2026-08-19｜第四轮（v12.1）生产复测  
> 交付包：微信 `nvfp4-kernels-delivery(1).zip` + `report_round4_v12.1_fix.md`  
> 执行环境：DGX Spark 生产 TP4（vllm-tp4-rank0, torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a）



---

## 一、测试结果总览

| 测试项                                    | 结果                                | 对比                                                |
| -------------------------------------- | --------------------------------- | ------------------------------------------------- |
| kernel① v12 pytest                     | ✅ **8/8 全 PASSED**                | 历史首次（v12 舍入修正）                                    |
| kernel① v12 **真实性能**（缓存 W 后 GEMM-only） | ✅ **20.5~45.8 TFLOPS**            | 比上一轮 1.7~4.3 提升 ~~10×，比 v11 16.6~~20.8 提升 ~1.2-2× |
| kernel② v12.1 linear pytest            | ✅ **7/7 全 PASSED**                | v12 曾 7/7 FAIL → **grid 64 修复彻底解决**               |
| kernel② v12.1 benchmark                | ✅ speedup 3.72~22.96×（avg 11.28×） | 受伴跑 vLLM 影响低于服务端 59.67× 基线                        |
| kernel② paged v11                      | ✅ 5/5 全 PASSED                    | 维持                                                |
| 生产影响                                   | ✅ 无                               | 4 rank healthy、GPU 0%、无残留                         |

---

## 二、kernel② v12.1 关键修复（grid 64）

第四轮报告根因定位准确：**MCP v2 生成器 grid 第二维用 32**，只覆盖 K/V 各前 16 组（K/V 各 512 元素 = 各 32 组），后 256 元素+scale 全缺失 → 49.3% mismatch。

**v12.1 修复**（生产已核验）：

```
grid 第二维 32 → 64
is_v: pid_g >= 16 → pid_g >= 32
g_half: pid_g - 16*is_v → pid_g - 32*is_v（0..31）
```

信封布局 [K_data,V_data,K_scale,V_scale,pad] 与 v11/PR#46329 完全一致。

**生产验证**：pytest 7/7 逐字节全 PASSED（atol=0），修复彻底。

## 三、kernel① v12 真实性能（P0 验证完成）

上一轮 benchmark 1.7~4.3 TFLOPS 的瓶颈确认是 **主机侧 W 重打包在 benchmark 循环内重复执行**。本轮**缓存 W 变换后只测 GEMM 内核**：

| shape          | GEMM-only TFLOPS | speedup vs ref |
| -------------- | ---------------- | -------------- |
| 256×4096×4096  | 29.9             | 24.28×         |
| 512×4096×4096  | 39.4             | 19.21×         |
| 1024×4096×4096 | 43.7             | 14.57×         |
| 256×8192×8192  | 20.5             | 14.85×         |
| 512×8192×8192  | 44.7             | 19.13×         |
| 1024×8192×4096 | 45.8             | 15.22×         |
| 256×4096×16384 | 42.4             | 30.27×         |

**结论**：

- ✅ GEMM 内核真实 TFLOPS = **20.5~45.8**，明显高于 v11（16.6~20.8）——分离量化架构生效
- ✅ 缓存 W 是**必须的部署动作**（vLLM `process_weights_after_loading()` 中缓存 `W_packed_rhs` + `W_scale_rhs`）
- ⚠️ 仍低于 400 TFLOPS 目标——后续需 SASS 确认 `mmaf_scaled` 是否真正生效（P0），若降级需换 Triton 版本/CUTLASS

## 四、推荐部署矩阵（最终）

| 算子                | 推荐版本                  | 状态                                                     |
| ----------------- | --------------------- | ------------------------------------------------------ |
| ① prefill_gemm    | **v12**               | ✅ 8/8 正确性 + 20-46 TFLOPS 真实性能；需 SASS 确认 FP4 MMA 后冲 400 |
| ② kv_linear       | **v12.1**（grid 64 修复） | ✅ **7/7 逐字节全 PASSED**，性能架构更优                           |
| ② kv_linear paged | **v11**               | ✅ 5/5 全精确                                              |

## 五、交付物

| 文件         | 位置                                                                                      |
| ---------- | --------------------------------------------------------------------------------------- |
| 本报告        | `deliverables/engineering-assurance/nvfp4-delivery-run-2026-08-19/v121_final_report.md` |
| v12.1 测试日志 | `.../v121_logs/`（k2_v121_t2 / k2_v121_bench / k1_v12_real_perf）                         |
| 容器侧 v12.1  | `01:/vllm-workspace/nvfp4-delivery-v12/`（kernel1=v12、kernel2=v12.1）                     |

## 六、生产状态

- 4 rank 全程 healthy，GPU 0%，无残留测试进程。
- 按用户要求**未恢复生产**，恢复时间由用户决定。
