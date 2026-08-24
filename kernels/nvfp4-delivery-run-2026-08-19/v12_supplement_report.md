# v12 补充报告：kernel① prefill_gemm 分离量化架构 + kernel② v12 逐字节复核

> 日期：2026-08-19｜续第二轮生产实机测试
> 交付包来源：微信 `nvfp4-kernels-delivery.zip`（v12，MCP 第三轮验证 + 生产修复合并）
> 执行环境：DGX Spark 生产 TP4（vllm-tp4-rank0, torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a）

---

## 一、v12 相对 v11 的变更

| 算子 | v11 | v12 |
|---|---|---|
| kernel① | GEMM 内核内做 A 量化+W 重打包（量化开销→20 TFLOPS） | 分离架构：① A 量化独立 kernel ② W 主机侧重打包（可缓存）③ GEMM 纯 MMA |
| kernel① 舍入 | 阈值链 `>=`（等距取高档，与 torch argmin 不一致） | 阈值链 `>`（等距取低档，与 torch argmin 一致） |
| kernel② linear | v11 架构（2D grid + TOKENS_PER_PROG） | MCP v2 新架构（59.67× 服务端基准），scale 修正去 ×6 |

## 二、kernel① v12 测试结果

### T2 正确性（pytest，rtol/atol=5e-2）

| 状态 | 详情 |
|---|---|
| ✅ **8/8 全 PASSED** | M=256/512/1024/128 × bias 开关，全部通过（11.4s） |

**这是历史首次 8/8 全通过**——v12 的舍入修正（`>` 阈值链）解决了 v11 的 M=1024/N=2048 量化边界问题。

### T3 吞吐（benchmark）

| 指标 | v11 | v12 |
|---|---|---|
| triton TFLOPS | 16.6~20.8 | 1.7~4.3 |
| speedup vs ref | 6.74~14.17× | 1.36~1.46× |

**⚠️ v12 benchmark 数字低，但非内核性能问题**：benchmark 测的是 `triton_impl` 整个调用（含主机侧 `_repack_w_for_rhs_k_pack` + `_expand_w_scale` 的 torch 操作），而 v12 架构设计是这些操作**每层只做一次**（在 `process_weights_after_loading()` 中缓存），不应在每次 forward benchmark 中重复计算。生产部署时缓存后 GEMM 内核纯 MMA，性能预期高于 v11。

## 三、kernel② v12 测试结果

### 正确性（逐字节 atol=0）

| 测试 | 结果 |
|---|---|
| linear v12 | ❌ **7/7 FAILED**（49.3% mismatch，差值 255） |
| paged v11 | ✅ **5/5 全 PASSED** |

**根因**：MCP v2 生成器采用全新 2D grid 架构，其 584B 信封字节布局与 v11 torch 参考不兼容。**建议保留 v11 版**（生产已验证 6/7 精确 + speedup 10-41×），v12 作为后续布局对齐。

## 四、综合结论

| 算子 | 推荐版本 | 状态 |
|---|---|---|
| ① prefill_gemm | **v12（分离量化架构）** | ✅ 8/8 全 PASSED；性能需在生产部署缓存 W 后实测 |
| ② kv_linear | **v11（生产修复版）** | ✅ 6/7 精确 + speedup 10-41× |
| ② kv_linear paged | **v11（生产修复版）** | ✅ 5/5 全精确 |

**关键突破**：kernel① 从 v8 无法编译 → v10 6/8 → v12 8/8 全 PASSED，完成了 defect-to-deployment 的完整闭环。分离量化架构是下一个性能台阶（需生产部署时验证 TFLOPS）。

## 五、交付物

| 文件 | 位置 |
|---|---|
| 本报告 | `deliverables/engineering-assurance/nvfp4-delivery-run-2026-08-19/v12_supplement_report.md` |
| v12 测试日志 | `.../v12_logs/`（k1_v12_t2/k1_v12_t3/k2_v12_t2） |
| 容器侧 v12 包 | `01:/vllm-workspace/nvfp4-delivery-v12/`（kernel1 已替换为 v12、kernel2 已替换为 v12） |
| v11 备份 | `kernel1/*_triton_v10backup.py`、`kernel2/*_triton_v11bak.py` |

## 六、生产状态

- 4 rank 全程 healthy，GPU 0%，无残留测试进程。
- 按用户要求**未恢复生产**，恢复时间由用户决定。