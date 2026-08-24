# NVFP4 双算子 完整测试分析资料 2026-08-19

## 目录结构
  00a_原始交付包_v10/  内核方首次交付（v10 prefill_gemm + kv_linear v4/paged）
  00b_原始交付包_v12/  内核方第二次交付（v12 分离量化架构 + MCP 验证报告）
  01_第一轮诊断_v8/    上包 v8 诊断：7 处缺陷 + 修复代码 + 失败日志
  02_第二轮双算子测试_v11/  生产实机测试产物（修复后内核 + 日志 + 完整报告）
  03_v12补充测试/       v12 分离量化架构测试（pytest 8/8 全 PASSED + 日志）

## 关键结论速览
  v8  → 无法编译（7 处缺陷，dot_scaled 接口不匹配）
  v10 → 6/8 PASSED（修复 rhs[K,N]布局 + uint8 scale + 不trans）
  v11 → 6/8 PASSED（舍入边界 M=1024 单行 0.1875 量化步长）
  v12 → 8/8 全 PASSED（分离量化架构 + 舍入修正 `>` 阈值链，历史首次全通过）
  kernel② kv_linear: v11 6/7 精确 + speedup 10-41× avg 22×
  kernel② paged: v11 5/5 全精确（逐字节 atol=0）
  生产 4 rank 全程 healthy，GPU 0%，未恢复

## 推荐部署版本
  kernel① prefill_gemm: v12 分离量化架构
  kernel② kv_linear:   v11 修复版
  kernel② kv_linear paged: v11 修复版
