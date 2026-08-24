# kernel② KV-Linear v17 替换手册（v11 → v17）

> 版本 v2026-08-20 | DGX Spark / vLLM 0.26 | kernel② = NVFP4 DS-MLA KV-Linear

## 一、验收结论

**v17 全面合格，同意替换 v11**（2026-08-20 生产实测）：
- 正确性 **8/8 逐字节**（shipped test）
- 带宽 **大 T 194~262 GB/s**（4680B/token，3.5~4.6× v11，T=1024 达 96% 理论）
- 边缘 case：zeros/±0/1e6/1e30/1e-30/6.0 与 torch ref 全 byte_equal（scale 24/24/24/144/224/24/127）

## 二、安全审计结果

- **4 项通过**：确定性、显存无增长、NaN 不崩溃、数值。
- **shipped 安全套件 3 FAIL = 测试脚本缺陷**（非内核）：
  - `test_saturation`：期望 scale=255，正确值 **144**（floor(log2(1e6/6))+127）
  - `test_sign_zero`：期望 scale=1，正确值 **24**（amax clamp 1e-30）
  - `test_boundary_T`：未 seed，v17 与 ref 各吃不同 randn，比对无效
- 修正期望值/seed 后全绿（探针已证）。

## 三、部署步骤

1. **文件**：`nvfp4_ds_mla_kv_linear_v17_triton.py` + `test_nvfp4_ds_mla_kv_linear_v17.py`（及 safety 变体）。来源：交付包 `/vllm-workspace/nvfp4-delivery-final/kernel2-nvfp4_ds_mla_kv_linear/`。
2. **放置**：四节点生产可引用路径（待 SRE 落地 <INSTALL_DIR>，见下方持久化）。
3. **切换**：调用点把 `nvfp4_ds_mla_kv_linear_triton`（v11）换成 `_v17_triton`。保留 v11 文件作为回退。
4. **验证**：
   ```bash
   python3 -m pytest test_nvfp4_ds_mla_kv_linear_v17.py -q    # 8/8
   python3 benchmark_nvfp4_ds_mla_kv_linear_v17.py            # GB/s
   ```
5. **四节点一致性**：分发到 01-04，md5 校验。

## 四、部署加固（R1/R2/R3）

- **R1**：可选加 `isfinite` 断言（NaN 防护），默认关闭（不引入额外开销）。
- **R2**：CUDA Graph 捕获前 warmup 一次（避免首次编译进 graph）。
- **R3**：paged v17 变体**待做**（当前 paged 维持 v11）。若需要 paged 加速，另行评估。

## 五、回退

- 恢复调用 `_triton`（v11）即回退；文件留存，无编译依赖破坏。

## 六、审计文档

- 官方：`kernel2_v17_safety_reliability.md`（交付包）
- 我方探针：`probe_v17_edge.py`

## 附：v17 带宽数据（4680B/token，理论 273）

| T | v11 | v17 | 提升 | 理论占比 |
|---|---|---|---|---|
| 1024 | 56.8 | **262.3** | 4.62× | 96% |
| 4096 | 61.0 | 248.9 | 4.08× | 91% |
| 16384 | 58.8 | 227.0 | 3.86× | 83% |
| 65536 | 55.2 | 194.3 | 3.52× | 71% |