# nvfp4-testkit 生产环境执行诊断报告

> 日期：2026-08-19｜执行环境：**DGX Spark 生产 TP4 集群（vllm-tp4-rank0~3）**
> 任务包：`dgxspark-nvfp4-testkit`（NVFP4 4W4A 算子 v8 方案，目标 ≥400 TFLOPS prefill）
> 结论：**测试未完成，卡在 7 处必须修复的缺陷**。生产环境运行正常，未做任何恢复。

---

## 一、执行摘要

| 测试项 | 状态 | 说明 |
|---|---|---|
| T0 环境检查 | ✅ **通过** | torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a，`dot_scaled` 存在 |
| T1.1/T1.2 权重转换 | ⏸️ **未执行**（按决策仅小样本；受 T2 卡住未推进） | checkpoint 已定位 `/models`，已有 08-13 的 NVFP4 目录 |
| T2 内核正确性 pytest | ❌ **8/8 全挂** | 脚手架 + 参考实现 + 内核共 7 处缺陷，最后卡在 `dot_scaled` 算法矛盾 |
| T3 内核吞吐基准 | ⏸️ **未执行** | 依赖 T2 修复后的内核 |
| 生产影响 | ✅ **无** | 4 rank 全程 healthy，GPU 0%，测试进程已清理 |

**关键教训**：测试包标注 "v2–v8 连续 7 轮正确性通过"，但那是在 KernelGen MCP 的 **CPU/GPU harness 脚手架**上验证的（报告自述 benchmark harness 有"CPU/GPU 索引脚手架问题"）。**移植到生产 vLLM 容器（torch 2.11/triton 3.6）后暴露了一整串真实编译错误**——说明该内核从未在 GPU 实机编译跑通。

---

## 二、T0 环境检查输出

```
torch 2.11.0+cu130 | triton 3.6.0 | NVIDIA GB10
cc: (12, 1) expect sm_121a (12, 1)
has dot_scaled: True
```
✅ 完全满足测试包环境标准（torch≥2.9 / triton≥3.6 / CC=(12,1)）。

---

## 三、缺陷清单（按发现顺序，共 7 处）

### 已修复（6 处，修复后均可推进到下一编译阶段）

| # | 文件 | 行 | 缺陷 | 根因 | 修复 |
|---|---|---|---|---|---|
| 1 | `test_nvfp4_4w4a_prefill_gemm.py` `make_weights` | 25 | `RuntimeError: repeat dims 越界` | `w_scale_f` 已是 2D `[K//16,N//128]`，`unsqueeze(0).repeat(K,1)` 把 2D 当 3D | 改为先 `repeat_interleave(128,dim=1)` 再 `repeat_interleave(16,dim=0)` 摊到 `[K,N]` |
| 2 | `nvfp4_4w4a_prefill_gemm_torch.py` | 58-59 | `indices should be on same device (cpu)` | `e2m1_table` 建在 CPU，GPU 张量做索引 | 补 `e2m1_table = e2m1_table.to(W_packed.device)` |
| 3 | `nvfp4_4w4a_prefill_gemm_triton.py` | 104 | `Cannot bitcast size 64 to 32` | `tl.max` 累加把 fp32→fp64，`bitcast` 失败 | 前插 `scale_val = scale_val.to(tl.float32)` |
| 4 | 同上 | 13 | `NameError: cannot access global _E8M0_BIAS` | 全局常量用类型注解 `tl.constexpr=127`，非真 constexpr 实例 | 改 `tl.constexpr(127)` |
| 5 | 同上 | 106 | `tl.clamp` 不支持 int32 | Triton `tl.clamp` 仅浮点 | 改 `tl.minimum(tl.maximum(x,0),255)` |
| 6 | 同上 | 163,213 | `split() got multiple values` | Triton 3.6 `tl.split(a,_semantic=None)` 只接受 1 位置参，不再支持 `tl.split(x,2)` | 改 `tl.split(x)`（沿最后一维=2 分裂） |

### 未修复（1 处，算法级）

| # | 文件 | 行 | 缺陷 | 根因 |
|---|---|---|---|---|
| 7 | `nvfp4_4w4a_prefill_gemm_triton.py` | 253 | `dot_scaled() got unexpected keyword 'lhs_type'` | **Triton 3.6 的 `dot_scaled` API 与内核假设完全不符（见下节）** |

---

## 四、根因：`dot_scaled` API 与算法的根本矛盾（缺陷 7）

### 4.1 Triton 3.6 `dot_scaled` 实测签名
```python
dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
           acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=fp32)
```
关键约束（Triton 3.6 源码 `help(tl.dot_scaled)` 确认）：
- `lhs_format`/`rhs_format` 是**必填位置参数**，可取 `e2m1`/`e4m3`/`e5m2`/`bf16`/`fp16`
- **e8m0 scale 的 group_size 硬性 = 32**：`lhs_scale` 须 `[M, K//32]`，`rhs_scale` 须 `[N, K//32]`（`rhs_scale` 不转置）
- `rhs_k_pack=True`（SM120/121 必需，Triton issue #9678）——这一点内核用对了

### 4.2 内核 v8 的错误假设
```python
tl.dot_scaled(lhs_g, lhs_scale_g, rhs_g, rhs_scale_g,
              out_dtype=tl.float32, lhs_type="e2m1", rhs_type="e2m1", rhs_k_pack=True)
```
三个不匹配：
1. 参数名 `lhs_type`/`rhs_type` 不存在 → 应为 `lhs_format`/`rhs_format` 且是**位置参数**
2. `lhs_format`/`rhs_format` 缺失导致参数**整体错位**（`rhs_g` 被当成 `lhs_format`）
3. **算法级矛盾**：内核用 `A_GROUP_K=16`（A 每 16 个 K 元素一个 scale），并在 k-tile 内 `for g in static_range` 逐 g 各调一次 `dot_scaled` 累加。而 Triton e8m0 强制 **32 元素分组**——16 ≠ 32，无法用单个 `dot_scaled` 表达，且内核把 scale 按 "group g 切片"（`lhs_scale_g` 形状 `[M//32,1]`）传入，也非 `dot_scaled` 期望的完整 `[M, K//32]`。

### 4.3 影响评估
这不是"补个参数名就能跑"的级别，而是**算子数据布局（A 用 16 分组 + 逐 g 累加）与 Triton 3.6 强制要求（e8m0 必须 32 分组、整块一次 MMA）的根本矛盾**。即使强行改成 32 分组对齐，也意味着**改变了官方 NVFP4 的标准布局语义**，是否仍与转换器 `convert_mxfp4_to_nvfp4.py`（NV_K_BLOCK=16）及 vLLM 集成约定一致，需内核作者/方案所有者确认。

**建议**（需方案所有者决策）：
- 方案 A（对齐官方）：A/W scale 改 32 分组，k-tile 整块一次 `dot_scaled(rhs_k_pack=True)`，与社区其他 NVFP4 内核统一。
- 方案 B（保留 16 分组）：改用不带 scale 语义的 `tl.dot` + 手动 scale，或升级到支持 16 分组的库（如 DeepGEMM 风格），需更大改造。
- 方案 C（交回作者）：将本诊断连同 diff 回传给 KernelGen 内核作者重新生成 v9。

---

## 五、生产环境核查（安全结论）

- **4 rank 容器全程 healthy**（`vllm-tp4-rank0~3` Up 3 hours healthy），GPU utilization 0%，无残留测试进程。
- 修改仅限：测试包内 `nvfp4_4w4a_prefill_gemm_{triton,torch}.py` + `test_nvfp4_4w4a_prefill_gemm.py`；在 rank0 容器内安装了 pytest 9.1.1（供 T2，不影响推理）。
- 未触碰：已有 08-13 的 `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4/`（172GiB）目录、原版 checkpoint、生产启动编排。
- **按你的要求，生产未做任何恢复**，测试环境处于"测试包已上传、pytest 已装、修复 diff 已留存"状态。

---

## 六、交付物

| 文件 | 位置 |
|---|---|
| 本报告 | `deliverables/engineering-assurance/nvfp4-testkit-run-2026-08-19/nvfp4-testkit-diagnosis-2026-08-19.md` |
| 修复后内核代码 | `deliverables/engineering-assurance/nvfp4-testkit-run-2026-08-19/kernels_fixed/` |
| T2 最新失败日志 | `deliverables/engineering-assurance/nvfp4-testkit-run-2026-08-19/t2_latest_fail.log` |
| 测试包已上传 | `01:/home/<USER>/nvfp4-testkit/`（+ `01:/vllm-workspace/nvfp4-testkit/`） |

---

## 七、下一步（待你决策）

1. 就**缺陷 7 的 3 种方案**择一，我据此改造内核并完成 T2/T3。
2. 内核 T2/T3 通过后，再回头执行 T1.2 全量权重转换校验（本次未做）。
3. 测试全部完成后，是否由你决定恢复时间（你说过不需要我来恢复）。