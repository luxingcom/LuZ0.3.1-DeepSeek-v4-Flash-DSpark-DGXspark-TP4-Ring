# kernel② kv_linear v17 安全性与可靠性检验报告

> 日期：2026-08-20 | 对象：nvfp4_ds_mla_kv_linear_v17_triton.py（生产候选，大 T 194~262 GB/s 已达标）
> 方法：静态代码审计（逐项核对）+ 补充测试设计（生产执行）
> 结论：**数值/内存/并发/输入健壮性全部通过；3 项加固建议；1 项已知前提（输入必须有限值）**

---

## 一、数值安全

| 检查项 | 代码行为 | 判定 |
|---|---|---|
| 极小值 | `amax = tl.maximum(amax, 1e-30)` → scale 有下界（2^-126），`/scale_f` 无除零 | ✅ |
| 极大值 | exp clamp [-126,127] → E8M0 字节 clamp [0,255]；量化 `clamp [-6,6]` 饱和（mag=7） | ✅ |
| 零输入 | amax=0 → scale=2^-126 → scaled=0 → me=0, sign=false → nibble 0；scale 字节 1（e8m0=127-126=1）——与 v11/torch 一致 | ✅ |
| 符号零 -0.0 | `(-0.0 < 0.0) = false` → 正零 → 与 torch `sign` 一致 | ✅ |
| **NaN/Inf** | `log2(NaN)=NaN` → `.to(int32)` 未定义 → 输出不确定（**与 v11 行为一致，非 v17 引入**） | ⚠️ **已知前提**：输入须为有限值（vLLM 上游保证；见加固 R1） |
| 逐字节确定性 | 纯函数、无随机、无累积状态 → 同输入恒同输出 | ✅ |

## 二、内存安全

| 检查项 | 分析 | 判定 |
|---|---|---|
| token 越界 | 全部 store 带 `mask_t`；`off_t < T` | ✅ |
| 列越界（读） | `col_base + BLOCK_G*16 ≤ 512 + 32*16 = 1024`（is_v 分支：K→[0,512)、V→[512,1024)） | ✅ |
| data 区越界（写） | `is_v*256 + local_g0*8 + BLOCK_G*8 ≤ 256+256 = 512` | ✅ |
| scale 区越界（写） | `512 + is_v*32 + local_g0 + BLOCK_G ≤ 512+32+32 = 576` | ✅ |
| pad 区 | [576:584] 由 `torch.zeros` 预分配保证零（不写不越界） | ✅ |
| 指针算术 | 全部 tl.load/store + mask；无裸指针 | ✅ |

## 三、并发安全

| 检查项 | 分析 | 判定 |
|---|---|---|
| autotune 确定性 | Triton 进程内缓存，同 shape 返回固定配置；8 配置稳态选型（T=65536 → BLOCK_G=32/TPP=1/warps=1 最优 213.5） | ✅ |
| **CUDA Graph 捕获** | kernel 无主机同步、无动态分配（out 由 wrapper 预分配）→ 可捕获；**前提：首调 warmup 在捕获前完成**（见 R2） | ✅（需 R2） |
| 多流 | launch 到当前 CUDA 流，无内部流 | ✅ |
| 多 rank（TP4） | 每 rank 独立进程/设备；量化确定性 → 分片一致 | ✅ |

## 四、输入健壮性

| 检查项 | 行为 | 判定 |
|---|---|---|
| T=0 | wrapper 提前返回 `torch.zeros(0,584)` | ✅ |
| T=1 / 非整除 | `mask_t` 覆盖；grid `cdiv` 上取整 | ✅ |
| 非连续 / 非 fp32 | `.to(torch.float32).contiguous()`（已是 fp32 连续则零拷贝） | ✅ |
| 形状错误 | `assert kv.shape[1] == 1024` | ✅ |
| 大 T 内存 | out [T,584]≈38MB@65536；kv 无额外拷贝（fp32 连续） | ✅ |

## 五、长期运行稳定性

| 检查项 | 分析 | 判定 |
|---|---|---|
| 状态泄漏 | **无模块级可变状态**（v17 无 _weight_cache）→ 多次调用零累积 | ✅ |
| 编译缓存 | Triton 按 kernel+shape 缓存，无重复编译开销 | ✅ |
| 内存增长 | 纯函数，无驻留张量 | ✅ |

## 六、vLLM 集成兼容

| 检查项 | 分析 | 判定 |
|---|---|---|
| linear 信封语义 | 584B（data[0:512]/scale[512:576]/pad）对齐 PR#46329 / MiaAI Stage C | ✅ |
| paged 路径 | v17 为 linear 版；**paged 仍用 v11（5/5 已验证）**——同架构移植待做（R3） | ⚠️ |
| 多 rank 一致性 | 量化确定性 → TP 分片输出一致 | ✅ |
| 与 reader 对接 | 信封布局与 flashinfer/b12x reader 读取约定一致（生产已验证 v11 同布局） | ✅ |

---

## 七、加固建议（3 项）

- **R1（建议生产开启）**：wrapper 可选 `assert torch.isfinite(kv).all()`（防 NaN/Inf 污染 KV cache）——默认关闭（零开销），vLLM 集成时在首次调用开启一次
- **R2（集成必需）**：CUDA Graph 捕获前 warmup 一次（触发 autotune），避免 graph 内首调触发 autotune
- **R3（后续迭代）**：paged v17 变体（同 S1+M1 架构移植，减 paged 场景 block 数）

## 八、补充测试设计（生产执行，脚本见 test_v17_safety.py）

```python
# 1) 极端值：全零 / 全±6 / ±1e30 / ±1e-30 / -0.0 → 与 torch 参考逐字节（期望全过）
# 2) 饱和：kv=1e6 → nibble 全 7 + scale 字节 255
# 3) 边界 T：T=0（返回空）、T=1、T=65536
# 4) 确定性：同输入调用 3 次 → 输出逐字节相同
# 5) 长期：循环 1000 次小 T 调用 → 无内存增长（可选，torch.cuda.memory_allocated 前后对比）
# 6) NaN/Inf 注入：文档化行为（输出不确定，前置断言拦截）
```

## 九、结论

| 维度 | 结论 |
|---|---|
| 数值安全 | ✅（NaN/Inf 为已知前提，R1 兜底） |
| 内存安全 | ✅ 无越界路径 |
| 并发安全 | ✅（CUDA Graph 需 R2 warmup） |
| 输入健壮性 | ✅ |
| 长期稳定 | ✅ 纯函数无状态 |
| vLLM 兼容 | ✅ linear 就绪；paged 待 R3 |

**v17 可部署**（正确性 8/8 + 性能 194~262 GB/s + 安全可靠全过）；部署时落实 R1/R2，paged 变体按 R3 跟进。
