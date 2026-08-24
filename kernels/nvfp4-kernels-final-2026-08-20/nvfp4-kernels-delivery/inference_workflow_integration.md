# 双算子纳入推理工作流集成方案 v2（依据生产加载报告 2026-08-20）

> 前提：双算子已进生产容器（PYTHONPATH 挂载，import 生效，零调用点风险）
> 生产实际（EngineeringAssuranceTeam 勘查）：MoE prefill 走 **B12xExperts**（w4a16，`_run_b12x_moe_fp4`）、KV 走 **fused_compress_quant_cache**（fp8_ds_mla paged [64,584]）——**均已功能完整**
> 结论：**本算子走"新建路径"（非替换）**，A/B 证明增量价值后才切换调用点

---

## 一、策略调整（依据生产报告）

| 原方案（v1） | 调整后（v2，依生产实际） |
|---|---|
| 插件替换 MoE/KV 调用点 | **新建旁路路径**，默认零侵入（环境变量开关 `VLLM_NVFP4_K1/K2`，默认 0） |
| 直接分派 prefill→v15 | **A/B 先行**：`ab_routeA_vs_b12x.py` 证明 v15 相对 B12X 的 prefill 加速比后才建调用路径 |
| v17 直接写 KV | **语义对齐先行**：v17 是 NVFP4 信封工具，仅 `--kv-cache-dtype nvfp4_ds_mla` 启用；fp8 路径 fused_compress 原样 |
| — | **架构设计 + 备份 + 灰度 + A/B 单独立项**（生产报告 §三 要求） |

---

## 二、落地包结构（nvfp4_vllm_plugin/）

```
nvfp4_vllm_plugin/
├── setup.py                        # vllm.general_plugins 注册（pip install -e .）
├── ab_routeA_vs_b12x.py            # 🆕 A/B①：v15(4W4A) vs B12X(w4a16 模拟) prefill TFLOPS
├── ab_v17_semantics.py             # 🆕 A/B②：v17 信封结构/逐字节/量化 roundtrip 对齐
└── nvfp4_vllm_plugin/
    ├── __init__.py                 # 环境变量开关 VLLM_NVFP4_K1/K2（默认关，惰性导入）
    ├── quant_config.py             # QuantizationConfig("nvfp4_4w4a_sm121") + M 阈值
    ├── moe_method.py               # FusedMoEMethodBase：prefill→v15，decode→原方法（分派）
    └── kv_writer.py                # v17 写回 hook + warmup（R2）；paged 散写骨架
```

**关键设计（对齐生产约束）**：
- `moe_method.py`：`_fallback_apply` 委托原 B12X/Marlin 方法——**decode 与 B12X 链路零改动**
- `_nvfp4_prefill` 为骨架（`NotImplementedError` 提示按生产 w13_packed/w2_packed 布局补全）——**先跑 A/B 再接入**
- `kv_writer.py`：`write_nvfp4_kv` 仅 NVFP4 dtype 时启用；R3 paged 单 kernel 落地前用 torch 散写（生产换 kernel）

---

## 三、执行路线（分三阶段，每阶段可回滚）

### Phase 0 — A/B 证明（当前，~1 小时）
```bash
# A/B①：routeA 增量价值（生产容器内）
python ab_routeA_vs_b12x.py
#   判据：v15 TFLOPS / B12X TFLOPS ≥ 1.2 才值得建 prefill 路径

# A/B②：v17 语义对齐（NVFP4 dtype 前提）
python ab_v17_semantics.py
#   判据：信封结构 PASS + 逐字节 PASS + roundtrip ~1e-2
```

### Phase 1 — 条件接入（A/B 达标后单独立项）
- **kernel①**：`pip install -e nvfp4_vllm_plugin` + `--quantization nvfp4_4w4a_sm121` + `VLLM_NVFP4_K1=1`：
  - prefill（M≥256）→ v15 4W4A；decode → B12X 原样
  - 权重：转换器 NVFP4 格式（`process_weights_after_loading` 预热 W 缓存）
  - CUDA Graph：prefill/decode 分 phase 捕获（分派在 Python 层，不污染 graph）
- **kernel②**：`--kv-cache-dtype nvfp4_ds_mla` + `VLLM_NVFP4_K2=1`：
  - KV 写回 → v17（linear 先；R3 paged 后替换散写）
  - reader 契约：584B 信封与 NVFP4 reader 对齐（已逐字节验证）

### Phase 2 — 灰度与长稳
- 数值一致性 ≤1%（vs B12X/fp8 基线）
- 128K 上下文 needle 回归（v17 信封 + NVFP4 reader）
- 4 rank 长跑 + 回滚演练（删挂载两行 / 关 env）

---

## 四、验收矩阵

| 指标 | 目标 | 依据 |
|---|---|---|
| routeA vs B12X prefill | ≥1.2× | Phase 0 A/B① |
| v17 语义对齐 | 结构/逐字节/roundtrip 全 PASS | Phase 0 A/B② |
| 端到端 prefill | +1.6~2.5×（v15 vs v11 基线） | 已实测 |
| KV 写回 | -73~79% 耗时（v17 vs v11） | 已实测 |
| 回滚 | 关闭 env / 还原 .bak | 生产报告 §六 |

---

## 五、待办（生产报告 §五 对齐）

- [ ] Phase 0 A/B 执行（routeA vs B12X / v17 语义）
- [ ] A/B 达标后：prefill MoE 新建路径（moe_method 补全 w13/w2 分组调用）
- [ ] R3：v17 paged 变体（对接 kv_cache[bid,slot,:] 单 kernel，替代 torch 散写）
- [ ] 生产容器 .bak-import 留档核对（四节点）
- [ ] docs/RFERENCE.md 更新 nvfp4 挂载 + PYTHONPATH
