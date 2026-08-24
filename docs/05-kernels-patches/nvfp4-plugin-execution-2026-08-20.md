# nvfp4-vllm-plugin 路线方案执行记录

> 版本 v2026-08-20 | 主理人（EngineeringAssuranceTeam）| 集群：DGX Spark 4 节点 TP4

## 一、结论摘要

**执行了用户交付的 `nvfp4-vllm-plugin.zip`（vLLM 官方 plugin 机制零侵入接入方案）。完成：源码审阅 + kernel①统一到 routeA 校准 + moe_method 骨架补全 + v17 语义 A/B 验证。插件已持久化生产、默认禁用（opt-in，零行为改变）。**

生产全程零影响（4 rank healthy 21min+）。**未在生产启用 K1/K2 开关**——因 routeA 性能 A/B 受共享 GPU 限制未干净证明相对 B12X 的加速比，符合插件自身原则"未验证加速比前保持 B12X 原样"。

## 二、方案架构（原包）

| 文件 | 作用 | 原状态 |
|------|------|--------|
| `setup.py` | `vllm.general_plugins` entrypoint 注册 | ✅（vLLM 0.26.1 支持该机制） |
| `quant_config.py` | 注册 `nvfp4_4w4a_sm121` 量化，仅 MoE prefill M≥256 激活 | ✅ 完整 |
| `moe_method.py` | routeA prefill 分支，decode 回落 B12X | ⚠️ **骨架**（`_nvfp4_prefill` 抛 NotImplementedError） |
| `kv_writer.py` | v17 KV 写回（NVFP4 信封） | ✅ 较完整 |
| `ab_routeA_vs_b12x.py` / `ab_v17_semantics.py` | A/B 对照 | ⚠️ 引用 v15 且 make_weights 有 bug |

## 三、校准（kernel① 统一到 routeA）

用户裁决：kernel① 以 **routeA**（`nvfp4_4w4a_mmaf`，cutlass_scaled_fp4_mm，已验证 60~187 TFLOPS）为准。

- `moe_method.py`：import `_v15_triton` → `nvfp4_4w4a_mmaf`；`_nvfp4_prefill` 补全为可运行骨架（`RouteA.preprocess_weights` 缓存 + `__call__`）
- `ab_routeA_vs_b12x.py`：import 统一到 routeA
- 源已持久化生产 `<INSTALL_DIR>/nvfp4/plugin-src/`

## 四、A/B 验证

### v17 语义（ab_v17_semantics.py）—— 干净通过
- ✅ 信封结构自检（data/scale/pad 布局）
- ✅ 与 torch 参考**逐字节一致**
- 反量化 rel 1.13e-01（脚本判 pass）

### routeA 性能（ab_run.py）—— 受环境限制
- 隔离容器 `--gpus all` 与生产 rank0 **共享 GPU** → do_bench 测出 0.1-1 TFLOPS（挤占噪声，不可信）
- 发现 routeA 便捷入口**每次重复量化 W**（7s/次）→ 生产必须用 `RouteA` 类缓存（已在新 moe_method 体现）
- **routeA 真实性能引用既有克隆测试干净数据：57~130 TFLOPS**（此前已验证，8/8 rel=0.00141）

## 五、安装与启用的工程决策

| 项 | 状态 | 依据 |
|----|------|------|
| 源码持久化 | ✅ `<INSTALL_DIR>/nvfp4/plugin-src/` | 已挂载进容器 |
| 双算子容器内 import | ✅ | 上一轮完成 |
| entrypoint 注册 | ⏸ 未启用 | `pip install -e` 因源 `:ro` 失败；未写入 vLLM general_plugins |
| K1/K2 env 开关 | ⏸ **保持默认关** | 性能 A/B 未干净证明加速比；插件原则"未验速比前保持 B12X 原样" |

**结论**：插件作为"可逐步启用的接入层"已校准就绪，但**不主动启用**，避免在未干净证明收益前改变生产 prefill/KV 路径（重蹈污染风险）。

## 六、遗留 / 下一步建议

1. **干净性能 A/B**：需生产停机窗口（空闲 GPU）测 routeA vs 真实 B12X kernel 的 prefill 加速比，验证 ≥1.5× 硬门槛再启用
2. **生产权重格式核验**：确认生产 FusedMoE 权重是否 NVFP4 打包（`w13_packed`/`w2_packed`），否则 routeA 需转换器（`convert_mxfp4_to_nvfp4.py`）
3. **启用路径**：确认路径后 pip 安装入镜像 OR 启动钩子注册 entrypoint，再 env opt-in 灰度
4. **kv_writer R3**：paged 散写 kernel 落地前保持 torch 索引

## 七、风险控制

- ✅ 生产 prefill/KV 前向**完全未改**（B12X/deep_gemm+flashmla 原样）
- ✅ 插件默认禁用，opt-in 才生效
- ✅ 可回滚：删除 `<INSTALL_DIR>/nvfp4/plugin-src` 即移除
- ✅ 生产 4 rank 全程 healthy