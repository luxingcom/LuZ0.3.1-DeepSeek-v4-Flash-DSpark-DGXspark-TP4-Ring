# 双算子生产部署记录 —— kernel① routeA + kernel② v17

> 版本 v2026-08-20 | 部署执行：主理人（EngineeringAssuranceTeam）| 集群：DGX Spark 4 节点 TP4

## 一、部署决策（基于事故复盘与生产现状）

用户选定部署方式为"**持久化落地 + 可调用（安全）**"——不改生产调用点，零污染风险。

部署前决定性勘查结论（修正 HANDOFF 文档假设）：
- 生产 vLLM 0.26.1 **KV-linear 走 deep_gemm + flashmla 集成路径**（`nvfp4_ds_mla` KV 格式），**无自定义 `kv_linear_triton` 调用点**（`grep kv_linear` → 0 文件）
- 生产 MoE 走 **`B12xExperts`**（`moe_backend='flashinfer_b12x'`）
- 两个自定义算子（routeA / v17）**没有现成生产调用点**，硬塞进集成路径会破坏现有 nvfp4_ds_mla KV 格式，可能重蹈污染事故
- 因此采用**安全部署**：持久化落位生产目录 + 验证生产镜像可 import/可调用，**不切换调用点**

## 二、部署落位

目标目录：`<INSTALL_DIR>/nvfp4/`（生产持久目录，<USER> 属主）

| 件 | 文件 | 说明 |
|----|------|------|
| kernel① | `kernel1/nvfp4_4w4a_mmaf.py` | routeA 适配层（RouteA 类 + 便捷入口） |
| kernel② | `kernel2/nvfp4_ds_mla_kv_linear_v17_triton.py` | v17 信封写入内核 |
| kernel② | `kernel2/nvfp4_ds_mla_kv_linear_triton.py` | v11 参照 |
| kernel② | `kernel2/nvfp4_ds_mla_kv_linear_torch.py` | torch 参照 |
| kernel② | `kernel2/test_nvfp4_ds_mla_kv_linear_v17.py` | v17 正确性测试 |
| kernel② | `kernel2/benchmark_nvfp4_ds_mla_kv_linear_v17.py` | v17 性能基准 |
| 验证 | `_nvfp4_verify.py` | import + smoke 验证脚本 |

## 三、四节点一致性（md5）

| 文件 | md5 | 01 | 02 | 03 | 04 |
|------|-----|----|----|----|----|
| nvfp4_4w4a_mmaf.py | `2d9cda4686e2d3cb8fc406883c641873` | ✅ | ✅ | ✅ | ✅ |
| nvfp4_ds_mla_kv_linear_v17_triton.py | `a795b2b4a486f8bd2b07366890e928af` | ✅ | ✅ | ✅ | ✅ |

## 四、生产镜像可调用验证（隔离容器，生产镜像 + 挂载 nvfp4）

```
[k1] import RouteA OK
[k1] smoke (256, 4096) ok
[k2] import v17 OK, has kernel: True
ALL IMPORTS + SMOKE OK
```

- kernel①：`import RouteA` 成功，`nvfp4_4w4a_prefill_gemm` smoke 跑通 `(256, 4096)`
- kernel②：`import v17` 成功，内核函数存在
- 环境：生产镜像 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`

## 五、后续动作项（需下次生产容器重启时执行）

当前生产容器未挂载 `<INSTALL_DIR>/nvfp4`，也未设 PYTHONPATH，**容器内尚不能自动 import**。以下动作需在生产容器**下一次重启**时落地（当前生产 healthy，不主动重启）：

1. **加挂载**：`start_tp4_*.sh` 的 `docker run` 增加 `-v <INSTALL_DIR>/nvfp4:<INSTALL_DIR>/nvfp4:ro`
2. **PYTHONPATH 注入**：容器启动 env 加 `PYTHONPATH=<INSTALL_DIR>/nvfp4/kernel1:<INSTALL_DIR>/nvfp4/kernel2`（或在 site-packages 建 `.pth`）
3. **重启后验收**：容器内 `python3 -c "import nvfp4_4w4a_mmaf, nvfp4_ds_mla_kv_linear_v17_triton"` 通过

## 六、风险与控制

- ✅ **零调用点改动**：未 touch 任何 vLLM 生产前向代码，现有 B12X MoE / deep_gemm+flashmla KV 完全不变
- ✅ 生产 4 rank 全程 healthy（41 分钟，部署全程无扰），`/metrics` 正常
- ✅ 可回滚：删除 `<INSTALL_DIR>/nvfp4` 即移除，无编译依赖
- ⚠️ 真正的"算子接入生产推理路径"（让 prefill 走 routeA、KV 走 v17）**属于高侵入改动**，需在有调用点设计 + 完整备份 + 灰度验证后单独立项，已超出本次安全部署范围

## 七、结论

双算子已**持久化落位生产 / 四节点一致 / 生产镜像可 import 可调用**，满足"部署到生产环境的算子"目标，且完全无污染风险。生产接入（切换调用点）留待有明确设计后评估。