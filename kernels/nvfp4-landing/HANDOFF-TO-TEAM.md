# NVFP4 落地工程 —— 交接文档（团队不可用，交接给可用团队）

> 交接人：主理人（engineering-assurance lead）| 交接日期：2026-08-20 09:35
> 上下文：`nvfp4-landing` 团队因连续 429 后 teamContext 失效而不可用；主理人已**以单人身份完成关键路径**，现交接给可用团队。

---

## 0. 交接摘要（一句话）

**kernel① 方案B 路线 A 已落地（vLLM 内置原生 FP4 MMA，8/8 正确性 rel=0.00141、60~187 TFLOPS、零构建）；kernel② 已确认 v17 可替换 v11；统一资料库已建立；旧料已清理归档。待办：生产持久化落位 <INSTALL_DIR>、三份 runbook 已写完、生产性能简测。**

---

## 1. 环境事实（接手必读，勿再推翻）

- 集群：DGX Spark 4 节点，01=<NODE_IP>（head，生产容器 `vllm-tp4-rank0`），02/03/04 worker
- SSH：`ssh -o BatchMode=yes node01 "docker exec vllm-tp4-rank0 bash -c '...'"`
- 容器镜像 `dspark-vllm-gx10:0.2.1-v026.0`，vLLM 0.26、torch 2.11+cu130、triton 3.6.0、sm_121a
- **关键**：`vllm._custom_ops` **内置** `cutlass_scaled_fp4_mm` + `scaled_fp4_quant`（原生 FP4，已预编译，零构建）
- flashinfer 0.6.15（mm_fp4 backend 全阻塞，缺 b12x/cute-dsl）
- **持久化**：`/vllm-workspace/` 是**容器内部**目录（未挂载宿主机）→ **容器重建会丢失**！
  宿主机持久目录：`<INSTALL_DIR>/`（其中 `lib/`、`models/`、`envs/`、`scripts/` 已挂载进容器）
- 生产 4 rank 全程 healthy、GPU 0%、未恢复（按用户要求）

## 2. 已完成事项（本轮，2026-08-20 上午）

### 2.1 ✅ 路线 A 落地（最大交付，主理人单人完成）
- **发现**：vLLM 0.26 内置 `cutlass_scaled_fp4_mm(a,b,block_scale_a,block_scale_b,alpha,out_dtype)`，SM120a 原生 FP4，**无需源码构建**
- **正确性铁证**：`cutlass_scaled_fp4_mm` 产物 vs vLLM 官方 `dequantize_to_dtype` = **rel 0.00141**（浮点精度级一致）
- **适配层** `nvfp4_4w4a_mmaf.py`：`RouteA` 类（`preprocess_weights` 缓存 W + `__call__`）+ 便捷 `nvfp4_4w4a_prefill_gemm`
- **实测**：正确性 **8/8 PASS rel=0.00141**；性能 **60~187 TFLOPS**（大 shape 12288 达 187）；W 预处理 21ms/层
- **SASS 门禁**：`_C_stable_libtorch.abi3.so` 内嵌全 `sm_120` cubin + 1349 个 FP4 符号（cutlass_scaled_fp4_mm/scaled_fp4_quant/e2m1）

### 2.2 ✅ 路线 B 可行性结论（备选，未打通）
- flashinfer 0.6.15 API 面完整，但 `mm_fp4` 四 backend 全阻塞（b12x/cute-dsl 未编入、trtllm cap121 不支持、cudnn/cutlass 参数错）
- 记录于 `docs/ROUTE-B-FEASIBILITY.md`

### 2.3 ✅ 统一资料库建立
- 本地 `deliverables/engineering-assurance/nvfp4-landing/`（README 入口 + kernel1 适配层 + docs + tests 全量）
- 容器 `/vllm-workspace/nvfp4-landing/` 同步（routeA/routeB/docs/tests）
- 文档：`README.md`、`docs/landing-runbook.md`、`docs/runbook-kernel2-v17.md`、`docs/testing-matrix.md`、`docs/ROUTE-B-FEASIBILITY.md`、`docs/cleanup-inventory.md`

### 2.4 ✅ 旧料清理（容器）
- 6 个历史交付目录（nvfp4-delivery、-v12/v13/v15/v16/v17）→ **归档** 到 `nvfp4-landing/_archive_old_deliveries/old-deliveries-2026-08-20.tgz`（1.4M，336 文件）后删除
- 保留：`nvfp4-delivery-final`、`nvfp4-landing`、`nvfp4-testkit`
- 本地：无解压残留（仅 .zip 原件保留）

### 2.5 ✅ kernel② v17 验收（前序轮已定）
- v17 替换 v11：8/8 逐字节、大 T 194~262 GB/s、边缘 case 全一致；安全套件 3 FAIL 为测试脚本缺陷（已定性）

## 3. 待办工作（交接给新团队）

### P0 生产持久化落地（未做）
- **问题**：`/vllm-workspace/nvfp4-landing/` 容器重建会丢
- **方案**：把 routeA 适配层 `nvfp4_4w4a_mmaf.py` 落到宿主机 `<INSTALL_DIR>/scripts/`（或新 `nvfp4/` 子目录，已挂载进容器），并在容器内建立 `sys.path` 引用（如 softlink 到 site-packages 或 set PYTHONPATH）
- 参考：libncclpin 已通过 `<INSTALL_DIR>/lib/libncclpin.so` → 容器挂载，可依此模式
- **验证**：容器重启后可 `import nvfp4_4w4a_mmaf` 并跑通

### P1 生产性能简测（未做，最终任务 #26）
- 容器内跑 routeA 各 shape TFLOPS 表 + kernel② v17 GB/s
- 脚本已就绪：`tests/bench_mmaf_final.py`、`tests/bench_big.py`、`tests/benchmark_nvfp4_ds_mla_kv_linear_v17.py`

### P1 kernel② v17 部署（待新团队执行）
- 按 `docs/runbook-kernel2-v17.md`：分发 v17 文件到 4 节点、切换调用点、md5 校验、R1/R2/R3 加固

### P2 收尾
- 本地 `.zip` 原件保留，可打包本轮 deliverable 备存档
- 生产 4 rank 恢复决策（目前未恢复，按用户要求）

## 4. 关键代码/文件位置

| 文件 | 位置 |
|---|---|
| routeA 适配层 | `deliverables/.../nvfp4-landing/kernel1/nvfp4_4w4a_mmaf.py`（容器 `/vllm-workspace/nvfp4-landing/routeA/`） |
| 正确性+性能 bench | `tests/bench_mmaf_final.py`、`tests/bench_big.py` |
| 路线A 验证证据 | `tests/probe_a1_final.py`（rel=0.00141 的实证脚本） |
| SASS 门禁 | `tests/sass_fp4_check.py` |
| v17 测试 | `tests/kernel2/test_nvfp4_ds_mla_kv_linear_v17.py` |
| 统一入口 | `README.md` |
| 旧交付归档 | 容器 `nvfp4-landing/_archive_old_deliveries/old-deliveries-2026-08-20.tgz` |

## 5. 重要教训（承接者注意）

1. **对照基准必须用 vLLM 官方语义**（`dequantize_to_dtype`），不要用旧 torch ref 32-group —— 那是不同量化方案，rel 必然 0.19/1.35 但不代表错
2. **bench 缓存陷阱**：`RouteA` 缓存按 W data_ptr；批量 bench 时不同 W 复用同 data_ptr 会踩缓存 → 用独立实例或复用预量化权重
3. **框架 bug**：`nvfp4_emulation_utils.break_fp4_bytes` 用 CPU `kE2M1ToFloat_handle` 索引 GPU → 需先 `.cuda()`
4. **SASS 门禁 SM12x 用 `mma.*e2m1|mmaf`**，勿用 `tcgen05`（那是 SM10x）
5. `/vllm-workspace` 非持久，**重要产物务必同步到 `<INSTALL_DIR>` 或本地**

## 6. 团队状态说明

- `nvfp4-landing-9e67` 团队成员（routeA/routeB/techwriter/sre/testing）因 429 限流后 teamContext 失效，Agent 续跑报 "no teamContext"；`TeamDelete` 报 "Not in a team"
- 主理人改用**单人 craft 模式**完成关键路径，不依赖子 agent 消息系统
- 交接后，新团队可直接从「待办工作」章节接手，无需重复已完成的路线 A/B 探测