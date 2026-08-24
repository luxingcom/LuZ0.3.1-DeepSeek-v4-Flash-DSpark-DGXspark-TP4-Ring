# ARCHITECTURE — NVFP4 低精度落地工程架构审查与接续方案

> **架构师**：阿奇（system architect） | **日期**：2026-08-20
> **范围**：DGX Spark GB10 TP4 集群（01-04 / sm_121a / vLLM 0.26）NVFP4 4W4A 落地，支撑「交接文档落实」
> **审查方式**：只读研究 + ADR 风格架构评估。输入依据 = `nvfp4-landing/` 统一资料库 + 今日交接 `HANDOFF-TO-TEAM.md` P0/P1/P2 待办。
> **前置硬约束（用户 08-15 已定，勿否定）**：运行环境必须全面转向低精度 4W4A；E8M0→fp32 upcast、torch fallback、cutlass fp32 scale 等 0.27 冒烟绕行补丁**不作生产候选**；正式运行路径 = NVFP4 原生低精度（nvfp4 weight + deep_gemm E8M0 + nvfp4_ds_mla KV）。

---

## 一、需求与目标

| 维度 | 内容 |
|---|---|
| **功能需求** | kernel① prefill GEMM 走 NVFP4 4W4A 原生 FP4 MMA；kernel② KV-Linear 走 NVFP4 DS-MLA 原生路径；0.27 评估/切换必须回归低精度路径 |
| **非功能需求** | 正确性对齐 vLLM 官方语义（rel<0.02）；性能 ≥200 TFLOPS（kernel①）/ ≥1.5× v15（对照）；SASS 显示原生 `mma.*e2m1`/`mmaf`；生产可重建恢复 |
| **约束** | GB10 sm_121a；运行环境必须转向低精度（用户红线）；临时绕行补丁不作生产路径；容器 `/vllm-workspace/` 非持久；生产须落在宿主机 `<INSTALL_DIR>/` |
| **接续目标（本文核心）** | 把 `HANDOFF-TO-TEAM.md` P0/P1/P2 待办落成**可执行的架构方案与执行路径**，供交接文档采纳 |

**架构结论（先行预览）**：路线 A（vLLM 内置 `cutlass_scaled_fp4_mm`）作为 kernel① 主路径已打通且可部署；kernel② v17 替换 v11 成立；生产持久化落位 `<INSTALL_DIR>/`；剩余 P0/P1/P2 为"工程化收尾"而非"方案不确定"——方案已定，只差落地执行。

---

## 二、高层设计

### 2.1 部署拓扑

```
宿主机 <INSTALL_DIR>/ (持久, 重建保活)
  ├── scripts/nvfp4/           → routeA 适配层 nvfp4_4w4a_mmaf.py (新, P0)
  ├── lib/                     → libncclpin.so (既有持久化模式蓝本)
  ├── models/ · envs/          → (既有挂载)
  └── kernel2/v17/            → nvfp4_ds_mla_kv_linear_v17_triton.py (新, P1)
              │  (bind-mount / volume 挂载进容器)
              ▼
容器 vllm-tp4-rank0 (02/03/04 worker)
  /vllm-workspace/ (临时, 重建丢失)
    └── nvfp4-landing/ (工作区, 同步副本)
```

**两点关键（证据支撑）**：
1. `/vllm-workspace/` 是容器内部目录未挂载 → 重建丢失 → 一切生产产物必须以 `<INSTALL_DIR>/` 为**唯一权威源**（single source of truth）。
2. `libncclpin` 已通过 `<INSTALL_DIR>/lib/libncclpin.so` → 容器挂载，是本方案的**成熟持久化模式蓝本**，复用而非自创。

### 2.2 主路径（路线 A）与备选（路线 B）定位

- **主路径 = 路线 A**：vLLM 0.26 `_custom_ops` 内置 `cutlass_scaled_fp4_mm`（SM120a 原生 FP4 **已预编译进 vLLM，零构建**）+ `scaled_fp4_quant`（硬件量化）。8/8 正确性 rel=0.00141、60~187 TFLOPS。**这是满足低精度红线、可直接部署的正规路径**，非临时补丁。
- **备选 = 路线 B**：FlashInfer `mm_fp4` 四 backend 全阻塞（b12x/cute-dsl 未编入、trtllm cap121 不支持、cudnn/cutlass 参数错）→ 当前 wheel 不可运行；记录为备用，待 FlashInfer TOT（含 sm12x NVFP4，双 arch JIT）或源码重编后可作为对照/冗余。
- **回退 = v15**（bf16 MMA，26~81 TFLOPS）：仅兜底，非低精度，不作长期。

### 2.3 核心理由：为何路线 A 是正视路径而非绕行补丁

0.27 冒烟绕行补丁（E8M0→fp32 upcast / torch fallback / cutlass fp32 scale）是**改数值语义**的临时手段。路线 A **不改语义**——A/W 均用 vLLM 官方 `scaled_fp4_quant` 量化 + CUTLASS 原生 FP4 GEMM，与 vLLM 官方 `dequantize_to_dtype` 数学完全一致（rel=0.00141）。这是**落到官方语义本身**，因此是生产候选，非补丁。

---

## 三、关键决策记录 (ADR)

### ADR-1：kernel① 主路径采用路线 A（vLLM 内置 cutlass_scaled_fp4_mm）
**状态**：Accepted
- **背景**：Triton 3.6 无原生 FP4 MMA codegen（v16 fp8 仅 0.1~0.2 TFLOPS）；v15 为 bf16 降级（26~81 TFLOPS，含精度损失），均不合 4W4A 红线。
- **选项**：
  | 选项 | 复杂度 | 成本 | 正确性 | 性能 | 判定 |
  |---|---|---|---|---|---|
  | A. vLLM 内置 cutlass FP4 | Low | 零构建 | 8/8 rel=0.00141 | 60~187 TFLOPS | ✅ 主路径 |
  | B. FlashInfer mm_fp4 | Med-High | 需重编 wheel | 阻塞 | 阻塞 | 🔶 备选 |
  | v15 bf16 回退 | Low | 0 | 8/8 | 26~81 | ↩ 回退 |
- **决策**：**路线 A**。理由：零构建、合规 4W4A、数值对齐官方语义、性能 1.5~7× v15、可独立 `.py` 部署且不改 vLLM 本体（删除即回退）。
- **影响**：生产精度与性能已验证，风险主要转为部署/持久化工程。

### ADR-2：kernel② KV-Linear v17 替换 v11
**状态**：Accepted
- **背景**：v11 带宽瓶颈 53~61 GB/s。
- **选项与决策**：v17（宽 tile、多 token 负载）8/8 逐字节一致、大 T 194~262 GB/s（3.5~4.6× v11，理论 273 的 71~96%）、边缘 case 全 byte_equal、安全审计 4 项通过。**采用 v17**，v11 保留作回退。
- **注意（非内核缺陷）**：shipped 安全套件 3 FAIL 是**测试脚本缺陷**（saturation 期望 255 实为 144、sign_zero 期望 1 实为 24、boundary_T 未 seed）→ 修正期望/seed 后全绿。
- **影响**：需四节点分发 + 切换调用点 + md5 校验；paged 路径维持 v11（R3 待做）。

### ADR-3：生产持久化单一权威源 = `<INSTALL_DIR>/`
**状态**：Accepted（落实 P0）
- **背景**：`/vllm-workspace/` 容器重建即丢。
- **决策**：routeA 适配层与 v17 内核文件落宿主机 `<INSTALL_DIR>/scripts/nvfp4/` 与 `<INSTALL_DIR>/kernel2/v17/`（已挂载进容器），复用 `libncclpin` 挂载蓝本。
- **影响**：容器重建后自动可见，满足生产保活。

### ADR-4：SASS 门禁判据 = SM12x `mma.*e2m1|mmaf`（非 tcgen05）
**状态**：Accepted
- **决策**：GB10 sm_121a 原生 FP4 以 `mma.*e2m1` / `mmaf` 出现为准；**勿用** `tcgen05`（那是 SM10x）。辅证：`_C_stable_libtorch.abi3.so` 内嵌 sm_120 cubin + 1349 个 FP4 符号、性能远超 bf16 上限。
- **影响**：防止用错门禁指令导致误判/漏判。

### ADR-5：对照基准必须用 vLLM 官方语义
**状态**：Accepted
- **决策**：kernel① 对照 = vLLM 官方 `dequantize_to_dtype`（rel<0.02），**不要**用旧 torch 32-group ref（那是不同量化方案，rel 0.19/1.35 属误报）。kernel② 对照 = 逐字节（`torch.equal`）。
- **影响**：统一验收口径，避免历史误判重演。

---

## 四、可运维性（可运维性 + 恢复）

### 4.1 生产持久化落位设计（落实 P0）

```
宿主机                   容器(挂载点)                     作用
<INSTALL_DIR>/
  scripts/nvfp4/nvfp4_4w4a_mmaf.py   → /opt/nvfp4/ nvfp4_4w4a_mmaf.py   routeA适配层
  kernel2/v17/*.py                   → /opt/nvfp4/kernel2/v17/           v17内核+测试
  docs/ (runbook/README)             → /opt/nvfp4/docs/                  文档随迁
```

**容器内引用（三选一，推荐 softlink 到 site-packages）**：
```bash
# 方式一（推荐）：softlink 到 site-packages → import 透明
ln -sf /opt/nvfp4/nvfp4_4w4a_mmaf.py \
  /usr/local/lib/python3.12/dist-packages/nvfp4_4w4a_mmaf.py
# 方式二：sitecustomize/site-packages .pth 追加 PYTHONPATH
# 方式三：容器启动脚本 export PYTHONPATH=/opt/nvfp4:$PYTHONPATH
```

**验证标准（P0 完成判据）**：
```bash
docker restart vllm-tp4-rank0
docker exec vllm-tp4-rank0 python3 -c \
  "import nvfp4_4w4a_mmaf; from nvfp4_4w4a_mmaf import RouteA; print('OK persist')"
# 并跑通快速验证：preprocess_weights + __call__ 出 [M,N]
```

### 4.2 回退方案（弹性边界）

| 层级 | 回退动作 | 代价 |
|---|---|---|
| kernel① | 删除 `nvfp4_4w4a_mmaf.py` 引用 → 落回 v15 | 0（不改 vLLM 本体） |
| kernel② | 调用点换回 `_triton`(v11) | 0（文件留存） |
| 全量 | 恢复 `<INSTALL_DIR>` 挂载 + 重建容器 | 低（源在宿主机） |

### 4.3 容器重建恢复（DRSOP）

1. 重建容器 `dspark-vllm-gx10:0.2.1-v026.0`，重挂 `<INSTALL_DIR>/{scripts,lib,models,envs}`。
2. 校验 routeA + v17 在 `/opt/nvfp4/` 自动可见、`import` 成功。
3. 跑三类回归：SASS 门禁 → 正确性(8/8) → 性能(≥200 TFLOPS / v17 GB/s)。
4. 通过即恢复生产；失败即按 §4.2 回退。

### 4.4 运行时运维要点
- **bench 缓存陷阱**：`RouteA` 缓存按 W data_ptr，批量 bench 不同 W 复用同 data_ptr 会踩缓存 → 用独立实例或复用预量化权重。
- **框架 bug 注记**：`nvfp4_emulation_utils.break_fp4_bytes` 用 CPU `kE2M1ToFloat_handle` 索引 GPU → 需先 `.cuda()`。
- **持久化纪律**：一切生产产物同步 `<INSTALL_DIR>` 或本地，勿依赖 `/vllm-workspace`。
- **4 rank 恢复**：目前 4 rank healthy、GPU 0%、未恢复（按用户要求）；恢复决策归入 P2。

---

## 五、测试策略（验收门禁）

### 5.1 kernel①（路线 A）
| 门 | 判据 | 脚本 |
|---|---|---|
| SASS | 出现 `mma.*e2m1` / `mmaf`（sm_120 cubin + FP4 符号） | `tests/sass_fp4_check.py` |
| 正确性 | 8/8，rel<0.02 vs 官方 `dequantize_to_dtype` | `tests/bench_mmaf_final.py` |
| 性能 | **≥200 TFLOPS**（当前最大 187，大 shape 需冲） | `tests/bench_big.py` |
| 对照 v15 | **≥1.5×** | `tests/compare_v15.py` |

> **性能差距提示**：当前峰值 187 TFLOPS 距 200 门槛还有 7%。路径：A 量化融合进 CUDA Graph、`scaled_fp4_quant` 改 cutlass backend、大 shape autotune。这是 P1 生产性能简测的首要关注点。

### 5.2 kernel②（v17）
| 门 | 判据 | 脚本 |
|---|---|---|
| 正确性 | 8/8 逐字节（`torch.equal`） | `test_nvfp4_ds_mla_kv_linear_v17.py` |
| 安全 | 4 项通过 + 3 项修期望/seed 后全绿 | `..._v17_safety.py` |
| 性能 | 大 T 194~262 GB/s（3.5~4.6× v11） | `benchmark_nvfp4_ds_mla_kv_linear_v17.py` |
| 门禁 | CI / 发版必跑 | — |

### 5.3 统一门禁矩阵
见 `docs/testing-matrix.md`：kernel① 路线A=部署 / 路线B=备用 / v15=回退；kernel② v17=替换 / v11=回退 / paged=v11 维持。

---

## 六、文档结构（交接文档应含哪些章节）

对标现有统一资料库，建议交接文档固定为以下章节（缺则补）：

1. **交接摘要**（一句话 + 状态灯）
2. **环境事实**（集群拓扑 / SSH / 容器镜像 / 持久化边界 / 关键内置能力）
3. **决策基线**（ADR 摘要：路线A主 / 路线B备 / v17 / 官方语义对照 / SASS判据）
4. **已交付成果与证据**（路线A落地数据、v17验收数据、资料库、清理记录）
5. **待办工作分优先级**（P0/P1/P2，每项含：是什么 / 方案 / 完成判据 / 脚本）
6. **可运维性**（持久化落位 <INSTALL_DIR>、回退、容器重建恢复 DRSOP）
7. **测试策略与验收矩阵**（SASS/正确性/性能/对照基线）
8. **风险与权衡**（见下）
9. **关键代码/文件位置索引**（表）
10. **重要教训**（缓存陷阱、框架 bug、SASS 指令、持久化纪律）

当前资料库已含大部分，**需补齐**：P0/P1/P2 可执行执行路径（含完成判据）、DRSOP 恢复流程、ADR 决策记录段、风险登记表。

---

## 七、风险与权衡

| 风险 | 等级 | 缓解 |
|---|---|---|
| **性能距 200 门槛 7%**（峰值 187） | Med | P1 简测定位；A量化融合/CUDA Graph/cutlass backend；若不可达需与用户确认 200 是否硬门槛 |
| **路线 A 依赖 vLLM 0.26 内置符号** | Med | 不做 vLLM 本体修改；适配层独立 .py 可回退；记录 API 契约以便将来 vLLM 升级核对 |
| **路线 B wheel 阻塞** | Low（备选） | 保持记录；FlashInfer TOT/重编后再评估 |
| **持久化中断**（工件仅存 /vllm-workspace） | Med | P0 立即落 <INSTALL_DIR>；全部产物双写宿主机 |
| **4 rank 生产恢复决策未定** | Low-Med | P2 收尾统一决策；当前按用户要求保持未恢复 |
| **v17 safe 套件误报** | Low | 已定性为脚本缺陷，修正期望/seed 后全绿 |
| **大规模 shape autotune 成本** | Low | 按需；仅在冲 200 失败时投入 |
| **技术栈锁定**（GB10 sm_121a/Triton 3.6 无原生 FP4 codegen） | Low | 路线 A 已绕开该限制；路线 B 为打散单点依赖的冗余 |

**权衡总结**：主路径路线 A 在"正确性×性能×零构建×合规低精度"上达成**当前最优可行解**；付出的是对 vLLM 0.26 内置符号的依赖与对 200 TFLOPS 门槛的小幅性能冲刺。持久化、回退、恢复三防线已设计齐备，工程风险集中于"执行与资源"，而非"方案不确定性"。

---

*本文供交接文档采纳，源码依据见 `nvfp4-landing/` 统一资料库。*