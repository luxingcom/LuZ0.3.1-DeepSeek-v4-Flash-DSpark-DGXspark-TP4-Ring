# 测试覆盖与验证策略评估 — DGX Spark TP4 + NVFP4 低精度落地

> **测试专家**：泰莎（Tessa） | **日期**：2026-08-20
> **范围**：DGX Spark GB10 TP4 四机集群（01-04 / sm_121a / vLLM 0.26）NVFP4 4W4A 落地
> **审查方式**：只读研究 + 覆盖率评估。输入依据 = `architecture-nvfp4-2026-08-20.md`（阿奇，ADR/测试策略）、`sre-ops-reliability-2026-08-20.md`（雷克斯，A–F 投产门禁）、`nvfp4-landing/` 统一资料库（testing-matrix / runbook-kernel2-v17 / kernel1 / tests 全量）、`HANDOFF-TO-TEAM.md`（P0/P1/P2）、`audit-doc-vs-server-tp4-2026-08-13.md`、长期记忆基准口径。
> **目标**：产出可嵌入交接文档的**投产验收测试矩阵**，支撑「交接文档落实」；识别现有覆盖缺口并按严重度排序，给出建议补测项。

---

## 一、现有覆盖盘点

### 1.1 5 类覆盖 × 现状汇总

| 覆盖类别 | 覆盖对象 | 现有证据 | 判据 | 状态 |
|---|---|---|---|---|
| **正确性** | kernel① 路线A | `bench_mmaf_final.py` 8/8, rel=**0.00141** vs 官方 `dequantize_to_dtype`（非旧 torch 32-group） | rel<0.02 | ✅ 达标，口径正确 |
| 正确性 | kernel② v17 | `test_..._v17.py` 8/8 逐字节（`torch.equal`, mismatch=0） | byte-equal | ✅ 达标 |
| 安全性 | kernel② v17 | `test_..._v17_safety.py` 4 项过（确定性/无泄漏/NaN不崩/数值），3 FAIL 已定性为脚本缺陷 | 修期望/seed 后应全绿 | ⚠️ 脚本缺陷未修 |
| **性能** | kernel① 路线A | `bench_big.py` 60~187 TFLOPS（峰值 187 < 200 门槛） | ≥200 TFLOPS | 🔴 未达门槛 |
| 性能 | kernel① 对照 v15 | `compare_v15.py` 用**bf16 matmul 近似 v15**（非真 v15 Triton 内核） | ≥1.5× | ⚠️ 判据代理不准 |
| 性能 | kernel② v17 | `benchmark_..._v17.py` 大T 194~262 GB/s（3.5~4.6× v11） | ≥120 GB/s | ✅ 达标（实测远超） |
| **SASS** | 路线A | `sass_fp4_check.py`/`sass_fp4_deep.py`：`_C_stable_libtorch.abi3.so` 内嵌 sm_120 cubin + 1349 FP4 符号 | `mma.*e2m1\|mmaf` | ✅ 主判据正确 |
| | 历史 SASS 脚本 | `tests/sass/sass_gate.sh`/`sass_check_prefill_gemm.sh` **仍含 tcgen05**、面向旧 Triton v16 路径 | 勿用 tcgen05 | ⚠️ 陈旧，须更新 |
| **回归** | — | 无 CI/门禁编排脚本串联 正确性→性能→SASS | — | 🔴 缺失 |
| **边界** | kernel② | safety：zeros/±0/1e6/1e30/1e-30/6.0 全 byte_equal | byte-equal | ✅ 已覆盖 |
| 边界 | kernel① | **无** edge-case 脚本（仅 random scale=0.5） | — | 🔴 缺失 |

### 1.2 关键算法规格核对（确认口径正确，防止历史误判重演）

- **对照基准** ✅：kernel① 用 vLLM 官方 `dequantize_to_dtype`（`bench_mmaf_final.py` 内联 `official_ref`），**未**用旧 torch 32-group ref。规避 rel 0.19/1.35 误报（ADR-5 成立）。`probe_a1_final.py` 用 `ref_nvfp4_quant(...,16)` 16-group + `swizzle_blockscale` 双路佐证。
- **SASS 门禁** ✅：主线用 `mma.*e2m1|mmaf`（SM12x 正确指令），未用 tcgen05（SM10x）。
- **量化规格** ✅：A/W 均 `scaled_fp4_quant`（16-group e4m3）；`prob/bench` 权重构造 `amax clamp 1e-9 → max(32×128 block) → floor(log2(x/6))+127 → clamp 0..255`，与官方 E8M0 语义一致；edge 探针 `probe_v17_edge.py` 证 scale 24/24/24/144/224/24/127。

---

## 二、覆盖缺口（按严重度排序）

### 🔴 严重（投产阻断 / 最大交付缺口）

**G1. P0 持久化 import 验收无自动化测试（最高优先）**
- **缺口**：`<INSTALL_DIR>` 落位 routeA + v17 后，容器重建 → `import nvfp4_4w4a_mmaf` → preprocess+`__call__` 冒烟，**只有手工命令（阿奇 §4.1 / 雷克斯 C3），无脚本、无判定退出码**。这是交接文档 P0 的"完成判据"，却是**唯一单点不可自动化验证的环节**。
- **建议补测**：写 `tests/persist-import-gate.sh`（exit 0/1 门禁）：
  ```bash
  docker exec vllm-tp4-rank0 python3 -c "import nvfp4_4w4a_mmaf"
  docker exec ... python3 -c "from nvfp4_4w4a_mmaf import RouteA; import torch; \
    r=RouteA(); A=torch.randn(256,4096,device='cuda'); \
    Wp=torch.randint(0,16,(4096,2048),dtype=torch.uint8,device='cuda'); \
    Ws=torch.full((128,32),127,dtype=torch.uint8,device='cuda'); \
    r.preprocess_weights(Wp,Ws); assert r(A,use_cached_w=True).shape==(256,4096); print('PERSIST_OK')"
  ```
  纳入 C 段门禁。

**G2. 性能验收「≥200 TFLOPS」无法通过/无门禁脚本**
- **缺口**：kernel① 实测峰值 187 < 200（差 7%）。`bench_big.py` **只打印 TFLOPS，无 PASS/FAIL 判定**，无最低门槛断言 → P1 冲刺前后无法量化是否达标。且 `bench_big.py` 大 shape **不跑正确性对照**（只测性能）。
- **建议补测**：给 `bench_big.py` 加 200 门槛断言 + 每 shape 附带 rel<0.02 正确性核查；`exit 1` if 峰值<200。明确记录 187 为已知偏差，P1 冲刺后复测（量化融合/CUDA Graph/cutlass backend）。

**G3. A–F 投产门禁（雷克斯）无可执行测试套件**
- **缺口**：A–F 是**手工 checklist**，虽有命令/判据，但**无 shell 脚本串联、无自动 Go/No-Go 汇总、无退出码**，无法在 CI/交接时一键执行。B3（NCCL banner）、F1（Authorization 头）、D4（SASS）、F3/F4/F5（benchmark）均可自动化。
- **建议补测**：建 `tests/prod-gate/`，把 A–F 每个检查项抽成 `check_*.sh`，统一入口 `run_prod_gate.sh A` ~ `run_prod_gate.sh F`，汇总 Go/No-Go + 退出码。重点自动化：
  - **B3**：`docker logs ... | grep "NCCL version"` 须含 `2.30.7+cuda13.0`，出现 `13.3` 即 No-Go（LD_PRELOAD 失效）。
  - **F1**：`curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/v1/models` == 200。

### 🟠 高（P1，接手窗口内需补）

**G4. kernel① edge-case / 负向覆盖缺失**
- **缺口**：edge case（zeros/±0/±6/1e30/1e-30）**仅在 kernel② 有覆盖**，kernel① 路线A 只有 random scale=0.5。量化下溢/饱和/符号零在 prefill GEMM 的 W/A 两侧均未验证。
- **建议补测**：仿 kernel② 写 `test_nvfp4_4w4a_edge.py`，对 W 注入 全零 / 全±6 / ±1e30 / ±1e-30 / -0.0，assert rel<0.02（或与官方 dequant byte 一致）。

**G5. `compare_v15.py` 对照判据代理不准**
- **缺口**：用 `Ah@Wb`（**raw torch bf16 matmul**）近似 v15 —— 非真实 v15 Triton 内核（26.7~81.4 TFLOPS 那个），会低估/高估 1.5× 比值。
- **建议补测**：改用真 `nvfp4_4w4a_prefill_gemm_v15_triton.py` 作为 v15 基线测同 shape，得到可信的 ≥1.5× 判据。

**G6. SASS 历史脚本与 ADR-4 冲突（陈旧）**
- **缺口**：`tests/sass/sass_gate.sh` 与 `sass_check_prefill_gemm.sh` 仍把 `tcgen05` 当作可接受 FP4 指令（`TCGEN>0` 成立即 PASS），且面向旧 Triton v16 `dot_scaled` 路径，**不是路线A 的 cutlass 路径**。README §sass 也提到 `tcgen05.mma`。与 ADR-4「勿用 tcgen05」矛盾，误用会漏判/误判。
- **建议补测**：SASS 门禁统一只认 `mma.*e2m1|mmaf`，删 tcgen05 分支；把路线A 的 `_C_stable...so` dump SASS 判据固化为全新 `sass_gate_routeA.sh`（针对 cutlass 预编译 cubin，而非 Triton 缓存）。

**G7. 自愈恢复 / 内存告警 / 4 rank 恢复 无回归测试**
- **缺口**：雷克斯 R2（monitor/healthcheck 从 disable 恢复）、R4（UMA avail<2G 告警）、R1/P2（4 rank 恢复冒烟）均为**待办描述，无验证脚本**。交接团队无法判断"是否已恢复/是否生效"。
- **建议补测**：
  - 自愈：`systemctl is-active vllm-tp4-head.service` + `is-enabled vllm-healthcheck.timer` 断言 active/enabled（F6 落地为脚本）。
  - 内存告警：验证 Prometheus alert 规则存在 `avail_mem_bytes` metric + 告警阈值（排障/告警规则检索 check）。
  - 4 rank 恢复冒烟：`start_tp4_cluster.sh` 后 `docker ps` 4 rank healthy + `nvidia-smi` 4 GPU + `curl /v1/models` 200（F1-F3）。

### 🟡 中（P2，收尾）

**G8. safety 套件 3 个脚本缺陷未修正**
- `test_saturation` 期望 255→应 144；`test_sign_zero` 期望 1→应 24；`test_boundary_T` 未 seed（ref/got 各吃不同 randn，比对无效）。已定性但**代码文件仍是错的**，跑起来红。
- **建议**：修正 shipped 文件（期望值 + 加 `torch.manual_seed`），使 `pytest` 全绿作为门禁真实可跑。

**G9. kernel① 无确定性/显存回归测试**
- routeA 无 `test_determinism` / `test_long_run_memory`（kernel② 有）。CUDA Graph 捕获、缓存按 data_ptr 的潜坑需覆盖。
- **建议**：为 RouteA 补确定性（多次调用 byte 一致）+ 无显存增长 + Warmup（R2 加固）测试。

**G10. `evidence/` 目录为空，无基准原始输出归档**
- `nvfp4-landing/evidence/` 标注"待补"无文件；`README/runbook` 引用 TFLOPS/GB/s 表但原始 bench 输出未留存。
- **建议**：把 `bench_mmaf_final`/`bench_big`/`compare_v15`/`benchmark_v17` 的一次干净输出落 `evidence/`，做交接信任锚点。

### 🟢 良好（已在位，保持）
- kernel① 8/8 rel=0.00141 vs 官方 dequant；kernel② 8/8 逐字节 + safety 覆盖 edge；性能 v17 194~262 GB/s 远超 120 门槛；SASS 主判据正确；历史事故教训已闭环并有回归锚点（MD5/rollback-anchors）。

---

## 三、投产验收测试矩阵（可嵌入交接文档）

> 用法：每一行 = 一个可执行检查项（含判据 + 脚本/命令路径）。状态列填 ✅PASS / ❌FAIL / ⬜待跑。任一 🔴 行 FAIL 即不投产。

### 阶段 A–B：环境 / 镜像 / 补丁
| 项 | 检查 | 判据 | 脚本/命令 | 优先级 |
|---|---|---|---|---|
| A1 | 4 机在线 | 4 ping 通 | `ping <NODE_IP>~<MGMT_OCTET>` | 🔴 |
| A3 | 隔离核 | cmdline 含 `isolcpus=8-9` | `cat /proc/cmdline` | 🔴 |
| A4 | 内存头寸 | 03/04 avail≥4G | `free -g` | 🔴 |
| A6 | NCCL/shim MD5 | `b7784b49`/`ce43c688` 四机一致 | `md5sum` | 🔴 |
| B3 | NCCL banner | `2.30.7+cuda13.0`，**无 13.3** | `docker logs vllm-tp4-rank0 \| grep "NCCL version"` | 🔴 |

### 阶段 C：持久化（P0 门禁，新增自动化）
| 项 | 检查 | 判据 | 脚本 | 优先级 |
|---|---|---|---|---|
| C1 | routeA 落宿主机 | 文件存在 | `test -f <INSTALL_DIR>/scripts/nvfp4/nvfp4_4w4a_mmaf.py` | 🔴 |
| C2 | v17 落宿主机 | 存在 + md5 四机一致 | `test -d <INSTALL_DIR>/kernel2/v17/` | 🔴 |
| C3 | **容器重建 import** | import + preprocess + out shape OK，exit 0 | **`tests/prod-gate/check_c3_persist_import.sh`（新增）** | 🔴 |
| C4 | 唯一权威源 | 无生产依赖 `/vllm-workspace` | `grep -r "vllm-workspace" <INSTALL_DIR>/scripts/` | 🔴 |

### 阶段 D / F：GPU / 回归（NVFP4 专项）
| 项 | 检查 | 判据 | 脚本/命令 | 优先级 |
|---|---|---|---|---|
| D4 | SASS 门禁 | `mma.*e2m1\|mmaf` 出现（**勿用 tcgen05**） | 固化 `tests/sass/sass_gate_routeA.sh`（针对 _C_stable .so cubin） | 🔴 |
| F1 | health（鉴权） | HTTP 200（带 `Authorization: Bearer $KEY`） | `curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/v1/models` | 🔴 |
| F2 | chat 冒烟 | `2+2=?` → "4" | 容器内 curl chat | 🔴 |
| F3 | kernel① 正确性 | 8/8 rel<0.02 vs 官方 dequant | `python tests/bench_mmaf_final.py` | 🔴 |
| F4a | kernel① 性能 | **峰值≥200 TFLOPS**（当前 187，已知偏差需冲刺后复测） | `python tests/bench_big.py`（**加 200 断言**） | 🔴 |
| F4b | kernel② 带宽 | 大 T ≥120 GB/s（实测 194~262） | `python tests/kernel2/benchmark_nvfp4_ds_mla_kv_linear_v17.py` | 🔴 |
| F5 | 对照 v15 | **≥1.5×（用真 v15 Triton 内核，勿用 bf16 matmul 代理）** | `python tests/compare_v15.py`（改基线） | 🔴 |
| F6 | 自愈在位 | head monitor active + healthcheck timer enabled | `systemctl is-active/is-enabled` | 🔴 |
| — | kernel② 正确性 | 8/8 逐字节 | `python -m pytest tests/kernel2/test_..._v17.py` | 🟠 |
| — | kernel② 安全 | 修脚本缺陷后全绿 | `python -m pytest tests/kernel2/test_..._v17_safety.py` | 🟠 |
| — | kernel① edge | zeros/±0/±6/1e30/1e-30 rel<0.02 | 新增 `tests/kernel1/test_nvfp4_4w4a_edge.py` | 🟠 |
| — | kernel① 确定性/泄漏 | byte 一致 + 无泄漏 | 新增 RouteA determinism/mem 测试 | 🟠 |

**判定规则**：阶段 A–C 全 Go（🔴 全 PASS）→ 可进入 D/F；F1–F2 Go → 基础可投产；F3–F6 为 NVFP4 专项深化，F4a 未达 200 需单独立项决策并记录偏差（不构成基础阻断，但 P1 冲刺必须启动）；F4a/F5 判据补齐前不得宣称"性能验收通过"。

---

## 四、事故复验清单（08-19 SEV1 复盘 → 回归验证）

| # | 复验项 | 判据 | 脚本/动作 | 状态 |
|---|---|---|---|---|
| RC1 | **事故格复验**（65536/coding/conc3） | conc3×65536 连续≥4 次不触发 UMA 耗尽，内存谷底未归零 | benchmark 分段恢复，事故格单独跑（雷克斯 §4.1.5 未闭环项） | ⬜ 待跑 |
| RC2 | **UMA 内存告警生效** | avail<2G 触发告警（防复发） | Prometheus alert 规则检索 + 触发验证 | ⬜ 待补 |
| RC3 | **NCCL 超时被动受害者识别** | 出现 300s 超时先查 avail（UMA）再查网；banner 非 13.3 | 排障 SOP 走查 + `grep "NCCL version"` 断言 | ⬜ 待跑 |
| RC4 | **monitor/healthcheck 自愈还原** | is-active/service + is-enabled/timer，显式 mask 定格 | `systemctl` 断言脚本 | ⬜ 待补 |
| RC5 | 4 rank 恢复冒烟 | `start_tp4_cluster.sh` → 4 rank healthy + 4 GPU + /v1/models 200 | prod-gate F 段 | ⬜ 待跑（P2 决策后） |
| RC6 | 停机窗口防误拉 | 先 stop timer+service → 停容器 → 完成后恢复 | 演练脚本 | ⬜ 待跑 |

---

## 五、测试就绪度评级

> **评级：🟡 有条件通过（核已验，壳未测）**
> 内核数值正确性/性能数据可信（kernel① rel=0.00141、kernel② 8/8 逐字节 194~262 GB/s），但**验收测试框架未就绪**：P0 持久化 import、A–F 投产门禁、200 TFLOPS 性能门槛、1.5× 对照（真 v15）、SASS 路线A 门禁、自愈/内存告警 均**无自动化可执行脚本**。

| 维度 | 就绪度 | 说明 |
|---|---|---|
| 正确性 | 🟢 高 | 官方语义口径正确；kernel①② 均已达标 |
| 边界/负向 | 🟡 中 | kernel② 强、kernel① 缺（G4）；事故格（RC1）待复验 |
| 性能 | 🟠 中低 | kernel② 达标；kernel① 距 200 差 7% 未冲刺（G2）；v15 对照代理不准（G5） |
| SASS | 🟢 高（主线）/ 🟡 中（脚本陈旧 G6） | 主判据正确；历史脚本须更新 |
| 回归/门禁自动化 | 🔴 低 | 无 CI/门禁套件（G1/G3/G7） |

**投产前必须补齐**（否则不能宣称"已投产"）：
1. C3 持久化 import 门禁脚本（G1）— 唯一单点锁定 P0；
2. F4a 200TFLOPS + F5 真v15 对照（G2/G5）；
3. `run_prod_gate.sh` A–F 一键门禁（G3）+ B3/F1 自动化；
4. SASS 路线A 门禁替换陈旧脚本（G6）；
5. 自愈/内存告警/4rank 恢复回归（G7）+ 事故格复验（RC1–RC6）。

---

## 附：建议新增/修正脚本清单（交 tech-writer 并入交接文档）

| 文件 | 类型 | 用途 | 对应缺口 |
|---|---|---|---|
| `tests/prod-gate/check_c3_persist_import.sh` | 新增 | P0 import 冒烟 | G1 |
| `tests/prod-gate/run_prod_gate.sh` | 新增 | A–F 一键门禁 + Go/No-Go 汇总 | G3 |
| `tests/sass/sass_gate_routeA.sh` | 新增 | 路线A cutlass cubin 门禁（只认 mma.*e2m1\|mmaf） | G6 |
| `tests/kernel1/test_nvfp4_4w4a_edge.py` | 新增 | kernel① edge-case | G4 |
| `tests/kernel1/test_routeA_determinism.py` | 新增 | 确定性/显存/暖机 | G9 |
| `tests/bench_big.py` | 修正 | 加 200 断言 + shape 附正确性 | G2 |
| `tests/compare_v15.py` | 修正 | 用真 v15 Triton 内核作基线 | G5 |
| `tests/kernel2/test_..._v17_safety.py` | 修正 | 期望值/seed 修正 | G8 |
| `tests/sass/sass_gate.sh`, `tests/sass/sass_check_prefill_gemm.sh` | 修正/归档 | 删 tcgen05、标注面向旧路径 | G6 |

*本报告由工程保障团队测试专家泰莎基于只读研究生成，未修改任何生产脚本。所有补测脚本待实现落地；200 TFLOPS 是否硬门槛、事故格复验时机请由人类工程负责人裁决。*