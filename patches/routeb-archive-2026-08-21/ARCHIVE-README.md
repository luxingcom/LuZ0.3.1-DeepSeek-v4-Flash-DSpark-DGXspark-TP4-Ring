# routeB 方案存档（2026-08-21）

> **状态**：已验证 standby——P4 判据未达（E2E 0.37×–1.02× vs routeA），用户裁决 kernel1 采纳 routeA，routeB 打包存档。
> **复活条件**：A 量化 kernel 化至 C++ 单 pass（~200GB/s）后，w1/w3 M∈[1024,4096] 窗口理论 E2E ~1.05-1.2×——届时再评估。

## 存档内容

### 本目录（代码工件，~1MB）
- `routeb-delivery/`——完整工作树：
  - `routeb_bench_blockscaled.py`（bench 编排器，修复后：c_dtype fp16 默认 + --c-dtype 参数 + 量化自检）
  - `routeb_official/`（vendored NVIDIA SM120 三件套：pingpong/cooperative/common，含 B-N1 护栏）
  - `routeb_prod_adapter.py`（P3 适配器：gemm(A_bf16, W_packed, W_scale)→fp16，含 atom-swizzle 双向 repack，md5 7c46209）
  - `sass_dump/`（SASS 门禁证据：cubin 41KB + 反汇编 295KB + PTX 85KB，128/128 原生 FP4 MMA）
  - B-N1 诊断工件（bn1_run.py/bn1_analyze.py/_make_probe.py）+ P3 诊断脚本 ×9 + P4 测试套件（p4/）
  - 原交付包文件（patch/setup/README/执行计划/技术规格，patch+setup 已修复验证）
- `_test_results.txt`——patch 脚本 T0-T7 验证记录

### 服务器侧（SRE 收编至 <INSTALL_DIR>/nvfp4/routeb-archive-20260821.tar.gz）
- 01:/tmp/routeb_task12/（bench 工作副本 + SASS 工件 + P4 results.json 全量 + 原始日志）
- 02:/tmp/routeb_p3/（P3 权重取证 + 诊断脚本运行副本）

### 关联报告（deliverables/engineering-assurance/）
| 报告 | 内容 |
|------|------|
| routeb-deploy-precheck-2026-08-20.md | 部署前检查（19+3 项发现，Go/No-Go） |
| routeb-fix-log-2026-08-20.md | 全部修复日志（17+2 项 + B-N1 根因 + SASS 门禁） |
| routeb-p3-semantic-2026-08-21.md | P3 语义对接（生产 MXFP4 直配 15/15 + **-hp 缺陷品证据链**） |
| routeb-p4-ab-perf-2026-08-21.md | P4 性能 A/B 全矩阵（判据未达 + 三个新发现） |
| dual-kernel-problem-list-2026-08-20.md | 双算子问题总清单（终态） |

## 技术要点速查（复活时用）
- **B-N1 铁律**：c_dtype 必须 fp16/bf16（16-bit）——f32 会静默产出 50% 垃圾（epilogue C-atom 约束）
- **SASS 取证**：`CUTE_DSL_KEEP=ptx,cubin` + `CUTE_DSL_DUMP_DIR`（编译期落盘）；grep 加 `-i`
- **性能特征**：峰值 368.1 TFLOPS @ 4096×14336×4096/128³；M=16384 崩塌 0.35×（persistent kernel 禁区）；K=14336 仅 0.42-0.60×；tile 锁定 128³（tile_k=256 MXF4 编译失败）
- **权重直配**：生产 MXFP4（[N,K//2] E2M1 + [N,K//32] E8M0）= kernel B 侧原生格式零重排（P3 rel≤4.26e-04）
- **E2E 瓶颈**：A 量化（triton 病理 20GB/s → 双 pass 修复 94-136GB/s，目标 C++ 单 pass ~200GB/s）
