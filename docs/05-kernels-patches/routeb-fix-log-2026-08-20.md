# routeB 修复日志 — 2026-08-20

## Task #13：patch + setup 脚本修复 + DSL 版本核验（SRE · 雷克斯）

**修复对象**:
- `C:\Users\novAI\WorkBuddy\集群部署\_routeb_extract\routeb-delivery\patch_cutlass_dsl_sm121a.py`
- `C:\Users\novAI\WorkBuddy\集群部署\_routeb_extract\routeb-delivery\setup_routeb_env.sh`

**约束遵守**: 只动上述两个本地文件 + 本地 mma.py 副本；服务器仅一次性容器只读取证（`docker run --rm`，无 GPU 绑定，用后即删）；未启动生产。

---

## 1. patch_cutlass_dsl_sm121a.py 修复明细

| 项 | 行动清单 | 修复内容 | 状态 |
|---|---|---|---|
| A2 (P0) | equality-check 变换 SyntaxError | 新 `EQ_PATTERN = r'if\s+(not\s+)?arch\s*([!=]=)\s*Arch\.sm_120a\s*:'`，按捕获组语义生成 `if arch in (Arch.sm_120a, Arch.sm_121a):`（==）或 `if arch not in (...)`（!= / not ==）；替换后天然不再匹配模式 → 幂等 | ✅ |
| A12 | 字符串拼接假定双引号 | `_insert_121a_into_list()`：枚举 `Arch.sm_120a` → `Arch.sm_120a, Arch.sm_121a`（优先）；带引号（单/双自适应保持风格）`'sm_120a'` → `'sm_120a', 'sm_121a'`；裸 token 兜底；列表已含 sm_121a 则跳过（幂等）；不含 sm_120a 的列表不动并告警 | ✅ |
| A13 | 非原子写 | `_atomic_write()`：写 `*.tmp-routeb` + `os.replace`；读写均 `newline=''` 防行尾漂移 | ✅ |
| A19 | revert 无备份静默 | 无 `.bak-routeb` 时 stderr 输出 + `sys.exit(1)` | ✅ |
| 附带1 | （新发现）原版只 patch 第一处 admissible_archs | 改为 `ADMISSIBLE_PATTERN.sub()` 处理**所有**含 sm_120a 且不含 sm_121a 的列表 | ✅ |
| 附带2 | （新发现）硬编码定位，无法对副本测试 | 新增 `--target PATH` 参数（缺失/不存在时 exit 2），`--revert` 可与之组合 | ✅ |
| 附带3 | 验证提示引用 BlockScaledMmaOp | 4.5.2 中该名已不存在（实测 ImportError），提示改为 `MmaMXF4Op`（继承 MmaSM120BlockScaledOp.admissible_archs） | ✅ |

## 2. setup_routeb_env.sh 修复明细

| 项 | 行动清单 | 修复内容 | 状态 |
|---|---|---|---|
| A10 | driver 版本仅 echo | `nvidia-smi` 不可用即 exit 1；`sort -V` 比较 `DRIVER_VER` ≥ 580.142，不满足 exit 1；无法取版本 exit 1 | ✅ |
| A11 | 备份可能覆盖干净备份 | `.bak-routeb` 已存在则跳过并提示"保留最早的干净备份" | ✅ |
| A16 | --no-deps 假设预装 CUDA13 runtime | 显式 `pip install --no-deps nvidia-cuda-runtime-cu13` | ✅ |
| A21 | cp 无 -p | `cp -p`（保留权限/时间戳） | ✅ |
| 附带 | （新增信息性检查，不拦截） | 安装前探测已装 nvidia-cutlass-dsl 版本，≠4.4.2 时输出降级警告（版本策略归 architect 决策） | ✅ |

## 3. 验证结果（_routeb_extract/_test_results.txt 全文留档）

| 测试 | 内容 | 结果 |
|---|---|---|
| T0 | `bash -n setup_routeb_env.sh` | ✅ SYNTAX_OK |
| T1 | patch 脚本 ast.parse | ✅ |
| T2 | 对**生产镜像导出的真实 mma.py 副本**（DSL 4.5.2）跑 patch | ✅ 无变更（no-op），副本 byte 级不变，仍可 ast.parse |
| T3 | 对合成 4.4 风格文件（双引号/单引号/枚举三种列表 + ==/!=/not== 三种 equality）跑 patch | ✅ 全部正确变换，产出可 ast.parse |
| T4 | 幂等性（重复跑 patch） | ✅ md5 前后一致 |
| T5 | diff 检查变更点 | ✅ 五处变更全部正确（引号风格保持、`in`/`not in` 语义正确、无语法破坏） |
| T6 | `--revert` 回滚 | ✅ 副本恢复至原始（diff -q 通过） |
| T7 | 无备份 revert | ✅ exit 1 + stderr 报错（非静默） |

关键 diff 样例（T5）：
```diff
-        "sm_120a",
+        "sm_120a", "sm_121a",
-        if arch == Arch.sm_120a:
+        if arch in (Arch.sm_120a, Arch.sm_121a):
-        if arch != Arch.sm_120a:
+        if arch not in (Arch.sm_120a, Arch.sm_121a):
-        if not arch == Arch.sm_120a:
+        if arch not in (Arch.sm_120a, Arch.sm_121a):
-    admissible_archs = ['sm_100a', 'sm_120a']
+    admissible_archs = ['sm_100a', 'sm_120a', 'sm_121a']
-        Arch.sm_120a,
+        Arch.sm_120a, Arch.sm_121a,
```

测试工件：`_routeb_extract/_testdata/`（mma_44_style_synth.py 合成夹具、mma_real_452.py 真实副本）；真实副本源：`_routeb_extract/mma_prod_copy.py`（自生产镜像 `nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/warp/mma.py` 导出，21445B，sha 留档于 _test_results.txt 语境）。

## 4. DSL 版本核验数据（供 architect 版本决策；一次性容器只读）

镜像：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（node01）

```
nvidia-cutlass-dsl            4.5.2
nvidia-cutlass-dsl-libs-base  4.5.2
nvidia-cutlass-dsl-libs-core  4.6.0   ← 混装
nvidia-cutlass-dsl-libs-cu12  4.6.0   ← 混装
nvidia-cutlass-dsl-libs-cu13  4.5.2
cutlass.__version__ = 4.5.2
cutlass.__file__ = /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
```

- `from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op` → **成功**（`<class 'cutlass.cute.nvgpu.warp.mma.MmaMXF4Op'>`）
- `from cutlass.cute.nvgpu.warp.mma import BlockScaledMmaOp` → **ImportError（4.5.2 已无此类）**——routeB 内核若 import 此名将直接失败
- 实际 import 的 mma.py：`nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/warp/mma.py`（另有 dsl_packages 同名副本树，非 import 根）
- **★ 该 mma.py 的 `MmaSM120BlockScaledOp.admissible_archs` 已含 `Arch.sm_121a`（L225-230：[sm_120a, sm_120f, sm_121a, sm_121f]），`MmaMXF8F6F4Op.admissible_archs` 亦含（L597-600）；全文件无 equality check（已是 `not in` 成员检查）**

### 对版本决策的三个关键含义
1. **issue #2800 patch 在 4.5.2 上不需要**——setup 若按钉版安装 `4.4.2` 属**降级**，会重新引入 patch 需求；建议 architect 评估直接用镜像内 4.5.2（省 Phase 0/1 全部动作，只留验证）。
2. **patch 脚本已修好且实测**：若最终决策仍是 4.4.2 路线，修复版脚本可安全使用（A2/A12/A13/A19 全过）。
3. **routeB 内核代码需核对 import 目标**：BlockScaledMmaOp 在 4.5.2 已不存在（改名/重构为 MmaSM120BlockScaledOp + MmaMXF4Op 族），4.4 写法的 import 路径在 4.5.2 上会炸。

## 5. 遗留/移交

- setup 钉版 `4.4.2` vs 镜像内 `4.5.2` 的取舍 → **移交 architect**（Task #12 上下文）
- 本任务执行中本机 Bash/PowerShell 工具 stdout 捕获通道故障（命令实际执行正常），取证改用「输出重定向落盘 + Read」模式，证据文件均在 `_routeb_extract/` 下（_dsl_verify.txt、_dsl_verify2.txt、_test_results.txt 等）

**—— Task #13 完，2026-08-20，雷克斯（SRE）**

---

## Task #12：routeB bench 脚本全量修复 + 官方 kernel 冒烟（架构 · 阿奇）

**修复对象**: `_routeb_extract/routeb-delivery/routeb_bench_blockscaled.py`（整文件重写为 v2 编排版）+ 新增 `routeb-delivery/routeb_official/` 三件套（官方示例 vendor）

**约束遵守**: 服务器仅 `docker run --rm --gpus all` 一次性容器（用后即删，未动生产容器/镜像/挂载）；工作目录 /tmp/routeb_task12（宿主机临时目录）；未启动生产。DSL 4.6.0 测试为容器内 pip 临时安装（容器即删，无持久影响）。

---

### 1. 修复明细（9 项全落地，其中 A1 方案有架构级偏离，见 §3）

| 项 | 行动清单 | 修复内容 | 验证方式 | 状态 |
|---|---|---|---|---|
| A1 | host launch NotImplementedError | **方案偏离**：v1 缺的不是 ~150 行 host launch，而是**整个设备侧 kernel（~1000 行）**。v2 改为 vendor 官方 SM120 完整实现（`routeb_official/` 三件套，NVIDIA/cutlass main 分支 `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`），bench 重写为编排器 | 容器内编译+执行成功（kernel 真实跑通） | ✅（见 §4 新阻塞） |
| A3 | encode_e8m0_32 漏 +127 | `floor(log2(clamp(amax,1e-30)/6)) + 127` 后 clamp [0,255]（clamp 顺序一并修正） | --check T1：零输入→24、1e6→144 ✅ | ✅ |
| A5 | SMEM 误计 acc_bytes | 删除 acc_bytes 项（F32 acc 驻留 RF）；smem_estimate 降级为信息性参考（官方 kernel 内部自算 stages） | 代码审查 + 打印参考值 | ✅ |
| A6 | W_packed 布局错误 | `pack_e2m1_n(W)` 直出 [K, N//2]（删除 v1 的 `.t()` 转置链） | --check T2：shape + 逐字节抽查 ✅ | ✅ |
| A7 | MmaMXF4Op dtype FP8 | **现场定案**：4.5.2 实测签名 `MmaMXF4Op(ab_dtype, acc_dtype, sf_type)` 三位置参数（v1 的 5 关键字写法必然 TypeError）；经官方 dispatch 构造 `MmaMXF4Op(Float4E2M1FN, Float32, Float8E8M0FNU)`，sf_vec_size=32 固化在类内 | 容器内 `inspect.signature` 实测输出留档（_smoke_bench.txt） | ✅ |
| A8 | W_scale 非 block-max | 新 `make_w_scale()`：32(K)×128(N) block-max，对齐生产 [K//32, N//128] | --check T3：构造性断言（6→127、96→131、零→24）✅ | ✅ |
| A14* | 绝对误差过松 | rel_err = max_err / ref.abs().max()（阈值 0.3，NVFP4+floor 语义区间 0.1-0.25） | --check T4：实测 0.2473 ✅ | ✅ |
| A15* | --check 仅 print | 5 组真数值断言（T1 编码校准向量 / T2 打包字节 / T3 block-max / T4 量化往返 / T5 确定性） | 生产容器实跑全过 ✅ | ✅ |
| A20 | 计时循环内重构 kernel | 官方 `cute.compile` 一次编译 + `cutlass.testing.benchmark` 工作区轮换计时（构造在计时外，架构性满足） | 代码结构审查 | ✅ |

*注：A14/A15 编号按 team-lead 派发单（对应 precheck 的 A15/A13）。

**附带修复（--check 真断言暴露的 v1 潜伏 bug，precheck 未列入）**：
- `torch_reference` 的 `A_deq @ W_deq.t()` 形状非法（[M,K]@[N,K]）——v1 因 host launch 从未执行到该行而漏检 → 改 `@ W_deq`
- `torch_reference` 的 W 侧只乘 scale 未做 E2M1 码本量化（量化从未发生）→ 补完整 W 量化路径（W/scale→clamp→码本→×scale）

### 2. 新增文件

| 文件 | 说明 |
|---|---|
| `routeb-delivery/routeb_official/dense_blockscaled_gemm_persistent_pingpong.py` | 官方 SM120 warp-level blockscaled GEMM（pingpong 调度）+ 2 处 Task#12 本地补丁（见 §3） |
| `routeb-delivery/routeb_official/dense_blockscaled_gemm_persistent_cooperative.py` | 官方 cooperative 调度变体 + 同款补丁 |
| `routeb-delivery/routeb_official/blockscaled_gemm_dispatch.py` | 官方 MMA op 分派（MXF4 路径：FP4+UE8M0+vec32 → MmaMXF4Op） |

### 3. 对官方示例的本地补丁（2 处 ×2 文件，均有注释标记）

1. **精确可表示输入**（patch 1/2）：A/B 取 E2M1 精确可表示值（0.5 整数倍）+ 随机符号；scale 取精确 2 的幂。原因：上游参考校验（einsum）不模拟 E2M1 量化/E8M0 取整，对随机输入必然失败（实测 86.5% mismatch）；补丁后参考校验数学上精确，成为真门禁（对齐 SM100 示例 emulated 语义）。
2. **诊断模式**（ROUTEB_DIAG=ones/scales 环境变量）：全 1 输入（期望 C=K 逐元素精确）/变 scale 模式，用于 §4 的排障。
3. 4.5.2 兼容 shim（bench 侧非示例侧）：`cutlass.testing` → `cutlass.cute.testing` 模块别名（4.5.2 无前者，后者含 JitArguments/benchmark/get_workspace_count/convert，逐一核实存在）。

### 4. 冒烟结果与【新发现的 P2 硬阻塞】

**✅ 通过项**：
- `--check` 量化自检 5/5 全过（生产镜像容器实测）
- DSL 环境探测：4.5.2 实测 `MmaMXF4Op(ab_dtype, acc_dtype, sf_type)`、`admissible_archs` **原生含 sm_121a**（P1 patch 在 DSL≥4.5 为 no-op，仅 4.4.x 需要——与 Rex Task#13 T2 结论互证）
- 官方 kernel 在 GB10/sm_121a/4.5.2 上**编译成功并真实执行**（自研骨架到可执行的最远推进）
- 端到端链路打通：shape 4096×14336×4096 完成 10 次迭代计时，162.2 TFLOPS（skip-ref-check，见下判读）

**🔴 新阻塞 B-N1（P2 级，比原 B1 更深）**：官方示例 kernel 输出**结构性 50% 为零**——每 tile 的 M 下半（epi 128 时 row 64 起；epi 64 时 row 32 起）从未写出。隔离矩阵：
| 实验 | 结果 |
|---|---|
| pingpong + MXF4(vec32) + tile 128³ | 50% 零 @ (64,0,0) |
| pingpong + NVFP4(vec16, 官方默认路径) | 50% 零 @ (64,0,0)（非路径特定） |
| cooperative + MXF4 | 50% 零 @ (64,0,0)（非调度特定） |
| tile 数 4→8→64（128³→1024²） | 恒 50%（非 persistent 调度边界） |
| epi_tile 128,128 → 64,32 | 43.8%（零区边界随 epi M 减半移动） |
| DSL 4.5.2 → 4.6.0（容器内临时装） | 完全相同（非版本漂移） |

判定：**共享基础设施级缺陷**（TiledMMA/permutation 构造或 epilogue 覆盖），影响官方两变体、两路径；MMA 数值通路本身正确（零区之外的输出逐位精确=1024，说明 mma.sync mxf4 + SFA 应用是对的）。GitHub 上 blackwell_geforce 示例无 CI 测试文件（已核实 main 树无 test），该目录 2026-08 新增，成熟度存疑。

**⚠️ 性能判读（162.2 TFLOPS 的两个折扣）**：
1. 半输出 bug → 实际计算量减半，真实效率估计 ~81 TFLOPS（若 bug 修复后时间翻倍）
2. 该数字取自 tile 128×128×128（本示例唯一可用 prefill 级 tile；baristankut 的 356 基线用 tile 256×128，**本 DSL 示例不支持**——交付包"256×128 最优 tile"来自 SGLang C++/builder 路径）。tile 128×128×256 尝试在 SMEM 布局构造处报错（tile_to_shape size=0，待查）。

**对 routeB 计划的影响（架构师意见，供主理人裁定）**：
- 官方 DSL 示例路径（routeB 当前唯一母本）在 GB10 上同时存在正确性 bug 与性能缺口（~81-162 vs 356 目标），P2 达标概率显著下调
- 356 单源不可复现性加深：baristankut 的 356 可能依赖 SGLang C++ collective builder 路径（GitHub commit 记录显示 SM120 ptr-array grouped GEMM 走 C++ builder，非本 DSL 示例）
- 备选路径排序建议更新：① SGLang/flashinfer 的 SM120 C++ builder 路径（community Docker 镜像 ghcr.io/btankut/sglang-spark-glm47 可直接取证）② 上游 issue 反馈官方示例 bug（附本次隔离矩阵）③ 维持 routeA

### 5. 执行环境留档

- 宿主机暂存：node01:/tmp/routeb_task12/（bench + routeb_official；一次性容器用后即删）
- 容器镜像：<NODE_IP>:5000/anemll/dspark-vllm-gx10:test-0.2.1-v027-fix121a-dg250-ijson-parser（生产镜像，未做任何修改）
- 本地留档：_routeb_extract/_smoke_check.txt（--check 5/5）、_smoke_bench.txt、_diag_*.txt（隔离矩阵）、_main_shape_perf.txt（162.2 TFLOPS 全文）、_perf_num.txt（微 shape 15.1）
- SASS 门禁未完成项：DSL 编译缓存为 compiled_cache.db（非独立 cubin 文件），nvdisasm 提取需从 db 中导出 blob——已列入待办（正确性未过前 SASS 门禁意义有限）

---

## kernel2 修复（Cody 重启执行，Task #11）

**背景**：前修复代理（kernel2-fixer）中途被终止。Cody（kernel2-fixer-2）重启执行：先逐文件核对前代理遗留编辑，再独立完成真机验证与回填。审查依据：`code-review-kernel2-2026-08-20.md`（13 项发现）。

### 1. 前代理遗留编辑核对结论（逐文件）

| 项 | 文件 | 状态 | 核对详情 |
|---|---|---|---|
| P0-1 | test_nvfp4_ds_mla_kv_linear_v17_safety.py | ✅ 已由前代理完成 | :44 `==144`（含不可达性注释）、:52 `==24`（错误注释已删）、:59-60 共用 kv + manual_seed(20260820)，三处全部到位 |
| P0-2 | kernel2_v17_safety_reliability.md | ✅ 已完成 | §一 :15 改 24、§八 :79 改 144、:14 clamp 可达域 [1,254] 勘误、§九修订注记均在 |
| P0-3 | paged_triton / paged_torch / test_paged | ✅ 已完成 | BLOCK_SIZE 从 `kv_cache.shape[1]` 派生下传（:158 硬编码已删）；safe_max 1e-30 / exp clamp [-126,127] / 除数防护已去除（triton+torch 双侧对齐）；测试 64 槽×5 + 256 槽×2 兼容 + 跨 linear/paged 解码值冒烟×1 |
| P1 | benchmark_v17 / README | ✅ 已完成 | BYTES_PER_TOKEN 分版本计（v11=584 / v17=1160 含 memset）；512MB 多缓冲轮换；README §一 L2 驻留口径注记 |
| P2 | 6 文件 | ✅ 已完成 | 版本注释（v5/v4/v6→现行）、README 计数（15 文件 / 7 bit-exact+1 smoke / 8 组）、v11 wrapper shape 断言（linear_triton.py:182）、paged 死代码（tle/num_blocks/DIM/ENVELOPE）与 evict_first 均落实 |

**红线核验**：v17 triton md5 `a795b2b4a486f8bd2b07366890e928af`（=生产锚点，全程未动）；v11 triton `4f3aa97b...` 与 linear_torch `aa31420b...` 内核逻辑零改动（仅注释 + wrapper 断言）。

### 2. 真机独立复核（node01 一次性容器，Cody 执行）

- 镜像：`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（生产镜像，`docker run --rm --gpus all`，容器内临时 pip 装 pytest/matplotlib/pandas，用后即删）
- 交付包上传 `/tmp/kernel2_fix_cody`（上传后服务器端 v17 md5 复核一致），验证完毕已 sudo 清理（容器以 root 写入 __pycache__）

| 测试 | 结果 | 关键输出 |
|---|---|---|
| test_v17_safety.py | **7/7 PASSED**（11.88s） | test_saturation/sign_zero/boundary_T（初版 3/7 失败点）全过 |
| test_v17.py | **8/8 PASSED** | 7 组 bit-exact + 1 kv 入口 smoke |
| test_linear.py（v11） | **7/7 PASSED** | 佐证 v11 红线文件在新增 shape 断言后行为不变 |
| test_paged.py | **8/8 PASSED**（417.40s） | 64 槽×5（生产块大小）+ 256 槽×2（兼容）+ 跨路径冒烟 |
| benchmark_v17.py | 修正口径复测 | T=256→68.2 / 1024→112.3 / 4096→179.8 / 16384→211.9 / **65536→211.1 GB/s**（HBM 口径，273 理论的 77%，3.9× v11）；v11 对照 39.2/49.9/57.7/56.6/54.7 |

### 3. 本轮 Cody 增量修改（前代理未完成的收尾）

1. **README 数字回填为独立复核值**：§一部署矩阵 183~217→**180~212 GB/s**（3.7~3.9× v11）；各 T 档表更新为上表数字；口径注记结尾改为本轮独立复核留档表述（删除无法溯源的"A/B 实验 do_bench 自带 L2 清理"表述）
2. **safety 报告残留旧口径数字勘误**：头部"194~262 GB/s 已达标"→"修正口径 180~212 GB/s"；§九结论"194~262"→"180~212"
3. **README §八修订记录**追加"Cody 重启执行·独立复核"条目

### 4. 结论

- 审查 13 项发现全部闭环：P0×2（含 paged Critical 落锤项：BLOCK_SIZE 参数化 + 语义统一 v17 金标准）、P1×2、P2×5
- v17 生产留任不受影响（md5 全程一致）；**paged 变体自本轮起解除禁部署状态**（64 槽真机逐字节 8/8 + 语义与 v17 金标准统一）
- 遗留（非本轮范围）：R3 paged 性能向 v17 架构移植；审查 #13 的 paged 重复 (seq,position) 冲突用例与 benchmark_paged ref 预分配（低优先级，已在审查报告记录）

**—— Task #11 完，2026-08-20，科迪（Cody）· 代码审查师（kernel2-fixer-2）**

---

## B-N1 深挖与修复：DSL 半零缺陷根因定位（Task #16 · 架构 · 阿奇，2026-08-20）

**结论先行：B-N1 根因不在官方 kernel，而在 bench 编排器传入 `c_dtype=Float32`。官方 SM120 示例的 epilogue C-atom（`StMatrix8x8x16bOp`）只对 16-bit 输出 dtype 成立；f32 使 tiled-copy 每线程值数减半（4→2）、M tiler 粒度减半（32→16 行），epilogue 填充循环按 32 行粒度索引 → 一半累加器值被丢弃、一半寄存器槽位未初始化 → 输出结构性 ~50% 错误。改 `c_dtype=Float16` 后 100% 逐位正确，主 shape 4096×14336×4096 @ tile 128³ 368.1 TFLOPS（参考校验通过，>350 门禁，>356 社区基线）。**

**修复对象**:
- `_routeb_extract/routeb-delivery/routeb_bench_blockscaled.py`（c_dtype Float32→Float16 + `--c-dtype` 参数 + 护栏说明）
- `_routeb_extract/routeb-delivery/routeb_official/dense_blockscaled_gemm_persistent_pingpong.py`（非 16-bit c_dtype 显式 raise，防静默半错）
- `_routeb_extract/routeb-delivery/routeb_official/dense_blockscaled_gemm_persistent_cooperative.py`（同款护栏）

**约束遵守**：服务器仅 `docker run --rm --gpus all --entrypoint bash` 一次性容器（全部用后即删，未动生产容器/镜像/挂载）；工作目录 /tmp/routeb_task12；未启动生产。

### 1. 排查路径（方法论对齐用户资料 dsl_half_zero_diagnosis.md）

1. **上游 diff**：vendored 三件套 vs NVIDIA/cutlass main 逐字节一致（仅 2 处 Task#12 文档化的 host 侧输入生成补丁，不触 kernel）→ 排除 vendor 引入差异；上游该文件仅 2 个 commit（#3272 新增 2026-05-27、#3315 v4.6 整理 2026-06-16），无后续修复。
2. **对照样本**：同目录非 blockscaled 的 `dense_gemm.py`（单 math warpgroup、`get_slice(tidx)` 无 `%128`、StMatrix x4）结构对照 → 锁定 epilogue 差异面。
3. **零形态精确化（§三 dump 法）**：sentinel-C（-777 预填）+ ROUTEB_DIAG=ones（期望 C≡K）实验（shape 256×256×512）：
   - **sentinel 残留 = 0** → TMA store 覆盖完整 epi tile，排除"TMA box 半覆盖/未写出"假设；
   - **zero = 0、other = 50%** → 坏区是**写入的错误值**而非未写/写零——纠正 Task#12"kernel 值=C 初始零"的判读；
   - 坏区值实为 **~1e-38/1e-41 量级非正规数**（未初始化寄存器/SMEM 位型，共 129 个唯一值），打印为 0.0000、肉眼/宽松判零即"半零"假象。**全案闭合**。
4. **trace 期 layout 内省**：probe 副本（`dense_blockscaled_gemm_persistent_pingpong_probe.py`，`_make_probe.py` 生成）在 jit 追踪期打印 tiled_mma/epilogue 全部静态形状，拿到 f32/f16 两组决定性形状对照。

### 2. 根因机制（行号级）

| 环节 | 位置 | 事实 |
|---|---|---|
| 触发点 | bench 旧 L351 `c_dtype=cutlass.Float32` | 官方示例 CLI 默认 Float16，bench 硬编码 f32 |
| C-atom 构造 | pingpong L1133-1139 `copy_atom_C = make_copy_atom(StMatrix8x8x16bOp(is_m_major, 2), self.c_dtype)` | **16-bit 专用 stmatrix 以 f32 实例化**：每线程值数 4（fp16）→2（f32） |
| 派生 tiler | `make_tiled_copy_C_atom`→`make_tiled_copy_S` | f32 下 Tiler MN 的 M 粒度 32→16 行（probe 实测 `(8,2):(1,16)` vs fp16 `(8,2,2):(1,16,8)`） |
| 失配现场 | probe 实测形状 | **f32**: `tRS_rAcc=((1,4),4,8)` vs `tRS_rD/tRS_sD=((1,2),(2,4),(2,4))`（R2S 4≠2、M 粒度 4×32行≠8×16行）；**fp16**: `((4,1),4,8)` vs `(((2,2),1),4,(2,(2,2)))` 同构 |
| 丢值循环 | pingpong L1188-1203 | `for elem_idx in range(size(tRS_rD_slice))`（=2）拷自 rAcc 切片（=4 值）→ 丢一半；`mma_m_in_epi∈[0,MmaMPerEpiM=4)` 只填 rD 8 个 M 槽位中的 4 个 → 另一半未初始化 |
| 形态预测验证 | epi 128→零界 row 64；epi 64,32→零界 row 32 | rD M 槽位 = epi_m/16 个，循环只填前 epi_m/32 个 → 零界恒 = epi_m/2，与 Task#12 隔离矩阵 6 组实验全部吻合 |

辅助事实：`sm120_get_smem_store_op`（DSL 4.5.2 blackwell_helpers.py L1667）对非 16-bit 返回 `CopyUniversalOp` 作 r2s atom，但 `copy_atom_C`（定义 C 分区的 atom）**无条件**用 StMatrix——上游示例对 f32 的"半支持"正是静默半错的温床。

### 3. 修复内容

1. **bench**：`c_dtype` 默认 Float16（生产 prefill GEMM 输出即 fp16/bf16，语义无损失），新增 `--c-dtype {Float16,BFloat16}`；文档注明 f32 不可用原因。
2. **vendored 两文件**：`run_bs` 内 dtype 解析后加 `c_dtype.width != 16 → ValueError`（含 B-N1 编号与日志指引），变静默半错为显式失败。
3. 诊断工件保留：`_make_probe.py`（probe 生成器）、`dense_blockscaled_gemm_persistent_pingpong_probe.py`、`bn1_run.py`/`bn1_analyze.py`（sentinel/零形态/坏值分析驱动）。

### 4. 复测矩阵（node01 一次性容器，生产镜像 DSL 4.5.2，sm_121a）

| 实验 | 配置 | 结果 |
|---|---|---|
| E1 | f32 + ones + sentinel | 50% other（坏区=非正规垃圾值），每 128 行带 top 64 行精确 ==K |
| E2 | **fp16** + ones + sentinel | **100% ==K（65536/65536），零 sentinel/零 zero/零 other** |
| V1 | fp16 + 精确可表示随机 + **全参考校验** 256³ | ✅ PASS（RC=0） |
| V1b | 同上 1024³ | ✅ PASS |
| V2 | **BFloat16** + 全参考校验 | ✅ PASS |
| V3 | **NVFP4 (vec16, E4M3)** + fp16 + 全参考校验 | ✅ PASS（隔离矩阵中 vec16"同败"实为 c_dtype 共因） |
| V4 | fp16 + epi 64,32 + ones + sentinel dump | **100% ==K**（epi 变体同修） |
| V5 | f32（修补后官方模块） | ✅ ValueError 护栏触发 |
| V9 | **cooperative** + fp16 + 全参考校验 | ✅ PASS（协作变体同修） |
| **V6** | **4096×14336×4096 @ 128×128×128，fp16，参考校验通过** | **368.1 TFLOPS**（1306.8 µs/iter；此前半错态 162.2 → 修复后全对且更快：fp16 epilogue 存储带宽减半） |
| V7 | tile 128×128×256 + sf_vec32 | ❌ SFA 拷贝视图 congruence 报错（pingpong L944，上游示例 K=256+vec32 限制，**与 B-N1 无关**；128³ 已达标故不阻塞） |
| V8 | f32 坏值取证 | 坏区唯一值 129 个，全部 ~1e-38/1e-41 非正规数（未初始化位型）→"半零"假象解释 |

证据文件（服务器 /tmp/routeb_task12/）：`bn1_e1_log.txt`、`bn1_e2_log.txt`、`bn1_suite_log.txt`、`bn1_perf_log.txt`、`bn1_C_*.pt`（dump）、`v4_epi64.pt`；本地 `_routeb_extract/` 同步留档。

### 5. 遗留与移交

1. **上游 issue 素材（待用户批准后再提交）**：NVIDIA/cutlass `blackwell_geforce/blockscaled_gemm` 示例对 `--c_dtype Float32` 静默产出 ~50% 错误输出（无校验/无报错；`sm120_get_smem_store_op` 的 CopyUniversalOp 分支暗示曾意图支持非 16-bit）。建议上游要么修复 f32 epilogue，要么在 CLI/文档拒绝非 16-bit。素材：本节 §2 机制链 + §4 E1/E2 对照数据。
2. **tile 128×128×256 + sf_vec32 报错**（L944 SFA copy view congruence）：上游限制，128³ 已达 368 TFLOPS 不阻塞；若后续要冲更高可转上游 issue 或等官方修。
3. routeB 达标状态更新：**主 shape 368.1 TFLOPS ≥350 门禁达成**（条件：官方 testing.benchmark、warmup 5 + iter 10、L2 热；与 Task#12 162.2 同口径可比）。SASS 门禁（nvdisasm 验 mma e2m1）仍未做——正确性已过后建议补做以彻底闭环。
4. vLLM 生产态半零（用户资料 H1-H4）与本 B-N1 无关（独立基准即复现，非生产上下文特有）；但该资料的"恰好一半=未处理/未写"先验在本案修正为"**恰好一半=写入垃圾（视觉为零）**"——sentinel 判别法应纳入后续排障标准动作。

**—— Task #16 完，2026-08-20，阿奇（架构）**

### 6. SASS 硬门禁收口（Task #17 续派，2026-08-20，阿奇）

**结论：Go。** GEMM 主 kernel SASS 中 **128/128（100%）条 MMA 指令为原生 FP4 block-scaled MMA**（`OMMA.SF.16864.F32.E2M1.E2M1.E8`，即 m16n8k64 · F32 累加 · E2M1×E2M1 操作数 · UE8M0 scale factor），无任何 bf16 回退。

**提取方法**（Task#12 记录的 compiled_cache.db 路线在本镜像默认配置下不存在磁盘缓存，改走 DSL 官方产物开关）：
1. `CUTE_DSL_KEEP=ptx,cubin` + `CUTE_DSL_DUMP_DIR=/work/dump`（env_manager.py L379-391 的官方 artifact 保留开关，token：ir/ir-debug/ptx/cubin）→ 编译期直接落盘 `.sm_121a.cubin` + `.sm_121a.ptx`
2. 备选路径留档：monkeypatch `cute.compile` 拦截 `CudaDialectJitCompiledFunction`（`__cubin__` 属性 / `dump_to_object()` host 对象），KEEP 开关更直接故为终选
3. 反汇编：`cuobjdump -sass`（CUDA 13.3；需把 tokenspeed_triton 自带 nvdisasm 加入 PATH，cuobjdump 内部调用它）

**数据**（shape 256×256×512 编译产物，与主 shape 同 kernel 同 tile/dtype 配置，SASS 与问题规模无关）：

| 文件（sm_121a） | 大小 | SASS 行数 | mma 总数 | mma.*e2m1 | 判定 |
|---|---|---|---|---|---|
| `...dense_blockscaled_gemm_persistent_pingpong....cubin`（**GEMM 主 kernel**） | 41216 B | 2041 | **128** | **128** | 原生 FP4 ✓ |
| `cutlass__convert_...1_8.cubin`（辅助转换） | 11632 B | 489 | 0 | 0 | 无 MMA（预期） |
| `cutlass__convert_...3_4.cubin`（辅助转换） | 17376 B | 1017 | 0 | 0 | 无 MMA（预期） |

SASS 样本（`OMMA.SF.16864.F32.E2M1.E2M1.E8 R4, R160, R164, R4, R0, R2, URZ`；`.SF`=block-scale、`16864`=m16n8k64、`.E2M1.E2M1`=双 FP4 操作数、`.E8`=UE8M0 scale）；PTX 佐证 128 条 `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`；ELF header 确认 `code for sm_121a`。
注：SASS opcode 为大写（OMMA…E2M1），precheck 门禁命令 `grep -c 'mma.*e2m1'` 需加 `-i` 才能命中，实际执行用 `grep -icE 'mma.*e2m1|e2m1.*mma'`。

**工件留档**：本地 `_routeb_extract/sass_dump/`（主 kernel cubin 41KB + SASS 全文 295KB + PTX 85KB + sass_gate_final.txt 全日志）；服务器 `/tmp/routeb_task12/{dump,sass}/`。工具脚本 `_routeb_extract/routeb-delivery/sass_gate2.py` + `sass_gate_run4.sh`。

**routeB 验收链至此完整闭环**：正确性（fp16 全参考校验 PASS + sentinel 判别 100% ==K）→ 性能（4096×14336×4096 @ 128³ = **368.1 TFLOPS** ≥350 门禁）→ SASS 硬门禁（**Go**，原生 FP4 MMA 128/128）。P0/P1/P2 三层门槛全过。

**—— Task #17 完，2026-08-20，阿奇（架构）**
