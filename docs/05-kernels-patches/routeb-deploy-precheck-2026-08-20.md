# RouteB 部署前检查报告（CUTLASS 4.4.0 Python DSL — NVFP4 4W4A）

**日期**：2026-08-20
**工作流**：部署前检查（Go/No-Go）+ 系统设计审查 + 代码审查
**参与成员**：Archi（架构）/ Cody（代码审查）/ Rex（SRE）/ Tessa（测试）
**审查对象**：`_routeb_extract/routeb-delivery/` 交付包（7 文件）
**目标**：routeB ≥350 TFLOPS（SM121 dense NVFP4）替代 routeA（vLLM 原生 cutlass，80–180 TFLOPS）
**环境**：DGX Spark 4 节点 TP4 / DeepSeek V4 Flash / CUDA 13.2 / vLLM 0.26 / CUTLASS 4.4.0 / torch 2.11 / sm_121a / 社区实证基线 356 TFLOPS（baristankut）

---

## 📌 TL;DR

routeB（CUTLASS 4.4.0 Python DSL，MXF4 E2M1×E2M1 + UE8M0 32 分组）方向**正确、社区有 356 TFLOPS 实证**，但**当前交付包不可直接执行**：P2 核心基准脚本 `routeb_bench_blockscaled.py` 的 host launch 段是 `raise NotImplementedError`（第 188 行），整条复现链在此断裂；patch 脚本的 equality-check 变换会产出 Python `SyntaxError` 直接破坏 import；量化 `encode_e8m0_32` 漏 `+127` 偏移导致 scale 系统性偏差。三项 P2 硬阻塞 + 16 项代码缺陷需在进入性能验证前修复（预算 1–1.5 天）。**结论：🟡 有条件通过——修复阻塞项后可继续，维持 routeA 现役期间零生产风险**（routeB 是纯计算层优化，对 TP4 通信/编排/B1 固化配置零侵入，回滚三级完备）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（Conditional Go） |
| 架构结论 | 🟡 有条件推荐——方向与路径成立，但 3 项 P2 硬阻塞须先修 |
| 代码审查 | 🔴 Request Changes——严重×2 / 高×9 / 中×5 / 低×3 |
| 阻塞项数量 | 3 项 P2 硬阻塞（架构）+ 2 项 🔴 严重（代码，与架构重叠） |
| 关键行动项 | 20 条（🔴×4 / 🟠×8 / 🟡×6 / 🟢×2，去重合并） |
| 工期修正 | P2 实际 1–1.5 天（非 0.5–1 天）；总工期 4.5–5.5 天（非 3–4 天） |
| 生产侵入 | 零——routeB 是纯 GEMM 计算层优化，不动 TP4 通信/编排/B1 配置 |
| 回滚保障 | 三级完备（撤销插件 → revert patch → pip uninstall），routeA 现役始终完好 |
| 停机窗口风险 | 全部 SEV4（无生产流量），生产启动后升 SEV1–SEV3 |
| 建议下一步 | 修复 4 项 P0 阻塞 → 走 P0→P2 正确性→SASS→性能 验证链；未达标则维持 routeA |

---

## 正文

### 一、架构评估（Archi）

#### 1.1 ADR 结论

| ADR | 决策 | 状态 |
|-----|------|------|
| 路径选择 | R2' CUTLASS 4.4.0 Python DSL 为 routeB 唯一定案路径 | ✅ 成立（社区 356 TFLOPS 实证支撑） |
| 为何唯一 | Triton 任何版本无原生 FP4 MMA codegen（bf16 回退 2.76×）；FlashInfer 生产不可用；手写 CUTLASS 3.9 C++ 三连编译阻塞（B1/B2/B3） | ✅ 排除充分 |
| 集成方式 | Option(a) 独立 kernel 接插件（quant_config._nvfp4_prefill → routeB，M 阈值分派） | ✅ 推荐（可控、易回滚） |
| v17 Triton 定位 | bf16 FALLBACK 路径，**永不得入生产**（违反 4W4A 原则）；仅作数值真值基准与 A 量化 kernel 复用源 | ⚠️ 边界明确 |
| SASS 门禁 | `mma.*e2m1` / `mmaf`（SM12x），**勿用 `tcgen05`**（SM10x） | ✅ ADR-4 对齐 |

#### 1.2 三项 P2 硬阻塞（已逐行核实源码）

| # | 阻塞点 | 源码位置 | 证据 | 修复预算 |
|---|--------|---------|------|---------|
| B1 | **host launch = NotImplementedError** | `routeb_bench_blockscaled.py:188` | 第 182–192 行为"生产替换点"注释块，紧接 `raise NotImplementedError("host launch 段需按官方...补齐")`。TMA descriptor + grid launch 缺失 → P2 核心门禁**不可执行** | 移植官方 `dense_blockscaled_gemm_persistent_pingpong.py` ~150 行，1–1.5 天 |
| B2 | **SMEM 估算误计累加器** | `routeb_bench_blockscaled.py:60-66` | 第 65 行 `acc_bytes = tile_m * tile_n * 4` 把 F32 累加器计入 SMEM；实际 F32 acc 驻留寄存器堆（RF）而非 SMEM。导致 256×128×128 被误判 204.5KB（实际 76.5KB），**脚本系统性跳过所有最优 tile** | 删除 `acc_bytes` 项 |
| B3 | **MmaMXF4Op dtype 签名未核对** | `routeb_bench_blockscaled.py:126-131` | 第 126–128 行 `a_dtype=torch.float8_e4m3fn`（**FP8 而非 FP4**），第 128 行 TODO 注释自承"以安装版 help 为准；若用元素类型枚举（cutlass.Float4E2M1FN）则替换"。FP4 路径却喂 FP8 dtype，**指令可能不发射原生 e2m1 MMA** | 改用 `cutlass.Float4E2M1FN`（按 4.4.x 实际签名核对） |

#### 1.3 其他架构结论

- **MXF4 N 向分组(128) 与纯 MXFP4 标准(1×32) 潜在不匹配**：routeB 的 MXF4 变体按 N 向 128 分组组织 scale，而纯 MXFP4 标准是 1×32 分组。需 P3 实测验证是否影响数值正确性。
- **工期修正**：交付包 README 声称"3–4 天"，但 P2 单独因 host launch 移植实际需 1–1.5 天，**总工期修正为 4.5–5.5 天**。
- **356 单源不可复现性**：社区 356 TFLOPS 基线仅 baristankut 单一来源（论坛 #359960），无第二独立复现——复现风险中等。

#### 1.4 架构风险矩阵

| 阶段 | 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|------|
| P0 | 缺 CUDA runtime libs（pip --no-deps 只装 libs-base） | 中 | 高 | 单独 `pip install nvidia-cuda-runtime-cu13`；验 LD_LIBRARY_PATH |
| P1 | regex 未匹配 admissible_archs/equality 写法 | 中 | 中 | patch 后跑官方示例 + `import` 验证 |
| P2 | host launch NotImplementedError（**确定致命**） | 确定 | 致命 | 移植官方 persistent_pingpong ~150 行 |
| P2 | SMEM 公式误计 acc（**确定高影响**） | 确定 | 高 | 删除 acc_bytes 项 |
| P2 | MmaMXF4Op dtype 用 FP8 | 高 | 高 | 改 Float4E2M1FN |
| P2 | 356 单源不可复现 | 中 | 中 | 失败则 tile sweep 扩展 + num_warps 调优 |
| P3 | MXF4 N 向不匹配 | 中 | 高 | P3 实测；备选 NVF4 16 分组 + 高精度转换器 |
| P4 | kernel 未达生产级 | 中高 | 中 | 任何阶段未达标维持 routeA 现役 |

#### 1.5 替代方案权衡

| 集成方案 | 可控性 | 回滚 | 推荐 |
|---------|--------|------|------|
| Option(a) 独立 kernel 接插件（M 阈值分派：prefill→routeB，decode→B12X 原路径） | 高 | 删插件即回退 | ✅ 首选 |
| Option(b) flashinfer-b12x backend（四层配方：patch + env CUTE_DSL_ARCH=sm_121a） | 中 | env 回退 | 备选 |

---

### 二、代码审查发现（Cody）

**审查结论：🔴 Request Changes**——严重度分布 🔴严重×2 / 🟠高×9 / 🟡中×5 / 🟢低×3，共 19 项。

> 已逐行核实源码，所有行号与交付包一致。

#### 2.1 19 项发现按严重度排序

| # | 级别 | 文件:行 | 问题 | 修复建议 |
|---|------|---------|------|---------|
| 1 | 🔴 | bench:188 | host launch `raise NotImplementedError`，P2 核心门禁不可执行 | 移植官方 persistent_pingpong ~150 行 |
| 2 | 🔴 | patch:79-83 | equality-check 变换产出 `if not arch == Arch.sm_120a, Arch.sm_121a:`——Python3 `SyntaxError`，直接破坏 import | 改为 `if arch not in (Arch.sm_120a, Arch.sm_121a):` 形式 |
| 3 | 🟠 | bench:213 | `W_packed = pack_e2m1_n(W.t()).t()` 布局错误——产出 `[K//2, N]` 而非注释所称 `[K, N//2]`（pack 沿末维 K 分组，转置后维度错位） | 按 MXF4 N 向打包语义重写，直出 `[K, N//2]` |
| 4 | 🟠 | bench:81-86 | `encode_e8m0_32` 漏 `+127`——`exp = floor(log2(amax/6))` 未加 127 偏移，但 `torch_reference:94` 解码用 `pow(2, scale-127)`，**scale 系统性偏差 127 幂次** | encode 末加 `+ 127`（对照 v17:51 已含 `+127.0`） |
| 5 | 🟠 | bench:214 | `W_scale = encode_e8m0_32(W).reshape(...).amax(-1)` 计算错误——encode 按 N 向分组而非 K 向（contraction dim），且 reshape+amax 非 32×128 block-max | 重写为 32(K)×128(N) block-max，对齐生产格式 |
| 6 | 🟠 | bench:60-66 | `smem_estimate` 误计 `acc_bytes = tile_m*tile_n*4`——F32 累加器驻留 RF 而非 SMEM，系统性跳过所有最优 tile | 删除 acc_bytes 项 |
| 7 | 🟠 | bench:126-131 | `MmaMXF4Op` 用 `float8_e4m3fn`（FP8）应改 `Float4E2M1FN`（FP4） | 按安装版签名核对替换 |
| 8 | 🟠 | v17:351 | `_a_quant_kernel` grid 硬编码 `BLOCK_M=128`，但 autotune 可选 32/64/128——非 128 配置时 ~50% 行未量化 | grid 按 autotune 实际 BLOCK_M 计算 |
| 9 | 🟠 | setup:10-12 | driver 版本仅 `echo` 不比较——无 fail-fast，低于 580.142 仍继续 | 加 `if` 比较 + `exit 1` |
| 10 | 🟠 | setup:40 | 备份 `cp` 无存在性检查——patch 后重跑覆盖干净备份 | 先 `test -f .bak` 跳过 |
| 11 | 🟠 | patch:63 | 字符串拼接 `old.replace("sm_120a", "sm_120a\", \"sm_121a")` 假定双引号——遇枚举/单引号即 SyntaxError | 用 AST 或正则重写为幂等安全替换 |
| 12 | 🟡 | bench:267-275 | `--check` 模式仅 `print` 不校验——仅打印形状，无数值断言，制造虚假正确性信心 | `--check` 加 `assert` 数值比对 |
| 13 | 🟡 | patch:92 | 非原子写——`open(path,"w").write()` 中断则 mma.py 损坏 | 写临时文件 + `os.replace` 原子替换 |
| 14 | 🟡 | bench:225 | `assert err < 1e-2` 绝对误差过松——输出 O(10²) 时 1e-2 ≈ 无校验 | 改相对误差 `rel_err = max_err/ref.abs().max() < 1e-2` |
| 15 | 🟡 | setup:16 | `--no-deps` 依赖容器预装 CUDA 13——脆弱假设，缺 runtime libs 则 import 失败 | 显式装 `nvidia-cuda-runtime-cu13` |
| 16 | 🟡 | v17:90-99 | autotune 在非 SM121 验证——configs 含 BLOCK_M=32/64/128，需在目标硬件复测 | sm_121a 实测复验 |
| 17 | 🟢 | patch:97-103 | `revert` 无备份时不 `exit` 非零——静默失败 | 无备份时 `sys.exit(1)` |
| 18 | 🟢 | bench:229-236 | 每 iter 重构 kernel——`make_blockscaled_kernel` 在计时循环内调用，污染计时 | 循环外构造一次 |
| 19 | 🟢 | setup:40 | `cp` 未加 `-p`——元数据（时间戳/权限）不一致 | `cp -p` |

#### 2.2 关键正面发现

- **v17 Triton 参考内核（`nvfp4_4w4a_prefill_gemm_v17_triton.py`）是整个交付包唯一数值正确且自洽的工件**：
  - E8M0 编码含 `+127`（第 51 行 `tl.floor(log2_val) + 127.0`）✓
  - E2M1 幅值查表正确（E2M1_MAG 表 + 阈值比较）✓
  - 指针 int64 + mask 齐备 ✓
  - 8 轮 MCP 验证通过，speedup 2.76×（bf16 路径）
  - **用途定位**：① routeB CUTLASS 的数值对标基（同输入同输出）② A 量化 kernel（`_a_quant_kernel`）复用源——但须先修 #8 grid 硬编码

---

### 三、部署检查清单（Rex）

#### 3.1 P0–P4 Go/No-Go 判据矩阵

| 阶段 | 执行范围 | Go 判据 | No-Go 动作 |
|------|---------|---------|-----------|
| **P0 环境** | 全 4 节点 | CUDA ≥13.0 + driver ≥580.142 + pip 隔离（`--no-deps`）+ `import cutlass 4.4.x` 成功 + mma.py 已备份 | 不装；修 runtime libs + LD_LIBRARY_PATH |
| **P1 patch** | 全 4 节点 | `admissible_archs` 含 `sm_121a` + 官方示例不报错 + diff 仅 2 处 + `--revert` 可用 | revert；人工核对 regex |
| **P2 基准** | 单节点 | ≥350 TFLOPS（基线 356）+ SASS `mma.*e2m1` 命中 + `max_err < 1e-2`（相对）+ SMEM ≤99KB | tile sweep 扩展；维持 routeA |
| **P3 语义** | 单节点 | pytest 8/8 + 生产权重直配 + `sf_vec=32` + A 量化复用 + 4 参接口 `(A, W_packed, W_scale, bias)` | 备选 NVF4 16 分组 + 转换器 |
| **P4 集成** | 全集群 | ≥1.5× routeA + ≥350 TFLOPS + pytest 8/8 + needle 128K 全绿 + 4 节点一致 | 维持 routeA 现役（零改动） |

**执行范围说明**：routeB 是纯计算层 GEMM 优化，对 TP4 通信拓扑/编排/B1 固化配置**零侵入**。P0/P1 必须全 4 节点执行（环境一致）；P2/P3 单节点即可（kernel 验证）；P4 需全集群（端到端 A/B）。

#### 3.2 回滚方案（三级完备，routeA 现役始终完好）

| 级别 | 动作 | 代价 | 触发条件 |
|------|------|------|---------|
| L1 撤销插件 | 删除 `quant_config._nvfp4_prefill` 引用 → 落回 routeA/B12X 原路径 | 0（不改 vLLM 本体） | 任何 P4 指标未达 |
| L2 revert patch | `python patch_cutlass_dsl_sm121a.py --revert`（恢复 .bak） | ~10s | P2/P3 数值错误或 crash |
| L3 pip uninstall | `pip uninstall nvidia-cutlass-dsl-libs-cu13` | ~30s | DSL 库本身污染依赖 |

> routeA（vLLM 原生 `cutlass_scaled_fp4_mm`，80–180 TFLOPS）在整条 routeB 验证链期间**始终现役**，任何阶段未达标均可零风险回退。

#### 3.3 回滚触发条件（生产启动后监控）

| 指标 | 阈值 | 持续 | 动作 |
|------|------|------|------|
| prefill P99 延迟 | >1.5× routeA 同 shape | 5 min | L1 撤销插件 |
| 数值偏差 | max_err >1e-2 | 即时 | L2 revert + 排查 |
| OOM | >0 次 | 即时 | L2 revert |
| needle-in-haystack | <95%（128K context） | 即时 | L2 revert + 降级长上下文 |

#### 3.4 SEV 风险评级

| 阶段 | 最高 SEV | 说明 |
|------|---------|------|
| 停机窗口（当前） | **SEV4** | 无生产流量，所有操作低危；生产未启动 |
| 生产启动后 | **SEV1** | routeB 数值错误/crash → 产出错误结果/服务中断 |
| 生产启动后 | **SEV2** | 性能回退（prefill P99 退化） |
| 生产启动后 | **SEV3** | 长上下文精度下降（needle <95%） |

#### 3.5 生产配置影响评估

- **零侵入确认**：routeB 不修改 NCCL/TP4 通信、不修改调度编排、不修改 B1 固化配置（隔离核 `isolcpus`、内存头寸、NCCL/shim MD5 四机一致等均不动）。
- **SASS 门禁是硬门槛**：`nvdisasm kernel | grep mma.*e2m1` 不出现即判定 bf16 回退，**直接 No-Go**——不可绕过。
- **MXF4 32 分组直配**：与生产 `W_scale [K//32, N//128]` 格式直配，主机侧零重排——这是 routeB 相对 NVF4 16 分组的关键优势（但需 P3 实测 N 向分组匹配性）。

---

### 四、测试策略（Tessa）

#### 4.1 56 项测试用例矩阵

| 测试组 | 用例数 | 范围 | 验证目标 |
|--------|--------|------|---------|
| P2 前置修复验证 | 24 | encode+127 / SMEM 公式 / W_packed 布局 / W_scale / MmaMXF4Op dtype / patch 语法 / grid | 阻塞项修复后回归 |
| P2 基准 | 11 | 正确性(torch 参考) → SASS(e2m1) → 性能(≥350 TFLOPS) | 复现 356 基线 |
| P3 语义 | 11 | 权重直配 / sf_vec=32 / A 量化复用 / 4 参接口 / MXF4 N 向匹配 | 生产语义对接 |
| P4 集成 | 10 | A/B 对照 / needle 128K / 4 节点一致 / 回滚演练 | 端到端投产 |

#### 4.2 优先级排序（修复 + 验证）

| 优先级 | 内容 | 阻塞链 |
|--------|------|--------|
| 🔴 P0 阻塞 | host launch 移植 + encode `+127` + SASS e2m1 门禁 | 阻塞整条测试链 |
| 🟠 P1 高 | patch SyntaxError + SMEM 公式 + W_packed 布局 + grid 硬编码 | 阻塞 P2 基准 |
| 🟡 P2 中 | dtype + W_scale + 误差度量(绝对→相对) + MXF4 N 向 | 阻塞 P3 语义 |
| 🟢 P3 低 | MoE 门槛 + 回滚演练 + 元数据一致性 | 收尾 |

#### 4.3 根阻塞点

**host launch ~150 行待移植**是整条测试链的根阻塞——`routeb_bench_blockscaled.py:188` 的 `NotImplementedError` 使 P2 基准（正确性/SASS/性能）全部无法执行。移植官方 `dense_blockscaled_gemm_persistent_pingpong.py` 的 TMA descriptor + grid launch 是解锁后续所有验证的前置。

#### 4.4 误差度量修正建议

- **现状**：`bench:225` `assert err < 1e-2`（绝对误差）——输出 O(10²) 时 1e-2 相对 ≈1e-4，近乎无校验。
- **修正**：改相对误差 `rel_err = max_err / ref.abs().max() < 1e-2`，与生产语义对齐（routeA 对照 rel<0.02）。

#### 4.5 v17 作为数值真值基准

- v17 的 `encode_e8m0_32` 含 `+127`（第 51 行）、E2M1 幅值查表正确、指针 int64 + mask 齐备——**已验证 8 轮**。
- 用途：routeB CUTLASS kernel 的数值对标基（同输入同输出比对）；A 量化 kernel `_a_quant_kernel` 复用源（须先修 #8 grid 硬编码）。

#### 4.6 测试执行顺序

```
P0 环境（CUDA/driver/import/备份）
  → P2 前置修复验证（24 项：encode/SMEM/W_packed/dtype/patch/grid）
    → P2 基准（正确性 → SASS e2m1 → 性能 ≥350）
      → P3 语义（权重直配 / sf_vec=32 / A 量化复用 / 4 参接口）
        → P4 集成 A/B（≥1.5× routeA + needle 128K + 4 节点一致）
```

---

### 五、综合风险矩阵（跨成员去重合并）

| # | 风险项 | 来源 | 概率 | 影响 | 阶段 | 缓解对策 | 状态 |
|---|--------|------|------|------|------|---------|------|
| R1 | host launch NotImplementedError | 架构+代码+SRE+测试 | **确定** | 致命 | P2 | 移植官方 persistent_pingpong ~150 行 | 🔴 待修 |
| R2 | patch equality-check SyntaxError | 代码 | 高 | 致命 | P1 | 改 `arch not in (...)` 形式 | 🔴 待修 |
| R3 | encode_e8m0_32 漏 +127 | 代码+测试 | **确定** | 高 | P2 | encode 末加 `+127`（对照 v17:51） | 🔴 待修 |
| R4 | SASS 无 e2m1（bf16 回退） | SRE+架构 | 中 | 致命 | P2 | `nvdisasm | grep mma.*e2m1` 硬门禁 | 🔴 门禁在位 |
| R5 | SMEM 误计 acc_bytes | 架构+代码 | **确定** | 高 | P2 | 删除 acc_bytes 项 | 🟠 待修 |
| R6 | W_packed 布局错误 | 代码 | 确定 | 高 | P2 | 按 MXF4 N 向重写 | 🟠 待修 |
| R7 | MmaMXF4Op 用 FP8 dtype | 架构+代码 | 高 | 高 | P2 | 改 Float4E2M1FN | 🟠 待修 |
| R8 | W_scale 计算非 block-max | 代码 | 确定 | 高 | P2 | 重写 32×128 block-max | 🟠 待修 |
| R9 | v17 grid 硬编码 BLOCK_M | 代码 | 高 | 中 | P3 | 按 autotune 实际计算 | 🟠 待修 |
| R10 | 缺 CUDA runtime libs | 架构 | 中 | 高 | P0 | 显式装 nvidia-cuda-runtime-cu13 | 🟡 待验 |
| R11 | patch regex 未匹配 | 架构+代码 | 中 | 中 | P1 | patch 后 import 验证 | 🟡 待验 |
| R12 | 356 单源不可复现 | 架构 | 中 | 中 | P2 | tile sweep 扩展 + num_warps | 🟡 待验 |
| R13 | MXF4 N 向(128) vs 标准(1×32) | 架构+测试 | 中 | 高 | P3 | P3 实测；备选 NVF4 16 分组 | 🟡 待测 |
| R14 | 绝对误差 1e-2 过松 | 代码+测试 | 确定 | 中 | P2 | 改相对误差 | 🟡 待修 |
| R15 | --check 仅 print 无校验 | 代码 | 确定 | 中 | P2 | 加 assert 数值比对 | 🟡 待修 |
| R16 | patch 非原子写 | 代码 | 低 | 中 | P1 | 写临时文件 + os.replace | 🟡 待修 |
| R17 | --no-deps 依赖假设 | 代码 | 中 | 低 | P0 | 显式装 runtime libs | 🟢 待办 |
| R18 | 每 iter 重构 kernel 污染计时 | 代码 | 确定 | 低 | P2 | 循环外构造一次 | 🟢 待修 |
| R19 | driver 无 fail-fast / cp 无 -p | 代码 | 中 | 低 | P0 | 加比较 exit / cp -p | 🟢 待修 |
| R20 | v17 autotune 非 SM121 验证 | 代码 | 中 | 低 | P3 | sm_121a 复测 | 🟢 待测 |

---

## ✅ 行动清单（按优先级排序）

> 编号 R* 对应综合风险矩阵。去重合并架构/代码/SRE/测试四份产出。

### 🔴 P0 阻塞（必须修复，否则整条链不可执行）

| # | 行动项 | 文件:行 | 负责人 | 预算 |
|---|--------|---------|--------|------|
| A1 | 移植官方 `dense_blockscaled_gemm_persistent_pingpong.py` 的 host launch（TMA descriptor + grid launch），替换 `routeb_bench_blockscaled.py:188` 的 `NotImplementedError` | bench:182-192 | 开发 | 1–1.5 天 |
| A2 | 修复 patch equality-check 变换：`patch:79-83` 改为 `if arch not in (Arch.sm_120a, Arch.sm_121a):` 形式，消除 SyntaxError | patch:79-83 | 开发 | 30 min |
| A3 | 修复 `encode_e8m0_32`：`bench:85` 末加 `+ 127`（对照 v17:51 `tl.floor(log2_val) + 127.0`） | bench:81-86 | 开发 | 15 min |
| A4 | 确认 SASS 门禁脚本就位：`nvdisasm kernel \| grep mma.*e2m1` 命中即 Go，未命中即 No-Go（硬门槛，不可绕过） | — | SRE | 已在位 |

### 🟠 P1 高（阻塞 P2 基准，停机窗口内优先）

| # | 行动项 | 文件:行 | 预算 |
|---|--------|---------|------|
| A5 | 删除 `smem_estimate` 的 `acc_bytes = tile_m*tile_n*4` 项（F32 acc 驻留 RF 非 SMEM） | bench:60-66 | 10 min |
| A6 | 重写 `W_packed` 布局：`pack_e2m1_n(W.t()).t()` → 直出 `[K, N//2]`（MXF4 N 向打包） | bench:213 | 1 h |
| A7 | 改 `MmaMXF4Op` dtype：`float8_e4m3fn` → `cutlass.Float4E2M1FN`（按 4.4.x 签名核对） | bench:126-131 | 30 min |
| A8 | 重写 `W_scale`：按 32(K)×128(N) block-max，对齐生产 `W_scale [K//32, N//128]` | bench:214 | 1 h |
| A9 | 修复 v17 `_a_quant_kernel` grid：按 autotune 实际 BLOCK_M（32/64/128）计算，非硬编码 128 | v17:351 | 1 h |
| A10 | setup driver 比较加 fail-fast：`bench:10-12` 加 `if` + `exit 1` | setup:10-12 | 15 min |
| A11 | setup 备份加存在性检查：先 `test -f .bak` 跳过，防覆盖干净备份 | setup:40 | 10 min |
| A12 | patch 字符串拼接改 AST/正则幂等替换，消除双引号假定 | patch:63 | 1 h |

### 🟡 P2 中（阻塞 P3 语义，收尾阶段处理）

| # | 行动项 | 文件:行 | 预算 |
|---|--------|---------|------|
| A13 | `--check` 模式加数值 `assert`（非仅 print 形状） | bench:267-275 | 30 min |
| A14 | patch 写改原子：临时文件 + `os.replace` | patch:92 | 20 min |
| A15 | 误差度量改相对：`rel_err = max_err / ref.abs().max() < 1e-2` | bench:225 | 15 min |
| A16 | setup 显式装 `nvidia-cuda-runtime-cu13`（去脆弱 `--no-deps` 假设） | setup:16 | 15 min |
| A17 | v17 autotune configs 在 sm_121a 复测（32/64/128 均验证） | v17:90-99 | 1 h |
| A18 | P3 实测 MXF4 N 向分组(128) vs 纯 MXFP4(1×32) 匹配性；不匹配则备选 NVF4 16 分组 + 高精度转换器 `--block-k 16` | — | 0.5–1 天 |

### 🟢 P3 低（收尾，不影响门禁）

| # | 行动项 | 文件:行 | 预算 |
|---|--------|---------|------|
| A19 | patch revert 无备份时 `sys.exit(1)`（非静默失败） | patch:97-103 | 5 min |
| A20 | bench 计时循环外构造 kernel 一次（防每 iter 重构污染计时） | bench:229-236 | 15 min |
| A21 | setup `cp` 加 `-p`（保元数据一致） | setup:40 | 2 min |

---

## ⚠️ 待完善 / 已知局限

1. **356 TFLOPS 单源依赖**：社区基线仅 baristankut 一人实证（论坛 #359960，CUTLASS 4.4.0 + CUDA 13.1），无第二独立复现。若 P2 复现失败，风险敞口无对冲——需 tile sweep 扩展（128×256 / 256×256 在 99KB 预算内）+ num_warps 调优兜底。
2. **MmaMXF4Op 签名待现场核对**：交付包自承"以安装版 `help(MmaMXF4Op)` 为准"（bench 第 128 行 TODO），4.4.0 与 4.6.0 Operator API 有差异——需在目标机器 `import` 后核对真实签名再定 dtype 枚举。
3. **MXF4 N 向分组匹配性未验证**：routeB MXF4 变体按 N 向 128 分组组织 scale，与纯 MXFP4 标准 1×32 分组语义存在潜在不匹配——须 P3 实测，最坏需切换 NVF4 16 分组路径（scale 重算，非简单重复）。
4. **v17 Triton 永不得入生产**：v17 是 bf16 FALLBACK（内核明确标注 `FALLBACK PATH - bf16 dequant`），违反 4W4A 红线。其唯一合法用途是数值对标基 + A 量化 kernel 复用源，**不可作为生产 GEMM 路径**。
5. **工期估算需对齐**：交付包 README 声称 3–4 天，架构审查修正为 4.5–5.5 天（P2 host launch 移植占 1–1.5 天）。停机窗口规划须按修正后工期排期。
6. **测试门禁自动化缺口**：P0 持久化 import、A–F 投产门禁、性能门槛断言目前多为手工命令/仅 print 无 PASS-FAIL——需补 `run_prod_gate.sh` 一键门禁 + 退出码（Tessa G1/G3）。
7. **本报告基于静态审查**：所有代码行号已逐行核实，但未在目标硬件（sm_121a + CUDA 13.2 容器）实跑；runtime 行为（import 成功、SASS 发射、tile 实测性能）须以现场执行为准。

---

## 📚 数据来源 & 成员产出索引

| 来源 | 角色 | 产出 | 依据文件 |
|------|------|------|---------|
| Archi | 架构师 | ADR + 三项 P2 硬阻塞 + 风险矩阵 + 替代方案权衡 | `architecture-nvfp4-2026-08-20.md`；交付包 `kernel1_routeB_improvement.md` / `routeB_execution_plan.md` |
| Cody | 代码审查 | 19 项发现（🔴×2/🟠×9/🟡×5/🟢×3）+ 正面发现 | `code-review-cluster-2026-08-20.md`；逐行核实 `_routeb_extract/routeb-delivery/` 全 7 文件 |
| Rex | SRE | P0–P4 Go/No-Go 矩阵 + 三级回滚 + SEV 评级 + 生产影响 | `sre-ops-reliability-2026-08-20.md` |
| Tessa | 测试 | 56 项测试矩阵 + 根阻塞点 + 误差度量修正 + 执行顺序 | `testing-strategy-2026-08-20.md` |

**交付包文件清单（`_routeb_extract/routeb-delivery/`）**：

| 文件 | 角色 | 关键问题 |
|------|------|---------|
| `README.md` | 执行顺序索引 | 工期 3–4 天（修正为 4.5–5.5） |
| `routeB_execution_plan.md` | 主执行计划 | P0–P4 + 风险矩阵 + 参考资源 |
| `kernel1_routeB_improvement.md` | 技术规格定案 v3 | 路线收敛依据 + MCP 验证结论 |
| `setup_routeb_env.sh` | P0 环境 | driver 无 fail-fast / 备份无存在性检查 / `--no-deps` / cp 无 -p |
| `patch_cutlass_dsl_sm121a.py` | P1 patch | equality-check SyntaxError / 字符串拼接双引号假定 / 非原子写 / revert 无 exit |
| `routeb_bench_blockscaled.py` | P2 基准 | host launch NotImplementedError / SMEM 误计 / encode 漏 +127 / W_packed 布局 / W_scale / dtype FP8 / 误差过松 / --check 无校验 / 每 iter 重构 |
| `nvfp4_4w4a_prefill_gemm_v17_triton.py` | P3 语义对标 | **唯一数值正确工件**；grid 硬编码 BLOCK_M=128 需修；bf16 FALLBACK 不得入生产 |

---

> ⚠️ **免责声明**：本报告由工程保障团队 AI 协作生成（架构/代码/SRE/测试四角色 + 文档汇编），基于对交付包源码的静态只读审查，未在目标硬件实跑。所有代码行号已逐行核实，但 runtime 行为（import 成功性、SASS 指令发射、tile 实测 TFLOPS、MXF4 分组匹配性）须以现场执行为准。**关键决策（356 TFLOPS 是否硬门槛、host launch 移植是否启动、v17 是否复用）请由人类工程负责人复核裁定。** 维持 routeA 现役期间所有 routeB 验证操作均为 SEV4（无生产流量），回滚三级完备，零生产风险。
