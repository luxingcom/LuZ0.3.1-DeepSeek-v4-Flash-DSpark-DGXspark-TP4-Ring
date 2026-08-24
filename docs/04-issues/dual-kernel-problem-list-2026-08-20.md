# 双算子问题总清单（kernel1 routeB 替换 + kernel2 v17 交付包）

**日期**：2026-08-20（22:40 汇编）
**汇编**：主理人甄宇航（基于五份成员产出交叉去重）
**来源**：routeb-deploy-precheck（预检）/ architecture-dual-kernel（架构）/ code-review-kernel2（代码审查）/ sre-dual-kernel-deploy（SRE 现场核查）/ routeb-fix-log（修复执行）

---

## 📌 TL;DR

- **kernel1（routeB 替换 routeA）**：预检 19+3 项代码问题已**全部修复**（17+2 项）；路径级阻塞 **B-N1 已根因修复**（根因=bench 编排器传 c_dtype=f32，官方 epilogue C-atom 仅支持 16-bit 输出 → 半输出为写入的垃圾值"视觉为零"；修复=c_dtype 改 fp16 + 护栏）；**主 shape 复测 368.1 TFLOPS（参考校验通过）——超过 350 门禁与 356 社区基线，routeB 目标达成**。B-N2（356 来源疑云）随 DSL 路径达标失去意义（SGLang 取证已放弃）。剩余：SASS 门禁收口（进行中）+ P3 语义对接 + P4 集成 A/B。
- **kernel2（v17 交付包）**：**v17 内核本体 Approve**（与生产 md5 一致、零缺陷）；交付包 13 项发现**已全部修复闭环并真机复核**（7/7+8/8+7/7+8/8 全过）；paged 变体解除禁部署（BLOCK_SIZE 参数化 + 语义统一到 v17 金标准 + 64 槽生产块大小真机 8/8）；benchmark 修正口径：HBM 真实带宽 T=65536=**211.1 GB/s（理论 77%，3.9× v11）**。
- **生产影响**：零——两算子均为"可 import 零调用点"挂载，生产容器停机中。

---

## 一、kernel1（routeB）问题清单

### 1.1 已修复（Task #12/#13，全部验证通过 ✅）

| # | 原编号 | 问题 | 修复方式 | 验证 |
|---|--------|------|---------|------|
| 1 | A1/B1 | bench host launch = NotImplementedError | **架构级偏离**：交付包实缺整个设备侧 kernel（~1000 行），改为 vendor 官方 SM120 完整实现（blackwell_geforce 三件套），bench 重写为编排器 | 官方 kernel GB10 编译+执行成功 |
| 2 | A3 | encode_e8m0_32 漏 +127 | 末加 +127 | 量化自检 5/5（零→24、1e6→144） |
| 3 | A5/B2 | SMEM 误计 acc_bytes | 删除该项 | tile 筛选恢复正确 |
| 4 | A6 | W_packed 布局错误 | 重写直出 [K, N//2] | 自检过 |
| 5 | A7/B3 | MmaMXF4Op 用 FP8 dtype | 按 4.5.2 实测签名定案（三位置参数） | 容器实测 |
| 6 | A8 | W_scale 非 block-max | 重写 32(K)×128(N) | 自检过 |
| 7 | A2 | patch equality-check SyntaxError | 改产 `not in (...)` 形式 | 真实 mma.py 副本实测 |
| 8 | A12 | patch 字符串拼接脆弱 | 引号风格自适应正则（幂等） | 9 种写法夹具全过 |
| 9 | A13 | patch 非原子写 | tmp + os.replace | 验证过 |
| 10 | A19 | revert 无备份静默 | sys.exit(1) | 验证过 |
| 11 | A10 | setup driver 无 fail-fast | sort -V 比较 + exit 1 | bash -n 过 |
| 12 | A11 | 备份无存在性检查 | test -f 跳过 | 验证过 |
| 13 | A16 | --no-deps 脆弱假设 | 显式装 runtime libs | 脚本更新 |
| 14 | A14 | 绝对误差过松 | 改相对误差 | 已改 |
| 15 | A15 | --check 仅 print | 加数值 assert | 5/5 自检 |
| 16 | A20 | 计时循环内重构 kernel | 移到循环外 | 已改 |
| 17 | A21 | cp 无 -p | 加 -p | 已改 |
| 18 | 附带 | torch_reference 形状非法（预检未列） | 修复 | 真断言暴露后修 |
| 19 | 附带 | W 侧量化从未发生（预检未列） | 修复 | 真断言暴露后修 |

### 1.2 路径级新阻塞（已解决 ✅）

> **2026-08-20 深夜更新：B-N1 已根因修复，routeB 达标。** 根因不在官方 kernel，在 bench 编排器传 `c_dtype=Float32`：官方 SM120 示例 epilogue 的 C-atom（StMatrix8x8x16bOp）仅对 16-bit 输出成立——f32 使 tiled-copy 每线程值数 4→2、M tiler 粒度 32→16 行，与累加器 retile（4 值×32 行）错配 → epilogue 每片只拷 2/4 值 → 一半累加器丢弃+一半寄存器槽未初始化 → ~50% 输出为垃圾值（~1e-38 未初始化位型，打印为 0.0000——"初始零"是视觉假象，sentinel 判别法实证）。判决实验：同 shape 仅改 c_dtype，fp16→100% 逐位精确。修复：c_dtype 默认 fp16 + `--c-dtype` 参数 + vendored kernel 非宽 16 位 ValueError 护栏。**复测 4096×14336×4096 @ 128³ fp16 参考校验通过：368.1 TFLOPS > 350 门禁 > 356 社区基线。**

| # | 原阻塞 | 状态 |
|---|--------|------|
| B-N1 | 官方 DSL 示例 kernel 半零输出 | ✅ **已修复**（根因 c_dtype f32；fp16 复测 100% 逐位精确 + 368.1 TFLOPS；BFloat16/NVFP4 vec16/cooperative/epi 64,32 全部复测 PASS） |
| B-N2 | 356 单源不可复现 | ✅ **失去意义**（DSL 路径实测 368.1 > 356；SGLang 取证已按用户指示放弃） |

遗留（非阻塞）：tile 128×128×256 + sf_vec32 报错 = 上游示例 K=256+vec32 限制（128³ 已达标）；上游 issue 素材已备（f32 静默半错应显式拒绝）——**对外提交待用户批准**；SASS 门禁（nvdisasm 验 mma e2m1）收口进行中。

### 1.3 环境事实（SRE 核查）

| 项 | 事实 | 含义 |
|---|------|------|
| CUTLASS DSL | 生产镜像已装 4.5.2（dsl/libs-base/libs-cu13），混装 libs-core/libs-cu12=4.6.0 | **P1 patch 阶段整体取消**（4.5.2 mma.py 原生含 sm_121a，admissible_archs=[sm_120a,sm_120f,sm_121a,sm_121f]）——工期简化 |
| Driver | 580.173.02 四节点一致 | ≥580.142 ✓ |
| 工期 | 架构师修正 4.5-6 天（P1 免除后可再下修） | 但 B-N1 未解前 DSL 路径整体不可推进 |

---

## 二、kernel2（v17 交付包）问题清单

### 2.1 结论

- **v17 内核本体：Approve**——与生产 md5（a795b2b4）一致，逐行审查零缺陷（列界推导成立/store 全掩码/int64 寻址/T=0 早退/确定性），生产留任不受影响
- **交付包：Request Changes**——13 项发现；README §四自验指令真机将 3/7 失败，"安全可靠全过"结论不成立

### 2.2 发现明细（含修复状态）

> **✅ 2026-08-20 23:5x 更新：13 项全部修复闭环（Cody 两轮接力 + 真机独立复核）**——test_v17_safety 7/7、test_v17 8/8、test_linear 7/7、test_paged 8/8（含 64 槽生产块大小用例）全过；paged 解除禁部署；v17 内核 md5 修复前后均与生产锚点 a795b2b4 一致（红线遵守）；benchmark 修正口径 HBM T=65536=211.1 GB/s。修复日志：routeb-fix-log-2026-08-20.md §kernel2 修复。下表"修复状态"列为发现时的初始状态记录。

| # | 严重度 | 位置 | 问题 | 修复状态 |
|---|--------|------|------|---------|
| 1 | High | test_v17_safety.py:44 | saturation 断言期望 255（实为 144；255 在 fp32 数学不可达，上限 252） | 🔄 修复中（kernel2-fixer） |
| 2 | High | test_v17_safety.py:52 | sign_zero 断言期望 1（实为 24；错误注释"127-126=1"） | 🔄 修复中 |
| 3 | High | test_v17_safety.py:59-60 | boundary_T 两次独立未 seed randn 比较（必失败） | 🔄 修复中 |
| 4 | **Critical（落锤）** | paged_triton.py:158 | BLOCK_SIZE=256 硬编码，生产=64 → **每个 token 写错块+错槽，静默 KV 损坏**；测试自建 256 槽缓存永远测不出 | 🔄 修复中（参数化 + 断言） |
| 5 | Medium | safety md:15,79 | 安全报告数值断言错误（"字节 1"/"255"应为 24/144）；§九"全过"建立于错误期望 | 🔄 修复中（文档勘误） |
| 6 | Medium | paged vs v17 多处 | 语义漂移：safe_max 1e-38 vs 1e-30、clamp [-127,128] vs [-126,127]、paged 独有除数防护（编码自洽缺陷：clamp 时解码 0.588× 偏差）；两族不可交叉校验 | 🔄 修复中（架构裁定统一到 v17 金标准） |
| 7 | Medium | benchmark_v17.py:13 | BYTES_PER_TOKEN 漏计 memset（实际 1160B/token，虚高 ~12.3%；v11 用 empty 无 memset——对比不公平） | 🔄 P1 修复中 |
| 8 | Medium | benchmark/README:16 | 小 T 档 L2 驻留膨胀（262.3"理论 96%"实为 L2 口径；克隆 436>273 物理上限即铁证）；HBM 真实口径=T=65536 的 194.3（71%） | 🔄 P1 修复中（README 加注） |
| 9 | Low | 多文件 | 陈旧版本号注释（"v5"/"v4"/"v6"，现行 v11/v17） | 🔄 P2 顺手修 |
| 10 | Low | README 多处 | 计数出入（14→15 文件、"8/8"→7 bit-exact+1 smoke、T=65536 vs 65535、213.5 vs 194.3 两口径打架） | 🔄 P2 勘误 |
| 11 | Low | linear_triton.py:178 | v11 wrapper 无 shape 断言（窄输入 OOB 读） | 🔄 P2 补断言 |
| 12 | Low | paged_triton.py 多处 | 死代码（tle 导入/num_blocks/DIM/ENVELOPE 死参数）、bid 无界检查、evict_last 与优化文档相悖 | 🔄 P2 清理 |
| 13 | Low | test_paged/benchmark | 无重复 (seq,position) 冲突用例（triton 并行未定义 vs torch 后者胜）；ref 计时含分配 | 🔄 P2 补用例 |

### 2.3 生产相关事实（SRE）

- 生产 v17 无 block_table 引用 → 生产实际走 **linear（非 paged）路径**，linear v17 = Go（与已部署基线一致，零变更）
- paged 变体从未部署到生产 → Critical 缺陷**无生产暴露**，属交付包质量问题
- 四节点 md5 一致（kernel1=2d9cda46 routeA、kernel2=a795b2b4 v17）

---

## 三、集群/生产侧问题（SRE 核查）

| # | 项 | 状态 | 建议 |
|---|---|------|------|
| 1 | 02/03/04 的 vllm-tp4-worker.service 为 enabled | ⚠️ 重启会无 head 自启（孤儿 worker） | 下个窗口 disable（本窗口不动） |
| 2 | monitor/healthcheck systemd 服务 stop（矛盾项） | 停机窗口预期状态 | 恢复生产前还原 |
| 3 | DSL 混装（cu12 4.6.0 vs cu13 4.5.2） | 不阻塞当前工作 | 记录，择机清理 |
| 4 | 03/04 GPU 有 anemll-embed 5750MiB | 与 TP4 无关（Qwen3-Embedding 常驻） | 知悉即可 |

---

## 四、修复与执行状态总览

| 轨道 | 状态 |
|------|------|
| kernel1 代码修复（17+2 项） | ✅ 全部完成并验证（routeb-fix-log-2026-08-20.md） |
| kernel1 B-N1（DSL 半零缺陷） | ✅ **已根因修复**（c_dtype f32 × 16-bit C-atom 错配；fp16 复测 100% 逐位精确 + **368.1 TFLOPS 超 350 门禁与 356 基线**） |
| kernel1 SGLang 取证 | ⛔ 按用户指示放弃（B-N2 随 DSL 达标失去意义） |
| kernel1 SASS 门禁收口 | ✅ **Go**——主 kernel SASS 128/128=100% 条 MMA 为原生 FP4 block-scaled（OMMA.SF.16864.F32.E2M1.E2M1.E8，无 bf16 回退）；PTX 佐证 128 条 mxf4 e2m1 ue8m0；工件 _routeb_extract/sass_dump/ |
| kernel2 修复包（P0/P1/P2） | ✅ **全部完成并真机复核**（13 项闭环；7/7+8/8+7/7+8/8 全过；paged 解除禁部署；修正口径 211.1 GB/s@T=65536；遗留仅低优先级 R3 性能移植与冲突用例） |
| 最终报告 | ✅ 本清单即最终交付（双算子全闭环）；P3/P4 推进与否待用户裁定后另行立项 |

---

## 六、routeB P0-P2 验收链终态（全过 ✅）

| 门禁 | 判据 | 结果 |
|------|------|------|
| P0 环境 | CUTLASS DSL 可用 + driver ≥580.142 | ✅ 镜像内 4.5.2 原生含 sm_121a（P1 patch 免除）；driver 580.173.02 |
| P2 正确性 | fp16 全参考校验 + sentinel 判别 | ✅ 256³/1024³ PASS；65536/65536 逐位精确 |
| P2 SASS | mma.*e2m1 命中（硬门槛） | ✅ **Go：128/128=100% 原生 FP4 MMA**（OMMA.SF.16864.F32.E2M1.E2M1.E8） |
| P2 性能 | ≥350 TFLOPS | ✅ **368.1**（4096×14336×4096 @ 128³，>356 社区基线） |

---

## 五、待用户裁定事项（08-21 凌晨更新）

1. **P4 终局裁定**：routeB 判据未达（详见 §七）——**主理人建议维持 routeA 现役、routeB 不进灰度**（routeA 零改动即可恢复生产）；若用户接受"窗口期有限收益"可另行评估
2. **-hp 缺陷资产处置**（P3 发现）：148G×2（01/02）+ NFS 03/04 挂载——隔离或删除待裁定；转换器缺陷建议上报用户团队
3. **kernel2 修复包同步生产**（可选低风险）：修复后的测试套件 + paged 变体（已解禁）是否同步到 <INSTALL_DIR>/nvfp4/kernel2/
4. **NVIDIA 上游 issue**：是否批准对外提交（素材齐备）
5. **A 量化 kernel 化（行动项 A1-A4）**：routeB 复活的唯一路径（C++ 单 pass ~200GB/s），是否立项

---

## 七、P3/P4 推进结果（08-21 凌晨，用户批准后执行）

### P3 语义对接：✅ 通过

- **routeB 直接消费生产 MXFP4（-0731）真实权重，零重排直配**：W_packed [N,K//2] E2M1 + W_scale [N,K//32] E8M0 正是 kernel B 侧原生格式——**routeB 部署无需任何权重转换**
- 15/15 shape rel_err ≤ 4.26e-04（判据 1e-2）；跨层/跨 expert 4/4；奇数 M 含
- Scale 布局契约闭环：kernel 必须 atom-swizzle（plain 不可行），重排公式对官方 cvt 100% 逐字节验证
- 交付：routeb_prod_adapter.py（md5 7c46209 三方一致）+ routeb-p3-semantic-2026-08-21.md
- **🔴 附带重大发现：-nvfp4-hp 权重（148G×2）是缺陷品**——scale 恒为字节 1（全模型 54 张抽查）+ 码本非 E2M1（均匀 13 级 INT4 式，4.0/6.0 档零出现）；根因=转换 roundtrip 自参考验证漏检系统性码本缺陷。routeB 不需要它（-0731 直配）；资产处置待用户裁定

### P4 性能 A/B：❌ 判据未达（主理人裁定：维持 routeA，不启动灰度）

| 判据 | 结果 | 判定 |
|------|------|------|
| ≥1.5× routeA（同 shape） | kernel-only 峰值 1.27×（w1/w3 @M=1024），全矩阵 0.35×–1.27× | ❌ |
| 大 shape ≥350 TFLOPS | 351.8 官方中位（临界过 0.5%）/ 347.1 事件口径 / 383.3 冷机 | ⚠️ 临界（热方差 ±9%） |
| 端到端（含 A 量化） | 0.37×–1.02× 全面未达 | ❌ |

**判据校准勘误（诚实记录）**：1.5× 判据按克隆测试 routeA 57-130 TFLOPS（生产高压同机 + 含量化口径）校准；干净 GPU kernel-only 下 routeA 实测 **155–313**——门槛实际被大幅抬高。此为克隆口径误用，非 routeB 退步（368.1 基线复现 OK）。

**三个新发现**：①routeB M=16384 崩塌 0.35×（persistent kernel 内在问题，数值正确）——超大 prefill 禁区；②K=14336 dense 投影仅 94-143（0.42-0.60×）——此类 shape 不可用；③E2E 瓶颈在 A 量化不在 GEMM（vLLM C++ 单 pass ~200GB/s vs triton 病理 20GB/s，双 pass 修复到 94-136GB/s 仍差 1.5-2×）。

**结论**：不存在 routeB E2E 胜出的 M 档；若 A 量化补齐至 C++ 单 pass，唯一潜在窗口 w1/w3 M∈[1024,4096]（理论 ~1.05-1.2×）——收益边际。**routeB 归档为已验证 standby**（368 TFLOPS 峰值 + SASS Go + P3 直配 + 全工件留档），维持 routeA 现役。

---

> 本清单由主理人汇编自五份成员产出；kernel2 修复与 SGLang 取证回传后将更新状态并入最终报告。
