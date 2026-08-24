# 高价值工程资产汇编报告 — NVFP4/MoE 性能工程攻坚线（2026-08-20 ~ 08-22）

**汇编**: 多库（Docu）· 技术文档师 ｜ **范围**: DGX Spark 4 节点 TP4 集群（DeepSeek V4 Flash 生产）8/20-8/22 两天攻坚全部交付报告
**用途**: 管理层汇报 ｜ **口径**: 量化优先、诚实标注（standby / No-Go / 证伪如实呈现），每项资产附出处供深挖

---

## 1. 执行摘要（TL;DR）

过去两天，团队围绕 DeepSeek V4 Flash 在 DGX Spark 上的 NVFP4/MoE 性能展开了一条完整攻坚线：从双算子（kernel1 routeB / kernel2 v17）的代码审查与修复闭环，到 routeA/W4A4 生产探索（三路径否决 + 插件 TP4 实测 No-Go），再到 merged-GEMM 间接寻址 kernel 的设计-验证-生产冒烟全链路（P0 → P1 → Phase B → Phase C 五项全过 → e2e 冒烟）。**当前生产以最优基线运行**（B12X W4A16，全面超参考部署：C8 +20%、PR +14-42%），自愈链已开启，两次生产回退全部干净验证、零事故。攻坚产出 **5 类高价值资产**：5 项已验证 kernel/算子（其中 merged-GEMM 332T/MFU 66.4%、99.3% 保留率为核心新资产）、6 项一次性修复的生产集成机制、一套可复用的测量数据基线（含 30,717 tokens 真实路由分布）、一套排障方法论（sentinel 判别法 + SASS 门禁 + 微基准↔e2e 校准铁律）、以及一批同样值钱的负结果（避免了 routeA/W4A4/层1 8240 三条已证伪路线的持续投入，发现并清除了 296GB 缺陷权重资产）。**下一步**：按已知修复路径（DSL 预编译 + A 量化器精化 + 派生瘦身）消除 e2e 冒烟的最后一个阻塞点后重跑 PR 测量；层 3（路由集中化）已放弃（模型重训练成本超出预算，2026-08-22 用户裁定），当前 >2× 探索方向为大 batch × merged kernel 组合（分析进行中），当前栈现实上限 ~1.4-1.6×。

---

## 2. 资产分类汇编（核心章节）

### A. 已验证的 kernel / 算子资产

| # | 资产 | 量化指标 | 状态 | 位置 / 出处 |
|---|---|---|---|---|
| A1 | **merged-GEMM 间接寻址 kernel（routeb_official_v2）** | merge-8 shape 3072×12288×4096：**332.1 TFLOPS / MFU 66.4%**（P0 锚）；v2 间接寻址 **329.9T，性能保留率 99.3%**（判据 ≥90%）；正确性矩阵 **16/16 PASS**（含与基线 kernel 逐位一致、M=8240 生产档直过）；散射大 stacked B（512MB）零衰减 | **验证完毕，待 e2e 收口**（阻塞在 DSL 冷编译，非 kernel 本身） | 01:/tmp/_routea_work/routeb_official_v2/；routeb-merged-p0 / phasec / kernel-design-2026-08-21.md |
| A2 | **routeB DSL kernel（routeb_official）** | 4096×14336×4096 @ 128³：**368.1 TFLOPS**（>350 门禁、>356 社区基线）；SASS 门禁 **Go：128/128 条 MMA 100% 原生 FP4**（OMMA.SF.16864.F32.E2M1.E2M1.E8）；生产权重直配 rel ≤ 4.26e-04（15/15 shape） | **已验证 standby 存档**（P4 E2E 判据未达 1.5×，主理人裁定维持 routeA 现役） | 01:<INSTALL_DIR>/nvfp4/routeb-archive-2026-08-21.tar.gz（125 文件，md5 留档）；dual-kernel-problem-list-2026-08-20.md |
| A3 | **fused prologue/combine Triton 管线** | prologue（gather+NVFP4 量化+pack+swizzle 一步融合）**0.416ms**（预算 0.5ms 内）；combine（行内 6 块加权求和）**0.386ms**（预算 0.7ms 内）；**开销/双 GEMM 计算 ≈ 43%**（判据 ≤1.5×）；量化字节与 torch 参考**逐位一致** | **原型级验证通过**（生产融合版只会更快） | 01:/tmp/_routea_work/phasec_pipeline.py；routeb-merged-phasec-2026-08-21.md §3 |
| A4 | **routea_weight_adapter（零拷贝权重派生器）** | 生产 -0731 MXFP4 payload **零拷贝直配**（E2M1 字节完全兼容）；E8M0→E4M3 LUT 精确扩展对全模型 35,328 张量 9.26GB scale **逐值精确**；真实 layer-0 权重 rel=**1.41e-03**（比 requant 路径精确 **56×**）；派生成本 ~0.2ms/expert、+4.33GB/rank | **已交付并三方验证**（派生链 checkpoint 级等价亦已证明：与原生 -nvfp4 logprob 差 0.12/最小配对） | routea_prod_adapter.py + routea_weight_adapter.py（本地 + 01:/tmp/routeb_task12，md5 三方一致）；routea-integration-design / routeb-p3-semantic-2026-08-21.md |
| A5 | **kernel2 v17（NVFP4 DS-MLA KV 写回）** | v17 内核逐行审查**零缺陷**，与生产 md5（a795b2b4）一致；交付包 13 项发现全部修复闭环，真机复核 7/7+8/8+7/7+8/8 全过；修正口径 HBM 带宽 T=65536 = **211.1 GB/s**（理论 77%，**3.9× v11**）；paged 变体解除禁部署（64 槽生产块大小 8/8） | **生产留任金标准 + paged 解禁** | <INSTALL_DIR>/nvfp4/kernel2/；code-review-kernel2 + dual-kernel-problem-list-2026-08-20.md |

**补充（同一攻坚线内的辅助算子资产）**：w2 K-concat combine 数学（combine 折叠进单次 GEMM，优于独立 kernel 方案）；routea mini 模型构建器（build_mini.py，真实 checkpoint 抽层，可复现为任意 W4A4 变更的回归基座）；EMULATION 后端确认为 W4A4 语义 ground truth。

### B. 生产集成机制资产

**B1. 六项一次性修复的生产集成障碍**（每项均已生产实证，routeb-merged-e2e-smoke-2026-08-21.md §5）：

| # | 障碍 | 修复 | 价值 |
|---|---|---|---|
| 1 | EngineCore spawn 子进程不继承 monkey-patch | pip install + `vllm.general_plugins` entry point（必须 callable `module:install`） | 所有后续 vLLM 插件的唯一正确装载机制 |
| 2 | nvfp4 挂载只读 → pip egg_info 不可写 | /tmp 拷贝安装 | 只读挂载约束下的标准部署手法 |
| 3 | 插件安装拖慢冷启 → rendezvous 301s 超时 | --distributed-timeout-seconds 300→900 | TP4 启动韧性 |
| 4 | Triton JIT 首编译发生在 CUDA graph 捕获中 → capture 崩 | _derive 期 warmup 预编译 | 捕获期零 JIT 通用做法 |
| 5 | `.any()` host 同步在捕获中 → cudaErrorStreamCaptureUnsupported | Python 布尔分流，捕获路径零 host 同步 | CG 兼容性关键技巧 |
| 6 | 运行时 topk 无效 expert id → illegal access（capture 测不出、replay 才触发） | clamp + 权重清零 + 桶逐出 | CG 测试盲区的真实生产风险防护 |

**B2. entry point 插件机制 + 零污染结构性保证**：`VLLM_MOE_MERGED=0` → install() 直接 return 无任何 patch；确定性对照（未生效跑与基线 logprob 逐位一致 0.000%）旁证。两套插件（plugin_a1 / plugin_merged）均以 env 门控、unset 重启即回退，**回退 SOP 已两次实操验证**（P2 TP4 + e2e 冒烟，回退后全指标回基线带内）。

**B3. 零拷贝喂入机制**：merged 路径直接消费 stacked NVFP4 vec16 权重（payload [256, N_e, K/2] 原样 + E4M3 scale），kernel 侧 B/SFB tile 间接寻址（tile→expert 全局 id 查表），**不 concat、不重排、不拷贝**——这是把 P1 实测的 270ms/step host 组装开销压到 ~0 的结构性方案。

**B4. NVFP4 scale swizzle 实证公式**（从 proven CVT 输出反推、bijection 验证、torch 公式与官方转换逐字节 100% 一致）：
`off(m,g) = (m%32)·16·rm·rk + ((m//32)%4)·4·rm·rk + (m//128)·4·rk + (g%4)·rk + g//4`
（注意与 FlashInfer 的 swizzle_block_scale 布局不同，不可混用——两套布局契约均已字节级闭环。）

**B5. 生产编排与自愈资产**：restart_run.sh head-first 手动编排（绕过 start_tp4_cluster.sh 的 B12X 门禁死锁，两次生产恢复一次成功）；systemd monitor + healthcheck timer 自愈链（安全 attach 模式，开启零扰动已验证）；四节点脚本改动全程 .bak 留档 + checker 校验。运维侧另有：01 nfsd 僵死根因处置（systemctl restart nfs-server + 诊断链留档）、-hp 缺陷资产四节点外科手术式删除（释放 ~309GB，-0731/-nvfp4 完好验证）、routeB 服务器工件 /tmp→持久收编归档（routea-prod-ops-2026-08-21.md）。

### C. 数据资产

| # | 数据集 | 内容与规模 | 出处 |
|---|---|---|---|
| C1 | **e2e 生产基线全套** | C1 92.8 / C4 237 / C8 343 / C12 408（tok/s 聚合）；PR 4K/16K/32K/64K = 2510/2500/2420/2270；Agent 代码 +35% / 工具调用 +20%；MTP 接受率采样（规律文本 6.47-7.13、规律段 78-88%）；panorama decode 全景 16 档矩阵 | e2e-baseline-prod-2026-08-21.md（原始日志 e2e-baseline-results/） |
| C2 | **真实路由分布** | 30,717 tokens × 4 层（3 hash + 1 dense）：hash 层 top-10 组合覆盖 62.2%、top-set 单组合频率 8,220；dense 层 7,937 组合、平均 3 token/组合；M_g 阈值-覆盖率曲线（≥256/1024/2048/3072 → hash 62/59/35/27%，dense 28/27/27/27%） | routeb-merged-p1a-2026-08-21.md（routing_capture.jsonl） |
| C3 | **M_g 效率曲线** | 256→157T、512→254T、768→314T、1024→338T、**1536→351T 平台**、4096→367T（DYN_MIN 校准依据：可降至 256-512） | routeb-merged-phasec-2026-08-21.md §4 |
| C4 | **host 开销实测锚点** | D2D 带宽 **202GB/s**（比预估低 26%）；朴素 concat 233ms/step、gather/scatter 187ms/step vs 计算收益 ~0.3ms —— **900× 开销/收益比**（kernel 侧零拷贝路线的立项依据） | routeb-merged-p1a-2026-08-21.md §1-2 |
| C5 | **A/B 对照数据组** | 层1 8240 全量 A/B（PR/C1-C12/KV/panorama 全矩阵）；W4A4 三运行 TP4 对照（基线/hybrid/full + 回退）；routeA per-expert 五档 M 矩阵（0.19-0.42×）；W4A4 vs W4A16 九档 M 矩阵（M=1 6.17× ～ M=2048 0.95×） | layer1-ab / routea-tp4-p2 / routea-integration-design-2026-08-21.md |
| C6 | **权重健康度档案** | -0731 全模型 scale 扫描（35,328 张量、9.26GB、字节范围 [118,126] ⊂ E4M3 精确域）；-nvfp4 结构完整性 + 量化配置实证（group_size=16、E2M1 码本 16/16 使用）；-hp 缺陷三层证据链 | routea-integration-design / routeb-p3-semantic / routea-prod-ops-2026-08-21.md |

### D. 方法论资产

| # | 方法论 | 内容 | 已发挥作用 |
|---|---|---|---|
| D1 | **sentinel 判别法** | 输出缓冲预填 sentinel（-777）区分"未写出 / 写零 / 写入垃圾值"三种形态——B-N1 案中纠正了"恰好一半=未处理"的错误先验（实为**写入 ~1e-38 垃圾值、打印为 0 的视觉假象**） | 定位 c_dtype f32 × 16-bit C-atom 错配根因，救活整条 DSL kernel 路线；建议纳入后续排障标准动作 |
| D2 | **SASS 硬门禁工具链** | CUTE_DSL_KEEP=ptx,cubin 官方开关落盘 → cuobjdump -sass → grep 统计 MMA 指令形态（注意大写 opcode 需 `-i`）——证明 kernel 真走原生 FP4 张量核心而非 bf16 回退 | routeB 验收链第三层门禁（128/128=100%） |
| D3 | **微基准↔e2e 校准铁律** | probe 单变量外推不可替代同口径 e2e 验证：层1 的 probe 1.38× 边际收益在 e2e 兑现为 -3%~+0.8%；W4A4 微基准 1.32× 被 CG=1 的 static workspace 路径吞噬为 -13% | 两次避免按微基准数字做生产决策 |
| D4 | **止损与零污染机制** | 50ms 止损线（P1）、判据前置（P0 ≥300T、Phase C ≥90% 保留率）、No-Go 即回退条款、env 门控零污染、回退后全指标带内验证 | 两次 No-Go（W4A4、e2e 冒烟）均干净退出，生产零事故 |
| D5 | **mini 模型 logprob A/B 法** | 真实 checkpoint 抽层构建 mini（build_mini.py）→ 多方对照矩阵 → **总 logprob 为有效质量判据**（随机权重/逐元素 rel 会放大噪声误导定性）；W4A4 wrapper 运行间不确定的发现进一步确立"统计口径对比"规范 | 把 Task#20 的"疑似缺陷"正确定性为非缺陷，避免了 1-2 天无效排障 |

### E. 负结果资产（同样值钱——每项都避免了真金白银的持续投入或生产风险）

| # | 负结果 | 数据 | 避免的损失 / 指导的决策 |
|---|---|---|---|
| E1 | **routeA per-expert 形态否决** | 真实 MoE 调度几何（topk=6/256 experts → M_e≈96）下比生产 B12X **慢 2.4-5.2×**（0.19-0.42×）；效率拐点 M_e≥768 生产 4096 chunk 不可达 | 否决路径 A/B/C 三条权重路线 + 2-3 天集成工期；转向 merged-GEMM（热组合加速层）正确方向 |
| E2 | **W4A4 生产 No-Go（当前实现形态）** | TP4 实测：TTFT 4K -13%、TTFT 2K -47%、decode -16~-19%、KV -52%、weight +28GB；hybrid 结构性不可启动（+42GB 吃光 KV 预算）；根因 = CG=1 锁定 static workspace 路径 | 维持 B12X-only 并经三运行对照确认为当前最优；后续仅存"wrapper 动态路径复测"低成本选项（预期上限 ~+10-13%） |
| E3 | **层 1（batched 8240）证伪回退** | e2e A/B：PR -3.0%~+0.8%（远低于 +10% 门槛）、C12 -9.8%、KV -37%、64K 出现驱逐抖动轮；唯一亮点 C1 +17% 不足以抵消 | 立即回退 4096；建立"probe 外推 ≠ e2e 结论"的团队认知（D3 铁律的实证来源） |
| E4 | **-hp 权重缺陷发现** | scale 全模型恒为字节 1（54 张抽查）+ 码本非 E2M1（4.0/6.0 档零出现）→ 幅度失真 ~47×，任何 E8M0 路径均不可消费 | 清除 296GB 缺陷资产（四节点，fstab/exports/symlink 全链清理）；避免基于缺陷权重的无效验证；转换器缺陷证据链已备上报素材 |
| E5 | **B-N1 根因（c_dtype f32）** | f32 使 16-bit 专用 C-atom 错配 → ~50% 输出为垃圾值（视觉为零） | 救活 routeB DSL 路线（162.2→368.1 TFLOPS）；产出上游 NVIDIA/cutlass issue 素材（f32 静默半错应显式拒绝，待批准提交） |
| E6 | **K-256 tile 证伪** | vec16 K-256 编译可过但性能 **-33%**（223.8 vs 332.1T） | 排除"杠杆 3"，tile 128³ 定版——避免一周量级的错误优化方向 |
| E7 | **朴素 host 组装 900× 否决 + 零拷贝区间分桶证伪** | 270ms/step 开销 vs 0.3ms 收益；连续区间零拷贝浪费 7.5-25× | 直接立项 kernel 侧间接寻址（唯一正解），省去 host 侧所有无效尝试 |
| E8 | **routeB M=16384 崩塌 + dense 投影不可用** | persistent kernel 超大 prefill 0.35×；K=14336 dense 投影 0.42-0.60× | 划定 routeB standby 的适用边界，防止误部署 |
| E9 | **>2× 杠杆栈边界（层 3 已放弃）** | 层1+层2 现实上限 1.35-1.5×（层1 已证伪后更低）；>2× 原唯一通路为模型层路由集中化（dense 覆盖需 83-100% vs 现状 27%，3-4× 集中度提升） | 把">2× 目标"校准为"1.4-1.6× 现实上限"交管理层裁决——**2026-08-22 用户裁定：放弃路由集中化方案（模型重训练成本超出预算）**；当前 >2× 探索方向转为大 batch × merged kernel 组合（分析进行中） |

---

## 3. 当前生产状态与基线

**生产终态**：四节点 TP4 基线运行（B12X_MXFP4 W4A16 + dspark n=7 投机解码），KV ~6.02-6.07M tokens，/health 200，自愈链（head/worker monitor + healthcheck timer）全部 active。两次窗口回退均干净验证（脚本 .bak 恢复、checker PASS、零 Merged/W4A4 日志、插件 env 门控零污染）。

**基线 vs 基准包参考部署（2026-08-21 实测，中位口径）**：

| 指标 | 我方 | vs 参考中位 | vs 参考最优 |
|---|---|---|---|
| C8 聚合 decode | 343 tok/s | **+20%** | +13% |
| C12 聚合 decode | 408 tok/s | **+19%** | +14% |
| C4 聚合 decode | 237 tok/s | +9% | +1% |
| C1 单流 | 92.8 tok/s | -4% | -25% |
| PR prefill 4K/16K/64K | 2510/2500/2270 | **+14% / +25% / +42%** | — |
| Agent 代码生成 | 132.9 tok/s | **+35%** | +30% |
| Agent 工具调用 | 126.9 tok/s | **+20%** | +16% |

> 一句话：**高并发聚合、prefill 吞吐、Agent 场景全面领先参考部署；唯一短板是单流 C1（-4%，与 MTP n=7 接受率敏感性相关，已列 P3 排查项）**。

---

## 4. 下一步路线

**主线：消除 e2e 冒烟最后一关，拿 PR 信号**（修复路径明确、无未知风险，routeb-merged-e2e-smoke §7）：

| 优先级 | 项 | 说明 |
|---|---|---|
| P0 | DSL 预编译（阻塞根因） | _derive 期预编译固定 M_pad 档位集（~10 次编译启动期吸收）或 AOT cache 烘焙随插件分发；另查生产容器 10× 编译慢根因。冷编译 45min 卡死首 prefill 是 PR 未取得的唯一原因——架构侧（部署/TP4 启动/CUDA graph/decode warmup）已全部打通 |
| P0 | A 量化器精化 | 对齐 flashinfer 两级 scale 方案，mini logprob 41.5% → 目标 ~1%（当前 0.36% 的插件骨架已不劣于已验收 W4A4 的 0.41%，此项消除全暴露口径的噪声放大） |
| P1 | 派生内存瘦身 | KV -29% 超 +9GB 预算（免 f32 壳驻留 / w2 缓存 cap 调优） |
| P1 | PR 四档重测 | 上述完成后重跑 smoke（六脚本全部就绪） |
| P2 | Triton decode 性能补测 | 本轮仅验证正确性（warmup 20×200 OK），C1/C8 快测补齐 |

**方向性判断（供决策）**：
- **大 batch × merged kernel 组合（当前 >2× 探索方向，分析进行中）**：M=8240 下 merged 覆盖档位整体上移（≥2048 从 0%→27%），MoE 加速 1.27×→1.50×——但层 1（8240）e2e 已证伪，该叠加模型需以真实 e2e 数据重新校准，不可按模拟数字立项。层 3 放弃后，此方向是 >2× 的主要探索线。
- **层 3（路由集中化）已放弃**：2026-08-22 用户裁定——"放弃路由集中化方案，模型重训练成本超出预算"（成本裁决而非暂缓）。路线图中"层 3 是唯一能推过 2× 的层"的旧口径（routeb-pr-2x-strategy-2026-08-21.md §4）自该裁定起失效；当前栈现实上限 ~1.4-1.6×。
- **待办收尾项**：start_tp4_cluster.sh B12X 门禁死锁修复（SRE）；02/04 systemd unit 磁盘版本差异核查；NVIDIA 上游 issue 提交待批准；kernel2 修复包是否同步生产（低风险可选项）。

---

## 5. 投入产出小结

**投入**：2 个维护窗口 + 连续两夜攻坚（约 6 名成员接力：架构师 ×2、SRE、代码审查师 ×2、汇编），生产两次变更两次干净回退，**零生产事故**。

**产出对照**：

| 资产类别 | 数量 | 代表性价值 |
|---|---|---|
| A kernel/算子 | 5 项 | merged-GEMM 332T/99.3% 保留率（核心增量）；routeB 368T standby；适配器 56× 精度提升 |
| B 集成机制 | 6 修复 + 4 机制 | 插件 entry point 装载/CG 兼容/零拷贝喂入/swizzle 公式——后续所有 kernel 生产化的可复用底座 |
| C 数据资产 | 6 套 | e2e 基线全套 + 30,717 tokens 路由分布（后续一切优化的对照锚点） |
| D 方法论 | 5 项 | sentinel 判别法、SASS 门禁、e2e 校准铁律——团队长期排障产能 |
| E 负结果 | 9 项 | 三条路线证伪止损 + 296GB 缺陷资产清除 + 上游 issue 素材 |

**一句话总结**：两天投入换来了「**一条已验证到 e2e 门口的 merged-GEMM 加速路线（唯一阻塞点修复路径明确）+ 一个全面超参考部署的生产基线 + 一整套防止重复踩坑的负结果库**」——无论后续 PR 判据是否达成，这批资产对下一条性能线都是直接可复用的启动资本。

---

## 附：源文档索引（深挖入口）

| 主题 | 文档（deliverables/engineering-assurance/） |
|---|---|
| e2e 冒烟终局 | routeb-merged-e2e-smoke-2026-08-21.md |
| Phase C 性能验证 | routeb-merged-phasec-2026-08-21.md |
| merged kernel 设计 + Phase B | routeb-merged-kernel-design-2026-08-21.md |
| P1 开销量化 + 路由分布 | routeb-merged-p1a-2026-08-21.md |
| P0 kernel 基准 | routeb-merged-p0-2026-08-21.md |
| routeA 适配/集成设计 | routea-integration-design-2026-08-21.md |
| W4A4 语义排障 | routea-w4a4-debug-2026-08-21.md |
| A′ 插件 P1 | routea-plugin-p1-2026-08-21.md |
| TP4 三运行 No-Go | routea-tp4-p2-2026-08-21.md |
| e2e 基线 | e2e-baseline-prod-2026-08-21.md |
| 层 1 A/B | layer1-ab-20260821.md |
| 双算子总清单 | dual-kernel-problem-list-2026-08-20.md |
| 修复日志（B-N1/SASS） | routeb-fix-log-2026-08-20.md |
| P3 语义 + -hp 缺陷 | routeb-p3-semantic-2026-08-21.md |
| kernel2 审查 | code-review-kernel2-2026-08-20.md |
| >2× 杠杆栈路线图 | routeb-pr-2x-strategy-2026-08-21.md（其层 3"待决策"口径已被 2026-08-22 用户裁定取代：层 3 放弃，见 §4 修订） |
| 运维（-hp 删除/nfsd） | routea-prod-ops-2026-08-21.md |
