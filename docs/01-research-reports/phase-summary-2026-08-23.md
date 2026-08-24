# DGX Spark 4 节点 TP4 集群（DeepSeek V4 Flash）攻坚项目
## 阶段性总结报告（2026-08-22 ~ 2026-08-23）

- **汇编**：多库（Docu）· 技术文档师（tech-writer-1）· 工程保障团队
- **范围**：2026-08-22 至 2026-08-23 全部工程保障报告（33 份主文档 + 3 个交付物目录）
- **用途**：督导阶段性总结 + 管理层汇报口径 + 后续排期依据
- **口径**：量化优先、诚实标注（✅完成 / ⛔No-Go / 🔄在途 / 📌待办），关键结论均附数据与出处（相对路径，见文末引用说明）

---

## 一、TL;DR（三句话总览）

1. **两天完成了一次"先深挖根因、再落地收益"的完整闭环**：从 08-22 的调度器钳制（threshold）与 AR/通信/内存三大根因调查，到 08-23 的 threshold 4096 生产采纳（PR +8.5~+13.5%）、W4A4 翻案（并发 prefill +11.5%/+13.0%）、FlashInfer 0.6.16 生产替换、以及 **LuZ0.3.1（= B2 组合 + util 0.82）正式生产落地**——两天内生产 PR 从 2510→~2950-3060 tok/s 区间，并在 08-23 晚形成"**填满甜点区 + 削减墙下流量**"的下一阶段叙事（现实上限 ~3250-3350 tok/s）。
2. **大量"诚实关闭"同样关键**：AR 四路线全 No-Go（占比校准为 ~13%）、MTP 扩档 No-Go（n=7 已最优）、budget 8192 关闭（M 供给在 4096 见顶）、Eugr attention b12x 化 No-Go（PR <5% + C12 -49%）、v5 环序补丁 No-Go（通道数无带宽收益）、b′ native 共享 No-Go（decode -25%/-34%）——每项关闭都有实测数据，避免继续无效投入。
3. **环境级问题已被钉到根因层**：环境级随机 AR stall 判决为"**单节点 worker 单核约束（CPU 8/9，即 NCCL proxy 目标核）**"（12/12 完美相关），与库/通道数无关；模式方差排除 H2/通道映射后转实验 C 已关闭，剩 E3（cpuset 监控 /--cpuset-cpus A/B / NCCL_IGNORE_CPU_AFFINITY A/B）**挂起待处理**。

---

## 二、成果清单总表（按主题分组）

> 全部引用为相对路径（`deliverables/engineering-assurance/…`），状态图例：✅ 完成/采纳｜⛔ No-Go/关闭｜🔄 在途/待窗口｜📌 待办/待裁定。

### 主题 1 · W4A4 翻案与采纳（threshold 主线）

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 1.1 | 08-22 | `threshold-retest-2026-08-22.md` | 昨日"-22.5%"系重启级模式方差混杂；threshold 2048 三对交错 3/3 正向；4096 探路显著正向 | 2048 合并 4K +6.4%/16K +11.1%；**4096 两臂 2849/2851（+12%）**，零慢轮零方差；M_e=96 进入 B12X 高效区间 | ✅ |
| 1.2 | 08-22 | `threshold-4096-adoption-2026-08-22.md` | **threshold 4096 生产采纳成功**，四机脚本固化，自愈链恢复 | PR 四档 2849/2829/2724/2462 = **+13.5%/+13.2%/+12.6%/+8.5%**；DE 归一 +5.0%/+3.2% 无回退；质量门 PASS | ✅ |
| 1.3 | 08-22 | `ws-dedup-patch-2026-08-22/ws-dedup-patch-2026-08-22.md` | b12x wrapper 几何键共享池补丁实施，L1 24/24 + L2 GPU 16/16 全过 | 每 wrapper workspace **455 MiB**（medium 几何）；43 层去重预期省 13-27GB——W4A4"+42GB 吃光 KV"死结结构性解除 | ✅ |
| 1.4 | 08-23 | `wsdedup-l3-combo-2026-08-23.md` | **W4A4 翻案成立（"值得深测"档）**：P2 的 -13% 系 threshold 1024 时代形态误差 | M3 full+on：weight **68.15→45.32 GiB（省 22.83）**、KV -51%→-8.5%、**PR 4K 2982（+8.3%）**、池性能零代价、质量门 4/4 逐字一致；剩余成本 decode 归一 -6~-9% | ✅ |
| 1.5 | 08-23 | `w4a4-ext-2026-08-23.md` | **W4A4 并发增益成立且大于单流**；threshold 8192 干净否决 | 并发聚合 **C6 3060（+11.5%）/ C12 3092（+13.0%）**；单流 4K 2994（+8.1%）、64K +14.7%；8192 无增量且 KV 3.48M 跌破 4.5M 底线；TTFT 改善 ~11% | ✅ |

### 主题 2 · b′ native 共享路线（设计 → 实现 → No-Go 储备）

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 2.1 | 08-23 | `a3-hybrid-slim-design-2026-08-23.md` | **旧结论"payload 共享不可行"被源码推翻**：b12x 0.15.3 内置非破坏性 native 路径恰为"A4 prefill + A16 decode 同进程共存"设计 | hybrid weight 79.82→**~45.3 GiB**、KV 1.53M→~5.48M（+e 可达 6.2-6.6M≥M1）；路线 a/c/d 否决（kernel 重写/零增益/LPDDR 带宽算术）；工作量 2-4 天 | ✅ |
| 2.2 | 08-23 | `bprime-impl-2026-08-23/`（含 §b′ 实施记录于 a3 报告） | b′ 三处规格改动 + 两处勘察新增全部实现，L1 CPU 68 + GPU 12 PASS | native 主 GEMM M=8/M=64 与 packed **逐位相等（max_rel=0.0）**；显存增量实测 = scale store only（0.8MB vs 7.1MB 小几何）；**新发现上游缺陷 e8m0×micro direct 100% NaN**（强制关闭防护） | ✅ |
| 2.3 | 08-23 | `bprime-window-2026-08-23.md` | **N1 门 FAIL → b′ No-Go**：内存收益全部兑现但 native staging 代价否决 decode | 内存门过：weight 45.32 GiB / KV 5.54M（A3 承诺兑现）；**DE C1 14.7（-25%）/ C12 61.8（-34%）** vs 门 -3%；mid-M staging 曲线 M=8 1.61×/96 1.74×/512 1.60×/2048 1.20×/3071 1.14×；prefill/并发/质量门全过 | ⛔（保留为已验证设计储备） |

### 主题 3 · LuZ0.3.1 生产落地

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 3.1 | 08-23 | `luz031-deployment-2026-08-23.md` | **LuZ0.3.1（= B2 组合 + util 0.82）采纳成立，生产终态全绿**；过程中发现并修复 FI 0.6.16 误回滚 | PR 四档 2950.5/2943.6/2834.2/2550.0（vs W4A16 基线 **+6.6~+15.1%**）；并发 C6 3057（+11.4%）/ C12 3056（+11.6%）；**KV 5.73M ≥5.7M 门过**；质量 4/4 + needle 3/3 + 回归 0 异常；补丁 6/7（E4 留窗口）；检查点+自包含恢复镜像（digest sha256:85f2149f…）+ restore_luz031.sh dry-run 演练 | ✅ |

### 主题 4 · 上游核对与性能上限

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 4.1 | 08-23 | `upstream-check-perf-ceiling-2026-08-23.md` | 上游 b12x 1.2.6 后零提交、6 项问题无一已修（PR #227 最接近）；**FI 0.6.17 实存且最对症**；性能释放最大空间 = bf16 稠密池（37% FLOPs）→ W4A4/routeB；**叙事切换为"填满甜点区 + 削减墙下流量"** | 全池转化推算 **PR +5~7%（2994→~3160-3210）**，现实上限 **~3250-3350**；shared experts 首发（M=4096 恰在 routeB 350T 平台）+ lm_head 唯一打 decode 带宽墙；唯一值得投的移拐点方向 = routeB 平台 M=1024 下探 512-768 | ✅ |

### 主题 5 · 环境 stall 调查与判决（模式方差 + AR stall 双线收束）

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 5.1 | 08-22 | `slowround-rootcause-2026-08-22.md` | 慢轮根因**未定论**（快臂 0 慢轮无样本）；H1（IRQ 风暴）大幅下修、H2（libncclpin 按名 pin 漏网）升头号；确立**慢轮判定看 steps/s 非 tput** | 阶段 2：14 轮 0 慢轮；de C1 tput 方差 36.6% 但 steps/s 恒定 19.6-20.1；arch_timer ~375-725Hz/核常驻含隔离核 | 🔄 |
| 5.2 | 08-23 | `expd-r123-2026-08-23.md` | **实验 D 复现模式方差 + R1-R3 卫生修复全落地**；H2 实锤但**与臂模式不相关**（排除为主因） | 7 次抽签：慢簇 4.25-4.43s×4 / 中簇 3.07-3.48s×3，快臂 0；稳态 PR 差 ~3.4%；H2 漏网率 4/7 重启（~57%）但慢臂 R1/R5 干净仍落慢簇；IRQ 84×4 迁 0-4,10-14 达标 | ✅（修复落地；模式方差转实验 C） |
| 5.3 | 08-23 | `ringonly-w1-2026-08-23.md` | **v5 = No-Go + ★P0 新发现环境级随机 AR stall**（与库/通道数/pin/embed 全无关） | 干净窗口 busbw：4ch 21.57 / 8ch 21.12 / 16ch 13.01（**通道数无带宽收益**），33.5MB 档三配置持平 ~19.8 → 生产 PR 增益 0%；stall 17-20ms/AR、1ms 量子化、per-run 全有或全无、随窗口时间恶化（25%→~5%） | ⛔（v5）+ 🔄（环境 stall 升级） |
| 5.4 | 08-23 | `envstall-rootcause-2026-08-23.md` | **GDR=0 四节点实锤**（AR 数据面 = host staging）；RC1 头号候选 = NCCL proxy 线程"~1ms 定时睡眠轮询"抽签；生产稳态未中招 | stall 期 proxy 核（CPU 8/9）出现 ~0.9-1.3kHz arch_timer（clean 无）；生产 expD R1-R7 全带内（若 17ms/AR 打生产 PR 将坍塌 >30% 未观测）；附带发现 Prometheus down 已处置 | 🔄（判决实验 E1-E6 清单列出） |
| 5.5 | 08-23 | `expverdict-verdict-2026-08-23.md` | **环境 stall 最终判决：根因 = 单节点 worker 单核约束（CPU 8/9，即 proxy 目标核），12/12 完美相关**；实验 C 排除 channel 映射；RC1 字面定义修正 | stall 率 6/10（med 17.99-21.18ms vs CLEAN 0.59ms）；受限节点随机（01/02/03/04 均出现）；stall/clean 稳定差异 = busy-spin（vol ~180/s）vs timed-poll（vol ~16-45/s）；`.sudo_pw` 四节点删除 | ✅（判决完成） |
| 5.6 | 📌 | — | **E3 挂起待处理**：cpuset 监控 / `--cpuset-cpus` A/B / `NCCL_IGNORE_CPU_AFFINITY` A/B（expverdict §6 建议） | 单核约束触发机制（cgroup cpuset 瞬时收窄 vs 调度器挤占）未闭环 | 📌 |

### 主题 6 · E4 KV 足迹

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 6.1 | 08-23 | `e4-kv-footprint-2026-08-23.md` | **"+58%/token KV 足迹" 是对指标语义的误读**：真实机制是"每请求 KV 预留随批大小上涨"，物理 per-token 字节不变 | 报告值 = max_concurrency × max_model_len（非物理池）；物理 per-token ≈9.4 KiB 不变（B2 49.27 vs B3 48.19 GiB 仅 -2.2%）；`max_in_flight=2×batched` 使 SWA 每请求预留 131→259 块；**满长并发 9.16→5.80（-36.7%）** | ✅ |

### 主题 7 · FI 0.6.17 评估 + bf16 稠密池 P0 拆账

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 7.1 | 08-23 | `fi017-p0-accounting-2026-08-23.md` | **U1（FI 0.6.16→0.6.17）Go（有条件）**：wheel 路径成立（CuTe-DSL 下限降回 4.5.2 与生产精确匹配）；P0 拆账推算 shared/lm_head/attn 三节点份额 | P0 拆账（推算）：attn 15-19 / shared 9-12 / lm_head 3-5 µs·token⁻¹；decode 带宽墙侧 lm_head 唯一可转化（0.27GB→0.07GB，C12 步时 -5~6%）；P1 顺序 = shared 首发 → lm_head 第二 → attn 第三 | ✅（窗口验证项待执行） |

### 主题 8 · 其他重要成果（支撑/边缘线）

| # | 日期 | 报告 | 核心结论 | 关键数据 | 状态 |
|---|------|------|---------|---------|------|
| 8.1 | 08-22 | `engineering-assets-report-2026-08-22.md` | 管理层资产汇编：5 类高价值资产 + 9 项负结果 | kernel×5（merged-GEMM 332T/MFU 66.4%、routeB 368T）、集成机制 6+4、数据×6、方法论×5、负结果×9 | ✅ |
| 8.2 | 08-22 | `large-batch-analysis-2026-08-22.md` | **层 1 失败根因定论：调度器把单请求 prefill 钳制 threshold=1024 tokens/步**——batched 不改 M，probe 1.38× 在基准中不存在 | C12 -9.8% 不可归因 8240（0 抢占/KV 13.53%/接受率更高）；KV 账 5.78KB/token，batched 4096→8240 代价 12.8GB 中 ~11GB 是非分解工作区；batch 上限 4096 主档 + 8264 上限档 | ✅ |
| 8.3 | 08-22 | `pr-de-bottleneck-analysis-2026-08-22.md` | **Roofline 瀑布 + 三大缺口**：AR 31%、MoE M_e=24 带宽几何 + B12X 62-69% 效率、稠密池/attention；**口径修正：4K/16K/32K/64K 实为 8.2K/32.8K/65.5K/131K** | PR 407ms/步：MoE 185-204ms（45-50%）+ AR ~127ms（31%）+ attention 40-85ms + 稠密池 ~30-35ms；理论上限 3560-3720（+fp8 AR 4450-4650）；C12 已顶 273GB/s 带宽墙 | ✅ |
| 8.4 | 08-22 | `p1-p2-research-2026-08-22.md` | 四项研究：threshold 异常=混杂头号 + 形状 JIT；workspace 去重**口径修正**（生产 W4A16 已共享，×43 仅属 FlashInfer wrapper 路径）；FI 0.6.16 rebase 1-2 周；attention b12x 化建议 Eugr 并行 A/B | 镜像 0.6.15 实为混合体（94 路径差异，41/64 修改文件与 0.6.16 逐字节同）；b12x 树未被补丁 | ✅ |
| 8.5 | 08-22 | `ar-optimization-2026-08-22.md` | **AR 四路线全 No-Go/已达物理最优 + 占比重大校准** | fp8 AR 端到端 1.522ms = bf16 2.6×（scale 同步 0.509ms ≥ fp8 省 0.27ms）；**AR 占比终校准 12.9%（52.3ms/步）非 31%**（busbw 实测 21.4GB/s）；8/16 通道定制库灾难劣化；C0 基线零漂移 | ⛔（AR 线关闭） |
| 8.6 | 08-22 | `mtp-tuning-2026-08-22.md` | **MTP 调优线关闭**：n=7 已是本负载最优；"acc_len 是 DE 唯一大杠杆"被修正 | n=10 C1 +2.6%（噪声级）/C12 -25%；n=5 C1 -8.5%；自适应 n 全面负（C12 -24%、PR -21%）；机制：扩 n → verify token 增 → distinct experts 增 → 权重读/步增（带宽墙）→ 步时 +11-18% 吃掉 acc_len 红利 | ⛔ |
| 8.7 | 08-22 | `routing-recapture-scheduler-ab-2026-08-22.md` | **真实路由重采集 + threshold 2048 A/B（当日被 threshold-retest 翻案前）**；合成语料失真双反 | 等样本量：hash ≥1024 覆盖 58.9%→16.1%、dense 26.8%→3.3%（高估 3.7~8×）；口径 3 单步 hash ≥64=17.9%/dense≈0-2%；merged 单步收益上限极低 | ⛔（Route 1 当日否定） |
| 8.8 | 08-22 | `upstream-tracking-2026-08-22.md` | b12x 上游身份确认（0.15.3 @ 05-30 vs 上游 1.2.6 @ 08-20，落后 3 个月）；三项最优采纳 | ①workspace 跨层去重（+28GB×43 对症，上游已解）②FI 0.6.16 ③vLLM main B12X 全家桶（Eugr 配方） | ✅ |
| 8.9 | 08-22 | `b12x-tail-path-strategy-2026-08-22.md` | **B12X 12 项形式清单 + Triton 4.3× 分解 + B-lite 推荐** | B12X 小 M 带宽最优机制本质（块 8 → 每 expert 权重读 1 遍）；Triton 重写上限仅 B12X 50-70%（fragment 布局不可修）；B-lite 按层选择性双表示 = KV 从悬崖变旋钮 | 🔄 |
| 8.10 | 08-22 | `freetoken-research-2026-08-22.md`（含 §P0 实验） | FreeToken 动态路由机制解析 + **P0 决定性实验**：merged 有条件复活为"跨块 MoE 聚合"（deferred MoE） | 明确不做路由预测/聚合（不能直接改写覆盖率数学）；w=4 M_e≥384 工作量占比 hash 68.1%/dense 81.3%（吞吐加权）；N-merge 永久关闭；TTFT +(w-1)×960ms 仅限 prefill | ✅ |
| 8.11 | 08-22 | `fi-rebase-eugr-prep-2026-08-22.md` | **rebase 工作量大幅下修（1-2 周 → 代码 1-2 天）**：24 个"冲突文件"几乎全部被 0.6.16 官方吸收 | 真 fork delta 仅 5 文件 + 58 新增文件；CPU import 冒烟 22/23（1 个 TVM 环境伪缺陷）；Eugr 镜像 34.5GB 已拉至 01 | ✅ |
| 8.12 | 08-22 | `routeb-merged-mainline-2026-08-22.md` | **merged-GEMM e2e No-Go 关账**：Triton W4A16 是架构死穴，尾部必须保留 B12X | 主臂 PR -77~-75%、∞ 臂 -78~-75%（归因 Triton grouped BM=16 小 tile ~3-4% 峰值效率，prefill/decode ×4.3）；NaN 根因+修复、py-spy watcher v2、三臂自动化链可复用 | ⛔ |
| 8.13 | 08-22 | `routeb-merged-fix-rerun-2026-08-22.md` | fix-engineer 系列收束：探针脚本 bug 制造"架构级阻塞"假象，数值侧全通 | 真实权重决定性验证：低 nibble=偶 k 约定确认（3.9e-3）；merged 派生链 1.8e-4；41.5% mini logprob 系不同 prompt 集伪影；45min 挂起 = watcher pid 发现 bug | ✅ |
| 8.14 | 08-22 | `ringonly-optimization-plan-2026-08-22.md` | ringonly 补丁清单还原 + v5 设计 + QPS2 捷径 + 窗口清单 | 生产库 = 官方 2.30.7 重建 + 4 补丁；v5 预期 busbw 21.4→25-28GB/s、33.5MB AR -15-24%、PR +2-3%；DE 小消息 98µs@196KB 贴环架构下界；2-hop 维持终审否定 | ✅ |
| 8.15 | 08-22/23 | `ringonly-v5-2026-08-23/ringonly-v5-brief-2026-08-23.md` | v5 补丁实现 + 构建成功 + W1 测试就绪 | md5 2b8669ec，GLIBC_MAX=2.34 达标，ABI 与生产库 diff=0；四机分发 md5 一致；生产库零接触 | ✅ |
| 8.16 | 08-23 | `windowA-fi-cg-budget-2026-08-23.md` | 短窗 A 三任务：**FI 0.6.16 GPU 冒烟 Go / cudagraph=0 无增益关闭 / budget 8192 关闭** | 冒烟 5/5（B12xMoEWrapper 与 0.6.15 逐位一致、JIT 磁盘缓存 317→32.4ms）；cudagraph=0 同模式 2826 vs 2850（-0.84%）；8192 PR 4K -1.4%、KV 6.06M→3.85M（-36.5%）——**M 供给在 4096 见顶，"budget+threshold 协同上推单流 M"证伪** | ✅（两关闭） |
| 8.17 | 08-23 | `fi016-replacement-2026-08-23.md` | **FlashInfer 0.6.16 生产替换三门全过，采纳保持** | 性能门 PR 四档 -1.8~-2.8% 全带内、DE C1 -3.8%/C12 -3.5% 带内；质量门 4/4 逐字；回归观察门 0 error + needle 64K 3/3 + 128K 2/2；目录级 bind mount + 持久 JIT 缓存 | ✅ |
| 8.18 | 08-23 | `eugr-ab-2026-08-23.md` | **Eugr attention b12x 化 No-Go（判据双双不过）**；同权重公平 A/B 成立 | PR 四档 +0.4~+4.0%（判据 ≥+5%）；**DE C12 -49%**（step_eff 46.1 vs 93.9，第二臂复现稳定）；C12 扩展 2.5× vs 生产 4.7×；6 次启动迭代工程发现留档 | ⛔ |

**覆盖统计**：主文档 33 份（08-22 计 18 份、08-23 计 15 份）+ 交付物目录 3 个（ws-dedup-patch / bprime-impl / ringonly-v5），与目录中 08-22~08-23 日期后缀文件一一对应。

---

## 三、主题章节详述

### 3.1 主线叙事：从"调度器钳制"到"LuZ0.3.1 落地"，再到"填满甜点区"

两天攻坚的主线可以用一条证据链串起：

1. **08-22 上午 batch-analyst 定论**（`large-batch-analysis`）：层 1 失败的根因是**调度器把单请求 prefill 步进钳制在 threshold=1024 tokens/步**——`max_num_batched_tokens` 只决定打包请求数、不改单请求 GEMM 的 M。这直接指出**调 threshold 是比调 batched 更直接作用于单流 PR 的新杠杆**。
2. **08-22 上午 window-engineer 的 threshold 2048 A/B 初判 -22.5%**（`routing-recapture-scheduler-ab`）一度让"扩 M"路线蒙上阴影，但 p12-researcher 的原始数据再分析（`p1-p2-research`）指出同配置 1024 两轮相差 26 个百分点——**混杂嫌疑**。
3. **08-22 下午 threshold-retester 带仪器交错复测**（`threshold-retest`）最终翻案：-22.5% 是**重启级模式方差**混杂，2048 三对交错 3/3 正向，**4096 探路 +12% 且零慢轮零方差**——扩 M 路线重开。
4. **08-22 晚 threshold-4096-adoption**（`threshold-4096-adoption`）将 4096 采纳上线：**PR 四档 +8.5~+13.5%**。这一步成为两天内所有后续 W4A4 收益的"地基"。
5. **08-23 W4A4 翻案**（`wsdedup-l3-combo` + `w4a4-ext`）：ws-dedup 补丁解除内存死结（省 22.83 GiB），threshold 4096 采纳使 chunk M=4096 恰入 W4A4 1.32× 甜点区——**"两天前的 threshold 采纳直接改写了 W4A4 的命运"**（P2 的 -13% → M2/M3/M4 三臂 +6~8%）。并发测试进一步确认 C6/C12 聚合 +11.5%/+13.0%（大于单流且随并发扩大）。
6. **08-23 晚 LuZ0.3.1 落地**（`luz031-deployment`）：用户批准 **B2 + util 0.82** 组合上生产，验收全过（PR +6.6~+15.1%、KV 5.73M≥5.7M、检查点+恢复镜像就位）。
7. **08-23 upstream-check-perf-ceiling** 给出下一阶段叙事：**"填满甜点区 + 削减墙下流量"**，现实上限 ~3250-3350 tok/s——即把 shared experts / lm_head / attn prefill 等"墙下流量"搬进 W4A4 甜点区（P1 shared experts 首发，P2 lm_head，P3 attn），唯一值得投的"移拐点"方向 = routeB 平台 M=1024 下探到 512-768。

### 3.2 主题 1：W4A4 翻案与采纳（详述）

**翻案的三层证据链**：
- **内存死结解除**：`ws-dedup-patch`（L1 24/24 + L2 16/16，per-wrapper 455 MiB）→ `wsdedup-l3-combo` M3 实测 weight 68.15→45.32 GiB（省 22.83，落预期 13-27GB 带内），KV -51%→-8.5%。此处还修正了一个**任务书笔误**：W4A4 插件实际位于 `plugin_a1/`（VLLM_MOE_W4A4 门控），`plugin_merged/` 是 routeb merged-GEMM 插件，两者是不同资产；且补丁池与插件路径原本不连通（插件从 flashinfer.fused_moe 直接构造 wrapper 不经补丁池），已做最小池集成（`.bak-wsdedupl3` 留档）。
- **性能翻正**：M2/M3/M4 同为 fast 模式直接可比，M3 PR 4K 2982（+8.3%），M4 hybrid 2999（+8.9%）。机理自洽：P2 测于 threshold 1024 时代（chunk M=1024 落 W4A4 0.79-0.95× 劣势区），4096 采纳后 chunk M=4096 恰入 1.32× 甜点区。
- **并发放大**：`w4a4-ext` 建立并发 prefill 聚合基准（prefill 计算饱和 → 聚合吞吐是"单位计算效率增益"的干净标尺），B2 并发增益 C6 +11.5%/C12 +13.0% 大于单流 +8.1% 且随并发扩大；**8192 双证伪关闭**（性能无增量 + KV 3.48M 跌破底线，见主题 6 的机制修订）。

**状态**：W4A4 并发强阳性已实证（`w4a4-ext` E1 行动项），但**最终采纳决策回到用户**：B2（full W4A4+池+4096：并发 +11.5~13%/decode 中性/KV -8.9%）vs 维持 W4A16 基线。08-23 晚用户已按"B2+util 0.82"拍板 → 即 LuZ0.3.1（见主题 3）。

### 3.3 主题 2：b′ native 共享路线（No-Go 与设计储备）

- **设计翻案**（`a3-hybrid-slim-design`）：hybrid 双表示的 +34.5 GiB 不是技术必然，是集成层 `_w4a16_weight_layout_for_source()`（tp_moe.py:759-771）对 serving 恒返 `"packed"` 的**政策选择**；b12x 0.15.3 内置非破坏性 native 路径（`prepare_w4a16_e8m0_native_weights`，docstring 明言"A4 prefill + A16 decode 同进程共存"）——恰是 hybrid 场景。路线 a/c/d 被源码级否决（nibble gather 非 TMA 可表达 / out-of-place 零增益 / UMA 带宽算术 +124ms/步）。
- **实现**（`bprime-impl`）：三处规格改动 + 两处勘察新增（`_w13_layout` 覆写、**强制 B12X_W4A16_SMALL_M_DIRECT=0**——L1 GPU 实证 e8m0×micro direct 多几何 100% NaN，上游缺陷 is_supported 误放行），L1 CPU 68 + GPU 12 PASS，native 主 GEMM 与 packed 逐位相等。
- **窗口判决**（`bprime-window`）：内存门全兑现（45.32 GiB / KV 5.54M）但 **N1 门 FAIL**：DE C1 14.7（-25%）/ C12 61.8（-34%），归因 `_stage_b_tile_modelopt_native` 逐 tile 索引 staging vs packed 扁平 cp_async——mid-M 微基准给出结构性证据曲线（M=8 1.61× ~ M=3071 1.14×），**非调参可救**（MIN_M 梯子无效，decode 天然小 M）。**判决：b′ 不可用，退回 B2 形态评估**；b′ 保留为已验证设计储备（插件四节点保留 + .bak 一键重放），建议上游报 2 issue。

### 3.4 主题 3：LuZ0.3.1 生产落地（含重大发现）

`luz031-deployment` 是 08-23 的收官性生产变更，构成 = **W4A4 full（VLLM_MOE_W4A4=2）+ 池补丁（SHARED=1）+ FI 0.6.16 bind-mount + threshold 4096 + util 0.82**。验收全过（见成果表）。**过程中发现前序窗口遗留的 FI 0.6.16 误回滚**：w4a4-ext 收尾恢复误用 phase3b 时代 `.bak-wsdedupl3`（00:56 前快照）→ 03:01 起生产实际跑 0.6.15 混合树约 2.5 小时。处置：按 fi016 报告原样补回挂载行 + 二次重建；**b′ No-Go 判决稳健成立**（两树数值逐位一致 + 版本差 ±3% + b12x 树两版未补丁完全相同）。教训已入 runbook §E（跨窗口恢复核对 .bak 时序 + 启动核验加 flashinfer 版本项）。

补丁 6/7 落实（quality_gate.py 固化 / systemd 重启姿势 / healthcheck 探针超时 10s→30s / checker 用法修复 / **Prometheus 现场恢复（Exited 137 2 天）+ restart=always** / worker daemon-reload），E4（KV 足迹）留窗口 → 08-23 晚已由 e4-kv-footprint 专项完成（主题 6）。

### 3.5 主题 4：上游核对与性能上限

`upstream-check-perf-ceiling` 是 08-23 的"方向性"报告，把下一阶段工作从"移动拐点"重定向为"**填满甜点区 + 削减墙下流量**"：
- 上游面：b12x 1.2.6 后零提交，我们 6 项问题无一已修（PR #227 最接近，e8m0×micro 同族）；FI 0.6.17（08-11）最对症（NVFP4 W4A4 GB10 parity + 精度修复 + W4A16 小 batch TC decode + 共享专家融合）；vLLM #41834（DSV4 SM12x）携带 prefix-cache 竞态 + DSpark 采样器越界修复，建议对 fork 做正确性对账（U5）。
- 性能面：bf16 稠密池（37% 线性 FLOPs，~29-34µs/token）→ W4A4/routeB 全池转化推算 **PR +5~7%（2994→~3160-3210）**；三节点分层 = shared experts 首发（M=4096 恰在 routeB 350T 平台）+ lm_head（唯一同时作用于 decode 带宽墙，W4A4 权重读 262→~75MB/rank/步，decode 步时省 ~5-6%，质量门最高）+ attn 投影（仅 prefill 半场）。
- 拐点判定：W4A4 甜点 = 已捕获资产（扩大覆盖）；M_e 768（模型几何锁死，需 threshold 32768）与 decode 273GB/s（LPDDR5x 硬件墙）= 物理边界；**唯一值得投的移拐点方向 = routeB 338T 平台从 M=1024 下探到 512-768**。
- 路线图：P0 profiler 拆账 + FI 0.6.17 升级 / P1 shared experts + lm_head 立项 / P2 attn prefill 量化 + issue 素材 + fork 对账 / P3 attention A/B。

### 3.6 主题 5：环境 stall 调查与判决（模式方差 + AR stall 双线收束）

这是两天内**调查链条最长、反转最多**的一条线，最终在 08-23 晚以"单节点单核约束"判决收官：

```
模式方差（±8-13% 臂间）       环境级随机 AR stall（17-20ms/AR）
      │                               │
slowround H1 IRQ 风暴 ──下修──┐       │
slowround H2 pin 漏网 ──升头号─┼──> expD 复现+H2 实锤（4/7）但非主因
NCCL channel→NIC 映射 ──并列──┘       │
      │                               │
  实验 C（expverdict）排除 channel 映射  │
      │                               ▼
      │                    ringonly-w1：发现环境级随机 AR stall
      │                    envstall：GDR=0 实锤 + RC1 proxy 睡眠假说
      ▼                               ▼
   模式方差=重启级现象（卫生修复 R1-R3 落地，非根治）
                                    expverdict E1 判决：
                          ★根因=单节点 worker 单核约束（CPU 8/9=proxy 目标核）
                          RC1 字面定义修正（未实证，为下游症状）
                          → E3（cpuset A/B）挂起待处理
```

关键数据与状态见成果表。**注意表达修订**：`envstall-rootcause` 的 RC1（proxy 线程 hrtimer_nanosleep + ~1kHz）在 `expverdict-verdict` 中按字面定义未实证（无 hrtimer_nanosleep、存在 wchan=0 仍 STALL 的反例），修正为"单核约束的下游症状"；`ringonly-w1` 建议的"交换机 PFC/ECN 排查"在 `envstall` 中被撤销（直连无交换机 + NIC 侧全阴性）。

### 3.7 主题 6：E4 KV 足迹（指标语义修订）

`e4-kv-footprint` 修正了 `w4a4-ext` 的"+58%/token"表述：启动日志 "GPU KV cache size" 是 **max_concurrency × max_model_len**（准入容量），不是物理 KV 池 token 容量；物理 per-token 字节不变（≈9.4 KiB）。8192 档真实机制 = `max_in_flight_tokens = 2×max_num_batched_tokens` 使 SWA 组每请求预留 131→259 块 → **满长并发 9.16→5.80（-36.7%）**。**结论不变（threshold 8192 继续 No-Go）但否决理由需按本报告修订**；对短上下文（4K~64K 生产形态）并发容量几乎不受影响。E4 后无需为"每 token 足迹"做优化——它从未变大。

### 3.8 主题 7：FI 0.6.17 评估 + P0 拆账

`fi017-p0-accounting` 把 upstream-check 的 U1 从"候选"推进到"**Go（有条件）**"：wheel 元数据在线核验（py3-none-any、Python 3.12、CUDA 13.0、aarch64/GB10、CuTe-DSL 下限 4.5.2 与生产精确匹配）；前置条件 = 5 fork 补丁重放 diff + 58 fork 文件对账 moe_ep + 容器冒烟三门；E2 专项（W4A4 decode 中性度两口径悬案定论）随之列档。P0 拆账给出 P1 顺序建议（shared 首发 / lm_head 第二 / attn 第三），并强调**所有 [推算] 份额须在 P0 profiler 后替换为实测**（池总量 29-34µs 为 M=1024 口径，LuZ0.3.1 已是 M=4096 需重标定）。

### 3.9 主题 8：其他重要成果

支撑线成果详见成果表 8.1~8.18。要点：
- **根因/方法论层**：large-batch 的"调度器钳制万能钥匙"（8.2）、pr-de 的 Roofline 瀑布（8.3）、p1-p2 的四项研究（8.4）构成 08-22 上午的"挖根因"产出；**ar（8.5）与 mtp（8.6）两线同归 No-Go 并留下"busbw 21.4GB/s / n=7 最优"的定标常数**。
- **merged-GEMM 线**：08-21 晚的 e2e No-Go（8.12）→ 08-22 的 Triton 形式深析（8.9）与真实路由重采集（8.7）确认"尾部必须保留 B12X / 单步合并覆盖≈0"→ FreeToken P0 实验（8.10）给出"跨块 MoE 聚合（deferred MoE）"的有条件复活形态（w=4 M_e≥384 覆盖 68-81%）——但 08-23 的 budget 8192 实验（8.16）已把其前提（M 供给上探）证伪，"budget+threshold 协同上推单流 M"关闭。
- **运维/生产链**：fi-rebase-eugr-prep（8.11）→ windowA 冒烟（8.16）→ fi016-replacement（8.17）是 FI 0.6.16 落地的三步链；eugr-ab（8.18）在 08-23 全天窗口后段把 attention b12x 化正式关闭。

---

## 四、关联图谱说明（依赖 / 时序 / 证据链）

```
【08-22】                                                      【08-23】
large-batch（钳制定论）──> routing-recapture（2048 初判 -22.5%）──> p1-p2（混杂嫌疑）──> threshold-retest（翻案 +12%）
                                                                        │
                                        threshold-4096-adoption（采纳 +13.5%）◄──────────┘
                                               │
     ws-dedup-patch（补丁 L1/L2）──────────────┤
                                               ├──> wsdedup-l3-combo（W4A4 翻案 +8.3%）
     pr-de Roofline（AR 31% 假设）──> ar-opt（校准 13%，AR 线关闭）    ├──> w4a4-ext（并发 +13%、8192 否决）
     mtp-tuning（n=7 最优，线关闭）                                    │
     freetoken（deferred MoE 前提：M 供给上探）                        ▼
                                               ├──> a3-hybrid-slim（b′ 设计）──> bprime-impl（L1 全绿）──> bprime-window（N1 FAIL）
                                               │
                                               └──> luz031-deployment（LuZ0.3.1 落地 = B2+util 0.82）◄── 用户裁定 B2+util0.82
                                               └──> e4-kv-footprint（8192 否决理由修订）◄── w4a4-ext E4 遗留
                                               └──> upstream-check / fi017-p0（FI 0.6.17 候选 + 甜点区叙事 + P0 拆账）

【环境调查链】slowround（H2 升头号）──> expd-r123（复现+H2 非主因）──> ringonly-w1（发现随机 AR stall）──> envstall（GDR=0 + RC1 假说）──> expverdict（单核约束判决）
```

**关键依赖（供排期引用）**：
1. **W4A4 翻案 ← threshold 4096 采纳 + ws-dedup 池补丁**（缺任一，M2/M3 不会成立；`wsdedup-l3-combo` §0 明确"两天前的 threshold 采纳直接改写了 W4A4 的命运"）。
2. **b′ No-Go ← N1 门数据（DE step_eff 门 ≥19.1/90.5）**；`a3-hybrid-slim` §4.3 预设"native decode 回归 >3% → b′ 退化为设计储备"，`bprime-window` 正是按此止损线执行。
3. **LuZ0.3.1 ← B2（w4a4-ext）+ util 0.82（a3 路线 e）**；验收门 KV ≥5.7M 源自 B2 的 -8.9% 代价 + util 回补的合成预期（实测回补 +0.23M 低于合成 +0.44M，记录不阻断）。
4. **expverdict ← envstall RC1 假说**（判决实验 E1 直接对 RC1 取证并修正）；**实验 C ← slowround/expD 的 channel 映射并列候选**（判决排除）。
5. **FI 0.6.16 生产替换 ← windowA GPU 冒烟（5/5）**；**LuZ0.3.1 的 FI 组件 ← fi016-replacement**（误回滚发现由此而来）。
6. **budget 8192 关闭（windowA）→ deferred MoE 立项判断弱化**（freetoken P0 建议"先测 budget 上探"，结果 M 供给在 4096 见顶 → 该路线的价值前提 M_e 增益不成立）。
7. **e4-kv-footprint ← w4a4-ext §5.2 E4 遗留 + luz031 §8.1 遗留**（报告自述"前置遗留"）。
8. **fi017-p0 ← upstream-check-perf-ceiling**（U1 细化）；P0 拆账的池总量账 ← pr-de-bottleneck（M=1024 口径，需按 LuZ0.3.1 的 M=4096 重标定）。

---

## 五、修订说明（表述不一致处 / 口径更正清单）

| # | 位置 | 原文表述 | 修订后表述 | 依据 |
|---|------|---------|-----------|------|
| R1 | threshold 2048 A/B | `routing-recapture-scheduler-ab`：2048 使 PR 4K -22.5%，Route 1 判定否定 | **-22.5% 系重启级模式方差混杂；2048 3/3 正向、4096 +12%，扩 M 路线重开** | `threshold-retest` §4.1；`p1-p2-research` §1.1 先行发现同配置 26pp 反差 |
| R2 | AR 占比 | `pr-de-bottleneck`：AR 占步时 ~31%（127ms），fp8 AR 即 PR +18% | **AR 实际 ~13%（52.3ms/步）**（busbw 实测 21.4GB/s，非假设 8.8-10）；AR 全消除 PR 上限仅 +15%；fp8 AR 端到端 2.6× 结构性死 | `ar-optimization` §0 校准 1/校准 2 |
| R3 | benchmark 档位标签 | 报告沿用"4K/16K/32K/64K"标称 | **实际 prompt 为 8.2K/32.8K/65.5K/131K tokens**；PR 2510 是真实 GPU 吞吐（逐轮行口径），汇总行 1254 是 label 假数 | `pr-de-bottleneck` §2.3 |
| R4 | W4A4 生产结论 | P2（08-21）：W4A4 生产 No-Go（prefill -13%） | **P2 结论系 threshold 1024 时代形态误差；4096 下翻正 +6~8%（M2/M3/M4 三臂同 fast 模式互证）** | `wsdedup-l3-combo` §4；`w4a4-ext` §2.2 |
| R5 | KV 足迹 8192 | `w4a4-ext` §2.5/§5.2：每 token KV 足迹 9.0→14.2 KiB（+58%） | **指标语义误读：物理 per-token 不变（≈9.4 KiB）；真实机制 = max_in_flight 翻倍使 SWA 每请求预留 131→259 块 → 满长并发 9.16→5.80（-36.7%）**；8192 否决理由按此修订 | `e4-kv-footprint` §3 |
| R6 | 生产权重口径 | 任务书/prep：生产用 modelopt NVFP4 checkpoint | **生产 M1 实际加载官方 0731 checkpoint（FP8 block linear + MXFP4 experts，ue8m0 scales）**；Eugr 源码原生 dispatch 同款 → 双臂同权重公平 A/B 无需 confound | `eugr-ab` §1 |
| R7 | FI 0.6.16 挂载状态 | 各窗口启动核验无 flashinfer 版本项 | **03:01 起生产实际跑 0.6.15 混合树 ~2.5h（w4a4-ext 收尾误用旧 .bak）**；已修复 + runbook §E 补"启动核验加 flashinfer 版本项" | `luz031-deployment` §2 |
| R8 | v5 根因假说 | `ringonly-optimization-plan` §2：多通道 search 产生非物理环序 → 恒定重传 stall | **stock 8 通道环序本来就全是物理序；stall 根因不在逻辑环序层（node 级 rail/ringRecv/ringSend 分配或 v1b 交互层 + 环境 stall 混杂）** | `ringonly-w1` §3.1/§6 |
| R9 | envstall RC1 | `envstall-rootcause`：RC1 = NCCL proxy 线程 hrtimer_nanosleep + ~1kHz | **按字面定义未实证（无 hrtimer_nanosleep、wchan=0 仍 STALL 反例）；proxy timed-poll 为单核约束的下游症状**；根因 = 单节点 worker 单核约束（CPU 8/9） | `expverdict-verdict` §二 |
| R10 | 交换机 PFC/ECN 排查 | `ringonly-w1` §4.4 行动项 1：交换机侧 PFC/ECN/队列调度 | **撤销**：环网直连无交换机 + NIC 侧 PFC/pause/ECN 全零 | `envstall-rootcause` §2.4 |
| R11 | 插件路径 | wsdedup L3 任务书："插件路径挂载补丁即生效" | **不成立：W4A4 插件（plugin_a1）直接从 flashinfer.fused_moe 构造 wrapper 不经补丁池**；已做最小池集成（`_get_pooled_wrapper`），任务书另将 plugin_merged 与 plugin_a1 混淆 | `wsdedup-l3-combo` §3.1 |
| R12 | merged 覆盖率 | 合成语料口径：merged 单步覆盖率高 | **真实流量口径 3：hash ≥64=17.9%/dense≈0-2%（高估 3.7~8×）**；merged 单步收益上限极低，复活需跨步专家级聚合（deferred MoE） | `routing-recapture-scheduler-ab` §2.3；`freetoken-research` §P0 |
| R13 | 慢轮判定口径 | 以 tput 判慢轮 | **必须看 steps/s**（tput 方差可由 MTP 接受率解释，C1 tput 方差 36.6% 但 steps/s 恒定） | `slowround-rootcause` §3.3 |
| R14 | "4ch 免疫"历史认知 | 历史认知：4ch 不中 stall | **推翻：生产等价配置（nccl-tests 形态 4ch）同概率中招**（18:47-18:52 生产库 4ch 0/10 clean） | `ringonly-w1` §4.3 |

---

## 六、待办与下一步

### 6.1 待用户裁定

| # | 项 | 依据 | 优先级 |
|---|----|------|--------|
| T1 | **W4A4 生产采纳**：LuZ0.3.1 已是 B2 形态（已采纳）；深测/长期监控（decode 中性度两口径定论 E2） | `w4a4-ext` E1/E2；`luz031` §8.2 | P1 |
| T2 | **上游 issue 对外提交**：①e8m0×micro 数值缺陷（bprime-impl B.4）②native staging 小 M 性能（bprime-window §6 曲线）③GB10 48-SM attention 盲区补充（可选） | `fi017-p0` §1.5/§1.6；`upstream-check` §1.5 U3 | P2（素材已备） |
| T3 | **E3 环境判决是否立项**（cpuset 监控 / `--cpuset-cpus` A/B / `NCCL_IGNORE_CPU_AFFINITY` A/B） | `expverdict-verdict` §六 | P1 |
| T4 | **FI 0.6.17 窗口验证与 P0 profiler 实测排期**（U1 前置条件 = 5 补丁重放 diff + moe_ep 对账 + 三门） | `fi017-p0` §1.3/§1.4 | P0 |
| T5 | **Eugr 镜像处置**（34.5GB×4 + registry 副本保留为备选方案，用户 08-23 已裁定保留） | `eugr-ab` §4.2 A6；`2026-08-23.md` 08:42 决策 | P3 |

### 6.2 建议下一步（按叙事"填满甜点区 + 削减墙下流量"）

| # | 项 | 前置 | 预期 | 状态 |
|---|----|------|------|------|
| N1 | **P0 profiler 拆账 bf16 池三节点**（M=4096 当前生产形态，替换 [推算] 份额） | 无 | 锁定 P1 顺序 | 📌 半天级 |
| N2 | **U1：FI 0.6.16 → 0.6.17** | N1 之外的独立低风险项 | W4A4 decode 中性度定论（E2 关闭）+ W4A16 小 batch decode + 共享专家融合探路 | 📌 |
| N3 | **P1 shared experts → W4A4/routeB**（零拷贝适配器 + MIN_M 分档，M=4096 甜点） | P0 拆账确认份额 | PR +1.5~2.5%（推算） | 📌 首发 |
| N4 | **P1 lm_head W4A4 立项**（校准 + KL 门先行） | P0 | decode 步时 -5~6%（推算）+ prefill 增量 + 显存 -190MB/rank | 📌 质量门最高 |
| N5 | **routeB 平台 M 下探 512-768 预研**（唯一"移拐点"方向） | N1/N3 资产 | P3 的 decode 半场 + P4 的 M_e 96 受益 | 📌 自研 kernel 工作量级 |
| N6 | **E3 环境判决 + 模式方差残余跟踪**（cpuset A/B；R4 libncclpin v9 降级卫生项） | 窗口 | 关闭环境级随机 stall 与重启级模式方差两条线 | 📌 待用户裁定后执行 |
| N7 | **vLLM fork 对账 #41834 正确性修复**（prefix-cache 竞态 / DSpark 采样器越界 / eager scratch pool） | 无 | 生产稳健性（正确性保险） | 📌 U5 |

### 6.3 生产终态（截至 08-23 收官）

**LuZ0.3.1 全绿运行**：W4A4 full（VLLM_MOE_W4A4=2）+ 池补丁（SHARED=1）+ FI 0.6.16 bind-mount（git 8da13a29）+ threshold 4096 + util 0.82 + 自愈链三件套 + Prometheus 恢复（restart=always）+ 检查点/恢复镜像就位（`<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`、`<INSTALL_DIR>/backup/luz031-checkpoint-20260823/`）。回滚链：`.bak-luz031-20260823` + `restore_luz031.sh`（dry-run 演练通过）。

---

## 七、引用与数据位置说明

- 本报告引用的全部报告均位于 `deliverables/engineering-assurance/`（相对路径，如 `deliverables/engineering-assurance/threshold-retest-2026-08-22.md`）；三处目录型交付物为 `deliverables/engineering-assurance/ws-dedup-patch-2026-08-22/`、`deliverables/engineering-assurance/bprime-impl-2026-08-23/`、`deliverables/engineering-assurance/ringonly-v5-2026-08-23/`。
- 服务器原始数据位置（node01）：`/tmp/_thrst/`、`/tmp/_thr4096/`、`/tmp/_ws_dedup/`、`/tmp/_wsdedup_l3/`、`/tmp/_w4a4_ext/`、`/tmp/_bprime/`、`/tmp/_bprime_win/`、`/tmp/_luz031/`、`/tmp/_fi016/`、`/tmp/_wA/`、`/tmp/_eugr_ab/`、`/tmp/_ringopt/`、`/tmp/_slowround/`、`/tmp/_expverdict/`、`/tmp/_envstall_*.py`、`/tmp/_mtp_tune/`、`/tmp/_ar_opt/`、`/tmp/_routea_work/` 等。
- 本报告为纯只读文档产出，未触碰 GPU/集群；所有数字均取自 08-22/08-23 报告原文与 `2026-08-22.md`/`2026-08-23.md` 工作日志，未做虚构或外推（[推算] 类数字均保留原报告口径标注）。

---

*本阶段性总结由工程保障团队（技术文档师 tech-writer-1）汇编；关键生产决策（W4A4 采纳、FI 0.6.17 升级、E3 立项、上游 issue 提交）请由人类工程负责人复核。*
