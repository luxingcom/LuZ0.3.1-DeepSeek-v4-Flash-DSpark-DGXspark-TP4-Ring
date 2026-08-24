# b′ native hybrid 窗口测试报告（N1 门判决 + 全矩阵验证）

- **执行人**：雷克斯（Rex）· SRE 工程师（bprime-window）
- **日期**：2026-08-23（停机窗口 04:12-04:55 UTC）
- **对象**：b′ native 共享 hybrid（plugin_a1_bprime，bprime-impl 交付 + L1 全绿）
- **判决**：**N1 门 FAIL —— b′ 不可用（decode 回退 -25%/-34%，远超 -3% 门），退回 B2 形态采纳评估**；内存门/PR/并发/质量门全过（b′ 的内存与 prefill 收益真实存在，被 native 主 GEMM staging 代价否决）
- **生产终态**：已恢复基线全绿（FI 0.6.16 + W4A16 + threshold 4096 + overlay env=0 + plugin_a1 entry point 恢复 + b′ 插件文件保留 + env 三开关全关 + 自愈链恢复）

---

## 1. 执行摘要

b′ 的双表示零拷贝共享**在工程上完全成立**：hybrid weight 79.82 → **45.32 GiB**（实测，与 M3 full W4A4 完全一致），KV 1.53M → **5.54M tokens**（实测），质量门 golden 4/4 逐字一致，prefill 四档与并发聚合均在 M4/B2 的 -3%/±1% 内。**但 W4A16 侧切换到 native（modelopt）布局后，主 GEMM 的 `_stage_b_tile_modelopt_native` 逐 tile 索引 staging 代价使 decode 全 M 段步时劣化 45-74%**（微基准实测），端到端 DE step_eff C1 14.7（门 ≥19.1）/ C12 61.8（门 ≥90.5）——A3 报告的头号风险以最不利方式兑现。判决：**b′ 不可用**；其内存收益无法以可接受的 decode 代价兑现，退回 B2（full W4A4，已验证形态）采纳评估。

## 2. 部署记录（04:12-04:13 UTC，四节点）

| 项 | 结果 |
|---|---|
| 停自愈链 | healthcheck.timer + head + 3 worker 停止，残留容器四节点 0（phase3b 教训落实） |
| plugin_a1_bprime 部署 | 四机 scp，md5 全一致（__init__ 913e3ae8 / w4a4_experts 20b977fc / setup 3f9e5434，与本地交付物一致） |
| plugin_a1 entry point 禁用 | SERVE_CMD 前缀显式 `pip uninstall -y routea-plugin-a1`（容器态互斥，§B 部署警示落实；原生产脚本本无插件安装前缀，belt-and-suspenders） |
| start 脚本注入 | 四机 `.bak-bprime-20260823` 留档；SERVE_CMD 前缀安装 bprime + env `VLLM_MOE_W4A4=1` `VLLM_MOE_W4A4_NATIVE=1` `VLLM_MOE_W4A4_MIN_M=3072` `VLLM_MOE_W4A4_CG=1` + `VLLM_B12X_SHARED_WRAPPER 0→1` |
| checker | 四机 4/4 PASS（脚本路径参数） |
| 重启 | head-first 显式启动（head 先、2s 后三 worker），rendezvous/加载正常 |

## 3. 启动验证（b′ 生效标志 + 内存门）【实测】

**生效标志（rank0+rank1 双确认）**：
- `[routea_plugin_a1_bprime] installed: mode=1 native=1 min_m=3072`（APIServer/EngineCore 双进程）
- `b' native layout policy installed: fp4_e8m0_k32 -> 'modelopt'`（monkeypatch 生效）
- `B12X_W4A16_SMALL_M_DIRECT forced to 0`（e8m0×micro 缺陷防护，告警在）
- `b' native W4A16 prepared: weight_layout=modelopt w13_layout=w13 (payload shared with W4A4 wrapper)` 43 层全打
- `B12x shared wrapper pool: reusing wrapper (pool size=1)`（池生效）

**内存门【实测——b′ 核心收益兑现】**：

| 指标 | M4 hybrid（旧） | b′ 实测 | 门 | B1 | B2 |
|---|---|---|---|---|---|
| Model loading | 79.82 GiB | **45.32 GiB** | 45.3±1 ✓ | 40.5 | 45.32 |
| GPU KV cache | 1.53M tokens | **5,537,238** | ≥5.3M ✓ | 6.04M | 5.50M |

**其他启动项**：cudagraph 三档完整（PIECEWISE 16/16 + FULL 12/12 + dspark 11/11）｜threshold 4096 不动｜dspark n=7｜双探针过（stall 3×短4K TTFT 2.8-3.4s 全 <6s 零重签；首 4K TTFT 2.807s）。

## 4. N1 门判决（DE，核心判决点）【实测：FAIL】

step_eff = tput_sum 中位 / tokens_per_step 中位（接受率归一），各 4 轮，同口径复核 M1/B1/B2 基线文件一致：

| 臂 | C1 step_eff | C12 step_eff |
|---|---|---|
| M1 基线 | 19.7 | 93.3 |
| **N1 门（-3%）** | **≥19.1** | **≥90.5** |
| **N1 b′ 实测** | **14.7（-25%）** | **61.8（-34%）** |
| B1（W4A16 生产基线） | 17.7 | 87.2 |
| B2（full W4A4） | 18.3 | 85.1 |

**归因（劣化全部来自步时，非投机质量）**：N1 接受率不降反升（C12 tokens/step 4.385 vs B1 4.135；C1 4.971 vs 4.491）；C12 tput 270.9 vs 360.4（-25%）伴随接受长度 +6% → 步时 +34%。旁证：stall 探针 itl_med 67ms（b′）vs 51ms（恢复基线后同探针）= **+31% ITL**，与 step_eff 回退量级吻合。C1/C12 双段一致回退排除 cudagraph 缺失类单点故障。

## 5. 全矩阵（prefill/并发/质量）【实测】

**PR 四档（单流 3 轮中位，tok/s）**：

| 档 | M1 | B1 | M4 | B2 | **N1 b′** | vs M4 |
|---|---|---|---|---|---|---|
| 4K | 2753 | 2768 | 2999 | 2994 | **2920** | -2.6% ✓（门 -3%） |
| 16K | 2777 | 2770 | 2980 | 2972 | **2955** | -0.8% ✓ |
| 32K | 2674 | 2565 | 2852 | 2830 | **2846** | -0.2% ✓ |
| 64K | 2454 | 2215 | 2545 | 2541 | **2536** | -0.4% ✓ |

**并发聚合（4K，3 轮中位，tok/s）**：

| 并发 | B1 | B2 | **N1 b′** | vs B2 |
|---|---|---|---|---|
| C6 | 2744 | 3060 | **3028** | -1.0% ✓ |
| C12 | 2737 | 3092 | **3057** | -1.1% ✓ |

→ prefill 路径（M≥3072 走 W4A4 wrapper，与 M4/B2 相同）无恙，**回退干净隔离在 W4A16 native decode 段**——与 A3 "W4A4 侧零风险" 推导一致。

**质量门**：golden 4 稳定 prompt（fox_repeat/count/code/list）vs B1 参考 **4/4 逐字一致** ✓（reason/zh 为已除名不稳定 prompt——本窗口在恢复后的基线自身配置下复核，两 prompt 对自身参考亦漂移，确证除名合理）。needle 64K 抽验 2/3 pass（mid ✓、late 1 fail——与 wsdedup L3 窗口 `_needle.json` 同型 late-fail，属已知统计波动，A6 已降级 smoke 口径）。

## 6. mid-M 微基准（native vs packed 主 GEMM staging 代价曲线）【实测-GPU】

一次性容器直调 `run_w4a16_moe`（随机权重+路由，CUDA event 计时，中位；`B12X_W4A16_SMALL_M_DIRECT=0` 复刻生产 b′ 行为；生产并行运行中，共享 GPU 纪律——drop_caches 后 free 0.64→11.35 GiB 才可执行）：

| M | E=64/H=2048/I=512 packed→native ms | ratio | E=128/H=4096/I=512（生产同 N/K 形状）packed→native ms | ratio |
|---|---|---|---|---|
| 8 | 0.390→0.567 | **1.45×** | 0.721→1.161 | **1.61×** |
| 96 | 0.664→1.033 | **1.56×** | 2.033→3.527 | **1.74×** |
| 512 | 0.878→1.359 | **1.55×** | 2.401→3.831 | **1.60×** |
| 2048 | 1.991→2.227 | 1.12× | 4.432→5.305 | 1.20× |
| 3071 | 2.889→3.213 | 1.11× | 5.761→6.569 | 1.14× |

**曲线解读**：staging 代价在 M≤512 段 +45~74%（decode C1 M=8、C12 M≈96 恰在此段），M≥2048 收窄至 +11~20%（GEMM 计算开始摊薄逐 tile 索引开销）。生产几何（E=128/H=4096/I=512，与生产仅 E 不同）在 M=96 达 **+73.5%**——43 层 MoE 步时主导，端到端 C12 step_eff -34% 完全自洽。**结论：`_stage_b_tile_modelopt_native` 的逐 tile 索引 staging 在小 M 段结构性劣于 packed 的扁平 cp_async，非调参可救（MIN_M 梯子无效——decode 天然小 M）**。这正是上游 serving 弃 native 选 packed 的原因（A3 §2.3 翻案证据的另一面）。

## 7. Go/No-Go 与生产建议

- **N1 门：No-Go**（DE C1/C12 均 3 倍劣于容差）。b′ 的 KV/内存收益真实（45.32 GiB / 5.54M 全兑现），但 decode 代价不可接受。
- **生产建议**：
  1. **b′ 不采纳**；若需 hybrid 级 KV（5.5M）+ prefill 增益，**退回 B2（full W4A4）形态采纳评估**（已验证：PR 2994/3060/3092、DE 18.3/85.1、KV 5.50M、weight 45.32、golden 过）——decode 相对 M1 -7%/-9% 是其已知代价，由用户裁定。
  2. b′ 保留为**已验证的设计储备**：插件文件四节点保留（`<INSTALL_DIR>/nvfp4/plugin_a1_bprime/`，md5 20b977fc），env 三开关全关（脚本已回滚，重启用 `.bak-bprime-20260823` 可一键重放）。若未来 b12x 上游优化 native staging（或 micro direct 缺陷修复后 M≤8 段受益），窗口数据可直接复用。
  3. 建议向上游 b12x 报两个 issue：①e8m0×micro direct 数值缺陷（bprime-impl B.4）；②`_stage_b_tile_modelopt_native` 小 M 段 staging 性能（本报告 §6 曲线可作附件）。
- **util 0.82 加测臂：N/A**（仅 b′ Go 才有意义；N1 败后不适用）。

## 8. 回滚链与生产终态核验【实测】

回滚链（<10 分钟级，全链可用）：start 脚本 `.bak-bprime-20260823` 回滚（env 三开关随脚本回滚自然全关）→ checker 4/4 PASS → plugin_a1 entry point 恢复（回滚后脚本无插件安装前缀 = 生产原状）→ bprime 文件保留 → head-first 重启。

终态核验（04:43-04:55 UTC）：

| 项 | 结果 |
|---|---|
| Model loading | **40.5 GiB**（基线口径 ✓，bprime 标志 0 条） |
| GPU KV cache | **6,021,604 tokens**（≈B1 6.04M ✓） |
| 容器 | 四节点 vllm-tp4-rank0/1/2/3 全 healthy（16 min） |
| 服务/自愈 | vllm-tp4-head active + healthcheck.timer active + /health 200 |
| 双探针 | stall 3×TTFT 2.9-3.4s SUSPECT=False；首 4K TTFT 2.871s |
| golden | 4 稳定 prompt vs B1 参考 4/4 MATCH |
| checker | 四机 PASS；SHARED_WRAPPER=0；bprime 行 0 |

## 9. 证据索引（服务器 01）

| 项 | 位置 |
|---|---|
| 窗口日志与工件 | `/tmp/_bprime_win/`（deploy.log / restore.log / rank0_boot.log / rank0_baseline_boot.log / logs/*.json） |
| 测量 JSON 汇总 | `/tmp/_bprime_win/evidence/`（n1_de/panorama/conc6/conc12/needle/stall/probe + base_stall/probe + midm_E64/E128） |
| mid-M 微基准 | `/tmp/_bprime/midm_bench.py`、`midm_E64.json`、`midm_E128.json`（log：`/tmp/_bprime_win/logs/midm_E64.log`、`midm_E128.log`） |
| 基线参考 | `/tmp/_w4a4_ext/logs/`（B1/B2/B3）、`/tmp/_wsdedup_l3/logs/`（M1/M4、_de.json、_needle.json）、`/tmp/_fi016/` |
| 部署/恢复脚本 | `/tmp/_bprime_win/deploy_bprime.sh`、`patch_bprime.py`、`restore_baseline.sh`（本地副本：集群部署/tmp/） |
| 设计依据 | a3-hybrid-slim-design-2026-08-23.md §b′ 实施记录（B.1-B.7） |

---

*本报告由工程保障团队（SRE）生成；B2 形态采纳与否请由用户裁定。窗口全程未发生计划外事故，回滚链一次成功。*
