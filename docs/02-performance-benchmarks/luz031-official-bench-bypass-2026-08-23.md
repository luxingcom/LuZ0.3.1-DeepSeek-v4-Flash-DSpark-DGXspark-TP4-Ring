# LuZ0.3.1 新基座全量基准执行方案（官方包 + 克隆镜像旁路）

- **执行**：雷克斯（Rex）· SRE 工程师（sre-engineer-2）
- **日期**：2026-08-23
- **状态**：**方案与脚本已就绪，未执行、未启动任何容器、未触碰生产启动资产**；执行须先获督导确认
- **任务来源**：主理人（工程总监）下发 — 基于**官方基准包**重做 LuZ0.3.1 全量基准；用户明确 ① 基准**不切换生产容器**（克隆镜像+新容器名旁路）② 窗口启动前**必须**创建生产镜像检查点 + 整套容器备份 ③ 基准包 = 官方包
- **被测对象**：LuZ0.3.1 新基座（克隆镜像 `LuZ0.3.1-bench-20260823`，克隆容器 `tp4-bench-rank0..3`）
- **对比锚点**：官方包 `data/测试数据汇总.md` 的 8/19 原基座参考数据（decode-only 中位|最优）

---

## 0. 一页结论

1. **官方口径定锚**：decode-only（自首 token `t_first` 计时，`(completion_tokens-1)/(t_last-t_first)`，排除 prefill）；每流取中位、并发聚合 = 中位×C；预热 20-30 请求；轮数 ≥5（对齐官方 ⭐最优数据 5-10 轮口径）。
2. **旁路形态**：生产停机态（四机无 vLLM 生产容器）下，用 `docker tag` 克隆 LuZ0.3.1 → `LuZ0.3.1-bench-20260823`，以**克隆启动脚本**（对生产 start 脚本只做「镜像 tag + 容器名」两处 sed，其余 env/args/mounts 逐字保留）拉起 `tp4-bench-rank0..3`，**零冲突、零改动生产资产**。
3. **窗口前置三件事**（P0 拆账启动前必做，本次全量基准同样执行）：① 镜像检查点（LuZ0.3.1 + 基座 digest/tag 锚定 + 克隆 tag）② 整套容器备份（四机启动脚本/.bak + overlay 目录 + secrets 权限确认 + md5 清单 + restore）③ 停 `vllm-healthcheck.timer`（防自愈链把生产容器拉起与 bench 抢资源——**本窗口关键**）。
4. **矩阵**：M1 单流 decode-only 4 项（S1 fox p512 10 轮 / S2 fox p256 / S3 编号列表 / S4 Agent 工具）＋ M2 并发 C1/C4/C8/C12（≥5 轮）＋ M3 Agent 5 场景（Math/JSON/Code/Communication/Narrative，≥5 轮）＋ M4 可选端到端单流 4 项。
5. **P0 拆账（方案 B）**并入同一旁路窗口排期，可拆独立 session：LuZ0.3.1-base 一次性容器微基准，三池节点 bf16 GEMM，M=4096 生产形态 + M=8/96 decode 形态。
6. **预计时长**：全量基准（含前置备份+克隆启动+测量+收尾）约 **2h**；P0 拆账约 **1~1.5h**；同窗串联 **3.5~4.5h**（建议拆两个 session）。

---

## 1. 被测对象与对比锚点

### 1.1 被测对象：LuZ0.3.1（克隆形态）

| 项 | 值 | 依据 |
|---|---|---|
| 生产形态 | W4A4 full（`VLLM_MOE_W4A4=2`）+ 池补丁（`VLLM_B12X_SHARED_WRAPPER=1`）+ FI 0.6.16 bind-mount + threshold 4096 + util 0.82 + MTP dspark n=7 | luz031-deployment-2026-08-23 / LuZ0.3.1-release-notes |
| 自包含镜像 | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1`（digest sha256:85f2149f…） | 检查点已有 |
| 基座镜像 | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1-base`（=0.2.1-v026.0，digest sha256:e100ddad568a） | 检查点已有 |
| **克隆镜像 tag** | `…:LuZ0.3.1-bench-20260823`（`docker tag` LuZ0.3.1 创建，四机） | 本方案 |
| **克隆容器名** | `tp4-bench-rank0`(01)/`rank1`(02)/`rank2`(04)/`rank3`(03) | 本方案；与生产 `vllm-tp4-rank*` 零冲突 |
| API | `http://127.0.0.1:8001/v1/chat/completions`（生产 8001；生产停机态下空闲） | 本方案 |
| Key 来源 | `<INSTALL_DIR>/secrets/vllm.env`（`VLLM_API_KEY`） | 既有惯例 luz031_run.sh |

> **形态保真声明**：克隆启动脚本由生产 start 脚本 sed 生成，仅改动 ① 镜像 tag（→克隆 tag）② 容器名（→tp4-bench-*）；**serve 参数 / env 全集 / overlay 挂载 / 端口 / NCCL / cudagraph 逐字保留** → bench 即 LuZ0.3.1 生产形态的忠实复制。

### 1.2 对比锚点：官方 8/19 原基座参考（`_bench_pkg_official/.../data/测试数据汇总.md`）

> 官方参考环境：4×DGX Spark TP4，W4A16 时代原基座，max_num_seqs=16、max_num_batched_tokens=8240、util 0.82、MTP dspark n=5。**配置差异在对比表中逐项标注（seqs/batched/threshold/MTP/W4A4），Δ 为两形态综合差异，供业务裁定。**

| 指标 | 8/19 参考（中位 \| 最优） | 轮数口径 |
|---|---|---|
| C1 单流（fox p512 decode-only） | 97.1 \| 124.0 | 10 轮 |
| C4 聚合 | 218.0 \| 233.7 | 5 轮 |
| C8 聚合 | 286.3 \| 302.9 | 5 轮 |
| C12 聚合 | 342.8 \| 358.2 | 5 轮 |
| Agent 工具调用 decode | 105.8 \| 109.4 | 3 轮 |
| Agent Math | 93.2 \| 95.4 | 5 轮 |
| Agent JSON | 97.0 \| 101.1 | 5 轮 |
| Agent Code | 98.6 \| 102.4 | 5 轮 |
| Agent Communication | 67.1 \| 72.7 | 5 轮 |
| Agent Narrative | 50.7 \| 51.5 | 5 轮 |
| **Agent 平均** | **81.3 \| 84.6** | 各场景中位/最优均值 |

---

## 2. 官方基准矩阵（LuZ0.3.1 全量）

> 脚本一律用**官方包 scripts/**（`conc_decode_only.py` / `c1_10rounds.py` / `bench_decode_only.py` / `bench_code_agent.py` / `bench_3rounds_vs_tp2.py` 的测量函数与 prompt 逐字沿用），包内新增薄封装仅做：API/key 注入、轮数提到 ≥5、输出对齐官方 data/ 汇总格式。测量运行在 `--network host` 一次性客户端容器（基镜像，requests 就绪）内，不落盘生产主机 Python。

### M1 单流 decode-only 4 项（≥5 轮，S1 取 10 轮）

| 项 | prompt | gen | 源脚本 | 参考值 | 轮数 |
|---|---|---|---|---|---|
| S1 fox p512 | fox×60 | 512 | c1_10rounds.py | 97.1 \| 124.0 | **10** |
| S2 fox p256 | fox×30 | 512 | bench_decode_only.py | （补充项） | 5 |
| S3 编号列表（MTP 高接受率规律文本） | 官方 LIST | 512 | bench_decode_only.py | （补充项） | 5 |
| S4 Agent 工具调用 | 官方 AGENT prompt | 310 | bench_code_agent.py | 105.8 \| 109.4 | 5 |

### M2 并发聚合 decode-only（C1/C4/C8/C12，≥5 轮）

| 项 | 源脚本 | 口径 | 参考值 | 轮数 |
|---|---|---|---|---|
| C1 | conc_decode_only.py | 每流中位×1 | 97.1 \| 124.0 | 5 |
| C4 | conc_decode_only.py | 每流中位×4 | 218.0 \| 233.7 | 5 |
| C8 | conc_decode_only.py | 每流中位×8 | 286.3 \| 302.9 | 5 |
| C12 | conc_decode_only.py | 每流中位×12 | 342.8 \| 358.2 | 5 |

> 可选项：`bench_c1to8.py` 全 C1-C8 扫（对齐官方 §2 并发聚合表，+5-10min）。因 LuZ0.3.1 util 0.82 高并发资源受限，C12 仍按官方口径必测；若 `max_num_seqs` 小于并发数导致排队，记录实况并如实标注（官方参考同样标注）。

### M3 Agent 5 场景 decode-only（≥5 轮）

| 场景 | 源脚本 prompt（bench_3rounds_vs_tp2.py 原文） | gen | 参考值 | 轮数 |
|---|---|---|---|---|
| Math | MATH 官方 prompt | 512 | 93.2 \| 95.4 | 5 |
| JSON | JSON 官方 prompt | 512 | 97.0 \| 101.1 | 5 |
| Code | CODE 官方 prompt | 512 | 98.6 \| 102.4 | 5 |
| Communication | COMM 官方 prompt | 512 | 67.1 \| 72.7 | 5 |
| Narrative | NARR 官方 prompt | 512 | 50.7 \| 51.5 | 5 |
| **平均** | — | — | **81.3 \| 84.6** | — |

> ⚠️ 口径说明：官方包 `bench_3rounds_vs_tp2.py` 的 Agent 段使用端到端 `ct/el`；官方 ⭐最优 Agent 数字（93.2 等）来自 decode-only 客户端。本矩阵按官方 ⭐最优口径（decode-only）重跑 Agent 5 场景，prompt 逐字沿用官方文件。

### M4 可选：端到端单流 4 项（官方 bench_3rounds_vs_tp2.py 单流段）

| 项 | prompt | gen | 口径 |
|---|---|---|---|
| p256/g64、p256/g256、p512/g64、p512/g256 | fox×30 / fox×60 | 64 / 256 | 端到端 ct/el（含 prefill） |

> 纯附加参考项（对齐官方包步骤⑤），**非 decode-only**，报告中单独标注，不与 decode-only 数据混排。

### 预热步骤（官方口径）

1. **全局预热**：`bench_luz031_warmup.py` 发 24 个 fox p512 stream 请求（对齐官方「20-30 请求 + 编译收敛」）。
2. **块级去冷启动**：M1/M2/M3 每个块首轮计入轮数（官方 ⭐最优数据即含首轮），全局预热后首轮已收敛。
3. 预热后判定：C8 首轮 ≥ 官方「预热前 242.6」即视为收敛完成（官方 8/19 教训）。

### 数据落点格式（对齐官方 data/ 汇总）

| 层 | 格式 | 文件 |
|---|---|---|
| 原始日志 | `bench_20260819_regression.log` 风格（环境头 + 每项轮次序列 + 中位\|最优） | `logs/bench_luz031_regression_<UTC>.log` |
| 结构化数据 | 每块 JSON（M1/M2/M3 逐轮 + 中位\|最优） | `logs/luz031_m1_single.json` / `m2_conc.json` / `m3_agent.json` |
| 汇总 | `测试数据汇总.md` 风格（§0.1/§0.2/§0.3 同构表） | `logs/luz031_汇总_<UTC>.md` |
| 对比 | 新基座 vs 8/19 原基座 逐项 Δ 表（§5 模板） | `logs/luz031_vs_official_<UTC>.md` |
| 本地副本 | 窗口结束后 scp 回 | `deliverables/engineering-assurance/_luz031_official_bench/data/` |

---

## 3. 克隆镜像旁路窗口流程

> 全程执行顺序：**督导确认 → 前置勘察 → 镜像检查点+克隆 tag → 整套容器备份 → 停 timer → bench 克隆启动 → 测量 → bench 停止 → 数据收集/对比 → 恢复检查点**。任何 `docker run` 前必须有督导明确批准（脚本内 `APPROVED=launch` 门禁）。

### 3.1 前置（~5 min）

- [ ] 督导批准窗口（本方案 v1.0 已送审）。
- [ ] 四机 `docker ps` 确认**无任何 vllm-tp4-* / tp4-bench-* 容器**（生产停机态基线）。
- [ ] 记录 `docker ps -a`、`nvidia-smi` 基线、8001 端口占用（`ss -ltnp | grep 8001`）。
- [ ] 确认 `systemctl is-active vllm-healthcheck.timer` 并**停用**（防自愈链拉起生产容器与 bench 抢 GPU/端口）。服务 vllm-tp4-head/worker.service 保持原态（生产停机态下通常 inactive，不主动启停）。

### 3.2 镜像检查点（P0 启动前必做；全量基准同样执行，~5-10 min）

由 `bench_preflight_backup.sh --image-checkpoint` 执行：

| # | 动作 | 说明 |
|---|---|---|
| C1 | 四机 `docker image inspect` LuZ0.3.1 / LuZ0.3.1-base，记录 **digest + tag + created + size** | 检查点锚定 |
| C2 | 四机 `docker images --digests` 快照 | 佐证各机一致性（四机 md5/digest 一致） |
| C3 | 四机 `docker tag LuZ0.3.1 LuZ0.3.1-bench-20260823` | **克隆镜像**（纯 tag 别名，非复制，秒级） |
| C4 | 校验四机克隆 tag digest == LuZ0.3.1 digest（sha256:85f2149f…） | 防错 tag |
| C5 | 可选 `docker save LuZ0.3.1-base` 到检查点（基座 ~10-15GB，磁盘允许时） | LuZ0.3.1 34.4GB 默认不 save（已在 registry + 四机） |

> 输出：`checkpoint/images_anchors_<UTC>.txt` + 四机 digest 一致性校验记录。**此步不启动任何容器。**

### 3.3 整套容器备份（P0 启动前必做；全量基准同样执行，~15-25 min）

由 `bench_preflight_backup.sh --container-backup` 执行。**备份模型**：每节点在**本机** `$BENCH_CP`（`<INSTALL_DIR>/backup/luz031-bench-checkpoint-20260823/`）建立本地检查点（备份本机启动资产/overlay），head 汇总各机 md5 清单做四机一致性核验；restore 由各机本地 `restore_bench_assets.sh` 执行（含 `--dry-run`）。

| # | 资产 | 动作 | 校验 |
|---|---|---|---|
| B1 | 四机启动脚本 + .bak | 01: `start_tp4_head.sh` + `*.bak-*`；02/03/04: `start_tp4_worker.sh` + `*.bak-*`；全部: `check_vllm_script.sh` + `*.bak-*` | 拷贝后四机 md5 比对（一致性） |
| B2 | overlay 目录 `<INSTALL_DIR>/nvfp4/` | tar.gz（plugin_a1 树 + flashinfer-0.6.16 树） | md5 + tar -tzf 清单 |
| B3 | overlay 目录 `<INSTALL_DIR>/overlay-wsdedup/` | tar.gz（`flashinfer_b12x_moe.py`） | md5 |
| B4 | secrets `vllm.env` | **权限确认**（mode/owner/group，应为 600 root 族）+ 记录 md5（**不落明文到检查点**） | 权限+md5 |
| B5 | 状态佐证 | 四机 `docker ps -a` 快照 + 镜像清单 + 端口快照 | 与前置一致 |
| B6 | md5 清单 | `md5-manifest.txt` 全资产 | `md5sum -c` 通过 |
| B7 | restore 脚本 | 生成 `restore_bench_assets.sh`（含 `--dry-run`） | 语法 + dry-run 演练 |

> **restore 步骤（恢复到备份时态）**：`cd <checkpoint> && md5sum -c md5-manifest.txt` → 分发 start 脚本/.bak/overlay → 恢复 secrets 权限 → 四机 md5 复核 → `check_vllm_script.sh` 四机 PASS → （如需生产）按 runbook §E.1 head-first 重建。**跨窗口恢复必须核对 .bak 时序（luz031 §2 教训），重建后核 flashinfer 版本项。**

### 3.4 bench 克隆集群启动（~10-16 min 冷启动）

由 `bench_clone_start.sh`（需 `APPROVED=launch`）执行：

| 步 | 动作 |
|---|---|
| S1 | 前置门禁：确认无 vllm-tp4-* 运行、克隆 tag 四机在位、8001 空闲、APPROVED=launch |
| S2 | **生成克隆启动脚本**（node01，读生产脚本→写 /tmp/_bench_luz031/scripts/）：`cp` 各机 start 脚本 → `sed`：`vllm-tp4-rank`→`tp4-bench-rank`、`:0.2.1-v026.0`→`:LuZ0.3.1-bench-20260823`（IMG 行）；**其余逐字保留** |
| S3 | **差异核验**：`diff` 生产 vs 克隆，**只允许上述两类差异**，否则 abort（保真门） |
| S4 | 克隆脚本自检：`check_vllm_script.sh <克隆脚本>` 四机（内容未变应 PASS，若 FAIL 则停，不启动容器） |
| S5 | **head-first 启动**：01 `NODE_RANK=0 VLLM_HOST_IP=<NODE_IP> bash start_tp4_head.bench.sh` → 02/04/03 依次 worker（rank1/2/3） |
| S6 | 轮询就绪：`docker logs tp4-bench-rank0 | grep "Application startup complete"`（≤16min，后台+轮询） |
| S7 | 启动核验：四容器 healthy + env（W4A4=2/SHARED=1）+ **flashinfer 0.6.16** + threshold 4096 + util 0.82 + MTP n=7 + `/health` 200 |

> 容器由克隆脚本内部 `docker run -d --name tp4-bench-rankN --restart no --network host ...` 创建，**不注册 systemd、不触碰生产服务**；停止即 `docker rm -f`，不遗留。

### 3.5 基准测量执行（~45-60 min）

`bench_luz031_full_run.sh` 按 §2 矩阵执行（预热 → M1 → M2 → M3 → M4 可选），数据落 `logs/`，结束自动调 `bench_luz031_compare.py` 出 Δ 表。

### 3.6 bench 停止与清理（~5 min）

`bench_clone_stop.sh`：四机 `docker rm -f tp4-bench-rank0..3`（`--restart no`，删除即不复活）；确认无残留；**不删除克隆 tag / 不删克隆启动脚本**（留档供复核）。

### 3.7 恢复与自愈链（~5 min）

- [ ] 若窗口前置停用了 `vllm-healthcheck.timer` → **恢复启用**（基准纪律，勿忘）。
- [ ] 确认生产 vllm-tp4-* 容器仍未启动（维持停机态基线，除非督导另有指示）。
- [ ] 自愈链三件套状态记录（head.service / worker.service / healthcheck.timer）。
- [ ] 数据 scp 回本地 `_luz031_official_bench/data/`。

---

## 4. P0 拆账（方案 B）并入旁路窗口排期

- **方案 B（fi017 §2.5 B）**：`LuZ0.3.1-base` 镜像一次性容器微基准，三池节点 bf16 GEMM，**M=4096 生产 prefill 形态** + M=8/96 decode 形态，产出 µs/token 与 ms/step。
- **三节点含义**：三**池节点**（shared experts / lm_head / attn 投影），非三台物理机；脚本 `p0_dense_pool_microbench.py` 在单节点（node01，主）+ 可选交叉（node01）各跑一次，取跨节点中位增强稳健性。
- **执行**：`run_p0_profiler_b.sh`（`APPROVED=launch` 门禁），一次性容器 `--gpus all` `<1GB 显存共享纪律`，即测即删。
- **排期**：同旁路窗口、独立 session（B）。**前置条件**：镜像检查点 + 整套容器备份已完成；bench 克隆集群已停止（GPU 空闲）；生产仍停机。
- **产出**：`p0/p0_micro_M4096.json` → 填入 `p0_accounting_data_template.md` → 与 fi017 推算带（attn 15-19 / shared 9-12 / lm_head 3-5µs）对比，偏差 >±30% 才动 P1 顺序。

| 池节点 | 几何（per-rank, TP4） | ×层 | 测量 |
|---|---|---|---|
| shared experts | [M×4096]×[4096×512]×2 + [M×512]×[512×4096] | 43 | µs/token（M=4096）+ ms/step（M=8/96） |
| lm_head | [M×4096]×[4096×32320] | 1 | 同上 |
| attn 投影（代理） | [M×4096]×[4096×4096] FLOPs 缩放 ×32.2 | 43 等效 | 同上 |

---

## 5. 对比表模板（LuZ0.3.1 新基座 vs 官方 8/19 原基座）

> 由 `bench_luz031_compare.py` 自动生成；空模板如下，判定阈值：Δ中位 ≥0 提升 ✅ / -5%~0 持平 ⚠ / <-5% 回退 🔴（decode 为 W4A4 full 已知代价带，回退带参考 luz031：phase3b -6~-9%，w4a4-ext ±3%）。

### 5.1 并发聚合（decode-only，t/s）

| 并发 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| C1 | 97.1 \| 124.0 | _ | _ | _ | _ |
| C4 | 218.0 \| 233.7 | _ | _ | _ | _ |
| C8 | 286.3 \| 302.9 | _ | _ | _ | _ |
| C12 | 342.8 \| 358.2 | _ | _ | _ | _ |

### 5.2 Agent 场景（decode-only，t/s）

| 场景 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| Math | 93.2 \| 95.4 | _ | _ | _ | _ |
| JSON | 97.0 \| 101.1 | _ | _ | _ | _ |
| Code | 98.6 \| 102.4 | _ | _ | _ | _ |
| Communication | 67.1 \| 72.7 | _ | _ | _ | _ |
| Narrative | 50.7 \| 51.5 | _ | _ | _ | _ |
| **平均** | **81.3 \| 84.6** | _ | _ | _ | _ |

### 5.3 单流（decode-only，t/s）

| 项 | 官方 8/19（中位\|最优） | LuZ0.3.1（中位\|最优） | Δ中位 | Δ最优 | 判定 |
|---|---|---|---|---|---|
| fox p512（10 轮） | 97.1 \| 124.0 | _ | _ | _ | _ |
| Agent 工具调用 | 105.8 \| 109.4 | _ | _ | _ | _ |

### 5.4 服务形态差异标注（对比时必须并列呈现）

| 维度 | 官方 8/19 原基座 | LuZ0.3.1 新基座 |
|---|---|---|
| MoE 量化 | W4A16 时代原基座 | W4A4 full（+ 池补丁 SHARED=1） |
| max_num_seqs | 16 | 克隆生产脚本现值（以克隆启动脚本为准） |
| max_num_batched_tokens | 8240 | 4096（threshold 4096） |
| MTP 投机 | dspark n=5 | dspark n=7 |
| FI | 参考环境版本 | 0.6.16 |
| util | 0.82 | 0.82 |

> 结论书写原则：Δ 为两形态综合差异（模型/运行时 + serving 参数）；单项回退须结合 W4A4 decode 已知代价带解释，不单独归因。

---

## 6. 窗口排期与预计时长

| Session | 阶段 | 内容 | 预计 |
|---|---|---|---|
| **A：全量基准** | A0 | 督导确认 + 前置勘察 + 停 timer | 5 min |
| | A1 | 镜像检查点 + 克隆 tag（3.2） | 5-10 min |
| | A2 | 整套容器备份 + md5 + restore 演练（3.3） | 15-25 min |
| | A3 | bench 克隆启动 + 就绪 + 启动核验（3.4） | 10-16 min |
| | A4 | 预热 + 全量测量 M1/M2/M3(+M4)（3.5） | 45-60 min |
| | A5 | bench 停止 + 数据收集 + 对比报告 | 15-20 min |
| | A6 | 恢复 timer + 自愈链核验 | 5 min |
| | **A 合计** | | **≈ 1.5-2 h** |
| **B：P0 拆账** | B0 | 前置（检查点/备份已完成、bench 已停、GPU 空闲） | 5 min |
| | B1 | 方案 B 微基准（三池节点 GEMM，01 主 + 02 交叉） | 45-60 min |
| | B2 | 数据填模板 + 对比 fi017 推算 | 15-20 min |
| | B3 | 清理一次性容器 + 报告 | 5 min |
| | **B 合计** | | **≈ 1-1.5 h** |

**总时长**：同窗串联 **3.5-4.5 h**；建议 **A / B 拆两个 session**（A 主、B 独立），任一 session 均可独立排期，不互相阻塞。

---

## 7. 纪律与红线

1. **不启动任何容器，直到督导明确批准**（脚本 `APPROVED` 门禁）。
2. **不触碰生产启动资产**：不改/不删/不覆盖 `start_tp4_*.sh`、`check_vllm_script.sh`、`.bak-*`、overlay、secrets；克隆脚本一律放 `/tmp/_bench_luz031/scripts/`。
3. **四机一致性**：所有分发资产 md5 比对；克隆 tag digest 四机一致。
4. **长命令后台 + 轮询**；测量期无并行构建/部署/生产启动。
5. **基准纪律（runbook §F）**：测量前停 healthcheck.timer、结束恢复；≥5 轮中位；唯一 nonce（官方脚本固定 prompt 有 prefix cache 风险，测量值以 decode-only 时序为准，官方同口径）；greedy（temp=0）；标注轮数/预热/文本/并发口径。
6. **跨窗口恢复核对 .bak 时序 + 重建后核 flashinfer 版本**（luz031 §2 教训）。
7. 生产停机态保持：本窗口结束后生产 vllm-tp4-* 仍不启动，除非督导指示。
8. secrets vllm.env 不落明文到任何交付物。

---

## 8. 交付文件清单

| 文件 | 用途 |
|---|---|
| `luz031-official-bench-bypass-2026-08-23.md`（本文档） | 方案总纲 |
| `_luz031_official_bench/README.md` | 窗口 runbook（部署→测量→对比→恢复） |
| `_luz031_official_bench/bench_luz031_config.env` | 配置（API/key/轮数/镜像 tag） |
| `_luz031_official_bench/bench_preflight_backup.sh` | 镜像检查点 + 整套容器备份 + md5 + restore 生成 |
| `_luz031_official_bench/bench_clone_start.sh` | 克隆启动脚本生成 + tp4-bench 四机启动 + 启动核验 |
| `_luz031_official_bench/bench_clone_stop.sh` | bench 停止清理 |
| `_luz031_official_bench/restore_bench_assets.sh` | 备份资产 restore（含 --dry-run） |
| `_luz031_official_bench/bench_luz031_full_run.sh` | 官方矩阵测量编排（预热/M1/M2/M3/M4） |
| `_luz031_official_bench/bench_luz031_warmup.py` | 预热 24 请求 |
| `_luz031_official_bench/bench_luz031_single4.py` | M1 单流 decode-only 4 项 |
| `_luz031_official_bench/bench_luz031_conc.py` | M2 并发 C1/C4/C8/C12 |
| `_luz031_official_bench/bench_luz031_agent5.py` | M3 Agent 5 场景 |
| `_luz031_official_bench/bench_luz031_compare.py` | 新基座 vs 官方 8/19 Δ 表 |
| `_luz031_official_bench/run_p0_profiler_b.sh` | P0 方案 B 执行封装 |
| `_luz031_official_bench/p0_dense_pool_microbench.py` | P0 三池节点 GEMM 微基准 |
| `_luz031_official_bench/p0_accounting_data_template.md` | P0 拆账数据表 |
| `_luz031_official_bench/data/` | 结果落点 |

---

## 9. 风险与回滚

| 风险 | 概率/影响 | 处置 |
|---|---|---|
| 克隆启动与生产不一致（sed 误伤） | 低/高 | S3 差异核验（仅允许两类差异）+ 自检 PASS 门禁，不符 abort |
| 自愈链拉起生产容器与 bench 抢资源 | 中/高 | 3.1 前置强制停 timer；3.7 恢复 |
| W4A4 decode 回退（已知代价带） | 高（预期内） | 对比表并列形态差异 + 判定带，不阻断 |
| 高并发 C12 排队（seqs 限制） | 中/低 | 如实记录实况，标注官方同口径 |
| 备份损坏 | 低/高 | md5 清单 + restore --dry-run 演练 + 与既有 luz031-checkpoint 双保险 |
| 端口 8001 被占 | 低/中 | 前置确认；若被占即停（生产停机态预期空闲） |

**回滚锚点**：LuZ0.3.1 自包含恢复镜像（85f2149f…）+ `restore_luz031.sh`（既有检查点）；本次新增 `luz031-bench-checkpoint-20260823/` 容器备份 + `restore_bench_assets.sh`；W4A16 基线回退 = 四机 `.bak-luz031-20260823` + head-first 重建。

---

*纪律遵守：方案与脚本就绪，未启动容器、未触碰生产启动资产；执行前送督导批准；任何 docker run 前先确认；四机 md5 一致；长命令后台+轮询；跨窗口恢复核对 .bak 时序。*
