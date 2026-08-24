# ringonly 定制 NCCL 算子优化方案（小消息 DE + threshold 4096 大消息 PR）

**执行**：ringonly-optimizer（Rex · SRE 工程师，工程保障团队）
**日期**：2026-08-22（UTC）
**任务来源**：用户指示——"ringonly 是我定制的算子，资料库内有开发文档，针对性对小消息优化可看服务器内存档资料；若能适配更大 threshold 并进一步优化延迟改善 PR 和 DE 速度，请给出方案。"
**口径标注**：【实测】= 服务器日志/微基准/RDMA 计数器直接测量；【源码】= NCCL 2.30.7 源码（hardened 归档）精读；【推断】= 实测×模型推导。全部关键结论按此三档标注。

---

## 0. 一页结论

1. **现状补丁集全貌已还原**（§1）：生产库 = 官方 NCCL 2.30.7 干净重建 + 4 个补丁（v1 环邻过滤 / v4 per-peer 双 dev 硬编码 / stageB per-size LL-Simple tuner / hardened 双分支加固），文件与函数级清单见 §1.2。
2. **8/16 通道灾难劣化机制已定位到候选根因**（§3）：所有消息尺寸恒定 ~17-20ms/AR（196KB 也是 17ms）——固定 stall 特征【实测】。源码层面 NCCL 2.30 的 rail 交替 + 多环 search 通道在 >4 通道时会产生非物理环序，v4 补丁的 `{0,0}` 哨兵使这些通道落到自动选 dev（无物理路径）→ 恒定重传【源码+特征推断，待窗口取证】。
3. **大消息方向（PR，配合 threshold 4096）是本方案主推**：修复通道映射（v5 环序强制补丁）解锁 8/16 通道，预期 busbw 21.4 → 25-28GB/s（+17-30%）【推断，依据 v3+MAXCH16 历史大档 -3~-70% 与逻辑口利用率分析】，33.5MB AR 时间 -15~24%，PR 在 threshold 4096 的 +12% 之上再叠加 +2-3%。
4. **小消息方向（DE）NCCL 层已基本到实证最优**：协议（40KB 交叉点）、通道数（4ch）、buffsize 均在实测最优点；理论残余空间（one-shot 类 2× 延迟）已有 2-hop 项目 S3 终审否定（新原语 5-10 人日、成功率 40-60%、价值上限 3-8% TPOT），不建议重开。
5. **"2048 臂流量 +10%" 之谜已闭环**：threshold-retest §3.1 全矩阵验证为采样窗口噪声（A3/B3/C1 流量区间完全重叠），无需处理【实测】。
6. **意外事件**：本任务尝试共享 GPU 微基准失败（CUDA 上下文 OOM）且与一次生产全集群自愈重启（14:04 UTC）时间相关，已停止测试、生产 14:13 完整恢复、教训与规矩见 §6。**全部基准验证转入需窗口清单**（§7），工具链已备好。

---

## 1. 现状：补丁集全貌（资料消化产出）

### 1.1 生产链与构建方式【源码+实测】

| 项 | 值 | 证据 |
|---|---|---|
| 生产库 | `/opt/nccl-ringonly/libnccl.so.2.30.7`，md5 `2be94172c1172734d00dee9ff7d788bd` | 8/22 md5sum 实测 |
| 源码基线 | 官方 NCCL 2.30.7（git HEAD 0dd44cd，分支 nccl-2307-hardened） | MD5-RECORD.txt |
| 归档位置 | `<INSTALL_DIR>/backup/nccl-official-2307-hardened-20260816/`（源码+patches+构建记录） | 8/22 ls 实测 |
| 构建方式 | 生产同源镜像 `anemll/dspark-vllm-gx10:0.2.1-v026.0` 容器内 `make -j src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`；**GLIBC 约束：产物 GLIBC_MAX ≤2.34**（生产镜像 glibc 2.35 才能 LD_PRELOAD）；编译前必须 `rm -rf build`（残留产物触发 nvlink 133 vs 130 崩溃） | 归档 README.md |
| 加载方式 | `LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2`（libncclpin=v8 CPU 钉核 shim：NCCL 线程→8-9，EngineCore→15-19） | B1 基线 §1.3 |

**版本史（/opt/nccl-ringonly/ 备份链）**：
| 库 | md5 | 说明 |
|---|---|---|
| .bak-v2 | 4cc43e3b | GLM-5.2 社区链 v2（v1 过滤 + NCCL_IB_PEER_HCA env per-peer 对口） |
| .bak-stageB / .bak-stageB-prod | b7784b49 | v3（PEER_HCA + chan 级日志），**曾支持 MAXCH16 并 E2E 获益**（8/15：PR@131K +21.7%） |
| .bak-hardened | 3d9cf539 | Stage B glibcfix（GLIBC 修复重编） |
| **现役** | **2be94172** | Stage B hardened（双分支加固），官方干净重建 + 4 补丁，**弃 PEER_HCA env 改 v4 硬编码** |

### 1.2 现役补丁清单（文件/函数级）【源码实证，patch 全文已读】

| # | 补丁 | 文件 / 函数 | 改动内容 | 作用 |
|---|---|---|---|---|
| 1 | v1-ring-only | `src/transport.cc` / `ncclTransportP2pConnect()` | 取 `channel->ring.prev/next`，跨机（hostHash 不同）且非本 channel 环前驱/后继的 peer 直接 `continue`（recv 循环 + send 循环各一处） | **解决 stock NCCL 在无交换机环网上必然失败**：NCCL 拓扑模型假设交换机全互联，对非直连 rank 建 QP → `ibv_modify_qp` errno 110 → init 崩溃（NVIDIA 论坛 GLM-5.2 帖独立复现此现象） |
| 2 | v4-netdev-hardcode | `src/transport/net.cc` / `ncclRingDevOverride()`（新静态函数 + sendSetup/recvSetup 各 1 行调用） | 4×4×2 硬编码 per-peer 双 dev 映射（环 01(0)-02(1)-04(2)-03(3)）：rank0→1={1,3}、0→3={0,2}、1→2={0,2}、2→3={1,3}（对称）；**even chan→pair[0]，odd chan→pair[1]**；非环邻对 `{0,0}` 哨兵→不改写（留给 NCCL 自动选 dev） | 环上同一 channel 的收发 peer 在不同环边，各需不同本地口；硬编码保证每 peer QP 落到物理对口 NIC |
| 3 | stageB-tuner | `src/enqueue.cc` / `ncclGetAlgoInfo()` else 分支 | allreduce ≤40KB→强制 LL、>40KB→强制 Simple（cost table 其他 proto 置 1e18，跳过 -1 IGNORE）；`NCCL_TUNER_THRESHOLD` env 可覆盖（默认 40960）；仅干预 allreduce，不碰 nChannels | decode 小 AR 走 LL（-9~-34%）、prefill 大 AR 走 Simple；40KB 交叉点有 8/16 全尺寸扫描实证 |
| 4 | stageB-hardened | `src/enqueue.cc` / `ncclPersizeTunerOverride()`（抽出为静态函数） | 覆盖逻辑在 if（SPCX tuner 在场）与 else（tuner==NULL）双分支均生效 | 消除对 `NCCL_NET_PLUGIN=none` 的隐式依赖（防外部 tuner 劫持使 per-size 失效——bench-SB 非单调事故的教训） |

### 1.3 生产 NCCL env（B1 v2.0 定版，8/22 重启后核验零漂移）【实测】

```bash
LD_PRELOAD=/opt/libncclpin.so /opt/nccl-ringonly/libnccl.so.2
NCCL_ALGO=RING  NCCL_BUFFSIZE=8388608  NCCL_MIN_NCHANNELS=4  NCCL_MAX_NCHANNELS=4
NCCL_TUNER_THRESHOLD=40960  NCCL_NET=IB  NCCL_NET_PLUGIN=none   # 无 NCCL_PROTO（env 会压过 tuner）
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1  NCCL_IB_GID_INDEX=3
NCCL_IB_MERGE_NICS=0  NCCL_IB_RETRY_CNT=7  NCCL_IB_TIMEOUT=1000  NCCL_IB_TOS=46
NCCL_IB_SUBNET_AWARE_ROUTING=1  NCCL_CROSS_NIC=1  NCCL_SOCKET_IFNAME=enP7s7
```

**基线常数（8/22 校准，直接采信）**：bf16 AR busbw **21.4GB/s**（8.4MB、4ch；= 双发口方向 wire 10.7GB/s ≈ 逻辑口利用率 43%）；每步 AR 流量 1119MB（threshold 1024，86 次 AR）；AR 占步时 12.9%；小消息参考：786KB=133µs、196KB=98µs；BUFFSIZE 32MB 无收益（0.602 vs 0.588ms）；fp8 AR 结构性 No-Go。

---

## 2. 关键机制发现：NCCL 2.30 通道/ring 生成逻辑（v4 补丁的脆弱点）【源码】

读 `src/graph/connect.cc` `ncclTopoPostset()`（hardened 归档）发现三个与通道数耦合的行为：

1. **rail 交替（环反向）**：`crossNicRing==2 && nChannels 为偶` 时，奇数节点的 channel 对 (c, c^1) **交换 ringPrev/ringNext/ringRecv/ringSend**——奇数通道环方向反转。反向环邻接关系不变（物理邻居相同），v1 过滤照常通过，v4 映射覆盖两个方向——**4 通道下安全**。
2. **compute channel 翻倍**：`minCompCap≥90 && nNodes>1 && graphs[RING]->bwIntra>45 && nChannels<16` 时 `copyChannels(n→2n)`（sm_121 满足 compCap 条件）。复制通道保持环序与奇偶性——若源通道全在物理环上，复制后仍安全。
3. **多环 search 通道**：ringRecv/ringSend/ringPrev/ringNext 按 `c*nranks` 索引——**每个 search channel 可以有独立的环序**。NCCL 用多环把带宽分摊到多个 NIC（本机 4 个 RDMA dev、MERGE_NICS=0）。>4 通道时 search 可能产生**非物理环序**（如 0-2-1-3：rank0 的环邻是 2 和 3，而 2 与 0 无直连）。
   - 此时 v1 过滤**照常放行**（peer 是该 channel 自己的环邻）；
   - v4 对 rank0→peer2 查表得 `{0,0}` 哨兵 → **不覆盖 → NCCL 自动选 dev** → 该 QP 落到与 peer2 无 L2 路径的口 → 数据面不可达。

**与实测特征吻合**：8/22 通道扫描中 8ch/16ch 的 AR 在**所有尺寸恒定 ~17-20ms**（196KB 小消息也是 16.99ms、8.4MB 是 18.12ms）——恒定值 stall 而非带宽衰减，符合"部分通道 QP 不可达 + 重传/超时后由健康通道兜底完成"的特征【实测特征+源码推断】。

**另一处必须写清的矛盾**：FINALBASE v1.0（8/17，MIN=4/MAX=16）E2E 并未灾难（decode 仅比 B1 慢 5-23%）。两种解释并存【推断】：① 按 connect.cc 逻辑 `min(MAX,search×2)` 后 `copyChannels(n, max(MIN,·))`，MIN=4/MAX=16 的有效通道数可能塌缩到 ≤4-8，远小于 16；② B1 vs FINALBASE 的 decode 差 (+5-23%) 与 threshold-retester 发现的"重启级模式方差 ±8-13%"同量级，可能是模式混杂而非纯通道效应。→ **窗口取证项 #1**：用 NCCL_DEBUG=INFO 直接数出各 MIN/MAX 组合的实际通道数与环序（init.cc:1493 有现成的 `Ring %02d : %d -> %d -> %d` 日志行，v4 有 `RING-ONLY v4 rank %d->%d chan %d` 行）。

---

## 3. 方向一（主推）：大消息 / threshold 4096 适配（PR 侧）

### 3.1 背景与新 regime

threshold-retester（8/22）：threshold 1024→4096 实测 PR +12%（4K/16K 档双臂稳健、剂量-反应单调 1024<2048<4096），扩 M 路线已重开、4096 是 budget=4096 下的最优剂量点。threshold 4096 下 prefill 主 AR 消息 8.4MB → **33.5MB**。AR 字节总量/token 不变（带宽型），busbw 成为 PR 通信时间的唯一杠杆。

**带宽余量分析【推断】**：当前 busbw 21.4GB/s = 方向 wire 10.7GB/s（2 个逻辑口 × ~100G，合计 25GB/s 方向容量）的 **43%**。理论 busbw 天花板 ≈33GB/s（100% 逻辑口利用，不现实）；按 70-80% 实际可达利用率算 **27-30GB/s**；runbook §D 既有"双口 v3 补丁预期 25GB/s"的锚点。当前 4 通道每个方向仅 2 QP/dev，多通道分摊（更多 QP/CTA 并发）是提带宽的正道——这正是 8/16 通道修复的价值。

### 3.2 v5 补丁设计：全通道强制物理环序（修复 chan 映射）

**目标**：任意通道数下所有通道都落在物理环 01-02-04-03-01 上，使 v4 的 even/odd dev 映射对 8/16 通道同样成立。

**改动点（3 处，均在既有补丁家族内）**：

| 改动 | 位置 | 内容 |
|---|---|---|
| v1b：物理邻接过滤 | `src/transport.cc`（改造 v1） | 过滤条件从 `channel->ring.prev/next` 改为**硬编码物理邻接**：`peer != (rank+1)%4 && peer != (rank+3)%4`（跨机）→ 与 channel 环序解耦，杜绝"环 B 邻居被放行但无物理路径" |
| v5：环序强制 | `src/graph/connect.cc` `ncclTopoPostset()` | 在 `ringPrev/ringNext` 数组填充循环之后、`connectRings()`/`ncclBuildRings()` 之前，把**所有 channel 的所有 rank** 的 ringPrev/ringNext 覆盖为物理环（rank r：prev=(r+3)%4，next=(r+1)%4；环序 0→1→2→3→0）。`ringRecv/ringSend`（节点级，1 rank/node 时退化为同值）同步覆盖 | 
| v4 保持 | `src/transport/net.cc` | even/odd parity 映射不变——环序统一后对任意通道数正确（8ch → 每 dev 4 QP，16ch → 8 QP） |

**一致性要点（设计核心风险控制）**：channel->ring.prev/next（连接期）、rings[] 数组（ncclBuildRings，供算法 plan 做 chunk 偏移）、以及图搜索的带宽模型三者必须一致——v5 在数据源头（数组）覆盖，下游全部自然继承，不会出现"plan 环 ≠ 连接环"的错误结果。

**预期收益（量化）【推断】**：
- busbw：21.4 → **25-28GB/s**（+17-30%）。依据：① v3+MAXCH16 历史实测大档 2M-32M 全改善（-3%~-70% 延迟，其中中档收益最大）；② 逻辑口利用率 43% → 60-70%；③ runbook 25GB/s 锚点。
- 33.5MB AR 时间：2.35ms → **1.8-2.0ms**（-15-24%）；8.4MB：0.588ms → ~0.45-0.50ms。
- PR 端到端：AR 占步时 ~13% → **PR +2-3%**（叠加 threshold 4096 的 +12% → 合计 ~+14-15%）；TTFT 同向改善。
- 附带 DE 收益：768KB decode 大 AR（133µs）可能受益于多通道（v3+MAXCH16 历史：368KB 410→173µs）——但 B1 E2E 曾显示 4ch 对 decode 更优（存疑：见 §2 模式方差），**DE 侧通道数需 e2e 裁决，不预设**。

**风险与对策**：
| 风险 | 等级 | 对策 |
|---|---|---|
| 多通道环序强制作业与 NCCL 内部假设冲突（正确性） | 高 | 窗口先做 4 节点正确性门（allreduce 结果 vs 单机参考、尺寸 64KB-64MB 全档），不过不上 e2e |
| CUDA graph 兼容（decode capture sizes 1-64） | 中 | e2e 冒烟 + dspark 接受率对比（±4.6% 门） |
| 16ch 反而劣化 decode（B1 教训） | 中 | env 分档：MIN=4/MAX=16 先行；若 DE 回归则 MAX=8 或回落 4；必要时上 per-size 通道数分档（见 §3.4） |
| GLIBC/构建 | 低 | 沿用 anemll 镜像构建 SOP + GLIBC_MAX≤2.34 检查（既有流程） |
| 回滚 | 低 | 新库存 `.bak-v5-YYYYMMDD`，回滚 = 换软链 + 重启；env 回滚独立 |

**工作量**：补丁编写+容器构建 0.5-1 人日；窗口验证 0.5-1 人日；e2e A/B+上线 0.5 人日。**合计 ~2 人日**。

### 3.3 零补丁捷径（先行验证）：NCCL_IB_QPS_PER_CONNECTION=2

不动代码、纯 env：4ch 不变，每 channel 每对等 2 QP——若 43% 利用率的主因是 QP 深度/NIC 并行度而非通道数，这一项可吃到部分收益。历史 T1b（8ch+QPS2+SPLIT）劣化是**在 8ch 已坏的配置上测的**，QPS2@4ch 从未单独测过【源码历史核对】。列入窗口 A/B，30 分钟出结果；若有效可作为 v5 落地前的过渡配置。

### 3.4 可选增强：per-size 通道数分档（v5 落地后再评估）

ncclPersizeTunerOverride 已按尺寸改协议 cost table；同思路可对 nChannels 做分档（小消息 AR cap 4ch、大消息放 16ch），避免"16ch 利好大消息但拖累小消息"的两难。改动在 enqueue.cc/plan 层，工作量 +0.5 人日。**仅当窗口数据显示 16ch 对 DE 有回归时才做**。

### 3.5 "2048 臂流量 +10%" 之谜：已闭环，无需处理【实测】

threshold-retest §3.1：全矩阵（A3/B3/C1 臂）流量区间完全重叠（7.7-9.0GB/轮），快轮慢轮与流量无关——首对观察到的 +10% 为**采样窗口噪声**（0.5s 采样 vs 3-13s 轮长）。机制问题不存在，不进任何行动项。

---

## 4. 方向二：小消息优化（DE 侧）——诚实的结论：NCCL 层已到实证最优

### 4.1 现状评估（各杠杆逐项核对）

| 杠杆 | 现值 | 证据 | 结论 |
|---|---|---|---|
| 协议 | ≤40KB LL / >40KB Simple | 8/16 全尺寸扫描：LL 在 ≤32KB 快 9-34%，≥48KB Simple 快，LL 在 ≥64KB 爆炸（131KB +448%、368KB 20×） | **40KB 即交叉点，已最优**。decode AR 尺寸 64KB-786KB 全部正确落 Simple；≤32KB（attention 等小 AR）已享 LL |
| 通道数 | 4 | T1a(2ch) vs T1aM4(4ch)：64KB 309→279µs（4ch 更优）；B1 e2e decode 全档优于 MAXCH16 | 4ch 已最优（v5 后需复验，见 §3.2） |
| BUFFSIZE | 8MB | 32MB 无收益（0.602 vs 0.588ms） | 已最优 |
| QP/线程/CHECKS | — | P0 扫描：NTHREADS/QPS2/CHECKS_DISABLE 均无收益 | 已排除 |
| Tree/CollNet/PAT | 不可用 | Tree 需非环邻连接（v1 过滤 + 物理环无对角边）；CollNet 需 MNNVL/NVSwitch（RoCE 无）；P0"Tree 不可行" | 结构性排除 |
| 延迟下界 | ~98µs@196KB | 3 跳环：3×L_hop(~15µs) + 传输(~7µs) + launch/clone/同步(~40µs) ≈ 90-100µs | **当前值贴着环架构下界**【推断】 |

### 4.2 one-shot 数学评估（用户点名项）与既有裁决

4 rank 环上小消息 AR 的替代算法数学【推断】：

- **朴素 one-shot**（每 rank RDMA 直写 3 个 peer 全量 S + 本地归约）：t ≈ L_hop + 3S/(2×B_port) + 归约 kernel。wire 放大 3S/0.75S=4×。
  - 64KB：~38µs vs 环 64µs → **-40%**
  - 196KB：~70µs vs 环 98µs → -29%
  - 393KB：~125µs vs 环 ~115µs → **开始劣化**（放大流量吃掉延迟优势，交叉点 ~150-250KB）
  - 仅对最小档（C1 decode 64-192KB）有效，且需要绕过 NCCL 自研（NCCL 无 one-shot AR 算法）。
- **2-hop（RS(2)+AG(2)）**——已由 2-hop 项目做到 S3 终审：延迟 2 跳、wire 放大仅 1.33×，196KB 理论 ~48µs（-51%），全 decode 尺寸段有效。**但终审裁定**：SIMPLE 协议下双 Primitives 单 thread block 机制级崩溃（两套 kernel 同点 illegal memory access）；LL/LL128 价值区从未实现；唯一续接路径=新 sendrecv 设备原语，**5-10 人日、成功率 40-60%、价值上限 3-8% TPOT**（86 AR × ~100µs ≈ 8.6ms/步 ≈ 13% 步时，理论砍半也只回收 ~4ms）。
- **本方案立场**：维持 2-hop S3 裁决（不重开）。在 MoE（185-204ms/步，45-50%）是第一优先的盘子下，把 2-3 人日投入投向 v5（PR 侧、可叠加 threshold 4096）性价比显著更高。若用户明确愿意为 DE 3-8% TPOT 定价 5-10 人日，2-hop"旧 A（新原语）"是唯一技术路径，硬门=LL 16K 正确性。

### 4.3 DE 侧保留的低成本动作（窗口清单内确认项）

1. **hardened 库上 LL/Simple 交叉点复验**（30 分钟）：proto 扫描是 v3 库时代数据，hardened 库上 64-786KB 档 NCCL_PROTO=LL/LL128 A/B 一次，确认 Simple 仍优。预期无变化，纯属低成本确认。
2. **NCCL_GDRO/GDR 路径核对**（15 分钟）：GB10 UMA 下 Simple 是否走 GDR 或 host staging 影响小消息路径，NCCL_DEBUG=INFO 看一次 `useGdr` 标志，为 v5 窗口顺带项。
3. （v5 落地后）768KB 档 8/16ch 复测——若 DE 大消息受益，DE 也有 +1-2% 空间。

---

## 5. 微基准执行情况与事故记录（如实）

### 5.1 共享 GPU 微基准结论：**当前内存水位下不可行**

- 两次尝试（14:00/14:03 UTC）均在 CUDA 上下文创建即 OOM：生产占 GPU 99.5GB/121.6GB，系统 available 仅 9-14GB、**swap 14/15GB 近满**；单进程最小容器可建上下文（CTX_OK），四机 torchrun 全环境不行——余量不足以支撑共存。
- **事故**：14:03:32 第二次尝试后，14:03:51-14:04:12 四节点 head/worker monitor 全部 fast-fail，自愈链全集群滚动重启。归因（无责口径，聚焦系统）：测试容器经 libncclpin 钉到 CPU 5-19，与生产 EngineCore（15-19）/NCCL watchdog（8-9）同核竞争，四机同时启动触发全节点健康检查超时；叠加 01 上其他 teammate 容器。**教训：生产运行期（尤其 swap >50%）禁止四机并发带 LD_PRELOAD libncclpin 的容器；微基准一律走窗口。**
- 恢复：14:04:12 head 重建 → 14:05:03 四 rank 全连 → 14:13 health 200、KV 99.7GB 满配、四机 healthy、preemption=0、env 零漂移（MIN=MAX=4/BUFFSIZE 8MB/TUNER 40960 核验）。**生产已完整恢复。**
- 小缓冲基准（延迟曲线、33.5MB 探测、8ch 取证）**全部转入窗口清单**。工具链已就绪并分发四机：`/tmp/_ringopt/{ringopt_scan.py, ringopt_node.sh, run4.sh}`（9 尺寸 64KB-33.5MB 扫描、可传任意 NCCL env、四机一键启动）。

### 5.2 已有实测数据盘点（无需重跑）

| 数据 | 来源 | 状态 |
|---|---|---|
| 8.4MB busbw 21.4GB/s、86×0.588ms、RDMA 流量 1120MB/步 | ar-optimizer 8/22 窗口 | 采信 |
| 8/16ch 全尺寸 17-20ms 恒定 stall | ar-optimizer scan JSON（/tmp/_ar_opt/scan_ch*.json） | 采信（机制分析输入） |
| LL/Simple 40KB 交叉点全尺寸表 | proto-threshold-scan 8/16 | 采信（v3 库，待 hardened 复验） |
| v3+MAXCH16 大档 2M-32M -3~-70% | maxch16-large-msg 8/16 | 采信（v5 收益预期的历史锚点） |
| threshold 4096 PR +12% | threshold-retester 8/22 | 采信 |
| 小消息 786KB=133µs / 196KB=98µs | ar-optimizer 8/22 | 采信 |

---

## 6. 需窗口验证清单（按优先级）

| # | 项 | 时长 | 判据 |
|---|---|---|---|
| 1 | **8ch 机制取证**：MIN=MAX=8 + NCCL_DEBUG=INFO（SUBSYS=INIT,GRAPH），数实际通道数与环序（`Ring %02d` + `RING-ONLY v4` 日志行）；顺带 MIN=4/MAX=16 组合的有效通道数 | 30 分钟 | 确认/否决 §2 假设（非物理环序通道存在） |
| 2 | **QPS=2@4ch env A/B**（零补丁捷径） | 30 分钟 | 8.4MB/33.5MB busbw vs 21.4GB/s 基线，>+5% 即值得过渡上生产 |
| 3 | **v5 补丁构建+正确性门**：补丁→anemll 容器构建→GLIBC 检查→4 节点正确性（64KB-64MB 全档 vs 单机参考） | 半天 | 全尺寸结果 bit 级一致（bf16 容差） |
| 4 | **v5 通道 A/B**：4/8/16ch × BUFFSIZE 8/16/32MB × 8.4/16.8/33.5MB + 小消息曲线（64KB-1MB）+ 768KB 档 | 1 小时 | busbw 目标 ≥25GB/s；小消息无回归 |
| 5 | **e2e A/B**（bench_v2 32 档 + C0 锚点三档）：v5+MAX16 vs 现役，含 DE 12 档 + dspark 接受率 | 1.5 小时 | PR 无回归且 ≥+1%；DE/dspark 无回归（接受率 ±4.6% 门） |
| 6 | hardened 库 LL/Simple 交叉点复验 + GDR 路径核对 | 30 分钟 | 确认项，预期无变化 |
| 7 | （与 threshold 4096 上线联合）threshold 4096 + v5 组合 e2e：PR 四档 + DE C1/C12 | 1 小时 | 合计 PR +14% 目标拆分归因 |

**窗口预算合计**：取证+捷径验证 1 小时；v5 全链路（构建+正确性+A/B+e2e）约 1 个停机窗口（3-4 小时）。

## 7. 实施路线图（建议）

```
本周窗口 W1（3-4h）：
  #1 取证 → #2 QPS2 捷径 → 若 QPS2 有效：过渡上生产（env 变更，30 分钟回滚）
  → #3 v5 构建+正确性门
下个窗口 W2（3-4h）：
  #4 v5 通道 A/B 定参（ch 数/buffsize）→ #5 e2e A/B → 上线（新库 .bak 链 + env MIN/MAX 调整）
与 threshold 4096 上线解耦（由 threshold 线自行择窗），W2 后可做 #7 组合验证
```

## 8. 生产加固建议（事故衍生的规矩，供 team-lead 采纳）

1. 生产运行期（available <20GB 或 swap >50%）**禁止**四机并发起带 `LD_PRELOAD=/opt/libncclpin.so` 的容器（CPU 钉核 5-19 与 EngineCore 15-19 / NCCL watchdog 8-9 冲突）；测试容器如需 preload，应改钉到空闲核区（如 10-14）或干脆不 preload。
2. 微基准/试验一律走停机窗口，用 `/tmp/_ringopt` 工具链（含显存自检：起容器前 `free -m` + swap 水位检查）。
3. 建议在 monitor_tp4_head/worker 加"连续 fast-fail ≥2 次才重建"或"重建前检查 swap 水位"的防抖，避免瞬时 CPU 竞争触发全集群滚动重启（本次自愈链工作正常但代价是 ~10 分钟不可用）。

---

## 9. 数据来源

- 服务器：`<INSTALL_DIR>/backup/nccl-official-2307-hardened-20260816/`（patches/ 四补丁全文、MD5-RECORD、README 构建方式）、`src/graph/connect.cc`（通道/ring 生成逻辑）、`<INSTALL_DIR>/docs/`（runbook-tp4-v1.5、nccl-function-doc-mapping-qa、sre-perf-twokernels）、`<INSTALL_DIR>/deliverables/engineering-assurance/`（nccl-ab-results / proto-threshold-scan / maxch16-e2e / maxch16-large-msg / large-msg-nonmonotonic / 2hop-s3-final-adjudication / final-performance-baseline-v2-B1 / ringonly-optimization-2026-08-15）
- 本地：ar-optimization-2026-08-22.md、threshold-retest-2026-08-22.md、analysis-tp2-tp4-communication-2026-08-09.md
- 实时：/opt/nccl-ringonly md5 核验、生产 env docker inspect 核验（8/22 14:13-14:20 UTC）、/tmp/_ar_opt/scan_ch*.json、四机 docker/journalctl 取证（14:04 事件）

> 本报告由工程保障团队 ringonly-optimizer 生成。v5 补丁设计与生产变更（通道数调整）请人类工程负责人复核后择窗执行。
