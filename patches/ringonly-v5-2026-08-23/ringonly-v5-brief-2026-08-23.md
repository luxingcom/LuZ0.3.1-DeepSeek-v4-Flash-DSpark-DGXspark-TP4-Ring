# ringonly v5 环序强制补丁 — 实现与构建简报 + W1 测试就绪

**执行**：v5-developer（阿奇 · 系统架构师，工程保障团队）
**日期**：2026-08-22/23（UTC）
**任务来源**：team-lead 指派——v5 环序强制补丁开发 + W1 测试准备（纯源码/构建，不占 GPU）
**口径标注**：【实测】= 服务器命令直接输出；【源码】= NCCL 2.30.7 源码精读；【推断】= 逻辑推导。

---

## 0. 一页结论

1. **v5 补丁已实现并构建成功**【实测】：connect.cc 数据源头强制所有 channel 环序 = 物理环（覆盖 rail 交替与多环 search 非物理序）+ v1b 硬编码物理邻接过滤。构建于生产同源镜像 CPU 容器（未占 GPU），GLIBC_MAX=2.34 达标，md5 `2b8669eceebd633120cd8053a5be3089`。
2. **ABI 与生产库完全一致**【实测】：动态符号名集合 diff=0（仅地址偏移），ncclGetVersion 在位，"RING-ONLY v5" 取证日志串在二进制中。
3. **四机分发完成**【实测】：/tmp/_ringopt/v5/（01/02/04/03 四节点 md5 一致），生产 /opt/nccl-ringonly 与全部生产配置零接触——无需回滚方案（未动生产）。
4. **W1 测试矩阵就绪**【实测】：w1_matrix.sh 一键跑 8 个臂（取证×3 + 正确性门 + 通道扫描×3 + 可选 QPS2），判据 J1-J5 内嵌于脚本注释与输出。
5. CPU 容器内 ctypes CDLL 加载两库（v5 与生产对照）均 SIGSEGV——**环境性现象（无 GPU driver），非 v5 特有**【实测，对照组复现】；功能门留给 W1 窗口四机实测。

## 1. v5 实现（commit 3c8f449，分支 nccl-2307-v5，基于 hardened 0dd44cd）

### 1.1 改动一：环序强制（src/graph/connect.cc · ncclTopoPostset()）

- **位置**：allTopoRanks→ringRecv/ringSend/ringPrev/ringNext 的 gather 循环之后、`connectRings()` 之前（connect.cc:428 附近插入 32 行）。
- **内容**：当 `nNodes==4 && nRanks==4`（本集群 1 rank/node）且 `NCCL_RINGONLY_V5≠0` 时，对每个 channel c：
  - `ringRecv[c*nNodes+n] = ringSend[c*nNodes+n] = firstRanks[n]`（节点级）
  - `ringPrev[c*nranks+firstRanks[n]] = firstRanks[(n+3)%4]`，`ringNext[...] = firstRanks[(n+1)%4]`（rank 级）
  - 即强制环 = 节点序/rank 序环 0-1-2-3-0 == 物理环 01-02-04-03（与 v4 per-peer dev 映射同一假设，生产 4ch 已验证）。
- **为什么在数据源头覆盖**：rail 交替（crossNicRing==2 的 c/c^1 交换）发生在覆盖之前被完全中和；下游 connectRings、`channel->ring.prev/next` 赋值、双倍 memcpy、copyChannels、ncclBuildRings 全部从这些数组派生 → **plan 环 == 连接环**，不会出现算法 plan 的 chunk 偏移环与连接环不一致的错误（前任报告 §3.2 一致性要点的落实）。
- **取证日志**：每 rank 每 channel 一行 `RING-ONLY v5 rank %d chan %d forced ring prev %d next %d`（仅 NCCL_DEBUG=INFO 时输出，生产默认 WARN 零噪声）。

### 1.2 改动二：v1b 物理邻接过滤（src/transport.cc · ncclTransportP2pConnect()）

- v1 原过滤用 `channel->ring.prev/next`（与 channel 环序耦合）；v1b 在 `nRanks==4 && nNodes==4` 时改用硬编码物理邻接 `(rank+3)%4 / (rank+1)%4`，与 channel 环序解耦。
- **定位为防御层**：v5 生效时两者等价（channel 环序已被强制为物理序）；`NCCL_RINGONLY_V5=0` 关闭 v5 时 v1b 仍拦截非物理环序通道的建连——正好构成 W1 的 A/B 取证臂（A1: v5-off 复现 stall + 抓 stock 环序；A2: v5-on 验证强制）。

### 1.3 与 v4 的配合（零改动）

v4 net.cc 的 per-peer even/odd→devA/devB parity 映射保持原样。环序统一后，该映射对任意通道数成立：8ch → 每 dev 4 QP，16ch → 8 QP。**这正是解锁 8/16 通道的机制**：非物理环序通道不再落进 v4 的 {0,0} 哨兵（自动选 dev → 无物理路径 QP → 恒定 17-20ms 重传 stall）。

### 1.4 安全设计

- 双 gate（nNodes==4 && nRanks==4）：单机/其他拓扑/多 rank-per-node 保持 stock+v1 语义，不影响通用性。
- `NCCL_RINGONLY_V5=0` 整体开关（默认开）：W1 A/B 与万一需要的运行时规避。

## 2. 构建记录【实测】

| 项 | 值 |
|---|---|
| 源码 | git clone hardened 归档 → /tmp/nccl-v5-src，分支 nccl-2307-v5（归档保持 pristine） |
| 镜像 | <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0-b12x-recovered-20260820（CPU 容器 --rm，无 --gpus，未占 GPU） |
| 命令 | `rm -rf build && make -j src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"` |
| 耗时 | ~10 分钟（15:41-15:51 UTC），日志 /tmp/nccl-v5-src/build.log（本地副本 build-v5.log） |
| 产物 | libnccl.so.2.30.7，60,538,392 字节，md5 `2b8669eceebd633120cd8053a5be3089`（本地副本 md5 复核一致） |
| GLIBC_MAX | **2.34**（约束 ≤2.34，PASS；2.32/2.33/2.34 各 1/1/51 个符号引用，无 2.35+） |
| ABI | 动态符号名集合与生产库（2be94172）diff=0；ncclGetVersion 在位 |
| 补丁产物 | v5-incremental.patch（65 行：connect.cc+32 / transport.cc+11）；v5-full-chain.patch（3803 行 = v1b+v4+stageB-tuner+hardened+v5 全链，对 vanilla 73cf112） |
| 加载冒烟 | LD_PRELOAD 进 python 进程 PASS；ctypes CDLL 在无 GPU 容器 SIGSEGV——生产库对照同样 SIGSEGV（环境性，非 v5 特有；归档时代的 LIB_LOAD_OK 应为带 GPU 容器所测） |

## 3. W1 测试就绪（GPU 测试等窗口）

### 3.1 分发布局（四机，md5 已核验一致）

```
/tmp/_ringopt/v5/
  libnccl.so.2.30.7        # v5 库（staging，生产库未动）
  MD5-RECORD-v5.txt        # 构建/校验记录
  v5-incremental.patch / v5-full-chain.patch
  ringopt_scan_v5.py       # 扫描：64KB-128MB 全档 + RINGOPT_SIZES 子集 + CHECK=1 正确性门
  ringopt_node_v5.sh       # 节点运行器：LD_PRELOAD=/v5lib/libnccl.so.2.30.7（挂载 staging 副本）
                            #   NO_PIN=1 去 libncclpin；extra env 透传（DEBUG/RINGONLY_V5=0/QPS2…）
  run4_v5.sh               # 四机一键启动（物理环序 01-02-04-03，日志落 logs/）
  w1_matrix.sh             # W1 窗口全矩阵 + 判据（在 01 上执行）
  deploy_v5.sh             # 幂等重分发（01 上）
```

### 3.2 W1 矩阵（w1_matrix.sh，预计 35-45 分钟）

| 臂 | 配置 | 目的 |
|---|---|---|
| A1 | v5-off（RINGONLY_V5=0）8ch + DEBUG=INFO SUBSYS=INIT,GRAPH | 复现 stall + 抓 NCCL 自身多环/rail 交替环序（"Ring %02d" 行）——根因假说直接取证 |
| A2 | v5-on 8ch + DEBUG=INFO | "RING-ONLY v5" 行验证全通道物理序 + stall 消失 |
| A3 | v5-on MIN=4/MAX=16 + DEBUG=INFO | 数 16 通道档的有效通道数 |
| B | v5 16ch + CHECK=1 全尺寸 64KB-128MB | 正确性门（vs 本地确定性期望和，bf16 容差 max rel err <5%） |
| C1-C3 | v5 4/8/16ch 全尺寸扫描 | busbw 对比基线 21.4GB/s@8.4MB；通道数定参 |
| D（可选） | QPS2@4ch | 零补丁捷径臂（注释态，择需启用） |

### 3.3 判据（J1-J5，内嵌于 w1_matrix.sh）

- **J1 stall 消失**：v5-on 的 8/16ch 在 8.4MB 档 AR 时间回到带宽量级（≤~1.5ms，对照 stall 恒定 17-20ms），小消息回 ~100-400µs 量级。
- **J2 收益**：8/16ch 最优 busbw@8.4MB ≥25GB/s（目标带 25-28）。
- **J3 不劣化**：v5 4ch busbw@8.4MB 在 21.4±5% 内。
- **J4 正确性**：臂 B 全尺寸 PASS。
- **J5 环序物理**：A2 日志每 channel prev/next = rank±1 (mod 4)。
- J1 失败且 v5-on → 假说被否证，升级上报，不进 e2e。

## 4. 执行边界遵守情况

- 未占 GPU（构建与加载冒烟均 CPU 容器 --rm）；未动 /opt/nccl-ringonly、未动生产配置与容器。
- 长命令（构建 10 分钟）后台 + 落盘日志（build.log）。
- 生产运行期未起任何四机带 preload 容器（§8 规矩；W1 脚本仅在窗口执行）。

## 5. 交付物索引

- 服务器：/tmp/_ringopt/v5/（四机，见 §3.1）；源码 /tmp/nccl-v5-src（分支 nccl-2307-v5，commit 3c8f449，build.log 在内）
- 本地：C:\Users\novAI\WorkBuddy\集群部署\deliverables\engineering-assurance\ringonly-v5-2026-08-23\
  - libnccl.so.2.30.7（v5 库副本）、v5-incremental.patch、v5-full-chain.patch、build-v5.log
  - apply_v5.py（补丁应用脚本，锚点断言式）、staging/（四机分发全套：MD5-RECORD-v5.txt + 5 个脚本）

## 6. 后续（W1 窗口执行时）

1. 窗口内跑 `bash /tmp/_ringopt/v5/w1_matrix.sh`（node01，约 40 分钟）→ 按判据 J1-J5 裁决。
2. 若 J1-J4 通过：W2 做 e2e A/B（bench_v2 32 档 + C0 锚点）后择机上生产（新库 .bak-v5 链 + env MIN/MAX 调整），与 threshold 4096 上线解耦。
3. 若 A1 取证显示 stock 环序其实是物理的（假说否证的另一形态），stall 根因需另找——v1b 防御层仍无害保留。

> 本简报与生产变更（v5 上线、通道数调整）请人类工程负责人复核后择窗执行。
