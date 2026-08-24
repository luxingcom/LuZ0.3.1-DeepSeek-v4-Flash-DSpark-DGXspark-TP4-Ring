# DGX Spark TP4 — 14 条现场裁决与执行计划（第二轮）

**日期**：2026-08-20
**督导**：甄宇航 Zhen（engineering-director）
**依据**：本轮对生产集群 4 节点运行态实测（SSH 直连 node01~04）+ 上一轮四位成员审查报告。
**说明**：用户对每条作出明确裁决/指令，本表逐条记录处置与执行设计。涉及生产变更项（②中的卸载）已在实际动手前与用户确认执行方式。

---

## 📋 14 条逐条裁决表

### ① 性能技术路线核对（W4A8 vs W4A4）
- **用户裁决**：W4A8 下峰值 187T 达标 ✅；**W4A4 要求 400T**，需核对技术路线是否达标。
- **处置**：以 **400T 为 routeA 在 W4A4 下的硬指标**。三组核对路线：
  1. **SASS/符号核对**：routeA 调用 `cutlass_scaled_fp4_mm`（sm_120a 原生 FP4）+ `scaled_fp4_quant`（16-group e4m3）——确认 4W4A 激活路径成立、非退化成 bf16。
  2. **实测候选**：W4A4 下大 shape 实测 TFLOPS（上一轮已测 60~187，12288 shape 峰值 187）；核对距 400T 差距与可优化点（A 量化融合进 CUDA Graph、scaled_fp4_quant 改 cutlass backend、大 shape autotune）。
  3. **差距结论**：若实测不可达 400T，记录差距与原因（编译器能力边界 vs 内核设计），交由用户决策（是否接受差距/换实现）。
- **状态**：执行中 → 见交付件 `architecture-nvfp4-2026-08-20.md` + 本轮实测补充。

### ② 生产 4 rank 恢复 + 两算子性能影响测试
- **用户裁决**：生产 4 rank **等待测试完成后恢复**；**已关闭自愈**；测试需**卸载生产环境腾出内存**。
- **实测确认**：4 节点 rank 容器全部 `Up 20 hours (healthy)`：01=rank0、02=rank1、04=rank2、03=rank3；rank0 内 `vllm serve` 在跑（--gpu-memory-utilization 0.80、--api-key 明文、--tensor-parallel-size 4 --nnodes 4）。= 确为**生产环境**。
- **执行设计（已与用户确认）**：
  1. 确认生产环境后 → `docker stop` 4 节点 rank 容器（Retention：容器保留可重启，勿删）。
  2. **用生产镜像 `dspark-vllm-gx10:0.2.1-v026.0`（非 v0.27）拉起独立 test 容器**，复用同一套挂载（<INSTALL_DIR> models/lib/ncclpin、/opt/nccl-ringonly、patch tilelang），创建 `vllm-tp4-test` 系列。
  3. 用生产 bench 脚本 `bench_prefill_decode_async.py`，**挑选 10 个典型负载**（由 --ctx 512/4096/16384/65536/131072 与 --concurrency 组合选取，覆盖 prefill/decode 典型档）。
  4. 测**两算子对总体性能的影响**：kernel① 路线A 接入 vs 不接入、kernel② v17 vs v11，A/B 对比总吞吐/prefill/decode 的 p50×conc。
  5. 测试完成后 **恢复生产 4 rank**（start_tp4_cluster.sh head-first），并按裁决②既定的恢复流程。
- **状态**：待执行（生产卸载已获用户 Go，动手前记录容器状态快照）。

### ③ shim 实际 v8（纠正我上轮误判）
- **用户裁决**：shim 实际是 **V8**，看的是过期文件；代码和文件已归档。
- **处置**：接受纠正。**以运行态实测为准**——上轮 S2 的"源码↔二进制漂移"判断源自工作区过期副本；生产 `<INSTALL_DIR>/lib/libncclpin.so` 为 v8（MD5 ce43c688，NCCL→8-9 / EngineCore→15-19），与 runbook 一致。**S2 降级为"需清理工作区源文件版本标注"**，非生产缺陷。
- **状态**：✅ 已核实纠正（生产 v8）。

### ④ `/vllm-workspace` 非持久问题详细说明
详见下方"专项说明④"。

### ⑤ 网段三套口径未统一问题详细说明
详见下方"专项说明⑤"。

### ⑥ 泄漏 API key 用途与安全风险
详见下方"专项说明⑥"。

### ⑦ 过时 .58/.60 TP2 注释需修正
- **用户裁决**：所有写成 `.58/.60 TP2` 拓扑的过时文件/注释都需修正。
- **处置**：调度 tech-writer 全工作区 grep 定位 `.58/.60`/TP2 陈旧引用 → 按 TP4 运行态（02=.187/03=.188/04=.189、8191 端口）修正。典型：prometheus.yml（仍旧 .58/.60）、文档/注释中 .60 等。
- **状态**：待执行（tech-writer）。

### ⑧ SSH 编排问题修复
- **用户裁决**：SSH 编排问题需修复（上轮 H4：缺 StrictHostKeyChecking/ServerAlive、复合命令拼接解析风险）。
- **处置**：code-reviewer 定位 start_tp4_cluster.sh 及各脚本 SSH 用法 → 补 `-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6`、修正远程复合命令拼接（引用/转义）。改后全节点连测。
- **状态**：待执行（code-reviewer）。

### ⑨ UMA 维持 util 0.8 不变 + 测试腾内存方式
- **用户裁决**：UMA 内存**维持 util 0.80 不变**（否决我上轮"降 0.70"建议）；测试需卸载生产腾出内存。
- **处置**：生产 serve 参数保持 `--gpu-memory-utilization 0.80` 不动；测试通过卸载生产 rank 容器腾出 UMA/显存（与②一致）。运行态确认 rank0 serve 已是 0.80 ✅。
- **状态**：✅ 生产参数确认 0.80 不动；测试执行同②。

### ⑩ 监控 job 修正
- **用户裁决**：监控 job 需修正（上轮 R5：job 仍旧命名、抓 188:8001 无效目标、冻结时 Prom 中断）。
- **处置**：sre-engineer 核对 prometheus.yml/告警规则 → 修正 job 名（对外 8191）、移除无效抓取目标 188:8001、加固节点冻结时采集可靠性。
- **状态**：待执行（sre-engineer）。

### ⑪ start_tp4_cluster 核实 set -e
- **用户裁决**：核实 set -e 影响，**无兼容问题可添加**。
- **处置**：code-reviewer 通读 start_tp4_cluster.sh 主路径，评估加 `set -e` 对既有控制流（GPU-gate、对端门禁、快速失败）的影响；若所有失败分支已有显式处理（不依赖 set -e 也无副作用）则添加，否则仅加固关键失败点。
- **状态**：待执行（code-reviewer）。

### ⑫ 配置文档漂移纠正
- **用户裁决**：配置文档漂移均为失准，**按运行态实测纠正文档**。
- **处置**：以本轮实测（rank 容器名、挂载、镜像 0.2.1-v026.0、util 0.80、shim v8、NCCL b7784b49、监控 8191、网段）为唯一权威基线，tech-writer 全量回填 runbook/交接文档/资料库，纠正 10+ 处失准。注明"以运行态实测为准"。
- **状态**：待执行（tech-writer + sre 供数）。

### ⑬ 停机 SOP 完善
- **用户裁决**：停机 SOP 完善（上轮 R7 停机须先停 monitor/timer；R12 kvssd 不可行不得重启）。
- **处置**：sre-engineer 完善停机 SOP：先停 monitor timer/service（防误伤自愈拉起）→ 停 rank 容器 → 维护 → 恢复顺序；补 kvssd 禁启、RTO 目标、恢复验证清单。
- **状态**：待执行（sre-engineer）。

### ⑭ Tessa 三问题均为测试问题
- **用户裁决**：Tessa 的 G8（safety 3 脚本缺陷）/G9（routeA 无确定性/泄漏测试）等为测试问题，**自行判定**。
- **处置**：接受。Tessa 上轮已定性 G8 为**测试脚本缺陷**（期望 255→144、1→24、boundary 未 seed），非内核缺陷（8/8 逐字节已过）。G9 补测列为收尾项。三处**不改被测代码**，仅修测试脚本判据。
- **状态**：✅ 已定性（测试问题，非内核）；待补确定性脚本。

---

## 专项说明④：/vllm-workspace 非持久问题

**现象**：生产容器 `vllm-tp4-rank0` 内 `/vllm-workspace/`（含 nvfp4-landing 资料库、bench 脚本副本、交付包）是**容器内部目录，未挂载宿主机的 `<INSTALL_DIR>`**。

**为什么是问题**：容器 `--restart no` + `/vllm-workspace` 不被卷绑定。**任何一次容器重建（docker rm + 重起，或 test 容器用完清理后）都会把 /vllm-workspace 里的全部内容抹掉**——包括：
- NVFP4 路线A 适配层、kernel② v17 文件、测试脚本 → **丢了就要重新拷**
- 各种交付包/测试中间产物

**RPO（恢复点目标）**：当前只要产物依赖 `/vllm-workspace` → **RPO=∞**（重建即丢，没有恢复点）。这是上轮架构审查的 **P0 R3 / ADR-3**。

**正确做法（ADR-3 已定）**：宿主机 `<INSTALL_DIR>`（`scripts/`、`lib/`、`models/`、`envs/` 已挂载进容器）才是**持久单一权威源**。落位：
- routeA 适配层 `nvfp4_4w4a_mmaf.py` → `<INSTALL_DIR>/scripts/nvfp4/`（已挂载进容器）
- kernel② v17 → `<INSTALL_DIR>/kernel2/v17/`
- 验证判据：**容器重建后 `import nvfp4_4w4a_mmaf` 成功 + quick check 跑通**。

**本轮验证（实测）**：rank0 `/vllm-workspace/nvfp4-landing/` 存在（routeA/routeB/docs/tests），但确认它是容器内非挂载目录 → 本轮**暂停在 /vllm-workspace 存生产机制**，以 <INSTALL_DIR> 为准。

## 专项说明⑤：网段三套口径未统一问题

**现象**：工作区/文档存在三套网段描述，运维易拿错 IP：
1. **控制面（管理网镜像）**：`<NODE_IP>/.187/.188/.189` —— TP4 分布式 TCPStore/master 走这个（实测 rank0 serve 的 `--master-addr <NODE_IP>`）。这也是 registry（<NODE_IP>:5000）所在网段。
2. **环网数据面**：`10.100.x.x`（RoCE，如 10.100.136~139）+ `10.20.0.x`（另一环线段）。
3. **生产 serve 参数**：`--master-addr <NODE_IP> --master-port 25999`（实测）。

**为什么是问题**：同一设备三套编号，且 MTU 配置只覆盖了部分（上轮发现 10.20.0.x 的 MTU 未被 netplan 覆盖）。运维"照文档配"极易拿错地址 → 组网失败 / NCCL 连不通。

**实测修正**：以 rank0 serve 实参为准打通权威表——**控制面=<NODE_IP>；数据面=环网 10.100.x + 10.20.0.x**。待 tech-writer 建单一权威 hosts/netplan 映射表 + 补 10.20.0.x MTU。

## 专项说明⑥：泄漏 API key 用途与安全风险

**实测**：rank0 serve 启动参数含 `--api-key <KEY_PREFIX_OLD>98d4cae30a416366729f09202b1f013a429a13679f973c09c5344594`（64-hex 生产密钥）。
- **用途**：vLLM `/v1` 服务的鉴权密钥——客户端访问 OpenAI 兼容接口必须带 `Authorization: Bearer <key>`。rank0 serve 用该 key 保护 8001 端口；2 节点 TP2/生产落盘脚本（start_tp4_head.sh:77 / worker.sh:76）会把它 `echo` 进启动日志。
- **安全风险**：
  1. **凭据入日志**：脚本把 `--api-key` 明写进 SERVE_CMD 并 echo → 生产密钥随容器/log 落盘，`journalctl`/日志文件可读。
  2. **泄露后果**：任何拿到日志的人可免鉴权调用 8001 推理接口 → 盗用算力、越权访问模型、无限推理（成本/合规风险）；若模型权重敏感，等同未授权访问。
  3. **无轮换机制**：长期不变，泄露窗口无限。
- **处置建议（P0）**：①脚本改为从环境变量/密钥文件读取 key，日志打码；②日志 chmod 600 + 定期清理；③建立密钥轮换（本次不动生产，列入 P0 供用户定轮换时机）。用户上轮已把"密钥轮换时机"列入待裁决项。

---

## ⚠️ 需用户最终裁决（更新）

1. **W4A4 400T 达标判定**：实测后若不可达，是否接受差距或换实现（等①实测数据）。
2. **生产 4 rank 恢复时机**：等②测试完成后恢复（用户已定）。
3. **密钥轮换时机**：未定（本批不轮换，仅防泄漏入日志）。
4. **v15 对照真实性**：上轮项，维持以官方 dequantize_to_dtype 对照为准。

## 下一批中风险待核对（用户要求处理完 14 条后再清点）

将在 14 条处置完成、测试出具结论后，由四位成员重新清点"中风险项"（上一轮 🟡 中项 M1-M6/R8-R12/G9 等）是否仍需处理，输出第二轮中风险清单。