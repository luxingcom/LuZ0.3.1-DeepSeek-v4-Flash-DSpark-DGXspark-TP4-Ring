# b12x Wrapper 跨层去重补丁——实施与验证报告（2026-08-22）

**执行**: ws-dedup-impl（系统架构师）· 研究二落地
**输入**: p1-p2-research-2026-08-22.md §2.4/§2.5 设计（方案 A）+ 生产镜像源码取证
**口径标注**: 【实证-源码】= 容器内代码直接验证（行号级）；【实证-数据】= GPU 实测；【推断】= 基于证据推断
**产物位置**: 本地 `deliverables/engineering-assurance/ws-dedup-patch-2026-08-22/` + 服务器 `node01:/tmp/_ws_dedup/`

---

## 0. 执行摘要

- **补丁已实施并通过 L1 全量验证**（24/24）：`flashinfer_b12x_moe.py` 增加模块级几何键共享池（env `VLLM_B12X_SHARED_WRAPPER` 门控，**默认 off，off 路径与原代码 AST 级等价**）。净增 ~79 行（含注释），交付 overlay 文件 + 四节点部署脚本（**未部署**，按铁律只落盘）。
- **设计核实结论**：§2.5 伪代码与 fork 实际源码结构完全一致，唯一偏差是 env 默认值（设计写 "1"，按任务铁律改为 "0"）。另有一个**利好新发现**：`moe_dispatch.py` 的 `_get_weight_views` 有模块级 `_WEIGHT_CACHE`（data_ptr 键控），共享后权重视图重算只是 µs 级 host 开销。
- **L2 GPU 实测全过（16/16）**：池机制（off 4 wrapper / on 1 wrapper）、去重量级直接实测（medium 几何 **off 2 wrapper = 915.2 MiB vs on 1 wrapper = 460.1 MiB，每 wrapper 455 MiB**）、数值正确性（共享不引入超出内核自身 ULP 级非确定性边界的误差——边界实测 0.03125 ≈ 2 ULP bf16）、显存预算 peak 915.2 MiB << 5GB。
- **战略意义确认**：本补丁是 W4A4/W4A8 路线（微基准 M≥4096 时 1.32×）的内存前提；与 threshold 4096 组合构成"算力利用"主线。

---

## 1. 设计核实（§2.5 vs 实际源码）【实证-源码】

一次性容器（`--entrypoint bash --rm`）读取生产镜像 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0` 内 fork 源码：

| 设计假设 | 实际源码 | 结论 |
|---|---|---|
| `_ensure_wrapper()` :249-266 每层 lazy 创建 wrapper | `flashinfer_b12x_moe.py:230-246`，构造参数 8 个与设计伪代码逐字一致 | ✓ 一致 |
| `apply()` 权重每调用传入、`output.copy_(wrapper_output)` 立即消费 | `:283-295` | ✓ 一致（共享安全论证成立） |
| `workspace_shapes()` 返回 `(1,),(0,)` 不变 | `:206-221` | ✓ 一致 |
| 每实例 `_allocate_buffers()` 分配 static+dynamic+_moe_output | flashinfer `b12x_moe.py:258-470`：nvfp4 路径分配 `_static_workspace` + 按需 `_dynamic_workspace` + `_moe_output` | ✓ ×43 根因坐实 |
| 权重缓存 data_ptr 键控 → 共享不串权重 | `b12x_moe.py:573-585`（weight_key 7 元组全 data_ptr）、`:614`（`_weight_views` 单槽缓存按 key 失效） | ✓ 一致 |
| 几何键 7 字段充分 | ctor 其余参数对本类恒定：quant dtype 构造时 assert nvfp4；device/output_dtype/swiglu_*/source_format 走默认值 | ✓ 键充分 |
| env 默认 "1" | **改**：任务铁律要求默认 off → `os.environ.get(..., "0") == "1"` 严格解析 | ✎ 唯一偏差（有意） |

**新发现（设计之外，利好）**【实证-源码】：`moe_dispatch.py:391-470` `_get_weight_views` 内部有**模块级 `_WEIGHT_CACHE`**（data_ptr 键控 + 驻留淘汰注册）。共享后 43 层交替调用使 wrapper 单槽 `_weight_views` 每次失效，但重算路径命中 `_WEIGHT_CACHE`——重量级操作（`convert_sf_from_mma_layout().contiguous()`、fp32 转换）全部全局缓存，实际重算只是 permute view + 2×make_ptr + namedtuple 构造（µs 级 host 开销，每层每步一次）。**共享的运行时代价可忽略**。

**上游参照核对**：b12x commit 77d1cfe（几何键跨层去重）、lablup module 级共享池——本补丁机制等价，做在 fork vLLM experts 层（升级 FlashInfer 0.6.16 不受影响，见 p1-p2 报告 §3.3 联动结论）。

## 2. 补丁实现

**文件**: `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`（原 295 行 → 374 行；`patch.diff` 133 行）

核心改动（全部在 env=1 时才激活）：

1. 模块级 `_B12X_WRAPPER_POOL: dict[tuple, Any]` + `_B12X_WRAPPER_POOL_LOCK`（threading.Lock，防多线程首用竞争）；
2. `_b12x_wrapper_pool_enabled()`：`VLLM_B12X_SHARED_WRAPPER=1` 且仅 "1" 才开启（"true"/"2"/""/unset/"0" 全部 off）；
3. `_ensure_wrapper()`：off 分支保持原构造调用（AST 逐节点等价）；on 分支按几何键 `(global_num_experts, topk, hidden_dim, intermediate_size_per_partition, max_num_tokens, num_local_experts, activation_str)` 查池，miss 时创建并注入，创建时打 `logger.info`（含几何与池大小，供启动日志确认 pool size == 不同几何数）；
4. 池容量锁定 `max_num_tokens`（`use_cuda_graph=True`），共享后禁止 resize（fail-fast 防 graph 地址漂移）；
5. 生命周期：池随进程存活，worker 退出即回收，无显式析构需求。

**注意（几何键设计说明）**：activation 必须进键——`relu2`（非门控）与 `silu`（门控）的 w1 行数不同（n vs 2n），workspace 形状不同。当前模型 43 层全 `silu`，实际池 size 预期 = 1。

## 3. L1 无 GPU 验证（一次性容器 CPU）——24/24 PASS

脚本 `l1_test.py`（双模块并载：pristine `src_orig/` vs 补丁版）：

| 组 | 测试 | 结果 |
|---|---|---|
| T1 导入 | 双模块加载（补丁新增 os/threading/init_logger 均无害） | PASS |
| T2 AST 等价 | (a) 类方法名集合一致；(b) 除 `_ensure_wrapper` 外全部方法 AST-identical；(c) off 分支与池分支的构造调用均与原文件构造调用 **AST 逐节点一致**（= off 路径构造语义字节级等价）；(d) 模块级新增严格白名单 6 项（os/threading/init_logger/logger/_POOL/_LOCK/_enabled），零删除 | PASS |
| T3 运行时 off 等价 | mock `B12xMoEWrapper` 录制构造调用：补丁版 4 实例 → 4 次调用，**kwargs 列表与原模块逐一相同**；wrapper 对象各异；池未使用 | PASS |
| T4 池逻辑（on） | 4 同几何实例 → 1 次构造、共享同一对象、池 size=1；第二几何（max_tokens 2048）→ 新条目；activation 变化 → 新条目；幂等（二次 `_ensure_wrapper` no-op）；env 严格解析 6 组用例 | PASS |

## 4. L2 共享 GPU 小验证 —— 16/16 PASS（最终轮 l2_final.log）

**环境**: 节点 01（生产 vllm-tp4-rank0 运行中），一次性容器 `--gpus all`，补丁文件 bind mount 到真实路径（同时验证 overlay 挂载点 + 生产路径导入）。脚本 `l2_gpu_test.py`（harness 经三轮迭代修正，见 §4.2）。

### 4.1 最终结果【实证-数据】

**池机制与去重（绝对读数口径，phase-end）**：

| 相位 | 配置 | allocated（phase 末） |
|---|---|---|
| 小几何（E=8,topk=2,K=512,N=256,maxt=256）off | 4 实例 → 4 wrapper | 9.3 MiB |
| 小几何 on（env=1） | 4 实例 → **1 共享 wrapper** | **5.0 MiB**（省 4.2 = 3 wrapper × 1.4） |
| medium 几何（E=256,topk=6,K=4096,N=512,maxt=1024，生产几何的 maxt=1024 版）off | 2 实例 → 2 wrapper | 915.2 MiB |
| medium 几何 on | 2 实例 → **1 共享 wrapper** | **460.1 MiB**（**省 455.1 = 恰好 1 wrapper**） |

- **每 wrapper workspace 实测**：小几何 1.41 MiB；medium 几何 **455.1 MiB**（debug 直测 ctor delta = 1,479,680 字节同口径印证小几何）。
- 池逻辑全过：on 相位 4/2 实例只创建 1 个 wrapper、全部共享同一对象、池 size=1、wrapper workspace 完整（moe_output (maxt, K) + static_workspace）。
- **显存预算**: peak allocated 915.2 MiB << 5GB ✓；对生产零干扰（只读挂载 + 独立进程）。

**数值正确性（包络判据）**：

| 测试 | 结果 |
|---|---|
| 输出有限性（合法 sf 构造） | PASS（nan=0，maxabs=6.34） |
| 内核固有非确定性边界（两个**独立** wrapper 同输入） | 实测 max&#124;diff&#124; = 0.03125 ≈ 2 ULP bf16 @ maxabs 6.34（**基线属性，与共享无关**——该内核存在 ULP 级 run-to-run 非确定性） |
| 同 wrapper 连续运行 | ≤ 边界 PASS |
| **workspace 残留无影响**（同 wrapper 被 layerA 数据污染后再算 layerB，clone 口径） | 0.03125 ≤ 边界 PASS（残留效应为零——差异全部来自内核自身非确定性） |
| **共享 wrapper vs 独立 wrapper 同输入** | 0.03125 ≤ 边界 PASS（**共享不引入任何额外误差**） |
| 共享 wrapper 双入口连续运行 | ≤ 边界 PASS |

**量级推算**【推断】：medium 实测 455 MiB@maxt=1024 与 P2 口径（生产 maxt=4096 时 0.3-0.65GB/层）自洽。43 层去重预期节省 **~13-27GB**（W4A4 路线），直接解除 W4A4 实验 "+42GB 吃光 KV" 的结构性封锁。

### 4.2 harness 三轮迭代记录（方法论沉淀）

1. **v1 NaN**：sf 直接用随机 fp32 做 mma 转换——对照 routea W4A4 修正版 harness（`/tmp/_routea_work/probe22_corrected_w4a4.py`）确认 sf 必须是 **float8_e4m3fn + swizzle_blockscale swizzled 布局**（零权重 NaN 数与随机权重完全相同证明与数据无关）。已修复。
2. **v2 view 陷阱**：`wrapper.run()` 返回 `_moe_output` 的 view——污染运行直接改写"参照输出"本身，残留测试必须 **clone**。已修复。
3. **v2/v3 内存 delta 异常**：wrapper 显存块**延迟释放**（phase 间 delta 被上一相位 wrapper 的延迟 free 抵消，on 相位 delta 显示 0.0）——phase 末**绝对读数**才是可靠信号（B_end = 3.9+1.4 恰为 1 wrapper ✓；M_on_end = 5.0+455.1 ✓）。已在报告与脚本注释中沉淀此口径。
4. **bit-exact 判据不可达**：内核自身 ULP 级非确定性（两个独立 wrapper 也差 2 ULP）→ 验收改为包络判据（共享 ≤ 固有边界）。这也修正了 §2.5 验证要点 3 的 "bit-exact" 预期——**窗口 W-3 的 golden 对比应采用同样包络判据**。
5. **GPU 余量间歇性 OOM**：14:12-14:28 期间 host available 12-14G 时新 CUDA 上下文建立间歇失败（生产负载波动，14:24 与 14:31 两轮成功均为余量瞬时回升）；14:16 生产容器被外部重启（RestartCount=0、ExitCode=0，serve 参数已带 `long_prefill_token_threshold: 4096`——非本任务动作，疑似团队协调的 threshold 采纳重启），等待 healthy 后完成验证。**教训入窗口清单：L2/W-1 类测试需重试循环或选生产低负载时段**。

## 5. L3 窗口验证清单（列出，不执行）

**前置**：与 threshold 4096 采纳窗口合并跑（用户已指示三项研究重点推进，threshold 4096 昨日实测 PR +12% 待采纳）。

| 步骤 | 内容 | 判据 |
|---|---|---|
| W-0 | `deploy_ws_dedup.sh` 四节点落盘（overlay + env=0 + bind + .bak + checker）——**不重启不生效** | checker 全过、md5 四节点一致 |
| W-1 | ~~跑 L2 v2~~ **已完成（16/16 PASS，见 §4）**——窗口如需复跑：`l2_gpu_test.py`（~1GB 显存峰值，~1 分钟；GPU 余量紧张时用重试循环，见 §4.2.5） | 已达成 |
| W-2 | W4A4 实验容器（nvfp4 插件路径）挂载补丁 + `VLLM_B12X_SHARED_WRAPPER=1`，跑全 43 层真实模型启动 | 启动日志 pool size == 1；`nvidia-smi`/`torch.cuda.memory_reserved` 对比 off：减 ~ (43-1)×每层 workspace（P2 口径 13-27GB）；cudagraph 捕获正常；无 workspace 相关报错 |
| W-3 | 数值回归：W4A4 on/off 各跑同一组 golden 请求，逐 token 输出对比 | 差异 ≤ 内核固有 ULP 级非确定性边界（**包络判据**，非 bit-exact——见 §4.2.4；边界可先在 W4A4 几何下用双独立 wrapper 实测标定） |
| W-4 | 组合矩阵（与 threshold 4096 窗口合并）：{threshold 1024/4096} × {wrapper off/on} × {W4A16/W4A4} 的 PR 四档 + DE C1/C12 | W4A4+on 不劣化吞吐；重点看 W4A4 的 KV/显存余量恢复后 prefill 表现（微基准 M≥4096 1.32× 能否兑现到 E2E） |
| W-5 | 吞吐 A/B：`VLLM_B12X_SHARED_WRAPPER=0/1` 同窗口交错各 ≥10 轮（吸取研究一方差教训） | on 臂吞吐无回归（±2% 内）；权重视图重算 host 开销不可见 |
| 回滚 | start 脚本 `VLLM_B12X_SHARED_WRAPPER=1`→`0` 重启，或恢复 `.bak-wsdedup-20260822` | 即时回滚 |

## 6. 交付物清单

本地 `deliverables/engineering-assurance/ws-dedup-patch-2026-08-22/`（同步 `node01:/tmp/_ws_dedup/`）：

| 文件 | 说明 |
|---|---|
| `flashinfer_b12x_moe.py` | 补丁后 overlay 文件（374 行，md5 见服务器） |
| `src_orig/flashinfer_b12x_moe.py` | 镜像原件留档（295 行，md5 5f887cf38153f6b4c6a8b66b7a1888fb） |
| `patch.diff` | 统一 diff（133 行） |
| `deploy_ws_dedup.sh` | 四节点部署脚本（overlay 分发 + env/bind 注入 + .bak + checker；幂等；**未执行**） |
| `l1_test.py` | L1 验证脚本（24 项，容器内已跑全过，`l1_run.log`） |
| `l2_gpu_test.py` | L2 最终版脚本（16/16 PASS，`l2_final.log`；含 sf e4m3+swizzle 构造、包络判据、绝对读数口径） |
| `l2_debug.py` | L2 debug 脚本（NaN 定位 + 池路径 delta 直接实测） |
| `probe_wrapper.py` / `probe_import.py` / `probe_import_mem.py` | 源码核实与内存探针 |
| `ws-dedup-patch-2026-08-22.md` | 本报告 |

## 7. 工作量与风险

**工作量**：补丁+验证+脚本本日完成（≈0.5 人日）；窗口验证（L3 清单 W-2~W-5）预计 0.5-1 窗口日（与 threshold 4096 窗口合并可摊薄）。较 p1-p2 估计的"2-4 天（含 W4A4 复测）"提前——因为 W4A4 全量复测归入窗口。

**风险**：
1. 【低】层间并行（LayerOverlap/async TP 未来引入）需按 lane 分池——b12x 文档已预警；当前 fork 无此路径，键中无 lane 维度，引入时需扩展。
2. 【低】EP 启用时 num_local_experts 已在键中（当前 wrapper 不支持 EP，`_supports_parallel_config` 挡住）。
3. 【低】共享后权重视图单槽缓存逐层失效 → 模块级 `_WEIGHT_CACHE` 兜底（µs 级），W-5 步骤吞吐 A/B 实证。
4. 【低→已基本关闭】W4A4 数值正确性：L2 小几何已证共享不引入超出内核固有边界的误差（残留效应为零）；剩余不确定性仅在 43 层真实权重 + cudagraph 捕获路径（W-2/W-3 窗口收尾）。
5. 【低】部署脚本 sed 注入依赖 start 脚本锚点行（`VLLM_TRITON_MLA_SPARSE` / `nccl-ringonly` 挂载行）——若未来脚本重构需同步更新锚点；checker + `bash -n` 已内置防护。

## 8. 局限声明

- L2 数值正确性基于小几何合成权重（真实 43 层权重的 golden 对比归窗口 W-3）；包络判据取代 bit-exact 的依据是内核固有 ULP 级非确定性实测（两个独立 wrapper 同输入差 2 ULP，基线属性）。
- medium 几何 workspace 455 MiB 是 max_num_tokens=1024 的实测；4096 的绝对值引用 P2 口径（0.3-0.65GB/层）双轨标注。
- GPU 余量 12-14G 时新 CUDA 上下文间歇性 OOM（生产负载波动）——窗口期同类测试需重试循环或选低负载时段（§4.2.5）。
- 本补丁只解锁 W4A4/W4A8 路线；生产 W4A16 路径（B12xExperts）不受影响也无所增益（其 scratch 已走全局 WorkspaceManager 共享）。
- 观察记录（非本任务动作）：14:16 生产容器被外部重启，serve 参数已含 `long_prefill_token_threshold: 4096`（threshold 4096 采纳疑似已被应用——建议 team-lead 与相关执行方核对）。

---

> 本报告由工程保障团队协作生成，关键决策请由人类工程负责人复核。
