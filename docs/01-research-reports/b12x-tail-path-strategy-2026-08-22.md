# B12X 尾路径策略报告：kernel 形式深析 + Triton 缺陷归因 + 三选项（2026-08-22）

**执行**: b12x-analyst（系统架构师）· 纯源码/文档分析，零 GPU 占用
**任务来源**: 主理人指派（用户定调："B12X 是优秀的内核……merged 桶的作用未能发挥是需分析改进之处；b12x+dspark 技术方案可移植"）
**数据源**: 生产镜像 `0.2.1-v026.0-b12x-recovered-20260820` 内源码（b12x 0.15.3 包 + vllm fork，一次性容器逐文件精读）、`<INSTALL_DIR>/nvfp4/plugin_merged/`（triton_moe.py v2.1 + merged_experts.py v2.1）、routeb-merged-mainline / routeb-merged-kernel-design / routea-plugin-p1 / large-batch-analysis 四份必读文档
**口径标注**: 全文逐条标注【源码实证】（给行号）/【推理】（机制链推导）/【估算】（数量级折算）

---

## 0. 一页结论（供裁决）

1. **B12X W4A16 快的根源不是单点技巧，而是 12 项形式的乘积**：自适应 route-packing（M 块 8/16/32/48/64 按填充率 0.9 目标选择）+ 持久 CTA + cp.async 4 级流水 + **PTX 位重排打包解包（每 4 个 FP4 仅 7 条整数指令，寄存器内完成）** + 每 k 组一次的 scale 寄存器化 + mma.m16n8k16 fragment 对齐布局 + **单次 launch 融合 FC1→SwiGLU→FC2→topk-sum**（grid 级 barrier + split-k 锁归约）+ 小 M 直通微内核。
2. **Triton 4.3× 差距的定量归因（修正昨日口径）**：解包**不是 LUT 查表，而是逐元素 fp32 算术链（含 exp2）**——triton_moe.py:69-79 实证；payload 以 **LDG.U8 字节粒度加载（offs_k//2 非仿射索引 → 编译器无法向量化，每字节重复加载 2×）**；scale 逐元素 gather（w13 swizzle 逐元素 int64 乘加 + w2 每 32 k 重复加载 32×）；固定 BM=16 使 prefill 每 expert 权重**重复读 2×**（ceil(24/16)=2 块 vs B12X ceil(24/32)=1 块）；FC1/SwiGLU/FC2/scatter 四段不融合（[cap,N13] bf16 三次 HBM 往返）。分解：**prefill ≈ 2×（权重重复读）× ~2.2×（内核效率）；decode ≈ 1×（权重流量与 B12X 相同！）× ~4×（纯内核效率）**。
3. **选项 B（merged 热桶 + B12X 尾）的 KV 代价可从"悬崖"改造成"旋钮"**：+35GB 双表示中，payload 克隆是**结构性不可省**（tcgen05 NVFP4 硬件 MMA 布局与 b12x ldmatrix 软解包布局物理不兼容，双向都无法零拷贝消费对方——源码级证实），但**双表示可以按层选择性实施**：每层双表示代价 ≈ 0.81GB ≈ 14 万 KV tokens。hash-3 层 = +2.4GB（KV 5.6M ✅）；全 43 层 = +35GB（KV → ~0，**且昨日文档"KV 2.2M"的算术有误**，见 §4.3）。
4. **推荐：B-lite（选择性双表示 + B12X 尾 + merged 仅高覆盖层）为主线，C（纯 B12X 基线）为保底与对照臂，A（Triton 重写）降级为备选**。判据门 = 另一代理的 GSM8K 真实路由覆盖：给出敏感性公式 PR 收益 = f(覆盖×层选择×加速比)，若 top-k 层加权收益 <+3% e2e 或 KV <4M → 放弃 B 维持 C。与调度器扩 M（路线①）**正向叠加**：扩 M 同时改善 B12X 尾效率与 merged 桶覆盖率/档位。

---

## 1. B12X W4A16 kernel 形式清单（"充分借鉴"的底册）

生产调用链【源码实证】：`Mxfp4MoEMethod → B12xExperts.apply`（b12x_mxfp4_moe.py:785-874）→ `b12x.integration.tp_moe.b12x_moe_fp4`（tp_moe.py:5112, w4a16 分支 5279-5389）→ `run_w4a16_moe`（b12x/moe/fused/w4a16/kernel.py:4744-5271）。权重一次性准备：`prepare_b12x_fp4_moe_weights(prepare_w4a16=True, reuse_input_storage=True)`（b12x_mxfp4_moe.py:660-676）→ `prepare_w4a16_packed_weights`（prepare.py:523+）。

**生产 per-rank 形状**【推理，与 33GB payload 锚点交叉验证】：E=256，K(hidden)=4096，I_rank=512（N13=1024，K2=512，N2=4096），MoE 层 43。w13 payload 537MB/层 + w2 268MB/层 = 805MB/层 × 43 ≈ 34.6GB ✓（与 routea-p1 "+33GB/rank" 一致；config hidden_size=4096 / moe_intermediate_size=2048 / n_routed_experts=256 / num_experts_per_tok=6 实证，TP4 分片 I）。

### 1.1 形式清单（12 条）

| # | 形式 | 机制（源码） | 为什么快 / GB10 适配点 |
|---|------|------------|----------------------|
| F1 | **自适应 route-packing 到稠密 M 块** | `select_route_block_size_m`：按 avg_routes/expert 选块 {8,16,32,48,64}，目标填充率 0.9（host.py:140-145）；Triton 排序/前缀和打包 kernel（route_pack.py:56-121） | 把 ragged 的 per-expert M_e 变成**规则稠密 GEMM 块**，decode M=96 → 块 8、prefill M=1024（M_e≈24）→ 块 32。关键效应：**ceil(M_e/块) 决定每 expert 权重读几遍**——块 ≥ M_e 时每 expert 权重只读 1 遍（带宽最优）|
| F2 | **持久 CTA + 静态占用模型** | grid = SMs × blocks_per_sm；blocks_per_sm 由每特化实测寄存器数（118-255，kernel.py:124-140 `_W4A16_REGS_SM121`）+ smem 足迹（:189-209）推出，上限 4/SM（:212-252）；工作按 (route_block × n_tile) tile 静态分发 + 尾部均衡（:919-933） | SM121 无 tcgen05 大 tile MMA 的场合下，用**占用率模型**保证带宽型负载满 SM；避免逐 tile launch |
| F3 | **cp.async 4 级 smem 流水** | `_STAGES=4`（kernel.py:81）；A(bf16)/B(打包 fp4)/scale 三路 smem 流水，cp_async4（16B/线程）+ commit/wait_group（:2396-2701） | GB10 统一内存带宽 ~273GB/s 是硬瓶颈——16B 粒度异步拷贝 + 4 级深流水把访存与 MMA 重叠到极限 |
| F4 | **寄存器内 PTX 位重排解包** | `packed_dequant_e2m1x4_to_bfloat2x2`：7 条整数指令（and/shr/or）把 1 个 b32（4 个 FP4）变成 2×bfloat2x2（fp4.py:1963-1998）；E4M3/E8M0 scale 同样打包×4 解码（kernel.py:698-717） | **每元素 ~1.75 条整数指令、零 exp2/查表/访存**——E2M1→BF16 是纯位重排（bf16 = s<<15 \| e<<8 \| m<<6 \| 0x3F00 的位域映射）。这是与 Triton 版差距的单点最大来源 |
| F5 | **scale 每 k 组一次、寄存器驻留** | `_load_b_scale_registers`：每 16/32 个 k 取 1 次 scale（ld_shared_v2_u32），解包后以 half2/bf16x2 乘法施加于解包 fragment（kernel.py:2160-2176, 1731-1732） | scale 访存缩小 16-32×；乘法在 bf16x2 packed 域完成（2 元素/指令）|
| F6 | **MMA fragment 对齐的寄存器布局** | b_frag 仅 [2,2] uint32 = **4 寄存器/线程**，恰好等于 mma.m16n8k16 B 操作数 fragment（kernel.py:1684, 1730-1748）；A 走 ldmatrix（:2020-2036） | 解包输出**直接就是** MMA 操作数布局——零 shuffle/零 smem 中转；寄存器压力极小（118-255 regs 全档可控）|
| F7 | **单 launch 全融合 FC1→SwiGLU→FC2→topk-sum** | `compile_w4a16_fused_moe`：一个持久 kernel 内 grid 级 barrier（workspace 原子计数 arrive-wait，:3479-3560）串联两级 GEMM；split-k 用 fp32 c_tmp + 锁（locks_i32，:929-933）+ 最终归约（:5199-5230） | intermediate [M×topk, N13] **全程不出寄存器/smem 级**——省掉三次 HBM 往返（Triton 版正是死在这里）；swiglu_limit clamp 融合在 FC1 epilogue |
| F8 | **小 M 直通微内核（M≤8）+ TC-decode 变体** | `_W4A16_SMALL_M_DIRECT_MAX_M=8`：MoEMicroKernelBackend 单 kernel + workspace barrier_count/epoch（kernel.py:93-105, 4836-4924, micro.py）；TC-decode 把 topk-sum 折进 FC2 store epilogue，用 red_add_global_bf16x2 原子累加（:4926-4983, 5165-5172） | 极小 batch 时免 route-pack + 免独立 sum launch；GB10 单流 decode 延迟最优路径 |
| F9 | **权重一次性预重排到 ldmatrix 直消费布局** | `_repack_weight` → `_repack_4bit_no_perm`：[N,K/2] u8 → [K/16, N/64×128] int32，int32 内 8 nibble 按 mma fragment (tc_row, offsets[0,1,8,9]) 排列（prepare.py:291-407, 434-470）；`reuse_input_storage=True` 时**就地覆写**原 payload（:434-437） | 让 F3/F4/F6 成立的物理前提；代价 = 一次性 repack（加载期）+ **原格式被销毁**（KV 双表示问题的根源，§4）|
| F10 | **M 档位化 tile 配置选择器** | small/large batch 两套候选 (tile_k, tile_n, threads)（kernel.py:141-150），`_select_tile_config` 按 N/K 整除 + smem 拟合 + blocks_per_sm 最大化选优（:284-346）；vllm 侧还留了 env 强制覆盖（b12x_mxfp4_moe.py:177-332） | 小 M 用 128×128/64×128，大 M 用 64×256——tile 随问题形状自适应而非写死 |
| F11 | **预规划 launch + CUDA graph 安全** | `_w4a16_preplanned_launches` 按容量预编译复用（tp_moe.py:5363-5368）；route-pack 全部 2^n 容量档在加载期预热（b12x_mxfp4_moe.py:402-472）；捕获期禁分配的显式防御（:379-392） | decode 捕获零 JIT/零动态 shape |
| F12 | **workspace 容量规划** | c_tmp fp32 = min(N×route_slots, SMs×4×bm×256)（host.py:183-196）——split-k 归约缓冲有界 | scratch 不随 M 无界膨胀，KV 预算友好 |

### 1.2 decode 115 t/s 档（M=96）的机制拆解【推理+源码】

decode 步 batch=96（12 seqs × MTP 8）→ 576 routes / 256 experts，avg M_e=2.25 → 块 8（F1）→ 每活跃 expert **权重恰好读 1 遍**（ceil(2.25/8)=1）。每层权重流量 = 活跃 expert 数 d × 3.14MB（w13 2.1 + w2 1.05）；d≈100 时 ≈ 314MB/层 × 43 层 ≈ 13.5GB/步 @ 273GB/s ≈ 49ms——与 C1 步时 ~41ms 量级吻合（large-batch §2.4 同口径）。**机制本质：B12X 在 M_e≪768 拐点时不是靠算力，而是靠"每 expert 权重单次读尽 + 满带宽"维持 decode**——这正是 Triton 版做不到的（§2）。

---

## 2. Triton W4A16 grouped 缺陷定量归因（对照 B12X 形式）

对象：`<INSTALL_DIR>/nvfp4/plugin_merged/routeb_merged_plugin/triton_moe.py` v2.1（226 行，与 /tmp/_routea_work 版一致）。调用形态：`triton_moe()`（:175-226）每层每步 = host 排序链（argsort/align_pairs ~10 个小 kernel）+ `grouped_linear` w13 + 独立 `swiglu` + `grouped_linear` w2 + 加权 scatter（index_add）。

### 2.1 缺陷根因表

| # | 缺陷 | 机制（源码行号） | 对照 B12X | 估算贡献 | Triton 内可修性 |
|---|------|----------------|-----------|---------|--------------|
| D1 | **payload 字节粒度加载，无法向量化** | `tl.load(w_payload_ptr + w_row0[:,None]*(K//2) + offs_k[None,:]//2)`（triton_moe.py:67-68）——`//2` 索引非仿射 → 编译器不能证明线程内连续 → **每元素一条 LDG.U8**；且每字节含 2 nibble 被偶/奇 k 各读一次 → **2× 冗余字节流量** | F3/F9：cp.async 16B/线程，一次带 16 nibble | 内存指令数 ~16-32×，有效带宽损失 ~2-4×【估算】 | ✅ 可修：按 int32 加载 packed（4 字节/元素→位移），Triton int 运算向量化 |
| D2 | **逐元素 fp32 算术解包（含 exp2）**——**修正昨日"LUT 解包"口径** | `code/mag_c/sgn/e2/m1/mag` 链 + `tl.exp2(e2-1.0)*(1.0+m1*0.5)`（:69-79）——每元素 ~10-14 条 fp32 指令（exp2 是 SFU 多周期） | F4：每 4 元素 7 条整数指令（位重排，无 exp2） | ALU 吞吐损失 ~4-8×/元素【估算】；与 D1 叠加成主导瓶颈 | ✅ 可修：E2M1→BF16 是位域映射，可用 `(code&8)<<12 \| (code&7)<<8 \| 0x3F00` 类 int 重排（~4 条/元素，无 SFU）|
| D3 | **scale 逐元素 gather** | w13：swizzle 逆运算逐元素构造 [BN,BK] int64 偏移 + 字节 gather（:81-88，每元素 4 次乘加）；w2：`offs_k//32` 使同一 scale 字节被 **32× 重复加载**（:89-93） | F5：每 16/32 k 一次 ld，寄存器驻留，bf16x2 域施加 | 地址 ALU + 冗余访存 ~1.3-1.6×【估算】 | ✅ 可修：scale 按 k 组加载进寄存器 + 广播 |
| D4 | **寄存器压力 → 低占用** | 解包链同时存活多个 [BN=64, BK=128] fp32 中间张量（code/mag/sf/w 等）≈ 每线程 64-192 regs → 溢出/占用坍缩；num_warps=4 | F6：b_frag 4 寄存器/线程，fragment 对齐 | 有效带宽再损 ~1.3-1.5×【估算】 | ⚠️ 部分可修（int 化 + 小 BK），但 Triton 布局不受控，达不到 fragment 对齐 |
| D5 | **固定 BM=16（+ BN=64/BK=128，:105）** | prefill M=1024：M_e≈24 → ceil(24/16)=2 块/expert → **每 expert 权重读 2 遍**（B12X 块 32 → 1 遍）；填充率 24/16 vs 24/32 | F1/F10：块 8-64 自适应 | **prefill 权重流量 ×2**【源码实证+推理】；decode ceil(2.25/16)=ceil(2.25/8)=1 → 流量持平 | ✅ 可修：M 档位选 BM（16/32/64） |
| D6 | **四段不融合 + HBM 往返** | inter [cap,N13] bf16 写→swiglu 读→写→w2 读（:103-144, 206-215）+ y fp32 化 + scatter（:217-225）；cap 含 padding（prefill M=1024 时 cap≈9984 vs 有效 6144） | F7：单 launch，intermediate 不出片上 | prefill ~1.1-1.2×；decode ~1.1×（CG 下 launch 隐藏但流量不隐藏）【估算】 | ⚠️ 部分可修：swiglu 可并入 w13 kernel epilogue；scatter 可原子化 |
| D7 | **无 split-k / 无持久调度** | grid=(n_blocks, N/BN) 一次性（:116）；K=4096 全长循环 per CTA | F2：持久 CTA + 尾部均衡 + split-k 锁归约 | 小网格时负载不均 ~1.05-1.15×【估算】 | ⚠️ 可用 atomic 模拟，笨拙 |
| D8 | **tl.trans(w) 每步寄存器重排**（:95） | 解包出的 [BN,BK] 需转置喂 tl.dot | F6：布局天生对齐 | ~1.05-1.1×【估算】 | ❌ Triton 自选布局 |
| D9 | host 侧每步排序/对齐链 ~10+ 小 kernel（torch.sort/searchsorted/cumsum，:189-203, 147-172） | prefill eager 下叠加 launch 开销 | route-pack Triton kernel 打包一次 | prefill ~1.05-1.15×【估算】；decode 被 CG 吸收 | ✅ 可合并 |
| D10 | A 行每 n-tile 重载（x 被 N/BN=16 个 pid_n 各读一遍，:64-65） | 无 A 的 smem 复用跨 n | F3 A 进 smem 流水 | 次要（A≪B 流量） | ⚠️ |

### 2.2 4.3× 的定量分解【估算，标注残差】

- **prefill（M=1024，权重流量主导）**：4.3× ≈ D5 权重重复读 **2.0×** × 内核效率 **~2.2×**（D1+D2+D3+D4 叠加，byte-load + exp2 链在低占用下的联合损失）× D6/D7/D9 ~1.0-1.1 残差。
- **decode（M=96，权重流量与 B12X 相同——都是每活跃 expert 读 1 遍）**：4.3× ≈ **纯内核效率 ~3.9×**（D1/D2/D3/D4）× D6 ~1.1。
- 与昨日 "~3-4% vs ~13% 峰值效率"（routeb-mainline §4）自洽：效率比 ≈ 3.3×，加 prefill 侧流量因子后两端都落在 4.3× 附近。
- **主根因排序：D1（字节粒度加载）≥ D2（逐元素 fp32/exp2 解包）> D5（BM=16 权重重复读，仅 prefill）> D3（scale gather）> D4/D6/D7**。

### 2.3 对昨日报告的两处口径修正【源码实证】

1. "逐元素 LUT 解包" → 实为**逐元素 fp32 算术解包（含 exp2）**（triton_moe.py:74-79 无任何查表）。LUT 反而是 B12X 侧的近似形态（PTX 位重排等效于 16 项 LUT 的硬件级实现）。
2. "decode 慢 4.3× 是小 tile 之故" → decode 的**权重流量与 B12X 完全相同**（块数同为每 expert 1 块）；差距全部来自内核实现效率，BM=16 的流量惩罚只在 prefill 显形。

---

## 3. 选项 A：参照 B12X 形式重写 Triton

### 3.1 可复刻 vs 受限

| B12X 形式 | Triton 可复刻性 |
|-----------|----------------|
| F4' 打包解包去 exp2（int 位重排 ~4 条/元素） | ✅ 完全可行（int32 位移/与/或 + bitcast bf16）——**单点最大收益** |
| F1' BM 档位自适应（16/32/64 按 M_e） | ✅ 完全可行（host 侧选择 constexpr 档）——**prefill 权重流量直接 ÷2** |
| D1' int32 向量化 payload 加载 | ✅ 可行（tl.load int32 + 位移解 nibble）——内存指令 ÷16 |
| F5' scale 每 k 组寄存器化 | ✅ 基本可行（小张量加载 + 广播） |
| F7' 融合 swiglu epilogue + 原子 scatter | ⚠️ 部分可行（swiglu 可并入 w13 kernel；topk-sum 原子化有数值顺序差） |
| F3' cp.async 深流水 | ⚠️ 自动（num_stages），深度/谓词控制弱于手写 |
| F6' fragment 对齐寄存器布局 | ❌ Triton 自选布局，解包后仍需 [BN,BK] 级张量物化——寄存器效率 ~2-3× 差于 B12X（D4 残留） |
| F2' 持久 CTA + split-k 锁 | ❌ 语言级不支持（可 atomic 模拟，复杂度失控） |
| F8' 小 M 直通微内核 | ❌ 同上 |

### 3.2 上限与工作量【估算】

- **上限 ≈ B12X 的 50-70%**（仍慢 1.4-2×）：D4/D7/D8 三项结构性残留（寄存器布局、持久调度、转置）合计估算 1.4-2×。即 Triton 重写成功后尾路径仍是净负于 B12X，只是从 -77% 收敛到约 -20~-40% 量级（e2e 口径按 MoE 占 55% 折算）。
- **工作量**：kernel 重写 + M 档位配置 + 融合 epilogue + 正确性矩阵（对齐 padding/哨兵/NaN 类回归全要重验）≈ **1.5-2 人周 + 1 个停机验证窗口**。
- **A 的唯一存在价值**：保住 v2.1 的内存画像（payload 零拷贝 + 仅 +2.9GB scale 壳，KV -0.5M）——即"不想付 KV 代价"时的备选。但既然 B12X 尾是现成的、零重写、零风险且更快，**A 只在"B 的选择性双表示被证明覆盖不足/收益不足、且又确有 merged 收益"的窄场景下才有意义**。

---

## 4. 选项 B：merged 热桶 + B12X 尾——KV 代价深析与优化

### 4.1 +35GB 的构成分解（重新审视 routea-plugin-p1 的"双表示必然"结论）

【源码实证】`prepare_w4a16_packed_weights(..., reuse_input_storage=True)`（b12x_mxfp4_moe.py:675 传入；prepare.py:434-437 就地 `weight.view(int32).reshape(packed_shape)` + 逐 expert 覆写）——**b12x 打包副本不是"另一份分配"，而是原 payload 的就地重排**；同理 scale 也就地 permute（`_pack_e8m0_k32_scales(reuse_input_storage=True)`，prepare.py:237-287）。结论：**"b12x 打包副本"与"原始 payload"无法共享物理存储——routea-p1 的结论在尾部用 B12X 的形态下依然成立**，但原因可以更精确地表述：

> **两种布局物理不兼容，且双向不可零拷贝**：merged DSL GEMM 走 tcgen05 NVFP4 硬件 MMA，要求 B 为 [N, K/2] 行序 vec16 + swizzled SFB（routeb-kernel-design §3.1）；b12x 走 ldmatrix 软件解包，要求 [K/16, N/64×128] int32 nibble-permute（prepare.py:291-407）。前者无法以 TMA box + 受限 swizzle 模式表达后者（nibble 排列是 mma.m16n8k16 fragment 序，非任何硬件 swizzle 模式）；后者反向同理。【推理，源码支撑】

因此 payload 双表示 = +34.6GB（w13 22.1 + w2 12.5）**结构性存在**。**但昨日 +35GB 账本里的 scale 部分（"scale 4.3 + swizzled 4.3"）高估了一倍**【推理，逐项重算】：

| 项 | 昨日 v1 估计 | 重算（源码形状） | 说明 |
|---|------------|----------------|------|
| payload 克隆 | 33GB | 34.6GB | 结构性，不可省 |
| w13 E4M3 k16 swizzled（merged w13 GEMM 用） | 计入 4.3+4.3 | **2.89GB** | E8M0→E4M3 派生，须在 b12x scale 就地 permute **之前**派生 |
| b12x scale | 计入 4.3 | **0GB** | `reuse_input_storage=True` 就地——不新增 |
| w2 E8M0 原始备份（combo 派生源） | — | **0.69GB** | 或从 b12x 打包格式逆 permute 派生（省此项，+0.5 天工作量） |
| w2 combo LRU 缓存（cap 8/层） | — | **≤2.4GB**（实测驻留取决于命中） | cap 可调 4 → 1.2GB |
| **合计** | ~+35GB | **+37-40GB（全 43 层）** | 与 v1 同量级——**全层双表示不省** |

### 4.2 关键优化：双表示按层选择性实施（KV 从悬崖变旋钮）

merged 桶的价值高度集中在**有覆盖的层**（hash 层 top-set 62% vs dense 层长尾 3 token/组合，routeb-design §1；合成语料下 dense 覆盖≈0 已被 ∞ 臂证实）。而双表示代价是**每层独立的线性量**：

- 每层双表示 = w13 克隆 0.537 + w2 克隆 0.268 + w13 E4M3 0.067 ≈ **0.87GB/层**（用"克隆 w2"简化版；若 combo 改从 b12x 格式逆 permute 派生则 0.80GB/层）。
- KV 代价率：0.87GB ÷ 5.78KB/token ≈ **15 万 KV tokens/层**。

| 双表示层选择 | 驻留增量 | KV tokens（基线 6.07M） | 判定（≥5.15M bar / ≥2.6M 底线） |
|------------|---------|----------------------|------|
| k=3（仅 hash 层） | +2.6GB | **≈5.6M** | ✅ 远离 bar |
| k=13（hash + top-10 dense） | +11.3GB | ≈4.1M | ⚠️ 低于 5.15M bar，高于 12 流底线（基准最坏负载 1.57M 有 2.6× 余量）|
| k=26 | +22.6GB | ≈2.2M | ❌ 贴底线 |
| k=43（全层） | +37GB | **≈0-0.9M** | ❌ 不可行 |

**实施形态（plugin v3 = B-lite）**：
1. `process_weights_after_loading`：先 `_derive`（stacked 零拷贝视图 + w13 E4M3 壳，复用 v2.1 代码）→ 对**入选层**克隆 w13/w2 payload → `super().process_weights_after_loading`（b12x 在克隆上就地打包；未入选层不动原始 payload、无 b12x 表示——注意：未入选层的尾/decode 必须走 merged-per-expert？**不**——未入选层无 merged 桶，全量走 super() 需要 b12x 表示…… **修正：decode 需要全部 43 层的 b12x 表示**（decode 每层都要算）。因此"选择性"只能作用于 **merged 侧的 stacked 副本**而非 b12x 侧：正确形态 = **全部层 b12x 就地打包（0 额外），入选层额外克隆 stacked 副本给 merged**（+0.87GB×k）——b12x 打包销毁的是原始，merged 侧持的是克隆。两种方向（克隆给 b12x vs 克隆给 merged）总量相同，但后者把"未入选层"的额外驻留压到 0。w2 combo 从 b12x 打包格式逆 permute 派生可再省 w2 克隆（0.27GB/层）。
2. `apply`：decode / M<MIN_M → 恒 `super()`（B12X，**DE 零风险**）；prefill → 桶分组（复用 v2.1 exact-set 逻辑）→ 命中桶走 `_merged_bucket`（读 stacked 克隆 + tile 表）→ 尾部行 `super()`（B12X 子批量）——**Triton 路径整体删除**。
3. 层选择策略：静态 env（`VLLM_MOE_MERGED_LAYERS`）+ **懒激活**（层首次出现 ≥MIN_M 桶才物化克隆，受全局 GB 预算钳制）——dense 层真实覆盖≈0 时自动不付代价。

### 4.3 昨日文档 KV 估计的算术勘误【估算】

routeb-kernel-design §5 "v1：+35GB → KV 6.08M→~2.2M（-64%）"——按 bytes/token=5.78KB（large-batch R2 定论）折算：35GB ÷ 5.78KB = **6.05M tokens**，即 6.08M − 6.05M ≈ 0，而非 2.2M。该估计与 routea-p1 mini 实测（hybrid +15.5GB → mini KV 85→69GB，即 −16GB KV 对 +15.5GB 驻留，比率 ~1.0）也不自洽。**全层双表示的真实 KV 后果比昨日文档更差（≈0-1M），不可行判定加固**；本报告 §4.2 表为重算口径。最终数字以停机窗口实测 `GPU KV cache size` 为准（large-batch R2 方法论）。

### 4.4 敏感性公式（接另一代理的 GSM8K 路由重采集）

设：α = MoE 占 prefill 步时比例 ≈ 0.55（large-batch §2.2 口径）；f_l = 层 l 的 merged 桶 token 覆盖率（≥MIN_M 的 exact-set 桶）；s = 桶内加速比 = T_B12X(M)/T_merged(M_g)（M_g∈[256,4096] 时 s≈2.4-8，phase C 曲线 157T@256→351T@1536 vs B12X 30-65T@1024【估算，保守取 2.4】）。则：

```
PR_ratio ≈ 1 / [ (1−α) + α·( 1 − Σ_l∈L_dual f_l·(1−1/s)/43 ) ]
e2e 增益 ≈ α·Σ_l f_l·(1−1/s)/43   （小增益线性化）
```

| 场景（真实覆盖待测） | e2e PR 增益【估算】 | KV |
|---------------------|------------------|-----|
| 全 43 层 f=0.27、s=2.4 | **+8~13%** | ❌ 不可行（§4.2） |
| k=13 层 f=0.27、s=2.4 | +2.4~3.9% | 4.1M ⚠️ |
| k=13 层 f=0.5、s=4（hash 高覆盖+大 M_g） | +4.5~7% | 4.1M ⚠️ |
| 仅 hash 3 层 f=0.62 | +1.5~2.5% | 5.6M ✅ |
| 真实覆盖 ≈0（合成语料中位数窗口情形） | ≈0 | —（懒激活自动退化为纯 B12X = 选项 C） |

**判据门（建议写入 plugin v3 验收）**：GSM8K 重采集后按上式计算 k=13/26 两档收益——若 **<+3% e2e 或 KV <4M → No-Go，维持选项 C**；若 hash+top-dense 加权 ≥+5% 且 KV≥4M → 进入 TP4 A/B（Phase 0 判据沿用 batch-analyst §3：PR ≥+10% 保留——注意全层理论 +8~13% 意味着**即便覆盖达标，Phase 0 的 +10% 门槛也接近上界**，建议主理人把门槛按敏感性公式折算后的预期值设定，而非一刀切 +10%）。

### 4.5 与路线①（调度器扩 M）的叠加关系

- 扩 M（threshold 1024→2048）→ 单请求 chunk M 翻倍：B12X 尾 M_e 24→48（仍 <768 拐点，probe 边际改善 ~1.1-1.2×【估算】）；merged 桶 M_g 翻倍 → 覆盖率 f_l 上升（≥256 门槛更易达到）且 s 上升（351T@1536 档）。
- 两路线**乘性叠加于 prefill**（尾更快 × 桶更多更快），decode 不受 threshold 影响（B12X 域）。
- 注意事项（承接 large-batch §4 长线项）：扩 threshold 需重估激活峰值与 ITL；且扩 M 会**稀释 merged 相对优势**（B12X 尾自身变快，s 下降）——组合后应用敏感性公式重算，不宜用单路线数字外推。

---

## 5. 选项 C：回归纯 B12X（基线对照项）

= 当前生产状态（routeb-mainline §5 回退后）。零风险、零工作量；merged 插件封存为库资产（332T kernel + 16/16 正确性矩阵 + 间接寻址改造均可复用）。**C 是 A/B 的对照臂与回退终点**；若 GSM8K 覆盖判据门不过，C 即终态（收益追求转向路线①调度器扩 M 与路线②的 kernel 资产沉淀）。

---

## 6. 推荐与组合策略

**推荐：B-lite（选项 B 的选择性双表示形态）为唯一值得投入工程量的 merged 复活路线，C 为保底，A 降级为条件备选。**

| 路线 | 投入 | 收益上限 | 风险 | 判定 |
|------|------|---------|------|------|
| ① 调度器扩 M（另代理在测） | 配置级 | prefill ~1.1-1.2×（probe 口径） | 激活峰值/ITL 需重估 | 并行推进 |
| ②-A Triton 重写 | 1.5-2 人周 | 尾仍慢 B12X 1.4-2× | 高（正确性矩阵全重验） | **条件备选**（仅当 B 判据门不过且确有 merged 收益时） |
| ②-B B-lite（选择性双表示 + B12X 尾） | 2-3 天编码 + 1 验证窗口 | +2~13% e2e（覆盖敏感） | KV（可控旋钮）/层选择策略 | **主线**（以 GSM8K 覆盖为门） |
| ③ B12X 优化形式移植到 merged 路径本身 | 已部分完成（332T/间接寻址） | — | — | 库资产沉淀 |

**执行序建议**：
1. 等 GSM8K 路由重采集（另一代理）→ 按公式算 k=13/26 收益 → 过判据门才动工 B-lite；
2. B-lite 实施要点：b12x 全层就地打包（0 额外）+ 入选层 stacked 克隆（0.87GB/层）+ w2 combo 逆 permute 派生（省 w2 备份）+ 懒激活 + Triton 路径删除 + decode 恒 super()；
3. 与扩 M 组合测试矩阵沿用 batch-analyst §3 Phase 0/1 框架，臂配置改为「B-lite(k) × threshold{1024, 2048}」2×2；
4. 全程 KV 真值以停机窗口 `GPU KV cache size` 实测记录（large-batch R2 方法论）。

---

## 7. 工件与引用索引

| 项 | 位置 |
|----|------|
| b12x 源码提取（本地分析副本） | 本地 `_b12x_analysis/imgsrc/`（b12x/{moe,cute,integration,quant} + vllm fork 两文件；来自镜像 0.2.1-v026.0-b12x-recovered-20260820 一次性容器，未加 --gpus） |
| Triton 插件源码 | 本地 `_b12x_analysis/plugin/routeb_merged_plugin/`（= 01:<INSTALL_DIR>/nvfp4/plugin_merged/，只读拷贝） |
| 镜像内关键文件 | vllm/.../experts/b12x_mxfp4_moe.py（877 行）/ flashinfer_b12x_moe.py（W4A4 变体，非生产路径）/ b12x/moe/fused/w4a16/{kernel,host,prepare,route_pack}.py / b12x/integration/tp_moe.py / b12x/cute/fp4.py |
| 上游文档 | routeb-merged-mainline-2026-08-22（三臂定论）/ routeb-merged-kernel-design-2026-08-21（§5 KV 估计勘误见本文 §4.3）/ routea-plugin-p1-2026-08-21（payload 共享实测，本文 §4.1 重审）/ large-batch-analysis-2026-08-22（α=0.55、bytes/token=5.78KB、调度器钳制） |
| 模型形状 | mini0731 config.json（hidden 4096 / moe_intermediate 2048 / E 256 / topk 6）+ 33GB payload 锚点交叉验证 |

**环境约束遵守**：全程一次性容器（--rm、无 --gpus）、服务器只读（tar 拷出）、未触碰 <INSTALL_DIR> 任何文件、未干扰 GPU 测试代理（romantic_chatterjee 容器仅读 config.json）。
