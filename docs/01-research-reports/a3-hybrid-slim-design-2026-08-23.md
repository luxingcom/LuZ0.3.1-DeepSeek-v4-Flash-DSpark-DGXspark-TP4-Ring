# A3 —— W4A4 hybrid 双表示瘦身设计研究报告

- **执行人**：阿奇（Archi）· 系统架构师（a3-slim-researcher）
- **日期**：2026-08-23
- **方法**：纯源码级设计研究（一次性 `--entrypoint bash` 容器读 fork 镜像 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0` 源码 + 张量形状推导；**不占 GPU、不动生产**，研究容器已清理）
- **口径标注**：【实证-源码】= 容器内代码行号级验证；【实证-数据】= wsdedup L3 四臂数据再分析；【推导】= 基于源码与几何的推导（未上机）；【估算】= 数量级估算
- **输入**：wsdedup-l3-combo-2026-08-23.md（M1-M4 四臂数据）、p1-p2-research-2026-08-22.md §2、upstream-tracking-2026-08-22.md

---

## 0. 执行摘要

**一句话结论：hybrid 双表示的 +34.5 GiB 不是技术必然，是 b12x 集成层的一个"政策选择"——把 W4A16 侧从破坏性就地重打包切换到 b12x 0.15.3 已内置的非破坏性 native（modelopt）权重路径，W4A4 wrapper 即可直接以 view 共享同一份 payload，hybrid 内存回到 full W4A4（M3）水平 45.3 GiB，KV 从 1.53M 恢复到 ~5.5M tokens。**

1. **旧结论"payload 共享不可行"被源码级推翻**：b12x 0.15.3 内置 `prepare_w4a16_e8m0_native_weights`（prepare.py:770-838），文档明言这是"**memory-safe path for GLM serving that needs A4 prefill and A16 decode in the same process**"——即上游本来就为"双 kernel 共存共享一份权重"造好了路径。当前破坏性来自集成层 `_w4a16_weight_layout_for_source()`（tp_moe.py:759-771）对 serving 一律返回 `"packed"` 的政策选择，而非 kernel 不支持。
2. **W4A4 wrapper 侧天然零拷贝**【实证-源码】：flashinfer `B12xMoEWrapper`（nvfp4 模式）对权重只缓存 **view**（`permute(1,2,0)`，moe_dispatch.py:447-448 注释 "view, no copy"），键为 data_ptr；我们的几何（intermediate/rank=512，128 对齐）不触发 padding 拷贝分支。**它需要的正是 checkpoint 原生 [E,N,K/2] 布局**——与 native W4A16 共享同一份 payload 完全兼容。
3. **推荐路线 b′（native 共享）**：hybrid weight 79.82 → **~45.3 GiB**（=M3 水平），KV 1.53M → **~5.48M tokens**（-8.5% vs M1 基线 6.0M）；叠加路线 e（util 0.80→0.85）可达 **~6.2-6.6M ≥ M1**。工作量 **2-4 天**（插件 + monkeypatch，~120 行，含一个测试窗口）。
4. **主要风险（必须 A/B）**：native 布局下 W4A16 主 GEMM 的权重 staging 路径不同（`_stage_b_tile_modelopt_native` vs packed 的扁平 cp_async），decode 中批（M 9-96，C12 场景）与 mid-M prefill chunk 性能未知。回退梯子：env 开关回滚 / MIN_M 调整 / 退回 M3。
5. 路线 a（W4A4 消费 packed 布局）**不可行**（nibble 级 gather 置换非步长可表达）；路线 c（out-of-place repack）**零增益**；路线 d（UMA/host 驻留）**被带宽算术否决**（每步全量权重读取，273 GB/s LPDDR 下 +124ms/步量级）。

---

## 1. 内存账精确分解

### 1.1 生产几何（TP4 每 rank）【实证-源码 + 推导】

模型 config（/models/config.json，P1/P2 已核验）：43 个 MoE 层，全部同构 E=256、topk=6、moe_intermediate_size=2048、hidden=4096。TP4 无 EP：256 专家全驻每 rank，intermediate 按 4 切分（512/rank）。

| 张量 | 形状（每层每 rank） | 字节 | 43 层合计 |
|---|---|---|---|
| w13（gate+up 融合）payload | [256, 1024, 2048] uint8 | 512 MiB | **21.5 GiB** |
| w2 payload | [256, 4096, 256] uint8 | 256 MiB | **10.75 GiB** |
| 专家 FP4 payload 小计 | — | 768 MiB/层 | **32.25 GiB** |
| E8M0 scale（K/32）s13+s2 | [256,1024,128]+[256,4096,16] | 48 MiB | **2.02 GiB** |
| E4M3 K/16 swz store（W4A4 用） | 线性 96 MiB/层（无 padding：1024%128=0、4096%128=0、256%4=0、32%4=0） | 96 MiB | **4.03 GiB** |
| 其他权重（dense FP8 linear/MTP/embedding） | M1 40.5 − 32.25 − 2.02 | — | **6.23 GiB**【推导】 |
| W4A4 wrapper workspace（池化后） | L2 实测 per-wrapper 0.543 GiB，池 size=1 | — | **0.54 GiB**【实证-数据】 |

### 1.2 四臂 + 目标形态的逐项账【实证-数据 + 推导】

| 组成（GiB/rank） | M1 W4A16 | M3 full W4A4 | M4 hybrid（现状） | **b′ native hybrid（目标）** |
|---|---|---|---|---|
| 专家 FP4 payload | 32.25（b12x packed，**就地**重打包） | 32.25（native，就地半交换） | 32.25（packed 就地）**+ 32.25（W4A4 副本）= 64.50** | **32.25（native，单份共享）** |
| E8M0 scale | 2.02（就地 packed） | 2.02（native 保留） | 2.02 + 2.02（副本）= 4.04 | 2.02（W4A16 packed grid，新分配；原 E8M0 派生后释放） |
| E4M3 swz store | — | 4.03 | 4.03 | 4.03 |
| wrapper（池化） | — | 0.54 | 0.54 | 0.54 |
| 其他权重 | 6.23 | 6.23 | 6.23 | 6.23 |
| 残差（alpha/unit/workspace 张量、allocator 粒度） | ~0 | ~0.25 | ~0.45 | ~0.25 |
| **weight 合计** | **40.5**（实测） | **45.32**（实测） | **79.82**（实测） | **~45.3 ± 0.5**【推导】 |
| **KV tokens** | 6.00M | 5.48M | **1.53M（-74.5%）** | **~5.4-5.5M**【推导】 |

**账目闭合核验**【推导】：
- M4 − M3 = 34.50 GiB（实测）vs 计算值 32.25（w13/w2 副本）+ 2.02（E8M0 副本）= 34.27 → 残差 0.23，与 M3 自身残差 0.25 同量级 ✓
- M3 − M1 = 4.82 GiB（实测）vs 计算值 4.03（swz）+ 0.54（wrapper）= 4.57 → 残差 0.25 ✓
- **结论：hybrid 的 +34.5 GiB = 纯粹的 W4A4 自持副本（插件 w4a4_experts.py:110-115 的 `torch.cat(...).contiguous()` + `w2.clone()` + scale 副本），没有其它隐藏膨胀。** b12x 重打包本身是**等大小置换**（见 §2.2），不贡献膨胀。

### 1.3 KV 换算系数【实证-数据 + 推导】

M1：53.18 GiB / 6.00M tokens ≈ **8.86 KiB/token**（fp8 KV + indexer cache）。节点 DRAM 121 GiB（`free -g` 实测），util 0.80 → 预算 96.8 GiB（M1 CUDA 侧 ~96.4 ✓ 自洽）。每 +0.02 util ≈ +2.42 GiB ≈ **+274K tokens**（我的算术）；任务书给定实测系数 +443K/0.02（口径可能不含 indexer cache）。取区间表述。

---

## 2. 机制链源码取证（路线评估的共同基础）

### 2.1 W4A4 侧：wrapper 对权重是纯 view，无拷贝无重打包【实证-源码】

`flashinfer/fused_moe/cute_dsl/b12x_moe.py`（镜像内 0.6.15 混合版，b12x 树未被打补丁）：
- `run()`（:472-）每次调用接收 `w1_weight/w2_weight` 传入；nvfp4 模式构造 `weight_key = (data_ptr × 6)`（:572-580）缓存 `_weight_views`（:614-617）——**缓存的是视图不是副本**。
- 权重 view 构造 `moe_dispatch.py:_get_weight_views`（:391-）：
  - `w13 = w1_fp4.permute(1, 2, 0)`（:447-448，注释 "Permute [E, w1_rows, k//2] -> [w1_rows, k//2, E] (view, no copy)"）——kernel 经单一 TMA descriptor 消费 **checkpoint 原生 [E,N,K/2] 布局**。
  - scale 侧 `convert_sf_from_mma_layout(...).contiguous()`（:430-441）：与插件的 `flashinfer_convert_sf_to_mma_layout` 构成 view 往返（to_mma 是 strided view，utils.py:419-421；from_mma 的逆 permute 落回原始连续存储，`.contiguous()` 为 no-op，utils.py:455）——M3 内存账 4.57≈4.82（含残差）反向印证无第二次 scale 拷贝【推导】。
  - intermediate%128≠0 才触发 `_pad_intermediate_to_tile` 拷贝（:582-605）；我们 512%128=0，不触发。

### 2.2 W4A16 侧：重打包是等大小置换，破坏性是开关行为【实证-源码】

`b12x/moe/fused/w4a16/prepare.py`：
- `_repack_4bit_no_perm`（:279-411）：`packed_shape = (K/16, N/64, 128)` int32（:300-301）= **N×K/2 字节 = 与原始 [N, K/2] uint8 等大小**。内容是 16×64 nibble tile 内的 gather 置换（`source_index/source_shift` 计算 :349-366，`pack_idx` 重排 :348）——**数据同一份、顺序重排、字节数不变**。
- `_repack_weight`（:414-470）：`reuse_input_storage=True` 时 `packed = weight.view(torch.int32).reshape(packed_shape)`（:429-433）就地覆写；`False` 时新分配 buffer（保留原始）。
- E8M0 scale 同理：`_pack_e8m0_k32_scales`（:236-283）有 `reuse_input_storage` 双形态。

**fork 桥接层调用点**（vllm `experts/b12x_mxfp4_moe.py`）：
- `B12xExperts._get_or_prepare_fp4_moe_weights`（:628-678）→ `prepare_b12x_fp4_moe_weights(source_format="fp4_e8m0_k32", w13_layout="w31", prepare_w4a16=True, **reuse_input_storage=True**)`（:660-676）。
- `process_weights_after_loading`（:541-570）随后 `_release_w4a16_source_scales` + `_release_w4a16_source_weights`（:567-568）——layer 参数置空，packed 存活于 prepared dataclass 引用。**生产 W4A16 = 单份 payload（就地 packed）+ 2.02 GiB 就地 packed scale**，账目与 M1 40.5 自洽 ✓。

### 2.3 关键翻案证据：b12x 0.15.3 内置非破坏性 native 路径【实证-源码】

`prepare.py` 两条 native API（均在 `__all__` 导出，:1031-1041）：
- `prepare_w4a16_modelopt_native_weights`（:647-726）：docstring（:653-657）——"**This is the memory-safe path for GLM serving that needs A4 prefill and A16 decode in the same process. It keeps the checkpoint FP4 tensors resident instead of materializing a second full W4A16 packed copy.**"
- `prepare_w4a16_e8m0_native_weights`（:770-838）：**正是我们的源格式**（fp4_e8m0_k32）。返回 `W4A16ModelOptWeights(weight_layout="modelopt")`，权重 `w13=w13_fp4` 原样保留；新分配 packed E8M0 grid（~2.02 GiB，:812-824）；"the kernel applies the matching source_n_rotation to the native weights"（:806-808）——w13 行序差异在 kernel 内处理。
- **kernel 级支持完整**：`kernel.py`（5290 行）主 GEMM 有 `weight_layout == "modelopt"` 的 CuTeDSL 编译期分支（:2143-2151 load 路径、:2567-2585 `_stage_b_tile_modelopt_native` staging）；运行入口 `run_w4a16_moe` 从 prepared 对象读取 layout（:4780-4796）。
- **小 M direct（micro）路径只支持 native**：`_small_m_direct_supported`（:3810-3846）要求 `weight_layout == "modelopt"`（:3832-3833），M≤8（`_W4A16_SMALL_M_DIRECT_MAX_M = 8`，:93）——我们的 C1 decode（M=8，dspark n=7）恰好命中。
- **plan 层支持**：`_prewarm_w4a16_planned_launches` 接受 weight_layout 参数，fused launch 键含 `(weight_layout, scale_format, token_count)`（tp_moe.py:3074）；运行时 `b12x_moe_fp4` binding 从 prepared 读 layout（tp_moe.py:5319-5324）。
- **为什么上游 serving 不用它**：`_w4a16_weight_layout_for_source`（tp_moe.py:759-771）docstring 明言——"All serving W4A16 sources are repacked to packed; small-M decode is served by the TC-decode path on that same packed object, **so no native modelopt copy is needed**... (The modelopt layout + micro decode kernel remain reachable for offline/benchmark use via the prepare API, just not auto-routed here.)"——即**上游在"packed + TC-decode（默认关）"与"native + micro"之间选了前者作为 serving 默认**，因为单 kernel 场景下 packed 是复制粘贴最优。**该选择的前提（只有一个 kernel）在我们 hybrid 双 kernel 场景下不成立。**

### 2.4 插件侧副本的成因【实证-源码】

`<INSTALL_DIR>/nvfp4/plugin_a1/routea_plugin_a1/w4a4_experts.py`（池化版，与 .bak 原版仅差 `_get_pooled_wrapper`）：
- hybrid 分支（:110-115）：`self._w13 = torch.cat([w13[:, n:], w13[:, :n]], dim=1).contiguous()`（**整份副本** + 行序交换 [w1;w3]→[w3;w1]）、`self._w2 = w2.clone()`、scale 副本——共 32.25+2.02 GiB/.rank。
- full 分支（:98-109）：**就地半交换、零拷贝**——证明 W4A4 kernel 消费的正是 native 布局（M3 golden 4/4 逐字一致背书）。
- `process_weights_after_loading`（:150-162）：hybrid 先 `_derive_w4a4`（做副本）再 `super().process_weights_after_loading(layer)`（就地销毁原始）——**副本存在的唯一原因是 super 的破坏性 prepare**。

### 2.5 两 kernel 的行序约定（实现要点，非阻断）【实证-源码】

- W4A4（flashinfer）：up-first [w3;w1]（插件注释 :7-8 + full 模式就地交换的数值验证）。
- W4A16（b12x）：kernel 原生消费 [gate;up]（fork `_w13_layout()` 返回 "w31"，b12x_mxfp4_moe.py:524-528 注释）。native 路径下 "w13" 标签（up/gate 物理序）由 kernel 施加 `source_n_rotation` 处理（prepare.py:806-808）。
- **方案**：payload 就地半交换为 up-first（同 full 模式），W4A16 native 用对应标签（kernel 内旋转）；标签↔物理序映射须在 L1 单测核对（golden 门兜底）。

---

## 3. 五条瘦身路线逐条评估

### 路线 a：反向派生（W4A4 直接消费 b12x packed 格式）——**不可行**

- **机制核查**【实证-源码】：b12x packed 布局 `(K/16, N/64, 128) int32` 是 16×64 nibble tile 内的 **gather 置换**（prepare.py:349-366 的 `source_index/source_shift/pack_idx` 计算）。W4A4 kernel 经 TMA descriptor + stride view 消费权重（moe_dispatch.py:399-448）——**TMA/stride 只能表达仿射映射，不能表达 tile 内 gather**。
- 要让 W4A4 消费 packed 格式 = 重写 flashinfer CuTeDSL kernel 的 B 操作数 layout atom 以匹配 b12x 的私有 tile 置换——周级 kernel 工程、与 flashinfer 上游完全敌对（上游无此布局）、并锁死未来升级。
- **判定：否决。**（附带结论：b12x packed → 原生的**逆向**转换数学上存在——置换可逆——但只能靠 gather kernel 物化，等于每次调用重打包，性能不可接受。）

### 路线 b：派生顺序反转 —— **修正为 b′（native 共享）后成立，是推荐路线**

- **原设想的机制核查**【实证-源码】：wrapper 确实"构造期只读一次"——但读出的是 **view（别名）而非拷贝**。若先建 view 再让 b12x 就地重打包，view 会静默读到已覆写的 packed 数据（**数据损坏而非报错**，golden 门能抓住）。原设想按字面执行 = 死路。
- **修正（b′）**：问题不在顺序，在**破坏性本身**。把 W4A16 侧切到 §2.3 的 native 路径（`prepare_w4a16_e8m0_native_weights`，非破坏、新分配的只有 2.02 GiB packed scale grid），payload 保持 native —— W4A4 wrapper 的 view（§2.1）与 W4A16 native kernel **共享同一份 32.25 GiB payload，双表示零副本**。
- **内存数学**【推导】：weight 79.82 → **~45.3 GiB**（§1.2 表末列；与 M3 相同的物理构成——packed scale grid 1:1 替换被释放的 E8M0 原始 scale）；KV 1.53M → **~5.4-5.5M**（-8.5% vs M1，同 M3）。
- **实现代价**：**2-4 天**（含测试窗口）。改动面：
  1. 插件 hybrid 分支改为 full 式就地半交换 + 直接引用（~20 行）；
  2. `process_weights_after_loading` 替换 super 调用为 native prepare 并塞入 `_prepared_fp4_moe_by_dtype`（~60-80 行，env 门控 `VLLM_MOE_W4A4_NATIVE=1`）；
  3. monkeypatch `b12x.integration.tp_moe._w4a16_weight_layout_for_source` 对 fp4_e8m0_k32 返回 "modelopt"（~10 行）——**必须**，否则 plan 层预编译 packed launch 与运行时 native prepared 失配（tp_moe.py:3259 vs :5319，miss 即 RuntimeError）；
  4. 不调用 `_release_w4a16_source_weights`（payload 共享）；E8M0 原始 scale 在双侧派生完成后释放（省 2.02 GiB）。
- **风险**：
  - **头号：native 主 GEMM 性能**（M 9-3071 档：decode 中批 C12 M≈96、mid-M prefill chunk、尾 chunk）。`_stage_b_tile_modelopt_native`（kernel.py:2567-2585）是逐 tile 计算索引的 staging，vs packed 的扁平 `cp_async4_shared_global`（:2568-2572）——**上传实测未知**。这正是上游 serving 弃 native 的原因。
  - micro direct（M≤8）换路径：现生产 M=8 走 packed 主 GEMM（direct-topk 上限 M=6、TC-decode 默认关），native 后走 micro direct——专为小 M 设计的路径，**可能更好也可能更差**，需实测；`_W4A16SmallMDirectKernel.is_supported` 对本几何（E=256/inter=512）需 L1 核验。
  - w13/w31 标签映射（§2.5）——golden 门兜底。
  - native 路径 serving 使用率低（上游标注 "offline/benchmark"）——边界情况暴露风险，靠扩大质量门覆盖。
- **回退梯子**：env 开关一键回滚（off 路径构造 kwargs 逐字一致的原则沿用 phase3b 池化插件）；native decode 回归超阈值 → 调 MIN_M / 退回 M3 / 维持 M1。
- **W4A4 侧零风险**：prefill M≥3072 路径与 M4 完全相同（wrapper + view + swz scale），M4 的 PR +8.9% 数据直接迁移【推导】。

### 路线 c：改 b12x prepare 不销毁（out-of-place）——**零增益，否决**

- **机制核查**【实证-源码】：`reuse_input_storage=False` 是现成参数（prepare.py:1964、tp_moe.py:1964-1997）——重打包到新 buffer、保留原始。W4A4 拿原始（零副本），但 W4A16 需要 packed **副本** 32.25+2.02 GiB。
- **内存数学**：总占用 = 原始 + packed 副本 = 与现状 hybrid 完全相同的 34.3 GiB 额外。**纯零增益**（只是把副本从 W4A4 侧挪到 W4A16 侧）。
- 上游 TPMoEWorkspacePool/bind_shared_arena 只覆盖 **workspace/scratch**（tp_moe.py:227-296 区域 + plan 层），**不覆盖权重 payload**——`_W4A16_PACKED_WEIGHT_CACHE` 是按 data_ptr 缓存 prepared 对象的机制，不是共享机制【实证-源码】。
- **判定：否决**（无任何场景下优于 b′）。

### 路线 d：CPU/UMA 侧驻留——**被带宽算术否决**

- **机制**：GB10 是 121 GiB UMA，host malloc 与 cudaMalloc 同一 LPDDR；把 34.5 GiB 副本放 CUDA 分配器之外，vLLM 的 KV 预算（基于 CUDA 侧 free）即放大。
- **致命算术**【估算】：UMA 无独立 GPU 带宽——LPDDR ~273 GB/s 共享。MoE 权重是**每步全量热读**：
  - W4A4 prefill M=4096：24576 routed rows 覆盖全部 256 专家 → 每步读 33 GiB → **~124 ms/步** 纯权重带宽（现步时 ~600ms 量级 → +20% 回归）；
  - W4A16 decode M=96：topk=6×96=576 激活，期望命中专家 ≈256×(1-e^(-96×6/256))≈229（~90%）→ 同量级灾难。
  - L2（~128MB）对 33 GiB 工作集无效。
- host 页上的指针能否进 CuTeDSL `make_ptr(gmem)`/TMA descriptor 亦未经验证（次级风险）。
- **判定：否决**（UMA 的"统一"恰恰意味着没有第二条带宽路径可薅）。仅当未来出现"冷权重"（如稀疏激活场景）才值得重开。

### 路线 e：KV 预算重平衡（util 0.80→0.82/0.85）——**成立，作为叠加项**

- **输入**【实证-数据】：节点 DRAM 121 GiB（`free -g`）；aicad 容器族占用 MB 级（docker stats 实测）；util 0.80 下 M1 CUDA ~96.4 GiB，**仍有 ~20 GiB DRAM 余量**。
- **数学**【推导 + 任务书系数】：每 +0.02 util = +2.42 GiB ≈ +274K tokens（自算，8.86 KiB/tok）/ +443K（任务书实测系数）。区间表述：**+0.02 → +274-443K；+0.05 → +0.7-1.1M**。
- 单独用于现状 hybrid：1.53M → 2.2M（@0.82）~2.6M（@0.85）——仍 -56% vs M1，**不足以独立救活 hybrid**。
- 叠加于 b′：5.48M → 5.8-5.9M（@0.82）~ **6.2-6.6M（@0.85）≥ M1 6.0M**。
- **代价**：纯配置（`--gpu-memory-utilization`），0.5 天含验证。**风险**：留给 OS/aicad 的余量从 ~20 GiB 降到 ~14 GiB（@0.85）——当前 aicad 占用极小，可行，但需在窗口实测 OOM 边界；建议 0.82 起步。
- **判定：采纳为 b′ 的叠加项**（先 0.82，视 DRAM 余量实测再上 0.85）。

---

## 4. 推荐与路线图

### 4.1 排序

| 排序 | 路线 | 判定 | 瘦身后 weight / KV | 工作量 |
|---|---|---|---|---|
| **1** | **b′ native 共享** | **推荐立项** | **45.3 GiB / ~5.48M**（@util 0.80） | 2-4 天 |
| 2 | e KV 重平衡（叠加） | 采纳 | 45.3 GiB / 5.8-5.9M（@0.82）→ 6.2-6.6M（@0.85） | 0.5 天 |
| 3 | d UMA 驻留 | 否决（带宽算术） | — | — |
| 4 | c out-of-place | 否决（零增益） | — | — |
| 5 | a 反向派生 | 否决（kernel 重写） | — | — |

### 4.2 最短可行路径（b′ 实施骨架）

**阶段 0（L1，无 GPU 容器，0.5 天）**
1. 插件新增 `VLLM_MOE_W4A4_NATIVE=1` 分支：hybrid 派生改 full 式就地半交换 + 直接引用；native prepare 封装（构造 `B12XPreparedFP4MoEWeights(w4a16=W4A16ModelOptWeights)` 塞入 `_prepared_fp4_moe_by_dtype`）。
2. monkeypatch `_w4a16_weight_layout_for_source`（env 门控，off 时恒等）。
3. 单测：①标签↔物理行序映射正确性（w13/w31 双向）；②`prepared.weight_layout=="modelopt"` 且 `prepared.w13.data_ptr()==layer.w13_weight.data_ptr()`（**零拷贝断言**）；③plan 侧（monkeypatch 后）与运行时 layout 一致；④off 路径 kwargs 逐字一致（沿用 phase3b 纪律）；⑤`_W4A16SmallMDirectKernel.is_supported` 本几何核验（需 GPU 容器，若 L1 无法则挪到窗口首项）。

**阶段 1（测试窗口，0.5-1 天，复用 phase3b 臂测试基建）**
- 臂 N1：native hybrid（mode=1 + native=1 + 池 on）。判据：
  1. **内存门**：weight ≈ 45.3±1 GiB（vs 79.82）；KV ≥ 5.3M tokens；
  2. **质量门**：golden 4 prompt 逐字一致 + logprob 对齐（M1 参考）；needle 降级为 smoke（A6 结论）；
  3. **性能门**：PR 四档 ≥ M4-3%（2999 基准）；**DE C1/C12 step_eff ≥ M1-3%（19.7/93.3 基准）**——此为 native W4A16 decode 的关键判据；
  4. mid-M 补测：M=96/512/2048/3071 的 W4A16 微基准（native vs packed），量化 `_stage_b_tile_modelopt_native` 的 staging 代价【新增项，直接回答头号风险】；
  5. cudagraph 三档捕获完整。
- 臂 N2（可选）：N1 + util 0.82。
- 回滚链：env 三开关（W4A4/NATIVE/SHARED）全独立，<10 分钟。

**阶段 2（若 N1 过门）**：MIN_M 扫描（3072 vs 2048 vs 4096，mid-M 微基准数据驱动）+ 正式化（A2 池化整合合并为单一补丁 + logger 修复一并做）。

### 4.3 收益论证：b′ 相对 M3（full W4A4，已可用形态）值多少投入

| 维度 | M3 full W4A4 | M4 hybrid（现状） | **b′ native hybrid** |
|---|---|---|---|
| weight | 45.32 GiB | 79.82 GiB | **~45.3 GiB** |
| KV | 5.48M（-8.5%） | 1.53M（-74.5%，不可产） | **~5.48M（-8.5%）**（+e 可 ≥6.0M） |
| PR 4K | 2982（+8.3%） | 2999（+8.9%） | 2999 持平（W4A4 路径不变）【推导】 |
| DE C1 step_eff | 19.0（-6.4%） | 19.7（-3.0%） | **目标 ≥19.7**（待 A/B，见风险） |
| DE C12 step_eff | 85.4（-9.1%） | 93.3（-0.6%） | **目标 ≥93.3**（待 A/B） |

- **投入 2-4 天换取**：decode 归一 +6~9% 恢复（M3 的主要剩余成本）+ KV 与 M3 持平 + prefill +8.9%——即 **M4 的全部性能收益、M3 的全部内存代价**。若 decode 业务（DE C12 类负载）占比可观，ROI 明确为正。
- **不确定性对冲**：native W4A16 decode 若回归 >3%，b′ 的核心卖点消失——此时 M3 仍是可用形态，b′ 退化为"已验证的设计储备"（2-4 天沉没成本可控）。**建议 N1 臂的 DE 门设为 go/no-go 决策点。**
- 与并行工作零冲突：b′ 只动插件 + monkeypatch，不触碰 flashinfer 包（与 FI 0.6.16 生产替换窗口正交；0.6.16 的 `B12xMoEWrapper` API 签名不变已被 P1/P2 核验）；不占生产。

### 4.4 上游对齐注记

- b′ 使用的全部是 b12x 0.15.3 **已导出的公开 API**（`prepare_w4a16_e8m0_native_weights`、kernel weight_layout 分支、plan weight_layout 参数）——未来 b12x 升级（1.x）漂移风险低；monkeypatch 点（`_w4a16_weight_layout_for_source`）是唯一私有符号依赖，升级时需复核。
- 更彻底的长期形态是向上游提 issue/PR：为 `prepare_b12x_fp4_moe_weights` 增加 `weight_layout="modelopt"` 显式参数（消掉 monkeypatch）——建议窗口验证通过后再做。

---

## 5. 证据索引

| 项 | 位置 |
|---|---|
| b12x native prepare API | 镜像 `b12x/moe/fused/w4a16/prepare.py:647-726`（modelopt native docstring :653-657）、`:770-838`（e8m0 native，含 micro scale 共享 :826-836）、`__all__ :1031-1041` |
| 等大小重打包 + 就地覆写 | `prepare.py:279-411`（`_repack_4bit_no_perm`，packed_shape :300-301，gather 索引 :349-366）、`:414-470`（`_repack_weight`，reuse view :429-433）、`:236-283`（E8M0 scale pack 双形态） |
| serving 布局政策（翻案核心） | `b12x/integration/tp_moe.py:759-771`（`_w4a16_weight_layout_for_source` 恒返 "packed" + docstring）、`:3259`（plan 侧调用）、`:5319-5324`（运行时从 prepared 读取） |
| fork 桥接调用链 | vllm `experts/b12x_mxfp4_moe.py:519-520`（fp4_e8m0_k32）、`:524-528`（w31）、`:628-678`（prepare 调用，reuse=True :675）、`:541-570`（process + release :567-568） |
| W4A16 kernel native 分支 | `b12x/moe/fused/w4a16/kernel.py:2143-2151`、`:2567-2585`（`_stage_b_tile_modelopt_native`）、`:3810-3846`（micro direct 仅 modelopt）、`:92-93`（M≤6/M≤8 常量）、`:4780-4796`（run 入口）、`:4937`（TC-decode 仅 packed） |
| W4A4 wrapper 零拷贝 | flashinfer `fused_moe/cute_dsl/b12x_moe.py:569-617`（weight_key/views 缓存、padding 分支 :582-605）、`blackwell_sm12x/moe_dispatch.py:391-470`（permute view :447-448、scale 转换 :430-441）、`cute_dsl/utils.py:354-423`（to_mma strided view）与 `:425-467`（from_mma 逆 view） |
| W4A4 插件副本成因 | 宿主 `<INSTALL_DIR>/nvfp4/plugin_a1/routea_plugin_a1/w4a4_experts.py:110-115`（hybrid 副本）、`:98-109`（full 就地半交换）、`:150-162`（process 顺序）、`:7-13`（派生链文档） |
| 内存/性能数据 | wsdedup-l3-combo-2026-08-23.md §2（四臂数据：40.5/45.32/79.82 GiB，KV 6.0/5.48/1.53M，DE step_eff） |
| 节点 DRAM 余量 | node01 `free -g`（121 GiB total）+ `docker stats`（aicad 族 MB 级）——2026-08-23 实测 |
| 几何 | /models/config.json：43 层 × E=256/topk=6/I=2048/H=4096（P1/P2 核验） |

## 6. 局限声明

1. **native W4A16 主 GEMM 性能未上机**（头号开放问题）——本报告的 KV/内存结论不依赖它，但 b′ 的 go/no-go 依赖 N1 臂 DE 门。
2. 内存账残差 ±0.3 GiB（alpha/unit/workspace 小张量与 allocator 粒度未逐项枚举）；b′ 目标值 45.3±0.5 GiB 为推导口径。
3. micro direct 路径对本几何的 `is_supported` 未核验（需 GPU 容器）；w13/w31 标签映射以 L1 单测 + golden 门兜底。
4. 路线 e 的 +443K 系数沿用任务书口径，与自算 +274K 并列呈区间；@0.85 的 DRAM 余量需窗口实测。
5. 全部源码取证经一次性 `--entrypoint bash` 容器（无 GPU、只读），研究容器 `a3-research` 已删除；生产与 FI 替换窗口零干扰。

---

*本报告由工程保障团队（系统架构师）生成；N1 臂 go/no-go（尤其 DE step_eff 门）请由人类工程负责人复核决策。*

---

# §b′ 实施记录（2026-08-23 · bprime-impl）

- **执行人**：阿奇（Archi）· 系统架构师（bprime-impl）
- **任务**：b′ native 共享路线实现与 L1 验证（B 方案，用户已批准）
- **基座**：池集成版插件（/tmp/_wsdedup_l3/w4a4_experts_pooled.py，phase3b L1/L3 已验证资产；现役 <INSTALL_DIR>/nvfp4/plugin_a1/ 已回滚原版 c2d1de3d，未动）
- **口径标注**：【实证-CPU】= 一次性 CPU 容器（--entrypoint bash，无 GPU）；【实证-GPU】= 共享 GPU 一次性容器（--gpus all --rm，生产并行运行中）；【推导】= 未上机推算

## B.1 交付物

| 项 | 位置 |
|---|---|
| 插件（服务器，新目录，生产零接触） | `<INSTALL_DIR>/nvfp4/plugin_a1_bprime/`（routea_plugin_a1_bprime/{__init__,w4a4_experts}.py + setup.py；md5 20b977fc/913e3ae8 与本地一致） |
| 插件（本地副本） | `deliverables/engineering-assurance/bprime-impl-2026-08-23/plugin_a1_bprime/` |
| L1 测试 | `test_l1_cpu.py`（68 项）、`test_l1_gpu.py`（12 项）；debug 取证 `debug_t2.py / debug_micro.py / debug_micro2.py` |
| 测试资产（服务器） | `/tmp/_bprime/`（含全部日志） |

规模：w4a4_experts.py 相对池化基座 **+126 行**（净增，含文档），`__init__.py` 87→**150 行**（native 政策 patch + micro 防护 + 文档）。A3 §4.2 预估 ~120 行总量同级。

## B.2 实现细节与设计偏差

三处规格改动全部落地 + 两处勘察后新增（共 5 个改动点）：

1. **hybrid 分支改 full 式就地引用**（规格 1）✓：`_derive_w4a4` 中 `mode==2 or (mode==1 and native)` 共用就地半交换分支（w13+scale 半交换、`self._w13=w13`、`self._w2=w2` 零拷贝、E4M3 派生同 full）。
2. **native prepare 封装**（规格 2）✓：`_prepare_native_w4a16()` 调 b12x 公开 API `prepare_w4a16_e8m0_native_weights(w13_layout="w13")`，包 `B12XPreparedFP4MoEWeights(source_format="fp4_e8m0_k32", w13_layout="w13", w4a16=native)` 塞入 `_prepared_fp4_moe_by_dtype[params_dtype]`；**fail-fast 断言** `weight_layout=="modelopt"` 且 `w13/w2.data_ptr()==payload`（b12x 升级破坏共享契约时在加载期暴露而非静默副本）。`process_weights_after_loading` native 分支编排顺序 = prepare → `_prewarm_b12x_route_pack` → `_release_w4a16_source_scales`（E8M0 原始 scale 双侧派生后释放）→ `_release_w4a16_source_weights`（payload 由插件 `_w13/_w2` 与 `prepared.w13/w2` 持有，layer 参数清空仅去引用）→ `_maybe_release_cuda_cache`（T1b 实证该顺序）。
3. **monkeypatch `_w4a16_weight_layout_for_source`**（规格 3）✓：包级 `install()` 内安装（native=1 且 mode=1 才装），对 `fp4_e8m0_k32` 返 `"modelopt"`、其他格式透传原函数；幂等（`_bprime_patched` 标记）；进程生命周期不撤销。作用域正确性：EngineCore 子进程经 vllm.general_plugins entry point 逐进程加载插件（与 plugin_a1 同机制），plan 侧（tp_moe.py:1334/:3259）与运行时（:5319 读 prepared）在模型加载前即一致。
4. **【新增·偏差】`_w13_layout()` 覆写**（A3 未列）：native hybrid 下 payload 物理序已半交换为 up-first，须声明 `"w13"`（up_gate）；fork 基座返回 `"w31"`。apply 内 plan/run 两处调用点（b12x_mxfp4_moe.py:776/:835）与 prepared.w13_layout 一致性由此保证。off 路径恒走 `super()._w13_layout()`（行为逐字等价）。
5. **【新增·偏差】强制 `B12X_W4A16_SMALL_M_DIRECT=0`**（native 激活时，显式覆盖并告警）：L1 GPU 实证 e8m0×micro direct 输出错误数值（见 B.4），正确性优先；native M≤8 走主 GEMM（实证与 packed 逐位相等）。

env 三开关独立门控 ✓：`VLLM_MOE_W4A4`（0/1/2）/`VLLM_B12X_SHARED_WRAPPER`（池，overlay 侧）/`VLLM_MOE_W4A4_NATIVE`（0/1，仅 mode=1 生效，full+native 显式告警忽略）。native off 时与池化版 plugin_a1 行为等价（B.3 T1 实证）。

其他实现差异（无行为影响）：logger 改挂 `vllm.` 命名空间（phase3b 工程发现 #1 的修复——docker logs 可见）；日志行增加 native 标志与 resident-extra 修正（native hybrid 的驻留增量 = E4M3 store 而非全量）。

**部署警示（窗口阶段必须遵守）**：本插件与 plugin_a1 读同一 env `VLLM_MOE_W4A4` 且都 patch `backend_to_kernel_cls`——**两插件不得同时经 entry point 激活**。N1 臂部署须先移除/禁用 plugin_a1 的 vllm.general_plugins 注册（pip uninstall routea-plugin-a1 或环境隔离），否则类解析顺序不确定。

## B.3 L1 验证结果【实证-CPU：68 PASS / 0 FAIL；实证-GPU：12 PASS / 0 FAIL】

**T1 off 路径零行为变化（CPU）**：native∈{未设, "0"} × mode∈{0,1,2} 全组合，bprime vs 池化版逐项一致——wrapper 构造 kwargs 逐一相同、派生张量（w13/w2/sf13/sf2）`torch.equal`、别名关系一致、分派决策（M=1/8/96/3071/3072/4096）一致、`_w13_layout()=="w31"` 一致；native=1 时仅 mode=1 行为改变（payload 别名共享 + layout="w13" + 就地半交换，值与 packed 派生 `torch.equal`）。方法论沿用 phase3b（stub wrapper 记录 kwargs + 对照构造）；口径注记：`flashinfer_convert_sf_to_mma_layout` 与 `swizzle_blockscale` 内部 `.cuda()` 在 CPU 容器以 no-op stub/patch 替代（两模块同 stub，对比有效性保持）。

**T2 native prepare 契约（CPU，`_make_workspace` stub）**：`weight_layout=="modelopt"`、`source_format=="fp4_e8m0_k32"`、w13/w2 零拷贝（data_ptr 断言）、packed E8M0 grid 新分配（dtype `float8_e8m0fnu`，形状 [E,K/32,N] 与源码一致）、micro_\* 与主 grid 同一对象（存储兼容单 grid）、行旋转等变性成立。**更强的契约事实（实证）**：`native(swapped, "w13").w13_scale == native(unswapped, "w31").w13_scale` **逐位相等**——row_rotation 在 prepare 内与 payload 半交换相消，两布局声明产出同一 logical grid，物理差异完全由 kernel `source_n_rotation` 补偿。G3 的逐位相等是该设计的运行时印证。

**T1b 编排顺序（CPU）**：native 分支调用序 = prepare → prewarm → release_scales → release_weights → cache（记录器实证）；prepared 契约/kwargs 正确；payload 以插件引用存活；off 路径不触碰 native prepare。

**T3 monkeypatch（CPU 子进程）**：native on → `fp4_e8m0_k32`→`"modelopt"`、`modelopt_nvfp4`→`"packed"`（透传）、幂等、插件 installed；native off → 不安装、政策原样 `"packed"`、`B12X_W4A16_SMALL_M_DIRECT` 未触碰；full+native → native 显式忽略（告警）+ 政策原样。

**G1 显存账（GPU 共享容器，小几何 E=32/H=512/I=256）**：native derive 增量 **0.8MB**（= E4M3 store ~0.4MB + mma/杂项；payload 6.3MB **零副本**，增量/payload=12.5%）；packed hybrid 对照增量 **7.1MB**（payload+E8M0 副本 6.7MB + store）；**省 6.3MB = 恰为 payload 全量**。生产外推【推导】：43 层 × 32.25GiB payload 零副本 → hybrid weight 79.82 → ~45.3GiB（与 A3 §1.2 账目吻合），全量内存门留窗口验证。

**G2 共享证据（GPU）**：native prepare 后 `prepared.w13.data_ptr() == 插件 _w13.data_ptr() == layer payload`（同一内存，零副本直接证据）；packed grid 独立分配。

**G3 数值 A/B（GPU，run_w4a16_moe 直调，随机权重+路由）**：packed(w31, gate-first, 破坏性重打包) vs native(w13, up-first 就地半交换)——**M=64 与 M=8 双侧主 GEMM 输出逐位相等（max_rel=0.00e+00）**；补充几何扫描（E=64/H=2048/I=512/topk=6、E=128 同构）M=8 亦逐位相等，M=4 孤立元素差异 ≤2 ULP bf16（p50=p99=0，累加序差的量级，包络内）。判据（≈2 ULP）全过。

**G4 生产几何 micro 静态核验（GPU import 级）**：E=256/H=4096/I=512/topk=6/e8m0_k32 下 `is_supported`：m=1/4/8 True、**m=7 False**（odd-M 不支持→回落主 GEMM）；packed 布局恒不走 micro（与现产一致）。注：因 B.4 缺陷插件已强制关闭 micro，本项降级为背景事实。

共享 GPU 纪律执行：实测 CUDA free 仅 1.6-2.0GiB（生产 worker TP0 占 ~100.3GiB，低于任务书预期 12-14GB 余量），小几何（≤100MB 级）受控执行未影响生产；更大几何（E=256 生产级）不可行，列入窗口清单。

## B.4 新发现：e8m0 × micro direct 上游缺陷（b12x 0.15.3）【实证-GPU】

- **现象**：native（modelopt）+ e8m0_k32 + M≤8 路由 micro direct 时输出错误：E=32/H=512/I=256 → 值全错（max_rel 2.9e3）；E=64/H=2048/I=512 与 E=128 → **100% NaN**；M=4 同坏。
- **根因甄别**：非"双重旋转"（w31 未旋转 grid 同样 p50 rel=1.12 全错）→ micro kernel 对 e8m0_k32 scale 格式的消费本身错误（疑似按 K/16 swizzled nvfp4 grid 解读 e8m0 packed grid）。`_small_m_direct_supported` 接受 `scale_format="e8m0_k32"` 属 **host-check 误放行**——与 A3 §2.3 "native 路径 serving 使用率低、边界情况暴露风险"的预判一致。
- **处置**：插件 native 模式强制 `B12X_W4A16_SMALL_M_DIRECT=0`（显式覆盖+告警）。native M≤8 走主 GEMM——**已证与 packed 逐位相等**（多几何），正确性无损；micro 的性能收益本就未知（A3 已标注"可能更好也可能更差"），现明确放弃。
- **附带影响**：b′ 的 decode 路径 = native 主 GEMM（M 9-3071 同）——DE 性能门的对象统一为 `_stage_b_tile_modelopt_native` staging，窗口 N1 臂 mid-M 微基准仍是头号开放问题。
- **建议**：向上游报 issue（e8m0×micro 数值缺陷 + is_supported 放行缺口）；修复并验证后再评估启用。

## B.5 已知限制与窗口清单

1. **native 主 GEMM 性能未测**（A3 头号风险，维持）：需窗口 mid-M 微基准（M=96/512/2048/3071，native vs packed）。
2. **全量内存账/KV 未上机**：小几何比例实证 + 生产算术外推 45.3±0.5GiB；窗口 N1 内存门确认。
3. **e2e 未跑**（需窗口）：golden 4 prompt、PR 四档、DE C1/C12、cudagraph 三档捕获。
4. **生产级几何 G3 未跑**（GPU 余量不足）；已由 4 组小几何 + 逐位相等 + 契约级证据（B.2/T2）覆盖到逻辑层。
5. **两插件互斥**（B.2 部署警示）：窗口部署流程必须先禁用 plugin_a1 entry point。
6. m=7 odd-M micro 不支持：已被强制关闭 micro 化解，仅作背景事实。
7. L1 CPU 口径注记：sf mma conversion/swizzle 的 `.cuda()` 以 stub/no-op 替代（B.3）；GPU 侧为真实路径。

## B.6 窗口测试方案要点（N1 臂，供主理人排期）

- **部署**：四节点禁用 plugin_a1 entry point → 部署 plugin_a1_bprime（pip install -e 或 PYTHONPATH）→ env `VLLM_MOE_W4A4=1 VLLM_MOE_W4A4_NATIVE=1 VLLM_B12X_SHARED_WRAPPER=1`（+overlay 已在位）→ restart。
- **判据**（沿用 A3 §4.2 阶段 1）：内存门 weight 45.3±1GiB（vs 79.82）+ KV ≥5.3M；质量门 golden 4 逐字一致（logprob 对齐 M1）；性能门 PR 四档 ≥ M4-3%（2999 基准）+ **DE C1/C12 step_eff ≥ M1-3%（19.7/93.3 基准——现为 native 主 GEMM 全 M 段，含 M≤8）**；mid-M 微基准（native vs packed，M=96/512/2048/3071）量化 staging 代价；cudagraph 三档完整。
- **回滚**：env 三开关全独立（NATIVE=0 即回池化 hybrid 形态；W4A4=0 回生产基座），<10 分钟；恢复 plugin_a1 entry point 即回现役资产。
- **观察项**：启动日志应见 `b' native layout policy installed` + `SMALLM_DIRECT forced to 0` + 每层 `b' native W4A16 prepared`（logger 已挂 vllm 命名空间，docker logs 可见——phase3b 取证缺口已补）。

## B.7 结论

b′ 三处规格改动 + 两处勘察新增（`_w13_layout` 覆写、micro 强制关闭）全部实现并过 L1：off 路径与池化版逐项等价（回滚安全性实证）、native prepare 契约与零拷贝共享成立（CPU+GPU 双口径）、**native 主 GEMM 与 packed 逐位/≤2ULP 数值等价**、显存增量实证 = scale store only（无 payload 副本）。**具备窗口测试条件**；唯一实质设计变更 = decode M≤8 放弃 micro direct（上游缺陷，以已证等价的主 GEMM 替代），DE 性能门对象不变。
