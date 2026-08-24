# routeB P3 语义对接报告：生产权重数值验证（2026-08-21）

**任务**: Task #17 · routeB kernel 消费真实生产权重的数值验证
**执行**: Archi（系统架构师）· node01（一次性容器，`<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`，生产镜像 DSL 4.5.2 / torch 2.11）
**判决**: ✅ **P3 通过** —— routeB kernel 以生产 MXF4 权重**字节直配**（零重排）完成数值验证：15/15 shape `rel_err ≤ 4.26e-04`（判据 1e-2），另 4 项跨层/跨 expert 抽查全过。

> **一页摘要（供决策）**
> 1. **重大发现（R13 落锤）**：任务指定的对接对象 `deepseek-v4-flash-0731-nvfp4-hp` checkpoint **是缺陷品**——全部 43 层 × 256 experts 的 scale 张量恒为字节 1（无任何逐块信息），且权重码本与 E2M1 不符（幅值码 6/7 即 4.0/6.0 全模型零出现）。**该 checkpoint 不可被任何 E8M0 反量化路径消费，也不可被 routeB（E2M1 MMA）消费。**
> 2. **正确的对接对象是生产 MXFP4 checkpoint 本体**（`deepseek-v4-flash-0731`）：其 expert 权重 `[N, K//2]` E2M1 打包 + `[N, K//32]` E8M0 逐行 scale **正是 routeB vendored kernel 的 B 侧原生格式**（b_major="k" + SFB 逐行 vec32）——权重**零重排直配**，比 -hp 路径（需转置重排 + 块 scale 扩展）更优。
> 3. **Scale 布局契约（P3 最大风险点，已闭环）**：kernel 对 SFA/SFB 只取数据指针并按 `BlockScaledBasicChunk` atom 布局重排——**必须 atom-swizzle，plain 直传不可行**。重排公式已实证（对官方 `cvt_sf_MKL_to_M32x4xrm_K4xrk_L` 逐字节 100% 一致）。
> 4. 适配器 `routeb_prod_adapter.py` 已交付（本地 + 02:/tmp/routeb_p3 + 01:/tmp/routeb_task12，md5 一致），P4 可直接集成。

---

## 1. 权重实证（任务 §1）

### 1.1 生产 MXF4（deepseek-v4-flash-0731）——真实对接对象 ✅

config：hidden_size=4096，moe_intermediate_size=2048，num_hidden_layers=43，n_routed_experts=256，num_experts_per_tok=6。expert 权重（layer 0 expert 0 实测，safetensors header + 手工解析 + safetensors 库交叉验证一致）：

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `*.w1.weight` | [2048, 2048] | U8 | N=2048(中间维), K=4096(hidden)，E2M1 打包 `[N, K//2]`，**低半字节=偶数 K** |
| `*.w1.scale` | [2048, 128] | **F8_E8M0**（字节存储） | `[N, K//32]` 逐行 E8M0：`W[n,k] = e2m1(码) × 2^(scale[n,k//32]−127)` |
| `*.w2.weight` | [4096, 1024] | U8 | N=4096(hidden), K=2048(中间维) |
| `*.w2.scale` | [4096, 64] | F8_E8M0 | 同上逐行语义 |
| `*.w3.weight` | [4096/…] | U8 | 同 w1 |

scale 健康度：uniq=4，值域 {119..122}（即 2⁻⁸…2⁻⁵），均值 120.6 —— **真实逐块分布**（与 -hp 的恒 1 形成对照）。反量化统计：`W_true` std=0.0245，absmax=0.1875。

**布局/半字节方向三重证据**：(a) 与 -hp 权重（[K, N//2] 方向）相关系数 0.951（转置解释下相关≈0）；(b) nibble 翻转使匹配率 0.165→0.030；(c) v17 kernel 文档约定一致（lo=偶列）。**orientation/nibble 判定可信**。

**这是 routeB 的天然直配格式**：kernel B 侧要求 `[N, K]` k-major 打包（`[N, K//2]`）+ SFB `(N, K//32)` 逐行粒度 —— 与生产格式逐项相同，**权重无需任何重排**（P2 bench 中 4096×14336×4096 的 B 构造路径同构）。

### 1.2 -nvfp4（deepseek-v4-flash-0731-nvfp4）——旁证，非对接对象

modelopt 产物：`weight [N, K//2]` + `weight_scale`（E4M3，[N, K//16] 组 16）+ `weight_scale_2`（**标量** F32）两级 scale。其反量化结果与生产 MXF4 反量化**逐值一致**（std/absmax 全同）——证实模型谱系：`-0731 (MXF4) → -nvfp4 (modelopt NVFP4) → -hp`。该格式组 16 E4M3 scale 与 routeB（E8M0 vec32）不匹配，不直接可消费（未深入，非本次范围）。

### 1.3 -hp（deepseek-v4-flash-0731-nvfp4-hp）——缺陷品 ❌（任务原定对象，重大风险落锤）

**实证结论：不可用。三层证据：**

1. **scale 张量全模型恒 1**（决定性）：抽查 layer 0/21/42 × experts {0,1,2,7,100,255} × w1/w2/w3 共 54 张 scale，全部 `uniq_len=1, value≡1`。手工解析与 safetensors 官方库交叉一致（排除读法错误）。E8M0 语义下 2^(1−127)≈1e-38，反量化幅度偏离真值 ~47×（std 1.157 vs 真值 0.0245）。
2. **权重码本与 E2M1 不符**：幅值码 6/7（4.0/6.0）在 w1 全张量 8.4M 半字节中**零出现**（仅用 0–5）；任何逐块 E8M0 scale（无论 floor(amax/6)、floor(amax/3) 还是 ceil 语义）都必然在部分块产生 4.0/6.0 码——与观测矛盾。数据结构（~3.9% 元素饱和在码 5、每 32×128 块必有饱和元）与"均匀 13 级（±3 步长 0.5，即 INT4 式码本）+ 块 scale ≈ amax/4"一致——即 **-hp 实际编码并非 E2M1 码本**，即使修好 scale 也无法被 routeB 的 E2M1 MMA 正确消费。
3. **编码语义与声明的格式规格不符**：其权重与真值（MXF4 反量化）相关性 0.951 但任何 `[K//32, N//128]` E8M0 块 scale 假设下重编码匹配率最高仅 38.7%（常数 scale 2⁻⁶）/ 17–28%（其它）——真实逐块 scale 信息既不在 scale 张量里，也无法从权重码恢复（连续幅度、且粒度细于 32×128）。

**结论**：-hp 转换器存在双重缺陷（scale 通道未写入 + 编码器码本/粒度与格式规格不一致）。其目标格式（[K, N//2] + [K//32, N//128] 块 scale）与 kernel2 v17 Triton 路径的输入格式吻合——-hp 应是为 v17 路径准备的。**建议**：routeB 线放弃 -hp，直配生产 MXF4；若 -hp 路线仍有需求，需转换器所有者重写（真实 E8M0 块 scale + E2M1 码本重编码，且需评估从 MXF4 二次量化的精度损失）。

---

## 2. A 侧量化（任务 §2）

干净 torch 实现（未采用 v17 Triton kernel：其 `_a_quant_kernel` 的 grid 硬编码 BLOCK_M=128 假设 bug 已知，autotune 选 32/64 时 ~50% 行未量化——直接弃用，语义对齐其数值行为）：

- **E8M0**（对齐 v17 + 金标准）：`byte = clamp(floor(log2(amax/6)) + 127, 0, 255)`，amax 下限 1e-30。校准向量实测：零输入→24，1e6→144 ✅（v17 用 1e-38 下限，全零组给字节 0；两者仅在全零 32 组上有差异且数值贡献均为 0，采用金标准 1e-30）。
- **E2M1 就近量化**：tie 取低档，阈值与 v17（>0.25/0.75/1.25/1.75/2.5/3.5/5.0）逐点一致；−0 归一为 +0（v17 语义）；clamp ±6。
- **打包**：`[M, K//2]`，低半字节=偶数 K —— 与 kernel 侧实测一致（probe P5：+0.5/−1.0 交替输入 → 字节 0xA1，lo=+0.5=偶 k，hi=0xA=−1.0=奇 k ✅）。

A 量化随适配器交付（`quantize_a`），并含 pack/unpack 往返自检。

---

## 3. ★Scale 布局契约（任务 §3，R13 落锤点）

**结论：vendored kernel 要求 atom-swizzle 布局，plain `[mn, K//32]` 直传不可行。**

证据链：
1. **源码级**：kernel `__call__` 内 `sfa_tensor = cute.make_tensor(sfa.iterator, tile_atom_to_shape_SF(a.shape, sf_vec_size))` —— 只取数据指针，按 `BlockScaledBasicChunk`（K-major atom `((32,4),(32,4))`，stride `((16,4),(0,1))`，即 `offset(m,kg) = (m%32)·16 + ((m//32)%4)·4 + kg%4`，atom 以 `(m//128, kg//4)` 平铺）重排。plain 布局的字节会被张冠李戴。
2. **实证级**（p3_probe_layout.py，GPU 容器）：以可逆字节模式注入官方 `cvt_sf_MKL_to_M32x4xrm_K4xrk_L`，导出真实缓冲排布，与纯 torch 公式**逐字节 100% 一致**（含平坦内存序 equal）：

```
buf[l, m//128, kg//4, m%32, (m//32)%4, kg%4] = plain[m, kg]
buf: (l, ceil(m/128), ceil(K/32/4), 32, 4, 4) 连续 uint8
m 补齐 128 倍数（补 0x7F 中性值）；sf_k = K//32 需 4 的倍数（K=2048/4096 → 64/128 ✓）
```

3. **数值级**：适配器 `sf_plain_to_atom`/`sf_atom_to_plain` 双向实现（含往返自检），配合生产真实 scale 字节（值域 119–122，非平凡）通过 §4 全部数值判决——若 swizzle 有任何坐标错误，rel_err 会是 O(1) 级而非 4e-4 级。

另：SFB 的生产 scale 粒度（逐行 `[N, K//32]`）与 kernel SFB 期望**逐项相同**，无需 -hp 路径所需的块 scale 广播扩展；SFA/SFB 缓冲均为小张量（M=4096,K=4096 时 SFA 192KB）。

---

## 4. 数值对照（任务 §4，判决）

**对象**：生产 MXF4 layer 0 expert 0 真实权重（w1: K=4096→N=2048；w2: K=2048→N=4096；w3: 同 w1）。
**参考**：`dequant_a(quant(A)) @ dequant_w(W)ᵀ`（f32，TF32 关闭；两侧同为量化语义，差异仅剩累加序 + fp16 出舍入）。
**判据**：`rel_err = max|out−ref| / max|ref| ≤ 1e-2`（对齐 routeA）。

| mat | M=64 | M=256 | M=1024 | M=4096 | M=257(奇) |
|---|---|---|---|---|---|
| w1 (K4096→N2048) | 3.56e-4 | 3.66e-4 | 3.91e-4 | 3.32e-4 | 3.89e-4 |
| w2 (K2048→N4096) | 2.87e-4 | 2.84e-4 | 2.59e-4 | 3.53e-4 | 2.53e-4 |
| w3 (K4096→N2048) | 4.26e-4 | 3.74e-4 | 3.75e-4 | 3.04e-4 | 3.91e-4 |

**15/15 PASS，最差 4.26e-04**（判据余量 ~23×）。误差量级恰为 fp16 输出舍入（~5e-4 相对）+ f32 累加序差的理论预期，无系统性偏差。

跨层/跨 expert 稳健性抽查（p3_robustness.py）：

| tensor | M | rel_err |
|---|---|---|
| layers.0.experts.7.w1 | 256 | 3.88e-4 ✅ |
| layers.21.experts.5.w2 | 256 | 2.65e-4 ✅ |
| layers.42.experts.200.w3 | 1024 | 3.01e-4 ✅ |
| layers.0.experts.255.w1 | 129(奇) | 4.02e-4 ✅ |

附带 transitively 验证：kernel 的 B 侧 nibble 约定（lo=偶 k）与生产 MXF4 打包一致（直配真实字节下 rel_err 4e-4，若约定不符将是 O(1) 误差）；M 预测（64<tile_m、257 奇数）正确；tile 128×128×128 / epi 128×128 / c_dtype fp16（B-N1 铁律遵守）。

---

## 5. 交付物

| 文件 | 位置 | 说明 |
|---|---|---|
| `routeb_prod_adapter.py` | 本地 `_routeb_extract/routeb-delivery/`、02:/tmp/routeb_p3/、**01:/tmp/routeb_task12/**（md5 `7c46209…` 三方一致） | 适配器：`RouteBProdGEMM.gemm(A_bf16, W_packed[N,K//2], W_scale[N,K//32]) → fp16 [M,N]`；内含 A 量化 + scale atom-swizzle + 直配 kernel 调用（编译按 shape 缓存）；`reference()` 提供判据参考；CPU 自检（E8M0 校准/swizzle 往返/pack 往返） |
| `p3_validate.py` | 同上 | 数值判决脚本（15 shape 全表，产出 p3_results.txt） |
| `p3_robustness.py` | 同上 | 跨层/跨 expert 抽查 |
| `p3_inspect_weights.py` 等 7 个诊断脚本 | 02:/tmp/routeb_p3/、本地 | §1 权重取证全程可复现（inspect/diag×5/probe×2） |

**适配器直配要点**（P4 集成注意）：
- B 侧 = 生产字节**零拷贝语义**（写入 kernel 张量前 m*k/2 字节，无转置/重打包）——生产加载时甚至可零拷贝视图。
- A 侧量化 + SFA swizzle 在 host/GPU torch 完成；编译按 (M,N,K,dtype,tile) 缓存，M 变化即重编译（官方示例 mark_compact_shape_dynamic 的动态维是 K 侧，M 复用编译未验证，P4 计时请用 `--skip-ref-check` 的 bench 路径为主）。
- 约束：K、N 需 128 倍数（生产 w1/w2/w3 全满足）；输出仅 fp16/bf16（B-N1）。
- 本适配器面向**数值正确性**；大规模性能口径仍以 `routeb_bench_blockscaled.py`（P2 已验证 368 TFLOPS）为准。

---

## 6. 遗留与建议

1. **-hp 转换器缺陷上报**（高优先）：建议向 -hp 转换器所有者反馈 §1.3 三层证据；routeB 线按生产 MXF4 直配推进（本报告已验证）。若生产 serving 栈当前实际加载的是 -hp，需评估其输出质量影响（幅度 ~47× 失真 + 码本错配，理论上不可能产出正常 logits——推测 serving 实际用的是 -0731 或 -nvfp4，建议核实生产加载路径）。
2. **A 量化 kernel 化**：当前 A 量化/SFA swizzle 为 torch 实现（数值正确优先）。生产 prefill 热路径需 Triton/CUDA 化（v17 `_a_quant_kernel` 修复 grid 硬编码后可复用，或按本适配器语义重写）；swizzle 本身是纯 permute，可并入量化 kernel 尾部。
3. **-nvfp4 路径**（可选）：若未来生产切换到 modelopt NVFP4（组 16 E4M3×标量两级 scale），routeB 现路径不匹配——需 KV 评估（组 16 → vec 16 走官方 NVFP4 路径 `sf_vec_size=16, Float8E4M3FN`，或转换权重）。本次仅实证其与 MXF4 数值等价，未做对接。
4. **bf16 输出**未单独跑表（kernel 支持，B-N1 仅要求 16-bit）；P4 如需 bf16 口径可一键切换 `out_dtype`。
5. 一次性容器约束全程遵守：`docker run --rm` 用后即删、/model* 只读挂载、未启动生产、02 GPU 仅小 shape 数值验证（单次峰值显存 < 1.5GB）、未触碰 01 GPU（P4 由另一代理占用）。

## 附：环境与复现

- 容器：`docker run --rm --gpus all -v /tmp/routeb_p3:/work -v .../deepseek-v4-flash-0731:/model_base:ro --entrypoint bash <NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（DSL 4.5.2 原生 sm_121a，无需 patch；`cutlass.testing` shim 同 P2）
- 复现顺序：`python3 routeb_prod_adapter.py`（自检）→ `p3_validate.py`（判决）→ `p3_robustness.py`（抽查）；取证链：`p3_inspect_weights.py` → `p3_diag_scale.py` → `p3_diag_cross.py` → `p3_diag_hpvsmx.py`/`p3_diag_recover.py`/`p3_diag_final.py`/`p3_diag_gran.py`/`p3_diag_codebook.py`（-hp 缺陷证据）→ `p3_probe_layout.py`/`p3_probe_tensor_api.py`（scale 契约 + 张量构造 API）
