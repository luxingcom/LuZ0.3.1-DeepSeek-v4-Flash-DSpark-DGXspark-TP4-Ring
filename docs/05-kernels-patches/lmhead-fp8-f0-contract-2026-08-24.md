# lm_head FP8 F0 契约核对报告（2026-08-24）

- **执行人**: 阿奇（Archi）· 系统架构师（architect-2）
- **任务**: lm_head FP8 系列第一硬前置 F0 —— head.weight BF16→FP8 E4M3 转化契约与生产内存布局核对（纯离线，只读 SSH + CPU-only 容器源码抽取，**未触碰 GPU/生产进程**）
- **上游输入**: lmhead-fp8-project-2026-08-23.md（F0 定义/F1 go-no-go）、fp8-quality-impact-2026-08-23.md（scale 布局裁定 = routeB 原生 [N,K/32]）、opt-routeb-fp8-2026-08-23.md（routeB FP8 kernel 契约）
- **勘察**: node01（<NODE_IP>，本机 SSH key，只读命令）；checkpoint `<INSTALL_DIR>/models/deepseek-v4-flash-0731/`；routeB kernel `<INSTALL_DIR>/nvfp4/routeb_official_v2/`；vLLM 源码经镜像 `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1` CPU-only 一次性容器 cat/grep（不启动 GPU）
- **口径标注**: 【实测-源码】= 服务器 checkpoint/容器源码直接验证；【实测】= 本团队既有实测数据；【推算】= 基于格式性质/形状/既有数据；【待窗口验证】= 必须由 F1 起 GPU 窗口实证

---

## 0. 一页结论

**F0 契约闭合结论：GO（条件性）——可进 F1 微基准**。六个子项中 5 项闭合（head.weight 现状、FP8 转化契约、routeB 契约主体、零 staging 可行性、golden 资产），**1 项为契约缺口需 F1 首测项（N=32320 % 128 ≠ 0 需 padding +64；M<128 小 M 尾 tile 未源码验证）**。未闭合项均可在 F1 一次性容器内闭环，不阻塞 F1 启动。

| 子项 | 判定 | 关键证据 |
|---|---|---|
| F0.1 head.weight 现状 | ✅ 闭合 | BF16 [129280,4096]，shard 45，K-contiguous；**row 0 (token 0) 异常值（K[0:224] 含 ~4e37 巨值）** |
| F0.2 scale 布局 | ✅ 闭合 | routeB 原生 [N, ceil(K/32)] E8M0（sf_vec=32），直接满足，零 staging |
| F0.3 routeB B 操作数契约 | ⚠️ 主体闭合 / 2 项待 F1 | B [N,K] K-major ✓、SFB [N,K/32] ✓、K%128 ✓、FP8 dispatch (FP8,FP8,E8M0,32)→MmaMXF8Op ✓；**N=32320 非 128 倍数 → 需 pad 32384**；M=8 尾 tile 行为未源码验证 |
| F0.4 零 staging 可行性 | ✅ 源码级成立 | lm_head 现役 = `UnquantizedEmbeddingMethod`（BF16 [32320,4096] K-contiguous，无转置/无重排）；FP8 payload+scale 布局与 routeB B/SFB 逐字节匹配 |
| F0.5 golden 资产 | ✅ 产出 | BF16 md5 + 转化脚本 + FP8/scale 样本 + manifest（见 §5） |
| F0.6 F1 go/no-go 前置 | ✅ 条件 GO | 契约缺口清单明确，F1 可直接测（见 §6） |

---

## 1. head.weight 现状确认（F0.1）【实测-源码】

### 1.1 来源与元数据

| 项 | 值 | 来源 |
|---|---|---|
| 张量 | `head.weight` | `model.safetensors.index.json` weight_map |
| 分片 | `model-00045-of-00048.safetensors` | 同上 |
| dtype | **BF16** | safetensors header |
| shape | **[129280, 4096]**（N=vocab_size=129280, K=hidden_size=4096） | header |
| data_offsets | [262164, 1059323924] | header |
| 字节 | 1059061760 B = 129280×4096×2（精确匹配） | 推算+header |
| 生产 config | `vocab_size=129280, hidden_size=4096, tie_word_embeddings=False` | config.json |

### 1.2 存储布局：K-contiguous 确认【实测-源码】

- safetensors 行主序存储 → `[129280,4096]` 的最后一个维度 K 是最内层连续 → **K-contiguous（= routeB B 操作数要求的 column-major "K"）**。
- 采样验证（15 行 × 4096 全 K）：值域正常（多数行 maxabs 0.47–3.8，均值近 0，全部非零），BF16 解码正确，无 NaN/Inf。

### 1.3 重要发现：row 0（token 0）异常行【实测-源码】

- **row 0 的 K[0:224]（前 7 个 32-组）含异常巨值**（maxabs 最高 4.0375e37，如 1.56e37、2.81e37 等；K[224:] 恢复正常 ~0.68）。全量稀疏扫描（每 100 行）仅 row 0 异常；row 1/2/127/128/129/129278/129279 均正常。
- 语义推测：token 0 为特殊/占位 token 的 lm_head 行，属 checkpoint 固有数据特征，非解析错误（BF16 解码在其余 12.9 万行全部正常）。
- **F0 契约影响**：
  1. E8M0 scale 动态范围：row 0 前 7 组 e≈122–125（byte 249–252），仍在 E8M0 合法范围（≤255）✓ 转化不会溢出；
  2. 该行 logits 无论 BF16/FP8 均为垃圾值（BF16 基线同样），FP8 不额外恶化；采样中 token 0 正常情况下不会被选中；
  3. golden 样本排除 row 0（rows 1..4096），并在 manifest 标注。
- 附注：checkpoint 另有 `hc_head_base/fn/scale`（shard 45）与 `mtp.2.*`（shard 48），`grep deepseek_v2.py` 未引用 → vLLM 不加载/不使用，**不在 lm_head FP8 范围内**。

---

## 2. FP8 转化契约（F0.2）

### 2.1 scale 布局裁定（沿用 fp8-quality-impact §3.1）

**采用 routeB 原生 `[N, ceil(K/32)]` E8M0（sf_vec=32，每行每 32-K 组一个 scale）**，三重占优：零 staging + 最细粒度 + 精度最优。K=4096 → 每行 128 组，SFB 形状 `[N,128]`。

- 与 128×128 块对比：本布局 K 方向粒度 32（vs 128），实测块内动态范围更小（采样 4 行：32-组中位动态范围 72–98 vs 128-块 241–594）→ 小值相对精度更优【实测-源码】。
- 与 per-channel [N,1] 对比：K 方向 128 组 vs 1 组，动态范围抑制更强，且天然满足 routeB sf_vec=32 契约。

### 2.2 转化算法（F0 golden 脚本实现，CPU 可跑）

```
对每行 n、每 32-K 组 g:
  scale[n,g] = 2^ceil(log2(max_abs(row[n, g*32:(g+1)*32])))     # E8M0: byte = e + 127
  payload[n, j] = round_to_nearest_E4M3(row[n,j] / scale[n,g])   # E4M3 normal + subnormal
反量化: w_hat = e4m3(payload) * scale
```

- E4M3 编码：1s+4e(bias 7)+3m；支持 normal + subnormal（min 2⁻⁹），<2⁻⁹ 冲刷为 0。
- E8M0：bias 127，byte = e+127；row 0 异常 e 最高 125 → byte 252，合法。
- **数据无关**：max-abs 幂次 scale，无校准集（与 fp8-quality-impact §3.2 一致）。

### 2.3 转化后字节账（TP4 分片 per-rank）【实测-源码 + 推算】

| 项 | 每 rank 字节 | 说明 |
|---|---|---|
| BF16 head.weight（现状） | 264,765,440 B ≈ **264.8 MB / 0.2466 GiB** | 32320×4096×2 |
| FP8 payload | 132,382,720 B ≈ **132.4 MB** | 32320×4096×1 |
| SFB scale [N,128] E8M0 | 4,136,960 B ≈ **4.1 MB** | 32320×128×1 |
| FP8 + scale 合计 | **136.5 MB** | — |
| **净节省** | **128.3 MB/rank**（4 rank ≈ 512 MB） | 264.8 − 136.5 |

> 修正立项口径：`lmhead-fp8-project` 写"显存 -135MB/rank"是 payload 半减（132.4MB）未扣 scale；**净省 ~128MB/rank**。生产 weight 账 45.32 GiB/rank（luZ0.3.1 终态）中 lm_head BF16 占 0.2466 GiB → FP8 后占 0.127 GiB，**净 −0.119 GiB/rank**。

### 2.4 精度包络【实测-源码 + 推算】

| 指标 | 值 | 依据 |
|---|---|---|
| 单元素最大相对误差（top-binade） | **2⁻⁴ = 6.25%**（半 ULP） | E4M3 格式性质 |
| 绝对误差界 | **≤ scale·2⁻⁴**（逐块有界） | 实测紧：max_abs_err=0.125 = 2·2⁻⁴（scale=2 块），16384 行验证 |
| RMS 绝对误差 | **0.0046**（64M 显著元素） | 实测 |
| 显著权重相对误差上限（\|w\|≥1e-2） | 16.4%（row 187，块内小值） | 实测：32-组块动态范围效应，非 bug；绝对误差仍有界 |
| logits 误差（端到端） | ~0.1–2% 量级【推算】 | fp8-quality-impact §1.3 模型 |

**诚实声明**：块内动态范围大时小值元素相对精度受损（已实测 16.4% 上界），这是 32-组布局的已知弱点；对 logits 的实际影响以 F2 参考集 KL/困惑度门为准【待窗口验证】。

---

## 3. routeB B 操作数契约核对（F0.3）【实测-源码】

勘察 `<INSTALL_DIR>/nvfp4/routeb_official_v2/`（`blockscaled_gemm_dispatch.py` + `dense_blockscaled_gemm_persistent_pingpong.py`）与容器内 cutlass `blockscaled_layout.py`：

### 3.1 契约匹配表

| 契约项 | routeB 要求（源码） | head.weight 转化后 | 匹配 |
|---|---|---|---|
| B dtype | `Float8E4M3FN`（FP8 E4M3） | FP8 E4M3 | ✅ |
| B 布局 | `NxKxL, B can only be column-major("K")` = K-contiguous | [N,K] K-contiguous | ✅ |
| SFB 布局 | `N × ceil_div(K, sf_vec_size)` plain scale | [N, 128]（sf_vec=32） | ✅ |
| sf_vec | FP8 强制 **32** | 32 | ✅ |
| sf_dtype | FP8 强制 `Float8E8M0FNU` | E8M0 | ✅ |
| 路由 | `(FP8, FP8, *, Float8E8M0FNU, 32) → MmaMXF8Op, use_mxf8f6f4=True, mma_K=32` | 命中 | ✅ |
| tile | FP8 仅允许 `(128,128,128)` | tile_K=128 | ✅ |
| tile_K 约束 | 128 的倍数（sf_vec=32） | K=4096=32×128 | ✅ |
| B 16B 对齐 | K-major 时 K % 16 == 0 | 4096 % 16 = 0 | ✅ |
| C 布局 | `MxN, row-major("N")` | logits [M,N] N-major | ✅ |
| C dtype | **16-bit 强校验**（Float16/BFloat16） | logits BF16 | ✅ |

### 3.2 缺口/待 F1 实证项【待窗口验证】

1. **N=32320 % 128 ≠ 0（契约缺口，需处理）**：
   - TP4 分片后 per-rank N=32320 = 128×252 + 64，非 128 倍数；
   - kernel 的 identity tile_map 为 `torch.arange(n // 128)`（252 项），而 grid 按 `zipped_divide`（ceil）算 253 个 N-tile —— 尾 64 行会丢/越界；
   - **建议方案：N padding 到 32384（+64 行，253×128）**，FP8 payload 增 64×4096=0.25MB/rank、scale 增 64×128=8KB/rank，可忽略；padding 行置 0，logits 侧由 vLLM vocab-mask 丢弃（ParallelLMHead 本就 pad 到 64 倍数，语义一致）。
   - F1 需实证：pad 32384 下正确性 + 性能；或 kernel 是否本就有 N-tail 处理（未在源码找到 mask 证据）。
2. **M<128（decode M=8）尾 tile 行为未源码验证**：
   - kernel epilogue 用编译期 `zipped_divide(gC, epi_tile)`，未发现运行时边界 mask；
   - **建议方案：A 侧 pad 到 128 行**（decode M=8 → [128,4096] 零填充，输出取前 8 行），由 F2 A-quant 适配器承接；
   - F1 需实测 M=8/96 的正确性与效率（16× M 计算浪费是否仍优于带宽受限 BF16）。
3. **运行级零拷贝（from_dlpack 对齐/divisibility/编译路径）**：源码级匹配，运行级待 F2 golden 实证（沿用 opt-routeb-fp8 判定口径）。

> **诚实标注**：第 3.1 节为源码级匹配（`run_bs` 参考校验 + dispatch 表 + layout 函数直接验证）；运行级 kernel 行为（M/N 尾 tile、FP8 same-dtype 实测吞吐）必须由 F1 一次性容器实证。

---

## 4. 零 staging 可行性（F0.4）【实测-源码】

### 4.1 head.weight 现役加载路径（vLLM）

- 模型：`DeepseekV2ForCausalLM`（deepseek_v2.py，registry 指向 deepseek_v4）→ `self.lm_head = ParallelLMHead(vocab_size, hidden_size, quant_config=quant_config)`。
- **`Fp8Config.get_quant_method` 对 `ParallelLMHead`（非 LinearBase/RoutedExperts/Attention）返回 None** → `VocabParallelEmbedding` 落 `UnquantizedEmbeddingMethod()`。
- `UnquantizedEmbeddingMethod.create_weights` 建 `torch.empty([num_embeddings_per_partition, embedding_dim], dtype=params_dtype)` = **[32320, 4096] BF16**；`process_weights_after_loading` 在 CUDA 侧**无操作**（仅 CPU 分支 dispatch）。
- 前向：`LogitsProcessor._apply_head` → `lm_head.quant_method.apply` → `dispatch_unquantized_gemm()(layer, x, layer.weight, bias)`（BF16 GEMM）。

**结论：生产内存中 head.weight = BF16 [32320,4096] K-contiguous，无转置、无重排、无量化。** 与 routeB B 操作数 [N,K] K-major 逐字节布局一致。

### 4.2 FP8 转化后的加载路径（设计）

| 路径 | 说明 | staging |
|---|---|---|
| 离线转化（golden 脚本） | checkpoint BF16 → FP8 payload + SFB scale，产出独立资产 | 一次性（CPU，非运行时） |
| 运行时 | vLLM lm_head 权重替换为 FP8 payload [N,K] + scale [N,128]，routeB adapter 经 `from_dlpack` 直配 | **零 per-step staging** |
| A 侧 | 激活 [M,4096] → A-quant（group=32 + E8M0 + swizzle），M<128 时 pad 到 128 行 | 每步 1 次（既有 Fp8LinearMethod A-quant 同族，F2 承接） |

**零 staging 成立的条件**：① F0.3 的 N-padding 在权重侧一次性完成（非 per-step）；② routeB adapter 直接从 in-memory FP8 权重读取（不复制 payload）。此两点均可在 F2 落地并 golden 实证。

### 4.3 与生产显存账衔接

- 生产 weight 基线 **45.32 GiB/rank**（luZ0.3.1 终态，W4A4B12xExperts hybrid）【实测】。
- lm_head 现状占 0.2466 GiB/rank；FP8 后占 0.127 GiB/rank（含 scale）→ **净 −0.119 GiB/rank（~128MB），四 rank 共 ~512MB**。
- 与 G1/native 派生显存账正交；无新增运行时副本。

---

## 5. 生产内存布局 golden（F0.5）【实测-源码 + 实测】

### 5.1 golden 资产（本报告同目录 `lmhead-fp8-f0-golden/`）

| 文件 | 内容 | 大小 | md5 |
|---|---|---|---|
| `golden_manifest.json` | 元数据 + 转化契约 + 字节账 + 验证结果 + 再生成命令 | 2.5KB | — |
| `head_fp8_rows1_4097.bin` | **FP8 E4M3 payload [4096,4096] K-contiguous**（rows 1..4096） | 16,777,216 B | `afebee1335743dc085b21d553064d2a1` |
| `head_scale_rows1_4097.bin` | **E8M0 scale [4096,128]**（rows 1..4096） | 524,288 B | `e4adaf4f07f7db585a68e10aa185a820` |
| `convert_lmhead_fp8.py` | CPU 转化/验证脚本（stdlib only） | 11KB | — |
| `verify_report.txt` | 验证输出 + 解释 | 1.7KB | — |
| `f0_sample_head.py` / `f0_row0_investigate.py` | 勘察采样脚本（溯源） | — | — |

### 5.2 BF16 参照（生产内存布局 golden）

```
tensor: head.weight  dtype: BF16  shape: (129280, 4096)
shard : model-00045-of-00048.safetensors  data_offsets: [262164, 1059323924]
md5_full_bf16 = a1241ccc196be2cb58aa80e7a2ba4b91   (全量 1059061760 B)
```

### 5.3 golden 验证结果（离线 CPU dequant 对比，16384 行 = 4 个 TP4 rank 切片范围）

- `max_abs_err = 0.125` = **精确等于 scale·2⁻⁴ 紧界**（scale=2 块）→ 转化实现正确；
- `max_rel_err（|w|≥1e-2）= 0.1636`（row 187 块内小值，32-组动态范围效应，有界绝对误差）；
- `rms_abs_err = 0.0046`（64M 显著元素）；
- row 0 异常已从样本排除并在 manifest 标注。

### 5.4 再生成命令（F1/F2 需全量时）

```
python3 convert_lmhead_fp8.py --mode=manifest                    # 全量 BF16 md5
python3 convert_lmhead_fp8.py --mode=convert --rows=N --row-start=R --out-dir=DIR
python3 convert_lmhead_fp8.py --mode=verify  --rows=N --row-start=R
```

---

## 6. F1 go/no-go 前置结论（F0.6）

### 6.1 契约闭合结论：**GO（条件性）**

| 项 | 闭合状态 | 备注 |
|---|---|---|
| head.weight 布局 | ✅ | BF16 [32320,4096] K-contiguous，无转置 |
| FP8 转化契约 | ✅ | routeB 原生 [N,K/32] E8M0，CPU 脚本 + golden 实证（误差界紧） |
| routeB 契约 | ⚠️ 2 项待 F1 | N-padding 方案就绪；M<128 尾 tile 需实证 |
| 零 staging | ✅ 源码级 | 运行级留 F2 golden |
| golden 资产 | ✅ | 见 §5 |

### 6.2 F1 必须覆盖的未闭合项清单（F0 契约缺口）

1. **N=32320 pad 到 32384 的 routeB FP8 正确性 + 性能**（+64 行零填充；或确认 kernel 自带 N-tail 处理——未在源码找到 mask 证据，倾向 pad）；
2. **M=8/96 decode 小 M 尾 tile**：A pad 128 后正确性 + E2E 效率（16× M 浪费 vs 带宽受限 BF16 0.27GB→0.135GB）；
3. FP8 same-dtype（FP8×FP8+E8M0+sf_vec=32）在 N=32384、K=4096、M∈{8,96,512,1024,4096} 的实际吞吐（routeB 350T 平台 vs 当前 cutlass 55–65T 外推）；
4. 运行级 from_dlpack 零拷贝 + SFB atom swizzle 编译路径（若 F2 提前）。

### 6.3 对 F1 go/no-go 门的建议（沿用 lmhead-fp8-project §3）

- `E2E(M=4096) ≥ 1.1× 当前 cutlass` + `A-quant delta < GEMM 增益` + **decode M=8 正确性（非性能门，正确性即可）+ N=32384 正确性**；
- 若 N-padding 不可行（性能崩塌）或 decode 小 M 正确性失败 → **lm_head FP8 降级设计储备**（沉没成本 = F0 半天 + F1 半天），回退维持 BF16。

---

## 7. 证据与假设分离清单

| 类型 | 内容 |
|---|---|
| 【实测-源码】 | head.weight=BF16[129280,4096]、shard 45、offsets、K-contiguous；row 0 异常（K[0:224] 巨值）；routeB dispatch `(FP8,FP8,E8M0,32)→MmaMXF8Op` + tile(128,128,128) + tile_K%128 + SFB [N,K/32] + B K-major + C 16-bit 强校验；`Fp8Config.get_quant_method` 对 ParallelLMHead 返回 None → UnquantizedEmbeddingMethod → BF16 [32320,4096] 无转置；config vocab=129280/hidden=4096/tie=False；hc_head/mtp.2 未被 deepseek_v2 引用 |
| 【实测】 | 15 行采样统计（正常行 maxabs 0.47–3.8）；row 0 巨值细查；每 100 行 outlier 扫描仅 row 0；32-组 vs 128-块动态范围（72–98 vs 241–594）；转化验证 16384 行：max_abs_err=0.125（紧界）、rms=0.0046、max_rel=0.1636；BF16 全量 md5 a1241ccc…；golden 样本 md5 |
| 【推算】 | 字节账（FP8 132.4MB + scale 4.1MB vs BF16 264.8MB，净省 ~128MB/rank）；logits 误差 0.1–2% 量级；N-padding 增量可忽略 |
| 【待窗口验证】 | N=32320→32384 的 routeB 正确性/性能；M<128 尾 tile 行为；运行级零拷贝 golden；FP8 same-dtype 实测吞吐；A-pad 对 decode E2E 的影响 |

**诚实声明**：
1. 本报告所有判定为源码级 + 离线 CPU 实测；routeB 运行级行为（尾 tile、吞吐）必须由 F1 GPU 容器实证，未闭合项已在 §6.2 显式列出。
2. row 0 异常行是 checkpoint 固有数据特征，BF16 基线同样存在垃圾 logits；FP8 不额外恶化，但**质量门参考集应避免以 token 0 为采样输出**。
3. 精度包络的 16.4% 显著元素相对误差上界是 32-组块动态范围效应（有界绝对误差），**不代表 logits 层面退化**；端到端以 F2 参考集 KL/困惑度门为准。
4. 所有【推算】数字在 F1/F4 实测后须替换为实测值。

---

## 8. 引用索引

- 本项目: `lmhead-fp8-project-2026-08-23.md`、`fp8-quality-impact-2026-08-23.md`、`opt-routeb-fp8-2026-08-23.md`（architect-1/2 同日产出）
- 生产账: `luz031-deployment-2026-08-23.md`、`a3-hybrid-slim-design-2026-08-23.md`、`arstall-production-closure-2026-08-23.md`（45.32 GiB weight 口径）
- 服务器: node01 `<INSTALL_DIR>/models/deepseek-v4-flash-0731/`、`<INSTALL_DIR>/nvfp4/routeb_official_v2/`；镜像 LuZ0.3.1 内 `vllm/model_executor/{models/deepseek_v2.py, layers/vocab_parallel_embedding.py, layers/logits_processor.py, layers/quantization/fp8.py}`、`flashinfer/data/cutlass/.../static_persistent_tile_scheduler.py`、`blockscaled_layout.py`

*本报告由工程保障团队（系统架构师 architect-2）生成；纯只读勘察 + 离线 CPU golden，未触碰 GPU/生产进程；F1 go/no-go 由人类工程负责人结合 F1 微基准裁定。*
